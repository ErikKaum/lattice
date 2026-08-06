"""Modal launchers for the tokenizer-corrected training reproduction.

The training cache remains canonical in the Hugging Face Storage Bucket. The
launcher syncs only the immutable training inputs to Modal's local SSD, runs
the unchanged trainer against that local copy, then syncs checkpoints and eval
results back to the bucket. This avoids random mmap reads through a FUSE mount.

Required Modal secrets:

* ``huggingface-token`` with ``HF_TOKEN`` (NanoBEIR dataset access)
* ``hf-bucket-s3`` with ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``,
  and ``AWS_REGION`` (checkpoint uploads)

Run the storage compatibility probe before allocating GPUs::

    modal run scripts/modal_train.py::bucket_smoke

Then launch the full reproduction::

    modal run --detach --timestamps scripts/modal_train.py::train_full

After stage 1 completes, launch the historical 10-epoch stage 2 recipe::

    modal run --detach --timestamps scripts/modal_train.py::train_stage2
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import modal


APP_NAME = "lattice-tokenizer-reproduction"
TRAINER_IMAGE = (
    "erikkaum/lattice-trainer@"
    "sha256:ba8c051c4aca9a2fae63b5b6a386931a101e6a767eab841e7d21df58e20da5d6"
)
HF_BUCKET = "hf://buckets/erikkaum/training-cache-static"
HF_S3_ENDPOINT = "https://s3.hf.co/erikkaum"
HF_S3_BUCKET = "training-cache-static"
FULL_RUN_NAME = "tokenizer_full_20260803_modal_a100x4_r1"
STAGE2_RUN_NAME = "tokenizer_stage2_10ep_20260803_modal_a100x4_r1"

app = modal.App(APP_NAME)

# Reuse the exact image that passed the corrected xs/small/medium runs. Clearing
# the image entrypoint lets Modal invoke the Python wrapper normally; the actual
# trainer remains an unchanged subprocess inside that image.
image = modal.Image.from_registry(
    TRAINER_IMAGE,
    # Keep Modal's function runtime isolated. Installing it into /opt/venv
    # would downgrade packages pinned by the already-validated trainer image.
    setup_dockerfile_commands=[
        "RUN uv venv --python /opt/venv/bin/python /opt/modal-venv",
        "RUN uv pip install --python /opt/modal-venv/bin/python pip awscli",
        "ENV PATH=/opt/modal-venv/bin:/opt/venv/bin:$PATH",
    ],
).entrypoint([]).add_local_file(
    "trainer/trainer/decontam_eval.py",
    "/app/trainer/trainer/decontam_eval.py",
).add_local_file(
    "trainer/trainer/eval.py",
    "/app/trainer/trainer/eval.py",
).add_local_file(
    "trainer/trainer/decontam_quant_eval.py",
    "/app/trainer/trainer/decontam_quant_eval.py",
).add_local_file(
    "trainer/trainer/cli.py",
    "/app/trainer/trainer/cli.py",
)

hf_token = modal.Secret.from_name(
    "huggingface-token",
    required_keys=["HF_TOKEN"],
)
hf_bucket_credentials = modal.Secret.from_name(
    "hf-bucket-s3",
    required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
)


@app.function(
    image=image,
    secrets=[hf_token, hf_bucket_credentials],
    timeout=10 * 60,
    cpu=1.0,
    memory=2048,
)
def bucket_smoke() -> str:
    """Prove native HF bucket reads and safetensors round-trips on Modal."""
    cache_root = Path("/app/smoke/cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    _run_hf("cp", f"{HF_BUCKET}/tiers.json", str(cache_root / "tiers.json"))
    _run_hf(
        "cp",
        f"{HF_BUCKET}/tokenizer.json",
        str(cache_root / "tokenizer.json"),
    )
    tiers = json.loads((cache_root / "tiers.json").read_text())
    full = next((tier for tier in tiers if tier.get("tier") == "full"), None)
    if full is None:
        raise RuntimeError("full tier is missing from /app/cache/tiers.json")

    # Read a second object so the probe covers more than bucket listing and
    # metadata lookup. The tokenizer is small but follows the same GET path as
    # the large token binaries used by stage-in.
    tokenizer_prefix = (cache_root / "tokenizer.json").read_bytes()[:64]
    if not tokenizer_prefix:
        raise RuntimeError("tokenizer.json was empty")

    probe_dir = Path("/app/smoke/probe")
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_path = probe_dir / "probe.safetensors"
    validator = """
import sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

path, mode = sys.argv[1:]
expected = torch.arange(1024, dtype=torch.float32).reshape(32, 32)
if mode == "write":
    save_file({"embedding.weight": expected}, path,
              metadata={"purpose": "modal-hf-bucket-write-smoke"})
else:
    with safe_open(path, framework="pt") as handle:
        actual = handle.get_tensor("embedding.weight")
        metadata = handle.metadata() or {}
    if not torch.equal(actual, expected):
        raise RuntimeError("safetensors round-trip changed the probe tensor")
    if metadata.get("purpose") != "modal-hf-bucket-write-smoke":
        raise RuntimeError("safetensors round-trip lost checkpoint metadata")
"""
    subprocess.run(
        ["/opt/venv/bin/python", "-c", validator, str(probe_path), "write"],
        check=True,
    )
    remote_probe = f"{HF_BUCKET}/runs/modal_bucket_smoke_20260803/probe.safetensors"
    _run_aws(
        "s3",
        "cp",
        str(probe_path),
        f"s3://{HF_S3_BUCKET}/runs/modal_bucket_smoke_20260803/probe.safetensors",
    )
    downloaded_probe = probe_dir / "downloaded.safetensors"
    _run_hf("cp", remote_probe, str(downloaded_probe))
    subprocess.run(
        [
            "/opt/venv/bin/python",
            "-c",
            validator,
            str(downloaded_probe),
            "validate",
        ],
        check=True,
    )

    result = {
        "status": "ok",
        "full_rows": full["actual_rows"],
        "probe_path": remote_probe,
        "probe_bytes": downloaded_probe.stat().st_size,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return json.dumps(result, sort_keys=True)


@app.function(
    image=image,
    secrets=[hf_token, hf_bucket_credentials],
    gpu="A100:4",
    timeout=4 * 60 * 60,
    # The corrected full tier stages about 310 GB. Modal's default is 512 GiB;
    # make that dependency explicit so a future platform-default change cannot
    # silently undersize this reproduction.
    ephemeral_disk=512 * 1024,
)
def train_full() -> str:
    """Run historical full stage 1 plus NanoBEIR on the corrected tokenizer."""
    cache_root = Path("/app/cache")
    out_dir = Path("/app/output/stage1")
    remote_out = f"{HF_BUCKET}/runs/{FULL_RUN_NAME}/stage1"

    print("syncing full-tier inputs to local SSD", flush=True)
    _run_hf(
        "sync",
        HF_BUCKET,
        str(cache_root),
        "--include",
        "tiers.json",
        "--include",
        "tokenizer.json",
        "--include",
        "partition.json",
        "--include",
        "subsets/**",
    )

    train_command = [
        "torchrun",
        "--nproc_per_node",
        "4",
        "-m",
        "trainer.cli",
        "train",
        "--tier",
        "full",
        "--cache-root",
        str(cache_root),
        "--out-dir",
        str(out_dir),
        "--batch-size",
        "2048",
        "--epochs",
        "1",
        "--log-every",
        "200",
        "--save-every",
        "10000",
        "--device",
        "cuda",
        "--dtype",
        "float32",
    ]
    eval_command = [
        "trainer",
        "eval",
        str(out_dir / "final.safetensors"),
        "--cache-root",
        "/app/cache",
        "--out-dir",
        str(out_dir / "eval"),
    ]

    stop_backup = threading.Event()
    backup_thread = threading.Thread(
        target=_checkpoint_backup_loop,
        args=(out_dir, FULL_RUN_NAME, "stage1", stop_backup),
        daemon=True,
    )
    backup_thread.start()
    try:
        print("launching: " + " ".join(train_command), flush=True)
        subprocess.run(train_command, cwd="/app", check=True)
    finally:
        stop_backup.set()
        backup_thread.join()
        if out_dir.exists():
            _sync_output_s3(out_dir, FULL_RUN_NAME, "stage1")

    print("launching: " + " ".join(eval_command), flush=True)
    subprocess.run(eval_command, cwd="/app", check=True)
    print("syncing NanoBEIR results to HF bucket", flush=True)
    _sync_output_s3(out_dir, FULL_RUN_NAME, "stage1")

    result = {
        "status": "completed",
        "checkpoint": f"{remote_out}/final.safetensors",
        "nanobeir": f"{remote_out}/eval/nanobeir.json",
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return json.dumps(result, sort_keys=True)


@app.function(
    image=image,
    secrets=[hf_token, hf_bucket_credentials],
    gpu="A100:4",
    timeout=2 * 60 * 60,
    # Stage 2 needs only about 22 GB, but this Modal runtime currently enforces
    # a 512 GiB minimum explicit disk request.
    ephemeral_disk=512 * 1024,
)
def train_stage2() -> str:
    """Run the historical winning 10-epoch stage 2 plus held-out eval."""
    cache_root = Path("/app/cache")
    init_dir = Path("/app/init")
    init_checkpoint = init_dir / "stage1_final.safetensors"
    out_dir = Path("/app/output/stage2")
    remote_out = f"{HF_BUCKET}/runs/{STAGE2_RUN_NAME}/stage2"

    print("syncing stage-2 training data and held-out surface to local SSD", flush=True)
    _run_hf(
        "sync",
        HF_BUCKET,
        str(cache_root),
        "--include",
        "tokenizer.json",
        "--include",
        "stage2/training/**",
        "--include",
        "stage2/eval_surface/**",
    )
    init_dir.mkdir(parents=True, exist_ok=True)
    _run_hf(
        "cp",
        f"{HF_BUCKET}/runs/{FULL_RUN_NAME}/stage1/final.safetensors",
        str(init_checkpoint),
    )

    train_command = [
        "torchrun",
        "--nproc_per_node",
        "4",
        "-m",
        "trainer.cli",
        "stage2-train",
        "--init-from",
        str(init_checkpoint),
        "--training-root",
        str(cache_root / "stage2" / "training"),
        "--out-dir",
        str(out_dir),
        "--batch-size",
        "256",
        "--epochs",
        "10",
        "--lr",
        "0.02",
        "--weight-decay",
        "0",
        "--warmup-ratio",
        "0.05",
        "--log-every",
        "20",
        "--save-every",
        "1000",
        "--seed",
        "42",
        "--n-neg-sample",
        "7",
        "--device",
        "cuda",
        "--dtype",
        "float32",
    ]
    eval_command = [
        "trainer",
        "eval-stage2",
        str(out_dir / "final.safetensors"),
        "--cache-root",
        str(cache_root),
        "--surface-dir",
        str(cache_root / "stage2" / "eval_surface"),
        "--out-dir",
        str(out_dir / "eval"),
    ]

    stop_backup = threading.Event()
    backup_thread = threading.Thread(
        target=_checkpoint_backup_loop,
        args=(out_dir, STAGE2_RUN_NAME, "stage2", stop_backup),
        daemon=True,
    )
    backup_thread.start()
    try:
        print("launching: " + " ".join(train_command), flush=True)
        subprocess.run(train_command, cwd="/app", check=True)
    finally:
        stop_backup.set()
        backup_thread.join()
        if out_dir.exists():
            _sync_output_s3(out_dir, STAGE2_RUN_NAME, "stage2")

    try:
        print("launching: " + " ".join(eval_command), flush=True)
        subprocess.run(eval_command, cwd="/app", check=True)
    finally:
        if out_dir.exists():
            _sync_output_s3(out_dir, STAGE2_RUN_NAME, "stage2")

    result = {
        "status": "completed",
        "checkpoint": f"{remote_out}/final.safetensors",
        "heldout": f"{remote_out}/eval/stage2_heldout.json",
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return json.dumps(result, sort_keys=True)


@app.function(
    image=image,
    secrets=[hf_token, hf_bucket_credentials],
    gpu="T4",
    timeout=30 * 60,
)
def eval_stage1_heldout() -> str:
    """Evaluate corrected full Stage 1 on the Stage 2 held-out surface."""
    cache_root = Path("/app/cache")
    init_dir = Path("/app/init")
    checkpoint = init_dir / "stage1_final.safetensors"
    out_dir = Path("/app/output/stage1")
    remote_out = f"{HF_BUCKET}/runs/{FULL_RUN_NAME}/stage1"

    _run_hf(
        "sync",
        HF_BUCKET,
        str(cache_root),
        "--include",
        "tokenizer.json",
        "--include",
        "stage2/eval_surface/**",
    )
    init_dir.mkdir(parents=True, exist_ok=True)
    _run_hf(
        "cp",
        f"{remote_out}/final.safetensors",
        str(checkpoint),
    )
    eval_command = [
        "trainer",
        "eval-stage2",
        str(checkpoint),
        "--cache-root",
        str(cache_root),
        "--surface-dir",
        str(cache_root / "stage2" / "eval_surface"),
        "--out-dir",
        str(out_dir / "eval"),
    ]
    try:
        print("launching: " + " ".join(eval_command), flush=True)
        subprocess.run(eval_command, cwd="/app", check=True)
    finally:
        if out_dir.exists():
            _sync_output_s3(out_dir, FULL_RUN_NAME, "stage1")

    result = {
        "status": "completed",
        "heldout": f"{remote_out}/eval/stage2_heldout.json",
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return json.dumps(result, sort_keys=True)


@app.function(
    image=image,
    secrets=[hf_token, hf_bucket_credentials],
    gpu="A100",
    cpu=16.0,
    memory=64 * 1024,
    timeout=3 * 60 * 60,
)
def eval_stage1_decontam() -> str:
    """Run the 14-task decontaminated BEIR suite on corrected Stage 1."""
    return _eval_decontam(
        run_name=FULL_RUN_NAME,
        stage="stage1",
        checkpoint_remote=(
            f"{HF_BUCKET}/runs/{FULL_RUN_NAME}/stage1/final.safetensors"
        ),
    )


@app.function(
    image=image,
    secrets=[hf_token, hf_bucket_credentials],
    gpu="A100",
    cpu=16.0,
    memory=64 * 1024,
    timeout=3 * 60 * 60,
)
def eval_stage2_decontam() -> str:
    """Run the 14-task decontaminated BEIR suite on corrected Stage 2."""
    return _eval_decontam(
        run_name=STAGE2_RUN_NAME,
        stage="stage2",
        checkpoint_remote=(
            f"{HF_BUCKET}/runs/{STAGE2_RUN_NAME}/stage2/final.safetensors"
        ),
    )


@app.function(
    image=image,
    secrets=[hf_token, hf_bucket_credentials],
    gpu="A100",
    cpu=16.0,
    memory=64 * 1024,
    timeout=90 * 60,
)
def eval_stage2_quant() -> str:
    """Run the 78-cell NanoBEIR PTQ sweep on corrected Stage 2."""
    cache_root = Path("/app/cache")
    init_dir = Path("/app/init")
    checkpoint = init_dir / "stage2_final.safetensors"
    stage_out = Path("/app/output/stage2")
    eval_out = stage_out / "quant_sweep"
    remote_out = f"{HF_BUCKET}/runs/{STAGE2_RUN_NAME}/stage2"
    partial_path = eval_out / "quantization_sweep.partial.json"

    cache_root.mkdir(parents=True, exist_ok=True)
    _run_hf(
        "cp",
        f"{HF_BUCKET}/tokenizer.json",
        str(cache_root / "tokenizer.json"),
    )
    init_dir.mkdir(parents=True, exist_ok=True)
    _run_hf(
        "cp",
        f"{remote_out}/final.safetensors",
        str(checkpoint),
    )
    eval_out.mkdir(parents=True, exist_ok=True)
    try:
        _run_aws(
            "s3",
            "cp",
            (
                f"s3://{HF_S3_BUCKET}/runs/{STAGE2_RUN_NAME}/stage2/"
                "quant_sweep/quantization_sweep.partial.json"
            ),
            str(partial_path),
        )
        print(f"restored resumable quant sweep progress from {remote_out}", flush=True)
    except subprocess.CalledProcessError:
        print("no resumable quant sweep progress found; starting fresh", flush=True)

    eval_command = [
        "trainer",
        "eval-quant",
        str(checkpoint),
        "--cache-root",
        str(cache_root),
        "--out-dir",
        str(eval_out),
    ]
    stop_backup = threading.Event()
    backup_thread = threading.Thread(
        target=_checkpoint_backup_loop,
        args=(stage_out, STAGE2_RUN_NAME, "stage2", stop_backup),
        daemon=True,
    )
    backup_thread.start()
    try:
        print("launching: " + " ".join(eval_command), flush=True)
        subprocess.run(eval_command, cwd="/app", check=True)
    finally:
        stop_backup.set()
        backup_thread.join()
        if stage_out.exists():
            _sync_output_s3(stage_out, STAGE2_RUN_NAME, "stage2")

    result = {
        "status": "completed",
        "quant_sweep": f"{remote_out}/quant_sweep/quantization_sweep.json",
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return json.dumps(result, sort_keys=True)


@app.function(
    image=image,
    secrets=[hf_token, hf_bucket_credentials],
    gpu="A100",
    cpu=16.0,
    memory=64 * 1024,
    timeout=3 * 60 * 60,
)
def eval_deployment_int8_dim() -> str:
    """Evaluate the selected int8-dim variants on decontaminated BEIR."""
    return _eval_deployment_quant_group("int8_dim")


@app.function(
    image=image,
    secrets=[hf_token, hf_bucket_credentials],
    gpu="A100",
    cpu=16.0,
    memory=64 * 1024,
    timeout=3 * 60 * 60,
)
def eval_deployment_int4_dim() -> str:
    """Evaluate the selected int4-dim variants on decontaminated BEIR."""
    return _eval_deployment_quant_group("int4_dim")


@app.function(
    image=image,
    secrets=[hf_token, hf_bucket_credentials],
    gpu="A100",
    cpu=16.0,
    memory=64 * 1024,
    timeout=3 * 60 * 60,
)
def eval_deployment_row() -> str:
    """Evaluate the selected row-quantized variants on decontaminated BEIR."""
    return _eval_deployment_quant_group("row")


def _eval_deployment_quant_group(group: str) -> str:
    cache_root = Path("/app/cache")
    init_dir = Path("/app/init")
    checkpoint = init_dir / "stage2_final.safetensors"
    stage_out = Path("/app/output/stage2")
    deployment_out = stage_out / "deployment"
    remote_out = f"{HF_BUCKET}/runs/{STAGE2_RUN_NAME}/stage2"

    cache_root.mkdir(parents=True, exist_ok=True)
    _run_hf(
        "cp",
        f"{HF_BUCKET}/tokenizer.json",
        str(cache_root / "tokenizer.json"),
    )
    init_dir.mkdir(parents=True, exist_ok=True)
    _run_hf(
        "cp",
        f"{remote_out}/final.safetensors",
        str(checkpoint),
    )
    deployment_out.mkdir(parents=True, exist_ok=True)
    _run_aws(
        "s3",
        "sync",
        f"s3://{HF_S3_BUCKET}/runs/{STAGE2_RUN_NAME}/stage2/deployment",
        str(deployment_out),
    )

    eval_command = [
        "trainer",
        "eval-deployment-quant",
        str(checkpoint),
        "--cache-root",
        str(cache_root),
        "--out-dir",
        str(deployment_out),
        "--group",
        group,
    ]
    stop_backup = threading.Event()
    backup_thread = threading.Thread(
        target=_checkpoint_backup_loop,
        args=(stage_out, STAGE2_RUN_NAME, "stage2", stop_backup),
        daemon=True,
    )
    backup_thread.start()
    try:
        print("launching: " + " ".join(eval_command), flush=True)
        subprocess.run(eval_command, cwd="/app", check=True)
    finally:
        stop_backup.set()
        backup_thread.join()
        if stage_out.exists():
            _sync_output_s3(stage_out, STAGE2_RUN_NAME, "stage2")

    result = {
        "status": "completed",
        "group": group,
        "deployment": f"{remote_out}/deployment",
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return json.dumps(result, sort_keys=True)


def _eval_decontam(
    run_name: str,
    stage: str,
    checkpoint_remote: str,
) -> str:
    cache_root = Path("/app/cache")
    init_dir = Path("/app/init")
    checkpoint = init_dir / f"{stage}_final.safetensors"
    stage_out = Path("/app/output") / stage
    eval_out = stage_out / "decontam"
    remote_out = f"{HF_BUCKET}/runs/{run_name}/{stage}"
    partial_path = eval_out / "decontam_beir.partial.json"

    cache_root.mkdir(parents=True, exist_ok=True)
    _run_hf(
        "cp",
        f"{HF_BUCKET}/tokenizer.json",
        str(cache_root / "tokenizer.json"),
    )
    init_dir.mkdir(parents=True, exist_ok=True)
    _run_hf("cp", checkpoint_remote, str(checkpoint))
    eval_out.mkdir(parents=True, exist_ok=True)
    try:
        _run_aws(
            "s3",
            "cp",
            (
                f"s3://{HF_S3_BUCKET}/runs/{run_name}/{stage}/"
                "decontam/decontam_beir.partial.json"
            ),
            str(partial_path),
        )
        print(f"restored resumable decontam progress from {remote_out}", flush=True)
    except subprocess.CalledProcessError:
        print("no resumable decontam progress found; starting fresh", flush=True)

    eval_command = [
        "trainer",
        "eval-beir",
        str(checkpoint),
        "--cache-root",
        str(cache_root),
        "--out-dir",
        str(eval_out),
    ]
    stop_backup = threading.Event()
    backup_thread = threading.Thread(
        target=_checkpoint_backup_loop,
        args=(stage_out, run_name, stage, stop_backup),
        daemon=True,
    )
    backup_thread.start()
    try:
        print("launching: " + " ".join(eval_command), flush=True)
        subprocess.run(eval_command, cwd="/app", check=True)
    finally:
        stop_backup.set()
        backup_thread.join()
        if stage_out.exists():
            _sync_output_s3(stage_out, run_name, stage)

    result = {
        "status": "completed",
        "decontam": f"{remote_out}/decontam/decontam_beir.json",
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return json.dumps(result, sort_keys=True)


def _run_hf(*args: str) -> None:
    command = ["/opt/venv/bin/hf", "buckets", *args, "--format", "agent"]
    print("launching: " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _run_aws(*args: str) -> None:
    command = [
        "/opt/modal-venv/bin/aws",
        "--endpoint-url",
        HF_S3_ENDPOINT,
        *args,
        "--no-progress",
    ]
    env = os.environ.copy()
    env["AWS_REQUEST_CHECKSUM_CALCULATION"] = "when_required"
    print("launching: " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def _sync_output_s3(out_dir: Path, run_name: str, stage: str) -> None:
    print(f"syncing {stage} artifacts to HF bucket", flush=True)
    _run_aws(
        "s3",
        "sync",
        str(out_dir),
        f"s3://{HF_S3_BUCKET}/runs/{run_name}/{stage}",
    )


def _checkpoint_backup_loop(
    out_dir: Path,
    run_name: str,
    stage: str,
    stop: threading.Event,
) -> None:
    while not stop.wait(60):
        if not out_dir.exists():
            continue
        try:
            _sync_output_s3(out_dir, run_name, stage)
        except subprocess.CalledProcessError as exc:
            print(f"checkpoint sync failed transiently: {exc}", flush=True)
