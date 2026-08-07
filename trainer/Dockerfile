# Trainer image for HF Jobs.
#
# Mounts the bucket at /app/cache (which holds `subsets/`, `tiers.json`,
# `tokenizer.json`, etc — produced locally by the Rust `pipeline tokenize`
# stage and uploaded). The trainer stages source binaries from the bucket
# into /app/scratch at startup so the dataloader mmaps from local disk, not
# FUSE — an earlier project wedged a bucket with concurrent random reads
# over FUSE, and this codepath is the fix.
#
# Build:  docker build -t <registry>/lattice-trainer:latest .
# Push:   docker push <registry>/lattice-trainer:latest
#
# Run on HF Jobs (single-GPU):
#   uvx hf jobs run --flavor a100-large --timeout 12h -s HF_TOKEN \
#     -v hf://buckets/<user>/<bucket>:/app/cache \
#     <registry>/lattice-trainer:latest \
#     trainer train --tier xs --scratch /app/scratch \
#     --out-dir /app/cache/runs/xs_$(date +%Y%m%d_%H%M%S)
#
# Run on HF Jobs (multi-GPU, single node, e.g. a100x4):
#   uvx hf jobs run --flavor a100x4 --timeout 12h -s HF_TOKEN \
#     -v hf://buckets/<user>/<bucket>:/app/cache \
#     <registry>/lattice-trainer:latest \
#     torchrun --nproc_per_node 4 -m trainer.cli train \
#         --tier small --scratch /app/scratch \
#         --out-dir /app/cache/runs/small_$(date +%Y%m%d_%H%M%S)
#
# Three things to note:
#
# 1. `uvx hf jobs run` overrides ENTRYPOINT and treats the trailing args as the
#    full command. Always pass `trainer ...` or `torchrun ...` explicitly.
#
# 2. Mount the bucket at /app/cache (== the CLI's `--cache-root` default).
#    Keeps everything under the project root in the container — no
#    hard-coded paths outside.
#
# 3. **Pass `--scratch /app/scratch` on every run.** The `subsets/` dirs are
#    multi-GB each; sparse mmap reads of those over a FUSE bucket mount
#    serialize under DDP contention (an earlier pipeline wedged a bucket with
#    three concurrent doc-side shards). Local scratch dodges this entirely.
#    Checkpoints are small (~125 MB each) and aligned-write, so they go
#    straight back to the bucket via `--out-dir /app/cache/runs/...`.

# CUDA 12.6 matches what the HF Jobs A100 fleet's driver supports based on
# earlier measurements. -runtime (not -devel) — we don't compile CUDA
# kernels here.
#
# `--platform linux/amd64` is explicit so the image builds correctly when
# the dev machine is arm64 (Apple Silicon) — HF Jobs hosts are x86_64.
# Without this, buildkit picks the host platform and `uv pip install`
# silently grabs ARM wheels that won't run on the GPU host.
FROM --platform=linux/amd64 pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
# HF caches go to the container-local default `/root/.cache/huggingface`,
# NOT on the bucket. The bucket would serialize the 1 GB NanoBEIR + tokenizer
# downloads. Container-local re-pays ~3-10 s per job — better trade.

# torch.compile shells out to gcc/g++ to build Triton kernels. Install minimal
# C compiler. (Skipping `-devel` saves ~5 GB and nvcc isn't needed.)
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc g++ \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /usr/local/bin/

WORKDIR /app

# Hatchling needs both the declared README and package source when preparing
# the editable install.
COPY pyproject.toml README.md /app/trainer/
COPY trainer /app/trainer/trainer
RUN cd /app/trainer \
 && uv venv /opt/venv \
 && uv pip install --python /opt/venv/bin/python -e . \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    --index-strategy unsafe-best-match
# `unsafe-best-match` lets uv consult PyPI for packages where the PyTorch
# index only carries a stale version (e.g. `requests` — the PyTorch index
# ships an old version that conflicts with `datasets>=2.20`'s requirement).
# The "unsafe" name is misleading: it just disables uv's strict
# first-index-wins behavior. We trust both indexes.

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/trainer

# Mount targets — some FUSE volume mounters fail when the path doesn't
# preexist in the image.
RUN mkdir -p /app/cache /app/scratch /app/cache/runs

# No ENTRYPOINT — HF Jobs overrides it anyway. Default to printing help.
CMD ["trainer", "--help"]
