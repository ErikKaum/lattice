"""Stage-2 fine-tuning loop.

Mirror of `train.py` (stage 1) with three differences:

- Loads initial weights from a stage-1 `.safetensors` checkpoint instead
  of random init — this is fine-tuning, not pre-training.
- Forward emits embeddings for query + positive + 7 hard negatives per
  example; loss uses the negative-aware `MatryoshkaLoss(MNR)` path.
- Default LR an order of magnitude lower than stage 1 (`2e-2` vs `2e-1`)
  to match plan2's "fine-tuning, not pre-training" framing. User can
  override.

Everything else — DDP `_ddp_env`, save_every / log_every cadence,
`_build_lr_lambda` linear warmup→decay, atomic safetensors checkpoint —
is the same as stage 1.

The init checkpoint is staged from the (potentially FUSE-mounted) cache
to a local path on rank 0 only, before any rank opens it. 4 ranks
mmap'ing the same FUSE-backed safetensors file deadlocked on the first
stage-2 launch — staging it locally first avoids the contention.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW

from .loss import DEFAULT_MATRYOSHKA_DIMS, MatryoshkaLoss, MultipleNegativesRankingLoss
from .model import BERT_BOUNDARY_TOKEN_IDS, StaticEmbeddingModel
from .stage2_dataloader import Stage2Dataloader
from .staging import stage_in_file, stage_in_stage2_training
from .train import _build_lr_lambda, _ddp_env, load_checkpoint, save_checkpoint


@dataclasses.dataclass
class Stage2TrainConfig:
    training_root: Path  # `<bucket>/stage2/training` — per-source dirs underneath
    out_dir: Path
    init_from: Path  # the stage-1 checkpoint we fine-tune from
    batch_size: int = 256
    num_epochs: int = 1
    max_steps: int | None = None
    learning_rate: float = 2e-2
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05  # shorter warmup than stage 1 — fine-tune
    log_every: int = 20
    save_every: int | None = 1000  # smaller default than stage 1 since stage 2 has
                                    # ~6K steps total, not ~80K
    seed: int = 42
    n_neg_sample: int = 7
    matryoshka_dims: tuple[int, ...] = DEFAULT_MATRYOSHKA_DIMS
    scale: float = 20.0
    device: str = "cuda"
    dtype: str = "float32"
    scratch_root: Path | None = None  # honored for future bucket-mounted runs


def stage2_train(cfg: Stage2TrainConfig) -> Path:
    rank, local_rank, world_size = _ddp_env()
    is_main = rank == 0

    torch.manual_seed(cfg.seed + rank)

    print(f"[rank{rank}] entered stage2_train (ws={world_size} local_rank={local_rank})", flush=True)

    if world_size > 1:
        # Same DDP init pattern as stage-1 (which is known to work on a100x4).
        dist.init_process_group(backend="nccl" if cfg.device == "cuda" else "gloo")
        if cfg.device == "cuda":
            torch.cuda.set_device(local_rank)
        print(f"[rank{rank}] init_process_group done", flush=True)

    # --------------------------------------------------------------
    # Stage-in. 18 GB of training binaries + 125 MB init checkpoint
    # both come from `/app/cache` (FUSE bucket). Reading them directly
    # under DDP contention deadlocks — 4 ranks all mmap'ing the same
    # FUSE-backed file produce zero-progress hangs that we observed on
    # the first 3 stage-2 launches. Stage everything to local scratch
    # using the same flock + sentinel discipline as stage-1.
    # --------------------------------------------------------------
    if cfg.scratch_root is None:
        # Local dev / single-GPU case — read directly. The deadlock only
        # appears under multi-rank FUSE pressure.
        read_training_root = cfg.training_root
        local_init = cfg.init_from
    else:
        scratch_root = cfg.scratch_root
        if is_main:
            print(
                f"stage-in: {cfg.training_root} (18 GB) + {cfg.init_from} (125 MB) "
                f"-> {scratch_root}",
                flush=True,
            )
        read_training_root = stage_in_stage2_training(
            cfg.training_root, scratch_root / "stage2" / "training",
        )
        local_init = stage_in_file(cfg.init_from, scratch_root / "init")
        print(f"[rank{rank}] stage-in done", flush=True)

    if world_size > 1:
        print(f"[rank{rank}] entering pre-loader barrier", flush=True)
        dist.barrier()
        print(f"[rank{rank}] passed barrier", flush=True)

    print(f"[rank{rank}] building Stage2Dataloader from {read_training_root}", flush=True)
    loader = Stage2Dataloader(
        training_root=read_training_root,
        batch_size=cfg.batch_size,
        n_neg_sample=cfg.n_neg_sample,
        seed=cfg.seed,
        rank=rank,
        world_size=world_size,
    )
    print(f"[rank{rank}] dataloader ready, {loader.steps_per_epoch()} steps/epoch", flush=True)
    steps_per_epoch = loader.steps_per_epoch()
    total_steps = steps_per_epoch * cfg.num_epochs
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)

    device = torch.device(
        f"{cfg.device}:{local_rank}" if cfg.device == "cuda" and world_size > 1
        else cfg.device
    )
    dtype = getattr(torch, cfg.dtype)

    print(f"[rank{rank}] loading init checkpoint from {local_init}", flush=True)
    init_ckpt = load_checkpoint(local_init)
    print(
        f"[rank{rank}] init checkpoint loaded "
        f"(vocab={init_ckpt['vocab_size']} dim={init_ckpt['embedding_dim']})",
        flush=True,
    )
    model = StaticEmbeddingModel(
        vocab_size=init_ckpt["vocab_size"],
        embedding_dim=init_ckpt["embedding_dim"],
        ignored_token_ids=(
            BERT_BOUNDARY_TOKEN_IDS if loader.add_special_tokens else ()
        ),
    )
    print(f"[rank{rank}] StaticEmbeddingModel constructed", flush=True)
    with torch.no_grad():
        model.embedding.weight.copy_(init_ckpt["embedding.weight"])
    model.zero_ignored_token_rows()
    print(f"[rank{rank}] init weight copied into model", flush=True)
    # Drop the source tensor so its memory isn't pinned for the rest of training.
    del init_ckpt
    print(f"[rank{rank}] moving to device {device}", flush=True)
    model = model.to(device=device, dtype=dtype)
    if world_size > 1:
        print(f"[rank{rank}] wrapping in DDP", flush=True)
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if cfg.device == "cuda" else None,
        )
        print(f"[rank{rank}] DDP ready", flush=True)

    criterion = MatryoshkaLoss(
        base_loss=MultipleNegativesRankingLoss(scale=cfg.scale),
        dims=cfg.matryoshka_dims,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _build_lr_lambda(total_steps, cfg.warmup_ratio),
    )

    if is_main:
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        _write_config(
            cfg, steps_per_epoch, world_size, loader.add_special_tokens
        )

    global_step = 0
    last_log = time.time()
    last_log_step = 0
    source_counter: Counter[str] = Counter()
    running_loss = 0.0
    running_count = 0

    if is_main:
        print(
            f"stage-2 train start: init_from={cfg.init_from} "
            f"steps_per_epoch={steps_per_epoch} total_steps={total_steps} "
            f"batch_size={cfg.batch_size} lr={cfg.learning_rate} "
            f"world_size={world_size} "
            f"cache_add_special_tokens={loader.add_special_tokens}",
            flush=True,
        )

    done = False
    for epoch in range(cfg.num_epochs):
        if done:
            break
        for batch in loader.epoch_iter(epoch):
            if cfg.max_steps is not None and global_step >= cfg.max_steps:
                done = True
                break

            q_ids = batch.query_input_ids.to(device, non_blocking=True)
            q_off = batch.query_offsets.to(device, non_blocking=True)
            p_ids = batch.positive_input_ids.to(device, non_blocking=True)
            p_off = batch.positive_offsets.to(device, non_blocking=True)
            n_ids = batch.negative_input_ids.to(device, non_blocking=True)
            n_off = batch.negative_offsets.to(device, non_blocking=True)

            anchors = model(q_ids, q_off)
            positives = model(p_ids, p_off)
            negatives_flat = model(n_ids, n_off)
            B, D = positives.shape
            negatives = negatives_flat.reshape(B, cfg.n_neg_sample, D)

            loss = criterion(anchors, positives, negatives)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if world_size > 1:
                loss_for_log = loss.detach().clone()
                dist.all_reduce(loss_for_log, op=dist.ReduceOp.AVG)
                running_loss += loss_for_log.item()
            else:
                running_loss += loss.item()
            running_count += 1
            source_counter[batch.source] += 1
            global_step += 1

            if is_main and global_step % cfg.log_every == 0:
                now = time.time()
                steps = global_step - last_log_step
                dt = now - last_log
                lr = optimizer.param_groups[0]["lr"]
                avg_loss = running_loss / running_count
                print(
                    f"step {global_step}/{total_steps} epoch {epoch} "
                    f"loss {avg_loss:.4f} lr {lr:.4g} "
                    f"steps/s {steps / max(dt, 1e-9):.2f}",
                    flush=True,
                )
                last_log = now
                last_log_step = global_step
                running_loss = 0.0
                running_count = 0

            if (
                is_main and cfg.save_every
                and global_step % cfg.save_every == 0
            ):
                save_checkpoint(model, cfg.out_dir / f"step_{global_step}.safetensors")

        if is_main:
            save_checkpoint(model, cfg.out_dir / f"epoch_{epoch}.safetensors")

    if is_main:
        save_checkpoint(model, cfg.out_dir / "final.safetensors")
        _write_source_distribution(cfg.out_dir, source_counter)
        print("stage-2 train complete", flush=True)
    if world_size > 1:
        dist.destroy_process_group()
    return cfg.out_dir / "final.safetensors"


def _write_config(
    cfg: Stage2TrainConfig,
    steps_per_epoch: int,
    world_size: int,
    cache_add_special_tokens: bool,
) -> None:
    data = dataclasses.asdict(cfg)
    data["training_root"] = str(cfg.training_root)
    data["out_dir"] = str(cfg.out_dir)
    data["init_from"] = str(cfg.init_from)
    data["scratch_root"] = str(cfg.scratch_root) if cfg.scratch_root else None
    data["resolved"] = {
        "steps_per_epoch_per_rank": steps_per_epoch,
        "world_size": world_size,
        "cache_add_special_tokens": cache_add_special_tokens,
        "ignored_token_ids": (
            list(BERT_BOUNDARY_TOKEN_IDS)
            if cache_add_special_tokens
            else []
        ),
    }
    (cfg.out_dir / "train_config.json").write_text(
        json.dumps(data, indent=2, default=str)
    )


def _write_source_distribution(out_dir: Path, counter: Counter[str]) -> None:
    total = sum(counter.values()) or 1
    rows = [
        {"source": s, "batches": n, "fraction": n / total}
        for s, n in sorted(counter.items(), key=lambda kv: -kv[1])
    ]
    (out_dir / "source_distribution.json").write_text(json.dumps(rows, indent=2))
