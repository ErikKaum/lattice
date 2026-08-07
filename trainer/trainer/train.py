"""Training loop for the static embedding model.

Mirrors the reference recipe from `sentence-transformers/static-retrieval-mrl-en-v1`:
- `MatryoshkaLoss` wrapping `MultipleNegativesRankingLoss` at
  `dims=[1024, 512, 256, 128, 64, 32]`, equal weights.
- AdamW.
- Linear warmup → linear decay schedule.
- One epoch over the filtered corpus.

The headline LR (`2e-1`) is correct for static embeddings trained from
scratch — the model is just an embedding table, not a pretrained transformer
being fine-tuned, so it needs an order-of-magnitude higher LR than the
sentence-transformers default of `2e-5`. This matches the public
static-embeddings blog (Tom Aarsen, 2024). If our run lands at a wildly off
loss, this is the first knob to verify against the reference's train.py.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW

from .dataloader import TierDataloader
from .io import Tier, load_tiers
from .loss import (
    DEFAULT_MATRYOSHKA_DIMS,
    MatryoshkaLoss,
    MultipleNegativesRankingLoss,
)
from .model import BERT_BOUNDARY_TOKEN_IDS, StaticEmbeddingModel
from .staging import stage_in_tier


@dataclasses.dataclass
class TrainConfig:
    cache_root: Path
    out_dir: Path
    tier: str = "xs"
    batch_size: int = 2048
    num_epochs: int = 1
    max_steps: int | None = None
    """Hard cap on training steps (across all epochs). Useful for smoke tests
    and DDP integration checks; None means "run to end of `num_epochs`"."""
    learning_rate: float = 2e-1
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    log_every: int = 20
    eval_every: int | None = None  # steps; None = end of epoch only
    save_every: int | None = 10000
    """Steps between intermediate checkpoints. None or 0 = end-of-epoch
    only. 10000 gives ~8 saves over a full run on a100x4 — enough to
    bound rewind on a late-stage failure without spamming the bucket."""
    seed: int = 42
    matryoshka_dims: tuple[int, ...] = DEFAULT_MATRYOSHKA_DIMS
    scale: float = 20.0
    device: str = "cuda"
    dtype: str = "float32"
    scratch_root: Path | None = None
    """If set, stage the tier's binaries from `cache_root` to `scratch_root`
    at startup and mmap from there. Essential on HF Jobs where `cache_root`
    is a FUSE-mounted bucket; sparse mmap reads of multi-GB token files
    over FUSE serialize catastrophically under multi-rank contention."""


def _build_lr_lambda(total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = float(step - warmup_steps) / float(decay_steps)
        return max(0.0, 1.0 - progress)

    return lr_lambda


def train(cfg: TrainConfig) -> Path:
    rank, local_rank, world_size = _ddp_env()
    is_main = rank == 0

    torch.manual_seed(cfg.seed + rank)  # different ranks see different RNG state

    if world_size > 1:
        dist.init_process_group(backend="nccl" if cfg.device == "cuda" else "gloo")
        if cfg.device == "cuda":
            torch.cuda.set_device(local_rank)

    # Stage-in. Every rank participates so the lock + sentinel serialize them.
    # When `scratch_root` is the same path on every rank (the recommended
    # config), only the first rank to acquire each per-source lock actually
    # pulls bytes from the bucket; the rest reuse the staged copy.
    read_root = cfg.cache_root
    tiers = load_tiers(cfg.cache_root)
    if cfg.tier not in tiers:
        raise KeyError(f"tier {cfg.tier} not in tiers.json (have {list(tiers)})")
    tier: Tier = tiers[cfg.tier]

    if cfg.scratch_root is not None:
        if is_main:
            print(f"stage-in: {cfg.cache_root} -> {cfg.scratch_root}", flush=True)
        read_root = stage_in_tier(cfg.cache_root, cfg.scratch_root, tier)
        # Reload tiers from scratch so downstream paths stay consistent.
        tier = load_tiers(read_root)[cfg.tier]

    if world_size > 1:
        dist.barrier()  # all ranks finished staging before any starts reading

    loader = TierDataloader(
        tier=tier,
        cache_root=read_root,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        rank=rank,
        world_size=world_size,
    )
    steps_per_epoch = loader.steps_per_epoch()
    total_steps = steps_per_epoch * cfg.num_epochs
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)

    device = torch.device(
        f"{cfg.device}:{local_rank}" if cfg.device == "cuda" and world_size > 1 else cfg.device
    )
    dtype = getattr(torch, cfg.dtype)

    ignored_token_ids = BERT_BOUNDARY_TOKEN_IDS if loader.add_special_tokens else ()
    model = StaticEmbeddingModel(
        ignored_token_ids=ignored_token_ids,
    ).to(device=device, dtype=dtype)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if cfg.device == "cuda" else None,
        )
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
        optimizer, _build_lr_lambda(total_steps, cfg.warmup_ratio)
    )

    if is_main:
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        _write_config(cfg, tier, steps_per_epoch, world_size, loader.add_special_tokens)

    global_step = 0
    last_log = time.time()
    last_log_step = 0
    source_counter: Counter[str] = Counter()
    running_loss = 0.0
    running_count = 0

    if is_main:
        print(
            f"train start: tier={cfg.tier} steps_per_epoch={steps_per_epoch} "
            f"total_steps={total_steps} batch_size={cfg.batch_size} "
            f"lr={cfg.learning_rate} world_size={world_size} "
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
            d_ids = batch.doc_input_ids.to(device, non_blocking=True)
            d_off = batch.doc_offsets.to(device, non_blocking=True)

            anchors = model(q_ids, q_off)
            positives = model(d_ids, d_off)
            loss = criterion(anchors, positives)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # For logging only: average loss across ranks so rank-0 prints
            # the global training loss, not just its disjoint slice's loss.
            # `clone().detach()` because all_reduce is in-place and we
            # don't want to perturb the autograd state.
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

            if is_main and cfg.save_every and global_step % cfg.save_every == 0:
                save_checkpoint(model, cfg.out_dir / f"step_{global_step}.safetensors")

        if is_main:
            save_checkpoint(model, cfg.out_dir / f"epoch_{epoch}.safetensors")

    if is_main:
        save_checkpoint(model, cfg.out_dir / "final.safetensors")
        _write_source_distribution(cfg.out_dir, source_counter)
        print("train complete", flush=True)
    if world_size > 1:
        dist.destroy_process_group()
    return cfg.out_dir / "final.safetensors"


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    """Save the embedding weights to `path` as safetensors.

    The reference `static-retrieval-mrl-en-v1` ships safetensors, and
    `torch.save` writes pickle — which is unsafe to load from untrusted
    sources and ~2x slower to memmap on read. `vocab_size` and
    `embedding_dim` ride along as string metadata in the safetensors
    header (the format's only-strings restriction), recovered as ints
    by `load_checkpoint`.

    DDP-wrapped models are unwrapped first.
    """
    inner = model.module if isinstance(model, DistributedDataParallel) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"embedding.weight": inner.embedding.weight.detach().cpu().contiguous()},
        str(path),
        metadata={
            "vocab_size": str(inner.vocab_size),
            "embedding_dim": str(inner.embedding_dim),
            "tokenizer": "bert-base-uncased",
            "ignored_token_ids": ",".join(str(token_id) for token_id in inner.ignored_token_ids),
        },
    )
    print(f"saved checkpoint -> {path}", flush=True)


def load_checkpoint(path: Path) -> dict:
    """Read a checkpoint produced by `save_checkpoint`.

    Returns `{"embedding.weight": Tensor, "vocab_size": int,
    "embedding_dim": int}`. Forces the weight off the safetensors mmap
    via `.clone()` — otherwise a downstream `.copy_()` page-faults
    against the source file, and on FUSE-mounted buckets that produces
    multi-rank contention that can deadlock (seen on the stage-2 launch
    where 4 ranks all loaded the same 125 MB file from `/app/cache`).
    """
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
        weight = f.get_tensor("embedding.weight").clone()
    return {
        "embedding.weight": weight,
        "vocab_size": int(meta.get("vocab_size", weight.shape[0])),
        "embedding_dim": int(meta.get("embedding_dim", weight.shape[1])),
        "ignored_token_ids": tuple(
            int(token_id) for token_id in meta.get("ignored_token_ids", "").split(",") if token_id
        ),
    }


def _ddp_env() -> tuple[int, int, int]:
    """Return `(rank, local_rank, world_size)` from torchrun env vars.
    Defaults to `(0, 0, 1)` for non-distributed runs."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])
        return rank, local_rank, world_size
    return 0, 0, 1


def _write_config(
    cfg: TrainConfig,
    tier: Tier,
    steps_per_epoch: int,
    world_size: int,
    cache_add_special_tokens: bool,
) -> None:
    data = dataclasses.asdict(cfg)
    data["cache_root"] = str(cfg.cache_root)
    data["out_dir"] = str(cfg.out_dir)
    data["scratch_root"] = str(cfg.scratch_root) if cfg.scratch_root else None
    data["resolved_tier"] = {
        "name": tier.name,
        "target_rows": tier.target_rows,
        "actual_rows": tier.actual_rows,
        "steps_per_epoch_per_rank": steps_per_epoch,
        "world_size": world_size,
        "cache_add_special_tokens": cache_add_special_tokens,
        "ignored_token_ids": (list(BERT_BOUNDARY_TOKEN_IDS) if cache_add_special_tokens else []),
    }
    (cfg.out_dir / "train_config.json").write_text(json.dumps(data, indent=2, default=str))


def _write_source_distribution(out_dir: Path, counter: Counter[str]) -> None:
    total = sum(counter.values()) or 1
    rows = [
        {"source": s, "batches": n, "fraction": n / total}
        for s, n in sorted(counter.items(), key=lambda kv: -kv[1])
    ]
    (out_dir / "source_distribution.json").write_text(json.dumps(rows, indent=2))
