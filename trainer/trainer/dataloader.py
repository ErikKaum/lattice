"""mmap-backed dataloader implementing batch-level source mixing.

For each batch:
- Sample a source proportional to its `take_rows` for the current tier.
- Pick a contiguous window of `batch_size` rows within that source.
- All rows in the batch come from the same source — in-batch negatives are
  realistic (the loss's hardest negatives are other items from the same
  domain).
- Across batches, source diversity is preserved by the proportional sampling.

Sampling is done via a *chunked shuffle*: for every source, enumerate the
batch-aligned window starts `[0, batch_size, 2*batch_size, ...]` up to
`(usable_rows // batch_size) * batch_size`. Tail rows < batch_size are
dropped per epoch — fine at the tier sizes we operate on. The flat list of
`(source_idx, window_start)` pairs is shuffled once per epoch; one pass over
that list is one epoch. This yields proportional source weighting for free
(big source → more windows → more chance per epoch), uniform coverage of
every row exactly once per epoch, and page-cache-friendly contiguous reads
within each batch.

Caller hot path:

    loader = TierDataloader(tier, cache_root, batch_size=4096)
    for epoch in range(n_epochs):
        for batch in loader:                  # reshuffles each epoch
            q_ids, q_offs, d_ids, d_offs, src = batch
            anchors = model(q_ids, q_offs)
            positives = model(d_ids, d_offs)
            loss = criterion(anchors, positives)

The tensors come out on CPU; the caller `.to(device)`s them. We don't pin
memory yet — measure first.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from .io import SourceReader, Tier, open_tier_sources


class Batch(NamedTuple):
    query_input_ids: torch.Tensor
    query_offsets: torch.Tensor
    doc_input_ids: torch.Tensor
    doc_offsets: torch.Tensor
    source: str
    """Name of the source this batch was drawn from, used for per-source
    distribution checks."""


class TierDataloader:
    def __init__(
        self,
        tier: Tier,
        cache_root: Path,
        batch_size: int,
        seed: int = 42,
        drop_last_per_source: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if batch_size <= 1:
            raise ValueError("batch_size > 1 required (MNR loss needs negatives)")
        if not (0 <= rank < world_size):
            raise ValueError(f"bad rank/world_size: {rank}/{world_size}")
        self.tier = tier
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last_per_source = drop_last_per_source
        self.rank = rank
        self.world_size = world_size
        self._sources: list[tuple[SourceReader, int]] = open_tier_sources(cache_root, tier)
        if not self._sources:
            raise ValueError(f"tier {tier.name} has no sources with take_rows > 0")
        token_policies = {reader.meta.add_special_tokens for reader, _ in self._sources}
        if len(token_policies) != 1:
            by_source = {reader.name: reader.meta.add_special_tokens for reader, _ in self._sources}
            raise ValueError(
                f"tier mixes caches with different add_special_tokens policies: {by_source}"
            )
        self.add_special_tokens = token_policies.pop()

    @property
    def sources(self) -> list[SourceReader]:
        return [r for (r, _) in self._sources]

    def total_rows(self) -> int:
        return sum(u for (_, u) in self._sources)

    def steps_per_epoch(self) -> int:
        """Steps this rank sees per epoch. With DDP (`world_size > 1`) we
        floor-divide so every rank does the same number of steps —
        matches the truncation in `epoch_iter` that prevents fast-finisher
        NCCL all_reduce timeouts."""
        bs = self.batch_size
        if self.drop_last_per_source:
            global_steps = sum(u // bs for (_, u) in self._sources)
        else:
            global_steps = sum((u + bs - 1) // bs for (_, u) in self._sources)
        return global_steps // self.world_size

    def _build_chunk_list(self) -> np.ndarray:
        """Return an (N, 2) int64 array of `(source_idx, window_start)`."""
        bs = self.batch_size
        rows: list[np.ndarray] = []
        for src_idx, (_, usable) in enumerate(self._sources):
            if self.drop_last_per_source:
                n_windows = usable // bs
                if n_windows == 0:
                    continue
                starts = np.arange(n_windows, dtype=np.int64) * bs
            else:
                n_windows = (usable + bs - 1) // bs
                if n_windows == 0:
                    continue
                starts = np.arange(n_windows, dtype=np.int64) * bs
            col_idx = np.full(starts.shape, src_idx, dtype=np.int64)
            rows.append(np.stack([col_idx, starts], axis=1))
        if not rows:
            return np.zeros((0, 2), dtype=np.int64)
        return np.concatenate(rows, axis=0)

    def epoch_iter(self, epoch: int) -> Iterator[Batch]:
        """One epoch's worth of batches, reshuffled per epoch.

        With DDP (`world_size > 1`), all ranks shuffle with the same seed
        (so the global chunk order is identical) and then each rank takes
        every `world_size`-th chunk. Disjoint slices, no all-gather, no
        ranks ever see the same `(source, row_start)` in the same epoch.

        After stride-slicing we truncate every rank to
        `global_chunks // world_size` batches. Without this, ranks with
        higher indices get one fewer chunk on the remainder, finish their
        loop early, stop calling the per-step `all_reduce` and DDP's
        gradient `all_reduce`, and the slower ranks hang on the next
        `all_reduce` until the NCCL watchdog kills the run at 600 s.
        Dropping up to `world_size - 1` chunks per epoch is invisible
        in practice (tens out of tens of thousands) and far cheaper
        than the alternative of `model.join()` plumbing.
        """
        chunks = self._build_chunk_list()
        rng = np.random.default_rng(self.seed + epoch)
        rng.shuffle(chunks)
        if self.world_size > 1:
            per_rank = chunks.shape[0] // self.world_size
            chunks = chunks[self.rank :: self.world_size][:per_rank]
        bs = self.batch_size

        for src_idx, start in chunks:
            src_idx = int(src_idx)
            start = int(start)
            reader, usable = self._sources[src_idx]
            n = min(bs, usable - start)
            if n <= 1:
                continue
            yield self._make_batch(reader, start, n)

    def __iter__(self) -> Iterator[Batch]:
        return self.epoch_iter(epoch=0)

    def _make_batch(self, reader: SourceReader, start: int, n: int) -> Batch:
        q_flat, q_off = reader.get_batch("query", start, n)
        d_flat, d_off = reader.get_batch("doc", start, n)
        # `nn.EmbeddingBag` wants int64 for both indices and offsets. The
        # u16→int64 copy is the main per-batch CPU cost; ~5 MB for a 4K batch
        # of ~300-token docs. Cheap.
        return Batch(
            query_input_ids=torch.from_numpy(q_flat.astype(np.int64, copy=False)),
            query_offsets=torch.from_numpy(q_off.astype(np.int64, copy=False)),
            doc_input_ids=torch.from_numpy(d_flat.astype(np.int64, copy=False)),
            doc_offsets=torch.from_numpy(d_off.astype(np.int64, copy=False)),
            source=reader.name,
        )
