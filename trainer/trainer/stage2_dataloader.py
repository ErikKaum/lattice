"""Stage-2 mmap dataloader.

Reads the per-pair binaries from `stage2_tokenize.py` and yields batches
in the shape the stage-2 training loop wants:

    Batch:
      query_input_ids       (Σ q_tok,) int64
      query_offsets         (B,)       int64
      positive_input_ids    (Σ p_tok,) int64
      positive_offsets      (B,)       int64
      negative_input_ids    (Σ n_tok,) int64   ← K=7 sampled per pair
      negative_offsets      (B * K,)   int64
      source                str

Each batch is drawn from a *single* source (same as stage-1's source-
mixing discipline — keeps in-batch negatives realistic, avoids letting
the model exploit formatting/domain shifts as the discriminative signal).

For each pair, we sample K negatives at access time from the 50 stored,
with an RNG seeded per `(epoch, source, batch_start)` so the seven each
batch sees varies across epochs but stays reproducible.

DDP: same chunk-shard-then-truncate pattern as the stage-1 loader —
build a flat list of `(source_idx, pair_start)` batches, shuffle with a
seed shared across ranks, then `chunks[rank::world_size][:per_rank]` so
every rank sees the same number of steps. The truncation is the
load-bearing property that prevents the NCCL all-reduce timeout we hit
in the medium DDP launch.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch


def _negative_seed(base_seed: int, epoch: int, source: str, start: int) -> int:
    """Stable RNG seed for one batch's hard-negative sample.

    Python's built-in ``hash()`` is salted per process, so it cannot be used
    for a seed that must survive fresh interpreters, machines, or DDP
    launches. Encoding every seed component into SHA-256 gives us a stable
    uint64 while keeping adjacent batches independent.
    """
    payload = f"{base_seed}:{epoch}:{source}:{start}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class Stage2Batch(NamedTuple):
    query_input_ids: torch.Tensor
    query_offsets: torch.Tensor
    positive_input_ids: torch.Tensor
    positive_offsets: torch.Tensor
    negative_input_ids: torch.Tensor
    negative_offsets: torch.Tensor
    source: str


class Stage2Source:
    """mmap view of one source's stage-2 binaries."""

    def __init__(self, source_dir: Path) -> None:
        self.dir = source_dir
        self.meta = json.loads((source_dir / "meta.json").read_text())
        self.n_pairs = int(self.meta["n_pairs"])
        self.n_neg_per_pair = int(self.meta["n_negatives_per_pair"])
        self.source = str(self.meta["source"])
        add_special_tokens = self.meta.get("add_special_tokens", True)
        if not isinstance(add_special_tokens, bool):
            raise ValueError(
                f"{source_dir}: add_special_tokens must be a boolean, "
                f"got {add_special_tokens!r}"
            )
        # Stage-2 caches predating this field were all tokenized with
        # `add_special_tokens=True`.
        self.add_special_tokens = add_special_tokens

        self._qt = np.memmap(source_dir / "query_tokens.bin",     dtype="<u2", mode="r")
        self._qo = np.memmap(source_dir / "query_offsets.bin",    dtype="<u8", mode="r")
        self._pt = np.memmap(source_dir / "positive_tokens.bin",  dtype="<u2", mode="r")
        self._po = np.memmap(source_dir / "positive_offsets.bin", dtype="<u8", mode="r")
        self._nt = np.memmap(source_dir / "negative_tokens.bin",  dtype="<u2", mode="r")
        self._no = np.memmap(source_dir / "negative_offsets.bin", dtype="<u8", mode="r")

        # Sanity: file shapes match meta.
        if self._qo.shape[0] != self.n_pairs + 1:
            raise ValueError(
                f"{source_dir.name}: query_offsets has {self._qo.shape[0]} "
                f"entries, expected {self.n_pairs + 1}"
            )
        if self._no.shape[0] != self.n_pairs * self.n_neg_per_pair + 1:
            raise ValueError(
                f"{source_dir.name}: negative_offsets has {self._no.shape[0]} "
                f"entries, expected {self.n_pairs * self.n_neg_per_pair + 1}"
            )

    def pair_window(self, start: int, n: int) -> tuple[
        np.ndarray, np.ndarray,    # q flat tokens, q local offsets (n,)
        np.ndarray, np.ndarray,    # p flat tokens, p local offsets (n,)
    ]:
        """Slice queries + positives for a contiguous window of `n` pairs.
        Returns concatenated flat tokens and local offsets (rebased to 0)
        — exactly the shape `nn.EmbeddingBag` wants."""
        q_first = int(self._qo[start])
        q_last = int(self._qo[start + n])
        q_flat = self._qt[q_first:q_last]
        q_local = self._qo[start : start + n].astype(np.int64) - q_first

        p_first = int(self._po[start])
        p_last = int(self._po[start + n])
        p_flat = self._pt[p_first:p_last]
        p_local = self._po[start : start + n].astype(np.int64) - p_first

        return q_flat, q_local, p_flat, p_local

    def sample_negatives(
        self, start: int, n: int, k: int, rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """For each of `n` pairs starting at `start`, sample `k` of the
        50 stored negatives. Returns `(flat_tokens, local_offsets)` with
        `local_offsets` of length `n*k`. The samples are without
        replacement within a pair; different pairs get independent draws.
        """
        N = self.n_neg_per_pair
        if k > N:
            raise ValueError(f"k={k} > stored n_negatives={N}")

        # Build the list of GLOBAL negative-record indices we want.
        # For pair p (offset by `start`), record j of N → global = (start+p)*N + j.
        # Vectorize with one rng.permutation per pair, take the first k.
        pair_indices = np.arange(n, dtype=np.int64)
        # rng.permuted over axis=1 of an (n, N) tile is the cleanest vectorized form.
        perms = rng.permuted(
            np.broadcast_to(np.arange(N, dtype=np.int64), (n, N)).copy(),
            axis=1,
        )
        chosen_local = perms[:, :k]  # (n, k) values in [0, N)
        global_recs = ((start + pair_indices)[:, None] * N + chosen_local).ravel()

        # `_no` is the offsets file with `n_pairs*N + 1` entries. Look up
        # start/end byte offsets per chosen global record.
        starts = self._no[global_recs].astype(np.int64)
        ends = self._no[global_recs + 1].astype(np.int64)
        lengths = ends - starts

        # Build local flat array. Concatenate row-by-row (vectorized via
        # an arange of in-flat positions).
        total = int(lengths.sum())
        flat = np.empty(total, dtype=np.uint16)
        pos = 0
        for s, e in zip(starts.tolist(), ends.tolist()):
            ln = e - s
            flat[pos : pos + ln] = self._nt[s:e]
            pos += ln

        local_offsets = np.empty(n * k, dtype=np.int64)
        local_offsets[0] = 0
        # cumsum-then-shift trick to get starting offsets.
        if n * k > 1:
            local_offsets[1:] = np.cumsum(lengths[:-1])
        return flat, local_offsets


class Stage2Dataloader:
    def __init__(
        self,
        training_root: Path,
        batch_size: int,
        n_neg_sample: int = 7,
        seed: int = 42,
        drop_last_per_source: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if batch_size <= 1:
            raise ValueError("batch_size > 1 required (MNR loss needs negatives)")
        if not (0 <= rank < world_size):
            raise ValueError(f"bad rank/world_size: {rank}/{world_size}")
        self.batch_size = batch_size
        self.n_neg_sample = n_neg_sample
        self.seed = seed
        self.drop_last_per_source = drop_last_per_source
        self.rank = rank
        self.world_size = world_size

        source_dirs = sorted(
            d for d in training_root.iterdir()
            if d.is_dir() and (d / "meta.json").exists()
        )
        if not source_dirs:
            raise ValueError(f"no tokenized sources under {training_root}")
        self.sources: list[Stage2Source] = [Stage2Source(d) for d in source_dirs]
        token_policies = {source.add_special_tokens for source in self.sources}
        if len(token_policies) != 1:
            by_source = {
                source.source: source.add_special_tokens
                for source in self.sources
            }
            raise ValueError(
                "stage-2 training data mixes add_special_tokens policies: "
                f"{by_source}"
            )
        self.add_special_tokens = token_policies.pop()

    def steps_per_epoch(self) -> int:
        bs = self.batch_size
        if self.drop_last_per_source:
            global_steps = sum(s.n_pairs // bs for s in self.sources)
        else:
            global_steps = sum((s.n_pairs + bs - 1) // bs for s in self.sources)
        return global_steps // self.world_size

    def _build_chunk_list(self) -> np.ndarray:
        """`(N, 2)` int64 array of `(source_idx, pair_start)`."""
        bs = self.batch_size
        rows: list[np.ndarray] = []
        for idx, src in enumerate(self.sources):
            n = src.n_pairs // bs if self.drop_last_per_source else (src.n_pairs + bs - 1) // bs
            if n == 0:
                continue
            starts = np.arange(n, dtype=np.int64) * bs
            col = np.full(starts.shape, idx, dtype=np.int64)
            rows.append(np.stack([col, starts], axis=1))
        if not rows:
            return np.zeros((0, 2), dtype=np.int64)
        return np.concatenate(rows, axis=0)

    def epoch_iter(self, epoch: int) -> Iterator[Stage2Batch]:
        chunks = self._build_chunk_list()
        rng_shuffle = np.random.default_rng(self.seed + epoch)
        rng_shuffle.shuffle(chunks)
        if self.world_size > 1:
            per_rank = chunks.shape[0] // self.world_size
            chunks = chunks[self.rank :: self.world_size][:per_rank]
        bs = self.batch_size

        for src_idx, start in chunks:
            src_idx = int(src_idx)
            start = int(start)
            src = self.sources[src_idx]
            n = min(bs, src.n_pairs - start)
            if n <= 1:
                continue
            yield self._make_batch(src, start, n, epoch)

    def _make_batch(
        self, src: Stage2Source, start: int, n: int, epoch: int,
    ) -> Stage2Batch:
        # Independent RNG per (epoch, source, start) so negative sampling is
        # reproducible and different each epoch.
        neg_rng = np.random.default_rng(
            _negative_seed(self.seed, epoch, src.source, start)
        )
        q_flat, q_off, p_flat, p_off = src.pair_window(start, n)
        n_flat, n_off = src.sample_negatives(start, n, self.n_neg_sample, neg_rng)

        return Stage2Batch(
            query_input_ids=torch.from_numpy(q_flat.astype(np.int64, copy=False)),
            query_offsets=torch.from_numpy(q_off),
            positive_input_ids=torch.from_numpy(p_flat.astype(np.int64, copy=False)),
            positive_offsets=torch.from_numpy(p_off),
            negative_input_ids=torch.from_numpy(n_flat.astype(np.int64, copy=False)),
            negative_offsets=torch.from_numpy(n_off),
            source=src.source,
        )
