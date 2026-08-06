"""mmap reader for the per-source binary layout produced by `pipeline tokenize`.

Layout under `cache/subsets/<source>/`:
    query_tokens.bin   — concatenated u16 little-endian token IDs
    query_offsets.bin  — u64 little-endian, length N+1, with sentinel
    doc_tokens.bin     — same for documents
    doc_offsets.bin    — same
    meta.json          — row counts, dtype tags

Top-level:
    cache/partition.json  — global schedule with per-source take_rows (full)
    cache/tiers.json      — per-tier per-source take_rows

The reader is mmap-only: opening a source pays for the file headers, not the
data. Random access to row `i` is O(1) — two offset reads and one slice into
the tokens mmap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


Side = Literal["query", "doc"]
SIDES: tuple[Side, ...] = ("query", "doc")


@dataclass(frozen=True)
class SourceMeta:
    source: str
    n_rows: int
    total_query_tokens: int
    total_doc_tokens: int
    tokenizer: str
    tokens_dtype: str
    offsets_dtype: str
    add_special_tokens: bool

    @classmethod
    def load(cls, path: Path) -> "SourceMeta":
        data = json.loads(path.read_text())
        add_special_tokens = data.get("add_special_tokens", True)
        if not isinstance(add_special_tokens, bool):
            raise ValueError(
                f"{path}: add_special_tokens must be a boolean, "
                f"got {add_special_tokens!r}"
            )
        return cls(
            source=data["source"],
            n_rows=int(data["n_rows"]),
            total_query_tokens=int(data["total_query_tokens"]),
            total_doc_tokens=int(data["total_doc_tokens"]),
            tokenizer=data["tokenizer"],
            tokens_dtype=data["tokens_dtype"],
            offsets_dtype=data["offsets_dtype"],
            # Caches produced before this field was introduced always used
            # `add_special_tokens=True`.
            add_special_tokens=add_special_tokens,
        )


class SourceReader:
    """mmap-backed view of one tokenized source. Cheap to construct; lazy on
    actual I/O via the OS page cache."""

    def __init__(self, source_dir: Path) -> None:
        self.dir = source_dir
        self.meta = SourceMeta.load(source_dir / "meta.json")
        if self.meta.tokens_dtype != "u16" or self.meta.offsets_dtype != "u64":
            raise ValueError(
                f"unexpected dtypes in {source_dir}: tokens={self.meta.tokens_dtype} "
                f"offsets={self.meta.offsets_dtype} (only u16/u64 supported)"
            )
        self._tokens: dict[Side, np.memmap] = {}
        self._offsets: dict[Side, np.memmap] = {}
        for side in SIDES:
            self._tokens[side] = np.memmap(
                source_dir / f"{side}_tokens.bin", dtype="<u2", mode="r"
            )
            self._offsets[side] = np.memmap(
                source_dir / f"{side}_offsets.bin", dtype="<u8", mode="r"
            )
            # Sanity: files must hold AT LEAST `n_rows + 1` offset entries and
            # AT LEAST `sentinel(@ row n_rows)` token entries. Bigger is fine
            # — it means a larger tier is currently being tokenized in the
            # background, OR a previous tokenize was killed. The trainer's
            # `open_tier_sources` enforces the load-bearing per-tier check.
            n_offsets = self._offsets[side].shape[0]
            if n_offsets < self.meta.n_rows + 1:
                raise ValueError(
                    f"{side}_offsets.bin has {n_offsets} entries; "
                    f"expected at least {self.meta.n_rows + 1} (rows + sentinel)"
                )
            sentinel = int(self._offsets[side][self.meta.n_rows])
            if self._tokens[side].shape[0] < sentinel:
                raise ValueError(
                    f"{side}_tokens.bin has {self._tokens[side].shape[0]} entries; "
                    f"sentinel for row {self.meta.n_rows} says ≥ {sentinel}"
                )

    @property
    def name(self) -> str:
        return self.meta.source

    @property
    def n_rows(self) -> int:
        return self.meta.n_rows

    def get_row(self, side: Side, i: int) -> np.ndarray:
        """Token IDs for row `i` of `side` as a (possibly zero-length) u16 array."""
        offsets = self._offsets[side]
        start = int(offsets[i])
        end = int(offsets[i + 1])
        return self._tokens[side][start:end]

    def get_batch(
        self, side: Side, start: int, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return `(tokens_flat, local_offsets)` for rows `[start, start+n)`.

        `tokens_flat` is the concatenated token IDs (u16) for all rows in the
        batch — already contiguous in the file, returned as a slice without
        copying. `local_offsets` is a length-`n` u64 array of starting offsets
        rebased to 0 — exactly what `torch.nn.EmbeddingBag` wants.
        """
        if start < 0 or start + n > self.meta.n_rows:
            raise IndexError(
                f"batch [{start}, {start + n}) out of range for source "
                f"{self.meta.source} (n_rows={self.meta.n_rows})"
            )
        offsets = self._offsets[side]
        first = int(offsets[start])
        last = int(offsets[start + n])
        tokens_flat = self._tokens[side][first:last]
        local_offsets = offsets[start : start + n].astype(np.int64) - first
        return tokens_flat, local_offsets


@dataclass(frozen=True)
class TierSource:
    source: str
    take_rows: int


@dataclass(frozen=True)
class Tier:
    name: str
    target_rows: int
    actual_rows: int
    per_source: tuple[TierSource, ...]


def load_tiers(cache_root: Path) -> dict[str, Tier]:
    raw = json.loads((cache_root / "tiers.json").read_text())
    out: dict[str, Tier] = {}
    for t in raw:
        per_source = tuple(
            TierSource(source=s["source"], take_rows=int(s["take_rows"]))
            for s in t["per_source"]
        )
        out[t["tier"]] = Tier(
            name=t["tier"],
            target_rows=int(t["target_rows"]),
            actual_rows=int(t["actual_rows"]),
            per_source=per_source,
        )
    return out


def source_dir(cache_root: Path, source: str) -> Path:
    return cache_root / "subsets" / source


def open_tier_sources(
    cache_root: Path, tier: Tier
) -> list[tuple[SourceReader, int]]:
    """Open every source mentioned in `tier` with `take_rows > 0`. Each returned
    tuple is `(reader, usable_rows)` — the tier's prefix length within that
    source. The reader may carry more rows on disk (if a larger tier was
    tokenized later) — `usable_rows` is the cap the dataloader should respect.
    """
    out: list[tuple[SourceReader, int]] = []
    for ts in tier.per_source:
        if ts.take_rows == 0:
            continue
        d = source_dir(cache_root, ts.source)
        if not (d / "meta.json").exists():
            raise FileNotFoundError(
                f"source {ts.source} required by tier {tier.name} not tokenized "
                f"(no {d / 'meta.json'}). Run `pipeline tokenize {tier.name}`."
            )
        reader = SourceReader(d)
        usable = min(ts.take_rows, reader.n_rows)
        if usable < ts.take_rows:
            raise RuntimeError(
                f"source {ts.source} has only {reader.n_rows} rows on disk; "
                f"tier {tier.name} expects {ts.take_rows}"
            )
        out.append((reader, usable))
    return out
