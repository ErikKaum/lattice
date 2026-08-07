"""Stage tokenized source binaries from a (possibly bucket-mounted) cache to
local scratch storage before training.

Why this exists: on HF Jobs we mount the data bucket at `/app/cache` via FUSE.
Sparse mmap reads of multi-GB `*_tokens.bin` from a bucket-mounted FUSE
serialize catastrophically under concurrent access — an earlier pipeline
wedged a bucket for an hour with three concurrent doc-side shards doing
random reads. The fix is the same here: copy the binaries once to local
disk at startup, then mmap from there.

Concurrent-safe across DDP ranks sharing one scratch dir. Uses
`fcntl.flock` + a per-source `.staged` sentinel. Only the winner of the
lock race pulls bytes from the bucket; the other ranks find the sentinel
and reuse the staged copy. This matches the previously validated pattern.

The bucket is still the canonical home for checkpoints — those are small
(~125 MB each) and infrequent, and aligned writes to bucket FUSE are
well-behaved (~300 MB/s in earlier measurements). So **reads stage,
writes don't**.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import time
from pathlib import Path

from .io import Tier

log = logging.getLogger(__name__)


_TOP_LEVEL_FILES = ("tiers.json", "tokenizer.json", "partition.json")
"""Small files at `cache_root/`. Copy unconditionally — they're a few hundred
KB total and the trainer + eval both consume them."""

_SIDES = ("query", "doc")
_TOKEN_BYTES = 2  # u16
_OFFSET_BYTES = 8  # u64
_COPY_BUFFER_BYTES = 8 << 20


def stage_in_tier(cache_root: Path, scratch_root: Path, tier: Tier) -> Path:
    """Copy every source binary needed by `tier` from `cache_root` to
    `scratch_root`. Returns the staged root (== `scratch_root`).

    Each source is copied as an atomic group: lock → check sentinel → copy
    → write sentinel. Sources are independent so concurrent ranks may stage
    *different* sources in parallel; collisions on the same source are
    serialized by `flock`.

    Top-level small files (`tiers.json`, `tokenizer.json`, `partition.json`)
    are also copied — under the same lock-per-file discipline.
    """
    scratch_root.mkdir(parents=True, exist_ok=True)

    for name in _TOP_LEVEL_FILES:
        src = cache_root / name
        if src.exists():
            _stage_file(src, scratch_root / name)

    for ts in tier.per_source:
        if ts.take_rows == 0:
            continue
        src = cache_root / "subsets" / ts.source
        dst = scratch_root / "subsets" / ts.source
        _stage_source_prefix(src, dst, ts.take_rows)

    return scratch_root


def _stage_source_prefix(src: Path, dst: Path, take_rows: int) -> None:
    """Stage exactly the prefix of a source required by one tier.

    A source directory can contain the full 660M-row corpus even when the
    selected tier is ``xs``. The row offsets make the byte boundary for a
    tier prefix explicit, so copy only ``take_rows + 1`` offsets and the
    token bytes ending at that final sentinel. Rewrite ``meta.json`` to
    describe the staged prefix so :class:`SourceReader` validates it.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    sentinel = dst.with_name(dst.name + ".staged")
    lock_path = dst.with_name(dst.name + ".lock")

    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)

        # Keep every bucket-backed read behind the lock. In particular, the
        # prefix boundary requires sparse reads from the offsets files; doing
        # those on every DDP rank recreates the FUSE contention that staging
        # is meant to avoid.
        meta_path = src / "meta.json"
        if not src.exists():
            raise FileNotFoundError(f"missing input: {src}")
        if not meta_path.exists():
            raise FileNotFoundError(f"missing input: {meta_path}")
        source_meta = json.loads(meta_path.read_text())
        source_rows = int(source_meta["n_rows"])
        if take_rows <= 0 or take_rows > source_rows:
            raise ValueError(f"cannot stage {take_rows} rows from {src} (source has {source_rows})")
        if source_meta.get("tokens_dtype") != "u16":
            raise ValueError(f"unexpected tokens dtype in {meta_path}")
        if source_meta.get("offsets_dtype") != "u64":
            raise ValueError(f"unexpected offsets dtype in {meta_path}")

        requested_marker = {
            "take_rows": take_rows,
            "source_rows": source_rows,
            "add_special_tokens": source_meta.get("add_special_tokens", True),
        }
        if dst.exists() and _marker_contains(sentinel, requested_marker):
            log.info("stage-in: %s already staged at %d rows, reusing", dst, take_rows)
            return

        offset_bytes = (take_rows + 1) * _OFFSET_BYTES
        token_counts: dict[str, int] = {}
        for side in _SIDES:
            offsets_path = src / f"{side}_offsets.bin"
            tokens_path = src / f"{side}_tokens.bin"
            token_counts[side] = _read_u64(offsets_path, take_rows)
            _require_size(offsets_path, offset_bytes)
            _require_size(tokens_path, token_counts[side] * _TOKEN_BYTES)

        stage_marker = {
            **requested_marker,
            "query_tokens": token_counts["query"],
            "doc_tokens": token_counts["doc"],
        }

        tmp = dst.with_name(dst.name + ".staging")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)

        size_bytes = offset_bytes * len(_SIDES) + sum(
            count * _TOKEN_BYTES for count in token_counts.values()
        )
        print(
            f"stage-in source: {src.name}  rows={take_rows:,}  bytes={size_bytes:,}",
            flush=True,
        )
        t0 = time.monotonic()
        try:
            for side in _SIDES:
                _copy_prefix(
                    src / f"{side}_offsets.bin",
                    tmp / f"{side}_offsets.bin",
                    offset_bytes,
                )
                _copy_prefix(
                    src / f"{side}_tokens.bin",
                    tmp / f"{side}_tokens.bin",
                    token_counts[side] * _TOKEN_BYTES,
                )

            staged_meta = dict(source_meta)
            staged_meta["n_rows"] = take_rows
            staged_meta["total_query_tokens"] = token_counts["query"]
            staged_meta["total_doc_tokens"] = token_counts["doc"]
            (tmp / "meta.json").write_text(json.dumps(staged_meta, indent=2, sort_keys=True) + "\n")

            if dst.exists():
                shutil.rmtree(dst)
            os.replace(tmp, dst)
            sentinel.write_text(json.dumps(stage_marker, sort_keys=True) + "\n")
        except BaseException:
            if tmp.exists():
                shutil.rmtree(tmp)
            raise

        elapsed = max(time.monotonic() - t0, 1e-3)
        print(
            f"stage-in source done: {src.name}  seconds={elapsed:.1f}  "
            f"MB/s={size_bytes / 1e6 / elapsed:.1f}",
            flush=True,
        )


def _read_u64(path: Path, index: int) -> int:
    """Read one little-endian u64 without mmap'ing a bucket-backed file."""
    with open(path, "rb") as f:
        f.seek(index * _OFFSET_BYTES)
        raw = f.read(_OFFSET_BYTES)
    if len(raw) != _OFFSET_BYTES:
        raise ValueError(f"{path} has no u64 entry at index {index}")
    return int.from_bytes(raw, byteorder="little", signed=False)


def _require_size(path: Path, minimum_bytes: int) -> None:
    size = path.stat().st_size
    if size < minimum_bytes:
        raise ValueError(f"{path} has {size} bytes; expected at least {minimum_bytes}")


def _copy_prefix(src: Path, dst: Path, n_bytes: int) -> None:
    """Sequentially copy exactly ``n_bytes`` from ``src`` to ``dst``."""
    remaining = n_bytes
    with open(src, "rb") as source, open(dst, "wb") as target:
        while remaining:
            chunk = source.read(min(remaining, _COPY_BUFFER_BYTES))
            if not chunk:
                raise ValueError(f"{src} ended with {remaining} bytes left to copy")
            target.write(chunk)
            remaining -= len(chunk)


def _marker_contains(path: Path, expected: dict[str, object]) -> bool:
    """Return whether every requested field matches a completed marker."""
    if not path.exists():
        return False
    try:
        actual = json.loads(path.read_text())
        return all(actual.get(key) == value for key, value in expected.items())
    except (OSError, ValueError, TypeError):
        return False


def _stage_dir(src: Path, dst: Path) -> None:
    """Copy `src` → `dst` once, under a per-destination flock + sentinel.
    Concurrent callers serialize on the lock; only the first does I/O."""
    if not src.exists():
        raise FileNotFoundError(f"missing input: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    sentinel = dst.with_name(dst.name + ".staged")
    lock_path = dst.with_name(dst.name + ".lock")

    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        if sentinel.exists():
            log.info("stage-in: %s already staged, reusing", dst)
            return
        if dst.exists():
            shutil.rmtree(dst)
        size_bytes = _dir_size(src)
        log.info("staging %s -> %s (%.1f MB)", src, dst, size_bytes / 1e6)
        t0 = time.monotonic()
        shutil.copytree(src, dst)
        elapsed = max(time.monotonic() - t0, 1e-3)
        log.info(
            "  done in %.1fs (%.1f MB/s)",
            elapsed,
            size_bytes / 1e6 / elapsed,
        )
        sentinel.touch()


def _stage_file(src: Path, dst: Path) -> None:
    """Same as `_stage_dir` but for a single file."""
    if not src.exists():
        raise FileNotFoundError(f"missing input: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    sentinel = dst.with_name(dst.name + ".staged")
    lock_path = dst.with_name(dst.name + ".lock")

    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        if sentinel.exists():
            return
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)
        sentinel.touch()


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def stage_in_stage2_training(src_training_root: Path, dst_training_root: Path) -> Path:
    """Stage the full `stage2/training/` tree (all 7 sources) from cache
    to local scratch. Same `flock` + `.staged` sentinel discipline as the
    stage-1 tier staging — concurrent DDP ranks share one scratch dir
    and only the lock winner per source actually copies bytes. Returns
    the staged root."""
    dst_training_root.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_training_root.iterdir()):
        if src.is_dir() and (src / "meta.json").exists():
            _stage_dir(src, dst_training_root / src.name)
    return dst_training_root


def stage_in_file(src: Path, dst_dir: Path) -> Path:
    """Stage one file (e.g. the init `.safetensors`) under `flock` so
    multi-rank reads converge on a local copy. Returns the local path."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    _stage_file(src, dst)
    return dst
