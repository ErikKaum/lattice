"""Stage-2 train/eval split builder (eval_plan.md Part 1, step 1).

Splits `lightonai/embeddings-fine-tuning` into a stage-2 training set and a
held-out in-domain eval set.

Design contract (eval_plan.md):

- **Query-level**: a query and all its positives go entirely to train *or*
  eval, never both. Splitting at the pair level would leak the same query
  across both halves. (Each query in this dataset has ~2.6 positives on
  average; FiQA has 14,166 pairs over 5,500 queries.)
- **Per-source stratification**: 20% held out from *each* of the 7 sources
  independently. A global 20% sample would let MSMARCO (502K queries)
  swamp the eval and could leave FiQA-equivalent sources sparsely
  measured. Stratify with an independent RNG per source.
- **Cap at `eval_cap` queries per source** (default 100; plan2 §1 "50-100
  per source"). If 20% exceeds the cap, downsample to the cap; if it's
  under, keep all of them. This makes the eval Nano-fast by construction.
- **Reproducibility**: fixed top-level seed, deterministic per-source RNG
  derived from `(seed, source)`. Split manifests written to disk include
  the seed they were produced with — refuse to overwrite without an
  explicit `--force` so we can't silently rebuild with a different seed
  and invalidate every existing stage-2 comparison.

Output layout:

    <out_root>/stage2/splits/<source>.json
        {"source": "fiqa",
         "seed": 42, "eval_fraction": 0.2, "eval_cap": 100,
         "n_total_queries": 5500,
         "n_train_queries": 5400,
         "n_eval_queries": 100,
         "train_query_ids": [...], "eval_query_ids": [...]}

    <out_root>/stage2/splits/manifest.json
        Top-level summary: per-source counts, seed, version stamp.

Both files are pure JSON so they can be git-committed if we want them
versioned.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DATASET = "lightonai/embeddings-fine-tuning"
SOURCES: tuple[str, ...] = ("fiqa", "nq", "hotpotqa", "msmarco", "fever", "squadv2", "trivia")

DEFAULT_SEED = 42
DEFAULT_EVAL_FRACTION = 0.20
DEFAULT_EVAL_CAP = 100  # plan2 §1: "50-100 per source"; pick upper end so
                        # calibration in Part 2 has more discriminative
                        # headroom — easy to tighten if it survives.


@dataclass(frozen=True)
class SplitManifest:
    source: str
    seed: int
    eval_fraction: float
    eval_cap: int
    n_total_queries: int
    n_train_queries: int
    n_eval_queries: int
    train_query_ids: list[int]
    eval_query_ids: list[int]


def _per_source_seed(base_seed: int, source: str) -> int:
    """Derive a per-source RNG seed deterministically from `(base_seed,
    source)`. Using `hashlib.sha256` instead of Python's `hash()` because
    `hash()` is salted per-process by default and would produce different
    seeds on different runs — defeating the entire point of a fixed seed.
    """
    h = hashlib.sha256(f"{base_seed}:{source}".encode()).digest()
    # First 8 bytes → uint64; mask to numpy's accepted 32-bit range.
    return int.from_bytes(h[:8], "big") & 0xFFFF_FFFF


def split_source(
    source: str,
    seed: int = DEFAULT_SEED,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    eval_cap: int = DEFAULT_EVAL_CAP,
) -> SplitManifest:
    """Produce a train/eval query-id split for one source."""
    from datasets import load_dataset

    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")

    queries_ds = load_dataset(DATASET, "queries", split=source)
    # Sort first so the RNG is the *only* source of order: any HF
    # cache/version that changes iteration order can't shift the split.
    qids = sorted(int(q) for q in queries_ds["query_id"])

    rng = np.random.default_rng(_per_source_seed(seed, source))
    perm = rng.permutation(len(qids))
    shuffled = [qids[i] for i in perm]

    n_total = len(shuffled)
    n_eval_target = int(round(n_total * eval_fraction))
    n_eval = min(n_eval_target, eval_cap)

    eval_qids = sorted(shuffled[:n_eval])
    train_qids = sorted(shuffled[n_eval:])

    return SplitManifest(
        source=source,
        seed=seed,
        eval_fraction=eval_fraction,
        eval_cap=eval_cap,
        n_total_queries=n_total,
        n_train_queries=len(train_qids),
        n_eval_queries=len(eval_qids),
        train_query_ids=train_qids,
        eval_query_ids=eval_qids,
    )


def write_split(manifest: SplitManifest, out_dir: Path, force: bool = False) -> Path:
    path = out_dir / f"{manifest.source}.json"
    if path.exists() and not force:
        existing = json.loads(path.read_text())
        if existing.get("seed") != manifest.seed:
            raise FileExistsError(
                f"{path} exists with seed={existing.get('seed')!r}, new seed={manifest.seed}. "
                "Refusing to overwrite — rebuilding with a different seed invalidates every "
                "existing stage-2 comparison. Pass force=True if this is intentional."
            )
        # Same seed → idempotent overwrite is fine.
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(manifest), indent=2))
    return path


def make_all_splits(
    out_dir: Path,
    sources: tuple[str, ...] = SOURCES,
    seed: int = DEFAULT_SEED,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    eval_cap: int = DEFAULT_EVAL_CAP,
    force: bool = False,
) -> dict:
    """Build splits for every source and write a top-level manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)

    per_source: dict[str, dict] = {}
    total_eval = 0
    total_train = 0
    for source in sources:
        m = split_source(source, seed=seed, eval_fraction=eval_fraction, eval_cap=eval_cap)
        write_split(m, out_dir, force=force)
        per_source[source] = {
            "n_total": m.n_total_queries,
            "n_train": m.n_train_queries,
            "n_eval": m.n_eval_queries,
            "eval_fraction_realized": m.n_eval_queries / m.n_total_queries,
        }
        total_eval += m.n_eval_queries
        total_train += m.n_train_queries
        print(
            f"  {source:>10s}: total={m.n_total_queries:>7d}  "
            f"train={m.n_train_queries:>7d}  eval={m.n_eval_queries:>4d} "
            f"({100 * m.n_eval_queries / m.n_total_queries:.2f}%)",
            flush=True,
        )

    summary = {
        "seed": seed,
        "eval_fraction": eval_fraction,
        "eval_cap": eval_cap,
        "sources": list(sources),
        "totals": {"train": total_train, "eval": total_eval},
        "per_source": per_source,
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(f"\ntotal: train={total_train}  eval={total_eval}", flush=True)
    return summary


def load_split(splits_dir: Path, source: str) -> SplitManifest:
    """Load a previously-written split manifest. Used by the eval-surface
    builder and the tokenize step to enforce eval/train disjointness."""
    data = json.loads((splits_dir / f"{source}.json").read_text())
    return SplitManifest(**data)
