"""Standalone analysis of the StaticEmbedding weight matrix to verify the
"silent rows" argument for why per-dim quantization fails at low bits.

The claim: under per-dim scaling, the scale for each output column `d` is
`max(|W[:, d]|) / qmax`. A nonzero token row `r` whose entries are all small —
specifically, where `|W[r, d]| < 0.5 * scale_d` for every `d` — gets
rounded to 0 across the board. The whole row is wiped out. Under
per-row scaling, every nonzero row has at least one ±qmax entry by
construction, so quantization cannot make an additional row silent.

Whether this matters depends on the *distribution* of per-row max-abs
values. If a few tokens have much louder rows than the median, the
per-dim scale gets dominated by those outliers and the quiet rows get
quantized to zero. We expect that's exactly what happens in BERT-style
vocab — common tokens vs rare tokens.

Run: `uv run python scripts/weight_variance.py <checkpoint.safetensors>`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from safetensors import safe_open


def load_weight(path: Path) -> np.ndarray:
    with safe_open(str(path), framework="np") as f:
        tensor_names = list(f.keys())
        if "embedding.weight" not in tensor_names:
            raise SystemExit(
                f"{path} has no 'embedding.weight' tensor; got {tensor_names}"
            )
        return f.get_tensor("embedding.weight")


def summarize(name: str, values: np.ndarray) -> None:
    """Print percentile-based summary of a 1-D positive array."""
    p = np.percentile(values, [0, 1, 50, 99, 100])
    print(
        f"  {name:>18}  min={p[0]:.5f}  p1={p[1]:.5f}  med={p[2]:.5f}  "
        f"p99={p[3]:.5f}  max={p[4]:.5f}  std={values.std():.5f}  "
        f"max/min={p[4] / max(p[0], 1e-12):.1f}×"
    )


def count_silent_rows(W: np.ndarray, bits: int) -> tuple[int, int]:
    """Count rows that quantize to all-zero under per-DIM symmetric int_n.

    Per-dim scale: `scale[d] = max(|W[:, d]|) / qmax`. A value rounds to 0
    iff its magnitude is < 0.5 * scale[d]. A row is "silent" iff every
    one of its entries rounds to 0.
    """
    qmax = (1 << (bits - 1)) - 1
    scale = np.abs(W).max(axis=0) / qmax  # shape (dim,)
    # Avoid div-by-zero on dead columns (shouldn't happen on a trained model).
    scale = np.clip(scale, 1e-12, None)
    # For each row, check if every entry's |value| < 0.5 * scale.
    threshold = 0.5 * scale[None, :]  # (1, dim) broadcasts over rows
    rounds_to_zero = np.abs(W) < threshold  # (vocab, dim) bool
    silent = rounds_to_zero.all(axis=1)  # (vocab,) bool
    return int(silent.sum()), int(W.shape[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", type=Path)
    args = ap.parse_args()

    W = load_weight(args.checkpoint)
    print(f"=== {args.checkpoint} ===")
    print(f"shape: {W.shape}  dtype: {W.dtype}")
    print(f"global max-abs: {np.abs(W).max():.5f}")
    print()

    abs_W = np.abs(W)
    row_max = abs_W.max(axis=1)  # (vocab,) — what per-row int_n scale uses
    col_max = abs_W.max(axis=0)  # (dim,)   — what per-dim int_n scale uses
    row_std = W.std(axis=1)
    col_std = W.std(axis=0)
    row_l2 = np.linalg.norm(W, axis=1)
    col_l2 = np.linalg.norm(W, axis=0)

    nonzero_rows = row_max > 0
    zero_row_ids = np.flatnonzero(~nonzero_rows)
    print(
        "=== Per-ROW stats (nonzero token rows,",
        int(nonzero_rows.sum()),
        "of",
        W.shape[0],
        ") ===",
    )
    print(f"  exactly-zero row ids: {zero_row_ids.tolist()}")
    summarize("max-abs", row_max[nonzero_rows])
    summarize("std", row_std[nonzero_rows])
    summarize("L2 norm", row_l2[nonzero_rows])
    print()

    print(
        "=== Per-DIM stats (one number per output column, length", W.shape[1], ") ==="
    )
    summarize("max-abs", col_max)
    summarize("std", col_std)
    summarize("L2 norm", col_l2)
    print()

    # The headline asymmetry: max-abs dynamic range.
    row_dyn = row_max[nonzero_rows].max() / row_max[nonzero_rows].min()
    col_dyn = col_max.max() / max(col_max.min(), 1e-12)
    print("=== Asymmetry (max-abs dynamic range) ===")
    print(f"  per-row max-abs range: {row_dyn:.1f}×")
    print(f"  per-dim max-abs range: {col_dyn:.1f}×")
    print(
        f"  ratio: {row_dyn / col_dyn:.1f}×  (per-row is this much wider than per-dim)"
    )
    print()

    print("=== Silent-row counts under per-dim quantization ===")
    print(f"  vocab size: {W.shape[0]}")
    for bits in (8, 4, 3, 2):
        n_silent, total = count_silent_rows(W, bits)
        pct = 100.0 * n_silent / total
        print(f"  int{bits}-dim: {n_silent:>5d} / {total} silent rows  ({pct:.2f}%)")
    print()
    print("  (per-row quantization creates 0 additional silent rows —")
    print("   each nonzero row's max-abs entry hits ±qmax exactly.)")


if __name__ == "__main__":
    main()
