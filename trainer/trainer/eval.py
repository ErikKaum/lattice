"""NanoBEIR evaluation. Plan.md primary metric: NanoBEIR mean NDCG@10
across 13 tasks, plus per-task breakdown and per-matryoshka-dim curve.

We round-trip through `sentence-transformers`' `SentenceTransformer` /
`StaticEmbedding` for evaluation. This is the canonical eval path the
reference model is benchmarked on, so it's the apples-to-apples comparison
we want against the 0.5032 baseline.

Eval is optional (it pulls a ~heavyweight stack). Only imported when the
`eval` command is invoked.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .loss import DEFAULT_MATRYOSHKA_DIMS


def _build_sentence_transformer(checkpoint_path: Path, tokenizer_path: Path):
    # Imports here so `trainer.eval` itself can be imported without
    # sentence-transformers/datasets installed. Use the post-3.x module paths
    # to dodge the deprecation warnings.
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import StaticEmbedding
    from tokenizers import Tokenizer

    from .train import load_checkpoint

    ckpt = load_checkpoint(checkpoint_path)
    weights: torch.Tensor = ckpt["embedding.weight"]

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    static = StaticEmbedding(tokenizer, embedding_weights=weights)
    return SentenceTransformer(modules=[static])


def evaluate(
    checkpoint_path: Path,
    tokenizer_path: Path,
    out_dir: Path,
    matryoshka_dims: tuple[int, ...] = DEFAULT_MATRYOSHKA_DIMS,
    dataset_names: tuple[str, ...] | None = None,
) -> dict:
    """Run NanoBEIR at every matryoshka dim. Returns the aggregated result
    dict and writes it to `out_dir / nanobeir.json`.

    `dataset_names=None` runs the full 13-task NanoBEIR per `plan.md`.
    Pass a subset (e.g., `('scifact',)`) to speed up smoke tests.
    """
    from sentence_transformers.sentence_transformer.evaluation import (
        NanoBEIREvaluator,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model = _build_sentence_transformer(checkpoint_path, tokenizer_path)

    kwargs: dict = {"show_progress_bar": False}
    if dataset_names is not None:
        kwargs["dataset_names"] = list(dataset_names)

    # Build the evaluator ONCE. `NanoBEIREvaluator.__init__` calls
    # `_load_dataset(...)` for each of the 13 tasks — pulling each task's
    # `corpus`/`queries`/`qrels` from the HF datasets cache. That's ~35 s
    # of pure I/O; doing it once per matryoshka dim (6 dims) would burn ~3
    # minutes on disk reads alone for a model that encodes in milliseconds.
    # Instead, we keep the loaded datasets and just rebind `truncate_dim`
    # on the outer evaluator and every inner per-task IR evaluator before
    # each call. Re-encoding still happens — but the static embedding
    # forward is microseconds per row, so encoding all 13 NanoBEIR corpora
    # 6 times costs a couple of seconds, not minutes.
    evaluator = NanoBEIREvaluator(truncate_dim=None, **kwargs)

    # Don't change `evaluator.name` between calls: the aggregator at
    # nano_beir.py:333 splits inner-evaluator result keys by
    # `self.name.count("_")` — adding a dim-suffix breaks key parsing.
    # The reported result keys are dim-agnostic; we attach the dim
    # ourselves in the per-dim JSON section.
    all_results: dict[str, dict] = {}
    for dim in matryoshka_dims:
        evaluator.truncate_dim = dim
        for inner in evaluator.evaluators:
            inner.truncate_dim = dim
        result = evaluator(model)
        all_results[str(dim)] = {k: float(v) for k, v in result.items()
                                 if isinstance(v, (int, float))}
        mean = result.get("NanoBEIR_mean_cosine_ndcg@10", float("nan"))
        print(f"dim={dim}  mean_ndcg@10={mean:.4f}", flush=True)

    (out_dir / "nanobeir.json").write_text(json.dumps(all_results, indent=2))
    return all_results


def evaluate_quantization_sweep(
    checkpoint_path: Path,
    tokenizer_path: Path,
    out_dir: Path,
    matryoshka_dims: tuple[int, ...] = DEFAULT_MATRYOSHKA_DIMS,
    bit_widths: tuple[int, ...] = (32, 8, 4, 3, 2),
    axes: tuple[str, ...] = ("row", "dim", "hybrid"),
) -> list[dict]:
    """Sweep (dim, bits, axis) and report NanoBEIR mean NDCG@10 per cell.

    Strategy: build the `NanoBEIREvaluator` once (~35 s of dataset I/O
    happens here only) and reuse its loaded per-task IR evaluators across
    cells. For each cell, slice the checkpoint to `dim` columns, fake-
    quantize at `bits` / `axis`, wrap in a fresh `StaticEmbedding` so the
    encoder sees the modified weight, and run the evaluator. Each cell
    is ~5 s of encoding work; the whole sweep finishes in a few minutes.

    fp32 cells skip the axis loop (axis is a no-op) so we don't double-
    report the same baseline.
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.evaluation import (
        NanoBEIREvaluator,
    )
    from sentence_transformers.sentence_transformer.modules import StaticEmbedding
    from tokenizers import Tokenizer

    from .quantize import QuantSpec, quantized_table
    from .train import load_checkpoint

    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(checkpoint_path)
    full_weights: torch.Tensor = ckpt["embedding.weight"]

    # Datasets load on construction — pay this ~35 s once.
    evaluator = NanoBEIREvaluator(truncate_dim=None, show_progress_bar=False)

    sweep_config = {
        "matryoshka_dims": list(matryoshka_dims),
        "bit_widths": list(bit_widths),
        "axes": list(axes),
    }
    partial_path = out_dir / "quantization_sweep.partial.json"
    rows = _load_quantization_sweep_progress(partial_path, sweep_config)
    completed = {
        (int(row["dim"]), int(row["bits"]), row.get("axis"))
        for row in rows
    }
    if rows:
        print(
            f"resuming quantization sweep with {len(rows)} completed cells",
            flush=True,
        )

    for dim in matryoshka_dims:
        for bits in bit_widths:
            cell_axes = (None,) if bits >= 32 else axes
            for axis in cell_axes:
                key = (dim, bits, axis)
                if key in completed:
                    print(
                        f"dim={dim:4d}  bits={bits:2d}  "
                        f"axis={axis or '-':>4}  already complete",
                        flush=True,
                    )
                    continue

                spec = QuantSpec(bits=bits, axis=axis or "dim")
                W_q = quantized_table(full_weights, dim, spec)

                tokenizer = Tokenizer.from_file(str(tokenizer_path))
                static = StaticEmbedding(tokenizer, embedding_weights=W_q)
                model = SentenceTransformer(modules=[static])

                evaluator.truncate_dim = None
                for inner in evaluator.evaluators:
                    inner.truncate_dim = None

                result = evaluator(model)
                mean = float(result.get("NanoBEIR_mean_cosine_ndcg@10", float("nan")))
                cell = {
                    "dim": dim,
                    "bits": bits,
                    "axis": (axis if bits < 32 else None),
                    "name": spec.name() if bits < 32 else "fp32",
                    "ndcg@10": mean,
                }
                rows.append(cell)
                completed.add(key)
                _write_json_atomic(
                    partial_path,
                    {"config": sweep_config, "rows": rows},
                )
                print(
                    f"dim={dim:4d}  bits={bits:2d}  axis={axis or '-':>4}  "
                    f"ndcg@10={mean:.4f}",
                    flush=True,
                )

    (out_dir / "quantization_sweep.json").write_text(json.dumps(rows, indent=2))
    return rows


def _load_quantization_sweep_progress(
    path: Path,
    expected_config: dict,
) -> list[dict]:
    """Load a compatible per-cell checkpoint, rejecting stale/malformed data."""
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ignoring unreadable quant sweep progress at {path}: {exc}", flush=True)
        return []
    if payload.get("config") != expected_config:
        print(f"ignoring incompatible quant sweep progress at {path}", flush=True)
        return []

    expected_keys = {
        (dim, bits, None if bits >= 32 else axis)
        for dim in expected_config["matryoshka_dims"]
        for bits in expected_config["bit_widths"]
        for axis in ([None] if bits >= 32 else expected_config["axes"])
    }
    rows = payload.get("rows")
    if not isinstance(rows, list):
        print(f"ignoring malformed quant sweep progress at {path}", flush=True)
        return []
    try:
        actual_keys = [
            (int(row["dim"]), int(row["bits"]), row.get("axis"))
            for row in rows
        ]
    except (KeyError, TypeError, ValueError):
        print(f"ignoring malformed quant sweep progress at {path}", flush=True)
        return []
    if len(set(actual_keys)) != len(actual_keys) or not set(actual_keys) <= expected_keys:
        print(f"ignoring invalid quant sweep cells at {path}", flush=True)
        return []
    return rows


def _write_json_atomic(path: Path, payload: object) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(path)
