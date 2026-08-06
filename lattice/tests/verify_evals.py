"""Re-run NanoBEIR on a handful of Rust-implementation artifacts and check
that the resulting mean NDCG@10 matches the corresponding cell in the
stage-2 quant sweep JSON. This gives an end-to-end confidence pass: not
just embedding-vector parity (already done) but downstream metric parity
on the real evaluation surface the blog cites.

Saves each run's per-task metrics to `data/<slug>/eval/nanobeir.json` so
the audit is reproducible.

Usage:
    .venv-py/bin/python tests/verify_evals.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers.sentence_transformer.evaluation import NanoBEIREvaluator

import lattice

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
QUANT_SWEEP = (
    REPO
    / "trainer"
    / "runs"
    / "tokenizer_stage2_10ep_20260803_modal_a100x4_r1"
    / "quant_sweep"
    / "quantization_sweep.json"
)


class LatticeST:
    """SentenceTransformer-shaped duck for the Rust kernel — exposes just
    what `NanoBEIREvaluator` actually calls on the model object.

    Each call goes through `lattice.Model.embed` (the same kernel that
    `bench` uses for throughput). No torch, no extra allocations beyond
    the per-call output ndarray.
    """

    def __init__(self, model_path: Path):
        self.rust_model = lattice.Model.load(str(model_path))
        tok_path = model_path.parent / "tokenizer.json"
        self.tokenizer = lattice.Tokenizer.load(str(tok_path))
        # NanoBEIREvaluator / IRE peek at a handful of attributes; provide
        # defaults that match what a SentenceTransformer would expose.
        self.similarity_fn_name = "cosine"
        self.truncate_dim = None
        self.prompts = {}
        self.default_prompt_name = None
        # `score_functions = {sim_name: model.similarity}` — IRE picks the
        # method off the model and calls it as `similarity(emb_q, emb_c)`.
        # We use sentence-transformers' own cosine helper so the math is
        # bit-equivalent to what the trainer's eval would compute.
        from sentence_transformers.util import cos_sim
        self.similarity = cos_sim
        # SentenceTransformer normally has a model_card_data attribute that
        # the evaluator pings via .set_evaluation_metrics() after the run.
        # We don't care about model cards here — a no-op stub keeps the
        # post-processing chain happy.
        class _NullCard:
            def set_evaluation_metrics(self, *a, **kw):
                pass
        self.model_card_data = _NullCard()

    @property
    def dim(self) -> int:
        return self.rust_model.dim

    def encode(
        self,
        sentences,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        convert_to_tensor=False,
        normalize_embeddings=False,
        **kwargs,
    ):
        if isinstance(sentences, str):
            sentences = [sentences]
        sentences = list(sentences)
        dim = self.rust_model.dim
        out = np.zeros((len(sentences), dim), dtype=np.float32)
        # Batch tokenize via HF's internal rayon — much faster than calling
        # encode() per sentence.
        all_tokens = self.tokenizer.encode_batch(sentences)
        for i, tokens in enumerate(all_tokens):
            out[i] = self.rust_model.embed(tokens, normalize=normalize_embeddings)
        if convert_to_tensor:
            import torch
            return torch.from_numpy(out)
        return out

    # Some evaluators check for these query/corpus variants. Proxy through.
    def encode_query(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def encode_corpus(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def encode_document(self, *args, **kwargs):
        return self.encode(*args, **kwargs)


def cell_in_sweep(sweep: list[dict], *, bits: int, axis, dim: int) -> float | None:
    """Find the matching cell in `quantization_sweep.json`. `axis` is None
    for fp32, otherwise 'dim' / 'row'."""
    for row in sweep:
        if row["bits"] == bits and row["axis"] == axis and row["dim"] == dim:
            return float(row["ndcg@10"])
    return None


def run_one(slug: str, *, bits: int, axis, dim: int, sweep: list[dict]) -> bool:
    model_path = DATA / slug / "model.safetensors"
    out_dir = DATA / slug / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "nanobeir.json"

    expected = cell_in_sweep(sweep, bits=bits, axis=axis, dim=dim)
    print(f"\n=== {slug} ===")
    print(f"  model:    {model_path}")
    print(f"  expected: {expected!r}  (from quant_sweep.json, cell bits={bits} axis={axis} dim={dim})")

    model = LatticeST(model_path)
    print(f"  variant: {model.rust_model.variant}  dim: {model.dim}")

    evaluator = NanoBEIREvaluator(truncate_dim=None, show_progress_bar=False)
    t0 = time.time()
    result = evaluator(model)
    elapsed = time.time() - t0

    # Filter to JSON-serializable scalars
    summary = {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
    measured = summary.get("NanoBEIR_mean_cosine_ndcg@10")
    out_file.write_text(json.dumps({str(dim): summary}, indent=2))

    delta = (measured - expected) if (measured is not None and expected is not None) else float("nan")
    abs_d = abs(delta) if delta == delta else float("nan")
    ok = abs_d < 0.001  # tolerance: 0.001 NDCG (i.e., 1 in 10k)
    marker = "OK " if ok else "MIS"
    print(f"  measured: {measured:.6f}  delta vs expected: {delta:+.6f}  elapsed: {elapsed:.1f}s  [{marker}]")
    return ok


def main() -> int:
    with open(QUANT_SWEEP) as f:
        sweep = json.load(f)

    # The 5 artifacts the user wants verified, with their (bits, axis, dim)
    # for matching against the quant_sweep cells.
    cases = [
        ("fp32-dim-1024",   {"bits": 32, "axis": None,  "dim": 1024}),
        ("int4-dim-1024",   {"bits": 4,  "axis": "dim", "dim": 1024}),
        ("int4-dim-512",    {"bits": 4,  "axis": "dim", "dim": 512}),
        ("int8-dim-256",    {"bits": 8,  "axis": "dim", "dim": 256}),
        ("int2-row-1024",   {"bits": 2,  "axis": "row", "dim": 1024}),
    ]

    n_ok = 0
    for slug, key in cases:
        if run_one(slug, sweep=sweep, **key):
            n_ok += 1

    print()
    print(f"{n_ok} / {len(cases)} variants match the trainer's quant sweep within 0.001 NDCG")
    return 0 if n_ok == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
