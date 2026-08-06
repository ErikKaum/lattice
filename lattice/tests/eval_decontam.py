"""Run decontaminated BEIR through the Rust kernel via the `LatticeST`
adapter, for one or many artifacts under `data/`. Saves per-variant JSON
at `data/<slug>/eval/decontam_beir.json`.

Why: gives us a quality (NDCG@10) coordinate for the throughput-vs-quality
scatter plot. Decontam BEIR is the generalization surface the blog uses
(not NanoBEIR — which has stage-2 contamination on most tasks).

Usage:
    .venv-py/bin/python tests/eval_decontam.py <slug> [<slug> ...]
    .venv-py/bin/python tests/eval_decontam.py --all
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import lattice

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"

# Same 12-task LightOn-comparable surface the trainer uses for the headline.
ALL_TASKS = (
    "arguana", "climate-fever", "dbpedia", "fever", "fiqa", "hotpotqa",
    "msmarco", "nfcorpus", "nq", "quora", "scidocs", "scifact",
    "trec-covid", "webis-touche2020",
)
LIGHTON_12_EXCLUDE = ("climate-fever", "fever")  # matches trainer/decontam_eval


class LatticeST:
    """Same minimal SentenceTransformer-shaped adapter as in verify_evals.py.
    Encode is batched through the Rust kernel; we tokenize in HF's parallel
    encode_batch and then embed one-by-one (kernel itself is single-threaded
    here since we're driven by the evaluator's encode calls)."""

    def __init__(self, model_path: Path):
        self.rust_model = lattice.Model.load(str(model_path))
        self.tokenizer = lattice.Tokenizer.load(str(model_path.parent / "tokenizer.json"))
        self.similarity_fn_name = "cosine"
        self.truncate_dim = None
        self.prompts = {}
        self.default_prompt_name = None
        from sentence_transformers.util import cos_sim
        self.similarity = cos_sim
        class _NullCard:
            def set_evaluation_metrics(self, *a, **kw): pass
        self.model_card_data = _NullCard()

    @property
    def dim(self) -> int:
        return self.rust_model.dim

    def encode(self, sentences, batch_size=256, show_progress_bar=False,
               convert_to_numpy=True, convert_to_tensor=False,
               normalize_embeddings=False, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]
        sentences = list(sentences)
        dim = self.rust_model.dim
        out = np.zeros((len(sentences), dim), dtype=np.float32)
        all_tokens = self.tokenizer.encode_batch(sentences)
        for i, tokens in enumerate(all_tokens):
            out[i] = self.rust_model.embed(tokens, normalize=normalize_embeddings)
        if convert_to_tensor:
            import torch
            return torch.from_numpy(out)
        return out

    def encode_query(self, *a, **kw): return self.encode(*a, **kw)
    def encode_corpus(self, *a, **kw): return self.encode(*a, **kw)
    def encode_document(self, *a, **kw): return self.encode(*a, **kw)


def evaluate_one_variant(slug: str) -> dict:
    """Run decontam BEIR on one variant's Rust model. Eval at the model's
    NATIVE dim only (no matryoshka sweep — the artifact is already at that
    dim). Returns the structured result + saves JSON to
    `data/<slug>/eval/decontam_beir.json`."""
    # Lazy import of trainer code — needs trainer/ on sys.path
    sys.path.insert(0, str(REPO / "trainer"))
    from trainer.decontam_eval import evaluate_decontam_beir

    model_path = DATA / slug / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model = LatticeST(model_path)
    print(f"\n=== {slug} (variant={model.rust_model.variant}, dim={model.dim}) ===", flush=True)

    t0 = time.time()
    results = evaluate_decontam_beir(
        model=model,
        matryoshka_dims=(model.dim,),    # just the native dim
        tasks=ALL_TASKS,
        headline_exclude=LIGHTON_12_EXCLUDE,
        show_progress=False,
        batch_size=512,
    )
    elapsed = time.time() - t0

    headline_12 = results["aggregates"]["12_mean_ndcg@10"].get(str(model.dim))
    headline_14 = results["aggregates"]["14_mean_ndcg@10"].get(str(model.dim))
    print(f"  12-mean NDCG@10 = {headline_12:.4f}", flush=True)
    print(f"  14-mean NDCG@10 = {headline_14:.4f}", flush=True)
    print(f"  elapsed: {elapsed:.1f}s", flush=True)

    out_dir = DATA / slug / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "decontam_beir.json"
    out_file.write_text(json.dumps(results, indent=2))
    return results


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print("usage: eval_decontam.py <slug> [<slug> ...]")
        print("       eval_decontam.py --all")
        return 2

    if args == ["--all"]:
        slugs = [d.name for d in sorted(DATA.iterdir())
                if d.is_dir() and (d / "model.safetensors").exists()
                and any(d.name.startswith(p) for p in ("fp32-", "int"))]
    else:
        slugs = args

    print(f"Variants to evaluate: {len(slugs)}")
    for s in slugs:
        print(f"  - {s}")

    for s in slugs:
        try:
            evaluate_one_variant(s)
        except Exception as e:
            print(f"\n!!! {s}: {type(e).__name__}: {e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
