"""Decontaminated BEIR evaluation for generated deployment variants."""

from __future__ import annotations

import gc
import json
from pathlib import Path

from .decontam_eval import evaluate_decontam_beir
from .quantize import QuantSpec, quantized_table

DEPLOYMENT_QUANT_GROUPS: dict[str, tuple[tuple[int, str, int], ...]] = {
    "int8_dim": tuple((8, "dim", dim) for dim in (1024, 512, 256, 128)),
    "int4_dim": tuple((4, "dim", dim) for dim in (1024, 512, 256, 128)),
    "row": (
        (8, "row", 256),
        (4, "row", 512),
        (2, "row", 1024),
        (2, "row", 512),
    ),
}


def deployment_slug(bits: int, axis: str, dim: int) -> str:
    return f"int{bits}-{axis}-{dim}"


def evaluate_deployment_quant_group(
    checkpoint_path: Path,
    tokenizer_path: Path,
    out_root: Path,
    group: str,
) -> dict[str, dict]:
    """Evaluate one four-cell deployment group, resuming at task boundaries."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import StaticEmbedding
    from tokenizers import Tokenizer

    from .train import load_checkpoint

    try:
        cells = DEPLOYMENT_QUANT_GROUPS[group]
    except KeyError as exc:
        raise ValueError(
            f"unknown deployment quant group {group!r}; "
            f"expected one of {sorted(DEPLOYMENT_QUANT_GROUPS)}"
        ) from exc

    full_weights = load_checkpoint(checkpoint_path)["embedding.weight"]
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    all_results: dict[str, dict] = {}

    for bits, axis, dim in cells:
        slug = deployment_slug(bits, axis, dim)
        eval_dir = out_root / slug / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        final_path = eval_dir / "decontam_beir.json"
        partial_path = eval_dir / "decontam_beir.partial.json"

        if final_path.is_file():
            saved = json.loads(final_path.read_text())
            if saved.get("config", {}).get("is_complete"):
                print(f"[{slug}] final result already complete; skipping", flush=True)
                all_results[slug] = saved
                continue

        print(f"[{slug}] building fake-quantized evaluation model", flush=True)
        spec = QuantSpec(bits=bits, axis=axis)
        quant_weights = quantized_table(full_weights, dim, spec)
        static = StaticEmbedding(tokenizer, embedding_weights=quant_weights)
        model = SentenceTransformer(modules=[static])

        results = evaluate_decontam_beir(
            model=model,
            matryoshka_dims=(dim,),
            progress_path=partial_path,
            show_progress=False,
            batch_size=512,
        )
        final_path.write_text(json.dumps(results, indent=2))
        all_results[slug] = results
        headline = results["aggregates"]["12_mean_ndcg@10"][str(dim)]
        print(f"[{slug}] complete: 12-mean ndcg@10={headline:.4f}", flush=True)

        del model, static, quant_weights
        gc.collect()

    return all_results
