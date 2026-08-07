"""Export a training checkpoint as a standard Sentence Transformers model."""

from __future__ import annotations

from pathlib import Path


def export_hf_model(
    checkpoint_path: Path,
    tokenizer_path: Path,
    out_dir: Path,
) -> Path:
    """Write a Hub-ready Sentence Transformers directory.

    The trainer checkpoint intentionally contains only ``embedding.weight``.
    This function wraps that tensor in ``StaticEmbedding`` and lets Sentence
    Transformers write its normal module/config layout. The output directory
    must be new or empty so an old export cannot leak stale files into an
    upload.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out_dir}; use a fresh directory")

    from .eval import _build_sentence_transformer

    model = _build_sentence_transformer(checkpoint_path, tokenizer_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), create_model_card=False)

    print(f"exported Hugging Face model -> {out_dir}", flush=True)
    return out_dir
