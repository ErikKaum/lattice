"""Parity test: Rust kernel output ≈ sentence-transformers `StaticEmbedding`.

Strategy
--------
For each (variant, dim) we want to verify, we load the same model two ways:

1. **Rust side**: load the slicer artifact (the `.safetensors` for that
   variant) through the `lattice` Python bindings. The Rust kernel does
   the dequant + mean-pool inline.

2. **Python side**: load fake-quantized weights into sentence-transformers'
   `StaticEmbedding`. For `fp32` the weights are unmodified. For
   quantized variants we call slicer's own `quantize_and_slice` followed
   by `dequantize` to materialize the same weight values the Rust kernel
   would produce — i.e. *bit-equal fake quant*. ST then runs its
   `EmbeddingBag(mean)` over those fp32 weights.

If both implementations are correct, embeddings should match to within
float rounding — `np.allclose(rust, st, atol=1e-5, rtol=1e-4)` is the
gate we use.

This single mechanism validates the full inference stack per variant:
- Tokenization (we share the HF `tokenizers` crate, but tokens flow
  through different code paths in Rust vs Python).
- The bit-packing layout (int4 nibble order, int2 byte layout).
- The dequant arithmetic (per-row vs per-dim scale application, bias
  correction for biased-code variants).
- The mean-pool + L2-normalize wrapper.

Run with:

    .venv-py/bin/python tests/test_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import StaticEmbedding
from tokenizers import Tokenizer

import lattice

# Make sure we can import slicer (sibling project).
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "slicer" / "src"))
from slicer.quantize import (  # noqa: E402
    dequantize,
    quantize_and_slice,
)

DATA = REPO / "data"
SOURCE_FP32 = (
    REPO
    / "trainer"
    / "runs"
    / "tokenizer_stage2_10ep_20260803_modal_a100x4_r1"
    / "final.safetensors"
)
TOKENIZER_JSON = DATA / "fp32-dim-1024" / "tokenizer.json"

# A small but distribution-realistic set of inputs. Short queries, longer
# passages, ASCII and non-ASCII, content of varying tokenization complexity.
TEST_INPUTS = [
    "hello world",
    "The quick brown fox jumps over the lazy dog.",
    "tokenization",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "Static embedding models trade quality for speed in retrieval.",
    "café résumé naïve façade",  # non-ASCII (HF slow path on the Rust side)
    "URL https://example.com/path?q=1&r=2",
    "supercalifragilisticexpialidocious",
    "12345 67890",
    "a",
    # A longer passage to exercise multi-chunk inner loops:
    " ".join(["Wikipedia article body text."] * 50),
]


def load_fp32_source_weights() -> np.ndarray:
    """Load the stage-2 fp32 reference weights (the source the slicer
    quantizes from). Returns `(vocab, full_dim)` float32."""
    src = load_file(str(SOURCE_FP32))
    return src["embedding.weight"]


def build_st_from_fp32_weights(weights_fp32: np.ndarray) -> SentenceTransformer:
    """Build a sentence-transformers model with the given fp32 weights
    plugged into its `StaticEmbedding`. Uses the same tokenizer.json so
    token sequences match exactly."""
    tokenizer = Tokenizer.from_file(str(TOKENIZER_JSON))
    static = StaticEmbedding(
        tokenizer,
        embedding_weights=torch.from_numpy(weights_fp32.astype(np.float32).copy()),
    )
    return SentenceTransformer(modules=[static])


def fake_quantize(weights_fp32: np.ndarray, *, variant: str, dim: int) -> np.ndarray:
    """Apply the slicer's exact quantization → dequantization round-trip.
    Returns fp32 weights of shape `(vocab, dim)` that match what the Rust
    kernel would compute internally for the corresponding integer
    artifact."""
    qz = quantize_and_slice(
        weight=weights_fp32, variant=variant, dim=dim, source_label="parity-test",
    )
    return dequantize(qz).astype(np.float32)


def compare(variant_label: str, rust_model_path: Path, expected_weights: np.ndarray) -> bool:
    """Run TEST_INPUTS through both implementations and compare element-wise.
    Returns True if all inputs pass."""
    print(f"\n=== {variant_label} ===")

    # Rust side
    rust = lattice.Model.load(str(rust_model_path))
    tk = lattice.Tokenizer.load(str(rust_model_path.parent / "tokenizer.json"))
    print(f"  rust:   {rust!r}")

    # Python side: ST with the expected (possibly fake-quantized) weights
    st = build_st_from_fp32_weights(expected_weights)

    n_ok = 0
    max_err_seen = 0.0
    for text in TEST_INPUTS:
        ids = tk.encode(text)
        rust_emb = rust.embed(ids, normalize=True)
        st_emb = st.encode(text, normalize_embeddings=True, show_progress_bar=False)
        st_emb = np.asarray(st_emb, dtype=np.float32)

        max_abs = float(np.abs(rust_emb - st_emb).max())
        max_err_seen = max(max_err_seen, max_abs)

        ok = np.allclose(rust_emb, st_emb, atol=1e-5, rtol=1e-4)
        marker = "OK " if ok else "MIS"
        snippet = (text[:50] + ("…" if len(text) > 50 else "")).ljust(53)
        print(f"  [{marker}] {snippet} max_abs_err={max_abs:.2e}  toks={len(ids)}")
        if ok:
            n_ok += 1
        else:
            cos = float(
                np.dot(rust_emb, st_emb)
                / (np.linalg.norm(rust_emb) * np.linalg.norm(st_emb) + 1e-12)
            )
            print(f"        cos_sim={cos:.6f}")

    print(
        f"  {n_ok}/{len(TEST_INPUTS)} inputs passed, "
        f"worst max_abs_err={max_err_seen:.2e}"
    )
    return n_ok == len(TEST_INPUTS)


def main() -> int:
    print("Loading fp32 source weights...")
    fp32_full = load_fp32_source_weights()
    print(f"  source: shape={fp32_full.shape} dtype={fp32_full.dtype}")

    cases: list[tuple[str, Path, np.ndarray]] = []

    # --- fp32 baseline -----------------------------------------------------
    cases.append((
        "fp32 dim=1024",
        DATA / "fp32-dim-1024" / "model.safetensors",
        fp32_full,  # ST gets unmodified fp32
    ))

    # --- quant variants we ship -------------------------------------------
    for variant, dim, label in [
        ("int8_dim", 256, "int8-dim dim=256"),
        ("int4_dim", 512, "int4-dim dim=512"),
        ("int4_dim", 1024, "int4-dim dim=1024"),
        ("int4_row", 512, "int4-row dim=512"),
        ("int8_row", 256, "int8-row dim=256"),
        ("int2_row", 1024, "int2-row dim=1024"),
    ]:
        artifact = DATA / label.replace(" dim=", "-").replace("_", "-").replace(
            "int", "int"
        ).replace(" ", "/")
        # The path encoding above is fragile — just construct it directly:
        slug = f"{variant.replace('_', '-')}-{dim}"
        artifact_path = DATA / slug / "model.safetensors"
        if not artifact_path.is_file():
            print(f"\n[skip {label}]  artifact not found: {artifact_path}")
            continue
        fake_w = fake_quantize(fp32_full, variant=variant, dim=dim)
        cases.append((label, artifact_path, fake_w))

    all_passed = True
    for label, path, expected_w in cases:
        if not compare(label, path, expected_w):
            all_passed = False

    print()
    if all_passed:
        print("✅ All parity checks passed.")
        return 0
    else:
        print("❌ Some parity checks failed (see above).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
