"""Real symmetric quantization + matryoshka slicing.

Slicing happens *before* quantizing — for per-row scales, the row's
max-abs is computed over the kept columns, which produces meaningfully
tighter scales than "quantize at 1024, slice at runtime" for the
smaller dims. The cost is one quantize-per-(dim, axis) artifact instead
of one storage variant with runtime slicing; we accept that cost in
exchange for an extra ~0.001 NDCG at d=256 and below.

For sub-byte storage (int4, int2) we pack into uint8 with a fixed
convention:

    int4 (q ∈ [-7, +7]):
        byte i = ((q[2i]   + 7) & 0xF)
               | (((q[2i+1] + 7) & 0xF) << 4)
        i.e. low nibble first, value biased by +qmax so storage stays
        in [0, 14] (15 is never used; symmetric range).

    int2 (q ∈ [-1, +1]):
        byte i = ((q[4i  ] + 1)     )
               | ((q[4i+1] + 1) << 2)
               | ((q[4i+2] + 1) << 4)
               | ((q[4i+3] + 1) << 6)
        same bias-by-qmax convention; storage values ∈ {0, 1, 2}.
        Code 3 is RESERVED — never emitted by this encoder (since q+1
        for q ∈ {-1, 0, +1} only takes {0, 1, 2}). Decoders may treat
        its presence as file corruption, but the canonical decode
        formula `(code & 0x3) - 1` still produces a numerically valid
        +2 for it; we leave that as the encoder's invariant rather
        than gating the hot path on validation.

Both pack conventions are recorded in the output safetensors metadata
as `pack_bias` and `pack_layout` so the loader (Rust or otherwise)
knows what to subtract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


SUPPORTED_DIMS: tuple[int, ...] = (32, 64, 128, 256, 512, 1024)

Variant = Literal[
    "fp32",
    "int8_row", "int8_dim",
    "int4_row", "int4_dim",
    "int2_row", "int2_dim",
]
SUPPORTED_VARIANTS: tuple[Variant, ...] = (
    "fp32",
    "int8_row", "int8_dim",
    "int4_row", "int4_dim",
    "int2_row", "int2_dim",
)


@dataclass(frozen=True)
class Quantized:
    """Result of a quantize-and-slice. `q` and `scale` are written to the
    safetensors output; the metadata dict joins the safetensors header."""
    q: np.ndarray            # quantized weight (int8 / packed uint8 / fp32)
    scale: np.ndarray | None # f32 per-row or per-dim scales (None for fp32)
    metadata: dict[str, str] # safetensors header metadata


def quantize_and_slice(
    weight: np.ndarray,
    variant: Variant,
    dim: int,
    source_label: str = "",
) -> Quantized:
    """Take a fp32 weight matrix (vocab, full_dim), slice to (vocab, dim),
    apply the requested quantization, return packed tensors + metadata."""
    if weight.ndim != 2:
        raise ValueError(f"expected 2-D weight, got {weight.ndim}-D")
    if dim not in SUPPORTED_DIMS:
        raise ValueError(f"dim={dim} not in {SUPPORTED_DIMS}")
    if dim > weight.shape[1]:
        raise ValueError(f"dim={dim} > source dim {weight.shape[1]}")
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"variant={variant!r} not in {SUPPORTED_VARIANTS}")

    sliced = weight[:, :dim].astype(np.float32, copy=False)

    if variant == "fp32":
        return Quantized(
            q=sliced.copy(),
            scale=None,
            metadata=_metadata(variant, bits=32, axis="none", dim=dim,
                               vocab=weight.shape[0], source=source_label),
        )

    bits = {"int8_row": 8, "int8_dim": 8,
            "int4_row": 4, "int4_dim": 4,
            "int2_row": 2, "int2_dim": 2}[variant]
    axis = "row" if variant.endswith("_row") else "dim"
    qmax = (1 << (bits - 1)) - 1   # 127, 7, 1
    q_signed, scale = _symmetric_quantize(sliced, axis, qmax)

    if bits == 8:
        q_stored = q_signed.astype(np.int8)
    elif bits == 4:
        q_stored = _pack_int4(q_signed, qmax)
    elif bits == 2:
        q_stored = _pack_int2(q_signed, qmax)
    else:
        raise AssertionError("unreachable")

    metadata = _metadata(variant, bits=bits, axis=axis, dim=dim,
                         vocab=weight.shape[0], source=source_label)
    if bits < 8:
        metadata["pack_bias"] = str(qmax)
        metadata["pack_layout"] = "low_to_high"
        metadata["packed_dtype"] = "u8"

    return Quantized(q=q_stored, scale=scale.astype(np.float32), metadata=metadata)


def _symmetric_quantize(
    W: np.ndarray, axis: str, qmax: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric quantize. Returns (signed-int q, fp32 scale).
    `axis="row"` → scale shape (vocab,).  `axis="dim"` → scale shape (dim,)."""
    reduce_axis = 1 if axis == "row" else 0
    amax = np.abs(W).max(axis=reduce_axis, keepdims=True)
    scale = np.clip(amax / qmax, np.finfo(W.dtype).tiny, None)
    q = np.round(W / scale).clip(-qmax, qmax).astype(np.int8)
    return q, scale.squeeze(reduce_axis)


def _pack_int4(q: np.ndarray, qmax: int) -> np.ndarray:
    """Pack int4 signed values (q in [-qmax, +qmax]) into uint8 — two
    nibbles per byte, low-then-high. Requires last-dim even."""
    vocab, dim = q.shape
    if dim % 2 != 0:
        raise ValueError(f"int4 packing requires even dim, got {dim}")
    biased = (q + qmax).astype(np.uint8)  # in [0, 2*qmax]
    low = biased[:, 0::2]
    high = biased[:, 1::2]
    return (low | (high << 4)).astype(np.uint8)


def _pack_int2(q: np.ndarray, qmax: int) -> np.ndarray:
    """Pack int2 signed values into uint8 — four 2-bit fields per byte,
    low-to-high. Requires last-dim divisible by 4."""
    vocab, dim = q.shape
    if dim % 4 != 0:
        raise ValueError(f"int2 packing requires dim divisible by 4, got {dim}")
    biased = (q + qmax).astype(np.uint8)  # in {0, 1, 2}
    a = biased[:, 0::4]
    b = biased[:, 1::4]
    c = biased[:, 2::4]
    d = biased[:, 3::4]
    return (a | (b << 2) | (c << 4) | (d << 6)).astype(np.uint8)


def _metadata(
    variant: str, bits: int, axis: str, dim: int, vocab: int, source: str,
) -> dict[str, str]:
    return {
        "lattice_variant": variant,
        "bits": str(bits),
        "axis": axis,
        "dim": str(dim),
        "vocab_size": str(vocab),
        "tokenizer": "bert-base-uncased",
        "source": source,
    }


# --------------------------------------------------------------------------
# Dequantization helpers — for the post-write round-trip sanity check.
# --------------------------------------------------------------------------


def dequantize(qz: Quantized) -> np.ndarray:
    """Reverse `quantize_and_slice` to fp32. Used by the CLI's verify step
    to catch bit-packing bugs. Not in the deployment hot path."""
    variant = qz.metadata["lattice_variant"]
    if variant == "fp32":
        return qz.q.copy()
    bits = int(qz.metadata["bits"])
    axis = qz.metadata["axis"]
    dim = int(qz.metadata["dim"])
    qmax = (1 << (bits - 1)) - 1

    if bits == 8:
        q_signed = qz.q.astype(np.int32)
    elif bits == 4:
        q_signed = _unpack_int4(qz.q, dim, qmax)
    elif bits == 2:
        q_signed = _unpack_int2(qz.q, dim, qmax)
    else:
        raise ValueError(f"unsupported bits {bits}")

    if axis == "row":
        return q_signed.astype(np.float32) * qz.scale[:, None]
    else:
        return q_signed.astype(np.float32) * qz.scale[None, :]


def _unpack_int4(packed: np.ndarray, dim: int, qmax: int) -> np.ndarray:
    vocab = packed.shape[0]
    out = np.empty((vocab, dim), dtype=np.int32)
    out[:, 0::2] = (packed & 0xF).astype(np.int32) - qmax
    out[:, 1::2] = ((packed >> 4) & 0xF).astype(np.int32) - qmax
    return out


def _unpack_int2(packed: np.ndarray, dim: int, qmax: int) -> np.ndarray:
    vocab = packed.shape[0]
    out = np.empty((vocab, dim), dtype=np.int32)
    out[:, 0::4] = (packed & 0x3).astype(np.int32) - qmax
    out[:, 1::4] = ((packed >> 2) & 0x3).astype(np.int32) - qmax
    out[:, 2::4] = ((packed >> 4) & 0x3).astype(np.int32) - qmax
    out[:, 3::4] = ((packed >> 6) & 0x3).astype(np.int32) - qmax
    return out
