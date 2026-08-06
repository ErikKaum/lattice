"""Symmetric weight-only quantization for the static embedding table.

The model is `nn.EmbeddingBag(vocab, dim, mode='mean')` — pure data
movement, no nonlinearities. That makes it a near-ideal PTQ target:

- Quantization error stays linear (no relu / softmax to amplify it).
- The mean over a doc's tokens averages noise down by ~`sqrt(N)`.
- Eval is cosine-similarity-based, which is shift-invariant — only
  direction matters, not magnitude.

This module provides fake quantization (round-then-dequantize, all in
fp32) so we can drop the result straight into the existing `StaticEmbedding`
+ `NanoBEIREvaluator` eval path with zero plumbing changes. Real int8
storage + integer accumulation is a separate engineering exercise; quality
is what we're measuring here.

Two scale granularities:

- **per_row**:  one scale per vocab row (30,522 scales). Best quality —
  each token gets its own dynamic-range budget. At deployment requires
  loading a per-token scale alongside the int8 row, partly defeating the
  cache-residency speedup.

- **per_dim**:  one scale per output dimension (`dim` scales). At
  deployment the inner loop stays in int — `output[d] = scale[d] *
  (1/N) * sum_tok Q[tok, d]` — one fp multiply at the end. Maximum
  speedup for the static-embedding case where the bottleneck is moving
  the table through cache.

Quantization is applied **after** matryoshka slicing: each `dim` gets
its own scales. This is also the natural choice for `per_row` where the
row-max changes with the slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


Axis = Literal["row", "dim", "hybrid"]


@dataclass(frozen=True)
class QuantSpec:
    bits: int  # 32 = no-op, otherwise 2..16
    axis: Axis  # "row" or "dim"; ignored when bits == 32

    def name(self) -> str:
        if self.bits >= 32:
            return "fp32"
        return f"int{self.bits}_{self.axis}"


def fake_quantize(W: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    """Return a fp32 tensor that has been round-tripped through `bits`-bit
    symmetric quantization along `axis`. Shape and dtype match the input.

    No-op (returns `W.clone()`) when `bits >= 32`.
    """
    if spec.bits >= 32:
        return W.clone()
    if spec.bits < 2:
        raise ValueError(f"bits must be >= 2, got {spec.bits}")
    if W.ndim != 2:
        raise ValueError(f"expected 2-D weight, got {W.ndim}-D")

    qmax = (1 << (spec.bits - 1)) - 1  # 127 for int8, 7 for int4, 1 for int2
    eps = torch.finfo(W.dtype).tiny

    if spec.axis == "hybrid":
        # Two-stage: per-row magnitude normalization, then per-dim quantize.
        # Matryoshka training makes per-dim magnitudes roughly uniform across
        # the vocab but does nothing about per-token magnitude variance
        # (rare tokens vs common ones differ by 10x+ in row norm). Absorbing
        # the per-token magnitude into `scale_row` first leaves a residual
        # with a much tighter dynamic range for the per-dim quantizer to fit.
        scale_row = W.detach().abs().amax(dim=1, keepdim=True).clamp(min=eps)  # (V, 1)
        W_normalized = W / scale_row                                            # ~[-1, 1]
        scale_dim = (
            W_normalized.detach().abs().amax(dim=0, keepdim=True) / qmax
        ).clamp(min=eps)                                                        # (1, D)
        q = (W_normalized / scale_dim).round().clamp(-qmax, qmax)
        return (q * scale_dim * scale_row).to(W.dtype)

    # `axis="row"`: reduce across columns → one scale per row.
    # `axis="dim"`: reduce across rows    → one scale per column.
    reduce_dim = 1 if spec.axis == "row" else 0
    amax = W.detach().abs().amax(dim=reduce_dim, keepdim=True)
    # `clamp(min=eps)` guards against zero-magnitude rows/cols
    # (e.g. unused vocab entries). Their quantized output is exactly 0,
    # which is what we want.
    scale = (amax / qmax).clamp(min=eps)

    q = (W / scale).round().clamp(-qmax, qmax)
    return (q * scale).to(W.dtype)


def quantized_table(W: torch.Tensor, dim: int, spec: QuantSpec) -> torch.Tensor:
    """Slice the embedding table to the first `dim` columns and apply
    fake quantization. The matryoshka view: every output dim is its own
    quantized model with its own scales."""
    if dim > W.shape[1]:
        raise ValueError(f"dim {dim} > embedding dim {W.shape[1]}")
    sliced = W[:, :dim].contiguous()
    return fake_quantize(sliced, spec)


def quantization_error(W: torch.Tensor, spec: QuantSpec) -> dict[str, float]:
    """Sanity helper. Returns per-element abs error stats vs `W`.
    Useful for diagnosing whether a regression is "the quantizer broke"
    or "the quality dropped but quantizer is correct"."""
    dq = fake_quantize(W, spec)
    diff = (dq - W).abs()
    # `torch.quantile` caps input at ~16M elements; the embedding table is
    # 30M+. Sample uniformly to compute p99 — accurate to ~0.1% on a
    # 256k-sample subset, which is plenty for diagnostics.
    flat = diff.flatten()
    if flat.numel() > 1 << 18:
        idx = torch.randint(0, flat.numel(), (1 << 18,), device=flat.device)
        flat = flat[idx]
    return {
        "max_abs_err": float(diff.max()),
        "mean_abs_err": float(diff.mean()),
        "p99_abs_err": float(flat.quantile(0.99)),
    }
