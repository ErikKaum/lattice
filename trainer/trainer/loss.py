"""Loss functions for the training run.

`MultipleNegativesRankingLoss` (a.k.a. InfoNCE with in-batch negatives) and
`MatryoshkaLoss` (apply the loss at multiple output dimensions and average).
Both written from scratch — they are a few lines each and match the
sentence-transformers reference behavior exactly (verified against the
formula in `sentence-transformers/sentence_transformers/losses/`).

Per `plan.md`:
- `matryoshka_dims = [1024, 512, 256, 128, 64, 32]` with equal weights
- `scale = 20.0` (sentence-transformers default)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_SCALE = 20.0
DEFAULT_MATRYOSHKA_DIMS: tuple[int, ...] = (1024, 512, 256, 128, 64, 32)


def cos_sim_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """All-pairs cosine similarity. `a`: (m, d), `b`: (n, d) → (m, n)."""
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    return a @ b.t()


class MultipleNegativesRankingLoss(nn.Module):
    """In-batch negatives. For (anchor, positive) pairs `(a_i, p_i)` in a
    batch, every other `p_j` (j != i) acts as a negative for `a_i`.

    Loss = CE(scale * cos_sim(A, P), labels=[0, 1, ..., B-1])

    Larger batches → richer negatives → stronger gradient. The whole point of
    pushing batch size as high as memory allows in `plan.md`.
    """

    def __init__(self, scale: float = DEFAULT_SCALE) -> None:
        super().__init__()
        self.scale = scale

    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        hard_negatives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`hard_negatives` of shape `(B, K, D)` are explicit per-anchor
        negatives (stage 2: K=7 NV-Retriever-mined hard negs per query).
        They join the positives as candidates so anchor `i`'s candidate
        pool is `[p_1, ..., p_B, n_1_1, ..., n_B_K]` (one true positive at
        index `i`, B-1 in-batch positives, plus B·K hard negatives). When
        `None` (stage 1) the loss collapses to plain in-batch MNR."""
        if hard_negatives is None:
            candidates = positives
        else:
            B, K, D = hard_negatives.shape
            if B != positives.size(0) or D != positives.size(1):
                raise ValueError(
                    f"hard_negatives shape {(B, K, D)} incompatible with "
                    f"positives shape {tuple(positives.shape)}"
                )
            candidates = torch.cat(
                [positives, hard_negatives.reshape(B * K, D)], dim=0
            )  # (B + B*K, D)
        logits = cos_sim_matrix(anchors, candidates) * self.scale
        labels = torch.arange(anchors.size(0), device=anchors.device)
        return F.cross_entropy(logits, labels)


class MatryoshkaLoss(nn.Module):
    """Wrap a base loss and apply it at multiple output truncations of the
    embedding. Each truncation is the leading slice `emb[..., :dim]` followed
    by an implicit L2 re-normalization (done inside `cos_sim_matrix` via the
    base loss).

    `plan.md` says equal weights; this is what `MatryoshkaLoss` defaults to
    in sentence-transformers if `matryoshka_weights` isn't passed (it sets
    every weight to 1.0 and then averages).
    """

    def __init__(
        self,
        base_loss: nn.Module,
        dims: tuple[int, ...] = DEFAULT_MATRYOSHKA_DIMS,
        weights: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        if not dims:
            raise ValueError("dims must be non-empty")
        self.dims = tuple(sorted(dims, reverse=True))
        if weights is None:
            weights = tuple(1.0 for _ in self.dims)
        if len(weights) != len(self.dims):
            raise ValueError(
                f"len(weights)={len(weights)} != len(dims)={len(self.dims)}"
            )
        self.register_buffer(
            "weights",
            torch.tensor(weights, dtype=torch.float32),
        )

    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        hard_negatives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        full_dim = anchors.size(-1)
        if full_dim < self.dims[0]:
            raise ValueError(
                f"embedding dim {full_dim} smaller than max matryoshka dim "
                f"{self.dims[0]}"
            )
        total = anchors.new_zeros(())
        for w, d in zip(self.weights.tolist(), self.dims):
            a_d = anchors[..., :d]
            p_d = positives[..., :d]
            hn_d = hard_negatives[..., :d] if hard_negatives is not None else None
            total = total + w * self.base_loss(a_d, p_d, hn_d)
        return total / float(self.weights.sum().item())
