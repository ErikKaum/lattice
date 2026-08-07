"""Static embedding model. Architecturally identical to the reference
`sentence-transformers/static-retrieval-mrl-en-v1` — a single `EmbeddingBag`
with `mode='mean'`.

The `EmbeddingBag` does the mean-pool itself, so the forward is one CUDA op:
look up each token's row in the embedding table and average within each
sample's offset range. No transformer, no attention, no positional encodings,
no LayerNorm. This is what makes the model so cheap to evaluate.

Vocab size and embedding dim are exposed as constructor args to ease future
ablations, but the defaults match the reference exactly.
"""

from __future__ import annotations

import torch
from torch import nn

# bert-base-uncased vocab.
DEFAULT_VOCAB_SIZE = 30_522
DEFAULT_EMBEDDING_DIM = 1024
# Boundary tokens inserted by bert-base-uncased when
# `add_special_tokens=True`. Legacy training caches contain these at the
# start/end of every sequence.
BERT_BOUNDARY_TOKEN_IDS = (101, 102)


class StaticEmbeddingModel(nn.Module):
    """`nn.EmbeddingBag(vocab_size, embedding_dim, mode='mean')`.

    Forward inputs:
        input_ids: 1-D long tensor of all token IDs in the batch, concatenated
        offsets:   1-D long tensor of length `batch_size` — the start of each
                   sample in `input_ids`, exactly what `EmbeddingBag` wants.

    Output: (batch_size, embedding_dim) float tensor. Not normalized — the
    loss applies cosine similarity which normalizes internally.
    """

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        padding_idx: int | None = None,
        ignored_token_ids: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        ignored_token_ids = tuple(sorted(set(ignored_token_ids)))
        if any(token_id < 0 or token_id >= vocab_size for token_id in ignored_token_ids):
            raise ValueError(f"ignored_token_ids must be in [0, {vocab_size}): {ignored_token_ids}")
        self.ignored_token_ids = ignored_token_ids
        self.embedding = nn.EmbeddingBag(
            vocab_size,
            embedding_dim,
            mode="mean",
            padding_idx=padding_idx,
        )
        self.register_buffer(
            "_ignored_token_ids_tensor",
            torch.tensor(ignored_token_ids, dtype=torch.long),
            persistent=False,
        )
        self.zero_ignored_token_rows()
        if ignored_token_ids:
            self.embedding.weight.register_hook(self._zero_ignored_token_gradients)

    @torch.no_grad()
    def zero_ignored_token_rows(self) -> None:
        """Zero compatibility-only token rows, including after checkpoint load."""
        if self._ignored_token_ids_tensor.numel():
            self.embedding.weight.index_fill_(0, self._ignored_token_ids_tensor, 0.0)

    def _zero_ignored_token_gradients(self, gradient: torch.Tensor) -> torch.Tensor:
        # The embedding gradient is dense (EmbeddingBag's default). Mutating
        # this hook input avoids allocating a full ~125 MB gradient copy.
        gradient.index_fill_(0, self._ignored_token_ids_tensor, 0.0)
        return gradient

    def forward(
        self,
        input_ids: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        return self.embedding(input_ids, offsets)

    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.Tensor,
        offsets: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Inference helper. L2-normalized by default since retrieval needs
        cosine-similar vectors."""
        out = self.forward(input_ids, offsets)
        if normalize:
            out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out


def truncate_to_dim(emb: torch.Tensor, dim: int) -> torch.Tensor:
    """First-`dim` slice of an embedding, with L2 re-normalization. Used by
    Matryoshka eval and at serve time when a smaller embedding is preferred."""
    sliced = emb[..., :dim]
    return torch.nn.functional.normalize(sliced, p=2, dim=-1)
