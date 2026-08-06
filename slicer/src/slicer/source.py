"""Resolve where the fp32 source weights and tokenizer come from.

Two paths:

- **HF Hub** (default): the published `lattice-retrieval` repo follows
  the current single-module `sentence-transformers` layout, which writes
  `model.safetensors` and `tokenizer.json` at the repository root. They are
  downloaded via `huggingface_hub` into the standard HF cache.

- **Local** (`--source <path>`): the user points us at the exact
  `.safetensors` file. The tokenizer, if requested, is expected to be
  sitting next to it in the same directory as `tokenizer.json`. This
  intentionally does *not* search alternate layouts — keep the local
  contract dead-simple, the user already knows where their file is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_REPO = "erikkaum/lattice-retrieval"
WEIGHTS_PATH_IN_REPO = "model.safetensors"
TOKENIZER_PATH_IN_REPO = "tokenizer.json"


@dataclass(frozen=True)
class Source:
    weights: Path
    tokenizer: Path | None
    label: str  # for log messages — repo id or local path


def resolve_source(
    source: Path | None,
    repo: str = DEFAULT_REPO,
    revision: str = "main",
    want_tokenizer: bool = True,
) -> Source:
    """Find `model.safetensors` (+ optionally `tokenizer.json`).

    `source=None` → fetch from HF Hub (`repo` @ `revision`).
    `source=<file>` → use this exact `.safetensors` file. If
        `want_tokenizer`, look for `tokenizer.json` next to it in the
        same directory.
    """
    if source is None:
        return _resolve_hub(repo, revision, want_tokenizer)
    return _resolve_local(source, want_tokenizer)


def _resolve_local(source: Path, want_tokenizer: bool) -> Source:
    if not source.is_file():
        raise FileNotFoundError(
            f"--source must point at a .safetensors file, got {source!r} "
            "(not a regular file)"
        )
    tokenizer: Path | None = None
    if want_tokenizer:
        tokenizer = source.parent / "tokenizer.json"
        if not tokenizer.is_file():
            raise FileNotFoundError(
                f"expected tokenizer.json next to {source.name} at "
                f"{tokenizer} — pass --omit-tokenizer if you don't need it "
                "bundled in the output"
            )
    return Source(weights=source, tokenizer=tokenizer, label=str(source))


def _resolve_hub(repo: str, revision: str, want_tokenizer: bool) -> Source:
    # Lazy import: huggingface_hub is heavy and we want `slicer --help`
    # to be snappy.
    from huggingface_hub import hf_hub_download

    weights = Path(hf_hub_download(repo_id=repo, filename=WEIGHTS_PATH_IN_REPO, revision=revision))
    tokenizer: Path | None = None
    if want_tokenizer:
        tokenizer = Path(
            hf_hub_download(repo_id=repo, filename=TOKENIZER_PATH_IN_REPO, revision=revision)
        )
    return Source(weights=weights, tokenizer=tokenizer, label=f"hf://{repo}@{revision}")
