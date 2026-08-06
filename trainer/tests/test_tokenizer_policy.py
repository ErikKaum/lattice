from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trainer.stage2_dataloader import Stage2Dataloader, _negative_seed
from trainer.stage2_tokenize import _tokenize_and_pack, tokenize_all


class _Encoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _RecordingTokenizer:
    def __init__(self) -> None:
        self.policies: list[bool] = []

    def encode_batch_fast(
        self, texts: list[str], add_special_tokens: bool
    ) -> list[_Encoding]:
        self.policies.append(add_special_tokens)
        return [_Encoding([len(text)]) for text in texts]


def _write_stage2_source(
    root: Path, source: str, add_special_tokens: bool | None
) -> None:
    source_dir = root / source
    source_dir.mkdir(parents=True)
    np.array([1, 2], dtype="<u2").tofile(source_dir / "query_tokens.bin")
    np.array([0, 1, 2], dtype="<u8").tofile(source_dir / "query_offsets.bin")
    np.array([3, 4], dtype="<u2").tofile(source_dir / "positive_tokens.bin")
    np.array([0, 1, 2], dtype="<u8").tofile(source_dir / "positive_offsets.bin")
    np.array([5, 6], dtype="<u2").tofile(source_dir / "negative_tokens.bin")
    np.array([0, 1, 2], dtype="<u8").tofile(source_dir / "negative_offsets.bin")
    meta: dict[str, object] = {
        "source": source,
        "n_pairs": 2,
        "n_negatives_per_pair": 1,
    }
    if add_special_tokens is not None:
        meta["add_special_tokens"] = add_special_tokens
    (source_dir / "meta.json").write_text(json.dumps(meta))


def test_stage2_tokenizer_omits_special_tokens() -> None:
    tokenizer = _RecordingTokenizer()
    concat, offsets = _tokenize_and_pack(
        tokenizer, ["one", "three"], chunk_size=1
    )
    assert tokenizer.policies == [False, False]
    assert concat.tolist() == [3, 5]
    assert offsets.tolist() == [0, 1, 2]


def test_stage2_legacy_metadata_defaults_true(tmp_path: Path) -> None:
    _write_stage2_source(tmp_path, "legacy", add_special_tokens=None)
    loader = Stage2Dataloader(tmp_path, batch_size=2)
    assert loader.add_special_tokens is True


def test_stage2_rejects_mixed_policies(tmp_path: Path) -> None:
    _write_stage2_source(tmp_path, "canonical", add_special_tokens=False)
    _write_stage2_source(tmp_path, "legacy", add_special_tokens=True)
    with pytest.raises(ValueError, match="mixes add_special_tokens"):
        Stage2Dataloader(tmp_path, batch_size=2)


def test_stage2_negative_seed_is_stable_and_batch_specific() -> None:
    assert _negative_seed(42, 3, "msmarco", 1024) == 3670835513354418040
    assert _negative_seed(42, 4, "msmarco", 1024) != _negative_seed(
        42, 3, "msmarco", 1024
    )
    assert _negative_seed(42, 3, "nq", 1024) != _negative_seed(
        42, 3, "msmarco", 1024
    )


def test_stage2_tokenize_refuses_legacy_plus_missing_source(
    tmp_path: Path,
) -> None:
    training = tmp_path / "stage2" / "training" / "legacy"
    training.mkdir(parents=True)
    (training / "meta.json").write_text(json.dumps({"source": "legacy"}))
    with pytest.raises(RuntimeError, match="fresh output root"):
        tokenize_all(
            splits_dir=tmp_path / "splits",
            tokenizer_path=tmp_path / "tokenizer.json",
            out_root=tmp_path,
            sources=("legacy", "missing"),
        )


def test_stage2_legacy_manifest_preserves_resolved_policy(
    tmp_path: Path,
) -> None:
    training = tmp_path / "stage2" / "training" / "legacy"
    training.mkdir(parents=True)
    (training / "meta.json").write_text(json.dumps({"source": "legacy"}))
    tokenize_all(
        splits_dir=tmp_path / "splits",
        tokenizer_path=tmp_path / "tokenizer.json",
        out_root=tmp_path,
        sources=("legacy",),
    )
    manifest = json.loads(
        (tmp_path / "stage2" / "training" / "manifest.json").read_text()
    )
    assert manifest["add_special_tokens"] is True
