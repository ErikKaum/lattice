from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from trainer.export_hf import export_hf_model


def test_export_hf_writes_loadable_root_layout(tmp_path: Path) -> None:
    checkpoint = tmp_path / "final.safetensors"
    save_file(
        {"embedding.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)},
        str(checkpoint),
        metadata={"vocab_size": "3", "embedding_dim": "4"},
    )

    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("# test model\n")
    out_dir = tmp_path / "hub-model"

    export_hf_model(checkpoint, tokenizer_path, out_dir, model_card)

    assert (out_dir / "model.safetensors").is_file()
    assert (out_dir / "tokenizer.json").is_file()
    assert (out_dir / "modules.json").is_file()
    assert (out_dir / "README.md").read_text() == "# test model\n"

    model = SentenceTransformer(str(out_dir))
    embeddings = model.encode(["hello world"], normalize_embeddings=True)
    assert embeddings.shape == (1, 4)
    torch.testing.assert_close(
        torch.from_numpy(embeddings).norm(dim=1),
        torch.ones(1),
    )
