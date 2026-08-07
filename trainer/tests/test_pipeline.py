"""Synthetic end-to-end tests for the Python side.

We build a fake `cache/` directory with two tiny "source" subdirectories, a
`tiers.json` over them, and walk it through `SourceReader → TierDataloader →
StaticEmbeddingModel → MatryoshkaLoss(MultipleNegativesRankingLoss)`. No
real tokenizer, no real data — the goal is to catch shape/wiring bugs before
plugging the real cache in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from trainer.dataloader import TierDataloader
from trainer.io import SourceReader, load_tiers, source_dir
from trainer.loss import MatryoshkaLoss, MultipleNegativesRankingLoss, cos_sim_matrix
from trainer.model import BERT_BOUNDARY_TOKEN_IDS, StaticEmbeddingModel


def _write_source(
    cache_root: Path,
    name: str,
    rows: list[tuple[list[int], list[int]]],
    add_special_tokens: bool | None = False,
) -> None:
    """Hand-pack a fake tokenized source. `rows` is `[(query_ids, doc_ids)]`."""
    d = source_dir(cache_root, name)
    d.mkdir(parents=True, exist_ok=True)

    for side, idx in (("query", 0), ("doc", 1)):
        tokens: list[int] = []
        offsets: list[int] = []
        cur = 0
        for r in rows:
            offsets.append(cur)
            tokens.extend(r[idx])
            cur += len(r[idx])
        offsets.append(cur)  # sentinel

        (d / f"{side}_tokens.bin").write_bytes(np.array(tokens, dtype="<u2").tobytes())
        (d / f"{side}_offsets.bin").write_bytes(np.array(offsets, dtype="<u8").tobytes())

    meta = {
        "source": name,
        "n_rows": len(rows),
        "total_query_tokens": sum(len(r[0]) for r in rows),
        "total_doc_tokens": sum(len(r[1]) for r in rows),
        "tokenizer": "bert-base-uncased",
        "tokens_dtype": "u16",
        "offsets_dtype": "u64",
    }
    if add_special_tokens is not None:
        meta["add_special_tokens"] = add_special_tokens
    (d / "meta.json").write_text(json.dumps(meta))


def _write_tiers(cache_root: Path, tiers: list[dict]) -> None:
    (cache_root / "tiers.json").write_text(json.dumps(tiers))


@pytest.fixture
def fake_cache(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    rows_a = [
        (
            [int(x) for x in rng.integers(1, 1000, size=rng.integers(3, 20))],
            [int(x) for x in rng.integers(1, 1000, size=rng.integers(10, 50))],
        )
        for _ in range(40)
    ]
    rows_b = [
        (
            [int(x) for x in rng.integers(1, 1000, size=rng.integers(2, 8))],
            [int(x) for x in rng.integers(1, 1000, size=rng.integers(5, 25))],
        )
        for _ in range(60)
    ]
    _write_source(tmp_path, "src_a", rows_a)
    _write_source(tmp_path, "src_b", rows_b)
    _write_tiers(
        tmp_path,
        [
            {
                "tier": "xs",
                "target_rows": 100,
                "actual_rows": 100,
                "per_source": [
                    {"source": "src_a", "take_rows": 40},
                    {"source": "src_b", "take_rows": 60},
                ],
            }
        ],
    )
    return tmp_path


class TestSourceReader:
    def test_round_trip(self, fake_cache):
        r = SourceReader(source_dir(fake_cache, "src_a"))
        assert r.n_rows == 40
        # Reading every row should produce a non-empty array of u16 ids.
        for i in range(40):
            q = r.get_row("query", i)
            d = r.get_row("doc", i)
            assert q.dtype == np.uint16
            assert d.dtype == np.uint16
            assert len(q) > 0
            assert len(d) > 0

    def test_get_batch_offsets_local(self, fake_cache):
        r = SourceReader(source_dir(fake_cache, "src_a"))
        flat, local = r.get_batch("query", 5, 10)
        assert local[0] == 0
        # Reconstruct each row's tokens via local offsets and compare with
        # per-row reads.
        for i, off in enumerate(local):
            end = local[i + 1] if i + 1 < len(local) else len(flat)
            sliced = flat[int(off) : int(end)]
            ref = r.get_row("query", 5 + i)
            assert np.array_equal(sliced, ref)

    def test_metadata_without_policy_is_recognized_as_legacy(self, tmp_path: Path):
        _write_source(
            tmp_path,
            "legacy",
            [([101, 10, 102], [101, 20, 102])],
            add_special_tokens=None,
        )
        reader = SourceReader(source_dir(tmp_path, "legacy"))
        assert reader.meta.add_special_tokens is True


class TestTierDataloader:
    def test_yields_correct_shapes_and_sources(self, fake_cache):
        tiers = load_tiers(fake_cache)
        loader = TierDataloader(tiers["xs"], fake_cache, batch_size=8, seed=1)
        batches = list(loader.epoch_iter(0))
        assert len(batches) == 5 + 7  # 40//8 + 60//8 = 5 + 7
        for b in batches:
            assert b.query_input_ids.dtype == torch.int64
            assert b.query_offsets.dtype == torch.int64
            assert b.query_offsets.numel() == 8
            assert b.doc_offsets.numel() == 8
            assert b.source in {"src_a", "src_b"}

    def test_source_distribution_proportional(self, fake_cache):
        tiers = load_tiers(fake_cache)
        loader = TierDataloader(tiers["xs"], fake_cache, batch_size=2, seed=1)
        # batch=2: src_a→20 batches, src_b→30 batches. Should match exactly.
        from collections import Counter

        counts = Counter(b.source for b in loader.epoch_iter(0))
        assert counts == {"src_a": 20, "src_b": 30}

    def test_reshuffle_per_epoch(self, fake_cache):
        tiers = load_tiers(fake_cache)
        loader = TierDataloader(tiers["xs"], fake_cache, batch_size=8, seed=1)
        seq_0 = [b.source for b in loader.epoch_iter(0)]
        seq_1 = [b.source for b in loader.epoch_iter(1)]
        # Same multiset, different order (very unlikely to collide for 12 items).
        assert sorted(seq_0) == sorted(seq_1)
        assert seq_0 != seq_1

    def test_rejects_mixed_tokenizer_policies(self, tmp_path: Path):
        rows = [([1, 2], [3, 4]), ([5, 6], [7, 8])]
        _write_source(tmp_path, "canonical", rows, add_special_tokens=False)
        _write_source(tmp_path, "legacy", rows, add_special_tokens=True)
        _write_tiers(
            tmp_path,
            [
                {
                    "tier": "xs",
                    "target_rows": 4,
                    "actual_rows": 4,
                    "per_source": [
                        {"source": "canonical", "take_rows": 2},
                        {"source": "legacy", "take_rows": 2},
                    ],
                }
            ],
        )
        with pytest.raises(ValueError, match="mixes caches"):
            TierDataloader(load_tiers(tmp_path)["xs"], tmp_path, batch_size=2)


class TestModel:
    def test_forward_shape(self):
        m = StaticEmbeddingModel(vocab_size=100, embedding_dim=16)
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
        offsets = torch.tensor([0, 3, 5], dtype=torch.long)  # 3 samples
        out = m(input_ids, offsets)
        assert out.shape == (3, 16)

    def test_legacy_special_mask_matches_content_only_cosine_and_gradients(self):
        torch.manual_seed(0)
        clean = StaticEmbeddingModel(vocab_size=200, embedding_dim=16)
        legacy = StaticEmbeddingModel(
            vocab_size=200,
            embedding_dim=16,
            ignored_token_ids=BERT_BOUNDARY_TOKEN_IDS,
        )
        with torch.no_grad():
            legacy.embedding.weight.copy_(clean.embedding.weight)
        legacy.zero_ignored_token_rows()

        clean_ids = torch.tensor([5, 6, 7, 8], dtype=torch.long)
        clean_offsets = torch.tensor([0, 2], dtype=torch.long)
        legacy_ids = torch.tensor([101, 5, 6, 102, 101, 7, 8, 102], dtype=torch.long)
        legacy_offsets = torch.tensor([0, 4], dtype=torch.long)

        clean_out = torch.nn.functional.normalize(clean(clean_ids, clean_offsets), dim=-1)
        legacy_out = torch.nn.functional.normalize(legacy(legacy_ids, legacy_offsets), dim=-1)
        torch.testing.assert_close(clean_out, legacy_out)

        clean_out.sum().backward()
        legacy_out.sum().backward()
        torch.testing.assert_close(
            clean.embedding.weight.grad[[5, 6, 7, 8]],
            legacy.embedding.weight.grad[[5, 6, 7, 8]],
        )
        assert torch.count_nonzero(legacy.embedding.weight.grad[list(BERT_BOUNDARY_TOKEN_IDS)]) == 0

    def test_zero_ignored_rows_after_checkpoint_copy(self):
        model = StaticEmbeddingModel(
            vocab_size=200,
            embedding_dim=4,
            ignored_token_ids=BERT_BOUNDARY_TOKEN_IDS,
        )
        with torch.no_grad():
            model.embedding.weight.fill_(1.0)
        model.zero_ignored_token_rows()
        assert torch.count_nonzero(model.embedding.weight[list(BERT_BOUNDARY_TOKEN_IDS)]) == 0

    def test_checkpoint_records_legacy_compatibility(self, tmp_path: Path):
        from trainer.train import load_checkpoint, save_checkpoint

        model = StaticEmbeddingModel(
            vocab_size=200,
            embedding_dim=4,
            ignored_token_ids=BERT_BOUNDARY_TOKEN_IDS,
        )
        path = tmp_path / "model.safetensors"
        save_checkpoint(model, path)
        loaded = load_checkpoint(path)
        assert loaded["ignored_token_ids"] == BERT_BOUNDARY_TOKEN_IDS


class TestLoss:
    def test_mnr_zero_on_aligned_pairs(self):
        # When anchors == positives (perfectly aligned) and they're all
        # orthogonal/distinct, the loss is small but not zero — the diagonal
        # is just way bigger than off-diagonals, but CE still has a small
        # value. Verify it's much smaller than a random batch.
        torch.manual_seed(0)
        d = 32
        # Make anchors orthogonal-ish by picking far-apart random vectors.
        emb = torch.randn(8, d)
        aligned_loss = MultipleNegativesRankingLoss()(emb, emb).item()
        # Random pairs should have higher loss.
        emb_a = torch.randn(8, d)
        emb_b = torch.randn(8, d)
        random_loss = MultipleNegativesRankingLoss()(emb_a, emb_b).item()
        assert aligned_loss < random_loss

    def test_matryoshka_averages_equal_weights(self):
        torch.manual_seed(0)
        d = 64
        a = torch.randn(4, d)
        b = torch.randn(4, d)
        base = MultipleNegativesRankingLoss()
        m = MatryoshkaLoss(base, dims=(64, 32, 16))
        mat = m(a, b).item()
        # Manually average the three losses.
        manual = (
            base(a[..., :64], b[..., :64]).item()
            + base(a[..., :32], b[..., :32]).item()
            + base(a[..., :16], b[..., :16]).item()
        ) / 3.0
        assert abs(mat - manual) < 1e-5

    def test_cos_sim_matrix_is_normalized(self):
        a = torch.randn(3, 5) * 100
        b = torch.randn(4, 5) * 0.01
        s = cos_sim_matrix(a, b)
        # All entries should be in [-1, 1].
        assert s.shape == (3, 4)
        assert s.abs().max().item() <= 1.0 + 1e-6


class TestEndToEnd:
    def test_one_step(self, fake_cache):
        tiers = load_tiers(fake_cache)
        loader = TierDataloader(tiers["xs"], fake_cache, batch_size=8, seed=1)
        model = StaticEmbeddingModel(vocab_size=1024, embedding_dim=32)
        criterion = MatryoshkaLoss(MultipleNegativesRankingLoss(), dims=(32, 16, 8))
        opt = torch.optim.AdamW(model.parameters(), lr=0.1)
        batch = next(loader.epoch_iter(0))
        a = model(batch.query_input_ids, batch.query_offsets)
        p = model(batch.doc_input_ids, batch.doc_offsets)
        loss = criterion(a, p)
        loss.backward()
        opt.step()
        assert torch.isfinite(loss)


class TestDDPSharding:
    def test_ranks_partition_chunks(self, fake_cache):
        tiers = load_tiers(fake_cache)
        # batch=4 → src_a: 10 chunks, src_b: 15 chunks → 25 chunks total.
        # world_size=3 → each rank gets floor(25/3) = 8 chunks (one chunk
        # gets dropped to keep ranks equal-sized for NCCL safety).
        world_size = 3
        per_rank_counts: list[int] = []
        seen_keys: set[tuple[int, int]] = set()
        for rank in range(world_size):
            loader = TierDataloader(
                tiers["xs"],
                fake_cache,
                batch_size=4,
                seed=99,
                rank=rank,
                world_size=world_size,
            )
            count = 0
            for b in loader.epoch_iter(0):
                key = (b.source, int(b.query_offsets[0].item()))
                seen_keys.add(key)
                count += 1
            per_rank_counts.append(count)
        # All ranks see the same number of batches — load-bearing property
        # for DDP's collective ops (`all_reduce` on loss + gradient sync).
        assert len(set(per_rank_counts)) == 1
        assert per_rank_counts[0] == 25 // world_size  # 8

    def test_single_rank_equals_world_size_one(self, fake_cache):
        tiers = load_tiers(fake_cache)
        base = TierDataloader(tiers["xs"], fake_cache, batch_size=4, seed=7)
        ddp = TierDataloader(
            tiers["xs"],
            fake_cache,
            batch_size=4,
            seed=7,
            rank=0,
            world_size=1,
        )
        base_sources = [b.source for b in base.epoch_iter(0)]
        ddp_sources = [b.source for b in ddp.epoch_iter(0)]
        assert base_sources == ddp_sources


class TestStaging:
    def test_stage_in_round_trip(self, fake_cache, tmp_path, monkeypatch):
        from trainer import staging
        from trainer.io import SourceReader
        from trainer.io import source_dir as src_dir
        from trainer.staging import stage_in_tier

        scratch = tmp_path / "scratch"
        tiers = load_tiers(fake_cache)
        stage_in_tier(fake_cache, scratch, tiers["xs"])

        # Sentinel file should exist for each staged source.
        for ts in tiers["xs"].per_source:
            sentinel = scratch / "subsets" / f"{ts.source}.staged"
            assert sentinel.exists()

        # Reader works against the staged dir.
        for ts in tiers["xs"].per_source:
            staged_dir = src_dir(scratch, ts.source)
            reader = SourceReader(staged_dir)
            assert reader.n_rows == ts.take_rows

        # Second call should be a no-op (sentinel present).
        before = list(scratch.rglob("*"))
        monkeypatch.setattr(
            staging,
            "_read_u64",
            lambda *_args: pytest.fail(
                "completed source must be reused before sparse offset reads"
            ),
        )
        stage_in_tier(fake_cache, scratch, tiers["xs"])
        after = list(scratch.rglob("*"))
        assert sorted(p.name for p in before) == sorted(p.name for p in after)

    def test_stage_in_copies_only_requested_prefix_and_can_upgrade(self, tmp_path: Path):
        from trainer.io import SourceReader
        from trainer.io import source_dir as src_dir
        from trainer.staging import stage_in_tier

        cache = tmp_path / "cache"
        rows = [([i, i + 100], [i + 200, i + 300, i + 400]) for i in range(10)]
        _write_source(cache, "source", rows)
        _write_tiers(
            cache,
            [
                {
                    "tier": "xs",
                    "target_rows": 3,
                    "actual_rows": 3,
                    "per_source": [{"source": "source", "take_rows": 3}],
                },
                {
                    "tier": "small",
                    "target_rows": 7,
                    "actual_rows": 7,
                    "per_source": [{"source": "source", "take_rows": 7}],
                },
            ],
        )
        tiers = load_tiers(cache)
        scratch = tmp_path / "scratch"

        stage_in_tier(cache, scratch, tiers["xs"])
        source = src_dir(cache, "source")
        staged = src_dir(scratch, "source")
        reader = SourceReader(staged)
        assert reader.n_rows == 3
        for side in ("query", "doc"):
            source_offsets = np.fromfile(source / f"{side}_offsets.bin", dtype="<u8")
            expected_tokens = int(source_offsets[3])
            assert (staged / f"{side}_offsets.bin").stat().st_size == 4 * 8
            assert (staged / f"{side}_tokens.bin").stat().st_size == expected_tokens * 2
            for row in range(3):
                assert np.array_equal(reader.get_row(side, row), rows[row][side == "doc"])

        marker = json.loads((scratch / "subsets" / "source.staged").read_text())
        assert marker["take_rows"] == 3

        # A larger tier in the same scratch root must replace, not reuse, the
        # completed smaller prefix.
        stage_in_tier(cache, scratch, tiers["small"])
        upgraded = SourceReader(staged)
        assert upgraded.n_rows == 7
        upgraded_marker = json.loads((scratch / "subsets" / "source.staged").read_text())
        assert upgraded_marker["take_rows"] == 7
