"""Stage-2 training-data tokenizer (eval_plan.md Sequencing §3).

Input: `lightonai/embeddings-fine-tuning` (queries / documents / scores
configs, per source) + the per-source `stage2_split.SplitManifest`.

Output, per source, under `<out_root>/stage2/training/<source>/`:

    query_tokens.bin       u16, packed
    query_offsets.bin      u64, length = n_pairs + 1     (per-pair query)
    positive_tokens.bin    u16, packed
    positive_offsets.bin   u64, length = n_pairs + 1     (per-pair positive)
    negative_tokens.bin    u16, packed
    negative_offsets.bin   u64, length = n_pairs * N_NEG + 1
    meta.json              n_pairs, n_neg_per_pair, totals, nv_threshold,
                           split_seed, tokenizer model, special-token policy

At training time, the dataloader mmaps these and for pair `i`:
    query     = query_tokens[query_offsets[i]:query_offsets[i+1]]
    positive  = positive_tokens[positive_offsets[i]:positive_offsets[i+1]]
    negative_j (j ∈ 0..N_NEG-1) =
        negative_tokens[negative_offsets[i*N_NEG + j]:negative_offsets[i*N_NEG + j + 1]]

The dataloader samples 7 of the 50 negatives per training step (random
each step, so the model sees varied hard negatives over an epoch).

The redundant-per-pair layout intentionally trades disk for runtime
simplicity: a given doc that's a negative for 30 different queries gets
its tokens written 30 times. We tokenize each unique doc *once* though
— deduplication happens at the encode step, not the emit step — so the
extra disk is just bytes, not extra CPU.

**Disjointness assertion** (eval_plan.md §3): we assert at the end of
each source that the set of query_ids actually emitted is a subset of
the training-split query_ids and has zero intersection with the eval
query_ids. The filter already enforces this by construction; the
assertion catches future regressions in the pipeline.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .stage2_split import DATASET, SOURCES, load_split


N_NEGATIVES_PER_PAIR = 50
NV_THRESHOLD = 0.95  # LightOn's example-code default. plan2 quoted 0.99;
                     # user picked 0.95 after seeing both.
ADD_SPECIAL_TOKENS = False


@dataclass(frozen=True)
class StageTwoTokenizeMeta:
    source: str
    n_pairs: int
    n_negatives_per_pair: int
    nv_threshold: float
    n_pairs_pre_filter: int
    n_pairs_dropped_insufficient_negs: int
    n_unique_queries: int
    n_unique_docs: int
    total_query_tokens: int
    total_positive_tokens: int
    total_negative_tokens: int
    tokenizer: str
    split_seed: int
    add_special_tokens: bool


# --------------------------------------------------------------------------
# Pair filtering (NV-Retriever)
# --------------------------------------------------------------------------


def _filter_pairs(
    source: str,
    train_qids: set[int],
    eval_qids: set[int],
    nv_threshold: float,
    n_neg: int,
) -> tuple[list[tuple[int, int, list[int]]], int, int]:
    """Walk `scores/{source}`, keep pairs whose query is in `train_qids`
    and which have at least `n_neg` mined candidates with score
    `< nv_threshold * positive_score`. Returns the surviving pairs plus
    counts for diagnostics.

    Each scores row is one `(query, positive)` pair. `document_ids[0]`
    is the positive (LightOn convention); `document_ids[1:]` are the
    2047 mined candidates with `scores[1:]` as their teacher reranker
    similarity to the query.
    """
    from datasets import load_dataset

    # Disjointness check on the splits themselves (not on the stream — eval
    # qids legitimately appear in the scores config; we just don't keep
    # their pairs). If train_qids and eval_qids overlap, stage2_split is
    # broken upstream and we should fail loud here.
    overlap = train_qids & eval_qids
    if overlap:
        raise AssertionError(
            f"{source}: train_qids and eval_qids overlap on {len(overlap)} ids "
            f"(first few: {sorted(overlap)[:5]}). stage2_split is broken."
        )

    scores_ds = load_dataset(DATASET, "scores", split=source)
    pairs: list[tuple[int, int, list[int]]] = []
    n_pre = 0
    n_dropped = 0

    for row in scores_ds:
        qid = int(row["query_id"])
        if qid not in train_qids:
            continue  # silently skip eval qids and any other unknowns
        n_pre += 1

        document_ids = row["document_ids"]
        scores = row["scores"]
        if not document_ids or not scores:
            n_dropped += 1
            continue

        positive_id = int(document_ids[0])
        threshold = nv_threshold * float(scores[0])

        # Collect (score, doc_id) pairs that pass the threshold.
        eligible: list[tuple[float, int]] = [
            (float(s), int(d))
            for d, s in zip(document_ids[1:], scores[1:])
            if float(s) < threshold
        ]
        if len(eligible) < n_neg:
            n_dropped += 1
            continue

        # Highest-scoring eligible (= hardest legitimate negatives) first.
        eligible.sort(reverse=True)
        neg_ids = [d for _, d in eligible[:n_neg]]

        pairs.append((qid, positive_id, neg_ids))

    return pairs, n_pre, n_dropped


# --------------------------------------------------------------------------
# Doc text lookup via numpy id-to-row (same pattern as stage2_eval.py)
# --------------------------------------------------------------------------


def _build_doc_lookup(docs_ds) -> tuple[np.ndarray, np.ndarray]:
    """Return `(sorted_ids, sort_idx)` for binary-searching doc_id → row."""
    ids_col = np.asarray(docs_ds["document_id"], dtype=np.int64)
    sort_idx = np.argsort(ids_col, kind="stable")
    sorted_ids = ids_col[sort_idx]
    return sorted_ids, sort_idx


def _resolve_doc_rows(
    needed_doc_ids: list[int],
    sorted_ids: np.ndarray,
    sort_idx: np.ndarray,
    source: str,
) -> np.ndarray:
    needed = np.array(needed_doc_ids, dtype=np.int64)
    positions = np.searchsorted(sorted_ids, needed)
    valid = (positions < sorted_ids.size) & (
        sorted_ids[positions.clip(max=sorted_ids.size - 1)] == needed
    )
    if not valid.all():
        missing = needed[~valid]
        raise RuntimeError(
            f"{source}: {len(missing)} needed document_ids not present "
            f"(first few: {missing[:5].tolist()})"
        )
    return sort_idx[positions]


# --------------------------------------------------------------------------
# Tokenize + pack
# --------------------------------------------------------------------------


def _tokenize_and_pack(
    tokenizer, texts: list[str], chunk_size: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode `texts` and pack into `(concat, offsets)` chunk-by-chunk,
    releasing each chunk's `Encoding` objects before the next chunk runs.

    For sources with millions of unique docs (e.g. trivia with ~10M),
    holding all `Encoding` objects between encode and pack would push
    peak memory into the tens of GB — each `Encoding` carries a Python
    list of ~100 ints which costs ~3 KB. Streaming caps peak at one
    chunk's worth (~150 MB for chunk_size=50K) plus the growing concat
    list of numpy arrays.
    """
    pack_chunks: list[np.ndarray] = []
    offsets_list: list[int] = [0]
    last = 0
    for i in range(0, len(texts), chunk_size):
        encs = tokenizer.encode_batch_fast(
            texts[i : i + chunk_size],
            add_special_tokens=ADD_SPECIAL_TOKENS,
        )
        # Pack this chunk into a single numpy array.
        ids_lists = [enc.ids for enc in encs]
        del encs  # free the Encoding objects before allocating the array
        lengths = [len(ids) for ids in ids_lists]
        total = sum(lengths)
        if total:
            arr = np.empty(total, dtype=np.uint16)
            pos = 0
            for ids in ids_lists:
                n = len(ids)
                arr[pos : pos + n] = ids  # numpy assignment from Python list
                pos += n
            pack_chunks.append(arr)
        for n in lengths:
            last += n
            offsets_list.append(last)
        del ids_lists

    concat = (
        np.concatenate(pack_chunks) if pack_chunks
        else np.empty(0, dtype=np.uint16)
    )
    offsets = np.array(offsets_list, dtype=np.uint64)
    return concat, offsets


def _write_bin(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(arr.tobytes())


def _atomic_write_json(path: Path, payload) -> None:
    """meta.json is the marker of a finished source — write atomically
    so a crash mid-write can't leave a half-finished JSON that the
    loader trusts. Same pattern as the Rust pipeline's `atomic_write`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


# --------------------------------------------------------------------------
# Per-source tokenize
# --------------------------------------------------------------------------


def tokenize_stage2_source(
    source: str,
    splits_dir: Path,
    tokenizer_path: Path,
    out_root: Path,
    nv_threshold: float = NV_THRESHOLD,
    n_neg: int = N_NEGATIVES_PER_PAIR,
) -> StageTwoTokenizeMeta:
    """Build the stage-2 training binaries for one source.

    Order: filter pairs (cheap), gather unique IDs (cheap), bulk encode
    each unique query and each unique doc *once* (the expensive step),
    then walk pairs and emit the per-pair binaries with the packed
    encodings.
    """
    from datasets import load_dataset
    from tokenizers import Tokenizer

    t_total = time.time()
    out_dir = out_root / "stage2" / "training" / source
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Pair filter ---------------------------------------------------
    print(f"[{source}] filtering pairs (nv_threshold={nv_threshold}, n_neg={n_neg})", flush=True)
    manifest = load_split(splits_dir, source)
    train_qids = set(manifest.train_query_ids)
    eval_qids = set(manifest.eval_query_ids)

    t0 = time.time()
    pairs, n_pre, n_dropped = _filter_pairs(source, train_qids, eval_qids, nv_threshold, n_neg)
    print(
        f"[{source}] pairs: kept={len(pairs)} of {n_pre} "
        f"(dropped {n_dropped} with <{n_neg} eligible negatives) "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )
    if not pairs:
        raise RuntimeError(f"{source}: no pairs survived filter (check threshold/data)")

    # ---- 2. Unique IDs ----------------------------------------------------
    unique_qids = sorted({p[0] for p in pairs})
    unique_doc_ids = sorted({p[1] for p in pairs} | {n for p in pairs for n in p[2]})
    print(
        f"[{source}] unique queries={len(unique_qids)}  "
        f"unique docs={len(unique_doc_ids)} (positives + 50×negatives, deduped)",
        flush=True,
    )

    # Disjointness assertion #1 — every emitted query is in train, none in eval.
    emitted_qids = set(unique_qids)
    if emitted_qids & eval_qids:
        bad = emitted_qids & eval_qids
        raise AssertionError(
            f"{source}: {len(bad)} eval qids leaked into emitted set "
            f"(first few: {sorted(bad)[:5]}). stage2_split → tokenize contract broken."
        )
    if not emitted_qids.issubset(train_qids):
        bad = emitted_qids - train_qids
        raise AssertionError(
            f"{source}: {len(bad)} emitted qids not in train_qids "
            f"(first few: {sorted(bad)[:5]})"
        )

    # ---- 3. Load texts ----------------------------------------------------
    t0 = time.time()
    queries_ds = load_dataset(DATASET, "queries", split=source)
    train_qid_set = train_qids  # alias for clarity
    query_text: dict[int, str] = {}
    for row in queries_ds:
        qid = int(row["query_id"])
        if qid in train_qid_set:
            query_text[qid] = row["query"]
    print(f"[{source}] loaded {len(query_text)} query texts in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    docs_ds = load_dataset(DATASET, "documents", split=source)
    sorted_ids, sort_idx = _build_doc_lookup(docs_ds)
    row_indices = _resolve_doc_rows(unique_doc_ids, sorted_ids, sort_idx, source)
    docs_subset = docs_ds.select(row_indices.tolist())
    # Materialize in unique_doc_ids order so doc_rank == index.
    doc_text_list: list[str] = [row["document"] for row in docs_subset]
    print(
        f"[{source}] loaded {len(doc_text_list)} doc texts in {time.time() - t0:.1f}s",
        flush=True,
    )

    # ---- 4. Bulk tokenize unique queries + unique docs --------------------
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    t0 = time.time()
    q_texts_in_order = [query_text[qid] for qid in unique_qids]
    q_concat, q_offsets_unique = _tokenize_and_pack(tokenizer, q_texts_in_order)
    q_rank = {qid: i for i, qid in enumerate(unique_qids)}
    print(
        f"[{source}] tokenized {len(unique_qids)} queries "
        f"({int(q_offsets_unique[-1])} tokens) in {time.time() - t0:.1f}s",
        flush=True,
    )

    t0 = time.time()
    d_concat, d_offsets_unique = _tokenize_and_pack(tokenizer, doc_text_list)
    d_rank = {did: i for i, did in enumerate(unique_doc_ids)}
    print(
        f"[{source}] tokenized {len(unique_doc_ids)} docs "
        f"({int(d_offsets_unique[-1])} tokens) in {time.time() - t0:.1f}s",
        flush=True,
    )

    # ---- 5. Walk pairs, emit per-pair binaries ----------------------------
    t0 = time.time()
    n_pairs = len(pairs)
    n_neg_total = n_pairs * n_neg

    # Pre-compute total sizes for one-pass allocation.
    total_query_tokens = sum(
        int(q_offsets_unique[q_rank[qid] + 1] - q_offsets_unique[q_rank[qid]])
        for qid, _, _ in pairs
    )
    total_pos_tokens = sum(
        int(d_offsets_unique[d_rank[pid] + 1] - d_offsets_unique[d_rank[pid]])
        for _, pid, _ in pairs
    )
    total_neg_tokens = 0
    for _, _, neg_ids in pairs:
        for nid in neg_ids:
            r = d_rank[nid]
            total_neg_tokens += int(d_offsets_unique[r + 1] - d_offsets_unique[r])

    q_tokens_out = np.empty(total_query_tokens, dtype=np.uint16)
    p_tokens_out = np.empty(total_pos_tokens, dtype=np.uint16)
    n_tokens_out = np.empty(total_neg_tokens, dtype=np.uint16)
    q_offsets_out = np.empty(n_pairs + 1, dtype=np.uint64)
    p_offsets_out = np.empty(n_pairs + 1, dtype=np.uint64)
    n_offsets_out = np.empty(n_neg_total + 1, dtype=np.uint64)

    q_off = 0
    p_off = 0
    n_off = 0
    q_offsets_out[0] = 0
    p_offsets_out[0] = 0
    n_offsets_out[0] = 0
    neg_idx = 0

    for i, (qid, pid, neg_ids) in enumerate(pairs):
        # Query.
        r = q_rank[qid]
        s = int(q_offsets_unique[r])
        e = int(q_offsets_unique[r + 1])
        q_tokens_out[q_off : q_off + (e - s)] = q_concat[s:e]
        q_off += e - s
        q_offsets_out[i + 1] = q_off

        # Positive.
        r = d_rank[pid]
        s = int(d_offsets_unique[r])
        e = int(d_offsets_unique[r + 1])
        p_tokens_out[p_off : p_off + (e - s)] = d_concat[s:e]
        p_off += e - s
        p_offsets_out[i + 1] = p_off

        # 50 negatives.
        for nid in neg_ids:
            r = d_rank[nid]
            s = int(d_offsets_unique[r])
            e = int(d_offsets_unique[r + 1])
            n_tokens_out[n_off : n_off + (e - s)] = d_concat[s:e]
            n_off += e - s
            neg_idx += 1
            n_offsets_out[neg_idx] = n_off

    print(f"[{source}] emitted per-pair binaries in {time.time() - t0:.1f}s", flush=True)

    # ---- 6. Write files ---------------------------------------------------
    _write_bin(out_dir / "query_tokens.bin", q_tokens_out)
    _write_bin(out_dir / "query_offsets.bin", q_offsets_out)
    _write_bin(out_dir / "positive_tokens.bin", p_tokens_out)
    _write_bin(out_dir / "positive_offsets.bin", p_offsets_out)
    _write_bin(out_dir / "negative_tokens.bin", n_tokens_out)
    _write_bin(out_dir / "negative_offsets.bin", n_offsets_out)

    meta = StageTwoTokenizeMeta(
        source=source,
        n_pairs=n_pairs,
        n_negatives_per_pair=n_neg,
        nv_threshold=nv_threshold,
        n_pairs_pre_filter=n_pre,
        n_pairs_dropped_insufficient_negs=n_dropped,
        n_unique_queries=len(unique_qids),
        n_unique_docs=len(unique_doc_ids),
        total_query_tokens=int(q_off),
        total_positive_tokens=int(p_off),
        total_negative_tokens=int(n_off),
        tokenizer="bert-base-uncased",
        split_seed=manifest.seed,
        add_special_tokens=ADD_SPECIAL_TOKENS,
    )
    _atomic_write_json(out_dir / "meta.json", dataclasses.asdict(meta))

    print(
        f"[{source}] DONE  n_pairs={n_pairs}  "
        f"q_tokens={q_off}  p_tokens={p_off}  n_tokens={n_off}  "
        f"({time.time() - t_total:.1f}s total)",
        flush=True,
    )
    return meta


def tokenize_all(
    splits_dir: Path,
    tokenizer_path: Path,
    out_root: Path,
    sources: tuple[str, ...] = SOURCES,
    nv_threshold: float = NV_THRESHOLD,
    n_neg: int = N_NEGATIVES_PER_PAIR,
) -> None:
    """Tokenize every source sequentially. Idempotent on the source-level —
    skips sources whose meta.json already exists (atomic-write means it's
    a reliable completion marker)."""
    out_root.mkdir(parents=True, exist_ok=True)
    existing_policies: dict[str, bool] = {}
    missing_sources: list[str] = []
    for source in sources:
        meta_path = out_root / "stage2" / "training" / source / "meta.json"
        if not meta_path.exists():
            missing_sources.append(source)
            continue
        existing = json.loads(meta_path.read_text())
        policy = existing.get("add_special_tokens", True)
        if not isinstance(policy, bool):
            raise ValueError(
                f"{meta_path}: add_special_tokens must be boolean, got {policy!r}"
            )
        existing_policies[source] = policy

    distinct_policies = set(existing_policies.values())
    if len(distinct_policies) > 1:
        raise RuntimeError(
            "existing stage-2 sources mix add_special_tokens policies: "
            f"{existing_policies}"
        )
    if distinct_policies == {True} and missing_sources:
        raise RuntimeError(
            "refusing to combine legacy special-token sources with newly "
            "tokenized no-special-token sources; use a fresh output root or "
            f"finish with the old code (missing: {missing_sources})"
        )

    summary: dict[str, dict] = {}
    for source in sources:
        meta_path = out_root / "stage2" / "training" / source / "meta.json"
        if meta_path.exists():
            existing = json.loads(meta_path.read_text())
            existing_policy = existing.get("add_special_tokens", True)
            policy_note = (
                "legacy cache; trainer compatibility mode will ignore "
                "[CLS]/[SEP]"
                if existing_policy != ADD_SPECIAL_TOKENS
                else "canonical token policy"
            )
            print(
                f"[{source}] skip — meta.json present ({policy_note})",
                flush=True,
            )
            summary[source] = existing
            continue
        meta = tokenize_stage2_source(
            source, splits_dir, tokenizer_path, out_root,
            nv_threshold=nv_threshold, n_neg=n_neg,
        )
        summary[source] = dataclasses.asdict(meta)

    resolved_policies = {
        meta.get("add_special_tokens", True) for meta in summary.values()
    }
    if len(resolved_policies) != 1:
        raise RuntimeError(
            "stage-2 output mixes add_special_tokens policies: "
            f"{resolved_policies}"
        )
    resolved_policy = resolved_policies.pop()

    _atomic_write_json(
        out_root / "stage2" / "training" / "manifest.json",
        {
            "nv_threshold": nv_threshold,
            "n_negatives_per_pair": n_neg,
            "add_special_tokens": resolved_policy,
            "sources": list(sources),
            "per_source": summary,
        },
    )
