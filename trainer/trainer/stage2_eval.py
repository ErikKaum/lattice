"""Stage-2 held-out in-domain evaluation surface (eval_plan.md Parts 1-2).

Two stages:

1. **Builder** (`build_eval_surface`): consumes the per-source split
   manifests from `stage2_split` and materializes a Nano-sized BEIR
   triple per source — `corpus` (eval positives + sampled distractors),
   `queries` (the 100 held-out per source), `qrels` (full coverage; every
   eval query has its positive(s) in the corpus). Pure data prep; no
   model. Output is a single JSON per source, ~few MB.

2. **Evaluator** (`evaluate_stage2_heldout`): loads the surface, wraps it
   in one `InformationRetrievalEvaluator` per source, walks the matryoshka
   dims by mutating `truncate_dim`. Mirrors the decontaminated harness's
   pattern for cheap dataset I/O. Reports per-source NDCG@10, an unweighted
   mean across sources, and a per-source-count-weighted mean.

Distractor design (per the design check before writing code): random
sample of `n_distractors` documents from `documents/{source}` per source,
plus the eval positives. No NV-Retriever 0.95 filter applied here — that
threshold is a *training* concept (decide which mined candidates to expose
as negatives during the loss step); for eval-corpus construction it doesn't
naturally apply. The calibration step (xs/small/medium ranking-preservation)
will tell us whether 3000 distractors is enough discriminative power; we
bump it if not.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .loss import DEFAULT_MATRYOSHKA_DIMS
from .stage2_split import DATASET, SOURCES, _per_source_seed, load_split

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


DEFAULT_N_DISTRACTORS = 3000  # eval_plan.md §1: "few thousand docs per source"


@dataclass(frozen=True)
class HeldoutTask:
    source: str
    corpus: dict[str, str]
    queries: dict[str, str]
    relevant_docs: dict[str, set[str]]

    @property
    def n_corpus(self) -> int:
        return len(self.corpus)

    @property
    def n_queries(self) -> int:
        return len(self.queries)

    @property
    def n_qrels(self) -> int:
        return sum(len(v) for v in self.relevant_docs.values())


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------


def build_eval_surface(
    source: str,
    splits_dir: Path,
    out_dir: Path,
    n_distractors: int = DEFAULT_N_DISTRACTORS,
    seed: int = 42,
) -> Path:
    """Build the held-out eval surface for one source and write a single
    JSON file to `out_dir/{source}.json`. Returns the path."""
    from datasets import load_dataset

    manifest = load_split(splits_dir, source)
    eval_qids = set(manifest.eval_query_ids)
    print(f"[{source}] {len(eval_qids)} held-out queries", flush=True)

    # 1. Query text.
    queries_ds = load_dataset(DATASET, "queries", split=source)
    queries_text: dict[int, str] = {}
    for row in queries_ds:
        qid = int(row["query_id"])
        if qid in eval_qids:
            queries_text[qid] = row["query"]
    if len(queries_text) != len(eval_qids):
        missing = eval_qids - set(queries_text)
        raise RuntimeError(
            f"{source}: {len(missing)} eval query_ids not found in queries config "
            f"(first few: {sorted(missing)[:5]})"
        )

    # 2. Positives — each row in scores is one (query, positive) pair;
    #    document_ids[0] / scores[0] is the positive per LightOn's
    #    convention (see README "How to use" snippet).
    scores_ds = load_dataset(DATASET, "scores", split=source)
    qrels: dict[int, set[int]] = {}
    for row in scores_ds:
        qid = int(row["query_id"])
        if qid in eval_qids:
            qrels.setdefault(qid, set()).add(int(row["document_ids"][0]))

    eval_positive_ids = {p for ps in qrels.values() for p in ps}
    print(
        f"[{source}] {len(qrels)} queries with positives; "
        f"{len(eval_positive_ids)} unique positive doc ids",
        flush=True,
    )

    # 3. Distractors — sample document_ids from the full pool, exclude positives.
    docs_ds = load_dataset(DATASET, "documents", split=source)
    n_docs = len(docs_ds)

    # We sample distractor document_ids from the document_id space — but
    # we have to read them out of `docs_ds`, which is indexed by row, not
    # by document_id. Build a numpy id-to-row lookup once (one column
    # read, no per-row Python iteration); then `.select(rows)` is O(needed).
    # This is much faster than the naive `for row in docs_ds` scan on
    # multi-million-row sources like trivia (21M) and msmarco (8.8M).
    ids_col = np.asarray(docs_ds["document_id"], dtype=np.int64)
    # Common fast-fast case: ids are 0..N-1 in order → row == document_id.
    if ids_col.size > 0 and ids_col[0] == 0 and ids_col[-1] == n_docs - 1 and (
        np.diff(ids_col[:: max(1, n_docs // 1000)]).min() > 0
    ):
        id_to_row = None  # signal "use document_id directly as row index"
    else:
        sort_idx = np.argsort(ids_col, kind="stable")
        sorted_ids = ids_col[sort_idx]
        id_to_row = (sort_idx, sorted_ids)

    # Distractor sampling — uniform over the document_id space (= rows).
    # Over-draw slightly so removing positives doesn't leave us short.
    rng = np.random.default_rng(_per_source_seed(seed * 1000 + 7, source))
    n_draw = min(n_distractors + len(eval_positive_ids) + 100, n_docs)
    drawn_rows = rng.choice(n_docs, size=n_draw, replace=False)
    drawn_ids = ids_col[drawn_rows]

    distractor_ids: list[int] = []
    for did in drawn_ids:
        d = int(did)
        if d in eval_positive_ids:
            continue
        distractor_ids.append(d)
        if len(distractor_ids) >= n_distractors:
            break

    needed_doc_ids = sorted(set(distractor_ids) | eval_positive_ids)

    if id_to_row is None:
        row_indices = needed_doc_ids
    else:
        sort_idx, sorted_ids = id_to_row
        needed_arr = np.array(needed_doc_ids, dtype=np.int64)
        positions = np.searchsorted(sorted_ids, needed_arr)
        # Guard: every needed doc_id must actually exist in the column.
        valid = (positions < sorted_ids.size) & (sorted_ids[positions.clip(max=sorted_ids.size - 1)] == needed_arr)
        if not valid.all():
            missing = needed_arr[~valid]
            raise RuntimeError(
                f"{source}: {len(missing)} needed document_ids not in documents config "
                f"(first few: {missing[:5].tolist()})"
            )
        row_indices = sort_idx[positions].tolist()

    subset = docs_ds.select(row_indices)
    corpus_text: dict[int, str] = {
        int(row["document_id"]): row["document"] for row in subset
    }

    # 4. Sanity: every positive must be in the corpus.
    missing_pos = eval_positive_ids - set(corpus_text)
    if missing_pos:
        raise RuntimeError(
            f"{source}: {len(missing_pos)} positives missing from corpus after build "
            f"(first few: {sorted(missing_pos)[:5]})"
        )

    # 5. Serialize. IDs are stringified at write time so the on-disk
    #    surface is BEIR-canonical and the loader can stay simple.
    payload = {
        "source": source,
        "meta": {
            "source": source,
            "n_corpus": len(corpus_text),
            "n_queries": len(queries_text),
            "n_qrels": sum(len(v) for v in qrels.values()),
            "n_distractors_requested": n_distractors,
            "n_distractors_realized": len(distractor_ids),
            "n_eval_positives": len(eval_positive_ids),
            "seed": seed,
            "split_seed": manifest.seed,
        },
        "corpus": {str(k): v for k, v in corpus_text.items()},
        "queries": {str(k): v for k, v in queries_text.items()},
        "qrels": {str(qid): sorted(str(c) for c in cids) for qid, cids in qrels.items()},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{source}.json"
    path.write_text(json.dumps(payload))  # no indent — these get big
    print(
        f"[{source}] wrote {path.name}: corpus={len(corpus_text)} "
        f"queries={len(queries_text)} qrels={sum(len(v) for v in qrels.values())}",
        flush=True,
    )
    return path


def build_all_eval_surfaces(
    splits_dir: Path,
    out_dir: Path,
    n_distractors: int = DEFAULT_N_DISTRACTORS,
    seed: int = 42,
    sources: tuple[str, ...] = SOURCES,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for source in sources:
        path = build_eval_surface(source, splits_dir, out_dir, n_distractors, seed)
        # Stash the meta block for the top-level manifest.
        with path.open() as f:
            data = json.load(f)
        summary[source] = data["meta"]

    (out_dir / "manifest.json").write_text(json.dumps({
        "n_distractors_per_source": n_distractors,
        "seed": seed,
        "sources": list(sources),
        "per_source": summary,
    }, indent=2))


# --------------------------------------------------------------------------
# Loader + evaluator
# --------------------------------------------------------------------------


def load_heldout_task(surface_dir: Path, source: str) -> HeldoutTask:
    """Read a previously-built surface JSON into IRE-ready dicts."""
    data = json.loads((surface_dir / f"{source}.json").read_text())
    return HeldoutTask(
        source=source,
        corpus=data["corpus"],
        queries=data["queries"],
        relevant_docs={qid: set(cids) for qid, cids in data["qrels"].items()},
    )


def _ndcg10_from_result(result: dict) -> float | None:
    for k in ("cosine_ndcg@10", "ndcg@10"):
        if k in result:
            return float(result[k])
    return None


def evaluate_stage2_heldout(
    model: "SentenceTransformer",
    surface_dir: Path,
    matryoshka_dims: tuple[int, ...] = DEFAULT_MATRYOSHKA_DIMS,
    sources: tuple[str, ...] = SOURCES,
    show_progress: bool = False,
    batch_size: int = 256,
) -> dict:
    """Run held-out in-domain eval. Per-source `cosine_ndcg@10` at every
    matryoshka dim, plus two aggregates: an unweighted mean (each source
    counts once — guards against MSMARCO swamping the mean by query
    count) and a weighted mean (by `n_queries`, the realistic mixed
    traffic number). Both are reported per eval_plan.md §1."""
    from sentence_transformers.sentence_transformer.evaluation import (
        InformationRetrievalEvaluator,
    )

    per_source: dict[str, dict] = {}
    for source in sources:
        t0 = time.time()
        task = load_heldout_task(surface_dir, source)
        print(
            f"[{source}] corpus={task.n_corpus} queries={task.n_queries} "
            f"qrels={task.n_qrels}",
            flush=True,
        )
        evaluator = InformationRetrievalEvaluator(
            queries=task.queries,
            corpus=task.corpus,
            relevant_docs=task.relevant_docs,
            show_progress_bar=show_progress,
            batch_size=batch_size,
            truncate_dim=None,
            write_csv=False,
            mrr_at_k=[10],
            ndcg_at_k=[10],
            accuracy_at_k=[10],
            precision_recall_at_k=[10],
            map_at_k=[10],
        )

        ndcg_per_dim: dict[str, float | None] = {}
        for dim in matryoshka_dims:
            evaluator.truncate_dim = dim
            result = evaluator(model)
            ndcg = _ndcg10_from_result(result)
            ndcg_per_dim[str(dim)] = ndcg
            print(
                f"  dim={dim:4d}  ndcg@10={ndcg:.4f}" if ndcg is not None
                else f"  dim={dim:4d}  ndcg@10=?",
                flush=True,
            )

        per_source[source] = {
            "n_corpus": task.n_corpus,
            "n_queries": task.n_queries,
            "n_qrels": task.n_qrels,
            "ndcg@10": ndcg_per_dim,
            "elapsed_s": time.time() - t0,
        }
        del task, evaluator
        import gc
        gc.collect()

    aggregates = _compute_aggregates(per_source, matryoshka_dims)

    return {
        "per_source": per_source,
        "aggregates": aggregates,
        "config": {
            "sources": list(sources),
            "matryoshka_dims": list(matryoshka_dims),
        },
    }


def _compute_aggregates(per_source: dict, dims: tuple[int, ...]) -> dict:
    """Per-dim {unweighted, weighted-by-query-count} means across sources."""
    unweighted: dict[str, float | None] = {}
    weighted: dict[str, float | None] = {}
    for dim in dims:
        d = str(dim)
        rows = [(s, r["ndcg@10"][d], r["n_queries"])
                for s, r in per_source.items()
                if r["ndcg@10"].get(d) is not None]
        if not rows:
            unweighted[d] = None
            weighted[d] = None
            continue
        unweighted[d] = sum(v for _, v, _ in rows) / len(rows)
        total_w = sum(w for _, _, w in rows)
        weighted[d] = sum(v * w for _, v, w in rows) / total_w if total_w else None
    return {
        "unweighted_mean_ndcg@10": unweighted,
        "weighted_mean_ndcg@10": weighted,
    }


def _print_summary(results: dict) -> None:
    dims = results["config"]["matryoshka_dims"]
    print()
    print(f"{'source':>12}  {'queries':>7}  " + "  ".join(f"d={d}" for d in dims))
    for name, row in results["per_source"].items():
        cells = "  ".join(
            f"{row['ndcg@10'][str(d)]:.4f}"
            if row['ndcg@10'].get(str(d)) is not None else "?".rjust(6)
            for d in dims
        )
        print(f"{name:>12}  {row['n_queries']:>7}  {cells}")
    print()
    for label, key in (("unweighted mean", "unweighted_mean_ndcg@10"),
                       ("weighted mean", "weighted_mean_ndcg@10")):
        agg = results["aggregates"][key]
        cells = "  ".join(
            f"{agg[str(d)]:.4f}" if agg.get(str(d)) is not None else "?".rjust(6)
            for d in dims
        )
        print(f"{label:>12}  {'':>7}  {cells}")


def evaluate_stage2_from_checkpoint(
    checkpoint_path: Path,
    tokenizer_path: Path,
    surface_dir: Path,
    out_dir: Path,
    **kwargs,
) -> dict:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import StaticEmbedding
    from tokenizers import Tokenizer

    from .train import load_checkpoint

    ckpt = load_checkpoint(checkpoint_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    static = StaticEmbedding(tokenizer, embedding_weights=ckpt["embedding.weight"])
    model = SentenceTransformer(modules=[static])

    out_dir.mkdir(parents=True, exist_ok=True)
    results = evaluate_stage2_heldout(model, surface_dir, **kwargs)
    (out_dir / "stage2_heldout.json").write_text(json.dumps(results, indent=2))
    _print_summary(results)
    return results
