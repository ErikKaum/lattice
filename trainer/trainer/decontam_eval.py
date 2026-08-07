"""Decontaminated BEIR evaluation harness.

Wraps `sentence_transformers`' `InformationRetrievalEvaluator` against the
14 `lightonai/*-decontaminated` splits. All three configs (`corpus`,
`queries`, `qrels-test`) follow the canonical BEIR triple-format so we
can plug straight in without an adapter.

Memory: `InformationRetrievalEvaluator` chunks the corpus internally
(`corpus_chunk_size=50000`), so even FEVER-scale corpora (~5M docs) don't
need to be embedded in one go. Encoding cost per matryoshka dim is one
full corpus pass — we reuse one evaluator instance per task and mutate
`truncate_dim` across dims to keep the loaded corpus/queries in memory.

Reported aggregates:
- **12-mean** (LightOn-comparable; excludes ClimateFEVER + FEVER) is the
  headline number, for direct comparison to DenseOn/LateOn published
  results.
- **14-mean** is reported for transparency.
- Per-task we report both `n_queries_scored` and `n_qrels` alongside
  ndcg@10. Query count is the clearest warning for unstable task-level
  deltas; qrel count separately describes how many positive judgments
  survive decontamination.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from .loss import DEFAULT_MATRYOSHKA_DIMS

# Canonical short name → HF dataset id.  Most follow the `<task>-decontaminated`
# pattern; `dbpedia` is the one exception (LightOn published it as
# `dbpedia-entity-decontaminated`).
TASK_DATASET_ID: dict[str, str] = {
    "arguana": "lightonai/arguana-decontaminated",
    "climate-fever": "lightonai/climate-fever-decontaminated",
    "dbpedia": "lightonai/dbpedia-entity-decontaminated",
    "fever": "lightonai/fever-decontaminated",
    "fiqa": "lightonai/fiqa-decontaminated",
    "hotpotqa": "lightonai/hotpotqa-decontaminated",
    "msmarco": "lightonai/msmarco-decontaminated",
    "nfcorpus": "lightonai/nfcorpus-decontaminated",
    "nq": "lightonai/nq-decontaminated",
    "quora": "lightonai/quora-decontaminated",
    "scidocs": "lightonai/scidocs-decontaminated",
    "scifact": "lightonai/scifact-decontaminated",
    "trec-covid": "lightonai/trec-covid-decontaminated",
    "webis-touche2020": "lightonai/webis-touche2020-decontaminated",
}

ALL_TASKS: tuple[str, ...] = tuple(TASK_DATASET_ID)

# LightOn computes their headline decontaminated average over 12 tasks,
# excluding ClimateFEVER and FEVER. We match this set so our numbers are
# directly comparable to their published DenseOn/LateOn deltas.
LIGHTON_12_EXCLUDE: tuple[str, ...] = ("climate-fever", "fever")


@dataclass(frozen=True)
class DecontamTask:
    name: str
    corpus: dict[str, str]
    queries: dict[str, str]
    relevant_docs: dict[str, set[str]]

    @property
    def n_corpus(self) -> int:
        return len(self.corpus)

    @property
    def n_qrels(self) -> int:
        return sum(len(v) for v in self.relevant_docs.values())

    @property
    def n_queries_scored(self) -> int:
        return len(self.relevant_docs)


def load_decontam_task(task: str) -> DecontamTask:
    """Load one decontaminated BEIR task into IRE-ready dicts.

    Concatenates `title + " " + text` for corpus entries (canonical BEIR
    convention; matches what the reference checkpoint was scored against
    originally). Queries use `text` only — title is empty in every
    decontaminated split we've inspected.

    Restricts the `queries` dict to qids that have at least one positive
    qrel. Queries without surviving qrels would be encoded for no reason
    (they contribute 0 to recall regardless of the model's prediction)
    and just inflate runtime — drop them at load time.
    """
    from datasets import load_dataset

    ds_id = TASK_DATASET_ID[task]
    corpus_ds = load_dataset(ds_id, "corpus", split="corpus")
    queries_ds = load_dataset(ds_id, "queries", split="queries")
    qrels_ds = load_dataset(ds_id, "qrels-test", split="test")

    # ID types vary across LightOn's decontam splits: ArguAna ships string
    # ids everywhere, SciFact has int qrels but string `_id` columns, etc.
    # Stringify all ids before using them as dict keys so we don't get the
    # silent `int 36 in {"0": ...}` → False filter that drops every query.
    corpus: dict[str, str] = {}
    for row in corpus_ds:
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        corpus[str(row["_id"])] = f"{title} {text}".strip() if title else text

    queries_all = {str(row["_id"]): (row.get("text") or "") for row in queries_ds}

    relevant_docs: dict[str, set[str]] = {}
    for row in qrels_ds:
        if int(row["score"]) > 0:
            relevant_docs.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))

    queries = {qid: queries_all[qid] for qid in relevant_docs if qid in queries_all}

    return DecontamTask(task, corpus, queries, relevant_docs)


def _ndcg10_from_result(result: dict) -> float | None:
    """Find the cosine ndcg@10 in an `InformationRetrievalEvaluator` result
    dict. Modern sentence-transformers uses `cosine_ndcg@10`; older
    versions or alternate score functions may use other keys. Try a few."""
    for k in ("cosine_ndcg@10", "ndcg@10"):
        if k in result:
            return float(result[k])
    return None


def evaluate_decontam_beir(
    model: SentenceTransformer,
    matryoshka_dims: tuple[int, ...] = DEFAULT_MATRYOSHKA_DIMS,
    tasks: tuple[str, ...] = ALL_TASKS,
    headline_exclude: tuple[str, ...] = LIGHTON_12_EXCLUDE,
    show_progress: bool = False,
    batch_size: int = 256,
    progress_path: Path | None = None,
) -> dict:
    """Run decontaminated BEIR across all matryoshka dims, per task.

    Output structure:

        {
          "per_task": {
            "arguana": {"n_corpus": ..., "n_qrels": ...,
                        "n_queries_scored": ...,
                        "ndcg@10": {"1024": 0.xxx, ..., "32": 0.xxx},
                        "elapsed_s": ...},
            ...
          },
          "aggregates": {
            "12_mean_ndcg@10": {"1024": ..., ...},  # LightOn-comparable
            "14_mean_ndcg@10": {"1024": ..., ...},  # all tasks
          },
          "config": {...}
        }
    """
    from sentence_transformers.sentence_transformer.evaluation import (
        InformationRetrievalEvaluator,
    )

    per_task = _load_partial_results(
        progress_path,
        tasks=tasks,
        matryoshka_dims=matryoshka_dims,
        headline_exclude=headline_exclude,
    )
    for task in tasks:
        if task in per_task:
            print(f"[{task}] already complete; resuming after it", flush=True)
            continue
        t0 = time.time()
        loaded = load_decontam_task(task)
        print(
            f"[{task}] corpus={loaded.n_corpus} qrels={loaded.n_qrels} "
            f"scored_queries={loaded.n_queries_scored}",
            flush=True,
        )

        evaluator = InformationRetrievalEvaluator(
            queries=loaded.queries,
            corpus=loaded.corpus,
            relevant_docs=loaded.relevant_docs,
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
                f"  dim={dim:4d}  ndcg@10={ndcg:.4f}"
                if ndcg is not None
                else f"  dim={dim:4d}  ndcg@10=?",
                flush=True,
            )

        per_task[task] = {
            "n_corpus": loaded.n_corpus,
            "n_qrels": loaded.n_qrels,
            "n_queries_scored": loaded.n_queries_scored,
            "ndcg@10": ndcg_per_dim,
            "elapsed_s": time.time() - t0,
        }
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(
                json.dumps(
                    _build_results(
                        per_task,
                        tasks=tasks,
                        matryoshka_dims=matryoshka_dims,
                        headline_exclude=headline_exclude,
                    ),
                    indent=2,
                )
            )
            print(f"[{task}] saved resumable progress -> {progress_path}", flush=True)
        # Free big corpus dict + evaluator's cached encodings before
        # loading the next task.
        del loaded, evaluator
        gc.collect()

    return _build_results(
        per_task,
        tasks=tasks,
        matryoshka_dims=matryoshka_dims,
        headline_exclude=headline_exclude,
    )


def _build_results(
    per_task: dict[str, dict],
    tasks: tuple[str, ...],
    matryoshka_dims: tuple[int, ...],
    headline_exclude: tuple[str, ...],
) -> dict:
    aggregates = {
        "12_mean_ndcg@10": _mean_per_dim(per_task, matryoshka_dims, headline_exclude),
        "14_mean_ndcg@10": _mean_per_dim(per_task, matryoshka_dims, exclude=()),
    }

    return {
        "per_task": per_task,
        "aggregates": aggregates,
        "config": {
            "tasks": list(tasks),
            "matryoshka_dims": list(matryoshka_dims),
            "headline_exclude": list(headline_exclude),
            "completed_tasks": [task for task in tasks if task in per_task],
            "is_complete": all(task in per_task for task in tasks),
        },
    }


def _load_partial_results(
    progress_path: Path | None,
    tasks: tuple[str, ...],
    matryoshka_dims: tuple[int, ...],
    headline_exclude: tuple[str, ...],
) -> dict[str, dict]:
    if progress_path is None or not progress_path.exists():
        return {}

    saved = json.loads(progress_path.read_text())
    config = saved.get("config", {})
    expected = {
        "tasks": list(tasks),
        "matryoshka_dims": list(matryoshka_dims),
        "headline_exclude": list(headline_exclude),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(
                f"partial decontam result has incompatible {key}: {config.get(key)!r} != {value!r}"
            )

    per_task = saved.get("per_task", {})
    completed = [task for task in tasks if task in per_task]
    print(
        f"resuming decontam eval from {progress_path}: "
        f"{len(completed)}/{len(tasks)} tasks complete",
        flush=True,
    )
    return {task: per_task[task] for task in completed}


def _mean_per_dim(
    per_task: dict[str, dict],
    dims: tuple[int, ...],
    exclude: tuple[str, ...],
) -> dict[str, float | None]:
    excluded = set(exclude)
    out: dict[str, float | None] = {}
    for dim in dims:
        d = str(dim)
        vals = [
            row["ndcg@10"][d]
            for name, row in per_task.items()
            if name not in excluded and row["ndcg@10"].get(d) is not None
        ]
        out[d] = (sum(vals) / len(vals)) if vals else None
    return out


def evaluate_decontam_from_checkpoint(
    checkpoint_path: Path,
    tokenizer_path: Path,
    out_dir: Path,
    **kwargs,
) -> dict:
    """Wrap our `.safetensors` checkpoint as a `SentenceTransformer`, run
    decontaminated BEIR, dump results to `out_dir / decontam_beir.json`."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import StaticEmbedding
    from tokenizers import Tokenizer

    from .train import load_checkpoint

    ckpt = load_checkpoint(checkpoint_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    static = StaticEmbedding(tokenizer, embedding_weights=ckpt["embedding.weight"])
    model = SentenceTransformer(modules=[static])

    out_dir.mkdir(parents=True, exist_ok=True)
    results = evaluate_decontam_beir(
        model,
        progress_path=out_dir / "decontam_beir.partial.json",
        **kwargs,
    )
    (out_dir / "decontam_beir.json").write_text(json.dumps(results, indent=2))
    _print_summary(results)
    return results


def evaluate_decontam_from_hub(model_id: str, out_dir: Path, **kwargs) -> dict:
    """Same, but for a Hub model id (e.g. `sentence-transformers/static-retrieval-mrl-en-v1`).
    Used to score the reference checkpoint on the same surface."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)

    out_dir.mkdir(parents=True, exist_ok=True)
    results = evaluate_decontam_beir(
        model,
        progress_path=out_dir / "decontam_beir.partial.json",
        **kwargs,
    )
    (out_dir / "decontam_beir.json").write_text(json.dumps(results, indent=2))
    _print_summary(results)
    return results


def _print_summary(results: dict) -> None:
    """Print a compact per-task table + the two aggregates."""
    per_task = results["per_task"]
    dims = results["config"]["matryoshka_dims"]
    print()
    print(f"{'task':>20}  {'qrels':>7}  " + "  ".join(f"d={d}" for d in dims))
    for name, row in per_task.items():
        cells = "  ".join(
            f"{row['ndcg@10'][str(d)]:.4f}"
            if row["ndcg@10"].get(str(d)) is not None
            else "?".rjust(6)
            for d in dims
        )
        print(f"{name:>20}  {row['n_qrels']:>7}  {cells}")

    print()
    for label, key in (("12-mean (LightOn)", "12_mean_ndcg@10"), ("14-mean", "14_mean_ndcg@10")):
        agg = results["aggregates"][key]
        cells = "  ".join(
            f"{agg[str(d)]:.4f}" if agg.get(str(d)) is not None else "?".rjust(6) for d in dims
        )
        print(f"{label:>20}  {'':>7}  {cells}")
