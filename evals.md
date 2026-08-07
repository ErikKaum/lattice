# Evaluation notes

## Stage-1 NanoBEIR

NanoBEIR is the fast scaling surface used to compare the `xs`, `small`,
`medium`, and `full` Stage-1 runs and to inspect the Matryoshka curve. The three
Stage-1 subsets with direct task overlap—`beir_dbpedia`, `msmarco`, and
`quora`—were excluded before training.

## Stage-2 held-out evaluation

The Stage-2 pipeline creates a seed-fixed, query-level held-out split from its
seven fine-tuning sources. Each source contributes at most 100 held-out
queries, evaluated against a few thousand candidates. The reported result is
the unweighted mean across sources.

This surface answers whether hard-negative fine-tuning improves the domains it
was trained on. It is not a broad generalization benchmark.

## Decontaminated BEIR

Stage 2 uses FiQA, NQ, HotpotQA, MS MARCO, FEVER, SQuAD v2, and TriviaQA, so
its absolute score on standard BEIR or NanoBEIR is contaminated. The main
generalization surface is LightOn's 14-task document-level decontaminated BEIR
release.

The headline **12-task mean** follows LightOn by excluding ClimateFEVER and
FEVER. The 14-task mean is retained as a secondary value. The suite removes
matching fine-tuning documents and queries that lose all relevant documents,
but it does not recreate the pristine original BEIR distributions. Some tasks
retain small candidate pools or few scored queries—for example, NQ has 26 and
MS MARCO 41—so the aggregate is most useful for paired model comparisons.

Stage-2 NanoBEIR remains useful for paired post-training-quantization checks
against the same fp32 checkpoint. It is not used as the release's headline
quality claim.

## Stage-1 data scaling

NanoBEIR NDCG@10 at 1,024 dimensions:

| Model | Training pairs | NanoBEIR | Decontaminated BEIR 12-mean |
|---|---:|---:|---:|
| `static-retrieval-mrl-en-v1` | ~80M | 0.5032 | 0.4334 |
| lattice xs | 10M | 0.4911 | — |
| lattice small | 100M | 0.5035 | — |
| lattice medium | 275M | 0.5143 | — |
| **lattice full** | **660M** | **0.5212** | **0.4581** |

The complete Stage-1 Matryoshka curve on NanoBEIR:

| Dimension | xs (10M) | small (100M) | medium (275M) | full (660M) | Reference |
|---|---:|---:|---:|---:|---:|
| 1024 | 0.4911 | 0.5035 | 0.5143 | 0.5212 | 0.5032 |
| 512 | 0.4872 | 0.5036 | 0.5113 | 0.5201 | — |
| 256 | 0.4808 | 0.4978 | 0.5017 | 0.5135 | — |
| 128 | 0.4526 | 0.4719 | 0.4732 | 0.4899 | — |
| 64 | 0.4087 | 0.4243 | 0.4283 | 0.4406 | — |
| 32 | 0.3420 | 0.3596 | 0.3554 | 0.3624 | — |

## Stage-2 lift

NDCG@10 at 1,024 dimensions:

| Evaluation | Stage 1 | Stage 2 | Change |
|---|---:|---:|---:|
| Held-out in-domain | 0.8107 | **0.8359** | +0.0252 |
| Decontaminated BEIR 12-mean | 0.4581 | **0.4749** | +0.0168 |
| Decontaminated BEIR 14-mean | 0.4368 | **0.4581** | +0.0213 |

The decontaminated Matryoshka results used by the release:

| Dimension | Reference 12 | Stage 1 12 | Stage 2 12 | Stage 2 14 |
|---|---:|---:|---:|---:|
| 1024 | 0.4334 | 0.4581 | **0.4749** | 0.4581 |
| 512 | 0.4297 | 0.4540 | 0.4697 | 0.4527 |
| 256 | 0.4213 | 0.4444 | 0.4624 | 0.4438 |
| 128 | 0.4101 | 0.4270 | 0.4402 | 0.4205 |
| 64 | 0.3663 | 0.3971 | 0.4148 | 0.3902 |
| 32 | 0.2991 | 0.3325 | 0.3262 | 0.3013 |

## Deployment matrix

Every row below was generated from the tokenizer-corrected Stage-2 checkpoint
and evaluated at the artifact's native dimension. Quality is the full
decontaminated BEIR 12-task mean. Sizes are decimal megabytes, matching the
blog and `data/quality_vs_throughput.csv`.

| Variant | Weight file | Decontaminated BEIR 12-mean | Throughput |
|---|---:|---:|---:|
| fp32-dim-1024 | 125.02 MB | 0.474926 | 7.55M tokens/sec |
| int8-dim-1024 | 31.26 MB | 0.474720 | 8.43M tokens/sec |
| int8-dim-512 | 15.63 MB | 0.469988 | 9.05M tokens/sec |
| fp32-dim-512 | 62.51 MB | 0.469748 | 8.81M tokens/sec |
| **int4-row-512** | **7.94 MB** | **0.469662** | **8.76M tokens/sec** |
| int4-dim-1024 | 15.63 MB | 0.465777 | 8.56M tokens/sec |
| int4-dim-512 | 7.82 MB | 0.462941 | 9.03M tokens/sec |
| int8-dim-256 | 7.82 MB | 0.462428 | 9.38M tokens/sec |
| fp32-dim-256 | 31.25 MB | 0.462411 | 9.32M tokens/sec |
| int8-row-256 | 7.94 MB | 0.462324 | 9.44M tokens/sec |
| int4-dim-256 | 3.91 MB | 0.449802 | 9.47M tokens/sec |
| fp32-dim-128 | 15.63 MB | 0.440249 | 9.52M tokens/sec |
| int8-dim-128 | 3.91 MB | 0.438987 | 9.73M tokens/sec |
| int4-dim-128 | 1.95 MB | 0.431234 | 9.43M tokens/sec |
| int2-row-512 | 4.03 MB | 0.419512 | 9.01M tokens/sec |
| int2-row-1024 | 7.94 MB | 0.418488 | 8.32M tokens/sec |
| fp32-dim-64 | 7.81 MB | 0.414774 | 9.67M tokens/sec |

The full per-task deployment results are checked in at
`data/<variant>/eval/decontam_beir.json`. Aggregate runtime inputs and outputs
are in:

- `data/throughput.csv`;
- `data/quality_vs_throughput.csv`;
- `data/quality_vs_throughput.png` and `.pdf`.

The runtime values are single measurements over the same 5,000-article corpus
with 12 workers on an 8-core Apple M2 MacBook Air. They show operating-point
trade-offs, not a hardware-independent ordering.

## Native-runtime verification

All 17 corrected artifacts load in the Rust runtime. The Python-versus-Rust
embedding parity suite passed 77/77 cases with worst maximum absolute error
`1.21e-6`.

Five representative artifacts also reproduced their corrected NanoBEIR sweep
cells within 0.001 NDCG@10:

| Variant | Rust NanoBEIR | Trainer sweep | Delta |
|---|---:|---:|---:|
| fp32-dim-1024 | 0.530600 | 0.530366 | +0.000233 |
| int4-dim-1024 | 0.529049 | 0.528847 | +0.000201 |
| int4-dim-512 | 0.524202 | 0.524000 | +0.000201 |
| int8-dim-256 | 0.518755 | 0.518553 | +0.000201 |
| int2-row-1024 | 0.495987 | 0.495774 | +0.000213 |

The corresponding checked-in files are
`data/<variant>/eval/nanobeir.json`. The audit harnesses live in
`lattice/tests/test_parity.py` and `lattice/tests/verify_evals.py`.

## Canonical local/object-store runs

The release was produced from:

- Stage 1: `tokenizer_full_20260803_modal_a100x4_r1`;
- Stage 2: `tokenizer_stage2_10ep_20260803_modal_a100x4_r1`.

The full run directories and checkpoints are deliberately not versioned. They
remain in object storage/local scratch, while this repository keeps the
maintained aggregate tables, per-variant evaluation JSON, plotting inputs, and
the exact code needed to reproduce them.
