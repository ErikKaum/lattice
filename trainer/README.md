# Trainer

This Python package contains the complete training and evaluation stack for
`lattice-retrieval`:

- Stage-1 contrastive training over the mmap cache produced by `pipeline/`;
- Matryoshka loss at dimensions 1,024, 512, 256, 128, 64, and 32;
- Stage-2 hard-negative splitting, tokenization, fine-tuning, and held-out
  evaluation;
- NanoBEIR, decontaminated BEIR, and post-training-quantization sweeps;
- export of a final checkpoint to a standard Sentence Transformers model.

Training outputs are written under `trainer/runs/` by convention. That entire
directory is local/object-store state and is intentionally ignored by Git.

## Setup

[`uv`](https://docs.astral.sh/uv/) is the supported environment manager.

```bash
cd trainer
uv sync --extra dev
uv run trainer --help
```

The package supports Python 3.10+. GPU training uses the PyTorch build
appropriate for your CUDA environment; the included Dockerfile pins the image
used for the A100 runs.

## Inputs

Stage 1 expects the layout produced by [`pipeline/`](../pipeline/README.md):

```text
../pipeline/cache/
├── tiers.json
├── tokenizer.json
└── subsets/<source>/{query,doc}_{tokens,offsets}.bin
```

Commands below assume they are run from `trainer/` and use
`../pipeline/cache` as the cache root.

## Stage 1

Single-process smoke or local run:

```bash
uv run trainer train \
  --tier xs \
  --cache-root ../pipeline/cache \
  --out-dir runs/xs \
  --batch-size 2048 \
  --epochs 1
```

Four-GPU DDP:

```bash
uv run torchrun --nproc_per_node 4 -m trainer.cli train \
  --tier full \
  --cache-root ../pipeline/cache \
  --scratch /local-ssd/lattice-cache \
  --out-dir runs/full \
  --batch-size 2048 \
  --epochs 1
```

`--scratch` copies only the selected tier to local storage before mmap. Use it
whenever `--cache-root` is a FUSE-mounted bucket; random mmap reads over a
remote mount are much slower than staging once to local SSD.

The final checkpoint is `runs/<name>/final.safetensors`. It contains the
single `embedding.weight` tensor and enough metadata to reconstruct the model.

## Stage 2

Stage 2 uses LightOn's seven-source hard-negative release. The workflow is
split into explicit, resumable steps.

### 1. Build the deterministic train/eval split

```bash
uv run trainer stage2-split \
  --out-dir ../pipeline/cache/stage2/splits \
  --seed 42
```

### 2. Materialize the held-out retrieval surfaces

```bash
uv run trainer stage2-build-eval \
  --splits-dir ../pipeline/cache/stage2/splits \
  --out-dir ../pipeline/cache/stage2/eval_surface
```

### 3. Tokenize the training split

```bash
uv run trainer stage2-tokenize \
  --splits-dir ../pipeline/cache/stage2/splits \
  --out-root ../pipeline/cache \
  --cache-root ../pipeline/cache \
  --nv-threshold 0.95 \
  --n-negatives 50
```

### 4. Fine-tune

```bash
uv run trainer stage2-train \
  --init-from runs/full/final.safetensors \
  --training-root ../pipeline/cache/stage2/training \
  --out-dir runs/stage2-10ep \
  --batch-size 256 \
  --epochs 10 \
  --lr 2e-2 \
  --n-neg-sample 7
```

For DDP, put `torchrun --nproc_per_node <N> -m trainer.cli` before the
`stage2-train` subcommand. `--scratch` has the same local-stage-in purpose as
in Stage 1.

## Evaluation

### NanoBEIR at all Matryoshka dimensions

```bash
uv run trainer eval \
  runs/stage2-10ep/final.safetensors \
  --cache-root ../pipeline/cache \
  --out-dir runs/stage2-10ep/eval
```

### Post-training quantization sweep on NanoBEIR

```bash
uv run trainer eval-quant \
  runs/stage2-10ep/final.safetensors \
  --cache-root ../pipeline/cache \
  --out-dir runs/stage2-10ep/quant-sweep
```

### Decontaminated BEIR

```bash
uv run trainer eval-beir \
  runs/stage2-10ep/final.safetensors \
  --cache-root ../pipeline/cache \
  --out-dir runs/stage2-10ep/decontam
```

Reference a published Hub model instead of a local checkpoint:

```bash
uv run trainer eval-beir-hub \
  sentence-transformers/static-retrieval-mrl-en-v1 \
  --out-dir runs/reference/decontam
```

### Stage-2 held-out surface

```bash
uv run trainer eval-stage2 \
  runs/stage2-10ep/final.safetensors \
  --cache-root ../pipeline/cache \
  --surface-dir ../pipeline/cache/stage2/eval_surface \
  --out-dir runs/stage2-10ep/heldout
```

See [`../evals.md`](../evals.md) before interpreting these numbers. In
particular, the Stage-2 data overlaps standard BEIR/NanoBEIR tasks, so the
absolute Stage-2 NanoBEIR score is not used as a generalization claim.

## Exporting the fp32 model for Hugging Face

The Hub export is the only model artifact intended for publication. It wraps
the final checkpoint in Sentence Transformers' normal `StaticEmbedding`
layout and copies the repository's model card.

```bash
uv run trainer export-hf \
  runs/stage2-10ep/final.safetensors \
  --tokenizer ../pipeline/cache/tokenizer.json \
  --out-dir ../hf_model
```

The command refuses to write into a non-empty directory. Inspect the result
locally before running the separate `hf upload` command documented in the
[root README](../README.md#exporting-the-canonical-hugging-face-model).

## Cloud reproduction

- `Dockerfile` builds the x86-64 CUDA trainer image used on Hugging Face Jobs.
- [`../scripts/modal_train.py`](../scripts/modal_train.py) reproduces the
  tokenizer-corrected Stage-1 and Stage-2 runs on Modal while using the
  Hugging Face storage bucket as durable input/output storage.

The Modal launcher names secrets but does not contain their values. See
[`../scripts/README.md`](../scripts/README.md) for setup and commands.

## Development

```bash
uv run --extra dev ruff check trainer tests
uv run --extra dev pytest
```

The synthetic tests create tiny caches under pytest's temporary directory;
they do not require the 660M-pair corpus or a real checkpoint.
