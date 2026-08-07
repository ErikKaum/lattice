# Utility scripts

These scripts support reproducibility and analysis around the four main
packages. They are not imported by the runtime.

| Script | Purpose |
|---|---|
| `modal_train.py` | Reproduce the Stage-1/Stage-2 training and evaluation jobs on Modal |
| `prepare_wikipedia.py` | Stream an English Wikipedia snapshot into one-article-per-line text for the Rust benchmark |
| `weight_variance.py` | Analyze row/dimension dynamic ranges and silent rows under low-bit per-dim quantization |

Run commands below from the repository root.

## Modal reproduction

`modal_train.py` keeps the canonical pre-tokenized cache and durable run
outputs in the Hugging Face storage bucket, but stages training inputs onto
Modal's local SSD before starting DDP. This avoids sparse mmap reads through a
remote bucket mount.

The script expects these Modal secrets:

- `huggingface-token`, containing `HF_TOKEN`;
- `hf-bucket-s3`, containing `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION`.

Secret values are configured in Modal and are never stored in this repository.
The exact validated trainer image digest is a constant near the top of the
script. Configure the Hugging Face account and bucket locally before running
the launcher:

```bash
export LATTICE_HF_NAMESPACE=your-hf-username
export LATTICE_HF_BUCKET=your-bucket-name
```

These values are used to derive both the native Hugging Face bucket URI and its
S3-compatible endpoint; they are not stored in the repository.

First prove read/write compatibility without allocating a GPU:

```bash
modal run scripts/modal_train.py::bucket_smoke
```

Then launch the canonical training stages:

```bash
modal run --detach --timestamps scripts/modal_train.py::train_full
modal run --detach --timestamps scripts/modal_train.py::train_stage2
```

Evaluation entry points are separate so they can be re-run without retraining:

```bash
modal run scripts/modal_train.py::eval_stage1_heldout
modal run scripts/modal_train.py::eval_stage1_decontam
modal run scripts/modal_train.py::eval_stage2_decontam
modal run scripts/modal_train.py::eval_stage2_quant
modal run scripts/modal_train.py::eval_deployment_int8_dim
modal run scripts/modal_train.py::eval_deployment_int4_dim
modal run scripts/modal_train.py::eval_deployment_row
```

The launcher uploads checkpoints and evaluation results back to the bucket.
Nothing under local `trainer/runs/` is required for the remote jobs.

## Wikipedia benchmark corpus

Create the 5,000-article development corpus:

```bash
uv run --with datasets python scripts/prepare_wikipedia.py \
  --limit 5000 \
  --out data/wiki/wiki-5k.txt
```

Create the complete `20231101.en` snapshot used for the headline benchmark:

```bash
uv run --with datasets python scripts/prepare_wikipedia.py \
  --out data/wiki/wiki-full.txt
```

The script uses streaming mode, flattens internal newlines/tabs, and writes one
article per line. Use `--min-chars` to drop very short articles or `--config`
to choose a different Wikimedia snapshot.

The resulting corpus is large and is ignored by Git.

## Weight-distribution analysis

```bash
uv run --with numpy --with safetensors python scripts/weight_variance.py \
  trainer/runs/stage2-10ep/final.safetensors
```

The output reports percentile summaries for token rows and embedding
dimensions, followed by the number of token rows that become entirely zero
under int8/int4/int3/int2 per-dim quantization. It reads the checkpoint without
modifying it.
