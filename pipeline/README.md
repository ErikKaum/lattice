# Training-data pipeline

This Rust crate turns
[`lightonai/embeddings-pre-training-curated`](https://huggingface.co/datasets/lightonai/embeddings-pre-training-curated)
into a local, pre-tokenized cache designed for sequential mmap reads during
training.

The pipeline excludes `beir_dbpedia`, `msmarco`, and `quora`, creates nested
training tiers, downloads the selected parquet shards, tokenizes with
`bert-base-uncased`, and stores token IDs and row offsets as flat binaries.
Special tokens are not added to match Sentence Transformers.

## Requirements

- A recent stable Rust toolchain.
- Network access to Hugging Face while planning and tokenizing.
- Enough storage for the requested tier. The full tokenized cache is hundreds
  of gigabytes.

## Commands

Run commands from this directory:

```bash
# Fetch source metadata and write cache/partition.json + cache/tiers.json.
cargo run --release -- plan

# Materialize one nested tier. The operation is resumable.
cargo run --release -- tokenize xs

# Inspect the generated plan or cache.
cargo run --release -- inspect partition
cargo run --release -- inspect subsets
cargo run --release -- inspect subset arxiv
```

Available tiers:

| Tier | Target pairs | Intended use |
|---|---:|---|
| `xs` | 10M | End-to-end validation and quick experiments |
| `small` | 100M | First scaling point |
| `medium` | 275M | Larger scaling confirmation |
| `full` | ~660M retained rows | Final Stage-1 training |

The tiers are nested: `xs ⊂ small ⊂ medium ⊂ full`. Tokenizing a larger tier
extends the same per-source files rather than creating a second copy.

## On-disk format

The default output root is `pipeline/cache/`:

```text
cache/
├── partition.json
├── tiers.json
├── tokenizer.json
└── subsets/
    └── <source>/
        ├── query_tokens.bin
        ├── query_offsets.bin
        ├── doc_tokens.bin
        ├── doc_offsets.bin
        └── meta.json
```

Token IDs are little-endian `u16`. Offsets are little-endian `u64` with one
sentinel entry, so row `i` is exactly:

```text
tokens[offsets[i] : offsets[i + 1]]
```

The trainer reads contiguous `batch_size` windows from one source at a time
and shuffles those windows globally. This retains in-domain in-batch negatives
without turning the cache access pattern into random I/O.

## Resuming and crash safety

Per-source binaries are append-only. JSON metadata is written atomically, and
an interrupted source is truncated back to its last finalized row boundary on
the next run. Re-running the same `tokenize` command skips complete sources
and resumes incomplete ones.

Do not mix caches created with different tokenization policies. The trainer
validates the `add_special_tokens` metadata and rejects mixed legacy/current
caches.

## Using the cache with the trainer

From `trainer/`, point `--cache-root` at this directory:

```bash
cd ../trainer
uv run trainer train \
  --tier xs \
  --cache-root ../pipeline/cache \
  --out-dir runs/xs
```

On cloud machines, copy the selected tier to local SSD before training. The
trainer's `--scratch` option performs that stage-in and prevents sparse mmap
reads from running through a FUSE-mounted object-store bucket.

## Container

The included `Dockerfile` builds the CPU-only pipeline binary for Linux
x86-64. It is suitable e.g. for a Hugging Face Job with a storage bucket mounted at
`/app/cache`:

```bash
docker build --platform linux/amd64 -t <registry>/lattice-pipeline:latest .

uvx hf jobs run \
  --flavor cpu-upgrade \
  --timeout 12h \
  -s HF_TOKEN \
  -v hf://buckets/<user>/<bucket>:/app/cache \
  <registry>/lattice-pipeline:latest \
  pipeline tokenize full
```

## Development

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
```

The generated `cache/` and Rust `target/` directories are intentionally
ignored by Git.
