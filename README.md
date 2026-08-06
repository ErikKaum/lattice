<div align="center">

<img src="https://cdn-uploads.huggingface.co/production/uploads/63148c4db031f7b1c7bc36f9/O83e3asl0ZPKpudwgkZjt.png" alt="lattice banner" />

---

# Lattice

Lattice is an end-to-end project for training, quantizing, and serving a static
English retrieval model. The final model is a single 30,522 × 1,024 embedding
table: tokenize, look up one row per token, mean-pool, and L2-normalize.

The project includes the large-scale preprocessing and training pipeline, a
post-training slicer/quantizer, and a pure-Rust inference runtime.

- [Model on Hugging Face](https://huggingface.co/erikkaum/lattice-retrieval)
- [Engineering write-up](blog_revised.md)
- [Evaluation methodology and full tables](evals.md)

## Results

The final model was:

- trained on roughly **660M curated query/document pairs**.
- after a fine-tune, reached **0.4749 NDCG@10** on the 12-task decontaminated
  BEIR mean, compared with 0.4334 for `static-retrieval-mrl-en-v1`.
- An int4 quantized model with 512 dimensions is only a 7.94 MB artifact and
  scores 0.4697 NDCG@10.
- The Rust runtime embedds **6.4M English Wikipedia articles in 7 minutes 26
  seconds** on an 8-core Apple M2 MacBook Air.

## Quickstart

### Use the fp32 model with Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("erikkaum/lattice-retrieval")
embeddings = model.encode(
    ["hello world", "static embeddings are fast"],
    normalize_embeddings=True,
)
print(embeddings.shape)  # (2, 1024)
```

### Generate a quantized artifact and run it in Rust

Requirements: Python 3.12+, [`uv`](https://docs.astral.sh/uv/), and a recent
Rust toolchain.

```bash
git clone https://github.com/ErikKaum/lattice
cd lattice/slicer

# Downloads the fp32 model from Hugging Face, slices (in this case) to 512 dimensions,
# and writes an int4 per-row deployment artifact locally.
uv run slicer slice \
  --dim 512 \
  --quant int4_row \
  --output-dir ../data/int4-row-512

cd ../lattice
cargo build --release --bin embed

echo "hello world" \
  | ./target/release/embed \
      --model ../data/int4-row-512/model.safetensors \
      --output /tmp/embedding.bin
```

The output is a flat little-endian fp32 matrix with one 512-dimensional row per
input line. The runtime also exposes a Rust library API and optional Python
bindings; see [`lattice/`](lattice/README.md).

## Repository layout

| Directory                         | Purpose                                                                                                                  |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| [`pipeline/`](pipeline/README.md) | Rust preprocessing pipeline for downloading, partitioning, and tokenizing the Stage-1 corpus into mmap-friendly binaries |
| [`trainer/`](trainer/README.md)   | Stage-1 and Stage-2 training, NanoBEIR/decontaminated-BEIR evaluation, quantization sweeps, and Hugging Face export      |
| [`slicer/`](slicer/README.md)     | Python CLI that downloads or reads the fp32 checkpoint and emits one sliced/quantized deployment artifact                |
| [`lattice/`](lattice/README.md)   | Pure-Rust model loader, tokenizer, SIMD kernels, CLI, benchmark binary, and Python bindings                              |
| [`data/`](data/README.md)         | Checked-in evaluation summaries and plots; generated weights and corpora stay local                                      |
| [`scripts/`](scripts/README.md)   | Reproduction utilities for Modal, Wikipedia preparation, and weight analysis                                             |

## Reproducing the project

The components are intentionally separable:

1. Run `pipeline plan`, then `pipeline tokenize <tier>` to build the
   pre-tokenized cache.
2. Use `trainer train` for Stage 1.
3. Build the held-out Stage-2 surfaces, tokenize the hard-negative data, and run
   `trainer stage2-train`.
4. Run NanoBEIR, held-out, and decontaminated-BEIR evaluations.
5. Export the final fp32 checkpoint to the Hugging Face layout.
6. Generate deployment variants with `slicer`, then validate and benchmark them
   with `lattice`.

The component READMEs contain exact commands. [`evals.md`](evals.md) explains
which results are comparable, why Stage-2 NanoBEIR is contaminated, and where
the aggregate numbers come from.

## Artifact policy

The Git repository keeps code, documentation, evaluation JSON, aggregate CSVs,
and the published plots. It does not include:

- training checkpoints or anything under `trainer/runs/`;
- generated `model.safetensors` or duplicate tokenizer files;
- pre-tokenized training caches;
- generated quantized deployment variants.

## Development

Each package has its own environment and test commands:

```bash
(cd pipeline && cargo test)
(cd trainer && uv run --extra dev pytest)
(cd slicer && uv run python -m compileall src)
(cd lattice && cargo test)
```

Some end-to-end runtime parity/evaluation scripts require locally generated
artifacts and are documented separately in [`lattice/`](lattice/README.md).

## Acknowledgements

The training data, Stage-2 recipe, and decontaminated evaluation splits come
from LightOn's
[DenseOn/LateOn release](https://huggingface.co/blog/lightonai/denseon-lateon).
The model architecture and original static-retrieval recipe come from Sentence
Transformers'
[`static-retrieval-mrl-en-v1`](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1).
