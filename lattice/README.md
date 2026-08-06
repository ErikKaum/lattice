# Lattice runtime

This crate is the deployment runtime for artifacts produced by
[`slicer`](../slicer/README.md). It provides:

- a zero-copy safetensors loader backed by `mmap`;
- the same Hugging Face tokenizer used during training;
- fp32 and packed integer mean-pooling kernels;
- `embed` and `bench` command-line binaries;
- a Rust library API;
- optional Python bindings built with `maturin`.

The runtime is designed for bulk CPU embedding. It streams newline-delimited
text and raw fp32 outputs rather than building an ANN index or defining a
network service.

## Create a model artifact first

The Git repository does not contain model binaries. Generate one from the
canonical Hugging Face fp32 model:

```bash
cd ../slicer
uv run slicer slice \
  --dim 512 \
  --quant int4_dim \
  --output-dir ../data/int4-dim-512
```

## Build

```bash
cd ../lattice
cargo build --release --bins
```

The hot loops use the portable `wide` SIMD crate and compile on normal Rust
targets. The published throughput number was measured on Apple Silicon; treat
it as an operating-point measurement, not a cross-machine CPU benchmark.

## `embed`: stream embeddings

Input is UTF-8 text with one document per line. Output is a flat
little-endian fp32 matrix.

```bash
echo "hello world" \
  | ./target/release/embed \
      --model ../data/int4-dim-512/model.safetensors \
      --output /tmp/embedding.bin
```

For a file:

```bash
./target/release/embed \
  --model ../data/int4-dim-512/model.safetensors \
  --input /path/to/documents.txt \
  --output /tmp/embeddings.bin
```

The tokenizer defaults to `tokenizer.json` next to the model. Override it with
`--tokenizer`. Embeddings are L2-normalized unless `--no-normalize` is set.

Read the output in NumPy:

```python
import numpy as np

embeddings = np.fromfile("/tmp/embeddings.bin", dtype="<f4").reshape(-1, 512)
```

Run `./target/release/embed --help` for the full interface.

## `bench`: measure the complete pipeline

```bash
./target/release/bench \
  --model ../data/int4-dim-512/model.safetensors \
  --corpus ../data/wiki/wiki-5k.txt \
  --threads 12 \
  --chunk-size 10000 \
  --no-write
```

The benchmark reports tokenization, model, write, and orchestration time. Use
`--output` to include actual fp32 output writes and `--limit` for a smaller
slice.

The headline Wikipedia run used int4-dim-512, 12 worker threads, and an 8-core
Apple M2 MacBook Air:

| Measurement | Result |
|---|---:|
| Articles | 6,407,814 |
| Wall time | 7 min 26 sec |
| Throughput | 9.52M tokens/sec |
| Tokenization share | 91.7% |

## Rust library API

```rust
use lattice::{kernel, LatticeTokenizer, Model};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let model = Model::load(Path::new(
        "data/int4-dim-512/model.safetensors",
    ))?;
    let tokenizer = LatticeTokenizer::load(Path::new(
        "data/int4-dim-512/tokenizer.json",
    ))?;

    let ids = tokenizer.encode("hello world")?;
    let mut scratch = model.scratch();
    let mut embedding = vec![0.0_f32; model.dim()];

    model.embed(&ids, &mut embedding, &mut scratch)?;
    kernel::l2_normalize(&mut embedding);

    println!("variant={} dim={}", model.variant(), model.dim());
    Ok(())
}
```

Allocate one scratch buffer per worker and reuse it. `Model` mmaps the weight
tensor, while the much smaller scale tensor is copied once at load time.

## Python bindings

```bash
uv venv .venv-py
source .venv-py/bin/activate
uv pip install maturin numpy
maturin develop --release --features python
```

```python
import numpy as np
import lattice

model = lattice.Model.load("../data/int4-dim-512/model.safetensors")
tokenizer = lattice.Tokenizer.load("../data/int4-dim-512/tokenizer.json")

texts = ["hello world", "static embeddings are fast"]
token_ids = tokenizer.encode_batch(texts)
embeddings = np.stack([
    model.embed(ids, normalize=True)
    for ids in token_ids
])

print(model.variant, model.dim)  # int4_dim 512
print(embeddings.shape)          # (2, 512)
```

The bindings call the same tokenizer and kernels as the Rust binaries.

## Supported artifacts

| Variant | Runtime support |
|---|---|
| fp32 | yes |
| int8 per-row / per-dim | yes |
| int4 per-row / per-dim | yes |
| int2 per-row | yes |
| int2 per-dim | intentionally rejected |

Int2 per-dim is excluded because its shared scales turn too many quiet token
rows into zeros and its retrieval quality is very poor.

## Tests and evaluation scripts

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
```

Unit tests that require generated model/tokenizer fixtures are marked ignored
in a clean clone. After creating the corresponding artifacts under `data/`,
run them explicitly with:

```bash
cargo test -- --ignored
```

The Python scripts under `tests/` perform heavier checks:

- `test_parity.py`: Rust embeddings versus fake-quantized Sentence
  Transformers embeddings;
- `verify_evals.py`: NanoBEIR parity against the trainer quantization sweep;
- `eval_decontam.py`: decontaminated BEIR for local deployment artifacts;
- `plot_quality_vs_throughput.py`: regenerate the aggregate CSV and plot.

They require local checkpoints/artifacts and the Python binding environment;
they are audit tools rather than clean-clone unit tests.
