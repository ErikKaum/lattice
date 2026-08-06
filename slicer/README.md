# Slicer

`slicer` converts the canonical fp32 `lattice-retrieval` model into one local
deployment artifact. Each invocation chooses a Matryoshka dimension and a
quantization recipe; generated artifacts are intentionally not stored in Git
or published as a matrix of Hugging Face models.

By default, the CLI downloads
[`erikkaum/lattice-retrieval`](https://huggingface.co/erikkaum/lattice-retrieval)
from the Hub. It can also read a local training checkpoint.

## Setup

Requirements: Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
cd slicer
uv sync
uv run slicer --help
```

For a persistent command in your current environment:

```bash
uv tool install .
slicer --help
```

## Create an artifact from the Hub model

```bash
uv run slicer slice \
  --dim 512 \
  --quant int4_row \
  --output-dir ../data/int4-row-512
```

This downloads the fp32 source weights and tokenizer through the normal
Hugging Face cache, then writes:

```text
data/int4-row-512/
├── model.safetensors
└── tokenizer.json
```

The command reloads and dequantizes the generated file as a round-trip sanity
check before returning.

## Create an artifact from a local checkpoint

```bash
uv run slicer slice \
  --dim 256 \
  --quant int8_dim \
  --source ../trainer/runs/stage2-10ep/final.safetensors \
  --output-dir ../data/int8-dim-256
```

When `--source` is provided, `tokenizer.json` must sit next to the checkpoint.
Pass `--omit-tokenizer` if the consumer manages tokenization separately.

Alternative Hub repositories and revisions can be selected with
`--source-repo` and `--source-revision`.

## Supported recipes

Dimensions: `32`, `64`, `128`, `256`, `512`, and `1024`.

| `--quant` | Storage | Scale layout |
|---|---|---|
| `fp32` | 32-bit float | none |
| `int8_row` | signed int8 | one scale per token row |
| `int8_dim` | signed int8 | one scale per embedding dimension |
| `int4_row` | two packed codes per byte | one scale per token row |
| `int4_dim` | two packed codes per byte | one scale per embedding dimension |
| `int2_row` | four packed codes per byte | one scale per token row |
| `int2_dim` | four packed codes per byte | one scale per embedding dimension |

Slicing happens before scales are calculated. A 256-dimensional artifact is
therefore quantized against the first 256 columns, not truncated from scales
computed at 1,024 dimensions.

The Rust runtime supports fp32, int8-row/dim, int4-row/dim, and int2-row.
`int2_dim` is available for analysis but intentionally rejected by the runtime
because it collapses too many quiet token rows and performs poorly.

## Inspect an artifact

```bash
uv run slicer inspect ../data/int4-row-512/model.safetensors
```

`inspect` prints tensor shapes, dtypes, and metadata, then verifies that the
file obeys the Lattice artifact contract. Exit status is 0 for a valid
artifact, 1 for an invalid/non-Lattice model, and 2 for file or format errors.

## Artifact format

`model.safetensors` contains:

- `weight`: fp32, int8, or packed uint8 data;
- `scale`: fp32 row/dimension scales for quantized variants;
- string metadata including `lattice_variant`, `bits`, `axis`, `dim`,
  `vocab_size`, packing layout, tokenizer, and source.

Int4 stores the low-dimension nibble first and biases signed `[-7, 7]` codes
by `+7`. Int2 stores four low-to-high two-bit fields and biases `[-1, 1]` by
`+1`. The format is implemented independently by the Python slicer and Rust
runtime, with parity tests covering the supported deployment recipes.

## Development

```bash
uv run python -m compileall src
uv run ruff check src  # if ruff is installed in the environment
```

Generated output should go under the repository's `data/` directory or
another ignored artifact directory, never into Git.
