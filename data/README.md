# Evaluation and benchmark data

This directory is not a model-artifact distribution. It keeps only the small
records needed to audit the blog's deployment claims:

- per-variant NanoBEIR and decontaminated-BEIR JSON outputs;
- the measured throughput table;
- the merged quality/throughput CSV;
- the published Pareto plot.

Generated weights, tokenizer copies, Wikipedia text, and output embeddings are
ignored by Git. The canonical fp32 model is hosted at
[`erikkaum/lattice-retrieval`](https://huggingface.co/erikkaum/lattice-retrieval),
and every deployment variant can be recreated with
[`slicer`](../slicer/README.md).

## Layout

```text
data/
├── <variant>/
│   └── eval/
│       ├── nanobeir.json          # selected audited variants
│       └── decontam_beir.json     # full deployment-quality coordinate
├── throughput.csv                 # runtime measurements
├── quality_vs_throughput.csv      # merged plotting table
├── quality_vs_throughput.png
└── quality_vs_throughput.pdf
```

Variant names encode precision, scale layout, and dimension, for example:

- `fp32-dim-1024`;
- `int8-dim-256`;
- `int4-row-512`;
- `int2-row-1024`.

`dim` and `row` mean per-dimension and per-token-row scaling respectively.
File sizes in the CSVs and article use decimal megabytes.

## Recreate a local artifact

```bash
cd ../slicer
uv run slicer slice \
  --dim 512 \
  --quant int4_row \
  --output-dir ../data/int4-row-512
```

This adds `model.safetensors` and `tokenizer.json` beside the checked-in eval
directory. Those generated files remain untracked.

## Re-run runtime parity and evaluations

Build the Python bindings first:

```bash
cd ../lattice
uv venv .venv-py
source .venv-py/bin/activate
uv pip install maturin numpy
maturin develop --release --features python
```

Then use the audit scripts in `lattice/tests/`:

```bash
.venv-py/bin/python tests/test_parity.py
.venv-py/bin/python tests/verify_evals.py
.venv-py/bin/python tests/eval_decontam.py --all
.venv-py/bin/python tests/plot_quality_vs_throughput.py
```

The first three commands also need the final local fp32 training checkpoint
under `trainer/runs/`; see the constants at the top of each audit script if
your run directory has a different name.

## Interpreting the numbers

Read [`../evals.md`](../evals.md) before comparing results. Stage-2 NanoBEIR
is used for paired quantization checks, not as an uncontaminated generalization
score. The quality coordinate in `quality_vs_throughput.csv` is the
decontaminated BEIR 12-task mean.

The throughput rows are individual runs on one 8-core Apple M2 MacBook Air
with 12 workers over the same 5,000-article corpus. Small differences between
neighboring points are directional rather than a hardware-independent ranking.
