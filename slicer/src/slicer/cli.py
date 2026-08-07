"""Command-line entry point.

Two subcommands:

    uvx slicer slice   --dim 256 --quant int8_row --output-dir out/
    uvx slicer slice   --dim 256 --quant int8_row --source path/m.safetensors --output-dir out/
    uvx slicer inspect path/model.safetensors

`slice` produces `out/model.safetensors` (and `out/tokenizer.json` unless
`--omit-tokenizer`). All variant metadata is in the safetensors header.

`inspect` prints what's inside a `.safetensors` file and tells you whether
it's a valid lattice deployment artifact.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from .inspect import inspect_file
from .quantize import (
    SUPPORTED_DIMS,
    SUPPORTED_VARIANTS,
    Quantized,
    dequantize,
    quantize_and_slice,
)
from .source import DEFAULT_REPO, resolve_source


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="slicer",
        description=(
            "Tools for the lattice-retrieval deployment artifact format. "
            "Use `slice` to produce one. Use `inspect` to verify one."
        ),
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---- slice -----------------------------------------------------------
    p_slice = sub.add_parser(
        "slice",
        help="slice + quantize into a deployment `model.safetensors`",
        description=(
            "Slice + quantize the lattice-retrieval static embedding model "
            "into a deployment artifact. One invocation produces one variant; "
            "variant info goes into the safetensors metadata strings."
        ),
    )
    p_slice.add_argument(
        "--dim",
        type=int,
        required=True,
        choices=SUPPORTED_DIMS,
        help="Output matryoshka dim. Must be one of the trained dims.",
    )
    p_slice.add_argument(
        "--quant",
        required=True,
        choices=SUPPORTED_VARIANTS,
        help="Quantization recipe. `fp32` means slice but don't quantize.",
    )
    p_slice.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write `model.safetensors` (and `tokenizer.json`).",
    )
    p_slice.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to a local `.safetensors` file. If omitted, the fp32 "
        "weights are downloaded from HF Hub. When set, the tokenizer "
        "is expected to live next to the file as `tokenizer.json` "
        "(unless --omit-tokenizer).",
    )
    p_slice.add_argument(
        "--source-repo",
        default=DEFAULT_REPO,
        help=f"HF Hub repo id (default: {DEFAULT_REPO}). Ignored if --source is set.",
    )
    p_slice.add_argument(
        "--source-revision",
        default="main",
        help="HF Hub revision (default: main).",
    )
    p_slice.add_argument(
        "--omit-tokenizer",
        action="store_true",
        help="Skip copying `tokenizer.json` into the output directory.",
    )
    p_slice.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the round-trip sanity check (load, dequant, compare).",
    )

    # ---- inspect ---------------------------------------------------------
    p_inspect = sub.add_parser(
        "inspect",
        help="report variant + dim of a `.safetensors` file; verify it's a valid lattice artifact",
        description=(
            "Inspect a `.safetensors` file: print its tensor shapes, dtypes, "
            "and metadata, then verify it's a valid lattice deployment "
            "artifact. Exit code 0 if valid, 1 if invalid or not a lattice "
            "model, 2 on file/format errors."
        ),
    )
    p_inspect.add_argument(
        "path",
        type=Path,
        help="Path to a `.safetensors` file produced by `slicer slice` "
        "(or any file — we'll tell you if it isn't one).",
    )

    return ap


def main() -> None:
    args = _build_parser().parse_args()
    if args.cmd == "slice":
        _do_slice(args)
    elif args.cmd == "inspect":
        sys.exit(inspect_file(args.path))
    else:
        raise SystemExit(f"unknown subcommand {args.cmd!r}")


def _do_slice(args: argparse.Namespace) -> None:
    source = resolve_source(
        source=args.source,
        repo=args.source_repo,
        revision=args.source_revision,
        want_tokenizer=not args.omit_tokenizer,
    )
    print(f"source: {source.label}", flush=True)
    print(f"  weights:   {source.weights}", flush=True)
    if source.tokenizer is not None:
        print(f"  tokenizer: {source.tokenizer}", flush=True)

    weights_dict = load_file(str(source.weights))
    # The lattice-retrieval safetensors stores the table as
    # `embedding.weight` (matches the StaticEmbedding module's parameter
    # name). Fall back to a single-tensor file if needed.
    if "embedding.weight" in weights_dict:
        W = weights_dict["embedding.weight"]
    elif len(weights_dict) == 1:
        W = next(iter(weights_dict.values()))
    else:
        raise SystemExit(
            f"could not pick a weight tensor from {source.weights}; keys: {list(weights_dict)}"
        )
    print(f"source weight: shape={W.shape} dtype={W.dtype}", flush=True)

    qz = quantize_and_slice(
        weight=W,
        variant=args.quant,
        dim=args.dim,
        source_label=source.label,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "model.safetensors"
    tensors = {"weight": qz.q}
    if qz.scale is not None:
        tensors["scale"] = qz.scale
    save_file(tensors, str(out_path), metadata=qz.metadata)
    print(
        f"wrote {out_path}  "
        f"({out_path.stat().st_size / 1e6:.2f} MB, "
        f"variant={args.quant}, dim={args.dim})",
        flush=True,
    )

    if source.tokenizer is not None:
        dst = args.output_dir / "tokenizer.json"
        shutil.copyfile(source.tokenizer, dst)
        print(f"wrote {dst}", flush=True)

    if not args.no_verify:
        _verify_roundtrip(out_path, W[:, : args.dim], args.quant)


def _verify_roundtrip(out_path: Path, W_ref: np.ndarray, variant: str) -> None:
    """Reload the file, dequantize, compare to the sliced reference.
    Catches bit-packing bugs, scale-dtype mistakes, and metadata drift."""
    from safetensors import safe_open

    with safe_open(str(out_path), framework="np") as f:
        meta = f.metadata() or {}
        tensor_names = list(f.keys())
        tensors = {key: f.get_tensor(key) for key in tensor_names}

    qz = Quantized(
        q=tensors["weight"],
        scale=tensors.get("scale"),
        metadata=meta,
    )
    W_dq = dequantize(qz).astype(np.float32)
    diff = np.abs(W_ref.astype(np.float32) - W_dq)
    bits = int(meta["bits"])
    print(
        f"round-trip verify: variant={variant}  "
        f"max_abs_err={diff.max():.5f}  "
        f"mean_abs_err={diff.mean():.5f}",
        flush=True,
    )
    if bits == 32:
        if diff.max() > 1e-6:
            print(
                f"  WARNING: fp32 round-trip not exact ({diff.max()})", file=sys.stderr
            )
    else:
        # Symmetric int_n round-trip error is bounded by scale/2 per element,
        # which for the loudest row/dim equals max_abs_in_W / (2*qmax).
        qmax = (1 << (bits - 1)) - 1
        expected_bound = float(np.abs(W_ref).max()) / (2 * qmax)
        if diff.max() > 1.5 * expected_bound:
            print(
                f"  WARNING: max_abs_err {diff.max():.5f} exceeds expected "
                f"bound {expected_bound:.5f} — possible packing bug",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
