"""Inspect a `.safetensors` file and report whether it's a valid lattice
deployment artifact.

A valid file has:
- `lattice_variant` in metadata (one of `fp32 | int8_row | int8_dim |
  int4_row | int4_dim | int2_row | int2_dim`).
- `bits`, `axis`, `dim`, `vocab_size` in metadata.
- A `weight` tensor with the right shape and dtype for the declared variant.
- A `scale` tensor for quantized variants only, with the right shape for
  the declared axis.

If the file isn't a lattice artifact (e.g. somebody points at a generic
HF checkpoint), `inspect` says so plainly instead of pretending.
"""

from __future__ import annotations

from pathlib import Path

from .quantize import SUPPORTED_DIMS, SUPPORTED_VARIANTS


def inspect_file(path: Path) -> int:
    """Print human-readable diagnostics. Returns process exit code:
    0 = valid lattice artifact, 1 = invalid or non-lattice, 2 = file error.
    """
    if not path.is_file():
        print(f"error: {path} is not a regular file")
        return 2

    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="np") as f:
            meta = dict(f.metadata() or {})
            keys = list(f.keys())
            shapes = {k: f.get_tensor(k).shape for k in keys}
            dtypes = {k: str(f.get_tensor(k).dtype) for k in keys}
    except Exception as e:  # noqa: BLE001 — we want to surface any safetensors error
        print(f"error: failed to open as safetensors: {e}")
        return 2

    size_mb = path.stat().st_size / 1e6
    print(f"file: {path}  ({size_mb:.2f} MB)")
    print()
    print("tensors:")
    for k in keys:
        print(f"  {k:<8} shape={shapes[k]}  dtype={dtypes[k]}")
    print()
    print("metadata:")
    if meta:
        for k, v in sorted(meta.items()):
            print(f"  {k}: {v}")
    else:
        print("  (none)")
    print()

    # ---- Lattice-validation -------------------------------------------------
    if "lattice_variant" not in meta:
        print("not a lattice model: missing `lattice_variant` in metadata")
        return 1

    variant = meta["lattice_variant"]
    errors: list[str] = []

    if variant not in SUPPORTED_VARIANTS:
        errors.append(
            f"unknown lattice_variant={variant!r}; expected one of {list(SUPPORTED_VARIANTS)}"
        )

    for required in ("bits", "axis", "dim", "vocab_size"):
        if required not in meta:
            errors.append(f"missing required metadata key: {required!r}")

    # Bail early if structural keys are missing — the shape checks below
    # would otherwise just throw.
    if errors:
        _report(variant, errors)
        return 1

    bits = int(meta["bits"])
    axis = meta["axis"]
    dim = int(meta["dim"])
    vocab = int(meta["vocab_size"])

    if dim not in SUPPORTED_DIMS:
        errors.append(f"dim={dim} not in supported set {list(SUPPORTED_DIMS)}")

    if "weight" not in keys:
        errors.append("missing `weight` tensor")
    else:
        w_shape = shapes["weight"]
        w_dtype = dtypes["weight"]
        if len(w_shape) != 2:
            errors.append(f"`weight` should be 2-D, got shape {w_shape}")
        else:
            if w_shape[0] != vocab:
                errors.append(
                    f"`weight` vocab dim mismatch: shape[0]={w_shape[0]} "
                    f"but metadata vocab_size={vocab}"
                )
            errors.extend(_check_weight_layout(bits, dim, w_shape, w_dtype))

    if bits == 32:
        if "scale" in keys:
            errors.append("fp32 file should not have a `scale` tensor")
    else:
        if "scale" not in keys:
            errors.append("missing `scale` tensor (required for quantized variants)")
        else:
            s_shape = shapes["scale"]
            s_dtype = dtypes["scale"]
            if s_dtype != "float32":
                errors.append(f"`scale` dtype {s_dtype}, expected float32")
            if axis == "row" and s_shape != (vocab,):
                errors.append(
                    f"per-row scale should be shape ({vocab},), got {s_shape}"
                )
            elif axis == "dim" and s_shape != (dim,):
                errors.append(f"per-dim scale should be shape ({dim},), got {s_shape}")
            elif axis not in ("row", "dim"):
                errors.append(
                    f"axis={axis!r} unrecognized for quantized variant; expected 'row' or 'dim'"
                )

    if bits < 8:
        for k in ("pack_bias", "pack_layout"):
            if k not in meta:
                errors.append(f"sub-byte variant should declare {k!r} in metadata")

    return _report(variant, errors, dim=dim, vocab=vocab)


def _check_weight_layout(
    bits: int,
    dim: int,
    w_shape: tuple[int, ...],
    w_dtype: str,
) -> list[str]:
    """Per-bitwidth weight shape + dtype checks. Returns a list of errors."""
    errs: list[str] = []
    cols = w_shape[1]
    if bits == 32:
        if w_dtype not in ("float32",):
            errs.append(f"fp32 `weight` dtype {w_dtype}, expected float32")
        if cols != dim:
            errs.append(f"fp32 `weight` cols={cols}, expected dim={dim}")
    elif bits == 8:
        if w_dtype != "int8":
            errs.append(f"int8 `weight` dtype {w_dtype}, expected int8")
        if cols != dim:
            errs.append(f"int8 `weight` cols={cols}, expected dim={dim}")
    elif bits == 4:
        if w_dtype != "uint8":
            errs.append(f"int4-packed `weight` dtype {w_dtype}, expected uint8")
        if cols != dim // 2:
            errs.append(f"int4-packed `weight` cols={cols}, expected dim/2={dim // 2}")
    elif bits == 2:
        if w_dtype != "uint8":
            errs.append(f"int2-packed `weight` dtype {w_dtype}, expected uint8")
        if cols != dim // 4:
            errs.append(f"int2-packed `weight` cols={cols}, expected dim/4={dim // 4}")
    else:
        errs.append(f"unsupported bits={bits}")
    return errs


def _report(
    variant: str, errors: list[str], *, dim: int | None = None, vocab: int | None = None
) -> int:
    if errors:
        print(f"INVALID lattice model (variant={variant}) — {len(errors)} issue(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"VALID lattice model: variant={variant} dim={dim} vocab={vocab}")
    return 0
