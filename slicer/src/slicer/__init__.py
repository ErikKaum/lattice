"""slicer — produce one deployment artifact (sliced + quantized
`model.safetensors`) from the lattice-retrieval static embedding model.

Variant matrix supported:
    dim   ∈ {32, 64, 128, 256, 512, 1024}   (the trained matryoshka dims)
    quant ∈ {fp32, int8_row, int8_dim, int4_row, int4_dim, int2_row, int2_dim}

Each invocation produces one `model.safetensors`; variant info lives in
the safetensors metadata strings, not in the filename (the filename
stays canonical to match HF model-card layout). Tokenizer is bundled
into the output dir unless `--omit-tokenizer`.
"""

from .quantize import Variant, quantize_and_slice
from .source import resolve_source

__all__ = ["Variant", "quantize_and_slice", "resolve_source"]
