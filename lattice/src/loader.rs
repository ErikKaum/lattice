//! Parse a `model.safetensors` produced by the slicer Python CLI.
//!
//! We mmap the file, read the safetensors header (which gives us both the
//! user metadata map and the `(start, end)` byte offsets of each tensor
//! within the data region), validate the lattice metadata, and record the
//! `weight` tensor's offsets so the kernel can read it zero-copy from the
//! mmap. The (much smaller) `scale` tensor is eagerly copied into a
//! `Vec<f32>` — at most `max(vocab, dim) * 4` bytes (~120KB) — so we don't
//! have to fight raw-byte alignment in the hot path.

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use memmap2::Mmap;
use safetensors::SafeTensors;
use safetensors::tensor::Dtype;

use crate::Axis;

// safetensors prefixes the header with an 8-byte little-endian u64 holding
// the header's JSON length. The tensor data region starts at `N_LEN +
// header_size`. `TensorInfo.data_offsets` are relative to that region.
const N_LEN: usize = 8;

pub struct ModelData {
    pub mmap: Mmap,
    pub bits: u8,
    pub axis: Axis,
    pub vocab: usize,
    pub dim: usize,
    pub weight_offset: usize,
    pub weight_len: usize,
    pub scale: Vec<f32>,
    pub variant: String,
}

pub fn load(path: &Path) -> Result<ModelData> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mmap = unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?;

    let (header_size, meta_obj) = SafeTensors::read_metadata(&mmap)
        .with_context(|| format!("parse safetensors header in {}", path.display()))?;
    let data_base = N_LEN + header_size;

    let user_meta = meta_obj
        .metadata()
        .as_ref()
        .ok_or_else(|| anyhow!("file has no metadata; not a lattice model"))?;

    let variant = user_meta
        .get("lattice_variant")
        .ok_or_else(|| anyhow!("not a lattice model: missing `lattice_variant`"))?
        .to_owned();
    let bits: u8 = user_meta
        .get("bits")
        .context("missing `bits` in metadata")?
        .parse()
        .context("`bits` is not a valid u8")?;
    let axis = match user_meta
        .get("axis")
        .context("missing `axis` in metadata")?
        .as_str()
    {
        "dim" => Axis::Dim,
        "row" => Axis::Row,
        // fp32 emits axis="none"; it reuses the dim dispatch because no scale
        // tensor is involved.
        "none" => Axis::Dim,
        other => bail!("unknown axis={:?}", other),
    };
    let dim: usize = user_meta
        .get("dim")
        .context("missing `dim`")?
        .parse()
        .context("`dim` not a usize")?;
    let vocab: usize = user_meta
        .get("vocab_size")
        .context("missing `vocab_size`")?
        .parse()
        .context("`vocab_size` not a usize")?;

    let weight_info = meta_obj
        .info("weight")
        .ok_or_else(|| anyhow!("missing `weight` tensor"))?;
    let (w_start, w_end) = weight_info.data_offsets;
    let weight_offset = data_base + w_start;
    let weight_len = w_end - w_start;

    // Physical row length is `ceil(dim * bits / 8)` bytes — validate the
    // table is exactly that big so a metadata/tensor drift fails loudly here,
    // not silently in the kernel.
    let expected_row_bytes = (dim * bits as usize).div_ceil(8);
    let expected_total = vocab * expected_row_bytes;
    if weight_len != expected_total {
        bail!(
            "weight size mismatch: got {} bytes, expected vocab*ceil(dim*bits/8) = {}*{} = {}",
            weight_len,
            vocab,
            expected_row_bytes,
            expected_total
        );
    }

    let scale = if bits == 32 {
        Vec::new()
    } else {
        let scale_info = meta_obj
            .info("scale")
            .ok_or_else(|| anyhow!("missing `scale` tensor (required for quantized variants)"))?;
        if scale_info.dtype != Dtype::F32 {
            bail!("`scale` dtype {:?}, expected F32", scale_info.dtype);
        }
        let (s_start, s_end) = scale_info.data_offsets;
        let s_bytes = &mmap[data_base + s_start..data_base + s_end];
        if s_bytes.len() % 4 != 0 {
            bail!("scale bytes len {} not multiple of 4", s_bytes.len());
        }
        let mut v: Vec<f32> = Vec::with_capacity(s_bytes.len() / 4);
        for chunk in s_bytes.chunks_exact(4) {
            v.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        let expected_scale_len = match axis {
            Axis::Dim => dim,
            Axis::Row => vocab,
        };
        if v.len() != expected_scale_len {
            bail!(
                "scale len {} != expected {} for axis={:?}",
                v.len(),
                expected_scale_len,
                axis
            );
        }
        v
    };

    Ok(ModelData {
        mmap,
        bits,
        axis,
        vocab,
        dim,
        weight_offset,
        weight_len,
        scale,
        variant,
    })
}
