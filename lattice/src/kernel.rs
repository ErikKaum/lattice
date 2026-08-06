//! Per-document embedding kernels.
//!
//! Two top-level kernels:
//!
//! - [`embed_dim`] — axis=dim. Same per-column scale for every row, so we
//!   accumulate biased codes across the document and apply `scale[d]` plus
//!   the bias correction in one final pass. Integer inner loop, one fp
//!   multiply per output dim at the end.
//! - [`embed_row`] — axis=row. Each row has its own scale; the inner loop
//!   broadcasts `scale[tok]` and folds it in per token, with an f32
//!   accumulator. More fp work per token but a strictly better quality/
//!   compression tradeoff at sub-byte storage (int2-row is the only
//!   deployable int2 variant; int2-dim collapses).
//!
//! All inner loops use the `wide` crate for portable SIMD — single source
//! that targets NEON / SSE / AVX / WASM SIMD with the same code. We measured
//! a 1.5-1.7× kernel-only slowdown vs hand-tuned aarch64 NEON intrinsics, but
//! kernel is <15% of end-to-end wall on real workloads, so the portable path
//! costs 2-3% on full pipeline. Trade chosen.
//!
//! ### Layout (axis=dim)
//!
//! For int4-dim (and analogously int2-dim if we ever enable it), the dim/2
//! bytes per row split into "even-position" nibbles (logical dims 0, 2,
//! 4, ...) and "odd-position" nibbles. We accumulate these into two halves
//! of the scratch buffer to avoid interleaved stores that would block SIMD.
//! For int8-dim there's no nibble split — each byte already targets one
//! logical dim.
//!
//! ### Bias trick (axis=dim)
//!
//! Symmetric quantization stores codes in `[0, 2·qmax]` (biased by
//! `+qmax`). Rather than subtracting the bias per element per token, we sum
//! unsigned biased codes during accumulation and apply `qmax·N` once per
//! output element in the final pass. For int8 storage (signed), we XOR
//! each byte with 0x80 to map `i8 ∈ [-128, 127]` → `u8 ∈ [0, 255]`, then
//! use the same bias-`128·N` trick. Replaces `dim · tokens` subtractions
//! with `dim` subtractions across all dim-axis kernels.
//!
//! ### Hoisted scratch
//!
//! The accumulator lives in [`EmbedScratch`], owned by the caller and
//! reused across `embed` calls. We allocate as `Vec<u32>` and reinterpret
//! to `&mut [f32]` (same alignment, same size) when the row-axis kernel
//! needs f32 accumulation. Per-doc allocation was ~3-5% of wall in the
//! samply baseline; hoisting eliminated it.

use wide::{f32x8, i32x8, u8x16, u16x8, u32x8};

/// Caller-owned scratch buffer. Allocate once with [`EmbedScratch::with_dim`]
/// (or via `Model::scratch()`), reuse across calls.
pub struct EmbedScratch {
    /// `dim` 32-bit elements. For axis=dim, used as `u32` (biased-code
    /// accumulator). For axis=row, reinterpreted as `f32` (scale-weighted
    /// accumulator). u32 and f32 have the same size and alignment.
    pub(crate) acc: Vec<u32>,
}

impl EmbedScratch {
    pub fn with_dim(dim: usize) -> Self {
        Self {
            acc: vec![0u32; dim],
        }
    }
}

// ==========================================================================
// Public entry points — one per axis. Both generic over `const BITS: u32`.
// ==========================================================================

/// Mean-pool the rows for `tokens` from a packed weight table with per-dim
/// scale. `BITS` is the per-element bit width of the storage; we implement
/// 4 and 8 (int4-dim and int8-dim).
pub fn embed_dim<const BITS: u32>(
    weight: &[u8],
    scale: &[f32],
    dim: usize,
    tokens: &[u32],
    out: &mut [f32],
    scratch: &mut EmbedScratch,
) {
    assert_eq!(out.len(), dim, "out buffer must have length = dim");
    if scratch.acc.len() != dim {
        scratch.acc.resize(dim, 0);
    }
    if tokens.is_empty() {
        out.fill(0.0);
        return;
    }
    // fp32 (BITS=32) takes the f32-accumulator path; everything else uses
    // the unsigned-integer accumulator with the bias trick.
    if BITS == 32 {
        let acc_f32: &mut [f32] = bytemuck::cast_slice_mut(&mut scratch.acc);
        for v in acc_f32.iter_mut() {
            *v = 0.0;
        }
        embed_dim_fp32(weight, dim, tokens, out, acc_f32);
        return;
    }
    scratch.acc.fill(0);
    match BITS {
        4 => embed_dim_int4(weight, scale, dim, tokens, out, &mut scratch.acc),
        8 => embed_dim_int8(weight, scale, dim, tokens, out, &mut scratch.acc),
        _ => panic!("embed_dim::<{}> not implemented; supported: 32, 8, 4", BITS),
    }
}

/// Mean-pool with a per-row (per-token) scale. f32 accumulator; the inner
/// loop broadcasts `scale[tok]` and folds it in per token. Used for
/// int2-row (the only deployable int2 variant).
pub fn embed_row<const BITS: u32>(
    weight: &[u8],
    scale: &[f32],
    dim: usize,
    tokens: &[u32],
    out: &mut [f32],
    scratch: &mut EmbedScratch,
) {
    assert_eq!(out.len(), dim, "out buffer must have length = dim");
    if scratch.acc.len() != dim {
        scratch.acc.resize(dim, 0);
    }
    if tokens.is_empty() {
        out.fill(0.0);
        return;
    }
    // Reinterpret the u32 scratch as f32 (same size/alignment). Zero it.
    let acc_f32: &mut [f32] = bytemuck::cast_slice_mut(&mut scratch.acc);
    for v in acc_f32.iter_mut() {
        *v = 0.0;
    }
    match BITS {
        2 => embed_row_int2(weight, scale, dim, tokens, out, acc_f32),
        4 => embed_row_int4(weight, scale, dim, tokens, out, acc_f32),
        8 => embed_row_int8(weight, scale, dim, tokens, out, acc_f32),
        _ => panic!("embed_row::<{}> not implemented; supported: 2, 4, 8", BITS),
    }
}

// ==========================================================================
// embed_dim<4> — int4-dim
// ==========================================================================

fn embed_dim_int4(
    weight: &[u8],
    scale: &[f32],
    dim: usize,
    tokens: &[u32],
    out: &mut [f32],
    acc: &mut [u32],
) {
    debug_assert_eq!(dim % 2, 0, "int4 packing requires even dim");
    let bytes_per_row = dim / 2;
    let (acc_even, acc_odd) = acc.split_at_mut(bytes_per_row);

    for &tok in tokens {
        let row_start = (tok as usize) * bytes_per_row;
        let row = &weight[row_start..row_start + bytes_per_row];
        accumulate_row_int4(row, acc_even, acc_odd);
    }

    // Final pass: bias = 7·N, interleave even/odd into dim output.
    let n = tokens.len() as i32;
    let bias = 7_i32 * n;
    let inv_n = 1.0_f32 / n as f32;
    for d in 0..bytes_per_row {
        let e = (acc_even[d] as i32 - bias) as f32;
        let o = (acc_odd[d] as i32 - bias) as f32;
        out[2 * d] = e * scale[2 * d] * inv_n;
        out[2 * d + 1] = o * scale[2 * d + 1] * inv_n;
    }
}

fn accumulate_row_int4(row: &[u8], acc_even: &mut [u32], acc_odd: &mut [u32]) {
    let len = row.len();
    let nibble_mask = u16x8::splat(0x000F);
    let mut b = 0;
    while b + 16 <= len {
        let chunk: [u8; 16] = row[b..b + 16].try_into().unwrap();
        let bytes = u8x16::new(chunk);

        let lo_half = u16x8::from_u8x16_low(bytes);
        let hi_half = u16x8::from_u8x16_high(bytes);

        let even_lo = lo_half & nibble_mask;
        let odd_lo = (lo_half >> 4u32) & nibble_mask;
        let even_hi = hi_half & nibble_mask;
        let odd_hi = (hi_half >> 4u32) & nibble_mask;

        let even_lo_u32: u32x8 = even_lo.into();
        let even_hi_u32: u32x8 = even_hi.into();
        let odd_lo_u32: u32x8 = odd_lo.into();
        let odd_hi_u32: u32x8 = odd_hi.into();

        let ae0 = u32x8::new(acc_even[b..b + 8].try_into().unwrap());
        let ae1 = u32x8::new(acc_even[b + 8..b + 16].try_into().unwrap());
        acc_even[b..b + 8].copy_from_slice(&(ae0 + even_lo_u32).to_array());
        acc_even[b + 8..b + 16].copy_from_slice(&(ae1 + even_hi_u32).to_array());

        let ao0 = u32x8::new(acc_odd[b..b + 8].try_into().unwrap());
        let ao1 = u32x8::new(acc_odd[b + 8..b + 16].try_into().unwrap());
        acc_odd[b..b + 8].copy_from_slice(&(ao0 + odd_lo_u32).to_array());
        acc_odd[b + 8..b + 16].copy_from_slice(&(ao1 + odd_hi_u32).to_array());

        b += 16;
    }
    while b < len {
        let byte = row[b];
        acc_even[b] += (byte & 0x0F) as u32;
        acc_odd[b] += (byte >> 4) as u32;
        b += 1;
    }
}

// ==========================================================================
// embed_dim<8> — int8-dim
// ==========================================================================

fn embed_dim_int8(
    weight: &[u8],
    scale: &[f32],
    dim: usize,
    tokens: &[u32],
    out: &mut [f32],
    acc: &mut [u32],
) {
    let bytes_per_row = dim;

    for &tok in tokens {
        let row_start = (tok as usize) * bytes_per_row;
        let row = &weight[row_start..row_start + bytes_per_row];
        accumulate_row_int8(row, acc);
    }

    // Storage is i8 ∈ [-128, 127]. We treat the bytes as u8 and XOR with
    // 0x80 inside the kernel to get u8 ∈ [0, 255]; this lets us reuse the
    // unsigned u32 accumulator with a single bias correction (128·N) at
    // the end.
    let n = tokens.len() as i32;
    let bias = 128_i32 * n;
    let inv_n = 1.0_f32 / n as f32;
    for d in 0..dim {
        let v = (acc[d] as i32 - bias) as f32;
        out[d] = v * scale[d] * inv_n;
    }
}

// ==========================================================================
// embed_dim<32> — fp32 (no quantization)
// ==========================================================================
//
// Storage is f32 directly. Mean-pool the rows of `tokens` into `out`. No
// scale, no bias, no unpack — used as the Rust-side oracle for parity
// tests against `sentence-transformers` (which loads the same fp32
// weights). LLVM auto-vectorizes the inner loop cleanly given the
// straight-line f32 add pattern.

fn embed_dim_fp32(weight: &[u8], dim: usize, tokens: &[u32], out: &mut [f32], acc: &mut [f32]) {
    let bytes_per_row = dim * 4;
    // Reinterpret the raw mmap bytes as a &[f32]. Safety: the loader has
    // validated that the weight tensor is `vocab * dim` f32s = `vocab *
    // dim * 4` bytes; alignment of a 4-byte type at the mmap offset is
    // guaranteed by the safetensors layout.
    let weight_f32: &[f32] = bytemuck::cast_slice(weight);

    for &tok in tokens {
        let row_start = (tok as usize) * dim;
        let row = &weight_f32[row_start..row_start + dim];
        for d in 0..dim {
            acc[d] += row[d];
        }
    }

    let inv_n = 1.0_f32 / tokens.len() as f32;
    for d in 0..dim {
        out[d] = acc[d] * inv_n;
    }
    // bytes_per_row is the natural per-row stride if we wanted to walk
    // raw bytes; kept for symmetry with the integer kernels.
    let _ = bytes_per_row;
}

fn accumulate_row_int8(row: &[u8], acc: &mut [u32]) {
    let len = row.len();
    let sign_flip = u8x16::splat(0x80);
    let mut b = 0;
    while b + 16 <= len {
        let chunk: [u8; 16] = row[b..b + 16].try_into().unwrap();
        let bytes = u8x16::new(chunk) ^ sign_flip; // i8 in -128..127 → u8 in 0..255

        let lo_u16 = u16x8::from_u8x16_low(bytes);
        let hi_u16 = u16x8::from_u8x16_high(bytes);

        let lo_u32: u32x8 = lo_u16.into();
        let hi_u32: u32x8 = hi_u16.into();

        let a0 = u32x8::new(acc[b..b + 8].try_into().unwrap());
        let a1 = u32x8::new(acc[b + 8..b + 16].try_into().unwrap());
        acc[b..b + 8].copy_from_slice(&(a0 + lo_u32).to_array());
        acc[b + 8..b + 16].copy_from_slice(&(a1 + hi_u32).to_array());

        b += 16;
    }
    while b < len {
        let v = (row[b] ^ 0x80) as u32; // sign-flip into unsigned
        acc[b] += v;
        b += 1;
    }
}

// ==========================================================================
// embed_row<2> — int2-row
// ==========================================================================
//
// Per-row scale (length = vocab). Each row stores dim/4 bytes, packing 4
// 2-bit codes per byte. Code value k ∈ {0, 1, 2}; signed value = k - 1.
//
// out[d] = (1/N) · Σ_t scale[t] · (code(t, d) - 1)
//        = (1/N) · ( Σ_t scale[t] · code(t, d)  -  Σ_t scale[t] )
//                                                  └─ same for every d
//
// Accumulator (f32, length dim) holds the weighted sum on the left.
// We also keep a scalar `total_scale = Σ_t scale[t]` to subtract once per
// output element at the end. Inner loop: broadcast scale[t], FMA into 4
// streams (one per byte-position), advance.

fn embed_row_int2(
    weight: &[u8],
    scale_per_token: &[f32],
    dim: usize,
    tokens: &[u32],
    out: &mut [f32],
    acc_f32: &mut [f32],
) {
    debug_assert_eq!(dim % 4, 0, "int2 packing requires dim divisible by 4");
    let bytes_per_row = dim / 4;
    // The acc layout splits into four "positions" within each byte:
    //   acc[0 .. dim/4]            ← position 0 (bits [0:2] of each byte)
    //   acc[dim/4 .. 2·dim/4]      ← position 1 (bits [2:4])
    //   acc[2·dim/4 .. 3·dim/4]    ← position 2 (bits [4:6])
    //   acc[3·dim/4 .. dim]        ← position 3 (bits [6:8])
    let (acc_p0, rest) = acc_f32.split_at_mut(bytes_per_row);
    let (acc_p1, rest) = rest.split_at_mut(bytes_per_row);
    let (acc_p2, acc_p3) = rest.split_at_mut(bytes_per_row);

    let mut total_scale: f32 = 0.0;
    for &tok in tokens {
        let tok_i = tok as usize;
        let scale_t = scale_per_token[tok_i];
        total_scale += scale_t;
        let row_start = tok_i * bytes_per_row;
        let row = &weight[row_start..row_start + bytes_per_row];
        accumulate_row_int2_with_scale(row, scale_t, acc_p0, acc_p1, acc_p2, acc_p3);
    }

    // Final pass: out[4b+p] = (acc_pp[b] - total_scale) / N
    let inv_n = 1.0_f32 / tokens.len() as f32;
    for b in 0..bytes_per_row {
        out[4 * b] = (acc_p0[b] - total_scale) * inv_n;
        out[4 * b + 1] = (acc_p1[b] - total_scale) * inv_n;
        out[4 * b + 2] = (acc_p2[b] - total_scale) * inv_n;
        out[4 * b + 3] = (acc_p3[b] - total_scale) * inv_n;
    }
}

fn accumulate_row_int2_with_scale(
    row: &[u8],
    scale_t: f32,
    acc_p0: &mut [f32],
    acc_p1: &mut [f32],
    acc_p2: &mut [f32],
    acc_p3: &mut [f32],
) {
    let len = row.len();
    let two_bit_mask = u16x8::splat(0x0003);
    let scale_v = f32x8::splat(scale_t);

    let mut b = 0;
    while b + 16 <= len {
        let chunk: [u8; 16] = row[b..b + 16].try_into().unwrap();
        let bytes = u8x16::new(chunk);

        let lo_half = u16x8::from_u8x16_low(bytes);
        let hi_half = u16x8::from_u8x16_high(bytes);

        // Extract 4 positions × 2 halves = 8 u16x8 of codes ∈ {0, 1, 2}.
        let p0_lo = lo_half & two_bit_mask;
        let p1_lo = (lo_half >> 2u32) & two_bit_mask;
        let p2_lo = (lo_half >> 4u32) & two_bit_mask;
        let p3_lo = (lo_half >> 6u32) & two_bit_mask;
        let p0_hi = hi_half & two_bit_mask;
        let p1_hi = (hi_half >> 2u32) & two_bit_mask;
        let p2_hi = (hi_half >> 4u32) & two_bit_mask;
        let p3_hi = (hi_half >> 6u32) & two_bit_mask;

        // Update 4 acc streams. Each stream gets one lo + one hi u16x8.
        fma_acc(acc_p0, b, p0_lo, p0_hi, scale_v);
        fma_acc(acc_p1, b, p1_lo, p1_hi, scale_v);
        fma_acc(acc_p2, b, p2_lo, p2_hi, scale_v);
        fma_acc(acc_p3, b, p3_lo, p3_hi, scale_v);

        b += 16;
    }
    while b < len {
        let byte = row[b];
        let c0 = (byte & 0x03) as f32;
        let c1 = ((byte >> 2) & 0x03) as f32;
        let c2 = ((byte >> 4) & 0x03) as f32;
        let c3 = (byte >> 6) as f32;
        acc_p0[b] += c0 * scale_t;
        acc_p1[b] += c1 * scale_t;
        acc_p2[b] += c2 * scale_t;
        acc_p3[b] += c3 * scale_t;
        b += 1;
    }
}

/// FMA helper: `acc[b..b+16] += widen(codes_lo, codes_hi) · scale_v`.
/// Inputs are u16x8 holding small non-negative values. We widen through
/// i32 → f32 (zero-extension safe here; codes are non-negative).
#[inline]
fn fma_acc(acc: &mut [f32], b: usize, codes_lo: u16x8, codes_hi: u16x8, scale_v: f32x8) {
    let lo_i32: i32x8 = i32x8::from_u16x8(codes_lo);
    let hi_i32: i32x8 = i32x8::from_u16x8(codes_hi);
    let lo_f32: f32x8 = f32x8::from_i32x8(lo_i32);
    let hi_f32: f32x8 = f32x8::from_i32x8(hi_i32);

    let a0 = f32x8::new(acc[b..b + 8].try_into().unwrap());
    let a1 = f32x8::new(acc[b + 8..b + 16].try_into().unwrap());
    let new_a0 = a0 + lo_f32 * scale_v;
    let new_a1 = a1 + hi_f32 * scale_v;
    acc[b..b + 8].copy_from_slice(&new_a0.to_array());
    acc[b + 8..b + 16].copy_from_slice(&new_a1.to_array());
}

// ==========================================================================
// embed_row<4> — int4-row
// ==========================================================================
//
// Same per-token broadcast-scale-and-FMA pattern as int2-row, but with the
// int4 nibble unpack and only 2 streams (even-position / odd-position) like
// the int4-dim path. Per-token f32 work makes this strictly slower than
// int4-dim at the same dim — included for matrix completeness, not for any
// quality advantage (stage-2 sweep has int4-dim 0.5314 ≥ int4-row 0.5267 at
// d=1024).

fn embed_row_int4(
    weight: &[u8],
    scale_per_token: &[f32],
    dim: usize,
    tokens: &[u32],
    out: &mut [f32],
    acc_f32: &mut [f32],
) {
    debug_assert_eq!(dim % 2, 0, "int4 packing requires even dim");
    let bytes_per_row = dim / 2;
    let (acc_even, acc_odd) = acc_f32.split_at_mut(bytes_per_row);

    let mut total_scale: f32 = 0.0;
    for &tok in tokens {
        let tok_i = tok as usize;
        let scale_t = scale_per_token[tok_i];
        total_scale += scale_t;
        let row_start = tok_i * bytes_per_row;
        let row = &weight[row_start..row_start + bytes_per_row];
        accumulate_row_int4_with_scale(row, scale_t, acc_even, acc_odd);
    }

    // Final pass: subtract 7·total_scale (per-element bias×scale-sum), interleave.
    let inv_n = 1.0_f32 / tokens.len() as f32;
    let bias = 7.0_f32 * total_scale;
    for b in 0..bytes_per_row {
        out[2 * b] = (acc_even[b] - bias) * inv_n;
        out[2 * b + 1] = (acc_odd[b] - bias) * inv_n;
    }
}

fn accumulate_row_int4_with_scale(
    row: &[u8],
    scale_t: f32,
    acc_even: &mut [f32],
    acc_odd: &mut [f32],
) {
    let len = row.len();
    let nibble_mask = u16x8::splat(0x000F);
    let scale_v = f32x8::splat(scale_t);
    let mut b = 0;
    while b + 16 <= len {
        let chunk: [u8; 16] = row[b..b + 16].try_into().unwrap();
        let bytes = u8x16::new(chunk);

        let lo_half = u16x8::from_u8x16_low(bytes);
        let hi_half = u16x8::from_u8x16_high(bytes);

        let even_lo = lo_half & nibble_mask;
        let odd_lo = (lo_half >> 4u32) & nibble_mask;
        let even_hi = hi_half & nibble_mask;
        let odd_hi = (hi_half >> 4u32) & nibble_mask;

        fma_acc(acc_even, b, even_lo, even_hi, scale_v);
        fma_acc(acc_odd, b, odd_lo, odd_hi, scale_v);

        b += 16;
    }
    while b < len {
        let byte = row[b];
        let lo = (byte & 0x0F) as f32;
        let hi = (byte >> 4) as f32;
        acc_even[b] += lo * scale_t;
        acc_odd[b] += hi * scale_t;
        b += 1;
    }
}

// ==========================================================================
// embed_row<8> — int8-row
// ==========================================================================
//
// Same sign-flip-XOR trick as int8-dim to remap i8 ∈ [-128, 127] → u8 ∈
// [0, 255], but accumulate in f32 with per-token scale broadcast. Final
// pass subtracts `128·total_scale` (the bias correction). Strictly slower
// than int8-dim at the same dim due to the per-element f32 FMA vs i32 add.

fn embed_row_int8(
    weight: &[u8],
    scale_per_token: &[f32],
    dim: usize,
    tokens: &[u32],
    out: &mut [f32],
    acc_f32: &mut [f32],
) {
    let bytes_per_row = dim;

    let mut total_scale: f32 = 0.0;
    for &tok in tokens {
        let tok_i = tok as usize;
        let scale_t = scale_per_token[tok_i];
        total_scale += scale_t;
        let row_start = tok_i * bytes_per_row;
        let row = &weight[row_start..row_start + bytes_per_row];
        accumulate_row_int8_with_scale(row, scale_t, acc_f32);
    }

    let inv_n = 1.0_f32 / tokens.len() as f32;
    let bias = 128.0_f32 * total_scale;
    for d in 0..dim {
        out[d] = (acc_f32[d] - bias) * inv_n;
    }
}

fn accumulate_row_int8_with_scale(row: &[u8], scale_t: f32, acc_f32: &mut [f32]) {
    let len = row.len();
    let sign_flip = u8x16::splat(0x80);
    let scale_v = f32x8::splat(scale_t);
    let mut b = 0;
    while b + 16 <= len {
        let chunk: [u8; 16] = row[b..b + 16].try_into().unwrap();
        let bytes = u8x16::new(chunk) ^ sign_flip;

        let lo_u16 = u16x8::from_u8x16_low(bytes);
        let hi_u16 = u16x8::from_u8x16_high(bytes);

        fma_acc(acc_f32, b, lo_u16, hi_u16, scale_v);

        b += 16;
    }
    while b < len {
        let v = (row[b] ^ 0x80) as f32;
        acc_f32[b] += v * scale_t;
        b += 1;
    }
}

// ==========================================================================
// L2 normalize
// ==========================================================================

/// L2-normalize a vector in place. No-op if the norm is below
/// `f32::EPSILON` (treats it as a zero vector — what mean-pooling an empty
/// doc produces).
pub fn l2_normalize(v: &mut [f32]) {
    let sumsq: f32 = v.iter().map(|x| x * x).sum();
    let norm = sumsq.sqrt();
    if norm < f32::EPSILON {
        return;
    }
    let inv = 1.0_f32 / norm;
    for x in v.iter_mut() {
        *x *= inv;
    }
}
