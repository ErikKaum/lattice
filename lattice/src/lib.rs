//! Pure-Rust inference for sliced + quantized lattice-retrieval static embeddings.
//!
//! The runtime supports fp32, int8 per-row/per-dim, int4 per-row/per-dim, and
//! int2 per-row artifacts. Int2 per-dim is intentionally rejected because its
//! retrieval quality collapses.

use std::path::Path;

use anyhow::{Result, bail};

pub mod kernel;
mod loader;
pub mod tokenize;

#[cfg(feature = "python")]
mod bindings;

pub use kernel::EmbedScratch;
pub use tokenize::LatticeTokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    Dim,
    Row,
}

pub struct Model {
    inner: loader::ModelData,
}

impl Model {
    pub fn load(path: &Path) -> Result<Self> {
        Ok(Self {
            inner: loader::load(path)?,
        })
    }

    pub fn bits(&self) -> u8 {
        self.inner.bits
    }
    pub fn axis(&self) -> Axis {
        self.inner.axis
    }
    pub fn dim(&self) -> usize {
        self.inner.dim
    }
    pub fn vocab(&self) -> usize {
        self.inner.vocab
    }
    pub fn variant(&self) -> &str {
        &self.inner.variant
    }

    /// Allocate a [`EmbedScratch`] sized for this model. Reuse across calls
    /// to [`Model::embed`] — the buffer is overwritten, never grown.
    pub fn scratch(&self) -> EmbedScratch {
        EmbedScratch::with_dim(self.dim())
    }

    /// Mean-pool the rows for `tokens` into `out` (length must equal `dim`).
    /// `scratch` must be from [`Model::scratch`] on a model with the same
    /// `dim`; the kernel overwrites it. Empty `tokens` → zero vector.
    pub fn embed(&self, tokens: &[u32], out: &mut [f32], scratch: &mut EmbedScratch) -> Result<()> {
        let weight = &self.inner.mmap
            [self.inner.weight_offset..self.inner.weight_offset + self.inner.weight_len];
        let scale = &self.inner.scale[..];
        let dim = self.inner.dim;

        match (self.inner.bits, self.inner.axis) {
            (32, _) => {
                // fp32 — scale slice is empty / unused; weight bytes are
                // reinterpreted as &[f32] inside the kernel.
                kernel::embed_dim::<32>(weight, scale, dim, tokens, out, scratch);
                Ok(())
            }
            (4, Axis::Dim) => {
                kernel::embed_dim::<4>(weight, scale, dim, tokens, out, scratch);
                Ok(())
            }
            (8, Axis::Dim) => {
                kernel::embed_dim::<8>(weight, scale, dim, tokens, out, scratch);
                Ok(())
            }
            (2, Axis::Row) => {
                kernel::embed_row::<2>(weight, scale, dim, tokens, out, scratch);
                Ok(())
            }
            (4, Axis::Row) => {
                kernel::embed_row::<4>(weight, scale, dim, tokens, out, scratch);
                Ok(())
            }
            (8, Axis::Row) => {
                kernel::embed_row::<8>(weight, scale, dim, tokens, out, scratch);
                Ok(())
            }
            (2, Axis::Dim) => bail!(
                "(bits=2, axis=Dim) is intentionally unimplemented — int2-dim collapses \
                 (NDCG ≈ 0.37); use int2-row instead"
            ),
            (b, a) => bail!("unsupported (bits={}, axis={:?}) combo", b, a),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = "../data/int4-dim-1024/model.safetensors";

    #[test]
    #[ignore = "requires a locally generated deployment artifact under data/"]
    fn load_fixture() {
        let m = Model::load(Path::new(FIXTURE)).expect("load fixture");
        assert_eq!(m.bits(), 4);
        assert_eq!(m.axis(), Axis::Dim);
        assert_eq!(m.dim(), 1024);
        assert_eq!(m.vocab(), 30522);
        assert_eq!(m.variant(), "int4_dim");
    }

    #[test]
    #[ignore = "requires a locally generated deployment artifact under data/"]
    fn smoke_embed() {
        let m = Model::load(Path::new(FIXTURE)).expect("load fixture");
        let tokens = vec![7592u32, 2088]; // "hello", "world" — content tokens
        let mut out = vec![0.0f32; m.dim()];
        let mut sc = m.scratch();
        m.embed(&tokens, &mut out, &mut sc).expect("embed");
        assert!(out.iter().all(|&x| x.is_finite()), "non-finite output");
        assert!(
            out.iter().any(|&x| x != 0.0),
            "all-zero output (unexpected)"
        );
    }

    #[test]
    #[ignore = "requires a locally generated deployment artifact under data/"]
    fn empty_doc_returns_zero() {
        let m = Model::load(Path::new(FIXTURE)).expect("load fixture");
        let mut out = vec![1.0f32; m.dim()];
        let mut sc = m.scratch();
        m.embed(&[], &mut out, &mut sc).expect("embed");
        assert!(
            out.iter().all(|&x| x == 0.0),
            "empty doc should produce a zero vector"
        );
    }

    #[test]
    fn l2_normalize_unit() {
        let mut v = vec![3.0_f32, 4.0, 0.0];
        kernel::l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "post-norm length should be 1, got {}",
            norm
        );
    }

    #[test]
    fn l2_normalize_zero_stays_zero() {
        let mut v = vec![0.0_f32; 16];
        kernel::l2_normalize(&mut v);
        assert!(v.iter().all(|&x| x == 0.0));
    }

    const TOKENIZER_FIXTURE: &str = "../data/int4-dim-1024/tokenizer.json";

    #[test]
    #[ignore = "requires a local bert-base-uncased tokenizer under data/"]
    fn tokenizer_content_tokens_only() {
        let tk = LatticeTokenizer::load(Path::new(TOKENIZER_FIXTURE)).expect("load tokenizer");
        let ids = tk.encode("hello world").expect("encode");
        // bert-base-uncased: hello=7592, world=2088. Special tokens
        // ([CLS]/[SEP]) are NOT in the bag — matches sentence-transformers
        // StaticEmbedding's behavior, which is what the model was trained on.
        assert_eq!(ids, vec![7592, 2088]);
    }

    #[test]
    #[ignore = "requires a locally generated deployment artifact under data/"]
    fn end_to_end_embed_text() {
        let m = Model::load(Path::new(FIXTURE)).expect("load model");
        let tk = LatticeTokenizer::load(Path::new(TOKENIZER_FIXTURE)).expect("load tokenizer");
        let ids = tk
            .encode("a quick brown fox jumps over the lazy dog")
            .expect("encode");
        let mut out = vec![0.0f32; m.dim()];
        let mut sc = m.scratch();
        m.embed(&ids, &mut out, &mut sc).expect("embed");
        kernel::l2_normalize(&mut out);
        let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "post-l2 norm should be 1.0, got {}",
            norm
        );
    }
}
