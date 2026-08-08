//! Thin wrapper over the HF `tokenizers` crate, with a fast IDs-only path
//! for the WordPiece pipeline lattice-retrieval actually ships.
//!
//! `Tokenizer::encode`/`encode_batch` build full `Encoding` objects (offsets,
//! alignment, masks, type IDs, special tokens) of which we use only
//! `get_ids()`. [`fast_wordpiece::FastWordPiece`] skips that machinery for
//! the BertNormalizer + BertPreTokenizer + WordPiece pipeline — the only
//! pipeline lattice-retrieval's slicer artifacts use — and falls back to the
//! crate for anything else, so correctness on non-WordPiece tokenizers is
//! unaffected.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rayon::prelude::*;
use tokenizers::Tokenizer;

use crate::fast_wordpiece::FastWordPiece;

pub struct LatticeTokenizer {
    inner: Tokenizer,
    fast: Option<FastWordPiece>,
}

impl LatticeTokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| anyhow!("{}", e))
            .with_context(|| format!("loading tokenizer from {}", path.display()))?;
        let fast = FastWordPiece::from_tokenizer(&inner);
        Ok(Self { inner, fast })
    }

    /// Encode one document. **Special tokens (`[CLS]`/`[SEP]`) are NOT
    /// added** — this matches what `sentence-transformers`'
    /// `StaticEmbedding` feeds the `EmbeddingBag`. Verified by the parity
    /// test against ST. (The tokenizer.json's TemplateProcessing
    /// post-processor still defines them; we just bypass it via
    /// `add_special_tokens=false`.)
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if let Some(fast) = &self.fast {
            return Ok(fast.encode_ids(text));
        }
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow!("tokenizer encode: {}", e))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Encode a batch. On the fast path we parallelize across texts with
    /// rayon ourselves (the crate's own `encode_batch` parallelizes
    /// internally, but `FastWordPiece::encode_ids` is single-threaded), so
    /// call this from a sequential context — calling it from within another
    /// rayon scope would nest two parallel iterations.
    pub fn encode_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<u32>>> {
        if let Some(fast) = &self.fast {
            return Ok(texts.par_iter().map(|t| fast.encode_ids(t)).collect());
        }
        let encs = self
            .inner
            .encode_batch(texts, false)
            .map_err(|e| anyhow!("tokenizer encode_batch: {}", e))?;
        Ok(encs.iter().map(|e| e.get_ids().to_vec()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = "../data/int4-dim-512-artifact/tokenizer.json";

    #[test]
    #[ignore = "requires a local bert-base-uncased tokenizer under data/"]
    fn no_special_tokens_in_bag() {
        let tk = LatticeTokenizer::load(Path::new(FIXTURE)).expect("load tokenizer");
        let ids = tk.encode("hello world").expect("encode");
        // bert-base-uncased: [CLS]=101, [SEP]=102, hello=7592, world=2088
        assert_eq!(
            ids,
            vec![7592, 2088],
            "content tokens only — no CLS/SEP (matches sentence-transformers StaticEmbedding)"
        );
    }

    #[test]
    #[ignore = "requires a local bert-base-uncased tokenizer under data/"]
    fn empty_returns_empty() {
        let tk = LatticeTokenizer::load(Path::new(FIXTURE)).expect("load tokenizer");
        let ids = tk.encode("").expect("encode empty");
        // With add_special_tokens=false, empty text → empty token list.
        assert_eq!(ids, Vec::<u32>::new());
    }
}
