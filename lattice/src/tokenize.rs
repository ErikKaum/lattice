//! Thin wrapper over the HF `tokenizers` crate.
//!
//! We use the HF crate as-is for now — it's correct on all Unicode inputs,
//! and replacing it with a custom WordPiece is a non-trivial undertaking
//! (Unicode normalization, accent stripping, CJK handling) that's been
//! punted to a follow-up research milestone. The single `encode` path is
//! what training and eval use, so output token IDs match exactly.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use tokenizers::Tokenizer;

pub struct LatticeTokenizer {
    inner: Tokenizer,
}

impl LatticeTokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| anyhow!("{}", e))
            .with_context(|| format!("loading tokenizer from {}", path.display()))?;
        Ok(Self { inner })
    }

    /// Encode one document. **Special tokens (`[CLS]`/`[SEP]`) are NOT
    /// added** — this matches what `sentence-transformers`'
    /// `StaticEmbedding` feeds the `EmbeddingBag`. Verified by the parity
    /// test against ST. (The tokenizer.json's TemplateProcessing
    /// post-processor still defines them; we just bypass it via
    /// `add_special_tokens=false`.)
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow!("tokenizer encode: {}", e))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Encode a batch. The HF tokenizer parallelizes internally via rayon,
    /// so call this from a sequential context — calling it from within
    /// another rayon scope would nest two parallel iterations.
    pub fn encode_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<u32>>> {
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

    const FIXTURE: &str = "../data/int4-dim-1024/tokenizer.json";

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
