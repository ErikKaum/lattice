//! IDs-only WordPiece tokenizer for lattice-retrieval's BertNormalizer +
//! BertPreTokenizer + WordPiece pipeline. See [`fast_wordpiece`] for why the
//! HF `tokenizers` crate isn't a runtime dependency at all — it's kept as a
//! dev-dependency purely for the parity test.

use std::path::Path;

use anyhow::Result;
use rayon::prelude::*;

use crate::fast_wordpiece::FastWordPiece;

pub struct LatticeTokenizer {
    fast: FastWordPiece,
}

impl LatticeTokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        Ok(Self { fast: FastWordPiece::load(path)? })
    }

    /// Encode one document. **Special tokens (`[CLS]`/`[SEP]`) are NOT
    /// added** — this matches what `sentence-transformers`'
    /// `StaticEmbedding` feeds the `EmbeddingBag`. Verified by the parity
    /// test against ST.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.fast.encode_ids(text))
    }

    /// Encode a batch, parallelized across texts with rayon (`FastWordPiece`
    /// is single-threaded per call). Call from a sequential context —
    /// calling from within another rayon scope would nest two parallel
    /// iterations.
    pub fn encode_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<u32>>> {
        Ok(texts.par_iter().map(|t| self.fast.encode_ids(t)).collect())
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
