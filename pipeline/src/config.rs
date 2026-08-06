//! All tunables live here. Edit and recompile — the CLI does not expose flags.

use clap::ValueEnum;

// ---- Dataset ---------------------------------------------------------------

pub const HF_DATASET: &str = "lightonai/embeddings-pre-training-curated";

/// Subsets that directly contaminate NanoBEIR eval tasks. Per `plan.md`:
/// `beir_dbpedia` (~2.17M) → NanoDBPedia
/// `msmarco` (~3.78M) → NanoMSMARCO
/// `quora` (~44.9k) → NanoQuoraRetrieval
pub const EXCLUDED_SUBSETS: &[&str] = &["beir_dbpedia", "msmarco", "quora"];

// ---- Tokenizer -------------------------------------------------------------

/// The reference model (`static-retrieval-mrl-en-v1`) uses this tokenizer.
/// Vocab size = 30,522, so token IDs fit in `u16`.
pub const TOKENIZER_MODEL: &str = "bert-base-uncased";

/// Sanity check on the loaded tokenizer's vocab size — anything beyond u16
/// would silently corrupt the token files.
pub const TOKENIZER_MAX_VOCAB: u32 = u16::MAX as u32 + 1;

// ---- Parquet schema --------------------------------------------------------

pub const QUERY_COLUMN: &str = "query";
pub const DOC_COLUMN: &str = "document";

/// RecordBatch size out of the parquet stream. Tokenizer's `encode_batch_fast`
/// uses rayon internally; larger batches amortize the rayon overhead.
pub const TOKENIZE_BATCH_SIZE: usize = 1024;

// ---- Local scratch ---------------------------------------------------------

pub const CACHE_ROOT: &str = "./cache";

// ---- Partition interleave --------------------------------------------------

/// Chunk size for the prefix-stable interleaved schedule (see `partition.rs`).
/// Smaller → finer-grained diversity at small tiers; larger → fewer stream
/// switches and less open-stream memory pressure. 10K matches audacity.
pub const INTERLEAVE_CHUNK: u64 = 10_000;

// ---- Tiers -----------------------------------------------------------------

/// `xs ⊂ small ⊂ medium ⊂ full` as row sets per `plan.md`. The schedule is
/// prefix-stable, so the first 10M rows of `full` are exactly the rows of
/// `xs`, etc. This lets the training-time dataloader switch tiers by changing
/// per-source `take_rows` only, never touching the on-disk binaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Tier {
    /// 10M pairs — pipeline validation.
    Xs,
    /// 100M pairs — rapid iteration.
    Small,
    /// 275M pairs — scaling confirmation (plan.md: "250-300M").
    Medium,
    /// All retained rows (~659M after exclusions) — production run.
    Full,
}

impl Tier {
    /// Target row count, or `None` for "everything in the source dataset".
    pub const fn target_rows(self) -> Option<u64> {
        match self {
            Tier::Xs => Some(10_000_000),
            Tier::Small => Some(100_000_000),
            Tier::Medium => Some(275_000_000),
            Tier::Full => None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Tier::Xs => "xs",
            Tier::Small => "small",
            Tier::Medium => "medium",
            Tier::Full => "full",
        }
    }

    pub const fn all() -> [Tier; 4] {
        [Tier::Xs, Tier::Small, Tier::Medium, Tier::Full]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Side {
    Query,
    Doc,
}

impl Side {
    pub const fn name(self) -> &'static str {
        match self {
            Side::Query => "query",
            Side::Doc => "doc",
        }
    }

    pub const fn column(self) -> &'static str {
        match self {
            Side::Query => QUERY_COLUMN,
            Side::Doc => DOC_COLUMN,
        }
    }

    pub const fn all() -> [Side; 2] {
        [Side::Query, Side::Doc]
    }
}
