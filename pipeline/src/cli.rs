//! Thin clap layer. All tunables live in `config.rs`; the CLI picks the stage
//! and which tier it operates on.

use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::config::Tier;

#[derive(Parser, Debug)]
#[command(
    name = "pipeline",
    about = "lattice static-embedding training data pipeline"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Build (or refresh) the partition plan and per-tier `take_rows` manifest.
    /// No tokenization — just reads HF metadata.
    Plan,

    /// Stream parquet, tokenize with BERT-base-uncased, append to per-source
    /// token binaries. Resumes safely if some sources are already tokenized
    /// to a lower tier — only the missing rows are processed.
    Tokenize { tier: Tier },

    /// Print stats over the cache (`partition`, `subsets`, `subset <name>`).
    Inspect {
        /// Concatenated as one string so `subset arxiv` works without quoting.
        #[arg(num_args = 1.., trailing_var_arg = true)]
        what: Vec<String>,
    },
}

pub async fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Plan => {
            let plan = crate::partition::build_or_load().await?;
            crate::partition::write_tiers(&plan)?;
            Ok(())
        }
        Command::Tokenize { tier } => crate::tokenize::run(tier).await,
        Command::Inspect { what } => crate::inspect::run(&what.join(" ")).await,
    }
}
