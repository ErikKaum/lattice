//! Sanity printer over the cache. `pipeline inspect <what>`:
//!
//! - `partition` — config sizes + per-tier per-source `take_rows`
//! - `subset <name>` — token / row counts for one source's binaries
//! - `subsets` — short summary table across all tokenized sources

use std::path::Path;

use anyhow::{Context, Result};
use tracing::info;

use crate::config::Tier;
use crate::partition::{build_or_load, source_dir};
use crate::tokenize::SourceMeta;

pub async fn run(what: &str) -> Result<()> {
    let parts: Vec<&str> = what.split_whitespace().collect();
    match parts.as_slice() {
        ["partition"] => inspect_partition().await,
        ["subsets"] => inspect_subsets().await,
        ["subset", name] => inspect_subset(name),
        _ => Err(anyhow::anyhow!(
            "unknown inspect target {what:?}; expected one of: partition, subsets, subset <name>"
        )),
    }
}

async fn inspect_partition() -> Result<()> {
    let plan = build_or_load().await?;
    println!("sources: {}", plan.n_sources());
    println!("total rows (after exclusions): {}", plan.total_rows);
    println!();
    println!("{:<24}  {:>14}", "source", "rows");
    for entry in &plan.entries {
        println!("{:<24}  {:>14}", entry.source, entry.source_size);
    }
    println!();
    println!("tiers:");
    for tier in Tier::all() {
        let target = tier.target_rows().unwrap_or(plan.total_rows);
        let takes = plan.take_rows_for_target(target);
        let actual: u64 = takes.iter().sum();
        println!(
            "  {:<6} target={:>12}  actual={:>12}  sources_touched={}",
            tier.name(),
            target,
            actual,
            takes.iter().filter(|t| **t > 0).count()
        );
    }
    Ok(())
}

async fn inspect_subsets() -> Result<()> {
    let plan = build_or_load().await?;
    println!(
        "{:<24}  {:>12}  {:>14}  {:>14}",
        "source", "rows", "query_tokens", "doc_tokens"
    );
    let mut total_rows = 0u64;
    let mut total_q = 0u64;
    let mut total_d = 0u64;
    for entry in &plan.entries {
        let dir = source_dir(&entry.source);
        match read_meta_opt(&dir)? {
            Some(meta) => {
                println!(
                    "{:<24}  {:>12}  {:>14}  {:>14}",
                    meta.source, meta.n_rows, meta.total_query_tokens, meta.total_doc_tokens
                );
                total_rows += meta.n_rows;
                total_q += meta.total_query_tokens;
                total_d += meta.total_doc_tokens;
            }
            None => {
                println!(
                    "{:<24}  {:>12}  {:>14}  {:>14}",
                    entry.source, "-", "-", "-"
                );
            }
        }
    }
    println!(
        "{:<24}  {:>12}  {:>14}  {:>14}",
        "TOTAL", total_rows, total_q, total_d
    );
    Ok(())
}

fn inspect_subset(name: &str) -> Result<()> {
    let dir = source_dir(name);
    let meta =
        read_meta_opt(&dir)?.with_context(|| format!("no tokenized data for source {name}"))?;
    println!("{}", serde_json::to_string_pretty(&meta)?);

    for side in crate::config::Side::all() {
        let tokens = dir.join(format!("{}_tokens.bin", side.name()));
        let offsets = dir.join(format!("{}_offsets.bin", side.name()));
        let t_bytes = std::fs::metadata(&tokens)?.len();
        let o_bytes = std::fs::metadata(&offsets)?.len();
        let n_offsets = o_bytes / 8;
        info!(
            side = side.name(),
            tokens_bytes = t_bytes,
            offsets_bytes = o_bytes,
            n_offsets,
            implied_rows = n_offsets.saturating_sub(1),
            "side"
        );
    }
    Ok(())
}

fn read_meta_opt(dir: &Path) -> Result<Option<SourceMeta>> {
    let p = dir.join("meta.json");
    if !p.exists() {
        return Ok(None);
    }
    let s = std::fs::read_to_string(&p)?;
    Ok(Some(serde_json::from_str(&s)?))
}
