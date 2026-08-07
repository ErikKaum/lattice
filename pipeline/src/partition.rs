//! Decide the global row order across source subsets — prefix-stable so that
//! `xs ⊂ small ⊂ medium ⊂ full` holds as row sets, not just sizes.
//!
//! Algorithm (adapted from an earlier pipeline): chunked
//! proportional round-robin. At each step the schedule picks the source whose
//! `taken / size` ratio is currently smallest (ties broken by source index)
//! and advances it by up to `INTERLEAVE_CHUNK` rows. Two targets share the
//! same first N decisions whenever the algorithm reaches `taken_total = N` —
//! which it always does deterministically as long as the source set is the
//! same.
//!
//! Persisted as `cache/partition.json` (the full-tier schedule). Per-tier
//! `take_rows` derivable as a prefix of that schedule.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::config::{CACHE_ROOT, EXCLUDED_SUBSETS, INTERLEAVE_CHUNK, Tier};
use crate::hf::{ParquetTree, dataset_sizes, discover_parquet_tree};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionEntry {
    pub source: String,
    pub urls: Vec<String>,
    pub source_size: u64,
    pub take_rows: u64,
}

/// One chunk of contiguous rows pulled from a single source, in global order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduleChunk {
    /// Index into `PartitionPlan::entries`.
    pub source_idx: u32,
    /// First row within that source (inclusive).
    pub row_start: u64,
    /// One past the last row within that source.
    pub row_end: u64,
}

impl ScheduleChunk {
    pub fn len(&self) -> u64 {
        self.row_end - self.row_start
    }

    pub fn is_empty(&self) -> bool {
        self.row_start == self.row_end
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionPlan {
    pub total_rows: u64,
    pub entries: Vec<PartitionEntry>,
    /// Global row order as a sequence of `(source_idx, row_range)` chunks.
    /// `entries[i].take_rows == sum of chunk.len() over chunks where chunk.source_idx == i`.
    pub schedule: Vec<ScheduleChunk>,
}

impl PartitionPlan {
    pub fn n_sources(&self) -> usize {
        self.entries.len()
    }

    /// Per-source `take_rows` if we walked `target` rows of the schedule.
    /// Prefix stability means the result for the same `target` is identical
    /// regardless of the larger schedule it was sliced from.
    pub fn take_rows_for_target(&self, target: u64) -> Vec<u64> {
        let mut takes = vec![0u64; self.entries.len()];
        let mut remaining = target;
        for chunk in &self.schedule {
            if remaining == 0 {
                break;
            }
            let step = chunk.len().min(remaining);
            takes[chunk.source_idx as usize] += step;
            remaining -= step;
        }
        takes
    }
}

pub fn cache_root() -> PathBuf {
    PathBuf::from(CACHE_ROOT)
}

pub fn partition_path() -> PathBuf {
    cache_root().join("partition.json")
}

pub fn tiers_path() -> PathBuf {
    cache_root().join("tiers.json")
}

/// `cache/subsets/<source>/`
pub fn source_dir(source: &str) -> PathBuf {
    cache_root().join("subsets").join(source)
}

pub async fn build_or_load() -> Result<PartitionPlan> {
    let path = partition_path();
    if path.exists() {
        let s = std::fs::read_to_string(&path)?;
        let plan: PartitionPlan = serde_json::from_str(&s)?;
        info!(
            sources = plan.n_sources(),
            total_rows = plan.total_rows,
            chunks = plan.schedule.len(),
            "Loaded cached partition"
        );
        return Ok(plan);
    }

    let tree = discover_parquet_tree().await?;
    let sizes = dataset_sizes().await?;
    let plan = build(&tree, &sizes);

    std::fs::create_dir_all(cache_root())?;
    crate::atomic::atomic_write(&path, serde_json::to_string_pretty(&plan)?.as_bytes())?;
    info!(
        sources = plan.n_sources(),
        total_rows = plan.total_rows,
        chunks = plan.schedule.len(),
        "Built partition plan -> {}",
        path.display()
    );
    Ok(plan)
}

fn build(tree: &ParquetTree, sizes: &BTreeMap<String, u64>) -> PartitionPlan {
    // BTreeMap iteration is sorted by key — keeps source ordering deterministic
    // across runs. Filter out exclusions and zero-sized configs here.
    let sources: Vec<(&String, u64)> = tree
        .configs
        .iter()
        .filter(|(name, _)| !EXCLUDED_SUBSETS.contains(&name.as_str()))
        .map(|(name, _urls)| {
            let n = sizes.get(name).copied().unwrap_or(0);
            (name, n)
        })
        .filter(|(_, n)| *n > 0)
        .collect();

    for (name, n) in &sources {
        info!("  source {name}: {n} rows");
    }
    for excl in EXCLUDED_SUBSETS {
        info!("  excluded {excl} (NanoBEIR contamination)");
    }

    let sizes_vec: Vec<u64> = sources.iter().map(|(_, s)| *s).collect();
    let target: u64 = sizes_vec.iter().sum();

    let schedule = build_schedule(&sizes_vec, target, INTERLEAVE_CHUNK);

    let mut takes = vec![0u64; sources.len()];
    for chunk in &schedule {
        takes[chunk.source_idx as usize] += chunk.len();
    }

    let entries: Vec<PartitionEntry> = sources
        .iter()
        .zip(takes.iter())
        .map(|((name, size), take)| PartitionEntry {
            source: (*name).clone(),
            urls: tree.configs.get(*name).cloned().unwrap_or_default(),
            source_size: *size,
            take_rows: *take,
        })
        .collect();

    PartitionPlan {
        total_rows: target,
        entries,
        schedule,
    }
}

/// Build a prefix-stable schedule pulling `target` rows across sources.
///
/// Pure function of `(sizes, target, chunk)` — same inputs give the same
/// schedule byte for byte. The first N rows of the schedule for target T
/// equal the schedule for target N, for any N ≤ T.
pub fn build_schedule(sizes: &[u64], target: u64, chunk: u64) -> Vec<ScheduleChunk> {
    let total_cap: u64 = sizes.iter().sum();
    let target = target.min(total_cap);
    let mut taken = vec![0u64; sizes.len()];
    let mut total = 0u64;
    let mut out: Vec<ScheduleChunk> = Vec::new();

    while total < target {
        let Some(c) = pick_next(sizes, &taken) else {
            break;
        };
        let headroom = sizes[c] - taken[c];
        let remaining = target - total;
        let take = chunk.min(headroom).min(remaining);
        // Coalesce with the previous chunk if it's from the same source —
        // happens only when one source is the only one with headroom left.
        if let Some(last) = out.last_mut()
            && last.source_idx as usize == c
        {
            last.row_end += take;
        } else {
            out.push(ScheduleChunk {
                source_idx: c as u32,
                row_start: taken[c],
                row_end: taken[c] + take,
            });
        }
        taken[c] += take;
        total += take;
    }

    out
}

/// Among the sources with headroom, pick the one with the smallest
/// `taken / size` ratio, ties broken by index. Returns `None` if every
/// source is exhausted.
fn pick_next(sizes: &[u64], taken: &[u64]) -> Option<usize> {
    (0..sizes.len())
        .filter(|&i| sizes[i] > 0 && taken[i] < sizes[i])
        .min_by(|&a, &b| {
            // Cross-multiply (u128 to avoid overflow on big sources).
            let lhs = taken[a] as u128 * sizes[b] as u128;
            let rhs = taken[b] as u128 * sizes[a] as u128;
            lhs.cmp(&rhs).then(a.cmp(&b))
        })
}

/// Persist per-tier per-source `take_rows` derived from the global schedule.
pub fn write_tiers(plan: &PartitionPlan) -> Result<()> {
    #[derive(Serialize)]
    struct TierEntry<'a> {
        source: &'a str,
        take_rows: u64,
    }
    #[derive(Serialize)]
    struct TierMap<'a> {
        tier: &'a str,
        target_rows: u64,
        actual_rows: u64,
        per_source: Vec<TierEntry<'a>>,
    }

    let mut tiers: Vec<TierMap<'_>> = Vec::new();
    for tier in Tier::all() {
        let target = tier.target_rows().unwrap_or(plan.total_rows);
        let takes = plan.take_rows_for_target(target);
        let actual: u64 = takes.iter().sum();
        let per_source: Vec<TierEntry<'_>> = plan
            .entries
            .iter()
            .zip(takes.iter())
            .map(|(e, t)| TierEntry {
                source: &e.source,
                take_rows: *t,
            })
            .collect();
        tiers.push(TierMap {
            tier: tier.name(),
            target_rows: target,
            actual_rows: actual,
            per_source,
        });
    }
    std::fs::create_dir_all(cache_root())?;
    crate::atomic::atomic_write(
        &tiers_path(),
        serde_json::to_string_pretty(&tiers)?.as_bytes(),
    )?;
    info!("Wrote tier map -> {}", tiers_path().display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schedule_to_takes(sched: &[ScheduleChunk], n: usize) -> Vec<u64> {
        let mut takes = vec![0u64; n];
        for c in sched {
            takes[c.source_idx as usize] += c.len();
        }
        takes
    }

    #[test]
    fn even_split_balanced_within_chunk_size() {
        let chunk = 10;
        let sched = build_schedule(&[1_000_000; 4], 100, chunk);
        let takes = schedule_to_takes(&sched, 4);
        assert_eq!(takes.iter().sum::<u64>(), 100);
        let max = *takes.iter().max().unwrap();
        let min = *takes.iter().min().unwrap();
        assert!(max - min <= chunk);
    }

    #[test]
    fn small_sources_capped() {
        let sched = build_schedule(&[100, 100, 10_000], 1000, 100);
        let takes = schedule_to_takes(&sched, 3);
        assert_eq!(takes[0], 100);
        assert_eq!(takes[1], 100);
        assert_eq!(takes[2], 800);
    }

    #[test]
    fn target_exceeds_total() {
        let sched = build_schedule(&[10, 20, 30], 1000, 100);
        let takes = schedule_to_takes(&sched, 3);
        assert_eq!(takes, vec![10, 20, 30]);
    }

    #[test]
    fn prefix_stability_small_to_large() {
        // The keystone: schedule[..N] for target T is the schedule for target N.
        let sizes = vec![5_000_000, 10_000_000, 20_000_000, 50_000_000, 100_000_000];
        let big = build_schedule(&sizes, 50_000_000, 10_000);

        for target in [100_000u64, 1_000_000, 10_000_000, 25_000_000] {
            let small = build_schedule(&sizes, target, 10_000);
            let small_total: u64 = small.iter().map(|c| c.len()).sum();
            assert_eq!(small_total, target);

            let mut walked = 0u64;
            let (mut si, mut bi, mut s_off, mut b_off) = (0usize, 0usize, 0u64, 0u64);
            while walked < target {
                let s_chunk = &small[si];
                let b_chunk = &big[bi];
                let step = (s_chunk.len() - s_off).min(b_chunk.len() - b_off);
                assert_eq!(s_chunk.source_idx, b_chunk.source_idx);
                assert_eq!(s_chunk.row_start + s_off, b_chunk.row_start + b_off);
                walked += step;
                s_off += step;
                b_off += step;
                if s_off == s_chunk.len() {
                    si += 1;
                    s_off = 0;
                }
                if b_off == b_chunk.len() {
                    bi += 1;
                    b_off = 0;
                }
            }
        }
    }

    #[test]
    fn take_rows_for_target_matches_prefix() {
        let sizes = vec![1_000u64, 2_000, 5_000];
        let full = build_schedule(&sizes, 8_000, 100);
        let entries: Vec<PartitionEntry> = sizes
            .iter()
            .enumerate()
            .map(|(i, s)| PartitionEntry {
                source: format!("s{i}"),
                urls: vec![],
                source_size: *s,
                take_rows: 0,
            })
            .collect();
        let plan = PartitionPlan {
            total_rows: 8_000,
            entries,
            schedule: full,
        };

        for target in [0u64, 500, 1_000, 4_000, 8_000, 99_999] {
            let derived = plan.take_rows_for_target(target);
            let bounded = target.min(8_000);
            let independent = schedule_to_takes(&build_schedule(&sizes, bounded, 100), sizes.len());
            assert_eq!(derived, independent, "mismatch at target={target}");
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let sizes = vec![3, 7, 11, 13];
        let a = build_schedule(&sizes, 25, 2);
        let b = build_schedule(&sizes, 25, 2);
        assert_eq!(a, b);
    }
}
