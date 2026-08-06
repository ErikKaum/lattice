//! Stream parquet → BERT-base-uncased tokenize → per-source token binaries.
//!
//! Layout under `cache/subsets/<source>/`:
//!   query_tokens.bin   — concatenated u16 token IDs, no padding
//!   query_offsets.bin  — u64 starting offset per row, plus a final sentinel
//!   doc_tokens.bin     — same for documents
//!   doc_offsets.bin    — same
//!   meta.json          — n_rows, total_{query,doc}_tokens, tokenizer info
//!
//! No truncation, no padding, per `plan.md`. Special tokens (`[CLS]`/`[SEP]`)
//! are not added: `sentence-transformers`'s `StaticEmbedding.tokenize` calls
//! the tokenizer with `add_special_tokens=false`.
//!
//! Tier-aware: `tokenize <tier>` walks every retained source and tokenizes
//! exactly `take_rows_for_target(tier)` rows from each. Append-safe: if a
//! source already has ≥ target rows on disk, it is skipped; if it has fewer,
//! the remaining rows are appended without re-tokenizing the existing ones.
//! This is what makes `xs → small → medium → full` cheap on upgrade.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use arrow::array::{Array, RecordBatch, StringArray};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use tokio::task;
use tracing::{info, warn};

use crate::config::{
    DOC_COLUMN, QUERY_COLUMN, Side, TOKENIZE_BATCH_SIZE, TOKENIZER_MAX_VOCAB, Tier,
};
use crate::hf::{build_http_client, load_tokenizer, open_stream};
use crate::partition::{PartitionEntry, build_or_load, source_dir, write_tiers};

type BatchStream = Pin<Box<dyn Stream<Item = parquet::errors::Result<RecordBatch>> + Send>>;

const ADD_SPECIAL_TOKENS: bool = false;

fn legacy_add_special_tokens() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMeta {
    pub source: String,
    pub n_rows: u64,
    pub total_query_tokens: u64,
    pub total_doc_tokens: u64,
    pub tokenizer: String,
    pub tokens_dtype: String,
    pub offsets_dtype: String,
    #[serde(default = "legacy_add_special_tokens")]
    pub add_special_tokens: bool,
}

impl SourceMeta {
    fn empty(source: &str, tokenizer: &str) -> Self {
        Self {
            source: source.to_string(),
            n_rows: 0,
            total_query_tokens: 0,
            total_doc_tokens: 0,
            tokenizer: tokenizer.to_string(),
            tokens_dtype: "u16".to_string(),
            offsets_dtype: "u64".to_string(),
            add_special_tokens: ADD_SPECIAL_TOKENS,
        }
    }
}

/// Appender for a single side of a single source. Maintains the offsets file's
/// sentinel invariant: every flush ends with `offsets[N] = total_tokens`.
struct SideAppender {
    tokens: BufWriter<File>,
    offsets: BufWriter<File>,
    cur_offset: u64,
    n_rows: u64,
}

impl SideAppender {
    /// Open the side's files for appending. `meta.json` (via
    /// `existing_rows` and `existing_tokens`) is the sole source of truth
    /// for "where the last finalized run left this side." Both files are
    /// unconditionally normalized to that state — extra bytes from a
    /// killed previous run, a half-stripped sentinel, or anything else
    /// past the meta.json boundary get truncated away. Anything *short* of
    /// the meta.json boundary is real data loss and hard-errors.
    fn open(dir: &Path, side: Side, existing_rows: u64, existing_tokens: u64) -> Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;

        let tokens_path = dir.join(format!("{}_tokens.bin", side.name()));
        let offsets_path = dir.join(format!("{}_offsets.bin", side.name()));

        if existing_rows == 0 {
            // Fresh start — overwrite anything stale.
            let tokens = BufWriter::with_capacity(1 << 20, File::create(&tokens_path)?);
            let offsets = BufWriter::with_capacity(1 << 16, File::create(&offsets_path)?);
            return Ok(Self {
                tokens,
                offsets,
                cur_offset: 0,
                n_rows: 0,
            });
        }

        // Offsets file: must have ≥ existing_rows entries (8 bytes each).
        // The (existing_rows+1)-th entry — the sentinel — may or may not
        // be present depending on whether the previous run finalized,
        // crashed pre-strip, or crashed post-strip. Either way, we
        // canonicalize to "exactly existing_rows entries, no sentinel"
        // and then append fresh; finalize() will write the new sentinel.
        let min_offsets_len = existing_rows * 8;
        let actual_offsets_len = std::fs::metadata(&offsets_path)?.len();
        if actual_offsets_len < min_offsets_len {
            bail!(
                "offsets file {} has {} bytes; expected at least {} for {} rows",
                offsets_path.display(),
                actual_offsets_len,
                min_offsets_len,
                existing_rows
            );
        }
        if actual_offsets_len != min_offsets_len {
            tracing::info!(
                path = %offsets_path.display(),
                actual = actual_offsets_len,
                target = min_offsets_len,
                "crash-recovery: normalizing offsets file to meta.json size"
            );
            OpenOptions::new()
                .write(true)
                .open(&offsets_path)?
                .set_len(min_offsets_len)?;
        }

        // Tokens file: must have ≥ existing_tokens u16s. Trim back to
        // exactly that — anything past is post-crash garbage.
        let expected_tokens_len = existing_tokens * 2; // u16 = 2 bytes
        let actual_tokens_len = std::fs::metadata(&tokens_path)?.len();
        if actual_tokens_len < expected_tokens_len {
            bail!(
                "tokens file {} has {} bytes; expected at least {} ({} u16s per meta.json)",
                tokens_path.display(),
                actual_tokens_len,
                expected_tokens_len,
                existing_tokens
            );
        }
        if actual_tokens_len != expected_tokens_len {
            tracing::info!(
                path = %tokens_path.display(),
                actual = actual_tokens_len,
                target = expected_tokens_len,
                "crash-recovery: normalizing tokens file to meta.json size"
            );
            OpenOptions::new()
                .write(true)
                .open(&tokens_path)?
                .set_len(expected_tokens_len)?;
        }

        let mut offsets_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&offsets_path)?;
        offsets_file.seek(SeekFrom::End(0))?;
        let offsets = BufWriter::with_capacity(1 << 16, offsets_file);

        let tokens_file = OpenOptions::new().append(true).open(&tokens_path)?;
        let tokens = BufWriter::with_capacity(1 << 20, tokens_file);

        let cur_offset = existing_tokens;

        Ok(Self {
            tokens,
            offsets,
            cur_offset,
            n_rows: existing_rows,
        })
    }

    fn append(&mut self, ids: &[u32]) -> Result<()> {
        // offsets[i] = starting offset of row i, written before tokens append.
        self.offsets.write_all(&self.cur_offset.to_le_bytes())?;
        // u32 → u16 narrow with explicit vocab check.
        for id in ids {
            if *id >= TOKENIZER_MAX_VOCAB {
                bail!(
                    "token id {} exceeds u16 vocab cap {}",
                    id,
                    TOKENIZER_MAX_VOCAB
                );
            }
            let id_u16 = *id as u16;
            self.tokens.write_all(&id_u16.to_le_bytes())?;
        }
        self.cur_offset += ids.len() as u64;
        self.n_rows += 1;
        Ok(())
    }

    fn finalize(mut self) -> Result<(u64, u64)> {
        // Sentinel.
        self.offsets.write_all(&self.cur_offset.to_le_bytes())?;
        self.tokens.flush()?;
        self.offsets.flush()?;
        Ok((self.n_rows, self.cur_offset))
    }
}

fn read_meta(dir: &Path) -> Result<Option<SourceMeta>> {
    let p = dir.join("meta.json");
    if !p.exists() {
        return Ok(None);
    }
    let s = std::fs::read_to_string(&p)?;
    Ok(Some(serde_json::from_str(&s)?))
}

fn write_meta(dir: &Path, meta: &SourceMeta) -> Result<()> {
    // Atomic write — meta.json is the source of truth for crash-recovery,
    // so a partial write here would brick the source on next open.
    crate::atomic::atomic_write(
        &dir.join("meta.json"),
        serde_json::to_string_pretty(meta)?.as_bytes(),
    )
}

fn validate_cache_token_policies(entries: &[PartitionEntry], takes: &[u64]) -> Result<()> {
    let mut observed: Option<(bool, &str)> = None;
    let mut source_needing_new_rows: Option<&str> = None;

    for (entry, requested_rows) in entries.iter().zip(takes.iter()) {
        let target_rows = (*requested_rows).min(entry.source_size);
        if target_rows == 0 {
            continue;
        }
        let existing = read_meta(&source_dir(&entry.source))?;
        let existing_rows = existing.as_ref().map_or(0, |meta| meta.n_rows);
        if existing_rows < target_rows {
            source_needing_new_rows.get_or_insert(&entry.source);
        }
        let Some(meta) = existing.filter(|meta| meta.n_rows > 0) else {
            continue;
        };
        if let Some((policy, first_source)) = observed {
            if policy != meta.add_special_tokens {
                bail!(
                    "tier mixes tokenizer policies: {} has \
                     add_special_tokens={} while {} has \
                     add_special_tokens={}; rebuild into one canonical cache",
                    first_source,
                    policy,
                    entry.source,
                    meta.add_special_tokens,
                );
            }
        } else {
            observed = Some((meta.add_special_tokens, &entry.source));
        }
        if existing_rows < target_rows && meta.add_special_tokens != ADD_SPECIAL_TOKENS {
            bail!(
                "cannot extend legacy cache for {} from {} to {} rows; \
                 rebuild this source with add_special_tokens=false",
                entry.source,
                existing_rows,
                target_rows,
            );
        }
    }

    if let (Some((true, legacy_source)), Some(new_source)) = (observed, source_needing_new_rows) {
        bail!(
            "cannot add canonical no-special-token rows for {} while tier \
             reuses legacy special-token cache {}; use a fresh cache root",
            new_source,
            legacy_source,
        );
    }
    Ok(())
}

/// Per-source parquet stream walking `urls` in order.
struct SourceStream {
    source_name: String,
    urls: Vec<String>,
    url_idx: usize,
    stream: Option<BatchStream>,
    leftover: Option<(RecordBatch, usize)>,
}

impl SourceStream {
    fn new(source_name: String, urls: Vec<String>) -> Self {
        Self {
            source_name,
            urls,
            url_idx: 0,
            stream: None,
            leftover: None,
        }
    }

    async fn pull(
        &mut self,
        want: usize,
        client: &reqwest::Client,
    ) -> Result<Vec<(String, String)>> {
        let mut out: Vec<(String, String)> = Vec::with_capacity(want);

        while out.len() < want {
            if let Some((batch, cursor)) = self.leftover.take() {
                let avail = batch.num_rows() - cursor;
                let take = (want - out.len()).min(avail);
                extract_pairs(&batch, cursor, take, &mut out)?;
                let new_cursor = cursor + take;
                if new_cursor < batch.num_rows() {
                    self.leftover = Some((batch, new_cursor));
                }
                continue;
            }

            if self.stream.is_none() {
                if self.url_idx >= self.urls.len() {
                    break;
                }
                let url = self.urls[self.url_idx].clone();
                let s = open_stream(
                    client.clone(),
                    &url,
                    &[QUERY_COLUMN, DOC_COLUMN],
                    TOKENIZE_BATCH_SIZE,
                )
                .await
                .with_context(|| format!("open stream {url}"))?;
                self.stream = Some(Box::pin(s));
            }

            match self.stream.as_mut().unwrap().next().await {
                Some(batch_result) => {
                    let batch = batch_result
                        .with_context(|| format!("read batch from {}", self.source_name))?;
                    if batch.num_rows() > 0 {
                        self.leftover = Some((batch, 0));
                    }
                }
                None => {
                    self.stream = None;
                    self.url_idx += 1;
                }
            }
        }

        Ok(out)
    }

    /// Pull `n` rows and discard them — used to fast-forward past rows already
    /// tokenized on a previous run.
    async fn skip(&mut self, n: u64, client: &reqwest::Client) -> Result<()> {
        let mut remaining = n;
        while remaining > 0 {
            let want = remaining.min(TOKENIZE_BATCH_SIZE as u64) as usize;
            let pairs = self.pull(want, client).await?;
            if pairs.is_empty() {
                bail!(
                    "source {} exhausted while skipping; wanted {} more",
                    self.source_name,
                    remaining
                );
            }
            remaining -= pairs.len() as u64;
        }
        Ok(())
    }
}

fn extract_pairs(
    batch: &RecordBatch,
    start: usize,
    take: usize,
    out: &mut Vec<(String, String)>,
) -> Result<()> {
    let q_col = batch
        .column_by_name(QUERY_COLUMN)
        .ok_or_else(|| anyhow!("missing column {QUERY_COLUMN}"))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow!("{QUERY_COLUMN} not Utf8"))?;
    let d_col = batch
        .column_by_name(DOC_COLUMN)
        .ok_or_else(|| anyhow!("missing column {DOC_COLUMN}"))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow!("{DOC_COLUMN} not Utf8"))?;
    for i in start..start + take {
        out.push((q_col.value(i).to_string(), d_col.value(i).to_string()));
    }
    Ok(())
}

async fn tokenize_source(
    entry: &PartitionEntry,
    target_rows: u64,
    tokenizer: &Tokenizer,
    client: &reqwest::Client,
) -> Result<()> {
    let dir = source_dir(&entry.source);
    let existing = read_meta(&dir)?.unwrap_or_else(|| SourceMeta::empty(&entry.source, ""));
    let existing_rows = existing.n_rows;
    if target_rows > entry.source_size {
        warn!(
            source = entry.source,
            target_rows,
            source_size = entry.source_size,
            "target exceeds source size; clamping"
        );
    }
    let target_rows = target_rows.min(entry.source_size);

    if existing_rows >= target_rows {
        if existing.add_special_tokens != ADD_SPECIAL_TOKENS {
            warn!(
                source = entry.source,
                existing_rows,
                "skip legacy cache with special tokens; trainer compatibility \
                 mode will ignore [CLS]/[SEP]"
            );
        }
        info!(
            source = entry.source,
            existing_rows, target_rows, "skip (already tokenized to target)"
        );
        return Ok(());
    }
    if existing_rows > 0 && existing.add_special_tokens != ADD_SPECIAL_TOKENS {
        bail!(
            "cannot append canonical no-special-token rows to legacy cache for \
             {} ({} existing rows); rebuild this source or keep using the \
             completed legacy cache through trainer compatibility mode",
            entry.source,
            existing_rows,
        );
    }
    let need = target_rows - existing_rows;

    let mut q_writer = SideAppender::open(
        &dir,
        Side::Query,
        existing_rows,
        existing.total_query_tokens,
    )?;
    let mut d_writer =
        SideAppender::open(&dir, Side::Doc, existing_rows, existing.total_doc_tokens)?;

    let mut stream = SourceStream::new(entry.source.clone(), entry.urls.clone());
    if existing_rows > 0 {
        info!(
            source = entry.source,
            existing_rows, "fast-forwarding parquet stream to resume"
        );
        stream.skip(existing_rows, client).await?;
    }

    let start = Instant::now();
    let mut written: u64 = 0;
    while written < need {
        let want = (need - written).min(TOKENIZE_BATCH_SIZE as u64) as usize;
        let pairs = stream.pull(want, client).await?;
        if pairs.is_empty() {
            bail!(
                "source {} exhausted at {} additional rows (wanted {})",
                entry.source,
                written,
                need
            );
        }

        let q_texts: Vec<String> = pairs.iter().map(|(q, _)| q.clone()).collect();
        let d_texts: Vec<String> = pairs.iter().map(|(_, d)| d.clone()).collect();
        let tok = tokenizer.clone();
        let (q_encs, d_encs) = task::spawn_blocking(move || {
            let q = tok
                .encode_batch_fast(q_texts, ADD_SPECIAL_TOKENS)
                .map_err(|e| anyhow!("encode q: {e}"))?;
            let d = tok
                .encode_batch_fast(d_texts, ADD_SPECIAL_TOKENS)
                .map_err(|e| anyhow!("encode d: {e}"))?;
            Ok::<_, anyhow::Error>((q, d))
        })
        .await??;

        for (qe, de) in q_encs.iter().zip(d_encs.iter()) {
            q_writer.append(qe.get_ids())?;
            d_writer.append(de.get_ids())?;
        }

        let n = pairs.len() as u64;
        written += n;

        if written.is_multiple_of(50_000) || written == need {
            let elapsed = start.elapsed().as_secs_f64();
            info!(
                source = entry.source,
                written,
                need,
                rows_per_sec = written as f64 / elapsed.max(0.001),
                "progress"
            );
        }
    }

    let (q_rows, q_total) = q_writer.finalize()?;
    let (d_rows, d_total) = d_writer.finalize()?;
    if q_rows != d_rows {
        bail!("side row counts diverged: query={q_rows} doc={d_rows}");
    }

    let meta = SourceMeta {
        source: entry.source.clone(),
        n_rows: q_rows,
        total_query_tokens: q_total,
        total_doc_tokens: d_total,
        tokenizer: crate::config::TOKENIZER_MODEL.to_string(),
        tokens_dtype: "u16".to_string(),
        offsets_dtype: "u64".to_string(),
        add_special_tokens: ADD_SPECIAL_TOKENS,
    };
    write_meta(&dir, &meta)?;

    info!(
        source = entry.source,
        n_rows = q_rows,
        total_query_tokens = q_total,
        total_doc_tokens = d_total,
        elapsed_s = start.elapsed().as_secs_f64(),
        "finished"
    );
    Ok(())
}

/// Tokenize every retained source up to its `take_rows` for `tier`.
pub async fn run(tier: Tier) -> Result<()> {
    let plan = build_or_load().await?;
    write_tiers(&plan)?;

    let target = tier.target_rows().unwrap_or(plan.total_rows);
    let takes = plan.take_rows_for_target(target);

    // Do this before loading the tokenizer or streaming any parquet so an
    // incompatible incremental run fails before writing a mixed cache.
    validate_cache_token_policies(&plan.entries, &takes)?;

    let tokenizer = load_tokenizer().await?;
    sanity_check_tokenizer(&tokenizer)?;
    let client = build_http_client()?;

    info!(
        tier = tier.name(),
        target_rows = target,
        actual_rows = takes.iter().sum::<u64>(),
        sources = plan.n_sources(),
        "tokenize start"
    );

    for (entry, target_rows) in plan.entries.iter().zip(takes.iter()) {
        if *target_rows == 0 {
            continue;
        }
        tokenize_source(entry, *target_rows, &tokenizer, &client).await?;
    }

    info!("tokenize complete");
    Ok(())
}

fn sanity_check_tokenizer(tokenizer: &Tokenizer) -> Result<()> {
    // `get_vocab_size(true)` includes added/special tokens.
    let vocab = tokenizer.get_vocab_size(true) as u32;
    if vocab > TOKENIZER_MAX_VOCAB {
        bail!(
            "tokenizer vocab {vocab} exceeds u16 cap {}; binary format assumes u16",
            TOKENIZER_MAX_VOCAB
        );
    }
    info!(vocab, "tokenizer loaded");
    Ok(())
}

/// `cache/subsets/<source>/` — useful for callers outside this module.
pub fn dir_for(source: &str) -> PathBuf {
    source_dir(source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    fn unique_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        temp_dir().join(format!("lattice-tokenize-test-{tag}-{nanos}"))
    }

    fn read_offsets(path: &Path) -> Vec<u64> {
        let bytes = std::fs::read(path).unwrap();
        bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn read_tokens(path: &Path) -> Vec<u16> {
        let bytes = std::fs::read(path).unwrap();
        bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn append_resumes_with_consistent_sentinel() {
        let dir = unique_dir("append");

        // First run: write 3 rows.
        let mut w = SideAppender::open(&dir, Side::Query, 0, 0).unwrap();
        w.append(&[10, 20, 30]).unwrap();
        w.append(&[40]).unwrap();
        w.append(&[50, 60, 70, 80]).unwrap();
        let (rows1, total1) = w.finalize().unwrap();
        assert_eq!(rows1, 3);
        assert_eq!(total1, 8);

        let tokens_path = dir.join("query_tokens.bin");
        let offsets_path = dir.join("query_offsets.bin");
        let offsets = read_offsets(&offsets_path);
        let tokens = read_tokens(&tokens_path);
        assert_eq!(offsets, vec![0, 3, 4, 8]); // 3 starts + sentinel
        assert_eq!(tokens, vec![10, 20, 30, 40, 50, 60, 70, 80]);

        // Second run: open in append mode with existing_rows=3.
        let mut w = SideAppender::open(&dir, Side::Query, rows1, total1).unwrap();
        w.append(&[90, 100]).unwrap();
        let (rows2, total2) = w.finalize().unwrap();
        assert_eq!(rows2, 4);
        assert_eq!(total2, 10);

        let offsets = read_offsets(&offsets_path);
        let tokens = read_tokens(&tokens_path);
        // Row 3 starts at offset 8; sentinel is total=10.
        assert_eq!(offsets, vec![0, 3, 4, 8, 10]);
        assert_eq!(tokens, vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);

        // Slice each row back out and compare.
        for (i, expected) in [
            vec![10u16, 20, 30],
            vec![40],
            vec![50, 60, 70, 80],
            vec![90, 100],
        ]
        .iter()
        .enumerate()
        {
            let start = offsets[i] as usize;
            let end = offsets[i + 1] as usize;
            assert_eq!(&tokens[start..end], expected.as_slice(), "row {i}");
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn append_recovers_from_killed_writer() {
        // Simulate a previous run that was killed mid-append: meta.json
        // says 3 rows + sentinel, but the offsets file has been growing
        // and is much larger; the tokens file also has trailing garbage.
        // Reopening with `existing_rows = 3` should silently truncate
        // both files to the clean state described by meta.json, then
        // resume appending cleanly.
        let dir = unique_dir("crash-recovery");

        // Clean baseline: 3 rows.
        let mut w = SideAppender::open(&dir, Side::Query, 0, 0).unwrap();
        w.append(&[10, 20, 30]).unwrap();
        w.append(&[40]).unwrap();
        w.append(&[50, 60]).unwrap();
        let (rows, total) = w.finalize().unwrap();
        assert_eq!(rows, 3);
        assert_eq!(total, 6);

        // Simulate a partial mid-append crash by appending bogus bytes
        // to BOTH files. The offsets file gets some extra `u64` entries;
        // the tokens file gets extra `u16` token IDs. meta.json on disk
        // still says n_rows=3, total_query_tokens=6.
        let offsets_path = dir.join("query_offsets.bin");
        let tokens_path = dir.join("query_tokens.bin");
        let mut f = OpenOptions::new().append(true).open(&offsets_path).unwrap();
        for v in [99u64, 999, 9999] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        drop(f);
        let mut f = OpenOptions::new().append(true).open(&tokens_path).unwrap();
        for v in [111u16, 222, 333, 444, 555] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        drop(f);

        // Reopen with existing_rows=3, existing_tokens=6 (from finalize above) —
        // should recover, not bail.
        let mut w = SideAppender::open(&dir, Side::Query, 3, 6).unwrap();
        w.append(&[70, 80]).unwrap();
        let (rows, total) = w.finalize().unwrap();
        assert_eq!(rows, 4);
        assert_eq!(total, 8);

        // Files should now contain only the clean 4 rows.
        let offsets = read_offsets(&offsets_path);
        let tokens = read_tokens(&tokens_path);
        assert_eq!(offsets, vec![0, 3, 4, 6, 8]);
        assert_eq!(tokens, vec![10, 20, 30, 40, 50, 60, 70, 80]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn append_detects_corrupt_offsets_file() {
        let dir = unique_dir("corrupt-offsets");
        let mut w = SideAppender::open(&dir, Side::Doc, 0, 0).unwrap();
        w.append(&[1, 2, 3]).unwrap();
        w.finalize().unwrap();

        // Reopen claiming 5 rows exist — file says 1. Should hard-error rather
        // than producing silently-wrong output.
        let res = SideAppender::open(&dir, Side::Doc, 5, 3);
        assert!(res.is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn append_rejects_oversized_token_id() {
        let dir = unique_dir("oversize");
        let mut w = SideAppender::open(&dir, Side::Query, 0, 0).unwrap();
        // 70_000 > 65_536 (u16 cap)
        let err = w.append(&[70_000]).err();
        assert!(err.is_some());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn empty_row_round_trip() {
        let dir = unique_dir("empty-row");
        let mut w = SideAppender::open(&dir, Side::Query, 0, 0).unwrap();
        w.append(&[]).unwrap();
        w.append(&[7, 8]).unwrap();
        w.append(&[]).unwrap();
        let (rows, total) = w.finalize().unwrap();
        assert_eq!(rows, 3);
        assert_eq!(total, 2);

        let offsets = read_offsets(&dir.join("query_offsets.bin"));
        assert_eq!(offsets, vec![0, 0, 2, 2]);
        let tokens = read_tokens(&dir.join("query_tokens.bin"));
        assert_eq!(tokens, vec![7, 8]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn legacy_meta_without_policy_defaults_to_special_tokens() {
        let raw = r#"{
            "source": "legacy",
            "n_rows": 1,
            "total_query_tokens": 3,
            "total_doc_tokens": 3,
            "tokenizer": "bert-base-uncased",
            "tokens_dtype": "u16",
            "offsets_dtype": "u64"
        }"#;
        let meta: SourceMeta = serde_json::from_str(raw).unwrap();
        assert!(meta.add_special_tokens);
    }

    #[test]
    fn fresh_meta_records_canonical_no_special_policy() {
        let meta = SourceMeta::empty("fresh", "bert-base-uncased");
        assert!(!meta.add_special_tokens);
        let serialized = serde_json::to_value(meta).unwrap();
        assert_eq!(serialized["add_special_tokens"], false);
    }
}
