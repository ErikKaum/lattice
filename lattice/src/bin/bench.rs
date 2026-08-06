//! Throughput benchmark for the embedding pipeline. Multi-threaded with
//! rayon, chunked streaming over the corpus so we never hold the whole
//! input or output in memory.
//!
//! Shaped around answering "how fast can we embed Wikipedia?" — feed it
//! the corpus produced by `scripts/prepare_wikipedia.py` (one article per
//! line). For development iteration, pass `--limit N`; for headline
//! numbers omit the limit so it runs over the whole file.
//!
//! Each chunk runs the same two-phase tokenize-then-embed dance:
//! - `tokenize`: HF `Tokenizer::encode_batch` over the chunk's docs
//!   (HF parallelizes internally via rayon)
//! - `embed`: rayon `par_chunks_mut` over the chunk's output buffer, then
//!   L2-normalize per row
//! - `write`: streaming append to the output file as each chunk completes;
//!   the per-chunk write cost is amortized across the run, and we never need
//!   an N·dim·4-byte output buffer sitting in RAM.
//!
//! Output (when written) matches the `embed` bin: flat little-endian f32,
//! `N · dim · 4` bytes, row-major.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use rayon::prelude::*;

use lattice::{LatticeTokenizer, Model, kernel};

// Replace macOS's system malloc with mimalloc. The bench creates many
// short-lived `Vec<u32>` token allocations across N rayon workers; the
// default allocator's per-CPU caching does not handle that contention
// well. mimalloc's per-thread heap drops it to ~zero.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

struct Args {
    model: PathBuf,
    tokenizer: Option<PathBuf>,
    corpus: PathBuf,
    output: Option<PathBuf>,
    limit: Option<usize>,
    threads: Option<usize>,
    chunk_size: usize,
    normalize: bool,
    no_write: bool,
}

fn main() -> Result<()> {
    let args = parse_args()?;

    let threads = args.threads.unwrap_or_else(num_cpus::get);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .context("building rayon thread pool")?;

    let model = Model::load(&args.model)
        .with_context(|| format!("loading model from {}", args.model.display()))?;
    let tk_path = args.tokenizer.clone().unwrap_or_else(|| {
        args.model
            .parent()
            .map(|d| d.join("tokenizer.json"))
            .unwrap_or_else(|| PathBuf::from("tokenizer.json"))
    });
    let tokenizer = LatticeTokenizer::load(&tk_path)
        .with_context(|| format!("loading tokenizer from {}", tk_path.display()))?;

    eprintln!(
        "model:       {} dim={} vocab={}",
        model.variant(),
        model.dim(),
        model.vocab()
    );
    eprintln!("tokenizer:   {}", tk_path.display());
    eprintln!("corpus:      {}", args.corpus.display());
    eprintln!(
        "limit:       {}",
        args.limit
            .map(|n| n.to_string())
            .unwrap_or_else(|| "(none — full file)".into())
    );
    eprintln!("threads:     {}", threads);
    eprintln!("chunk_size:  {}", args.chunk_size);
    eprintln!("normalize:   {}", args.normalize);
    eprintln!("write:       {}", if args.no_write { "no" } else { "yes" });
    eprintln!();

    let dim = model.dim();

    // Open the corpus as a streaming reader. We never read the whole file
    // into memory — only a chunk_size-sized batch at a time.
    let corpus_file = File::open(&args.corpus)
        .with_context(|| format!("open corpus {}", args.corpus.display()))?;
    let mut lines = BufReader::with_capacity(1 << 20, corpus_file).lines();

    // Open the output sink (if any). Writes are streamed per chunk.
    let mut output: Option<BufWriter<File>> = match (&args.output, args.no_write) {
        (_, true) => None,
        (Some(p), false) => Some(BufWriter::with_capacity(
            1 << 22,
            File::create(p).with_context(|| format!("create output {}", p.display()))?,
        )),
        (None, false) => None,
    };

    // Pre-allocate the per-chunk embedding buffer once; reuse across chunks.
    let mut emb_buf: Vec<f32> = vec![0.0; args.chunk_size * dim];

    let mut n_docs_total: u64 = 0;
    let mut n_tokens_total: u64 = 0;
    let mut n_empty_total: u64 = 0;
    let mut t_tokenize_total = Duration::ZERO;
    let mut t_embed_total = Duration::ZERO;
    let mut t_write_total = Duration::ZERO;
    let mut n_chunks: usize = 0;

    let wall_start = Instant::now();
    let mut last_progress = Instant::now();

    let mut eof = false;
    while !eof {
        // ---- Read a chunk of docs ----------------------------------------
        // Fresh allocation each iteration so `encode_batch` can consume the
        // Vec. With mimalloc this is essentially free at chunk_size=10k.
        let mut chunk_docs: Vec<String> = Vec::with_capacity(args.chunk_size);
        for _ in 0..args.chunk_size {
            if let Some(cap) = args.limit
                && (n_docs_total as usize + chunk_docs.len()) >= cap
            {
                eof = true;
                break;
            }
            match lines.next() {
                Some(line_result) => {
                    chunk_docs.push(line_result.context("reading corpus line")?);
                }
                None => {
                    eof = true;
                    break;
                }
            }
        }
        let chunk_n = chunk_docs.len();
        if chunk_n == 0 {
            break;
        }

        // ---- Phase 1: tokenize (HF parallelizes internally) --------------
        let t0 = Instant::now();
        let all_tokens: Vec<Vec<u32>> = tokenizer.encode_batch(chunk_docs)?;
        t_tokenize_total += t0.elapsed();
        n_tokens_total += all_tokens.iter().map(|t| t.len() as u64).sum::<u64>();
        n_empty_total += all_tokens.iter().filter(|t| t.is_empty()).count() as u64;

        // ---- Phase 2: embed (rayon over the chunk's dim-sized rows) -----
        let t1 = Instant::now();
        let emb_slice = &mut emb_buf[..chunk_n * dim];
        pool.install(|| {
            emb_slice
                .par_chunks_mut(dim)
                .zip(all_tokens.par_iter())
                .for_each_init(
                    || model.scratch(),
                    |scratch, (out_row, tokens)| {
                        model
                            .embed(tokens, out_row, scratch)
                            .expect("kernel dispatch should not fail for loaded model");
                        if args.normalize {
                            kernel::l2_normalize(out_row);
                        }
                    },
                )
        });
        t_embed_total += t1.elapsed();

        // ---- Phase 3: write (streaming append) --------------------------
        if let Some(ref mut w) = output {
            let t2 = Instant::now();
            // Safety: f32 → little-endian bytes. On all targets we care
            // about, f32 is already little-endian in memory.
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(emb_slice.as_ptr() as *const u8, emb_slice.len() * 4)
            };
            w.write_all(bytes)?;
            t_write_total += t2.elapsed();
        }

        n_docs_total += chunk_n as u64;
        n_chunks += 1;

        // ---- Progress log (every ~5s of wall, or every chunk if slow) ---
        if last_progress.elapsed() >= Duration::from_secs(5) {
            let elapsed = wall_start.elapsed().as_secs_f64();
            let docs_s = n_docs_total as f64 / elapsed.max(1e-6);
            let tok_s = n_tokens_total as f64 / elapsed.max(1e-6);
            eprintln!(
                "  [{:>10} docs / {:>3} chunks / {:>5.0}s wall / {:>7.0} docs/s / {:>7.2}M tok/s]",
                n_docs_total,
                n_chunks,
                elapsed,
                docs_s,
                tok_s / 1e6,
            );
            last_progress = Instant::now();
        }
    }

    if let Some(ref mut w) = output {
        let t = Instant::now();
        w.flush()?;
        t_write_total += t.elapsed();
    }

    let wall = wall_start.elapsed();
    let wall_s = wall.as_secs_f64();
    let docs_per_s = if wall_s > 0.0 {
        n_docs_total as f64 / wall_s
    } else {
        0.0
    };
    let tok_per_s = if wall_s > 0.0 {
        n_tokens_total as f64 / wall_s
    } else {
        0.0
    };
    let mean_tok_per_doc = if n_docs_total > 0 {
        n_tokens_total as f64 / n_docs_total as f64
    } else {
        0.0
    };

    eprintln!();
    eprintln!("---");
    eprintln!("docs:           {}", n_docs_total);
    eprintln!("tokens:         {}", n_tokens_total);
    eprintln!("mean toks/doc:  {:.1}", mean_tok_per_doc);
    if n_empty_total > 0 {
        eprintln!("empty docs:     {}", n_empty_total);
    }
    eprintln!("chunks:         {}", n_chunks);
    eprintln!("wall:           {:.3}s", wall_s);
    eprintln!("docs/sec:       {:.0}", docs_per_s);
    eprintln!("tokens/sec:     {:.0}", tok_per_s);
    eprintln!();
    eprintln!("phase           sec       %wall");
    let pct = |d: Duration| 100.0 * d.as_secs_f64() / wall_s.max(1e-6);
    eprintln!(
        "  tokenize     {:>7.3}    {:>5.1}%",
        t_tokenize_total.as_secs_f64(),
        pct(t_tokenize_total)
    );
    eprintln!(
        "  embed        {:>7.3}    {:>5.1}%",
        t_embed_total.as_secs_f64(),
        pct(t_embed_total)
    );
    if !args.no_write {
        eprintln!(
            "  write        {:>7.3}    {:>5.1}%",
            t_write_total.as_secs_f64(),
            pct(t_write_total)
        );
    }
    let accounted = t_tokenize_total + t_embed_total + t_write_total;
    let other = wall.saturating_sub(accounted);
    eprintln!(
        "  other        {:>7.3}    {:>5.1}%    (corpus read, harness overhead)",
        other.as_secs_f64(),
        pct(other),
    );
    Ok(())
}

fn parse_args() -> Result<Args> {
    let mut model: Option<PathBuf> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut corpus: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut limit: Option<usize> = None;
    let mut threads: Option<usize> = None;
    let mut chunk_size: usize = 10_000;
    let mut normalize: bool = true;
    let mut no_write: bool = false;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => {
                model = Some(PathBuf::from(
                    it.next().ok_or_else(|| anyhow!("--model needs a path"))?,
                ))
            }
            "--tokenizer" => {
                tokenizer = Some(PathBuf::from(
                    it.next()
                        .ok_or_else(|| anyhow!("--tokenizer needs a path"))?,
                ))
            }
            "--corpus" => {
                corpus = Some(PathBuf::from(
                    it.next().ok_or_else(|| anyhow!("--corpus needs a path"))?,
                ))
            }
            "--output" => {
                output = Some(PathBuf::from(
                    it.next().ok_or_else(|| anyhow!("--output needs a path"))?,
                ))
            }
            "--limit" => {
                limit = Some(
                    it.next()
                        .ok_or_else(|| anyhow!("--limit needs an integer"))?
                        .parse()?,
                );
            }
            "--threads" => {
                threads = Some(
                    it.next()
                        .ok_or_else(|| anyhow!("--threads needs an integer"))?
                        .parse()?,
                );
            }
            "--chunk-size" => {
                chunk_size = it
                    .next()
                    .ok_or_else(|| anyhow!("--chunk-size needs an integer"))?
                    .parse()?;
                if chunk_size == 0 {
                    bail!("--chunk-size must be > 0");
                }
            }
            "--no-normalize" => normalize = false,
            "--normalize" => normalize = true,
            "--no-write" => no_write = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => bail!("unknown argument: {}", other),
        }
    }

    Ok(Args {
        model: model.ok_or_else(|| anyhow!("--model is required"))?,
        tokenizer,
        corpus: corpus.ok_or_else(|| anyhow!("--corpus is required"))?,
        output,
        limit,
        threads,
        chunk_size,
        normalize,
        no_write,
    })
}

fn print_help() {
    eprintln!("Usage: bench --model PATH --corpus PATH [options]");
    eprintln!();
    eprintln!("Multi-threaded throughput benchmark with chunked streaming I/O.");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --model PATH       Path to model.safetensors   [required]");
    eprintln!(
        "  --tokenizer PATH   tokenizer.json              [default: <model_dir>/tokenizer.json]"
    );
    eprintln!("  --corpus PATH      one doc/line text file      [required]");
    eprintln!("  --output PATH      f32 LE output, optional");
    eprintln!("  --limit N          cap docs processed");
    eprintln!("  --threads N        worker threads              [default: num_cpus]");
    eprintln!("  --chunk-size N     docs per streaming chunk    [default: 10000]");
    eprintln!("  --no-write         skip output (pure compute mode)");
    eprintln!("  --no-normalize     skip L2 normalization       [default: on]");
    eprintln!("  -h, --help         print this help");
}
