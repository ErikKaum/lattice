//! CLI: read newline-delimited documents, tokenize, embed, write raw f32 LE
//! vectors. Doubles as the bulk-embedding path; stats go to stderr and
//! embeddings go to stdout or a file.
//!
//! Output format is `N * dim * 4` bytes of little-endian f32, row-major. No
//! header — the caller knows `dim` from the model. Trivially loadable as
//! `np.fromfile(path, dtype=np.float32).reshape(-1, dim)` on the Python side.
//!
//! Defaults: tokenizer found next to the model (`<model_dir>/tokenizer.json`);
//! input from stdin if `--input` not given; output to stdout otherwise.
//! L2-normalization on by default (matches training/eval).

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write, stdin, stdout};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};

use lattice::{LatticeTokenizer, Model, kernel};

struct Args {
    model: PathBuf,
    tokenizer: Option<PathBuf>,
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    normalize: bool,
}

fn main() -> Result<()> {
    let args = parse_args()?;

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
        "model: variant={} dim={} vocab={}",
        model.variant(),
        model.dim(),
        model.vocab()
    );
    eprintln!("tokenizer: {}", tk_path.display());
    eprintln!("normalize: {}", args.normalize);

    let input: Box<dyn BufRead> = match &args.input {
        Some(p) => Box::new(BufReader::new(
            File::open(p).with_context(|| format!("open input {}", p.display()))?,
        )),
        None => Box::new(BufReader::new(stdin())),
    };
    let mut output: Box<dyn Write> = match &args.output {
        Some(p) => Box::new(BufWriter::new(
            File::create(p).with_context(|| format!("create output {}", p.display()))?,
        )),
        None => Box::new(BufWriter::new(stdout())),
    };

    let dim = model.dim();
    let mut emb = vec![0.0f32; dim];
    let mut scratch = model.scratch();
    let mut n_docs: u64 = 0;
    let mut n_tokens: u64 = 0;
    let mut n_empty: u64 = 0;

    let start = Instant::now();
    for line in input.lines() {
        let text = line.context("reading input line")?;
        let ids = tokenizer
            .encode(&text)
            .with_context(|| format!("tokenizing doc {}", n_docs))?;
        if ids.is_empty() {
            n_empty += 1;
        }
        n_tokens += ids.len() as u64;
        model.embed(&ids, &mut emb, &mut scratch)?;
        if args.normalize {
            kernel::l2_normalize(&mut emb);
        }
        // write `dim` little-endian f32s. We accumulate per-vector into a
        // small scratch buffer of bytes to amortize the system-call cost of
        // many tiny writes (BufWriter helps too, but a single write_all per
        // vector is still cheaper than `dim` write_all calls).
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(emb.as_ptr() as *const u8, emb.len() * 4) };
        output.write_all(bytes)?;
        n_docs += 1;
    }
    output.flush()?;
    let elapsed = start.elapsed();
    let secs = elapsed.as_secs_f64();
    let docs_per_s = if secs > 0.0 {
        n_docs as f64 / secs
    } else {
        0.0
    };
    let tok_per_s = if secs > 0.0 {
        n_tokens as f64 / secs
    } else {
        0.0
    };

    eprintln!("---");
    eprintln!("docs:        {}", n_docs);
    eprintln!("tokens:      {}", n_tokens);
    if n_empty > 0 {
        eprintln!("empty docs:  {}  (zero-vector output)", n_empty);
    }
    eprintln!("elapsed:     {:.3}s", secs);
    eprintln!("docs/sec:    {:.0}", docs_per_s);
    eprintln!("tokens/sec:  {:.0}", tok_per_s);
    Ok(())
}

fn parse_args() -> Result<Args> {
    let mut model: Option<PathBuf> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut normalize: bool = true;

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
            "--input" => {
                input = Some(PathBuf::from(
                    it.next().ok_or_else(|| anyhow!("--input needs a path"))?,
                ))
            }
            "--output" => {
                output = Some(PathBuf::from(
                    it.next().ok_or_else(|| anyhow!("--output needs a path"))?,
                ))
            }
            "--no-normalize" => normalize = false,
            "--normalize" => normalize = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => bail!("unknown argument: {}", other),
        }
    }

    Ok(Args {
        model: model.ok_or_else(|| anyhow!("--model is required (see --help)"))?,
        tokenizer,
        input,
        output,
        normalize,
    })
}

fn print_help() {
    eprintln!("Usage: embed --model PATH [options]");
    eprintln!();
    eprintln!("Read newline-delimited documents, embed, write raw f32 LE vectors.");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --model PATH       Path to model.safetensors  [required]");
    eprintln!(
        "  --tokenizer PATH   Path to tokenizer.json     [default: <model_dir>/tokenizer.json]"
    );
    eprintln!("  --input PATH       Input file, one doc/line   [default: stdin]");
    eprintln!("  --output PATH      Output file, raw f32 LE    [default: stdout]");
    eprintln!("  --no-normalize     Skip L2 normalization      [default: on]");
    eprintln!("  -h, --help         Print this help");
}
