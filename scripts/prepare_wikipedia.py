"""Download a slice of English Wikipedia and dump it as newline-delimited text.

One article per line; line breaks/tabs inside the article body are flattened
to single spaces so each doc is a clean stream-of-text on one line. This is
the input format the Rust `bench` binary expects.

Usage:
    uv run scripts/prepare_wikipedia.py --limit 10000 --out data/wiki/wiki-10k.txt
    uv run scripts/prepare_wikipedia.py --out data/wiki/wiki-full.txt   # full ~6.7M

We use `datasets` in streaming mode so a small limit doesn't trigger a
full ~20GB download. The full run pulls every parquet shard; budget accordingly.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="max articles to emit (default: full dataset)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output .txt path (one article per line)",
    )
    ap.add_argument(
        "--config",
        default="20231101.en",
        help="HF wikimedia/wikipedia config (default: 20231101.en)",
    )
    ap.add_argument(
        "--min-chars",
        type=int,
        default=0,
        help="skip articles shorter than this (default: 0 = keep all)",
    )
    args = ap.parse_args()

    from datasets import load_dataset

    args.out.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "wikimedia/wikipedia",
        args.config,
        split="train",
        streaming=True,
    )

    start = time.time()
    n_written = 0
    n_seen = 0
    n_bytes = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for row in ds:
            n_seen += 1
            text: str = row["text"]
            if len(text) < args.min_chars:
                continue
            # one logical doc per line — flatten any internal newlines/tabs
            flat = text.replace("\n", " ").replace("\t", " ")
            fout.write(flat + "\n")
            n_written += 1
            n_bytes += len(flat) + 1
            if args.limit is not None and n_written >= args.limit:
                break
            if n_written % 50_000 == 0:
                elapsed = time.time() - start
                print(
                    f"  {n_written:>9,} articles  ({n_bytes / 1e6:.1f} MB)  {elapsed:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )

    elapsed = time.time() - start
    print(
        f"wrote {args.out}  ({n_written:,} articles, {n_bytes / 1e6:.1f} MB, "
        f"skipped {n_seen - n_written}, {elapsed:.1f}s)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
