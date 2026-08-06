# Pipeline image for HF Jobs. CPU-only Rust binary that streams parquet
# from HF, tokenizes with bert-base-uncased, and writes per-source binary
# files. Runs as a regular HF Job with the data bucket mounted at
# `/app/cache` — all writes go straight to the bucket via FUSE.
#
# Bucket-safety contract (matters because a job can be killed mid-run):
#
# - Binary files (`<src>/{query,doc}_{tokens,offsets}.bin`) grow append-only.
#   A partial append leaves bytes past what `meta.json` claims, which the
#   local `SideAppender::open` crash-recovery silently truncates on next
#   open. Net effect of a kill: the source rewinds to its last finalized
#   row count, never to a corrupt or undefined state.
#
# - All JSON metadata (`meta.json`, `partition.json`, `tiers.json`,
#   `sources.json`) is written via `atomic_write` — tmp + fsync + rename.
#   Readers always see either the pre-call content or the new content,
#   never a torn write. So `meta.json` (the source of truth) cannot end
#   up as invalid JSON even if the process dies between bytes.
#
# - Therefore the job is safe to launch, kill, and re-launch. Each run
#   skips finished sources, recovers any half-appended source, and
#   resumes from where the last successful `finalize` left things.
#
# Build:  docker build --platform linux/amd64 -t <user>/lattice-pipeline:<tag> .
# Push:   docker push <user>/lattice-pipeline:<tag>
#
# Run:
#   uvx hf jobs run --flavor cpu-upgrade --timeout 12h -s HF_TOKEN \
#     -v hf://buckets/<user>/<bucket>:/app/cache \
#     <user>/lattice-pipeline:<tag> \
#     pipeline tokenize full
#
# Notes:
# - CPU-only flavor is correct: tokenize never touches a GPU.
# - No `--scratch` needed: tokenize's writes are append-sequential, which
#   bucket FUSE handles at ~300 MB/s sustained (per audacity's
#   measurements). Sparse random reads — the audacity gotcha — don't
#   happen during tokenize.

# ----- Stage 1: build the Rust binary for Linux x86_64 -----------------
FROM --platform=linux/amd64 rust:1.88-slim AS builder

WORKDIR /build

RUN apt-get update \
 && apt-get install -y --no-install-recommends pkg-config build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --bin pipeline

# ----- Stage 2: minimal runtime ---------------------------------------
FROM --platform=linux/amd64 debian:bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/pipeline /usr/local/bin/pipeline

WORKDIR /app
# Pre-create the bucket mount target so `hf jobs run -v ...:/app/cache`
# never has to materialize a new directory in the image at mount time.
RUN mkdir -p /app/cache

ENV RUST_LOG=info

CMD ["pipeline", "--help"]
