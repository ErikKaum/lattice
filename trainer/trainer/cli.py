"""Thin click-free CLI for training and evaluating the static embedding model."""

from __future__ import annotations

import argparse
from pathlib import Path

from .train import TrainConfig, train


def _default_cache_root() -> Path:
    # Mirror the Rust pipeline default: `./cache` relative to cwd.
    return Path("./cache")


def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tier", default="xs", choices=["xs", "small", "medium", "full"])
    p.add_argument("--cache-root", type=Path, default=_default_cache_root())
    p.add_argument("--out-dir", type=Path, default=Path("./runs/latest"))
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after this many steps (across all epochs). Useful for "
        "smoke tests; otherwise leave unset.",
    )
    p.add_argument("--lr", type=float, default=2e-1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=_default_device())
    p.add_argument("--dtype", default="float32")
    p.add_argument(
        "--scratch",
        type=Path,
        default=None,
        help=(
            "Stage tier binaries to this local dir before mmap. Essential on "
            "HF Jobs with a FUSE-bucket `--cache-root` — sparse mmap reads "
            "of multi-GB token files over FUSE serialize catastrophically "
            "under DDP. Safe to share across ranks (flock + sentinel)."
        ),
    )


def _add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--cache-root", type=Path, default=_default_cache_root())
    p.add_argument("--out-dir", type=Path, default=Path("./runs/latest/eval"))


def main() -> None:
    ap = argparse.ArgumentParser(prog="trainer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="train on a tier's pre-tokenized cache")
    _add_train_args(p_train)

    p_eval = sub.add_parser("eval", help="run NanoBEIR on a checkpoint")
    _add_eval_args(p_eval)

    p_q = sub.add_parser(
        "eval-quant",
        help="sweep (dim × bits × axis) PTQ on a checkpoint, NanoBEIR per cell",
    )
    _add_eval_args(p_q)

    p_beir = sub.add_parser(
        "eval-beir",
        help="run decontaminated BEIR (LightOn 14-task suite) on a checkpoint",
    )
    _add_eval_args(p_beir)
    p_beir.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional subset of tasks (default: all 14). Names: arguana, "
        "climate-fever, dbpedia, fever, fiqa, hotpotqa, msmarco, "
        "nfcorpus, nq, quora, scidocs, scifact, trec-covid, "
        "webis-touche2020.",
    )

    p_beir_hub = sub.add_parser(
        "eval-beir-hub",
        help="run decontaminated BEIR against a HF Hub model id (for reference)",
    )
    p_beir_hub.add_argument("model_id")
    p_beir_hub.add_argument("--out-dir", type=Path, default=Path("./runs/reference/decontam"))
    p_beir_hub.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional subset of tasks (default: all 14).",
    )

    p_deploy_quant = sub.add_parser(
        "eval-deployment-quant",
        help="run decontaminated BEIR for a generated quantized artifact group",
    )
    _add_eval_args(p_deploy_quant)
    p_deploy_quant.add_argument(
        "--group",
        required=True,
        choices=["int8_dim", "int4_dim", "row"],
    )

    p_split = sub.add_parser(
        "stage2-split",
        help="build the stage-2 train/eval query-id split (per-source, seed-fixed)",
    )
    p_split.add_argument(
        "--out-dir",
        type=Path,
        default=Path("../pipeline/cache/stage2/splits"),
        help="Where to write per-source split JSONs + manifest.json.",
    )
    p_split.add_argument("--seed", type=int, default=42)
    p_split.add_argument("--eval-fraction", type=float, default=0.20)
    p_split.add_argument("--eval-cap", type=int, default=100)
    p_split.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing split files even if they were built with a different seed. "
        "Don't do this casually — every existing stage-2 comparison was made against "
        "the existing split.",
    )

    p_build_surface = sub.add_parser(
        "stage2-build-eval",
        help="materialize per-source held-out eval surfaces (BEIR triple)",
    )
    p_build_surface.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("../pipeline/cache/stage2/splits"),
    )
    p_build_surface.add_argument(
        "--out-dir",
        type=Path,
        default=Path("../pipeline/cache/stage2/eval_surface"),
    )
    p_build_surface.add_argument("--n-distractors", type=int, default=3000)
    p_build_surface.add_argument("--seed", type=int, default=42)

    p_eval_s2 = sub.add_parser(
        "eval-stage2",
        help="eval a checkpoint on the held-out in-domain surface",
    )
    p_eval_s2.add_argument("checkpoint", type=Path)
    p_eval_s2.add_argument(
        "--cache-root",
        type=Path,
        default=Path("../pipeline/cache"),
    )
    p_eval_s2.add_argument(
        "--surface-dir",
        type=Path,
        default=Path("../pipeline/cache/stage2/eval_surface"),
    )
    p_eval_s2.add_argument("--out-dir", type=Path, default=Path("./runs/latest/stage2"))

    p_tok2 = sub.add_parser(
        "stage2-tokenize",
        help="tokenize the stage-2 training split into (query, positive, 50×negative) binaries",
    )
    p_tok2.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("../pipeline/cache/stage2/splits"),
    )
    p_tok2.add_argument(
        "--out-root",
        type=Path,
        default=Path("../pipeline/cache"),
        help="Root above the `stage2/training/<source>/` per-source dirs.",
    )
    p_tok2.add_argument(
        "--cache-root",
        type=Path,
        default=Path("../pipeline/cache"),
        help="Where tokenizer.json lives.",
    )
    p_tok2.add_argument("--nv-threshold", type=float, default=0.95)
    p_tok2.add_argument("--n-negatives", type=int, default=50)
    p_tok2.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Optional subset (default: all 7).",
    )

    p_s2_train = sub.add_parser(
        "stage2-train",
        help="fine-tune a stage-1 checkpoint with hard negatives",
    )
    p_s2_train.add_argument(
        "--init-from",
        type=Path,
        required=True,
        help="Stage-1 .safetensors checkpoint to fine-tune from.",
    )
    p_s2_train.add_argument(
        "--training-root",
        type=Path,
        default=Path("../pipeline/cache/stage2/training"),
    )
    p_s2_train.add_argument("--out-dir", type=Path, default=Path("./runs/stage2/latest"))
    p_s2_train.add_argument("--batch-size", type=int, default=256)
    p_s2_train.add_argument("--epochs", type=int, default=1)
    p_s2_train.add_argument("--max-steps", type=int, default=None)
    p_s2_train.add_argument("--lr", type=float, default=2e-2)
    p_s2_train.add_argument("--weight-decay", type=float, default=0.0)
    p_s2_train.add_argument("--warmup-ratio", type=float, default=0.05)
    p_s2_train.add_argument("--log-every", type=int, default=20)
    p_s2_train.add_argument("--save-every", type=int, default=1000)
    p_s2_train.add_argument("--seed", type=int, default=42)
    p_s2_train.add_argument("--n-neg-sample", type=int, default=7)
    p_s2_train.add_argument("--device", default=_default_device())
    p_s2_train.add_argument("--dtype", default="float32")
    p_s2_train.add_argument("--scratch", type=Path, default=None)

    p_export = sub.add_parser(
        "export-hf",
        help="export a checkpoint as a Hub-ready Sentence Transformers model",
    )
    p_export.add_argument("checkpoint", type=Path)
    p_export.add_argument("--tokenizer", type=Path, required=True)
    p_export.add_argument("--out-dir", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "train":
        cfg = TrainConfig(
            cache_root=args.cache_root,
            out_dir=args.out_dir,
            tier=args.tier,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            log_every=args.log_every,
            save_every=args.save_every,
            seed=args.seed,
            device=args.device,
            dtype=args.dtype,
            scratch_root=args.scratch,
        )
        train(cfg)
    elif args.cmd == "eval":
        from .eval import evaluate

        tokenizer_path = args.cache_root / "tokenizer.json"
        evaluate(args.checkpoint, tokenizer_path, args.out_dir)
    elif args.cmd == "eval-quant":
        from .eval import evaluate_quantization_sweep

        tokenizer_path = args.cache_root / "tokenizer.json"
        evaluate_quantization_sweep(args.checkpoint, tokenizer_path, args.out_dir)
    elif args.cmd == "eval-beir":
        from .decontam_eval import (
            ALL_TASKS,
            evaluate_decontam_from_checkpoint,
        )

        tokenizer_path = args.cache_root / "tokenizer.json"
        tasks = tuple(args.tasks) if args.tasks else ALL_TASKS
        evaluate_decontam_from_checkpoint(
            args.checkpoint,
            tokenizer_path,
            args.out_dir,
            tasks=tasks,
        )
    elif args.cmd == "eval-beir-hub":
        from .decontam_eval import ALL_TASKS, evaluate_decontam_from_hub

        tasks = tuple(args.tasks) if args.tasks else ALL_TASKS
        evaluate_decontam_from_hub(args.model_id, args.out_dir, tasks=tasks)
    elif args.cmd == "eval-deployment-quant":
        from .decontam_quant_eval import evaluate_deployment_quant_group

        tokenizer_path = args.cache_root / "tokenizer.json"
        evaluate_deployment_quant_group(
            checkpoint_path=args.checkpoint,
            tokenizer_path=tokenizer_path,
            out_root=args.out_dir,
            group=args.group,
        )
    elif args.cmd == "stage2-split":
        from .stage2_split import make_all_splits

        make_all_splits(
            out_dir=args.out_dir,
            seed=args.seed,
            eval_fraction=args.eval_fraction,
            eval_cap=args.eval_cap,
            force=args.force,
        )
    elif args.cmd == "stage2-build-eval":
        from .stage2_eval import build_all_eval_surfaces

        build_all_eval_surfaces(
            splits_dir=args.splits_dir,
            out_dir=args.out_dir,
            n_distractors=args.n_distractors,
            seed=args.seed,
        )
    elif args.cmd == "eval-stage2":
        from .stage2_eval import evaluate_stage2_from_checkpoint

        tokenizer_path = args.cache_root / "tokenizer.json"
        evaluate_stage2_from_checkpoint(
            checkpoint_path=args.checkpoint,
            tokenizer_path=tokenizer_path,
            surface_dir=args.surface_dir,
            out_dir=args.out_dir,
        )
    elif args.cmd == "stage2-tokenize":
        from .stage2_split import SOURCES
        from .stage2_tokenize import tokenize_all

        tokenizer_path = args.cache_root / "tokenizer.json"
        sources = tuple(args.sources) if args.sources else SOURCES
        tokenize_all(
            splits_dir=args.splits_dir,
            tokenizer_path=tokenizer_path,
            out_root=args.out_root,
            sources=sources,
            nv_threshold=args.nv_threshold,
            n_neg=args.n_negatives,
        )
    elif args.cmd == "stage2-train":
        from .stage2_train import Stage2TrainConfig, stage2_train

        cfg = Stage2TrainConfig(
            training_root=args.training_root,
            out_dir=args.out_dir,
            init_from=args.init_from,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            log_every=args.log_every,
            save_every=args.save_every,
            seed=args.seed,
            n_neg_sample=args.n_neg_sample,
            device=args.device,
            dtype=args.dtype,
            scratch_root=args.scratch,
        )
        stage2_train(cfg)
    elif args.cmd == "export-hf":
        from .export_hf import export_hf_model

        export_hf_model(
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer,
            out_dir=args.out_dir,
        )
    else:
        ap.error(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()
