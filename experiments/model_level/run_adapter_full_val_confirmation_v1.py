"""Full Train-Val confirmation for one Model-Level Adapter.

This runner intentionally performs Adapter-only inference.
It reuses the frozen validation, prediction, resume, and metric
implementation from run_adapter_dev_evaluation_v1.py.

No Generic inference is performed here.
No Test rows are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from experiments.model_level import run_adapter_dev_evaluation_v1 as base


EXPERIMENT = "model_level_adapter_full_val_confirmation_v1"


def row_id_manifest_sha256(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["row_id"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-val", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)

    parser.add_argument("--author", default="Agent Phage")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260822)

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--beam-size", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=250)

    parser.add_argument("--label", required=True)
    parser.add_argument("--adapter", type=Path, required=True)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    if not args.adapter.is_file():
        raise FileNotFoundError(args.adapter)

    author_rows, population_audit = base.load_val_rows(
        args.train_val,
        author=args.author,
    )

    # Keep the same deterministic ordering rule as the calibration runner,
    # but retain the complete author population.
    rows = base.deterministic_subset(
        author_rows,
        seed=args.seed,
        max_rows=None,
    )

    if args.author == "Agent Phage" and len(rows) != 13741:
        raise RuntimeError(
            f"Expected 13741 Agent Phage Train-Val rows, got {len(rows)}"
        )

    manifest_sha = row_id_manifest_sha256(rows)

    config = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "analysis_scope": "FULL TRAIN-VAL CONFIRMATION; no Test inference",
        "author": args.author,
        "selected_rows": len(rows),
        "selection_seed": args.seed,
        "selected_row_id_manifest_sha256": manifest_sha,
        "train_val": str(args.train_val),
        "checkpoint": str(args.checkpoint),
        "adapter": str(args.adapter),
        "adapter_sha256": base.sha256_file(args.adapter),
        "label": args.label,
        "pinyin_condition": "full",
        "target_type": "short",
        "beam_size": args.beam_size,
        "top_k": args.top_k,
        "generic_inference_performed": False,
        "used_test": False,
    }

    base.write_json(
        args.output_root / "evaluation_config.json",
        config,
    )

    print("===== FULL TRAIN-VAL ADAPTER CONFIRMATION =====", flush=True)
    print(f"author={args.author}", flush=True)
    print(f"rows={len(rows)}", flush=True)
    print(f"label={args.label}", flush=True)
    print(f"beam_size={args.beam_size}", flush=True)
    print(f"top_k={args.top_k}", flush=True)
    print(f"row_manifest_sha256={manifest_sha}", flush=True)
    print(flush=True)

    backend = base.PinyinGPTConcatBackend(
        args.checkpoint,
        device=args.device,
    )

    parameter_audit = base.install_adapters(
        backend.model,
        base.SerialAdapterConfig(),
    )

    metadata = base.load_adapter_bundle(
        backend.model,
        args.adapter,
    )

    base.set_adapter_training_mode(
        backend.model,
        enabled=False,
    )

    if metadata.get("author") != args.author:
        raise RuntimeError(
            f"Adapter author mismatch: "
            f"{metadata.get('author')!r} != {args.author!r}"
        )

    if metadata.get("checkpoint_revision") != base.CHECKPOINT_REVISION:
        raise RuntimeError("Adapter checkpoint revision mismatch")

    predictions = base.evaluate_backend(
        backend,
        rows,
        condition=args.label,
        output_path=args.output_root / f"{args.label}_predictions.jsonl",
        top_k=args.top_k,
        beam_size=args.beam_size,
        log_every=args.log_every,
    )

    metrics = base.compute_metrics(predictions)

    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "complete",
        "analysis_scope": "FULL TRAIN-VAL CONFIRMATION; no Test inference",
        "author": args.author,
        "selected_rows": len(rows),
        "population": population_audit,
        "selection_seed": args.seed,
        "selected_row_id_manifest_sha256": manifest_sha,
        "pinyin_condition": "full",
        "target_type": "short",
        "beam_size": args.beam_size,
        "top_k": args.top_k,
        "condition": {
            "label": args.label,
            "adapter_path": str(args.adapter),
            "adapter_sha256": base.sha256_file(args.adapter),
            "adapter_metadata": metadata,
            "parameter_audit": parameter_audit,
            "metrics": metrics,
        },
        "generic_inference_performed": False,
        "used_test": False,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": args.device,
            "visible_cuda_devices": torch.cuda.device_count(),
            "device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "CPU"
            ),
        },
    }

    base.write_json(
        args.output_root / "evaluation_result.json",
        result,
    )

    base.write_json(
        args.output_root / f"{args.label}_metrics.json",
        {
            "label": args.label,
            "metrics": metrics,
        },
    )

    micro = metrics["micro"]

    print()
    print("===== FULL TRAIN-VAL RESULT =====")
    print(f"N={len(rows)}")
    print(
        f"{args.label} "
        f"Top1={micro['top1']:.9f} "
        f"Top3={micro['top3']:.9f} "
        f"Top5={micro['top5']:.9f} "
        f"Top10={micro['top10']:.9f} "
        f"MRR={micro['mrr_at_10']:.9f} "
        f"Missing={micro['missing_at_10_rate']:.9f}"
    )

    base.release_backend(backend)


if __name__ == "__main__":
    main()
