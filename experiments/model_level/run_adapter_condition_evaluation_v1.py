"""Evaluate one trained Adapter under Full or Initial Pinyin.

The underlying Train-Val validation, Adapter loading, Beam16 generation,
resume validation, trailing-record recovery, and ranking metrics are reused
from run_adapter_dev_evaluation_v1.py.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

from experiments.model_level import run_adapter_dev_evaluation_v1 as base
from src.model_level.pinyin_modes import full_to_initial


EXPERIMENT = "model_level_adapter_condition_evaluation_v1"


def row_id_manifest_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["row_id"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_selected_rows(
    path: Path,
    author_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    canonical = {
        str(row["row_id"]): row
        for row in author_rows
    }

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            supplied = json.loads(line)
            row_id = str(supplied["row_id"])

            if row_id in seen:
                raise RuntimeError(
                    f"Duplicate row_id {row_id!r} in {path}"
                )
            seen.add(row_id)

            expected = canonical.get(row_id)
            if expected is None:
                raise RuntimeError(
                    f"Unknown row_id {row_id!r} in {path}"
                )

            if str(supplied["gold"]) != str(expected["gold"]):
                raise RuntimeError(
                    f"Gold mismatch for {row_id}"
                )

            if tuple(map(str, supplied["pinyin_segments"])) != tuple(
                map(str, expected["pinyin_segments"])
            ):
                raise RuntimeError(
                    f"Pinyin mismatch for {row_id}"
                )

            selected.append(expected)

    if not selected:
        raise RuntimeError(f"No rows loaded from {path}")

    return selected


def transform_mode(
    rows: list[dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    if mode == "full":
        return [copy.deepcopy(row) for row in rows]

    if mode != "initial":
        raise ValueError(mode)

    transformed: list[dict[str, Any]] = []

    for source_row in rows:
        row = copy.deepcopy(source_row)
        initials = full_to_initial(row["pinyin_segments"])

        row["pinyin_segments"] = list(initials)
        row["pinyin_input"] = " ".join(initials)
        row["pinyin_type"] = "initial"

        transformed.append(row)

    return transformed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-val", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)

    parser.add_argument("--author", required=True)
    parser.add_argument("--label", required=True)

    parser.add_argument(
        "--pinyin-mode",
        choices=("full", "initial"),
        required=True,
    )

    parser.add_argument(
        "--selected-rows",
        type=Path,
        default=None,
        help="Frozen Dev manifest. If omitted, use full author Train-Val.",
    )

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--beam-size", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=250)

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

    if args.selected_rows is not None:
        if not args.selected_rows.is_file():
            raise FileNotFoundError(args.selected_rows)

        base_rows = load_selected_rows(
            args.selected_rows,
            author_rows,
        )

        evaluation_scope = "fixed_dev"
        selected_rows_sha256 = base.sha256_file(
            args.selected_rows
        )
    else:
        base_rows = base.deterministic_subset(
            author_rows,
            seed=args.seed,
            max_rows=None,
        )

        evaluation_scope = "full_train_val"
        selected_rows_sha256 = None

    rows = transform_mode(
        base_rows,
        args.pinyin_mode,
    )

    manifest_sha = row_id_manifest_sha256(base_rows)

    config = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "analysis_scope": evaluation_scope,
        "author": args.author,
        "selected_rows": len(rows),
        "selected_rows_file": (
            str(args.selected_rows)
            if args.selected_rows is not None
            else None
        ),
        "selected_rows_file_sha256": selected_rows_sha256,
        "selected_row_id_manifest_sha256": manifest_sha,
        "pinyin_mode": args.pinyin_mode,
        "target_type": "short",
        "beam_size": args.beam_size,
        "top_k": args.top_k,
        "adapter": str(args.adapter),
        "adapter_sha256": base.sha256_file(args.adapter),
        "label": args.label,
        "used_test": False,
    }

    base.write_json(
        args.output_root / "evaluation_config.json",
        config,
    )

    print("===== CONDITION EVALUATION =====", flush=True)
    print(f"author={args.author}", flush=True)
    print(f"scope={evaluation_scope}", flush=True)
    print(f"rows={len(rows)}", flush=True)
    print(f"pinyin_mode={args.pinyin_mode}", flush=True)
    print(f"label={args.label}", flush=True)
    print(f"manifest_sha256={manifest_sha}", flush=True)
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
        raise RuntimeError(
            "Adapter checkpoint revision mismatch"
        )

    predictions = base.evaluate_backend(
        backend,
        rows,
        condition=args.label,
        output_path=(
            args.output_root
            / f"{args.label}_predictions.jsonl"
        ),
        top_k=args.top_k,
        beam_size=args.beam_size,
        log_every=args.log_every,
    )

    metrics = base.compute_metrics(predictions)

    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "complete",
        "analysis_scope": evaluation_scope,
        "author": args.author,
        "selected_rows": len(rows),
        "population": population_audit,
        "selected_rows_file": (
            str(args.selected_rows)
            if args.selected_rows is not None
            else None
        ),
        "selected_rows_file_sha256": selected_rows_sha256,
        "selected_row_id_manifest_sha256": manifest_sha,
        "pinyin_mode": args.pinyin_mode,
        "target_type": "short",
        "beam_size": args.beam_size,
        "top_k": args.top_k,
        "adapter_path": str(args.adapter),
        "adapter_sha256": base.sha256_file(args.adapter),
        "adapter_metadata": metadata,
        "parameter_audit": parameter_audit,
        "metrics": metrics,
        "used_test": False,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
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
            "author": args.author,
            "scope": evaluation_scope,
            "pinyin_mode": args.pinyin_mode,
            "metrics": metrics,
        },
    )

    micro = metrics["micro"]

    print()
    print("===== RESULT =====")
    print(
        f"{args.label} "
        f"N={len(rows)} "
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
