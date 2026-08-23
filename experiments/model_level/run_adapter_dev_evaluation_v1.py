"""Development evaluation for Model-Level Adapter V1 LR calibration.

This runner evaluates Generic PinyinGPT2-Concat and one or more per-user
Adapter checkpoints on the frozen Clean3 Train-Val population.

Primary LR-calibration protocol:
- author: Agent Phage
- frozen Clean3 Train-Val
- Full Pinyin / Short target only
- deterministic held-out subset
- Beam 16, returned Top-10
- Generic and every Adapter use exactly the same rows
- no Test rows are read
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping

import torch

from src.evaluation.ranking import compute_metrics
from src.model_level.pinyingpt_adapter import (
    SerialAdapterConfig,
    install_adapters,
    load_adapter_bundle,
    set_adapter_training_mode,
)
from src.reference_backend_pinyingpt.backend import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    OFFICIAL_CODE_REVISION,
    PinyinGPTConcatBackend,
)


EXPERIMENT = "model_level_adapter_dev_evaluation_v1"

EXPECTED_VAL_SHA256 = (
    "d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220"
)
EXPECTED_VAL_ROWS = 34_416

DEFAULT_AUTHOR = "Agent Phage"
DEFAULT_SEED = 20_260_822
DEFAULT_MAX_ROWS = 2_048
DEFAULT_TOP_K = 10
DEFAULT_BEAM_SIZE = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_val_rows(path: Path, *, author: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    val_sha = sha256_file(path)
    if val_sha != EXPECTED_VAL_SHA256:
        raise RuntimeError(
            f"Train-Val SHA mismatch: expected {EXPECTED_VAL_SHA256}, got {val_sha}"
        )

    total = 0
    author_rows: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue

            row = json.loads(line)
            total += 1

            if str(row.get("source_split", "")).lower() == "test":
                raise RuntimeError("STOP: Test row detected in Train-Val input")

            if (
                str(row.get("condition")) != "full_short"
                or str(row.get("target_type")) != "short"
                or str(row.get("pinyin_type")) != "full"
                or str(row.get("standardized_partition")) != "train_val"
            ):
                raise RuntimeError(
                    f"Train-Val schema violation at row {row.get('row_id')}"
                )

            segments = tuple(map(str, row["pinyin_segments"]))
            gold = str(row["gold"])

            if len(gold) != len(segments):
                raise RuntimeError(
                    f"Gold/Pinyin alignment failure at row {row.get('row_id')}"
                )

            if str(row.get("author")) == author:
                author_rows.append(row)

    if total != EXPECTED_VAL_ROWS:
        raise RuntimeError(
            f"Train-Val row-count mismatch: expected {EXPECTED_VAL_ROWS}, got {total}"
        )

    if not author_rows:
        raise RuntimeError(f"No Train-Val rows found for author {author!r}")

    audit = {
        "train_val_sha256": val_sha,
        "train_val_total_rows": total,
        "author": author,
        "author_train_val_rows": len(author_rows),
        "used_test": False,
    }
    return author_rows, audit


def deterministic_subset(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
    max_rows: int | None,
) -> list[dict[str, Any]]:
    materialized = list(rows)

    def key(row: Mapping[str, Any]) -> tuple[str, str]:
        row_id = str(row["row_id"])
        digest = hashlib.sha256(
            f"{seed}\0{row_id}".encode("utf-8")
        ).hexdigest()
        return digest, row_id

    selected = sorted(materialized, key=key)

    if max_rows is not None:
        if max_rows < 1:
            raise ValueError("--max-rows must be positive")
        selected = selected[:max_rows]

    return selected


def prediction_row(
    row: Mapping[str, Any],
    backend: PinyinGPTConcatBackend,
    *,
    condition: str,
    top_k: int,
    beam_size: int,
) -> dict[str, Any]:
    context = str(row["context"])
    pinyin = list(map(str, row["pinyin_segments"]))
    gold = str(row["gold"])

    started = time.perf_counter()

    with torch.inference_mode():
        generated = backend.generate(
            context,
            pinyin,
            top_k=top_k,
            beam_size=beam_size,
        )

    elapsed = time.perf_counter() - started

    candidates = [candidate.text for candidate in generated.candidates]
    scores = [float(candidate.log_probability) for candidate in generated.candidates]
    rank = candidates.index(gold) + 1 if gold in candidates else None

    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "condition": condition,
        "interaction_id": str(row["row_id"]),
        "row_id": str(row["row_id"]),
        "user_id": str(row["author"]),
        "author": str(row["author"]),
        "gold": gold,
        "context": context,
        "segmented_pinyin": pinyin,
        "typed_pinyin": " ".join(pinyin),
        "top10_candidates": candidates,
        "top10_candidate_scores": scores,
        "gold_top10_rank": rank,
        "top1_correct": rank == 1,
        "top3_correct": rank is not None and rank <= 3,
        "top5_correct": rank is not None and rank <= 5,
        "top10_present": rank is not None,
        "reciprocal_rank_at_10": 0.0 if rank is None else 1.0 / rank,
        "inference_seconds": elapsed,
        "beam_size": beam_size,
        "top_k": top_k,
        "runtime_device": generated.runtime_device,
    }


def load_existing_predictions(
    output_path: Path,
    rows: list[dict[str, Any]],
    *,
    condition: str,
    top_k: int,
    beam_size: int,
) -> dict[str, dict[str, Any]]:
    """Load and strictly validate resumable prediction rows."""

    if not output_path.exists():
        return {}

    expected = {str(row["row_id"]): row for row in rows}
    existing: dict[str, dict[str, Any]] = {}

    # Read in binary mode so an interrupted final JSONL write can be
    # distinguished from corruption in an earlier completed record.
    with output_path.open("rb") as source:
        file_size = source.seek(0, 2)
        source.seek(0)

        line_number = 0
        last_good_offset = 0

        while True:
            raw = source.readline()
            if not raw:
                break

            line_number += 1
            current_offset = source.tell()

            if not raw.strip():
                last_good_offset = current_offset
                continue

            try:
                line = raw.decode("utf-8")
                prediction = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if current_offset == file_size:
                    print(
                        f"{condition}: removing incomplete trailing "
                        f"JSONL record from {output_path}",
                        flush=True,
                    )
                    with output_path.open("r+b") as destination:
                        destination.truncate(last_good_offset)
                    break

                raise RuntimeError(
                    f"Corrupt non-final resume record in {output_path} "
                    f"at line {line_number}: {exc}"
                ) from exc

            last_good_offset = current_offset

            row_id = str(prediction.get("row_id"))

            if row_id not in expected:
                raise RuntimeError(
                    f"Resume file contains unknown row_id "
                    f"{row_id!r}: {output_path}"
                )

            if row_id in existing:
                raise RuntimeError(
                    f"Duplicate row_id {row_id!r} "
                    f"in resume file: {output_path}"
                )

            row = expected[row_id]

            if prediction.get("condition") != condition:
                raise RuntimeError(
                    f"Resume condition mismatch for {row_id}: "
                    f"{prediction.get('condition')!r} != {condition!r}"
                )

            if prediction.get("gold") != str(row["gold"]):
                raise RuntimeError(
                    f"Resume gold mismatch for {row_id}"
                )

            expected_pinyin = list(map(str, row["pinyin_segments"]))

            if prediction.get("segmented_pinyin") != expected_pinyin:
                raise RuntimeError(
                    f"Resume Pinyin mismatch for {row_id}"
                )

            if int(prediction.get("top_k", -1)) != top_k:
                raise RuntimeError(
                    f"Resume top_k mismatch for {row_id}"
                )

            if int(prediction.get("beam_size", -1)) != beam_size:
                raise RuntimeError(
                    f"Resume beam_size mismatch for {row_id}"
                )

            existing[row_id] = prediction

    return existing


def evaluate_backend(
    backend: PinyinGPTConcatBackend,
    rows: list[dict[str, Any]],
    *,
    condition: str,
    output_path: Path,
    top_k: int,
    beam_size: int,
    log_every: int,
) -> list[dict[str, Any]]:
    started = time.perf_counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing_predictions(
        output_path,
        rows,
        condition=condition,
        top_k=top_k,
        beam_size=beam_size,
    )

    total = len(rows)
    completed = len(existing)
    remaining = total - completed

    print(
        f"{condition}: resume_validated={completed}/{total} "
        f"remaining={remaining}",
        flush=True,
    )

    if remaining:
        with output_path.open(
            "a",
            encoding="utf-8",
            newline="\n",
            buffering=1,
        ) as destination:
            for row in rows:
                row_id = str(row["row_id"])

                if row_id in existing:
                    continue

                prediction = prediction_row(
                    row,
                    backend,
                    condition=condition,
                    top_k=top_k,
                    beam_size=beam_size,
                )

                existing[row_id] = prediction
                destination.write(
                    canonical_json(prediction) + "\n"
                )

                completed += 1

                if (
                    completed == total
                    or (
                        log_every
                        and completed % log_every == 0
                    )
                ):
                    print(
                        f"{condition}: {completed}/{total} "
                        f"elapsed_this_run="
                        f"{time.perf_counter() - started:.1f}s",
                        flush=True,
                    )

    if len(existing) != total:
        raise RuntimeError(
            f"{condition}: resume finished with "
            f"{len(existing)}/{total} predictions"
        )

    return [
        existing[str(row["row_id"])]
        for row in rows
    ]


def paired_top1_outcomes(
    generic: Iterable[Mapping[str, Any]],
    adapted: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    left = {str(row["interaction_id"]): row for row in generic}
    right = {str(row["interaction_id"]): row for row in adapted}

    if set(left) != set(right):
        raise RuntimeError("Generic and Adapter interaction IDs are not aligned")

    result = {
        "rescue": 0,
        "harm": 0,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
    }

    for interaction_id in sorted(left):
        generic_correct = left[interaction_id]["gold_top10_rank"] == 1
        adapted_correct = right[interaction_id]["gold_top10_rank"] == 1

        if not generic_correct and adapted_correct:
            result["rescue"] += 1
        elif generic_correct and not adapted_correct:
            result["harm"] += 1
        elif generic_correct:
            result["unchanged_correct"] += 1
        else:
            result["unchanged_wrong"] += 1

    result["net"] = result["rescue"] - result["harm"]
    return result


def metric_delta(
    generic_metrics: Mapping[str, Any],
    adapted_metrics: Mapping[str, Any],
) -> dict[str, float]:
    result: dict[str, float] = {}

    for name in ("top1", "top3", "top5", "top10", "mrr_at_10"):
        result[name] = (
            float(adapted_metrics["micro"][name])
            - float(generic_metrics["micro"][name])
        )

    result["missing_at_10_rate"] = (
        float(adapted_metrics["micro"]["missing_at_10_rate"])
        - float(generic_metrics["micro"]["missing_at_10_rate"])
    )
    return result


def release_backend(backend: Any) -> None:
    del backend
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_adapter_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--adapter must use LABEL=/path/to/adapter.safetensors"
        )

    label, path = value.split("=", 1)
    label = label.strip()
    path = Path(path.strip())

    if not label:
        raise argparse.ArgumentTypeError("Adapter label cannot be empty")

    if path.suffix != ".safetensors":
        raise argparse.ArgumentTypeError(
            "Adapter path must end in .safetensors"
        )

    return label, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-val", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)

    parser.add_argument("--author", default=DEFAULT_AUTHOR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help="Deterministic held-out subset size; omit by passing 0 for all author rows.",
    )

    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--beam-size", type=int, default=DEFAULT_BEAM_SIZE)
    parser.add_argument("--log-every", type=int, default=100)

    parser.add_argument(
        "--adapter",
        action="append",
        type=parse_adapter_spec,
        required=True,
        metavar="LABEL=PATH",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.top_k != 10:
        raise ValueError("Dev Evaluation V1 freezes top_k=10")
    if args.beam_size != 16:
        raise ValueError("Dev Evaluation V1 freezes beam_size=16")

    max_rows = None if args.max_rows == 0 else args.max_rows

    args.output_root.mkdir(parents=True, exist_ok=True)

    author_rows, population_audit = load_val_rows(
        args.train_val,
        author=args.author,
    )

    rows = deterministic_subset(
        author_rows,
        seed=args.seed,
        max_rows=max_rows,
    )

    if not rows:
        raise RuntimeError("No evaluation rows selected")

    selected_manifest = args.output_root / "selected_rows.jsonl"
    with selected_manifest.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(
                canonical_json(
                    {
                        "row_id": row["row_id"],
                        "author": row["author"],
                        "gold": row["gold"],
                        "context": row["context"],
                        "pinyin_segments": row["pinyin_segments"],
                        "standardized_partition": row["standardized_partition"],
                    }
                )
                + "\n"
            )

    config = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "checkpoint": CHECKPOINT_ID,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "official_code_revision": OFFICIAL_CODE_REVISION,
        "checkpoint_path": str(args.checkpoint),
        "train_val_path": str(args.train_val),
        **population_audit,
        "selected_rows": len(rows),
        "selection_seed": args.seed,
        "max_rows": max_rows,
        "full_author_evaluation": max_rows is None,
        "pinyin_condition": "full",
        "target_type": "short",
        "beam_size": args.beam_size,
        "top_k": args.top_k,
        "adapter_specs": [
            {"label": label, "path": str(path)}
            for label, path in args.adapter
        ],
        "used_test": False,
    }
    write_json(args.output_root / "evaluation_config.json", config)

    print("===== GENERIC =====", flush=True)

    generic_backend = PinyinGPTConcatBackend(
        args.checkpoint,
        device=args.device,
    )

    generic_predictions = evaluate_backend(
        generic_backend,
        rows,
        condition="generic",
        output_path=args.output_root / "generic_predictions.jsonl",
        top_k=args.top_k,
        beam_size=args.beam_size,
        log_every=args.log_every,
    )

    generic_metrics = compute_metrics(generic_predictions)

    write_json(
        args.output_root / "generic_metrics.json",
        generic_metrics,
    )

    release_backend(generic_backend)

    conditions: dict[str, Any] = {}

    for label, adapter_path in args.adapter:
        print()
        print(f"===== ADAPTER {label} =====", flush=True)

        if not adapter_path.is_file():
            raise FileNotFoundError(adapter_path)

        backend = PinyinGPTConcatBackend(
            args.checkpoint,
            device=args.device,
        )

        parameter_audit = install_adapters(
            backend.model,
            SerialAdapterConfig(),
        )

        metadata = load_adapter_bundle(
            backend.model,
            adapter_path,
        )
        set_adapter_training_mode(
            backend.model,
            enabled=False,
        )

        if metadata.get("author") != args.author:
            raise RuntimeError(
                f"Adapter {label} author mismatch: "
                f"{metadata.get('author')!r} != {args.author!r}"
            )

        if metadata.get("checkpoint_revision") != CHECKPOINT_REVISION:
            raise RuntimeError(
                f"Adapter {label} checkpoint revision mismatch"
            )

        predictions = evaluate_backend(
            backend,
            rows,
            condition=label,
            output_path=args.output_root / f"{label}_predictions.jsonl",
            top_k=args.top_k,
            beam_size=args.beam_size,
            log_every=args.log_every,
        )

        metrics = compute_metrics(predictions)
        delta = metric_delta(generic_metrics, metrics)
        outcomes = paired_top1_outcomes(
            generic_predictions,
            predictions,
        )

        condition_result = {
            "label": label,
            "adapter_path": str(adapter_path),
            "adapter_sha256": sha256_file(adapter_path),
            "adapter_metadata": metadata,
            "parameter_audit": parameter_audit,
            "metrics": metrics,
            "delta_vs_generic": delta,
            "top1_outcomes_vs_generic": outcomes,
        }

        write_json(
            args.output_root / f"{label}_metrics.json",
            condition_result,
        )

        conditions[label] = condition_result

        micro = metrics["micro"]
        print(
            f"{label}: "
            f"Top1={micro['top1']:.6f} "
            f"Top3={micro['top3']:.6f} "
            f"Top5={micro['top5']:.6f} "
            f"Top10={micro['top10']:.6f} "
            f"MRR={micro['mrr_at_10']:.6f} "
            f"Missing={micro['missing_at_10_rate']:.6f} "
            f"rescue={outcomes['rescue']} "
            f"harm={outcomes['harm']} "
            f"net={outcomes['net']}",
            flush=True,
        )

        release_backend(backend)

    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "complete",
        "analysis_scope": "DEVELOPMENT LR CALIBRATION; not final Test inference",
        "author": args.author,
        "selected_rows": len(rows),
        "population": population_audit,
        "selection_seed": args.seed,
        "pinyin_condition": "full",
        "target_type": "short",
        "beam_size": args.beam_size,
        "top_k": args.top_k,
        "generic_metrics": generic_metrics,
        "conditions": conditions,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": args.device,
            "device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() and args.device.startswith("cuda")
                else "CPU"
            ),
            "visible_cuda_devices": torch.cuda.device_count(),
        },
        "selected_manifest_sha256": sha256_file(selected_manifest),
        "used_test": False,
    }

    write_json(
        args.output_root / "evaluation_result.json",
        result,
    )

    generic_micro = generic_metrics["micro"]

    print()
    print("===== DEV EVALUATION SUMMARY =====")
    print(
        "Generic "
        f"Top1={generic_micro['top1']:.6f} "
        f"Top3={generic_micro['top3']:.6f} "
        f"Top5={generic_micro['top5']:.6f} "
        f"Top10={generic_micro['top10']:.6f} "
        f"MRR={generic_micro['mrr_at_10']:.6f} "
        f"Missing={generic_micro['missing_at_10_rate']:.6f}"
    )

    for label, condition in conditions.items():
        micro = condition["metrics"]["micro"]
        outcomes = condition["top1_outcomes_vs_generic"]
        print(
            f"{label} "
            f"Top1={micro['top1']:.6f} "
            f"Top3={micro['top3']:.6f} "
            f"Top5={micro['top5']:.6f} "
            f"Top10={micro['top10']:.6f} "
            f"MRR={micro['mrr_at_10']:.6f} "
            f"Missing={micro['missing_at_10_rate']:.6f} "
            f"rescue={outcomes['rescue']} "
            f"harm={outcomes['harm']} "
            f"net={outcomes['net']}"
        )


if __name__ == "__main__":
    main()
