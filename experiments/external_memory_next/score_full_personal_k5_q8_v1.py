"""Exact historical Q8 scoring on the frozen Full-Pinyin Personal-K5 pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

from src.personalisation.task_specific_biencoder import refuse_closed_path, sha256_file, write_json


EXPECTED_VAL = "d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220"
EXPECTED_FEATURES = "e311261cb0c5ea252ce593fdaa43ff87826d19eef440f39e2efc67ddc4310b13"
EXPECTED_GENERIC = "cf4ae382fa23e5ec1154bf28320d13ac1d6ca9600e9dcf8a6aa599600bc28eab"
EXPECTED_ROWS = 34_416
EXPECTED_ELIGIBLE = 3_556
EXPECTED_PAIRS = 6_942


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def index_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {str(row["row_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate row IDs")
    return result


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(map(float, values))
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--generic", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    for path in (args.val, args.features, args.generic, args.checkpoint, args.output_root):
        refuse_closed_path(path)
    for path, expected in ((args.val, EXPECTED_VAL), (args.features, EXPECTED_FEATURES), (args.generic, EXPECTED_GENERIC)):
        if sha256_file(path) != expected:
            raise ValueError(f"frozen input hash changed: {path}")
    val_rows = read_jsonl(args.val)
    feature_rows = read_jsonl(args.features)
    generic_rows = read_jsonl(args.generic)
    if not (len(val_rows) == len(feature_rows) == len(generic_rows) == EXPECTED_ROWS):
        raise ValueError("Full row count changed")
    order = [str(row["row_id"]) for row in val_rows]
    if order != [str(row["row_id"]) for row in feature_rows] or order != [str(row["row_id"]) for row in generic_rows]:
        raise ValueError("Full row order changed")
    if any(row.get("used_dev3000") or row.get("used_test") or row.get("pilot_partition") == "test" for row in [*val_rows, *feature_rows, *generic_rows]):
        raise ValueError("closed-resource row found")
    features = index_rows(feature_rows)
    generic = index_rows(generic_rows)
    eligible = [row_id for row_id in order if features[row_id]["personal_k5"]]
    if len(eligible) != EXPECTED_ELIGIBLE or sum(len(features[row_id]["personal_k5"]) for row_id in eligible) != EXPECTED_PAIRS:
        raise ValueError("Full Personal-K5 surface changed")

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / "full_q8_scores.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if output_path.is_file():
        for row in read_jsonl(output_path):
            row_id = str(row["row_id"])
            if row_id in completed or row_id not in features:
                raise ValueError("stale or duplicate Q8 cache row")
            expected = list(map(str, features[row_id]["personal_k5"]))
            if [str(item["candidate"]) for item in row["scores"]] != expected:
                raise ValueError("Q8 cache candidate mismatch")
            completed[row_id] = row
    pending = [row_id for row_id in eligible if row_id not in completed]
    print(f"Full Q8 eligible={len(eligible):,} complete={len(completed):,} pending={len(pending):,}", flush=True)
    if pending:
        from src.reference_backend_pinyingpt.backend import PinyinGPTConcatBackend

        backend = PinyinGPTConcatBackend(args.checkpoint, device=args.device)
        mode = "a" if output_path.is_file() and output_path.stat().st_size else "w"
        with output_path.open(mode, encoding="utf-8", newline="\n") as destination:
            for number, row_id in enumerate(pending, start=1):
                candidates = list(map(str, features[row_id]["personal_k5"]))
                context = str(generic[row_id]["model_used_context"])[-8:]
                started = time.perf_counter()
                scored = backend.score_candidates(
                    context=context,
                    typed_pinyin=tuple(map(str, generic[row_id]["pinyin_segments"])),
                    candidates=candidates,
                )
                elapsed = time.perf_counter() - started
                by_text = {str(value.text): value for value in scored}
                if set(by_text) != set(candidates):
                    raise ValueError(f"Q8 candidate mismatch at {row_id}")
                row = {
                    "schema_version": 1,
                    "track": "full",
                    "row_id": row_id,
                    "author": str(generic[row_id]["author"]),
                    "context_chars": 8,
                    "row_inference_seconds": elapsed,
                    "scores": [
                        {
                            "candidate": candidate,
                            "fixed_log_probability": float(by_text[candidate].log_probability),
                            "fixed_mean_log_probability": float(by_text[candidate].mean_log_probability),
                        }
                        for candidate in candidates
                    ],
                    "gold_used_for_scoring": False,
                    "used_dev3000": False,
                    "used_test": False,
                }
                destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                if number % args.progress_every == 0 or number == len(pending):
                    destination.flush()
                    print(f"Full Q8 {number:,}/{len(pending):,}", flush=True)
    final = index_rows(read_jsonl(output_path))
    if set(final) != set(eligible):
        raise ValueError("Full Q8 cache incomplete")
    latency = [1000.0 * float(final[row_id]["row_inference_seconds"]) for row_id in eligible]
    summary = {
        "schema_version": 1,
        "status": "complete",
        "track": "full",
        "eligible_rows": len(eligible),
        "candidate_pairs": sum(len(final[row_id]["scores"]) for row_id in eligible),
        "latency_ms": {
            "n": len(latency),
            "mean": statistics.fmean(latency),
            "p50": percentile(latency, .5),
            "p95": percentile(latency, .95),
            "p99": percentile(latency, .99),
        },
        "scores_sha256": sha256_file(output_path),
        "used_dev3000": False,
        "used_test": False,
    }
    write_json(args.output_root / "full_q8_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
