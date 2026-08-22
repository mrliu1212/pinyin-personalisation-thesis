"""Read-only closeout audit for post-hoc Task-BiEncoder artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.context_comparison import run_full_transfer_initial_final_v1 as base
from src.personalisation.task_specific_biencoder import refuse_closed_path


ROWS = 34_416


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def index_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {str(row["row_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate row IDs")
    return result


def audit_chronology(fit: Sequence[Mapping[str, Any]], val: Sequence[Mapping[str, Any]]) -> int:
    history = base.CausalHistoryIndex([*fit, *val])
    maximum = 0
    for row in val:
        author = str(row["author"])
        position = int(row["chronological_position"])
        pinyin = base.pinyin_of(row)
        visible = history.visible_same_pinyin(author=author, position=position, pinyin=pinyin)
        raw_count = history.raw_visible_count(author=author, position=position)
        maximum = max(maximum, raw_count)
        if raw_count > 5000:
            raise ValueError("H5000 raw-history budget exceeded")
        if any(
            item.record.author != author
            or item.record.position >= position
            or item.record.pinyin != pinyin
            or item.age < 0
            or item.age >= 5000
            for item in visible
        ):
            raise ValueError("causal same-Pinyin history invariant failed")
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--initial-fit", type=Path, required=True)
    parser.add_argument("--initial-val", type=Path, required=True)
    parser.add_argument("--initial-frequency", type=Path, required=True)
    parser.add_argument("--initial-support", type=Path, required=True)
    parser.add_argument("--full-fit", type=Path, required=True)
    parser.add_argument("--full-val", type=Path, required=True)
    parser.add_argument("--full-stage1", type=Path, required=True)
    parser.add_argument("--full-support", type=Path, required=True)
    args = parser.parse_args()
    for path in vars(args).values():
        refuse_closed_path(path)

    result = json.loads(args.result.read_text(encoding="utf-8"))
    if result["used_dev3000"] or result["used_test"]:
        raise ValueError("closed resource marker is true")
    if result["initial"]["rows"] != ROWS or result["full"]["rows"] != ROWS:
        raise ValueError("evaluation population changed")

    for methods in result["initial"]["fixed_surface"].values():
        missing = {round(item["metrics"]["overall"]["missing10"], 15) for item in methods.values()}
        if len(missing) != 1:
            raise ValueError("Initial fixed-surface Missing@10 changed")
    full_missing = {
        round(item["metrics"]["overall"]["missing10"], 15)
        for item in result["full"]["fixed_surface"].values()
    }
    if len(full_missing) != 1:
        raise ValueError("Full fixed-surface Missing@10 changed")

    initial_fit = read_jsonl(args.initial_fit)
    initial_val = read_jsonl(args.initial_val)
    full_fit = read_jsonl(args.full_fit)
    full_val = read_jsonl(args.full_val)
    predictions = read_jsonl(args.predictions)
    if len(predictions) != 4 * ROWS:
        raise ValueError("selected prediction count changed")
    orders = {
        "initial": [str(row["row_id"]) for row in initial_val],
        "full": [str(row["row_id"]) for row in full_val],
    }
    for track in ("initial", "full"):
        for method in ("generic", "task"):
            actual = [
                str(row["row_id"])
                for row in predictions
                if row["track"] == track and row["method"] == method
            ]
            if actual != orders[track]:
                raise ValueError(f"selected prediction order changed: {track}/{method}")

    for source_path, support_path, generic_key in (
        (args.initial_frequency, args.initial_support, "frequency_candidates"),
        (args.full_stage1, args.full_support, "generic_frequency_candidates"),
    ):
        support = index_rows(read_jsonl(support_path))
        for row in read_jsonl(source_path):
            row_id = str(row["row_id"])
            union = set(map(str, support[row_id]["candidate_union"]))
            generic = {str(item["candidate"]) for item in row[generic_key]}
            personal = set(map(str, row.get("personal_k5", ())))
            if not generic.issubset(union) or not personal.issubset(union):
                raise ValueError(f"candidate union incomplete: {row_id}")

    payload = {
        "status": "pass",
        "track_rows": {"initial": len(initial_val), "full": len(full_val)},
        "selected_predictions": len(predictions),
        "chronology_h5000": {
            "initial_max_raw": audit_chronology(initial_fit, initial_val),
            "full_max_raw": audit_chronology(full_fit, full_val),
        },
        "prediction_order_exact": True,
        "candidate_union_complete": True,
        "fixed_surface_missing10_invariant": True,
        "used_dev3000": False,
        "used_test": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
