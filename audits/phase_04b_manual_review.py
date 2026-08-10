"""Prepare and summarize manual review of existing Phase 4B audit samples."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


DEFAULT_AUDIT_DIR = Path("results/audits/phase_04b")
DEFAULT_CSV = DEFAULT_AUDIT_DIR / "manual_review.csv"

SAMPLE_FILES = (
    ("polyphonic_flagged", "polyphonic_flagged_sample.jsonl"),
    ("polyphonic_unflagged", "polyphonic_unflagged_sample.jsonl"),
    ("top10_missing", "top10_missing_sample.jsonl"),
)

PINYIN_VALUES = ("correct", "incorrect", "uncertain")
SEGMENTATION_VALUES = ("reasonable", "unreasonable", "uncertain")
MISSING_CAUSE_VALUES = (
    "proper_name",
    "rare_or_literary_vocabulary",
    "traditional_or_variant_form",
    "segmentation_problem",
    "pinyin_problem",
    "likely_rank_beyond_top10",
    "other",
    "uncertain",
)

FIELDNAMES = (
    "sample_type",
    "interaction_id",
    "work_title",
    "source_start_offset",
    "source_end_offset",
    "raw_context",
    "derived_context",
    "target",
    "generated_pinyin",
    "pinyin_syllables",
    "polyphonic_flag",
    "base_candidates",
    "target_present",
    "target_rank",
    "pinyin_judgement",
    "segmentation_judgement",
    "missing_cause",
    "notes",
)


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on {path}:{line_number}") from error
    return records


def candidate_text(candidates: Iterable[dict]) -> str:
    return " | ".join(
        f"{candidate['base_rank']}:{candidate['text']}" for candidate in candidates
    )


def prepare_review_csv(audit_dir: Path, output_path: Path) -> int:
    rows: list[dict[str, object]] = []
    for expected_type, filename in SAMPLE_FILES:
        for record in read_jsonl(audit_dir / filename):
            actual_type = record.get("audit_sample_category")
            if actual_type != expected_type:
                raise ValueError(
                    f"{filename} contains sample type {actual_type!r}; "
                    f"expected {expected_type!r}"
                )
            rows.append(
                {
                    "sample_type": expected_type,
                    "interaction_id": record["interaction_id"],
                    "work_title": record["work_title"],
                    "source_start_offset": record.get("source_start_offset", ""),
                    "source_end_offset": record.get("source_end_offset", ""),
                    "raw_context": record["raw_context"],
                    "derived_context": record["derived_context"],
                    "target": record["target_candidate"],
                    "generated_pinyin": record.get("pinyin", ""),
                    "pinyin_syllables": " ".join(
                        record.get("pinyin_syllables") or []
                    ),
                    "polyphonic_flag": str(
                        bool(record["polyphonic_review_required"])
                    ).lower(),
                    "base_candidates": candidate_text(
                        record["ordered_base_candidates"]
                    ),
                    "target_present": str(bool(record["target_present"])).lower(),
                    "target_rank": (
                        "" if record["target_rank"] is None else record["target_rank"]
                    ),
                    "pinyin_judgement": "",
                    "segmentation_judgement": "",
                    "missing_cause": "",
                    "notes": "",
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def read_review_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        present_columns = reader.fieldnames or []
        missing_columns = [field for field in FIELDNAMES if field not in present_columns]
        if missing_columns:
            raise ValueError(f"missing CSV columns: {', '.join(missing_columns)}")
        return list(reader)


def validate_label(value: str, allowed: tuple[str, ...], field: str, row: int) -> str:
    normalized = value.strip()
    if normalized and normalized not in allowed:
        raise ValueError(
            f"row {row}: invalid {field} {normalized!r}; "
            f"allowed values are {', '.join(allowed)} or blank"
        )
    return normalized


def distribution(
    rows: list[dict[str, str]], field: str, allowed: tuple[str, ...]
) -> dict[str, object]:
    counts = Counter(row[field] for row in rows if row[field])
    labelled = sum(counts.values())
    return {
        "total_rows": len(rows),
        "labelled_rows": labelled,
        "unlabelled_rows": len(rows) - labelled,
        "values": {
            value: {
                "count": counts[value],
                "percentage_of_labelled": (
                    counts[value] / labelled if labelled else None
                ),
            }
            for value in allowed
        },
    }


def summarize_review(path: Path) -> dict[str, object]:
    rows = read_review_csv(path)
    expected_types = {sample_type for sample_type, _ in SAMPLE_FILES}
    normalized_rows: list[dict[str, str]] = []
    for row_number, row in enumerate(rows, 2):
        sample_type = row["sample_type"].strip()
        if sample_type not in expected_types:
            raise ValueError(
                f"row {row_number}: invalid sample_type {sample_type!r}"
            )
        normalized = dict(row)
        normalized["sample_type"] = sample_type
        normalized["pinyin_judgement"] = validate_label(
            row["pinyin_judgement"], PINYIN_VALUES, "pinyin_judgement", row_number
        )
        normalized["segmentation_judgement"] = validate_label(
            row["segmentation_judgement"],
            SEGMENTATION_VALUES,
            "segmentation_judgement",
            row_number,
        )
        normalized["missing_cause"] = validate_label(
            row["missing_cause"], MISSING_CAUSE_VALUES, "missing_cause", row_number
        )
        normalized_rows.append(normalized)

    flagged = [
        row for row in normalized_rows if row["sample_type"] == "polyphonic_flagged"
    ]
    unflagged = [
        row
        for row in normalized_rows
        if row["sample_type"] == "polyphonic_unflagged"
    ]
    missing = [
        row for row in normalized_rows if row["sample_type"] == "top10_missing"
    ]
    return {
        "percentage_denominator": "manually labelled rows in each distribution",
        "polyphonic_flagged_pinyin": distribution(
            flagged, "pinyin_judgement", PINYIN_VALUES
        ),
        "polyphonic_unflagged_pinyin": distribution(
            unflagged, "pinyin_judgement", PINYIN_VALUES
        ),
        "top10_missing_segmentation": distribution(
            missing, "segmentation_judgement", SEGMENTATION_VALUES
        ),
        "top10_missing_cause": distribution(
            missing, "missing_cause", MISSING_CAUSE_VALUES
        ),
    }


def display_summary(summary: dict[str, object]) -> str:
    sections = (
        ("Polyphonic flagged — Pinyin", "polyphonic_flagged_pinyin"),
        ("Polyphonic unflagged — Pinyin", "polyphonic_unflagged_pinyin"),
        ("Top-10 missing — segmentation", "top10_missing_segmentation"),
        ("Top-10 missing — cause", "top10_missing_cause"),
    )
    lines = [
        "Manual review summary",
        "Percentages use manually labelled rows within each distribution.",
    ]
    for title, key in sections:
        result = summary[key]
        lines.extend(
            [
                "",
                title,
                f"Labelled: {result['labelled_rows']}/{result['total_rows']} "
                f"(blank: {result['unlabelled_rows']})",
            ]
        )
        for value, metrics in result["values"].items():
            percentage = metrics["percentage_of_labelled"]
            formatted = "n/a" if percentage is None else f"{percentage:.2%}"
            lines.append(f"  {value}: {metrics['count']} ({formatted})")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="create the blank review CSV")
    prepare.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    prepare.add_argument("--output", type=Path, default=DEFAULT_CSV)

    summarize = subparsers.add_parser(
        "summarize", help="aggregate manually entered labels"
    )
    summarize.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    summarize.add_argument("--json-output", type=Path)

    args = parser.parse_args()
    if args.command == "prepare":
        count = prepare_review_csv(args.audit_dir, args.output)
        print(f"Wrote {count} blank review rows to {args.output}")
        return

    summary = summarize_review(args.csv)
    print(display_summary(summary))
    if args.json_output:
        args.json_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
