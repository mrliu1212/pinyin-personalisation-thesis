"""Prepare and summarize the final Phase 4B.7 manual data-quality audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path(
    "data/processed/interactions/zhu_ziqing_simplified_rime/interactions.jsonl"
)
DEFAULT_COMPARISON = Path(
    "data/processed/interactions/zhu_ziqing_simplified_rime/phase_04b6_comparison.json"
)
DEFAULT_OUTPUT_DIR = Path("results/audits/phase_04b7")
DEFAULT_SEED = 40407
DEFAULT_SAMPLE_SIZE = 100

PINYIN_VALUES = ("correct", "incorrect", "uncertain")
MISSING_CAUSE_VALUES = (
    "proper_name",
    "rare_or_literary_vocabulary",
    "segmentation_problem",
    "pinyin_problem",
    "candidate_coverage_problem",
    "traditional_variant_residual",
    "other",
    "uncertain",
)

PROVENANCE_FIELDS = (
    "interaction_id",
    "work_id",
    "work_title",
    "source_start_offset",
    "source_end_offset",
    "source_original_target",
    "source_original_processed_file",
)

POLYPHONIC_FIELDS = PROVENANCE_FIELDS + (
    "context",
    "target",
    "generated_pinyin",
    "pinyin_syllables",
    "polyphonic_flag",
    "candidates",
    "target_rank",
    "pinyin_judgement",
    "notes",
)

MISSING_FIELDS = PROVENANCE_FIELDS + (
    "context",
    "target",
    "generated_pinyin",
    "pinyin_syllables",
    "candidates",
    "target_rank",
    "polyphonic_flag",
    "missing_cause",
    "notes",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def deterministic_sample(
    records: Iterable[dict[str, Any]],
    *,
    seed: int,
    sample_name: str,
    sample_size: int,
) -> list[dict[str, Any]]:
    pool = list(records)
    if len(pool) < sample_size:
        raise ValueError(
            f"{sample_name} has {len(pool)} eligible rows; cannot sample "
            f"{sample_size} without replacement"
        )

    def key(record: dict[str, Any]) -> tuple[str, str]:
        interaction_id = str(record["interaction_id"])
        digest = hashlib.sha256(
            f"{seed}:{sample_name}:{interaction_id}".encode("utf-8")
        ).hexdigest()
        return digest, interaction_id

    return sorted(pool, key=key)[:sample_size]


def candidate_text(candidates: Iterable[dict[str, Any]]) -> str:
    return " | ".join(
        f"{candidate['base_rank']}:{candidate['text']}" for candidate in candidates
    )


def common_review_values(record: dict[str, Any]) -> dict[str, Any]:
    provenance = record.get("normalization_provenance", {})
    return {
        "interaction_id": record["interaction_id"],
        "work_id": record["work_id"],
        "work_title": record["work_title"],
        "source_start_offset": record["source_start_offset"],
        "source_end_offset": record["source_end_offset"],
        "source_original_target": record.get("source_original_target", ""),
        "source_original_processed_file": provenance.get(
            "source_original_processed_file", ""
        ),
        "context": record["derived_context"],
        "target": record["target_candidate"],
        "generated_pinyin": record["pinyin"],
        "pinyin_syllables": " ".join(record.get("pinyin_syllables") or []),
        "polyphonic_flag": str(
            bool(record["polyphonic_review_required"])
        ).lower(),
        "candidates": candidate_text(record["candidates"]),
        "target_rank": "" if record["target_rank"] is None else record["target_rank"],
    }


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare(
    input_path: Path,
    comparison_path: Path,
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> dict[str, Any]:
    input_sha256 = sha256_file(input_path)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    expected_sha256 = comparison["output_interactions_sha256"]
    if input_sha256 != expected_sha256:
        raise ValueError("Phase 4B.6 interaction checksum does not match diagnostics")

    records = read_jsonl(input_path)
    polyphonic_pool = [
        record for record in records if record["polyphonic_review_required"]
    ]
    missing_pool = [record for record in records if not record["target_present"]]
    polyphonic_sample = deterministic_sample(
        polyphonic_pool,
        seed=seed,
        sample_name="polyphonic_flagged",
        sample_size=sample_size,
    )
    missing_sample = deterministic_sample(
        missing_pool,
        seed=seed,
        sample_name="top10_missing",
        sample_size=sample_size,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    polyphonic_rows = []
    for record in polyphonic_sample:
        row = common_review_values(record)
        row.update({"pinyin_judgement": "", "notes": ""})
        polyphonic_rows.append(row)
    missing_rows = []
    for record in missing_sample:
        row = common_review_values(record)
        row.update({"missing_cause": "", "notes": ""})
        missing_rows.append(row)

    polyphonic_path = output_dir / "polyphonic_review_sample.csv"
    missing_path = output_dir / "missing_review_sample.csv"
    write_csv(polyphonic_path, POLYPHONIC_FIELDS, polyphonic_rows)
    write_csv(missing_path, MISSING_FIELDS, missing_rows)

    manifest = {
        "schema_version": 1,
        "source_interactions": str(input_path),
        "source_interactions_sha256": input_sha256,
        "fixed_seed": seed,
        "sampling_method": (
            "SHA-256 rank of seed, sample name, and interaction_id; "
            "sampling without replacement"
        ),
        "sample_size": sample_size,
        "eligible_population_counts": {
            "polyphonic_flagged": len(polyphonic_pool),
            "top10_missing": len(missing_pool),
        },
        "outputs": {
            "polyphonic_review": polyphonic_path.name,
            "missing_review": missing_path.name,
        },
        "selected_interaction_ids": {
            "polyphonic_flagged": [
                record["interaction_id"] for record in polyphonic_sample
            ],
            "top10_missing": [record["interaction_id"] for record in missing_sample],
        },
        "automatic_labels_assigned": False,
    }
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def read_csv(path: Path, required_fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        present = reader.fieldnames or []
        missing = [field for field in required_fields if field not in present]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        return list(reader)


def validate_labels(
    rows: list[dict[str, str]],
    *,
    field: str,
    allowed: tuple[str, ...],
    path: Path,
) -> list[dict[str, str]]:
    normalized = []
    for row_number, row in enumerate(rows, 2):
        value = row[field].strip()
        if value and value not in allowed:
            raise ValueError(
                f"{path}:{row_number}: invalid {field} {value!r}; allowed values "
                f"are {', '.join(allowed)} or blank"
            )
        item = dict(row)
        item[field] = value
        normalized.append(item)
    return normalized


def distribution(
    rows: list[dict[str, str]], field: str, allowed: tuple[str, ...]
) -> dict[str, Any]:
    counts = Counter(row[field] for row in rows if row[field])
    labelled = sum(counts.values())
    return {
        "total_rows": len(rows),
        "labelled_rows": labelled,
        "blank_rows": len(rows) - labelled,
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


def summarize(output_dir: Path) -> dict[str, Any]:
    polyphonic_path = output_dir / "polyphonic_review_sample.csv"
    missing_path = output_dir / "missing_review_sample.csv"
    polyphonic = validate_labels(
        read_csv(polyphonic_path, POLYPHONIC_FIELDS),
        field="pinyin_judgement",
        allowed=PINYIN_VALUES,
        path=polyphonic_path,
    )
    missing = validate_labels(
        read_csv(missing_path, MISSING_FIELDS),
        field="missing_cause",
        allowed=MISSING_CAUSE_VALUES,
        path=missing_path,
    )
    return {
        "percentage_denominator": "manually labelled rows only",
        "polyphonic_pinyin_judgement": distribution(
            polyphonic, "pinyin_judgement", PINYIN_VALUES
        ),
        "missing_cause": distribution(missing, "missing_cause", MISSING_CAUSE_VALUES),
    }


def display_distribution(title: str, result: dict[str, Any]) -> list[str]:
    lines = [
        title,
        f"Labelled: {result['labelled_rows']}/{result['total_rows']} "
        f"(blank: {result['blank_rows']})",
    ]
    for value, metrics in result["values"].items():
        percentage = metrics["percentage_of_labelled"]
        formatted = "n/a" if percentage is None else f"{percentage:.2%}"
        lines.append(f"  {value}: {metrics['count']} ({formatted})")
    return lines


def display_summary(summary: dict[str, Any]) -> str:
    lines = ["Phase 4B.7 manual review summary", "Percentages use labelled rows only."]
    lines.extend(["", *display_distribution(
        "Polyphonic pronunciation judgement",
        summary["polyphonic_pinyin_judgement"],
    )])
    lines.extend(["", *display_distribution(
        "Top-10 missing cause",
        summary["missing_cause"],
    )])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    prepare_parser.add_argument(
        "--comparison", type=Path, default=DEFAULT_COMPARISON
    )
    prepare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare_parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    summarize_parser.add_argument("--json-output", type=Path)

    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare(
            args.input,
            args.comparison,
            args.output_dir,
            seed=args.seed,
            sample_size=args.sample_size,
        )
        print(
            f"Prepared {manifest['sample_size']} polyphonic and "
            f"{manifest['sample_size']} missing-target review rows."
        )
        print(f"Fixed seed: {manifest['fixed_seed']}")
        print(f"Output: {args.output_dir}")
        return

    summary = summarize(args.output_dir)
    print(display_summary(summary))
    if args.json_output:
        args.json_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
