from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOTS = [
    Path("/home/3160454/work/model-level-assets-20260822"),
    Path("/home/3160454/work/incoming-model-level-20260822"),
    Path("/home/3160454/work/thesis-model-level"),
]

OUT = Path(
    "results/model_level/"
    "long_term_qualification_v1/"
    "temporal_coverage_audit_v1"
)

# Strictly do not open files whose path looks like a Test artifact.
TEST_PATTERN = re.compile(
    r"(^|[/_.-])test([/_.-]|$)",
    re.IGNORECASE,
)

# Evaluation prediction dumps are not source-history candidates.
PREDICTION_PATTERN = re.compile(
    r"(prediction|predictions)",
    re.IGNORECASE,
)


def is_test_path(path: Path) -> bool:
    return TEST_PATTERN.search(str(path)) is not None


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: list[str] = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def first_json_row(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    x = json.loads(line)
                    return x if isinstance(x, dict) else None
    except Exception:
        return None

    return None


def classify_candidate(path: Path, first: dict[str, Any]) -> str:
    name = str(path).lower()

    if "train_fit" in name:
        return "train_fit"

    if "train_val" in name:
        return "train_val"

    if "dev" in name:
        return "dev_or_history"

    if "history" in name:
        return "history"

    if "source" in name:
        return "source_like"

    if "work_creation_date" in first:
        return "dated_other"

    return "other"


def scan_dated_jsonl(path: Path) -> dict[str, Any]:
    n = 0
    bad_json = 0

    authors = Counter()
    partitions = Counter()
    source_splits = Counter()

    dates: list[str] = []
    work_ids = set()
    work_indices = set()
    row_ids = set()

    min_chrono = None
    max_chrono = None

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                x = json.loads(line)
            except Exception:
                bad_json += 1
                continue

            if not isinstance(x, dict):
                continue

            n += 1

            author = x.get("author", x.get("author_id"))
            if author is not None:
                authors[str(author)] += 1

            partition = x.get("standardized_partition")
            if partition is not None:
                partitions[str(partition)] += 1

            split = x.get("source_split")
            if split is not None:
                source_splits[str(split)] += 1

            date = x.get("work_creation_date")
            if date:
                dates.append(str(date))

            work_id = x.get("work_id")
            if work_id is not None:
                work_ids.add(str(work_id))

            work_idx = x.get("work_chronological_index")
            if work_idx is not None:
                work_indices.add(str(work_idx))

            row_id = x.get("row_id")
            if row_id is not None:
                row_ids.add(str(row_id))

            chrono = x.get("chronological_position")
            if isinstance(chrono, int):
                if min_chrono is None or chrono < min_chrono:
                    min_chrono = chrono
                if max_chrono is None or chrono > max_chrono:
                    max_chrono = chrono

    return {
        "rows": n,
        "bad_json_lines": bad_json,
        "unique_row_ids": len(row_ids),
        "unique_work_ids": len(work_ids),
        "unique_work_indices": len(work_indices),
        "earliest_date": min(dates) if dates else "",
        "latest_date": max(dates) if dates else "",
        "min_chronological_position": (
            min_chrono if min_chrono is not None else ""
        ),
        "max_chronological_position": (
            max_chrono if max_chrono is not None else ""
        ),
        "authors": json.dumps(
            dict(authors),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "standardized_partitions": json.dumps(
            dict(partitions),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "source_splits": json.dumps(
            dict(source_splits),
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    all_jsonl: set[Path] = set()

    for root in ROOTS:
        if not root.exists():
            continue

        for path in root.rglob("*.jsonl"):
            if path.is_file():
                all_jsonl.add(path.resolve())

    inventory_rows = []
    scanned_rows = []
    sealed_test_rows = []

    for path in sorted(all_jsonl):
        size = path.stat().st_size
        test_like = is_test_path(path)

        inventory_rows.append(
            {
                "path": str(path),
                "size_bytes": size,
                "size_mb": size / (1024 * 1024),
                "sealed_test": test_like,
            }
        )

        if test_like:
            sealed_test_rows.append(
                {
                    "path": str(path),
                    "size_bytes": size,
                    "size_mb": size / (1024 * 1024),
                    "content_read": False,
                    "reason": "Test path sealed by audit policy",
                }
            )
            continue

        if PREDICTION_PATTERN.search(path.name):
            continue

        first = first_json_row(path)
        if first is None:
            continue

        # Only fully scan data that actually carries chronology.
        if "work_creation_date" not in first:
            continue

        row = {
            "path": str(path),
            "classification": classify_candidate(path, first),
            "size_mb": size / (1024 * 1024),
        }

        row.update(scan_dated_jsonl(path))
        scanned_rows.append(row)

    write_csv(
        OUT / "jsonl_inventory.csv",
        inventory_rows,
    )

    write_csv(
        OUT / "sealed_test_inventory.csv",
        sealed_test_rows,
    )

    write_csv(
        OUT / "dated_non_test_datasets.csv",
        scanned_rows,
    )

    # Focused author-level summaries.
    author_summary = []

    for row in scanned_rows:
        authors = json.loads(row["authors"])

        for author, count in authors.items():
            author_summary.append(
                {
                    "path": row["path"],
                    "classification": row["classification"],
                    "author": author,
                    "author_rows": count,
                    "dataset_rows": row["rows"],
                    "earliest_date": row["earliest_date"],
                    "latest_date": row["latest_date"],
                    "unique_work_ids": row["unique_work_ids"],
                    "unique_work_indices": row["unique_work_indices"],
                    "standardized_partitions": (
                        row["standardized_partitions"]
                    ),
                    "source_splits": row["source_splits"],
                }
            )

    write_csv(
        OUT / "author_temporal_coverage.csv",
        author_summary,
    )

    agent_rows = [
        row
        for row in author_summary
        if row["author"] == "Agent Phage"
    ]

    summary = {
        "schema_version": 1,
        "status": "AUDIT_ONLY_NOT_FROZEN",
        "policy": {
            "test_content_read": False,
            "test_paths_only_inventory": True,
            "non_test_dated_jsonl_scanned": True,
            "predictions_skipped": True,
        },
        "roots": [str(x) for x in ROOTS],
        "jsonl_files_found": len(all_jsonl),
        "sealed_test_files_found": len(sealed_test_rows),
        "dated_non_test_files_scanned": len(scanned_rows),
        "agent_dated_sources_found": len(agent_rows),
    }

    dump_json(
        OUT / "audit_summary.json",
        summary,
    )

    print("============================================================")
    print(" TEMPORAL COVERAGE AUDIT")
    print("============================================================")
    print()
    print("JSONL files found       =", len(all_jsonl))
    print("sealed Test files       =", len(sealed_test_rows))
    print("dated non-Test datasets =", len(scanned_rows))
    print("Agent dated sources     =", len(agent_rows))

    print()
    print("===== AGENT PHAGE TEMPORAL COVERAGE =====")

    if not agent_rows:
        print("No dated Agent Phage datasets found.")
    else:
        for row in sorted(
            agent_rows,
            key=lambda x: (
                x["earliest_date"],
                x["latest_date"],
                x["path"],
            ),
        ):
            print()
            print("classification =", row["classification"])
            print("rows           =", row["author_rows"])
            print(
                "dates          =",
                row["earliest_date"],
                "->",
                row["latest_date"],
            )
            print(
                "works(ids/idx) =",
                row["unique_work_ids"],
                "/",
                row["unique_work_indices"],
            )
            print(
                "partitions     =",
                row["standardized_partitions"],
            )
            print(
                "source_splits  =",
                row["source_splits"],
            )
            print("path           =", row["path"])

    print()
    print("===== SEALED TEST INVENTORY =====")

    if not sealed_test_rows:
        print("No Test-like JSONL paths found.")
    else:
        for row in sealed_test_rows:
            print(
                f'{row["size_mb"]:.2f} MB  '
                f'{row["path"]}'
            )

    print()
    print("===== POLICY CHECK =====")
    print("Test content read: NO")
    print("Test dates inspected: NO")
    print("Test rows inspected: NO")
    print()
    print("AUDIT COMPLETE -- NOTHING FROZEN")


if __name__ == "__main__":
    main()
