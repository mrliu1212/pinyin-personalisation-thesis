from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


AUTHOR = "Agent Phage"

CANDIDATES = [
    Path(
        "/home/3160454/work/model-level-assets-20260822/"
        "clean3_train_fit_v1.jsonl"
    ),
    Path(
        "/home/3160454/work/model-level-assets-20260822/"
        "clean3_train_val_v1.jsonl"
    ),
    Path(
        "results/model_level/fixed_dev1000_v1/"
        "agent_phage_dev1000.jsonl"
    ),
    Path(
        "/home/3160454/work/incoming-model-level-20260822/"
        "multi-m0/results/"
        "m0_standard_train_val_population_v1/"
        "multi_full_train_val_v1.jsonl"
    ),
    Path(
        "/home/3160454/work/incoming-model-level-20260822/"
        "multi-m0/train-val-v2-20260823/"
        "results/personalisation/multi_full_anatomy/"
        "m0_standard_train_val_population_v2/"
        "multi_full_train_val_v2.jsonl"
    ),
]

OUT = Path(
    "results/model_level/"
    "long_term_qualification_v1/"
    "temporal_coverage_audit_v2"
)


def author_of(x):
    return x.get("author", x.get("author_id"))


def scan(path: Path):
    rows = 0
    dates = []
    work_ids = set()
    work_indices = set()
    row_ids = set()
    chrono = []
    partitions = Counter()
    source_splits = Counter()

    first_rows = []
    last_rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            x = json.loads(line)

            if author_of(x) != AUTHOR:
                continue

            rows += 1

            if len(first_rows) < 3:
                first_rows.append(x)

            last_rows.append(x)
            if len(last_rows) > 3:
                last_rows.pop(0)

            if x.get("work_creation_date"):
                dates.append(str(x["work_creation_date"]))

            if x.get("work_id") is not None:
                work_ids.add(str(x["work_id"]))

            if x.get("work_chronological_index") is not None:
                work_indices.add(
                    int(x["work_chronological_index"])
                )

            if x.get("row_id") is not None:
                row_ids.add(str(x["row_id"]))

            if isinstance(x.get("chronological_position"), int):
                chrono.append(x["chronological_position"])

            if x.get("standardized_partition"):
                partitions[str(x["standardized_partition"])] += 1

            if x.get("source_split"):
                source_splits[str(x["source_split"])] += 1

    return {
        "path": str(path.resolve()),
        "rows": rows,
        "unique_row_ids": len(row_ids),
        "earliest_date": min(dates) if dates else "",
        "latest_date": max(dates) if dates else "",
        "unique_work_ids": len(work_ids),
        "unique_work_indices": len(work_indices),
        "min_work_index": min(work_indices) if work_indices else "",
        "max_work_index": max(work_indices) if work_indices else "",
        "min_chrono": min(chrono) if chrono else "",
        "max_chrono": max(chrono) if chrono else "",
        "partitions": dict(partitions),
        "source_splits": dict(source_splits),
        "row_ids": row_ids,
        "first_rows": first_rows,
        "last_rows": last_rows,
    }


def edge_view(x):
    return {
        "row_id": x.get("row_id"),
        "date": x.get("work_creation_date"),
        "work_idx": x.get("work_chronological_index"),
        "chrono": x.get("chronological_position"),
        "pinyin": x.get(
            "pinyin_segments",
            x.get("pinyin_input"),
        ),
        "gold": x.get(
            "gold",
            x.get("target_candidate", x.get("target")),
        ),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    results = []

    for path in CANDIDATES:
        if not path.exists():
            continue

        x = scan(path)
        results.append(x)

    print("============================================================")
    print(" AGENT-ONLY TEMPORAL COVERAGE V2")
    print("============================================================")

    for x in results:
        print()
        print("path       =", x["path"])
        print("rows       =", x["rows"])
        print(
            "dates      =",
            x["earliest_date"],
            "->",
            x["latest_date"],
        )
        print(
            "work idx   =",
            x["min_work_index"],
            "->",
            x["max_work_index"],
            f'({x["unique_work_indices"]} unique)',
        )
        print(
            "work ids   =",
            x["unique_work_ids"],
        )
        print(
            "chrono     =",
            x["min_chrono"],
            "->",
            x["max_chrono"],
        )
        print(
            "partitions =",
            json.dumps(
                x["partitions"],
                ensure_ascii=False,
                sort_keys=True,
            ),
        )

        print("first:")
        for row in x["first_rows"]:
            print(
                " ",
                json.dumps(
                    edge_view(row),
                    ensure_ascii=False,
                ),
            )

        print("last:")
        for row in x["last_rows"]:
            print(
                " ",
                json.dumps(
                    edge_view(row),
                    ensure_ascii=False,
                ),
            )

    by_name = {
        Path(x["path"]).name: x
        for x in results
    }

    fit = by_name.get("clean3_train_fit_v1.jsonl")
    val = by_name.get("clean3_train_val_v1.jsonl")

    print()
    print("============================================================")
    print(" TRAIN-FIT VS TRAIN-VAL TEMPORAL RELATION")
    print("============================================================")

    if fit and val:
        print(
            "Train-Fit:",
            fit["earliest_date"],
            "->",
            fit["latest_date"],
            "rows=",
            fit["rows"],
        )

        print(
            "Train-Val:",
            val["earliest_date"],
            "->",
            val["latest_date"],
            "rows=",
            val["rows"],
        )

        print(
            "row_id overlap =",
            len(fit["row_ids"] & val["row_ids"]),
        )

        print(
            "work-index overlap =",
            len(
                set(range(
                    fit["min_work_index"],
                    fit["max_work_index"] + 1,
                ))
                &
                set(range(
                    val["min_work_index"],
                    val["max_work_index"] + 1,
                ))
            ),
            "(range-level diagnostic only)",
        )

    print()
    print("============================================================")
    print(" DUPLICATE / DERIVED TRAIN-VAL CHECK")
    print("============================================================")

    if val:
        for x in results:
            if x is val:
                continue

            if "train_val" not in x["path"]:
                continue

            overlap = len(
                val["row_ids"] & x["row_ids"]
            )

            print()
            print(Path(x["path"]).name)
            print("rows =", x["rows"])
            print(
                "exact row_id overlap with clean3_train_val =",
                overlap,
            )
            print(
                "dates =",
                x["earliest_date"],
                "->",
                x["latest_date"],
            )

    rows_out = []

    for x in results:
        rows_out.append({
            "path": x["path"],
            "rows": x["rows"],
            "earliest_date": x["earliest_date"],
            "latest_date": x["latest_date"],
            "unique_work_ids": x["unique_work_ids"],
            "unique_work_indices": x["unique_work_indices"],
            "min_work_index": x["min_work_index"],
            "max_work_index": x["max_work_index"],
            "min_chrono": x["min_chrono"],
            "max_chrono": x["max_chrono"],
            "partitions": json.dumps(
                x["partitions"],
                ensure_ascii=False,
                sort_keys=True,
            ),
        })

    with (
        OUT / "agent_temporal_coverage_v2.csv"
    ).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows_out[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows_out)

    summary = {
        "schema_version": 2,
        "status": "AUDIT_ONLY_NOT_FROZEN",
        "author": AUTHOR,
        "important_fix": (
            "All dates/work/chronology statistics are "
            "computed after filtering to Agent Phage."
        ),
        "test_content_used": False,
        "datasets": rows_out,
    }

    (
        OUT / "audit_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("AUDIT COMPLETE -- NOTHING FROZEN")


if __name__ == "__main__":
    main()
