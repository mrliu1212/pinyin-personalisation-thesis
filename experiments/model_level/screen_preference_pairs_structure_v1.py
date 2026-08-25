from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

AUTHOR = "Agent Phage"

FIT = Path(
    "/home/3160454/work/model-level-assets-20260822/"
    "clean3_train_fit_v1.jsonl"
)
VAL = Path(
    "/home/3160454/work/model-level-assets-20260822/"
    "clean3_train_val_v1.jsonl"
)

OUT = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "pair_screening_structure_v1"
)


def get_pinyin(row: dict) -> tuple[str, ...]:
    value = (
        row.get("segmented_pinyin")
        or row.get("pinyin_input")
        or row.get("pinyin")
        or row.get("typed_pinyin")
    )

    if isinstance(value, list):
        return tuple(str(x).strip().lower() for x in value if str(x).strip())

    if isinstance(value, str):
        return tuple(x for x in value.strip().lower().split() if x)

    raise ValueError(
        f"Cannot identify pinyin field for row {row.get('row_id')}"
    )


def get_target(row: dict) -> str:
    for key in ("target", "gold", "text"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError(
        f"Cannot identify target field for row {row.get('row_id')}"
    )


def load_rows(path: Path, partition: str) -> list[dict]:
    rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if str(row.get("source_split", "")).lower() == "test":
                raise RuntimeError("STOP: Test row detected")

            if row.get("author") != AUTHOR:
                continue

            # Keep the same short-target population used by model-level work.
            if row.get("target_type") not in (None, "short"):
                continue

            if row.get("condition") not in (None, "full_short"):
                continue

            row = dict(row)
            row["_partition"] = partition
            row["_pinyin"] = get_pinyin(row)
            row["_target"] = get_target(row)

            rows.append(row)

    return rows


rows = (
    load_rows(FIT, "train_fit")
    + load_rows(VAL, "train_val")
)

by_pinyin: dict[tuple[str, ...], list[dict]] = defaultdict(list)

for row in rows:
    pinyin = row["_pinyin"]
    target = row["_target"]

    # Core structural compatibility required by score_candidates().
    if len(target) != len(pinyin):
        continue

    by_pinyin[pinyin].append(row)


pair_rows = []

for pinyin, group in by_pinyin.items():
    target_counts = Counter(row["_target"] for row in group)

    if len(target_counts) < 2:
        continue

    # Make all unordered A/B pairs from real observed targets.
    targets = sorted(
        target_counts,
        key=lambda x: (-target_counts[x], x),
    )

    metadata = {}

    for target in targets:
        target_rows = [
            row for row in group
            if row["_target"] == target
        ]

        works = {
            str(
                row.get("work_id")
                or row.get("work")
                or row.get("source_work")
                or ""
            )
            for row in target_rows
        }
        works.discard("")

        dates = sorted(
            str(row.get("date"))
            for row in target_rows
            if row.get("date") is not None
        )

        contexts = {
            str(row.get("context", ""))
            for row in target_rows
        }

        metadata[target] = {
            "count": len(target_rows),
            "works": works,
            "n_works": len(works),
            "n_contexts": len(contexts),
            "first_date": dates[0] if dates else None,
            "last_date": dates[-1] if dates else None,
            "fit_count": sum(
                row["_partition"] == "train_fit"
                for row in target_rows
            ),
            "val_count": sum(
                row["_partition"] == "train_val"
                for row in target_rows
            ),
        }

    for i in range(len(targets)):
        for j in range(i + 1, len(targets)):
            a = targets[i]
            b = targets[j]

            ma = metadata[a]
            mb = metadata[b]

            pair_rows.append(
                {
                    "pinyin": " ".join(pinyin),
                    "syllables": len(pinyin),
                    "candidate_a": a,
                    "candidate_b": b,
                    "count_a": ma["count"],
                    "count_b": mb["count"],
                    "min_count": min(
                        ma["count"],
                        mb["count"],
                    ),
                    "total_count": (
                        ma["count"]
                        + mb["count"]
                    ),
                    "count_ratio": (
                        min(ma["count"], mb["count"])
                        / max(ma["count"], mb["count"])
                    ),
                    "works_a": ma["n_works"],
                    "works_b": mb["n_works"],
                    "contexts_a": ma["n_contexts"],
                    "contexts_b": mb["n_contexts"],
                    "fit_count_a": ma["fit_count"],
                    "fit_count_b": mb["fit_count"],
                    "val_count_a": ma["val_count"],
                    "val_count_b": mb["val_count"],
                    "first_date_a": ma["first_date"],
                    "last_date_a": ma["last_date"],
                    "first_date_b": mb["first_date"],
                    "last_date_b": mb["last_date"],
                }
            )


# Prefer pairs where both alternatives actually have meaningful evidence.
pair_rows.sort(
    key=lambda x: (
        -x["min_count"],
        -x["count_ratio"],
        -min(x["works_a"], x["works_b"]),
        -x["total_count"],
        x["pinyin"],
        x["candidate_a"],
        x["candidate_b"],
    )
)

OUT.mkdir(parents=True, exist_ok=True)

csv_path = OUT / "preference_pairs_structure_all_v1.csv"

if pair_rows:
    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(pair_rows[0]),
        )
        writer.writeheader()
        writer.writerows(pair_rows)

audit = {
    "schema_version": 1,
    "author": AUTHOR,
    "test_used": False,
    "source_rows": len(rows),
    "unique_pinyin": len(by_pinyin),
    "multi_target_pinyin": sum(
        len({
            row["_target"]
            for row in group
        }) >= 2
        for group in by_pinyin.values()
    ),
    "candidate_pairs": len(pair_rows),
    "pairs_min_count_ge_2": sum(
        row["min_count"] >= 2
        for row in pair_rows
    ),
    "pairs_min_count_ge_3": sum(
        row["min_count"] >= 3
        for row in pair_rows
    ),
    "pairs_min_count_ge_5": sum(
        row["min_count"] >= 5
        for row in pair_rows
    ),
    "output_csv": str(csv_path),
}

(OUT / "screening_audit_structure_v1.json").write_text(
    json.dumps(
        audit,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print("===== CONTROLLED PREFERENCE PAIR STRUCTURE SCREEN =====")
for key, value in audit.items():
    print(f"{key} = {value}")

print()
print("===== TOP 40 STRUCTURAL PAIRS =====")

for n, row in enumerate(pair_rows[:40], 1):
    print(
        f"{n:02d}. "
        f"{row['pinyin']} | "
        f"{row['candidate_a']} ({row['count_a']}) "
        f"vs "
        f"{row['candidate_b']} ({row['count_b']}) | "
        f"works={row['works_a']}/{row['works_b']} | "
        f"contexts={row['contexts_a']}/{row['contexts_b']}"
    )
