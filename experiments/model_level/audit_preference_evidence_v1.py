from __future__ import annotations

import csv
import json
from collections import defaultdict
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

PAIRS = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "frozen_pairs_v1/"
    "frozen_pairs.json"
)

OUT = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "evidence_audit_v1"
)


def load_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def get_pinyin(row: dict) -> str:
    value = (
        row.get("segmented_pinyin")
        or row.get("pinyin_input")
        or row.get("pinyin")
        or row.get("typed_pinyin")
    )

    if isinstance(value, list):
        return " ".join(
            str(x).lower()
            for x in value
        )

    return " ".join(
        str(value).lower().split()
    )


def get_target(row: dict) -> str:
    for key in (
        "target",
        "gold",
        "text",
    ):
        value = row.get(key)
        if isinstance(value, str):
            return value

    raise RuntimeError(
        "Missing target"
    )


def load_rows(path: Path):
    rows = []

    with path.open(
        encoding="utf-8"
    ) as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if row.get("author") != AUTHOR:
                continue

            if str(
                row.get("source_split", "")
            ).lower() == "test":
                raise RuntimeError(
                    "Test data detected"
                )

            rows.append(row)

    return rows


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    frozen = load_json(PAIRS)

    pairs = frozen["pairs"]

    rows = (
        load_rows(FIT)
        +
        load_rows(VAL)
    )

    index = defaultdict(list)

    for row in rows:
        key = (
            get_pinyin(row),
            get_target(row),
        )

        index[key].append(row)


    results = []

    for pair in pairs:

        pinyin = pair["pinyin"]
        a = pair["candidate_a"]
        b = pair["candidate_b"]

        rows_a = index.get(
            (pinyin, a),
            []
        )

        rows_b = index.get(
            (pinyin, b),
            []
        )

        works_a = {
            str(
                r.get(
                    "work_id",
                    r.get("work", "")
                )
            )
            for r in rows_a
        }

        works_b = {
            str(
                r.get(
                    "work_id",
                    r.get("work", "")
                )
            )
            for r in rows_b
        }

        results.append(
            {
                "pair_id":
                    pair["pair_id"],

                "pinyin":
                    pinyin,

                "candidate_a":
                    a,

                "candidate_b":
                    b,

                "count_a":
                    len(rows_a),

                "count_b":
                    len(rows_b),

                "works_a":
                    len(works_a),

                "works_b":
                    len(works_b),

                "usable":
                    (
                        len(rows_a) >= 5
                        and
                        len(rows_b) >= 5
                    ),

                "status":
                    (
                        "READY"
                        if (
                            len(rows_a) >= 5
                            and
                            len(rows_b) >= 5
                        )
                        else
                        "INSUFFICIENT"
                    ),
            }
        )


    csv_path = (
        OUT /
        "evidence_pairs.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                results[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(results)


    audit = {

        "schema_version": 1,

        "experiment":
            "controlled_preference_dynamics_v1",

        "author":
            AUTHOR,

        "test_used":
            False,

        "frozen_pairs":
            len(pairs),

        "ready_pairs":
            sum(
                x["usable"]
                for x in results
            ),

        "insufficient_pairs":
            sum(
                not x["usable"]
                for x in results
            ),

        "output":
            str(csv_path),
    }


    (OUT /
     "evidence_audit.json").write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    print(
        "===== R4 EVIDENCE AUDIT ====="
    )

    for k, v in audit.items():
        print(
            f"{k} = {v}"
        )


    print()

    for row in results[:10]:
        print(
            row["pair_id"],
            row["pinyin"],
            row["candidate_a"],
            "vs",
            row["candidate_b"],
            "A=",
            row["count_a"],
            "B=",
            row["count_b"],
            row["status"],
        )


if __name__ == "__main__":
    main()
