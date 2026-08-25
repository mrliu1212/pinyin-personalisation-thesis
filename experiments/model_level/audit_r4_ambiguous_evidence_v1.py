from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import defaultdict


PAIRS = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "frozen_pairs_v1/"
    "frozen_pairs.json"
)

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
    "r4_ambiguous_evidence_audit_v1"
)


def load_json(path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_rows(path):
    rows = []

    with path.open(
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)

                if row.get("author") == "Agent Phage":
                    rows.append(row)

    return rows


def get_pinyin(row):
    value = (
        row.get("segmented_pinyin")
        or row.get("pinyin_input")
        or row.get("pinyin")
    )

    if isinstance(value, list):
        return " ".join(
            str(x).lower()
            for x in value
        )

    return " ".join(
        str(value).lower().split()
    )


def get_target(row):
    for k in (
        "target",
        "gold",
        "text",
    ):
        if isinstance(row.get(k), str):
            return row[k]

    return None


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    frozen = load_json(PAIRS)

    rows = (
        load_rows(FIT)
        +
        load_rows(VAL)
    )

    index = defaultdict(list)

    for row in rows:
        index[
            (
                get_pinyin(row),
                get_target(row),
            )
        ].append(row)


    results = []

    for pair in frozen["pairs"]:

        a_rows = index[
            (
                pair["pinyin"],
                pair["candidate_a"],
            )
        ]

        b_rows = index[
            (
                pair["pinyin"],
                pair["candidate_b"],
            )
        ]

        train_a = max(
            0,
            len(a_rows) - 5
        )

        train_b = max(
            0,
            len(b_rows) - 5
        )

        eval_a = min(
            5,
            len(a_rows)
        )

        eval_b = min(
            5,
            len(b_rows)
        )

        status = (
            "READY"
            if (
                train_a >= 10
                and
                train_b >= 10
                and
                eval_a >= 5
                and
                eval_b >= 5
            )
            else
            "LIMITED"
        )

        results.append(
            {
                "pair_id": pair["pair_id"],
                "pinyin": pair["pinyin"],
                "candidate_a": pair["candidate_a"],
                "candidate_b": pair["candidate_b"],

                "total_a": len(a_rows),
                "total_b": len(b_rows),

                "train_a": train_a,
                "train_b": train_b,

                "eval_a": eval_a,
                "eval_b": eval_b,

                "status": status,
            }
        )


    out = OUT / "evidence_availability.csv"

    with out.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(results)


    print("===== R4 AMBIGUOUS AUDIT =====")

    print(
        "pairs =",
        len(results)
    )

    print(
        "ready =",
        sum(
            x["status"] == "READY"
            for x in results
        )
    )

    print(
        "limited =",
        sum(
            x["status"] != "READY"
            for x in results
        )
    )

    print()

    for x in results[:10]:
        print(
            x["pair_id"],
            x["candidate_a"],
            "vs",
            x["candidate_b"],
            "A=",
            x["total_a"],
            "B=",
            x["total_b"],
            x["status"],
        )


if __name__ == "__main__":
    main()
