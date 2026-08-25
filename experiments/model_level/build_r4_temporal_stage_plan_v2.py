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
    "r4_temporal_stage_plan_v1"
)

AUTHOR = "Agent Phage"

STAGES = {
    "stage_01_A": "A",
    "stage_02_A": "A",
    "stage_03_A": "A",
    "stage_04_B": "B",
    "stage_05_B": "B",
    "stage_06_B": "B",
    "stage_07_A": "A",
    "stage_08_A": "A",
}

TOTAL_ROWS_PER_STAGE = 5000
CONTROL_ROWS_PER_STAGE = 500
CONTROL_PER_PAIR = 20


def read_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_rows(path: Path):
    rows = []

    with path.open(
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)

                if row.get("author") == AUTHOR:
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
    for key in (
        "target",
        "gold",
        "text",
    ):
        if isinstance(row.get(key), str):
            return row[key]

    return None


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    frozen = read_json(PAIRS)

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


    pair_plan = []

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

        pair_plan.append(
            {
                "pair_id": pair["pair_id"],
                "pinyin": pair["pinyin"],

                "candidate_a":
                    pair["candidate_a"],

                "candidate_b":
                    pair["candidate_b"],

                "real_a":
                    len(a_rows),

                "real_b":
                    len(b_rows),

                "required_per_stage":
                    CONTROL_PER_PAIR,

                "a_oversample_needed":
                    max(
                        0,
                        CONTROL_PER_PAIR
                        -
                        len(a_rows)
                    ),

                "b_oversample_needed":
                    max(
                        0,
                        CONTROL_PER_PAIR
                        -
                        len(b_rows)
                    ),

                "trainable":
                    True,
            }
        )


    stages = []

    for stage, preference in STAGES.items():

        stages.append(
            {
                "stage": stage,

                "total_rows":
                    TOTAL_ROWS_PER_STAGE,

                "background_rows":
                    (
                        TOTAL_ROWS_PER_STAGE
                        -
                        CONTROL_ROWS_PER_STAGE
                    ),

                "control_rows":
                    CONTROL_ROWS_PER_STAGE,

                "preference_direction":
                    preference,

                "control_policy":
                    (
                        "20 examples per pair; "
                        "real rows preferred; "
                        "oversample existing rows "
                        "when insufficient"
                    ),
            }
        )


    result = {

        "schema_version": 1,

        "experiment":
            "controlled_preference_dynamics_v1",

        "protocol":
            "three_phase_temporal_ABA",

        "stages":
            stages,

        "pairs":
            pair_plan,

        "constraints":
            {
                "stage_rows":
                    TOTAL_ROWS_PER_STAGE,

                "control_rows":
                    CONTROL_ROWS_PER_STAGE,

                "pairs":
                    len(pair_plan),
            },

        "test_used":
            False,
    }


    out = (
        OUT /
        "r4_stage_plan.json"
    )

    out.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    csv_out = (
        OUT /
        "pair_oversampling_plan.csv"
    )

    with csv_out.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                pair_plan[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(pair_plan)


    print(
        "===== R4 STAGE PLAN ====="
    )

    print(
        "pairs =",
        len(pair_plan),
    )

    print(
        "stages =",
        len(stages),
    )

    print(
        "rows/stage =",
        TOTAL_ROWS_PER_STAGE,
    )

    print(
        "control/stage =",
        CONTROL_ROWS_PER_STAGE,
    )

    print(
        "output =",
        out,
    )


if __name__ == "__main__":
    main()
