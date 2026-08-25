from __future__ import annotations

import json
import random
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
    "r4_aba_stage_data_v1"
)

AUTHOR = "Agent Phage"

SEED = 20260824

STAGES = {
    "stage_01_A": "A",
    "stage_02_B": "B",
    "stage_03_A": "A",
}

TOTAL_ROWS = 5000
CONTROL_ROWS = 600
BACKGROUND_ROWS = 4400
CONTROL_PER_PAIR = 20


def read_json(path):
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

    random.seed(SEED)

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


    used_ids = set()

    background_pool = rows.copy()

    stage_summary = []


    for stage, direction in STAGES.items():

        stage_rows = []

        background = random.sample(
            background_pool,
            BACKGROUND_ROWS,
        )

        stage_rows.extend(
            background
        )


        for pair in frozen["pairs"]:

            candidate = (
                pair["candidate_a"]
                if direction == "A"
                else
                pair["candidate_b"]
            )

            examples = index[
                (
                    pair["pinyin"],
                    candidate,
                )
            ]

            if not examples:
                continue

            for i in range(CONTROL_PER_PAIR):

                stage_rows.append(
                    examples[
                        i % len(examples)
                    ]
                )


        random.shuffle(
            stage_rows
        )

        if len(stage_rows) != TOTAL_ROWS:
            raise RuntimeError(
                f"{stage}: expected {TOTAL_ROWS}, got {len(stage_rows)}"
            )


        output = (
            OUT /
            f"{stage}.jsonl"
        )

        with output.open(
            "w",
            encoding="utf-8",
        ) as f:

            for row in stage_rows:
                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )


        stage_summary.append(
            {
                "stage": stage,
                "direction": direction,
                "rows": len(stage_rows),
                "background_rows": BACKGROUND_ROWS,
                "control_rows": CONTROL_ROWS,
            }
        )


    (OUT / "stage_plan.json").write_text(
        json.dumps(
            {
                "experiment":
                    "controlled_preference_dynamics_r4_aba_v1",
                "seed": SEED,
                "stages": stage_summary,
                "total_training_rows":
                    len(STAGES) * TOTAL_ROWS,
                "test_used": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    print(
        "===== R4 ABA STAGE DATA ====="
    )

    for x in stage_summary:
        print(
            x
        )


if __name__ == "__main__":
    main()
