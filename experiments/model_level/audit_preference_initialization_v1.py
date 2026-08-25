from __future__ import annotations

import csv
import json
from pathlib import Path


PAIRS = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "frozen_pairs_v1/"
    "frozen_pairs.json"
)

INIT_DIR = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "preference_init_smoke_v1"
)

OUT = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "preference_init_audit_v1"
)


def read_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    frozen = read_json(PAIRS)

    rows = []

    for pair in frozen["pairs"]:

        ckpt = (
            INIT_DIR
            /
            f"{pair['pair_id']}.safetensors"
        )

        rows.append(
            {
                "pair_id": pair["pair_id"],
                "pinyin": pair["pinyin"],
                "candidate_a": pair["candidate_a"],
                "candidate_b": pair["candidate_b"],

                "generic_margin": (
                    pair[
                        "generic_margin_a_minus_b"
                    ]
                ),

                "checkpoint_exists": (
                    ckpt.is_file()
                ),

                "status": (
                    "READY"
                    if ckpt.is_file()
                    else "MISSING_CHECKPOINT"
                ),

                "checkpoint": str(ckpt),
            }
        )

    csv_path = OUT / "preference_init_audit.csv"

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


    summary = {
        "schema_version": 1,
        "experiment": (
            "controlled_preference_dynamics_v1"
        ),
        "pairs": len(rows),

        "ready_pairs": sum(
            x["status"] == "READY"
            for x in rows
        ),

        "missing_pairs": sum(
            x["status"] != "READY"
            for x in rows
        ),

        "output": str(csv_path),
    }

    (OUT / "audit_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


    print(
        "===== R4 INITIALIZATION AUDIT ====="
    )

    for k, v in summary.items():
        print(
            f"{k} = {v}"
        )


if __name__ == "__main__":
    main()
