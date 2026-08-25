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

EVIDENCE = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "r4_ambiguous_evidence_audit_v1/"
    "evidence_availability.csv"
)

OUT = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "r4_control_budget_audit_v1"
)


# target controlled examples per pair per stage
TARGET_CONTROL_PER_PAIR = 20


def read_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_csv(path: Path):
    rows = {}

    with path.open(
        encoding="utf-8-sig"
    ) as f:
        for row in csv.DictReader(f):
            rows[row["pair_id"]] = row

    return rows


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    frozen = read_json(PAIRS)

    evidence = load_csv(EVIDENCE)

    results = []

    for pair in frozen["pairs"]:

        pid = pair["pair_id"]

        ev = evidence[pid]

        real_a = int(ev["total_a"])
        real_b = int(ev["total_b"])

        oversample_a = max(
            0,
            TARGET_CONTROL_PER_PAIR - real_a,
        )

        oversample_b = max(
            0,
            TARGET_CONTROL_PER_PAIR - real_b,
        )

        results.append(
            {
                "pair_id": pid,

                "pinyin":
                    pair["pinyin"],

                "candidate_a":
                    pair["candidate_a"],

                "candidate_b":
                    pair["candidate_b"],

                "real_a":
                    real_a,

                "real_b":
                    real_b,

                "target_control_each_stage":
                    TARGET_CONTROL_PER_PAIR,

                "oversample_a_needed":
                    oversample_a,

                "oversample_b_needed":
                    oversample_b,

                "training_status":
                    (
                        "REAL_ONLY"
                        if (
                            oversample_a == 0
                            and
                            oversample_b == 0
                        )
                        else
                        "NEEDS_OVERSAMPLING"
                    ),
            }
        )


    output = OUT / "control_budget_audit.csv"

    with output.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(results)


    summary = {
        "schema_version": 1,

        "experiment":
            "controlled_preference_dynamics_v1",

        "target_control_per_pair":
            TARGET_CONTROL_PER_PAIR,

        "pairs":
            len(results),

        "real_only_pairs":
            sum(
                x["training_status"]
                == "REAL_ONLY"
                for x in results
            ),

        "oversampling_required_pairs":
            sum(
                x["training_status"]
                == "NEEDS_OVERSAMPLING"
                for x in results
            ),

        "output":
            str(output),
    }


    (OUT / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    print(
        "===== R4 CONTROL BUDGET AUDIT ====="
    )

    for k, v in summary.items():
        print(
            f"{k} = {v}"
        )

    print()

    for row in results[:10]:
        print(
            row["pair_id"],
            row["candidate_a"],
            "vs",
            row["candidate_b"],
            "real_A=",
            row["real_a"],
            "real_B=",
            row["real_b"],
            "oversample_A=",
            row["oversample_a_needed"],
            "oversample_B=",
            row["oversample_b_needed"],
        )


if __name__ == "__main__":
    main()
