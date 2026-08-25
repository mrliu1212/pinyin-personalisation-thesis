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
    "evidence_audit_v1/"
    "evidence_pairs.csv"
)

OUT = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "temporal_shift_audit_v1"
)


def read_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_evidence():
    rows = {}

    with EVIDENCE.open(
        encoding="utf-8-sig"
    ) as f:
        for row in csv.DictReader(f):
            rows[row["pair_id"]] = row

    return rows


def build_stream(count_a: int, count_b: int):
    total = count_a + count_b

    if total == 0:
        return []

    return {
        "gradual": [
            {
                "step": "S0",
                "a_ratio": 1.0,
                "b_ratio": 0.0,
            },
            {
                "step": "S1",
                "a_ratio": 0.8,
                "b_ratio": 0.2,
            },
            {
                "step": "S2",
                "a_ratio": 0.6,
                "b_ratio": 0.4,
            },
            {
                "step": "S3",
                "a_ratio": 0.4,
                "b_ratio": 0.6,
            },
            {
                "step": "S4",
                "a_ratio": 0.2,
                "b_ratio": 0.8,
            },
        ],
        "abrupt": [
            {
                "step": "S0",
                "sequence": "A",
            },
            {
                "step": "S1",
                "sequence": "B",
            },
        ],
        "temporary": [
            {
                "step": "S0",
                "sequence": "A",
            },
            {
                "step": "S1",
                "sequence": "B",
            },
            {
                "step": "S2",
                "sequence": "A",
            },
        ],
    }


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    frozen = read_json(PAIRS)
    evidence = load_evidence()

    results = []

    for pair in frozen["pairs"]:

        pid = pair["pair_id"]

        ev = evidence[pid]

        streams = build_stream(
            int(ev["count_a"]),
            int(ev["count_b"]),
        )

        results.append(
            {
                "pair_id": pid,
                "pinyin": pair["pinyin"],
                "candidate_a": pair["candidate_a"],
                "candidate_b": pair["candidate_b"],

                "available_a_examples": int(
                    ev["count_a"]
                ),

                "available_b_examples": int(
                    ev["count_b"]
                ),

                "streams": streams,
            }
        )


    output = {
        "schema_version": 1,
        "experiment": (
            "controlled_preference_dynamics_v1"
        ),
        "protocol": (
            "temporal_preference_drift_v1"
        ),
        "pairs": len(results),
        "test_used": False,
        "results": results,
    }


    path = (
        OUT /
        "temporal_shift_streams.json"
    )

    path.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    print(
        "===== R4 TEMPORAL SHIFT AUDIT ====="
    )

    print(
        "pairs =",
        len(results),
    )

    print(
        "output =",
        path,
    )

    for x in results[:5]:
        print(
            x["pair_id"],
            x["candidate_a"],
            "->",
            x["candidate_b"],
        )
        print(
            " gradual steps =",
            len(
                x["streams"]["gradual"]
            )
        )


if __name__ == "__main__":
    main()
