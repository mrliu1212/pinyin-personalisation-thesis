from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


SRC = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "pair_screening_generic_v1/"
    "preference_pairs_generic_shortlist_v1.csv"
)

OUT = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "frozen_pairs_v1"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    with SRC.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    rows.sort(
        key=lambda x: (
            float(x["abs_mean_margin"]),
            x["pinyin"],
            x["candidate_a"],
            x["candidate_b"],
        )
    )

    frozen = []

    for idx, row in enumerate(rows, 1):
        frozen.append(
            {
                "pair_id": f"R4_PAIR_{idx:03d}",
                "pinyin": row["pinyin"],
                "candidate_a": row["candidate_a"],
                "candidate_b": row["candidate_b"],
                "source_pair_id": row["pair_id"],

                "count_a": int(row["count_a"]),
                "count_b": int(row["count_b"]),

                "generic_margin_a_minus_b": float(
                    row["mean_margin_a_minus_b"]
                ),

                "generic_median_margin": float(
                    row["median_margin_a_minus_b"]
                ),

                "generic_std_margin": float(
                    row["std_margin"]
                ),

                "a_win_rate": float(
                    row["a_win_rate"]
                ),

                "valid_contexts": int(
                    row["valid_contexts"]
                ),

                "status": "FROZEN_FOR_R4",
            }
        )

    payload = {
        "schema_version": 1,
        "experiment": (
            "controlled_preference_dynamics_v1"
        ),
        "author": "Agent Phage",

        "selection_source": str(SRC),

        "selection_source_sha256": sha256(SRC),

        "pair_count": len(frozen),

        "selection_policy": (
            "All pairs from generic shortlist; "
            "sorted by absolute generic margin; "
            "no manual filtering"
        ),

        "test_used": False,

        "pairs": frozen,
    }

    json_path = OUT / "frozen_pairs.json"

    json_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = OUT / "frozen_pairs.csv"

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(frozen[0].keys())
        )
        writer.writeheader()
        writer.writerows(frozen)

    audit = {
        "schema_version": 1,
        "status": "FROZEN",

        "input_sha256": sha256(SRC),

        "pairs": len(frozen),

        "test_used": False,

        "outputs": [
            str(json_path),
            str(csv_path),
        ],
    }

    (OUT / "freeze_audit.json").write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("===== R4 FROZEN PAIRS =====")
    print("pairs =", len(frozen))
    print("source_sha256 =", sha256(SRC))
    print("status = FROZEN")


if __name__ == "__main__":
    main()
