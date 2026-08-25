from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


AUTHOR = "Agent Phage"
N_UPDATES = 12
PROBE_ROWS = 500

TRAIN_FIT = Path(
    "/home/3160454/work/model-level-assets-20260822/"
    "clean3_train_fit_v1.jsonl"
)

TRAIN_VAL = Path(
    "/home/3160454/work/model-level-assets-20260822/"
    "clean3_train_val_v1.jsonl"
)

CHECKPOINT = Path(
    "/home/3160454/work/model-level-assets-20260822/"
    "pinyingpt2-concat"
)


def chrono_key(row: dict[str, Any]) -> tuple:
    return (
        int(row["work_chronological_index"]),
        int(row["chronological_position"]),
        int(row["source_position_start"]),
        int(row["source_position_end"]),
        str(row["row_id"]),
    )


def target_of(row: dict[str, Any]) -> str:
    return str(row.get("gold", row.get("target", "")))


def load(path: Path, partition: str) -> list[dict[str, Any]]:
    rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            x = json.loads(line)

            if x.get("author") != AUTHOR:
                continue

            x = dict(x)
            x["_partition"] = partition
            rows.append(x)

    return rows


def compatible(tokenizer, target: str) -> bool:
    return all(
        tokenizer.convert_tokens_to_ids(c) != tokenizer.unk_token_id
        for c in target
    )


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT,
        local_files_only=True,
    )

    rows = (
        load(TRAIN_FIT, "train_fit")
        + load(TRAIN_VAL, "train_val")
    )
    rows.sort(key=chrono_key)

    rows = [
        r
        for r in rows
        if compatible(tokenizer, target_of(r))
    ]

    assert len(rows) == 69664

    # Hypothetical equal update positions WITHOUT reserving probes.
    # This audit only asks: after each equally spaced learning position,
    # what would the immediate next 500 chronological rows look like?
    update_target = (len(rows) - 3000) / N_UPDATES

    print("============================================================")
    print(" POTENTIAL 500-ROW FUTURE PROBES")
    print("============================================================")
    print()
    print(
        f"{'AFTER':<8}"
        f"{'POS':>8}"
        f"{'WORKS':>12}"
        f"{'NWORK':>8}"
        f"{'DOM%':>8}"
        f"{'FIT':>8}"
        f"{'VAL':>8}"
        f"  DATES"
    )

    candidates = []

    for u in range(1, N_UPDATES + 1):
        pos = round(u * update_target)

        if pos + PROBE_ROWS > len(rows):
            continue

        probe = rows[pos:pos + PROBE_ROWS]

        works = Counter(
            int(r["work_chronological_index"])
            for r in probe
        )

        partitions = Counter(
            r["_partition"]
            for r in probe
        )

        dominant_work, dominant_n = works.most_common(1)[0]

        first_work = int(
            probe[0]["work_chronological_index"]
        )
        last_work = int(
            probe[-1]["work_chronological_index"]
        )

        dominant_share = dominant_n / PROBE_ROWS

        row = {
            "after_update": u,
            "position": pos,
            "first_work": first_work,
            "last_work": last_work,
            "n_works": len(works),
            "dominant_work": dominant_work,
            "dominant_rows": dominant_n,
            "dominant_share": dominant_share,
            "train_fit": partitions["train_fit"],
            "train_val": partitions["train_val"],
            "first_date": probe[0]["work_creation_date"],
            "last_date": probe[-1]["work_creation_date"],
        }

        candidates.append(row)

        work_range = f"{first_work}-{last_work}"

        print(
            f"U{u:<7}"
            f"{pos:>8}"
            f"{work_range:>12}"
            f"{len(works):>8}"
            f"{100 * dominant_share:>7.1f}%"
            f"{partitions['train_fit']:>8}"
            f"{partitions['train_val']:>8}"
            f"  "
            f"{probe[0]['work_creation_date']} -> "
            f"{probe[-1]['work_creation_date']}"
        )

    print()
    print("===== MOST WORK-DIVERSE POSITIONS =====")

    for x in sorted(
        candidates,
        key=lambda z: (
            z["dominant_share"],
            -z["n_works"],
            z["after_update"],
        ),
    ):
        print(
            f"after U{x['after_update']}: "
            f"works={x['first_work']}-{x['last_work']} "
            f"n_works={x['n_works']} "
            f"dominant_work={x['dominant_work']} "
            f"dominant_share={x['dominant_share']:.3f}"
        )


if __name__ == "__main__":
    main()
