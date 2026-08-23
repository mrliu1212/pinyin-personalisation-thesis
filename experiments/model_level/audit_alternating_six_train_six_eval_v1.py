from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


AUTHOR = "Agent Phage"
N_BLOCKS = 12
BATCH_SIZE = 8

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

OUT = Path(
    "results/model_level/"
    "long_term_qualification_v1/"
    "alternating_six_train_six_eval_candidate_v1"
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


def load_rows(path: Path, partition: str) -> list[dict[str, Any]]:
    rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if row.get("author") != AUTHOR:
                continue

            row = dict(row)
            row["_partition"] = partition
            rows.append(row)

    return rows


def compatible(tokenizer, target: str) -> bool:
    for char in target:
        token_id = tokenizer.convert_tokens_to_ids(char)
        token = tokenizer.convert_ids_to_tokens(token_id)

        if token_id == tokenizer.unk_token_id or token != char:
            return False

    return True


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT,
        local_files_only=True,
    )

    rows = (
        load_rows(TRAIN_FIT, "train_fit")
        + load_rows(TRAIN_VAL, "train_val")
    )
    rows.sort(key=chrono_key)

    effective = [
        row
        for row in rows
        if compatible(tokenizer, target_of(row))
    ]

    assert len(rows) == 69667
    assert len(effective) == 69664

    total = len(effective)

    # Equal chronological twelfths.
    blocks = []

    for i in range(N_BLOCKS):
        start = (i * total) // N_BLOCKS
        end = ((i + 1) * total) // N_BLOCKS

        selected = effective[start:end]

        # Odd-numbered block = training.
        # Even-numbered block = future evaluation.
        role = "train" if (i + 1) % 2 == 1 else "eval"

        blocks.append(
            {
                "block": i + 1,
                "role": role,
                "rows": selected,
            }
        )

    assert sum(len(x["rows"]) for x in blocks) == total

    all_ids = []
    summary_rows = []
    assignments = []

    for block in blocks:
        selected = block["rows"]

        all_ids.extend(str(r["row_id"]) for r in selected)

        works = sorted(
            {
                int(r["work_chronological_index"])
                for r in selected
            }
        )

        partitions = Counter(
            r["_partition"] for r in selected
        )

        steps = (
            math.ceil(len(selected) / BATCH_SIZE)
            if block["role"] == "train"
            else 0
        )

        summary_rows.append(
            {
                "block": block["block"],
                "role": block["role"],
                "rows": len(selected),
                "optimizer_steps": steps,
                "first_work": works[0],
                "last_work": works[-1],
                "n_works": len(works),
                "first_date": selected[0]["work_creation_date"],
                "last_date": selected[-1]["work_creation_date"],
                "train_fit_rows": partitions["train_fit"],
                "train_val_rows": partitions["train_val"],
            }
        )

        for row in selected:
            assignments.append(
                {
                    "block": block["block"],
                    "role": block["role"],
                    "row_id": row["row_id"],
                    "partition": row["_partition"],
                    "work_index": row["work_chronological_index"],
                    "date": row["work_creation_date"],
                    "chronological_position": row[
                        "chronological_position"
                    ],
                }
            )

    assert len(all_ids) == total
    assert len(set(all_ids)) == total

    train_rows = sum(
        len(x["rows"])
        for x in blocks
        if x["role"] == "train"
    )

    eval_rows = sum(
        len(x["rows"])
        for x in blocks
        if x["role"] == "eval"
    )

    train_steps = sum(
        math.ceil(len(x["rows"]) / BATCH_SIZE)
        for x in blocks
        if x["role"] == "train"
    )

    # Strict chronology.
    for left, right in zip(blocks, blocks[1:]):
        assert chrono_key(left["rows"][-1]) < chrono_key(
            right["rows"][0]
        )

    result = {
        "schema_version": 1,
        "status": "AUDIT_ONLY_NOT_FROZEN",
        "author": AUTHOR,
        "design": {
            "blocks": 12,
            "policy": (
                "equal chronological twelfths; odd blocks train, "
                "even blocks held out for future evaluation"
            ),
            "train_blocks": [1, 3, 5, 7, 9, 11],
            "eval_blocks": [2, 4, 6, 8, 10, 12],
            "batch_size": BATCH_SIZE,
        },
        "population": {
            "effective_rows": total,
            "train_rows": train_rows,
            "eval_rows": eval_rows,
            "train_fraction": train_rows / total,
            "eval_fraction": eval_rows / total,
            "total_optimizer_steps": train_steps,
        },
        "blocks": summary_rows,
        "test_used": False,
        "supersedes_candidate_design": (
            "twelve_update_six_probe_v1 "
            "for future longitudinal execution"
        ),
    }

    (OUT / "audit_summary.json").write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    write_csv(
        OUT / "block_summary.csv",
        summary_rows,
    )

    write_csv(
        OUT / "row_assignments.csv",
        assignments,
    )

    print("============================================================")
    print(" ALTERNATING 6 TRAIN / 6 EVAL AUDIT")
    print("============================================================")
    print()
    print("effective rows =", total)
    print("train rows     =", train_rows)
    print("eval rows      =", eval_rows)
    print("train fraction =", f"{train_rows / total:.4%}")
    print("eval fraction  =", f"{eval_rows / total:.4%}")
    print("train steps    =", train_steps)
    print()

    print(
        f"{'BLOCK':<7}"
        f"{'ROLE':<8}"
        f"{'ROWS':>8}"
        f"{'STEPS':>8}"
        f"{'WORKS':>12}"
        f"{'FIT':>8}"
        f"{'VAL':>8}"
        f"  DATES"
    )

    for x in summary_rows:
        work_range = f"{x['first_work']}-{x['last_work']}"

        print(
            f"B{x['block']:<6}"
            f"{x['role']:<8}"
            f"{x['rows']:>8}"
            f"{x['optimizer_steps']:>8}"
            f"{work_range:>12}"
            f"{x['train_fit_rows']:>8}"
            f"{x['train_val_rows']:>8}"
            f"  {x['first_date']} -> {x['last_date']}"
        )

    print()
    print("STATUS = AUDIT_ONLY_NOT_FROZEN")
    print("TEST USED = false")


if __name__ == "__main__":
    main()
