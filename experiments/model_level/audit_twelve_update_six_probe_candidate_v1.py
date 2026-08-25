from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


AUTHOR = "Agent Phage"
N_UPDATES = 12
N_PROBES = 6
PROBE_ROWS = 500
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
    "twelve_update_six_probe_candidate_v1"
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
            row["_source_partition"] = partition
            rows.append(row)

    return rows


def tokenizer_compatible(tokenizer, target: str) -> bool:
    for char in target:
        token_id = tokenizer.convert_tokens_to_ids(char)
        if token_id == tokenizer.unk_token_id:
            return False
    return True


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

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

    compatible = [
        row
        for row in rows
        if tokenizer_compatible(
            tokenizer,
            target_of(row),
        )
    ]

    assert len(rows) == 69667, len(rows)
    assert len(compatible) == 69664, len(compatible)

    total = len(compatible)
    total_probe_rows = N_PROBES * PROBE_ROWS
    total_update_rows = total - total_probe_rows

    assert total_probe_rows == 3000
    assert total_update_rows == 66664

    # Distribute update rows as evenly as possible over 12 updates.
    base_update = total_update_rows // N_UPDATES
    remainder = total_update_rows % N_UPDATES

    update_sizes = [
        base_update + (1 if i < remainder else 0)
        for i in range(N_UPDATES)
    ]

    assert sum(update_sizes) == total_update_rows

    # Chronological layout:
    #
    # U1 U2 P1 U3 U4 P2 ... U11 U12 P6
    #
    cursor = 0
    blocks: list[dict[str, Any]] = []

    for update_index in range(1, N_UPDATES + 1):
        size = update_sizes[update_index - 1]
        selected = compatible[cursor : cursor + size]
        cursor += size

        assert len(selected) == size

        blocks.append(
            {
                "kind": "update",
                "index": update_index,
                "rows": selected,
            }
        )

        if update_index % 2 == 0:
            probe_index = update_index // 2
            selected_probe = compatible[
                cursor : cursor + PROBE_ROWS
            ]
            cursor += PROBE_ROWS

            assert len(selected_probe) == PROBE_ROWS

            blocks.append(
                {
                    "kind": "probe",
                    "index": probe_index,
                    "after_update": update_index,
                    "rows": selected_probe,
                }
            )

    assert cursor == total, (cursor, total)

    all_ids = []
    assignment_rows = []
    update_summary = []
    probe_summary = []

    work_placements = defaultdict(
        lambda: defaultdict(int)
    )

    for block in blocks:
        selected = block["rows"]

        ids = [str(r["row_id"]) for r in selected]
        all_ids.extend(ids)

        first = selected[0]
        last = selected[-1]

        works = sorted(
            {
                int(r["work_chronological_index"])
                for r in selected
            }
        )

        partitions = Counter(
            r["_source_partition"]
            for r in selected
        )

        if block["kind"] == "update":
            summary = {
                "update": block["index"],
                "rows": len(selected),
                "steps_one_pass": math.ceil(
                    len(selected) / BATCH_SIZE
                ),
                "first_work": works[0],
                "last_work": works[-1],
                "n_works": len(works),
                "first_date": first["work_creation_date"],
                "last_date": last["work_creation_date"],
                "train_fit_rows": partitions["train_fit"],
                "train_val_rows": partitions["train_val"],
            }

            update_summary.append(summary)

            label = f"U{block['index']}"

        else:
            summary = {
                "probe": block["index"],
                "after_update": block["after_update"],
                "rows": len(selected),
                "first_work": works[0],
                "last_work": works[-1],
                "n_works": len(works),
                "first_date": first["work_creation_date"],
                "last_date": last["work_creation_date"],
                "train_fit_rows": partitions["train_fit"],
                "train_val_rows": partitions["train_val"],
            }

            probe_summary.append(summary)

            label = f"P{block['index']}"

        for row in selected:
            work = int(row["work_chronological_index"])
            work_placements[work][label] += 1

            assignment_rows.append(
                {
                    "block": label,
                    "kind": block["kind"],
                    "index": block["index"],
                    "row_id": row["row_id"],
                    "partition": row["_source_partition"],
                    "work_index": work,
                    "date": row["work_creation_date"],
                    "chronological_position": row[
                        "chronological_position"
                    ],
                }
            )

    assert len(all_ids) == total
    assert len(set(all_ids)) == total

    # Strict chronology between every adjacent block.
    for left, right in zip(blocks, blocks[1:]):
        assert chrono_key(left["rows"][-1]) < chrono_key(
            right["rows"][0]
        )

    # Inspect works that are shared across multiple blocks.
    work_split_summary = []

    for work in sorted(work_placements):
        placements = work_placements[work]

        work_split_summary.append(
            {
                "work_index": work,
                "blocks": ",".join(placements.keys()),
                "n_blocks": len(placements),
                "total_rows": sum(placements.values()),
                "split_across_blocks": len(placements) > 1,
            }
        )

    # Explicitly summarize every evaluation checkpoint.
    checkpoint_summary = []

    for probe_index in range(1, N_PROBES + 1):
        u1 = update_summary[(probe_index * 2) - 2]
        u2 = update_summary[(probe_index * 2) - 1]
        p = probe_summary[probe_index - 1]

        checkpoint_summary.append(
            {
                "checkpoint": probe_index,
                "after_updates": (
                    f"U{probe_index * 2 - 1},"
                    f"U{probe_index * 2}"
                ),
                "pair_update_rows": (
                    u1["rows"] + u2["rows"]
                ),
                "pair_update_steps": (
                    u1["steps_one_pass"]
                    + u2["steps_one_pass"]
                ),
                "probe_rows": p["rows"],
                "probe_first_work": p["first_work"],
                "probe_last_work": p["last_work"],
                "probe_first_date": p["first_date"],
                "probe_last_date": p["last_date"],
            }
        )

    summary = {
        "schema_version": 1,
        "status": "AUDIT_ONLY_NOT_FROZEN",
        "author": AUTHOR,
        "design": {
            "updates": N_UPDATES,
            "evaluation_checkpoints": N_PROBES,
            "probe_rows_per_checkpoint": PROBE_ROWS,
            "layout": (
                "U1 U2 P1 U3 U4 P2 ... "
                "U11 U12 P6"
            ),
            "update_policy": (
                "all non-probe compatible rows distributed "
                "as evenly as possible across 12 "
                "chronological update blocks"
            ),
            "batch_size": BATCH_SIZE,
        },
        "population": {
            "nominal_rows": len(rows),
            "compatible_rows": total,
            "total_update_rows": total_update_rows,
            "total_probe_rows": total_probe_rows,
        },
        "update_sizes": update_sizes,
        "updates": update_summary,
        "probes": probe_summary,
        "checkpoints": checkpoint_summary,
        "split_works": [
            x
            for x in work_split_summary
            if x["split_across_blocks"]
        ],
        "test_used": False,
    }

    (OUT / "audit_summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    write_csv(
        OUT / "update_summary.csv",
        update_summary,
    )

    write_csv(
        OUT / "probe_summary.csv",
        probe_summary,
    )

    write_csv(
        OUT / "checkpoint_summary.csv",
        checkpoint_summary,
    )

    write_csv(
        OUT / "work_split_summary.csv",
        work_split_summary,
    )

    write_csv(
        OUT / "row_assignments.csv",
        assignment_rows,
    )

    print()
    print("============================================================")
    print(" 12 UPDATE + 6 PROBE CANDIDATE AUDIT")
    print("============================================================")
    print()
    print("effective rows =", total)
    print("update rows    =", total_update_rows)
    print("probe rows     =", total_probe_rows)
    print()
    print("update sizes   =", update_sizes)
    print()

    print(
        f"{'UPDATE':<8}"
        f"{'ROWS':>8}"
        f"{'STEPS':>8}"
        f"{'WORKS':>12}"
        f"{'FIT':>8}"
        f"{'VAL':>8}"
        f"  DATES"
    )

    for x in update_summary:
        works = f"{x['first_work']}-{x['last_work']}"

        print(
            f"U{x['update']:<7}"
            f"{x['rows']:>8}"
            f"{x['steps_one_pass']:>8}"
            f"{works:>12}"
            f"{x['train_fit_rows']:>8}"
            f"{x['train_val_rows']:>8}"
            f"  {x['first_date']} -> {x['last_date']}"
        )

    print()
    print("===== PROBES =====")

    for x in probe_summary:
        works = f"{x['first_work']}-{x['last_work']}"

        print(
            f"P{x['probe']} "
            f"after U{x['after_update']} "
            f"rows={x['rows']} "
            f"works={works} "
            f"fit={x['train_fit_rows']} "
            f"val={x['train_val_rows']} "
            f"dates={x['first_date']}->{x['last_date']}"
        )

    print()
    print("===== CHECKPOINTS =====")

    for x in checkpoint_summary:
        print(
            f"C{x['checkpoint']} "
            f"after={x['after_updates']} "
            f"pair_rows={x['pair_update_rows']} "
            f"pair_steps={x['pair_update_steps']} "
            f"probe={x['probe_rows']} "
            f"probe_work="
            f"{x['probe_first_work']}-"
            f"{x['probe_last_work']}"
        )

    print()
    print("===== SPLIT WORKS =====")

    split = [
        x
        for x in work_split_summary
        if x["split_across_blocks"]
    ]

    for x in split:
        print(
            "work",
            x["work_index"],
            "blocks=",
            x["blocks"],
            "rows=",
            x["total_rows"],
        )

    print()
    print("split works =", len(split))
    print()
    print("STATUS = AUDIT_ONLY_NOT_FROZEN")
    print("TEST USED = false")
    print("OUT =", OUT)


if __name__ == "__main__":
    main()
