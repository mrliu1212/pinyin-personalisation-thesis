from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


AUTHOR = "Agent Phage"
N_CYCLES = 6
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
    "six_cycle_candidate_audit_v1"
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


def load_rows(
    path: Path,
    partition: str,
) -> list[dict[str, Any]]:
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

    # Group effective rows by complete work.
    by_work: dict[int, list[dict[str, Any]]] = {}

    for row in compatible:
        work = int(row["work_chronological_index"])
        by_work.setdefault(work, []).append(row)

    works = sorted(by_work)
    assert works == list(range(45)), works

    work_groups = [by_work[w] for w in works]

    cumulative = []
    running = 0

    for group in work_groups:
        running += len(group)
        cumulative.append(running)

    total = len(compatible)

    # Choose five macro-cycle boundaries at complete work boundaries,
    # each as close as possible to equal-row sixths.
    cuts: list[int] = []
    previous = -1

    for k in range(1, N_CYCLES):
        target = total * k / N_CYCLES

        min_j = previous + 1
        max_j = len(work_groups) - (N_CYCLES - k) - 1

        candidates = list(range(min_j, max_j + 1))

        best = min(
            candidates,
            key=lambda j: (
                abs(cumulative[j] - target),
                j,
            ),
        )

        cuts.append(best)
        previous = best

    macro_cycles: list[list[dict[str, Any]]] = []
    start_group = 0

    for cut in cuts + [len(work_groups) - 1]:
        block = []

        for group in work_groups[start_group : cut + 1]:
            block.extend(group)

        macro_cycles.append(block)
        start_group = cut + 1

    assert len(macro_cycles) == N_CYCLES
    assert sum(map(len, macro_cycles)) == total

    summaries = []
    assignment_rows = []

    all_update_ids = set()
    all_probe_ids = set()

    for cycle_index, block in enumerate(macro_cycles, 1):
        assert len(block) > PROBE_ROWS

        update = block[:-PROBE_ROWS]
        probe = block[-PROBE_ROWS:]

        update_ids = {str(r["row_id"]) for r in update}
        probe_ids = {str(r["row_id"]) for r in probe}

        assert not (update_ids & probe_ids)
        assert not (all_update_ids & update_ids)
        assert not (all_probe_ids & probe_ids)
        assert not (all_update_ids & probe_ids)
        assert not (all_probe_ids & update_ids)

        all_update_ids |= update_ids
        all_probe_ids |= probe_ids

        macro_works = sorted(
            {
                int(r["work_chronological_index"])
                for r in block
            }
        )

        update_works = sorted(
            {
                int(r["work_chronological_index"])
                for r in update
            }
        )

        probe_works = sorted(
            {
                int(r["work_chronological_index"])
                for r in probe
            }
        )

        # Probe begins after every update row by the full chronology key.
        assert chrono_key(update[-1]) < chrono_key(probe[0])

        partition_counts_update = Counter(
            r["_source_partition"]
            for r in update
        )
        partition_counts_probe = Counter(
            r["_source_partition"]
            for r in probe
        )

        probe_starts_mid_work = (
            int(update[-1]["work_chronological_index"])
            == int(probe[0]["work_chronological_index"])
        )

        summaries.append(
            {
                "cycle": cycle_index,
                "macro_rows": len(block),
                "update_rows": len(update),
                "probe_rows": len(probe),
                "update_steps_one_pass": math.ceil(
                    len(update) / BATCH_SIZE
                ),
                "first_work": macro_works[0],
                "last_work": macro_works[-1],
                "n_works": len(macro_works),
                "first_date": block[0]["work_creation_date"],
                "last_date": block[-1]["work_creation_date"],
                "update_last_work": update_works[-1],
                "probe_first_work": probe_works[0],
                "probe_last_work": probe_works[-1],
                "probe_starts_mid_work": probe_starts_mid_work,
                "update_train_fit": partition_counts_update["train_fit"],
                "update_train_val": partition_counts_update["train_val"],
                "probe_train_fit": partition_counts_probe["train_fit"],
                "probe_train_val": partition_counts_probe["train_val"],
            }
        )

        for role, selected in (
            ("update", update),
            ("probe", probe),
        ):
            for row in selected:
                assignment_rows.append(
                    {
                        "cycle": cycle_index,
                        "role": role,
                        "row_id": row["row_id"],
                        "partition": row["_source_partition"],
                        "work_index": row[
                            "work_chronological_index"
                        ],
                        "date": row["work_creation_date"],
                        "chronological_position": row[
                            "chronological_position"
                        ],
                    }
                )

    assert len(all_probe_ids) == N_CYCLES * PROBE_ROWS
    assert len(all_update_ids) == total - len(all_probe_ids)
    assert not (all_update_ids & all_probe_ids)

    summary = {
        "schema_version": 1,
        "status": "AUDIT_ONLY_NOT_FROZEN",
        "author": AUTHOR,
        "design": {
            "cycles": N_CYCLES,
            "macro_boundary_policy": (
                "nearest equal-row target at complete work boundaries"
            ),
            "probe_policy": (
                "last 500 chronological compatible rows "
                "inside each macro-cycle"
            ),
            "probe_rows_per_cycle": PROBE_ROWS,
            "batch_size": BATCH_SIZE,
        },
        "population": {
            "nominal_rows": len(rows),
            "compatible_rows": len(compatible),
            "works": len(works),
            "first_work": works[0],
            "last_work": works[-1],
            "total_update_rows": len(all_update_ids),
            "total_probe_rows": len(all_probe_ids),
        },
        "cycles": summaries,
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
        OUT / "cycle_summary.csv",
        summaries,
    )

    write_csv(
        OUT / "row_assignments.csv",
        assignment_rows,
    )

    print()
    print("============================================================")
    print(" SIX-CYCLE CANDIDATE AUDIT")
    print("============================================================")
    print()
    print("effective rows =", len(compatible))
    print("cycles         =", N_CYCLES)
    print("probe/cycle    =", PROBE_ROWS)
    print("total update   =", len(all_update_ids))
    print("total probe    =", len(all_probe_ids))
    print()
    print(
        f"{'CYCLE':<7}"
        f"{'WORKS':<10}"
        f"{'MACRO':>8}"
        f"{'UPDATE':>9}"
        f"{'STEPS':>8}"
        f"{'PROBE':>8}"
        f"{'MIDWORK':>10}"
        f"  DATES"
    )

    for x in summaries:
        print(
            f"{x['cycle']:<7}"
            f"{x['first_work']}-{x['last_work']:<7}"
            f"{x['macro_rows']:>8}"
            f"{x['update_rows']:>9}"
            f"{x['update_steps_one_pass']:>8}"
            f"{x['probe_rows']:>8}"
            f"{str(x['probe_starts_mid_work']):>10}"
            f"  {x['first_date']} -> {x['last_date']}"
        )

    print()
    print("STATUS = AUDIT_ONLY_NOT_FROZEN")
    print("TEST USED = false")
    print()
    print("OUT =", OUT)


if __name__ == "__main__":
    main()
