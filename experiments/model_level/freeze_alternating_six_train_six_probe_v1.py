from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


AUTHOR = "Agent Phage"
PROBE_ROWS = 500
BATCH_SIZE = 8

SOURCE = Path(
    "results/model_level/"
    "long_term_qualification_v1/"
    "alternating_six_train_six_eval_candidate_v1/"
    "row_assignments.csv"
)

OUT = Path(
    "results/model_level/"
    "long_term_qualification_v1/"
    "alternating_six_train_six_probe_v1"
)

OLD_PROTOCOL = Path(
    "results/model_level/"
    "long_term_qualification_v1/"
    "twelve_update_six_probe_v1/"
    "frozen_protocol.json"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def uniform_indices(n: int, k: int) -> list[int]:
    """
    Deterministic coverage of the full chronological block.

    Uses k equal strata and takes the midpoint row of each stratum.
    No randomness and no result-dependent selection.
    """
    if k < 1 or k > n:
        raise ValueError((n, k))

    indices = []

    for i in range(k):
        start = (i * n) // k
        end = ((i + 1) * n) // k

        if end <= start:
            raise RuntimeError(
                f"Empty stratum i={i} n={n} k={k}"
            )

        index = (start + end - 1) // 2
        indices.append(index)

    if len(indices) != k:
        raise AssertionError(len(indices))

    if len(set(indices)) != k:
        raise AssertionError("Duplicate probe indices")

    if indices != sorted(indices):
        raise AssertionError("Probe indices not chronological")

    return indices


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    OUT.mkdir(parents=True, exist_ok=True)

    rows = []

    with SOURCE.open(
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            x = dict(row)
            x["block"] = int(x["block"])
            x["work_index"] = int(x["work_index"])
            x["chronological_position"] = int(
                x["chronological_position"]
            )
            rows.append(x)

    assert len(rows) == 69664, len(rows)
    assert len({x["row_id"] for x in rows}) == 69664

    by_block: dict[int, list[dict[str, Any]]] = {}

    for row in rows:
        by_block.setdefault(row["block"], []).append(row)

    assert sorted(by_block) == list(range(1, 13))

    train_blocks = [1, 3, 5, 7, 9, 11]
    eval_blocks = [2, 4, 6, 8, 10, 12]

    manifest_records = []

    all_train_ids: set[str] = set()
    all_probe_ids: set[str] = set()
    reserve_ids: set[str] = set()

    # ---------------------------------------------------------
    # Freeze all six complete training blocks.
    # ---------------------------------------------------------
    for cycle, block_id in enumerate(train_blocks, start=1):
        selected = by_block[block_id]

        assert all(x["role"] == "train" for x in selected)
        assert len(selected) in (5805, 5806)

        ids = [x["row_id"] for x in selected]

        if all_train_ids & set(ids):
            raise AssertionError("Train overlap")

        all_train_ids.update(ids)

        path = OUT / f"cycle_{cycle:02d}_train_row_ids.json"

        write_json(path, ids)

        manifest_records.append(
            {
                "cycle": cycle,
                "source_block": block_id,
                "role": "train",
                "rows": len(selected),
                "optimizer_steps_one_pass_batch8": 726,
                "first_row_id": selected[0]["row_id"],
                "last_row_id": selected[-1]["row_id"],
                "first_work": selected[0]["work_index"],
                "last_work": selected[-1]["work_index"],
                "first_date": selected[0]["date"],
                "last_date": selected[-1]["date"],
                "manifest": str(path),
                "sha256": sha256_file(path),
            }
        )

    # ---------------------------------------------------------
    # Freeze one 500-row deterministic uniformly-covered probe
    # from each corresponding future evaluation block.
    # ---------------------------------------------------------
    probe_selection_audit = []

    for cycle, block_id in enumerate(eval_blocks, start=1):
        block = by_block[block_id]

        assert all(x["role"] == "eval" for x in block)
        assert len(block) in (5805, 5806)

        indices = uniform_indices(
            len(block),
            PROBE_ROWS,
        )

        selected = [block[i] for i in indices]
        selected_id_set = {
            x["row_id"]
            for x in selected
        }

        assert len(selected) == PROBE_ROWS
        assert len(selected_id_set) == PROBE_ROWS

        if all_probe_ids & selected_id_set:
            raise AssertionError("Probe overlap")

        if all_train_ids & selected_id_set:
            raise AssertionError("Train/probe overlap")

        all_probe_ids.update(selected_id_set)

        block_ids = {
            x["row_id"]
            for x in block
        }

        reserve = block_ids - selected_id_set
        reserve_ids.update(reserve)

        path = OUT / f"cycle_{cycle:02d}_probe_row_ids.json"

        write_json(
            path,
            [x["row_id"] for x in selected],
        )

        index_path = (
            OUT
            / f"cycle_{cycle:02d}_probe_selection_indices.json"
        )

        write_json(
            index_path,
            {
                "source_block": block_id,
                "source_block_rows": len(block),
                "probe_rows": PROBE_ROWS,
                "selection_policy": (
                    "500 equal chronological strata; "
                    "take midpoint row of each stratum"
                ),
                "zero_based_indices": indices,
            },
        )

        works = sorted(
            {
                x["work_index"]
                for x in selected
            }
        )

        probe_selection_audit.append(
            {
                "cycle": cycle,
                "source_block": block_id,
                "source_block_rows": len(block),
                "probe_rows": len(selected),
                "reserve_rows": len(reserve),
                "probe_first_index": indices[0],
                "probe_last_index": indices[-1],
                "probe_first_work": works[0],
                "probe_last_work": works[-1],
                "probe_n_works": len(works),
                "probe_first_date": selected[0]["date"],
                "probe_last_date": selected[-1]["date"],
                "probe_manifest": str(path),
                "probe_sha256": sha256_file(path),
                "selection_manifest": str(index_path),
                "selection_sha256": sha256_file(index_path),
            }
        )

        manifest_records.append(
            {
                "cycle": cycle,
                "source_block": block_id,
                "role": "probe",
                "rows": len(selected),
                "first_row_id": selected[0]["row_id"],
                "last_row_id": selected[-1]["row_id"],
                "first_work": works[0],
                "last_work": works[-1],
                "n_works": len(works),
                "first_date": selected[0]["date"],
                "last_date": selected[-1]["date"],
                "manifest": str(path),
                "sha256": sha256_file(path),
            }
        )

    assert len(all_train_ids) == 34832
    assert len(all_probe_ids) == 3000
    assert len(reserve_ids) == 31832

    assert not (all_train_ids & all_probe_ids)
    assert not (all_train_ids & reserve_ids)
    assert not (all_probe_ids & reserve_ids)

    assert (
        len(
            all_train_ids
            | all_probe_ids
            | reserve_ids
        )
        == 69664
    )

    # ---------------------------------------------------------
    # Explicitly mark the earlier 66,664-train design as
    # superseded, without deleting its provenance.
    # ---------------------------------------------------------
    supersession = {
        "schema_version": 1,
        "status": "SUPERSEDED_NOT_EXECUTED",
        "superseded_protocol": str(OLD_PROTOCOL),
        "replacement_protocol": (
            str(OUT / "frozen_protocol.json")
        ),
        "reason": (
            "Revised before execution to match the intended "
            "alternating longitudinal design: six approximately "
            "5.8K training episodes interleaved with six held-out "
            "future blocks, rather than training on 66,664 of "
            "69,664 effective rows."
        ),
    }

    write_json(
        OUT / "supersedes_previous_protocol.json",
        supersession,
    )

    frozen = {
        "schema_version": 1,
        "status": "FROZEN",
        "experiment": (
            "model_level_alternating_six_train_"
            "six_probe_v1"
        ),
        "author": AUTHOR,
        "design": {
            "chronological_blocks": 12,
            "cycles": 6,
            "train_blocks": train_blocks,
            "future_eval_blocks": eval_blocks,
            "layout": (
                "B1_train B2_future | "
                "B3_train B4_future | "
                "B5_train B6_future | "
                "B7_train B8_future | "
                "B9_train B10_future | "
                "B11_train B12_future"
            ),
            "training_policy": (
                "all rows from odd chronological blocks"
            ),
            "probe_policy": (
                "500 deterministic rows per even future block; "
                "divide the full block into 500 chronological "
                "strata and select the midpoint row of each stratum"
            ),
            "probe_selection_random": False,
            "probe_selection_result_dependent": False,
            "batch_size": BATCH_SIZE,
            "optimizer_steps_per_cycle": 726,
        },
        "population": {
            "effective_rows": 69664,
            "training_rows": len(all_train_ids),
            "primary_probe_rows": len(all_probe_ids),
            "held_out_reserve_rows": len(reserve_ids),
            "training_fraction": len(all_train_ids) / 69664,
            "primary_probe_fraction": len(all_probe_ids) / 69664,
            "all_nontraining_rows": (
                len(all_probe_ids) + len(reserve_ids)
            ),
            "test_used": False,
        },
        "interpretation": {
            "primary_probe_meaning": (
                "held-out future-slice prediction after each "
                "chronological training episode"
            ),
            "reserve_policy": (
                "non-probe rows in even blocks remain held out "
                "from primary training and primary evaluation; "
                "they may only be used later for explicitly "
                "labelled robustness analysis"
            ),
        },
        "source_assignments": str(SOURCE),
        "source_assignments_sha256": sha256_file(SOURCE),
        "manifests": manifest_records,
        "probe_selection_audit": probe_selection_audit,
        "supersedes": (
            "model_level_twelve_update_six_probe_v1"
        ),
    }

    protocol_path = OUT / "frozen_protocol.json"
    write_json(protocol_path, frozen)

    with (
        OUT / "cycle_summary.csv"
    ).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        fields = [
            "cycle",
            "train_block",
            "train_rows",
            "train_steps",
            "probe_block",
            "probe_rows",
            "reserve_rows",
            "probe_first_work",
            "probe_last_work",
            "probe_n_works",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        train_by_cycle = {
            x["cycle"]: x
            for x in manifest_records
            if x["role"] == "train"
        }

        probe_by_cycle = {
            x["cycle"]: x
            for x in probe_selection_audit
        }

        for cycle in range(1, 7):
            t = train_by_cycle[cycle]
            p = probe_by_cycle[cycle]

            writer.writerow(
                {
                    "cycle": cycle,
                    "train_block": t["source_block"],
                    "train_rows": t["rows"],
                    "train_steps": (
                        t[
                            "optimizer_steps_one_pass_batch8"
                        ]
                    ),
                    "probe_block": p["source_block"],
                    "probe_rows": p["probe_rows"],
                    "reserve_rows": p["reserve_rows"],
                    "probe_first_work": p[
                        "probe_first_work"
                    ],
                    "probe_last_work": p[
                        "probe_last_work"
                    ],
                    "probe_n_works": p[
                        "probe_n_works"
                    ],
                }
            )

    print("============================================================")
    print(" FROZEN ALTERNATING 6-TRAIN / 6-PROBE V1")
    print("============================================================")
    print()
    print("effective rows       =", 69664)
    print("training rows        =", len(all_train_ids))
    print("primary probe rows   =", len(all_probe_ids))
    print("held-out reserve     =", len(reserve_ids))
    print("training steps total =", 6 * 726)
    print()

    for cycle in range(1, 7):
        t = [
            x
            for x in manifest_records
            if x["role"] == "train"
            and x["cycle"] == cycle
        ][0]

        p = probe_selection_audit[cycle - 1]

        print(
            f"Cycle {cycle}: "
            f"B{t['source_block']} train="
            f"{t['rows']} rows / 726 steps "
            f"-> B{p['source_block']} probe="
            f"{p['probe_rows']} rows "
            f"(works "
            f"{p['probe_first_work']}-"
            f"{p['probe_last_work']}, "
            f"{p['probe_n_works']} works)"
        )

    print()
    print(
        "source assignment sha256 =",
        frozen["source_assignments_sha256"],
    )
    print(
        "frozen protocol sha256   =",
        sha256_file(protocol_path),
    )
    print()
    print("STATUS = FROZEN")
    print("PREVIOUS = SUPERSEDED_NOT_EXECUTED")
    print("TEST USED = false")


if __name__ == "__main__":
    main()
