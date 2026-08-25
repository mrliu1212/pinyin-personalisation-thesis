from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SOURCE = Path(
    "results/model_level/"
    "long_term_qualification_v1/"
    "twelve_update_six_probe_candidate_v1/"
    "row_assignments.csv"
)

OUT = Path(
    "results/model_level/"
    "long_term_qualification_v1/"
    "twelve_update_six_probe_v1"
)

EXPECTED_UPDATES = {
    1: 5556,
    2: 5556,
    3: 5556,
    4: 5556,
    5: 5555,
    6: 5555,
    7: 5555,
    8: 5555,
    9: 5555,
    10: 5555,
    11: 5555,
    12: 5555,
}

EXPECTED_PROBES = {
    1: 500,
    2: 500,
    3: 500,
    4: 500,
    5: 500,
    6: 500,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


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


def main() -> None:
    if not SOURCE.exists():
        raise RuntimeError(
            f"Missing candidate assignments: {SOURCE}"
        )

    OUT.mkdir(parents=True, exist_ok=True)

    rows = []

    with SOURCE.open(
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            row = dict(row)
            row["index"] = int(row["index"])
            row["work_index"] = int(row["work_index"])
            row["chronological_position"] = int(
                row["chronological_position"]
            )
            rows.append(row)

    assert len(rows) == 69664, len(rows)

    ids = [r["row_id"] for r in rows]
    assert len(set(ids)) == len(ids)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[row["block"]].append(row)

    manifest_records = []

    # Freeze all 12 update manifests.
    for i in range(1, 13):
        label = f"U{i}"
        selected = grouped[label]

        assert len(selected) == EXPECTED_UPDATES[i]

        path = OUT / f"update_{i:02d}_row_ids.json"

        write_json(
            path,
            [r["row_id"] for r in selected],
        )

        manifest_records.append(
            {
                "label": label,
                "role": "update",
                "index": i,
                "rows": len(selected),
                "optimizer_steps_one_pass_batch8": 695,
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

    # Freeze all 6 probe manifests.
    for i in range(1, 7):
        label = f"P{i}"
        selected = grouped[label]

        assert len(selected) == EXPECTED_PROBES[i]

        path = OUT / f"probe_{i:02d}_row_ids.json"

        write_json(
            path,
            [r["row_id"] for r in selected],
        )

        manifest_records.append(
            {
                "label": label,
                "role": "probe",
                "index": i,
                "after_update": i * 2,
                "rows": len(selected),
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

    expected_sequence = []
    for i in range(1, 13):
        expected_sequence.append(f"U{i}")
        if i % 2 == 0:
            expected_sequence.append(f"P{i // 2}")

    actual_sequence = []
    previous = None

    for row in rows:
        block = row["block"]

        if block != previous:
            actual_sequence.append(block)
            previous = block

    assert actual_sequence == expected_sequence, (
        actual_sequence,
        expected_sequence,
    )

    update_ids = {
        row["row_id"]
        for row in rows
        if row["kind"] == "update"
    }

    probe_ids = {
        row["row_id"]
        for row in rows
        if row["kind"] == "probe"
    }

    assert len(update_ids) == 66664
    assert len(probe_ids) == 3000
    assert not (update_ids & probe_ids)
    assert len(update_ids | probe_ids) == 69664

    source_sha = sha256_file(SOURCE)

    frozen = {
        "schema_version": 1,
        "status": "FROZEN",
        "experiment": (
            "model_level_twelve_update_six_probe_v1"
        ),
        "author": "Agent Phage",
        "design": {
            "layout": (
                "U1 U2 P1 U3 U4 P2 U5 U6 P3 "
                "U7 U8 P4 U9 U10 P5 U11 U12 P6"
            ),
            "updates": 12,
            "evaluation_checkpoints": 6,
            "probe_rows_per_checkpoint": 500,
            "batch_size": 8,
            "optimizer_steps_per_update": 695,
            "evaluation_after_updates": [
                2,
                4,
                6,
                8,
                10,
                12,
            ],
            "selection_principle": (
                "regular chronological checkpoints; "
                "checkpoint positions were not selected "
                "for favorable work diversity"
            ),
        },
        "population": {
            "effective_rows": 69664,
            "update_rows": 66664,
            "probe_rows": 3000,
            "test_used": False,
        },
        "known_stream_structure": {
            "large_work_22_rows": 16756,
            "large_work_26_rows": 9195,
            "probe_2_work": 22,
            "probe_3_work": 22,
            "probe_4_work": 26,
            "interpretation_note": (
                "Immediate-future probes preserve the natural "
                "chronological stream and may be concentrated "
                "within large works. They are not treated alone "
                "as evidence of work-independent preference."
            ),
        },
        "source_candidate_assignments": str(SOURCE),
        "source_candidate_assignments_sha256": source_sha,
        "manifests": manifest_records,
    }

    write_json(
        OUT / "frozen_protocol.json",
        frozen,
    )

    # Human-readable manifest table.
    with (
        OUT / "manifest_summary.csv"
    ).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        fields = [
            "label",
            "role",
            "index",
            "rows",
            "first_work",
            "last_work",
            "first_date",
            "last_date",
            "sha256",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        for x in manifest_records:
            writer.writerow(
                {
                    key: x.get(key, "")
                    for key in fields
                }
            )

    print("============================================================")
    print(" FROZEN 12-UPDATE / 6-PROBE PROTOCOL")
    print("============================================================")
    print()
    print("effective rows =", 69664)
    print("update rows    =", len(update_ids))
    print("probe rows     =", len(probe_ids))
    print("layout         =", frozen["design"]["layout"])
    print()
    print("UPDATE COUNTS")

    for i in range(1, 13):
        print(
            f"U{i:02d} = "
            f"{len(grouped[f'U{i}'])} rows, "
            f"695 steps"
        )

    print()
    print("PROBE COUNTS")

    for i in range(1, 7):
        selected = grouped[f"P{i}"]
        print(
            f"P{i} = {len(selected)} rows "
            f"after U{i * 2}, "
            f"work "
            f"{selected[0]['work_index']}-"
            f"{selected[-1]['work_index']}"
        )

    print()
    print("source assignment sha256 =", source_sha)
    print(
        "frozen protocol sha256   =",
        sha256_file(
            OUT / "frozen_protocol.json"
        ),
    )
    print()
    print("STATUS = FROZEN")
    print("TEST USED = false")


if __name__ == "__main__":
    main()
