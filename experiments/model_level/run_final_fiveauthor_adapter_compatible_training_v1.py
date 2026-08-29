"""Formal final-five-author Adapter trainer on the frozen compatible Fit subset."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.model_level import run_adapter_training_v1 as base
from experiments.model_level import run_final_fiveauthor_adapter_training_v1 as final


COMPATIBLE_PATH = Path(
    "results/finalmodel_fiveauthor_v1/"
    "adapter_input_preparation_v1/"
    "adapter_fit_compatible_row_ids_frozen_v1.json"
)

EXPECTED_COMPATIBLE_SHA256 = (
    "64403217bd0af78f211c238b50db74a1b0144cf939e83908de8b4701f1433d3d"
)

EXPECTED_COUNTS = {
    "Re_spectators": 29951,
    "Etinjat": 29131,
    "Agent Phage": 29991,
    "QBLevi": 29967,
    "breaddddd": 29981,
}


if base.sha256_file(COMPATIBLE_PATH) != EXPECTED_COMPATIBLE_SHA256:
    raise RuntimeError(
        "STOP: compatible row-ID freeze SHA256 changed"
    )

payload = json.loads(
    COMPATIBLE_PATH.read_text(encoding="utf-8")
)

COMPATIBLE = {
    author: set(payload["authors"][author])
    for author in EXPECTED_COUNTS
}


def load_compatible_author_rows(
    path: Path,
    *,
    author: str,
):
    rows, total = final.load_final_author_rows(
        path,
        author=author,
    )

    allowed = COMPATIBLE[author]

    selected = [
        row
        for row in rows
        if str(row["row_id"]) in allowed
    ]

    actual_ids = {
        str(row["row_id"])
        for row in selected
    }

    if actual_ids != allowed:
        missing = sorted(allowed - actual_ids)
        extra = sorted(actual_ids - allowed)

        raise RuntimeError(
            f"{author}: compatible population mismatch; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    expected = EXPECTED_COUNTS[author]

    if len(selected) != expected:
        raise RuntimeError(
            f"{author}: expected {expected} compatible rows, "
            f"got {len(selected)}"
        )

    return selected, total


base.load_author_rows = load_compatible_author_rows

# Preserve the exact frozen Pinyin manifestation logic.
base.choose_pinyin_representation = (
    final.choose_frozen_pinyin_representation
)

base.EXPERIMENT = (
    "final_fiveauthor_adapter_compatible_v1_training"
)


if __name__ == "__main__":
    base.main()
