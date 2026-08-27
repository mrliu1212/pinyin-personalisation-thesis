"""Final five-author Adapter training wrapper.

Reuses the validated Adapter V1 trainer unchanged except for:
1. loading the frozen final-five-author Fit schema;
2. accepting the frozen 150,000-row Fit population;
3. preserving each row's already-frozen Full/Initial/Mixed Pinyin input.

No Val or Test rows are accepted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.model_level import run_adapter_training_v1 as base
from src.model_level.pinyin_modes import PinyinRepresentation


EXPECTED_FIT_ROWS = 150_000

EXPECTED_AUTHOR_ROWS = {
    "Re_spectators": 30_000,
    "Etinjat": 30_000,
    "Agent Phage": 30_000,
    "QBLevi": 30_000,
    "breaddddd": 30_000,
}

EXPECTED_FIT_SHA256 = (
    "789d0e47e18120d1a9c5136c4f185d2ceb7c760b00f2c9796d4d356032cd4739"
)

# Filled when the Fit file is loaded.
FROZEN_REPRESENTATIONS: dict[str, dict[str, Any]] = {}


def _segments(value: str) -> tuple[str, ...]:
    return tuple(x for x in str(value).strip().split() if x)


def _derive_mask(
    *,
    full: tuple[str, ...],
    segmented: tuple[str, ...],
    requested: str,
    effective: str,
) -> tuple[bool, ...]:

    if len(full) != len(segmented):
        raise RuntimeError(
            "Full/frozen Pinyin segment-count mismatch"
        )

    for f, s in zip(full, segmented):
        if s not in {f, f[0]}:
            raise RuntimeError(
                f"Frozen Pinyin segment {s!r} "
                f"is neither Full nor Initial of {f!r}"
            )

    if requested == "full":
        if segmented != full:
            raise RuntimeError(
                "Frozen Full row contains abbreviation"
            )
        return (False,) * len(full)

    if requested == "initial":
        initials = tuple(x[0] for x in full)
        if segmented != initials:
            raise RuntimeError(
                "Frozen Initial row is not all-initial"
            )
        return (True,) * len(full)

    if requested != "mixed":
        raise RuntimeError(
            f"Unknown frozen typing mode: {requested!r}"
        )

    # Text alone cannot distinguish Full versus Initial when
    # a full syllable itself has length 1 (e.g. "a").
    # This does not change the actual model input.  Build a mask
    # consistent with the frozen effective mode where possible.
    mask = [s != f for f, s in zip(full, segmented)]
    ambiguous = [
        i
        for i, (f, s) in enumerate(zip(full, segmented))
        if len(f) == 1 and s == f
    ]

    if effective == "full":
        mask = [False] * len(full)

    elif effective == "initial":
        mask = [True] * len(full)

    elif effective == "mixed":
        if len(full) < 2:
            raise RuntimeError(
                "Single-syllable row cannot have effective=mixed"
            )

        if not any(mask):
            if not ambiguous:
                raise RuntimeError(
                    "Cannot reconstruct frozen Mixed mask"
                )
            mask[ambiguous[0]] = True

        if all(mask):
            if not ambiguous:
                raise RuntimeError(
                    "Cannot reconstruct frozen Mixed mask"
                )
            mask[ambiguous[0]] = False

    else:
        raise RuntimeError(
            f"Unknown effective typing mode: {effective!r}"
        )

    return tuple(mask)


def load_final_author_rows(
    path: Path,
    *,
    author: str,
) -> tuple[list[dict[str, Any]], int]:

    if base.sha256_file(path) != EXPECTED_FIT_SHA256:
        raise RuntimeError(
            "STOP: frozen Fit SHA256 changed"
        )

    if author not in EXPECTED_AUTHOR_ROWS:
        raise RuntimeError(
            f"Unexpected final author: {author!r}"
        )

    rows: list[dict[str, Any]] = []
    total = 0

    FROZEN_REPRESENTATIONS.clear()

    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue

            source_row = json.loads(line)
            total += 1

            split = str(source_row.get("split", "")).lower()

            if split == "test":
                raise RuntimeError(
                    "STOP: Test row detected in Adapter Fit input"
                )

            if split != "fit":
                raise RuntimeError(
                    f"STOP: non-Fit row detected: {split!r}"
                )

            row_author = str(source_row["author_name"])

            full = _segments(source_row["full_pinyin"])
            frozen = _segments(source_row["pinyin_input"])

            gold = str(source_row["gold"])

            if len(gold) != len(full):
                raise RuntimeError(
                    "Gold/Full-Pinyin alignment failure for "
                    f"{source_row['row_id']}"
                )

            if len(gold) != len(frozen):
                raise RuntimeError(
                    "Gold/Frozen-Pinyin alignment failure for "
                    f"{source_row['row_id']}"
                )

            requested = str(source_row["typing_mode"])
            effective = str(source_row["effective_typing_mode"])

            mask = _derive_mask(
                full=full,
                segmented=frozen,
                requested=requested,
                effective=effective,
            )

            row_id = str(source_row["row_id"])

            if row_id in FROZEN_REPRESENTATIONS:
                raise RuntimeError(
                    f"Duplicate row_id: {row_id}"
                )

            FROZEN_REPRESENTATIONS[row_id] = {
                "requested_mode": requested,
                "effective_mode": effective,
                "full_segments": full,
                "segmented_input": frozen,
                "mask": mask,
            }

            if row_author != author:
                continue

            # Canonical aliases expected by the validated trainer.
            row = dict(source_row)

            row.update(
                {
                    "author": row_author,
                    "source_split": "fit",
                    "condition": "full_short",
                    "target_type": "short",
                    "standardized_partition": "train_fit",
                    "pinyin_segments": list(full),
                    "target": gold,
                }
            )

            rows.append(row)

    if total != EXPECTED_FIT_ROWS:
        raise RuntimeError(
            f"Fit row count changed: "
            f"{total} != {EXPECTED_FIT_ROWS}"
        )

    expected = EXPECTED_AUTHOR_ROWS[author]

    if len(rows) != expected:
        raise RuntimeError(
            f"{author} Fit count changed: "
            f"{len(rows)} != {expected}"
        )

    return rows, total


def choose_frozen_pinyin_representation(
    full_segments,
    *,
    row_id: str,
    epoch: int,
    seed: int,
    forced_mode=None,
    weights=(3, 1, 2),
) -> PinyinRepresentation:

    if row_id not in FROZEN_REPRESENTATIONS:
        raise RuntimeError(
            f"No frozen Pinyin representation for {row_id}"
        )

    value = FROZEN_REPRESENTATIONS[row_id]

    full = tuple(value["full_segments"])

    if tuple(map(str, full_segments)) != full:
        raise RuntimeError(
            f"Full Pinyin provenance drift for {row_id}"
        )

    mask = tuple(value["mask"])

    abbreviated_positions = tuple(
        i
        for i, abbreviated in enumerate(mask)
        if abbreviated
    )

    return PinyinRepresentation(
        schema_version=1,
        row_id=row_id,
        epoch=epoch,
        seed=seed,
        requested_mode=value["requested_mode"],
        effective_mode=value["effective_mode"],
        full_segments=full,
        segmented_input=tuple(value["segmented_input"]),
        abbreviated_mask=mask,
        abbreviated_positions=abbreviated_positions,

        # No sampling weights are used in this final wrapper.
        mode_weights=(0, 0, 0),

        selection_policy=(
            "frozen_final_fiveauthor_v1_exact_pinyin_input"
        ),
    )


# ------------------------------------------------------------
# Patch only the population/Pinyin-input interface.
# Everything below remains the validated historical trainer.
# ------------------------------------------------------------

base.load_author_rows = load_final_author_rows
base.choose_pinyin_representation = (
    choose_frozen_pinyin_representation
)

base.EXPECTED_FIT_ROWS = EXPECTED_FIT_ROWS
base.EXPECTED_AUTHOR_ROWS = EXPECTED_AUTHOR_ROWS
base.EXPECTED_FIT_SHA256 = EXPECTED_FIT_SHA256

base.EXPERIMENT = (
    "final_fiveauthor_adapter_v1_training"
)


if __name__ == "__main__":
    base.main()
