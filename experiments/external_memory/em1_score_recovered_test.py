"""EM-1 frozen Test recovered-candidate exact scoring.

Frozen from Dev:
- Full+Short
- H5000
- Authors: Etinjat, Re_spectators, breaddddd
- Recovery K = 1
- Frequency lambda = 4

This runner does not use Gold for candidate selection and performs
no Test-time parameter tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from src.reference_backend_pinyingpt.backend import (
    PinyinGPTConcatBackend,
)


AUTHORS = (
    "Etinjat",
    "Re_spectators",
    "breaddddd",
)

EXPECTED_ROWS = 3000

PREDICTIONS_SHA256 = (
    "764db39887f3db04b913d1739d9dbd46295f0e46e5a2bffa649f1563b56ee4e2"
)

TEST_STATES_SHA256 = (
    "2912d32b8cd88843e825cb5592dfbc0a06e88e4a58831c632a126d2b8452b061"
)

FROZEN_RECOVERY_K = 1
FROZEN_FREQUENCY_LAMBDA = 4.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as source:
        return [
            json.loads(line)
            for line in source
            if line.strip()
        ]


def load_predictions(
    path: Path,
) -> dict[str, dict[str, Any]]:
    actual = sha256_file(path)

    if actual != PREDICTIONS_SHA256:
        raise RuntimeError(
            "Frozen prediction SHA mismatch:\n"
            f"expected={PREDICTIONS_SHA256}\n"
            f"actual={actual}"
        )

    output = {}

    for row in load_jsonl(path):
        if row.get("condition") != "full_short":
            continue

        if str(row.get("author")) not in AUTHORS:
            continue

        row_id = str(row["condition_id"])

        if row_id in output:
            raise RuntimeError(
                f"Duplicate Test prediction: {row_id}"
            )

        output[row_id] = row

    if len(output) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} predictions; "
            f"found {len(output)}"
        )

    return output


def load_states(
    path: Path,
) -> dict[str, dict[str, Any]]:
    actual = sha256_file(path)

    if actual != TEST_STATES_SHA256:
        raise RuntimeError(
            "Frozen H5000 Test state SHA mismatch:\n"
            f"expected={TEST_STATES_SHA256}\n"
            f"actual={actual}"
        )

    output = {}

    for row in load_jsonl(path):
        if str(row.get("author")) not in AUTHORS:
            continue

        row_id = str(row["row_id"])

        if row_id in output:
            raise RuntimeError(
                f"Duplicate Test state: {row_id}"
            )

        output[row_id] = row

    if len(output) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} states; "
            f"found {len(output)}"
        )

    return output


def load_existing(
    path: Path,
) -> dict[str, dict[str, Any]]:
    output = {}

    for row in load_jsonl(path):
        row_id = str(row["row_id"])

        if row_id in output:
            raise RuntimeError(
                f"Duplicate cached row: {row_id}"
            )

        output[row_id] = row

    return output


def compatible(
    backend: PinyinGPTConcatBackend,
    candidate: str,
    pinyin: tuple[str, ...],
) -> bool:
    characters = list(candidate)

    if len(characters) != len(pinyin):
        return False

    token_ids = backend.tokenizer.convert_tokens_to_ids(
        characters
    )

    return all(
        token_id in backend.allowed_token_ids[segment]
        for token_id, segment in zip(
            token_ids,
            pinyin,
        )
    )


def select_first_compatible(
    backend: PinyinGPTConcatBackend,
    state: dict[str, Any],
) -> tuple[str | None, int | None]:
    """K=1 after backend-compatibility filtering."""

    pinyin = tuple(
        str(value)
        for value in state["pinyin"]
    )

    for raw_rank, value in enumerate(
        state.get("personal_only_targets", []),
        start=1,
    ):
        candidate = str(value)

        if compatible(
            backend,
            candidate,
            pinyin,
        ):
            return candidate, raw_rank

    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--test-states",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    args = parser.parse_args()

    predictions = load_predictions(
        args.predictions
    )

    states = load_states(
        args.test_states
    )

    if set(predictions) != set(states):
        raise RuntimeError(
            "Prediction condition_id surface and "
            "Test-state row_id surface differ"
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        args.output_root
        / "recovered_candidate_scores.jsonl"
    )

    existing = load_existing(
        output_path
    )

    print(
        f"Loading Frozen PinyinGPT: {args.checkpoint}"
    )

    backend = PinyinGPTConcatBackend(
        args.checkpoint,
        device=args.device,
    )

    eligible = []

    for row_id in sorted(states):
        state = states[row_id]
        prediction = predictions[row_id]

        candidate, raw_rank = (
            select_first_compatible(
                backend,
                state,
            )
        )

        if candidate is None:
            continue

        generic_surface = {
            str(value["text"])
            for value in prediction[
                "top10_candidates"
            ]
        }

        if candidate in generic_surface:
            raise RuntimeError(
                "Personal-only candidate unexpectedly "
                f"already in Generic Top10: "
                f"{row_id} {candidate!r}"
            )

        eligible.append(
            (
                row_id,
                prediction,
                state,
                candidate,
                int(raw_rank),
            )
        )

    eligible_ids = {
        item[0]
        for item in eligible
    }

    stale = set(existing) - eligible_ids

    if stale:
        raise RuntimeError(
            f"Existing cache contains {len(stale)} "
            "rows outside current frozen K=1 surface"
        )

    pending = [
        item
        for item in eligible
        if item[0] not in existing
    ]

    print()
    print(
        "EM-1 Test recovered candidate scoring"
    )
    print("Condition: Full+Short")
    print("Partition: Frozen Test")
    print("History semantics: H5000")
    print(
        "Authors: "
        + ", ".join(AUTHORS)
    )
    print(
        f"Frozen recovery K: "
        f"{FROZEN_RECOVERY_K}"
    )
    print(
        "Frozen frequency lambda: "
        f"{FROZEN_FREQUENCY_LAMBDA:g}"
    )
    print(
        f"Test rows: {len(predictions)}"
    )
    print(
        "Rows with >=1 backend-compatible "
        f"personal-only candidate: {len(eligible)}"
    )
    print(
        f"Already cached: {len(existing)}"
    )
    print(
        f"Pending: {len(pending)}"
    )
    print()

    if pending:
        start = time.perf_counter()

        with output_path.open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as destination:
            for index, (
                row_id,
                prediction,
                state,
                candidate,
                raw_rank,
            ) in enumerate(
                pending,
                start=1,
            ):
                pinyin = tuple(
                    str(value)
                    for value in state["pinyin"]
                )

                scored = backend.score_candidates(
                    context=str(
                        prediction[
                            "model_used_context"
                        ]
                    ),
                    typed_pinyin=pinyin,
                    candidates=(candidate,),
                )

                if len(scored) != 1:
                    raise RuntimeError(
                        f"Expected one score for "
                        f"{row_id}"
                    )

                value = scored[0]

                result = {
                    "schema_version": 1,
                    "experiment": (
                        "em1_recovered_candidate_scoring_test"
                    ),
                    "partition": "test",
                    "condition": "full_short",
                    "row_id": row_id,
                    "anchor_id": str(
                        prediction["anchor_id"]
                    ),
                    "author": str(
                        prediction["author"]
                    ),
                    "pinyin_segments": list(
                        pinyin
                    ),
                    "model_used_context_tokens": int(
                        prediction[
                            "model_used_context_tokens"
                        ]
                    ),
                    "frozen_recovery_k": 1,
                    "frozen_frequency_lambda": 4.0,
                    "selected_raw_personal_rank": (
                        raw_rank
                    ),
                    "scored_candidate_count": 1,
                    "scores": [
                        {
                            "candidate": (
                                value.text
                            ),
                            "personal_candidate_rank": 1,
                            "raw_personal_candidate_rank": (
                                raw_rank
                            ),
                            "fixed_log_probability": (
                                value.log_probability
                            ),
                            "fixed_mean_log_probability": (
                                value.mean_log_probability
                            ),
                        }
                    ],
                }

                destination.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )

                destination.flush()

                if (
                    index % 25 == 0
                    or index == len(pending)
                ):
                    elapsed = (
                        time.perf_counter()
                        - start
                    )

                    rate = (
                        index / elapsed
                        if elapsed
                        else 0.0
                    )

                    print(
                        f"{index}/{len(pending)} rows; "
                        f"rate={rate:.2f} rows/s"
                    )

    final_cache = load_existing(
        output_path
    )

    if set(final_cache) != eligible_ids:
        raise RuntimeError(
            "Final Test score cache does not "
            "match eligible K=1 surface"
        )

    total_scores = sum(
        int(row["scored_candidate_count"])
        for row in final_cache.values()
    )

    summary = {
        "schema_version": 1,
        "experiment": (
            "em1_recovered_candidate_scoring_test"
        ),
        "condition": "Full+Short",
        "partition": "frozen_test",
        "history_budget": 5000,
        "authors": list(AUTHORS),
        "test_rows": len(predictions),
        "frozen_recovery_k": 1,
        "frozen_frequency_lambda": 4.0,
        "eligible_rows": len(eligible),
        "cached_rows_total": len(
            final_cache
        ),
        "candidate_scores_total": (
            total_scores
        ),
        "used_gold_for_candidate_selection": (
            False
        ),
        "used_test_for_parameter_tuning": (
            False
        ),
        "provenance": {
            "predictions_sha256": (
                sha256_file(
                    args.predictions
                )
            ),
            "test_states_sha256": (
                sha256_file(
                    args.test_states
                )
            ),
        },
    }

    with (
        args.output_root
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as destination:
        json.dump(
            summary,
            destination,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        destination.write("\n")

    print()
    print(
        "=== EM-1 Test Recovered Candidate "
        "Scoring Summary ==="
    )
    print(
        f"Test rows: "
        f"{summary['test_rows']}"
    )
    print(
        f"Eligible rows: "
        f"{summary['eligible_rows']}"
    )
    print(
        f"Cached rows total: "
        f"{summary['cached_rows_total']}"
    )
    print(
        f"Candidate scores total: "
        f"{summary['candidate_scores_total']}"
    )
    print(
        "Gold used for candidate selection: "
        f"{summary['used_gold_for_candidate_selection']}"
    )
    print(
        "Test used for parameter tuning: "
        f"{summary['used_test_for_parameter_tuning']}"
    )


if __name__ == "__main__":
    main()
