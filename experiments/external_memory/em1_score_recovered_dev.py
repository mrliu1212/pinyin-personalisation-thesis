from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from src.reference_backend_pinyingpt.backend import PinyinGPTConcatBackend


AUTHORS = (
    "Etinjat",
    "Re_spectators",
    "breaddddd",
)

MAX_K = 5
HISTORY_BUDGET = 5000

GENERIC_SHA256 = (
    "588aa84c6397e8cb1a13576c0d5dfecd9dd2c4305b45be351328dd83ef62007d"
)

PV_DEV_STATES_SHA256 = (
    "5d367b1bf2294e0d9ff4102d26cb4dd4732d1c1d520a20a86086377d3b0bcbc5"
)

PV_DEV_STATES_ROWS = 16171


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_generic(path: Path) -> dict[str, dict[str, Any]]:
    actual_hash = sha256_file(path)
    if actual_hash != GENERIC_SHA256:
        raise RuntimeError(
            "Frozen Generic Dev cache SHA mismatch:\n"
            f"expected={GENERIC_SHA256}\n"
            f"actual={actual_hash}"
        )

    rows: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            row = json.loads(line)

            if row.get("pilot_partition") != "tune":
                continue

            author = str(row.get("author", ""))
            if author not in AUTHORS:
                continue

            row_id = str(row["row_id"])

            if row_id in rows:
                raise RuntimeError(
                    f"Duplicate Generic row_id at line {line_number}: {row_id}"
                )

            rows[row_id] = row

    return rows


def load_states(path: Path) -> dict[str, dict[str, Any]]:
    actual_hash = sha256_file(path)
    if actual_hash != PV_DEV_STATES_SHA256:
        raise RuntimeError(
            "Frozen PV Dev state cache SHA mismatch:\n"
            f"expected={PV_DEV_STATES_SHA256}\n"
            f"actual={actual_hash}"
        )

    states: dict[str, dict[str, Any]] = {}
    total_rows = 0

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            total_rows += 1
            state = json.loads(line)

            author = str(state.get("author", ""))
            if author not in AUTHORS:
                continue

            row_id = str(state["row_id"])

            if row_id in states:
                raise RuntimeError(
                    f"Duplicate PV row_id at line {line_number}: {row_id}"
                )

            states[row_id] = state

    if total_rows != PV_DEV_STATES_ROWS:
        raise RuntimeError(
            f"PV Dev state row count changed: "
            f"expected={PV_DEV_STATES_ROWS} actual={total_rows}"
        )

    return states


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}

    if not path.is_file():
        return existing

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            row = json.loads(line)
            row_id = str(row["row_id"])

            if row_id in existing:
                raise RuntimeError(
                    f"Duplicate cached row_id at line {line_number}: {row_id}"
                )

            existing[row_id] = row

    return existing


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "EM-1 Dev recovered-candidate fixed scoring. "
            "Scores personal_only_targets[:5] only."
        )
    )

    parser.add_argument(
        "--generic-cache",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dev-states",
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

    generic = load_generic(args.generic_cache)
    states = load_states(args.dev_states)

    if set(generic) != set(states):
        missing_state = set(generic) - set(states)
        missing_generic = set(states) - set(generic)

        raise RuntimeError(
            "Three-author Generic/PV row IDs differ: "
            f"missing_state={len(missing_state)} "
            f"missing_generic={len(missing_generic)}"
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        args.output_root
        / "recovered_candidate_scores.jsonl"
    )

    existing = load_existing(output_path)

    eligible_rows = []
    incompatible_candidates_total = 0
    rows_with_incompatible_candidates = 0

    def compatible_personal_targets(
        row: dict[str, Any],
        state: dict[str, Any],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        pinyin = tuple(
            str(value)
            for value in row["pinyin_segments"]
        )

        compatible: list[str] = []
        incompatible: list[str] = []

        for target in state.get(
            "personal_only_targets",
            []
        ):
            target = str(target)
            characters = list(target)

            if len(characters) != len(pinyin):
                incompatible.append(target)
                continue

            ids = backend.tokenizer.convert_tokens_to_ids(
                characters
            )

            valid = True

            for token_id, segment in zip(
                ids,
                pinyin,
            ):
                allowed = backend.allowed_token_ids.get(
                    segment,
                    ()
                )

                if token_id not in allowed:
                    valid = False
                    break

            if valid:
                compatible.append(target)
            else:
                incompatible.append(target)

            if len(compatible) >= MAX_K:
                break

        return (
            tuple(compatible[:MAX_K]),
            tuple(incompatible),
        )

    # Compatibility uses the frozen PinyinGPT tokenizer and
    # constrained Pinyin vocabulary, so load the backend before
    # constructing the final recovery pool.


    print(f"Loading PinyinGPT: {args.checkpoint}")
    backend = PinyinGPTConcatBackend(
        args.checkpoint,
        device=args.device,
    )

    for row_id in sorted(generic):
        state = states[row_id]

        targets, incompatible = (
            compatible_personal_targets(
                generic[row_id],
                state,
            )
        )

        if incompatible:
            rows_with_incompatible_candidates += 1
            incompatible_candidates_total += len(
                incompatible
            )

        if targets:
            eligible_rows.append(
                (
                    row_id,
                    generic[row_id],
                    targets,
                )
            )

    eligible_ids = {
        item[0]
        for item in eligible_rows
    }

    stale_cached_ids = (
        set(existing) - eligible_ids
    )

    if stale_cached_ids:
        raise RuntimeError(
            "Existing recovered-score cache contains "
            f"{len(stale_cached_ids)} rows that are no longer "
            "eligible under backend-compatible recovery semantics. "
            "Do not delete automatically; inspect the cache first."
        )

    pending = [
        item
        for item in eligible_rows
        if item[0] not in existing
    ]

    print("EM-1 recovered candidate scoring")
    print("Condition: Full+Short")
    print("Partition: Dev tune")
    print(f"History semantics: H{HISTORY_BUDGET}")
    print(f"Authors: {', '.join(AUTHORS)}")
    print(f"Max recovery K: {MAX_K}")
    print(f"Three-author rows: {len(generic)}")
    print(
        "Rows with at least one personal-only candidate: "
        f"{len(eligible_rows)}"
    )
    print(f"Already cached: {len(existing)}")
    print(f"Pending: {len(pending)}")
    print()

    if not pending:
        print("Nothing to score.")
        return

    print(
        "Rows containing incompatible personal candidates: "
        f"{rows_with_incompatible_candidates}"
    )
    print(
        "Incompatible personal candidates encountered: "
        f"{incompatible_candidates_total}"
    )
    print()

    mode = (
        "a"
        if output_path.is_file()
        and output_path.stat().st_size > 0
        else "w"
    )

    started = time.perf_counter()

    scored_rows = 0
    scored_candidates = 0

    with output_path.open(
        mode,
        encoding="utf-8",
        newline="\n",
    ) as destination:

        for number, (
            row_id,
            row,
            targets,
        ) in enumerate(
            pending,
            start=1,
        ):
            scores = backend.score_candidates(
                context=str(
                    row["model_used_context"]
                ),
                typed_pinyin=tuple(
                    row["pinyin_segments"]
                ),
                candidates=targets,
            )

            score_by_text = {
                score.text: score
                for score in scores
            }

            if set(score_by_text) != set(targets):
                raise RuntimeError(
                    f"{row_id}: scorer returned "
                    "different candidate identities"
                )

            scored = []

            for personal_rank, target in enumerate(
                targets,
                start=1,
            ):
                score = score_by_text[target]

                scored.append(
                    {
                        "candidate": target,
                        "personal_candidate_rank": (
                            personal_rank
                        ),
                        "fixed_log_probability": (
                            float(
                                score.log_probability
                            )
                        ),
                        "fixed_mean_log_probability": (
                            float(
                                score.mean_log_probability
                            )
                        ),
                    }
                )

            output = {
                "schema_version": 1,
                "experiment": (
                    "em1_recovered_candidate_scoring_dev"
                ),
                "row_id": row_id,
                "condition_id": str(
                    row["condition_id"]
                ),
                "anchor_id": str(
                    row["anchor_id"]
                ),
                "author": str(
                    row["author"]
                ),
                "pilot_partition": "tune",
                "pinyin_segments": list(
                    row["pinyin_segments"]
                ),
                "model_used_context_tokens": (
                    row.get(
                        "model_used_context_tokens"
                    )
                ),
                "max_recovery_k": MAX_K,
                "personal_only_available": len(
                    states[row_id].get(
                        "personal_only_targets",
                        []
                    )
                ),
                "scored_candidate_count": len(
                    scored
                ),
                "scores": scored,
            }

            destination.write(
                json.dumps(
                    output,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

            scored_rows += 1
            scored_candidates += len(
                scored
            )

            if (
                number % 25 == 0
                or number == len(pending)
            ):
                destination.flush()

                elapsed = (
                    time.perf_counter()
                    - started
                )

                rate = (
                    number / elapsed
                    if elapsed
                    else 0.0
                )

                print(
                    f"{number}/{len(pending)} "
                    f"rows; "
                    f"candidates={scored_candidates}; "
                    f"rate={rate:.2f} rows/s",
                    flush=True,
                )

    elapsed = (
        time.perf_counter()
        - started
    )

    total_cached = load_existing(
        output_path
    )

    total_candidate_scores = sum(
        int(row["scored_candidate_count"])
        for row in total_cached.values()
    )

    summary = {
        "schema_version": 1,
        "experiment": (
            "em1_recovered_candidate_scoring_dev"
        ),
        "status": "complete",
        "condition": "Full+Short",
        "partition": "dev_tune",
        "authors": list(AUTHORS),
        "history_budget": HISTORY_BUDGET,
        "max_recovery_k": MAX_K,
        "three_author_rows": len(
            generic
        ),
        "eligible_rows": len(
            eligible_rows
        ),
        "cached_rows_total": len(
            total_cached
        ),
        "candidate_scores_total": (
            total_candidate_scores
        ),
        "rows_scored_this_run": (
            scored_rows
        ),
        "candidates_scored_this_run": (
            scored_candidates
        ),
        "elapsed_seconds_this_run": (
            elapsed
        ),
        "generic_cache_sha256": (
            GENERIC_SHA256
        ),
        "pv_dev_states_sha256": (
            PV_DEV_STATES_SHA256
        ),
        "pv_dev_states_rows": (
            PV_DEV_STATES_ROWS
        ),
        "test_rows_used": 0,
        "gold_used_for_scoring": False,
        "old_pv_boundary_score_used": (
            False
        ),
        "fixed_candidate_scoring_used": (
            True
        ),
    }

    summary_path = (
        args.output_root
        / "summary.json"
    )

    with summary_path.open(
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
        "=== Recovered Candidate Scoring Summary ==="
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
        f"Rows scored this run: "
        f"{summary['rows_scored_this_run']}"
    )
    print(
        f"Candidates scored this run: "
        f"{summary['candidates_scored_this_run']}"
    )


if __name__ == "__main__":
    main()


