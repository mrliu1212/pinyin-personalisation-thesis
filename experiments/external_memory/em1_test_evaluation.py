"""Final frozen EM-1 Test evaluation.

Three-author Full+Short / H5000 evaluation.

Frozen on Dev:
- Recovery K = 1
- Frequency lambda = 4

No Test-time parameter selection is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.personalisation.external_memory import (
    rank_of,
    rank_recovery_frequency,
    rank_recovery_only,
    unified_pool,
)
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

FROZEN_K = 1
FROZEN_LAMBDA = 4.0

CONFIGS = (
    "G0",
    "F",
    "R",
    "R+F",
)


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
                f"Duplicate prediction row: {row_id}"
            )

        output[row_id] = row

    if len(output) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} Test rows; "
            f"found {len(output)}"
        )

    return output


def load_states(
    path: Path,
) -> dict[str, dict[str, Any]]:
    actual = sha256_file(path)

    if actual != TEST_STATES_SHA256:
        raise RuntimeError(
            "Frozen Test-state SHA mismatch:\n"
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
            f"Expected {EXPECTED_ROWS} Test states; "
            f"found {len(output)}"
        )

    return output


def load_recovered(
    path: Path,
) -> dict[str, dict[str, Any]]:
    output = {}

    for row in load_jsonl(path):
        row_id = str(row["row_id"])

        if row_id in output:
            raise RuntimeError(
                f"Duplicate recovered-score row: {row_id}"
            )

        scores = row.get("scores", [])

        if len(scores) != 1:
            raise RuntimeError(
                f"Frozen K=1 row {row_id} does not "
                "contain exactly one score"
            )

        output[row_id] = row

    return output


def lexicon_counts(
    state: Mapping[str, Any],
) -> dict[str, int]:
    return {
        str(entry["target"]): int(entry["count"])
        for entry in state.get("lexicon", [])
    }


def subset_flags(
    state: Mapping[str, Any],
    gold: str,
) -> dict[str, bool]:
    counts = lexicon_counts(state)

    history_available = bool(counts)
    ambiguous = len(counts) >= 2

    winner = None

    if counts:
        maximum = max(counts.values())

        winners = sorted(
            target
            for target, count in counts.items()
            if count == maximum
        )

        if len(winners) == 1:
            winner = winners[0]

    conflict = (
        ambiguous
        and winner is not None
        and gold != winner
    )

    return {
        "history_available": history_available,
        "ambiguous": ambiguous,
        "conflict": conflict,
    }


def generic_rank(
    prediction: Mapping[str, Any],
    gold: str,
) -> int | None:
    for row in prediction["top10_candidates"]:
        if str(row["text"]) == gold:
            return int(row["rank"])

    return None


def frequency_rank(
    state: Mapping[str, Any],
    gold: str,
) -> int | None:
    for row in state["generic_frequency_ranked"]:
        if str(row["candidate"]) == gold:
            return int(row["rank"])

    return None


def backend_compatible_gold(
    backend: PinyinGPTConcatBackend,
    state: Mapping[str, Any],
    gold: str,
) -> bool:
    pinyin = tuple(
        str(value)
        for value in state["pinyin"]
    )

    characters = list(gold)

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


def metric_values(
    ranks: Sequence[int | None],
) -> dict[str, Any]:
    if not ranks:
        return {
            "n": 0,
            "top1": None,
            "top3": None,
            "mrr_at_10": None,
            "missing_at_10": None,
        }

    n = len(ranks)

    return {
        "n": n,
        "top1": sum(
            rank == 1
            for rank in ranks
        ) / n,
        "top3": sum(
            rank is not None and rank <= 3
            for rank in ranks
        ) / n,
        "mrr_at_10": sum(
            0.0 if rank is None else 1.0 / rank
            for rank in ranks
        ) / n,
        "missing_at_10": sum(
            rank is None
            for rank in ranks
        ) / n,
    }


def aggregate(
    rows: Sequence[Mapping[str, Any]],
    config: str,
    predicate,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if predicate(row)
    ]

    values = metric_values(
        [
            row["ranks"][config]
            for row in selected
        ]
    )

    author_top1 = {}

    for author in AUTHORS:
        author_rows = [
            row
            for row in selected
            if row["author"] == author
        ]

        if not author_rows:
            continue

        author_top1[author] = (
            sum(
                row["ranks"][config] == 1
                for row in author_rows
            )
            / len(author_rows)
        )

    values["author_top1"] = author_top1

    values["macro_author_top1"] = (
        statistics.fmean(
            author_top1.values()
        )
        if author_top1
        else None
    )

    return values


def transition(
    rows: Sequence[Mapping[str, Any]],
    base: str,
    new: str,
    predicate,
) -> dict[str, int]:
    selected = [
        row
        for row in rows
        if predicate(row)
    ]

    rescue = sum(
        row["ranks"][base] != 1
        and row["ranks"][new] == 1
        for row in selected
    )

    harm = sum(
        row["ranks"][base] == 1
        and row["ranks"][new] != 1
        for row in selected
    )

    return {
        "n": len(selected),
        "rescue": rescue,
        "harm": harm,
        "net": rescue - harm,
    }


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
        "--recovered-scores",
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

    args = parser.parse_args()

    predictions = load_predictions(
        args.predictions
    )

    states = load_states(
        args.test_states
    )

    recovered = load_recovered(
        args.recovered_scores
    )

    if set(predictions) != set(states):
        raise RuntimeError(
            "Prediction/Test-state surfaces differ"
        )

    if not set(recovered).issubset(states):
        raise RuntimeError(
            "Recovered-score cache contains unknown Test rows"
        )

    print(
        "Loading Frozen PinyinGPT for "
        "Gold reachability audit only..."
    )

    backend = PinyinGPTConcatBackend(
        args.checkpoint,
        device="cpu",
    )

    rows = []

    for index, row_id in enumerate(
        sorted(predictions),
        start=1,
    ):
        prediction = predictions[row_id]
        state = states[row_id]

        gold = str(prediction["gold"])

        generic_candidates = (
            prediction["top10_candidates"]
        )

        recovered_row = recovered.get(
            row_id
        )

        recovered_scores = (
            recovered_row["scores"]
            if recovered_row is not None
            else []
        )

        pool = unified_pool(
            generic_candidates,
            recovered_scores,
            k_recovery=FROZEN_K,
        )

        counts = lexicon_counts(
            state
        )

        ranked_r = rank_recovery_only(
            pool
        )

        ranked_rf = rank_recovery_frequency(
            pool,
            counts,
            lambda_frequency=FROZEN_LAMBDA,
        )

        g0_rank = generic_rank(
            prediction,
            gold,
        )

        f_rank = frequency_rank(
            state,
            gold,
        )

        r_rank = rank_of(
            ranked_r,
            gold,
        )

        rf_rank = rank_of(
            ranked_rf,
            gold,
        )

        reachable = backend_compatible_gold(
            backend,
            state,
            gold,
        )

        flags = subset_flags(
            state,
            gold,
        )

        pool_texts = {
            str(row["candidate"])
            for row in pool
        }

        rows.append(
            {
                "row_id": row_id,
                "author": str(
                    prediction["author"]
                ),
                "gold": gold,
                "backend_reachable": reachable,
                **flags,
                "generic_missing": (
                    g0_rank is None
                ),
                "gold_recovered_to_pool": (
                    g0_rank is None
                    and gold in pool_texts
                ),
                "ranks": {
                    "G0": g0_rank,
                    "F": f_rank,
                    "R": r_rank,
                    "R+F": rf_rank,
                },
            }
        )

        if (
            index % 1000 == 0
            or index == EXPECTED_ROWS
        ):
            print(
                f"Evaluated: "
                f"{index}/{EXPECTED_ROWS}"
            )

    subsets = {
        "overall": lambda row: True,
        "backend_reachable": (
            lambda row: row[
                "backend_reachable"
            ]
        ),
        "history_available": (
            lambda row: row[
                "history_available"
            ]
        ),
        "ambiguous": (
            lambda row: row[
                "ambiguous"
            ]
        ),
        "conflict": (
            lambda row: row[
                "conflict"
            ]
        ),
    }

    metrics = {}

    for config in CONFIGS:
        metrics[config] = {}

        for name, predicate in subsets.items():
            metrics[config][name] = (
                aggregate(
                    rows,
                    config,
                    predicate,
                )
            )

    transitions = {}

    for base, new in (
        ("G0", "F"),
        ("G0", "R"),
        ("G0", "R+F"),
        ("F", "R+F"),
    ):
        key = f"{base}_to_{new}"

        transitions[key] = {
            name: transition(
                rows,
                base,
                new,
                predicate,
            )
            for name, predicate in subsets.items()
        }

    reachable_missing = [
        row
        for row in rows
        if row["backend_reachable"]
        and row["generic_missing"]
    ]

    pool_recovered = [
        row
        for row in reachable_missing
        if row["gold_recovered_to_pool"]
    ]

    recovery = {
        "raw_generic_missing": sum(
            row["generic_missing"]
            for row in rows
        ),
        "backend_reachable_generic_missing": len(
            reachable_missing
        ),
        "backend_unreachable_generic_missing": sum(
            row["generic_missing"]
            and not row["backend_reachable"]
            for row in rows
        ),
        "recovered_to_pool": len(
            pool_recovered
        ),
        "recovered_to_pool_rate": (
            len(pool_recovered)
            / len(reachable_missing)
            if reachable_missing
            else 0.0
        ),
        "recovered_to_top10": sum(
            row["ranks"]["R+F"] is not None
            for row in pool_recovered
        ),
        "recovered_to_top3": sum(
            row["ranks"]["R+F"] is not None
            and row["ranks"]["R+F"] <= 3
            for row in pool_recovered
        ),
        "recovered_to_top1": sum(
            row["ranks"]["R+F"] == 1
            for row in pool_recovered
        ),
    }

    summary = {
        "schema_version": 1,
        "experiment": "em1_frozen_test_evaluation",
        "condition": "Full+Short",
        "partition": "test",
        "history_budget": 5000,
        "authors": list(AUTHORS),
        "rows": EXPECTED_ROWS,
        "frozen_configuration": {
            "recovery_k": FROZEN_K,
            "lambda_frequency": (
                FROZEN_LAMBDA
            ),
            "selected_on": (
                "three-author Dev Macro-author "
                "Overall Top1"
            ),
        },
        "used_test_for_parameter_tuning": False,
        "metrics": metrics,
        "transitions": transitions,
        "recovery": recovery,
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
            "recovered_scores_sha256": (
                sha256_file(
                    args.recovered_scores
                )
            ),
        },
    }

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    with (
        args.output_root
        / "rows.jsonl"
    ).open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as destination:
        for row in rows:
            destination.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    print()
    print("=== EM-1 Frozen Test Evaluation ===")
    print(
        "Config      Micro Top1   Macro Top1   "
        "Top3       MRR@10   Missing@10"
    )
    print("-" * 72)

    for config in CONFIGS:
        value = metrics[
            config
        ]["overall"]

        print(
            f"{config:<10}"
            f"{100 * value['top1']:>9.3f}%   "
            f"{100 * value['macro_author_top1']:>9.3f}%   "
            f"{100 * value['top3']:>7.3f}%   "
            f"{value['mrr_at_10']:>7.4f}   "
            f"{100 * value['missing_at_10']:>8.3f}%"
        )

    print()
    print("Per-author Top1:")

    for config in CONFIGS:
        values = metrics[
            config
        ]["overall"]["author_top1"]

        rendered = " | ".join(
            f"{author}={100 * values[author]:.3f}%"
            for author in AUTHORS
        )

        print(
            f"  {config}: {rendered}"
        )

    print()
    print("Recovery:")
    print(
        "  Raw Generic Missing: "
        f"{recovery['raw_generic_missing']}"
    )
    print(
        "  Reachable Generic Missing: "
        f"{recovery['backend_reachable_generic_missing']}"
    )
    print(
        "  Unreachable Generic Missing: "
        f"{recovery['backend_unreachable_generic_missing']}"
    )
    print(
        "  Recovered to pool: "
        f"{recovery['recovered_to_pool']}"
    )
    print(
        "  Recovered to Top10: "
        f"{recovery['recovered_to_top10']}"
    )
    print(
        "  Recovered to Top3: "
        f"{recovery['recovered_to_top3']}"
    )
    print(
        "  Recovered to Top1: "
        f"{recovery['recovered_to_top1']}"
    )

    print()
    print("Incremental F -> R+F:")

    for name in (
        "overall",
        "history_available",
        "ambiguous",
        "conflict",
    ):
        value = transitions[
            "F_to_R+F"
        ][name]

        print(
            f"  {name}: "
            f"rescue={value['rescue']} "
            f"harm={value['harm']} "
            f"net={value['net']}"
        )

    print()
    print(
        "Frozen Test evaluation complete. "
        "No Test retuning permitted."
    )


if __name__ == "__main__":
    main()
