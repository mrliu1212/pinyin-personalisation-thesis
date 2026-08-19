"""EM-2E1 Hidden-M1 Dev grid.

Controlled ablation:
    original M1 logic
    with BGE retrieval representation replaced by
    Frozen PinyinGPT hidden-state retrieval.

Dev only. No Test.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from src.personalisation.context_memory import (
    macro_author_metrics,
    metric_values,
    rank_from_retrieved,
    rank_of,
    subset_membership,
)
from src.personalisation.pilot_a import (
    HistoryIndex,
    PilotRunner,
)


AUTHORS = (
    "Etinjat",
    "Re_spectators",
    "breaddddd",
)

HISTORY_BUDGET = 5000

TOP_NS = (1, 3, 5, 10, 20)

LAMBDAS = (
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
)

EXPECTED_TUNE = 5608
EXPECTED_HISTORY_AVAILABLE = 3625
EXPECTED_HIDDEN_ROWS = 11475
EXPECTED_HIDDEN_SIZE = 768

EXPECTED_GENERIC_SHA256 = (
    "588aa84c6397e8cb1a13576c0d5dfecd"
    "9dd2c4305b45be351328dd83ef62007d"
)

EXPECTED_HIDDEN_SHA256 = (
    "9a80a3314c184ccf3f0540916203c651"
    "474fad162dc3dab1fc97f7451f441df1"
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"Missing file: {path}")

    with path.open(
        "r",
        encoding="utf-8",
    ) as source:
        return [
            json.loads(line)
            for line in source
            if line.strip()
        ]


def gold_of(row: Mapping[str, Any]) -> str:
    if "gold" in row:
        return str(row["gold"])

    if "target" in row:
        return str(row["target"])

    raise RuntimeError(
        f"No Gold field for {row.get('row_id')}"
    )


def load_generic(
    path: Path,
    expected_ids: set[str],
) -> dict[str, dict[str, Any]]:
    actual = sha256_file(path)

    if actual != EXPECTED_GENERIC_SHA256:
        raise RuntimeError(
            "Frozen Generic Dev cache SHA differs:\n"
            f"expected={EXPECTED_GENERIC_SHA256}\n"
            f"actual={actual}"
        )

    result = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as source:
        for line in source:
            if not line.strip():
                continue

            row = json.loads(line)

            matched_id = None

            for key in (
                "row_id",
                "condition_id",
            ):
                value = row.get(key)

                if (
                    value is not None
                    and str(value) in expected_ids
                ):
                    matched_id = str(value)
                    break

            if matched_id is None:
                continue

            if matched_id in result:
                raise RuntimeError(
                    f"Duplicate Generic row: {matched_id}"
                )

            result[matched_id] = row

    missing = expected_ids - set(result)

    if missing:
        raise RuntimeError(
            "Frozen Generic cache is incomplete for "
            f"three-author tune surface: {len(missing)} missing"
        )

    if len(result) != EXPECTED_TUNE:
        raise RuntimeError(
            f"Unexpected Generic surface: {len(result)}"
        )

    return result


def load_hidden(
    path: Path,
) -> dict[str, np.ndarray]:
    actual = sha256_file(path)

    if actual != EXPECTED_HIDDEN_SHA256:
        raise RuntimeError(
            "Frozen EM-2B hidden cache SHA differs:\n"
            f"expected={EXPECTED_HIDDEN_SHA256}\n"
            f"actual={actual}"
        )

    connection = sqlite3.connect(
        f"file:{path.resolve()}?mode=ro",
        uri=True,
    )

    vectors = {}

    try:
        for (
            row_id,
            hidden_size,
            blob,
        ) in connection.execute(
            """
            SELECT row_id, hidden_size, vector
            FROM hidden_states
            """
        ):
            if int(hidden_size) != EXPECTED_HIDDEN_SIZE:
                raise RuntimeError(
                    f"Hidden size differs for {row_id}"
                )

            vector = np.frombuffer(
                blob,
                dtype="<f4",
            ).astype(
                np.float32,
                copy=True,
            )

            if vector.shape != (
                EXPECTED_HIDDEN_SIZE,
            ):
                raise RuntimeError(
                    f"Bad hidden shape for {row_id}: "
                    f"{vector.shape}"
                )

            norm = float(
                np.linalg.norm(vector)
            )

            if norm == 0.0:
                raise RuntimeError(
                    f"Zero hidden vector: {row_id}"
                )

            vectors[str(row_id)] = (
                vector / norm
            )

    finally:
        connection.close()

    if len(vectors) != EXPECTED_HIDDEN_ROWS:
        raise RuntimeError(
            f"Hidden cache surface changed: {len(vectors)}"
        )

    return vectors


def retrieve_hidden(
    query: Any,
    visible: Sequence[Mapping[str, Any]],
    vectors: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], ...]:
    query_vector = vectors.get(
        query.row_id
    )

    if query_vector is None:
        raise RuntimeError(
            f"Missing query hidden vector: {query.row_id}"
        )

    retrieved = []

    for history in visible:
        history_id = str(
            history["row_id"]
        )

        history_vector = vectors.get(
            history_id
        )

        if history_vector is None:
            raise RuntimeError(
                f"Missing historical vector: {history_id}"
            )

        similarity = float(
            np.dot(
                query_vector,
                history_vector,
            )
        )

        retrieved.append(
            {
                "historical_interaction_id": history_id,
                "historical_target": str(
                    history["target"]
                ),
                "similarity": similarity,
                "weight": max(
                    similarity,
                    0.0,
                ),
                "chronological_position": int(
                    history[
                        "chronological_position"
                    ]
                ),
            }
        )

    retrieved.sort(
        key=lambda row: (
            -float(row["similarity"]),
            int(
                row["chronological_position"]
            ),
            str(
                row[
                    "historical_interaction_id"
                ]
            ),
        )
    )

    return tuple(retrieved)


def micro_macro(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ranks = [
        row.get("rank")
        for row in rows
    ]

    return {
        "micro": metric_values(ranks),
        **macro_author_metrics(
            rows,
            "rank",
        ),
    }


def subset_rows(
    rows: Sequence[Mapping[str, Any]],
    name: str,
) -> list[Mapping[str, Any]]:
    if name == "overall":
        return list(rows)

    key = {
        "history_available": "history_available",
        "ambiguous": "ambiguous",
        "conflict": "conflict",
    }[name]

    return [
        row
        for row in rows
        if bool(row[key])
    ]


def full_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        name: micro_macro(
            subset_rows(
                rows,
                name,
            )
        )
        for name in (
            "overall",
            "history_available",
            "ambiguous",
            "conflict",
        )
    }


def decision_delta(
    baseline: Sequence[Mapping[str, Any]],
    method: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = {
        str(row["row_id"]): row
        for row in baseline
    }

    method_by_id = {
        str(row["row_id"]): row
        for row in method
    }

    result = {}

    for subset in (
        "overall",
        "history_available",
        "ambiguous",
        "conflict",
    ):
        ids = [
            str(row["row_id"])
            for row in baseline
            if (
                subset == "overall"
                or bool(row[subset])
            )
        ]

        rescue = 0
        harm = 0

        for row_id in ids:
            before = (
                baseline_by_id[row_id]
                .get("rank")
                == 1
            )

            after = (
                method_by_id[row_id]
                .get("rank")
                == 1
            )

            if not before and after:
                rescue += 1

            if before and not after:
                harm += 1

        result[subset] = {
            "n": len(ids),
            "rescue": rescue,
            "harm": harm,
            "net": rescue - harm,
        }

    return result


def evaluate(
    states: Sequence[Mapping[str, Any]],
    *,
    top_n: int,
    lambda_hidden: float,
    save_evidence: bool = False,
) -> list[dict[str, Any]]:
    outputs = []

    for state in states:
        selected = tuple(
            state["retrieved"][:top_n]
        )

        ranked = rank_from_retrieved(
            state["candidates"],
            selected,
            lambda_memory=lambda_hidden,
        )

        result = {
            "row_id": state["row_id"],
            "author": state["author"],
            "gold": state["gold"],
            "rank": rank_of(
                ranked,
                state["gold"],
            ),
            "history_available": state[
                "history_available"
            ],
            "ambiguous": state[
                "ambiguous"
            ],
            "conflict": state[
                "conflict"
            ],
        }

        if save_evidence:
            result[
                "visible_history_count"
            ] = state[
                "visible_history_count"
            ]

            result[
                "distinct_historical_targets"
            ] = state[
                "distinct_historical_targets"
            ]

            result[
                "selected_history"
            ] = [
                dict(item)
                for item in selected
            ]

            result[
                "ranking"
            ] = [
                dict(item)
                for item in ranked
            ]

        outputs.append(result)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pilot-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--generic-cache",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--hidden-cache",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    history = read_jsonl(
        args.pilot_root
        / "history_manifest.jsonl"
    )

    dev = read_jsonl(
        args.pilot_root
        / "dev_manifest.jsonl"
    )

    if any(
        row.get("source_split") == "test"
        for row in history + dev
    ):
        raise RuntimeError(
            "STOP: Test row detected."
        )

    tune = [
        row
        for row in dev
        if (
            row.get("pilot_partition")
            == "tune"
            and str(row["author"])
            in AUTHORS
        )
    ]

    if len(tune) != EXPECTED_TUNE:
        raise RuntimeError(
            f"Dev tune surface changed: {len(tune)}"
        )

    tune_ids = {
        str(row["row_id"])
        for row in tune
    }

    print("Loading Frozen Generic Dev cache...")

    generic = load_generic(
        args.generic_cache,
        tune_ids,
    )

    print(
        f"Generic rows loaded: {len(generic)}"
    )

    print("Loading Frozen EM-2 hidden cache...")

    vectors = load_hidden(
        args.hidden_cache
    )

    print(
        f"Hidden vectors loaded: {len(vectors)}"
    )
    print()

    history_index = HistoryIndex(
        history + dev,
        HISTORY_BUDGET,
    )

    states = []

    history_available_count = 0

    for number, row in enumerate(
        tune,
        start=1,
    ):
        query = PilotRunner._query(row)
        visible = history_index.visible(
            query
        )

        if visible:
            history_available_count += 1

        gold = gold_of(row)

        flags = subset_membership(
            query,
            gold,
            visible,
        )

        generic_row = generic[
            query.row_id
        ]

        candidates = (
            PilotRunner._candidates(
                generic_row
            )
        )

        retrieved = (
            retrieve_hidden(
                query,
                visible,
                vectors,
            )
            if visible
            else ()
        )

        states.append(
            {
                "row_id": query.row_id,
                "author": query.author,
                "gold": gold,
                "candidates": candidates,
                "retrieved": retrieved,
                "history_available": bool(
                    flags[
                        "history_available"
                    ]
                ),
                "visible_history_count": int(
                    flags[
                        "visible_history_count"
                    ]
                ),
                "distinct_historical_targets": int(
                    flags[
                        "distinct_historical_targets"
                    ]
                ),
                "ambiguous": bool(
                    flags["ambiguous"]
                ),
                "conflict": bool(
                    flags["conflict"]
                ),
            }
        )

        if (
            number % 500 == 0
            or number == len(tune)
        ):
            print(
                f"Hidden-M1 state prep: "
                f"{number}/{len(tune)}",
                flush=True,
            )

    if (
        history_available_count
        != EXPECTED_HISTORY_AVAILABLE
    ):
        raise RuntimeError(
            "History-available population changed: "
            f"{history_available_count}"
        )

    print()
    print("Running Hidden-M1 Dev grid...")

    grid = []

    best = None

    for top_n in TOP_NS:
        for lambda_hidden in LAMBDAS:
            evaluated = evaluate(
                states,
                top_n=top_n,
                lambda_hidden=lambda_hidden,
            )

            macro_top1 = (
                macro_author_metrics(
                    evaluated,
                    "rank",
                )[
                    "macro_author"
                ]["top1"]
            )

            micro_top1 = (
                metric_values(
                    [
                        row["rank"]
                        for row in evaluated
                    ]
                )["top1"]
            )

            result = {
                "top_n": top_n,
                "lambda_hidden": (
                    lambda_hidden
                ),
                "macro_author_overall_top1": (
                    macro_top1
                ),
                "micro_overall_top1": (
                    micro_top1
                ),
            }

            grid.append(result)

            print(
                f"TopN={top_n:>2} "
                f"lambda={lambda_hidden:>4} "
                f"MacroTop1="
                f"{macro_top1:.6f}",
                flush=True,
            )

            key = (
                -float(macro_top1),
                float(lambda_hidden),
                int(top_n),
            )

            if (
                best is None
                or key < best[0]
            ):
                best = (
                    key,
                    result,
                )

    assert best is not None

    selected = dict(best[1])

    selected_top_n = int(
        selected["top_n"]
    )

    selected_lambda = float(
        selected["lambda_hidden"]
    )

    generic_rows = evaluate(
        states,
        top_n=1,
        lambda_hidden=0.0,
    )

    selected_rows = evaluate(
        states,
        top_n=selected_top_n,
        lambda_hidden=selected_lambda,
        save_evidence=True,
    )

    generic_metrics = full_metrics(
        generic_rows
    )

    hidden_metrics = full_metrics(
        selected_rows
    )

    delta = decision_delta(
        generic_rows,
        selected_rows,
    )

    boundary_check_required = (
        selected_lambda
        == max(LAMBDAS)
    )

    summary = {
        "schema_version": 1,
        "experiment": (
            "em2e1_hidden_m1_dev"
        ),
        "partition": "dev_tune_only",
        "condition": "Full+Short",
        "authors": list(AUTHORS),
        "history_budget": (
            HISTORY_BUDGET
        ),
        "representation": (
            "Frozen PinyinGPT final-layer "
            "hidden state at final prompt [SEP]"
        ),
        "retrieval": "cosine",
        "aggregation": (
            "original M1 positive-similarity "
            "Top-N target support"
        ),
        "candidate_surface": (
            "Frozen Generic Top10 only"
        ),
        "grid": {
            "top_n": list(TOP_NS),
            "lambda_hidden": list(
                LAMBDAS
            ),
        },
        "primary_metric": (
            "Macro-author Overall Top1"
        ),
        "tie_break": (
            "lower lambda_hidden, then lower Top-N"
        ),
        "selected": selected,
        "boundary_check_lambda8_required": (
            boundary_check_required
        ),
        "population": {
            "queries": len(states),
            "history_available": (
                history_available_count
            ),
            "ambiguous": sum(
                row["ambiguous"]
                for row in states
            ),
            "conflict": sum(
                row["conflict"]
                for row in states
            ),
        },
        "generic": generic_metrics,
        "hidden_m1": hidden_metrics,
        "generic_to_hidden_m1": (
            delta
        ),
        "used_test": False,
        "frequency_used": False,
        "recovery_used": False,
        "provenance": {
            "generic_cache_sha256": (
                sha256_file(
                    args.generic_cache
                )
            ),
            "hidden_cache_sha256": (
                sha256_file(
                    args.hidden_cache
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
        / "grid.json"
    ).open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as destination:
        json.dump(
            grid,
            destination,
            ensure_ascii=False,
            indent=2,
        )
        destination.write("\n")

    with (
        args.output_root
        / "selected_rows.jsonl"
    ).open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as destination:
        for row in selected_rows:
            destination.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
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

    print()
    print(
        "=== EM-2E1 Hidden-M1 Dev Selection ==="
    )
    print(
        f"Selected Top-N: "
        f"{selected_top_n}"
    )
    print(
        f"Selected lambda_hidden: "
        f"{selected_lambda}"
    )
    print(
        "Primary Macro Overall Top1: "
        f"{selected['macro_author_overall_top1']:.6f}"
    )

    print()
    print("Macro-author Top1:")

    for subset in (
        "overall",
        "history_available",
        "ambiguous",
        "conflict",
    ):
        g = (
            generic_metrics[
                subset
            ]["macro_author"]["top1"]
        )

        h = (
            hidden_metrics[
                subset
            ]["macro_author"]["top1"]
        )

        print(
            f"  {subset:18s} "
            f"G0={g:.6f} "
            f"Hidden-M1={h:.6f} "
            f"delta={(h-g):+.6f}"
        )

    print()
    print("G0 -> Hidden-M1 decisions:")

    for subset in (
        "overall",
        "history_available",
        "ambiguous",
        "conflict",
    ):
        value = delta[subset]

        print(
            f"  {subset:18s} "
            f"rescue={value['rescue']} "
            f"harm={value['harm']} "
            f"net={value['net']:+d}"
        )

    print()
    print(
        "Lambda=8 boundary check required: "
        f"{boundary_check_required}"
    )
    print("Test used: False")
    print("Frequency used: False")
    print("Recovery used: False")


if __name__ == "__main__":
    main()
