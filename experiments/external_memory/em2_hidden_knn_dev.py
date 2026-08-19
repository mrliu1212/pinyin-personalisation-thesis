"""EM-2C: Frozen PinyinGPT hidden-state kNN retrieval diagnostic.

Dev only.

No tuning is performed.

Primary metric:
    Macro-author Ambiguous R@1,
    conditional on the Gold being present in legal history.

The legal history surface is exactly:
    HistoryIndex(history + dev, H5000).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from src.personalisation.context_memory import (
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

EXPECTED_QUERIES = 5608
EXPECTED_HISTORY_AVAILABLE = 3625
EXPECTED_CACHE_ROWS = 11475
EXPECTED_HIDDEN_SIZE = 768

KS = (1, 5, 10)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as source:
        return [
            json.loads(line)
            for line in source
            if line.strip()
        ]


def row_gold(row: Mapping[str, Any]) -> str:
    if "gold" in row:
        return str(row["gold"])

    if "target" in row:
        return str(row["target"])

    raise RuntimeError(
        f"No Gold field in row {row.get('row_id')}"
    )


def load_vectors(
    path: Path,
) -> dict[str, np.ndarray]:
    connection = sqlite3.connect(path)

    vectors = {}

    try:
        rows = connection.execute(
            """
            SELECT row_id, hidden_size, vector
            FROM hidden_states
            """
        )

        for row_id, hidden_size, blob in rows:
            if int(hidden_size) != EXPECTED_HIDDEN_SIZE:
                raise RuntimeError(
                    f"Unexpected hidden size for {row_id}"
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
                    f"Unexpected vector shape for {row_id}"
                )

            norm = float(
                np.linalg.norm(vector)
            )

            if norm == 0.0:
                raise RuntimeError(
                    f"Zero vector for {row_id}"
                )

            vectors[str(row_id)] = (
                vector / norm
            )

    finally:
        connection.close()

    if len(vectors) != EXPECTED_CACHE_ROWS:
        raise RuntimeError(
            "Frozen EM-2B cache row count differs: "
            f"{len(vectors)}"
        )

    return vectors


def retrieval_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result = {
        "n": len(rows),
    }

    for k in KS:
        if not rows:
            result[f"r@{k}"] = None
        else:
            result[f"r@{k}"] = (
                sum(
                    bool(row[f"hit_at_{k}"])
                    for row in rows
                )
                / len(rows)
            )

    return result


def macro_author(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_author = defaultdict(list)

    for row in rows:
        by_author[
            str(row["author"])
        ].append(row)

    per_author = {
        author: retrieval_metrics(
            by_author.get(author, [])
        )
        for author in AUTHORS
    }

    macro = {}

    for k in KS:
        values = [
            item[f"r@{k}"]
            for item in per_author.values()
            if item[f"r@{k}"] is not None
        ]

        macro[f"r@{k}"] = (
            statistics.fmean(values)
            if values
            else None
        )

    macro["n"] = len(rows)
    macro["authors_with_rows"] = sum(
        bool(by_author.get(author))
        for author in AUTHORS
    )

    return {
        "macro_author": macro,
        "per_author": per_author,
    }


def evaluate_subset(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "micro": retrieval_metrics(rows),
        **macro_author(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pilot-root",
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
        if row.get("pilot_partition") == "tune"
        and str(row["author"]) in AUTHORS
    ]

    if len(tune) != EXPECTED_QUERIES:
        raise RuntimeError(
            f"Dev population changed: {len(tune)}"
        )

    index = HistoryIndex(
        history + dev,
        HISTORY_BUDGET,
    )

    print(
        "Loading EM-2B hidden vectors..."
    )

    vectors = load_vectors(
        args.hidden_cache
    )

    print(
        f"Vectors loaded: {len(vectors)}"
    )
    print()

    outputs = []

    for number, row in enumerate(
        tune,
        start=1,
    ):
        query = PilotRunner._query(row)
        gold = row_gold(row)

        visible = index.visible(query)

        flags = subset_membership(
            query,
            gold,
            visible,
        )

        gold_history_available = any(
            str(item["target"]) == gold
            for item in visible
        )

        retrieved = []

        if visible:
            query_vector = vectors.get(
                query.row_id
            )

            if query_vector is None:
                raise RuntimeError(
                    f"Missing query vector: {query.row_id}"
                )

            for item in visible:
                history_id = str(
                    item["row_id"]
                )

                history_vector = (
                    vectors.get(history_id)
                )

                if history_vector is None:
                    raise RuntimeError(
                        f"Missing history vector: {history_id}"
                    )

                similarity = float(
                    np.dot(
                        query_vector,
                        history_vector,
                    )
                )

                retrieved.append(
                    {
                        "historical_interaction_id": (
                            history_id
                        ),
                        "historical_target": str(
                            item["target"]
                        ),
                        "similarity": similarity,
                        "chronological_position": int(
                            item[
                                "chronological_position"
                            ]
                        ),
                    }
                )

            retrieved.sort(
                key=lambda item: (
                    -float(item["similarity"]),
                    int(
                        item[
                            "chronological_position"
                        ]
                    ),
                    str(
                        item[
                            "historical_interaction_id"
                        ]
                    ),
                )
            )

        result = {
            "row_id": query.row_id,
            "author": query.author,
            "gold": gold,
            "history_available": bool(
                flags["history_available"]
            ),
            "visible_history_count": int(
                flags["visible_history_count"]
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
            "gold_history_available": bool(
                gold_history_available
            ),
            "retrieved_top10": (
                retrieved[:10]
            ),
        }

        for k in KS:
            result[f"hit_at_{k}"] = any(
                item["historical_target"]
                == gold
                for item in retrieved[:k]
            )

        outputs.append(result)

        if (
            number % 500 == 0
            or number == len(tune)
        ):
            print(
                f"EM-2C retrieval: "
                f"{number}/{len(tune)}",
                flush=True,
            )

    history_available = [
        row
        for row in outputs
        if row["history_available"]
    ]

    gold_history = [
        row
        for row in outputs
        if row["gold_history_available"]
    ]

    ambiguous = [
        row
        for row in gold_history
        if row["ambiguous"]
    ]

    conflict = [
        row
        for row in gold_history
        if row["conflict"]
    ]

    if (
        len(history_available)
        != EXPECTED_HISTORY_AVAILABLE
    ):
        raise RuntimeError(
            "History-available population differs: "
            f"{len(history_available)}"
        )

    summary = {
        "schema_version": 1,
        "experiment": (
            "em2c_hidden_knn_dev_retrieval"
        ),
        "partition": "dev_tune_only",
        "condition": "Full+Short",
        "authors": list(AUTHORS),
        "history_budget": HISTORY_BUDGET,
        "representation": (
            "Frozen PinyinGPT final-layer "
            "hidden state at final prompt [SEP]"
        ),
        "similarity": "cosine",
        "retrieval_ks": list(KS),
        "primary_metric": (
            "Macro-author Ambiguous R@1, "
            "conditional on Gold being present "
            "in legal history"
        ),
        "population": {
            "all_queries": len(outputs),
            "history_available": len(
                history_available
            ),
            "gold_history_available": len(
                gold_history
            ),
            "ambiguous_gold_history": len(
                ambiguous
            ),
            "conflict_gold_history": len(
                conflict
            ),
        },
        "retrieval": {
            "overall_gold_history": (
                evaluate_subset(
                    gold_history
                )
            ),
            "ambiguous_gold_history": (
                evaluate_subset(
                    ambiguous
                )
            ),
            "conflict_gold_history": (
                evaluate_subset(
                    conflict
                )
            ),
        },
        "used_test": False,
        "tuning_performed": False,
    }

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        args.output_root
        / "rows.jsonl"
    ).open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as destination:
        for row in outputs:
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

    overall = summary["retrieval"][
        "overall_gold_history"
    ]

    amb = summary["retrieval"][
        "ambiguous_gold_history"
    ]

    con = summary["retrieval"][
        "conflict_gold_history"
    ]

    print()
    print(
        "=== EM-2C Hidden-State kNN "
        "Dev Retrieval ==="
    )

    print()
    print("Population:")
    for key, value in summary[
        "population"
    ].items():
        print(f"  {key}: {value}")

    print()
    print("Overall Gold-History:")
    print(
        "  Micro "
        f"R@1={overall['micro']['r@1']:.6f} "
        f"R@5={overall['micro']['r@5']:.6f} "
        f"R@10={overall['micro']['r@10']:.6f}"
    )
    print(
        "  Macro "
        f"R@1={overall['macro_author']['r@1']:.6f} "
        f"R@5={overall['macro_author']['r@5']:.6f} "
        f"R@10={overall['macro_author']['r@10']:.6f}"
    )

    print()
    print("Ambiguous Gold-History:")
    print(
        "  Micro "
        f"R@1={amb['micro']['r@1']:.6f} "
        f"R@5={amb['micro']['r@5']:.6f} "
        f"R@10={amb['micro']['r@10']:.6f}"
    )
    print(
        "  Macro "
        f"R@1={amb['macro_author']['r@1']:.6f} "
        f"R@5={amb['macro_author']['r@5']:.6f} "
        f"R@10={amb['macro_author']['r@10']:.6f}"
    )

    print()
    print("Conflict Gold-History:")
    print(
        "  Micro "
        f"R@1={con['micro']['r@1']:.6f} "
        f"R@5={con['micro']['r@5']:.6f} "
        f"R@10={con['micro']['r@10']:.6f}"
    )
    print(
        "  Macro "
        f"R@1={con['macro_author']['r@1']:.6f} "
        f"R@5={con['macro_author']['r@5']:.6f} "
        f"R@10={con['macro_author']['r@10']:.6f}"
    )

    print()
    print(
        "Primary metric:"
    )
    print(
        "  Macro-author Ambiguous R@1 = "
        f"{amb['macro_author']['r@1']:.6f}"
    )

    print()
    print("Test used: False")
    print("Tuning performed: False")


if __name__ == "__main__":
    main()
