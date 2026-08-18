from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.personalisation.context_memory import (
    PredictionQuery,
    retrieve_memory,
)
from src.personalisation.pilot_a import (
    EmbeddingCache,
    EmbeddingLookup,
    HistoryIndex,
)


AUTHORS = (
    "Etinjat",
    "Re_spectators",
    "breaddddd",
)

CONDITIONS = (
    "full_short",
    "initial_short",
    "full_multi3",
    "initial_multi3",
)

KS = (1, 3, 5, 10, 20)
HISTORY_BUDGET = 5000

DEEP_ROOT = Path(
    r"C:\Users\chiar\Desktop\LBH\thesis-deep-author"
)

OLD_MATRIX_ROOT = Path(
    r"C:\Users\chiar\Desktop\LBH\thesis-personalisation"
    r"\results\personalisation\reranking_matrix"
)

T1_CONDITIONS = (
    DEEP_ROOT
    / "results/evaluation/deep_author_v2/design/t1_condition_manifest.jsonl"
)

WORK_SPLIT = (
    DEEP_ROOT
    / "results/evaluation/deep_author_v2/design/work_split_manifest.csv"
)

EMBEDDING_CACHE = Path(
    r"C:\Users\chiar\Desktop\LBH\thesis-personalisation"
    r"\results\personalisation\pilot_a_context_memory"
    r"\cache\embedding_cache.sqlite3"
)

OUTPUT_ROOT = Path(
    r"results\personalisation\context_lab"
    r"\diagnostic_a1_retrieval"
)


def read_work_split() -> dict[str, dict[str, Any]]:
    with WORK_SPLIT.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    return {
        str(row["work_id"]): {
            "split": str(row["split"]),
            "chronological_index": int(
                row["chronological_index"]
            ),
        }
        for row in rows
    }


def load_test_rows() -> dict[str, list[dict[str, Any]]]:
    work = read_work_split()

    grouped: dict[str, list[dict[str, Any]]] = {
        condition: []
        for condition in CONDITIONS
    }

    with T1_CONDITIONS.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            value = json.loads(line)

            condition = str(value["condition"])

            if condition not in grouped:
                continue

            split = work[str(value["work_id"])]

            if split["split"] != "test":
                raise RuntimeError(
                    "Frozen T1 condition belongs "
                    "to a non-Test work"
                )

            row = {
                **value,
                "row_id": str(value["condition_id"]),
                "pinyin_segments": str(
                    value["pinyin_input"]
                ).split(),
                "target": str(value["gold"]),
                "source_split": "test",
                "pilot_partition": "test",
                "chronological_position": (
                    int(split["chronological_index"])
                    * 1_000_000_000
                    + int(value["source_position_start"])
                ),
            }

            grouped[condition].append(row)

    for condition, rows in grouped.items():
        counts = Counter(
            str(row["author"])
            for row in rows
        )

        if len(rows) != 6000:
            raise RuntimeError(
                f"{condition}: expected 6000 "
                f"Test rows, got {len(rows)}"
            )

        if any(
            counts[author] != 1000
            for author in AUTHORS
        ):
            raise RuntimeError(
                f"{condition}: selected author "
                "Test counts are not 1000 each"
            )

    return grouped


def load_history(
    condition: str,
) -> list[dict[str, Any]]:
    path = (
        OLD_MATRIX_ROOT
        / "manifests"
        / f"history_{condition}.jsonl"
    )

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            row = json.loads(line)

            if str(row["author"]) in AUTHORS:
                rows.append(row)

    return rows


def query_from_row(
    row: Mapping[str, Any],
) -> PredictionQuery:
    return PredictionQuery(
        row_id=str(row["row_id"]),
        author=str(row["author"]),
        work_id=str(row["work_id"]),
        chronological_position=int(
            row["chronological_position"]
        ),
        context=str(row["context"]),
        pinyin=tuple(row["pinyin_segments"]),
    )


def history_flags(
    visible: Sequence[Mapping[str, Any]],
    gold: str,
) -> dict[str, Any]:
    counts = Counter(
        str(row["target"])
        for row in visible
    )

    ambiguous = len(counts) >= 2

    frequency_winner = None
    frequency_winner_tied = False

    if counts:
        maximum = max(counts.values())

        winners = sorted(
            target
            for target, count in counts.items()
            if count == maximum
        )

        frequency_winner_tied = (
            len(winners) != 1
        )

        if not frequency_winner_tied:
            frequency_winner = winners[0]

    conflict = (
        ambiguous
        and frequency_winner is not None
        and gold != frequency_winner
    )

    return {
        "history_available": bool(visible),
        "visible_history_count": len(visible),
        "distinct_historical_targets": len(counts),
        "gold_history_exists": counts[gold] > 0,
        "gold_history_count": counts[gold],
        "ambiguous": ambiguous,
        "frequency_winner": frequency_winner,
        "frequency_winner_tied": (
            frequency_winner_tied
        ),
        "conflict": conflict,
    }


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    total = len(rows)

    history_count = sum(
        bool(row["history_available"])
        for row in rows
    )

    gold_rows = [
        row
        for row in rows
        if bool(row["gold_history_exists"])
    ]

    result: dict[str, Any] = {
        "rows": total,
        "history_available": history_count,
        "history_available_rate": (
            history_count / total
            if total else None
        ),
        "gold_history_exists": len(gold_rows),
        "gold_history_exists_rate": (
            len(gold_rows) / total
            if total else None
        ),
        "gold_history_exists_given_history": (
            len(gold_rows) / history_count
            if history_count else None
        ),
    }

    for k in KS:
        result[f"recall_at_{k}"] = (
            sum(
                row["gold_retrieval_rank"] is not None
                and int(
                    row["gold_retrieval_rank"]
                ) <= k
                for row in gold_rows
            )
            / len(gold_rows)
            if gold_rows else None
        )

    return result


def subset_rows(
    rows: Sequence[Mapping[str, Any]],
    subset: str,
) -> list[Mapping[str, Any]]:
    if subset == "overall":
        return list(rows)

    if subset == "history_available":
        return [
            row
            for row in rows
            if row["history_available"]
        ]

    if subset == "ambiguous":
        return [
            row
            for row in rows
            if row["ambiguous"]
        ]

    if subset == "conflict":
        return [
            row
            for row in rows
            if row["conflict"]
        ]

    raise ValueError(subset)


def metrics_for_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    subsets = (
        "overall",
        "history_available",
        "ambiguous",
        "conflict",
    )

    output: dict[str, Any] = {}

    for subset in subsets:
        values = subset_rows(rows, subset)

        per_author = {
            author: summarize_rows(
                [
                    row
                    for row in values
                    if row["author"] == author
                ]
            )
            for author in AUTHORS
        }

        micro = summarize_rows(values)

        macro: dict[str, Any] = {
            "authors": len(AUTHORS),
        }

        for metric in (
            "history_available_rate",
            "gold_history_exists_rate",
            "gold_history_exists_given_history",
            *(
                f"recall_at_{k}"
                for k in KS
            ),
        ):
            available = [
                float(value[metric])
                for value in per_author.values()
                if value[metric] is not None
            ]

            macro[metric] = (
                statistics.fmean(available)
                if available
                else None
            )

        output[subset] = {
            "micro": micro,
            "macro_author": macro,
            "per_author": per_author,
        }

    return output


def audit_cache(
    test_by_condition:
        Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    cache = EmbeddingCache(EMBEDDING_CACHE)

    audit: dict[str, Any] = {}

    try:
        for condition in CONDITIONS:
            print(
                f"\n[AUDIT] {condition}",
                flush=True,
            )

            history = load_history(condition)

            index = HistoryIndex(
                history,
                HISTORY_BUDGET,
            )

            test_rows = [
                row
                for row in test_by_condition[
                    condition
                ]
                if row["author"] in AUTHORS
            ]

            required: set[str] = set()

            for number, row in enumerate(
                test_rows,
                start=1,
            ):
                query = query_from_row(row)

                required.add(query.context)

                visible = index.visible(query)

                required.update(
                    str(item["context"])
                    for item in visible
                )

                if (
                    number % 500 == 0
                    or number == len(test_rows)
                ):
                    print(
                        f"  required contexts "
                        f"{number}/{len(test_rows)}",
                        flush=True,
                    )

            missing = [
                context
                for context in required
                if cache.get(context) is None
            ]

            audit[condition] = {
                "test_rows": len(test_rows),
                "selected_authors": list(AUTHORS),
                "history_rows_loaded": len(history),
                "required_unique_contexts": len(
                    required
                ),
                "cache_hits": (
                    len(required) - len(missing)
                ),
                "cache_misses": len(missing),
                "missing_context_examples": [
                    value[-160:]
                    for value in missing[:10]
                ],
            }

            print(
                "  required_unique_contexts =",
                len(required),
            )

            print(
                "  cache_misses =",
                len(missing),
            )

    finally:
        cache.close()

    return audit


def run_condition(
    condition: str,
    test_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    print(
        f"\n[RUN] {condition}",
        flush=True,
    )

    history = load_history(condition)

    index = HistoryIndex(
        history,
        HISTORY_BUDGET,
    )

    selected_test = [
        row
        for row in test_rows
        if row["author"] in AUTHORS
    ]

    cache = EmbeddingCache(EMBEDDING_CACHE)
    embeddings = EmbeddingLookup(cache)

    outputs: list[dict[str, Any]] = []

    started = time.perf_counter()

    try:
        for number, row in enumerate(
            selected_test,
            start=1,
        ):
            query = query_from_row(row)

            visible = index.visible(query)

            flags = history_flags(
                visible,
                str(row["gold"]),
            )

            retrieved = (
                retrieve_memory(
                    query,
                    visible,
                    embeddings,
                )
                if visible
                else ()
            )

            gold_rank = next(
                (
                    rank
                    for rank, item in enumerate(
                        retrieved,
                        start=1,
                    )
                    if str(
                        item["historical_target"]
                    ) == str(row["gold"])
                ),
                None,
            )

            outputs.append(
                {
                    "condition": condition,
                    "row_id": str(row["row_id"]),
                    "anchor_id": str(
                        row["anchor_id"]
                    ),
                    "author": str(row["author"]),
                    "work_id": str(row["work_id"]),
                    "chronological_position": int(
                        row[
                            "chronological_position"
                        ]
                    ),
                    "pinyin_segments": list(
                        row["pinyin_segments"]
                    ),
                    "gold": str(row["gold"]),
                    **flags,
                    "gold_retrieval_rank": (
                        gold_rank
                    ),
                }
            )

            if (
                number % 250 == 0
                or number == len(selected_test)
            ):
                elapsed = (
                    time.perf_counter()
                    - started
                )

                rate = (
                    number / elapsed
                    if elapsed else 0.0
                )

                print(
                    f"  {number}/"
                    f"{len(selected_test)} "
                    f"({rate:.2f} rows/s)",
                    flush=True,
                )

    finally:
        cache.close()

    metrics = metrics_for_rows(outputs)

    return outputs, metrics


def write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        required=True,
        choices=("audit", "run"),
    )

    args = parser.parse_args()

    test_by_condition = load_test_rows()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.phase == "audit":
        audit = audit_cache(
            test_by_condition
        )

        write_json(
            OUTPUT_ROOT
            / "cache_audit.json",
            {
                "status": "complete",
                "research_result": False,
                "history_budget": (
                    HISTORY_BUDGET
                ),
                "authors": list(AUTHORS),
                "conditions": audit,
            },
        )

        print(
            "\n=== CACHE AUDIT SUMMARY ==="
        )

        for condition in CONDITIONS:
            row = audit[condition]

            print(
                condition,
                "test_rows=",
                row["test_rows"],
                "required=",
                row[
                    "required_unique_contexts"
                ],
                "misses=",
                row["cache_misses"],
            )

        return

    audit_path = (
        OUTPUT_ROOT / "cache_audit.json"
    )

    if not audit_path.is_file():
        raise RuntimeError(
            "Run --phase audit first"
        )

    audit = json.loads(
        audit_path.read_text(
            encoding="utf-8"
        )
    )

    misses = {
        condition: int(
            audit["conditions"][
                condition
            ]["cache_misses"]
        )
        for condition in CONDITIONS
    }

    if any(misses.values()):
        raise RuntimeError(
            "Embedding cache is incomplete "
            f"for A1: {misses}"
        )

    all_metrics: dict[str, Any] = {}

    for condition in CONDITIONS:
        rows, metrics = run_condition(
            condition,
            test_by_condition[condition],
        )

        write_jsonl(
            OUTPUT_ROOT
            / f"rows_{condition}.jsonl",
            rows,
        )

        write_json(
            OUTPUT_ROOT
            / f"metrics_{condition}.json",
            metrics,
        )

        all_metrics[condition] = metrics

    write_json(
        OUTPUT_ROOT
        / "summary.json",
        {
            "schema_version": 1,
            "status": "complete",
            "experiment": (
                "context_lab_diagnostic_a1"
            ),
            "population": (
                "Frozen T1 Test; "
                "three exploratory authors"
            ),
            "authors": list(AUTHORS),
            "conditions": list(
                CONDITIONS
            ),
            "history_budget": (
                HISTORY_BUDGET
            ),
            "history_budget_before_pinyin_filter": True,
            "retrieval": (
                "existing BGE context cosine "
                "retrieval"
            ),
            "recall_denominator": (
                "rows where legal visible "
                "Gold-target history exists"
            ),
            "generic_predictions_used": False,
            "test_gold_used_for_parameter_tuning": False,
            "metrics": all_metrics,
        },
    )

    print(
        "\n=== DIAGNOSTIC A1 COMPLETE ==="
    )

    for condition in CONDITIONS:
        overall = all_metrics[
            condition
        ]["overall"]

        print(
            "\n",
            condition,
        )

        print(
            "micro =",
            json.dumps(
                overall["micro"],
                ensure_ascii=False,
            ),
        )

        print(
            "macro_author =",
            json.dumps(
                overall[
                    "macro_author"
                ],
                ensure_ascii=False,
            ),
        )


if __name__ == "__main__":
    main()
