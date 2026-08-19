"""Fixed G+F+C fusion on the EM-2 three-author Dev tune surface.

G = Frozen Generic z-score
F = exact normalized log-frequency support
C = Frozen PinyinGPT-hidden Top-3 context support

No Test.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.external_memory import em2_hidden_m1_dev as em2

from src.personalisation.context_memory import (
    frequency_support,
    macro_author_metrics,
    normalize_generic_scores,
    rank_of,
    subset_membership,
)

from src.personalisation.pilot_a import (
    HistoryIndex,
    PilotRunner,
)


AUTHORS = em2.AUTHORS
HISTORY_BUDGET = 5000

# Frozen by Hidden-M1.
HIDDEN_TOP_N = 3

# Registered before Fixed Fusion results.
WEIGHTS = (
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
)

EXPECTED_QUERIES = 5608


def context_support(
    retrieved: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    selected = tuple(
        retrieved[:HIDDEN_TOP_N]
    )

    total = sum(
        float(row["weight"])
        for row in selected
    )

    support: dict[str, float] = defaultdict(float)

    if total:
        for row in selected:
            support[
                str(row["historical_target"])
            ] += (
                float(row["weight"])
                / total
            )

    return dict(support)


def fusion_rank(
    candidates,
    frequency: Mapping[str, float],
    context: Mapping[str, float],
    *,
    lambda_f: float,
    lambda_c: float,
):
    generic = normalize_generic_scores(
        candidates
    )

    rows = []

    for candidate, g in zip(
        candidates,
        generic,
    ):
        f = float(
            frequency.get(
                candidate.text,
                0.0,
            )
        )

        c = float(
            context.get(
                candidate.text,
                0.0,
            )
        )

        rows.append(
            {
                "candidate":
                    candidate.text,
                "generic_rank":
                    candidate.generic_rank,
                "generic_score":
                    candidate.generic_score,
                "normalized_generic_score":
                    float(g),
                "frequency_support": f,
                "context_support": c,
                "final_score":
                    float(g)
                    + lambda_f * f
                    + lambda_c * c,
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row["final_score"]),
            int(row["generic_rank"]),
        )
    )

    for rank, row in enumerate(
        rows,
        start=1,
    ):
        row["rank"] = rank

    return tuple(rows)


def subset_rows(rows, name):
    if name == "overall":
        return list(rows)

    return [
        row
        for row in rows
        if bool(row[name])
    ]


def macro_top1(rows, name):
    return macro_author_metrics(
        subset_rows(rows, name),
        "rank",
    )["macro_author"]["top1"]


def decision_delta(before, after, subset):
    before_by_id = {
        row["row_id"]: row
        for row in before
    }

    after_by_id = {
        row["row_id"]: row
        for row in after
    }

    selected = subset_rows(
        before,
        subset,
    )

    rescue = 0
    harm = 0

    for row in selected:
        row_id = row["row_id"]

        b = (
            before_by_id[row_id]["rank"]
            == 1
        )

        a = (
            after_by_id[row_id]["rank"]
            == 1
        )

        if not b and a:
            rescue += 1

        if b and not a:
            harm += 1

    return {
        "n": len(selected),
        "rescue": rescue,
        "harm": harm,
        "net": rescue - harm,
    }


def prediction_visible_features(
    visible,
    retrieved,
):
    counts = Counter(
        str(row["target"])
        for row in visible
    )

    ordered_counts = sorted(
        counts.values(),
        reverse=True,
    )

    n = len(visible)

    winner_share = (
        ordered_counts[0] / n
        if n
        else 0.0
    )

    second = (
        ordered_counts[1]
        if len(ordered_counts) >= 2
        else 0
    )

    frequency_margin = (
        (
            ordered_counts[0]
            - second
        ) / n
        if n
        else 0.0
    )

    selected = tuple(
        retrieved[:HIDDEN_TOP_N]
    )

    similarities = [
        float(row["similarity"])
        for row in selected
    ]

    top1_similarity = (
        similarities[0]
        if similarities
        else None
    )

    similarity_margin = (
        similarities[0]
        - similarities[1]
        if len(similarities) >= 2
        else None
    )

    retrieved_counts = Counter(
        str(row["historical_target"])
        for row in selected
    )

    agreement = (
        max(
            retrieved_counts.values()
        ) / len(selected)
        if selected
        else None
    )

    return {
        "visible_history_count": n,
        "distinct_target_count":
            len(counts),
        "frequency_winner_share":
            winner_share,
        "frequency_margin":
            frequency_margin,
        "retrieval_top1_similarity":
            top1_similarity,
        "retrieval_similarity_margin":
            similarity_margin,
        "retrieved_target_agreement":
            agreement,
    }


def main():
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

    history = em2.read_jsonl(
        args.pilot_root
        / "history_manifest.jsonl"
    )

    dev = em2.read_jsonl(
        args.pilot_root
        / "dev_manifest.jsonl"
    )

    if any(
        str(row.get("source_split"))
        == "test"
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

    if len(tune) != EXPECTED_QUERIES:
        raise RuntimeError(
            f"Unexpected tune size: "
            f"{len(tune)}"
        )

    tune_ids = {
        str(row["row_id"])
        for row in tune
    }

    print("Loading Frozen Generic...")

    generic = em2.load_generic(
        args.generic_cache,
        tune_ids,
    )

    print("Loading Frozen Hidden states...")

    vectors = em2.load_hidden(
        args.hidden_cache
    )

    index = HistoryIndex(
        history + dev,
        HISTORY_BUDGET,
    )

    states = []

    for number, row in enumerate(
        tune,
        start=1,
    ):
        query = PilotRunner._query(row)

        visible = index.visible(
            query
        )

        gold = em2.gold_of(row)

        candidates = (
            PilotRunner._candidates(
                generic[query.row_id]
            )
        )

        retrieved = (
            em2.retrieve_hidden(
                query,
                visible,
                vectors,
            )
            if visible
            else ()
        )

        _, f_support = frequency_support(
            [
                candidate.text
                for candidate in candidates
            ],
            visible,
        )

        c_support = context_support(
            retrieved
        )

        flags = subset_membership(
            query,
            gold,
            visible,
        )

        states.append(
            {
                "row_id": query.row_id,
                "author": query.author,
                "gold": gold,
                "candidates": candidates,
                "frequency_support":
                    f_support,
                "context_support":
                    c_support,
                "history_available":
                    bool(
                        flags[
                            "history_available"
                        ]
                    ),
                "ambiguous":
                    bool(
                        flags["ambiguous"]
                    ),
                "conflict":
                    bool(
                        flags["conflict"]
                    ),
                "features":
                    prediction_visible_features(
                        visible,
                        retrieved,
                    ),
            }
        )

        if (
            number % 500 == 0
            or number == len(tune)
        ):
            print(
                f"GFC state prep: "
                f"{number}/{len(tune)}",
                flush=True,
            )

    def evaluate(
        lambda_f,
        lambda_c,
        *,
        save_ranking=False,
    ):
        outputs = []

        for state in states:
            ranked = fusion_rank(
                state["candidates"],
                state[
                    "frequency_support"
                ],
                state[
                    "context_support"
                ],
                lambda_f=lambda_f,
                lambda_c=lambda_c,
            )

            output = {
                "row_id":
                    state["row_id"],
                "author":
                    state["author"],
                "rank": rank_of(
                    ranked,
                    state["gold"],
                ),
                "history_available":
                    state[
                        "history_available"
                    ],
                "ambiguous":
                    state["ambiguous"],
                "conflict":
                    state["conflict"],
            }

            if save_ranking:
                output.update(
                    state["features"]
                )

                output["ranking"] = [
                    dict(row)
                    for row in ranked
                ]

            outputs.append(output)

        return outputs

    print()
    print("Running Fixed G+F+C grid...")

    grid = []
    rows_by_grid = {}

    for lambda_f in WEIGHTS:
        for lambda_c in WEIGHTS:
            rows = evaluate(
                lambda_f,
                lambda_c,
            )

            rows_by_grid[
                (
                    float(lambda_f),
                    float(lambda_c),
                )
            ] = rows

            value = macro_top1(
                rows,
                "overall",
            )

            grid.append(
                {
                    "lambda_f":
                        float(lambda_f),
                    "lambda_c":
                        float(lambda_c),
                    "macro_author_overall_top1":
                        value,
                    "is_true_fusion":
                        (
                            lambda_f > 0
                            and lambda_c > 0
                        ),
                }
            )

            print(
                f"lambdaF={lambda_f:>4} "
                f"lambdaC={lambda_c:>4} "
                f"MacroTop1={value:.6f}",
                flush=True,
            )

    # Primary fixed-fusion selection:
    # both personal signals must be active.
    fusion_candidates = [
        row
        for row in grid
        if row["is_true_fusion"]
    ]

    selected = max(
        fusion_candidates,
        key=lambda row: (
            float(
                row[
                    "macro_author_overall_top1"
                ]
            ),
            -(
                float(row["lambda_f"])
                + float(row["lambda_c"])
            ),
            -float(row["lambda_c"]),
            -float(row["lambda_f"]),
        ),
    )

    # Diagnostic best point even if it collapses
    # to G/F/C-only.
    best_any = max(
        grid,
        key=lambda row: (
            float(
                row[
                    "macro_author_overall_top1"
                ]
            ),
            -(
                float(row["lambda_f"])
                + float(row["lambda_c"])
            ),
            -float(row["lambda_c"]),
            -float(row["lambda_f"]),
        ),
    )

    selected_key = (
        float(selected["lambda_f"]),
        float(selected["lambda_c"]),
    )

    # Re-evaluate selected once with detailed output.
    selected_rows = evaluate(
        *selected_key,
        save_ranking=True,
    )

    # Exact axis controls.
    g_rows = rows_by_grid[
        (0.0, 0.0)
    ]

    f_rows = rows_by_grid[
        (4.0, 0.0)
    ]

    hidden_rows = rows_by_grid[
        (0.0, 4.0)
    ]

    equal44_rows = rows_by_grid[
        (4.0, 4.0)
    ]

    methods = {
        "G": g_rows,
        "F": f_rows,
        "Hidden-M1": hidden_rows,
        "GFC-4-4": equal44_rows,
        "GFC-selected":
            selected_rows,
    }

    print()
    print(
        "=== FIXED G+F+C DEV SELECTION ==="
    )

    print(
        "Selected lambda_F:",
        selected["lambda_f"],
    )

    print(
        "Selected lambda_C:",
        selected["lambda_c"],
    )

    print(
        "Best-any point:",
        f"F={best_any['lambda_f']} "
        f"C={best_any['lambda_c']} "
        f"Top1="
        f"{best_any['macro_author_overall_top1']:.6f}"
    )

    print()
    print(
        f"{'Method':15s}"
        f"{'Overall':>12s}"
        f"{'History':>12s}"
        f"{'Ambiguous':>12s}"
        f"{'Conflict':>12s}"
    )

    metrics = {}

    for method, rows in methods.items():
        values = {
            subset: macro_top1(
                rows,
                subset,
            )
            for subset in (
                "overall",
                "history_available",
                "ambiguous",
                "conflict",
            )
        }

        metrics[method] = values

        print(
            f"{method:15s}"
            f"{values['overall']:12.6f}"
            f"{values['history_available']:12.6f}"
            f"{values['ambiguous']:12.6f}"
            f"{values['conflict']:12.6f}"
        )

    print()
    print("=== F -> SELECTED GFC ===")

    f_delta = {}

    for subset in (
        "overall",
        "history_available",
        "ambiguous",
        "conflict",
    ):
        value = decision_delta(
            f_rows,
            selected_rows,
            subset,
        )

        f_delta[subset] = value

        print(
            f"{subset:18s} "
            f"rescue={value['rescue']} "
            f"harm={value['harm']} "
            f"net={value['net']:+d}"
        )

    print()
    print(
        "=== HIDDEN-M1 -> SELECTED GFC ==="
    )

    h_delta = {}

    for subset in (
        "overall",
        "history_available",
        "ambiguous",
        "conflict",
    ):
        value = decision_delta(
            hidden_rows,
            selected_rows,
            subset,
        )

        h_delta[subset] = value

        print(
            f"{subset:18s} "
            f"rescue={value['rescue']} "
            f"harm={value['harm']} "
            f"net={value['net']:+d}"
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "status":
            "dev_selection_complete",
        "experiment":
            "fixed_g_f_hidden_c",
        "partition":
            "dev_tune_only",
        "authors":
            list(AUTHORS),
        "queries":
            len(states),
        "history_budget":
            HISTORY_BUDGET,
        "hidden_top_n":
            HIDDEN_TOP_N,
        "weight_grid":
            list(WEIGHTS),
        "primary_selection":
            "Macro-author Overall Top1 "
            "among lambda_F>0 and lambda_C>0",
        "selected":
            selected,
        "best_any":
            best_any,
        "metrics":
            metrics,
        "f_to_selected":
            f_delta,
        "hidden_to_selected":
            h_delta,
        "test_used":
            False,
        "gold_used_for_scoring":
            False,
        "future_adaptive_features_frozen_before_result":
            [
                "visible_history_count",
                "distinct_target_count",
                "frequency_winner_share",
                "frequency_margin",
                "retrieval_top1_similarity",
                "retrieval_similarity_margin",
                "retrieved_target_agreement",
            ],
    }

    (
        args.output_root
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        args.output_root
        / "grid.json"
    ).write_text(
        json.dumps(
            grid,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

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

    print()
    print("Test used: False")


if __name__ == "__main__":
    main()
