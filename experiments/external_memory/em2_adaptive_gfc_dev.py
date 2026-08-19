"""Transparent confidence-aware Adaptive G+F+C fusion.

Dev tune only.
Uses prediction-visible features frozen before Adaptive results.
No Test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from experiments.external_memory import em2_hidden_m1_dev as em2

from src.personalisation.context_memory import (
    macro_author_metrics,
)


AUTHORS = em2.AUTHORS

SCALES = (
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
)

HISTORY_SHRINK_K = 5.0

EXPECTED_QUERIES = 5608


def clamp01(value: float) -> float:
    return max(
        0.0,
        min(1.0, float(value)),
    )


def history_confidence(
    n: int,
    *,
    use_count: bool,
) -> float:
    if n <= 0:
        return 0.0

    if not use_count:
        return 1.0

    return float(n) / (
        float(n)
        + HISTORY_SHRINK_K
    )


def gate(
    row: Mapping[str, Any],
    *,
    scale: float,
    use_count: bool,
) -> dict[str, float]:
    n = int(
        row["visible_history_count"]
    )

    h = history_confidence(
        n,
        use_count=use_count,
    )

    frequency_margin = clamp01(
        float(
            row.get(
                "frequency_margin",
                0.0,
            )
            or 0.0
        )
    )

    top1_similarity_raw = row.get(
        "retrieval_top1_similarity"
    )

    top1_similarity = (
        clamp01(
            float(top1_similarity_raw)
        )
        if top1_similarity_raw
        is not None
        else 0.0
    )

    agreement_raw = row.get(
        "retrieved_target_agreement"
    )

    agreement = (
        clamp01(
            float(agreement_raw)
        )
        if agreement_raw
        is not None
        else 0.0
    )

    confidence_f = (
        h
        * frequency_margin
    )

    confidence_c = (
        h
        * top1_similarity
        * agreement
    )

    total_confidence = (
        confidence_f
        + confidence_c
    )

    if total_confidence <= 0.0:
        return {
            "history_confidence": h,
            "confidence_f":
                confidence_f,
            "confidence_c":
                confidence_c,
            "lambda_f": 0.0,
            "lambda_c": 0.0,
        }

    strength = (
        float(scale)
        * max(
            confidence_f,
            confidence_c,
        )
    )

    share_f = (
        confidence_f
        / total_confidence
    )

    share_c = (
        confidence_c
        / total_confidence
    )

    return {
        "history_confidence": h,
        "confidence_f":
            confidence_f,
        "confidence_c":
            confidence_c,
        "lambda_f":
            strength * share_f,
        "lambda_c":
            strength * share_c,
    }


def rerank(
    source_row: Mapping[str, Any],
    *,
    scale: float,
    use_count: bool,
):
    gate_values = gate(
        source_row,
        scale=scale,
        use_count=use_count,
    )

    rows = []

    for candidate in source_row[
        "ranking"
    ]:
        g = float(
            candidate[
                "normalized_generic_score"
            ]
        )

        f = float(
            candidate[
                "frequency_support"
            ]
        )

        c = float(
            candidate[
                "context_support"
            ]
        )

        score = (
            g
            + gate_values["lambda_f"]
            * f
            + gate_values["lambda_c"]
            * c
        )

        rows.append(
            {
                "candidate":
                    str(
                        candidate[
                            "candidate"
                        ]
                    ),
                "generic_rank":
                    int(
                        candidate[
                            "generic_rank"
                        ]
                    ),
                "final_score":
                    score,
            }
        )

    rows.sort(
        key=lambda item: (
            -float(
                item["final_score"]
            ),
            int(
                item["generic_rank"]
            ),
        )
    )

    for rank, item in enumerate(
        rows,
        start=1,
    ):
        item["rank"] = rank

    return rows, gate_values


def gold_rank(
    ranking,
    gold: str,
):
    for row in ranking:
        if str(
            row["candidate"]
        ) == gold:
            return int(
                row["rank"]
            )

    return None


def static_rank(
    source_row,
    gold,
    *,
    lambda_f,
    lambda_c,
):
    rows = []

    for candidate in source_row[
        "ranking"
    ]:
        score = (
            float(
                candidate[
                    "normalized_generic_score"
                ]
            )
            + float(lambda_f)
            * float(
                candidate[
                    "frequency_support"
                ]
            )
            + float(lambda_c)
            * float(
                candidate[
                    "context_support"
                ]
            )
        )

        rows.append(
            {
                "candidate":
                    candidate[
                        "candidate"
                    ],
                "generic_rank":
                    candidate[
                        "generic_rank"
                    ],
                "final_score":
                    score,
            }
        )

    rows.sort(
        key=lambda item: (
            -float(
                item["final_score"]
            ),
            int(
                item["generic_rank"]
            ),
        )
    )

    for rank, item in enumerate(
        rows,
        start=1,
    ):
        item["rank"] = rank

    return gold_rank(
        rows,
        gold,
    )


def subset_rows(
    rows,
    name,
):
    if name == "overall":
        return list(rows)

    return [
        row
        for row in rows
        if bool(row[name])
    ]


def macro_top1(
    rows,
    name,
):
    return macro_author_metrics(
        subset_rows(
            rows,
            name,
        ),
        "rank",
    )[
        "macro_author"
    ]["top1"]


def transitions(
    before,
    after,
    subset,
):
    before_map = {
        row["row_id"]: row
        for row in before
    }

    after_map = {
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
            before_map[
                row_id
            ]["rank"]
            == 1
        )

        a = (
            after_map[
                row_id
            ]["rank"]
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pilot-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--fixed-gfc-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    dev = em2.read_jsonl(
        args.pilot_root
        / "dev_manifest.jsonl"
    )

    tune = [
        row
        for row in dev
        if (
            row.get(
                "pilot_partition"
            )
            == "tune"
            and str(
                row["author"]
            )
            in AUTHORS
        )
    ]

    if len(tune) != EXPECTED_QUERIES:
        raise RuntimeError(
            f"Unexpected tune size: "
            f"{len(tune)}"
        )

    gold_by_id = {
        str(row["row_id"]):
            em2.gold_of(row)
        for row in tune
    }

    source_path = (
        args.fixed_gfc_root
        / "selected_rows.jsonl"
    )

    source_rows = []

    with source_path.open(
        encoding="utf-8"
    ) as source:
        for line in source:
            if line.strip():
                source_rows.append(
                    json.loads(line)
                )

    if (
        len(source_rows)
        != EXPECTED_QUERIES
    ):
        raise RuntimeError(
            "Fixed-GFC detailed surface "
            f"changed: {len(source_rows)}"
        )

    if set(
        str(row["row_id"])
        for row in source_rows
    ) != set(gold_by_id):
        raise RuntimeError(
            "Row-id surface mismatch."
        )

    def evaluate(
        scale,
        *,
        use_count,
        detailed=False,
    ):
        outputs = []

        for source_row in source_rows:
            row_id = str(
                source_row["row_id"]
            )

            gold = gold_by_id[
                row_id
            ]

            ranking, gate_values = (
                rerank(
                    source_row,
                    scale=scale,
                    use_count=use_count,
                )
            )

            output = {
                "row_id": row_id,
                "author":
                    source_row["author"],
                "rank":
                    gold_rank(
                        ranking,
                        gold,
                    ),
                "history_available":
                    bool(
                        source_row[
                            "history_available"
                        ]
                    ),
                "ambiguous":
                    bool(
                        source_row[
                            "ambiguous"
                        ]
                    ),
                "conflict":
                    bool(
                        source_row[
                            "conflict"
                        ]
                    ),
            }

            if detailed:
                output.update(
                    {
                        "visible_history_count":
                            source_row[
                                "visible_history_count"
                            ],
                        **gate_values,
                    }
                )

            outputs.append(output)

        return outputs

    grid = []
    rows_by_scale = {}

    print(
        "Running primary Adaptive "
        "count-aware grid..."
    )

    for scale in SCALES:
        rows = evaluate(
            scale,
            use_count=True,
        )

        rows_by_scale[
            float(scale)
        ] = rows

        value = macro_top1(
            rows,
            "overall",
        )

        grid.append(
            {
                "scale":
                    float(scale),
                "macro_author_overall_top1":
                    value,
            }
        )

        print(
            f"L={scale:>4} "
            f"MacroTop1={value:.6f}",
            flush=True,
        )

    selected = max(
        grid,
        key=lambda row: (
            float(
                row[
                    "macro_author_overall_top1"
                ]
            ),
            -float(
                row["scale"]
            ),
        ),
    )

    selected_scale = float(
        selected["scale"]
    )

    adaptive_rows = evaluate(
        selected_scale,
        use_count=True,
        detailed=True,
    )

    print()
    print(
        "Running no-count "
        "diagnostic control..."
    )

    no_count_grid = []
    no_count_rows_by_scale = {}

    for scale in SCALES:
        rows = evaluate(
            scale,
            use_count=False,
        )

        no_count_rows_by_scale[
            float(scale)
        ] = rows

        value = macro_top1(
            rows,
            "overall",
        )

        no_count_grid.append(
            {
                "scale":
                    float(scale),
                "macro_author_overall_top1":
                    value,
            }
        )

        print(
            f"NoCount L={scale:>4} "
            f"MacroTop1={value:.6f}",
            flush=True,
        )

    no_count_selected = max(
        no_count_grid,
        key=lambda row: (
            float(
                row[
                    "macro_author_overall_top1"
                ]
            ),
            -float(
                row["scale"]
            ),
        ),
    )

    no_count_rows = (
        no_count_rows_by_scale[
            float(
                no_count_selected[
                    "scale"
                ]
            )
        ]
    )

    # Same-surface baselines reconstructed
    # from the stored G/F/C candidate evidence.

    baseline_rows = {
        "G": [],
        "F": [],
        "Hidden-M1": [],
        "Fixed-GFC": [],
    }

    for source_row in source_rows:
        row_id = str(
            source_row["row_id"]
        )

        gold = gold_by_id[
            row_id
        ]

        common = {
            "row_id": row_id,
            "author":
                source_row["author"],
            "history_available":
                bool(
                    source_row[
                        "history_available"
                    ]
                ),
            "ambiguous":
                bool(
                    source_row[
                        "ambiguous"
                    ]
                ),
            "conflict":
                bool(
                    source_row[
                        "conflict"
                    ]
                ),
        }

        configs = {
            "G":
                (0.0, 0.0),
            "F":
                (4.0, 0.0),
            "Hidden-M1":
                (0.0, 4.0),
            "Fixed-GFC":
                (0.5, 4.0),
        }

        for name, (
            lambda_f,
            lambda_c,
        ) in configs.items():
            baseline_rows[
                name
            ].append(
                {
                    **common,
                    "rank":
                        static_rank(
                            source_row,
                            gold,
                            lambda_f=
                                lambda_f,
                            lambda_c=
                                lambda_c,
                        ),
                }
            )

    methods = {
        **baseline_rows,
        "Adaptive":
            adaptive_rows,
        "Adaptive-NoCount":
            no_count_rows,
    }

    print()
    print(
        "=== ADAPTIVE G+F+C "
        "DEV SELECTION ==="
    )

    print(
        "Selected L:",
        selected_scale,
    )

    print(
        "No-count selected L:",
        no_count_selected[
            "scale"
        ],
    )

    print()

    print(
        f"{'Method':18s}"
        f"{'Overall':>12s}"
        f"{'History':>12s}"
        f"{'Ambiguous':>12s}"
        f"{'Conflict':>12s}"
    )

    metrics = {}

    for name, rows in methods.items():
        values = {
            subset:
                macro_top1(
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

        metrics[name] = values

        print(
            f"{name:18s}"
            f"{values['overall']:12.6f}"
            f"{values['history_available']:12.6f}"
            f"{values['ambiguous']:12.6f}"
            f"{values['conflict']:12.6f}"
        )

    comparisons = {}

    for baseline in (
        "F",
        "Hidden-M1",
        "Fixed-GFC",
    ):
        print()
        print(
            f"=== {baseline} -> "
            "ADAPTIVE ==="
        )

        values = {}

        for subset in (
            "overall",
            "history_available",
            "ambiguous",
            "conflict",
        ):
            delta = transitions(
                baseline_rows[
                    baseline
                ],
                adaptive_rows,
                subset,
            )

            values[
                subset
            ] = delta

            print(
                f"{subset:18s} "
                f"rescue="
                f"{delta['rescue']} "
                f"harm="
                f"{delta['harm']} "
                f"net="
                f"{delta['net']:+d}"
            )

        comparisons[
            baseline
        ] = values

    lambda_fs = [
        float(
            row["lambda_f"]
        )
        for row in adaptive_rows
    ]

    lambda_cs = [
        float(
            row["lambda_c"]
        )
        for row in adaptive_rows
    ]

    gate_stats = {
        "mean_lambda_f":
            statistics.mean(
                lambda_fs
            ),
        "mean_lambda_c":
            statistics.mean(
                lambda_cs
            ),
        "median_lambda_f":
            statistics.median(
                lambda_fs
            ),
        "median_lambda_c":
            statistics.median(
                lambda_cs
            ),
        "lambda_f_gt_lambda_c":
            sum(
                f > c
                for f, c
                in zip(
                    lambda_fs,
                    lambda_cs,
                )
            ),
        "lambda_c_gt_lambda_f":
            sum(
                c > f
                for f, c
                in zip(
                    lambda_fs,
                    lambda_cs,
                )
            ),
        "lambda_equal":
            sum(
                f == c
                for f, c
                in zip(
                    lambda_fs,
                    lambda_cs,
                )
            ),
        "zero_personalisation":
            sum(
                f == 0.0
                and c == 0.0
                for f, c
                in zip(
                    lambda_fs,
                    lambda_cs,
                )
            ),
    }

    print()
    print("=== GATE STATISTICS ===")

    for key, value in (
        gate_stats.items()
    ):
        print(
            f"{key}: {value}"
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "status":
            "dev_selection_complete",
        "experiment":
            "adaptive_g_f_hidden_c",
        "partition":
            "dev_tune_only",
        "queries":
            len(source_rows),
        "authors":
            list(AUTHORS),
        "hidden_top_n":
            3,
        "history_shrink_k":
            HISTORY_SHRINK_K,
        "scale_grid":
            list(SCALES),
        "selected":
            selected,
        "no_count_selected":
            no_count_selected,
        "metrics":
            metrics,
        "comparisons":
            comparisons,
        "gate_statistics":
            gate_stats,
        "test_used":
            False,
        "gold_used_for_gate":
            False,
        "conflict_used_for_gate":
            False,
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
            {
                "adaptive":
                    grid,
                "no_count":
                    no_count_grid,
            },
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
        for row in adaptive_rows:
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
    print("Adaptive design complete.")


if __name__ == "__main__":
    main()
