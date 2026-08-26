from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


EXPECTED_QUERIES = 94590
EXPECTED_BASE_TOP1 = 0.5980019029495718

LEVELS = [-2, -1, 0, 1, 2]
LOG2 = math.log(2.0)

CONTROLS = {
    "personalisation_strength": {
        "primitive": "phi_personal",
        "interpretation": (
            "positive = stronger personal evidence; "
            "negative = weaker personal evidence"
        ),
    },
    "exact_vs_composition": {
        "primitive": "phi_exact_minus_composition",
        "interpretation": (
            "positive = prefer Exact; "
            "negative = prefer Composition"
        ),
    },
    "composition_tolerance": {
        "primitive": "phi_fragmentation",
        "interpretation": (
            "positive = tolerate fragmented composition; "
            "negative = conservative composition"
        ),
    },
}


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def gold_rank(candidates, scores):
    order = sorted(
        range(len(candidates)),
        key=lambda i: (
            -scores[i],
            i,
        ),
    )

    for rank, i in enumerate(
        order,
        start=1,
    ):
        if candidates[i][
            "is_gold_posthoc"
        ]:
            return rank

    return None


def top_index(scores):
    # Stable tie rule: earlier frozen pool order wins.
    best_i = 0
    best_score = scores[0]

    for i in range(
        1,
        len(scores),
    ):
        if scores[i] > best_score:
            best_i = i
            best_score = scores[i]

    return best_i


def empty_metric():
    return {
        "queries": 0,
        "top1_correct": 0,
        "top3_correct": 0,
        "top5_correct": 0,
        "mrr_sum": 0.0,
        "missing": 0,
        "top1_personal_slot": 0,
        "top1_has_personal_evidence": 0,
        "top1_has_exact": 0,
        "top1_has_composition": 0,
        "selected_primitive_sum": 0.0,
        "selected_recovery_confidence_sum": 0.0,
        "selected_fragmentation_sum": 0.0,
        "switch_from_baseline": 0,
    }


def finalize(metric):
    n = metric["queries"]

    return {
        "n": n,
        "top1": (
            metric["top1_correct"]
            / n
        ),
        "top3": (
            metric["top3_correct"]
            / n
        ),
        "top5": (
            metric["top5_correct"]
            / n
        ),
        "mrr": (
            metric["mrr_sum"]
            / n
        ),
        "missing": (
            metric["missing"]
            / n
        ),
        "top1_personal_slot_rate": (
            metric["top1_personal_slot"]
            / n
        ),
        "top1_personal_evidence_rate": (
            metric[
                "top1_has_personal_evidence"
            ]
            / n
        ),
        "top1_exact_support_rate": (
            metric["top1_has_exact"]
            / n
        ),
        "top1_composition_support_rate": (
            metric[
                "top1_has_composition"
            ]
            / n
        ),
        "mean_selected_control_primitive": (
            metric[
                "selected_primitive_sum"
            ]
            / n
        ),
        "mean_selected_recovery_confidence": (
            metric[
                "selected_recovery_confidence_sum"
            ]
            / n
        ),
        "mean_selected_fragmentation_ratio": (
            metric[
                "selected_fragmentation_sum"
            ]
            / n
        ),
        "switch_rate_vs_default": (
            metric[
                "switch_from_baseline"
            ]
            / n
        ),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--surface",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--r7a-summary",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    args = ap.parse_args()

    if (
        args.output_root.exists()
        and any(
            args.output_root.iterdir()
        )
    ):
        raise RuntimeError(
            "Refusing to overwrite non-empty R7-B output"
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    r7a = json.loads(
        args.r7a_summary.read_text(
            encoding="utf-8-sig"
        )
    )

    if r7a["status"] != "complete":
        raise RuntimeError(
            "R7-A is not complete"
        )

    print(
        "===== M4-R7B CONTROL INTERVENTION CURVES ====="
    )

    print(
        "Frozen R6-v2 + frozen R7-A control surface"
    )

    print(
        "No model training / no feature scaling"
    )

    print(
        "One control unit = ln(2) maximum log-odds shift"
    )

    print(
        "Sweep = -2, -1, 0, +1, +2"
    )

    print(
        "DEV3000=CLOSED TEST=CLOSED"
    )

    metrics = {
        control: {
            level: empty_metric()
            for level in LEVELS
        }
        for control in CONTROLS
    }

    monotonic_violations = {
        control: 0
        for control in CONTROLS
    }

    query_count = 0

    with gzip.open(
        args.surface,
        "rt",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            query_count += 1

            candidates = row[
                "candidates"
            ]

            if not candidates:
                raise RuntimeError(
                    "Empty candidate pool: "
                    + row["row_id"]
                )

            base_scores = [
                float(
                    c["z_base"]
                )
                for c in candidates
            ]

            base_top = top_index(
                base_scores
            )

            primitives = {
                "personalisation_strength": [
                    float(
                        c["phi_personal"]
                    )
                    for c in candidates
                ],
                "exact_vs_composition": [
                    float(
                        c["phi_exact"]
                    )
                    - float(
                        c["phi_composition"]
                    )
                    for c in candidates
                ],
                "composition_tolerance": [
                    float(
                        c["phi_fragmentation"]
                    )
                    for c in candidates
                ],
            }

            for control in CONTROLS:
                selected_values = []

                phi = primitives[
                    control
                ]

                for level in LEVELS:
                    controlled_scores = [
                        base_scores[i]
                        + LOG2
                        * level
                        * phi[i]
                        for i in range(
                            len(candidates)
                        )
                    ]

                    top_i = top_index(
                        controlled_scores
                    )

                    rank = gold_rank(
                        candidates,
                        controlled_scores,
                    )

                    top = candidates[
                        top_i
                    ]

                    metric = metrics[
                        control
                    ][
                        level
                    ]

                    metric[
                        "queries"
                    ] += 1

                    if rank is None:
                        metric[
                            "missing"
                        ] += 1
                    else:
                        if rank <= 1:
                            metric[
                                "top1_correct"
                            ] += 1

                        if rank <= 3:
                            metric[
                                "top3_correct"
                            ] += 1

                        if rank <= 5:
                            metric[
                                "top5_correct"
                            ] += 1

                        metric[
                            "mrr_sum"
                        ] += (
                            1.0 / rank
                        )

                    if (
                        top["source"]
                        == "personal_recovery"
                    ):
                        metric[
                            "top1_personal_slot"
                        ] += 1

                    if float(
                        top[
                            "recovery_confidence"
                        ]
                    ) > 0:
                        metric[
                            "top1_has_personal_evidence"
                        ] += 1

                    if float(
                        top["p_exact"]
                    ) > 0:
                        metric[
                            "top1_has_exact"
                        ] += 1

                    if float(
                        top[
                            "p_composition"
                        ]
                    ) > 0:
                        metric[
                            "top1_has_composition"
                        ] += 1

                    metric[
                        "selected_primitive_sum"
                    ] += phi[
                        top_i
                    ]

                    metric[
                        "selected_recovery_confidence_sum"
                    ] += float(
                        top[
                            "recovery_confidence"
                        ]
                    )

                    metric[
                        "selected_fragmentation_sum"
                    ] += float(
                        top[
                            "fragmentation_ratio"
                        ]
                    )

                    if top_i != base_top:
                        metric[
                            "switch_from_baseline"
                        ] += 1

                    selected_values.append(
                        phi[
                            top_i
                        ]
                    )

                # For argmax of affine functions:
                # selected phi must be non-decreasing
                # as its coefficient increases.
                for a, b in zip(
                    selected_values,
                    selected_values[1:],
                ):
                    if b + 1e-12 < a:
                        monotonic_violations[
                            control
                        ] += 1
                        break

            if (
                query_count
                % 10000
                == 0
            ):
                print(
                    f"control {query_count}/"
                    f"{EXPECTED_QUERIES}",
                    flush=True,
                )

    if query_count != EXPECTED_QUERIES:
        raise RuntimeError(
            "Query count mismatch: "
            f"{query_count}"
        )

    curves = {
        control: {
            str(level): finalize(
                metrics[
                    control
                ][
                    level
                ]
            )
            for level in LEVELS
        }
        for control in CONTROLS
    }

    # --------------------------------------------------
    # Default fidelity.
    # --------------------------------------------------

    default_top1_values = []

    for control in CONTROLS:
        default = curves[
            control
        ]["0"]

        default_top1_values.append(
            default[
                "top1"
            ]
        )

        if abs(
            default["top1"]
            - EXPECTED_BASE_TOP1
        ) > 1e-15:
            raise RuntimeError(
                f"Default Top1 drift for {control}: "
                f"{default['top1']}"
            )

        if (
            default[
                "switch_rate_vs_default"
            ]
            != 0.0
        ):
            raise RuntimeError(
                f"Default ranking drift: {control}"
            )

    # --------------------------------------------------
    # Behavioral monotonicity gates.
    # --------------------------------------------------

    for control, count in (
        monotonic_violations.items()
    ):
        if count != 0:
            raise RuntimeError(
                f"Monotonicity violation "
                f"{control}: {count}"
            )

    print()
    print(
        "===== CONTROL CURVES ====="
    )

    for control in CONTROLS:
        print()
        print(
            control.upper()
        )

        for level in LEVELS:
            x = curves[
                control
            ][
                str(level)
            ]

            print(
                f"u={level:+d} "
                f"Top1={x['top1']:.6f} "
                f"Top3={x['top3']:.6f} "
                f"Switch="
                f"{x['switch_rate_vs_default']:.6f} "
                f"PersonalTop1="
                f"{x['top1_personal_slot_rate']:.6f} "
                f"PersonalEvidence="
                f"{x['top1_personal_evidence_rate']:.6f} "
                f"Exact="
                f"{x['top1_exact_support_rate']:.6f} "
                f"Comp="
                f"{x['top1_composition_support_rate']:.6f} "
                f"SelectedPhi="
                f"{x['mean_selected_control_primitive']:.6f}"
            )

    print()
    print(
        "===== MONOTONICITY ====="
    )

    for control, count in (
        monotonic_violations.items()
    ):
        values = [
            curves[
                control
            ][
                str(level)
            ][
                "mean_selected_control_primitive"
            ]
            for level in LEVELS
        ]

        print(
            control,
            "QUERY_VIOLATIONS=",
            count,
            "MEAN_SELECTED_PHI=",
            values,
        )

    csv_path = (
        args.output_root
        / "control_curves.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "control",
                "level",
                "top1",
                "top3",
                "top5",
                "mrr",
                "switch_rate_vs_default",
                "top1_personal_slot_rate",
                "top1_personal_evidence_rate",
                "top1_exact_support_rate",
                "top1_composition_support_rate",
                "mean_selected_control_primitive",
                "mean_selected_recovery_confidence",
                "mean_selected_fragmentation_ratio",
            ]
        )

        for control in CONTROLS:
            for level in LEVELS:
                x = curves[
                    control
                ][
                    str(level)
                ]

                writer.writerow(
                    [
                        control,
                        level,
                        x["top1"],
                        x["top3"],
                        x["top5"],
                        x["mrr"],
                        x[
                            "switch_rate_vs_default"
                        ],
                        x[
                            "top1_personal_slot_rate"
                        ],
                        x[
                            "top1_personal_evidence_rate"
                        ],
                        x[
                            "top1_exact_support_rate"
                        ],
                        x[
                            "top1_composition_support_rate"
                        ],
                        x[
                            "mean_selected_control_primitive"
                        ],
                        x[
                            "mean_selected_recovery_confidence"
                        ],
                        x[
                            "mean_selected_fragmentation_ratio"
                        ],
                    ]
                )

    summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment": (
            "m4_r7b_control_intervention_curves_v1"
        ),
        "control_formula": (
            "z_ctrl = z_base + ln(2) * u * phi; "
            "one unit changes odds by at most x2 "
            "for phi=1"
        ),
        "levels": LEVELS,
        "controls": CONTROLS,
        "curves": curves,
        "behavioral_monotonicity": {
            control: {
                "query_level_violations": (
                    monotonic_violations[
                        control
                    ]
                ),
                "passed": (
                    monotonic_violations[
                        control
                    ]
                    == 0
                ),
            }
            for control in CONTROLS
        },
        "default_fidelity": {
            "expected_r6_v2_top1": (
                EXPECTED_BASE_TOP1
            ),
            "observed_top1": (
                default_top1_values[0]
            ),
            "ranking_switch_rate": 0.0,
            "passed": True,
        },
        "scientific_interpretation": {
            "accuracy_is_secondary": True,
            "primary_endpoint": (
                "behavioral fidelity and monotonic "
                "response to explicit user controls"
            ),
            "lambdaMART_modified": False,
            "controls_are_post_model_log_odds_shifts": True,
        },
        "hashes": {
            "runner": sha256_file(
                Path(__file__)
            ),
            "r7a_surface": sha256_file(
                args.surface
            ),
            "r7a_summary": sha256_file(
                args.r7a_summary
            ),
            "curve_csv": sha256_file(
                csv_path
            ),
        },
        "used_dev3000": False,
        "used_test": False,
    }

    summary_path = (
        args.output_root
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "DEFAULT_TOP1 =",
        default_top1_values[0],
    )

    print(
        "CONTROL_CURVES_SHA =",
        summary[
            "hashes"
        ][
            "curve_csv"
        ],
    )

    print(
        "RUNNER_SHA =",
        summary[
            "hashes"
        ][
            "runner"
        ],
    )

    print(
        "SUMMARY_SHA =",
        sha256_file(
            summary_path
        ),
    )

    print()
    print(
        "DEFAULT_CONTROL_FIDELITY=PASS"
    )

    print(
        "PERSONALISATION_MONOTONICITY=PASS"
    )

    print(
        "EXACT_COMPOSITION_MONOTONICITY=PASS"
    )

    print(
        "COMPOSITION_TOLERANCE_MONOTONICITY=PASS"
    )

    print(
        "DEV3000_TEST_CLOSED=PASS"
    )

    print(
        "M4_R7B_CONTROL_INTERVENTION_GATE=PASS"
    )


if __name__ == "__main__":
    main()