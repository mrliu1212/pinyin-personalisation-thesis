from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binom


ROOT = Path(
    "results/finalmodel_fiveauthor_v1"
)

ROW_RANKS = (
    ROOT
    / "five_system_test_comparison_v1"
    / "row_level_ranks_v1.csv"
)

GENERIC = (
    ROOT
    / "generic_baseline_v1"
    / "run_v2"
    / "generic_test_predictions.jsonl"
)

ADAPTER_ROOT = (
    ROOT
    / "test_end_to_end_v1"
    / "adapter_test_top10_v1"
)

GENERIC_EXPLICIT = (
    ROOT
    / "generic_explicit_test_rich30_v1"
    / "final_fiveauthor_generic_explicit_test_rich30_surface_v1.jsonl"
)

FINAL = (
    ROOT
    / "test_end_to_end_v1"
    / "rich30_test_v1"
    / "final_fiveauthor_test_rich30_surface_v1.jsonl"
)

OUT = (
    ROOT
    / "statistical_robustness_and_recovery_v1"
)

EXPECTED_ROWS = 40_000
BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 20260829

COMPARISONS = [
    (
        "Generic_vs_Adapter",
        "Generic",
        "Adapter",
    ),
    (
        "GenericFrequency_vs_GenericExplicit",
        "Generic+Frequency",
        "Generic+Explicit",
    ),
    (
        "Adapter_vs_Final",
        "Adapter",
        "Adapter+Explicit",
    ),
]


# ============================================================
# Basic loaders
# ============================================================

def load_jsonl(path: Path):
    with path.open(
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def prediction_map(path: Path):
    out = {}

    for r in load_jsonl(path):
        rid = str(r["row_id"])

        if rid in out:
            raise RuntimeError(
                f"Duplicate row_id: {rid}"
            )

        out[rid] = r

    return out


def load_adapter_map():
    out = {}

    paths = sorted(
        ADAPTER_ROOT.glob(
            "*_test_top10_v1.jsonl"
        )
    )

    if not paths:
        raise RuntimeError(
            f"No Adapter prediction files under "
            f"{ADAPTER_ROOT}"
        )

    for path in paths:
        for r in load_jsonl(path):
            rid = str(r["row_id"])

            if rid in out:
                raise RuntimeError(
                    f"Duplicate Adapter row: {rid}"
                )

            out[rid] = r

    if len(out) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Adapter rows={len(out)} "
            f"!= {EXPECTED_ROWS}"
        )

    return out


# ============================================================
# Statistical robustness
# ============================================================

def parse_rank(v):
    if v is None:
        return None

    s = str(v).strip()

    if not s:
        return None

    if s.lower() in {
        "none",
        "nan",
        "null",
    }:
        return None

    return int(float(s))


def paired_bootstrap_mean_ci(
    values: np.ndarray,
    reps: int,
    seed: int,
):
    """
    Exact ordinary nonparametric paired bootstrap
    for a scalar paired difference.

    Instead of materialising reps x N row indices,
    compress identical difference values and draw
    bootstrap multinomial counts. This is equivalent
    to resampling the paired rows for the mean.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    unique, counts = np.unique(
        values,
        return_counts=True,
    )

    n = len(values)

    probs = (
        counts.astype(np.float64)
        / float(n)
    )

    rng = np.random.default_rng(
        seed
    )

    boot_counts = rng.multinomial(
        n,
        probs,
        size=reps,
    )

    boot_means = (
        boot_counts @ unique
    ) / float(n)

    low, high = np.quantile(
        boot_means,
        [0.025, 0.975],
    )

    return {
        "estimate":
            float(values.mean()),
        "ci95_low":
            float(low),
        "ci95_high":
            float(high),
        "bootstrap_reps":
            int(reps),
        "unique_difference_values":
            int(len(unique)),
    }


def exact_mcnemar(
    left_correct: np.ndarray,
    right_correct: np.ndarray,
):
    """
    b = left correct, right wrong
    c = left wrong, right correct

    Under McNemar H0:
        b and c are equally likely among
        discordant pairs.

    For p=0.5, exact two-sided p is
    2 * lower-tail probability of min(b,c).
    """

    left_correct = np.asarray(
        left_correct,
        dtype=bool,
    )
    right_correct = np.asarray(
        right_correct,
        dtype=bool,
    )

    b = int(np.sum(
        left_correct
        & ~right_correct
    ))

    c = int(np.sum(
        ~left_correct
        & right_correct
    ))

    discordant = b + c

    if discordant == 0:
        return {
            "left_correct_right_wrong":
                b,
            "left_wrong_right_correct":
                c,
            "discordant":
                0,
            "exact_two_sided_p":
                1.0,
            "log10_p":
                0.0,
        }

    k = min(b, c)

    log_p = (
        math.log(2.0)
        + float(
            binom.logcdf(
                k,
                discordant,
                0.5,
            )
        )
    )

    # Exact two-sided value cannot exceed 1.
    log_p = min(
        0.0,
        log_p,
    )

    if log_p > -745:
        p = math.exp(log_p)
    else:
        p = 0.0

    return {
        "left_correct_right_wrong":
            b,
        "left_wrong_right_correct":
            c,
        "discordant":
            discordant,
        "exact_two_sided_p":
            p,
        "log10_p":
            log_p
            / math.log(10.0),
    }


def run_statistical_analysis():
    with ROW_RANKS.open(
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"row_level_ranks rows="
            f"{len(rows)} "
            f"!= {EXPECTED_ROWS}"
        )

    row_ids = [
        str(r["row_id"])
        for r in rows
    ]

    if len(set(row_ids)) != EXPECTED_ROWS:
        raise RuntimeError(
            "Duplicate row_id in row-level "
            "comparison surface"
        )

    print(
        "STATISTICAL_ROW_UNIVERSE=PASS"
    )

    results = {}

    for index, (
        name,
        left,
        right,
    ) in enumerate(
        COMPARISONS
    ):
        left_rank = np.array(
            [
                (
                    parse_rank(r[left])
                    if parse_rank(
                        r[left]
                    ) is not None
                    else 0
                )
                for r in rows
            ],
            dtype=np.int16,
        )

        right_rank = np.array(
            [
                (
                    parse_rank(r[right])
                    if parse_rank(
                        r[right]
                    ) is not None
                    else 0
                )
                for r in rows
            ],
            dtype=np.int16,
        )

        left_top1 = (
            left_rank == 1
        )
        right_top1 = (
            right_rank == 1
        )

        top1_diff = (
            right_top1.astype(
                np.float64
            )
            - left_top1.astype(
                np.float64
            )
        )

        left_rr = np.where(
            left_rank > 0,
            1.0
            / np.maximum(
                left_rank,
                1,
            ),
            0.0,
        )

        right_rr = np.where(
            right_rank > 0,
            1.0
            / np.maximum(
                right_rank,
                1,
            ),
            0.0,
        )

        mrr_diff = (
            right_rr
            - left_rr
        )

        top1_boot = (
            paired_bootstrap_mean_ci(
                top1_diff,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED
                + index * 10
                + 1,
            )
        )

        mrr_boot = (
            paired_bootstrap_mean_ci(
                mrr_diff,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED
                + index * 10
                + 2,
            )
        )

        mcnemar = exact_mcnemar(
            left_top1,
            right_top1,
        )

        left_top1_rate = float(
            left_top1.mean()
        )
        right_top1_rate = float(
            right_top1.mean()
        )

        left_mrr = float(
            left_rr.mean()
        )
        right_mrr = float(
            right_rr.mean()
        )

        results[name] = {
            "left_system":
                left,
            "right_system":
                right,
            "N":
                EXPECTED_ROWS,

            "left_top1":
                left_top1_rate,
            "right_top1":
                right_top1_rate,

            "delta_top1":
                top1_boot[
                    "estimate"
                ],
            "delta_top1_pp":
                100.0
                * top1_boot[
                    "estimate"
                ],
            "delta_top1_ci95_low":
                top1_boot[
                    "ci95_low"
                ],
            "delta_top1_ci95_high":
                top1_boot[
                    "ci95_high"
                ],
            "delta_top1_ci95_low_pp":
                100.0
                * top1_boot[
                    "ci95_low"
                ],
            "delta_top1_ci95_high_pp":
                100.0
                * top1_boot[
                    "ci95_high"
                ],

            "mcnemar":
                mcnemar,

            "left_mrr":
                left_mrr,
            "right_mrr":
                right_mrr,
            "delta_mrr":
                mrr_boot[
                    "estimate"
                ],
            "delta_mrr_ci95_low":
                mrr_boot[
                    "ci95_low"
                ],
            "delta_mrr_ci95_high":
                mrr_boot[
                    "ci95_high"
                ],

            "bootstrap_reps":
                BOOTSTRAP_REPS,
            "bootstrap_seed":
                BOOTSTRAP_SEED,
        }

    return results


# ============================================================
# Exact / Composition recovery-source analysis
# ============================================================

def get_top10(row):
    for key in (
        "top10_candidates",
        "generic_top10",
        "adapter_top10",
    ):
        if key in row:
            return list(
                map(
                    str,
                    row[key],
                )
            )[:10]

    raise RuntimeError(
        f"Cannot find Top10 field. "
        f"Keys={sorted(row)}"
    )


def classify_candidate(candidate):
    has_exact = (
        candidate.get(
            "p_exact"
        )
        is not None
    )

    has_comp = (
        candidate.get(
            "p_composition"
        )
        is not None
    )

    if has_exact and has_comp:
        return "exact_and_composition"

    if has_exact:
        return "exact_only"

    if has_comp:
        return "composition_only"

    raise RuntimeError(
        "Recovered personal candidate "
        "has neither Exact nor Composition "
        f"support: {candidate}"
    )


def find_gold_personal_candidate(
    surface_row,
    gold,
):
    personal = (
        surface_row.get(
            "personal_recovery_top5"
        )
    )

    if personal is None:
        raise RuntimeError(
            "Surface missing "
            "personal_recovery_top5"
        )

    matches = [
        c
        for c in personal
        if str(c["text"]) == gold
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f'{surface_row["row_id"]}: '
            f"expected exactly one Gold "
            f"candidate in personal_recovery_top5, "
            f"found {len(matches)}"
        )

    return matches[0]


def analyse_recovery_source(
    *,
    name,
    neural_map,
    surface_path,
    expected_recovered,
):
    counts = Counter()

    by_M = defaultdict(
        Counter
    )

    examples = {
        "exact_only": [],
        "composition_only": [],
        "exact_and_composition": [],
    }

    surface_rows = 0
    neural_hit = 0
    newly_recovered = 0

    for r in load_jsonl(
        surface_path
    ):
        surface_rows += 1

        rid = str(r["row_id"])

        if rid not in neural_map:
            raise RuntimeError(
                f"{name}: surface row "
                f"missing neural row {rid}"
            )

        nrow = neural_map[rid]

        gold = str(
            nrow["gold"]
        )

        if str(r["gold"]) != gold:
            raise RuntimeError(
                f"{name}: Gold mismatch "
                f"for {rid}"
            )

        neural_top10 = (
            get_top10(nrow)
        )

        is_neural_hit = (
            gold in neural_top10
        )

        if is_neural_hit:
            neural_hit += 1

        final_pool = list(
            map(
                str,
                r["final_pool"],
            )
        )

        is_new_recovery = (
            not is_neural_hit
            and gold in final_pool
        )

        if not is_new_recovery:
            continue

        newly_recovered += 1

        cand = (
            find_gold_personal_candidate(
                r,
                gold,
            )
        )

        source = classify_candidate(
            cand
        )

        counts[source] += 1

        M = str(r["M"])
        by_M[M][source] += 1

        if len(
            examples[source]
        ) < 10:
            examples[source].append({
                "row_id":
                    rid,
                "gold":
                    gold,
                "M":
                    int(r["M"]),
                "author_name":
                    r.get(
                        "author_name"
                    ),
                "typing_mode":
                    r.get(
                        "typing_mode"
                    ),
                "effective_typing_mode":
                    r.get(
                        "effective_typing_mode"
                    ),
                "p_exact":
                    cand.get(
                        "p_exact"
                    ),
                "p_composition":
                    cand.get(
                        "p_composition"
                    ),
                "piece_count":
                    cand.get(
                        "piece_count"
                    ),
                "provenance":
                    cand.get(
                        "provenance"
                    ),
            })

    if surface_rows != EXPECTED_ROWS:
        raise RuntimeError(
            f"{name}: surface rows="
            f"{surface_rows} "
            f"!= {EXPECTED_ROWS}"
        )

    if newly_recovered != expected_recovered:
        raise RuntimeError(
            f"{name}: newly recovered="
            f"{newly_recovered}, expected "
            f"{expected_recovered}"
        )

    total_source = sum(
        counts.values()
    )

    if total_source != expected_recovered:
        raise RuntimeError(
            f"{name}: source sum="
            f"{total_source}, expected "
            f"{expected_recovered}"
        )

    ordered_sources = [
        "exact_only",
        "composition_only",
        "exact_and_composition",
    ]

    result = {
        "system":
            name,
        "N":
            EXPECTED_ROWS,
        "neural_top10_hit":
            neural_hit,
        "neural_top10_hit_rate":
            neural_hit
            / EXPECTED_ROWS,
        "newly_recovered":
            newly_recovered,
        "newly_recovered_rate":
            newly_recovered
            / EXPECTED_ROWS,
        "recovery_support": {},
        "by_M": {},
        "examples":
            examples,
    }

    for source in ordered_sources:
        n = counts[source]

        result[
            "recovery_support"
        ][source] = {
            "count": n,
            "share_of_newly_recovered":
                (
                    n
                    / newly_recovered
                    if newly_recovered
                    else 0.0
                ),
            "share_of_all_test":
                n
                / EXPECTED_ROWS,
        }

    for M in sorted(
        by_M,
        key=int,
    ):
        subtotal = sum(
            by_M[M].values()
        )

        result["by_M"][M] = {
            "total_newly_recovered":
                subtotal,
        }

        for source in ordered_sources:
            n = by_M[M][source]

            result[
                "by_M"
            ][M][source] = {
                "count":
                    n,
                "share_within_M_recovery":
                    (
                        n / subtotal
                        if subtotal
                        else 0.0
                    ),
            }

    return result


def run_recovery_analysis():
    generic = prediction_map(
        GENERIC
    )

    adapter = load_adapter_map()

    if len(generic) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Generic rows={len(generic)}"
        )

    if set(generic) != set(adapter):
        raise RuntimeError(
            "Generic/Adapter Test "
            "row universe mismatch"
        )

    print(
        "RECOVERY_NEURAL_UNIVERSE=PASS"
    )

    generic_result = (
        analyse_recovery_source(
            name="Generic+Explicit",
            neural_map=generic,
            surface_path=
                GENERIC_EXPLICIT,
            expected_recovered=3490,
        )
    )

    final_result = (
        analyse_recovery_source(
            name="Adapter+Explicit",
            neural_map=adapter,
            surface_path=FINAL,
            expected_recovered=669,
        )
    )

    return {
        "Generic+Explicit":
            generic_result,
        "Adapter+Explicit":
            final_result,
    }


# ============================================================
# Output
# ============================================================

def write_statistics_csv(
    results
):
    path = (
        OUT
        / "statistical_robustness_v1.csv"
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        w = csv.writer(f)

        w.writerow([
            "comparison",
            "N",
            "left_system",
            "right_system",
            "left_top1",
            "right_top1",
            "delta_top1_pp",
            "delta_top1_ci95_low_pp",
            "delta_top1_ci95_high_pp",
            "mcnemar_left_correct_right_wrong",
            "mcnemar_left_wrong_right_correct",
            "mcnemar_discordant",
            "mcnemar_exact_p",
            "mcnemar_log10_p",
            "left_mrr",
            "right_mrr",
            "delta_mrr",
            "delta_mrr_ci95_low",
            "delta_mrr_ci95_high",
            "bootstrap_reps",
        ])

        for name, x in results.items():
            m = x["mcnemar"]

            w.writerow([
                name,
                x["N"],
                x["left_system"],
                x["right_system"],
                x["left_top1"],
                x["right_top1"],
                x["delta_top1_pp"],
                x[
                    "delta_top1_ci95_low_pp"
                ],
                x[
                    "delta_top1_ci95_high_pp"
                ],
                m[
                    "left_correct_right_wrong"
                ],
                m[
                    "left_wrong_right_correct"
                ],
                m["discordant"],
                m[
                    "exact_two_sided_p"
                ],
                m["log10_p"],
                x["left_mrr"],
                x["right_mrr"],
                x["delta_mrr"],
                x[
                    "delta_mrr_ci95_low"
                ],
                x[
                    "delta_mrr_ci95_high"
                ],
                x["bootstrap_reps"],
            ])

    return path


def write_recovery_csv(
    recovery
):
    path = (
        OUT
        / "exact_composition_recovery_v1.csv"
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        w = csv.writer(f)

        w.writerow([
            "system",
            "N",
            "newly_recovered",
            "support_type",
            "count",
            "share_of_newly_recovered",
            "share_of_all_test",
        ])

        for system, x in (
            recovery.items()
        ):
            for source, s in (
                x[
                    "recovery_support"
                ].items()
            ):
                w.writerow([
                    system,
                    x["N"],
                    x[
                        "newly_recovered"
                    ],
                    source,
                    s["count"],
                    s[
                        "share_of_newly_recovered"
                    ],
                    s[
                        "share_of_all_test"
                    ],
                ])

    return path


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=============================================="
    )
    print(
        "FINAL TEST STATISTICAL ROBUSTNESS"
    )
    print(
        "=============================================="
    )

    stats = (
        run_statistical_analysis()
    )

    print()
    print(
        "=============================================="
    )
    print(
        "EXACT / COMPOSITION RECOVERY SOURCE"
    )
    print(
        "=============================================="
    )

    recovery = (
        run_recovery_analysis()
    )

    payload = {
        "schema_version": 1,
        "test_queries":
            EXPECTED_ROWS,
        "bootstrap": {
            "method":
                "paired nonparametric percentile bootstrap",
            "repetitions":
                BOOTSTRAP_REPS,
            "seed":
                BOOTSTRAP_SEED,
            "ci":
                0.95,
        },
        "statistical_robustness":
            stats,
        "recovery_source_analysis":
            recovery,
        "interpretation_boundary": (
            "Recovery-source counts describe whether "
            "newly recovered Gold candidates have Exact, "
            "Composition, or both forms of support. "
            "They are not an ablation estimate of the "
            "independent Top-1 causal contribution of "
            "Composition."
        ),
    }

    json_path = (
        OUT
        / "statistical_robustness_and_recovery_v1.json"
    )

    json_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    stats_csv = (
        write_statistics_csv(
            stats
        )
    )

    recovery_csv = (
        write_recovery_csv(
            recovery
        )
    )

    print()
    print(
        "===== STATISTICAL RESULTS ====="
    )

    for name, x in stats.items():
        m = x["mcnemar"]

        print()
        print(name)
        print(
            "  Top1:",
            f'{100*x["left_top1"]:.3f}%'
            " -> "
            f'{100*x["right_top1"]:.3f}%'
        )
        print(
            "  Delta Top1 =",
            f'{x["delta_top1_pp"]:+.3f} pp'
        )
        print(
            "  95% paired bootstrap CI =",
            f'[{x["delta_top1_ci95_low_pp"]:+.3f}, '
            f'{x["delta_top1_ci95_high_pp"]:+.3f}] pp'
        )
        print(
            "  McNemar left-correct/right-wrong =",
            m[
                "left_correct_right_wrong"
            ],
        )
        print(
            "  McNemar left-wrong/right-correct =",
            m[
                "left_wrong_right_correct"
            ],
        )
        print(
            "  McNemar exact two-sided p =",
            m[
                "exact_two_sided_p"
            ],
        )
        print(
            "  McNemar log10(p) =",
            f'{m["log10_p"]:.3f}',
        )
        print(
            "  MRR:",
            f'{x["left_mrr"]:.6f}'
            " -> "
            f'{x["right_mrr"]:.6f}'
        )
        print(
            "  Delta MRR =",
            f'{x["delta_mrr"]:+.6f}'
        )
        print(
            "  95% paired bootstrap CI =",
            f'[{x["delta_mrr_ci95_low"]:+.6f}, '
            f'{x["delta_mrr_ci95_high"]:+.6f}]'
        )

    print()
    print(
        "===== RECOVERY SOURCE RESULTS ====="
    )

    for system, x in (
        recovery.items()
    ):
        print()
        print(system)
        print(
            "  Newly recovered =",
            x["newly_recovered"],
        )

        for source, s in (
            x[
                "recovery_support"
            ].items()
        ):
            print(
                f"  {source:24s}",
                f'{s["count"]:5d}',
                f'({100*s["share_of_newly_recovered"]:.3f}% '
                f'of recovered)',
            )

        print(
            "  --- by M ---"
        )

        for M, m in (
            x["by_M"].items()
        ):
            print(
                f"  M{M}:",
                "total=",
                m[
                    "total_newly_recovered"
                ],
                "exact_only=",
                m[
                    "exact_only"
                ]["count"],
                "composition_only=",
                m[
                    "composition_only"
                ]["count"],
                "both=",
                m[
                    "exact_and_composition"
                ]["count"],
            )

    print()
    print(
        "STATISTICAL_ROBUSTNESS_GATE=PASS"
    )
    print(
        "RECOVERY_SOURCE_GATE=PASS"
    )
    print(
        "NO_MODEL_TRAINING=PASS"
    )
    print(
        "NO_MODEL_UPDATE=PASS"
    )

    print()
    print(
        "JSON =", json_path
    )
    print(
        "STATISTICS_CSV =",
        stats_csv,
    )
    print(
        "RECOVERY_CSV =",
        recovery_csv,
    )


if __name__ == "__main__":
    main()
