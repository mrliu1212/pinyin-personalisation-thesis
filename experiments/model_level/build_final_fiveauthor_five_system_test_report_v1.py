from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(
    "results/finalmodel_fiveauthor_v1"
)

GENERIC = (
    ROOT
    / "generic_baseline_v1/run_v2/"
    "generic_test_predictions.jsonl"
)

ADAPTER_ROOT = (
    ROOT
    / "test_end_to_end_v1/"
    "adapter_test_top10_v1"
)

FREQ = (
    ROOT
    / "generic_frequency_test_v1/"
    "final_fiveauthor_generic_frequency_test_surface_v1.jsonl"
)

GENERIC_EXPLICIT = (
    ROOT
    / "generic_explicit_test_rich30_v1/"
    "final_fiveauthor_generic_explicit_test_rich30_surface_v1.jsonl"
)

FINAL = (
    ROOT
    / "test_end_to_end_v1/rich30_test_v1/"
    "final_fiveauthor_test_rich30_surface_v1.jsonl"
)

OUT = (
    ROOT
    / "five_system_test_comparison_v1"
)

EXPECTED = 40_000

SYSTEMS = [
    "Generic",
    "Generic+Frequency",
    "Generic+Explicit",
    "Adapter",
    "Adapter+Explicit",
]


def load_jsonl(path):
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def map_rows(rows):
    out = {}
    for r in rows:
        rid = str(r["row_id"])
        if rid in out:
            raise RuntimeError(
                f"Duplicate row_id {rid}"
            )
        out[rid] = r
    return out


def load_adapter():
    out = {}

    for path in sorted(
        ADAPTER_ROOT.glob(
            "*_test_top10_v1.jsonl"
        )
    ):
        for r in load_jsonl(path):
            rid = str(r["row_id"])

            if rid in out:
                raise RuntimeError(
                    f"Duplicate Adapter row {rid}"
                )

            out[rid] = r

    if len(out) != EXPECTED:
        raise RuntimeError(
            f"Adapter rows={len(out)}"
        )

    return out


def rank_of(ranking, gold):
    try:
        return ranking.index(gold) + 1
    except ValueError:
        return None


def new_stats():
    return {
        "N": 0,
        "top1": 0,
        "top3": 0,
        "top5": 0,
        "top10": 0,
        "mrr_sum": 0.0,
        "mrr5_sum": 0.0,
        "pool_hit": 0,
        "missing": 0,
    }


def add(stats, rank):
    stats["N"] += 1

    if rank is None:
        stats["missing"] += 1
        return

    stats["pool_hit"] += 1
    stats["mrr_sum"] += 1.0 / rank

    if rank <= 5:
        stats["mrr5_sum"] += (
            1.0 / rank
        )

    if rank <= 1:
        stats["top1"] += 1
    if rank <= 3:
        stats["top3"] += 1
    if rank <= 5:
        stats["top5"] += 1
    if rank <= 10:
        stats["top10"] += 1


def finish(x):
    n = x["N"]

    return {
        "N": n,
        "top1": x["top1"] / n,
        "top3": x["top3"] / n,
        "top5": x["top5"] / n,
        "top10": x["top10"] / n,
        "mrr": x["mrr_sum"] / n,
        "mrr_at_5":
            x["mrr5_sum"] / n,
        "pool_hit":
            x["pool_hit"] / n,
        "missing":
            x["missing"] / n,
        "top1_count": x["top1"],
        "top3_count": x["top3"],
        "top5_count": x["top5"],
        "top10_count": x["top10"],
        "pool_hit_count":
            x["pool_hit"],
        "missing_count":
            x["missing"],
    }


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    generic = map_rows(
        load_jsonl(GENERIC)
    )
    adapter = load_adapter()
    freq = map_rows(
        load_jsonl(FREQ)
    )
    gexp = map_rows(
        load_jsonl(
            GENERIC_EXPLICIT
        )
    )
    final = map_rows(
        load_jsonl(FINAL)
    )

    universes = {
        "Generic": set(generic),
        "Adapter": set(adapter),
        "Frequency": set(freq),
        "GenericExplicit": set(gexp),
        "Final": set(final),
    }

    for name, ids in universes.items():
        if len(ids) != EXPECTED:
            raise RuntimeError(
                f"{name} rows="
                f"{len(ids)}"
            )

    base_ids = set(generic)

    for name, ids in universes.items():
        if ids != base_ids:
            raise RuntimeError(
                f"{name} row universe mismatch"
            )

    print(
        "FIVE_SYSTEM_ROW_UNIVERSE=PASS"
    )

    stats = {
        system: {
            "overall":
                new_stats(),
            "author":
                defaultdict(new_stats),
            "M":
                defaultdict(new_stats),
            "typing":
                defaultdict(new_stats),
            "effective_typing":
                defaultdict(new_stats),
        }
        for system in SYSTEMS
    }

    condition_counts = {
        "author": defaultdict(int),
        "M": defaultdict(int),
        "typing":
            defaultdict(int),
        "effective_typing":
            defaultdict(int),
    }

    recovery = {
        "Generic+Frequency": {
            "neural_hit": 0,
            "pool_hit": 0,
            "newly_recovered": 0,
            "pool_but_not_top10": 0,
        },
        "Generic+Explicit": {
            "neural_hit": 0,
            "pool_hit": 0,
            "newly_recovered": 0,
            "pool_but_not_top10": 0,
        },
        "Adapter+Explicit": {
            "neural_hit": 0,
            "pool_hit": 0,
            "newly_recovered": 0,
            "pool_but_not_top10": 0,
        },
    }

    row_surface = []

    for rid in sorted(base_ids):
        g = generic[rid]
        a = adapter[rid]
        f = freq[rid]
        ge = gexp[rid]
        fi = final[rid]

        gold = str(g["gold"])

        # Cross-artifact identity gates.
        for name, r in [
            ("Adapter", a),
            ("Frequency", f),
            ("GenericExplicit", ge),
            ("Final", fi),
        ]:
            if str(r["gold"]) != gold:
                raise RuntimeError(
                    f"{rid}: Gold mismatch "
                    f"in {name}"
                )

        author = str(
            g["author_name"]
        )
        M = str(g["M"])
        typing = str(
            g["typing_mode"]
        )
        effective = str(
            g[
                "effective_typing_mode"
            ]
        )

        condition_counts[
            "author"
        ][author] += 1
        condition_counts[
            "M"
        ][M] += 1
        condition_counts[
            "typing"
        ][typing] += 1
        condition_counts[
            "effective_typing"
        ][effective] += 1

        rankings = {
            "Generic":
                list(
                    map(
                        str,
                        g[
                            "top10_candidates"
                        ],
                    )
                )[:10],

            "Generic+Frequency":
                list(
                    map(
                        str,
                        f[
                            "final_ranking"
                        ],
                    )
                ),

            "Generic+Explicit":
                list(
                    map(
                        str,
                        ge[
                            "final_ranking"
                        ],
                    )
                ),

            "Adapter":
                list(
                    map(
                        str,
                        a[
                            "top10_candidates"
                        ],
                    )
                )[:10],

            "Adapter+Explicit":
                list(
                    map(
                        str,
                        fi[
                            "final_ranking"
                        ],
                    )
                ),
        }

        ranks = {
            system:
                rank_of(
                    ranking,
                    gold,
                )
            for system, ranking
            in rankings.items()
        }

        for system in SYSTEMS:
            r = ranks[system]

            add(
                stats[system][
                    "overall"
                ],
                r,
            )
            add(
                stats[system][
                    "author"
                ][author],
                r,
            )
            add(
                stats[system][
                    "M"
                ][M],
                r,
            )
            add(
                stats[system][
                    "typing"
                ][typing],
                r,
            )
            add(
                stats[system][
                    "effective_typing"
                ][effective],
                r,
            )

        # ----------------------------------
        # Candidate recovery
        # ----------------------------------
        comparisons = [
            (
                "Generic+Frequency",
                rankings["Generic"],
                rankings[
                    "Generic+Frequency"
                ],
                f["final_pool"],
            ),
            (
                "Generic+Explicit",
                rankings["Generic"],
                rankings[
                    "Generic+Explicit"
                ],
                ge["final_pool"],
            ),
            (
                "Adapter+Explicit",
                rankings["Adapter"],
                rankings[
                    "Adapter+Explicit"
                ],
                fi["final_pool"],
            ),
        ]

        for (
            name,
            neural,
            fused_ranking,
            pool,
        ) in comparisons:
            neural_hit = (
                gold in neural
            )
            pool_hit = (
                gold in pool
            )
            top10_hit = (
                gold
                in fused_ranking[:10]
            )

            if neural_hit:
                recovery[name][
                    "neural_hit"
                ] += 1

            if pool_hit:
                recovery[name][
                    "pool_hit"
                ] += 1

            if (
                not neural_hit
                and pool_hit
            ):
                recovery[name][
                    "newly_recovered"
                ] += 1

            if (
                pool_hit
                and not top10_hit
            ):
                recovery[name][
                    "pool_but_not_top10"
                ] += 1

        row_surface.append({
            "row_id": rid,
            "author_name": author,
            "M": int(M),
            "typing_mode": typing,
            "effective_typing_mode":
                effective,
            "gold": gold,
            **{
                system:
                    ranks[system]
                for system
                in SYSTEMS
            },
        })

    # --------------------------------------
    # Finish statistics
    # --------------------------------------
    report = {}

    for system in SYSTEMS:
        report[system] = {
            "overall":
                finish(
                    stats[system][
                        "overall"
                    ]
                ),
            "by_author": {
                k: finish(v)
                for k, v in sorted(
                    stats[system][
                        "author"
                    ].items()
                )
            },
            "by_M": {
                k: finish(v)
                for k, v in sorted(
                    stats[system][
                        "M"
                    ].items(),
                    key=lambda x:
                        int(x[0]),
                )
            },
            "by_typing_mode": {
                k: finish(v)
                for k, v in sorted(
                    stats[system][
                        "typing"
                    ].items()
                )
            },
            "by_effective_typing_mode": {
                k: finish(v)
                for k, v in sorted(
                    stats[system][
                        "effective_typing"
                    ].items()
                )
            },
        }

    # --------------------------------------
    # Pairwise gains
    # --------------------------------------
    pairs = [
        (
            "Generic",
            "Generic+Frequency",
        ),
        (
            "Generic",
            "Generic+Explicit",
        ),
        (
            "Generic",
            "Adapter",
        ),
        (
            "Adapter",
            "Adapter+Explicit",
        ),
        (
            "Generic+Frequency",
            "Generic+Explicit",
        ),
    ]

    pairwise = {}

    for left, right in pairs:
        a = report[left][
            "overall"
        ]
        b = report[right][
            "overall"
        ]

        pairwise[
            f"{left} -> {right}"
        ] = {
            "delta_top1":
                b["top1"]
                - a["top1"],
            "delta_top3":
                b["top3"]
                - a["top3"],
            "delta_top5":
                b["top5"]
                - a["top5"],
            "delta_top10":
                b["top10"]
                - a["top10"],
            "delta_mrr":
                b["mrr"]
                - a["mrr"],
            "delta_missing":
                b["missing"]
                - a["missing"],
        }

    for name in recovery:
        for k in list(
            recovery[name]
        ):
            recovery[name][
                k + "_rate"
            ] = (
                recovery[name][k]
                / EXPECTED
            )

    final_json = {
        "schema_version": 1,
        "test_queries":
            EXPECTED,
        "systems":
            SYSTEMS,
        "condition_counts": {
            dimension:
                dict(
                    sorted(
                        values.items()
                    )
                )
            for dimension, values
            in condition_counts.items()
        },
        "results":
            report,
        "pairwise_gains":
            pairwise,
        "candidate_recovery":
            recovery,
    }

    json_path = (
        OUT
        / "five_system_test_comparison_v1.json"
    )

    json_path.write_text(
        json.dumps(
            final_json,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------
    # Overall CSV
    # --------------------------------------
    overall_csv = (
        OUT
        / "overall_v1.csv"
    )

    with overall_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)

        w.writerow([
            "system",
            "N",
            "top1",
            "top3",
            "top5",
            "top10",
            "mrr",
            "mrr_at_5",
            "pool_hit",
            "missing",
        ])

        for system in SYSTEMS:
            x = report[system][
                "overall"
            ]

            w.writerow([
                system,
                x["N"],
                x["top1"],
                x["top3"],
                x["top5"],
                x["top10"],
                x["mrr"],
                x["mrr_at_5"],
                x["pool_hit"],
                x["missing"],
            ])

    # --------------------------------------
    # Condition CSV
    # --------------------------------------
    condition_csv = (
        OUT
        / "conditions_v1.csv"
    )

    dimension_map = [
        (
            "author",
            "by_author",
        ),
        (
            "M",
            "by_M",
        ),
        (
            "typing_mode",
            "by_typing_mode",
        ),
        (
            "effective_typing_mode",
            "by_effective_typing_mode",
        ),
    ]

    with condition_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)

        w.writerow([
            "dimension",
            "condition",
            "system",
            "N",
            "top1",
            "top3",
            "top5",
            "top10",
            "mrr",
            "mrr_at_5",
            "pool_hit",
            "missing",
        ])

        for (
            dimension,
            key,
        ) in dimension_map:

            conditions = (
                report[
                    SYSTEMS[0]
                ][key].keys()
            )

            for condition in conditions:
                for system in SYSTEMS:
                    x = (
                        report[
                            system
                        ][key][
                            condition
                        ]
                    )

                    w.writerow([
                        dimension,
                        condition,
                        system,
                        x["N"],
                        x["top1"],
                        x["top3"],
                        x["top5"],
                        x["top10"],
                        x["mrr"],
                        x["mrr_at_5"],
                        x["pool_hit"],
                        x["missing"],
                    ])

    # --------------------------------------
    # Row-level aligned rank surface
    # --------------------------------------
    row_csv = (
        OUT
        / "row_level_ranks_v1.csv"
    )

    with row_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = [
            "row_id",
            "author_name",
            "M",
            "typing_mode",
            "effective_typing_mode",
            "gold",
            *SYSTEMS,
        ]

        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        w.writeheader()

        for r in row_surface:
            w.writerow(r)

    print()
    print("===== OVERALL =====")

    for system in SYSTEMS:
        x = report[system][
            "overall"
        ]

        print(
            system,
            "N=", x["N"],
            "Top1=", f'{x["top1"]:.6f}',
            "Top3=", f'{x["top3"]:.6f}',
            "Top5=", f'{x["top5"]:.6f}',
            "Top10=", f'{x["top10"]:.6f}',
            "MRR=", f'{x["mrr"]:.6f}',
            "Missing=", f'{x["missing"]:.6f}',
        )

    print()
    print("===== CONDITION COUNTS =====")
    print(json.dumps(
        final_json[
            "condition_counts"
        ],
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ))

    print()
    print("===== PAIRWISE GAINS =====")
    print(json.dumps(
        pairwise,
        indent=2,
        sort_keys=True,
    ))

    print()
    print("===== CANDIDATE RECOVERY =====")
    print(json.dumps(
        recovery,
        indent=2,
        sort_keys=True,
    ))

    print()
    print(
        "FIVE_SYSTEM_TEST_COMPARISON_GATE=PASS"
    )
    print(
        "JSON =", json_path
    )
    print(
        "OVERALL_CSV =", overall_csv
    )
    print(
        "CONDITION_CSV =", condition_csv
    )
    print(
        "ROW_CSV =", row_csv
    )


if __name__ == "__main__":
    main()
