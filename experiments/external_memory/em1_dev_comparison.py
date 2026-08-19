from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.personalisation.external_memory import (
    rank_of,
    rank_recovery_frequency,
    rank_recovery_only,
    unified_pool,
)


AUTHORS = (
    "Etinjat",
    "Re_spectators",
    "breaddddd",
)

K_VALUES = (1, 3, 5)

LAMBDA_VALUES = (
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
)

GENERIC_SHA256 = (
    "588aa84c6397e8cb1a13576c0d5dfecd9dd2c4305b45be351328dd83ef62007d"
)

PV_SHA256 = (
    "5d367b1bf2294e0d9ff4102d26cb4dd4732d1c1d520a20a86086377d3b0bcbc5"
)

EXPECTED_ROWS = 5608
EXPECTED_RECOVERY_ROWS = 941
EXPECTED_RECOVERY_SCORES = 1970


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as source:
        return [
            json.loads(line)
            for line in source
            if line.strip()
        ]


def load_generic(
    path: Path,
) -> dict[str, dict[str, Any]]:
    actual_hash = sha256_file(path)

    if actual_hash != GENERIC_SHA256:
        raise RuntimeError(
            "Generic cache SHA mismatch:\n"
            f"expected={GENERIC_SHA256}\n"
            f"actual={actual_hash}"
        )

    output = {}

    for row in load_jsonl(path):
        if row.get("pilot_partition") != "tune":
            continue

        if str(row.get("author")) not in AUTHORS:
            continue

        output[str(row["row_id"])] = row

    if len(output) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} three-author rows; "
            f"found {len(output)}"
        )

    return output


def load_states(
    path: Path,
) -> dict[str, dict[str, Any]]:
    actual_hash = sha256_file(path)

    if actual_hash != PV_SHA256:
        raise RuntimeError(
            "PV state SHA mismatch:\n"
            f"expected={PV_SHA256}\n"
            f"actual={actual_hash}"
        )

    output = {}

    for row in load_jsonl(path):
        if str(row.get("author")) not in AUTHORS:
            continue

        output[str(row["row_id"])] = row

    if len(output) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} three-author PV states; "
            f"found {len(output)}"
        )

    return output


def load_recovered(
    path: Path,
) -> dict[str, dict[str, Any]]:
    output = {}

    candidate_scores = 0

    for row in load_jsonl(path):
        row_id = str(row["row_id"])

        if row_id in output:
            raise RuntimeError(
                f"Duplicate recovered-score row: {row_id}"
            )

        output[row_id] = row

        candidate_scores += int(
            row["scored_candidate_count"]
        )

    if len(output) != EXPECTED_RECOVERY_ROWS:
        raise RuntimeError(
            "Recovered score cache row count differs: "
            f"expected={EXPECTED_RECOVERY_ROWS} "
            f"actual={len(output)}"
        )

    if candidate_scores != EXPECTED_RECOVERY_SCORES:
        raise RuntimeError(
            "Recovered candidate score count differs: "
            f"expected={EXPECTED_RECOVERY_SCORES} "
            f"actual={candidate_scores}"
        )

    return output


def load_reachability(
    path: Path,
) -> dict[str, bool]:
    output = {}

    for row in load_jsonl(path):
        output[str(row["row_id"])] = bool(
            row["gold_backend_compatible"]
        )

    if len(output) != EXPECTED_ROWS:
        raise RuntimeError(
            "Reachability row count differs: "
            f"expected={EXPECTED_ROWS} "
            f"actual={len(output)}"
        )

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
    row: Mapping[str, Any],
    gold: str,
) -> int | None:
    for candidate in row["top10_candidates"]:
        if str(candidate["text"]) == gold:
            return int(candidate["rank"])

    return None


def frequency_rank(
    state: Mapping[str, Any],
    gold: str,
) -> int | None:
    for candidate in state[
        "generic_frequency_ranked"
    ]:
        if str(candidate["candidate"]) == gold:
            return int(candidate["rank"])

    return None


def metric_values(
    ranks: Sequence[int | None],
) -> dict[str, float | int | None]:
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
            0.0
            if rank is None
            else 1.0 / rank
            for rank in ranks
        ) / n,
        "missing_at_10": sum(
            rank is None
            for rank in ranks
        ) / n,
    }


def aggregate_metrics(
    rows: Sequence[Mapping[str, Any]],
    config: str,
    predicate=None,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if predicate is None or predicate(row)
    ]

    ranks = [
        row["ranks"][config]
        for row in selected
    ]

    micro = metric_values(ranks)

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

    macro_top1 = (
        statistics.fmean(
            author_top1.values()
        )
        if author_top1
        else None
    )

    return {
        **micro,
        "macro_author_top1": macro_top1,
        "author_top1": author_top1,
    }


def recovery_metrics(
    rows: Sequence[Mapping[str, Any]],
    config: str,
    *,
    k: int,
) -> dict[str, Any]:
    reachable_missing = [
        row
        for row in rows
        if row["backend_reachable"]
        and row["generic_rank"] is None
    ]

    pool_recovered = [
        row
        for row in reachable_missing
        if row[
            f"gold_in_pool_k{k}"
        ]
    ]

    top10 = sum(
        row["ranks"][config] is not None
        for row in pool_recovered
    )

    top3 = sum(
        row["ranks"][config] is not None
        and row["ranks"][config] <= 3
        for row in pool_recovered
    )

    top1 = sum(
        row["ranks"][config] == 1
        for row in pool_recovered
    )

    rescue = sum(
        row["generic_rank"] != 1
        and row["ranks"][config] == 1
        for row in rows
    )

    harm = sum(
        row["generic_rank"] == 1
        and row["ranks"][config] != 1
        for row in rows
    )

    covered_top10_dropped = sum(
        row["generic_rank"] is not None
        and row["ranks"][config] is None
        for row in rows
    )

    return {
        "reachable_generic_missing": len(
            reachable_missing
        ),
        "pool_recovered": len(
            pool_recovered
        ),
        "pool_recovered_rate": (
            len(pool_recovered)
            / len(reachable_missing)
            if reachable_missing
            else 0.0
        ),
        "recovered_to_top10": top10,
        "recovered_to_top3": top3,
        "recovered_to_top1": top1,
        "top10_conversion_of_pool": (
            top10 / len(pool_recovered)
            if pool_recovered
            else 0.0
        ),
        "top3_conversion_of_pool": (
            top3 / len(pool_recovered)
            if pool_recovered
            else 0.0
        ),
        "top1_conversion_of_pool": (
            top1 / len(pool_recovered)
            if pool_recovered
            else 0.0
        ),
        "rescue_vs_generic_top1": rescue,
        "harm_vs_generic_top1": harm,
        "net_top1_change": rescue - harm,
        "generic_covered_top10_dropped": (
            covered_top10_dropped
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

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
        "--recovered-scores",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--reachability",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    generic = load_generic(
        args.generic_cache
    )

    states = load_states(
        args.dev_states
    )

    recovered = load_recovered(
        args.recovered_scores
    )

    reachability = load_reachability(
        args.reachability
    )

    expected_ids = set(generic)

    if set(states) != expected_ids:
        raise RuntimeError(
            "Generic/PV row IDs differ"
        )

    if set(reachability) != expected_ids:
        raise RuntimeError(
            "Generic/reachability row IDs differ"
        )

    rows = []

    config_names = [
        "G0",
        "F",
    ]

    for k in K_VALUES:
        config_names.append(
            f"R_K{k}"
        )

    for k in K_VALUES:
        for value in LAMBDA_VALUES:
            config_names.append(
                f"RF_K{k}_L{value:g}"
            )

    for row_id in sorted(generic):
        row = generic[row_id]
        state = states[row_id]

        gold = str(row["target"])

        generic_candidates = (
            row["top10_candidates"]
        )

        recovered_row = recovered.get(
            row_id
        )

        recovered_scores = (
            recovered_row["scores"]
            if recovered_row is not None
            else []
        )

        counts = lexicon_counts(
            state
        )

        flags = subset_flags(
            state,
            gold,
        )

        ranks: dict[str, int | None] = {}

        ranks["G0"] = generic_rank(
            row,
            gold,
        )

        ranks["F"] = frequency_rank(
            state,
            gold,
        )

        gold_in_pool = {}

        for k in K_VALUES:
            pool = unified_pool(
                generic_candidates,
                recovered_scores,
                k_recovery=k,
            )

            pool_texts = {
                str(candidate["candidate"])
                for candidate in pool
            }

            gold_in_pool[k] = (
                gold in pool_texts
                and ranks["G0"] is None
            )

            ranked_r = rank_recovery_only(
                pool
            )

            ranks[
                f"R_K{k}"
            ] = rank_of(
                ranked_r,
                gold,
            )

            for value in LAMBDA_VALUES:
                ranked_rf = (
                    rank_recovery_frequency(
                        pool,
                        counts,
                        lambda_frequency=value,
                    )
                )

                ranks[
                    f"RF_K{k}_L{value:g}"
                ] = rank_of(
                    ranked_rf,
                    gold,
                )

        rows.append(
            {
                "row_id": row_id,
                "author": str(
                    row["author"]
                ),
                "gold": gold,
                "generic_rank": (
                    ranks["G0"]
                ),
                "backend_reachable": (
                    reachability[row_id]
                ),
                **flags,
                **{
                    f"gold_in_pool_k{k}": (
                        gold_in_pool[k]
                    )
                    for k in K_VALUES
                },
                "ranks": ranks,
            }
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

    results = {}

    for config in config_names:
        config_result = {
            "metrics": {}
        }

        for subset_name, predicate in subsets.items():
            config_result[
                "metrics"
            ][subset_name] = aggregate_metrics(
                rows,
                config,
                predicate,
            )

        if config.startswith("R_K"):
            k = int(
                config.split("K")[1]
            )

            config_result[
                "recovery"
            ] = recovery_metrics(
                rows,
                config,
                k=k,
            )

        elif config.startswith("RF_K"):
            prefix = config.split("_")[1]
            k = int(
                prefix[1:]
            )

            config_result[
                "recovery"
            ] = recovery_metrics(
                rows,
                config,
                k=k,
            )

        results[config] = (
            config_result
        )

    rf_configs = []

    for k in K_VALUES:
        for value in LAMBDA_VALUES:
            name = (
                f"RF_K{k}_L{value:g}"
            )

            macro = results[
                name
            ]["metrics"]["overall"][
                "macro_author_top1"
            ]

            rf_configs.append(
                (
                    name,
                    k,
                    value,
                    float(macro),
                )
            )

    selected_name, selected_k, selected_lambda, selected_macro = sorted(
        rf_configs,
        key=lambda value: (
            -value[3],
            value[2],
            value[1],
        ),
    )[0]

    summary = {
        "schema_version": 1,
        "experiment": (
            "em1_dev_comparison"
        ),
        "condition": "Full+Short",
        "partition": "dev_tune",
        "history_budget": 5000,
        "authors": list(AUTHORS),
        "rows": len(rows),
        "k_grid": list(K_VALUES),
        "lambda_frequency_grid": list(
            LAMBDA_VALUES
        ),
        "selection_rule": (
            "Primary: Macro-author Overall Top1 "
            "among R+F configurations. "
            "Exact ties: lower lambda_frequency, "
            "then lower K."
        ),
        "selected_r_plus_f": {
            "config": selected_name,
            "k": selected_k,
            "lambda_frequency": (
                selected_lambda
            ),
            "macro_author_overall_top1": (
                selected_macro
            ),
        },
        "results": results,
        "provenance": {
            "generic_cache_sha256": (
                sha256_file(
                    args.generic_cache
                )
            ),
            "pv_states_sha256": (
                sha256_file(
                    args.dev_states
                )
            ),
            "recovered_scores_sha256": (
                sha256_file(
                    args.recovered_scores
                )
            ),
            "reachability_sha256": (
                sha256_file(
                    args.reachability
                )
            ),
        },
        "test_rows_used_for_parameter_selection": 0,
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
    print(
        "=== EM-1 Dev Comparison ==="
    )

    print(
        "Config                  "
        "Micro Top1   Macro Top1   "
        "Top3       MRR@10"
    )

    print("-" * 68)

    for config in config_names:
        metrics = results[
            config
        ]["metrics"]["overall"]

        print(
            f"{config:<23}"
            f"{100 * metrics['top1']:>8.3f}%   "
            f"{100 * metrics['macro_author_top1']:>8.3f}%   "
            f"{100 * metrics['top3']:>8.3f}%   "
            f"{metrics['mrr_at_10']:>7.4f}"
        )

    print()
    print(
        "Selected R+F:"
    )

    print(
        f"  config={selected_name}"
    )

    print(
        f"  K={selected_k}"
    )

    print(
        f"  lambda_frequency="
        f"{selected_lambda:g}"
    )

    print(
        "  Macro-author Overall Top1="
        f"{100 * selected_macro:.3f}%"
    )

    selected_recovery = results[
        selected_name
    ]["recovery"]

    print()
    print(
        "Selected R+F Recovery:"
    )

    print(
        "  Reachable Generic Missing: "
        f"{selected_recovery['reachable_generic_missing']}"
    )

    print(
        "  Recovered to pool: "
        f"{selected_recovery['pool_recovered']}"
    )

    print(
        "  Recovered to Top10: "
        f"{selected_recovery['recovered_to_top10']}"
    )

    print(
        "  Recovered to Top3: "
        f"{selected_recovery['recovered_to_top3']}"
    )

    print(
        "  Recovered to Top1: "
        f"{selected_recovery['recovered_to_top1']}"
    )

    print(
        "  Rescue / Harm / Net: "
        f"{selected_recovery['rescue_vs_generic_top1']} / "
        f"{selected_recovery['harm_vs_generic_top1']} / "
        f"{selected_recovery['net_top1_change']}"
    )

    print(
        "  Generic-covered Top10 dropped: "
        f"{selected_recovery['generic_covered_top10_dropped']}"
    )

    print()
    print(
        "Dev selection complete. "
        "Do not inspect Test for parameter choice."
    )


if __name__ == "__main__":
    main()
