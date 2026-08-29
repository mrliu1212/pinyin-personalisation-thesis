from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_ROWS = 40_000
FREQ_BUDGET = 5


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        "frozen_val_rich30_v1",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def event_target(event):
    for key in ("target", "gold", "text"):
        if key in event:
            return str(event[key])
    raise RuntimeError(
        f"Cannot find target field: {sorted(event)}"
    )


def event_segments(event):
    for key in (
        "raw_segments",
        "segmented_pinyin",
        "full_pinyin",
        "pinyin",
    ):
        if key in event:
            v = event[key]
            if isinstance(v, str):
                return v.split()
            return [str(x) for x in v]

    raise RuntimeError(
        f"Cannot find Pinyin field: {sorted(event)}"
    )


def query_segments(generic_row):
    if "segmented_pinyin" in generic_row:
        v = generic_row["segmented_pinyin"]
    else:
        v = generic_row["pinyin_input"]

    if isinstance(v, str):
        return v.split()

    return [str(x) for x in v]


def segment_compatible(q, full):
    q = str(q).lower()
    full = str(full).lower()

    return (
        q == full
        or (
            len(q) == 1
            and len(full) >= 1
            and q == full[0]
        )
    )


def compatible(query_parts, full_parts):
    if len(query_parts) != len(full_parts):
        return False

    return all(
        segment_compatible(q, f)
        for q, f in zip(
            query_parts,
            full_parts,
        )
    )


def new_stats():
    return {
        "queries": 0,
        "top1": 0,
        "top3": 0,
        "top5": 0,
        "top10": 0,
        "mrr_sum": 0.0,
        "pool_hit": 0,
        "missing": 0,
    }


def add_result(stats, rank):
    stats["queries"] += 1

    if rank is None:
        stats["missing"] += 1
        return

    stats["pool_hit"] += 1
    stats["mrr_sum"] += 1.0 / rank

    if rank <= 1:
        stats["top1"] += 1
    if rank <= 3:
        stats["top3"] += 1
    if rank <= 5:
        stats["top5"] += 1
    if rank <= 10:
        stats["top10"] += 1


def finish_stats(x):
    n = x["queries"]

    return {
        **x,
        "top1_rate": x["top1"] / n,
        "top3_rate": x["top3"] / n,
        "top5_rate": x["top5"] / n,
        "top10_rate": x["top10"] / n,
        "mrr": x["mrr_sum"] / n,
        "pool_hit_rate": x["pool_hit"] / n,
        "missing_rate": x["missing"] / n,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--generic-predictions",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--h5000",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--val-module",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--raw-history",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--work-split",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    args = ap.parse_args()
    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    vm = load_module(args.val_module)

    # --------------------------------------------------
    # Generic Test
    # --------------------------------------------------
    generic = {}

    for r in load_jsonl(
        args.generic_predictions
    ):
        rid = str(r["row_id"])

        if rid in generic:
            raise RuntimeError(
                f"Duplicate Generic row {rid}"
            )

        if str(r["split"]).lower() != "test":
            raise RuntimeError(
                f"{rid}: non-Test Generic row"
            )

        if bool(r.get("used_adapter", False)):
            raise RuntimeError(
                f"{rid}: Generic used adapter"
            )

        if bool(r.get("used_h5000", False)):
            raise RuntimeError(
                f"{rid}: Generic used H5000"
            )

        if bool(r.get("used_rich30", False)):
            raise RuntimeError(
                f"{rid}: Generic used Rich30"
            )

        generic[rid] = r

    if len(generic) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Generic Test rows={len(generic)}"
        )

    print(
        "GENERIC_TEST_ROWS = 40000",
        flush=True,
    )

    # --------------------------------------------------
    # Raw chronological history
    # --------------------------------------------------
    print(
        "===== LOAD RAW HISTORY =====",
        flush=True,
    )

    work_order = vm.load_work_order(
        args.work_split
    )

    raw_by_author = vm.load_raw_history(
        args.raw_history,
        work_order,
    )

    print(
        "RAW_HISTORY_READY=PASS",
        flush=True,
    )

    overall = new_stats()
    by_author = defaultdict(new_stats)
    by_M = defaultdict(new_stats)
    by_typing = defaultdict(new_stats)
    by_effective = defaultdict(new_stats)

    generic_top10_hits = 0
    fused_pool_hits = 0
    frequency_recovered = 0

    seen = set()

    surface_path = (
        args.output_root
        / "final_fiveauthor_generic_frequency_test_surface_v1.jsonl"
    )

    with surface_path.open(
        "w",
        encoding="utf-8",
    ) as surface:

        for number, h in enumerate(
            load_jsonl(args.h5000),
            start=1,
        ):
            rid = str(h["row_id"])

            if rid in seen:
                raise RuntimeError(
                    f"Duplicate H5000 row {rid}"
                )
            seen.add(rid)

            if rid not in generic:
                raise RuntimeError(
                    f"H5000 row absent from Generic: {rid}"
                )

            if str(h["split"]).lower() != "test":
                raise RuntimeError(
                    f"{rid}: non-Test H5000 row"
                )

            if h.get("used_current_test"):
                raise RuntimeError(
                    f"{rid}: current-Test leakage"
                )

            if h.get("used_future_test"):
                raise RuntimeError(
                    f"{rid}: future-Test leakage"
                )

            g = generic[rid]

            if str(g["gold"]) != str(h["gold"]):
                raise RuntimeError(
                    f"{rid}: Gold mismatch"
                )

            author = str(g["author_name"])
            gold = str(g["gold"])

            generic_texts = list(
                map(
                    str,
                    g["top10_candidates"],
                )
            )[:10]

            generic_rank = {
                text: rank
                for rank, text
                in enumerate(
                    generic_texts,
                    start=1,
                )
            }

            if gold in generic_texts:
                generic_top10_hits += 1

            query_parts = query_segments(g)

            start = int(
                h["history_start_ordinal"]
            )
            stop = int(
                h["history_stop_ordinal"]
            )

            history = raw_by_author[author][
                start:stop
            ]

            counts = Counter()

            for event in history:
                full_parts = event_segments(
                    event
                )

                if compatible(
                    query_parts,
                    full_parts,
                ):
                    counts[
                        event_target(event)
                    ] += 1

            freq_ranked = sorted(
                counts,
                key=lambda text: (
                    -counts[text],
                    text,
                ),
            )

            freq_rank = {
                text: rank
                for rank, text
                in enumerate(
                    freq_ranked,
                    start=1,
                )
            }

            freq_only = [
                text
                for text in freq_ranked
                if text not in generic_rank
            ]

            injected = freq_only[
                :FREQ_BUDGET
            ]

            pool = list(generic_texts)

            for text in injected:
                if text not in pool:
                    pool.append(text)

            if len(pool) > 15:
                raise RuntimeError(
                    f"{rid}: pool={len(pool)}"
                )

            scores = {}

            for text in pool:
                score = 0.0

                if text in generic_rank:
                    score += (
                        1.0
                        / generic_rank[text]
                    )

                if text in freq_rank:
                    score += (
                        1.0
                        / freq_rank[text]
                    )

                scores[text] = score

            # Frozen untrained reciprocal-rank fusion.
            # Tie:
            # 1 generic rank
            # 2 frequency count
            # 3 text
            ranking = sorted(
                pool,
                key=lambda text: (
                    -scores[text],
                    generic_rank.get(
                        text,
                        10**9,
                    ),
                    -counts.get(
                        text,
                        0,
                    ),
                    text,
                ),
            )

            rank = (
                ranking.index(gold) + 1
                if gold in ranking
                else None
            )

            gold_in_pool = (
                gold in pool
            )

            if gold_in_pool:
                fused_pool_hits += 1

            if (
                gold not in generic_texts
                and gold_in_pool
            ):
                frequency_recovered += 1

            add_result(
                overall,
                rank,
            )
            add_result(
                by_author[author],
                rank,
            )
            add_result(
                by_M[str(g["M"])],
                rank,
            )
            add_result(
                by_typing[
                    str(g["typing_mode"])
                ],
                rank,
            )
            add_result(
                by_effective[
                    str(
                        g[
                            "effective_typing_mode"
                        ]
                    )
                ],
                rank,
            )

            surface.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "experiment":
                            "final_fiveauthor_generic_frequency_test_v1",
                        "row_id": rid,
                        "author_name": author,
                        "split": "test",
                        "gold": gold,
                        "M": int(g["M"]),
                        "typing_mode":
                            g["typing_mode"],
                        "effective_typing_mode":
                            g[
                                "effective_typing_mode"
                            ],
                        "generic_top10":
                            generic_texts,
                        "frequency_top5":
                            freq_ranked[:5],
                        "frequency_injected":
                            injected,
                        "final_pool":
                            pool,
                        "pool_size":
                            len(pool),
                        "gold_in_pool":
                            gold_in_pool,
                        "final_ranking":
                            ranking,
                        "gold_final_rank":
                            rank,
                        "used_test":
                            True,
                        "used_current_test":
                            False,
                        "used_future_test":
                            False,
                        "trained_reranker":
                            False,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

            if (
                number % 1000 == 0
                or number == EXPECTED_ROWS
            ):
                print(
                    "GENERIC_FREQUENCY_TEST",
                    number,
                    "/40000",
                    flush=True,
                )

    if len(seen) != EXPECTED_ROWS:
        raise RuntimeError(
            f"H5000 Test rows={len(seen)}"
        )

    if seen != set(generic):
        raise RuntimeError(
            "Generic/H5000 row universe mismatch"
        )

    summary = {
        "schema_version": 1,
        "experiment":
            "final_fiveauthor_generic_frequency_test_v1",
        "test_queries":
            EXPECTED_ROWS,
        "frequency_budget":
            FREQ_BUDGET,
        "trained_reranker":
            False,
        "lambdaMART_training_on_test":
            False,
        "neural_parameter_updates_on_test":
            False,
        "current_test_visible":
            0,
        "future_test_visible":
            0,
        "generic_top10_hit":
            generic_top10_hits,
        "generic_top10_hit_rate":
            generic_top10_hits
            / EXPECTED_ROWS,
        "fused_pool_hit":
            fused_pool_hits,
        "fused_pool_hit_rate":
            fused_pool_hits
            / EXPECTED_ROWS,
        "frequency_recovered_queries":
            frequency_recovered,
        "frequency_recovery_gain":
            frequency_recovered
            / EXPECTED_ROWS,
        "overall":
            finish_stats(overall),
        "by_author": {
            k: finish_stats(v)
            for k, v
            in sorted(
                by_author.items()
            )
        },
        "by_M": {
            k: finish_stats(v)
            for k, v
            in sorted(
                by_M.items(),
                key=lambda kv: int(
                    kv[0]
                ),
            )
        },
        "by_typing_mode": {
            k: finish_stats(v)
            for k, v
            in sorted(
                by_typing.items()
            )
        },
        "by_effective_typing_mode": {
            k: finish_stats(v)
            for k, v
            in sorted(
                by_effective.items()
            )
        },
        "surface_path":
            str(surface_path),
    }

    summary_path = (
        args.output_root
        / "final_fiveauthor_generic_frequency_test_summary_v1.json"
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
        "===== GENERIC + FREQUENCY TEST RESULT ====="
    )
    print(
        json.dumps(
            summary["overall"],
            indent=2,
            sort_keys=True,
        )
    )
    print()
    print(
        "GENERIC_TOP10_HIT_RATE =",
        summary[
            "generic_top10_hit_rate"
        ],
    )
    print(
        "FUSED_POOL_HIT_RATE =",
        summary[
            "fused_pool_hit_rate"
        ],
    )
    print(
        "FREQUENCY_RECOVERED_QUERIES =",
        frequency_recovered,
    )
    print(
        "FREQUENCY_RECOVERY_GAIN =",
        summary[
            "frequency_recovery_gain"
        ],
    )
    print(
        "TRAINED_RERANKER = 0"
    )
    print(
        "CURRENT_TEST_VISIBLE = 0"
    )
    print(
        "FUTURE_TEST_VISIBLE = 0"
    )
    print(
        "GENERIC_FREQUENCY_TEST_GATE=PASS"
    )
    print(
        "SUMMARY =",
        summary_path,
    )


if __name__ == "__main__":
    main()
