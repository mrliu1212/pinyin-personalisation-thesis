from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_ROWS = 20_000
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
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def event_target(event):
    for key in ("target", "gold", "text"):
        if key in event:
            return str(event[key])
    raise RuntimeError(
        f"Cannot find target field in raw event: "
        f"{sorted(event)}"
    )


def event_segments(event):
    # load_raw_history() in the frozen implementation
    # normally exposes raw_segments. The fallbacks below
    # are only schema-safe aliases.
    if "raw_segments" in event:
        v = event["raw_segments"]
    elif "segmented_pinyin" in event:
        v = event["segmented_pinyin"]
    elif "full_pinyin" in event:
        v = event["full_pinyin"]
    elif "pinyin" in event:
        v = event["pinyin"]
    else:
        raise RuntimeError(
            f"Cannot find Pinyin segments in raw event: "
            f"{sorted(event)}"
        )

    if isinstance(v, str):
        return v.split()

    return [str(x) for x in v]


def compatible(query_parts, full_parts):
    if len(query_parts) != len(full_parts):
        return False

    for q, full in zip(query_parts, full_parts):
        q = str(q).lower()
        full = str(full).lower()

        if q == full:
            continue

        if len(q) == 1 and full and q == full[0]:
            continue

        return False

    return True


def new_stats():
    return {
        "queries": 0,
        "top1": 0,
        "top3": 0,
        "top5": 0,
        "top10": 0,
        "mrr_sum": 0.0,
        "missing": 0,
    }


def update(stats, ranking, gold):
    stats["queries"] += 1

    if gold not in ranking[:10]:
        stats["missing"] += 1

    if gold not in ranking:
        return

    rank = ranking.index(gold) + 1

    if rank <= 1:
        stats["top1"] += 1
    if rank <= 3:
        stats["top3"] += 1
    if rank <= 5:
        stats["top5"] += 1
    if rank <= 10:
        stats["top10"] += 1

    if rank <= 10:
        stats["mrr_sum"] += 1.0 / rank


def finish(stats):
    n = stats["queries"]

    return {
        "queries": n,
        "top1": stats["top1"] / n,
        "top3": stats["top3"] / n,
        "top5": stats["top5"] / n,
        "top10": stats["top10"] / n,
        "mrr": stats["mrr_sum"] / n,
        "missing": stats["missing"] / n,
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
        "--output",
        type=Path,
        required=True,
    )

    args = ap.parse_args()

    vm = load_module(args.val_module)

    print("===== LOAD GENERIC VAL =====")

    generic_rows = load_jsonl(
        args.generic_predictions
    )

    if len(generic_rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Generic rows={len(generic_rows)}"
        )

    generic = {}

    for r in generic_rows:
        rid = str(r["row_id"])

        if str(r["split"]).lower() != "val":
            raise RuntimeError(
                f"{rid}: non-Val Generic row"
            )

        if bool(r.get("used_adapter", False)):
            raise RuntimeError(
                f"{rid}: Generic used Adapter"
            )

        if bool(r.get("used_h5000", False)):
            raise RuntimeError(
                f"{rid}: Generic already used H5000"
            )

        generic[rid] = r

    print("GENERIC_VAL_GATE=PASS")

    print()
    print("===== LOAD H5000 BOUNDARIES =====")

    hrows = load_jsonl(args.h5000)

    h5000 = {
        str(r["row_id"]): r
        for r in hrows
    }

    if len(h5000) != EXPECTED_ROWS:
        raise RuntimeError(
            f"H5000 rows={len(h5000)}"
        )

    if set(generic) != set(h5000):
        raise RuntimeError(
            "Generic/H5000 row-id mismatch"
        )

    print("H5000_VAL_GATE=PASS")

    print()
    print("===== LOAD RAW HISTORY =====")

    work_order = vm.load_work_order(
        args.work_split
    )

    raw_by_author = vm.load_raw_history(
        args.raw_history,
        work_order,
    )

    print("RAW_HISTORY_READY=PASS")

    overall = new_stats()
    by_author = defaultdict(new_stats)
    by_M = defaultdict(new_stats)
    by_typing = defaultdict(new_stats)

    generic_hits = 0
    fused_pool_hits = 0
    freq_recovered = 0

    total_injected = 0
    queries_with_injection = 0
    compatible_history_counts = []

    audit_rows = []

    for number, rid in enumerate(
        sorted(generic),
        start=1,
    ):
        g = generic[rid]
        h = h5000[rid]

        author = str(g["author_name"])
        gold = str(g["gold"])

        generic_top10 = [
            str(x)
            for x in g["top10_candidates"][:10]
        ]

        query_parts = [
            str(x)
            for x in g["segmented_pinyin"]
        ]

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

        compatible_rows = 0

        for event in history:
            full_parts = event_segments(event)

            if not compatible(
                query_parts,
                full_parts,
            ):
                continue

            compatible_rows += 1

            target = event_target(event)
            counts[target] += 1

        compatible_history_counts.append(
            compatible_rows
        )

        # Deterministic frequency ranking:
        # frequency descending, then text.
        freq_ranked = sorted(
            counts,
            key=lambda text: (
                -counts[text],
                text,
            ),
        )

        # Only frequency candidates absent from
        # Generic consume the fixed budget.
        freq_only = [
            text
            for text in freq_ranked
            if text not in generic_top10
        ]

        injected = freq_only[:FREQ_BUDGET]

        if injected:
            queries_with_injection += 1

        total_injected += len(injected)

        pool = list(generic_top10)

        for text in injected:
            if text not in pool:
                pool.append(text)

        if len(pool) > 15:
            raise RuntimeError(
                f"{rid}: pool={len(pool)}"
            )

        generic_rank = {
            text: rank
            for rank, text in enumerate(
                generic_top10,
                start=1,
            )
        }

        freq_rank = {
            text: rank
            for rank, text in enumerate(
                freq_ranked,
                start=1,
            )
        }

        # Simple untrained reciprocal-rank fusion.
        def score(text):
            s = 0.0

            if text in generic_rank:
                s += (
                    1.0
                    / generic_rank[text]
                )

            if text in freq_rank:
                s += (
                    1.0
                    / freq_rank[text]
                )

            return s

        # Tie-breaking remains deterministic.
        ranking = sorted(
            pool,
            key=lambda text: (
                -score(text),
                generic_rank.get(
                    text,
                    10**9,
                ),
                -counts.get(text, 0),
                text,
            ),
        )

        generic_hit = (
            gold in generic_top10
        )

        fused_hit = (
            gold in pool
        )

        generic_hits += int(
            generic_hit
        )

        fused_pool_hits += int(
            fused_hit
        )

        freq_recovered += int(
            fused_hit
            and not generic_hit
        )

        update(
            overall,
            ranking,
            gold,
        )

        update(
            by_author[author],
            ranking,
            gold,
        )

        update(
            by_M[int(g["M"])],
            ranking,
            gold,
        )

        update(
            by_typing[
                str(
                    g[
                        "effective_typing_mode"
                    ]
                )
            ],
            ranking,
            gold,
        )

        # Small audit sample only.
        if len(audit_rows) < 30:
            audit_rows.append({
                "row_id": rid,
                "author_name": author,
                "gold": gold,
                "generic_top10":
                    generic_top10,
                "frequency_top5":
                    freq_ranked[:5],
                "frequency_counts": {
                    x: counts[x]
                    for x in freq_ranked[:5]
                },
                "injected":
                    injected,
                "final_top10":
                    ranking[:10],
                "generic_gold_hit":
                    generic_hit,
                "fused_pool_gold_hit":
                    fused_hit,
                "compatible_history_rows":
                    compatible_rows,
            })

        if (
            number % 1000 == 0
            or number == EXPECTED_ROWS
        ):
            print(
                "GENERIC_FREQUENCY",
                number,
                "/20000",
                flush=True,
            )

    if overall["queries"] != EXPECTED_ROWS:
        raise RuntimeError(
            "Query count changed"
        )

    result = {
        "schema_version": 1,
        "experiment":
            "final_fiveauthor_generic_frequency_val_v1",

        "population": "Val",

        "baseline_definition": {
            "neural_source":
                "Generic PinyinGPT Top10",
            "history_window":
                "latest 5000 strictly prior raw same-author interactions",
            "history_filter":
                "same Full/Initial/Mixed Pinyin compatibility rule",
            "frequency_score":
                "raw occurrence count only",
            "context_matching":
                False,
            "recency_weighting":
                False,
            "composition":
                False,
            "rich30":
                False,
            "trained_reranker":
                False,
            "frequency_candidate_budget":
                FREQ_BUDGET,
            "max_pool_size":
                15,
            "ranking":
                "untrained reciprocal-rank fusion of Generic rank and frequency rank",
        },

        "queries":
            EXPECTED_ROWS,

        "generic_top10_gold_queries":
            generic_hits,

        "generic_top10_hit_rate":
            generic_hits
            / EXPECTED_ROWS,

        "fused_pool_gold_queries":
            fused_pool_hits,

        "fused_pool_hit_rate":
            fused_pool_hits
            / EXPECTED_ROWS,

        "frequency_recovered_queries":
            freq_recovered,

        "frequency_recovery_gain":
            freq_recovered
            / EXPECTED_ROWS,

        "queries_with_frequency_injection":
            queries_with_injection,

        "mean_injected_candidates":
            total_injected
            / EXPECTED_ROWS,

        "mean_compatible_history_rows":
            sum(
                compatible_history_counts
            )
            / EXPECTED_ROWS,

        "overall":
            finish(overall),

        "by_author": {
            k: finish(v)
            for k, v
            in sorted(
                by_author.items()
            )
        },

        "by_M": {
            str(k): finish(v)
            for k, v
            in sorted(
                by_M.items()
            )
        },

        "by_effective_typing_mode": {
            k: finish(v)
            for k, v
            in sorted(
                by_typing.items()
            )
        },

        "audit_sample":
            audit_rows,

        "used_test":
            False,

        "test_rows_used_as_queries":
            0,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "========================================"
    )
    print(
        "GENERIC + FREQUENCY VAL RESULT"
    )
    print(
        "========================================"
    )

    print(
        json.dumps(
            result["overall"],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        "GENERIC_TOP10_HIT_RATE =",
        result[
            "generic_top10_hit_rate"
        ],
    )

    print(
        "FUSED_POOL_HIT_RATE =",
        result[
            "fused_pool_hit_rate"
        ],
    )

    print(
        "FREQUENCY_RECOVERED_QUERIES =",
        result[
            "frequency_recovered_queries"
        ],
    )

    print(
        "FREQUENCY_RECOVERY_GAIN =",
        result[
            "frequency_recovery_gain"
        ],
    )

    print(
        "TRAINED_RERANKER = 0"
    )

    print(
        "TEST_ROWS_USED_AS_QUERIES = 0"
    )

    print(
        "GENERIC_FREQUENCY_VAL_GATE=PASS"
    )

    print(
        "OUTPUT =",
        args.output,
    )


if __name__ == "__main__":
    main()
