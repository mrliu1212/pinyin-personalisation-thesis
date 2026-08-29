from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np


EXPECTED_ROWS = 20_000


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


def new_stats():
    return {
        "queries": 0,
        "top1": 0,
        "top3": 0,
        "top5": 0,
        "top10": 0,
        "missing": 0,
        "mrr_sum": 0.0,
    }


def update(stats, rank, gold_in_pool):
    stats["queries"] += 1

    if not gold_in_pool:
        stats["missing"] += 1
        return

    stats["top1"] += int(rank <= 1)
    stats["top3"] += int(rank <= 3)
    stats["top5"] += int(rank <= 5)
    stats["top10"] += int(rank <= 10)
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
        "--val-module",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--adapter-root",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--h5000",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--full-manifest",
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
        "--model",
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

    if len(vm.FEATURES) != 30:
        raise RuntimeError("Rich30 feature count changed")

    print("===== LOAD FROZEN VAL ARTIFACTS =====")

    adapter = vm.map_adapter_predictions(
        args.adapter_root
    )

    h5000_rows = vm.load_jsonl(
        args.h5000
    )

    h5000 = {
        str(r["row_id"]): r
        for r in h5000_rows
    }

    if len(h5000) != EXPECTED_ROWS:
        raise RuntimeError(
            f"H5000 rows={len(h5000)}"
        )

    full_meta = {}

    with args.full_manifest.open(
        encoding="utf-8"
    ) as f:
        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)

            if str(r["split"]).lower() != "val":
                continue

            full_meta[
                str(r["row_id"])
            ] = r

    if len(full_meta) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Val metadata={len(full_meta)}"
        )

    if (
        set(adapter) != set(h5000)
        or set(adapter) != set(full_meta)
    ):
        raise RuntimeError(
            "Val row-id universe mismatch"
        )

    print("VAL_ARTIFACT_UNIVERSE_GATE=PASS")

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

    print()
    print("===== LOAD FROZEN LAMBDAMART =====")

    model = lgb.Booster(
        model_file=str(args.model)
    )

    if list(model.feature_name()) != list(vm.FEATURES):
        raise RuntimeError(
            "LambdaMART feature order mismatch"
        )

    print(
        "MODEL_SHA256 =",
        vm.sha256_file(args.model),
    )
    print("MODEL_LOAD_GATE=PASS")
    print("LAMBDAMART_TRAINING_CALLS=0")

    overall = new_stats()
    by_author = defaultdict(new_stats)
    by_M = defaultdict(new_stats)
    by_typing = defaultdict(new_stats)
    by_effective_typing = defaultdict(new_stats)

    pool_size_counts = defaultdict(int)
    empty_pool_rows = []

    positive_pool_queries = 0

    for number, rid in enumerate(
        sorted(adapter),
        start=1,
    ):
        a = adapter[rid]
        h = h5000[rid]

        personal = vm.merge_personal(h)

        adapter_texts = list(
            map(
                str,
                a["top10_candidates"],
            )
        )[:10]

        adapter_scores = list(
            map(
                float,
                a["top10_candidate_scores"],
            )
        )[:10]

        adapter_map = {
            text: {
                "rank": rank,
                "score": score,
            }
            for rank, (text, score)
            in enumerate(
                zip(
                    adapter_texts,
                    adapter_scores,
                ),
                start=1,
            )
        }

        personal_map = {
            x["text"]: x
            for x in personal
        }

        # IMPORTANT:
        # overlap candidates do not consume
        # Personal5 injection budget.
        personal_only = [
            x for x in personal
            if x["text"] not in adapter_map
        ]

        selected = (
            personal_only[
                :vm.PERSONAL_BUDGET
            ]
        )

        pool = list(adapter_texts)

        for x in selected:
            if x["text"] not in pool:
                pool.append(x["text"])

        if len(pool) > 15:
            raise RuntimeError(
                f"{rid}: pool={len(pool)}"
            )

        pool_size = len(pool)
        pool_size_counts[pool_size] += 1

        gold = str(a["gold"])
        gold_in_pool = gold in pool

        positive_pool_queries += int(
            gold_in_pool
        )

        # Confirmed unsupported edge case:
        # an empty candidate pool is a missing query,
        # with rank=None and MRR contribution 0.
        if not pool:
            rank = None

            empty_pool_rows.append({
                "row_id": rid,
                "author_name":
                    str(a["author_name"]),
                "gold": gold,
            })

        else:
            top_score = (
                adapter_scores[0]
                if adapter_scores
                else 0.0
            )

            history_ev = (
                vm.history_features_for_query(
                    h,
                    full_meta,
                    pool,
                    raw_by_author,
                )
            )

            features = []

            for text in pool:
                ad = adapter_map.get(text)
                pe = personal_map.get(text)

                has_adapter = (
                    ad is not None
                )

                is_personal = (
                    pe is not None
                )

                if has_adapter:
                    arank = int(ad["rank"])
                    ascore = float(ad["score"])
                    arr = 1.0 / arank

                    amean = (
                        ascore
                        / max(
                            1,
                            len(text),
                        )
                    )

                    agap = (
                        top_score
                        - ascore
                    )
                else:
                    arank = 0
                    ascore = 0.0
                    arr = 0.0
                    amean = 0.0
                    agap = 0.0

                if is_personal:
                    prank = int(
                        pe[
                            "personal_rank_raw"
                        ]
                    )

                    has_exact = (
                        pe["p_exact"]
                        is not None
                    )

                    has_comp = (
                        pe["p_composition"]
                        is not None
                    )

                else:
                    prank = 0
                    has_exact = False
                    has_comp = False

                base = [
                    float(has_adapter),
                    float(arank),
                    float(arr),
                    float(ascore),
                    float(amean),
                    float(agap),
                    float(is_personal),
                    float(prank),
                    float(has_exact),

                    vm.safe(
                        pe["exact_raw_score"]
                        if pe
                        else None
                    ),

                    vm.safe(
                        pe["p_exact"]
                        if pe
                        else None
                    ),

                    float(has_comp),

                    vm.safe(
                        pe["composition_raw_score"]
                        if pe
                        else None
                    ),

                    vm.safe(
                        pe["p_composition"]
                        if pe
                        else None
                    ),

                    vm.safe(
                        pe["recovery_confidence"]
                        if pe
                        else None
                    ),

                    vm.safe(
                        pe["piece_count"]
                        if pe
                        else None
                    ),

                    vm.safe(
                        pe["fragmentation_ratio"]
                        if pe
                        else None
                    ),

                    vm.safe(
                        pe["longest_span_ratio"]
                        if pe
                        else None
                    ),

                    float(
                        has_adapter
                        and is_personal
                    ),

                    float(a["M"]),
                    float(pool_size),
                ]

                ev = history_ev[text]

                rich = [
                    vm.safe(ev[name])
                    for name
                    in vm.RICH_FEATURES
                ]

                feat = base + rich

                if len(feat) != 30:
                    raise RuntimeError(
                        f"{rid}: feature width "
                        f"{len(feat)}"
                    )

                features.append(feat)

            X = np.asarray(
                features,
                dtype=np.float32,
            )

            if (
                X.ndim != 2
                or X.shape[1] != 30
            ):
                raise RuntimeError(
                    f"{rid}: X shape={X.shape}"
                )

            if not np.isfinite(X).all():
                raise RuntimeError(
                    f"{rid}: non-finite matrix"
                )

            scores = np.asarray(
                model.predict(X),
                dtype=float,
            )

            # Stable tie handling:
            # equal scores preserve original pool order.
            order = sorted(
                range(len(pool)),
                key=lambda i:
                    -float(scores[i]),
            )

            ranking = [
                pool[i]
                for i in order
            ]

            rank = (
                ranking.index(gold) + 1
                if gold_in_pool
                else None
            )

        update(
            overall,
            rank,
            gold_in_pool,
        )

        update(
            by_author[
                str(a["author_name"])
            ],
            rank,
            gold_in_pool,
        )

        update(
            by_M[
                int(a["M"])
            ],
            rank,
            gold_in_pool,
        )

        update(
            by_typing[
                str(a["typing_mode"])
            ],
            rank,
            gold_in_pool,
        )

        update(
            by_effective_typing[
                str(
                    a["effective_typing_mode"]
                )
            ],
            rank,
            gold_in_pool,
        )

        if (
            number % 1000 == 0
            or number == EXPECTED_ROWS
        ):
            print(
                f"VAL_RERANK "
                f"{number}/20000",
                flush=True,
            )

    if overall["queries"] != EXPECTED_ROWS:
        raise RuntimeError(
            "Val query count changed"
        )

    result = {
        "schema_version": 1,

        "experiment":
            "final_fiveauthor_val_rich30_diagnostic_v1",

        "IMPORTANT":
            (
                "These are LambdaMART training-set "
                "diagnostics on Val, not held-out "
                "generalisation results."
            ),

        "val_queries":
            EXPECTED_ROWS,

        "feature_count":
            30,

        "model_path":
            str(args.model),

        "model_sha256":
            vm.sha256_file(
                args.model
            ),

        "lambdaMART_training_in_diagnostic":
            False,

        "fused_pool_gold_queries":
            positive_pool_queries,

        "fused_pool_hit_rate":
            positive_pool_queries
            / EXPECTED_ROWS,

        "pool_size_counts": {
            str(k): v
            for k, v
            in sorted(
                pool_size_counts.items()
            )
        },

        "empty_pool_count":
            len(empty_pool_rows),

        "empty_pool_rows":
            empty_pool_rows,

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

        "by_typing_mode": {
            k: finish(v)
            for k, v
            in sorted(
                by_typing.items()
            )
        },

        "by_effective_typing_mode": {
            k: finish(v)
            for k, v
            in sorted(
                by_effective_typing.items()
            )
        },
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
    print("====================================")
    print("FINAL VAL TRAINING-SET DIAGNOSTIC")
    print("====================================")

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
        "FUSED_POOL_HIT_RATE =",
        result[
            "fused_pool_hit_rate"
        ],
    )

    print(
        "EMPTY_POOL_COUNT =",
        result[
            "empty_pool_count"
        ],
    )

    print(
        "LAMBDAMART_TRAINING_IN_DIAGNOSTIC = 0"
    )

    print(
        "VAL_RICH30_DIAGNOSTIC_GATE=PASS"
    )

    print(
        "OUTPUT =",
        args.output,
    )


if __name__ == "__main__":
    main()
