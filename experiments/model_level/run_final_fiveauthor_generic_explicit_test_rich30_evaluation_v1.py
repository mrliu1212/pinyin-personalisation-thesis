from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np


AUTHORS = [
    "Re_spectators",
    "Etinjat",
    "Agent Phage",
    "QBLevi",
    "breaddddd",
]

EXPECTED_ROWS = 40_000
PER_AUTHOR = 8_000


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import module: {path}"
        )

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_generic_test(path: Path):
    rows = list(load_jsonl(path))

    if len(rows) != 40000:
        raise RuntimeError(
            f"Generic Test rows={len(rows)} != 40000"
        )

    out = {}

    for r in rows:
        rid = str(r["row_id"])

        if rid in out:
            raise RuntimeError(
                f"Duplicate Generic Test row_id: {rid}"
            )

        if str(r["split"]).lower() != "test":
            raise RuntimeError(
                f"{rid}: non-Test row in Generic predictions"
            )

        if bool(r.get("used_adapter", False)):
            raise RuntimeError(
                f"{rid}: Generic prediction used adapter"
            )

        if bool(r.get("used_h5000", False)):
            raise RuntimeError(
                f"{rid}: Generic prediction used H5000"
            )

        if bool(r.get("used_rich30", False)):
            raise RuntimeError(
                f"{rid}: Generic prediction used Rich30"
            )

        candidates = list(
            map(str, r["top10_candidates"])
        )[:10]

        scores = list(
            map(float, r["top10_candidate_scores"])
        )[:10]

        if len(candidates) != len(scores):
            raise RuntimeError(
                f"{rid}: Generic candidate/score length mismatch"
            )

        out[rid] = r

    return out


def load_test_meta(path: Path):
    out = {}

    for r in load_jsonl(path):
        if str(r["split"]).lower() != "test":
            continue

        rid = str(r["row_id"])

        if rid in out:
            raise RuntimeError(
                f"Duplicate Test metadata: {rid}"
            )

        out[rid] = r

    if len(out) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Test metadata={len(out)}"
        )

    return out


def history_features_for_test(
    vm,
    h5000_row,
    full_meta,
    pool,
    raw_by_author,
):
    rid = str(h5000_row["row_id"])
    meta = full_meta[rid]

    author = str(meta["author_name"])

    if str(
        h5000_row["author_name"]
    ) != author:
        raise RuntimeError(
            f"{rid}: H5000 author mismatch"
        )

    if str(
        h5000_row["split"]
    ).lower() != "test":
        raise RuntimeError(
            f"{rid}: non-Test H5000 row"
        )

    query_parts = tuple(
        str(
            meta["pinyin_input"]
        ).split()
    )

    query_context = str(
        meta.get("context")
        or ""
    )

    start = int(
        h5000_row[
            "history_start_ordinal"
        ]
    )

    stop = int(
        h5000_row[
            "history_stop_ordinal"
        ]
    )

    raw = raw_by_author[author]

    if not (
        0 <= start <= stop <= len(raw)
    ):
        raise RuntimeError(
            f"{rid}: invalid H5000 window"
        )

    visible_raw = raw[start:stop]
    raw_count = stop - start

    if (
        int(
            h5000_row[
                "history_raw_count"
            ]
        )
        != raw_count
    ):
        raise RuntimeError(
            f"{rid}: raw count mismatch"
        )

    raw_with_age = [
        (
            event,
            stop
            - 1
            - (
                start
                + local_ordinal
            ),
        )
        for local_ordinal, event
        in enumerate(visible_raw)
    ]

    visible = [
        (
            event,
            age,
        )
        for event, age
        in raw_with_age
        if vm.pinyin_sequence_compatible(
            query_parts,
            event["pinyin"],
        )
    ]

    counts = Counter(
        event["target"]
        for event, _age in visible
    )

    same_count = len(visible)

    entropy = (
        vm.entropy_concentration(
            counts
        )
    )

    candidate_set = set(pool)

    candidate_history = [
        (
            event,
            age,
        )
        for event, age in visible
        if event["target"]
        in candidate_set
    ]

    effective_n = 0
    matched = candidate_history

    for order in range(
        min(
            vm.NGRAM_MAX_N,
            len(query_context),
        ),
        0,
        -1,
    ):
        current = [
            (
                event,
                age,
            )
            for event, age
            in candidate_history
            if vm.suffix_match(
                query_context,
                event["context"],
                order,
            )
        ]

        if current:
            effective_n = order
            matched = current
            break

    raw_scores = {
        candidate: 0.0
        for candidate in pool
    }

    for event, age in matched:
        raw_scores[
            event["target"]
        ] += vm.recency_weight(age)

    values = vm.normalize(
        [
            raw_scores[candidate]
            for candidate in pool
        ],
        uniform_if_zero=True,
    )

    ngram = dict(
        zip(
            pool,
            values,
        )
    )

    top_ngram = max(
        ngram.values(),
        default=0.0,
    )

    total = sum(
        counts.values()
    )

    result = {}

    for text in pool:
        freq = int(
            counts.get(text, 0)
        )

        choice = (
            freq / total
            if total
            else 0.0
        )

        ng = float(
            ngram.get(
                text,
                0.0,
            )
        )

        result[text] = {
            "history_frequency_count":
                float(freq),

            "history_choice_share":
                float(choice),

            "history_entropy_concentration":
                float(entropy),

            "history_log1p_same_pinyin_count":
                math.log1p(
                    same_count
                ),

            "history_log1p_raw_count":
                math.log1p(
                    raw_count
                ),

            "history_ngram_support":
                ng,

            "history_ngram_effective_n":
                float(effective_n),

            "history_log1p_ngram_matched_count":
                math.log1p(
                    len(matched)
                ),

            "history_ngram_gap_to_top":
                float(
                    top_ngram - ng
                ),
        }

    return result


def new_stats():
    return {
        "queries": 0,
        "top1": 0,
        "top3": 0,
        "top5": 0,
        "top10": 0,
        "pool_hit": 0,
        "missing": 0,
        "mrr_sum": 0.0,
        "mrr5_sum": 0.0,
    }


def add_result(stats, rank, gold_in_pool):
    stats["queries"] += 1
    stats["pool_hit"] += int(gold_in_pool)
    stats["missing"] += int(not gold_in_pool)

    if rank is None:
        return

    stats["top1"] += int(rank <= 1)
    stats["top3"] += int(rank <= 3)
    stats["top5"] += int(rank <= 5)
    stats["top10"] += int(rank <= 10)

    stats["mrr_sum"] += (
        1.0 / rank
    )

    if rank <= 5:
        stats["mrr5_sum"] += (
            1.0 / rank
        )


def finish_stats(x):
    n = x["queries"]

    if not n:
        return {
            **x,
            "top1_rate": 0.0,
            "top3_rate": 0.0,
            "top5_rate": 0.0,
            "top10_rate": 0.0,
            "pool_hit_rate": 0.0,
            "missing_rate": 0.0,
            "mrr": 0.0,
            "mrr_at_5": 0.0,
        }

    return {
        **x,
        "top1_rate":
            x["top1"] / n,
        "top3_rate":
            x["top3"] / n,
        "top5_rate":
            x["top5"] / n,
        "top10_rate":
            x["top10"] / n,
        "pool_hit_rate":
            x["pool_hit"] / n,
        "missing_rate":
            x["missing"] / n,
        "mrr":
            x["mrr_sum"] / n,
        "mrr_at_5":
            x["mrr5_sum"] / n,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--val-module",
        type=Path,
        required=True,
    )

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
        "--output-root",
        type=Path,
        required=True,
    )

    args = ap.parse_args()

    vm = load_module(
        args.val_module,
        "frozen_val_rich30_v1",
    )

    if len(vm.FEATURES) != 30:
        raise RuntimeError(
            "Frozen Rich30 feature count changed"
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    adapter = load_generic_test(
        args.generic_predictions
    )

    full_meta = load_test_meta(
        args.full_manifest
    )

    if set(adapter) != set(full_meta):
        raise RuntimeError(
            "Generic/Test metadata universe mismatch"
        )

    print(
        "TEST_GENERIC_ROWS = 40000",
        flush=True,
    )

    print(
        "===== LOAD RAW HISTORY =====",
        flush=True,
    )

    work_order = vm.load_work_order(
        args.work_split
    )

    raw_by_author = (
        vm.load_raw_history(
            args.raw_history,
            work_order,
        )
    )

    print(
        "RAW_HISTORY_READY=PASS",
        flush=True,
    )

    if not args.model.is_file():
        raise FileNotFoundError(
            args.model
        )

    model = lgb.Booster(
        model_file=str(args.model)
    )

    model_features = list(
        model.feature_name()
    )

    if model_features != list(
        vm.FEATURES
    ):
        raise RuntimeError(
            "Frozen LambdaMART feature order mismatch"
        )

    print(
        "MODEL_LOADED_FROM_VAL=PASS",
        flush=True,
    )

    print(
        "LAMBDAMART_TRAINING_CALLS=0",
        flush=True,
    )

    surface_path = (
        args.output_root
        / "final_fiveauthor_generic_explicit_test_rich30_surface_v1.jsonl"
    )

    overall = new_stats()
    by_author = defaultdict(new_stats)
    by_M = defaultdict(new_stats)
    by_typing = defaultdict(new_stats)
    by_effective_typing = defaultdict(new_stats)

    seen = set()
    test_history_queries = 0

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
                    f"Duplicate H5000 Test row: {rid}"
                )

            seen.add(rid)

            if rid not in adapter:
                raise RuntimeError(
                    f"H5000 row not in Generic: {rid}"
                )

            if str(
                h["split"]
            ).lower() != "test":
                raise RuntimeError(
                    "Non-Test H5000 row"
                )

            if h.get(
                "used_current_test"
            ):
                raise RuntimeError(
                    "Current-Test leakage"
                )

            if h.get(
                "used_future_test"
            ):
                raise RuntimeError(
                    "Future-Test leakage"
                )

            if int(
                h.get(
                    "visible_prior_test_raw",
                    0,
                )
            ) > 0:
                test_history_queries += 1

            a = adapter[rid]

            personal = (
                vm.merge_personal(h)
            )

            adapter_texts = list(
                map(
                    str,
                    a[
                        "top10_candidates"
                    ],
                )
            )[:10]

            adapter_scores = list(
                map(
                    float,
                    a[
                        "top10_candidate_scores"
                    ],
                )
            )[:10]

            adapter_map = {
                text: {
                    "rank": rank,
                    "score": score,
                }
                for rank, (
                    text,
                    score,
                )
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

            personal_only = [
                x for x in personal
                if x["text"]
                not in adapter_map
            ]

            selected = (
                personal_only[
                    :vm.PERSONAL_BUDGET
                ]
            )

            pool = list(
                adapter_texts
            )

            for x in selected:
                if x["text"] not in pool:
                    pool.append(
                        x["text"]
                    )

            if len(pool) > 15:
                raise RuntimeError(
                    f"{rid}: pool={len(pool)}"
                )

            top_score = (
                adapter_scores[0]
                if adapter_scores
                else 0.0
            )

            pool_size = len(pool)

            history_ev = (
                history_features_for_test(
                    vm,
                    h,
                    full_meta,
                    pool,
                    raw_by_author,
                )
            )

            features = []

            for text in pool:
                ad = adapter_map.get(
                    text
                )

                pe = personal_map.get(
                    text
                )

                has_adapter = (
                    ad is not None
                )

                is_personal = (
                    pe is not None
                )

                if has_adapter:
                    arank = int(
                        ad["rank"]
                    )
                    ascore = float(
                        ad["score"]
                    )
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
                    vm.safe(
                        ev[name]
                    )
                    for name
                    in vm.RICH_FEATURES
                ]

                feat = (
                    base
                    + rich
                )

                if len(feat) != 30:
                    raise RuntimeError(
                        "Feature width changed"
                    )

                features.append(
                    feat
                )

            X = np.asarray(
                features,
                dtype=np.float32,
            )

            if not np.isfinite(
                X
            ).all():
                raise RuntimeError(
                    f"{rid}: non-finite features"
                )

            predictions = list(
                map(
                    float,
                    model.predict(X),
                )
            )

            # Stable descending sort. Equal scores retain
            # frozen pool order.
            order = sorted(
                range(len(pool)),
                key=lambda i:
                    -predictions[i],
            )

            final_ranking = [
                pool[i]
                for i in order
            ]

            final_scores = [
                predictions[i]
                for i in order
            ]

            gold = str(
                a["gold"]
            )

            gold_in_pool = (
                gold in pool
            )

            rank = (
                final_ranking.index(
                    gold
                ) + 1
                if gold in final_ranking
                else None
            )

            add_result(
                overall,
                rank,
                gold_in_pool,
            )

            add_result(
                by_author[
                    str(a["author_name"])
                ],
                rank,
                gold_in_pool,
            )

            add_result(
                by_M[
                    str(a["M"])
                ],
                rank,
                gold_in_pool,
            )

            add_result(
                by_typing[
                    str(a["typing_mode"])
                ],
                rank,
                gold_in_pool,
            )

            add_result(
                by_effective_typing[
                    str(
                        a[
                            "effective_typing_mode"
                        ]
                    )
                ],
                rank,
                gold_in_pool,
            )

            surface.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "experiment":
                            "final_fiveauthor_generic_explicit_test_rich30_evaluation_v1",

                        "row_id": rid,
                        "author_name":
                            a["author_name"],
                        "split": "test",
                        "gold": gold,
                        "M": int(a["M"]),
                        "typing_mode":
                            a["typing_mode"],
                        "effective_typing_mode":
                            a[
                                "effective_typing_mode"
                            ],

                        "generic_top10":
                            adapter_texts,

                        "personal_recovery_top5":
                            selected,

                        "final_pool":
                            pool,

                        "pool_size":
                            pool_size,

                        "gold_in_pool":
                            gold_in_pool,

                        "final_ranking":
                            final_ranking,

                        "final_scores":
                            final_scores,

                        "gold_final_rank":
                            rank,

                        "used_test":
                            True,

                        "used_current_test":
                            False,

                        "used_future_test":
                            False,

                        "model_updated":
                            False,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

            if (
                number % 250 == 0
                or number == EXPECTED_ROWS
            ):
                print(
                    "TEST_RICH30",
                    number,
                    "/40000",
                    flush=True,
                )

    if len(seen) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Test H5000 rows="
            f"{len(seen)}"
        )

    if seen != set(adapter):
        raise RuntimeError(
            "Generic/Test row-id universe mismatch"
        )

    summary = {
        "schema_version": 1,
        "experiment":
            "final_fiveauthor_generic_explicit_test_rich30_evaluation_v1",

        "test_queries":
            EXPECTED_ROWS,

        "feature_count":
            30,

        "features":
            list(vm.FEATURES),

        "model_path":
            str(args.model),

        "model_sha256":
            vm.sha256_file(
                args.model
            ),

        "lambdaMART_training_on_test":
            False,

        "neural_parameter_updates_on_test":
            False,

        "queries_with_visible_strictly_prior_test_history":
            test_history_queries,

        "current_test_visible":
            0,

        "future_test_visible":
            0,

        "overall":
            finish_stats(
                overall
            ),

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
                key=lambda x:
                    int(x[0]),
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
                by_effective_typing.items()
            )
        },

        "surface_path":
            str(surface_path),

        "surface_sha256":
            vm.sha256_file(
                surface_path
            ),
    }

    summary_path = (
        args.output_root
        / "final_fiveauthor_generic_explicit_test_rich30_summary_v1.json"
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
        "TEST_QUERIES = 40000",
        flush=True,
    )

    print(
        "FEATURE_COUNT = 30",
        flush=True,
    )

    print(
        "LAMBDAMART_TRAINING_ON_TEST = 0",
        flush=True,
    )

    print(
        "MODEL_LOADED_FROM_VAL = PASS",
        flush=True,
    )

    print(
        "CURRENT_TEST_VISIBLE = 0",
        flush=True,
    )

    print(
        "FUTURE_TEST_VISIBLE = 0",
        flush=True,
    )

    print(
        "FINAL_PERSONALISED_TEST_GATE=PASS",
        flush=True,
    )

    print(
        "SUMMARY =",
        summary_path,
        flush=True,
    )


if __name__ == "__main__":
    main()
