from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

import lightgbm as lgb
import numpy as np

import audit_m2_runtime_candidate_lattice_v1 as m2rt
import run_multi_m1c_pv_expansion_frozen_v2 as m1c
import run_m4_r1_bounded_composition_beam16_v1 as r1


R1_EXPECTED_SHA = (
    "65793fce44cd8765e936bddcd24c0cdaa"
    "257bec0cf65eaebc87b0a417ab869b7"
)

FEATURE_NAMES = [
    "weighted_log_frequency",
    "weighted_choice_share",
    "weighted_entropy_concentration",
    "piece_count",
    "fragmentation_ratio",
    "longest_span_ratio",
    "average_piece_length",
    "log1p_min_piece_frequency",
    "log1p_max_piece_frequency",
    "min_piece_choice_share",
    "max_piece_choice_share",
    "min_piece_entropy",
    "max_piece_entropy",
    "log1p_number_of_derivations",
    "c0_score",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def features(
    candidate: Mapping[str, Any],
) -> list[float]:
    return [
        float(candidate["weighted_log_frequency"]),
        float(candidate["weighted_choice_share"]),
        float(candidate["weighted_entropy_concentration"]),
        float(candidate["piece_count"]),
        float(candidate["fragmentation_ratio"]),
        float(candidate["longest_span_ratio"]),
        float(candidate["average_piece_length"]),
        math.log1p(
            float(candidate["min_piece_frequency"])
        ),
        math.log1p(
            float(candidate["max_piece_frequency"])
        ),
        float(candidate["min_piece_choice_share"]),
        float(candidate["max_piece_choice_share"]),
        float(candidate["min_piece_entropy"]),
        float(candidate["max_piece_entropy"]),
        math.log1p(
            float(candidate["number_of_derivations"])
        ),
        float(candidate["c0_score"]),
    ]


def make_query(
    *,
    multi: Mapping[str, Any],
    anchor: Mapping[str, Any],
) -> tuple[str, int, tuple[str, ...], str]:
    qrow = m1c.make_query_row(
        multi,
        anchor,
    )

    author = str(qrow["author"])

    position = int(
        qrow["chronological_position"]
    )

    pinyin = tuple(
        str(x)
        for x in (
            multi.get("pinyin_segments")
            or str(
                multi["pinyin_input"]
            ).split()
        )
    )

    gold = str(multi["gold"])

    if len(gold) != len(pinyin):
        raise RuntimeError(
            "Gold/Pinyin alignment mismatch: "
            f"{multi['row_id']}"
        )

    return (
        author,
        position,
        pinyin,
        gold,
    )


def generate(
    *,
    lookup: m2rt.RuntimeExpandedLookup,
    history: Any,
    multi: Mapping[str, Any],
    anchor: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
]:
    (
        author,
        position,
        pinyin,
        gold,
    ) = make_query(
        multi=multi,
        anchor=anchor,
    )

    start_raw, stop_raw = lookup.raw_window(
        author=author,
        position=position,
    )

    raw_count = stop_raw - start_raw

    expected_raw = history.raw_visible_count(
        author=author,
        position=position,
    )

    if raw_count != expected_raw:
        raise RuntimeError(
            "H5000 mismatch: "
            f"{multi['row_id']}"
        )

    edges, _ = r1.build_composition_edges(
        lookup=lookup,
        author=author,
        position=position,
        pinyin=pinyin,
    )

    candidates, _ = r1.bounded_decode(
        n=len(pinyin),
        edges=edges,
    )

    return candidates, edges, gold


def candidate_matrix(
    candidates: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    return np.asarray(
        [
            features(candidate)
            for candidate in candidates
        ],
        dtype=np.float32,
    )


def labels_for(
    candidates: Sequence[Mapping[str, Any]],
    gold: str,
) -> np.ndarray:
    return np.asarray(
        [
            1 if str(candidate["text"]) == gold else 0
            for candidate in candidates
        ],
        dtype=np.int8,
    )


def rank_from_labels(
    labels: np.ndarray,
) -> int | None:
    hits = np.flatnonzero(labels == 1)

    if len(hits) == 0:
        return None

    if len(hits) != 1:
        raise RuntimeError(
            "Expected at most one gold after same-text Viterbi"
        )

    return int(hits[0]) + 1


def lmart_rank(
    scores: np.ndarray,
    labels: np.ndarray,
) -> int | None:
    if len(scores) == 0:
        return None

    # Primary: descending LambdaMART.
    # Tie: frozen C0 order.
    original_order = np.arange(
        len(scores),
        dtype=np.int64,
    )

    order = np.lexsort(
        (
            original_order,
            -scores,
        )
    )

    ranked_labels = labels[order]

    hits = np.flatnonzero(
        ranked_labels == 1
    )

    if len(hits) == 0:
        return None

    return int(hits[0]) + 1


def quantile(
    values: Sequence[float],
    q: float,
) -> float | None:
    if not values:
        return None

    xs = sorted(float(x) for x in values)

    if len(xs) == 1:
        return xs[0]

    pos = q * (len(xs) - 1)

    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return xs[lo]

    w = pos - lo

    return (
        xs[lo] * (1.0 - w)
        + xs[hi] * w
    )


def stats(
    values: Sequence[int | float],
) -> dict[str, float | None]:
    xs = [float(x) for x in values]

    if not xs:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }

    return {
        "mean": statistics.fmean(xs),
        "median": quantile(xs, 0.5),
        "p95": quantile(xs, 0.95),
        "max": max(xs),
    }


def summarize(
    rows: Sequence[Mapping[str, Any]],
    rank_key: str,
) -> dict[str, Any]:
    n = len(rows)

    if n == 0:
        return {"n": 0}

    ranks = [
        row[rank_key]
        for row in rows
    ]

    out: dict[str, Any] = {
        "n": n,
        "candidate_survival": sum(
            rank is not None
            for rank in ranks
        ),
        "candidate_survival_rate": (
            sum(
                rank is not None
                for rank in ranks
            )
            / n
        ),
        "composable": sum(
            bool(row["composable"])
            for row in rows
        ),
        "composable_rate": (
            sum(
                bool(row["composable"])
                for row in rows
            )
            / n
        ),
    }

    for k in (1, 3, 5, 10):
        hits = sum(
            rank is not None
            and int(rank) <= k
            for rank in ranks
        )

        out[f"hits_at_{k}"] = hits
        out[f"recall_at_{k}"] = (
            hits / n
        )

    out["mrr"] = (
        sum(
            (
                1.0 / int(rank)
                if rank is not None
                else 0.0
            )
            for rank in ranks
        )
        / n
    )

    composable = int(
        out["composable"]
    )

    if composable:
        out[
            "recall_at_5_given_composable"
        ] = (
            sum(
                bool(row["composable"])
                and row[rank_key]
                is not None
                and int(
                    row[rank_key]
                ) <= 5
                for row in rows
            )
            / composable
        )
    else:
        out[
            "recall_at_5_given_composable"
        ] = None

    return out


def comparison(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    rescue = 0
    harm = 0
    unchanged_correct = 0
    unchanged_wrong = 0

    for row in rows:
        c0 = (
            row["c0_rank"] == 1
        )

        c1 = (
            row["c1_rank"] == 1
        )

        if (not c0) and c1:
            rescue += 1
        elif c0 and (not c1):
            harm += 1
        elif c0 and c1:
            unchanged_correct += 1
        else:
            unchanged_wrong += 1

    return {
        "n": len(rows),
        "rescue": rescue,
        "harm": harm,
        "net": rescue - harm,
        "unchanged_correct": unchanged_correct,
        "unchanged_wrong": unchanged_wrong,
    }


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--fit",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--val",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--fit-multi",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--val-multi",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--m0-rows",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--r1-script",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=20000,
    )

    args = ap.parse_args()

    if sha256_file(
        args.r1_script
    ) != R1_EXPECTED_SHA:
        raise RuntimeError(
            "Frozen R1 SHA mismatch"
        )

    if (
        args.output_root.exists()
        and any(
            args.output_root.iterdir()
        )
    ):
        raise RuntimeError(
            "Refusing to overwrite non-empty output"
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    fit = read_jsonl(args.fit)
    val = read_jsonl(args.val)
    fit_multi = read_jsonl(
        args.fit_multi
    )
    val_multi = read_jsonl(
        args.val_multi
    )
    m0_rows = read_jsonl(
        args.m0_rows
    )

    fit_by_id = {
        str(row["row_id"]): row
        for row in fit
    }

    val_by_id = {
        str(row["row_id"]): row
        for row in val
    }

    m0_by_id = {
        str(row["row_id"]): row
        for row in m0_rows
    }

    # -------------------------------------------------
    # TRAIN-FIT
    # -------------------------------------------------

    print(
        "===== BUILD TRAIN-FIT COMPOSITION DATA =====",
        flush=True,
    )

    train_history = (
        m1c.ExpandedCausalHistoryIndex(
            fit,
            fit_multi,
        )
    )

    train_lookup = (
        m2rt.RuntimeExpandedLookup(
            train_history
        )
    )

    train_X_chunks = []
    train_y_chunks = []
    train_groups = []

    train_queries_seen = 0
    train_queries_nonempty = 0
    train_queries_gold_survived = 0

    train_started = time.perf_counter()

    for number, multi in enumerate(
        fit_multi,
        start=1,
    ):
        source_id = str(
            multi["source_standard_row_id"]
        )

        anchor = fit_by_id.get(
            source_id
        )

        if anchor is None:
            raise RuntimeError(
                f"Missing Fit anchor: {source_id}"
            )

        candidates, _, gold = generate(
            lookup=train_lookup,
            history=train_history,
            multi=multi,
            anchor=anchor,
        )

        train_queries_seen += 1

        if candidates:
            train_queries_nonempty += 1

        if not candidates:
            if (
                args.progress_every > 0
                and number
                % args.progress_every == 0
            ):
                print(
                    f"fit={number}/{len(fit_multi)}",
                    flush=True,
                )
            continue

        y = labels_for(
            candidates,
            gold,
        )

        # LambdaRank groups with no positive document
        # provide no ranking signal.
        if int(y.sum()) != 1:
            if (
                args.progress_every > 0
                and number
                % args.progress_every == 0
            ):
                print(
                    f"fit={number}/{len(fit_multi)}",
                    flush=True,
                )
            continue

        X = candidate_matrix(
            candidates
        )

        train_X_chunks.append(X)
        train_y_chunks.append(y)
        train_groups.append(
            len(candidates)
        )

        train_queries_gold_survived += 1

        if (
            args.progress_every > 0
            and number
            % args.progress_every == 0
        ):
            print(
                f"fit={number}/{len(fit_multi)} "
                f"positive_groups="
                f"{train_queries_gold_survived}",
                flush=True,
            )

    if not train_X_chunks:
        raise RuntimeError(
            "No positive Train-Fit groups"
        )

    X_train = np.concatenate(
        train_X_chunks,
        axis=0,
    )

    y_train = np.concatenate(
        train_y_chunks,
        axis=0,
    )

    group_train = np.asarray(
        train_groups,
        dtype=np.int32,
    )

    del train_X_chunks
    del train_y_chunks
    del train_lookup
    del train_history

    gc.collect()

    train_generation_seconds = (
        time.perf_counter()
        - train_started
    )

    print(
        "TRAIN_MATRIX",
        X_train.shape,
        "groups=",
        len(group_train),
        "positives=",
        int(y_train.sum()),
        flush=True,
    )

    # -------------------------------------------------
    # FIXED SMALL COMPOSITION LAMBDAMART
    # -------------------------------------------------

    print(
        "===== TRAIN COMPOSITION LAMBDAMART =====",
        flush=True,
    )

    dataset = lgb.Dataset(
        X_train,
        label=y_train,
        group=group_train,
        feature_name=FEATURE_NAMES,
        free_raw_data=False,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5, 10],
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 200,
        "lambda_l2": 1.0,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "max_bin": 255,
        "seed": 1729,
        "feature_fraction_seed": 1729,
        "bagging_seed": 1729,
        "data_random_seed": 1729,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }

    model_started = time.perf_counter()

    model = lgb.train(
        params,
        dataset,
        num_boost_round=100,
    )

    model_seconds = (
        time.perf_counter()
        - model_started
    )

    model_path = (
        args.output_root
        / "composition_lambdamart.txt"
    )

    model.save_model(
        str(model_path)
    )

    model_sha = sha256_file(
        model_path
    )

    print(
        "MODEL_SHA256=",
        model_sha,
        flush=True,
    )

    del dataset
    del X_train
    del y_train

    gc.collect()

    # -------------------------------------------------
    # TRAIN-VAL — HELD OUT FROM RANKER FIT
    # -------------------------------------------------

    print(
        "===== BUILD / EVALUATE TRAIN-VAL =====",
        flush=True,
    )

    eval_history = (
        m1c.ExpandedCausalHistoryIndex(
            [
                *fit,
                *val,
            ],
            [
                *fit_multi,
                *val_multi,
            ],
        )
    )

    eval_lookup = (
        m2rt.RuntimeExpandedLookup(
            eval_history
        )
    )

    rows = []

    eval_started = time.perf_counter()

    for number, multi in enumerate(
        val_multi,
        start=1,
    ):
        row_id = str(
            multi["row_id"]
        )

        source_id = str(
            multi["source_standard_row_id"]
        )

        anchor = val_by_id.get(
            source_id
        )

        if anchor is None:
            raise RuntimeError(
                f"Missing Val anchor: {source_id}"
            )

        m0 = m0_by_id.get(
            row_id
        )

        if m0 is None:
            raise RuntimeError(
                f"Missing M0 row: {row_id}"
            )

        candidates, edges, gold = generate(
            lookup=eval_lookup,
            history=eval_history,
            multi=multi,
            anchor=anchor,
        )

        labels = labels_for(
            candidates,
            gold,
        )

        c0_rank = rank_from_labels(
            labels
        )

        if candidates:
            X = candidate_matrix(
                candidates
            )

            scores = model.predict(
                X,
                num_iteration=model.best_iteration,
            )

            scores = np.asarray(
                scores,
                dtype=np.float64,
            )

            c1_rank = lmart_rank(
                scores,
                labels,
            )
        else:
            c1_rank = None

        (
            author,
            position,
            pinyin,
            _,
        ) = make_query(
            multi=multi,
            anchor=anchor,
        )

        composable = r1.gold_composable(
            n=len(pinyin),
            gold=gold,
            edges=edges,
        )

        generic_survived = (
            m2rt.m0_gold_survived(
                m0
            )
        )

        rows.append(
            {
                "row_id": row_id,
                "family_id": str(
                    multi["family_id"]
                ),
                "author": author,
                "multi_token_length": int(
                    multi[
                        "multi_token_length"
                    ]
                ),
                "candidate_count": len(
                    candidates
                ),
                "composable": bool(
                    composable
                ),
                "generic_beam_pruned": (
                    not bool(
                        generic_survived
                    )
                ),
                "c0_rank": c0_rank,
                "c1_rank": c1_rank,
            }
        )

        if (
            args.progress_every > 0
            and number
            % args.progress_every == 0
        ):
            print(
                f"val={number}/{len(val_multi)}",
                flush=True,
            )

    eval_seconds = (
        time.perf_counter()
        - eval_started
    )

    gp_rows = [
        row
        for row in rows
        if row["generic_beam_pruned"]
    ]

    summary = {
        "experiment": (
            "m4_r2_composition_lambdamart_v1"
        ),
        "runner_sha256": sha256_file(
            Path(__file__)
        ),
        "frozen_r1_sha256": (
            R1_EXPECTED_SHA
        ),
        "feature_names": FEATURE_NAMES,
        "training": {
            "split": "Train-Fit",
            "queries_seen": (
                train_queries_seen
            ),
            "queries_nonempty": (
                train_queries_nonempty
            ),
            "positive_ranking_groups": (
                train_queries_gold_survived
            ),
            "candidate_rows": int(
                sum(train_groups)
            ),
            "group_size": stats(
                train_groups
            ),
            "generation_seconds": (
                train_generation_seconds
            ),
            "model_seconds": (
                model_seconds
            ),
            "parameters": params,
            "num_boost_round": 100,
        },
        "evaluation": {
            "split": "Train-Val",
            "n": len(rows),
            "generic_pruned_n": len(
                gp_rows
            ),
            "generation_and_scoring_seconds": (
                eval_seconds
            ),
        },
        "model": {
            "path": str(model_path),
            "sha256": model_sha,
        },
        "overall": {
            "c0": summarize(
                rows,
                "c0_rank",
            ),
            "c1_lambdamart": summarize(
                rows,
                "c1_rank",
            ),
            "top1_transition": comparison(
                rows
            ),
        },
        "generic_pruned": {
            "c0": summarize(
                gp_rows,
                "c0_rank",
            ),
            "c1_lambdamart": summarize(
                gp_rows,
                "c1_rank",
            ),
            "top1_transition": comparison(
                gp_rows
            ),
        },
        "by_multi_token_length": {},
        "feature_importance_gain": dict(
            sorted(
                zip(
                    FEATURE_NAMES,
                    [
                        float(x)
                        for x in model.feature_importance(
                            importance_type="gain"
                        )
                    ],
                ),
                key=lambda item: -item[1],
            )
        ),
        "gold_used_for_candidate_generation": False,
        "gold_used_for_candidate_scoring": False,
        "gold_used_for_training_labels": True,
        "ranker_train_split": "Train-Fit",
        "ranker_eval_split": "Train-Val",
        "used_dev3000": False,
        "used_test": False,
    }

    for length in range(1, 6):
        subset = [
            row
            for row in rows
            if int(
                row[
                    "multi_token_length"
                ]
            ) == length
        ]

        gp_subset = [
            row
            for row in subset
            if row[
                "generic_beam_pruned"
            ]
        ]

        summary[
            "by_multi_token_length"
        ][str(length)] = {
            "overall": {
                "c0": summarize(
                    subset,
                    "c0_rank",
                ),
                "c1_lambdamart": summarize(
                    subset,
                    "c1_rank",
                ),
            },
            "generic_pruned": {
                "c0": summarize(
                    gp_subset,
                    "c0_rank",
                ),
                "c1_lambdamart": summarize(
                    gp_subset,
                    "c1_rank",
                ),
            },
        }

    write_json(
        args.output_root
        / "summary.json",
        summary,
    )

    # Compact evaluation rows only:
    # no giant candidate traces needed.
    with (
        args.output_root
        / "eval_rows.jsonl"
    ).open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    print()
    print(
        "===== M4-R2 RESULT ====="
    )

    for name, subset in (
        ("Overall", rows),
        ("Generic-pruned", gp_rows),
    ):
        c0 = summarize(
            subset,
            "c0_rank",
        )

        c1 = summarize(
            subset,
            "c1_rank",
        )

        trans = comparison(
            subset
        )

        print()
        print(name)

        print(
            "C0:",
            f"R1={c0['recall_at_1']:.6f}",
            f"R3={c0['recall_at_3']:.6f}",
            f"R5={c0['recall_at_5']:.6f}",
            f"R10={c0['recall_at_10']:.6f}",
            f"MRR={c0['mrr']:.6f}",
        )

        print(
            "C1:",
            f"R1={c1['recall_at_1']:.6f}",
            f"R3={c1['recall_at_3']:.6f}",
            f"R5={c1['recall_at_5']:.6f}",
            f"R10={c1['recall_at_10']:.6f}",
            f"MRR={c1['mrr']:.6f}",
        )

        print(
            "Top1 transition:",
            trans,
        )

    print()
    print("===== BY MULTI LENGTH =====")

    for length in range(1, 6):
        x = summary[
            "by_multi_token_length"
        ][str(length)]

        a = x["overall"]["c0"]
        b = x[
            "overall"
        ]["c1_lambdamart"]

        gp_a = x[
            "generic_pruned"
        ]["c0"]

        gp_b = x[
            "generic_pruned"
        ]["c1_lambdamart"]

        print(
            f"M{length}: "
            f"C0_R1={a['recall_at_1']:.6f} "
            f"C1_R1={b['recall_at_1']:.6f} "
            f"C0_R5={a['recall_at_5']:.6f} "
            f"C1_R5={b['recall_at_5']:.6f} "
            f"GP_C0_R5={gp_a.get('recall_at_5', 0):.6f} "
            f"GP_C1_R5={gp_b.get('recall_at_5', 0):.6f}"
        )

    print()
    print(
        "FEATURE_IMPORTANCE_GAIN="
    )

    for name, gain in summary[
        "feature_importance_gain"
    ].items():
        print(
            name,
            gain,
        )

    print()
    print(
        "DEV3000=CLOSED"
    )
    print(
        "TEST=CLOSED"
    )
    print(
        "M4_R2_COMPOSITION_LAMBDAMART=PASS"
    )


if __name__ == "__main__":
    main()
