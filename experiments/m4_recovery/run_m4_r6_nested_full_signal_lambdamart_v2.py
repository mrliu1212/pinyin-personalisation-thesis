from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize


N_FOLDS = 5
EXPECTED_QUERIES = 94590
EXPECTED_FAMILIES = 18918
EXPECTED_GP = 25443
EXPECTED_COMP = 525027
EXPECTED_EXACT = 3099

FEATURE_NAMES = [
    "is_generic",
    "generic_rank",
    "generic_reciprocal_rank",
    "generic_log_probability",
    "generic_mean_log_probability",
    "generic_gap_to_top",
    "has_full_score",
    "full_score",
    "full_rank",
    "full_reciprocal_rank",
    "full_gap_to_top",
    "full_source_personal",
    "is_personal_slot",
    "personal_rank",
    "has_exact",
    "exact_raw_score",
    "p_exact",
    "has_composition",
    "composition_raw_score",
    "p_composition",
    "recovery_confidence",
    "composition_piece_count",
    "composition_fragmentation_ratio",
    "composition_longest_span_ratio",
    "multi_token_length",
    "pool_size",
]


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with Path(path).open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sigmoid(z):
    z = np.clip(
        np.asarray(z, dtype=np.float64),
        -40.0,
        40.0,
    )
    return 1.0 / (
        1.0 + np.exp(-z)
    )


def fit_platt(scores, labels):
    x = np.asarray(
        scores,
        dtype=np.float64,
    )
    y = np.asarray(
        labels,
        dtype=np.float64,
    )

    if len(x) == 0:
        raise RuntimeError(
            "Empty calibration population"
        )

    positives = float(y.sum())

    if not (
        0 < positives < len(y)
    ):
        raise RuntimeError(
            "Calibration requires both classes"
        )

    mean = float(x.mean())
    std = float(x.std())

    if std < 1e-12:
        std = 1.0

    z = (x - mean) / std

    base = float(y.mean())

    intercept0 = math.log(
        base / (1.0 - base)
    )

    def objective(theta):
        slope = float(theta[0])
        intercept = float(theta[1])

        eta = (
            slope * z
            + intercept
        )

        p = sigmoid(eta)

        loss = float(
            np.logaddexp(
                0.0,
                eta,
            ).sum()
            - np.dot(
                y,
                eta,
            )
        )

        residual = p - y

        grad = np.asarray(
            [
                np.dot(
                    residual,
                    z,
                ),
                residual.sum(),
            ],
            dtype=np.float64,
        )

        return loss, grad

    kwargs = dict(
        fun=lambda theta:
            objective(theta)[0],
        x0=np.asarray(
            [1.0, intercept0],
            dtype=np.float64,
        ),
        jac=lambda theta:
            objective(theta)[1],
        bounds=[
            (1e-8, None),
            (None, None),
        ],
    )

    result = minimize(
        method="L-BFGS-B",
        options={
            "maxiter": 200,
            "ftol": 1e-12,
            "gtol": 1e-8,
        },
        **kwargs,
    )

    optimizer = "L-BFGS-B"

    if not result.success:
        result = minimize(
            method="SLSQP",
            options={
                "maxiter": 2000,
                "ftol": 1e-12,
            },
            **kwargs,
        )

        optimizer = "SLSQP_FALLBACK"

    if not result.success:
        raise RuntimeError(
            "Platt fit failed: "
            + str(result.message)
        )

    slope_std = float(
        result.x[0]
    )

    intercept_std = float(
        result.x[1]
    )

    a = slope_std / std

    b = (
        intercept_std
        - slope_std * mean / std
    )

    if not (
        math.isfinite(a)
        and math.isfinite(b)
        and a > 0
    ):
        raise RuntimeError(
            "Invalid Platt parameters"
        )

    return {
        "a": a,
        "b": b,
        "optimizer": optimizer,
        "n": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": float(
            y.mean()
        ),
    }


def apply_platt(raw, params):
    return float(
        sigmoid(
            np.asarray(
                [
                    float(params["a"])
                    * float(raw)
                    + float(params["b"])
                ]
            )
        )[0]
    )


def metric_template():
    return {
        "n": 0,
        "top1": 0,
        "top3": 0,
        "top5": 0,
        "top10": 0,
        "top15": 0,
        "rr": 0.0,
        "missing": 0,
    }


def add_metric(metric, rank):
    metric["n"] += 1

    if rank is None:
        metric["missing"] += 1
        return

    for k in (
        1,
        3,
        5,
        10,
        15,
    ):
        if rank <= k:
            metric[f"top{k}"] += 1

    metric["rr"] += (
        1.0 / rank
    )


def finalize(metric):
    n = metric["n"]

    if n == 0:
        return {
            "n": 0,
            "top1": None,
            "top3": None,
            "top5": None,
            "top10": None,
            "top15": None,
            "mrr": None,
            "missing": None,
        }

    return {
        "n": n,
        "top1": metric["top1"] / n,
        "top3": metric["top3"] / n,
        "top5": metric["top5"] / n,
        "top10": metric["top10"] / n,
        "top15": metric["top15"] / n,
        "mrr": metric["rr"] / n,
        "missing": metric["missing"] / n,
    }


def rank_from_scores(labels, scores):
    labels = np.asarray(
        labels,
        dtype=np.int8,
    )

    if not np.any(labels):
        return None

    order = np.argsort(
        -np.asarray(
            scores,
            dtype=np.float64,
        ),
        kind="stable",
    )

    for rank, index in enumerate(
        order,
        start=1,
    ):
        if labels[index]:
            return rank

    raise RuntimeError(
        "Gold label disappeared"
    )


def latency_summary(values):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(
            np.median(x)
        ),
        "p95": float(
            np.percentile(x, 95)
        ),
        "p99": float(
            np.percentile(x, 99)
        ),
        "max": float(
            x.max()
        ),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--m0",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--r4",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--exact",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--composition",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--r5",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--r5-freeze",
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
            "Refusing to overwrite R6-v2 output"
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    freeze = json.loads(
        args.r5_freeze.read_text(
            encoding="utf-8-sig"
        )
    )

    if freeze["status"] != "FROZEN":
        raise RuntimeError(
            "R5 is not frozen"
        )

    print(
        "===== M4-R6-v2 NESTED FULL-SIGNAL FINAL RERANKER ====="
    )

    print(
        "Outer evaluation: 5-fold family OOF"
    )

    print(
        "Calibration is re-fit inside each outer fold"
    )

    print(
        "Test-fold fusion must reproduce frozen R5 exactly"
    )

    print(
        "Frozen Full + Generic LM + M4 Recovery signals"
    )

    print(
        "No hyperparameter sweep"
    )

    print(
        "DEV3000=CLOSED TEST=CLOSED"
    )

    # --------------------------------------------------
    # Compact M0.
    # --------------------------------------------------

    m0 = {}

    for row in read_jsonl(
        args.m0
    ):
        row_id = str(
            row["row_id"]
        )

        generic = []

        top = row[
            "top10_distinct_candidates"
        ]

        top_logp = (
            float(
                top[0][
                    "log_probability"
                ]
            )
            if top
            else 0.0
        )

        for cand in top:
            generic.append(
                {
                    "text": str(
                        cand["text"]
                    ),
                    "rank": int(
                        cand["rank"]
                    ),
                    "logp": float(
                        cand[
                            "log_probability"
                        ]
                    ),
                    "mean_logp": float(
                        cand[
                            "mean_log_probability"
                        ]
                    ),
                    "gap": float(
                        cand[
                            "log_probability"
                        ]
                    ) - top_logp,
                }
            )

        m0[row_id] = {
            "author": str(
                row["author"]
            ),
            "family_id": str(
                row["family_id"]
            ),
            "gold": str(
                row["gold"]
            ),
            "length": int(
                row[
                    "multi_token_length"
                ]
            ),
            "generic": generic,
        }

    if len(m0) != EXPECTED_QUERIES:
        raise RuntimeError(
            "M0 row count mismatch"
        )

    # --------------------------------------------------
    # Compact Frozen Full.
    # --------------------------------------------------

    full = {}

    for row in read_jsonl(
        args.r4
    ):
        row_id = str(
            row["row_id"]
        )

        candidates = row[
            "zero_shot_candidates"
        ]

        by_text = {}

        top_score = (
            float(
                candidates[0][
                    "lambdamart_score"
                ]
            )
            if candidates
            else 0.0
        )

        ranked_texts = []

        for cand in candidates:
            text = str(
                cand["text"]
            )

            ranked_texts.append(
                text
            )

            score = float(
                cand[
                    "lambdamart_score"
                ]
            )

            by_text[text] = {
                "score": score,
                "rank": int(
                    cand["rank"]
                ),
                "gap": (
                    score
                    - top_score
                ),
                "source": str(
                    cand["source"]
                ),
            }

        full[row_id] = {
            "by_text": by_text,
            "ranked_texts": ranked_texts,
        }

    if set(full) != set(m0):
        raise RuntimeError(
            "M0/R4 row IDs differ"
        )

    # --------------------------------------------------
    # Compact Exact branch.
    # --------------------------------------------------

    exact = {}

    exact_scores = []
    exact_labels = []
    exact_folds = []

    for row in read_jsonl(
        args.exact
    ):
        row_id = str(
            row["row_id"]
        )

        gold = str(
            row["gold_posthoc"]
        )

        fold = int(
            row["calibration_fold"]
        )

        candidates = []

        for cand in row[
            "exact_candidates"
        ]:
            raw = float(
                cand[
                    "exact_raw_score"
                ]
            )

            text = str(
                cand["text"]
            )

            candidates.append(
                {
                    "text": text,
                    "raw": raw,
                }
            )

            exact_scores.append(
                raw
            )

            exact_labels.append(
                int(
                    text == gold
                )
            )

            exact_folds.append(
                fold
            )

        exact[row_id] = {
            "fold": fold,
            "candidates": candidates,
        }

    if len(exact_scores) != EXPECTED_EXACT:
        raise RuntimeError(
            "Exact candidate count mismatch"
        )

    # --------------------------------------------------
    # Compact Composition branch and define query order.
    # --------------------------------------------------

    composition = {}
    query_ids = []

    comp_scores = []
    comp_labels = []
    comp_folds = []

    family_ids = set()

    for row in read_jsonl(
        args.composition
    ):
        row_id = str(
            row["row_id"]
        )

        query_ids.append(
            row_id
        )

        gold = str(
            row["gold"]
        )

        fold = int(
            row[
                "calibration_fold"
            ]
        )

        family_ids.add(
            str(
                row["family_id"]
            )
        )

        candidates = []

        for cand in row[
            "candidates"
        ]:
            raw = float(
                cand[
                    "c1_raw_score"
                ]
            )

            text = str(
                cand["text"]
            )

            item = {
                "text": text,
                "raw": raw,
                "piece_count": int(
                    cand[
                        "piece_count"
                    ]
                ),
                "fragmentation": float(
                    cand[
                        "fragmentation_ratio"
                    ]
                ),
                "longest_span": float(
                    cand[
                        "longest_span_ratio"
                    ]
                ),
            }

            candidates.append(
                item
            )

            comp_scores.append(
                raw
            )

            comp_labels.append(
                int(
                    text == gold
                )
            )

            comp_folds.append(
                fold
            )

        composition[row_id] = {
            "fold": fold,
            "candidates": candidates,
        }

    if len(query_ids) != EXPECTED_QUERIES:
        raise RuntimeError(
            "Composition query count mismatch"
        )

    if len(family_ids) != EXPECTED_FAMILIES:
        raise RuntimeError(
            "Family count mismatch"
        )

    if len(comp_scores) != EXPECTED_COMP:
        raise RuntimeError(
            "Composition candidate count mismatch"
        )

    if (
        set(query_ids) != set(m0)
        or set(query_ids)
        != set(exact)
    ):
        raise RuntimeError(
            "Branch row IDs differ"
        )

    exact_scores = np.asarray(
        exact_scores,
        dtype=np.float64,
    )

    exact_labels = np.asarray(
        exact_labels,
        dtype=np.int8,
    )

    exact_folds = np.asarray(
        exact_folds,
        dtype=np.int8,
    )

    comp_scores = np.asarray(
        comp_scores,
        dtype=np.float64,
    )

    comp_labels = np.asarray(
        comp_labels,
        dtype=np.int8,
    )

    comp_folds = np.asarray(
        comp_folds,
        dtype=np.int8,
    )

    # --------------------------------------------------
    # Frozen R5 test-fold Top5 reference.
    # --------------------------------------------------

    frozen_r5 = {}

    for row in read_jsonl(
        args.r5
    ):
        row_id = str(
            row["row_id"]
        )

        frozen_r5[row_id] = {
            "fold": int(
                row[
                    "calibration_fold"
                ]
            ),
            "generic_pruned": bool(
                row[
                    "generic_beam_pruned"
                ]
            ),
            "top5": [
                str(x["text"])
                for x in row[
                    "personal_recovery_top5"
                ]
            ],
        }

    if set(frozen_r5) != set(
        query_ids
    ):
        raise RuntimeError(
            "R5 row IDs differ"
        )

    # --------------------------------------------------
    # Helper: build calibrated recovery pool.
    # --------------------------------------------------

    def build_pool(
        row_id,
        exact_platt,
        comp_platt,
    ):
        base = m0[row_id]

        generic = base[
            "generic"
        ]

        generic_set = {
            x["text"]
            for x in generic
        }

        exact_map = {}

        for cand in exact[
            row_id
        ]["candidates"]:
            p = apply_platt(
                cand["raw"],
                exact_platt,
            )

            text = cand[
                "text"
            ]

            current = exact_map.get(
                text
            )

            if (
                current is None
                or p > current["p"]
            ):
                exact_map[text] = {
                    "raw": cand[
                        "raw"
                    ],
                    "p": p,
                }

        comp_map = {}

        for cand in composition[
            row_id
        ]["candidates"]:
            p = apply_platt(
                cand["raw"],
                comp_platt,
            )

            text = cand[
                "text"
            ]

            current = comp_map.get(
                text
            )

            if (
                current is None
                or p > current["p"]
            ):
                comp_map[text] = {
                    "raw": cand[
                        "raw"
                    ],
                    "p": p,
                    "piece_count": cand[
                        "piece_count"
                    ],
                    "fragmentation": cand[
                        "fragmentation"
                    ],
                    "longest_span": cand[
                        "longest_span"
                    ],
                }

        merged = {}

        for text, item in exact_map.items():
            merged[text] = {
                "text": text,
                "exact": item,
                "comp": None,
            }

        for text, item in comp_map.items():
            if text not in merged:
                merged[text] = {
                    "text": text,
                    "exact": None,
                    "comp": item,
                }
            else:
                merged[text][
                    "comp"
                ] = item

        personal = []

        for text, item in merged.items():
            if text in generic_set:
                continue

            pe = (
                item["exact"]["p"]
                if item["exact"]
                is not None
                else 0.0
            )

            pc = (
                item["comp"]["p"]
                if item["comp"]
                is not None
                else 0.0
            )

            item[
                "confidence"
            ] = max(
                pe,
                pc,
            )

            personal.append(
                item
            )

        personal.sort(
            key=lambda x: (
                -float(
                    x[
                        "confidence"
                    ]
                ),
                x["text"],
            )
        )

        return (
            generic,
            personal[:5],
            exact_map,
            comp_map,
        )

    def feature_vector(
        *,
        row_id,
        text,
        generic_info,
        personal_rank,
        exact_info,
        comp_info,
        pool_size,
    ):
        base = m0[row_id]

        full_info = full[
            row_id
        ]["by_text"].get(
            text
        )

        is_generic = (
            generic_info
            is not None
        )

        has_full = (
            full_info
            is not None
        )

        has_exact = (
            exact_info
            is not None
        )

        has_comp = (
            comp_info
            is not None
        )

        pe = (
            float(
                exact_info["p"]
            )
            if has_exact
            else 0.0
        )

        pc = (
            float(
                comp_info["p"]
            )
            if has_comp
            else 0.0
        )

        return [
            float(is_generic),
            float(
                generic_info["rank"]
                if is_generic
                else 0
            ),
            (
                1.0
                / generic_info[
                    "rank"
                ]
                if is_generic
                else 0.0
            ),
            float(
                generic_info[
                    "logp"
                ]
                if is_generic
                else 0.0
            ),
            float(
                generic_info[
                    "mean_logp"
                ]
                if is_generic
                else 0.0
            ),
            float(
                generic_info[
                    "gap"
                ]
                if is_generic
                else 0.0
            ),
            float(has_full),
            float(
                full_info[
                    "score"
                ]
                if has_full
                else 0.0
            ),
            float(
                full_info[
                    "rank"
                ]
                if has_full
                else 0
            ),
            (
                1.0
                / full_info[
                    "rank"
                ]
                if has_full
                else 0.0
            ),
            float(
                full_info[
                    "gap"
                ]
                if has_full
                else 0.0
            ),
            float(
                has_full
                and full_info[
                    "source"
                ]
                == "personal_recovery"
            ),
            float(
                personal_rank
                is not None
            ),
            float(
                personal_rank
                if personal_rank
                is not None
                else 0
            ),
            float(has_exact),
            float(
                exact_info[
                    "raw"
                ]
                if has_exact
                else 0.0
            ),
            pe,
            float(has_comp),
            float(
                comp_info[
                    "raw"
                ]
                if has_comp
                else 0.0
            ),
            pc,
            max(
                pe,
                pc,
            ),
            float(
                comp_info[
                    "piece_count"
                ]
                if has_comp
                else 0
            ),
            float(
                comp_info[
                    "fragmentation"
                ]
                if has_comp
                else 0.0
            ),
            float(
                comp_info[
                    "longest_span"
                ]
                if has_comp
                else 0.0
            ),
            float(
                base["length"]
            ),
            float(pool_size),
        ]

    def build_matrix_for_ids(
        ids,
        exact_platt,
        comp_platt,
        verify_frozen=False,
    ):
        X = []
        y = []
        groups = []
        meta = []

        frozen_match = 0

        for row_id in ids:
            base = m0[
                row_id
            ]

            (
                generic,
                personal,
                exact_map,
                comp_map,
            ) = build_pool(
                row_id,
                exact_platt,
                comp_platt,
            )

            if verify_frozen:
                rebuilt = [
                    x["text"]
                    for x in personal
                ]

                expected = frozen_r5[
                    row_id
                ]["top5"]

                if rebuilt != expected:
                    raise RuntimeError(
                        "Held-out R5 Top5 "
                        "reproduction mismatch: "
                        f"{row_id}\n"
                        f"rebuilt={rebuilt}\n"
                        f"frozen={expected}"
                    )

                frozen_match += 1

            pool_size = (
                len(generic)
                + len(personal)
            )

            start = len(y)

            for g in generic:
                text = g[
                    "text"
                ]

                X.append(
                    feature_vector(
                        row_id=row_id,
                        text=text,
                        generic_info=g,
                        personal_rank=None,
                        exact_info=(
                            exact_map.get(
                                text
                            )
                        ),
                        comp_info=(
                            comp_map.get(
                                text
                            )
                        ),
                        pool_size=(
                            pool_size
                        ),
                    )
                )

                y.append(
                    int(
                        text
                        == base["gold"]
                    )
                )

            for rank, item in enumerate(
                personal,
                start=1,
            ):
                text = item[
                    "text"
                ]

                X.append(
                    feature_vector(
                        row_id=row_id,
                        text=text,
                        generic_info=None,
                        personal_rank=rank,
                        exact_info=(
                            item["exact"]
                        ),
                        comp_info=(
                            item["comp"]
                        ),
                        pool_size=(
                            pool_size
                        ),
                    )
                )

                y.append(
                    int(
                        text
                        == base["gold"]
                    )
                )

            end = len(y)

            groups.append(
                end - start
            )

            meta.append(
                {
                    "row_id": row_id,
                    "start": start,
                    "end": end,
                    "personal_texts": [
                        x["text"]
                        for x in personal
                    ],
                }
            )

        return (
            np.asarray(
                X,
                dtype=np.float32,
            ),
            np.asarray(
                y,
                dtype=np.int8,
            ),
            groups,
            meta,
            frozen_match,
        )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [
            1,
            3,
            5,
            10,
        ],
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 200,
        "lambda_l2": 1.0,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "seed": 1729,
        "feature_fraction_seed": 1729,
        "bagging_seed": 1729,
        "data_random_seed": 1729,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
        "num_threads": -1,
    }

    rounds = 100

    full_metric = (
        metric_template()
    )

    r6_metric = (
        metric_template()
    )

    gp_metric = (
        metric_template()
    )

    recovered_metric = (
        metric_template()
    )

    by_length = {
        i: metric_template()
        for i in range(1, 6)
    }

    by_author = defaultdict(
        metric_template
    )

    rescue = 0
    harm = 0
    unchanged_correct = 0
    unchanged_wrong = 0

    recovered_n = 0
    frozen_test_pool_matches = 0

    fold_results = []

    query_result_rows = []

    print()
    print(
        "===== NESTED 5-FOLD OOF ====="
    )

    for outer_fold in range(
        N_FOLDS
    ):
        train_ids = [
            row_id
            for row_id
            in query_ids
            if composition[
                row_id
            ]["fold"]
            != outer_fold
        ]

        test_ids = [
            row_id
            for row_id
            in query_ids
            if composition[
                row_id
            ]["fold"]
            == outer_fold
        ]

        exact_train_mask = (
            exact_folds
            != outer_fold
        )

        comp_train_mask = (
            comp_folds
            != outer_fold
        )

        exact_platt = fit_platt(
            exact_scores[
                exact_train_mask
            ],
            exact_labels[
                exact_train_mask
            ],
        )

        comp_platt = fit_platt(
            comp_scores[
                comp_train_mask
            ],
            comp_labels[
                comp_train_mask
            ],
        )

        print(
            f"fold={outer_fold} "
            f"exact_a="
            f"{exact_platt['a']:.8f} "
            f"exact_b="
            f"{exact_platt['b']:.8f} "
            f"comp_a="
            f"{comp_platt['a']:.8f} "
            f"comp_b="
            f"{comp_platt['b']:.8f}",
            flush=True,
        )

        (
            X_train,
            y_train,
            group_train,
            _,
            _,
        ) = build_matrix_for_ids(
            train_ids,
            exact_platt,
            comp_platt,
            verify_frozen=False,
        )

        (
            X_test,
            y_test,
            group_test,
            meta_test,
            match_count,
        ) = build_matrix_for_ids(
            test_ids,
            exact_platt,
            comp_platt,
            verify_frozen=True,
        )

        frozen_test_pool_matches += (
            match_count
        )

        if X_train.shape[1] != len(
            FEATURE_NAMES
        ):
            raise RuntimeError(
                "Feature dimension mismatch"
            )

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            group=group_train,
            feature_name=FEATURE_NAMES,
            free_raw_data=True,
        )

        t0 = time.perf_counter()

        booster = lgb.train(
            params,
            train_data,
            num_boost_round=rounds,
        )

        train_seconds = (
            time.perf_counter()
            - t0
        )

        model_path = (
            args.output_root
            / f"outer_fold_{outer_fold}.txt"
        )

        booster.save_model(
            str(model_path)
        )

        p0 = time.perf_counter()

        scores = np.asarray(
            booster.predict(
                X_test
            ),
            dtype=np.float64,
        )

        predict_seconds = (
            time.perf_counter()
            - p0
        )

        for q_index, row_id in enumerate(
            test_ids
        ):
            base = m0[
                row_id
            ]

            info = meta_test[
                q_index
            ]

            start = int(
                info["start"]
            )

            end = int(
                info["end"]
            )

            r6_rank = rank_from_scores(
                y_test[
                    start:end
                ],
                scores[
                    start:end
                ],
            )

            full_texts = full[
                row_id
            ]["ranked_texts"]

            try:
                full_rank = (
                    full_texts.index(
                        base["gold"]
                    )
                    + 1
                )
            except ValueError:
                full_rank = None

            add_metric(
                full_metric,
                full_rank,
            )

            add_metric(
                r6_metric,
                r6_rank,
            )

            add_metric(
                by_length[
                    base["length"]
                ],
                r6_rank,
            )

            add_metric(
                by_author[
                    base["author"]
                ],
                r6_rank,
            )

            old_correct = (
                full_rank == 1
            )

            new_correct = (
                r6_rank == 1
            )

            if (
                not old_correct
                and new_correct
            ):
                rescue += 1

            elif (
                old_correct
                and not new_correct
            ):
                harm += 1

            elif (
                old_correct
                and new_correct
            ):
                unchanged_correct += 1

            else:
                unchanged_wrong += 1

            generic_pruned = (
                frozen_r5[
                    row_id
                ][
                    "generic_pruned"
                ]
            )

            if generic_pruned:
                add_metric(
                    gp_metric,
                    r6_rank,
                )

                if (
                    base["gold"]
                    in info[
                        "personal_texts"
                    ]
                ):
                    recovered_n += 1

                    add_metric(
                        recovered_metric,
                        r6_rank,
                    )

            query_result_rows.append(
                {
                    "row_id": row_id,
                    "family_id": base[
                        "family_id"
                    ],
                    "outer_fold": (
                        outer_fold
                    ),
                    "multi_token_length": (
                        base["length"]
                    ),
                    "generic_pruned": (
                        generic_pruned
                    ),
                    "frozen_full_gold_rank": (
                        full_rank
                    ),
                    "r6_gold_rank": (
                        r6_rank
                    ),
                    "gold_in_personal_top5": (
                        base["gold"]
                        in info[
                            "personal_texts"
                        ]
                    ),
                }
            )

        fold_results.append(
            {
                "fold": outer_fold,
                "train_queries": len(
                    train_ids
                ),
                "test_queries": len(
                    test_ids
                ),
                "train_candidates": int(
                    len(y_train)
                ),
                "test_candidates": int(
                    len(y_test)
                ),
                "test_pool_reproduction_matches": (
                    match_count
                ),
                "exact_platt": (
                    exact_platt
                ),
                "composition_platt": (
                    comp_platt
                ),
                "train_seconds": (
                    train_seconds
                ),
                "predict_seconds": (
                    predict_seconds
                ),
                "predict_ms_per_query": (
                    1000.0
                    * predict_seconds
                    / len(test_ids)
                ),
                "model_sha256": (
                    sha256_file(
                        model_path
                    )
                ),
            }
        )

        print(
            f"fold={outer_fold} "
            f"train_q={len(train_ids)} "
            f"test_q={len(test_ids)} "
            f"train_c={len(y_train)} "
            f"test_c={len(y_test)} "
            f"pool_match={match_count} "
            f"train_s={train_seconds:.3f} "
            f"predict_s={predict_seconds:.3f}",
            flush=True,
        )

        del (
            X_train,
            y_train,
            X_test,
            y_test,
            scores,
            train_data,
            booster,
        )

    if (
        frozen_test_pool_matches
        != EXPECTED_QUERIES
    ):
        raise RuntimeError(
            "Frozen R5 held-out pool "
            "reproduction incomplete"
        )

    if recovered_n != 6209:
        raise RuntimeError(
            "Recovered subset mismatch: "
            f"{recovered_n}"
        )

    full_result = finalize(
        full_metric
    )

    r6_result = finalize(
        r6_metric
    )

    gp_result = finalize(
        gp_metric
    )

    recovered_result = finalize(
        recovered_metric
    )

    by_length_result = {
        f"M{i}": finalize(
            by_length[i]
        )
        for i in range(1, 6)
    }

    author_results = {
        author: finalize(
            metric
        )
        for author, metric
        in by_author.items()
    }

    macro_top1 = float(
        np.mean(
            [
                x["top1"]
                for x in author_results.values()
            ]
        )
    )

    print()
    print(
        "===== R6-v2 NESTED OOF RESULTS ====="
    )

    print(
        "FROZEN_FULL_TOP1 =",
        full_result["top1"],
    )

    print(
        "FROZEN_FULL_TOP3 =",
        full_result["top3"],
    )

    print(
        "FROZEN_FULL_TOP5 =",
        full_result["top5"],
    )

    print(
        "FROZEN_FULL_TOP10 =",
        full_result["top10"],
    )

    print(
        "FROZEN_FULL_MRR =",
        full_result["mrr"],
    )

    print()
    print(
        "R6_V2_MICRO_TOP1 =",
        r6_result["top1"],
    )

    print(
        "R6_V2_MICRO_TOP3 =",
        r6_result["top3"],
    )

    print(
        "R6_V2_MICRO_TOP5 =",
        r6_result["top5"],
    )

    print(
        "R6_V2_MICRO_TOP10 =",
        r6_result["top10"],
    )

    print(
        "R6_V2_MRR =",
        r6_result["mrr"],
    )

    print(
        "R6_V2_MISSING =",
        r6_result["missing"],
    )

    print(
        "R6_V2_MACRO_TOP1 =",
        macro_top1,
    )

    print()
    print(
        "TOP1_RESCUE_VS_FULL =",
        rescue,
    )

    print(
        "TOP1_HARM_VS_FULL =",
        harm,
    )

    print(
        "TOP1_NET_VS_FULL =",
        rescue - harm,
    )

    print()
    print(
        "GENERIC_PRUNED_N =",
        gp_result["n"],
    )

    print(
        "GP_R6_V2_TOP1 =",
        gp_result["top1"],
    )

    print(
        "GP_R6_V2_TOP3 =",
        gp_result["top3"],
    )

    print(
        "GP_R6_V2_TOP5 =",
        gp_result["top5"],
    )

    print()
    print(
        "RECOVERED_N =",
        recovered_result["n"],
    )

    print(
        "RECOVERED_CONDITIONAL_TOP1 =",
        recovered_result["top1"],
    )

    print(
        "RECOVERED_CONDITIONAL_TOP3 =",
        recovered_result["top3"],
    )

    print(
        "RECOVERED_CONDITIONAL_TOP5 =",
        recovered_result["top5"],
    )

    print()
    print(
        "PER_LENGTH =",
        by_length_result,
    )

    # --------------------------------------------------
    # Final operational model for future unseen split.
    # All-Val branch calibrators + all-Val R6.
    # --------------------------------------------------

    print()
    print(
        "===== FINAL ALL-VAL OPERATIONAL MODEL ====="
    )

    final_exact_platt = fit_platt(
        exact_scores,
        exact_labels,
    )

    final_comp_platt = fit_platt(
        comp_scores,
        comp_labels,
    )

    (
        X_final,
        y_final,
        group_final,
        meta_final,
        _,
    ) = build_matrix_for_ids(
        query_ids,
        final_exact_platt,
        final_comp_platt,
        verify_frozen=False,
    )

    final_data = lgb.Dataset(
        X_final,
        label=y_final,
        group=group_final,
        feature_name=FEATURE_NAMES,
        free_raw_data=True,
    )

    t0 = time.perf_counter()

    final_model = lgb.train(
        params,
        final_data,
        num_boost_round=rounds,
    )

    final_train_seconds = (
        time.perf_counter()
        - t0
    )

    final_model_path = (
        args.output_root
        / "final_r6_v2_lambdamart.txt"
    )

    final_model.save_model(
        str(final_model_path)
    )

    importance = {
        name: float(value)
        for name, value
        in sorted(
            zip(
                FEATURE_NAMES,
                final_model.feature_importance(
                    importance_type="gain"
                ),
            ),
            key=lambda x:
                -x[1],
        )
    }

    latency_values = []

    for q in range(
        min(
            2000,
            EXPECTED_QUERIES,
        )
    ):
        info = meta_final[q]

        start = int(
            info["start"]
        )

        end = int(
            info["end"]
        )

        t0 = time.perf_counter_ns()

        _ = final_model.predict(
            X_final[
                start:end
            ]
        )

        latency_values.append(
            (
                time.perf_counter_ns()
                - t0
            )
            / 1_000_000.0
        )

    latency = latency_summary(
        latency_values
    )

    query_result_path = (
        args.output_root
        / "oof_query_results.jsonl"
    )

    with query_result_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        for row in query_result_rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    summary = {
        "schema_version": 2,
        "status": "complete",
        "experiment": (
            "m4_r6_nested_full_signal_"
            "lambdamart_v2"
        ),
        "evaluation_protocol": (
            "5-fold family-level outer OOF; "
            "Exact and Composition Platt "
            "calibrators re-fit exclusively "
            "inside each outer training fold; "
            "held-out fusion pools reproduce "
            "Frozen R5 OOF pools exactly"
        ),
        "features": (
            FEATURE_NAMES
        ),
        "folds": (
            fold_results
        ),
        "frozen_r5_test_pool_matches": (
            frozen_test_pool_matches
        ),
        "frozen_full_baseline": (
            full_result
        ),
        "r6_v2_oof_micro": (
            r6_result
        ),
        "r6_v2_macro_author_top1": (
            macro_top1
        ),
        "per_multi_length": (
            by_length_result
        ),
        "generic_pruned": (
            gp_result
        ),
        "generic_pruned_recovered_subset": (
            recovered_result
        ),
        "top1_transition_vs_frozen_full": {
            "rescue": rescue,
            "harm": harm,
            "net": rescue - harm,
            "unchanged_correct": (
                unchanged_correct
            ),
            "unchanged_wrong": (
                unchanged_wrong
            ),
        },
        "final_operational_model": {
            "exact_platt": (
                final_exact_platt
            ),
            "composition_platt": (
                final_comp_platt
            ),
            "model_sha256": (
                sha256_file(
                    final_model_path
                )
            ),
            "train_seconds": (
                final_train_seconds
            ),
        },
        "feature_importance_gain": (
            importance
        ),
        "latency_ms_per_query": {
            "final_ranker_predict_only": (
                latency
            ),
            "scope": (
                "LambdaMART scoring only over "
                "preconstructed <=15 candidate pool"
            ),
        },
        "historical_provenance_note": {
            "current_windows_r4_m1_top1": (
                0.8808542129189132
            ),
            "historical_hpc_m1_top1": (
                0.880801353
            ),
            "difference_in_correct_rows": 1,
            "interpretation": (
                "Recorded as a one-row reconstruction "
                "discrepancy; not silently reconciled."
            ),
        },
        "training": {
            "objective": "lambdarank",
            "rounds": rounds,
            "params": params,
            "hyperparameter_sweep": False,
        },
        "hashes": {
            "runner": sha256_file(
                Path(__file__)
            ),
            "m0": sha256_file(
                args.m0
            ),
            "r4": sha256_file(
                args.r4
            ),
            "exact": sha256_file(
                args.exact
            ),
            "composition": (
                sha256_file(
                    args.composition
                )
            ),
            "r5": sha256_file(
                args.r5
            ),
            "r5_freeze": (
                sha256_file(
                    args.r5_freeze
                )
            ),
            "oof_query_results": (
                sha256_file(
                    query_result_path
                )
            ),
        },
        "gold_used_for_candidate_generation": False,
        "gold_used_for_heldout_calibration": False,
        "gold_used_for_heldout_fusion": False,
        "gold_used_for_heldout_ranking": False,
        "gold_used_for_training_labels": True,
        "gold_used_for_posthoc_evaluation": True,
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
        "FEATURE_IMPORTANCE_GAIN =",
        importance,
    )

    print(
        "R6_V2_SCORING_LATENCY_MS =",
        latency,
    )

    print(
        "FINAL_MODEL_SHA =",
        summary[
            "final_operational_model"
        ][
            "model_sha256"
        ],
    )

    print(
        "SUMMARY_SHA =",
        sha256_file(
            summary_path
        ),
    )

    print(
        "RUNNER_SHA =",
        summary[
            "hashes"
        ][
            "runner"
        ],
    )

    print()
    print(
        "HELDOUT_R5_POOL_REPRODUCTION=PASS"
    )

    print(
        "NESTED_CALIBRATION_NO_OUTER_FOLD_LEAKAGE=PASS"
    )

    print(
        "NO_R6_HYPERPARAMETER_SWEEP=PASS"
    )

    print(
        "DEV3000_TEST_CLOSED=PASS"
    )

    print(
        "M4_R6_V2_FINAL_RANKING_GATE=PASS"
    )


if __name__ == "__main__":
    main()