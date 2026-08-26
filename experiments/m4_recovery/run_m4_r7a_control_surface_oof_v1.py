from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize


EXPECTED_QUERIES = 94590
EXPECTED_FAMILIES = 18918
EXPECTED_CANDIDATES = 1146931
EXPECTED_POSITIVES = 74367
N_FOLDS = 5

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


def sigmoid(x):
    x = np.clip(
        np.asarray(
            x,
            dtype=np.float64,
        ),
        -40.0,
        40.0,
    )

    return 1.0 / (
        1.0 + np.exp(-x)
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

    positives = float(
        y.sum()
    )

    if not (
        0 < positives < len(y)
    ):
        raise RuntimeError(
            "Platt split lacks both classes"
        )

    mean = float(
        x.mean()
    )

    std = float(
        x.std()
    )

    if std < 1e-12:
        std = 1.0

    z = (
        x - mean
    ) / std

    base_rate = float(
        y.mean()
    )

    intercept0 = math.log(
        base_rate
        / (
            1.0 - base_rate
        )
    )

    def objective(theta):
        slope = float(
            theta[0]
        )

        intercept = float(
            theta[1]
        )

        eta = (
            slope * z
            + intercept
        )

        p = sigmoid(
            eta
        )

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

        residual = (
            p - y
        )

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

        return (
            loss,
            grad,
        )

    kwargs = dict(
        fun=lambda t:
            objective(t)[0],
        x0=np.asarray(
            [
                1.0,
                intercept0,
            ],
            dtype=np.float64,
        ),
        jac=lambda t:
            objective(t)[1],
        bounds=[
            (1e-8, None),
            (None, None),
        ],
    )

    result = minimize(
        method="L-BFGS-B",
        options={
            "maxiter": 300,
            "ftol": 1e-12,
            "gtol": 1e-8,
        },
        **kwargs,
    )

    optimizer = (
        "L-BFGS-B"
    )

    if not result.success:
        result = minimize(
            method="SLSQP",
            options={
                "maxiter": 2000,
                "ftol": 1e-12,
            },
            **kwargs,
        )

        optimizer = (
            "SLSQP_FALLBACK"
        )

    if not result.success:
        raise RuntimeError(
            "R7 Platt fit failed: "
            + str(
                result.message
            )
        )

    slope_std = float(
        result.x[0]
    )

    intercept_std = float(
        result.x[1]
    )

    a = (
        slope_std
        / std
    )

    b = (
        intercept_std
        - slope_std
        * mean
        / std
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
        "a": float(a),
        "b": float(b),
        "optimizer": optimizer,
        "training_candidates": int(
            len(y)
        ),
        "training_positives": int(
            y.sum()
        ),
        "training_base_rate": float(
            y.mean()
        ),
    }


def metrics(labels, probs):
    y = np.asarray(
        labels,
        dtype=np.float64,
    )

    p = np.clip(
        np.asarray(
            probs,
            dtype=np.float64,
        ),
        1e-12,
        1.0 - 1e-12,
    )

    return {
        "n": int(
            len(y)
        ),
        "positives": int(
            y.sum()
        ),
        "base_rate": float(
            y.mean()
        ),
        "mean_probability": float(
            p.mean()
        ),
        "brier": float(
            np.mean(
                (
                    p - y
                ) ** 2
            )
        ),
        "logloss": float(
            -np.mean(
                y * np.log(p)
                + (
                    1.0 - y
                )
                * np.log(
                    1.0 - p
                )
            )
        ),
    }


def rank_of_gold(
    candidates,
    score_key,
):
    ordered = sorted(
        enumerate(
            candidates
        ),
        key=lambda x: (
            -float(
                x[1][
                    score_key
                ]
            ),
            x[0],
        ),
    )

    for rank, (
        _,
        cand,
    ) in enumerate(
        ordered,
        start=1,
    ):
        if cand[
            "is_gold_posthoc"
        ]:
            return rank

    return None


def feature_vector(
    *,
    generic_info,
    full_info,
    personal_info,
    comp_info,
    length,
    pool_size,
):
    is_generic = (
        generic_info
        is not None
    )

    has_full = (
        full_info
        is not None
    )

    is_personal = (
        personal_info
        is not None
    )

    p_exact = (
        float(
            personal_info.get(
                "p_exact",
                0.0,
            )
            or 0.0
        )
        if is_personal
        else 0.0
    )

    exact_raw = (
        float(
            personal_info.get(
                "exact_raw_score",
                0.0,
            )
            or 0.0
        )
        if is_personal
        else 0.0
    )

    if is_personal:
        p_comp = float(
            personal_info.get(
                "p_composition",
                0.0,
            )
            or 0.0
        )

        comp_raw = float(
            personal_info.get(
                "composition_raw_score",
                0.0,
            )
            or 0.0
        )

        piece_count = float(
            personal_info.get(
                "piece_count",
                0,
            )
            or 0
        )

        fragmentation = float(
            personal_info.get(
                "fragmentation_ratio",
                0.0,
            )
            or 0.0
        )

        longest_span = float(
            personal_info.get(
                "longest_span_ratio",
                0.0,
            )
            or 0.0
        )

    elif comp_info is not None:
        p_comp = float(
            comp_info[
                "p"
            ]
        )

        comp_raw = float(
            comp_info[
                "raw"
            ]
        )

        piece_count = float(
            comp_info[
                "piece_count"
            ]
        )

        fragmentation = float(
            comp_info[
                "fragmentation"
            ]
        )

        longest_span = float(
            comp_info[
                "longest_span"
            ]
        )

    else:
        p_comp = 0.0
        comp_raw = 0.0
        piece_count = 0.0
        fragmentation = 0.0
        longest_span = 0.0

    recovery_confidence = max(
        p_exact,
        p_comp,
    )

    return [
        float(
            is_generic
        ),
        float(
            generic_info["rank"]
            if is_generic
            else 0
        ),
        (
            1.0
            / generic_info["rank"]
            if is_generic
            else 0.0
        ),
        float(
            generic_info["logp"]
            if is_generic
            else 0.0
        ),
        float(
            generic_info["mean_logp"]
            if is_generic
            else 0.0
        ),
        float(
            generic_info["gap"]
            if is_generic
            else 0.0
        ),
        float(
            has_full
        ),
        float(
            full_info["score"]
            if has_full
            else 0.0
        ),
        float(
            full_info["rank"]
            if has_full
            else 0
        ),
        (
            1.0
            / full_info["rank"]
            if has_full
            else 0.0
        ),
        float(
            full_info["gap"]
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
            is_personal
        ),
        float(
            personal_info["rank"]
            if is_personal
            else 0
        ),
        float(
            p_exact > 0
        ),
        exact_raw,
        p_exact,
        float(
            p_comp > 0
        ),
        comp_raw,
        p_comp,
        recovery_confidence,
        piece_count,
        fragmentation,
        longest_span,
        float(
            length
        ),
        float(
            pool_size
        ),
    ]


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
        "--r6-root",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--r6-summary",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--r6-query-results",
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
            "Refusing to overwrite "
            "non-empty R7-A output"
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    r6_summary = json.loads(
        args.r6_summary.read_text(
            encoding="utf-8-sig"
        )
    )

    if (
        r6_summary["status"]
        != "complete"
    ):
        raise RuntimeError(
            "R6-v2 is not complete"
        )

    print(
        "===== M4-R7A OOF CONTROL SURFACE ====="
    )

    print(
        "Frozen R6-v2 ranking models only"
    )

    print(
        "No ranker refit / no hyperparameter tuning"
    )

    print(
        "R6 raw score -> nested family-OOF Platt"
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

        raw_top = row[
            "top10_distinct_candidates"
        ]

        top_logp = (
            float(
                raw_top[0][
                    "log_probability"
                ]
            )
            if raw_top
            else 0.0
        )

        generic = {}

        for item in raw_top:
            text = str(
                item["text"]
            )

            generic[text] = {
                "rank": int(
                    item["rank"]
                ),
                "logp": float(
                    item[
                        "log_probability"
                    ]
                ),
                "mean_logp": float(
                    item[
                        "mean_log_probability"
                    ]
                ),
                "gap": (
                    float(
                        item[
                            "log_probability"
                        ]
                    )
                    - top_logp
                ),
            }

        m0[row_id] = {
            "gold": str(
                row["gold"]
            ),
            "family_id": str(
                row["family_id"]
            ),
            "length": int(
                row[
                    "multi_token_length"
                ]
            ),
            "generic": generic,
        }

    if len(
        m0
    ) != EXPECTED_QUERIES:
        raise RuntimeError(
            "M0 query count mismatch"
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

        cands = row[
            "zero_shot_candidates"
        ]

        top_score = (
            float(
                cands[0][
                    "lambdamart_score"
                ]
            )
            if cands
            else 0.0
        )

        by_text = {}

        for item in cands:
            text = str(
                item["text"]
            )

            score = float(
                item[
                    "lambdamart_score"
                ]
            )

            by_text[text] = {
                "score": score,
                "rank": int(
                    item["rank"]
                ),
                "gap": (
                    score
                    - top_score
                ),
                "source": str(
                    item["source"]
                ),
            }

        full[row_id] = by_text

    if set(
        full
    ) != set(
        m0
    ):
        raise RuntimeError(
            "M0/R4 row IDs differ"
        )

    # --------------------------------------------------
    # Frozen R6 OOF reference ranks.
    # --------------------------------------------------

    r6_reference = {}

    for row in read_jsonl(
        args.r6_query_results
    ):
        r6_reference[
            str(
                row["row_id"]
            )
        ] = row

    if len(
        r6_reference
    ) != EXPECTED_QUERIES:
        raise RuntimeError(
            "R6 reference query count mismatch"
        )

    # --------------------------------------------------
    # Load frozen outer-fold rankers.
    # --------------------------------------------------

    models = {}

    for fold in range(
        N_FOLDS
    ):
        path = (
            args.r6_root
            / f"outer_fold_{fold}.txt"
        )

        if not path.exists():
            raise RuntimeError(
                f"Missing fold model: {path}"
            )

        model = lgb.Booster(
            model_file=str(
                path
            )
        )

        if list(
            model.feature_name()
        ) != FEATURE_NAMES:
            raise RuntimeError(
                f"Feature mismatch fold={fold}"
            )

        models[fold] = model

    # --------------------------------------------------
    # First pass:
    # reconstruct exact frozen held-out pool + raw score.
    # --------------------------------------------------

    raw_surface_path = (
        args.output_root
        / "control_surface_raw.jsonl.gz"
    )

    all_scores = []
    all_labels = []
    all_folds = []

    query_count = 0
    candidate_count = 0
    positive_count = 0
    rank_match_count = 0

    family_ids = set()

    r5_iter = read_jsonl(
        args.r5
    )

    comp_iter = read_jsonl(
        args.composition
    )

    with gzip.open(
        raw_surface_path,
        "wt",
        encoding="utf-8",
        newline="\n",
        compresslevel=6,
    ) as sink:

        for r5row, comprow in (
            itertools.zip_longest(
                r5_iter,
                comp_iter,
            )
        ):
            if (
                r5row is None
                or comprow is None
            ):
                raise RuntimeError(
                    "R5/Composition row count mismatch"
                )

            query_count += 1

            row_id = str(
                r5row[
                    "row_id"
                ]
            )

            if row_id != str(
                comprow[
                    "row_id"
                ]
            ):
                raise RuntimeError(
                    "R5/Composition order mismatch"
                )

            base = m0.get(
                row_id
            )

            if base is None:
                raise RuntimeError(
                    f"Missing M0 row: {row_id}"
                )

            family_id = base[
                "family_id"
            ]

            family_ids.add(
                family_id
            )

            fold = int(
                r5row[
                    "calibration_fold"
                ]
            )

            if fold != int(
                comprow[
                    "calibration_fold"
                ]
            ):
                raise RuntimeError(
                    f"Fold mismatch: {row_id}"
                )

            if fold != int(
                r6_reference[
                    row_id
                ][
                    "outer_fold"
                ]
            ):
                raise RuntimeError(
                    f"R6 fold mismatch: {row_id}"
                )

            comp_map = {}

            for item in comprow[
                "candidates"
            ]:
                text = str(
                    item["text"]
                )

                p = float(
                    item[
                        "oof_calibrated_probability"
                    ]
                )

                current = (
                    comp_map.get(
                        text
                    )
                )

                if (
                    current is None
                    or p
                    > current["p"]
                ):
                    comp_map[
                        text
                    ] = {
                        "p": p,
                        "raw": float(
                            item[
                                "c1_raw_score"
                            ]
                        ),
                        "piece_count": int(
                            item[
                                "piece_count"
                            ]
                        ),
                        "fragmentation": float(
                            item[
                                "fragmentation_ratio"
                            ]
                        ),
                        "longest_span": float(
                            item[
                                "longest_span_ratio"
                            ]
                        ),
                    }

            generic_texts = [
                str(x)
                for x in r5row[
                    "generic_top10"
                ]
            ]

            personal = r5row[
                "personal_recovery_top5"
            ]

            pool_size = (
                len(
                    generic_texts
                )
                + len(
                    personal
                )
            )

            if pool_size > 15:
                raise RuntimeError(
                    f"Pool >15: {row_id}"
                )

            candidates = []
            X = []

            for text in generic_texts:
                generic_info = (
                    base[
                        "generic"
                    ].get(
                        text
                    )
                )

                if (
                    generic_info
                    is None
                ):
                    raise RuntimeError(
                        "Generic metadata missing: "
                        f"{row_id} {text}"
                    )

                full_info = (
                    full[
                        row_id
                    ].get(
                        text
                    )
                )

                comp_info = (
                    comp_map.get(
                        text
                    )
                )

                X.append(
                    feature_vector(
                        generic_info=(
                            generic_info
                        ),
                        full_info=(
                            full_info
                        ),
                        personal_info=None,
                        comp_info=(
                            comp_info
                        ),
                        length=(
                            base[
                                "length"
                            ]
                        ),
                        pool_size=(
                            pool_size
                        ),
                    )
                )

                p_comp = (
                    float(
                        comp_info[
                            "p"
                        ]
                    )
                    if comp_info
                    is not None
                    else 0.0
                )

                candidates.append(
                    {
                        "text": text,
                        "source": "generic",
                        "generic_rank": int(
                            generic_info[
                                "rank"
                            ]
                        ),
                        "personal_rank": None,
                        "p_exact": 0.0,
                        "p_composition": (
                            p_comp
                        ),
                        "recovery_confidence": (
                            p_comp
                        ),
                        "fragmentation_ratio": (
                            float(
                                comp_info[
                                    "fragmentation"
                                ]
                            )
                            if comp_info
                            is not None
                            else 0.0
                        ),
                        "is_gold_posthoc": (
                            text
                            == base[
                                "gold"
                            ]
                        ),
                    }
                )

            for item in personal:
                text = str(
                    item["text"]
                )

                full_info = (
                    full[
                        row_id
                    ].get(
                        text
                    )
                )

                X.append(
                    feature_vector(
                        generic_info=None,
                        full_info=(
                            full_info
                        ),
                        personal_info=(
                            item
                        ),
                        comp_info=None,
                        length=(
                            base[
                                "length"
                            ]
                        ),
                        pool_size=(
                            pool_size
                        ),
                    )
                )

                pe = float(
                    item.get(
                        "p_exact",
                        0.0,
                    )
                    or 0.0
                )

                pc = float(
                    item.get(
                        "p_composition",
                        0.0,
                    )
                    or 0.0
                )

                candidates.append(
                    {
                        "text": text,
                        "source": (
                            "personal_recovery"
                        ),
                        "generic_rank": None,
                        "personal_rank": int(
                            item[
                                "rank"
                            ]
                        ),
                        "provenance": (
                            item.get(
                                "provenance"
                            )
                        ),
                        "p_exact": pe,
                        "p_composition": pc,
                        "recovery_confidence": max(
                            pe,
                            pc,
                        ),
                        "fragmentation_ratio": float(
                            item.get(
                                "fragmentation_ratio",
                                0.0,
                            )
                            or 0.0
                        ),
                        "is_gold_posthoc": (
                            text
                            == base[
                                "gold"
                            ]
                        ),
                    }
                )

            X = np.asarray(
                X,
                dtype=np.float32,
            )

            if (
                X.shape[1]
                != len(
                    FEATURE_NAMES
                )
            ):
                raise RuntimeError(
                    "Feature dimension mismatch"
                )

            scores = np.asarray(
                models[
                    fold
                ].predict(
                    X
                ),
                dtype=np.float64,
            )

            for cand, score in zip(
                candidates,
                scores,
            ):
                cand[
                    "r6_raw_score"
                ] = float(
                    score
                )

                all_scores.append(
                    float(
                        score
                    )
                )

                label = int(
                    cand[
                        "is_gold_posthoc"
                    ]
                )

                all_labels.append(
                    label
                )

                all_folds.append(
                    fold
                )

                positive_count += (
                    label
                )

            candidate_count += len(
                candidates
            )

            observed_rank = (
                rank_of_gold(
                    candidates,
                    "r6_raw_score",
                )
            )

            expected_rank = (
                r6_reference[
                    row_id
                ][
                    "r6_gold_rank"
                ]
            )

            if (
                observed_rank
                != expected_rank
            ):
                raise RuntimeError(
                    "Frozen R6 rank reproduction "
                    f"failed: {row_id}: "
                    f"{observed_rank} != "
                    f"{expected_rank}"
                )

            rank_match_count += 1

            sink.write(
                json.dumps(
                    {
                        "row_id": row_id,
                        "family_id": (
                            family_id
                        ),
                        "outer_fold": (
                            fold
                        ),
                        "multi_token_length": (
                            base[
                                "length"
                            ]
                        ),
                        "generic_beam_pruned": bool(
                            r5row[
                                "generic_beam_pruned"
                            ]
                        ),
                        "gold_posthoc": (
                            base[
                                "gold"
                            ]
                        ),
                        "candidates": (
                            candidates
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

            if (
                query_count
                % 10000
                == 0
            ):
                print(
                    "surface "
                    f"{query_count}/"
                    f"{EXPECTED_QUERIES}",
                    flush=True,
                )

    if (
        query_count
        != EXPECTED_QUERIES
    ):
        raise RuntimeError(
            "Query count mismatch"
        )

    if (
        len(
            family_ids
        )
        != EXPECTED_FAMILIES
    ):
        raise RuntimeError(
            "Family count mismatch"
        )

    if (
        candidate_count
        != EXPECTED_CANDIDATES
    ):
        raise RuntimeError(
            "Candidate count mismatch: "
            f"{candidate_count}"
        )

    if (
        positive_count
        != EXPECTED_POSITIVES
    ):
        raise RuntimeError(
            "Positive count mismatch: "
            f"{positive_count}"
        )

    if (
        rank_match_count
        != EXPECTED_QUERIES
    ):
        raise RuntimeError(
            "R6 rank reproduction incomplete"
        )

    print()
    print(
        "FROZEN_R6_QUERY_RANK_MATCHES =",
        rank_match_count,
    )

    print(
        "CANDIDATE_OBSERVATIONS =",
        candidate_count,
    )

    print(
        "POSITIVE_OBSERVATIONS =",
        positive_count,
    )

    scores = np.asarray(
        all_scores,
        dtype=np.float64,
    )

    labels = np.asarray(
        all_labels,
        dtype=np.int8,
    )

    folds = np.asarray(
        all_folds,
        dtype=np.int8,
    )

    del (
        all_scores,
        all_labels,
        all_folds,
    )

    # --------------------------------------------------
    # Nested family-OOF calibration of final R6 score.
    # --------------------------------------------------

    print()
    print(
        "===== R6 SCORE FAMILY-OOF PLATT ====="
    )

    oof_prob = np.full(
        len(
            scores
        ),
        np.nan,
        dtype=np.float64,
    )

    fold_platts = []

    for fold in range(
        N_FOLDS
    ):
        train = (
            folds
            != fold
        )

        test = (
            folds
            == fold
        )

        params = fit_platt(
            scores[
                train
            ],
            labels[
                train
            ],
        )

        z = (
            float(
                params[
                    "a"
                ]
            )
            * scores[
                test
            ]
            + float(
                params[
                    "b"
                ]
            )
        )

        oof_prob[
            test
        ] = sigmoid(
            z
        )

        fold_platts.append(
            {
                "fold": fold,
                **params,
                "test_candidates": int(
                    test.sum()
                ),
            }
        )

        print(
            f"fold={fold} "
            f"a={params['a']:.8f} "
            f"b={params['b']:.8f} "
            f"optimizer="
            f"{params['optimizer']}",
            flush=True,
        )

    if not np.isfinite(
        oof_prob
    ).all():
        raise RuntimeError(
            "Incomplete R6 OOF calibration"
        )

    calibration = metrics(
        labels,
        oof_prob,
    )

    base_rate = float(
        labels.mean()
    )

    base_prob = np.full(
        len(labels),
        base_rate,
        dtype=np.float64,
    )

    base_metrics = metrics(
        labels,
        base_prob,
    )

    print()
    print(
        "OOF_MEAN_PROBABILITY =",
        calibration[
            "mean_probability"
        ],
    )

    print(
        "BASE_RATE =",
        calibration[
            "base_rate"
        ],
    )

    print(
        "OOF_BRIER =",
        calibration[
            "brier"
        ],
    )

    print(
        "BASE_BRIER =",
        base_metrics[
            "brier"
        ],
    )

    print(
        "OOF_LOGLOSS =",
        calibration[
            "logloss"
        ],
    )

    print(
        "BASE_LOGLOSS =",
        base_metrics[
            "logloss"
        ],
    )

    # --------------------------------------------------
    # Second pass: attach OOF calibrated base probability
    # and log-odds control coordinate.
    # --------------------------------------------------

    calibrated_path = (
        args.output_root
        / "control_surface_oof.jsonl.gz"
    )

    index = 0
    calibrated_rank_matches = 0

    with gzip.open(
        raw_surface_path,
        "rt",
        encoding="utf-8",
    ) as source, gzip.open(
        calibrated_path,
        "wt",
        encoding="utf-8",
        newline="\n",
        compresslevel=6,
    ) as sink:

        for line in source:
            if not line.strip():
                continue

            row = json.loads(
                line
            )

            fold = int(
                row[
                    "outer_fold"
                ]
            )

            params = (
                fold_platts[
                    fold
                ]
            )

            for cand in row[
                "candidates"
            ]:
                raw = float(
                    cand[
                        "r6_raw_score"
                    ]
                )

                z_base = (
                    float(
                        params[
                            "a"
                        ]
                    )
                    * raw
                    + float(
                        params[
                            "b"
                        ]
                    )
                )

                p_base = float(
                    sigmoid(
                        np.asarray(
                            [
                                z_base
                            ]
                        )
                    )[0]
                )

                cand[
                    "z_base"
                ] = float(
                    z_base
                )

                cand[
                    "p_base"
                ] = (
                    p_base
                )

                # Frozen control primitives.
                cand[
                    "phi_personal"
                ] = float(
                    cand[
                        "recovery_confidence"
                    ]
                )

                cand[
                    "phi_exact"
                ] = float(
                    cand[
                        "p_exact"
                    ]
                )

                cand[
                    "phi_composition"
                ] = float(
                    cand[
                        "p_composition"
                    ]
                )

                cand[
                    "phi_fragmentation"
                ] = float(
                    cand[
                        "p_composition"
                    ]
                    * cand[
                        "fragmentation_ratio"
                    ]
                )

                index += 1

            raw_rank = rank_of_gold(
                row[
                    "candidates"
                ],
                "r6_raw_score",
            )

            calibrated_rank = (
                rank_of_gold(
                    row[
                        "candidates"
                    ],
                    "z_base",
                )
            )

            if (
                raw_rank
                != calibrated_rank
            ):
                raise RuntimeError(
                    "R6 calibration changed "
                    "candidate ranking: "
                    + row[
                        "row_id"
                    ]
                )

            calibrated_rank_matches += 1

            sink.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    if (
        index
        != EXPECTED_CANDIDATES
    ):
        raise RuntimeError(
            "Second-pass candidate "
            "count mismatch"
        )

    if (
        calibrated_rank_matches
        != EXPECTED_QUERIES
    ):
        raise RuntimeError(
            "Calibration ranking "
            "fidelity incomplete"
        )

    calibrator_path = (
        args.output_root
        / "r6_score_oof_platt.json"
    )

    calibrator_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "folds": (
                    fold_platts
                ),
                "calibration": (
                    calibration
                ),
                "constant_base": (
                    base_metrics
                ),
                "positive_slope_required": (
                    True
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment": (
            "m4_r7a_control_surface_oof_v1"
        ),
        "scientific_role": (
            "Leakage-safe OOF base-score "
            "surface for explicit post-model "
            "control interventions"
        ),
        "query_count": (
            query_count
        ),
        "family_count": len(
            family_ids
        ),
        "candidate_observations": (
            candidate_count
        ),
        "positive_observations": (
            positive_count
        ),
        "frozen_r6_rank_reproduction": {
            "queries_matched": (
                rank_match_count
            ),
            "queries_total": (
                query_count
            ),
        },
        "calibration_ranking_fidelity": {
            "queries_matched": (
                calibrated_rank_matches
            ),
            "queries_total": (
                query_count
            ),
            "ranking_changed": False,
        },
        "r6_score_calibration": {
            "folds": (
                fold_platts
            ),
            "oof": (
                calibration
            ),
            "constant_base": (
                base_metrics
            ),
        },
        "control_coordinate": (
            "z_base = logit(p_base); "
            "future controls add bounded "
            "explicit log-odds shifts "
            "without modifying LambdaMART"
        ),
        "control_primitives": {
            "personalisation_strength": (
                "phi_personal = "
                "recovery_confidence"
            ),
            "exact_preference": (
                "phi_exact = p_exact"
            ),
            "composition_preference": (
                "phi_composition = "
                "p_composition"
            ),
            "fragmentation_tolerance": (
                "phi_fragmentation = "
                "p_composition * "
                "fragmentation_ratio"
            ),
        },
        "hashes": {
            "runner": sha256_file(
                Path(
                    __file__
                )
            ),
            "m0": sha256_file(
                args.m0
            ),
            "r4": sha256_file(
                args.r4
            ),
            "composition": (
                sha256_file(
                    args.composition
                )
            ),
            "r5": sha256_file(
                args.r5
            ),
            "r6_summary": (
                sha256_file(
                    args.r6_summary
                )
            ),
            "r6_query_results": (
                sha256_file(
                    args.r6_query_results
                )
            ),
            "raw_surface": (
                sha256_file(
                    raw_surface_path
                )
            ),
            "oof_surface": (
                sha256_file(
                    calibrated_path
                )
            ),
            "calibrator": (
                sha256_file(
                    calibrator_path
                )
            ),
        },
        "gold_used_for_candidate_generation": False,
        "gold_used_for_base_ranking": False,
        "gold_used_for_heldout_score_calibration": False,
        "gold_used_for_calibration_training_labels": True,
        "gold_used_for_posthoc_validation": True,
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
        "FROZEN_R6_RANK_REPRODUCTION =",
        rank_match_count,
        "/",
        query_count,
    )

    print(
        "CALIBRATION_RANK_REPRODUCTION =",
        calibrated_rank_matches,
        "/",
        query_count,
    )

    print(
        "CONTROL_SURFACE_SHA =",
        summary[
            "hashes"
        ][
            "oof_surface"
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
        "R6_SCORE_OOF_CALIBRATION=PASS"
    )

    print(
        "BASELINE_CONTROL_FIDELITY=PASS"
    )

    print(
        "DEV3000_TEST_CLOSED=PASS"
    )

    print(
        "M4_R7A_CONTROL_SURFACE_GATE=PASS"
    )


if __name__ == "__main__":
    main()