from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize

import audit_m2_runtime_candidate_lattice_v1 as m2rt
import run_multi_m1c_pv_expansion_frozen_v2 as m1c
import run_m4_r1_bounded_composition_beam16_v1 as r1


R1_SHA = (
    "65793fce44cd8765e936bddcd24c0cdaa"
    "257bec0cf65eaebc87b0a417ab869b7"
)

R2_MODEL_SHA = (
    "9917cd51a47f591e2170bd52671f06cc"
    "bced49492ede30980a7993cb3c9e5cb4"
)

N_FOLDS = 5

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


def family_fold(family_id: str) -> int:
    digest = hashlib.sha256(
        family_id.encode("utf-8")
    ).digest()

    value = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return value % N_FOLDS


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


def matrix(
    candidates: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    if not candidates:
        return np.zeros(
            (0, len(FEATURE_NAMES)),
            dtype=np.float32,
        )

    return np.asarray(
        [
            features(candidate)
            for candidate in candidates
        ],
        dtype=np.float32,
    )


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(
        z,
        -40.0,
        40.0,
    )

    return 1.0 / (
        1.0 + np.exp(-z)
    )


def fit_platt(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """
    Monotone Platt calibration:
        p = sigmoid(a * raw_score + b)

    a is constrained positive so calibration cannot reverse
    the frozen Composition LambdaMART ranking.
    """
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
            "Empty calibration training data"
        )

    positives = float(y.sum())

    if positives <= 0:
        raise RuntimeError(
            "Calibration split has no positives"
        )

    if positives >= len(y):
        raise RuntimeError(
            "Calibration split has no negatives"
        )

    mean = float(x.mean())

    std = float(x.std())

    if std < 1e-12:
        std = 1.0

    z = (
        x - mean
    ) / std

    base_rate = float(y.mean())

    intercept0 = math.log(
        base_rate
        / (1.0 - base_rate)
    )

    def objective(
        theta: np.ndarray,
    ) -> tuple[float, np.ndarray]:
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

    result = minimize(
        fun=lambda theta: objective(theta)[0],
        x0=np.asarray(
            [1.0, intercept0],
            dtype=np.float64,
        ),
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
        bounds=[
            (1e-8, None),
            (None, None),
        ],
        options={
            "maxiter": 200,
            "ftol": 1e-12,
            "gtol": 1e-8,
        },
    )

    if not result.success:
        first_result = result

        result = minimize(
            fun=lambda theta: objective(theta)[0],
            x0=np.asarray(
                first_result.x,
                dtype=np.float64,
            ),
            jac=lambda theta: objective(theta)[1],
            method="L-BFGS-B",
            bounds=[
                (1e-8, None),
                (None, None),
            ],
            options={
                "maxiter": 200,
                "ftol": 1e-12,
                "gtol": 1e-8,
                "maxls": 100,
            },
        )

        if not result.success:
            raise RuntimeError(
                "Platt fit failed after deterministic "
                "L-BFGS-B line-search retry: "
                f"first={first_result.message}; "
                f"retry={result.message}; "
                f"first_fun={first_result.fun}; "
                f"retry_fun={result.fun}; "
                f"retry_jac={result.jac}"
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

    if a <= 0:
        raise RuntimeError(
            "Non-positive Platt slope"
        )

    return {
        "a": float(a),
        "b": float(b),
        "raw_score_mean": mean,
        "raw_score_std": std,
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


def apply_platt(
    scores: np.ndarray,
    params: Mapping[str, float],
) -> np.ndarray:
    return sigmoid(
        float(params["a"])
        * np.asarray(
            scores,
            dtype=np.float64,
        )
        + float(params["b"])
    )


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(
        labels,
        dtype=np.float64,
    )

    p = np.clip(
        np.asarray(
            probabilities,
            dtype=np.float64,
        ),
        1e-12,
        1.0 - 1e-12,
    )

    brier = float(
        np.mean(
            (p - y) ** 2
        )
    )

    logloss = float(
        -np.mean(
            y * np.log(p)
            + (1.0 - y)
            * np.log(1.0 - p)
        )
    )

    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": float(
            y.mean()
        ),
        "mean_probability": float(
            p.mean()
        ),
        "brier": brier,
        "logloss": logloss,
    }


def reliability(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 15,
) -> dict[str, Any]:
    y = np.asarray(
        labels,
        dtype=np.float64,
    )

    p = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if len(y) == 0:
        return {
            "ece": None,
            "max_gap": None,
            "bins": [],
        }

    order = np.argsort(
        p,
        kind="stable",
    )

    chunks = np.array_split(
        order,
        n_bins,
    )

    bins = []

    weighted_gap = 0.0
    max_gap = 0.0

    for number, idx in enumerate(
        chunks,
        start=1,
    ):
        if len(idx) == 0:
            continue

        mean_p = float(
            p[idx].mean()
        )

        empirical = float(
            y[idx].mean()
        )

        gap = abs(
            mean_p - empirical
        )

        weighted_gap += (
            len(idx)
            / len(y)
            * gap
        )

        max_gap = max(
            max_gap,
            gap,
        )

        bins.append(
            {
                "bin": number,
                "n": int(len(idx)),
                "mean_probability": mean_p,
                "empirical_gold_rate": empirical,
                "absolute_gap": gap,
                "min_probability": float(
                    p[idx].min()
                ),
                "max_probability": float(
                    p[idx].max()
                ),
            }
        )

    return {
        "ece": float(
            weighted_gap
        ),
        "max_gap": float(
            max_gap
        ),
        "bins": bins,
    }


def quantile(
    values: Sequence[float],
    q: float,
) -> float | None:
    if not values:
        return None

    xs = sorted(
        float(x)
        for x in values
    )

    if len(xs) == 1:
        return xs[0]

    pos = q * (
        len(xs) - 1
    )

    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return xs[lo]

    weight = pos - lo

    return (
        xs[lo] * (
            1.0 - weight
        )
        + xs[hi] * weight
    )


def latency_stats(
    values: Sequence[float],
) -> dict[str, float | None]:
    xs = [
        float(x)
        for x in values
    ]

    if not xs:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    return {
        "mean": float(
            statistics.fmean(xs)
        ),
        "median": quantile(
            xs,
            0.50,
        ),
        "p95": quantile(
            xs,
            0.95,
        ),
        "p99": quantile(
            xs,
            0.99,
        ),
        "max": max(xs),
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
        "--r2-model",
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
        default=10000,
    )

    args = ap.parse_args()

    if sha256_file(
        args.r1_script
    ) != R1_SHA:
        raise RuntimeError(
            "Frozen R1 SHA mismatch"
        )

    if sha256_file(
        args.r2_model
    ) != R2_MODEL_SHA:
        raise RuntimeError(
            "Frozen R2 model SHA mismatch"
        )

    if (
        args.output_root.exists()
        and any(
            args.output_root.iterdir()
        )
    ):
        raise RuntimeError(
            "Refusing to overwrite "
            "non-empty R3 output"
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    fit = read_jsonl(
        args.fit
    )

    val = read_jsonl(
        args.val
    )

    fit_multi = read_jsonl(
        args.fit_multi
    )

    val_multi = read_jsonl(
        args.val_multi
    )

    m0_rows = read_jsonl(
        args.m0_rows
    )

    val_by_id = {
        str(row["row_id"]): row
        for row in val
    }

    m0_by_id = {
        str(row["row_id"]): row
        for row in m0_rows
    }

    print(
        "Building frozen M1-C history ...",
        flush=True,
    )

    history = (
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

    lookup = (
        m2rt.RuntimeExpandedLookup(
            history
        )
    )

    model = lgb.Booster(
        model_file=str(
            args.r2_model
        )
    )

    if (
        list(
            model.feature_name()
        )
        != FEATURE_NAMES
    ):
        raise RuntimeError(
            "R2 feature order mismatch"
        )

    raw_path = (
        args.output_root
        / "composition_raw_candidates.jsonl"
    )

    score_chunks = []
    label_chunks = []
    fold_chunks = []
    gp_chunks = []

    retrieval_ms = []
    search_ms = []
    scoring_ms = []

    query_count = 0
    candidate_count = 0

    generation_started = (
        time.perf_counter()
    )

    print(
        "===== GENERATE TRAIN-VAL "
        "COMPOSITION SCORES =====",
        flush=True,
    )

    with raw_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as raw_file:

        for number, multi in enumerate(
            val_multi,
            start=1,
        ):
            row_id = str(
                multi["row_id"]
            )

            family_id = str(
                multi["family_id"]
            )

            source_id = str(
                multi[
                    "source_standard_row_id"
                ]
            )

            anchor = val_by_id.get(
                source_id
            )

            if anchor is None:
                raise RuntimeError(
                    "Missing Val anchor: "
                    f"{source_id}"
                )

            m0 = m0_by_id.get(
                row_id
            )

            if m0 is None:
                raise RuntimeError(
                    "Missing M0 row: "
                    f"{row_id}"
                )

            qrow = m1c.make_query_row(
                multi,
                anchor,
            )

            author = str(
                qrow["author"]
            )

            position = int(
                qrow[
                    "chronological_position"
                ]
            )

            pinyin = tuple(
                str(x)
                for x in (
                    multi.get(
                        "pinyin_segments"
                    )
                    or str(
                        multi[
                            "pinyin_input"
                        ]
                    ).split()
                )
            )

            gold = str(
                multi["gold"]
            )

            if len(gold) != len(
                pinyin
            ):
                raise RuntimeError(
                    "Gold/Pinyin mismatch: "
                    f"{row_id}"
                )

            r0 = time.perf_counter_ns()

            edges, _ = (
                r1.build_composition_edges(
                    lookup=lookup,
                    author=author,
                    position=position,
                    pinyin=pinyin,
                )
            )

            r1_ns = (
                time.perf_counter_ns()
            )

            candidates, _ = (
                r1.bounded_decode(
                    n=len(pinyin),
                    edges=edges,
                )
            )

            r2_ns = (
                time.perf_counter_ns()
            )

            X = matrix(
                candidates
            )

            if len(candidates):
                scores = np.asarray(
                    model.predict(X),
                    dtype=np.float64,
                )
            else:
                scores = np.zeros(
                    0,
                    dtype=np.float64,
                )

            r3_ns = (
                time.perf_counter_ns()
            )

            retrieval_ms.append(
                (
                    r1_ns - r0
                ) / 1_000_000.0
            )

            search_ms.append(
                (
                    r2_ns - r1_ns
                ) / 1_000_000.0
            )

            scoring_ms.append(
                (
                    r3_ns - r2_ns
                ) / 1_000_000.0
            )

            labels = np.asarray(
                [
                    1
                    if str(
                        candidate["text"]
                    ) == gold
                    else 0
                    for candidate
                    in candidates
                ],
                dtype=np.int8,
            )

            fold = family_fold(
                family_id
            )

            generic_pruned = (
                not m2rt.m0_gold_survived(
                    m0
                )
            )

            if len(candidates):
                score_chunks.append(
                    scores
                )

                label_chunks.append(
                    labels
                )

                fold_chunks.append(
                    np.full(
                        len(candidates),
                        fold,
                        dtype=np.int8,
                    )
                )

                gp_chunks.append(
                    np.full(
                        len(candidates),
                        generic_pruned,
                        dtype=np.bool_,
                    )
                )

            original_order = np.arange(
                len(candidates),
                dtype=np.int64,
            )

            if len(candidates):
                c1_order = np.lexsort(
                    (
                        original_order,
                        -scores,
                    )
                )

                inverse = np.empty(
                    len(candidates),
                    dtype=np.int64,
                )

                inverse[
                    c1_order
                ] = np.arange(
                    1,
                    len(candidates) + 1,
                )
            else:
                inverse = np.zeros(
                    0,
                    dtype=np.int64,
                )

            candidate_rows = []

            for i, candidate in enumerate(
                candidates
            ):
                candidate_rows.append(
                    {
                        "text": str(
                            candidate["text"]
                        ),
                        "c0_rank": (
                            i + 1
                        ),
                        "c0_score": float(
                            candidate[
                                "c0_score"
                            ]
                        ),
                        "c1_rank": int(
                            inverse[i]
                        ),
                        "c1_raw_score": float(
                            scores[i]
                        ),
                        "piece_count": int(
                            candidate[
                                "piece_count"
                            ]
                        ),
                        "fragmentation_ratio": float(
                            candidate[
                                "fragmentation_ratio"
                            ]
                        ),
                        "longest_span_ratio": float(
                            candidate[
                                "longest_span_ratio"
                            ]
                        ),
                    }
                )

            raw_row = {
                "row_id": row_id,
                "family_id": family_id,
                "multi_token_length": int(
                    multi[
                        "multi_token_length"
                    ]
                ),
                "calibration_fold": fold,
                "generic_beam_pruned": bool(
                    generic_pruned
                ),
                "gold": gold,
                "candidate_count": len(
                    candidates
                ),
                "candidates": candidate_rows,
            }

            raw_file.write(
                json.dumps(
                    raw_row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

            query_count += 1
            candidate_count += len(
                candidates
            )

            if (
                args.progress_every > 0
                and number
                % args.progress_every
                == 0
            ):
                print(
                    f"val={number}/"
                    f"{len(val_multi)} "
                    f"candidates="
                    f"{candidate_count}",
                    flush=True,
                )

    generation_seconds = (
        time.perf_counter()
        - generation_started
    )

    if not score_chunks:
        raise RuntimeError(
            "No Composition candidates"
        )

    scores = np.concatenate(
        score_chunks
    )

    labels = np.concatenate(
        label_chunks
    )

    folds = np.concatenate(
        fold_chunks
    )

    gp_mask = np.concatenate(
        gp_chunks
    )

    del score_chunks
    del label_chunks
    del fold_chunks
    del gp_chunks

    if len(scores) != candidate_count:
        raise RuntimeError(
            "Candidate count mismatch"
        )

    print()
    print(
        "candidate_observations =",
        len(scores),
    )

    print(
        "gold_candidate_observations =",
        int(labels.sum()),
    )

    print(
        "candidate_gold_base_rate =",
        float(labels.mean()),
    )

    print()
    print(
        "===== 5-FOLD FAMILY OOF "
        "PLATT CALIBRATION ====="
    )

    oof_prob = np.full(
        len(scores),
        np.nan,
        dtype=np.float64,
    )

    fold_parameters = {}

    calibration_fit_started = (
        time.perf_counter()
    )

    for fold in range(
        N_FOLDS
    ):
        train_mask = (
            folds != fold
        )

        test_mask = (
            folds == fold
        )

        params = fit_platt(
            scores[train_mask],
            labels[train_mask],
        )

        fold_parameters[
            str(fold)
        ] = params

        oof_prob[
            test_mask
        ] = apply_platt(
            scores[test_mask],
            params,
        )

        print(
            f"fold={fold} "
            f"train_n="
            f"{int(train_mask.sum())} "
            f"test_n="
            f"{int(test_mask.sum())} "
            f"a={params['a']:.8f} "
            f"b={params['b']:.8f}",
            flush=True,
        )

    calibration_fit_seconds = (
        time.perf_counter()
        - calibration_fit_started
    )

    if np.isnan(
        oof_prob
    ).any():
        raise RuntimeError(
            "Incomplete OOF calibration"
        )

    final_params = fit_platt(
        scores,
        labels,
    )

    final_prob = apply_platt(
        scores,
        final_params,
    )

    print()
    print(
        "FINAL_PLATT:",
        "a=",
        final_params["a"],
        "b=",
        final_params["b"],
    )

    overall_metrics = (
        binary_metrics(
            labels,
            oof_prob,
        )
    )

    overall_reliability = (
        reliability(
            labels,
            oof_prob,
        )
    )

    gp_metrics = binary_metrics(
        labels[gp_mask],
        oof_prob[gp_mask],
    )

    gp_reliability = reliability(
        labels[gp_mask],
        oof_prob[gp_mask],
    )

    base_rate = float(
        labels.mean()
    )

    base_prob = np.full(
        len(labels),
        base_rate,
        dtype=np.float64,
    )

    base_metrics = binary_metrics(
        labels,
        base_prob,
    )

    calibrated_path = (
        args.output_root
        / "composition_calibrated_oof.jsonl"
    )

    calibration_apply_ms = []

    top1_confidence = []
    top1_correct = []

    cursor = 0

    print()
    print(
        "===== WRITE OOF CALIBRATED "
        "QUERY ARTIFACT ====="
    )

    with (
        raw_path.open(
            "r",
            encoding="utf-8",
        )
    ) as src, (
        calibrated_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        )
    ) as dst:

        for line in src:
            row = json.loads(
                line
            )

            candidates = row[
                "candidates"
            ]

            n = len(
                candidates
            )

            query_scores = scores[
                cursor:cursor + n
            ]

            t0 = (
                time.perf_counter_ns()
            )

            query_final_prob = (
                apply_platt(
                    query_scores,
                    final_params,
                )
            )

            t1 = (
                time.perf_counter_ns()
            )

            calibration_apply_ms.append(
                (
                    t1 - t0
                ) / 1_000_000.0
            )

            query_oof = oof_prob[
                cursor:cursor + n
            ]

            for i, candidate in enumerate(
                candidates
            ):
                candidate[
                    "oof_calibrated_probability"
                ] = float(
                    query_oof[i]
                )

                candidate[
                    "final_calibrated_probability"
                ] = float(
                    query_final_prob[i]
                )

            candidates.sort(
                key=lambda x: (
                    int(
                        x["c1_rank"]
                    ),
                    str(
                        x["text"]
                    ),
                )
            )

            if candidates:
                top = candidates[0]

                top1_confidence.append(
                    float(
                        top[
                            "oof_calibrated_probability"
                        ]
                    )
                )

                top1_correct.append(
                    1
                    if str(
                        top["text"]
                    ) == str(
                        row["gold"]
                    )
                    else 0
                )

            row[
                "candidates"
            ] = candidates

            row[
                "gold_used_for_candidate_generation"
            ] = False

            row[
                "gold_used_for_candidate_scoring"
            ] = False

            row[
                "gold_used_for_calibration_label_only"
            ] = True

            dst.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

            cursor += n

    if cursor != len(
        scores
    ):
        raise RuntimeError(
            "Calibration output cursor mismatch"
        )

    top1_metrics = (
        binary_metrics(
            np.asarray(
                top1_correct,
                dtype=np.int8,
            ),
            np.asarray(
                top1_confidence,
                dtype=np.float64,
            ),
        )
    )

    top1_reliability = (
        reliability(
            np.asarray(
                top1_correct,
                dtype=np.int8,
            ),
            np.asarray(
                top1_confidence,
                dtype=np.float64,
            ),
        )
    )

    total_ms = [
        retrieval_ms[i]
        + search_ms[i]
        + scoring_ms[i]
        + calibration_apply_ms[i]
        for i in range(
            len(retrieval_ms)
        )
    ]

    calibrator_path = (
        args.output_root
        / "composition_platt_calibrator.json"
    )

    calibrator = {
        "experiment": (
            "m4_r3_composition_calibration_v1"
        ),
        "method": (
            "monotone_platt_logistic"
        ),
        "probability_definition": (
            "P(candidate_is_gold | "
            "candidate proposed by "
            "Composition branch, "
            "C1 LambdaMART raw score)"
        ),
        "cross_validation": {
            "method": (
                "5-fold family-level "
                "deterministic cross-fitting"
            ),
            "n_folds": N_FOLDS,
            "fold_parameters": (
                fold_parameters
            ),
        },
        "final_fit_for_future_unseen_data": (
            final_params
        ),
        "important": (
            "Train-Val reporting uses OOF "
            "probabilities only. "
            "The final all-Train-Val fit is "
            "reserved for later unseen data."
        ),
    }

    write_json(
        calibrator_path,
        calibrator,
    )

    summary = {
        "experiment": (
            "m4_r3_composition_calibration_v1"
        ),
        "runner_sha256": (
            sha256_file(
                Path(__file__)
            )
        ),
        "frozen_r1_sha256": (
            R1_SHA
        ),
        "frozen_r2_model_sha256": (
            R2_MODEL_SHA
        ),
        "query_count": query_count,
        "candidate_observations": int(
            len(scores)
        ),
        "gold_candidate_observations": int(
            labels.sum()
        ),
        "calibration": {
            "method": (
                "monotone Platt logistic"
            ),
            "evaluation": (
                "5-fold family-level OOF "
                "on Train-Val"
            ),
            "overall_candidate_level": {
                "oof": overall_metrics,
                "reliability": (
                    overall_reliability
                ),
                "constant_base_rate_baseline": (
                    base_metrics
                ),
            },
            "generic_pruned_candidate_level": {
                "oof": gp_metrics,
                "reliability": (
                    gp_reliability
                ),
            },
            "top1_candidate_confidence": {
                "oof": top1_metrics,
                "reliability": (
                    top1_reliability
                ),
            },
            "fold_parameters": (
                fold_parameters
            ),
            "final_parameters": (
                final_params
            ),
            "fit_seconds": (
                calibration_fit_seconds
            ),
        },
        "latency_ms_per_query": {
            "personal_span_retrieval": (
                latency_stats(
                    retrieval_ms
                )
            ),
            "composition_beam16_search": (
                latency_stats(
                    search_ms
                )
            ),
            "composition_c1_lambdamart": (
                latency_stats(
                    scoring_ms
                )
            ),
            "platt_calibration": (
                latency_stats(
                    calibration_apply_ms
                )
            ),
            "composition_total": (
                latency_stats(
                    total_ms
                )
            ),
        },
        "generation_wall_seconds": (
            generation_seconds
        ),
        "ranker_train_split": (
            "Train-Fit"
        ),
        "calibration_evaluation_split": (
            "Train-Val OOF"
        ),
        "used_dev3000": False,
        "used_test": False,
        "gold_used_for_candidate_generation": False,
        "gold_used_for_candidate_scoring": False,
        "gold_used_for_calibration_label_only": True,
    }

    summary_path = (
        args.output_root
        / "summary.json"
    )

    write_json(
        summary_path,
        summary,
    )

    print()
    print(
        "===== M4-R3 CALIBRATION RESULT ====="
    )

    print(
        "candidate_n =",
        overall_metrics["n"],
    )

    print(
        "candidate_base_rate =",
        overall_metrics[
            "base_rate"
        ],
    )

    print(
        "OOF mean_probability =",
        overall_metrics[
            "mean_probability"
        ],
    )

    print(
        "OOF Brier =",
        overall_metrics[
            "brier"
        ],
    )

    print(
        "OOF LogLoss =",
        overall_metrics[
            "logloss"
        ],
    )

    print(
        "OOF ECE =",
        overall_reliability[
            "ece"
        ],
    )

    print(
        "Base-rate Brier =",
        base_metrics[
            "brier"
        ],
    )

    print(
        "Base-rate LogLoss =",
        base_metrics[
            "logloss"
        ],
    )

    print()
    print(
        "===== LATENCY ms/query ====="
    )

    for name, value in (
        summary[
            "latency_ms_per_query"
        ].items()
    ):
        print(
            name,
            json.dumps(
                value,
                sort_keys=True,
            ),
        )

    print()
    print(
        "RANKING_ORDER_CHANGED_BY_CALIBRATION=False"
    )

    print(
        "TRAIN_FIT_RANKER=PASS"
    )

    print(
        "TRAIN_VAL_FAMILY_OOF_CALIBRATION=PASS"
    )

    print(
        "DEV3000=CLOSED"
    )

    print(
        "TEST=CLOSED"
    )

    print(
        "M4_R3_COMPOSITION_CALIBRATION=PASS"
    )


if __name__ == "__main__":
    main()

