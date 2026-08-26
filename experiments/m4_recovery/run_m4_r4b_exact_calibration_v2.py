from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np


EXPECTED_EXACT_ROWS_SHA = (
    "7cfe23b1aaa8a64f02432938bdb8f84f"
    "bae0d8a4bbcb791105f1488b6fa323dd"
)

EXPECTED_QUERY_COUNT = 94_590
EXPECTED_FAMILY_COUNT = 18_918
EXPECTED_SOURCE = "personal_recovery"
N_FOLDS = 5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_r3(path: Path):
    spec = importlib.util.spec_from_file_location(
        "m4_r3_calibration_reference",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import frozen R3 calibration")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def latency_summary(values):
    x = np.asarray(values, dtype=np.float64)

    if len(x) == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
    }




def fit_platt_robust(
    r3,
    scores: np.ndarray,
    labels: np.ndarray,
):
    """
    Same Platt model and likelihood as frozen R3:

        p = sigmoid(a * raw_score + b), a > 0

    First use the frozen R3 L-BFGS-B implementation.
    If SciPy terminates abnormally, solve the identical
    standardized objective with SLSQP.

    This is a numerical fallback only; it does not change
    the calibration model, labels, folds, or ranking semantics.
    """
    try:
        params = dict(
            r3.fit_platt(
                scores,
                labels,
            )
        )

        params["optimizer"] = "R3_L-BFGS-B"
        params["fallback_used"] = False

        return params

    except RuntimeError as exc:
        if "Platt fit failed" not in str(exc):
            raise

        print(
            "R3_PLATT_NUMERICAL_FALLBACK:",
            str(exc),
            flush=True,
        )

    from scipy.optimize import minimize

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

    z = (x - mean) / std

    base_rate = float(y.mean())

    intercept0 = math.log(
        base_rate / (1.0 - base_rate)
    )

    def objective(theta):
        slope = float(theta[0])
        intercept = float(theta[1])

        eta = slope * z + intercept

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

        residual = r3.sigmoid(eta) - y

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
        method="SLSQP",
        bounds=[
            (1e-8, None),
            (None, None),
        ],
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
            "disp": False,
        },
    )

    if (
        not result.success
        or not np.isfinite(
            result.x
        ).all()
    ):
        raise RuntimeError(
            "Fallback Platt fit failed: "
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

    if (
        not math.isfinite(a)
        or not math.isfinite(b)
        or a <= 0
    ):
        raise RuntimeError(
            "Invalid fallback Platt parameters"
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
        "optimizer": "SLSQP_FALLBACK",
        "fallback_used": True,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--exact-rows",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--r3-script",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    args = ap.parse_args()

    if sha256_file(args.exact_rows) != EXPECTED_EXACT_ROWS_SHA:
        raise RuntimeError(
            "Frozen R4 Exact rows SHA mismatch"
        )

    if (
        args.output_root.exists()
        and any(args.output_root.iterdir())
    ):
        raise RuntimeError(
            "Refusing to overwrite non-empty output"
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    r3 = load_r3(args.r3_script)

    if int(r3.N_FOLDS) != N_FOLDS:
        raise RuntimeError(
            f"R3 fold count changed: {r3.N_FOLDS}"
        )

    print(
        "Calibration implementation: frozen R3 functions",
        flush=True,
    )

    print(
        "Branch population: source=personal_recovery only",
        flush=True,
    )

    print(
        "DEV3000=CLOSED TEST=CLOSED",
        flush=True,
    )

    rows = []

    score_list = []
    label_list = []
    fold_list = []
    gp_list = []

    candidate_locations = []

    family_ids = set()

    query_count = 0
    active_query_count = 0

    for row_index, row in enumerate(
        read_jsonl(args.exact_rows)
    ):
        query_count += 1

        family_id = str(row["family_id"])
        family_ids.add(family_id)

        fold = int(
            r3.family_fold(family_id)
        )

        if not 0 <= fold < N_FOLDS:
            raise RuntimeError(
                f"Invalid fold {fold}"
            )

        gold = str(row["gold"])

        generic_pruned = not bool(
            row["generic_gold_survived_final_beam"]
        )

        personal = []

        for cand in row.get(
            "zero_shot_candidates",
            [],
        ):
            if str(cand.get("source")) != EXPECTED_SOURCE:
                continue

            raw_score = float(
                cand["lambdamart_score"]
            )

            text = str(cand["text"])

            label = int(text == gold)

            obs_index = len(score_list)

            score_list.append(raw_score)
            label_list.append(label)
            fold_list.append(fold)
            gp_list.append(generic_pruned)

            personal.append(
                {
                    "text": text,
                    "exact_raw_score": raw_score,
                    "input_rank": int(cand["rank"]),
                    "label": label,
                    "observation_index": obs_index,
                }
            )

            candidate_locations.append(
                (
                    row_index,
                    len(personal) - 1,
                )
            )

        if personal:
            active_query_count += 1

        rows.append(
            {
                "row_id": str(row["row_id"]),
                "family_id": family_id,
                "multi_token_length": int(
                    row["multi_token_length"]
                ),
                "calibration_fold": fold,
                "generic_beam_pruned": generic_pruned,
                "gold": gold,
                "exact_candidates": personal,
            }
        )

    if query_count != EXPECTED_QUERY_COUNT:
        raise RuntimeError(
            f"Query count mismatch: {query_count}"
        )

    if len(family_ids) != EXPECTED_FAMILY_COUNT:
        raise RuntimeError(
            f"Family count mismatch: {len(family_ids)}"
        )

    scores = np.asarray(
        score_list,
        dtype=np.float64,
    )

    labels = np.asarray(
        label_list,
        dtype=np.int8,
    )

    folds = np.asarray(
        fold_list,
        dtype=np.int8,
    )

    gp_mask = np.asarray(
        gp_list,
        dtype=np.bool_,
    )

    if len(scores) == 0:
        raise RuntimeError(
            "No Exact personal-recovery candidates"
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

    print(
        "active_queries =",
        active_query_count,
    )

    print()
    print(
        "===== 5-FOLD FAMILY OOF PLATT CALIBRATION ====="
    )

    oof_prob = np.full(
        len(scores),
        np.nan,
        dtype=np.float64,
    )

    fold_results = []

    for fold in range(N_FOLDS):
        test_mask = folds == fold
        train_mask = ~test_mask

        if not test_mask.any():
            raise RuntimeError(
                f"Fold {fold} has no test candidates"
            )

        params = fit_platt_robust(r3, 
            scores[train_mask],
            labels[train_mask],
        )

        if float(params["a"]) <= 0:
            raise RuntimeError(
                "Platt slope is not positive"
            )

        test_prob = r3.apply_platt(
            scores[test_mask],
            params,
        )

        oof_prob[test_mask] = test_prob

        fold_metric = r3.binary_metrics(
            labels[test_mask],
            test_prob,
        )

        fold_results.append(
            {
                "fold": fold,
                "n_train": int(
                    train_mask.sum()
                ),
                "n_test": int(
                    test_mask.sum()
                ),
                "train_positives": int(
                    labels[train_mask].sum()
                ),
                "test_positives": int(
                    labels[test_mask].sum()
                ),
                "a": float(
                    params["a"]
                ),
                "b": float(
                    params["b"]
                ),
                "optimizer": str(
                    params.get(
                        "optimizer",
                        "R3_L-BFGS-B",
                    )
                ),
                "fallback_used": bool(
                    params.get(
                        "fallback_used",
                        False,
                    )
                ),
                "test_metrics": fold_metric,
            }
        )

        print(
            f"fold={fold} "
            f"n_train={train_mask.sum()} "
            f"n_test={test_mask.sum()} "
            f"a={params['a']:.8f} "
            f"b={params['b']:.8f}",
            flush=True,
        )

    if not np.isfinite(oof_prob).all():
        raise RuntimeError(
            "OOF calibration incomplete"
        )

    final_params = fit_platt_robust(r3, 
        scores,
        labels,
    )

    if float(final_params["a"]) <= 0:
        raise RuntimeError(
            "Final Platt slope is not positive"
        )

    print()
    print(
        "FINAL_PLATT_A =",
        final_params["a"],
    )

    print(
        "FINAL_PLATT_B =",
        final_params["b"],
    )

    # --------------------------------------------------
    # Candidate-level calibration metrics
    # --------------------------------------------------

    overall_metrics = r3.binary_metrics(
        labels,
        oof_prob,
    )

    overall_reliability = r3.reliability(
        labels,
        oof_prob,
    )

    base_prob = np.full(
        len(labels),
        float(labels.mean()),
        dtype=np.float64,
    )

    base_metrics = r3.binary_metrics(
        labels,
        base_prob,
    )

    def subset_report(mask):
        y = labels[mask]
        p = oof_prob[mask]

        if len(y) == 0:
            return None

        subset_base = np.full(
            len(y),
            float(y.mean()),
            dtype=np.float64,
        )

        return {
            "oof": r3.binary_metrics(
                y,
                p,
            ),
            "oof_reliability": r3.reliability(
                y,
                p,
            ),
            "subset_base_rate_baseline": (
                r3.binary_metrics(
                    y,
                    subset_base,
                )
            ),
        }

    gp_report = subset_report(
        gp_mask
    )

    survived_report = subset_report(
        ~gp_mask
    )

    print()
    print(
        "OOF_MEAN_PROBABILITY =",
        overall_metrics["mean_probability"],
    )

    print(
        "OOF_BRIER =",
        overall_metrics["brier"],
    )

    print(
        "BASE_BRIER =",
        base_metrics["brier"],
    )

    print(
        "OOF_LOGLOSS =",
        overall_metrics["logloss"],
    )

    print(
        "BASE_LOGLOSS =",
        base_metrics["logloss"],
    )

    print(
        "OOF_ECE =",
        overall_reliability["ece"],
    )

    # --------------------------------------------------
    # Ranking preservation
    # All candidates in one query share one fold/calibrator.
    # Positive slope must preserve Exact ordering.
    # --------------------------------------------------

    ranking_changed = False

    for row in rows:
        cands = row["exact_candidates"]

        if len(cands) <= 1:
            continue

        idx = [
            c["observation_index"]
            for c in cands
        ]

        raw = scores[idx]
        prob = oof_prob[idx]

        raw_order = sorted(
            range(len(cands)),
            key=lambda i: (
                -float(raw[i]),
                int(cands[i]["input_rank"]),
                str(cands[i]["text"]),
            ),
        )

        prob_order = sorted(
            range(len(cands)),
            key=lambda i: (
                -float(prob[i]),
                int(cands[i]["input_rank"]),
                str(cands[i]["text"]),
            ),
        )

        if raw_order != prob_order:
            ranking_changed = True
            break

    print(
        "RANKING_ORDER_CHANGED_BY_CALIBRATION =",
        ranking_changed,
    )

    if ranking_changed:
        raise RuntimeError(
            "Calibration changed Exact ranking"
        )

    # --------------------------------------------------
    # Attach OOF probabilities and write fusion-ready rows.
    # Gold/label retained only for post-hoc audit.
    # --------------------------------------------------

    calibrated_path = (
        args.output_root
        / "exact_calibrated_candidates.jsonl"
    )

    with calibrated_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as sink:
        for row in rows:
            out_candidates = []

            for exact_rank, cand in enumerate(
                row["exact_candidates"],
                start=1,
            ):
                obs = cand["observation_index"]

                out_candidates.append(
                    {
                        "text": cand["text"],
                        "exact_rank": exact_rank,
                        "original_final_rank": (
                            cand["input_rank"]
                        ),
                        "exact_raw_score": (
                            cand["exact_raw_score"]
                        ),
                        "exact_oof_probability": float(
                            oof_prob[obs]
                        ),
                        "source": EXPECTED_SOURCE,
                        "is_gold_posthoc": bool(
                            cand["label"]
                        ),
                    }
                )

            out = {
                "schema_version": 1,
                "row_id": row["row_id"],
                "family_id": row["family_id"],
                "multi_token_length": (
                    row["multi_token_length"]
                ),
                "calibration_fold": (
                    row["calibration_fold"]
                ),
                "generic_beam_pruned": (
                    row["generic_beam_pruned"]
                ),
                "gold_posthoc": row["gold"],
                "exact_candidate_count": len(
                    out_candidates
                ),
                "exact_candidates": out_candidates,
                "used_dev3000": False,
                "used_test": False,
            }

            sink.write(
                json.dumps(
                    out,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    # --------------------------------------------------
    # Tiny inference-cost benchmark for final calibrator.
    # --------------------------------------------------

    active_latency_ms = []
    all_latency_ms = []

    a = float(final_params["a"])
    b = float(final_params["b"])

    for row in rows:
        values = [
            float(c["exact_raw_score"])
            for c in row["exact_candidates"]
        ]

        started = time.perf_counter_ns()

        for value in values:
            z = max(
                -40.0,
                min(
                    40.0,
                    a * value + b,
                ),
            )
            _ = 1.0 / (
                1.0 + math.exp(-z)
            )

        elapsed = (
            time.perf_counter_ns()
            - started
        ) / 1_000_000.0

        all_latency_ms.append(
            elapsed
        )

        if values:
            active_latency_ms.append(
                elapsed
            )

    summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment": (
            "m4_r4b_exact_personal_recovery_"
            "family_oof_platt_v1"
        ),
        "scientific_role": (
            "Calibrate frozen Exact LambdaMART "
            "personal-recovery scores onto a "
            "probability scale comparable with "
            "Composition."
        ),
        "branch_definition": (
            'zero_shot_candidates where '
            'source == "personal_recovery" only'
        ),
        "queries": query_count,
        "families": len(family_ids),
        "active_queries": active_query_count,
        "candidate_observations": int(
            len(scores)
        ),
        "positive_observations": int(
            labels.sum()
        ),
        "candidate_gold_base_rate": float(
            labels.mean()
        ),
        "folds": fold_results,
        "final_platt": final_params,
        "overall": {
            "oof": overall_metrics,
            "oof_reliability": (
                overall_reliability
            ),
            "base_rate_baseline": (
                base_metrics
            ),
        },
        "generic_beam_pruned": gp_report,
        "generic_beam_survived": (
            survived_report
        ),
        "ranking_order_changed_by_calibration": (
            ranking_changed
        ),
        "latency_ms_per_query": {
            "all_queries": latency_summary(
                all_latency_ms
            ),
            "active_exact_queries": (
                latency_summary(
                    active_latency_ms
                )
            ),
        },
        "hashes": {
            "exact_rows": (
                sha256_file(
                    args.exact_rows
                )
            ),
            "r3_calibration_reference": (
                sha256_file(
                    args.r3_script
                )
            ),
            "calibrated_rows": (
                sha256_file(
                    calibrated_path
                )
            ),
            "runner": (
                sha256_file(
                    Path(__file__)
                )
            ),
        },
        "calibration_training_population": (
            "Train-Val Exact personal-recovery "
            "candidate observations"
        ),
        "evaluation_protocol": (
            "deterministic 5-fold family-level OOF"
        ),
        "final_calibrator_use": (
            "fit on all Train-Val Exact recovery "
            "observations for future unseen split; "
            "OOF probabilities only for current "
            "Train-Val fusion/evaluation"
        ),
        "gold_used_for_candidate_construction_or_scoring": False,
        "gold_used_for_calibration_label": True,
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

    calibration_path = (
        args.output_root
        / "exact_platt_calibrator.json"
    )

    calibration_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "branch": "exact_personal_recovery",
                "a": float(
                    final_params["a"]
                ),
                "b": float(
                    final_params["b"]
                ),
                "raw_score": (
                    "frozen Full LambdaMART "
                    "lambdamart_score"
                ),
                "model": (
                    "p = sigmoid(a * raw_score + b)"
                ),
                "note": (
                    "For future unseen data only. "
                    "Use OOF probabilities for "
                    "current Train-Val evaluation."
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "CALIBRATED_ROWS_SHA =",
        summary["hashes"]["calibrated_rows"],
    )

    print(
        "RUNNER_SHA =",
        summary["hashes"]["runner"],
    )

    print()
    print(
        "TRAIN_VAL_FAMILY_OOF_EXACT_CALIBRATION=PASS"
    )
    print(
        "DEV3000_TEST_CLOSED=PASS"
    )
    print(
        "M4_R4B_EXACT_CALIBRATION_GATE=PASS"
    )


if __name__ == "__main__":
    main()