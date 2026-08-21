from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

EXPECTED_ROWS = 34416
HISTORY_BUDGET = 5000
EXPECTED_SURFACE_SHA256 = "205c0ba01cd0678d7a4341c503fa2e74cf126a70182ff582687025e4946764b2"
EXPECTED_B2_PREDICTIONS_SHA256 = "7fd8aa158d8cd50bced36b55610f8d932bc65e3aae1dbbd5bd65907ff1707ea7"
EXPECTED_B2_SELECTION_SHA256 = "39e4756e805a525ae1c7a1be577741629daaf1f322f3eea402e1469fe51a00d2"
FROZEN_K = 1
FROZEN_LAMBDA_FREQUENCY = 4.0
FROZEN_LAMBDA_PV1 = 4.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "row_id" not in row:
                raise RuntimeError(f"Missing row_id at {path}:{line_no}")
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def target_of(row: Mapping[str, Any]) -> str:
    value = row.get("target", row.get("gold"))
    if value is None:
        raise RuntimeError(f"No target/gold for {row.get('row_id')}")
    return str(value)


def normalize_history_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row.setdefault("target", target_of(source))
        output.append(row)
    return output


def rank_metrics(ranks: Sequence[int | None]) -> dict[str, float | int]:
    n = len(ranks)
    if not n:
        return {"n": 0, "top1": 0.0, "top3": 0.0, "mrr10": 0.0, "missing10": 0.0}
    return {
        "n": n,
        "top1": sum(rank == 1 for rank in ranks) / n,
        "top3": sum(rank is not None and rank <= 3 for rank in ranks) / n,
        "mrr10": sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / n,
        "missing10": sum(rank is None for rank in ranks) / n,
    }


def author_metrics(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    by_author: dict[str, list[int | None]] = defaultdict(list)
    all_ranks: list[int | None] = []
    for row in rows:
        rank = row.get(key)
        rank = None if rank is None else int(rank)
        by_author[str(row["author"])].append(rank)
        all_ranks.append(rank)
    per_author = {author: rank_metrics(ranks) for author, ranks in sorted(by_author.items())}
    macro = {
        metric: statistics.fmean(float(values[metric]) for values in per_author.values())
        for metric in ("top1", "top3", "mrr10", "missing10")
    } if per_author else {metric: 0.0 for metric in ("top1", "top3", "mrr10", "missing10")}
    return {"micro": rank_metrics(all_ranks), "macro_author": macro, "per_author": per_author}


def transition(rows: Sequence[Mapping[str, Any]], base_key: str, new_key: str) -> dict[str, int]:
    rescue = harm = unchanged_correct = unchanged_wrong = 0
    for row in rows:
        base = row.get(base_key) == 1
        new = row.get(new_key) == 1
        if not base and new:
            rescue += 1
        elif base and not new:
            harm += 1
        elif base and new:
            unchanged_correct += 1
        else:
            unchanged_wrong += 1
    return {
        "rescue": rescue,
        "harm": harm,
        "net": rescue - harm,
        "unchanged_correct": unchanged_correct,
        "unchanged_wrong": unchanged_wrong,
    }


def validate_selection(path: Path) -> dict[str, Any]:
    actual = sha256_file(path)
    if actual != EXPECTED_B2_SELECTION_SHA256:
        raise RuntimeError(
            f"B2 selection SHA mismatch: expected {EXPECTED_B2_SELECTION_SHA256}, got {actual}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("candidate_surface_sha256") != EXPECTED_SURFACE_SHA256:
        raise RuntimeError("B2 selection points to another candidate surface")
    if value.get("dev3000_used") is not False or value.get("test_used") is not False:
        raise RuntimeError("B2 selection crossed Dev3000/Test boundary")
    if int(value["pv1"]["selected_k_pv"]) != FROZEN_K:
        raise RuntimeError("B2 selected K differs from frozen K=1")
    if float(value["frequency"]["selected_lambda_frequency"]) != FROZEN_LAMBDA_FREQUENCY:
        raise RuntimeError("B2 selected Frequency lambda differs from 4.0")
    if float(value["pv1"]["selected_lambda_pv"]) != FROZEN_LAMBDA_PV1:
        raise RuntimeError("B2 selected PV1 lambda differs from 4.0")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B3b: evaluate EM1-R and EM1-R+F on the exact same frozen Initial K=1 candidate surface as PV1."
    )
    parser.add_argument("--initial-train-fit", type=Path, required=True)
    parser.add_argument("--initial-train-val", type=Path, required=True)
    parser.add_argument("--candidate-surface", type=Path, required=True)
    parser.add_argument("--b2-predictions", type=Path, required=True)
    parser.add_argument("--b2-selection", type=Path, required=True)
    parser.add_argument("--exact-scores", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    if sha256_file(args.candidate_surface) != EXPECTED_SURFACE_SHA256:
        raise RuntimeError("Frozen B1 candidate surface SHA256 differs")
    if sha256_file(args.b2_predictions) != EXPECTED_B2_PREDICTIONS_SHA256:
        raise RuntimeError("Frozen B2 predictions SHA256 differs")
    selection = validate_selection(args.b2_selection)

    fit_raw = load_jsonl(args.initial_train_fit)
    val_raw = load_jsonl(args.initial_train_val)
    surface = load_jsonl(args.candidate_surface)
    b2 = load_jsonl(args.b2_predictions)
    exact_rows = load_jsonl(args.exact_scores)
    if len(val_raw) != EXPECTED_ROWS or len(surface) != EXPECTED_ROWS or len(b2) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} Val/surface/B2 rows; got {len(val_raw)}/{len(surface)}/{len(b2)}"
        )

    val_by_id = {str(row["row_id"]): row for row in val_raw}
    surface_by_id = {str(row["row_id"]): row for row in surface}
    b2_by_id = {str(row["row_id"]): row for row in b2}
    exact_by_id = {str(row["row_id"]): row for row in exact_rows}
    if len(val_by_id) != EXPECTED_ROWS or len(surface_by_id) != EXPECTED_ROWS or len(b2_by_id) != EXPECTED_ROWS:
        raise RuntimeError("Duplicate row IDs in standardized B3 inputs")
    if set(val_by_id) != set(surface_by_id) or set(val_by_id) != set(b2_by_id):
        raise RuntimeError("Train-Val/B1/B2 row-ID surfaces differ")

    eligible_ids = {
        str(row["row_id"])
        for row in surface
        if row.get("personal_candidate_texts_top5")
    }
    if set(exact_by_id) != eligible_ids:
        missing = len(eligible_ids - set(exact_by_id))
        stale = len(set(exact_by_id) - eligible_ids)
        raise RuntimeError(
            f"Exact-score cache does not equal frozen K=1 eligible surface: missing={missing} stale={stale}"
        )
    for rid, row in exact_by_id.items():
        if row.get("candidate_surface_sha256") != EXPECTED_SURFACE_SHA256:
            raise RuntimeError(f"Exact-score row belongs to another surface: {rid}")
        if row.get("gold_used_for_candidate_selection") is not False or row.get("gold_used_for_scoring") is not False:
            raise RuntimeError(f"Exact-score row has invalid Gold provenance: {rid}")
        if row.get("dev3000_used") is not False or row.get("test_used") is not False:
            raise RuntimeError(f"Exact-score row crossed Dev3000/Test boundary: {rid}")

    from src.personalisation.context_memory import PredictionQuery
    from src.personalisation.external_memory import (
        rank_of,
        rank_recovery_frequency,
        rank_recovery_only,
        unified_pool,
    )
    from src.personalisation.pilot_a import HistoryIndex

    fit = normalize_history_rows(fit_raw)
    val = normalize_history_rows(val_raw)
    history_index = HistoryIndex(fit + val, HISTORY_BUDGET)

    output_rows: list[dict[str, Any]] = []
    generic_only_r_mismatch = 0
    generic_only_rf_mismatch = 0
    surface_identity_failures = 0

    for number, row in enumerate(val, 1):
        rid = str(row["row_id"])
        srow = surface_by_id[rid]
        b2row = b2_by_id[rid]
        gold = target_of(row)
        if b2row.get("candidate_surface_sha256") != EXPECTED_SURFACE_SHA256:
            raise RuntimeError(f"B2 prediction belongs to another surface: {rid}")
        if int(b2row["selected_k_pv"]) != FROZEN_K:
            raise RuntimeError(f"B2 row K differs: {rid}")
        if float(b2row["selected_lambda_frequency"]) != FROZEN_LAMBDA_FREQUENCY:
            raise RuntimeError(f"B2 row Frequency lambda differs: {rid}")
        if float(b2row["selected_lambda_pv"]) != FROZEN_LAMBDA_PV1:
            raise RuntimeError(f"B2 row PV1 lambda differs: {rid}")

        query = PredictionQuery(
            row_id=rid,
            author=str(row["author"]),
            work_id=str(row["work_id"]),
            chronological_position=int(row["chronological_position"]),
            context=str(row["context"]),
            pinyin=tuple(str(value) for value in row["pinyin_segments"]),
        )
        visible = history_index.visible(query)
        counts = Counter(target_of(history_row) for history_row in visible)
        generic_candidates = [dict(item) for item in srow["generic_candidates"]]
        recovered_scores = exact_by_id[rid]["scores"] if rid in exact_by_id else []
        pool = unified_pool(generic_candidates, recovered_scores, k_recovery=FROZEN_K)

        expected_pool = [str(item["text"]) for item in generic_candidates]
        personal = [str(value) for value in srow.get("personal_candidate_texts_top5", [])[:FROZEN_K]]
        for candidate in personal:
            if candidate not in expected_pool:
                expected_pool.append(candidate)
        actual_pool = [str(item["candidate"]) for item in pool]
        if set(actual_pool) != set(expected_pool):
            surface_identity_failures += 1
            if surface_identity_failures <= 5:
                print(f"Surface mismatch: {rid}", flush=True)

        ranked_r = rank_recovery_only(pool)
        ranked_rf = rank_recovery_frequency(
            pool,
            counts,
            lambda_frequency=FROZEN_LAMBDA_FREQUENCY,
        )
        r_rank = rank_of(ranked_r, gold)
        rf_rank = rank_of(ranked_rf, gold)

        if not personal:
            if r_rank != b2row.get("generic_rank"):
                generic_only_r_mismatch += 1
            if rf_rank != b2row.get("frequency_rank"):
                generic_only_rf_mismatch += 1

        pool_texts = set(actual_pool)
        output_rows.append(
            {
                "schema_version": 1,
                "row_id": rid,
                "anchor_id": srow.get("anchor_id"),
                "author": str(row["author"]),
                "work_id": str(row["work_id"]),
                "chronological_position": int(row["chronological_position"]),
                "gold": gold,
                "history_available": bool(b2row["history_available"]),
                "ambiguous": bool(b2row["ambiguous"]),
                "conflict": bool(b2row["conflict"]),
                "generic_missing": bool(b2row["generic_missing"]),
                "compatible_recoverable_missing": bool(b2row["compatible_recoverable_missing"]),
                "selected_surface_contains_gold": bool(b2row["selected_surface_contains_gold"]),
                "generic_rank": b2row.get("generic_rank"),
                "frequency_rank": b2row.get("frequency_rank"),
                "pv1_rank": b2row.get("pv1_rank"),
                "em1_r_rank": r_rank,
                "em1_rf_rank": rf_rank,
                "em1_pool_contains_gold": gold in pool_texts,
                "personal_candidates": personal,
                "em1_r_candidates": [dict(value) for value in ranked_r],
                "em1_rf_candidates": [dict(value) for value in ranked_rf],
                "candidate_surface_sha256": EXPECTED_SURFACE_SHA256,
                "frozen_k": FROZEN_K,
                "frozen_lambda_frequency": FROZEN_LAMBDA_FREQUENCY,
                "dev3000_used": False,
                "test_used": False,
            }
        )
        if number % 2000 == 0 or number == len(val):
            print(f"B3 same-surface evaluation: {number}/{len(val)}", flush=True)

    if surface_identity_failures:
        raise RuntimeError(f"B3 unified-pool candidate identity failed on {surface_identity_failures} rows")
    if generic_only_r_mismatch or generic_only_rf_mismatch:
        raise RuntimeError(
            "Historical EM1 ranking helpers do not reproduce G/F when no personal candidate is injected: "
            f"R-vs-G={generic_only_r_mismatch}, RF-vs-F={generic_only_rf_mismatch}"
        )

    subsets = {
        "overall": lambda row: True,
        "history_available": lambda row: bool(row["history_available"]),
        "ambiguous": lambda row: bool(row["ambiguous"]),
        "conflict": lambda row: bool(row["conflict"]),
        "generic_missing": lambda row: bool(row["generic_missing"]),
        "compatible_recoverable_missing": lambda row: bool(row["compatible_recoverable_missing"]),
        "selected_surface_contains_gold": lambda row: bool(row["selected_surface_contains_gold"]),
    }
    methods = {
        "G": "generic_rank",
        "F": "frequency_rank",
        "PV1": "pv1_rank",
        "EM1-R": "em1_r_rank",
        "EM1-R+F": "em1_rf_rank",
    }
    metrics: dict[str, Any] = {}
    transitions: dict[str, Any] = {}
    for subset_name, predicate in subsets.items():
        subset = [row for row in output_rows if predicate(row)]
        metrics[subset_name] = {
            method: author_metrics(subset, key)
            for method, key in methods.items()
        }
        transitions[subset_name] = {
            "F_to_PV1": transition(subset, "frequency_rank", "pv1_rank"),
            "F_to_EM1-R+F": transition(subset, "frequency_rank", "em1_rf_rank"),
            "PV1_to_EM1-R+F": transition(subset, "pv1_rank", "em1_rf_rank"),
            "G_to_EM1-R": transition(subset, "generic_rank", "em1_r_rank"),
        }

    selected_recoverable = [row for row in output_rows if row["selected_surface_contains_gold"]]
    recovery = {
        "selected_surface_recoverable_n": len(selected_recoverable),
        "PV1": {
            "top10": sum(row["pv1_rank"] is not None for row in selected_recoverable),
            "top3": sum(row["pv1_rank"] is not None and row["pv1_rank"] <= 3 for row in selected_recoverable),
            "top1": sum(row["pv1_rank"] == 1 for row in selected_recoverable),
        },
        "EM1-R": {
            "top10": sum(row["em1_r_rank"] is not None for row in selected_recoverable),
            "top3": sum(row["em1_r_rank"] is not None and row["em1_r_rank"] <= 3 for row in selected_recoverable),
            "top1": sum(row["em1_r_rank"] == 1 for row in selected_recoverable),
        },
        "EM1-R+F": {
            "top10": sum(row["em1_rf_rank"] is not None for row in selected_recoverable),
            "top3": sum(row["em1_rf_rank"] is not None and row["em1_rf_rank"] <= 3 for row in selected_recoverable),
            "top1": sum(row["em1_rf_rank"] == 1 for row in selected_recoverable),
        },
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_root / "predictions.jsonl"
    write_jsonl(predictions_path, output_rows)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_short_standardized_b3_em1_same_surface_v1",
        "rows": len(output_rows),
        "comparison": "PV1 vs EM1 on identical frozen B1 K=1 personal-candidate surface",
        "scoring_change": "PV1 boundary approximation -> EM1 exact fixed-candidate PinyinGPT score",
        "frozen_k": FROZEN_K,
        "frozen_lambda_frequency": FROZEN_LAMBDA_FREQUENCY,
        "frozen_lambda_pv1": FROZEN_LAMBDA_PV1,
        "parameter_tuning": False,
        "metrics": metrics,
        "transitions": transitions,
        "recovery": recovery,
        "invariants": {
            "candidate_surface_identity_failures": surface_identity_failures,
            "generic_only_em1_r_vs_g_mismatches": generic_only_r_mismatch,
            "generic_only_em1_rf_vs_f_mismatches": generic_only_rf_mismatch,
            "status": "passed",
        },
        "runtime_seconds": time.perf_counter() - started,
        "provenance": {
            "initial_train_fit_sha256": sha256_file(args.initial_train_fit),
            "initial_train_val_sha256": sha256_file(args.initial_train_val),
            "candidate_surface_sha256": EXPECTED_SURFACE_SHA256,
            "b2_predictions_sha256": EXPECTED_B2_PREDICTIONS_SHA256,
            "b2_selection_sha256": EXPECTED_B2_SELECTION_SHA256,
            "exact_scores_sha256": sha256_file(args.exact_scores),
            "predictions_sha256": sha256_file(predictions_path),
        },
        "selection_population": selection["selection_population"],
        "gold_used_for_candidate_selection": False,
        "dev3000_used": False,
        "test_used": False,
    }
    summary_path = args.output_root / "metrics_summary.json"
    write_json(summary_path, summary)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "artifact": "Initial+Short standardized B3 same-surface EM1 Train-Val evaluation",
        "candidate_surface_sha256": EXPECTED_SURFACE_SHA256,
        "b2_predictions_sha256": EXPECTED_B2_PREDICTIONS_SHA256,
        "exact_scores_sha256": summary["provenance"]["exact_scores_sha256"],
        "metrics_summary_sha256": sha256_file(summary_path),
        "predictions_sha256": summary["provenance"]["predictions_sha256"],
        "parameter_tuning": False,
        "dev3000_used": False,
        "test_used": False,
    }
    write_json(args.output_root / "manifest.json", manifest)

    overall = metrics["overall"]
    print("\n=== B3 INITIAL EM1 SAME-SURFACE COMPLETE ===")
    for method in ("G", "F", "PV1", "EM1-R", "EM1-R+F"):
        values = overall[method]
        print(
            f"{method} Macro Top1={values['macro_author']['top1']:.6f} "
            f"Micro Top1={values['micro']['top1']:.6f} "
            f"Top3={values['micro']['top3']:.6f} "
            f"MRR@10={values['micro']['mrr10']:.6f} "
            f"Missing@10={values['micro']['missing10']:.6f}"
        )
    print("F -> PV1:", transitions["overall"]["F_to_PV1"])
    print("F -> EM1-R+F:", transitions["overall"]["F_to_EM1-R+F"])
    print("PV1 -> EM1-R+F:", transitions["overall"]["PV1_to_EM1-R+F"])
    print("Conflict PV1 -> EM1-R+F:", transitions["conflict"]["PV1_to_EM1-R+F"])
    print("Recovery:", recovery)
    print("Predictions SHA256:", summary["provenance"]["predictions_sha256"])
    print("Parameter tuning: false")
    print("Dev3000 used: false")
    print("Test used: false")


if __name__ == "__main__":
    main()
