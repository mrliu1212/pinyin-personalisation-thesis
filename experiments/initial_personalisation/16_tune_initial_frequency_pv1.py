from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

EXPECTED_ROWS = 34416
HISTORY_BUDGET = 5000
EXPECTED_SURFACE_SHA256 = "205c0ba01cd0678d7a4341c503fa2e74cf126a70182ff582687025e4946764b2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
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
            destination.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def generic_score(item: Mapping[str, Any]) -> float:
    for key in ("log_probability", "generic_score", "score"):
        if key in item and item[key] is not None:
            return float(item[key])
    raise RuntimeError(f"Generic candidate has no score: {item}")


def rank_of_candidates(rows: Sequence[Mapping[str, Any]], gold: str) -> int | None:
    for index, row in enumerate(rows, 1):
        text = row.get("candidate", row.get("text"))
        if str(text) == gold:
            return int(row.get("rank", index))
    return None


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B2: standardized Initial Frequency + PV1 tuning on the frozen B1 candidate surface."
    )
    parser.add_argument("--initial-train-fit", type=Path, required=True)
    parser.add_argument("--initial-train-val", type=Path, required=True)
    parser.add_argument("--candidate-surface", type=Path, required=True)
    parser.add_argument("--candidate-surface-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    fit_raw = load_jsonl(args.initial_train_fit)
    val_raw = load_jsonl(args.initial_train_val)
    surface = load_jsonl(args.candidate_surface)

    if len(val_raw) != EXPECTED_ROWS or len(surface) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} Train-Val/surface rows; got {len(val_raw)}/{len(surface)}")

    surface_sha = sha256_file(args.candidate_surface)
    if surface_sha != EXPECTED_SURFACE_SHA256:
        raise RuntimeError(
            "Frozen B1 surface SHA256 differs. Expected "
            f"{EXPECTED_SURFACE_SHA256}, got {surface_sha}"
        )

    manifest = json.loads(args.candidate_surface_manifest.read_text(encoding="utf-8"))
    if manifest.get("surface_sha256") != surface_sha:
        raise RuntimeError("Candidate-surface manifest SHA does not match the surface file")
    if manifest.get("gold_used_for_candidate_selection") is not False:
        raise RuntimeError("Candidate surface was not prediction-visible")
    if manifest.get("dev3000_used") is not False or manifest.get("test_used") is not False:
        raise RuntimeError("B1 surface provenance crossed Dev3000/Test boundary")

    val_by_id = {str(row["row_id"]): row for row in val_raw}
    surface_by_id = {str(row["row_id"]): row for row in surface}
    if len(val_by_id) != EXPECTED_ROWS or len(surface_by_id) != EXPECTED_ROWS:
        raise RuntimeError("Duplicate row_id in Train-Val or B1 surface")
    if set(val_by_id) != set(surface_by_id):
        raise RuntimeError("Train-Val and B1 surface row IDs differ")

    from src.personalisation.context_memory import (
        Candidate,
        PredictionQuery,
        frequency_support,
        normalize_generic_scores,
        rank_frequency,
    )
    from src.personalisation.personal_vocabulary import (
        PV1_K_GRID,
        PV1_LAMBDA_GRID,
        PersonalVocabularyState,
        build_personal_lexicon,
        rank_pv1,
    )
    from src.personalisation.pilot_a import FREQUENCY_LAMBDAS, HistoryIndex

    fit = normalize_history_rows(fit_raw)
    val = normalize_history_rows(val_raw)
    history_index = HistoryIndex(fit + val, HISTORY_BUDGET)

    def query_for(row: Mapping[str, Any]) -> PredictionQuery:
        return PredictionQuery(
            row_id=str(row["row_id"]),
            author=str(row["author"]),
            work_id=str(row["work_id"]),
            chronological_position=int(row["chronological_position"]),
            context=str(row["context"]),
            pinyin=tuple(str(value) for value in row["pinyin_segments"]),
        )

    def generic_for(surface_row: Mapping[str, Any]) -> tuple[Candidate, ...]:
        values = surface_row["generic_candidates"]
        candidates = tuple(
            Candidate(str(item["text"]), int(item["rank"]), generic_score(item))
            for item in values
        )
        if len(candidates) != 10:
            raise RuntimeError(f"Expected Generic Top10 for {surface_row['row_id']}; got {len(candidates)}")
        return candidates

    # Pass 1: re-tune Frequency on Initial Train-Val only.
    frequency_rows: dict[float, list[dict[str, Any]]] = {float(v): [] for v in FREQUENCY_LAMBDAS}
    generic_metric_rows: list[dict[str, Any]] = []
    for number, row in enumerate(val, 1):
        rid = str(row["row_id"])
        srow = surface_by_id[rid]
        query = query_for(row)
        visible = history_index.visible(query)
        candidates = generic_for(srow)
        gold = str(row.get("gold", row.get("target")))
        generic_metric_rows.append({
            "row_id": rid,
            "author": row["author"],
            "generic_rank": rank_of_candidates(
                [{"candidate": c.text, "rank": c.generic_rank} for c in candidates], gold
            ),
        })
        for value in FREQUENCY_LAMBDAS:
            ranked = rank_frequency(query, candidates, visible, lambda_frequency=float(value))
            frequency_rows[float(value)].append({
                "row_id": rid,
                "author": row["author"],
                "rank": rank_of_candidates(ranked, gold),
            })
        if number % 2000 == 0 or number == len(val):
            print(f"B2 Frequency tune: {number}/{len(val)}", flush=True)

    frequency_search: list[dict[str, Any]] = []
    for value in FREQUENCY_LAMBDAS:
        metrics = author_metrics(frequency_rows[float(value)], "rank")
        frequency_search.append({
            "lambda_frequency": float(value),
            "macro_author_top1": metrics["macro_author"]["top1"],
            "micro_top1": metrics["micro"]["top1"],
            "micro_top3": metrics["micro"]["top3"],
            "micro_mrr10": metrics["micro"]["mrr10"],
            "micro_missing10": metrics["micro"]["missing10"],
        })
    selected_frequency = max(
        frequency_search,
        key=lambda row: (float(row["macro_author_top1"]), -float(row["lambda_frequency"])),
    )
    lambda_frequency = float(selected_frequency["lambda_frequency"])

    # Pass 2: tune PV1 over the exact frozen B1 personal candidate prefixes.
    pv1_rows: dict[tuple[int, float], list[dict[str, Any]]] = {
        (int(k), float(v)): [] for k in PV1_K_GRID for v in PV1_LAMBDA_GRID
    }
    for number, row in enumerate(val, 1):
        rid = str(row["row_id"])
        srow = surface_by_id[rid]
        query = query_for(row)
        visible = history_index.visible(query)
        candidates = generic_for(srow)
        gold = str(row.get("gold", row.get("target")))

        frequency_ranked = rank_frequency(
            query, candidates, visible, lambda_frequency=lambda_frequency
        )
        lexicon = build_personal_lexicon(query, visible)
        lexicon_targets = {entry.target for entry in lexicon}
        generic_texts = {candidate.text for candidate in candidates}
        personal_targets = tuple(str(value) for value in srow["personal_candidate_texts_top5"])

        if len(personal_targets) != len(set(personal_targets)):
            raise RuntimeError(f"Duplicate personal target on frozen surface: {rid}")
        if any(target not in lexicon_targets for target in personal_targets):
            raise RuntimeError(f"Frozen B1 target missing from legal history: {rid}")
        if any(target in generic_texts for target in personal_targets):
            raise RuntimeError(f"Frozen B1 personal target overlaps Generic Top10: {rid}")

        _, personal_frequency = frequency_support(personal_targets, visible)
        state = PersonalVocabularyState(
            row_id=rid,
            author=str(row["author"]),
            pinyin=query.pinyin,
            generic_candidates=candidates,
            generic_frequency_ranked=tuple(frequency_ranked),
            lexicon=lexicon,
            personal_only_targets=personal_targets,
            personal_frequency_support=personal_frequency,
            personal_context_support={},
            generic_boundary_score=min(normalize_generic_scores(candidates)),
        )

        for k in PV1_K_GRID:
            for value in PV1_LAMBDA_GRID:
                ranked = rank_pv1(state, k_pv=int(k), lambda_pv=float(value))
                pv1_rows[(int(k), float(value))].append({
                    "row_id": rid,
                    "author": row["author"],
                    "rank": rank_of_candidates(ranked, gold),
                })
        if number % 2000 == 0 or number == len(val):
            print(f"B2 PV1 tune: {number}/{len(val)}", flush=True)

    pv1_search: list[dict[str, Any]] = []
    for k in PV1_K_GRID:
        for value in PV1_LAMBDA_GRID:
            metrics = author_metrics(pv1_rows[(int(k), float(value))], "rank")
            pv1_search.append({
                "k_pv": int(k),
                "lambda_pv": float(value),
                "frozen_lambda_frequency": lambda_frequency,
                "macro_author_top1": metrics["macro_author"]["top1"],
                "micro_top1": metrics["micro"]["top1"],
                "micro_top3": metrics["micro"]["top3"],
                "micro_mrr10": metrics["micro"]["mrr10"],
                "micro_missing10": metrics["micro"]["missing10"],
            })
    selected_pv1 = max(
        pv1_search,
        key=lambda row: (
            float(row["macro_author_top1"]),
            -float(row["lambda_pv"]),
            -int(row["k_pv"]),
        ),
    )
    selected_k = int(selected_pv1["k_pv"])
    selected_lambda_pv = float(selected_pv1["lambda_pv"])

    # Pass 3: materialize selected predictions and diagnostic subsets.
    predictions: list[dict[str, Any]] = []
    for number, row in enumerate(val, 1):
        rid = str(row["row_id"])
        srow = surface_by_id[rid]
        query = query_for(row)
        visible = history_index.visible(query)
        candidates = generic_for(srow)
        gold = str(row.get("gold", row.get("target")))

        frequency_ranked = rank_frequency(
            query, candidates, visible, lambda_frequency=lambda_frequency
        )
        lexicon = build_personal_lexicon(query, visible)
        personal_targets = tuple(str(value) for value in srow["personal_candidate_texts_top5"])
        _, personal_frequency = frequency_support(personal_targets, visible)
        state = PersonalVocabularyState(
            row_id=rid,
            author=str(row["author"]),
            pinyin=query.pinyin,
            generic_candidates=candidates,
            generic_frequency_ranked=tuple(frequency_ranked),
            lexicon=lexicon,
            personal_only_targets=personal_targets,
            personal_frequency_support=personal_frequency,
            personal_context_support={},
            generic_boundary_score=min(normalize_generic_scores(candidates)),
        )
        pv1 = rank_pv1(state, k_pv=selected_k, lambda_pv=selected_lambda_pv)

        g_rank = rank_of_candidates(
            [{"candidate": c.text, "rank": c.generic_rank} for c in candidates], gold
        )
        f_rank = rank_of_candidates(frequency_ranked, gold)
        pv1_rank = rank_of_candidates(pv1, gold)
        selected_personal_targets = list(personal_targets[:selected_k])

        predictions.append({
            "schema_version": 1,
            "row_id": rid,
            "anchor_id": srow.get("anchor_id"),
            "author": str(row["author"]),
            "work_id": str(row["work_id"]),
            "chronological_position": int(row["chronological_position"]),
            "gold": gold,
            "pinyin_segments": list(query.pinyin),
            "generic_rank": g_rank,
            "frequency_rank": f_rank,
            "pv1_rank": pv1_rank,
            "history_available": int(srow["same_initial_history_count"]) > 0,
            "ambiguous": bool(srow["ambiguous"]),
            "conflict": bool(srow["conflict"]),
            "generic_missing": bool(srow["generic_missing"]),
            "compatible_recoverable_missing": bool(srow["gold_compatible_recoverable_any"]),
            "selected_surface_contains_gold": bool(
                srow["generic_missing"] and gold in selected_personal_targets
            ),
            "selected_k_pv": selected_k,
            "selected_lambda_frequency": lambda_frequency,
            "selected_lambda_pv": selected_lambda_pv,
            "personal_candidates": selected_personal_targets,
            "frequency_candidates": [dict(value) for value in frequency_ranked],
            "pv1_candidates": [dict(value) for value in pv1],
            "candidate_surface_sha256": surface_sha,
            "dev3000_used": False,
            "test_used": False,
        })
        if number % 2000 == 0 or number == len(val):
            print(f"B2 selected materialization: {number}/{len(val)}", flush=True)

    subsets = {
        "overall": lambda row: True,
        "history_available": lambda row: bool(row["history_available"]),
        "ambiguous": lambda row: bool(row["ambiguous"]),
        "conflict": lambda row: bool(row["conflict"]),
        "generic_missing": lambda row: bool(row["generic_missing"]),
        "compatible_recoverable_missing": lambda row: bool(row["compatible_recoverable_missing"]),
        "selected_surface_contains_gold": lambda row: bool(row["selected_surface_contains_gold"]),
    }
    metric_keys = {
        "G": "generic_rank",
        "F": "frequency_rank",
        "PV1": "pv1_rank",
    }
    subset_metrics: dict[str, Any] = {}
    for subset_name, predicate in subsets.items():
        subset_rows = [row for row in predictions if predicate(row)]
        subset_metrics[subset_name] = {
            method: author_metrics(subset_rows, key)
            for method, key in metric_keys.items()
        }

    generic_missing = [row for row in predictions if row["generic_missing"]]
    selected_recoverable = [row for row in generic_missing if row["selected_surface_contains_gold"]]
    recovery = {
        "generic_missing_n": len(generic_missing),
        "selected_k_surface_recoverable_n": len(selected_recoverable),
        "selected_k_surface_recoverable_given_missing": (
            len(selected_recoverable) / len(generic_missing) if generic_missing else None
        ),
        "pv1_recovered_to_top10_n": sum(row["pv1_rank"] is not None for row in selected_recoverable),
        "pv1_recovered_to_top3_n": sum(
            row["pv1_rank"] is not None and row["pv1_rank"] <= 3 for row in selected_recoverable
        ),
        "pv1_recovered_to_top1_n": sum(row["pv1_rank"] == 1 for row in selected_recoverable),
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_root / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    write_csv(args.output_root / "frequency_hyperparameter_search.csv", frequency_search)
    write_csv(args.output_root / "pv1_hyperparameter_search.csv", pv1_search)

    selection = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_short_standardized_b2_frequency_pv1_v1",
        "selection_population": "standardized Clean3 Train-Val only",
        "selection_metric": "Macro-author Top1",
        "frequency": {
            "grid": [float(value) for value in FREQUENCY_LAMBDAS],
            "selected_lambda_frequency": lambda_frequency,
            "tie_break": "lower lambda_frequency",
        },
        "pv1": {
            "k_grid": [int(value) for value in PV1_K_GRID],
            "lambda_pv_grid": [float(value) for value in PV1_LAMBDA_GRID],
            "selected_k_pv": selected_k,
            "selected_lambda_pv": selected_lambda_pv,
            "frozen_lambda_frequency": lambda_frequency,
            "tie_break": "lower lambda_pv, then lower k_pv",
        },
        "candidate_surface_sha256": surface_sha,
        "candidate_surface_used_gold": False,
        "dev3000_used": False,
        "test_used": False,
    }
    selection_path = args.output_root / "selected_hyperparameters.json"
    write_json(selection_path, selection)

    metrics_summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_short_standardized_b2_frequency_pv1_v1",
        "rows": len(predictions),
        "selected_hyperparameters": selection,
        "metrics": subset_metrics,
        "transitions": {
            "G_to_F": transition(predictions, "generic_rank", "frequency_rank"),
            "G_to_PV1": transition(predictions, "generic_rank", "pv1_rank"),
            "F_to_PV1": transition(predictions, "frequency_rank", "pv1_rank"),
        },
        "recovery": recovery,
        "runtime_seconds": time.perf_counter() - started,
        "provenance": {
            "initial_train_fit": str(args.initial_train_fit.resolve()),
            "initial_train_fit_sha256": sha256_file(args.initial_train_fit),
            "initial_train_val": str(args.initial_train_val.resolve()),
            "initial_train_val_sha256": sha256_file(args.initial_train_val),
            "candidate_surface": str(args.candidate_surface.resolve()),
            "candidate_surface_sha256": surface_sha,
            "candidate_surface_manifest": str(args.candidate_surface_manifest.resolve()),
            "candidate_surface_manifest_sha256": sha256_file(args.candidate_surface_manifest),
            "predictions": str(predictions_path.resolve()),
            "predictions_sha256": sha256_file(predictions_path),
            "selection_sha256": sha256_file(selection_path),
        },
        "dev3000_used": False,
        "test_used": False,
    }
    metrics_path = args.output_root / "metrics_summary.json"
    write_json(metrics_path, metrics_summary)

    run_manifest = {
        "schema_version": 1,
        "artifact": "Initial+Short standardized B2 Frequency + PV1 Train-Val selection",
        "status": "complete",
        "candidate_surface_sha256": surface_sha,
        "selected_hyperparameters_sha256": sha256_file(selection_path),
        "metrics_summary_sha256": sha256_file(metrics_path),
        "predictions_sha256": sha256_file(predictions_path),
        "history_budget": HISTORY_BUDGET,
        "history_semantics": "same author -> strictly prior -> latest 5000 RAW -> exact Initial match",
        "gold_used_for_candidate_selection": False,
        "dev3000_used": False,
        "test_used": False,
    }
    write_json(args.output_root / "manifest.json", run_manifest)

    overall = subset_metrics["overall"]
    print("\n=== B2 INITIAL FREQUENCY + PV1 COMPLETE ===")
    print("Rows:", len(predictions))
    print("B1 surface SHA256:", surface_sha)
    print("Selected Frequency lambda:", lambda_frequency)
    print("Selected PV1 K:", selected_k)
    print("Selected PV1 lambda:", selected_lambda_pv)
    for method in ("G", "F", "PV1"):
        print(
            f"{method} Macro Top1={overall[method]['macro_author']['top1']:.6f} "
            f"Micro Top1={overall[method]['micro']['top1']:.6f} "
            f"Top3={overall[method]['micro']['top3']:.6f} "
            f"MRR@10={overall[method]['micro']['mrr10']:.6f} "
            f"Missing@10={overall[method]['micro']['missing10']:.6f}"
        )
    print("F -> PV1 rescue/harm/net:", metrics_summary["transitions"]["F_to_PV1"])
    print("Selected-K recoverable missing:", recovery["selected_k_surface_recoverable_n"])
    print("PV1 recovered Top10/Top3/Top1:", recovery["pv1_recovered_to_top10_n"], recovery["pv1_recovered_to_top3_n"], recovery["pv1_recovered_to_top1_n"])
    print("Predictions SHA256:", metrics_summary["provenance"]["predictions_sha256"])
    print("Dev3000 used: false")
    print("Test used: false")


if __name__ == "__main__":
    main()
