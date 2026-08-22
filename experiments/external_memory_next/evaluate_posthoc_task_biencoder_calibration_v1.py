"""Evaluate frozen Initial/Full post-hoc context and recovery calibration grids."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

from experiments.context_comparison import run_full_transfer_initial_final_v1 as base
from src.personalisation.posthoc_context_calibration import (
    merge_personal_recovery,
    metric_summary,
    rank_of,
    recovery_summary,
    rerank_fixed_surface,
    restrict_and_normalize,
    selection_key,
    transition_counts,
)
from src.personalisation.task_specific_biencoder import refuse_closed_path, sha256_file, write_json, write_jsonl


INITIAL_BASES = ("K5+Entropy", "4P+4CS+2E", "6P+2CS+.25E")
INITIAL_N = (0.0, .25, .5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0)
INITIAL_E = (0.0, .25, .5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0)
FULL_N = (0.0, 2.0, 4.0, 6.0, 8.0)
FULL_E = (0.0, 2.0, 4.0, 6.0, 8.0)
ALPHA_F = (0.0, .25, .5, .75, 1.0)
EXPECTED_INITIAL_GENERIC = {
    "K5+Entropy": (6.0, 8.0, .436767),
    "4P+4CS+2E": (4.0, 6.0, .437058),
    "6P+2CS+.25E": (4.0, 6.0, .436477),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def index_rows(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    result = {str(row["row_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate row IDs in {label}")
    return result


def candidate_text(item: Mapping[str, Any]) -> str:
    return str(item["candidate"])


def softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    maximum = max(map(float, values))
    exp = [math.exp(float(value) - maximum) for value in values]
    total = sum(exp)
    return [value / total for value in exp]


def ranked_candidates(candidates: Sequence[str], supports: Sequence[tuple[float, Mapping[str, float]]]) -> list[str]:
    scores = {candidate: sum(float(weight) * float(support[candidate]) for weight, support in supports) for candidate in candidates}
    return sorted(map(str, candidates), key=lambda candidate: (-scores[candidate], candidates.index(candidate), candidate))


def subset_bundle(rows: Sequence[Mapping[str, Any]], rank_key: str) -> dict[str, Any]:
    definitions = {
        "overall": list(rows),
        "ambiguous": [row for row in rows if row["ambiguous"]],
        "conflict": [row for row in rows if row["conflict"]],
        "recoverable": [row for row in rows if row["generic_missing"] and row["gold_in_personal_k5"]],
        "generic_missing": [row for row in rows if row["generic_missing"]],
        "generic_covered": [row for row in rows if not row["generic_missing"]],
    }
    return {name: metric_summary(values, rank_key) for name, values in definitions.items()}


def evaluate_config(
    rows: Sequence[Mapping[str, Any]],
    *,
    surface_key: str,
    ngram_key: str,
    encoder_key: str,
    lambda_n: float,
    lambda_e: float,
    rank_key: str,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        surface = row[surface_key]
        candidates = [candidate_text(item) for item in surface]
        ngram = row[ngram_key]
        encoder = restrict_and_normalize(row["raw_support"][encoder_key], candidates) if candidates else {}
        ranking = rerank_fixed_surface(surface, [(lambda_n, ngram), (lambda_e, encoder)]) if candidates else []
        copied = dict(row)
        copied[rank_key] = rank_of(ranking, str(row["gold"]))
        copied[rank_key.replace("_rank", "_top10")] = [candidate_text(item) for item in ranking]
        output.append(copied)
    return output


def tune_family(
    rows: Sequence[Mapping[str, Any]],
    *,
    surface_key: str,
    ngram_key: str,
    encoder_key: str,
    n_grid: Sequence[float],
    e_grid: Sequence[float],
    family: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if family == "ngram":
        configs = [(value, 0.0) for value in n_grid]
    elif family == "encoder":
        configs = [(0.0, value) for value in e_grid]
    elif family == "joint":
        configs = [(n, e) for n in n_grid for e in e_grid]
    else:
        raise ValueError(family)
    grid = []
    for lambda_n, lambda_e in configs:
        evaluated = evaluate_config(
            rows, surface_key=surface_key, ngram_key=ngram_key, encoder_key=encoder_key,
            lambda_n=lambda_n, lambda_e=lambda_e, rank_key="grid_rank",
        )
        grid.append({"lambda_n": lambda_n, "lambda_e": lambda_e, "metrics": metric_summary(evaluated, "grid_rank")})
    selected = max(grid, key=selection_key)
    return selected, grid


def fixed_track(
    track: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    base_name: str,
    n_grid: Sequence[float],
    e_grid: Sequence[float],
    baseline_key: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    surface_key = "surface"
    ngram_key = "ngram_support"
    methods: dict[str, Any] = {}
    grids: list[dict[str, Any]] = []
    selected_predictions: dict[str, list[dict[str, Any]]] = {}
    base_rows = []
    for row in rows:
        copied = dict(row)
        copied["Base_rank"] = rank_of(row[surface_key], str(row["gold"]))
        base_rows.append(copied)
    methods[base_name] = {
        "lambda_n": 0.0,
        "lambda_e": 0.0,
        "metrics": subset_bundle(base_rows, "Base_rank"),
        "transition_from_frozen": transition_counts(base_rows, baseline_key, "Base_rank"),
    }
    specs = (
        ("NGramRecency", "generic_recency", "ngram"),
        ("GenericBGE", "generic_plain", "encoder"),
        ("GenericBGERecency", "generic_recency", "encoder"),
        ("TaskBiEncoder", "task_plain", "encoder"),
        ("TaskBiEncoderRecency", "task_recency", "encoder"),
        ("NGram+GenericBGERecency", "generic_recency", "joint"),
        ("NGram+TaskBiEncoderRecency", "task_recency", "joint"),
    )
    for method, encoder_key, family in specs:
        selected, family_grid = tune_family(
            rows, surface_key=surface_key, ngram_key=ngram_key, encoder_key=encoder_key,
            n_grid=n_grid, e_grid=e_grid, family=family,
        )
        rank_key = f"{method}_rank"
        evaluated = evaluate_config(
            rows, surface_key=surface_key, ngram_key=ngram_key, encoder_key=encoder_key,
            lambda_n=float(selected["lambda_n"]), lambda_e=float(selected["lambda_e"]), rank_key=rank_key,
        )
        methods[method] = {
            "lambda_n": selected["lambda_n"],
            "lambda_e": selected["lambda_e"],
            "metrics": subset_bundle(evaluated, rank_key),
            "transition_from_frozen": transition_counts(evaluated, baseline_key, rank_key),
        }
        grids.extend({"track": track, "base": base_name, "method": method, **item} for item in family_grid)
        if method in ("NGram+GenericBGERecency", "NGram+TaskBiEncoderRecency"):
            selected_predictions[method] = evaluated
    return methods, grids, selected_predictions


def initial_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    val_rows = read_jsonl(args.initial_val)
    features = index_rows(read_jsonl(args.initial_stage1), "Initial Stage1")
    ngram = index_rows(read_jsonl(args.initial_ngram), "Initial NGram")
    support = index_rows(read_jsonl(args.initial_support), "Initial support")
    frequency = index_rows(read_jsonl(args.initial_frequency), "Initial Frequency")
    q8 = index_rows(read_jsonl(args.initial_q8), "Initial Q8")
    order = [str(row["row_id"]) for row in val_rows]
    if not all(set(order) == set(values) for values in (features, ngram, support, frequency)):
        raise ValueError("Initial population mismatch")
    rows = []
    for val in val_rows:
        row_id = str(val["row_id"])
        feature = features[row_id]
        rows.append({
            "row_id": row_id,
            "author": str(val["author"]),
            "gold": str(feature["gold"]),
            "ambiguous": bool(val.get("ambiguous")),
            "conflict": bool(val.get("conflict")),
            "generic_missing": bool(feature["generic_missing"]),
            "gold_in_personal_k5": bool(feature["gold_in_personal_k5"]),
            "personal_k5": list(map(str, feature["personal_k5"])),
            "p_ng": {str(k): float(v) for k, v in feature["interpolated_ngram_support"].items()},
            "frequency_support": {str(k): float(v) for k, v in feature["personal_frequency_support"].items()},
            "generic_candidates": [dict(item, source="generic") for item in frequency[row_id]["frequency_candidates"]],
            "bases": feature["bases"],
            "ngram_bases": ngram[row_id]["bases"],
            "raw_support": support[row_id]["raw_support"],
            "q8": q8.get(row_id),
        })
    return rows, {"val_rows": val_rows}


def full_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fit_rows = read_jsonl(args.full_fit)
    val_rows = read_jsonl(args.full_val)
    features = index_rows(read_jsonl(args.full_stage1), "Full Stage1")
    stage2 = index_rows(read_jsonl(args.full_stage2), "Full Stage2")
    support = index_rows(read_jsonl(args.full_support), "Full support")
    frozen = index_rows(read_jsonl(args.full_frozen), "Full frozen")
    q8 = index_rows(read_jsonl(args.full_q8), "Full Q8")
    history = base.CausalHistoryIndex([*fit_rows, *val_rows])
    order = [str(row["row_id"]) for row in val_rows]
    if not all(order == [str(row["row_id"]) for row in read_jsonl(path)] for path in (args.full_stage1, args.full_stage2, args.full_support, args.full_frozen)):
        raise ValueError("Full row/order mismatch")
    rows = []
    for val in val_rows:
        row_id = str(val["row_id"])
        feature = features[row_id]
        srow = stage2[row_id]
        frow = frozen[row_id]
        visible = history.visible_same_pinyin(
            author=str(val["author"]),
            position=int(val["chronological_position"]),
            pinyin=base.pinyin_of(val),
        )
        counts: dict[str, int] = defaultdict(int)
        for item in visible:
            counts[str(item.record.target)] += 1
        frequency_raw = {candidate: math.log1p(counts[candidate]) for candidate in feature["personal_k5"]}
        frequency_max = max(frequency_raw.values(), default=0.0)
        frequency = {
            candidate: (value / frequency_max if frequency_max else 0.0)
            for candidate, value in frequency_raw.items()
        }
        rows.append({
            "row_id": row_id,
            "author": str(val["author"]),
            "gold": str(frow["gold"]),
            "ambiguous": bool(frow["ambiguous"]),
            "conflict": bool(frow["conflict"]),
            "generic_missing": bool(feature["generic_missing"]),
            "gold_in_personal_k5": bool(feature["gold_in_personal_k5"]),
            "personal_k5": list(map(str, feature["personal_k5"])),
            "p_ng": {str(k): float(v) for k, v in feature["p_ng"].items()},
            "frequency_support": frequency,
            "generic_candidates": feature["generic_frequency_candidates"],
            "surface": srow["retuned_stage1_candidates"],
            "ngram_support": srow["retuned_ngram_support"],
            "raw_support": support[row_id]["raw_support"],
            "Frozen_rank": frow["RetunedFinal_rank"],
            "q8": q8.get(row_id),
        })
    return rows, {"fit_rows": fit_rows, "val_rows": val_rows}


def prepare_initial_base(rows: Sequence[Mapping[str, Any]], name: str, baseline_ranks: Mapping[str, int | None]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        copied = dict(row)
        copied["surface"] = row["bases"][name]["candidates"]
        copied["ngram_support"] = row["ngram_bases"][name]["support"]
        copied["Frozen_rank"] = baseline_ranks[str(row["row_id"])]
        output.append(copied)
    return output


def candidate_only(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    recoverable = [row for row in rows if row["generic_missing"] and row["gold_in_personal_k5"]]
    if not recoverable:
        return {"recoverable_n": 0}
    evaluated = []
    for row in recoverable:
        candidates = row["personal_k5"]
        supports = {
            "Frequency": row["frequency_support"],
            "P_NG": row["p_ng"],
            "GenericBGE": restrict_and_normalize(row["raw_support"]["generic_plain"], candidates),
            "GenericBGERecency": restrict_and_normalize(row["raw_support"]["generic_recency"], candidates),
            "TaskBiEncoder": restrict_and_normalize(row["raw_support"]["task_plain"], candidates),
            "TaskBiEncoderRecency": restrict_and_normalize(row["raw_support"]["task_recency"], candidates),
        }
        copied = dict(row)
        for method, support in supports.items():
            copied[f"{method}_rank"] = rank_of(ranked_candidates(candidates, [(1.0, support)]), str(row["gold"]))
        if row["q8"] is not None:
            q8_values = softmax([float(item["fixed_mean_log_probability"]) for item in row["q8"]["scores"]])
            q8_support = dict(zip(candidates, q8_values))
            q8_frequency = restrict_and_normalize(row["frequency_support"], candidates)
            copied["Q8_rank"] = rank_of(ranked_candidates(candidates, [(1.0, q8_support)]), str(row["gold"]))
            for alpha in ALPHA_F:
                copied[f"Q8F_{alpha:g}_rank"] = rank_of(
                    ranked_candidates(candidates, [(1.0 - alpha, q8_support), (alpha, q8_frequency)]), str(row["gold"])
                )
        evaluated.append(copied)

    def summaries(population: Sequence[Mapping[str, Any]], rank_key: str) -> dict[str, Any]:
        return {"ranking": metric_summary(population, rank_key), "recovery": recovery_summary(population, rank_key)}

    populations = {
        "generic_missing_recoverable": evaluated,
        "generic_missing_recoverable_k2plus": [row for row in evaluated if len(row["personal_k5"]) >= 2],
    }
    methods = {}
    keys = ["Frequency", "P_NG", "GenericBGE", "GenericBGERecency", "TaskBiEncoder", "TaskBiEncoderRecency", "Q8"]
    for key in keys:
        rank_key = f"{key}_rank"
        if rank_key in evaluated[0]:
            methods[key] = {
                name: summaries(population, rank_key)
                for name, population in populations.items()
            }
    alpha_rows = []
    for alpha in ALPHA_F:
        key = f"Q8F_{alpha:g}_rank"
        alpha_rows.append({
            "alpha_f": alpha,
            "generic_missing_recoverable": metric_summary(evaluated, key),
            "generic_missing_recoverable_k2plus": metric_summary(populations["generic_missing_recoverable_k2plus"], key),
        })
    selected_q8f = max(
        alpha_rows,
        key=lambda row: (
            row["generic_missing_recoverable_k2plus"]["macro_author_top1"],
            row["generic_missing_recoverable_k2plus"]["mrr_at_10"],
            -abs(row["alpha_f"] - .75),
        ),
    )
    methods["Q8+F"] = {
        "alpha_f": selected_q8f["alpha_f"],
        **{
            name: summaries(population, f"Q8F_{selected_q8f['alpha_f']:g}_rank")
            for name, population in populations.items()
        },
    }

    for encoder in ("generic_recency", "task_recency"):
        name = "P_NG+" + ("GenericBGERecency" if encoder.startswith("generic") else "TaskBiEncoderRecency")
        grid = []
        n_grid, e_grid = (INITIAL_N, INITIAL_E) if len(rows) == 34_416 and len(recoverable) == 4_910 else (FULL_N, FULL_E)
        for ln in n_grid:
            for le in e_grid:
                key = f"tmp_{name}_{ln}_{le}"
                for item in evaluated:
                    candidates = item["personal_k5"]
                    support = restrict_and_normalize(item["raw_support"][encoder], candidates)
                    item[key] = rank_of(ranked_candidates(candidates, [(ln, item["p_ng"]), (le, support)]), str(item["gold"]))
                metrics = metric_summary(evaluated, key)
                grid.append({"lambda_n": ln, "lambda_e": le, "metrics": metrics})
        selected = max(grid, key=selection_key)
        key = f"tmp_{name}_{selected['lambda_n']}_{selected['lambda_e']}"
        methods[name] = {
            "lambda_n": selected["lambda_n"],
            "lambda_e": selected["lambda_e"],
            **{population_name: summaries(population, key) for population_name, population in populations.items()},
        }
    return {
        "populations": {name: len(population) for name, population in populations.items()},
        "methods": methods,
        "q8f_grid": alpha_rows,
    }


def recovery_grid(rows: Sequence[Mapping[str, Any]], encoder_key: str, n_grid: Sequence[float], e_grid: Sequence[float]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    grid = []
    selected_rows: list[dict[str, Any]] = []
    for ln in n_grid:
        for le in e_grid:
            evaluated = []
            for row in rows:
                personal = row["personal_k5"]
                encoder = restrict_and_normalize(row["raw_support"][encoder_key], personal) if personal else {}
                generic = row["generic_candidates"]
                boundary = min((float(item["normalized_generic_score"]) for item in generic), default=0.0)
                ranking = merge_personal_recovery(
                    generic_candidates=generic,
                    personal_k5=personal,
                    personal_supports=[(ln, row["p_ng"]), (le, encoder)],
                    boundary=boundary,
                    tiebreak_support=row["p_ng"],
                ) if generic else []
                copied = dict(row)
                copied["grid_rank"] = rank_of(ranking, str(row["gold"]))
                copied["grid_surface"] = ranking
                evaluated.append(copied)
            grid.append({"lambda_n": ln, "lambda_e": le, "metrics": metric_summary(evaluated, "grid_rank")})
    selected = max(grid, key=selection_key)
    for row in rows:
        personal = row["personal_k5"]
        encoder = restrict_and_normalize(row["raw_support"][encoder_key], personal) if personal else {}
        generic = row["generic_candidates"]
        boundary = min((float(item["normalized_generic_score"]) for item in generic), default=0.0)
        ranking = merge_personal_recovery(
            generic_candidates=generic,
            personal_k5=personal,
            personal_supports=[(selected["lambda_n"], row["p_ng"]), (selected["lambda_e"], encoder)],
            boundary=boundary,
            tiebreak_support=row["p_ng"],
        ) if generic else []
        copied = dict(row)
        copied["Recovery_rank"] = rank_of(ranking, str(row["gold"]))
        copied["Recovery_surface"] = ranking
        selected_rows.append(copied)
    return selected, grid, selected_rows


def downstream_after_recovery(rows: Sequence[Mapping[str, Any]], *, encoder_key: str, lambda_n: float, lambda_e: float, fit_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    history = base.CausalHistoryIndex([*fit_rows, *val_rows])
    val = index_rows(val_rows, "val")
    output = []
    for row in rows:
        row_id = str(row["row_id"])
        surface = row["Recovery_surface"]
        candidates = [candidate_text(item) for item in surface]
        if candidates:
            vrow = val[row_id]
            visible = history.visible_same_pinyin(author=str(vrow["author"]), position=int(vrow["chronological_position"]), pinyin=base.pinyin_of(vrow))
            ngram, _effective, _matched = base.ngram_recency_support(query_context=base.context_of(vrow), candidates=candidates, visible=visible)
            encoder = restrict_and_normalize(row["raw_support"][encoder_key], candidates)
            ranking = rerank_fixed_surface(surface, [(lambda_n, ngram), (lambda_e, encoder)])
        else:
            ranking = []
        copied = dict(row)
        copied["RecoveryContext_rank"] = rank_of(ranking, str(row["gold"]))
        copied["RecoveryContext_top10"] = [candidate_text(item) for item in ranking]
        output.append(copied)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-fit", type=Path, required=True)
    parser.add_argument("--initial-val", type=Path, required=True)
    parser.add_argument("--initial-stage1", type=Path, required=True)
    parser.add_argument("--initial-ngram", type=Path, required=True)
    parser.add_argument("--initial-support", type=Path, required=True)
    parser.add_argument("--initial-frequency", type=Path, required=True)
    parser.add_argument("--initial-q8", type=Path, required=True)
    parser.add_argument("--full-fit", type=Path, required=True)
    parser.add_argument("--full-val", type=Path, required=True)
    parser.add_argument("--full-stage1", type=Path, required=True)
    parser.add_argument("--full-stage2", type=Path, required=True)
    parser.add_argument("--full-support", type=Path, required=True)
    parser.add_argument("--full-frozen", type=Path, required=True)
    parser.add_argument("--full-q8", type=Path, required=True)
    parser.add_argument("--intrinsic-result", type=Path, required=True)
    parser.add_argument("--lambdamart-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for value in vars(args).values():
        if isinstance(value, Path):
            refuse_closed_path(value)
    started = time.perf_counter()
    initial, initial_data = initial_rows(args)
    full, full_data = full_rows(args)
    if len(initial) != 34_416 or len(full) != 34_416:
        raise ValueError("track population changed")

    # Primary frozen Initial comparator: balanced N=4/B=6.
    balanced = prepare_initial_base(initial, "4P+4CS+2E", {str(row["row_id"]): None for row in initial})
    baseline_eval = evaluate_config(
        balanced, surface_key="surface", ngram_key="ngram_support", encoder_key="generic_recency",
        lambda_n=4.0, lambda_e=6.0, rank_key="Frozen_rank",
    )
    initial_baseline = {str(row["row_id"]): row["Frozen_rank"] for row in baseline_eval}
    initial_recovery_input = [
        {**row, "Frozen_rank": initial_baseline[str(row["row_id"])]}
        for row in initial
    ]
    initial_fixed = {}
    all_grids = []
    initial_selected = {}
    for name in INITIAL_BASES:
        prepared = prepare_initial_base(initial, name, initial_baseline)
        methods, grids, selected = fixed_track(
            "initial", prepared, base_name=name, n_grid=INITIAL_N, e_grid=INITIAL_E, baseline_key="Frozen_rank",
        )
        expected_n, expected_e, expected_macro = EXPECTED_INITIAL_GENERIC[name]
        generic = methods["NGram+GenericBGERecency"]
        if (generic["lambda_n"], generic["lambda_e"]) != (expected_n, expected_e) or abs(generic["metrics"]["overall"]["macro_author_top1"] - expected_macro) > 1e-6:
            raise RuntimeError(f"Initial Generic-BGE comparator failed for {name}: {generic}")
        initial_fixed[name] = methods
        all_grids.extend(grids)
        initial_selected[name] = selected

    full_methods, full_grids, full_selected = fixed_track(
        "full", full, base_name="RetunedStage1", n_grid=FULL_N, e_grid=FULL_E, baseline_key="Frozen_rank",
    )
    all_grids.extend(full_grids)
    generic_full = full_methods["NGram+GenericBGERecency"]
    if (generic_full["lambda_n"], generic_full["lambda_e"]) != (6.0, 6.0) or abs(generic_full["metrics"]["overall"]["macro_author_top1"] - .7960049265502147) > 1e-12:
        raise RuntimeError("Full Generic-BGE comparator failed")

    initial_candidate = candidate_only(initial)
    full_candidate = candidate_only(full)
    initial_q8 = initial_candidate["methods"]["Q8"]["generic_missing_recoverable_k2plus"]["ranking"]
    initial_q8f = initial_candidate["methods"]["Q8+F"]["generic_missing_recoverable_k2plus"]["ranking"]
    if initial_candidate["populations"]["generic_missing_recoverable_k2plus"] != 4_471:
        raise RuntimeError("historical Initial Q8 K2+ population changed")
    if abs(initial_q8["macro_author_top1"] - .6365306668058097) > 1e-12:
        raise RuntimeError("historical Initial Q8 result failed reconstruction")
    if initial_candidate["methods"]["Q8+F"]["alpha_f"] != .75 or abs(initial_q8f["macro_author_top1"] - .6691641408627556) > 1e-12:
        raise RuntimeError("historical Initial Q8+F result failed reconstruction")

    initial_recovery = {}
    full_recovery = {}
    recovery_predictions = []
    for label, key in (("generic", "generic_recency"), ("task", "task_recency")):
        selected, grid, rows = recovery_grid(initial_recovery_input, key, INITIAL_N, INITIAL_E)
        fixed = initial_fixed["4P+4CS+2E"][f"NGram+{'GenericBGERecency' if label == 'generic' else 'TaskBiEncoderRecency'}"]
        downstream = downstream_after_recovery(
            rows, encoder_key=key, lambda_n=float(fixed["lambda_n"]), lambda_e=float(fixed["lambda_e"]),
            fit_rows=read_jsonl(args.initial_fit), val_rows=initial_data["val_rows"],
        )
        initial_recovery[label] = {
            "selected": selected,
            "stage1_metrics": subset_bundle(rows, "Recovery_rank"),
            "downstream_fixed_context": {"lambda_n": fixed["lambda_n"], "lambda_e": fixed["lambda_e"], "metrics": subset_bundle(downstream, "RecoveryContext_rank"), "transition_from_frozen": transition_counts(downstream, "Frozen_rank", "RecoveryContext_rank")},
        }
        all_grids.extend({"track": "initial", "base": "PersonalK5Recovery", "method": label, **item} for item in grid)
        recovery_predictions.append(("initial", label, downstream))

        selected_f, grid_f, rows_f = recovery_grid(full, key, FULL_N, FULL_E)
        fixed_f = full_methods[f"NGram+{'GenericBGERecency' if label == 'generic' else 'TaskBiEncoderRecency'}"]
        downstream_f = downstream_after_recovery(
            rows_f, encoder_key=key, lambda_n=float(fixed_f["lambda_n"]), lambda_e=float(fixed_f["lambda_e"]),
            fit_rows=full_data["fit_rows"], val_rows=full_data["val_rows"],
        )
        full_recovery[label] = {
            "selected": selected_f,
            "stage1_metrics": subset_bundle(rows_f, "Recovery_rank"),
            "downstream_fixed_context": {"lambda_n": fixed_f["lambda_n"], "lambda_e": fixed_f["lambda_e"], "metrics": subset_bundle(downstream_f, "RecoveryContext_rank"), "transition_from_frozen": transition_counts(downstream_f, "Frozen_rank", "RecoveryContext_rank")},
        }
        all_grids.extend({"track": "full", "base": "PersonalK5Recovery", "method": label, **item} for item in grid_f)
        recovery_predictions.append(("full", label, downstream_f))

    intrinsic = json.loads(args.intrinsic_result.read_text(encoding="utf-8"))["intrinsic"]
    lambdamart = json.loads(args.lambdamart_result.read_text(encoding="utf-8"))["metrics"]
    result = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "posthoc_task_biencoder_calibration_v1",
        "protocol": {"initial_lambda_n": INITIAL_N, "initial_lambda_e": INITIAL_E, "full_lambda_n": FULL_N, "full_lambda_e": FULL_E, "selection": "Macro Top1, MRR, smaller total, encoder, NGram"},
        "initial": {"rows": len(initial), "frozen_primary": subset_bundle(baseline_eval, "Frozen_rank"), "fixed_surface": initial_fixed, "candidate_scoring": initial_candidate, "recovery": initial_recovery},
        "full": {"rows": len(full), "fixed_surface": full_methods, "candidate_scoring": full_candidate, "recovery": full_recovery, "historical_lambdamart": lambdamart},
        "intrinsic_full": intrinsic,
        "runtime_seconds": time.perf_counter() - started,
        "used_dev3000": False,
        "used_test": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "result.json", result)
    write_json(args.output_root / "grid.json", all_grids)
    predictions = []
    for track, label, rows in recovery_predictions:
        for row in rows:
            predictions.append({"track": track, "method": label, "row_id": row["row_id"], "author": row["author"], "gold": row["gold"], "Frozen_rank": row["Frozen_rank"], "rank": row["RecoveryContext_rank"], "top10": row["RecoveryContext_top10"], "used_dev3000": False, "used_test": False})
    write_jsonl(args.output_root / "selected_predictions.jsonl", predictions)
    write_json(args.output_root / "artifact_checksums.json", {"result": sha256_file(args.output_root / "result.json"), "grid": sha256_file(args.output_root / "grid.json"), "predictions": sha256_file(args.output_root / "selected_predictions.jsonl"), "used_dev3000": False, "used_test": False})
    print(json.dumps({"status": "complete", "runtime_seconds": result["runtime_seconds"], "result": str(args.output_root / "result.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
