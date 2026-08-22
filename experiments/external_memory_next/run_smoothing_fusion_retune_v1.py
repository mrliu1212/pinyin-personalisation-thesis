"""Sequential fixed-surface coefficient retune after Choice Share smoothing."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.external_memory_next import run_choice_share_smoothing_v1 as smooth
from experiments.external_memory_next.reproduce_full_retuned_baseline_v1 import candidate_text, final_rerank


ALPHAS = (0.0, 2.0, 8.0, 32.0, 128.0, 512.0)
W_CS_GRID = (2.0, 4.0, 6.0, 8.0, 12.0)
LAMBDA_GRID = (0.0, 2.0, 4.0, 6.0, 8.0)
REFERENCE_STAGE_A = (128.0, 6.0)
REFERENCE_STAGE_B = (6.0, 6.0)


def metric_summary(meta: Sequence[Mapping[str, Any]], ranks: Sequence[int | None],
                   subset: str = "overall") -> dict[str, Any]:
    selected = [(row, rank) for row, rank in zip(meta, ranks)
                if subset == "overall" or bool(row[subset])]
    by_author: dict[str, list[int | None]] = defaultdict(list)
    values = []
    for row, rank in selected:
        values.append(rank)
        by_author[str(row["author"])].append(rank)

    def top(items: Sequence[int | None], k: int) -> float:
        return sum(rank is not None and rank <= k for rank in items) / len(items)

    per_author = {author: top(items, 1) for author, items in sorted(by_author.items())}
    return {"n": len(values), "macro_author_top1": statistics.fmean(per_author.values()),
            "micro_top1": top(values, 1), "top3": top(values, 3), "top5": top(values, 5),
            "mrr_at_10": sum(0 if rank is None else 1 / rank for rank in values) / len(values),
            "missing10": sum(rank is None for rank in values) / len(values),
            "per_author_top1": per_author}


def transition_summary(before: Sequence[int | None], after: Sequence[int | None]) -> dict[str, int]:
    result = {"n": len(before), "rescue": 0, "harm": 0,
              "unchanged_correct": 0, "unchanged_wrong": 0}
    for old, new in zip(before, after):
        old_correct, new_correct = old == 1, new == 1
        key = "rescue" if new_correct and not old_correct else "harm" if old_correct and not new_correct else "unchanged_correct" if old_correct else "unchanged_wrong"
        result[key] += 1
    result["net"] = result["rescue"] - result["harm"]
    return result


def selection_key(metrics: Mapping[str, Any], params: Sequence[float], reference: Sequence[float]) -> tuple[Any, ...]:
    return (-float(metrics["macro_author_top1"]), -float(metrics["micro_top1"]),
            -float(metrics["mrr_at_10"]),
            sum(abs(a - b) for a, b in zip(params, reference)), tuple(params))


def rank_of(rows: Sequence[Mapping[str, Any]], gold: str) -> int | None:
    return next((index for index, row in enumerate(rows, start=1)
                 if candidate_text(row) == gold), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in smooth.EXPECTED:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in smooth.EXPECTED}
    provenance = {}
    for name, path in paths.items():
        actual = smooth.sha256_file(path)
        if actual != smooth.EXPECTED[name]:
            raise RuntimeError(f"{name} SHA mismatch: {actual}")
        provenance[name] = {"path": str(path.resolve()), "sha256": actual, "bytes": path.stat().st_size}
    config = json.loads(args.config.read_text(encoding="utf-8"))
    base_weights = {**{key: float(config["selected_stage1"][key]) for key in ("w_p", "w_cs", "w_e")},
                    **{key: float(config["selected_stage2"][key]) for key in ("lambda_n", "lambda_b")}}
    if base_weights != {"w_p": 2.0, "w_cs": 6.0, "w_e": 4.0, "lambda_n": 6.0, "lambda_b": 6.0}:
        raise RuntimeError(f"Frozen weights changed: {base_weights}")
    prior_counts, prior_totals = smooth.build_prior(smooth.iter_jsonl(args.fit))

    points_a = tuple(itertools.product(ALPHAS, W_CS_GRID))
    ranks_a: dict[tuple[float, float], list[int | None]] = {point: [] for point in points_a}
    meta: list[dict[str, Any]] = []
    iterators = (smooth.iter_jsonl(args.val), smooth.iter_jsonl(args.stage1),
                 smooth.iter_jsonl(args.stage2), smooth.iter_jsonl(args.predictions))
    for number, group in enumerate(itertools.zip_longest(*iterators), start=1):
        if any(row is None for row in group):
            raise RuntimeError("Input row counts differ")
        val, feature, support, prediction = group
        if len({str(row["row_id"]) for row in group}) != 1:
            raise RuntimeError(f"Input row order differs at {number}")
        gold = str(feature["gold"])
        meta.append({"row_id": feature["row_id"], "author": feature["author"],
                     "ambiguous": bool(feature["ambiguous"]), "conflict": bool(feature["conflict"]),
                     "n": int(feature["same_pinyin_history_count"]),
                     "generic_missing": bool(feature["generic_missing"]),
                     "gold_in_personal_k5": bool(feature["gold_in_personal_k5"])})
        for alpha, w_cs in points_a:
            weights = {**base_weights, "w_cs": w_cs}
            stage1 = smooth.fixed_surface_stage1(feature, support, prior_counts, prior_totals,
                                                  smooth.pinyin_of(val), alpha, weights)
            final = final_rerank(stage1, support["retuned_ngram_support"],
                                 support["retuned_bge_support"], weights)
            rank = rank_of(final, gold)
            ranks_a[(alpha, w_cs)].append(rank)
            if alpha == 0 and w_cs == 6:
                top10 = [candidate_text(row) for row in final]
                if top10 != list(map(str, prediction["RetunedFinal_top10"])) or rank != prediction.get("RetunedFinal_rank"):
                    raise RuntimeError(f"Raw baseline mismatch: {feature['row_id']}")
    if len(meta) != smooth.EXPECTED_ROWS:
        raise RuntimeError(f"Row count changed: {len(meta)}")

    grid_a = []
    for point, ranks in ranks_a.items():
        overall = metric_summary(meta, ranks)
        grid_a.append({"alpha": point[0], "w_cs": point[1], "metrics": overall})
    grid_a.sort(key=lambda row: selection_key(row["metrics"], (row["alpha"], row["w_cs"]), REFERENCE_STAGE_A))
    selected_a = grid_a[0]
    selected_a_point = (float(selected_a["alpha"]), float(selected_a["w_cs"]))

    points_b = tuple(itertools.product(LAMBDA_GRID, LAMBDA_GRID))
    ranks_b: dict[tuple[float, float], list[int | None]] = {point: [] for point in points_b}
    iterators = (smooth.iter_jsonl(args.val), smooth.iter_jsonl(args.stage1), smooth.iter_jsonl(args.stage2))
    for number, group in enumerate(itertools.zip_longest(*iterators), start=1):
        if any(row is None for row in group):
            raise RuntimeError("Input row counts differ in Stage B")
        val, feature, support = group
        if len({str(row["row_id"]) for row in group}) != 1:
            raise RuntimeError(f"Stage-B row order differs at {number}")
        weights_a = {**base_weights, "w_cs": selected_a_point[1]}
        stage1 = smooth.fixed_surface_stage1(feature, support, prior_counts, prior_totals,
                                              smooth.pinyin_of(val), selected_a_point[0], weights_a)
        gold = str(feature["gold"])
        for lambda_n, lambda_b in points_b:
            weights = {**weights_a, "lambda_n": lambda_n, "lambda_b": lambda_b}
            final = final_rerank(stage1, support["retuned_ngram_support"],
                                 support["retuned_bge_support"], weights)
            ranks_b[(lambda_n, lambda_b)].append(rank_of(final, gold))
    grid_b = []
    for point, ranks in ranks_b.items():
        overall = metric_summary(meta, ranks)
        grid_b.append({"lambda_n": point[0], "lambda_b": point[1], "metrics": overall})
    grid_b.sort(key=lambda row: selection_key(row["metrics"], (row["lambda_n"], row["lambda_b"]), REFERENCE_STAGE_B))
    selected_b = grid_b[0]
    selected_b_point = (float(selected_b["lambda_n"]), float(selected_b["lambda_b"]))
    selected_ranks = ranks_b[selected_b_point]
    baseline_ranks = ranks_a[(0.0, 6.0)]

    selected_predictions = []
    selected_weights = {**base_weights, "w_cs": selected_a_point[1],
                        "lambda_n": selected_b_point[0], "lambda_b": selected_b_point[1]}
    iterators = (smooth.iter_jsonl(args.val), smooth.iter_jsonl(args.stage1), smooth.iter_jsonl(args.stage2))
    for index, group in enumerate(itertools.zip_longest(*iterators)):
        if any(row is None for row in group):
            raise RuntimeError("Input row counts differ during finalization")
        val, feature, support = group
        stage1 = smooth.fixed_surface_stage1(feature, support, prior_counts, prior_totals,
                                              smooth.pinyin_of(val), selected_a_point[0], selected_weights)
        final = final_rerank(stage1, support["retuned_ngram_support"],
                             support["retuned_bge_support"], selected_weights)
        selected_predictions.append({**meta[index], "gold": str(feature["gold"]),
                                     "baseline_rank": baseline_ranks[index], "rank": selected_ranks[index],
                                     "top10": [candidate_text(row) for row in final],
                                     "used_dev3000": False, "used_test": False})

    breakdown = {name: metric_summary(meta, selected_ranks, name)
                 for name in ("overall", "ambiguous", "conflict")}
    history_bins = {}
    for name, lower, upper in smooth.N_BINS:
        indexes = [i for i, row in enumerate(meta) if int(row["n"]) >= lower and (upper is None or int(row["n"]) <= upper)]
        history_bins[name] = metric_summary([meta[i] for i in indexes], [selected_ranks[i] for i in indexes])
    result = {"schema_version": 1, "status": "complete",
              "experiment": "choice_share_smoothing_fusion_retune_fixed_surface_v1",
              "stage_a": {"grid": grid_a, "selected": selected_a,
                          "alphas": list(ALPHAS), "w_cs_grid": list(W_CS_GRID)},
              "stage_b": {"grid": grid_b, "selected": selected_b,
                          "lambda_n_grid": list(LAMBDA_GRID), "lambda_b_grid": list(LAMBDA_GRID)},
              "selected_config": {"alpha": selected_a_point[0], **selected_weights},
              "metrics": breakdown, "history_bins": history_bins,
              "transition_from_raw_baseline": transition_summary(baseline_ranks, selected_ranks),
              "baseline_metrics": metric_summary(meta, baseline_ranks),
              "fixed_candidate_surface": True, "provenance": provenance,
              "used_dev3000": False, "used_test": False}
    args.output_root.mkdir(parents=True, exist_ok=True)
    smooth.write_json(args.output_root / "result.json", result)
    smooth.write_json(args.output_root / "stage_a_grid.json", grid_a)
    smooth.write_json(args.output_root / "stage_b_grid.json", grid_b)
    smooth.write_jsonl(args.output_root / "selected_predictions.jsonl", selected_predictions)
    smooth.write_json(args.output_root / "artifact_checksums.json", {
        "runner": smooth.sha256_file(Path(__file__)), "smoothing_core": smooth.sha256_file(Path(smooth.__file__)),
        "result.json": smooth.sha256_file(args.output_root / "result.json"),
        "stage_a_grid.json": smooth.sha256_file(args.output_root / "stage_a_grid.json"),
        "stage_b_grid.json": smooth.sha256_file(args.output_root / "stage_b_grid.json"),
        "selected_predictions.jsonl": smooth.sha256_file(args.output_root / "selected_predictions.jsonl"),
        "used_dev3000": False, "used_test": False})
    print(json.dumps({"status": "complete", "selected_config": result["selected_config"],
                      "metrics": breakdown["overall"],
                      "transition": result["transition_from_raw_baseline"],
                      "output": str(args.output_root / "result.json")}, indent=2))


if __name__ == "__main__":
    main()
