"""Fixed-surface empirical-Bayes Choice Share ablation on Full Train-Val."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiments.external_memory_next.reproduce_full_retuned_baseline_v1 import (
    candidate_text,
    final_rerank,
)


EXPECTED_ROWS = 34416
EXPECTED = {
    "fit": "547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6",
    "val": "d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220",
    "stage1": "e311261cb0c5ea252ce593fdaa43ff87826d19eef440f39e2efc67ddc4310b13",
    "stage2": "d413d02650a759c0a759e1845212e68b2d948c1a1d20fc88abfd89ea7973bc64",
    "predictions": "f3e902e5a9e7d25e62799b9abb719026c336381eacc42999d1e7edccf2731b22",
    "config": "3dc3fb908aeeaa853526ad71cf85de7400f47d261ed7c09acdd8197446f5fa3d",
}
ALPHAS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
EXPERIMENT = "choice_share_smoothing_fixed_surface_v1"
RUNNER_PATH = Path(__file__)
CORE_RUNNER_PATH = Path(__file__)
N_BINS = (("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5),
          ("6-10", 6, 10), ("11-20", 11, 20), ("21-50", 21, 50),
          ("51-100", 51, 100), (">100", 101, None))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("source_split", "")).lower() == "test" or bool(row.get("used_test", False)):
                raise RuntimeError(f"Test row in {path}:{number}")
            yield row


def pinyin_of(row: Mapping[str, Any]) -> str:
    return " ".join(map(str, row["pinyin_segments"]))


def build_prior(rows: Iterable[Mapping[str, Any]]) -> tuple[Counter[tuple[str, str]], Counter[str]]:
    counts: Counter[tuple[str, str]] = Counter()
    totals: Counter[str] = Counter()
    for row in rows:
        if row.get("standardized_partition") != "train_fit":
            raise RuntimeError("Prior input is not Train-Fit")
        pinyin = pinyin_of(row)
        counts[(pinyin, str(row["target"]))] += 1
        totals[pinyin] += 1
    return counts, totals


def smooth_choice(n_c: int, n: int, prior: float, alpha: float) -> float:
    if n_c < 0 or n < n_c or alpha < 0 or not 0 <= prior <= 1:
        raise ValueError("Invalid smoothing input")
    if alpha == 0:
        return n_c / n if n else 0.0
    return (n_c + alpha * prior) / (n + alpha)


def n_bucket(value: int) -> str:
    for name, lower, upper in N_BINS:
        if value >= lower and (upper is None or value <= upper):
            return name
    raise AssertionError(value)


def fixed_surface_stage1(feature: Mapping[str, Any], support: Mapping[str, Any],
                         prior_counts: Mapping[tuple[str, str], int], prior_totals: Mapping[str, int],
                         pinyin: str, alpha: float, weights: Mapping[str, float]) -> list[dict[str, Any]]:
    surface = [dict(row) for row in support["retuned_stage1_candidates"]]
    if not surface:
        return []
    generic = feature["generic_frequency_candidates"]
    boundary = min(float(row["normalized_generic_score"]) for row in generic)
    raw_choice = {str(key): float(value) for key, value in feature["choice_share"].items()}
    p_ng = {str(key): float(value) for key, value in feature["p_ng"].items()}
    n = int(feature["same_pinyin_history_count"])
    total = int(prior_totals.get(pinyin, 0))
    for row in surface:
        if row["source"] != "personal_recovery":
            continue
        candidate = candidate_text(row)
        raw_count = raw_choice[candidate] * n
        n_c = int(round(raw_count))
        if not math.isclose(raw_count, n_c, rel_tol=0, abs_tol=1e-9):
            raise RuntimeError(f"Choice Share does not reconstruct an integer count: {feature['row_id']}")
        prior = prior_counts.get((pinyin, candidate), 0) / total if total else 0.0
        smoothed = smooth_choice(n_c, n, prior, alpha)
        row["raw_choice_share"] = raw_choice[candidate]
        row["choice_share"] = smoothed
        row["choice_prior"] = prior
        row["choice_alpha"] = alpha
        row["final_score"] = (boundary + weights["w_p"] * p_ng[candidate]
                              + weights["w_cs"] * smoothed
                              + weights["w_e"] * float(feature["entropy_concentration"]))
    surface.sort(key=lambda row: (-float(row["final_score"]),
                                  0 if row["source"] == "generic_frequency" else 1,
                                  int(row.get("generic_rank") or row.get("personal_candidate_rank") or row.get("rank") or 0),
                                  candidate_text(row)))
    for rank, row in enumerate(surface, start=1):
        row["rank"] = rank
        row["base_rank"] = rank
    return surface


def metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_author: dict[str, list[int | None]] = defaultdict(list)
    ranks = []
    for row in rows:
        rank = None if row.get("rank") is None else int(row["rank"])
        ranks.append(rank)
        by_author[str(row["author"])].append(rank)

    def top(values: Sequence[int | None], k: int) -> float:
        return sum(rank is not None and rank <= k for rank in values) / len(values)

    per_author = {author: top(values, 1) for author, values in sorted(by_author.items())}
    return {"n": len(rows), "macro_author_top1": statistics.fmean(per_author.values()),
            "micro_top1": top(ranks, 1), "top3": top(ranks, 3), "top5": top(ranks, 5),
            "mrr_at_10": sum(0 if rank is None else 1 / rank for rank in ranks) / len(ranks),
            "missing10": sum(rank is None for rank in ranks) / len(ranks),
            "per_author_top1": per_author}


def transitions(rows: Sequence[Mapping[str, Any]], before: str, after: str) -> dict[str, int]:
    out = {"n": len(rows), "rescue": 0, "harm": 0, "unchanged_correct": 0, "unchanged_wrong": 0}
    for row in rows:
        old = row.get(before) == 1
        new = row.get(after) == 1
        key = "rescue" if new and not old else "harm" if old and not new else "unchanged_correct" if old else "unchanged_wrong"
        out[key] += 1
    out["net"] = out["rescue"] - out["harm"]
    return out


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in EXPECTED:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in EXPECTED}
    provenance = {}
    for name, path in paths.items():
        actual = sha256_file(path)
        if actual != EXPECTED[name]:
            raise RuntimeError(f"{name} SHA mismatch: {actual}")
        provenance[name] = {"path": str(path.resolve()), "sha256": actual, "bytes": path.stat().st_size}
    config = json.loads(args.config.read_text(encoding="utf-8"))
    weights = {**{key: float(config["selected_stage1"][key]) for key in ("w_p", "w_cs", "w_e")},
               **{key: float(config["selected_stage2"][key]) for key in ("lambda_n", "lambda_b")}}
    if weights != {"w_p": 2.0, "w_cs": 6.0, "w_e": 4.0, "lambda_n": 6.0, "lambda_b": 6.0}:
        raise RuntimeError(f"Frozen weights changed: {weights}")
    prior_counts, prior_totals = build_prior(iter_jsonl(args.fit))

    rows_by_alpha: dict[float, list[dict[str, Any]]] = {alpha: [] for alpha in ALPHAS}
    iterators = (iter_jsonl(args.val), iter_jsonl(args.stage1), iter_jsonl(args.stage2), iter_jsonl(args.predictions))
    for number, group in enumerate(itertools.zip_longest(*iterators), start=1):
        if any(row is None for row in group):
            raise RuntimeError("Input row counts differ")
        val, feature, support, prediction = group
        row_ids = {str(row["row_id"]) for row in group}
        if len(row_ids) != 1:
            raise RuntimeError(f"Input row order differs at {number}: {row_ids}")
        gold = str(feature["gold"])
        for alpha in ALPHAS:
            stage1 = fixed_surface_stage1(feature, support, prior_counts, prior_totals,
                                          pinyin_of(val), alpha, weights)
            final = final_rerank(stage1, support["retuned_ngram_support"],
                                 support["retuned_bge_support"], weights)
            top10 = [candidate_text(row) for row in final]
            rank = next((i for i, candidate in enumerate(top10, start=1) if candidate == gold), None)
            if alpha == 0 and (top10 != list(map(str, prediction["RetunedFinal_top10"]))
                               or rank != prediction.get("RetunedFinal_rank")):
                raise RuntimeError(f"Alpha-zero baseline mismatch: {feature['row_id']}")
            rows_by_alpha[alpha].append({"row_id": feature["row_id"], "author": feature["author"],
                                         "ambiguous": bool(feature["ambiguous"]),
                                         "conflict": bool(feature["conflict"]),
                                         "generic_missing": bool(feature["generic_missing"]),
                                         "gold_in_personal_k5": bool(feature["gold_in_personal_k5"]),
                                         "n": int(feature["same_pinyin_history_count"]), "rank": rank,
                                         "top10": top10})
    if any(len(rows) != EXPECTED_ROWS for rows in rows_by_alpha.values()):
        raise RuntimeError("Output row count changed")

    grid = []
    baseline_rows = rows_by_alpha[0.0]
    for alpha in ALPHAS:
        rows = rows_by_alpha[alpha]
        record = {"alpha": alpha,
                  "metrics": {name: metrics([row for row in rows if name == "overall" or bool(row[name])])
                              for name in ("overall", "ambiguous", "conflict")},
                  "history_bins": {name: metrics([row for row in rows if n_bucket(int(row["n"])) == name])
                                   for name, _, _ in N_BINS},
                  "transition_from_alpha0": transitions(
                      [{**row, "baseline_rank": baseline_rows[index]["rank"]}
                       for index, row in enumerate(rows)], "baseline_rank", "rank")}
        grid.append(record)
    grid.sort(key=lambda row: (-row["metrics"]["overall"]["macro_author_top1"],
                               -row["metrics"]["overall"]["micro_top1"],
                               -row["metrics"]["overall"]["mrr_at_10"], row["alpha"]))
    selected = grid[0]
    selected_rows = rows_by_alpha[float(selected["alpha"])]
    result = {"schema_version": 1, "status": "complete",
              "experiment": EXPERIMENT,
              "estimator": "(n_c + alpha * all-author-Train-Fit-P(c|p)) / (N + alpha)",
              "unseen_policy": "zero prior mass", "alphas": list(ALPHAS),
              "selection_rule": "Macro-author Top1, Micro Top1, MRR@10, then smaller alpha",
              "fixed_weights": weights, "fixed_candidate_surface": True,
              "selected": selected, "grid": grid, "baseline_exact": True,
              "provenance": provenance, "used_dev3000": False, "used_test": False}
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "result.json", result)
    write_json(args.output_root / "grid_results.json", grid)
    write_jsonl(args.output_root / "selected_predictions.jsonl", selected_rows)
    write_json(args.output_root / "artifact_checksums.json", {
        "runner": sha256_file(RUNNER_PATH), "core_runner": sha256_file(CORE_RUNNER_PATH),
        "result.json": sha256_file(args.output_root / "result.json"),
        "grid_results.json": sha256_file(args.output_root / "grid_results.json"),
        "selected_predictions.jsonl": sha256_file(args.output_root / "selected_predictions.jsonl"),
        "used_dev3000": False, "used_test": False})
    print(json.dumps({"status": "complete", "selected_alpha": selected["alpha"],
                      "selected_metrics": selected["metrics"]["overall"],
                      "baseline_metrics": next(row for row in grid if row["alpha"] == 0)["metrics"]["overall"],
                      "output": str(args.output_root / "result.json")}, indent=2))


if __name__ == "__main__":
    main()
