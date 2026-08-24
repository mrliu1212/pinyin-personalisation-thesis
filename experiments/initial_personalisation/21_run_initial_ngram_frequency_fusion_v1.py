"""Train-Val-only fusion of fast lexical candidate scorers with Frequency.

Consumes the completed adaptive N-gram K5/K10 experiment and performs no
PinyinGPT/BGE inference.  It tests:

  HardBackoffNGramRecency + F
  InterpolatedNGramRecency + F

for K5 and K10 using

  S = (1 - alpha) * NGramSupport + alpha * FrequencySupport

where both component supports are already non-negative and normalized over the
current Personal candidate pool.

Selection protocol
------------------
- alpha grid: {0, .25, .5, .75, 1} by default
- select separately for each scorer and candidate K
- primary: Macro-author Top1 on Train-Val, Gold-in-current-Personal-K, K>=2
- exact tie: lower alpha (less Frequency injection)
- Dev3000 is not read; Test is not read
- Gold is used only for Train-Val selection/evaluation, never scoring

For K10, the selected method is also evaluated on the shared frozen-K5
population so K5-vs-K10 ranking pollution can be compared apples-to-apples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXPECTED_VAL_ROWS = 34416


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def index_rows(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row["row_id"])
        if row_id in out:
            raise RuntimeError(f"Duplicate row_id in {label}: {row_id}")
        out[row_id] = dict(row)
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    clipped = [max(0.0, float(value)) for value in values]
    total = sum(clipped)
    if total <= 0:
        return [1.0 / len(clipped)] * len(clipped)
    return [value / total for value in clipped]


def rank_candidates(candidates: Sequence[str], distribution: Sequence[float]) -> list[str]:
    if len(candidates) != len(distribution):
        raise RuntimeError("candidate/support length mismatch")
    return [
        candidates[i]
        for i in sorted(
            range(len(candidates)),
            key=lambda i: (-float(distribution[i]), i, candidates[i]),
        )
    ]


def gold_rank(ranking: Sequence[str], gold: str) -> int | None:
    for index, candidate in enumerate(ranking, start=1):
        if candidate == gold:
            return index
    return None


def metric_summary(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    by_author: dict[str, list[int]] = defaultdict(list)
    ranks: list[int] = []
    for row in rows:
        rank = row.get("rank")
        if rank is None:
            continue
        rank_i = int(rank)
        ranks.append(rank_i)
        by_author[str(row["author"])].append(rank_i)

    per_author = {
        author: sum(rank == 1 for rank in author_ranks) / len(author_ranks)
        for author, author_ranks in sorted(by_author.items())
        if author_ranks
    }
    n = len(rows)
    return {
        "method": method,
        "n": n,
        "authors": len(per_author),
        "macro_author_top1": statistics.fmean(per_author.values()) if per_author else None,
        "micro_top1": sum(rank == 1 for rank in ranks) / n if n else None,
        "top3": sum(rank <= 3 for rank in ranks) / n if n else None,
        "top5": sum(rank <= 5 for rank in ranks) / n if n else None,
        "mrr_at_10": sum(1.0 / rank for rank in ranks if rank <= 10) / n if n else None,
        "mean_gold_rank": statistics.fmean(ranks) if ranks else None,
        "per_author_top1": per_author,
    }


def choose_alpha(grid: Mapping[float, Mapping[str, Any]]) -> float:
    # Same principle as the earlier Q8+F fusion: primary Macro-author Top1;
    # exact tie chooses lower alpha.
    return max(
        grid,
        key=lambda alpha: (
            float(grid[alpha]["macro_author_top1"] or -1.0),
            -float(alpha),
        ),
    )


def transitions(base_rows: Mapping[str, Mapping[str, Any]], new_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "rescue": 0,
        "harm": 0,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
    }
    for row_id in sorted(set(base_rows) & set(new_rows)):
        base_correct = int(base_rows[row_id]["rank"]) == 1
        new_correct = int(new_rows[row_id]["rank"]) == 1
        if not base_correct and new_correct:
            counts["rescue"] += 1
        elif base_correct and not new_correct:
            counts["harm"] += 1
        elif base_correct and new_correct:
            counts["unchanged_correct"] += 1
        else:
            counts["unchanged_wrong"] += 1
    counts["net"] = counts["rescue"] - counts["harm"]
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument(
        "--adaptive-dir",
        type=Path,
        required=True,
        help="Directory containing scores.jsonl, comparison.json, candidate_surface_k5_k10.jsonl",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    args = parser.parse_args()

    alphas = tuple(sorted(set(float(value) for value in args.alphas)))
    if not alphas or any(value < 0 or value > 1 for value in alphas):
        raise ValueError("alphas must be in [0,1]")

    score_path = args.adaptive_dir / "scores.jsonl"
    comparison_path = args.adaptive_dir / "comparison.json"
    surface_path = args.adaptive_dir / "candidate_surface_k5_k10.jsonl"
    for path in (args.val, score_path, comparison_path, surface_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    val_rows = index_rows(read_jsonl(args.val), "Train-Val")
    score_rows = index_rows(read_jsonl(score_path), "adaptive scores")
    surface_rows = index_rows(read_jsonl(surface_path), "K5/K10 surface")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))

    if len(val_rows) != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"Unexpected Train-Val rows: {len(val_rows)}")
    if set(val_rows) != set(score_rows) or set(val_rows) != set(surface_rows):
        raise RuntimeError("Train-Val / score / surface row IDs differ")
    if comparison.get("status") != "complete":
        raise RuntimeError("Adaptive comparison is not complete")
    if comparison.get("provenance", {}).get("dev3000_used") is not False:
        raise RuntimeError("Unexpected Dev3000 provenance")
    if comparison.get("provenance", {}).get("test_used") is not False:
        raise RuntimeError("Unexpected Test provenance")

    selected = comparison["selected_by_k"]
    scorer_fields = {
        "HardBackoffNGramRecency": "best_hard_backoff",
        "InterpolatedNGramRecency": "best_interpolated",
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / "ngram_frequency_fusion_comparison.json"

    print("\n=== N-GRAM RECENCY + FREQUENCY FUSION ===")
    print(f"Train-Val rows: {len(val_rows)}")
    print(f"alpha grid (Frequency weight): {alphas}")
    print("Formula: (1-alpha)*NGram + alpha*F")
    print("Gold used for scoring: false")
    print("Gold used for Train-Val alpha selection/evaluation only: true")
    print("Dev3000 used: false")
    print("Test used: false")

    results: dict[str, Any] = {}

    for k in (5, 10):
        k_label = f"K{k}"
        frequency_key = str(selected[k_label]["frequency"])
        k_results: dict[str, Any] = {}

        for scorer_label, selector_field in scorer_fields.items():
            ngram_key = str(selected[k_label][selector_field])
            alpha_rows: dict[float, list[dict[str, Any]]] = {alpha: [] for alpha in alphas}
            shared_rows: dict[float, list[dict[str, Any]]] = {alpha: [] for alpha in alphas}
            component_ngram_rows: dict[str, dict[str, Any]] = {}
            component_frequency_rows: dict[str, dict[str, Any]] = {}

            fusion_arithmetic_ms: list[float] = []

            for row_id in sorted(val_rows):
                candidates = tuple(str(value) for value in surface_rows[row_id][f"personal_k{k}"])
                gold = str(val_rows[row_id].get("gold", val_rows[row_id].get("target", "")))
                if len(candidates) < 2 or gold not in candidates:
                    continue
                author = str(val_rows[row_id]["author"])
                frozen_k5 = tuple(str(value) for value in surface_rows[row_id]["personal_k5"])
                shared_k5_eligible = len(frozen_k5) >= 2 and gold in frozen_k5

                methods = score_rows[row_id]["methods"]
                ngram = methods[ngram_key]
                frequency = methods[frequency_key]
                ngram_candidates = tuple(str(value) for value in ngram["candidates"])
                frequency_candidates = tuple(str(value) for value in frequency["candidates"])
                if ngram_candidates != candidates or frequency_candidates != candidates:
                    raise RuntimeError(f"Candidate mismatch at {row_id}")

                n_support = normalize(ngram["support"])
                f_support = normalize(frequency["support"])

                n_rank = gold_rank(rank_candidates(candidates, n_support), gold)
                f_rank = gold_rank(rank_candidates(candidates, f_support), gold)
                component_ngram_rows[row_id] = {"row_id": row_id, "rank": n_rank}
                component_frequency_rows[row_id] = {"row_id": row_id, "rank": f_rank}

                started = time.perf_counter()
                fused_by_alpha = {
                    alpha: normalize([
                        (1.0 - alpha) * n_value + alpha * f_value
                        for n_value, f_value in zip(n_support, f_support)
                    ])
                    for alpha in alphas
                }
                fusion_arithmetic_ms.append((time.perf_counter() - started) * 1000.0)

                for alpha, fused in fused_by_alpha.items():
                    ranking = rank_candidates(candidates, fused)
                    record = {
                        "row_id": row_id,
                        "author": author,
                        "gold": gold,
                        "rank": gold_rank(ranking, gold),
                        "winner": ranking[0] if ranking else None,
                    }
                    alpha_rows[alpha].append(record)
                    if shared_k5_eligible:
                        shared_rows[alpha].append(record)

            metrics_grid = {
                alpha: metric_summary(rows, f"{ngram_key}+F@{alpha:g}")
                for alpha, rows in alpha_rows.items()
            }
            selected_alpha = choose_alpha(metrics_grid)
            selected_metrics = metrics_grid[selected_alpha]
            selected_shared_metrics = metric_summary(
                shared_rows[selected_alpha],
                f"{ngram_key}+F@{selected_alpha:g}|sharedK5",
            )

            selected_rows_by_id = {
                str(row["row_id"]): row for row in alpha_rows[selected_alpha]
            }
            ngram_transitions = transitions(component_ngram_rows, selected_rows_by_id)
            frequency_transitions = transitions(component_frequency_rows, selected_rows_by_id)

            k_results[scorer_label] = {
                "ngram_method": ngram_key,
                "frequency_method": frequency_key,
                "formula": "(1-alpha)*NGram + alpha*F",
                "alpha_is_frequency_weight": True,
                "alpha_grid": list(alphas),
                "selection_population": "Train-Val Gold-in-current-Personal-K and K>=2",
                "selection_metric": "Macro-author Top1; exact tie -> lower alpha",
                "selected_alpha": selected_alpha,
                "grid_metrics_k_specific": {f"{alpha:g}": metrics_grid[alpha] for alpha in alphas},
                "selected_metrics_k_specific": selected_metrics,
                "selected_metrics_shared_frozen_k5": selected_shared_metrics,
                "transition_ngram_to_fusion": ngram_transitions,
                "transition_frequency_to_fusion": frequency_transitions,
                "fusion_arithmetic_mean_ms_for_full_alpha_grid": (
                    statistics.fmean(fusion_arithmetic_ms) if fusion_arithmetic_ms else None
                ),
            }

            print(f"\n{k_label} {scorer_label}")
            print(f"  base NGram: {ngram_key}")
            for alpha in alphas:
                m = metrics_grid[alpha]
                print(
                    f"  alpha={alpha:>4g}  "
                    f"MacroTop1={m['macro_author_top1']:.6f} "
                    f"MicroTop1={m['micro_top1']:.6f} "
                    f"Top3={m['top3']:.6f} MRR10={m['mrr_at_10']:.6f}"
                )
            print(f"  SELECTED alpha={selected_alpha:g}")
            print(f"  NGram -> fusion: {ngram_transitions}")

        results[k_label] = k_results

    output = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_ngram_frequency_fusion_v1",
        "formula": "(1-alpha)*NGramSupport + alpha*FrequencySupport",
        "alpha_is_frequency_weight": True,
        "selection_population": "Train-Val Gold-in-current-Personal-K and K>=2",
        "selection_metric": "Macro-author Top1; exact tie -> lower alpha",
        "gold_used_for_scoring": False,
        "gold_used_for_train_val_selection_evaluation": True,
        "dev3000_used": False,
        "test_used": False,
        "results": results,
        "provenance": {
            "val": str(args.val.resolve()),
            "val_sha256": sha256_file(args.val),
            "adaptive_dir": str(args.adaptive_dir.resolve()),
            "scores_sha256": sha256_file(score_path),
            "comparison_sha256": sha256_file(comparison_path),
            "surface_sha256": sha256_file(surface_path),
        },
    }
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nsaved: {output_path}")


if __name__ == "__main__":
    main()
