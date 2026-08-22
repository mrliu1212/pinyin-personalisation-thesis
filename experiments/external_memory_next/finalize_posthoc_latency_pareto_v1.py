"""Create the post-hoc candidate-scoring latency/Pareto summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.personalisation.task_specific_biencoder import refuse_closed_path, sha256_file, write_json


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pareto(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["pareto"] = not any(
            other["mean_latency_ms"] <= row["mean_latency_ms"]
            and other["macro_author_top1"] >= row["macro_author_top1"]
            and (
                other["mean_latency_ms"] < row["mean_latency_ms"]
                or other["macro_author_top1"] > row["macro_author_top1"]
            )
            for other in rows
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--task-latency", type=Path, required=True)
    parser.add_argument("--historical-summary", type=Path, required=True)
    parser.add_argument("--support-initial", type=Path, required=True)
    parser.add_argument("--support-full", type=Path, required=True)
    parser.add_argument("--q8-full", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    for value in vars(args).values():
        if isinstance(value, Path):
            refuse_closed_path(value)

    evaluation = load(args.evaluation)
    task_latency = load(args.task_latency)
    historical = load(args.historical_summary)
    support_initial = load(args.support_initial)
    support_full = load(args.support_full)
    q8_full = load(args.q8_full)
    historical_by_method = {row["method"]: row for row in historical["candidate_only"]}
    candidate = evaluation["initial"]["candidate_scoring"]["methods"]

    ngram_mean = float(historical_by_method["Interpolated NGramRecency"]["mean_latency_ms"])
    ngram_p95 = float(historical_by_method["Interpolated NGramRecency"]["p95_latency_ms"])
    frequency_mean = float(historical_by_method["Frequency"]["mean_latency_ms"])
    frequency_p95 = float(historical_by_method["Frequency"]["p95_latency_ms"])
    generic_mean = float(historical_by_method["BGE64"]["mean_latency_ms"])
    generic_p95 = float(historical_by_method["BGE64"]["p95_latency_ms"])
    q8_mean = float(historical_by_method["Q8"]["mean_latency_ms"])
    q8_p95 = float(historical_by_method["Q8"]["p95_latency_ms"])
    task_mean = float(task_latency["online_total"]["mean_ms"])
    task_p95 = float(task_latency["online_total"]["p95_ms"])

    def accuracy(method: str) -> float:
        return float(candidate[method]["generic_missing_recoverable_k2plus"]["ranking"]["macro_author_top1"])

    rows = [
        {"method": "Frequency", "macro_author_top1": accuracy("Frequency"), "mean_latency_ms": frequency_mean, "p95_latency_ms": frequency_p95, "latency_source": "historical exact Initial K5 run"},
        {"method": "P_NG", "macro_author_top1": accuracy("P_NG"), "mean_latency_ms": ngram_mean, "p95_latency_ms": ngram_p95, "latency_source": "historical exact Initial K5 interpolated NGram run"},
        {"method": "GenericBGERecency", "macro_author_top1": accuracy("GenericBGERecency"), "mean_latency_ms": generic_mean, "p95_latency_ms": generic_p95, "latency_source": "historical exact Initial BGE64 online run"},
        {"method": "TaskBiEncoderRecency", "macro_author_top1": accuracy("TaskBiEncoderRecency"), "mean_latency_ms": task_mean, "p95_latency_ms": task_p95, "latency_source": "current 500-query warm batch-1 benchmark"},
        {"method": "P_NG+GenericBGERecency", "macro_author_top1": accuracy("P_NG+GenericBGERecency"), "mean_latency_ms": ngram_mean + generic_mean, "p95_latency_ms": ngram_p95 + generic_p95, "latency_source": "component sum; not jointly timed"},
        {"method": "P_NG+TaskBiEncoderRecency", "macro_author_top1": accuracy("P_NG+TaskBiEncoderRecency"), "mean_latency_ms": ngram_mean + task_mean, "p95_latency_ms": ngram_p95 + task_p95, "latency_source": "component sum; not jointly timed"},
        {"method": "Q8", "macro_author_top1": accuracy("Q8"), "mean_latency_ms": q8_mean, "p95_latency_ms": q8_p95, "latency_source": "historical exact Initial Q8 online score-call run"},
        {"method": "Q8+F", "macro_author_top1": accuracy("Q8+F"), "mean_latency_ms": q8_mean + frequency_mean, "p95_latency_ms": q8_p95 + frequency_p95, "latency_source": "component sum; Q8 exact, Frequency historical"},
    ]
    pareto(rows)
    result = {
        "schema_version": 1,
        "status": "complete",
        "population": "Initial Generic-missing recoverable Personal-K5 K>=2 (n=4471)",
        "candidate_scoring": rows,
        "offline_preprocessing": {
            "initial_support_seconds": support_initial["runtime_seconds"],
            "full_support_seconds": support_full["runtime_seconds"],
            "full_q8_accumulated_score_call_seconds": (
                float(q8_full["latency_ms"]["mean"]) * int(q8_full["latency_ms"]["n"]) / 1000.0
            ),
        },
        "task_latency_benchmark": task_latency,
        "input_sha256": {
            "evaluation": sha256_file(args.evaluation),
            "task_latency": sha256_file(args.task_latency),
            "historical_summary": sha256_file(args.historical_summary),
        },
        "used_dev3000": False,
        "used_test": False,
    }
    write_json(args.output, result)

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    for row in rows:
        axis.scatter(row["mean_latency_ms"], row["macro_author_top1"], marker="o" if row["pareto"] else "x")
        axis.annotate(row["method"], (row["mean_latency_ms"], row["macro_author_top1"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    axis.set_xscale("log")
    axis.set_xlabel("Mean online latency (ms/query, log scale)")
    axis.set_ylabel("Macro-author Top1")
    axis.set_title("Initial Personal-K5 candidate scoring: accuracy/latency")
    axis.grid(True, which="both", alpha=.25)
    figure.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.plot, dpi=160)
    print(json.dumps({"result": str(args.output), "plot": str(args.plot)}, indent=2))


if __name__ == "__main__":
    main()
