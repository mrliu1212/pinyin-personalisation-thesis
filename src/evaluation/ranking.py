"""Model-independent Top-K ranking metrics for reusable IME evaluation."""

from __future__ import annotations

from collections import defaultdict
import math
import random
from statistics import mean
from typing import Any, Callable, Iterable, Mapping, Sequence


TOP_KS = (1, 3, 5, 10)


def _rank(row: Mapping[str, Any]) -> int | None:
    value = row.get("gold_top10_rank")
    return int(value) if value is not None else None


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        return {
            "interaction_count": 0,
            "top1": None,
            "top3": None,
            "top5": None,
            "top10": None,
            "coverage_at_10": None,
            "mrr_at_10": None,
            "mean_rank_given_top10": None,
            "missing_at_10_count": 0,
            "missing_at_10_rate": None,
        }
    ranks = [_rank(row) for row in rows]
    present = [rank for rank in ranks if rank is not None]
    values: dict[str, Any] = {"interaction_count": count}
    for top_k in TOP_KS:
        values[f"top{top_k}"] = sum(
            rank is not None and rank <= top_k for rank in ranks
        ) / count
    values["coverage_at_10"] = values["top10"]
    values["mrr_at_10"] = sum(
        0.0 if rank is None else 1.0 / rank for rank in ranks
    ) / count
    values["mean_rank_given_top10"] = mean(present) if present else None
    values["missing_at_10_count"] = count - len(present)
    values["missing_at_10_rate"] = (count - len(present)) / count
    return values


def compute_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute micro and equally weighted macro-user Top-10 metrics."""

    materialized = list(rows)
    by_user: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in materialized:
        by_user[str(row["user_id"])].append(row)
    per_user = {
        user_id: _aggregate(user_rows)
        for user_id, user_rows in sorted(by_user.items())
    }
    macro: dict[str, Any] = {"user_count": len(per_user)}
    metric_names = (
        "top1",
        "top3",
        "top5",
        "top10",
        "coverage_at_10",
        "mrr_at_10",
        "mean_rank_given_top10",
        "missing_at_10_rate",
    )
    for name in metric_names:
        values = [metrics[name] for metrics in per_user.values() if metrics[name] is not None]
        macro[name] = mean(values) if values else None
    macro["missing_at_10_count_mean"] = (
        mean(metrics["missing_at_10_count"] for metrics in per_user.values())
        if per_user
        else None
    )
    return {
        "metric_scope_note": (
            "MRR@10 assigns zero when gold is absent; MeanRank|Top10 is conditional "
            "on gold appearing in the returned Top-10."
        ),
        "micro": _aggregate(materialized),
        "macro_user": macro,
        "per_user": per_user,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def context_gain(
    pinyin_only_rows: Iterable[Mapping[str, Any]],
    contextual_rows: Iterable[Mapping[str, Any]],
    *,
    seed: int = 40408,
    bootstrap_resamples: int = 10_000,
) -> dict[str, Any]:
    """Compare paired conditions and bootstrap macro-user Top-1 differences."""

    only = {str(row["interaction_id"]): row for row in pinyin_only_rows}
    contextual = {str(row["interaction_id"]): row for row in contextual_rows}
    if set(only) != set(contextual):
        missing_only = sorted(set(contextual) - set(only))[:5]
        missing_contextual = sorted(set(only) - set(contextual))[:5]
        raise ValueError(
            "prediction interaction IDs are not aligned; "
            f"missing pinyin-only={missing_only}, missing contextual={missing_contextual}"
        )
    pairs = [(only[key], contextual[key]) for key in sorted(only)]
    counts = {"rescued_by_context": 0, "harmed_by_context": 0, "both_correct": 0, "both_wrong": 0}
    for left, right in pairs:
        left_correct = _rank(left) == 1
        right_correct = _rank(right) == 1
        if not left_correct and right_correct:
            counts["rescued_by_context"] += 1
        elif left_correct and not right_correct:
            counts["harmed_by_context"] += 1
        elif left_correct:
            counts["both_correct"] += 1
        else:
            counts["both_wrong"] += 1

    only_metrics = compute_metrics(only.values())
    contextual_metrics = compute_metrics(contextual.values())
    paired_differences = {}
    for name in ("top1", "top3", "top5", "top10", "mrr_at_10"):
        paired_differences[name] = (
            contextual_metrics["micro"][name] - only_metrics["micro"][name]
        )

    users = sorted(contextual_metrics["per_user"])
    user_differences = {
        user_id: contextual_metrics["per_user"][user_id]["top1"]
        - only_metrics["per_user"][user_id]["top1"]
        for user_id in users
    }
    observed = mean(user_differences.values()) if user_differences else None
    samples: list[float] = []
    if users and bootstrap_resamples:
        rng = random.Random(seed)
        for _ in range(bootstrap_resamples):
            samples.append(
                mean(user_differences[rng.choice(users)] for _ in range(len(users)))
            )
    return {
        "analysis_label": "DEVELOPMENT analysis; not final confirmatory inference",
        "interaction_count": len(pairs),
        "paired_micro_differences_contextual_minus_pinyin_only": paired_differences,
        "top1_outcome_counts": counts,
        "macro_user_top1_difference": observed,
        "paired_user_bootstrap": {
            "resamples": bootstrap_resamples,
            "seed": seed,
            "confidence": 0.95,
            "lower": _percentile(samples, 0.025) if samples else None,
            "upper": _percentile(samples, 0.975) if samples else None,
        },
        "per_user_top1_difference": user_differences,
    }


def evaluate_breakdown(
    rows: Iterable[Mapping[str, Any]],
    group: Callable[[Mapping[str, Any]], str],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group(row)].append(row)
    return {
        key: compute_metrics(group_rows)["micro"]
        for key, group_rows in sorted(grouped.items())
    }
