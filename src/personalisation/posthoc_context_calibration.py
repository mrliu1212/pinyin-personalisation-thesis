"""Shared arithmetic for the frozen post-hoc context calibration study."""

from __future__ import annotations

from collections import defaultdict
import math
import statistics
from typing import Any, Mapping, Sequence

import numpy as np


def normalize_support(values: Mapping[str, float]) -> dict[str, float]:
    clipped = {str(key): max(0.0, float(value)) for key, value in values.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        return {key: (1.0 / len(clipped) if clipped else 0.0) for key in clipped}
    return {key: value / total for key, value in clipped.items()}


def cosine_top5_support(
    *,
    query_vector: np.ndarray,
    candidates: Sequence[str],
    visible: Sequence[Any],
    vectors: Mapping[str, np.ndarray],
    context_chars: int = 64,
    tau: float | None = None,
    normalize: bool = True,
) -> dict[str, float]:
    """Frozen cosine-only Top-5 retrieval with optional recency aggregation."""
    grouped: dict[str, list[Any]] = {str(candidate): [] for candidate in candidates}
    for item in visible:
        if item.record.target in grouped:
            grouped[item.record.target].append(item)

    raw = {str(candidate): 0.0 for candidate in candidates}
    for candidate, histories in grouped.items():
        history_vectors = []
        for item in histories:
            key = item.record.context[-context_chars:]
            if key not in vectors:
                raise KeyError(f"missing context vector for {item.record.row_id}")
            history_vectors.append(np.asarray(vectors[key], dtype=np.float32))
        if not history_vectors:
            continue
        # Preserve the frozen batched matrix operation. Separate vector dot
        # products can round near-ties differently and change Top-5 membership.
        similarities = np.vstack(history_vectors) @ query_vector
        order = sorted(
            range(len(histories)),
            key=lambda index: (
                -float(similarities[index]),
                int(histories[index].record.position),
                str(histories[index].record.row_id),
            ),
        )[:5]
        for index in order:
            similarity = float(similarities[index])
            age = int(histories[index].age)
            weight = 1.0 if tau is None else math.exp(-float(age) / float(tau))
            raw[candidate] += max(0.0, similarity) * weight
    return normalize_support(raw) if normalize else raw


def restrict_and_normalize(raw: Mapping[str, float], candidates: Sequence[str]) -> dict[str, float]:
    if not set(map(str, candidates)).issubset(set(map(str, raw))):
        raise ValueError("raw support does not cover requested candidate set")
    return normalize_support({str(candidate): float(raw[str(candidate)]) for candidate in candidates})


def rerank_fixed_surface(
    candidates: Sequence[Mapping[str, Any]],
    supports: Sequence[tuple[float, Mapping[str, float]]],
) -> list[dict[str, Any]]:
    names = [str(item["candidate"]) for item in candidates]
    for _weight, support in supports:
        if set(names) != set(map(str, support)):
            raise ValueError("support candidate set differs from frozen surface")
    output = []
    for index, item in enumerate(candidates, start=1):
        row = dict(item)
        name = str(item["candidate"])
        base_rank = int(item.get("base_rank", item.get("rank", index)))
        row["base_rank"] = base_rank
        row["final_score"] = float(item["final_score"]) + sum(
            float(weight) * float(support[name]) for weight, support in supports
        )
        output.append(row)
    output.sort(key=lambda row: (-float(row["final_score"]), int(row["base_rank"]), str(row["candidate"])))
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    return output


def merge_personal_recovery(
    *,
    generic_candidates: Sequence[Mapping[str, Any]],
    personal_k5: Sequence[str],
    personal_supports: Sequence[tuple[float, Mapping[str, float]]],
    boundary: float,
    tiebreak_support: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Frozen Generic/personal merge with a substituted recovery score."""
    generic_names = {str(item["candidate"]) for item in generic_candidates}
    if generic_names.intersection(personal_k5):
        raise ValueError("Personal-K5 overlaps Generic candidates")
    personal_names = set(map(str, personal_k5))
    for _weight, support in personal_supports:
        if set(map(str, support)) != personal_names:
            raise ValueError("personal support differs from frozen Personal-K5")
    tie_order = sorted(
        map(str, personal_k5),
        key=lambda name: (-float(tiebreak_support[name]), personal_k5.index(name), name),
    )
    tie_rank = {name: index for index, name in enumerate(tie_order, start=1)}
    rows = [dict(item) for item in generic_candidates]
    for original_rank, candidate in enumerate(map(str, personal_k5), start=1):
        rows.append(
            {
                "candidate": candidate,
                "source": "personal_recovery",
                "generic_rank": None,
                "personal_candidate_rank": tie_rank[candidate],
                "original_personal_frequency_rank": original_rank,
                "base_tiebreak_rank": tie_rank[candidate],
                "final_score": float(boundary)
                + sum(float(weight) * float(support[candidate]) for weight, support in personal_supports),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["final_score"]),
            0 if row.get("source") == "generic" else 1,
            int(row.get("generic_rank") or row.get("personal_candidate_rank") or 0),
            str(row["candidate"]),
        )
    )
    rows = rows[:10]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["base_rank"] = rank
    return rows


def rank_of(ranking: Sequence[Mapping[str, Any]] | Sequence[str], gold: str) -> int | None:
    for index, item in enumerate(ranking, start=1):
        text = str(item["candidate"]) if isinstance(item, Mapping) else str(item)
        if text == gold:
            return index
    return None


def metric_summary(rows: Sequence[Mapping[str, Any]], rank_key: str) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    per_author: dict[str, list[int | None]] = defaultdict(list)
    ranks = []
    for row in rows:
        value = row.get(rank_key)
        rank = None if value is None else int(value)
        ranks.append(rank)
        per_author[str(row["author"])].append(rank)

    def summarize(values: Sequence[int | None]) -> dict[str, float | int | None]:
        n = len(values)
        present = [rank for rank in values if rank is not None]
        return {
            "n": n,
            "micro_top1": sum(rank == 1 for rank in values) / n,
            "top3": sum(rank is not None and rank <= 3 for rank in values) / n,
            "top5": sum(rank is not None and rank <= 5 for rank in values) / n,
            "mrr_at_10": sum(0.0 if rank is None else 1.0 / rank for rank in values) / n,
            "missing10": sum(rank is None for rank in values) / n,
            "mean_present_rank": statistics.fmean(present) if present else None,
        }

    author_metrics = {author: summarize(values) for author, values in sorted(per_author.items())}
    result = summarize(ranks)
    result["macro_author_top1"] = statistics.fmean(float(value["micro_top1"]) for value in author_metrics.values())
    result["per_author_top1"] = {author: float(value["micro_top1"]) for author, value in author_metrics.items()}
    return result


def recovery_summary(rows: Sequence[Mapping[str, Any]], rank_key: str) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    ranks = [None if row.get(rank_key) is None else int(row[rank_key]) for row in rows]
    present = [rank for rank in ranks if rank is not None]
    per_author: dict[str, list[int | None]] = defaultdict(list)
    for row, rank in zip(rows, ranks):
        per_author[str(row["author"])].append(rank)
    return {
        "n": len(rows),
        "recovery_at_1": sum(rank == 1 for rank in ranks) / len(rows),
        "recovery_at_3": sum(rank is not None and rank <= 3 for rank in ranks) / len(rows),
        "recovery_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / len(rows),
        "recovery_at_10": sum(rank is not None and rank <= 10 for rank in ranks) / len(rows),
        "mrr": sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / len(rows),
        "mean_rank": statistics.fmean(present) if present else None,
        "per_author_recovery_at_5": {
            author: sum(rank is not None and rank <= 5 for rank in values) / len(values)
            for author, values in sorted(per_author.items())
        },
    }


def transition_counts(rows: Sequence[Mapping[str, Any]], before: str, after: str) -> dict[str, int]:
    rescue = sum(row.get(before) != 1 and row.get(after) == 1 for row in rows)
    harm = sum(row.get(before) == 1 and row.get(after) != 1 for row in rows)
    return {"n": len(rows), "rescue": rescue, "harm": harm, "net": rescue - harm}


def selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = row["metrics"]
    lambda_n = float(row.get("lambda_n", 0.0))
    lambda_e = float(row.get("lambda_e", 0.0))
    return (
        float(metrics["macro_author_top1"]),
        float(metrics["mrr_at_10"]),
        -(lambda_n + lambda_e),
        -lambda_e,
        -lambda_n,
    )
