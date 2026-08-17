"""Transparent chronological frequency and context-memory candidate ranking."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import statistics
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PredictionQuery:
    """Prediction-visible fields. Gold is deliberately absent."""

    row_id: str
    author: str
    work_id: str
    chronological_position: int
    context: str
    pinyin: tuple[str, ...]


@dataclass(frozen=True)
class Candidate:
    text: str
    generic_rank: int
    generic_score: float


def visible_same_pinyin_history(
    query: PredictionQuery,
    history: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Return only strictly prior, same-author, same-Pinyin records."""

    values = [
        record
        for record in history
        if str(record["author"]) == query.author
        and tuple(record["pinyin_segments"]) == query.pinyin
        and int(record["chronological_position"]) < query.chronological_position
    ]
    return tuple(sorted(values, key=lambda row: (int(row["chronological_position"]), str(row["row_id"]))))


def normalize_generic_scores(candidates: Sequence[Candidate]) -> tuple[float, ...]:
    """Per-query population z-score; a constant candidate set maps to zero."""

    values = [candidate.generic_score for candidate in candidates]
    mean = statistics.fmean(values)
    deviation = statistics.pstdev(values)
    if deviation == 0.0:
        return tuple(0.0 for _ in values)
    return tuple((value - mean) / deviation for value in values)


def _ranked(
    candidates: Sequence[Candidate],
    normalized: Sequence[float],
    personal: Mapping[str, float],
    weight: float,
    *,
    extra: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    rows = []
    for candidate, generic_component in zip(candidates, normalized):
        personal_component = float(personal.get(candidate.text, 0.0))
        rows.append(
            {
                "candidate": candidate.text,
                "generic_rank": candidate.generic_rank,
                "generic_score": candidate.generic_score,
                "normalized_generic_score": generic_component,
                "personal_score": personal_component,
                "final_score": generic_component + weight * personal_component,
                **(dict(extra.get(candidate.text, {})) if extra else {}),
            }
        )
    rows.sort(key=lambda row: (-float(row["final_score"]), int(row["generic_rank"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return tuple(rows)


def _generic_fallback(candidates: Sequence[Candidate]) -> tuple[dict[str, Any], ...]:
    normalized = normalize_generic_scores(candidates)
    return _ranked(candidates, normalized, {}, 0.0)


def rank_frequency(
    query: PredictionQuery,
    candidates: Sequence[Candidate],
    history: Sequence[Mapping[str, Any]],
    *,
    lambda_frequency: float,
) -> tuple[dict[str, Any], ...]:
    """Rank by normalized log(1+count) from visible same-Pinyin history."""

    visible = visible_same_pinyin_history(query, history)
    if not visible:
        return _generic_fallback(candidates)
    counts = Counter(str(record["target"]) for record in visible)
    raw = {candidate.text: math.log1p(counts[candidate.text]) for candidate in candidates}
    maximum = max(raw.values(), default=0.0)
    support = {text: value / maximum for text, value in raw.items()} if maximum else {}
    extra = {candidate.text: {"frequency_count": counts[candidate.text]} for candidate in candidates}
    return _ranked(candidates, normalize_generic_scores(candidates), support, lambda_frequency, extra=extra)


def _normalized(vector: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(float(value) ** 2 for value in vector))
    if norm == 0.0:
        raise ValueError("zero-length embedding vector")
    return tuple(float(value) / norm for value in vector)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    left_normalized = _normalized(left)
    right_normalized = _normalized(right)
    if len(left_normalized) != len(right_normalized):
        raise ValueError("embedding dimensions differ")
    return sum(a * b for a, b in zip(left_normalized, right_normalized))


def rank_memory(
    query: PredictionQuery,
    candidates: Sequence[Candidate],
    history: Sequence[Mapping[str, Any]],
    embeddings: Mapping[str, Sequence[float]],
    *,
    top_n: int,
    lambda_memory: float,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Retrieve deterministic Top-N history and rank by bounded positive support."""

    visible = visible_same_pinyin_history(query, history)
    if not visible:
        return _generic_fallback(candidates), ()
    retrieved = retrieve_memory(query, visible, embeddings)
    selected = tuple(retrieved[:top_n])
    ranked = rank_from_retrieved(candidates, selected, lambda_memory=lambda_memory)
    return ranked, selected


def retrieve_memory(
    query: PredictionQuery,
    visible: Sequence[Mapping[str, Any]],
    embeddings: Mapping[str, Sequence[float]],
) -> tuple[dict[str, Any], ...]:
    """Score and deterministically sort already visibility-filtered history."""

    query_vector = embeddings[query.context]
    retrieved = []
    for record in visible:
        similarity = cosine_similarity(query_vector, embeddings[str(record["context"])])
        retrieved.append(
            {
                "historical_interaction_id": str(record["row_id"]),
                "historical_target": str(record["target"]),
                "similarity": similarity,
                "weight": max(similarity, 0.0),
                "chronological_position": int(record["chronological_position"]),
            }
        )
    retrieved.sort(
        key=lambda row: (
            -float(row["similarity"]),
            int(row["chronological_position"]),
            str(row["historical_interaction_id"]),
        )
    )
    return tuple(retrieved)


def rank_from_retrieved(
    candidates: Sequence[Candidate],
    retrieved: Sequence[Mapping[str, Any]],
    *,
    lambda_memory: float,
) -> tuple[dict[str, Any], ...]:
    """Aggregate a fixed retrieved surface and apply the memory weight."""

    total_weight = sum(float(row["weight"]) for row in retrieved)
    support: dict[str, float] = defaultdict(float)
    if total_weight:
        for row in retrieved:
            support[str(row["historical_target"])] += float(row["weight"]) / total_weight
    return _ranked(candidates, normalize_generic_scores(candidates), support, lambda_memory)


def subset_membership(
    query: PredictionQuery,
    gold: str,
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute post-prediction diagnostic subsets; ties are excluded from Conflict."""

    visible = visible_same_pinyin_history(query, history)
    counts = Counter(str(record["target"]) for record in visible)
    ambiguous = len(counts) >= 2
    winner = None
    tied = False
    if counts:
        maximum = max(counts.values())
        winners = sorted(target for target, count in counts.items() if count == maximum)
        tied = len(winners) != 1
        if not tied:
            winner = winners[0]
    conflict = ambiguous and winner is not None and gold != winner
    return {
        "history_available": bool(visible),
        "visible_history_count": len(visible),
        "distinct_historical_targets": len(counts),
        "ambiguous": ambiguous,
        "frequency_winner": winner,
        "frequency_winner_tied": tied,
        "conflict": conflict,
    }


def assert_candidate_pool(
    generic: Sequence[Candidate],
    frequency: Sequence[Mapping[str, Any]],
    memory: Sequence[Mapping[str, Any]],
) -> None:
    expected = {candidate.text for candidate in generic}
    if {str(row["candidate"]) for row in frequency} != expected:
        raise AssertionError("Frequency changed the frozen Generic candidate set")
    if {str(row["candidate"]) for row in memory} != expected:
        raise AssertionError("Memory changed the frozen Generic candidate set")


def rank_of(rows: Sequence[Mapping[str, Any]], gold: str) -> int | None:
    return next((int(row["rank"]) for row in rows if str(row["candidate"]) == gold), None)


def metric_values(ranks: Sequence[int | None]) -> dict[str, float | int | None]:
    if not ranks:
        return {"n": 0, "top1": None, "top3": None, "mrr_at_10": None, "missing_at_10": None, "mean_rank_given_top10": None}
    found = [rank for rank in ranks if rank is not None]
    count = len(ranks)
    return {
        "n": count,
        "top1": sum(rank == 1 for rank in ranks) / count,
        "top3": sum(rank is not None and rank <= 3 for rank in ranks) / count,
        "mrr_at_10": sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / count,
        "missing_at_10": sum(rank is None for rank in ranks) / count,
        "mean_rank_given_top10": statistics.fmean(found) if found else None,
    }


def macro_author_metrics(rows: Sequence[Mapping[str, Any]], rank_key: str) -> dict[str, Any]:
    by_author: dict[str, list[int | None]] = defaultdict(list)
    for row in rows:
        by_author[str(row["author"])].append(row.get(rank_key))
    per_author = {author: metric_values(ranks) for author, ranks in sorted(by_author.items())}
    names = ("top1", "top3", "mrr_at_10", "missing_at_10", "mean_rank_given_top10")
    macro = {
        name: statistics.fmean(float(values[name]) for values in per_author.values() if values[name] is not None)
        if any(values[name] is not None for values in per_author.values())
        else None
        for name in names
    }
    macro["n"] = len(rows)
    macro["authors_with_rows"] = len(per_author)
    return {"macro_author": macro, "per_author": per_author}
