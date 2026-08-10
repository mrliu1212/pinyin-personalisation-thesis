"""Deterministic metrics and chronological evaluation across ranking conditions."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from .base_ranker import InMemoryBaseRanker
from .data import Interaction
from .personal_model import EvidenceWeights, FrequencyPersonalModel
from .reranker import LinearReranker


@dataclass(frozen=True)
class EvaluationMetrics:
    evaluated_count: int
    top1_accuracy: float
    top3_accuracy: float
    mrr: float
    mean_target_rank: float | None
    missing_target_count: int


@dataclass(frozen=True)
class RerankingCounts:
    helpful: int = 0
    harmful: int = 0
    unchanged: int = 0


@dataclass(frozen=True)
class EvaluationRecord:
    user_id: str
    timestamp: datetime
    context: str
    pinyin: str
    target_candidate: str
    wrong_user_id: str
    correct_history_size: int
    wrong_history_size: int
    base_rank: int | None
    correct_user_rank: int | None
    wrong_user_rank: int | None


@dataclass(frozen=True)
class ConditionEvaluation:
    metrics: EvaluationMetrics
    reranking_counts: RerankingCounts | None = None


@dataclass(frozen=True)
class EvaluationComparison:
    base: ConditionEvaluation
    correct_user: ConditionEvaluation
    wrong_user: ConditionEvaluation
    records: tuple[EvaluationRecord, ...]


def target_rank(target: str, ranking: Sequence[str]) -> int | None:
    """Return the one-based target rank, or None when the target is absent."""
    try:
        return ranking.index(target) + 1
    except ValueError:
        return None


def compute_metrics(ranks: Sequence[int | None]) -> EvaluationMetrics:
    """Aggregate ranks with an explicit missing-target policy.

    Missing targets are incorrect for Top-K and contribute zero reciprocal rank.
    They are excluded from mean target rank and reported separately, avoiding an
    invented numeric rank.
    """
    count = len(ranks)
    if count == 0:
        return EvaluationMetrics(0, 0.0, 0.0, 0.0, None, 0)

    present_ranks = [rank for rank in ranks if rank is not None]
    return EvaluationMetrics(
        evaluated_count=count,
        top1_accuracy=sum(rank == 1 for rank in ranks) / count,
        top3_accuracy=sum(rank is not None and rank <= 3 for rank in ranks) / count,
        mrr=sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / count,
        mean_target_rank=(
            sum(present_ranks) / len(present_ranks) if present_ranks else None
        ),
        missing_target_count=count - len(present_ranks),
    )


def classify_rank_change(base_rank: int | None, personal_rank: int | None) -> str:
    """Classify a personalized rank without assigning a rank to absence."""
    if base_rank is None and personal_rank is None:
        return "unchanged"
    if base_rank is None:
        return "helpful"
    if personal_rank is None:
        return "harmful"
    if personal_rank < base_rank:
        return "helpful"
    if personal_rank > base_rank:
        return "harmful"
    return "unchanged"


def count_rank_changes(
    base_ranks: Sequence[int | None], personal_ranks: Sequence[int | None]
) -> RerankingCounts:
    if len(base_ranks) != len(personal_ranks):
        raise ValueError("base and personal rank sequences must have equal length")
    counts = {"helpful": 0, "harmful": 0, "unchanged": 0}
    for base_rank, personal_rank in zip(base_ranks, personal_ranks):
        counts[classify_rank_change(base_rank, personal_rank)] += 1
    return RerankingCounts(**counts)


def evaluate_chronologically(
    interactions: Sequence[Interaction],
    base_ranker: InMemoryBaseRanker,
    wrong_user_by_user: Mapping[str, str],
    *,
    alpha: float = 0.5,
    evidence_weights: EvidenceWeights | None = None,
) -> EvaluationComparison:
    """Evaluate every interaction using histories strictly before its timestamp."""
    ordered = sorted(
        interactions,
        key=lambda item: (
            item.timestamp,
            item.user_id,
            item.pinyin,
            item.context,
            item.target_candidate,
        ),
    )
    records: list[EvaluationRecord] = []

    for interaction in ordered:
        if interaction.user_id not in wrong_user_by_user:
            raise ValueError(f"missing wrong-user control for {interaction.user_id!r}")
        wrong_user_id = wrong_user_by_user[interaction.user_id]
        if wrong_user_id == interaction.user_id:
            raise ValueError("wrong-user control must identify a different user")

        earlier = [item for item in ordered if item.timestamp < interaction.timestamp]
        correct_history = [
            item for item in earlier if item.user_id == interaction.user_id
        ]
        wrong_history = [item for item in earlier if item.user_id == wrong_user_id]

        base_texts = [
            candidate.text
            for candidate in base_ranker.rank(interaction.context, interaction.pinyin)
        ]
        correct_model = FrequencyPersonalModel(evidence_weights).fit(
            ordered, interaction.user_id, before=interaction.timestamp
        )
        wrong_model = FrequencyPersonalModel(evidence_weights).fit(
            ordered, wrong_user_id, before=interaction.timestamp
        )
        correct_texts = [
            candidate.text
            for candidate in LinearReranker(base_ranker, correct_model, alpha).rank(
                interaction.context, interaction.pinyin
            )
        ]
        wrong_texts = [
            candidate.text
            for candidate in LinearReranker(base_ranker, wrong_model, alpha).rank(
                interaction.context, interaction.pinyin
            )
        ]

        records.append(
            EvaluationRecord(
                user_id=interaction.user_id,
                timestamp=interaction.timestamp,
                context=interaction.context,
                pinyin=interaction.pinyin,
                target_candidate=interaction.target_candidate,
                wrong_user_id=wrong_user_id,
                correct_history_size=len(correct_history),
                wrong_history_size=len(wrong_history),
                base_rank=target_rank(interaction.target_candidate, base_texts),
                correct_user_rank=target_rank(interaction.target_candidate, correct_texts),
                wrong_user_rank=target_rank(interaction.target_candidate, wrong_texts),
            )
        )

    base_ranks = [record.base_rank for record in records]
    correct_ranks = [record.correct_user_rank for record in records]
    wrong_ranks = [record.wrong_user_rank for record in records]
    return EvaluationComparison(
        base=ConditionEvaluation(metrics=compute_metrics(base_ranks)),
        correct_user=ConditionEvaluation(
            metrics=compute_metrics(correct_ranks),
            reranking_counts=count_rank_changes(base_ranks, correct_ranks),
        ),
        wrong_user=ConditionEvaluation(
            metrics=compute_metrics(wrong_ranks),
            reranking_counts=count_rank_changes(base_ranks, wrong_ranks),
        ),
        records=tuple(records),
    )

