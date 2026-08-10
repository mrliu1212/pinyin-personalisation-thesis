"""Interpretable frequency-based personal scoring."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from .data import Interaction


@dataclass(frozen=True)
class EvidenceWeights:
    """Weights for the three directly inspectable frequency components."""

    global_weight: float = 0.1
    pinyin_weight: float = 0.3
    context_weight: float = 0.6

    def __post_init__(self) -> None:
        if min(self.global_weight, self.pinyin_weight, self.context_weight) < 0:
            raise ValueError("evidence weights must be non-negative")
        if self.global_weight + self.pinyin_weight + self.context_weight == 0:
            raise ValueError("at least one evidence weight must be positive")


@dataclass(frozen=True)
class PersonalScore:
    global_evidence: float
    pinyin_evidence: float
    context_evidence: float
    combined_score: float


class FrequencyPersonalModel:
    """Combine global, Pinyin, and exact-context selection frequencies."""

    def __init__(self, weights: EvidenceWeights | None = None) -> None:
        self.weights = weights or EvidenceWeights()
        self._global_counts: Counter[str] = Counter()
        self._pinyin_counts: Counter[tuple[str, str]] = Counter()
        self._context_counts: Counter[tuple[str, str, str]] = Counter()

    def fit(
        self,
        history: Iterable[Interaction],
        user_id: str,
        before: datetime | None = None,
    ) -> "FrequencyPersonalModel":
        self._global_counts.clear()
        self._pinyin_counts.clear()
        self._context_counts.clear()
        for interaction in sorted(history, key=lambda item: item.timestamp):
            if interaction.user_id != user_id:
                continue
            if before is not None and interaction.timestamp >= before:
                continue
            candidate = interaction.target_candidate
            self._global_counts[candidate] += 1
            self._pinyin_counts[(interaction.pinyin, candidate)] += 1
            self._context_counts[(interaction.context, interaction.pinyin, candidate)] += 1
        return self

    def score(self, pinyin: str, candidate: str, context: str = "") -> float:
        """Return the combined score while preserving the Phase 1 interface."""
        return self.score_details(pinyin, candidate, context).combined_score

    def score_details(
        self, pinyin: str, candidate: str, context: str = ""
    ) -> PersonalScore:
        global_evidence = float(self._global_counts[candidate])
        pinyin_evidence = float(self._pinyin_counts[(pinyin, candidate)])
        context_evidence = float(self._context_counts[(context, pinyin, candidate)])
        combined_score = (
            self.weights.global_weight * global_evidence
            + self.weights.pinyin_weight * pinyin_evidence
            + self.weights.context_weight * context_evidence
        )
        return PersonalScore(
            global_evidence=global_evidence,
            pinyin_evidence=pinyin_evidence,
            context_evidence=context_evidence,
            combined_score=combined_score,
        )
