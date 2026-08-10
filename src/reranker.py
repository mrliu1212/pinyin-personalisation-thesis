"""Linear interpolation of base and personal candidate scores."""

from .base_ranker import InMemoryBaseRanker
from .data import RankedCandidate
from .personal_model import FrequencyPersonalModel


def _min_max(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if low == high:
        return [0.0] * len(values)
    return [(value - low) / (high - low) for value in values]


class LinearReranker:
    def __init__(
        self,
        base_ranker: InMemoryBaseRanker,
        personal_model: FrequencyPersonalModel,
        alpha: float = 0.5,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self._base_ranker = base_ranker
        self._personal_model = personal_model
        self._alpha = alpha

    def rank(
        self, context: str, pinyin: str, top_k: int | None = None
    ) -> list[RankedCandidate]:
        base_candidates = self._base_ranker.rank(context, pinyin)
        base_scores = [candidate.base_score for candidate in base_candidates]
        personal_details = [
            self._personal_model.score_details(pinyin, candidate.text, context)
            for candidate in base_candidates
        ]
        personal_scores = [detail.combined_score for detail in personal_details]
        normalized_base = _min_max(base_scores)
        normalized_personal = _min_max(personal_scores)

        ranked = [
            RankedCandidate(
                text=candidate.text,
                base_score=candidate.base_score,
                global_evidence=detail.global_evidence,
                pinyin_evidence=detail.pinyin_evidence,
                context_evidence=detail.context_evidence,
                personal_score=detail.combined_score,
                final_score=(
                    self._alpha * base_score
                    + (1.0 - self._alpha) * personal_score_normalized
                ),
            )
            for candidate, detail, base_score, personal_score_normalized in zip(
                base_candidates,
                personal_details,
                normalized_base,
                normalized_personal,
            )
        ]
        ranked.sort(
            key=lambda candidate: (candidate.final_score, candidate.base_score), reverse=True
        )
        return ranked if top_k is None else ranked[:top_k]
