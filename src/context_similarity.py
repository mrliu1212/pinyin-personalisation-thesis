"""Deterministic character TF-IDF retrieval for transparent user memory."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import math
from typing import Iterable, Mapping, Sequence


NGRAM_RANGE = (1, 2)
RETRIEVAL_K = 5


def character_ngrams(text: str) -> list[str]:
    """Return character unigrams and bigrams using the frozen analyzer."""
    return [
        text[start : start + size]
        for size in range(NGRAM_RANGE[0], NGRAM_RANGE[1] + 1)
        for start in range(0, len(text) - size + 1)
    ]


class CharacterTfidf:
    """Small sparse TF-IDF implementation fitted only on supplied contexts."""

    def __init__(self) -> None:
        self.idf: dict[str, float] = {}
        self.document_count = 0

    def fit(self, contexts: Sequence[str]) -> "CharacterTfidf":
        self.document_count = len(contexts)
        document_frequency: Counter[str] = Counter()
        for context in contexts:
            document_frequency.update(set(character_ngrams(context)))
        self.idf = {
            term: math.log(
                (1.0 + self.document_count) / (1.0 + document_frequency[term])
            )
            + 1.0
            for term in sorted(document_frequency)
        }
        return self

    def transform(self, context: str) -> dict[str, float]:
        counts = Counter(character_ngrams(context))
        weighted = {
            term: float(count) * self.idf[term]
            for term, count in counts.items()
            if term in self.idf
        }
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        if norm == 0.0:
            return {}
        return {term: value / norm for term, value in weighted.items()}

    @staticmethod
    def cosine(
        left: Mapping[str, float], right: Mapping[str, float]
    ) -> float:
        if not left or not right:
            return 0.0
        if len(left) > len(right):
            left, right = right, left
        similarity = sum(value * right.get(term, 0.0) for term, value in left.items())
        return min(1.0, max(0.0, similarity))


@dataclass(frozen=True)
class MemoryInteraction:
    interaction_id: str
    user_id: str
    timestamp: datetime
    context: str
    pinyin: str
    selected_candidate: str
    work_id: str = ""


@dataclass(frozen=True)
class RetrievedInteraction:
    interaction: MemoryInteraction
    similarity: float


class ContextualMemory:
    """A single-user, frozen history with same-Pinyin Top-K retrieval."""

    def __init__(
        self, interactions: Iterable[MemoryInteraction], *, user_id: str
    ) -> None:
        ordered = tuple(
            sorted(interactions, key=lambda item: (item.timestamp, item.interaction_id))
        )
        if any(item.user_id != user_id for item in ordered):
            raise ValueError("contextual memory cannot mix users")
        self.user_id = user_id
        self.interactions = ordered
        self.tfidf = CharacterTfidf().fit([item.context for item in ordered])
        self._vectors = {
            item.interaction_id: self.tfidf.transform(item.context) for item in ordered
        }
        by_pinyin: dict[str, list[MemoryInteraction]] = defaultdict(list)
        for item in ordered:
            by_pinyin[item.pinyin].append(item)
        self._by_pinyin = {key: tuple(value) for key, value in by_pinyin.items()}

    def eligible_count(self, pinyin: str) -> int:
        return len(self._by_pinyin.get(pinyin, ()))

    def retrieve(
        self, current_context: str, current_pinyin: str, *, k: int = RETRIEVAL_K
    ) -> tuple[RetrievedInteraction, ...]:
        if k != RETRIEVAL_K:
            raise ValueError(f"Phase 4D retrieval K is frozen at {RETRIEVAL_K}")
        query = self.tfidf.transform(current_context)
        scored = []
        for item in self._by_pinyin.get(current_pinyin, ()):
            similarity = self.tfidf.cosine(query, self._vectors[item.interaction_id])
            if similarity > 0.0:
                scored.append(RetrievedInteraction(item, similarity))
        scored.sort(
            key=lambda result: (
                -result.similarity,
                result.interaction.timestamp,
                result.interaction.interaction_id,
            )
        )
        return tuple(scored[:RETRIEVAL_K])


def contextual_candidate_evidence(
    retrieved: Sequence[RetrievedInteraction], candidates: Sequence[str]
) -> dict[str, float]:
    denominator = sum(item.similarity for item in retrieved)
    if denominator == 0.0:
        return {candidate: 0.0 for candidate in candidates}
    contributions: dict[str, float] = defaultdict(float)
    for item in retrieved:
        contributions[item.interaction.selected_candidate] += item.similarity
    return {
        candidate: min(1.0, max(0.0, contributions[candidate] / denominator))
        for candidate in candidates
    }
