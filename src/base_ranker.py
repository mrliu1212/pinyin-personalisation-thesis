"""A deterministic in-memory base candidate provider."""

from collections.abc import Mapping, Sequence

from .data import BaseCandidate


class InMemoryBaseRanker:
    def __init__(self, lexicon: Mapping[str, Sequence[BaseCandidate]]) -> None:
        self._lexicon = {pinyin: tuple(candidates) for pinyin, candidates in lexicon.items()}

    def rank(
        self, context: str, pinyin: str, top_k: int | None = None
    ) -> list[BaseCandidate]:
        del context  # Reserved for a future context-sensitive base ranker.
        candidates = sorted(
            self._lexicon.get(pinyin, ()), key=lambda candidate: candidate.base_score, reverse=True
        )
        return candidates if top_k is None else candidates[:top_k]

