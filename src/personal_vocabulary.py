"""Deterministic personal-vocabulary augmentation for Phase 4E."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Sequence


MAX_PERSONAL_VOCAB_INJECTIONS = 3


@dataclass(frozen=True)
class PersonalVocabularyCandidate:
    candidate: str
    candidate_source: str
    same_pinyin_count: int
    most_recent_history_index: int
    provenance_interaction_ids: tuple[str, ...]


class PersonalVocabulary:
    def __init__(self, records: Sequence[dict[str, Any]], *, user_id: str) -> None:
        if any(record.get("author_id") != user_id for record in records):
            raise ValueError("personal vocabulary cannot mix users")
        self.user_id = user_id
        counts: Counter[tuple[str, str]] = Counter()
        recent: dict[tuple[str, str], int] = {}
        provenance: dict[tuple[str, str], list[str]] = defaultdict(list)
        for index, record in enumerate(records):
            key = (record["pinyin"], record["target_candidate"])
            counts[key] += 1
            recent[key] = index
            provenance[key].append(record["interaction_id"])
        self._counts = counts
        self._recent = recent
        self._provenance = provenance

    def inject(
        self,
        pinyin: str,
        luna_candidates: Sequence[str],
        *,
        maximum: int = MAX_PERSONAL_VOCAB_INJECTIONS,
    ) -> tuple[PersonalVocabularyCandidate, ...]:
        if maximum != MAX_PERSONAL_VOCAB_INJECTIONS:
            raise ValueError(
                "Phase 4E personal-vocabulary maximum is frozen at "
                f"{MAX_PERSONAL_VOCAB_INJECTIONS}"
            )
        native = set(luna_candidates)
        eligible = [
            candidate
            for (candidate_pinyin, candidate), count in self._counts.items()
            if candidate_pinyin == pinyin and candidate not in native and count > 0
        ]
        eligible.sort(
            key=lambda candidate: (
                -self._counts[(pinyin, candidate)],
                -self._recent[(pinyin, candidate)],
                candidate,
            )
        )
        return tuple(
            PersonalVocabularyCandidate(
                candidate=candidate,
                candidate_source="personal_vocabulary",
                same_pinyin_count=self._counts[(pinyin, candidate)],
                most_recent_history_index=self._recent[(pinyin, candidate)],
                provenance_interaction_ids=tuple(self._provenance[(pinyin, candidate)]),
            )
            for candidate in eligible[:MAX_PERSONAL_VOCAB_INJECTIONS]
        )
