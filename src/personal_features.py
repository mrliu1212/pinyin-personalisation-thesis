"""Transparent chronological behavioural features for Phase 4E."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Sequence


@dataclass(frozen=True)
class BehaviourFeatures:
    global_count: int
    same_pinyin_count: int
    log1p_global_count: float
    log1p_same_pinyin_count: float
    same_pinyin_selection_share: float
    candidate_seen_same_pinyin: float
    recency: float


class BehaviouralHistory:
    """Features computed from one already-ordered active-user history."""

    def __init__(self, records: Sequence[dict[str, Any]], *, user_id: str) -> None:
        if any(record.get("author_id") != user_id for record in records):
            raise ValueError("behavioural history cannot mix users")
        self.records = tuple(records)
        self.user_id = user_id
        self.global_counts = Counter(record["target_candidate"] for record in records)
        self.pinyin_counts = Counter(
            (record["pinyin"], record["target_candidate"]) for record in records
        )
        self.pinyin_totals = Counter(record["pinyin"] for record in records)
        self.last_selection_index: dict[tuple[str, str], int] = {}
        for index, record in enumerate(records):
            self.last_selection_index[(record["pinyin"], record["target_candidate"])] = index

    def features(self, pinyin: str, candidate: str) -> BehaviourFeatures:
        global_count = self.global_counts[candidate]
        same_pinyin_count = self.pinyin_counts[(pinyin, candidate)]
        denominator = self.pinyin_totals[pinyin]
        last_index = self.last_selection_index.get((pinyin, candidate))
        if last_index is None:
            recency = 0.0
        else:
            interactions_since = len(self.records) - 1 - last_index
            recency = 1.0 / (1.0 + interactions_since)
        return BehaviourFeatures(
            global_count=global_count,
            same_pinyin_count=same_pinyin_count,
            log1p_global_count=math.log1p(global_count),
            log1p_same_pinyin_count=math.log1p(same_pinyin_count),
            same_pinyin_selection_share=(
                same_pinyin_count / denominator if denominator else 0.0
            ),
            candidate_seen_same_pinyin=float(same_pinyin_count > 0),
            recency=recency,
        )
