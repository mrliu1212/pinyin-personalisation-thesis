"""Construct traceable lexical interactions without train/test splitting."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from .candidates import CandidateGenerator
from .linguistic import (
    JiebaSegmenter,
    TargetPolicy,
    convert_full_pinyin,
    derived_context,
    exclusion_reason,
)


@dataclass(frozen=True)
class ConstructionResult:
    interactions: tuple[dict[str, Any], ...]
    exclusions: dict[str, int]
    eligible_before_limit: int
    pinyin_failures: int


def target_rank(target: str, candidates: list[dict[str, Any]]) -> int | None:
    for candidate in candidates:
        if candidate["text"] == target:
            return int(candidate["base_rank"])
    return None


def construct_work_interactions(
    text: str,
    work: dict[str, Any],
    segmenter: JiebaSegmenter,
    candidate_generator: CandidateGenerator,
    policy: TargetPolicy,
    *,
    max_interactions: int | None = None,
) -> ConstructionResult:
    exclusions: Counter[str] = Counter()
    eligible = []
    for token in segmenter.segment(text):
        reason = exclusion_reason(token.text, policy)
        if reason:
            exclusions[reason] += 1
        else:
            eligible.append(token)
    selected = eligible if max_interactions is None else eligible[:max_interactions]
    interactions: list[dict[str, Any]] = []
    pinyin_failures = 0

    for token in selected:
        try:
            conversion = convert_full_pinyin(token.text)
        except ValueError:
            exclusions["pinyin_conversion_failed"] += 1
            pinyin_failures += 1
            continue
        candidates = [asdict(item) for item in candidate_generator.candidates(conversion.normalized)]
        rank = target_rank(token.text, candidates)
        identity = (
            f"{work['author_id']}|{work['work_id']}|{token.start}|{token.end}|"
            f"{token.text}|{conversion.normalized}"
        )
        raw_context = text[: token.start]
        interactions.append(
            {
                "schema_version": 1,
                "interaction_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
                "author_id": work["author_id"],
                "author_name": work["author_name"],
                "work_id": work["work_id"],
                "work_title": work["work_title"],
                "work_date": work["chronology"]["value"],
                "work_date_precision": work["chronology"]["precision"],
                "work_date_certainty": work["chronology"]["certainty"],
                "source_processed_file": work["processed_file"],
                "source_page_url": work["source_page_url"],
                "source_revision_id": work["source_revision_id"],
                "source_start_offset": token.start,
                "source_end_offset": token.end,
                "raw_context": raw_context,
                "derived_context": derived_context(
                    raw_context, policy.derived_context_characters
                ),
                "target_candidate": token.text,
                "target_length": len(token.text),
                "pinyin": conversion.normalized,
                "pinyin_syllables": list(conversion.syllables),
                "pinyin_method": "pypinyin.Style.NORMAL; tone-free full Pinyin",
                "polyphonic_review_required": bool(conversion.polyphonic_characters),
                "polyphonic_characters": list(conversion.polyphonic_characters),
                "candidates": candidates,
                "candidate_list_size": len(candidates),
                "target_rank": rank,
                "target_present": rank is not None,
            }
        )
    return ConstructionResult(
        interactions=tuple(interactions),
        exclusions=dict(sorted(exclusions.items())),
        eligible_before_limit=len(eligible),
        pinyin_failures=pinyin_failures,
    )

