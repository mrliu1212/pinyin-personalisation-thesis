"""Stable, serialisable decision and provenance records for Phase 4F."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence


def stable_digest(*parts: object, bytes_count: int = 12) -> str:
    payload = "\n".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[: bytes_count * 2]


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class GeneratedCandidate:
    text: str
    rank: int
    generation_score: float | None
    seed: int
    source: str
    raw_text: str


@dataclass(frozen=True)
class TriggerDecision:
    should_retrieve: bool
    query: str | None
    method: str
    raw_evidence: str | None = None


@dataclass(frozen=True)
class RetrievedMemory:
    memory_id: str
    user_id: str
    vector_label: int
    plaintext: str
    raw_cosine: float
    similarity_01: float
    lexical_score: float
    combined_score: float
    source_interaction_ids: tuple[str, ...]


@dataclass(frozen=True)
class TimingBreakdown:
    direct_generation_ms: float = 0.0
    query_embedding_ms: float = 0.0
    hnsw_search_ms: float = 0.0
    grounded_generation_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(frozen=True)
class PredictionResult:
    query_id: str
    user_id: str
    preceding_text: str
    pinyin_or_keystrokes: str
    external_context: str | None
    candidates: tuple[GeneratedCandidate, ...]
    memory_trigger: TriggerDecision
    retrieved_memories: tuple[RetrievedMemory, ...]
    supplied_memory_ids: tuple[str, ...]
    supplied_memory_plaintext: tuple[str, ...]
    model_runtime: Mapping[str, Any]
    timing: TimingBreakdown
    cache_status: Mapping[str, Any]
    direct_candidates: tuple[GeneratedCandidate, ...] = ()
    grounded_candidates: tuple[GeneratedCandidate, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_query_id(
    user_id: str,
    preceding_text: str,
    pinyin_or_keystrokes: str,
    external_context: str | None,
    request_nonce: str = "",
) -> str:
    return stable_digest(
        "phase_04f_query",
        user_id,
        preceding_text,
        pinyin_or_keystrokes,
        external_context or "",
        request_nonce,
    )


def stable_interaction_id(
    user_id: str,
    chronological_position: str,
    preceding_text: str,
    pinyin_or_keystrokes: str,
    selected_text: str | None,
) -> str:
    return stable_digest(
        "phase_04f_interaction",
        user_id,
        chronological_position,
        preceding_text,
        pinyin_or_keystrokes,
        selected_text or "",
    )


def tupled(values: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(values or ())
