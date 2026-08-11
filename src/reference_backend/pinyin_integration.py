"""Separate-channel Pinyin + HuoziIME integration grounded in upstream UI semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any

from .backend import ReferencePersonalisedIMEBackend
from .pinyin_decoder import (
    PinyinDecoder,
    PinyinDecoderCandidate,
    PinyinDecoderResult,
    normalize_pinyin,
)
from .provenance import GeneratedCandidate, PredictionResult


HUOZIIME_DIRECT_SOURCE = "HUOZIIME_DIRECT"
HUOZIIME_MEMORY_GROUNDED_SOURCE = "HUOZIIME_MEMORY_GROUNDED"
HUOZIIME_NO_MEMORY_RERUN_SOURCE = "HUOZIIME_NO_MEMORY_RERUN"
INTEGRATION_MODE = "FAITHFUL_DESKTOP_ADAPTATION_SEPARATE_CHANNELS"


@dataclass(frozen=True)
class MultiSourceCandidateProvenance:
    text: str
    sources: tuple[str, ...]
    pinyin_rank: int | None
    huoziime_direct_rank: int | None
    huoziime_grounded_rank: int | None
    huoziime_no_memory_rerun_rank: int | None
    grounded_memory_ids: tuple[str, ...]


@dataclass(frozen=True)
class IntegratedPredictionResult:
    normalized_pinyin: str
    pinyin_decoder: PinyinDecoderResult
    huoziime: PredictionResult
    pinyin_candidates: tuple[PinyinDecoderCandidate, ...]
    huoziime_direct_suggestions: tuple[GeneratedCandidate, ...]
    huoziime_grounded_suggestions: tuple[GeneratedCandidate, ...]
    huoziime_final_suggestions: tuple[GeneratedCandidate, ...]
    candidate_provenance: tuple[MultiSourceCandidateProvenance, ...]
    integration_mode: str
    channels_unified: bool
    latency_breakdown: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def integrate_separate_channels(
    pinyin_result: PinyinDecoderResult,
    huoziime_result: PredictionResult,
    *,
    total_ms: float | None = None,
) -> IntegratedPredictionResult:
    """Join channel records for provenance only; never create a shared ranking."""
    if normalize_pinyin(huoziime_result.pinyin_or_keystrokes) != pinyin_result.normalized_pinyin:
        raise ValueError("Pinyin and HuoziIME channel records belong to different queries")
    sources: dict[str, set[str]] = {}
    pinyin_ranks: dict[str, int] = {}
    direct_ranks: dict[str, int] = {}
    grounded_ranks: dict[str, int] = {}
    no_memory_rerun_ranks: dict[str, int] = {}
    order: list[str] = []

    def observe(text: str, source: str) -> None:
        if text not in sources:
            sources[text] = set()
            order.append(text)
        sources[text].add(source)

    for candidate in pinyin_result.candidates:
        observe(candidate.text, "PINYIN_DECODER")
        pinyin_ranks.setdefault(candidate.text, candidate.rank)
    for candidate in huoziime_result.direct_candidates:
        observe(candidate.text, HUOZIIME_DIRECT_SOURCE)
        direct_ranks.setdefault(candidate.text, candidate.rank)
    for candidate in huoziime_result.grounded_candidates:
        observe(candidate.text, HUOZIIME_MEMORY_GROUNDED_SOURCE)
        grounded_ranks.setdefault(candidate.text, candidate.rank)
    if huoziime_result.provenance.get("path") == "memory_rerun_no_hit":
        for candidate in huoziime_result.candidates:
            observe(candidate.text, HUOZIIME_NO_MEMORY_RERUN_SOURCE)
            no_memory_rerun_ranks.setdefault(candidate.text, candidate.rank)

    provenance = tuple(
        MultiSourceCandidateProvenance(
            text=text,
            sources=tuple(sorted(sources[text])),
            pinyin_rank=pinyin_ranks.get(text),
            huoziime_direct_rank=direct_ranks.get(text),
            huoziime_grounded_rank=grounded_ranks.get(text),
            huoziime_no_memory_rerun_rank=no_memory_rerun_ranks.get(text),
            grounded_memory_ids=(
                huoziime_result.supplied_memory_ids if text in grounded_ranks else ()
            ),
        )
        for text in order
    )
    return IntegratedPredictionResult(
        normalized_pinyin=pinyin_result.normalized_pinyin,
        pinyin_decoder=pinyin_result,
        huoziime=huoziime_result,
        pinyin_candidates=pinyin_result.candidates,
        huoziime_direct_suggestions=huoziime_result.direct_candidates,
        huoziime_grounded_suggestions=huoziime_result.grounded_candidates,
        huoziime_final_suggestions=huoziime_result.candidates,
        candidate_provenance=provenance,
        integration_mode=INTEGRATION_MODE,
        channels_unified=False,
        latency_breakdown={
            "pinyin_decoder_ms": pinyin_result.latency_ms,
            "huoziime": asdict(huoziime_result.timing),
            "total_ms": (
                total_ms
                if total_ms is not None
                else pinyin_result.latency_ms + huoziime_result.timing.total_ms
            ),
        },
    )


class PinyinIntegratedReferenceBackend:
    """Public two-channel API preserving the validated HuoziIME core."""

    def __init__(
        self,
        *,
        pinyin_decoder: PinyinDecoder,
        huoziime_backend: ReferencePersonalisedIMEBackend,
    ) -> None:
        self.pinyin_decoder = pinyin_decoder
        self.huoziime_backend = huoziime_backend

    def predict(
        self,
        user_id: str,
        preceding_text: str,
        pinyin_or_keystrokes: str,
        top_k: int,
        external_context: str | None = None,
        **huoziime_options: Any,
    ) -> IntegratedPredictionResult:
        start = time.perf_counter()
        decoded = self.pinyin_decoder.decode(pinyin_or_keystrokes, top_k=top_k)
        huoziime = self.huoziime_backend.predict(
            user_id,
            preceding_text,
            pinyin_or_keystrokes,
            top_k,
            external_context=external_context,
            **huoziime_options,
        )
        return integrate_separate_channels(
            decoded,
            huoziime,
            total_ms=(time.perf_counter() - start) * 1000.0,
        )
