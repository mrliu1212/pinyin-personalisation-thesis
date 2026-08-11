"""Mature Pinyin-decoder boundary for the Phase 4F.1 desktop adaptation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import time
from typing import Any, Protocol
import unicodedata

from interactions.candidates import RimeCliCandidateGenerator


PINYIN_DECODER_SOURCE = "PINYIN_DECODER"
_NORMALIZED_PINYIN = re.compile(r"[a-z']+")


def normalize_pinyin(value: str) -> str:
    """Normalize already tone-free full Pinyin without guessing a pronunciation."""
    if not isinstance(value, str):
        raise TypeError("Pinyin input must be a string")
    normalized = unicodedata.normalize("NFKC", value).lower().replace("u:", "v")
    normalized = normalized.replace("ü", "v")
    normalized = "".join(normalized.split()).strip("'")
    if not normalized or _NORMALIZED_PINYIN.fullmatch(normalized) is None:
        raise ValueError("Pinyin must be non-empty, tone-free letters/apostrophes")
    return normalized


@dataclass(frozen=True)
class PinyinDecoderCandidate:
    text: str
    rank: int
    source: str = PINYIN_DECODER_SOURCE


@dataclass(frozen=True)
class PinyinDecoderResult:
    raw_input: str
    normalized_pinyin: str
    consumed_input: str
    candidates: tuple[PinyinDecoderCandidate, ...]
    latency_ms: float
    decoder: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PinyinDecoder(Protocol):
    max_candidates: int

    def decode(self, pinyin_or_keystrokes: str, *, top_k: int) -> PinyinDecoderResult: ...


class LibrimeLunaPinyinDecoder:
    """Pinned Luna Pinyin through the same mature librime API family as YuyanIME.

    This is explicitly a faithful desktop adaptation, not Yuyan's exact Android
    dictionary or an official unified HuoziIME ranking.
    """

    INTEGRATION_STATUS = "FAITHFUL DESKTOP ADAPTATION"
    SCHEMA_ID = "luna_pinyin"
    SIMPLIFIED_OPTION = "zh_hans"
    DICTIONARY = "rime-luna-pinyin (pinned by data/rime/setup_manifest.json)"
    DICTIONARY_REVISION = "rime-luna-pinyin@56b934b099dfbeab842320f13aa8b461a6ab3e42"

    def __init__(
        self,
        *,
        executable: Path,
        shared_data: Path,
        prebuilt_data: Path,
        version: str,
        max_candidates: int = 10,
        setup_manifest: str = "data/rime/setup_manifest.json",
    ) -> None:
        self.max_candidates = max_candidates
        self.version = version
        self.setup_manifest = setup_manifest
        self._generator = RimeCliCandidateGenerator(
            executable=executable,
            shared_data=shared_data,
            prebuilt_data=prebuilt_data,
            version=version,
            schema_id=self.SCHEMA_ID,
            max_candidates=max_candidates,
            enabled_options=(self.SIMPLIFIED_OPTION,),
        )

    def decode(self, pinyin_or_keystrokes: str, *, top_k: int) -> PinyinDecoderResult:
        if not 1 <= top_k <= self.max_candidates:
            raise ValueError(f"top_k must be between 1 and {self.max_candidates}")
        normalized = normalize_pinyin(pinyin_or_keystrokes)
        start = time.perf_counter()
        decoded = self._generator.candidates(normalized)[:top_k]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        candidates = tuple(
            PinyinDecoderCandidate(text=item.text, rank=index)
            for index, item in enumerate(decoded, start=1)
        )
        return PinyinDecoderResult(
            raw_input=pinyin_or_keystrokes,
            normalized_pinyin=normalized,
            consumed_input=normalized,
            candidates=candidates,
            latency_ms=elapsed_ms,
            decoder=self.info(),
        )

    def info(self) -> dict[str, Any]:
        return {
            "implementation": "RimeCliCandidateGenerator over desktop librime C API",
            "version": self.version,
            "schema": self.SCHEMA_ID,
            "dictionary": self.DICTIONARY,
            "dictionary_revision": self.DICTIONARY_REVISION,
            "script_mode": "Simplified Chinese",
            "enabled_options": [self.SIMPLIFIED_OPTION],
            "candidate_count": self.max_candidates,
            "configuration": self.setup_manifest,
            "status": self.INTEGRATION_STATUS,
            "official_android_equivalence": False,
            "deviation": (
                "Same Rime/librime full-Pinyin conversion role; pinned Luna data "
                "replaces the APK's Android-only compiled Yuyan pinyin dictionary."
            ),
        }

    def close(self) -> None:
        self._generator.close()

    def __enter__(self) -> "LibrimeLunaPinyinDecoder":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
