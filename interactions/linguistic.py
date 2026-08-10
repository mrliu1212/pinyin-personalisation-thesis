"""Isolated segmentation, filtering, context, and full-Pinyin conversion."""

from __future__ import annotations

import re
from dataclasses import dataclass

import jieba
from pypinyin import Style, lazy_pinyin, pinyin


CHINESE_RE = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+$")
CHINESE_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


@dataclass(frozen=True)
class TargetPolicy:
    min_characters: int = 2
    max_characters: int = 4
    derived_context_characters: int = 12


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class PinyinConversion:
    normalized: str
    syllables: tuple[str, ...]
    polyphonic_characters: tuple[dict[str, object], ...]


class JiebaSegmenter:
    def __init__(self) -> None:
        self._tokenizer = jieba.Tokenizer()

    def segment(self, text: str) -> list[Token]:
        return [
            Token(word, start, end)
            for word, start, end in self._tokenizer.tokenize(text, mode="default")
        ]


def exclusion_reason(token: str, policy: TargetPolicy) -> str | None:
    if not CHINESE_RE.fullmatch(token):
        return "non_chinese_or_punctuation"
    if len(token) < policy.min_characters:
        return "below_minimum_length"
    if len(token) > policy.max_characters:
        return "above_maximum_length"
    return None


def derived_context(raw_context: str, character_count: int) -> str:
    chinese = CHINESE_CHAR_RE.findall(raw_context)
    return "".join(chinese[-character_count:])


def convert_full_pinyin(target: str) -> PinyinConversion:
    syllables = tuple(
        syllable.lower().replace("ü", "v")
        for syllable in lazy_pinyin(target, style=Style.NORMAL, strict=True)
    )
    if not syllables or any(not re.fullmatch(r"[a-z]+", item) for item in syllables):
        raise ValueError(f"could not normalize full Pinyin for {target!r}")
    flagged: list[dict[str, object]] = []
    for index, character in enumerate(target):
        readings = sorted(
            {
                value.lower().replace("ü", "v")
                for values in pinyin(
                    character, style=Style.NORMAL, heteronym=True, strict=True
                )
                for value in values
            }
        )
        if len(readings) > 1:
            flagged.append(
                {"character": character, "character_index": index, "readings": readings}
            )
    return PinyinConversion(
        normalized="".join(syllables),
        syllables=syllables,
        polyphonic_characters=tuple(flagged),
    )

