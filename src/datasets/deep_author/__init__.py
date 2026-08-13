"""Deep Author dataset preparation without model inference."""

from .pipeline import (
    AUTHOR_CONFIG,
    DeepAuthorBuilder,
    author_id,
    clean_text,
    full_pinyin,
    initial_pinyin,
    work_id,
)

__all__ = [
    "AUTHOR_CONFIG",
    "DeepAuthorBuilder",
    "author_id",
    "clean_text",
    "full_pinyin",
    "initial_pinyin",
    "work_id",
]
