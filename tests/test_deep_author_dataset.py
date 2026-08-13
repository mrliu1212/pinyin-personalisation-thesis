from __future__ import annotations

import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.datasets.deep_author.pipeline import (
    DeepAuthorBuilder,
    author_id,
    clean_text,
    full_pinyin,
    initial_pinyin,
    interaction_id,
    make_interactions,
    attribution_exclusion_reason,
    preliminary_exclusion_reason,
    segment_text,
    work_id,
)
import src.datasets.deep_author.pipeline as pipeline


def example_work(text: str = "这个方法非常实用而且没有那么复杂。") -> dict:
    return {
        "author_id": author_id("Example", 1),
        "author_name": "Example",
        "work_id": work_id(123),
        "page_title": "Example",
        "cleaned_text": text,
        "creation_date": "2019-01-01",
        "SHA256": "a" * 64,
    }


def test_ids_are_deterministic_and_distinct() -> None:
    assert author_id("Agent Phage", 2950095) == author_id("Agent Phage", 2950095)
    assert author_id("Agent Phage", 2950095) != author_id("Agent Phage", 1)
    assert work_id(66105575) == "da-work-66105575"
    assert interaction_id("work", 1, 2, "short") == interaction_id("work", 1, 2, "short")


def test_cleaning_is_deterministic_and_preserves_boundaries_and_prose() -> None:
    raw = "第一段。\r\n\r\n Loading... \r\n\r\n第二段  原文。\r\n"
    expected = "第一段。\n\n第二段 原文。\n"
    assert clean_text(raw) == expected
    assert clean_text(raw) == clean_text(raw)


def test_segmentation_offsets_are_stable_and_recover_source() -> None:
    text = "这个方法非常实用。"
    first = segment_text(text)
    assert first == segment_text(text)
    for token in first:
        assert text[token["start"] : token["end"]] == token["text"]


def test_pinyin_alignment_and_official_initials() -> None:
    syllables = full_pinyin("实用")
    assert syllables == ["shi", "yong"]
    assert initial_pinyin(syllables) == ["s", "y"]
    assert initial_pinyin(["shi", "zhong", "chi"]) == ["s", "z", "c"]
    with pytest.raises(ValueError):
        full_pinyin("实用!")


def test_short_context_and_multi_same_start_same_context() -> None:
    work = example_work()
    interactions, failures = make_interactions(work, segment_text(work["cleaned_text"]))
    assert not failures
    short_by_start = {
        row["source_position_start"]: row
        for row in interactions
        if row["composition_type"] == "short"
    }
    for multi in (row for row in interactions if row["composition_type"] == "multi"):
        short = short_by_start[multi["source_position_start"]]
        assert multi["context"] == short["context"]
        assert multi["gold"].startswith(short["gold"])
        assert multi["source_position_end"] > short["source_position_end"]
        context_start = short["context_source_position_start"]
        assert work["cleaned_text"][context_start : short["source_position_start"]] == short["context"]
        assert len(short["context"]) <= 512


def test_multi_stops_at_sentence_and_paragraph_boundaries() -> None:
    for text in ("前文实用。而且没有。", "前文实用\n\n而且没有。"):
        work = example_work(text)
        interactions, _ = make_interactions(work, segment_text(text))
        for row in interactions:
            if row["composition_type"] == "multi" and row["gold"].startswith("实用"):
                assert "而且" not in row["gold"]


def test_immutable_raw_preservation_rejects_change() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "config/deep_author").mkdir(parents=True)
        (root / "config/deep_author/authors_v1.json").write_text(
            json.dumps({"authors": [], "reserve_authors": []}), encoding="utf-8"
        )
        builder = DeepAuthorBuilder(root=root)
        path, checksum, size = builder._preserve_raw(Path("x.json"), {"value": 1})
        assert path.exists() and checksum and size > 0
        builder._preserve_raw(Path("x.json"), {"value": 1})
        with pytest.raises(RuntimeError, match="immutable raw file differs"):
            builder._preserve_raw(Path("x.json"), {"value": 2})


def test_eligibility_rejects_translation_coauthor_and_out_of_window() -> None:
    base = {
        "firstRevisionAt": "2019-01-01T00:00:00Z",
        "tags": ["原创", "故事"],
        "url": "https://scp-wiki-cn.wikidot.com/example",
    }
    assert preliminary_exclusion_reason(base, "2014-01-01", "2021-12-31") == ""
    assert preliminary_exclusion_reason({**base, "tags": ["原创", "翻译"]}, "2014-01-01", "2021-12-31") == "translation"
    assert preliminary_exclusion_reason({**base, "firstRevisionAt": "2022-01-01"}, "2014-01-01", "2021-12-31") == "out_of_window"
    assert preliminary_exclusion_reason({**base, "url": "https://scp-wiki-cn.wikidot.com/component:x"}, "2014-01-01", "2021-12-31") == "structural_or_non_work_page"

    one = [{"type": "SUBMITTER", "displayName": "Etinjat"}]
    assert attribution_exclusion_reason(one, {"Etinjat"}, ["Etinjat"]) == ""
    two = one + [{"type": "SUBMITTER", "displayName": "Other"}]
    assert attribution_exclusion_reason(two, {"Etinjat"}, ["Etinjat"]) == "coauthored_or_unclear_attribution"


def test_dataset_build_has_no_model_inference_or_personalisation() -> None:
    source = inspect.getsource(pipeline).casefold()
    forbidden = ("pinyingpt", "torch", "transformers", "candidate ranking", "personal_model")
    assert not any(term in source for term in forbidden)
