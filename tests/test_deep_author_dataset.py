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
    clean_text_with_offsets,
    changed_character_count,
    full_pinyin,
    find_uncertain_blocks,
    initial_pinyin,
    interaction_id,
    make_interactions,
    metadata_prefix_end,
    attribution_exclusion_reason,
    preliminary_exclusion_reason,
    segment_text,
    simplify_text,
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
        "simplified_cleaned_text": text,
        "source_offsets": list(range(len(text))),
        "creation_date": "2019-01-01",
        "SHA256": "a" * 64,
        "original_cleaned_sha256": "b" * 64,
        "simplified_cleaned_sha256": "c" * 64,
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


def test_source_confirmed_credit_block_is_removed_but_normal_prose_is_retained() -> None:
    source = """[[include :scp-wiki-cn:credit:start]]
**作者：**[[*user Example]]
[[include :scp-wiki-cn:credit:end]]

正常故事中的作者觉得这个方法实用。
"""
    rendered = "著作信息\n作者：Example\n返回\n正常故事中的作者觉得这个方法实用。\n"
    end = metadata_prefix_end(source, rendered)
    assert end == rendered.index("正常故事")
    cleaned, offsets = clean_text_with_offsets(rendered, end)
    assert cleaned == "正常故事中的作者觉得这个方法实用。\n"
    assert offsets[0] == end


def test_uncertain_metadata_like_line_is_retained_and_logged() -> None:
    rendered = "正文里的作者不是网站元数据。\n\n图像信息：\n这一块没有可验证的源结构。\n"
    cleaned, _ = clean_text_with_offsets(rendered)
    uncertain = find_uncertain_blocks(rendered, 0)
    assert "正文里的作者" in cleaned
    assert "图像信息：" in cleaned
    assert [row["block_text"] for row in uncertain] == ["图像信息："]


def test_opencc_t2s_is_deterministic() -> None:
    original = "這個方法非常實用"
    assert simplify_text(original) == "这个方法非常实用"
    assert simplify_text(original) == simplify_text(original)
    assert changed_character_count(original, simplify_text(original)) == 3


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


@pytest.mark.parametrize("separator", ["，", "(", ")", "Site-19", "123", "\n\n", "🤖"])
def test_non_han_is_a_hard_boundary_and_never_enters_gold_or_context(separator: str) -> None:
    text = f"前文实用{separator}而且没有"
    work = example_work(text)
    interactions, failures = make_interactions(work, segment_text(text))
    assert not failures
    assert all(all(pipeline.is_han(character) for character in row["gold"]) for row in interactions)
    assert all(all(pipeline.is_han(character) for character in row["context"]) for row in interactions)
    for row in interactions:
        if row["composition_type"] == "multi" and row["gold"].startswith("实用"):
            assert "而且" not in row["gold"]


def test_short_and_multi_keep_same_source_anchor_and_stable_ids() -> None:
    text = "前文这个方法非常实用"
    first, _ = make_interactions(example_work(text), segment_text(text))
    second, _ = make_interactions(example_work(text), segment_text(text))
    assert [row["interaction_id"] for row in first] == [row["interaction_id"] for row in second]
    short_by_start = {row["source_position_start"]: row for row in first if row["composition_type"] == "short"}
    for multi in (row for row in first if row["composition_type"] == "multi"):
        short = short_by_start[multi["source_position_start"]]
        assert short["context"] == multi["context"]
        assert short["boundary_span_id"] == multi["boundary_span_id"]


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


def test_v1_1_run_checksums_entire_raw_snapshot_and_has_no_model_inference() -> None:
    source = inspect.getsource(DeepAuthorBuilder.run)
    assert "self.raw_root.rglob" in source
    assert "self.discover()" in source and "self.acquire(discoveries)" in source
    assert "model" not in source.casefold()
