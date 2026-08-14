from __future__ import annotations

from collections import Counter
import inspect

from src.evaluation.deep_author_v2 import (
    AUTHORS,
    CONDITIONS,
    DesignBuilder,
    anchor_id,
    balanced_sample,
    choose_split,
    condition_id,
    conditions_for_anchor,
    valid_anchors_for_work,
)
import src.evaluation.deep_author_v2 as evaluation


def _work(text: str = "前文这个方法非常实用而且稳定") -> dict:
    return {
        "author_name": AUTHORS[0], "author_id": "author", "work_id": "work",
        "page_title": "title", "creation_date": "2020-01-01", "SHA256": "a" * 64,
        "cleaned_sha256": "b" * 64, "cleaned_text": text,
    }


def test_ids_are_stable_and_condition_specific() -> None:
    assert anchor_id("work", 3) == anchor_id("work", 3)
    assert condition_id("anchor", "full_short") != condition_id("anchor", "initial_short")


def test_exact_three_token_anchor_and_four_paired_conditions() -> None:
    tokens = [
        {"text": "前文", "start": 0, "end": 2, "is_han": True},
        {"text": "这个", "start": 2, "end": 4, "is_han": True},
        {"text": "方法", "start": 4, "end": 6, "is_han": True},
        {"text": "实用", "start": 6, "end": 8, "is_han": True},
    ]
    compatibility = {key: list("前文这个方法实用") for key in ("q", "w", "z", "g", "f", "s", "y", "qian", "wen", "ge", "zhe", "fang", "fa", "shi", "yong")}
    anchors = valid_anchors_for_work(_work(), tokens, compatibility)
    assert anchors
    assert all(row["multi3_token_count"] == 3 for row in anchors)
    conditions = conditions_for_anchor(anchors[0])
    assert {row["condition"] for row in conditions} == set(CONDITIONS)
    assert conditions[0]["context"] == conditions[-1]["context"]


def test_split_is_chronological_and_obeys_minimums() -> None:
    works = [{"eligible_anchor_count": value} for value in [1000] * 10]
    history_end, dev_end = choose_split(works)
    assert history_end >= 5
    assert dev_end - history_end >= 2
    assert len(works) - dev_end >= 3


def test_balanced_sampling_is_deterministic_and_work_balanced() -> None:
    rows = [{"anchor_id": f"{work}-{index}", "work_id": work, "source_position_start": index} for work in ("a", "b", "c") for index in range(10)]
    first = balanced_sample(rows, 12, 40408)
    assert first == balanced_sample(rows, 12, 40408)
    assert Counter(row["work_id"] for row in first) == Counter({"a": 4, "b": 4, "c": 4})


def test_design_has_no_model_or_personalisation_inference() -> None:
    source = inspect.getsource(evaluation).casefold()
    assert "torch" not in source
    assert "transformers" not in source
    assert "pinyingptconcatbackend" not in source
    run_source = inspect.getsource(DesignBuilder.run).casefold()
    assert "model_inference\": false" in run_source
