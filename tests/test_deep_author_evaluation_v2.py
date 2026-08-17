from __future__ import annotations

from collections import Counter
import inspect

from src.evaluation.deep_author_v2 import (
    AUTHORS,
    CONDITIONS,
    DesignBuilder,
    T1Runner,
    anchor_id,
    balanced_sample,
    choose_split,
    condition_id,
    conditions_for_anchor,
    valid_anchors_for_work,
    metric_values,
    aggregate_metrics,
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
    source = inspect.getsource(DesignBuilder).casefold()
    assert "torch" not in source
    assert "transformers" not in source
    assert "pinyingptconcatbackend" not in source
    run_source = inspect.getsource(DesignBuilder.run).casefold()
    assert "model_inference\": false" in run_source


def test_metrics_use_exact_rank_and_missing_definitions() -> None:
    rows = [
        {"gold_rank": 1, "top1_correct": True, "top3_correct": True, "reciprocal_rank": 1.0, "missing_at_10": False},
        {"gold_rank": 3, "top1_correct": False, "top3_correct": True, "reciprocal_rank": 1 / 3, "missing_at_10": False},
        {"gold_rank": None, "top1_correct": False, "top3_correct": False, "reciprocal_rank": 0.0, "missing_at_10": True},
    ]
    values = metric_values(rows)
    assert values["top1"] == 1 / 3
    assert values["top3"] == 2 / 3
    assert values["missing_at_10"] == 1 / 3
    assert values["mean_rank_given_top10"] == 2


def test_primary_metric_averages_authors_equally() -> None:
    rows = []
    for author_index, author in enumerate(AUTHORS):
        for condition in CONDITIONS:
            repetitions = 20 if author_index == 0 else 1
            for _ in range(repetitions):
                correct = author_index == 0
                rows.append({
                    "author": author,
                    "condition": condition,
                    "gold_rank": 1 if correct else None,
                    "top1_correct": correct,
                    "top3_correct": correct,
                    "reciprocal_rank": 1.0 if correct else 0.0,
                    "missing_at_10": not correct,
                })
    metrics = aggregate_metrics(rows)
    assert metrics["primary_macro_author"]["top1"] == 1 / 6
    assert metrics["secondary_micro"]["overall"]["top1"] != 1 / 6


def test_cached_prediction_must_match_frozen_provenance() -> None:
    frozen = {
        "condition_id": "condition",
        "anchor_id": "anchor",
        "author": AUTHORS[0],
        "work_id": "work",
        "condition": CONDITIONS[0],
        "context": "context",
        "pinyin_input": "pin yin",
        "gold": "拼音",
    }
    cached = {
        **frozen,
        "checkpoint_revision": evaluation.CHECKPOINT_REVISION,
        "official_code_revision": evaluation.OFFICIAL_CODE_REVISION,
        "beam_size": 16,
        "top_k": 10,
        "top10_candidates": [
            {"rank": 1, "text": frozen["gold"], "log_probability": -1.0}
        ],
        "gold_rank": 1,
        "top1_correct": True,
        "top3_correct": True,
        "top10_present": True,
        "missing_at_10": False,
        "reciprocal_rank": 1.0,
        "model_used_context": frozen["context"],
    }
    T1Runner.validate_cached_prediction(cached, frozen)
    cached["beam_size"] = 15
    try:
        T1Runner.validate_cached_prediction(cached, frozen)
    except RuntimeError as error:
        assert "decoding parameters" in str(error)
    else:
        raise AssertionError("non-frozen cached decoding parameters were accepted")
