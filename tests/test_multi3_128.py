from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.personalisation.multi3_128 import (
    AUTHORS,
    CONDITIONS,
    EXPERIMENT_POSITION_CAP,
    InteractionContextPolicy,
    Multi3AuditRunner,
    ROW_DIAGNOSTIC_FIELDS,
)
from src.reference_backend_pinyingpt.backend import PinyinGPTConcatBackend


class CharacterTokenizer:
    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(value)))


def test_interaction_policy_uses_total_budget_and_recent_suffix() -> None:
    policy = InteractionContextPolicy(CharacterTokenizer(), 1024)
    result = policy.apply("x" * 200, ("a", "b", "c", "d"))
    assert result.effective_maximum_positions == 128
    assert result.effective_context_tokens == 118
    assert result.effective_context_128 == "x" * 118
    assert result.complete_generation_positions == 128
    assert result.context_truncated is True


def test_each_current_or_historical_interaction_gets_own_pinyin_budget() -> None:
    policy = InteractionContextPolicy(CharacterTokenizer(), 1024)
    current = policy.contextualize_interaction(
        {"context": "c" * 200, "pinyin_segments": ["a", "b", "c"]}
    )
    historical = policy.contextualize_interaction(
        {"context": "h" * 200, "pinyin_segments": ["a"]}
    )
    assert current["context"] == "c" * 120
    assert historical["context"] == "h" * 124
    assert current["stored_context"] == "c" * 200
    assert historical["stored_context"] == "h" * 200


def test_backend_default_remains_model_maximum_and_v2_cap_is_opt_in() -> None:
    backend = object.__new__(PinyinGPTConcatBackend)
    backend.tokenizer = CharacterTokenizer()
    backend.model = SimpleNamespace(config=SimpleNamespace(n_positions=1024))
    backend.segment_pinyin = lambda value: tuple(value)
    old_default = backend.truncate_context_for_generation("x" * 1100, ("a", "b"))
    v2 = backend.truncate_context_for_generation(
        "x" * 1100, ("a", "b"), maximum_positions=128
    )
    assert old_default == ("x" * 1018, 1100, 1018, True)
    assert v2 == ("x" * 122, 1100, 122, True)


def test_model_smaller_than_experimental_cap_remains_the_hard_limit() -> None:
    policy = InteractionContextPolicy(CharacterTokenizer(), 64)
    result = policy.apply("x" * 100, ("a", "b"))
    assert result.effective_maximum_positions == 64
    assert result.complete_generation_positions == 64


def test_audit_sample_is_deterministic_and_five_per_stratum() -> None:
    rows = [
        {"condition": condition, "author": author, "condition_id": f"{condition}-{author}-{index}"}
        for condition in CONDITIONS
        for author in AUTHORS
        for index in range(10)
    ]
    first = Multi3AuditRunner._sample(rows)
    second = Multi3AuditRunner._sample(rows)
    assert [row["condition_id"] for row in first] == [row["condition_id"] for row in second]
    assert len(first) == 60
    assert {
        (condition, author): sum(
            row["condition"] == condition and row["author"] == author for row in first
        )
        for condition in CONDITIONS for author in AUTHORS
    } == {(condition, author): 5 for condition in CONDITIONS for author in AUTHORS}


def test_context_statistics_assertable_under_128() -> None:
    rows = [
        {
            "stored_context_characters": 512,
            "original_context_tokens": 450,
            "effective_context_tokens": 120,
            "complete_generation_positions": EXPERIMENT_POSITION_CAP,
            "context_truncated": True,
        }
    ]
    stats = Multi3AuditRunner._context_stats(rows)
    assert stats["rows_exceeding_128_positions"] == 0
    assert stats["truncation_rate"] == 1.0


def test_formal_run_refuses_without_human_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = Multi3AuditRunner(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path)
    monkeypatch.setattr(
        runner,
        "preflight",
        lambda: {"human_audit_approved": False, "eligible_for_formal_run": False},
    )
    with pytest.raises(RuntimeError, match="human_audit_approved != true"):
        runner.formal_run()


def test_future_row_schema_contains_comparison_and_context_fields() -> None:
    required = {
        "effective_context_128", "history_target_frequency_counts",
        "generic_top10", "f_wrong_m2_correct", "f_correct_m2_wrong",
    }
    assert required <= set(ROW_DIAGNOSTIC_FIELDS)
