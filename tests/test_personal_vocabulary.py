from __future__ import annotations

from pathlib import Path

import pytest

from src.personalisation.context_memory import Candidate, PredictionQuery, frequency_support, rank_frequency, rank_of
from src.personalisation.personal_vocabulary import (
    FROZEN_M1_TOP_N,
    PV1_K_GRID,
    PV1_LAMBDA_GRID,
    PV2_CONTEXT_LAMBDA_GRID,
    PersonalVocabularyState,
    build_personal_lexicon,
    prepare_personal_vocabulary_state,
    rank_pv1,
    rank_pv2,
    transition_counts,
)
from src.personalisation.pv_h5000 import GENERIC_MISSING, M1_EXPECTED, M2_EXPECTED
from src.personalisation.pilot_a import HistoryIndex


ROOT = Path(__file__).resolve().parents[1]


def query(*, author: str = "alice", position: int = 100, context: str = "current") -> PredictionQuery:
    return PredictionQuery("query-id", author, "query-work", position, context, ("shi", "yong"))


def record(
    row_id: str,
    target: str,
    position: int,
    *,
    author: str = "alice",
    pinyin: tuple[str, ...] = ("shi", "yong"),
    context: str | None = None,
) -> dict:
    return {
        "row_id": row_id,
        "author": author,
        "work_id": "history-work",
        "chronological_position": position,
        "context": context or f"context-{row_id}",
        "pinyin_segments": list(pinyin),
        "target": target,
    }


def candidates() -> tuple[Candidate, ...]:
    return (
        Candidate("使用", 1, -1.0),
        Candidate("实用", 2, -2.0),
        Candidate("试用", 3, -3.0),
    )


def embeddings(history: list[dict], current: tuple[float, float] = (1.0, 0.0)) -> dict[str, tuple[float, float]]:
    values = {"current": current}
    for index, row in enumerate(history, start=1):
        values[row["context"]] = (1.0, index / 100.0)
    return values


def state(history: list[dict]) -> PersonalVocabularyState:
    return prepare_personal_vocabulary_state(query(), candidates(), history, embeddings(history))


def test_lexicon_uses_same_user_strictly_prior_exact_pinyin_only() -> None:
    history = [
        record("valid", "适用", 1),
        record("other-user", "适用", 2, author="bob"),
        record("future", "适用", 101),
        record("other-pinyin", "适用", 3, pinyin=("shi",)),
    ]
    lexicon = build_personal_lexicon(query(), history)
    assert [entry.interaction_ids for entry in lexicon] == [("valid",)]


def test_h5000_is_applied_before_same_pinyin_for_personal_vocabulary() -> None:
    rows = [record("old-match", "适用", 1)] + [record(str(value), "是用", value, pinyin=("si",)) for value in range(2, 5002)]
    visible = HistoryIndex(rows, history_budget=5000).visible(query(position=6000))
    assert visible == ()
    assert build_personal_lexicon(query(position=6000), visible) == ()


def test_personal_lexicon_has_traceable_first_last_and_count() -> None:
    lexicon = build_personal_lexicon(query(), [record("second", "适用", 2), record("first", "适用", 1)])
    entry = lexicon[0]
    assert entry.count == 2
    assert entry.first_history_id == "first"
    assert entry.last_history_id == "second"
    assert entry.interaction_ids == ("first", "second")


def test_shared_state_has_no_gold_field() -> None:
    value = state([record("one", "适用", 1)])
    assert not hasattr(value, "gold")
    assert "gold" not in value.to_dict()


def test_personal_only_excludes_generic_surfaces() -> None:
    value = state([record("generic", "使用", 1), record("personal", "适用", 2)])
    assert value.personal_only_targets == ("适用",)


def test_personal_only_order_is_frequency_then_stable_surface() -> None:
    value = state([
        record("a1", "自用", 1), record("a2", "自用", 2),
        record("b", "适用", 3), record("c", "致用", 4),
    ])
    assert value.personal_only_targets == ("自用", "致用", "适用")


def test_frequency_support_is_the_exact_shared_f_component() -> None:
    history = [record("a", "使用", 1), record("b", "使用", 2), record("c", "实用", 3)]
    counts, support = frequency_support([candidate.text for candidate in candidates()], history)
    ranked = rank_frequency(query(), candidates(), history, lambda_frequency=4.0)
    assert counts["使用"] == 2
    assert {row["candidate"]: row["personal_score"] for row in ranked} == support


def test_shared_generic_ranking_matches_f_exactly() -> None:
    history = [record("a", "使用", 1), record("b", "实用", 2), record("c", "适用", 3)]
    value = state(history)
    expected = rank_frequency(query(), candidates(), history, lambda_frequency=4.0)
    assert value.generic_frequency_ranked == expected


def test_boundary_is_minimum_normalized_generic_score() -> None:
    value = state([record("one", "适用", 1)])
    assert value.generic_boundary_score == min(row["normalized_generic_score"] for row in value.generic_frequency_ranked)


def test_pv1_uses_boundary_plus_frequency_for_personal_only() -> None:
    value = state([record("one", "适用", 1)])
    ranked = rank_pv1(value, k_pv=1, lambda_pv=2.0)
    personal = next(row for row in ranked if row["candidate"] == "适用")
    assert personal["final_score"] == pytest.approx(value.generic_boundary_score + 2.0 * value.personal_frequency_support["适用"])


def test_kpv_limit_is_enforced() -> None:
    value = state([record("a", "适用", 1), record("b", "自用", 2), record("c", "致用", 3)])
    ranked = rank_pv1(value, k_pv=1, lambda_pv=4.0)
    assert sum(row["source"] == "personal_vocabulary" for row in ranked) <= 1


def test_merge_has_no_duplicate_surface_and_at_most_ten() -> None:
    value = state([record("a", "适用", 1), record("b", "使用", 2)])
    ranked = rank_pv1(value, k_pv=3, lambda_pv=4.0)
    assert len({row["candidate"] for row in ranked}) == len(ranked)
    assert len(ranked) <= 10


def test_candidate_expansion_can_recover_missing_gold() -> None:
    value = state([record("one", "适用", 1)])
    assert rank_of(value.generic_frequency_ranked, "适用") is None
    assert rank_of(rank_pv1(value, k_pv=1, lambda_pv=4.0), "适用") is not None


def test_pv2_uses_frozen_top5_positive_cosine_context() -> None:
    rows = [record(str(value), "适用", value) for value in range(1, 8)]
    value = state(rows)
    assert FROZEN_M1_TOP_N == 5
    assert value.personal_context_support["适用"] == pytest.approx(1.0)


def test_pv2_formula_adds_context_only_to_personal_candidates() -> None:
    value = state([record("one", "适用", 1)])
    pv1 = rank_pv1(value, k_pv=1, lambda_pv=2.0)
    pv2 = rank_pv2(value, k_pv=1, lambda_pv=2.0, lambda_ctx=0.5)
    generic_pv1 = {row["candidate"]: row["final_score"] for row in pv1 if row["source"] == "generic"}
    generic_pv2 = {row["candidate"]: row["final_score"] for row in pv2 if row["source"] == "generic"}
    assert generic_pv1 == generic_pv2
    personal1 = next(row for row in pv1 if row["candidate"] == "适用")
    personal2 = next(row for row in pv2 if row["candidate"] == "适用")
    assert personal2["final_score"] == pytest.approx(personal1["final_score"] + 0.5 * value.personal_context_support["适用"])


def test_personal_vocabulary_does_not_import_or_use_m2_cross_encoder() -> None:
    source = (ROOT / "src/personalisation/personal_vocabulary.py").read_text(encoding="utf-8")
    assert "candidate_memory_m2" not in source
    assert "BGEReranker" not in source


def test_grids_are_exactly_frozen() -> None:
    assert PV1_K_GRID == (1, 3, 5)
    assert PV1_LAMBDA_GRID == (0.5, 1.0, 2.0, 4.0)
    assert PV2_CONTEXT_LAMBDA_GRID == (0.5, 1.0, 2.0, 4.0)


def test_state_round_trip_is_reproducible() -> None:
    original = state([record("one", "适用", 1)])
    restored = PersonalVocabularyState.from_dict(original.to_dict())
    assert restored == original
    assert rank_pv1(restored, k_pv=1, lambda_pv=4.0) == rank_pv1(original, k_pv=1, lambda_pv=4.0)


def test_transition_accounting_is_paired_and_complete() -> None:
    values = transition_counts([1, 2, 1, None], [1, 1, 2, None])
    assert values == {"unchanged_correct": 1, "helped": 1, "harmed": 1, "unchanged_wrong": 1}
    assert sum(values.values()) == 4


def test_original_generic_missing_denominator_is_frozen() -> None:
    assert GENERIC_MISSING == 538


def test_previous_result_hashes_are_frozen_constants() -> None:
    assert M1_EXPECTED["metrics_summary.json"] == "e35fb9efbe3bdd31d7f8354c227efbed2aa178855061955b3ac16a70137e424d"
    assert M2_EXPECTED["metrics_summary.json"] == "9ad6acecf41b9f36aa1a1bf1bd702cfc729322c4226a4a6a9e3fde4082c6f6d8"
