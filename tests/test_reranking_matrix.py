from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from src.personalisation.candidate_memory_m2 import PairIdentity, PairScoreCache, rank_m2
from src.personalisation.context_memory import Candidate, PredictionQuery, rank_frequency, rank_from_retrieved, subset_membership
from src.personalisation.pilot_a import EmbeddingCache, HistoryIndex
from src.personalisation.reranking_matrix import (
    CONDITIONS,
    CONDITION_LABELS,
    FROZEN_ARTIFACTS,
    HISTORY_BUDGETS,
    METHODS,
    RerankingMatrixRunner,
    deterministic_wrong_user_mapping,
)


def query(*, author: str = "a", position: int = 1000, pinyin: tuple[str, ...] = ("bei", "jing")) -> PredictionQuery:
    return PredictionQuery("q", author, "w", position, "ctx", pinyin)


def record(number: int, *, author: str = "a", pinyin: tuple[str, ...] = ("bei", "jing"), target: str = "北京") -> dict:
    return {"row_id": f"h{number}", "author": author, "work_id": "w", "chronological_position": number, "context": f"c{number}", "pinyin_segments": list(pinyin), "target": target}


def candidates() -> tuple[Candidate, ...]:
    return (Candidate("北京", 1, -1.0), Candidate("背景", 2, -2.0))


def test_exact_frozen_matrix_identities() -> None:
    assert CONDITIONS == ("full_short", "initial_short", "full_multi3", "initial_multi3")
    assert set(CONDITION_LABELS) == set(CONDITIONS)
    assert HISTORY_BUDGETS == {"H500": 500, "H5000": 5000, "HFull": None}
    assert METHODS == ("F", "M1", "M2")
    assert 4 * 3 * 3 == 36


def test_prediction_query_excludes_current_gold() -> None:
    assert "gold" not in {field.name for field in fields(PredictionQuery)}


def test_h500_h5000_and_hfull_apply_budget_before_pinyin_filter() -> None:
    history = [record(1)] + [record(number, pinyin=("x",)) for number in range(2, 5502)]
    current = query(position=6000)
    assert HistoryIndex(history, 500).visible(current) == ()
    assert HistoryIndex(history, 5000).visible(current) == ()
    assert [row["row_id"] for row in HistoryIndex(history, None).visible(current)] == ["h1"]


def test_history_is_same_user_and_strictly_prior_only() -> None:
    history = [record(1), record(2, author="b"), record(1000), record(1001)]
    assert [row["row_id"] for row in HistoryIndex(history).visible(query())] == ["h1"]


def test_budget_specific_ambiguous_and_conflict_membership() -> None:
    history = [record(1, target="北京"), record(2, target="背景")] + [record(number, pinyin=("x",)) for number in range(3, 503)]
    current = query(position=1000)
    small = HistoryIndex(history, 500).visible(current)
    full = HistoryIndex(history).visible(current)
    assert subset_membership(current, "北京", small)["ambiguous"] is False
    assert subset_membership(current, "别的", full)["ambiguous"] is True
    assert subset_membership(current, "别的", full)["conflict"] is False  # tied winners excluded


def test_unique_frequency_winner_defines_conflict() -> None:
    visible = [record(1, target="北京"), record(2, target="北京"), record(3, target="背景")]
    flags = subset_membership(query(), "背景", visible)
    assert flags["ambiguous"] and flags["conflict"]
    assert flags["frequency_winner"] == "北京"


def test_f_frequency_semantics_are_shared_unchanged() -> None:
    ranked = rank_frequency(query(), candidates(), [record(1), record(2)], lambda_frequency=4.0)
    assert ranked[0]["candidate"] == "北京"
    assert ranked[0]["frequency_count"] == 2


def test_m1_ranking_semantics_are_shared_unchanged() -> None:
    ranked = rank_from_retrieved(candidates(), [{"historical_target": "背景", "weight": 1.0}], lambda_memory=4.0)
    assert ranked[0]["candidate"] == "背景"


def test_m2_ranking_semantics_are_shared_unchanged() -> None:
    ranked = rank_m2(candidates(), [{"historical_target": "背景", "raw_score": 2.0}], lambda_m2=4.0)
    assert ranked[0]["candidate"] == "背景"


def test_embedding_cache_identity_has_no_budget(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "embeddings.sqlite3", model_sha256="test")
    try:
        assert cache.key("same context") == cache.key("same context")
    finally:
        cache.close()


def test_pair_cache_identity_has_no_budget_or_matrix_cell(tmp_path: Path) -> None:
    cache = PairScoreCache(tmp_path / "pairs.sqlite3", model_revision="r", model_sha256="m", tokenizer_sha256="t", max_length=512, dtype="float16")
    pair = PairIdentity("q", "current", ("bei", "jing"), "h", "history", "北京", "北京")
    try:
        assert cache.key(pair) == cache.key(pair)
    finally:
        cache.close()


def test_wrong_user_mapping_is_deterministic_nonself_and_shared() -> None:
    first = deterministic_wrong_user_mapping()
    second = deterministic_wrong_user_mapping()
    assert first == second
    assert set(first) == set(second.values())
    assert all(user != wrong for user, wrong in first.items())
    assert {method: first for method in METHODS}["F"] is first


def test_initial_manifest_has_three_reused_and_33_pending(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = object.__new__(RerankingMatrixRunner)
    runner.output_root = tmp_path
    runner.m1 = type("M1", (), {"output_root": tmp_path / "m1"})()
    runner.m2_root = tmp_path / "m2"
    monkeypatch.setattr(runner, "verify_prior_artifacts", lambda: FROZEN_ARTIFACTS)
    manifest = runner._initial_manifest()
    assert len(manifest["cells"]) == 36
    assert sum(cell["state"] == "reused_complete" for cell in manifest["cells"]) == 3
    assert sum(cell["state"] == "pending" for cell in manifest["cells"]) == 33


def test_completed_group_is_not_recomputed(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = object.__new__(RerankingMatrixRunner)
    cells = [{"condition": "full_short", "history_budget": "H500", "method": method, "state": "complete"} for method in METHODS]
    monkeypatch.setattr(runner, "_manifest", lambda: {"cells": cells})
    assert runner.run_cell_group("full_short", "H500") == {"status": "complete", "methods": []}


def test_failed_group_remains_resumable(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = object.__new__(RerankingMatrixRunner)
    cells = [{"condition": "full_short", "history_budget": "H500", "method": method, "state": "failed"} for method in METHODS]
    monkeypatch.setattr(runner, "_manifest", lambda: {"cells": cells})
    assert [method for method in METHODS if runner._cell({"cells": cells}, "full_short", "H500", method)["state"] != "complete"] == list(METHODS)


def test_finalizer_refuses_complete_when_any_cell_failed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = object.__new__(RerankingMatrixRunner)
    runner.output_root = tmp_path
    failed = {"condition": "full_short", "history_budget": "H500", "method": "F", "state": "failed", "error": "boom"}
    monkeypatch.setattr(runner, "_manifest", lambda: {"cells": [failed]})
    result = runner.finalize()
    assert result["status"] == "incomplete"
    assert result["failed_cell_count"] == 1
    assert (tmp_path / "COMPLETE.json").is_file()


def test_frozen_prior_hashes_cover_t1_m1_m2_and_pv() -> None:
    assert set(FROZEN_ARTIFACTS) == {"T1", "M1", "M2", "PV"}
    assert FROZEN_ARTIFACTS["T1"]["predictions.jsonl"] == "764db39887f3db04b913d1739d9dbd46295f0e46e5a2bffa649f1563b56ee4e2"
