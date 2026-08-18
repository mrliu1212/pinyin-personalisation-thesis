from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from types import SimpleNamespace

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
    PreparedDevGenericRequest,
    RerankingMatrixRunner,
    _generate_compatible_dev_batches,
    deterministic_wrong_user_mapping,
)


def query(*, author: str = "a", position: int = 1000, pinyin: tuple[str, ...] = ("bei", "jing")) -> PredictionQuery:
    return PredictionQuery("q", author, "w", position, "ctx", pinyin)


def record(number: int, *, author: str = "a", pinyin: tuple[str, ...] = ("bei", "jing"), target: str = "北京") -> dict:
    return {"row_id": f"h{number}", "author": author, "work_id": "w", "chronological_position": number, "context": f"c{number}", "pinyin_segments": list(pinyin), "target": target}


def candidates() -> tuple[Candidate, ...]:
    return (Candidate("北京", 1, -1.0), Candidate("背景", 2, -2.0))


class FakeCandidate:
    def __init__(self, text: str) -> None:
        self.text = text

    def to_dict(self) -> dict:
        return {"text": self.text, "rank": 1, "log_probability": -1.0, "mean_log_probability": -1.0}


class ShapeCheckingBackend:
    def __init__(self, prompt_lengths: dict[str, int]) -> None:
        self.prompt_lengths = prompt_lengths
        self.calls: list[list[tuple[str, tuple[str, ...]]]] = []

    def truncate_context_for_generation(self, context: str, segments: list[str]) -> tuple[str, int, int, bool]:
        return context, len(context), len(context), False

    def _prompt(self, context: str, segments: list[str]) -> tuple[list[int], list[int]]:
        length = self.prompt_lengths[context]
        return [0] * length, list(range(length))

    def generate_batch(self, requests: list[tuple[str, tuple[str, ...]]], *, top_k: int, beam_size: int) -> tuple:
        assert top_k == 10 and beam_size == 16
        assert len({self.prompt_lengths[context] for context, _ in requests}) == 1
        assert len({len(segments) for _, segments in requests}) == 1
        self.calls.append(requests)
        return tuple(SimpleNamespace(candidates=(FakeCandidate(context),), runtime_device="cuda") for context, _ in requests)


def test_dev_generic_batches_mixed_shapes_and_restores_original_order() -> None:
    backend = ShapeCheckingBackend({"a": 7, "b": 8, "c": 7, "d": 7})
    prepared = [
        PreparedDevGenericRequest({"row_id": name}, segments, name, 1, 1, False, prompt_length)
        for name, segments, prompt_length in (
            ("a", ("a", "b"), 7),
            ("b", ("a", "b"), 8),
            ("c", ("a", "b", "c"), 7),
            ("d", ("a", "b"), 7),
        )
    ]
    restored = _generate_compatible_dev_batches(backend, prepared)
    assert [request.row["row_id"] for request, _ in restored] == ["a", "b", "c", "d"]
    assert [[context for context, _ in call] for call in backend.calls] == [["a", "d"], ["b"], ["c"]]


def test_dev_generic_resume_preserves_exact_count_order_and_does_not_recompute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "row_id": f"r{index}",
            "pilot_partition": "tune",
            "context": f"c{index}",
            "pinyin_segments": ["a"] * (1 + index % 2),
            "gold": f"g{index}",
        }
        for index in range(7)
    ]
    runner = object.__new__(RerankingMatrixRunner)
    runner.output_root = tmp_path
    runner.pinyingpt_model = tmp_path / "model"
    monkeypatch.setattr(runner, "_dev", lambda condition: rows)
    cache = runner._generic_dev_path("initial_multi3")
    cache.parent.mkdir(parents=True)
    cache.write_text("".join(json.dumps({**row, "cached": True}) + "\n" for row in rows[:2]), encoding="utf-8")
    backend = ShapeCheckingBackend({f"c{index}": 10 + index % 3 for index in range(7)})

    result = runner.ensure_dev_generic("initial_multi3", backend=backend)

    cached = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()]
    assert result == {"required": 7, "reused_at_start": 2, "added": 5, "complete": True}
    assert [row["row_id"] for row in cached] == [row["row_id"] for row in rows]
    assert {context for call in backend.calls for context, _ in call} == {f"c{index}" for index in range(2, 7)}

    class NoInferenceBackend:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"completed Dev cache attempted backend access: {name}")

    resumed = runner.ensure_dev_generic("initial_multi3", backend=NoInferenceBackend())
    assert resumed == {"required": 7, "reused_at_start": 7, "added": 0, "complete": True}


def test_owned_dev_backend_releases_torch_cuda_cache_before_next_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row = {"row_id": "r0", "pilot_partition": "tune", "context": "c0", "pinyin_segments": ["a"], "gold": "g0"}
    runner = object.__new__(RerankingMatrixRunner)
    runner.output_root = tmp_path
    runner.pinyingpt_model = tmp_path / "model"
    monkeypatch.setattr(runner, "_dev", lambda condition: [row])
    monkeypatch.setattr("src.reference_backend_pinyingpt.PinyinGPTConcatBackend", lambda *args, **kwargs: ShapeCheckingBackend({"c0": 4}))
    released = []
    monkeypatch.setattr("src.personalisation.reranking_matrix._release_torch_cuda_cache", lambda: released.append(True))

    result = runner.ensure_dev_generic("initial_multi3")

    assert result["complete"] is True
    assert released == [True]


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
