from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path

import numpy as np
import pytest

from src.personalisation.context_memory import (
    Candidate,
    PredictionQuery,
    assert_candidate_pool,
    rank_frequency,
    rank_memory,
    rank_of,
    subset_membership,
    visible_same_pinyin_history,
)
from src.personalisation.pilot_a import (
    BACKEND_INTEGRATION_REVISION,
    BACKEND_SOURCE_REVISION,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_SHA256,
    GENERIC_CONTEXT_SEMANTICS,
    EmbeddingCache,
    PilotRunner,
    split_dev_works,
)
from src.evaluation.deep_author_v2 import CHECKPOINT_REVISION, OFFICIAL_CODE_REVISION, canonical_json


def query(position: int = 10) -> PredictionQuery:
    return PredictionQuery("current", "alice", "dev-work", position, "当前上下文", ("shi",))


def candidates() -> tuple[Candidate, ...]:
    return (
        Candidate("是", 1, -1.0),
        Candidate("事", 2, -1.4),
        Candidate("时", 3, -2.0),
    )


def record(row_id: str, author: str, pinyin: str, position: int, target: str, context: str = "历史") -> dict:
    return {
        "row_id": row_id,
        "author": author,
        "work_id": "history-work",
        "pinyin_segments": [pinyin],
        "chronological_position": position,
        "target": target,
        "context": context,
    }


def test_history_is_strictly_prior_same_user_and_same_pinyin() -> None:
    history = [
        record("valid", "alice", "shi", 9, "事"),
        record("future", "alice", "shi", 11, "时"),
        record("current", "alice", "shi", 10, "时"),
        record("wrong-user", "bob", "shi", 8, "时"),
        record("wrong-pinyin", "alice", "si", 7, "四"),
    ]
    assert [row["row_id"] for row in visible_same_pinyin_history(query(), history)] == ["valid"]


def test_zero_history_frequency_and_memory_exactly_fall_back_to_generic() -> None:
    generic = candidates()
    frequency = rank_frequency(query(), generic, [], lambda_frequency=4.0)
    memory, evidence = rank_memory(query(), generic, [], {}, top_n=20, lambda_memory=4.0)
    expected = [candidate.text for candidate in generic]
    assert [row["candidate"] for row in frequency] == expected
    assert [row["candidate"] for row in memory] == expected
    assert evidence == ()


def test_frequency_formula_can_rerank_without_changing_candidates() -> None:
    history = [record("a", "alice", "shi", 1, "事"), record("b", "alice", "shi", 2, "事")]
    ranked = rank_frequency(query(), candidates(), history, lambda_frequency=4.0)
    assert ranked[0]["candidate"] == "事"
    assert next(row for row in ranked if row["candidate"] == "事")["frequency_count"] == 2
    assert {row["candidate"] for row in ranked} == {candidate.text for candidate in candidates()}


def test_memory_retrieval_is_deterministic_and_preserves_provenance() -> None:
    history = [
        record("later", "alice", "shi", 2, "事", "later-context"),
        record("earlier", "alice", "shi", 1, "时", "earlier-context"),
    ]
    embeddings = {
        "当前上下文": [1.0, 0.0],
        "later-context": [1.0, 0.0],
        "earlier-context": [1.0, 0.0],
    }
    first, first_evidence = rank_memory(query(), candidates(), history, embeddings, top_n=2, lambda_memory=1.0)
    second, second_evidence = rank_memory(query(), candidates(), list(reversed(history)), embeddings, top_n=2, lambda_memory=1.0)
    assert first == second
    assert first_evidence == second_evidence
    assert [row["historical_interaction_id"] for row in first_evidence] == ["earlier", "later"]
    assert all({"historical_target", "similarity", "chronological_position"} <= set(row) for row in first_evidence)


def test_ambiguous_and_conflict_definitions_and_tie_exclusion() -> None:
    unique_winner = [
        record("a", "alice", "shi", 1, "是"),
        record("b", "alice", "shi", 2, "是"),
        record("c", "alice", "shi", 3, "事"),
    ]
    flags = subset_membership(query(), "事", unique_winner)
    assert flags["ambiguous"] is True
    assert flags["frequency_winner"] == "是"
    assert flags["conflict"] is True
    tied = subset_membership(query(), "时", unique_winner[1:])
    assert tied["ambiguous"] is True
    assert tied["frequency_winner_tied"] is True
    assert tied["conflict"] is False


def test_candidate_pool_and_missing_at_10_are_invariant() -> None:
    generic = candidates()
    history = [record("a", "alice", "shi", 1, "事")]
    frequency = rank_frequency(query(), generic, history, lambda_frequency=2.0)
    memory, _ = rank_memory(
        query(), generic, history, {"当前上下文": [1.0], "历史": [1.0]}, top_n=1, lambda_memory=2.0
    )
    assert_candidate_pool(generic, frequency, memory)
    assert rank_of(frequency, "不存在") is None
    assert rank_of(memory, "不存在") is None


def test_current_gold_is_not_a_prediction_query_field() -> None:
    assert "gold" not in {field.name for field in fields(PredictionQuery)}


def test_whole_work_tune_partition_precedes_and_is_disjoint_from_evaluation() -> None:
    tune, evaluation = split_dev_works(("work-1", "work-2", "work-3", "work-4", "work-5"))
    assert tune == ("work-1", "work-2")
    assert evaluation == ("work-3", "work-4", "work-5")
    assert set(tune).isdisjoint(evaluation)
    with pytest.raises(ValueError):
        split_dev_works(("only",))


def test_test_data_is_rejected_by_pilot_manifest_loader(tmp_path: Path) -> None:
    output = tmp_path / "results"
    output.mkdir()
    (output / "history_manifest.jsonl").write_text("", encoding="utf-8")
    row = {"row_id": "bad", "source_split": "test"}
    (output / "dev_manifest.jsonl").write_text(canonical_json(row) + "\n", encoding="utf-8")
    runner = PilotRunner(tmp_path, tmp_path, tmp_path, tmp_path, output)
    with pytest.raises(RuntimeError, match="Test rows"):
        runner._manifests()


def _dev_row() -> dict:
    return {
        "row_id": "row",
        "author": "alice",
        "work_id": "work",
        "chronological_position": 2,
        "context": "上下文",
        "pinyin_input": "shi",
        "pinyin_segments": ["shi"],
        "gold": "是",
        "pilot_partition": "tune",
        "source_split": "dev",
        "target": "是",
    }


def _generic_row() -> dict:
    return {
        **_dev_row(),
        "checkpoint_revision": CHECKPOINT_REVISION,
        "official_code_revision": OFFICIAL_CODE_REVISION,
        "beam_size": 16,
        "top_k": 10,
        "runtime_device": "cuda",
        "backend_source_revision": BACKEND_SOURCE_REVISION,
        "backend_integration_revision": BACKEND_INTEGRATION_REVISION,
        "context_semantics": GENERIC_CONTEXT_SEMANTICS,
        "top10_candidates": [{"rank": 1, "text": "是", "log_probability": -1.0}],
    }


def test_generic_cache_resume_integrity_and_provenance(tmp_path: Path) -> None:
    output = tmp_path / "results"
    output.mkdir()
    (output / "history_manifest.jsonl").write_text("", encoding="utf-8")
    (output / "dev_manifest.jsonl").write_text(canonical_json(_dev_row()) + "\n", encoding="utf-8")
    cache = output / "generic_predictions.jsonl"
    cache.write_text(canonical_json(_generic_row()) + "\n", encoding="utf-8")
    runner = PilotRunner(tmp_path, tmp_path, tmp_path, tmp_path, output)
    assert set(runner._load_generic([_dev_row()], require_complete=True)) == {"row"}
    corrupted = _generic_row()
    corrupted["checkpoint_revision"] = "wrong"
    cache.write_text(canonical_json(corrupted) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="provenance"):
        runner._load_generic([_dev_row()], require_complete=True)


def test_embedding_cache_is_deterministic_and_rejects_provenance_change(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.sqlite3"
    cache = EmbeddingCache(path)
    vector = np.ones(EMBEDDING_DIMENSION, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    cache.put("上下文", vector)
    cache.commit()
    first_key = cache.key("上下文")
    cache.close()
    reopened = EmbeddingCache(path)
    assert reopened.key("上下文") == first_key
    assert np.allclose(reopened["上下文"], vector)
    reopened.close()
    with pytest.raises(RuntimeError, match="provenance"):
        EmbeddingCache(path, model_sha256="0" * len(EMBEDDING_MODEL_SHA256))


def test_tune_and_evaluate_pipeline_keeps_populations_separate(tmp_path: Path) -> None:
    output = tmp_path / "results"
    output.mkdir()
    history = []
    dev = []
    generic = []
    for author_index, author in enumerate(("alice", "bob")):
        base = author_index * 100
        history.append({
            "row_id": f"{author}-history", "author": author, "work_id": f"{author}-history-work",
            "chronological_position": base + 1, "context": f"{author}-history-context",
            "pinyin_segments": ["shi"], "target": "是", "gold": "是", "source_split": "history",
        })
        for partition, offset in (("tune", 10), ("evaluation", 20)):
            row = {
                "row_id": f"{author}-{partition}", "author": author, "work_id": f"{author}-{partition}-work",
                "chronological_position": base + offset, "context": f"{author}-{partition}-context",
                "pinyin_input": "shi", "pinyin_segments": ["shi"], "gold": "事", "target": "事",
                "pilot_partition": partition, "source_split": "dev",
            }
            dev.append(row)
            candidates_value = [
                {"rank": 1, "text": "是", "log_probability": -1.0},
                {"rank": 2, "text": "事", "log_probability": -1.2},
            ]
            generic.append({
                **row, "checkpoint_revision": CHECKPOINT_REVISION,
                "official_code_revision": OFFICIAL_CODE_REVISION, "beam_size": 16, "top_k": 10,
                "runtime_device": "cuda", "top10_candidates": candidates_value, "gold_rank": 2,
                "backend_source_revision": BACKEND_SOURCE_REVISION,
                "backend_integration_revision": BACKEND_INTEGRATION_REVISION,
                "context_semantics": GENERIC_CONTEXT_SEMANTICS,
            })
    (output / "history_manifest.jsonl").write_text("".join(canonical_json(row) + "\n" for row in history), encoding="utf-8")
    (output / "dev_manifest.jsonl").write_text("".join(canonical_json(row) + "\n" for row in dev), encoding="utf-8")
    (output / "generic_predictions.jsonl").write_text("".join(canonical_json(row) + "\n" for row in generic), encoding="utf-8")
    cache = EmbeddingCache(output / "embedding_cache.sqlite3")
    vector = np.ones(EMBEDDING_DIMENSION, dtype=np.float32)
    for row in history + dev:
        cache.put(row["context"], vector)
    cache.commit()
    cache.close()
    runner = PilotRunner(tmp_path, tmp_path, tmp_path, tmp_path, output)
    (output / "generic_runtime.json").write_text("{}", encoding="utf-8")
    (output / "embedding_runtime.json").write_text("{}", encoding="utf-8")
    selection = runner.tune()
    assert set(selection["tune_work_ids"]) == {"alice-tune-work", "bob-tune-work"}
    assert set(selection["evaluation_work_ids"]) == {"alice-evaluation-work", "bob-evaluation-work"}
    result = runner.evaluate()
    assert result["rows"] == 2
    assert result["test_rows_used"] == 0
    assert result["candidate_pool_invariant"] is True
    assert len(set(result["missing_counts"].values())) == 1
