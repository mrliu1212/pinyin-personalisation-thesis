from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from src.evaluation.deep_author_v2 import AUTHORS, CHECKPOINT_REVISION, OFFICIAL_CODE_REVISION, canonical_json
from src.personalisation.context_memory import PredictionQuery
from src.personalisation.h5000 import H5000Runner, HISTORY_BUDGET, T1_MANIFEST_SHA256
from src.personalisation.pilot_a import EMBEDDING_MODEL_SHA256, EmbeddingCache, HistoryIndex


ROOT = Path(__file__).resolve().parents[1]


def _query(position: int = 100) -> PredictionQuery:
    return PredictionQuery("query", "alice", "test-work", position, "current", ("shi",))


def _history(row_id: str, position: int, pinyin: str = "shi", author: str = "alice") -> dict:
    return {
        "row_id": row_id,
        "author": author,
        "work_id": "history-work",
        "chronological_position": position,
        "context": f"context-{row_id}",
        "pinyin_segments": [pinyin],
        "target": "是",
    }


def test_frozen_t1_full_short_population_is_exactly_6000_and_balanced() -> None:
    path = ROOT / "results/evaluation/deep_author_v2/design/t1_condition_manifest.jsonl"
    text = path.read_text(encoding="utf-8")
    normalized_hash = hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()
    rows = [json.loads(line) for line in text.splitlines()]
    full_short = [row for row in rows if row["condition"] == "full_short"]
    assert normalized_hash == T1_MANIFEST_SHA256
    assert len(rows) == 24_000
    assert len(full_short) == 6_000
    assert len({row["anchor_id"] for row in full_short}) == 6_000
    assert Counter(row["author"] for row in full_short) == Counter({author: 1_000 for author in AUTHORS})
    assert all(row["pinyin_type"] == "full" and row["target_type"] == "short" for row in full_short)


def test_h5000_caps_total_same_user_history_before_pinyin_filtering() -> None:
    records = [
        _history("old-match", 1, "shi"),
        _history("old-other", 2, "si"),
        _history("recent-other-a", 3, "si"),
        _history("recent-match", 4, "shi"),
        _history("recent-other-b", 5, "si"),
        _history("other-user", 6, "shi", "bob"),
        _history("future", 101, "shi"),
    ]
    index = HistoryIndex(records, history_budget=3)
    assert [row["row_id"] for row in index.visible(_query())] == ["recent-match"]


def test_h5000_limit_keeps_most_recent_5000_strictly_prior_records() -> None:
    records = [_history(str(position), position, "shi" if position in {1, 1001, 6000} else "si") for position in range(1, 6001)]
    visible = HistoryIndex(records, HISTORY_BUDGET).visible(_query(position=7000))
    assert [row["row_id"] for row in visible] == ["1001", "6000"]


def _condition(condition_id: str, condition: str) -> dict:
    return {
        "condition_id": condition_id,
        "anchor_id": "anchor",
        "author": "alice",
        "work_id": "work",
        "condition": condition,
        "target_type": "short",
        "pinyin_type": "full" if condition == "full_short" else "initial",
        "context": "context",
        "pinyin_input": "shi" if condition == "full_short" else "s",
        "gold": "是",
        "gold_char_length": 1,
        "source_position_start": 1,
        "source_position_end": 2,
        "source_hash": "source",
        "cleaned_text_hash": "cleaned",
    }


def _prediction(condition: dict) -> dict:
    return {
        **condition,
        "beam_size": 16,
        "top_k": 10,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "official_code_revision": OFFICIAL_CODE_REVISION,
        "runtime_device": "cuda",
        "model_used_context": "context",
        "top10_candidates": [{"rank": 1, "text": "是", "log_probability": -1.0}],
        "gold_rank": 1,
        "top1_correct": True,
        "top3_correct": True,
        "top10_present": True,
        "missing_at_10": False,
        "reciprocal_rank": 1.0,
    }


def test_valid_t1_generic_cache_is_read_only_reused_without_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conditions = [_condition("full", "full_short"), _condition("initial", "initial_short")]
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("".join(canonical_json(_prediction(row)) + "\n" for row in conditions), encoding="utf-8")
    digest = hashlib.sha256(predictions.read_bytes()).hexdigest()
    monkeypatch.setattr("src.personalisation.h5000.T1_PREDICTIONS_SHA256", digest)
    runner = H5000Runner(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / "results", predictions)
    monkeypatch.setattr(runner, "_conditions", lambda: conditions)
    reused = runner._load_t1_generic(expected_total=2, expected_full_short=1)
    assert set(reused) == {"anchor"}
    assert reused["anchor"]["top10_candidates"][0]["text"] == "是"


def test_t1_generic_provenance_mismatch_stops_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    condition = _condition("full", "full_short")
    prediction = _prediction(condition)
    prediction["checkpoint_revision"] = "wrong"
    path = tmp_path / "predictions.jsonl"
    path.write_text(canonical_json(prediction) + "\n", encoding="utf-8")
    monkeypatch.setattr("src.personalisation.h5000.T1_PREDICTIONS_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    runner = H5000Runner(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / "results", path)
    monkeypatch.setattr(runner, "_conditions", lambda: [condition])
    with pytest.raises(RuntimeError, match="non-frozen model"):
        runner._load_t1_generic(expected_total=1, expected_full_short=1)


def test_embedding_identity_is_reusable_across_history_budgets(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "embeddings.sqlite3")
    key_h500 = cache.key("same context")
    key_h5000 = cache.key("same context")
    key_hfull = cache.key("same context")
    assert key_h500 == key_h5000 == key_hfull
    assert "5000" not in key_h5000
    cache.close()
    with pytest.raises(RuntimeError, match="provenance"):
        EmbeddingCache(tmp_path / "embeddings.sqlite3", model_sha256="0" * len(EMBEDDING_MODEL_SHA256))
