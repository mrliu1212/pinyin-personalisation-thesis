from __future__ import annotations

from pathlib import Path

import pytest

from src.personalisation.candidate_memory_m2 import (
    CandidateAwareTemplate,
    PairIdentity,
    PairScoreCache,
    PreparedPair,
    monotonic_support,
    rank_m2,
)
from src.personalisation.context_memory import Candidate, PredictionQuery, rank_of
from src.personalisation.m2_h5000 import M2H5000Runner
from src.personalisation.pilot_a import HistoryIndex


MODEL_HASH = "a" * 64
TOKENIZER_HASH = "b" * 64


class CharacterTokenizer:
    model_max_length = 256

    def encode(self, text: str, add_special_tokens: bool = False, **_: object) -> list[int]:
        del add_special_tokens
        return [ord(value) for value in text]

    def decode(self, values: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(value) for value in values)

    def __call__(self, text: str, text_pair: str, **_: object) -> dict[str, list[int]]:
        return {"input_ids": [0] + self.encode(text) + [2] + self.encode(text_pair) + [2]}


def query(*, context: str = "current context", position: int = 100, author: str = "alice") -> PredictionQuery:
    return PredictionQuery("current-id", author, "current-work", position, context, ("shi", "yong"))


def history(
    row_id: str = "history-id",
    *,
    context: str = "history context",
    target: str = "使用",
    position: int = 50,
    author: str = "alice",
    pinyin: tuple[str, ...] = ("shi", "yong"),
) -> dict:
    return {
        "row_id": row_id,
        "author": author,
        "work_id": "history-work",
        "chronological_position": position,
        "context": context,
        "pinyin_segments": list(pinyin),
        "target": target,
    }


def identity(**changes: str) -> PairIdentity:
    values = {
        "current_id": "current-id",
        "current_context": "current context",
        "pinyin": ("shi", "yong"),
        "historical_id": "history-id",
        "historical_context": "history context",
        "historical_target": "使用",
        "candidate": "使用",
    }
    values.update(changes)
    return PairIdentity(**values)


def cache(path: Path, **changes: object) -> PairScoreCache:
    values = {
        "model_revision": "revision",
        "model_sha256": MODEL_HASH,
        "tokenizer_sha256": TOKENIZER_HASH,
        "max_length": 256,
        "dtype": "float16",
    }
    values.update(changes)
    return PairScoreCache(path, **values)


def test_m2_history_is_same_user_strictly_prior_and_budgeted_before_pinyin() -> None:
    records = [
        history("old-match", position=1),
        history("old-other", position=2, pinyin=("si",)),
        history("recent-other", position=98, pinyin=("si",)),
        history("recent-match", position=99),
        history("wrong-user", position=99, author="bob"),
        history("future", position=101),
    ]
    visible = HistoryIndex(records, history_budget=3).visible(query())
    assert [row["row_id"] for row in visible] == ["recent-match"]


def test_stage1_k10_is_prefix_of_k20() -> None:
    records = [history(str(number), context=str(number), position=number) for number in range(1, 21)]
    embeddings = {"current context": [1.0, 0.0], **{str(number): [1.0, number / 100.0] for number in range(1, 21)}}
    visible = HistoryIndex(records, history_budget=5000).visible(query())
    top20 = M2H5000Runner._stage1(query(), visible, embeddings, 20)
    top10 = M2H5000Runner._stage1(query(), visible, embeddings, 10)
    assert [row["row_id"] for row in top10] == [row["row_id"] for row in top20[:10]]


def test_pair_input_is_candidate_aware_and_has_no_gold_or_future_text() -> None:
    prepared = CandidateAwareTemplate(CharacterTokenizer()).prepare(identity())
    assert "candidate: 使用" in prepared.query_text
    assert "selected: 使用" in prepared.history_text
    assert "GOLD_SENTINEL" not in prepared.query_text + prepared.history_text
    assert "FUTURE_SENTINEL" not in prepared.query_text + prepared.history_text


def test_current_context_changes_pair_input_and_key(tmp_path: Path) -> None:
    first = identity()
    second = identity(current_context="different current context")
    template = CandidateAwareTemplate(CharacterTokenizer())
    store = cache(tmp_path / "scores.sqlite3")
    assert template.prepare(first).query_text != template.prepare(second).query_text
    assert store.key(first) != store.key(second)
    store.close()


def test_history_context_changes_pair_input_and_key(tmp_path: Path) -> None:
    first = identity()
    second = identity(historical_context="different history context")
    template = CandidateAwareTemplate(CharacterTokenizer())
    store = cache(tmp_path / "scores.sqlite3")
    assert template.prepare(first).history_text != template.prepare(second).history_text
    assert store.key(first) != store.key(second)
    store.close()


def test_candidate_identity_changes_pair_input_and_key(tmp_path: Path) -> None:
    first = identity()
    second = identity(candidate="实用")
    template = CandidateAwareTemplate(CharacterTokenizer())
    store = cache(tmp_path / "scores.sqlite3")
    assert template.prepare(first).query_text != template.prepare(second).query_text
    assert store.key(first) != store.key(second)
    store.close()


def test_historical_target_changes_pair_input_and_key(tmp_path: Path) -> None:
    first = identity()
    second = identity(historical_target="实用")
    template = CandidateAwareTemplate(CharacterTokenizer())
    store = cache(tmp_path / "scores.sqlite3")
    assert template.prepare(first).history_text != template.prepare(second).history_text
    assert store.key(first) != store.key(second)
    store.close()


def test_truncation_retains_recent_context_and_all_mandatory_fields() -> None:
    pair = identity(current_context="A" * 100 + "CURRENT_RECENT", historical_context="B" * 100 + "HISTORY_RECENT")
    prepared = CandidateAwareTemplate(CharacterTokenizer(), max_length=128).prepare(pair)
    combined = prepared.query_text + prepared.history_text
    assert "CURRENT_RECENT" in combined
    assert "HISTORY_RECENT" in combined
    assert "pinyin: shi yong" in combined
    assert "candidate: 使用" in combined
    assert "selected: 使用" in combined
    assert prepared.current_context_truncated
    assert prepared.historical_context_truncated
    assert prepared.input_tokens <= 128


def test_pair_cache_key_is_reusable_across_history_budget_names(tmp_path: Path) -> None:
    store = cache(tmp_path / "scores.sqlite3")
    key = store.key(identity())
    assert "H500" not in key and "H5000" not in key and "HFull" not in key
    assert key == store.key(identity())
    store.close()


def test_pair_cache_provenance_mismatch_prevents_reuse(tmp_path: Path) -> None:
    path = tmp_path / "scores.sqlite3"
    cache(path).close()
    with pytest.raises(RuntimeError, match="provenance"):
        cache(path, model_sha256="c" * 64)


def test_pair_cache_resume_does_not_duplicate_scores(tmp_path: Path) -> None:
    path = tmp_path / "scores.sqlite3"
    pair = identity()
    prepared = PreparedPair("query", "history", False, False, 10)
    store = cache(path)
    store.put(pair, prepared, 1.5)
    store.put(pair, prepared, 2.5)
    store.close()
    resumed = cache(path)
    assert resumed.count() == 1
    assert resumed.get(pair)["raw_score"] == pytest.approx(1.5)
    resumed.close()


def test_monotonic_support_is_non_negative_and_order_preserving() -> None:
    values = [monotonic_support(value) for value in (-10.0, -1.0, 0.0, 1.0, 10.0)]
    assert values == sorted(values)
    assert all(0.0 <= value <= 1.0 for value in values)


def test_m2_ranking_does_not_change_candidate_pool() -> None:
    candidates = (Candidate("使用", 1, -1.0), Candidate("实用", 2, -1.1), Candidate("试用", 3, -1.2))
    evidence = ({"historical_target": "实用", "raw_score": 10.0},)
    ranked = rank_m2(candidates, evidence, lambda_m2=4.0)
    assert {row["candidate"] for row in ranked} == {candidate.text for candidate in candidates}
    assert rank_of(ranked, "实用") == 1


def test_zero_history_is_exact_generic_order() -> None:
    candidates = (Candidate("使用", 1, -1.0), Candidate("实用", 2, -2.0))
    ranked = rank_m2(candidates, (), lambda_m2=4.0)
    assert [row["candidate"] for row in ranked] == ["使用", "实用"]


def test_out_of_pool_history_cannot_expand_candidate_surface() -> None:
    candidates = (Candidate("使用", 1, -1.0), Candidate("实用", 2, -2.0))
    evidence = ({"historical_target": "适用", "raw_score": 100.0},)
    ranked = rank_m2(candidates, evidence, lambda_m2=4.0)
    assert [row["candidate"] for row in ranked] == ["使用", "实用"]


def test_prediction_query_has_no_gold_field() -> None:
    assert not hasattr(query(), "gold")
