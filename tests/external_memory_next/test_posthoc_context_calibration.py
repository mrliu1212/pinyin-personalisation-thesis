from dataclasses import dataclass

import numpy as np
import pytest

from experiments.external_memory_next.prepare_posthoc_context_support_v1 import union_candidates
from src.personalisation.posthoc_context_calibration import (
    cosine_top5_support,
    merge_personal_recovery,
    rerank_fixed_surface,
    restrict_and_normalize,
    selection_key,
)


@dataclass(frozen=True)
class Record:
    row_id: str
    position: int
    target: str
    context: str


@dataclass(frozen=True)
class Visible:
    record: Record
    age: int


def test_recency_changes_aggregation_not_cosine_top5_membership():
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    visible = [Visible(Record(str(i), i, "a", str(i)), age=100 - i) for i in range(6)]
    vectors = {str(i): np.asarray([1.0 - i * .05, 0.0], dtype=np.float32) for i in range(6)}
    vectors["5"] = np.asarray([0.1, 0.0], dtype=np.float32)
    plain = cosine_top5_support(
        query_vector=query, candidates=["a", "b"], visible=visible, vectors=vectors,
        tau=None, normalize=False,
    )
    recency = cosine_top5_support(
        query_vector=query, candidates=["a", "b"], visible=visible, vectors=vectors,
        tau=10.0, normalize=False,
    )
    assert plain["a"] == pytest.approx(sum([1.0, .95, .9, .85, .8]))
    assert recency["a"] < plain["a"]
    assert plain["b"] == recency["b"] == 0.0


def test_fixed_surface_rejects_candidate_drift_and_preserves_ties():
    surface = [
        {"candidate": "a", "final_score": 1.0, "rank": 1},
        {"candidate": "b", "final_score": 1.0, "rank": 2},
    ]
    assert [row["candidate"] for row in rerank_fixed_surface(surface, [(1.0, {"a": 0.5, "b": 0.5})])] == ["a", "b"]
    with pytest.raises(ValueError, match="candidate set"):
        rerank_fixed_surface(surface, [(1.0, {"a": 1.0})])


def test_recovery_merge_keeps_frozen_pool_and_generic_tie_preference():
    generic = [{"candidate": "g", "final_score": 0.0, "generic_rank": 1, "source": "generic"}]
    ranking = merge_personal_recovery(
        generic_candidates=generic,
        personal_k5=["p2", "p1"],
        personal_supports=[(1.0, {"p1": 0.0, "p2": 0.0})],
        boundary=0.0,
        tiebreak_support={"p1": 1.0, "p2": 0.0},
    )
    assert [row["candidate"] for row in ranking] == ["g", "p1", "p2"]


def test_equal_grid_selection_uses_declared_tie_breaks():
    metrics = {"macro_author_top1": 0.5, "mrr_at_10": 0.7}
    rows = [
        {"lambda_n": 2.0, "lambda_e": 2.0, "metrics": metrics},
        {"lambda_n": 1.0, "lambda_e": 2.0, "metrics": metrics},
        {"lambda_n": 2.0, "lambda_e": 1.0, "metrics": metrics},
    ]
    assert max(rows, key=selection_key) == rows[2]


def test_restrict_and_normalize_uses_only_current_surface():
    assert restrict_and_normalize({"a": 2.0, "b": 1.0, "c": 9.0}, ["a", "b"]) == pytest.approx({"a": 2 / 3, "b": 1 / 3})


def test_initial_support_union_includes_original_generic_recovery_surface():
    feature = {
        "personal_k5": ["personal"],
        "bases": {
            "K5+Entropy": {"top10": ["stage1-a"]},
            "4P+4CS+2E": {"top10": ["stage1-b"]},
            "6P+2CS+.25E": {"top10": ["stage1-c"]},
        },
    }
    frequency = {"frequency_candidates": [{"candidate": "generic-only"}]}
    assert union_candidates("initial", feature, {}, frequency) == [
        "personal", "generic-only", "stage1-a", "stage1-b", "stage1-c",
    ]
