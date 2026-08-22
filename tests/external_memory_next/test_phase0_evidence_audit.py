import pytest

from experiments.external_memory_next.audit_phase0_evidence_v1 import all_author_prior, n_bucket, quantiles, rank_metrics, score_margin


def test_quantiles_are_deterministic() -> None:
    result = quantiles([0, 1, 2, 3, 4])
    assert result["min"] == 0
    assert result["p50"] == 2
    assert result["max"] == 4
    assert result["mean"] == 2


def test_history_buckets_cover_boundaries() -> None:
    values = (0, 1, 2, 3, 5, 6, 10, 11, 20, 21, 50, 51, 100, 101)
    expected = ("0", "1", "2", "3-5", "3-5", "6-10", "6-10", "11-20", "11-20", "21-50", "21-50", "51-100", "51-100", ">100")
    assert tuple(map(n_bucket, values)) == expected


def test_score_margin() -> None:
    assert score_margin({}) == 0
    assert score_margin({"a": .4}) == .4
    assert score_margin({"a": .7, "b": .2, "c": .1}) == pytest.approx(.5)


def test_rank_metrics() -> None:
    rows = [{"author": "a", "rank": 1}, {"author": "a", "rank": 2},
            {"author": "b", "rank": 1}, {"author": "b", "rank": None}]
    result = rank_metrics(rows, "rank")
    assert result["macro_author_top1"] == .5
    assert result["micro_top1"] == .5
    assert result["top3"] == .75
    assert result["missing_at_10"] == .25


def test_all_author_prior() -> None:
    prior = {("a", "bei jing", "北京"): 2, ("b", "bei jing", "北京"): 1}
    totals = {("a", "bei jing"): 4, ("b", "bei jing"): 2}
    assert all_author_prior("bei jing", "北京", prior, totals, ("a", "b")) == (3, 6, .5)
