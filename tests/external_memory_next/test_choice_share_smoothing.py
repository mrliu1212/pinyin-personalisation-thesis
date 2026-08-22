import pytest

from experiments.external_memory_next.run_choice_share_smoothing_v1 import smooth_choice
from experiments.external_memory_next.run_smoothing_fusion_retune_v1 import selection_key


def test_alpha_zero_is_raw_share() -> None:
    assert smooth_choice(1, 4, .9, 0) == .25


def test_unseen_candidate_shrinks_toward_zero() -> None:
    assert smooth_choice(1, 1, 0, 4) == .2


def test_population_prior_contributes_pseudocounts() -> None:
    assert smooth_choice(1, 4, .5, 2) == pytest.approx(1 / 3)


def test_invalid_smoothing_input_is_rejected() -> None:
    with pytest.raises(ValueError):
        smooth_choice(2, 1, .5, 1)


def test_selection_key_prefers_primary_metric_then_reference() -> None:
    better = {"macro_author_top1": .8, "micro_top1": .7, "mrr_at_10": .7}
    worse = {"macro_author_top1": .79, "micro_top1": .9, "mrr_at_10": .9}
    assert selection_key(better, (128, 6), (128, 6)) < selection_key(worse, (128, 6), (128, 6))
    assert selection_key(better, (128, 6), (128, 6)) < selection_key(better, (32, 6), (128, 6))
