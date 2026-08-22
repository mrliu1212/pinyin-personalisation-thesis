from pathlib import Path

import pytest
import torch

from src.personalisation.task_specific_biencoder import (
    assign_inner_split,
    context64,
    mean_pool,
    ranking_metrics,
    refuse_closed_path,
    select_epoch,
    split_position_cutoffs,
    transition_counts,
)


def test_chronological_split_keeps_position_blocks_together() -> None:
    queries = (
        [(f"a{index}", "A", index) for index in range(9)]
        + [("a9", "A", 9), ("a10", "A", 9)]
        + [(f"b{index}", "B", index) for index in range(10)]
    )
    cutoffs = split_position_cutoffs(queries)
    assert assign_inner_split("A", 8, cutoffs) == "inner_fit"
    assert assign_inner_split("A", 9, cutoffs) == "inner_gate"
    assert cutoffs["B"] == 9


def test_epoch_selection_uses_frozen_ties() -> None:
    records = [
        {"epoch": 2, "metrics": {"macro_author_recall_at_1": .7, "micro_recall_at_1": .8, "mrr": .9}},
        {"epoch": 1, "metrics": {"macro_author_recall_at_1": .7, "micro_recall_at_1": .8, "mrr": .9}},
    ]
    assert select_epoch(records)["epoch"] == 1


def test_mean_pool_masks_padding_and_normalizes() -> None:
    hidden = torch.tensor([[[3.0, 0.0], [0.0, 4.0], [100.0, 100.0]]])
    pooled = mean_pool(hidden, torch.tensor([[1, 1, 0]]))
    expected = torch.tensor([[.6, .8]])
    assert torch.allclose(pooled, expected)


def test_ranking_metrics_and_transitions_preserve_missing() -> None:
    rows = [
        {"author": "A", "old": 1, "new": 2},
        {"author": "A", "old": None, "new": 1},
        {"author": "B", "old": 2, "new": 1},
    ]
    metrics = ranking_metrics(rows, "new")
    assert metrics["macro_author_top1"] == pytest.approx(.75)
    assert metrics["missing10"] == 0.0
    assert transition_counts(rows, "old", "new") == {"n": 3, "rescue": 2, "harm": 1, "net": 1}


def test_last64_and_closed_path_guard() -> None:
    assert context64("x" * 70) == "x" * 64
    refuse_closed_path(Path("train_fit.jsonl"))
    with pytest.raises(ValueError):
        refuse_closed_path(Path("dev3000.jsonl"))
    with pytest.raises(ValueError):
        refuse_closed_path(Path("test.jsonl"))
