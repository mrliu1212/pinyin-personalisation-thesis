"""Frozen chronological evaluation design for the Phase 4C author benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import json
from typing import Any, Iterable, Sequence

from .base_ranker import InMemoryBaseRanker
from .data import BaseCandidate, Interaction, RankedCandidate
from .personal_model import EvidenceWeights, FrequencyPersonalModel
from .reranker import LinearReranker


ZHU_TRAIN_WORK_IDS = (
    "congcong",
    "qinhuai_river",
    "beiying",
    "ahe",
    "moonlight_over_lotus_pond",
)
ZHU_TEST_WORK_IDS = ("to_my_late_wife", "spring")
LU_TRAIN_WORK_IDS = (
    "madmans_diary",
    "kong_yiji",
    "medicine",
    "hometown",
    "new_years_sacrifice",
)
LU_TEST_WORK_IDS = ("takeism", "have_chinese_lost_self_confidence")


@dataclass(frozen=True)
class RankingMetrics:
    evaluated_count: int
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    top10_accuracy: float
    mrr: float
    mean_target_rank: float | None
    missing_target_count: int


@dataclass(frozen=True)
class RankChangeCounts:
    improved: int = 0
    unchanged: int = 0
    harmed: int = 0


@dataclass(frozen=True)
class FrozenSplit:
    train: tuple[dict[str, Any], ...]
    test: tuple[dict[str, Any], ...]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def parse_work_date(value: str) -> datetime:
    """Map catalog year/month precision to its deterministic first day."""
    parts = [int(part) for part in value.split("-")]
    if len(parts) == 1:
        parts.extend((1, 1))
    elif len(parts) == 2:
        parts.append(1)
    if len(parts) != 3:
        raise ValueError(f"unsupported work date: {value!r}")
    return datetime(*parts)


def frozen_split(
    records: Sequence[dict[str, Any]],
    train_work_ids: Sequence[str],
    test_work_ids: Sequence[str],
) -> FrozenSplit:
    train_ids = set(train_work_ids)
    test_ids = set(test_work_ids)
    if train_ids & test_ids:
        raise ValueError("train and test work IDs must be disjoint")
    present_ids = {record["work_id"] for record in records}
    expected_ids = train_ids | test_ids
    if present_ids != expected_ids:
        raise ValueError(
            f"dataset work IDs do not match frozen split: "
            f"missing={sorted(expected_ids - present_ids)}, "
            f"unexpected={sorted(present_ids - expected_ids)}"
        )

    train = tuple(record for record in records if record["work_id"] in train_ids)
    test = tuple(record for record in records if record["work_id"] in test_ids)
    if not train or not test:
        raise ValueError("frozen split requires non-empty train and test partitions")
    latest_train = max(parse_work_date(record["work_date"]) for record in train)
    earliest_test = min(parse_work_date(record["work_date"]) for record in test)
    if latest_train >= earliest_test:
        raise ValueError("training history must be strictly earlier than every test row")
    return FrozenSplit(train=train, test=test)


def record_to_interaction(record: dict[str, Any], user_id: str) -> Interaction:
    return Interaction(
        user_id=user_id,
        timestamp=parse_work_date(record["work_date"]),
        context=record["derived_context"],
        pinyin=record["pinyin"],
        target_candidate=record["target_candidate"],
    )


def build_personal_model(
    records: Iterable[dict[str, Any]],
    user_id: str,
    *,
    before: datetime,
    evidence_weights: EvidenceWeights | None = None,
) -> FrequencyPersonalModel:
    history = [record_to_interaction(record, user_id) for record in records]
    if any(interaction.timestamp >= before for interaction in history):
        raise ValueError("personal history contains a test-time or future interaction")
    return FrequencyPersonalModel(evidence_weights).fit(history, user_id, before=before)


def compute_metrics(ranks: Sequence[int | None]) -> RankingMetrics:
    count = len(ranks)
    if not count:
        return RankingMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, None, 0)
    present = [rank for rank in ranks if rank is not None]
    return RankingMetrics(
        evaluated_count=count,
        top1_accuracy=sum(rank == 1 for rank in ranks) / count,
        top3_accuracy=sum(rank is not None and rank <= 3 for rank in ranks) / count,
        top5_accuracy=sum(rank is not None and rank <= 5 for rank in ranks) / count,
        top10_accuracy=sum(rank is not None and rank <= 10 for rank in ranks) / count,
        mrr=sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / count,
        mean_target_rank=(sum(present) / len(present) if present else None),
        missing_target_count=count - len(present),
    )


def target_rank(target: str, candidates: Sequence[str]) -> int | None:
    try:
        return candidates.index(target) + 1
    except ValueError:
        return None


def count_rank_changes(
    base_ranks: Sequence[int | None], personal_ranks: Sequence[int | None]
) -> RankChangeCounts:
    if len(base_ranks) != len(personal_ranks):
        raise ValueError("rank sequences must have equal length")
    improved = unchanged = harmed = 0
    for base_rank, personal_rank in zip(base_ranks, personal_ranks):
        if base_rank is None and personal_rank is None:
            unchanged += 1
        elif base_rank is None:
            improved += 1
        elif personal_rank is None:
            harmed += 1
        elif personal_rank < base_rank:
            improved += 1
        elif personal_rank > base_rank:
            harmed += 1
        else:
            unchanged += 1
    return RankChangeCounts(improved, unchanged, harmed)


def _base_candidates(record: dict[str, Any]) -> list[BaseCandidate]:
    """Represent Rime's ordinal rank without inventing a probability score."""
    count = len(record["candidates"])
    return [
        BaseCandidate(
            text=candidate["text"],
            base_score=float(count - int(candidate["base_rank"]) + 1),
        )
        for candidate in sorted(
            record["candidates"], key=lambda item: int(item["base_rank"])
        )
    ]


def _ranked_detail(candidate: RankedCandidate, final_rank: int) -> dict[str, Any]:
    return {
        "candidate": candidate.text,
        "base_score": candidate.base_score,
        "global_evidence": candidate.global_evidence,
        "pinyin_evidence": candidate.pinyin_evidence,
        "context_evidence": candidate.context_evidence,
        "personal_score": candidate.personal_score,
        "final_score": candidate.final_score,
        "final_rank": final_rank,
    }


def _evaluate_row(
    record: dict[str, Any],
    correct_model: FrequencyPersonalModel,
    wrong_model: FrequencyPersonalModel,
    alpha: float,
) -> dict[str, Any]:
    candidates = _base_candidates(record)
    ranker = InMemoryBaseRanker({record["pinyin"]: candidates})
    base_texts = [candidate.text for candidate in ranker.rank("", record["pinyin"])]
    correct = LinearReranker(ranker, correct_model, alpha).rank(
        record["derived_context"], record["pinyin"]
    )
    wrong = LinearReranker(ranker, wrong_model, alpha).rank(
        record["derived_context"], record["pinyin"]
    )
    target = record["target_candidate"]
    return {
        "interaction_id": record["interaction_id"],
        "work_id": record["work_id"],
        "work_date": record["work_date"],
        "context": record["derived_context"],
        "pinyin": record["pinyin"],
        "target": target,
        "base_rank": target_rank(target, base_texts),
        "correct_user_rank": target_rank(target, [item.text for item in correct]),
        "wrong_user_rank": target_rank(target, [item.text for item in wrong]),
        "base_candidates": [
            {
                "candidate": candidate.text,
                "base_rank": index,
                "base_score": candidate.base_score,
            }
            for index, candidate in enumerate(candidates, start=1)
        ],
        "correct_user_candidates": [
            _ranked_detail(candidate, index)
            for index, candidate in enumerate(correct, start=1)
        ],
        "wrong_user_candidates": [
            _ranked_detail(candidate, index)
            for index, candidate in enumerate(wrong, start=1)
        ],
    }


def _subset(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    base_ranks = [row["base_rank"] for row in rows]
    correct_ranks = [row["correct_user_rank"] for row in rows]
    wrong_ranks = [row["wrong_user_rank"] for row in rows]
    return {
        "base": {"metrics": asdict(compute_metrics(base_ranks))},
        "correct_user": {
            "metrics": asdict(compute_metrics(correct_ranks)),
            "rank_changes": asdict(count_rank_changes(base_ranks, correct_ranks)),
        },
        "wrong_user": {
            "metrics": asdict(compute_metrics(wrong_ranks)),
            "rank_changes": asdict(count_rank_changes(base_ranks, wrong_ranks)),
        },
    }


def _change(base_rank: int | None, personal_rank: int | None) -> str:
    counts = count_rank_changes([base_rank], [personal_rank])
    if counts.improved:
        return "improved"
    if counts.harmed:
        return "harmed"
    return "unchanged"


def _examples(
    rows: Sequence[dict[str, Any]], rank_key: str, candidate_key: str
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for category in ("improved", "unchanged", "harmed"):
        matches = sorted(
            (
                row
                for row in rows
                if _change(row["base_rank"], row[rank_key]) == category
            ),
            key=lambda row: row["interaction_id"],
        )
        for row in matches[:2]:
            selected.append(
                {
                    "change": category,
                    "interaction_id": row["interaction_id"],
                    "work_id": row["work_id"],
                    "context": row["context"],
                    "pinyin": row["pinyin"],
                    "target": row["target"],
                    "base_rank": row["base_rank"],
                    "personalised_rank": row[rank_key],
                    "base_candidates": row["base_candidates"],
                    "personalised_candidates": row[candidate_key],
                }
            )
    return selected


def evaluate_phase_04c(
    zhu_records: Sequence[dict[str, Any]],
    lu_records: Sequence[dict[str, Any]],
    *,
    alpha: float = 0.5,
    evidence_weights: EvidenceWeights | None = None,
) -> dict[str, Any]:
    """Evaluate frozen histories against later Zhu interactions."""
    zhu = frozen_split(zhu_records, ZHU_TRAIN_WORK_IDS, ZHU_TEST_WORK_IDS)
    lu = frozen_split(lu_records, LU_TRAIN_WORK_IDS, LU_TEST_WORK_IDS)
    before = min(parse_work_date(record["work_date"]) for record in zhu.test)
    weights = evidence_weights or EvidenceWeights()
    correct_model = build_personal_model(
        zhu.train, "zhu_ziqing", before=before, evidence_weights=weights
    )
    wrong_model = build_personal_model(
        lu.train, "lu_xun", before=before, evidence_weights=weights
    )
    ordered_test = sorted(
        zhu.test,
        key=lambda record: (
            parse_work_date(record["work_date"]),
            record["work_id"],
            record["source_start_offset"],
            record["interaction_id"],
        ),
    )
    rows = [
        _evaluate_row(record, correct_model, wrong_model, alpha)
        for record in ordered_test
    ]
    rerankable = [row for row in rows if row["base_rank"] is not None]
    return {
        "configuration": {
            "alpha": alpha,
            "evidence_weights": {
                "global": weights.global_weight,
                "pinyin": weights.pinyin_weight,
                "context": weights.context_weight,
            },
            "base_score_representation": (
                "rank-derived ordinal utility (candidate_count - base_rank + 1); "
                "not a probability or librime numeric score"
            ),
            "online_test_updates": False,
        },
        "splits": {
            "zhu_ziqing": {
                "train_work_ids": list(ZHU_TRAIN_WORK_IDS),
                "test_work_ids": list(ZHU_TEST_WORK_IDS),
                "train_interactions": len(zhu.train),
                "test_interactions": len(zhu.test),
            },
            "lu_xun": {
                "train_work_ids": list(LU_TRAIN_WORK_IDS),
                "test_work_ids": list(LU_TEST_WORK_IDS),
                "train_interactions": len(lu.train),
                "test_interactions": len(lu.test),
                "test_partition_used_for_wrong_user_history": False,
            },
        },
        "subsets": {
            "full_benchmark": _subset(rows),
            "rerankable": _subset(rerankable),
        },
        "transparency_examples": {
            "correct_user": _examples(
                rows, "correct_user_rank", "correct_user_candidates"
            ),
            "wrong_user": _examples(rows, "wrong_user_rank", "wrong_user_candidates"),
        },
    }
