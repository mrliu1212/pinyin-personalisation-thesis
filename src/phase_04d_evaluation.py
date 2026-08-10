"""Frozen Phase 4D contextual-memory scoring and evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime
from typing import Any, Mapping, Sequence

from .context_similarity import (
    NGRAM_RANGE,
    RETRIEVAL_K,
    ContextualMemory,
    MemoryInteraction,
    RetrievedInteraction,
    contextual_candidate_evidence,
)
from .phase_04c_evaluation import (
    LU_TEST_WORK_IDS,
    LU_TRAIN_WORK_IDS,
    ZHU_TEST_WORK_IDS,
    ZHU_TRAIN_WORK_IDS,
    compute_metrics,
    count_rank_changes,
    evaluate_phase_04c,
    frozen_split,
    parse_work_date,
    target_rank,
)


ALPHA = 0.5
GLOBAL_FREQUENCY_WEIGHT = 0.25
PINYIN_FREQUENCY_WEIGHT = 0.75
NO_GATE = "phase_04d_no_gate"
FULL = "phase_04d_full"


def normalize_frequency_counts(
    candidates: Sequence[str], counts: Mapping[str, int]
) -> dict[str, float]:
    """Normalize one count channel by its candidate-list maximum."""
    maximum = max((counts.get(candidate, 0) for candidate in candidates), default=0)
    if maximum == 0:
        return {candidate: 0.0 for candidate in candidates}
    return {
        candidate: counts.get(candidate, 0) / maximum for candidate in candidates
    }


def interpolate_personal_evidence(
    frequency: float, context: float, confidence: float, *, condition: str
) -> float:
    if not 0.0 <= frequency <= 1.0:
        raise ValueError("frequency evidence must be in [0,1]")
    if not 0.0 <= context <= 1.0:
        raise ValueError("context evidence must be in [0,1]")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("context confidence must be in [0,1]")
    if condition == NO_GATE:
        return context
    if condition == FULL:
        return (1.0 - confidence) * frequency + confidence * context
    raise ValueError(f"unsupported Phase 4D condition: {condition!r}")


def _min_max(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if low == high:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _memory_interaction(record: dict[str, Any], user_id: str) -> MemoryInteraction:
    if record.get("author_id") != user_id:
        raise ValueError(
            f"history record author {record.get('author_id')!r} does not match {user_id!r}"
        )
    return MemoryInteraction(
        interaction_id=record["interaction_id"],
        user_id=user_id,
        timestamp=parse_work_date(record["work_date"]),
        context=record["derived_context"],
        pinyin=record["pinyin"],
        selected_candidate=record["target_candidate"],
        work_id=record["work_id"],
    )


class ContextualPersonalizer:
    """Single-user frozen frequency counts plus contextual retrieval memory."""

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        *,
        user_id: str,
        before: datetime,
    ) -> None:
        interactions = tuple(_memory_interaction(record, user_id) for record in records)
        if any(item.timestamp >= before for item in interactions):
            raise ValueError("personal history contains a test-time or future interaction")
        self.user_id = user_id
        self.before = before
        self.memory = ContextualMemory(interactions, user_id=user_id)
        self.global_counts = Counter(item.selected_candidate for item in interactions)
        self.pinyin_counts = Counter(
            (item.pinyin, item.selected_candidate) for item in interactions
        )

    def score(
        self,
        record: dict[str, Any],
        *,
        condition: str,
    ) -> dict[str, Any]:
        candidates = sorted(
            record["candidates"], key=lambda candidate: int(candidate["base_rank"])
        )
        texts = [candidate["text"] for candidate in candidates]
        ordinal = [float(len(candidates) - int(item["base_rank"]) + 1) for item in candidates]
        normalized_base = _min_max(ordinal)

        global_raw = {candidate: self.global_counts[candidate] for candidate in texts}
        pinyin_raw = {
            candidate: self.pinyin_counts[(record["pinyin"], candidate)]
            for candidate in texts
        }
        global_normalized = normalize_frequency_counts(texts, global_raw)
        pinyin_normalized = normalize_frequency_counts(texts, pinyin_raw)
        frequency = {
            candidate: (
                GLOBAL_FREQUENCY_WEIGHT * global_normalized[candidate]
                + PINYIN_FREQUENCY_WEIGHT * pinyin_normalized[candidate]
            )
            for candidate in texts
        }

        retrieved = self.memory.retrieve(
            record["derived_context"], record["pinyin"], k=RETRIEVAL_K
        )
        context = contextual_candidate_evidence(retrieved, texts)
        confidence = max((item.similarity for item in retrieved), default=0.0)
        personal = {
            candidate: interpolate_personal_evidence(
                frequency[candidate],
                context[candidate],
                confidence,
                condition=condition,
            )
            for candidate in texts
        }
        final_scores = {
            candidate: ALPHA * base + (1.0 - ALPHA) * personal[candidate]
            for candidate, base in zip(texts, normalized_base)
        }
        base_rank = {candidate: index for index, candidate in enumerate(texts, start=1)}
        final_order = sorted(
            texts, key=lambda candidate: (-final_scores[candidate], base_rank[candidate])
        )
        final_rank = {
            candidate: index for index, candidate in enumerate(final_order, start=1)
        }

        retrieval_trace = [self._retrieval_trace(item) for item in retrieved]
        candidate_rows = []
        for candidate, base_ordinal, base_normalized in zip(
            texts, ordinal, normalized_base
        ):
            candidate_rows.append(
                {
                    "candidate": candidate,
                    "base_rank": base_rank[candidate],
                    "base_ordinal_utility": base_ordinal,
                    "normalized_base_utility": base_normalized,
                    "global_count": global_raw[candidate],
                    "normalized_global_evidence": global_normalized[candidate],
                    "pinyin_count": pinyin_raw[candidate],
                    "normalized_pinyin_evidence": pinyin_normalized[candidate],
                    "frequency_evidence": frequency[candidate],
                    "context_evidence": context[candidate],
                    "context_contributors": [
                        trace
                        for trace in retrieval_trace
                        if trace["historical_selected_candidate"] == candidate
                    ],
                    "context_confidence": confidence,
                    "personal_evidence": personal[candidate],
                    "final_score": final_scores[candidate],
                    "final_rank": final_rank[candidate],
                }
            )
        candidate_rows.sort(key=lambda item: item["final_rank"])
        return {
            "condition": condition,
            "user_id": self.user_id,
            "eligible_same_pinyin_history_count": self.memory.eligible_count(
                record["pinyin"]
            ),
            "retrieved_contexts": retrieval_trace,
            "context_confidence": confidence,
            "candidates": candidate_rows,
            "ranking": final_order,
        }

    @staticmethod
    def _retrieval_trace(item: RetrievedInteraction) -> dict[str, Any]:
        return {
            "interaction_id": item.interaction.interaction_id,
            "work_id": item.interaction.work_id,
            "historical_context": item.interaction.context,
            "historical_pinyin": item.interaction.pinyin,
            "historical_selected_candidate": item.interaction.selected_candidate,
            "similarity": item.similarity,
        }


def _evaluate_row(
    record: dict[str, Any],
    personalizer: ContextualPersonalizer,
    condition: str,
) -> dict[str, Any]:
    trace = personalizer.score(record, condition=condition)
    base_texts = [
        candidate["text"]
        for candidate in sorted(
            record["candidates"], key=lambda item: int(item["base_rank"])
        )
    ]
    target = record["target_candidate"]
    return {
        "interaction_id": record["interaction_id"],
        "work_id": record["work_id"],
        "work_date": record["work_date"],
        "context": record["derived_context"],
        "pinyin": record["pinyin"],
        "target": target,
        "base_rank": target_rank(target, base_texts),
        "personalised_rank": target_rank(target, trace["ranking"]),
        **trace,
    }


def _diagnostics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    eligible = sum(row["eligible_same_pinyin_history_count"] > 0 for row in rows)
    nonzero = sum(row["context_confidence"] > 0.0 for row in rows)
    similarities = [
        item["similarity"] for row in rows for item in row["retrieved_contexts"]
    ]
    max_similarities = [row["context_confidence"] for row in rows]
    changed_context = improved_context = harmed_context = 0
    for row in rows:
        if row["context_confidence"] == 0.0:
            continue
        base, personal = row["base_rank"], row["personalised_rank"]
        if base == personal:
            continue
        changed_context += 1
        if base is None or (personal is not None and personal < base):
            improved_context += 1
        else:
            harmed_context += 1
    return {
        "evaluated_queries": count,
        "queries_with_eligible_same_pinyin_history": eligible,
        "queries_with_nonzero_contextual_similarity": nonzero,
        "percentage_with_nonzero_contextual_similarity": (
            nonzero / count if count else 0.0
        ),
        "mean_retrieved_similarity": (
            sum(similarities) / len(similarities) if similarities else 0.0
        ),
        "mean_maximum_similarity": (
            sum(max_similarities) / count if count else 0.0
        ),
        "ranking_changes_with_nonzero_contextual_evidence": changed_context,
        "improved_cases_with_contextual_evidence": improved_context,
        "harmed_cases_with_contextual_evidence": harmed_context,
    }


def _condition_result(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    base_ranks = [row["base_rank"] for row in rows]
    personal_ranks = [row["personalised_rank"] for row in rows]
    return {
        "metrics": asdict(compute_metrics(personal_ranks)),
        "rank_changes": asdict(count_rank_changes(base_ranks, personal_ranks)),
        "diagnostics": _diagnostics(rows),
    }


def _selected_changed(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for change in ("improved", "harmed"):
        matches = []
        for row in rows:
            base, personal = row["base_rank"], row["personalised_rank"]
            is_improved = (base is None and personal is not None) or (
                personal is not None and base is not None and personal < base
            )
            is_harmed = (base is not None and personal is None) or (
                base is not None and personal is not None and personal > base
            )
            if (change == "improved" and is_improved) or (
                change == "harmed" and is_harmed
            ):
                matches.append(row)
        for row in sorted(matches, key=lambda item: item["interaction_id"])[:2]:
            selected.append({"change": change, **row})
    return selected


def evaluate_phase_04d(
    zhu_records: Sequence[dict[str, Any]], lu_records: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Evaluate the five frozen Phase 4D comparison conditions."""
    zhu = frozen_split(zhu_records, ZHU_TRAIN_WORK_IDS, ZHU_TEST_WORK_IDS)
    lu = frozen_split(lu_records, LU_TRAIN_WORK_IDS, LU_TEST_WORK_IDS)
    before = min(parse_work_date(record["work_date"]) for record in zhu.test)
    correct = ContextualPersonalizer(
        zhu.train, user_id="zhu_ziqing", before=before
    )
    wrong = ContextualPersonalizer(lu.train, user_id="lu_xun", before=before)
    ordered_test = sorted(
        zhu.test,
        key=lambda record: (
            parse_work_date(record["work_date"]),
            record["work_id"],
            record["source_start_offset"],
            record["interaction_id"],
        ),
    )
    rows = {
        "phase_04d_no_gate_correct_user": [
            _evaluate_row(record, correct, NO_GATE) for record in ordered_test
        ],
        "phase_04d_full_correct_user": [
            _evaluate_row(record, correct, FULL) for record in ordered_test
        ],
        "phase_04d_full_wrong_user": [
            _evaluate_row(record, wrong, FULL) for record in ordered_test
        ],
    }
    phase_04c = evaluate_phase_04c(zhu_records, lu_records)

    subsets: dict[str, Any] = {}
    for subset_name in ("full_benchmark", "rerankable"):
        if subset_name == "full_benchmark":
            condition_rows = rows
        else:
            condition_rows = {
                name: [row for row in value if row["base_rank"] is not None]
                for name, value in rows.items()
            }
        phase_04c_subset = phase_04c["subsets"][subset_name]
        subsets[subset_name] = {
            "base": phase_04c_subset["base"],
            "phase_04c_frequency_personalisation": phase_04c_subset[
                "correct_user"
            ],
            **{
                name: _condition_result(value)
                for name, value in condition_rows.items()
            },
        }

    return {
        "configuration": {
            "context": {
                "source": "existing derived 12-Chinese-character preceding context",
                "analyzer": "character",
                "ngram_range": list(NGRAM_RANGE),
                "idf": "log((1+n_documents)/(1+document_frequency))+1",
                "term_frequency": "raw count",
                "normalization": "L2",
            },
            "retrieval_k": RETRIEVAL_K,
            "same_pinyin_only": True,
            "positive_similarity_only": True,
            "frequency_weights": {
                "global": GLOBAL_FREQUENCY_WEIGHT,
                "pinyin": PINYIN_FREQUENCY_WEIGHT,
            },
            "alpha": ALPHA,
            "no_gate_formula": "U(y)=C(y)",
            "full_formula": "U(y)=(1-q)*F(y)+q*C(y)",
            "final_formula": "S_final(y)=0.5*Base(y)+0.5*U(y)",
            "online_test_updates": False,
        },
        "splits": phase_04c["splits"],
        "subsets": subsets,
        "transparency_examples": {
            name: _selected_changed(value) for name, value in rows.items()
        },
        "evaluation_rows": rows,
    }
