"""Frozen Phase 4E hybrid neural-transparent training and evaluation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest

from .learned_reranker import (
    GENERIC_CONTEXT_FEATURES,
    HYBRID_PERSONAL_FEATURES,
    CandidateTrainingSet,
    PairwiseLinearReranker,
    deterministic_rank,
    factor_counterfactuals,
)
from .personal_features import BehaviouralHistory
from .personal_vocabulary import PersonalVocabulary
from .phase_04c_evaluation import (
    LU_TEST_WORK_IDS,
    LU_TRAIN_WORK_IDS,
    ZHU_TEST_WORK_IDS,
    ZHU_TRAIN_WORK_IDS,
    compute_metrics,
    count_rank_changes,
    frozen_split,
    parse_work_date,
    target_rank,
)
from .semantic_lm import CausalLMCandidateScorer, semantic_context_64
from .semantic_memory import (
    CachedEmbeddingModel,
    SemanticMemoryInteraction,
    SemanticPersonalMemory,
    SemanticRetrievedInteraction,
    memory_features,
)


BOOTSTRAP_SEED = 40408
BOOTSTRAP_RESAMPLES = 10_000
PHASE_04C_CHECKSUM = "c9a03ae4cdc18bba0facff7bcdd4ec9a0221906859cd001781719d8d646456ff"
PHASE_04D_CHECKSUM = "17c3ef37a416afba87b01de3741cd0c2131b50ad59ef737bdd136c10316d9620"
PHASE_04B6_CHECKSUM = "2d0df837fed3cf6b1a141b9f43677733671cf1f08cb72ca3b9e2f0f2f13f5077"


CONDITION_NAMES = (
    "base",
    "phase_04d_no_gate_correct_user",
    "phase_04e_generic_context",
    "phase_04e_hybrid_fixed_correct_user",
    "phase_04e_hybrid_fixed_wrong_user",
    "phase_04e_hybrid_augmented_correct_user",
    "phase_04e_hybrid_augmented_wrong_user",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            parse_work_date(record["work_date"]),
            record["source_start_offset"],
            record["interaction_id"],
        ),
    )


def _timestamp(record: dict[str, Any]):
    return parse_work_date(record["work_date"]) + timedelta(
        microseconds=int(record["source_start_offset"])
    )


def _semantic_memory_interactions(
    records: Sequence[dict[str, Any]], user_id: str
) -> tuple[SemanticMemoryInteraction, ...]:
    if any(record.get("author_id") != user_id for record in records):
        raise ValueError("semantic history cannot mix users")
    return tuple(
        SemanticMemoryInteraction(
            interaction_id=record["interaction_id"],
            user_id=user_id,
            timestamp=_timestamp(record),
            context=semantic_context_64(record["raw_context"]),
            pinyin=record["pinyin"],
            selected_candidate=record["target_candidate"],
            work_id=record["work_id"],
        )
        for record in records
    )


def _min_max(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if low == high:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _native_pool(record: dict[str, Any]) -> list[dict[str, Any]]:
    ordered = sorted(
        record["candidates"], key=lambda candidate: int(candidate["base_rank"])
    )
    ordinal = [float(len(ordered) - int(item["base_rank"]) + 1) for item in ordered]
    normalized = _min_max(ordinal)
    return [
        {
            "candidate": item["text"],
            "candidate_source": "luna",
            "base_rank": int(item["base_rank"]),
            "base_ordinal_utility": utility,
            "normalized_base_utility": normalized_utility,
            "personal_vocab_injected": 0.0,
            "personal_vocabulary_provenance": [],
            "pool_index": index,
        }
        for index, (item, utility, normalized_utility) in enumerate(
            zip(ordered, ordinal, normalized)
        )
    ]


class Phase04EFeatureExtractor:
    def __init__(
        self,
        lm_scorer: CausalLMCandidateScorer,
        embedding_model: CachedEmbeddingModel,
    ) -> None:
        self.lm_scorer = lm_scorer
        self.embedding_model = embedding_model

    def candidate_pool(
        self,
        record: dict[str, Any],
        history: Sequence[dict[str, Any]],
        *,
        user_id: str,
        augmented: bool,
    ) -> list[dict[str, Any]]:
        pool = _native_pool(record)
        if not augmented:
            return pool
        vocabulary = PersonalVocabulary(history, user_id=user_id)
        injections = vocabulary.inject(
            record["pinyin"], [item["candidate"] for item in pool]
        )
        for item in injections:
            pool.append(
                {
                    "candidate": item.candidate,
                    "candidate_source": item.candidate_source,
                    "base_rank": None,
                    "base_ordinal_utility": 0.0,
                    "normalized_base_utility": 0.0,
                    "personal_vocab_injected": 1.0,
                    "personal_vocabulary_provenance": list(
                        item.provenance_interaction_ids
                    ),
                    "pool_index": len(pool),
                }
            )
        return pool

    def extract(
        self,
        record: dict[str, Any],
        history: Sequence[dict[str, Any]],
        *,
        user_id: str,
        augmented: bool,
        include_personal: bool,
    ) -> dict[str, Any]:
        context_12 = record["derived_context"]
        context_64 = semantic_context_64(record["raw_context"])
        pool = self.candidate_pool(
            record, history, user_id=user_id, augmented=augmented
        )
        candidates = [item["candidate"] for item in pool]
        semantic = {
            item.candidate: item
            for item in self.lm_scorer.score_candidates(context_64, candidates)
        }

        behaviour = None
        retrieved: tuple[SemanticRetrievedInteraction, ...] = ()
        semantic_memory = None
        if include_personal:
            behaviour = BehaviouralHistory(history, user_id=user_id)
            semantic_memory = SemanticPersonalMemory(
                _semantic_memory_interactions(history, user_id),
                self.embedding_model,
                user_id=user_id,
            )
            retrieved = semantic_memory.retrieve(context_64, record["pinyin"])

        retrieval_trace = [self._retrieval_trace(item) for item in retrieved]
        target_behaviour = (
            behaviour.features(record["pinyin"], record["target_candidate"])
            if behaviour is not None
            else None
        )
        candidate_rows = []
        for pool_item in pool:
            candidate = pool_item["candidate"]
            semantic_item = semantic[candidate]
            memory = memory_features(retrieved, candidate)
            behavioural = (
                behaviour.features(record["pinyin"], candidate)
                if behaviour is not None
                else None
            )
            complete_features = {
                "normalized_base_utility": pool_item["normalized_base_utility"],
                "candidate_char_length": float(len(candidate)),
                "personal_vocab_injected": pool_item["personal_vocab_injected"],
                "normalized_lm_conditional": semantic_item.normalized_lm_conditional,
                "normalized_lm_context_gain": semantic_item.normalized_lm_context_gain,
                "memory_weighted_share": memory.memory_weighted_share,
                "memory_max_similarity": memory.memory_max_similarity,
                "memory_support_count": float(memory.memory_support_count),
                "memory_any_support": memory.memory_any_support,
                "log1p_global_count": (
                    behavioural.log1p_global_count if behavioural else 0.0
                ),
                "log1p_same_pinyin_count": (
                    behavioural.log1p_same_pinyin_count if behavioural else 0.0
                ),
                "same_pinyin_selection_share": (
                    behavioural.same_pinyin_selection_share if behavioural else 0.0
                ),
                "candidate_seen_same_pinyin": (
                    behavioural.candidate_seen_same_pinyin if behavioural else 0.0
                ),
                "recency": behavioural.recency if behavioural else 0.0,
            }
            candidate_rows.append(
                {
                    **pool_item,
                    "candidate_char_length": len(candidate),
                    "lm": asdict(semantic_item),
                    "retrieved_memory": retrieval_trace,
                    "memory": asdict(memory),
                    "behaviour": asdict(behavioural) if behavioural else None,
                    "ranking_features": complete_features,
                }
            )
        return {
            "interaction_id": record["interaction_id"],
            "work_id": record["work_id"],
            "work_date": record["work_date"],
            "context_12": context_12,
            "semantic_context_64": context_64,
            "pinyin": record["pinyin"],
            "target": record["target_candidate"],
            "personal_user_id": user_id if include_personal else None,
            "history_size": len(history) if include_personal else 0,
            "eligible_same_pinyin_history_count": (
                semantic_memory.eligible_count(record["pinyin"])
                if semantic_memory is not None
                else 0
            ),
            "retrieved_memory": retrieval_trace,
            "target_behaviour": (
                asdict(target_behaviour) if target_behaviour is not None else None
            ),
            "candidates": candidate_rows,
        }

    @staticmethod
    def _retrieval_trace(item: SemanticRetrievedInteraction) -> dict[str, Any]:
        return {
            "interaction_id": item.interaction.interaction_id,
            "work_id": item.interaction.work_id,
            "historical_context": item.interaction.context,
            "historical_pinyin": item.interaction.pinyin,
            "historical_selected_candidate": item.interaction.selected_candidate,
            "cosine_similarity": item.similarity,
            "nonnegative_weight": max(item.similarity, 0.0),
        }


def build_training_sets(
    author_training_records: Mapping[str, Sequence[dict[str, Any]]],
    extractor: Phase04EFeatureExtractor,
) -> tuple[list[CandidateTrainingSet], list[CandidateTrainingSet]]:
    """Build generic and hybrid rows with strict author-specific prefixes."""
    generic_sets: list[CandidateTrainingSet] = []
    hybrid_sets: list[CandidateTrainingSet] = []
    for user_id in sorted(author_training_records):
        history: list[dict[str, Any]] = []
        for record in ordered_records(author_training_records[user_id]):
            generic = extractor.extract(
                record,
                (),
                user_id=user_id,
                augmented=False,
                include_personal=False,
            )
            generic_sets.append(
                CandidateTrainingSet(
                    target=record["target_candidate"],
                    candidate_features={
                        item["candidate"]: {
                            name: item["ranking_features"][name]
                            for name in GENERIC_CONTEXT_FEATURES
                        }
                        for item in generic["candidates"]
                    },
                )
            )
            hybrid = extractor.extract(
                record,
                history,
                user_id=user_id,
                augmented=True,
                include_personal=True,
            )
            if hybrid["history_size"] != len(history):
                raise AssertionError("training feature history is not the strict prefix")
            hybrid_sets.append(
                CandidateTrainingSet(
                    target=record["target_candidate"],
                    candidate_features={
                        item["candidate"]: {
                            name: item["ranking_features"][name]
                            for name in HYBRID_PERSONAL_FEATURES
                        }
                        for item in hybrid["candidates"]
                    },
                )
            )
            history.append(record)
    return generic_sets, hybrid_sets


def train_rerankers(
    author_training_records: Mapping[str, Sequence[dict[str, Any]]],
    extractor: Phase04EFeatureExtractor,
) -> tuple[PairwiseLinearReranker, PairwiseLinearReranker]:
    generic_sets, hybrid_sets = build_training_sets(
        author_training_records, extractor
    )
    generic = PairwiseLinearReranker(GENERIC_CONTEXT_FEATURES).fit(generic_sets)
    hybrid = PairwiseLinearReranker(HYBRID_PERSONAL_FEATURES).fit(hybrid_sets)
    return generic, hybrid


def _score_extracted(
    extracted: dict[str, Any],
    reranker: PairwiseLinearReranker,
) -> dict[str, Any]:
    scored = []
    for item in extracted["candidates"]:
        model_score = reranker.score(item["ranking_features"])
        scored.append(
            {
                **item,
                **model_score,
            }
        )
    ranked = deterministic_rank(scored)
    target = extracted["target"]
    return {
        **{key: value for key, value in extracted.items() if key != "candidates"},
        "base_rank": next(
            (
                item["base_rank"]
                for item in extracted["candidates"]
                if item["candidate"] == target
            ),
            None,
        ),
        "personalised_rank": next(
            (item["final_rank"] for item in ranked if item["candidate"] == target),
            None,
        ),
        "candidate_coverage": float(any(item["candidate"] == target for item in ranked)),
        "candidates": ranked,
    }


def _base_row(record: dict[str, Any]) -> dict[str, Any]:
    candidates = _native_pool(record)
    target = record["target_candidate"]
    rank = next(
        (item["base_rank"] for item in candidates if item["candidate"] == target), None
    )
    return {
        "interaction_id": record["interaction_id"],
        "work_id": record["work_id"],
        "target": target,
        "base_rank": rank,
        "personalised_rank": rank,
        "candidate_coverage": float(rank is not None),
        "candidates": candidates,
    }


def _condition_summary(
    rows: Sequence[dict[str, Any]], base_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    ranks = [row["personalised_rank"] for row in rows]
    base_ranks = [row["base_rank"] for row in base_rows]
    return {
        "metrics": {
            **asdict(compute_metrics(ranks)),
            "candidate_coverage": sum(rank is not None for rank in ranks) / len(ranks)
            if ranks
            else 0.0,
        },
        "rank_changes": asdict(count_rank_changes(base_ranks, ranks)),
    }


def _augmented_recovery(
    rows: Sequence[dict[str, Any]], base_rows: Sequence[dict[str, Any]]
) -> dict[str, int]:
    absent = [
        (row, base)
        for row, base in zip(rows, base_rows)
        if base["base_rank"] is None
    ]
    ranks = [row["personalised_rank"] for row, _ in absent]
    return {
        "targets_absent_from_luna_top10": len(absent),
        "targets_recovered_into_augmented_pool": sum(rank is not None for rank in ranks),
        "recovered_targets_reaching_top1": sum(rank == 1 for rank in ranks),
        "recovered_targets_reaching_top3": sum(
            rank is not None and rank <= 3 for rank in ranks
        ),
        "recovered_targets_reaching_top5": sum(
            rank is not None and rank <= 5 for rank in ranks
        ),
        "recovered_targets_reaching_top10": sum(
            rank is not None and rank <= 10 for rank in ranks
        ),
    }


def _neural_diagnostics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    present = []
    target_conditional_highest = 0
    target_gain_highest = 0
    for row in rows:
        target = next(
            (item for item in row["candidates"] if item["candidate"] == row["target"]),
            None,
        )
        if target is None:
            continue
        present.append(target)
        max_conditional = max(
            item["lm"]["lm_conditional_logprob"] for item in row["candidates"]
        )
        max_gain = max(item["lm"]["lm_context_gain"] for item in row["candidates"])
        target_conditional_highest += (
            target["lm"]["lm_conditional_logprob"] == max_conditional
        )
        target_gain_highest += target["lm"]["lm_context_gain"] == max_gain
    return {
        "rows_scored_by_lm": len(rows),
        "rows_with_target_in_pool": len(present),
        "mean_target_lm_conditional_score": (
            sum(item["lm"]["lm_conditional_logprob"] for item in present)
            / len(present)
            if present
            else None
        ),
        "mean_target_lm_context_gain": (
            sum(item["lm"]["lm_context_gain"] for item in present) / len(present)
            if present
            else None
        ),
        "percentage_target_highest_lm_conditional": (
            target_conditional_highest / len(present) if present else 0.0
        ),
        "percentage_target_highest_lm_context_gain": (
            target_gain_highest / len(present) if present else 0.0
        ),
    }


def _personal_diagnostics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    similarities = [
        memory["cosine_similarity"]
        for row in rows
        for memory in row["retrieved_memory"]
    ]
    maxima = [
        max(
            (memory["cosine_similarity"] for memory in row["retrieved_memory"]),
            default=0.0,
        )
        for row in rows
    ]
    target_support = sum(
        memory["historical_selected_candidate"] == row["target"]
        for row in rows
        for memory in row["retrieved_memory"]
    )
    competitor_support = sum(
        memory["historical_selected_candidate"] != row["target"]
        for row in rows
        for memory in row["retrieved_memory"]
    )
    return {
        "queries_with_eligible_same_pinyin_history": sum(
            row["eligible_same_pinyin_history_count"] > 0 for row in rows
        ),
        "queries_with_semantic_memory_support": sum(
            any(memory["nonnegative_weight"] > 0 for memory in row["retrieved_memory"])
            for row in rows
        ),
        "mean_retrieved_similarity": (
            sum(similarities) / len(similarities) if similarities else 0.0
        ),
        "mean_maximum_similarity": sum(maxima) / len(maxima) if maxima else 0.0,
        "target_supported_retrieval_count": target_support,
        "competitor_supported_retrieval_count": competitor_support,
        "queries_with_previous_same_pinyin_target_selection": sum(
            row["target_behaviour"] is not None
            and row["target_behaviour"]["candidate_seen_same_pinyin"] > 0
            for row in rows
        ),
        "queries_with_nonzero_recency_evidence": sum(
            any(
                item["behaviour"] is not None
                and item["behaviour"]["recency"] > 0
                for item in row["candidates"]
            )
            for row in rows
        ),
    }


def _reciprocal(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def _paired_bootstrap(
    reference: Sequence[int | None], comparison: Sequence[int | None]
) -> dict[str, Any]:
    if len(reference) != len(comparison):
        raise ValueError("paired bootstrap inputs differ in length")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    count = len(reference)
    mrr_effects = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    rank_effects: list[float] = []
    reference_mrr = np.asarray([_reciprocal(rank) for rank in reference])
    comparison_mrr = np.asarray([_reciprocal(rank) for rank in comparison])
    reference_rank = np.asarray(
        [np.nan if rank is None else float(rank) for rank in reference]
    )
    comparison_rank = np.asarray(
        [np.nan if rank is None else float(rank) for rank in comparison]
    )
    for sample in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, count, size=count)
        mrr_effects[sample] = comparison_mrr[indices].mean() - reference_mrr[
            indices
        ].mean()
        sampled_comparison = comparison_rank[indices]
        sampled_reference = reference_rank[indices]
        if np.isfinite(sampled_comparison).any() and np.isfinite(sampled_reference).any():
            rank_effects.append(
                float(
                    np.nanmean(sampled_comparison) - np.nanmean(sampled_reference)
                )
            )
    reference_present = reference_rank[np.isfinite(reference_rank)]
    comparison_present = comparison_rank[np.isfinite(comparison_rank)]
    return {
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "valid_mean_rank_resamples": len(rank_effects),
        "mrr_effect_comparison_minus_reference": float(
            comparison_mrr.mean() - reference_mrr.mean()
        ),
        "mrr_95_percent_ci": [
            float(value) for value in np.quantile(mrr_effects, [0.025, 0.975])
        ],
        "mean_rank_effect_comparison_minus_reference": (
            float(comparison_present.mean() - reference_present.mean())
            if len(comparison_present) and len(reference_present)
            else None
        ),
        "mean_rank_95_percent_ci": [
            float(value) for value in np.quantile(rank_effects, [0.025, 0.975])
        ]
        if rank_effects
        else None,
    }


def _paired_statistics(
    reference_rows: Sequence[dict[str, Any]], comparison_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    reference = [row["personalised_rank"] for row in reference_rows]
    comparison = [row["personalised_rank"] for row in comparison_rows]
    reference_only = sum(
        ref == 1 and comp != 1 for ref, comp in zip(reference, comparison)
    )
    comparison_only = sum(
        comp == 1 and ref != 1 for ref, comp in zip(reference, comparison)
    )
    discordant = reference_only + comparison_only
    pvalue = (
        float(
            binomtest(
                min(reference_only, comparison_only),
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    return {
        "top1": {
            "test": "paired exact McNemar",
            "reference_correct_comparison_incorrect": reference_only,
            "reference_incorrect_comparison_correct": comparison_only,
            "top1_effect_comparison_minus_reference": (
                sum(rank == 1 for rank in comparison) / len(comparison)
                - sum(rank == 1 for rank in reference) / len(reference)
            ),
            "exact_pvalue": pvalue,
        },
        "bootstrap": _paired_bootstrap(reference, comparison),
    }


def _memory_deletion_counterfactuals(
    row: dict[str, Any], reranker: PairwiseLinearReranker
) -> list[dict[str, Any]]:
    if not row["retrieved_memory"]:
        return []
    original_rank = row["personalised_rank"]
    results = []
    for removed in row["retrieved_memory"]:
        remaining = [
            item
            for item in row["retrieved_memory"]
            if item["interaction_id"] != removed["interaction_id"]
        ]
        denominator = sum(item["nonnegative_weight"] for item in remaining)
        rescored = []
        for candidate in row["candidates"]:
            supported = [
                item
                for item in remaining
                if item["historical_selected_candidate"] == candidate["candidate"]
            ]
            supported_weights = [item["nonnegative_weight"] for item in supported]
            features = dict(candidate["ranking_features"])
            if denominator:
                features.update(
                    memory_weighted_share=sum(supported_weights) / denominator,
                    memory_max_similarity=max(supported_weights, default=0.0),
                    memory_support_count=float(len(supported)),
                    memory_any_support=float(bool(supported)),
                )
            else:
                features.update(
                    memory_weighted_share=0.0,
                    memory_max_similarity=0.0,
                    memory_support_count=0.0,
                    memory_any_support=0.0,
                )
            score = reranker.score(features)
            rescored.append(
                {
                    "candidate": candidate["candidate"],
                    "pool_index": candidate["pool_index"],
                    **score,
                }
            )
        ranked = deterministic_rank(rescored)
        counterfactual_rank = next(
            (
                item["final_rank"]
                for item in ranked
                if item["candidate"] == row["target"]
            ),
            None,
        )
        results.append(
            {
                "removed_memory_interaction_id": removed["interaction_id"],
                "original_rank": original_rank,
                "rank_without_memory": counterfactual_rank,
                "rank_delta": (
                    None
                    if original_rank is None or counterfactual_rank is None
                    else counterfactual_rank - original_rank
                ),
            }
        )
    return results


def _selected_explanations(
    rows: Sequence[dict[str, Any]], reranker: PairwiseLinearReranker
) -> list[dict[str, Any]]:
    changed = sorted(
        (row for row in rows if row["base_rank"] != row["personalised_rank"]),
        key=lambda row: (not bool(row["retrieved_memory"]), row["interaction_id"]),
    )[:6]
    explanations = []
    for row in changed:
        candidates = [
            {
                "candidate": item["candidate"],
                "pool_index": item["pool_index"],
                "final_score": item["final_score"],
                "factor_contributions": item["factor_contributions"],
            }
            for item in row["candidates"]
        ]
        explanations.append(
            {
                "interaction_id": row["interaction_id"],
                "work_id": row["work_id"],
                "context_12": row["context_12"],
                "semantic_context_64": row["semantic_context_64"],
                "pinyin": row["pinyin"],
                "target": row["target"],
                "candidates": row["candidates"],
                "factor_counterfactuals": factor_counterfactuals(
                    candidates, row["target"]
                ),
                "memory_deletion_counterfactuals": _memory_deletion_counterfactuals(
                    row, reranker
                ),
            }
        )
    return explanations


def evaluate_phase_04e(
    zhu_records: Sequence[dict[str, Any]],
    lu_records: Sequence[dict[str, Any]],
    phase_04d_result: dict[str, Any],
    extractor: Phase04EFeatureExtractor,
) -> dict[str, Any]:
    zhu = frozen_split(zhu_records, ZHU_TRAIN_WORK_IDS, ZHU_TEST_WORK_IDS)
    lu = frozen_split(lu_records, LU_TRAIN_WORK_IDS, LU_TEST_WORK_IDS)
    generic_model, hybrid_model = train_rerankers(
        {"zhu_ziqing": zhu.train, "lu_xun": lu.train}, extractor
    )
    zhu_history = ordered_records(zhu.train)
    lu_history = ordered_records(lu.train)
    tests = ordered_records(zhu.test)
    base_rows = [_base_row(record) for record in tests]

    rows: dict[str, list[dict[str, Any]]] = {
        "base": base_rows,
        "phase_04e_generic_context": [],
        "phase_04e_hybrid_fixed_correct_user": [],
        "phase_04e_hybrid_fixed_wrong_user": [],
        "phase_04e_hybrid_augmented_correct_user": [],
        "phase_04e_hybrid_augmented_wrong_user": [],
    }
    for record in tests:
        generic = extractor.extract(
            record,
            (),
            user_id="zhu_ziqing",
            augmented=False,
            include_personal=False,
        )
        rows["phase_04e_generic_context"].append(
            _score_extracted(generic, generic_model)
        )
        for condition, history, user_id, augmented in (
            (
                "phase_04e_hybrid_fixed_correct_user",
                zhu_history,
                "zhu_ziqing",
                False,
            ),
            (
                "phase_04e_hybrid_fixed_wrong_user",
                lu_history,
                "lu_xun",
                False,
            ),
            (
                "phase_04e_hybrid_augmented_correct_user",
                zhu_history,
                "zhu_ziqing",
                True,
            ),
            (
                "phase_04e_hybrid_augmented_wrong_user",
                lu_history,
                "lu_xun",
                True,
            ),
        ):
            extracted = extractor.extract(
                record,
                history,
                user_id=user_id,
                augmented=augmented,
                include_personal=True,
            )
            rows[condition].append(_score_extracted(extracted, hybrid_model))

    phase_04d_condition = {
        subset: json.loads(
            json.dumps(
                phase_04d_result["subsets"][subset][
                    "phase_04d_no_gate_correct_user"
                ]
            )
        )
        for subset in ("full_benchmark", "rerankable")
    }
    subsets = {}
    for subset in ("full_benchmark", "rerankable"):
        indices = (
            list(range(len(tests)))
            if subset == "full_benchmark"
            else [index for index, row in enumerate(base_rows) if row["base_rank"] is not None]
        )
        selected_base = [base_rows[index] for index in indices]
        computed_base = _condition_summary(selected_base, selected_base)
        imported_base_metrics = phase_04d_result["subsets"][subset]["base"]["metrics"]
        for name, value in imported_base_metrics.items():
            if computed_base["metrics"].get(name) != value:
                raise ValueError(
                    f"Phase 4D imported Base metric differs for {subset}/{name}"
                )
        imported_phase_04d = phase_04d_condition[subset]
        imported_phase_04d["metrics"]["candidate_coverage"] = (
            1.0
            - imported_phase_04d["metrics"]["missing_target_count"]
            / imported_phase_04d["metrics"]["evaluated_count"]
        )
        subsets[subset] = {
            "base": computed_base,
            "phase_04d_no_gate_correct_user": imported_phase_04d,
        }
        for condition in rows:
            if condition == "base":
                continue
            selected = [rows[condition][index] for index in indices]
            summary = _condition_summary(selected, selected_base)
            summary["neural_context_diagnostics"] = _neural_diagnostics(selected)
            if "hybrid" in condition:
                summary["personal_diagnostics"] = _personal_diagnostics(selected)
            if "augmented" in condition:
                summary["augmentation_diagnostics"] = _augmented_recovery(
                    selected, selected_base
                )
            subsets[subset][condition] = summary

    comparisons = {
        "base_vs_phase_04e_generic_context": _paired_statistics(
            base_rows, rows["phase_04e_generic_context"]
        ),
        "base_vs_phase_04e_hybrid_fixed_correct_user": _paired_statistics(
            base_rows, rows["phase_04e_hybrid_fixed_correct_user"]
        ),
        "base_vs_phase_04e_hybrid_augmented_correct_user": _paired_statistics(
            base_rows, rows["phase_04e_hybrid_augmented_correct_user"]
        ),
        "hybrid_fixed_correct_vs_wrong_user": _paired_statistics(
            rows["phase_04e_hybrid_fixed_wrong_user"],
            rows["phase_04e_hybrid_fixed_correct_user"],
        ),
    }
    phase_04d_rows = phase_04d_result["evaluation_rows"][
        "phase_04d_no_gate_correct_user"
    ]
    by_work = {
        work_id: {
            **{
                condition: _condition_summary(
                    [row for row in condition_rows if row["work_id"] == work_id],
                    [row for row in base_rows if row["work_id"] == work_id],
                )
                for condition, condition_rows in rows.items()
            },
            "phase_04d_no_gate_correct_user": _condition_summary(
                [row for row in phase_04d_rows if row["work_id"] == work_id],
                [row for row in base_rows if row["work_id"] == work_id],
            ),
        }
        for work_id in ZHU_TEST_WORK_IDS
    }

    return {
        "configuration": {
            "semantic_context_characters": 64,
            "semantic_memory_k": 5,
            "maximum_personal_vocabulary_injections": 3,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "online_test_updates": False,
            "conditions": list(CONDITION_NAMES),
        },
        "splits": phase_04d_result["splits"],
        "learned_models": {
            "generic_context_model": generic_model.model_manifest(),
            "hybrid_personal_model": hybrid_model.model_manifest(),
        },
        "subsets": subsets,
        "per_work": by_work,
        "statistical_validation": comparisons,
        "evaluation_rows": {
            "base": rows["base"],
            "phase_04d_no_gate_correct_user": json.loads(json.dumps(phase_04d_rows)),
            **{condition: values for condition, values in rows.items() if condition != "base"},
        },
        "transparency_examples": {
            "phase_04e_generic_context": _selected_explanations(
                rows["phase_04e_generic_context"], generic_model
            ),
            "phase_04e_hybrid_fixed_correct_user": _selected_explanations(
                rows["phase_04e_hybrid_fixed_correct_user"], hybrid_model
            ),
            "phase_04e_hybrid_augmented_correct_user": _selected_explanations(
                rows["phase_04e_hybrid_augmented_correct_user"], hybrid_model
            ),
        },
        "imported_phase_04d": {
            "source_checksum": PHASE_04D_CHECKSUM,
            "condition": "phase_04d_no_gate_correct_user",
        },
    }
