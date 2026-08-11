"""Interpretable standardized linear pairwise reranking for Phase 4E."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


GENERIC_CONTEXT_FEATURES = (
    "normalized_base_utility",
    "candidate_char_length",
    "normalized_lm_conditional",
    "normalized_lm_context_gain",
)

HYBRID_PERSONAL_FEATURES = (
    "normalized_base_utility",
    "candidate_char_length",
    "personal_vocab_injected",
    "normalized_lm_conditional",
    "normalized_lm_context_gain",
    "memory_weighted_share",
    "memory_max_similarity",
    "memory_support_count",
    "memory_any_support",
    "log1p_global_count",
    "log1p_same_pinyin_count",
    "same_pinyin_selection_share",
    "candidate_seen_same_pinyin",
    "recency",
)

FACTOR_GROUPS = {
    "base_ime": (
        "normalized_base_utility",
        "candidate_char_length",
    ),
    "semantic_context": (
        "normalized_lm_conditional",
        "normalized_lm_context_gain",
    ),
    "personal_memory": (
        "memory_weighted_share",
        "memory_max_similarity",
        "memory_support_count",
        "memory_any_support",
    ),
    "historical_behaviour": (
        "log1p_global_count",
        "log1p_same_pinyin_count",
        "same_pinyin_selection_share",
        "candidate_seen_same_pinyin",
        "recency",
    ),
    "personal_vocabulary": ("personal_vocab_injected",),
}


@dataclass(frozen=True)
class CandidateTrainingSet:
    target: str
    candidate_features: Mapping[str, Mapping[str, float]]


def feature_matrix(
    candidates: Sequence[Mapping[str, float]], feature_names: Sequence[str]
) -> np.ndarray:
    return np.asarray(
        [[float(candidate[name]) for name in feature_names] for candidate in candidates],
        dtype=float,
    )


def construct_pairwise_examples(
    training_sets: Sequence[CandidateTrainingSet],
    scaler: StandardScaler,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    differences: list[np.ndarray] = []
    labels: list[int] = []
    for interaction in training_sets:
        if interaction.target not in interaction.candidate_features:
            continue
        names = list(interaction.candidate_features)
        values = feature_matrix(
            [interaction.candidate_features[name] for name in names], feature_names
        )
        standardized = scaler.transform(values)
        by_name = dict(zip(names, standardized))
        target = by_name[interaction.target]
        for name in names:
            if name == interaction.target:
                continue
            difference = target - by_name[name]
            differences.extend((difference, -difference))
            labels.extend((1, 0))
    if not differences:
        raise ValueError("pairwise training produced no target/non-target pairs")
    return np.vstack(differences), np.asarray(labels, dtype=int)


class PairwiseLinearReranker:
    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names = tuple(feature_names)
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            fit_intercept=False,
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=40408,
        )
        self._is_fit = False

    def fit(self, training_sets: Sequence[CandidateTrainingSet]) -> "PairwiseLinearReranker":
        eligible = [
            interaction
            for interaction in training_sets
            if interaction.target in interaction.candidate_features
        ]
        if not eligible:
            raise ValueError("no training interaction has its target in the active pool")
        candidate_rows = [
            features
            for interaction in eligible
            for features in interaction.candidate_features.values()
        ]
        self.scaler.fit(feature_matrix(candidate_rows, self.feature_names))
        pairwise, labels = construct_pairwise_examples(
            eligible, self.scaler, self.feature_names
        )
        self.model.fit(pairwise, labels)
        self._is_fit = True
        return self

    @property
    def coefficients(self) -> dict[str, float]:
        self._require_fit()
        return dict(zip(self.feature_names, self.model.coef_[0]))

    def _require_fit(self) -> None:
        if not self._is_fit:
            raise RuntimeError("pairwise reranker is not fitted")

    def score(self, features: Mapping[str, float]) -> dict[str, Any]:
        self._require_fit()
        raw = feature_matrix([features], self.feature_names)
        standardized = self.scaler.transform(raw)[0]
        coefficients = self.model.coef_[0]
        contributions = standardized * coefficients
        final_score = float(contributions.sum())
        feature_details = {
            name: {
                "raw_value": float(features[name]),
                "standardized_value": float(value),
                "coefficient": float(coefficient),
                "contribution": float(contribution),
            }
            for name, value, coefficient, contribution in zip(
                self.feature_names, standardized, coefficients, contributions
            )
        }
        factor_contributions = {
            factor: sum(
                feature_details[name]["contribution"]
                for name in names
                if name in feature_details
            )
            for factor, names in FACTOR_GROUPS.items()
        }
        if not math.isclose(
            sum(item["contribution"] for item in feature_details.values()),
            final_score,
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise AssertionError("feature contributions do not reconstruct score")
        if not math.isclose(
            sum(factor_contributions.values()),
            final_score,
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise AssertionError("factor contributions do not reconstruct score")
        return {
            "features": feature_details,
            "factor_contributions": factor_contributions,
            "final_score": final_score,
        }

    def model_manifest(self) -> dict[str, Any]:
        self._require_fit()
        return {
            "feature_names": list(self.feature_names),
            "coefficients": [
                {"feature": name, "raw_coefficient": self.coefficients[name]}
                for name in self.feature_names
            ],
            "training_feature_statistics": [
                {
                    "feature": name,
                    "mean": float(mean),
                    "scale": float(scale),
                }
                for name, mean, scale in zip(
                    self.feature_names, self.scaler.mean_, self.scaler.scale_
                )
            ],
            "logistic_regression": {
                "fit_intercept": False,
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": 40408,
            },
        }


def deterministic_rank(scored_candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        scored_candidates,
        key=lambda item: (-item["final_score"], item["pool_index"], item["candidate"]),
    )
    return [dict(item, final_rank=index) for index, item in enumerate(ordered, start=1)]


def factor_counterfactuals(
    scored_candidates: Sequence[dict[str, Any]], target: str
) -> dict[str, dict[str, int | None]]:
    original = deterministic_rank(scored_candidates)
    original_rank = next(
        (item["final_rank"] for item in original if item["candidate"] == target), None
    )
    result = {}
    for factor in FACTOR_GROUPS:
        counterfactual = [
            {
                **item,
                "final_score": item["final_score"]
                - item["factor_contributions"].get(factor, 0.0),
            }
            for item in scored_candidates
        ]
        reranked = deterministic_rank(counterfactual)
        rank_without = next(
            (item["final_rank"] for item in reranked if item["candidate"] == target),
            None,
        )
        result[factor] = {
            "original_rank": original_rank,
            "rank_without_factor": rank_without,
            "rank_delta": (
                None
                if original_rank is None or rank_without is None
                else rank_without - original_rank
            ),
        }
    return result
