import unittest

from src.learned_reranker import (
    HYBRID_PERSONAL_FEATURES,
    CandidateTrainingSet,
    PairwiseLinearReranker,
    construct_pairwise_examples,
    factor_counterfactuals,
)


def features(base, semantic, personal):
    values = {name: 0.0 for name in HYBRID_PERSONAL_FEATURES}
    values.update(
        normalized_base_utility=base,
        candidate_char_length=2.0,
        normalized_lm_conditional=semantic,
        normalized_lm_context_gain=semantic,
        memory_weighted_share=personal,
        memory_any_support=float(personal > 0),
        log1p_same_pinyin_count=personal,
        candidate_seen_same_pinyin=float(personal > 0),
    )
    return values


class LearnedRerankerTests(unittest.TestCase):
    def training(self):
        return [
            CandidateTrainingSet(
                target="甲",
                candidate_features={
                    "甲": features(1.0, 1.0, 1.0),
                    "乙": features(0.0, 0.0, 0.0),
                },
            ),
            CandidateTrainingSet(
                target="乙",
                candidate_features={
                    "甲": features(0.0, 0.1, 0.0),
                    "乙": features(1.0, 0.9, 1.0),
                },
            ),
        ]

    def test_pair_construction_and_train_only_scaler(self):
        reranker = PairwiseLinearReranker(HYBRID_PERSONAL_FEATURES)
        training = self.training()
        reranker.fit(training)
        pairs, labels = construct_pairwise_examples(
            training, reranker.scaler, HYBRID_PERSONAL_FEATURES
        )
        self.assertEqual(pairs.shape[0], 4)
        self.assertEqual(labels.tolist(), [1, 0, 1, 0])
        self.assertAlmostEqual(reranker.scaler.mean_[0], 0.5)

    def test_coefficients_are_deterministic_and_score_decomposes(self):
        first = PairwiseLinearReranker(HYBRID_PERSONAL_FEATURES).fit(self.training())
        second = PairwiseLinearReranker(HYBRID_PERSONAL_FEATURES).fit(self.training())
        self.assertEqual(first.coefficients, second.coefficients)
        scored = first.score(features(1.0, 0.8, 1.0))
        self.assertAlmostEqual(
            scored["final_score"],
            sum(item["contribution"] for item in scored["features"].values()),
        )
        self.assertAlmostEqual(
            scored["final_score"], sum(scored["factor_contributions"].values())
        )

    def test_factor_counterfactual_zeroes_only_selected_family_without_retraining(self):
        reranker = PairwiseLinearReranker(HYBRID_PERSONAL_FEATURES).fit(self.training())
        scored = []
        for index, (candidate, row) in enumerate(
            [("甲", features(1.0, 0.8, 1.0)), ("乙", features(0.0, 0.2, 0.0))]
        ):
            result = reranker.score(row)
            scored.append({"candidate": candidate, "pool_index": index, **result})
        coefficients_before = reranker.coefficients.copy()
        counterfactuals = factor_counterfactuals(scored, "甲")
        self.assertEqual(set(counterfactuals), set(scored[0]["factor_contributions"]))
        self.assertEqual(reranker.coefficients, coefficients_before)


if __name__ == "__main__":
    unittest.main()
