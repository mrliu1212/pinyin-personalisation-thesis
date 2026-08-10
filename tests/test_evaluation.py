import unittest
from datetime import datetime, timezone

from experiments.synthetic_phase_03 import (
    WRONG_USER_BY_USER,
    build_base_ranker,
    build_interactions,
)
from src.base_ranker import InMemoryBaseRanker
from src.data import BaseCandidate, Interaction
from src.evaluation import (
    classify_rank_change,
    compute_metrics,
    count_rank_changes,
    evaluate_chronologically,
)
from src.personal_model import FrequencyPersonalModel


class MetricTest(unittest.TestCase):
    def test_top_k_mrr_and_mean_rank(self) -> None:
        metrics = compute_metrics([1, 2, 4, None])

        self.assertEqual(metrics.evaluated_count, 4)
        self.assertEqual(metrics.top1_accuracy, 0.25)
        self.assertEqual(metrics.top3_accuracy, 0.5)
        self.assertEqual(metrics.mrr, 0.4375)
        self.assertAlmostEqual(metrics.mean_target_rank, 7 / 3)
        self.assertEqual(metrics.missing_target_count, 1)

    def test_helpful_harmful_and_unchanged_classification(self) -> None:
        self.assertEqual(classify_rank_change(3, 1), "helpful")
        self.assertEqual(classify_rank_change(1, 2), "harmful")
        self.assertEqual(classify_rank_change(2, 2), "unchanged")
        self.assertEqual(classify_rank_change(None, 2), "helpful")
        self.assertEqual(classify_rank_change(2, None), "harmful")
        self.assertEqual(classify_rank_change(None, None), "unchanged")

        counts = count_rank_changes([3, 1, 2], [1, 2, 2])
        self.assertEqual((counts.helpful, counts.harmful, counts.unchanged), (1, 1, 1))

    def test_missing_targets_have_explicit_metric_policy(self) -> None:
        metrics = compute_metrics([None, None])

        self.assertEqual(metrics.top1_accuracy, 0.0)
        self.assertEqual(metrics.top3_accuracy, 0.0)
        self.assertEqual(metrics.mrr, 0.0)
        self.assertIsNone(metrics.mean_target_rank)
        self.assertEqual(metrics.missing_target_count, 2)


class ChronologicalEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base_ranker = InMemoryBaseRanker(
            {
                "shiyong": [
                    BaseCandidate("实用", 0.9),
                    BaseCandidate("使用", 0.8),
                    BaseCandidate("试用", 0.7),
                ]
            }
        )

    @staticmethod
    def item(day: int, user: str, candidate: str) -> Interaction:
        return Interaction(
            user,
            datetime(2026, 3, day, tzinfo=timezone.utc),
            "我们可以",
            "shiyong",
            candidate,
        )

    def test_history_is_strictly_earlier_for_both_user_conditions(self) -> None:
        interactions = [
            self.item(1, "user-a", "使用"),
            self.item(1, "user-b", "实用"),
            self.item(2, "user-a", "使用"),
            self.item(3, "user-b", "试用"),
        ]
        result = evaluate_chronologically(
            interactions,
            self.base_ranker,
            {"user-a": "user-b", "user-b": "user-a"},
            alpha=0.4,
        )
        record = next(
            item
            for item in result.records
            if item.user_id == "user-a"
            and item.timestamp == datetime(2026, 3, 2, tzinfo=timezone.utc)
        )

        self.assertEqual(record.correct_history_size, 1)
        self.assertEqual(record.wrong_history_size, 1)
        self.assertEqual(record.correct_user_rank, 1)
        self.assertEqual(record.wrong_user_rank, 2)

    def test_interaction_at_same_timestamp_is_not_history(self) -> None:
        timestamp = datetime(2026, 3, 2, tzinfo=timezone.utc)
        interactions = [
            Interaction("user-a", timestamp, "我们可以", "shiyong", "使用"),
            Interaction("user-a", timestamp, "我们可以", "shiyong", "使用"),
            self.item(1, "user-b", "实用"),
        ]
        result = evaluate_chronologically(
            interactions,
            self.base_ranker,
            {"user-a": "user-b", "user-b": "user-a"},
            alpha=0.4,
        )
        user_a_records = [item for item in result.records if item.user_id == "user-a"]

        self.assertTrue(all(item.correct_history_size == 0 for item in user_a_records))
        self.assertTrue(all(item.correct_user_rank == 2 for item in user_a_records))

    def test_absent_target_is_reported_without_an_invented_rank(self) -> None:
        interactions = [
            Interaction(
                "user-a",
                datetime(2026, 3, 1, tzinfo=timezone.utc),
                "",
                "shiyong",
                "适用",
            )
        ]
        result = evaluate_chronologically(
            interactions,
            self.base_ranker,
            {"user-a": "user-b"},
        )

        self.assertIsNone(result.records[0].base_rank)
        self.assertIsNone(result.records[0].correct_user_rank)
        self.assertEqual(result.base.metrics.missing_target_count, 1)
        self.assertIsNone(result.base.metrics.mean_target_rank)


class SyntheticMultiUserEvaluationTest(unittest.TestCase):
    def test_correct_and_wrong_user_conditions_are_both_evaluated(self) -> None:
        result = evaluate_chronologically(
            build_interactions(),
            build_base_ranker(),
            WRONG_USER_BY_USER,
            alpha=0.4,
        )

        self.assertEqual(result.base.metrics.evaluated_count, len(build_interactions()))
        self.assertEqual(
            result.correct_user.metrics.evaluated_count,
            result.wrong_user.metrics.evaluated_count,
        )
        self.assertTrue(
            any(
                record.correct_user_rank != record.wrong_user_rank
                for record in result.records
            )
        )
        self.assertIsNotNone(result.correct_user.reranking_counts)
        self.assertIsNotNone(result.wrong_user.reranking_counts)

    def test_synthetic_fixture_contains_competing_evidence_components(self) -> None:
        cutoff = datetime(2026, 2, 7, tzinfo=timezone.utc)
        model = FrequencyPersonalModel().fit(
            build_interactions(), "user-a", before=cutoff
        )

        broad = model.score_details("xing", "行", "这个款式")
        contextual = model.score_details("xing", "型", "这个款式")
        self.assertGreater(broad.global_evidence, contextual.global_evidence)
        self.assertGreater(contextual.pinyin_evidence, broad.pinyin_evidence)
        self.assertEqual(broad.context_evidence, 0.0)
        self.assertGreater(contextual.context_evidence, 0.0)
        self.assertGreater(contextual.combined_score, broad.combined_score)


if __name__ == "__main__":
    unittest.main()

