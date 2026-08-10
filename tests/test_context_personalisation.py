import unittest
from datetime import datetime, timezone

from src.base_ranker import InMemoryBaseRanker
from src.data import BaseCandidate, Interaction
from src.personal_model import FrequencyPersonalModel
from src.reranker import LinearReranker


class ContextPersonalisationTest(unittest.TestCase):
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
    def interaction(day: int, user: str, context: str, candidate: str) -> Interaction:
        return Interaction(
            user,
            datetime(2026, 1, day, tzinfo=timezone.utc),
            context,
            "shiyong",
            candidate,
        )

    def rank(self, model: FrequencyPersonalModel, context: str) -> list[str]:
        candidates = LinearReranker(self.base_ranker, model, alpha=0.4).rank(
            context, "shiyong"
        )
        return [candidate.text for candidate in candidates]

    def test_same_pinyin_can_have_different_preferences_by_context(self) -> None:
        history = [
            self.interaction(1, "user-a", "我们可以", "使用"),
            self.interaction(2, "user-a", "我们可以", "使用"),
            self.interaction(3, "user-a", "这个软件很", "实用"),
            self.interaction(4, "user-a", "这个软件很", "实用"),
        ]
        model = FrequencyPersonalModel().fit(history, "user-a")

        self.assertEqual(self.rank(model, "我们可以")[0], "使用")
        self.assertEqual(self.rank(model, "这个软件很")[0], "实用")

    def test_users_can_learn_different_preferences_for_same_context(self) -> None:
        history = [
            self.interaction(1, "user-a", "我们可以", "使用"),
            self.interaction(2, "user-a", "我们可以", "使用"),
            self.interaction(1, "user-b", "我们可以", "实用"),
            self.interaction(2, "user-b", "我们可以", "实用"),
        ]
        user_a = FrequencyPersonalModel().fit(history, "user-a")
        user_b = FrequencyPersonalModel().fit(history, "user-b")

        self.assertEqual(self.rank(user_a, "我们可以")[0], "使用")
        self.assertEqual(self.rank(user_b, "我们可以")[0], "实用")

    def test_future_context_evidence_is_excluded(self) -> None:
        cutoff = datetime(2026, 1, 2, tzinfo=timezone.utc)
        history = [
            self.interaction(1, "user-a", "我们可以", "使用"),
            self.interaction(2, "user-a", "我们可以", "实用"),
            self.interaction(1, "user-b", "我们可以", "试用"),
        ]
        model = FrequencyPersonalModel().fit(history, "user-a", before=cutoff)

        use = model.score_details("shiyong", "使用", "我们可以")
        practical = model.score_details("shiyong", "实用", "我们可以")
        trial = model.score_details("shiyong", "试用", "我们可以")
        self.assertEqual(
            (use.global_evidence, use.pinyin_evidence, use.context_evidence),
            (1.0, 1.0, 1.0),
        )
        self.assertEqual(practical.combined_score, 0.0)
        self.assertEqual(trial.combined_score, 0.0)

    def test_unseen_context_falls_back_to_pinyin_evidence(self) -> None:
        history = [
            self.interaction(1, "user-a", "我们应该", "使用"),
            self.interaction(2, "user-a", "请继续", "使用"),
            self.interaction(3, "user-a", "你可以", "使用"),
        ]
        model = FrequencyPersonalModel().fit(history, "user-a")
        ranking = LinearReranker(self.base_ranker, model, alpha=0.4).rank(
            "从未见过的上下文", "shiyong"
        )

        self.assertEqual(ranking[0].text, "使用")
        self.assertEqual(ranking[0].context_evidence, 0.0)
        self.assertEqual(ranking[0].pinyin_evidence, 3.0)
        self.assertGreater(ranking[0].personal_score, 0.0)

    def test_ranked_candidates_expose_all_scoring_components(self) -> None:
        model = FrequencyPersonalModel().fit(
            [self.interaction(1, "user-a", "我们可以", "使用")], "user-a"
        )
        candidate = LinearReranker(self.base_ranker, model, alpha=0.4).rank(
            "我们可以", "shiyong"
        )[0]

        self.assertEqual(candidate.global_evidence, 1.0)
        self.assertEqual(candidate.pinyin_evidence, 1.0)
        self.assertEqual(candidate.context_evidence, 1.0)
        self.assertGreater(candidate.personal_score, 0.0)
        self.assertGreater(candidate.final_score, 0.0)


if __name__ == "__main__":
    unittest.main()
