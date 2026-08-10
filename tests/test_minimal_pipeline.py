import unittest
from datetime import datetime, timezone

from src.base_ranker import InMemoryBaseRanker
from src.data import BaseCandidate, Interaction
from src.personal_model import EvidenceWeights, FrequencyPersonalModel
from src.reranker import LinearReranker


class MinimalPipelineTest(unittest.TestCase):
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

    def test_history_reranks_a_base_candidate_to_top_one(self) -> None:
        history = [
            Interaction(
                "user-a",
                datetime(2026, 1, day, tzinfo=timezone.utc),
                "我们可以",
                "shiyong",
                "使用",
            )
            for day in range(1, 4)
        ]
        model = FrequencyPersonalModel().fit(history, user_id="user-a")

        base = self.base_ranker.rank("我们可以", "shiyong")
        personal = LinearReranker(self.base_ranker, model, alpha=0.4).rank(
            "我们可以", "shiyong", top_k=3
        )

        self.assertEqual([candidate.text for candidate in base], ["实用", "使用", "试用"])
        self.assertEqual([candidate.text for candidate in personal], ["使用", "实用", "试用"])

    def test_fit_excludes_other_users_and_future_interactions(self) -> None:
        cutoff = datetime(2026, 1, 2, tzinfo=timezone.utc)
        history = [
            Interaction("user-a", datetime(2026, 1, 1, tzinfo=timezone.utc), "", "shiyong", "使用"),
            Interaction("user-a", cutoff, "", "shiyong", "实用"),
            Interaction("user-b", datetime(2026, 1, 1, tzinfo=timezone.utc), "", "shiyong", "试用"),
        ]
        model = FrequencyPersonalModel().fit(history, user_id="user-a", before=cutoff)

        details = model.score_details("shiyong", "使用")
        self.assertEqual(details.global_evidence, 1.0)
        self.assertEqual(details.pinyin_evidence, 1.0)
        self.assertEqual(details.context_evidence, 1.0)
        self.assertEqual(model.score("shiyong", "使用"), details.combined_score)
        self.assertEqual(model.score("shiyong", "实用"), 0.0)
        self.assertEqual(model.score("shiyong", "试用"), 0.0)

    def test_evidence_weights_are_explicit_and_configurable(self) -> None:
        weights = EvidenceWeights(
            global_weight=1.0, pinyin_weight=2.0, context_weight=3.0
        )
        interaction = Interaction(
            "user-a",
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            "我们可以",
            "shiyong",
            "使用",
        )
        model = FrequencyPersonalModel(weights).fit([interaction], "user-a")

        details = model.score_details("shiyong", "使用", "我们可以")
        self.assertEqual(details.combined_score, 6.0)


if __name__ == "__main__":
    unittest.main()
