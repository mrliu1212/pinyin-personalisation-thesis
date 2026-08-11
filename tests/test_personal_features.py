import unittest

from src.personal_features import BehaviouralHistory


def record(index, pinyin, candidate, user="zhu_ziqing"):
    return {
        "author_id": user,
        "interaction_id": str(index),
        "pinyin": pinyin,
        "target_candidate": candidate,
    }


class PersonalFeatureTests(unittest.TestCase):
    def test_counts_share_recency_and_unseen_behavior(self):
        history = BehaviouralHistory(
            [
                record(0, "shi", "是"),
                record(1, "ta", "他"),
                record(2, "shi", "是"),
                record(3, "shi", "事"),
            ],
            user_id="zhu_ziqing",
        )
        seen = history.features("shi", "是")
        self.assertEqual(seen.global_count, 2)
        self.assertEqual(seen.same_pinyin_count, 2)
        self.assertAlmostEqual(seen.same_pinyin_selection_share, 2 / 3)
        self.assertEqual(seen.candidate_seen_same_pinyin, 1.0)
        self.assertEqual(seen.recency, 0.5)
        unseen = history.features("shi", "诗")
        self.assertEqual(unseen.same_pinyin_count, 0)
        self.assertEqual(unseen.recency, 0.0)

    def test_history_rejects_other_user(self):
        with self.assertRaisesRegex(ValueError, "cannot mix users"):
            BehaviouralHistory(
                [record(0, "shi", "是", user="lu_xun")], user_id="zhu_ziqing"
            )


if __name__ == "__main__":
    unittest.main()
