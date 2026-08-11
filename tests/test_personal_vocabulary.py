import unittest

from src.personal_vocabulary import PersonalVocabulary


def record(index, candidate, user="zhu_ziqing"):
    return {
        "author_id": user,
        "interaction_id": f"history-{index}",
        "pinyin": "ceshi",
        "target_candidate": candidate,
    }


class PersonalVocabularyTests(unittest.TestCase):
    def test_maximum_order_no_duplicates_and_provenance(self):
        history = [
            record(0, "甲"),
            record(1, "乙"),
            record(2, "甲"),
            record(3, "丙"),
            record(4, "丁"),
            record(5, "乙"),
        ]
        vocabulary = PersonalVocabulary(history, user_id="zhu_ziqing")
        injected = vocabulary.inject("ceshi", ["丁"])
        self.assertEqual([item.candidate for item in injected], ["乙", "甲", "丙"])
        self.assertEqual(len(injected), 3)
        self.assertTrue(all(item.candidate_source == "personal_vocabulary" for item in injected))
        self.assertEqual(injected[0].provenance_interaction_ids, ("history-1", "history-5"))
        self.assertNotIn("丁", [item.candidate for item in injected])

    def test_prefix_history_prevents_future_vocabulary_and_wrong_user_mix(self):
        prefix = PersonalVocabulary([record(0, "甲")], user_id="zhu_ziqing")
        self.assertEqual([item.candidate for item in prefix.inject("ceshi", [])], ["甲"])
        with self.assertRaisesRegex(ValueError, "cannot mix users"):
            PersonalVocabulary(
                [record(0, "甲", user="lu_xun")], user_id="zhu_ziqing"
            )


if __name__ == "__main__":
    unittest.main()
