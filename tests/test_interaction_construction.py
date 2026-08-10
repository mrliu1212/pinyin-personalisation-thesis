import unittest

from interactions.candidates import Candidate
from interactions.construction import construct_work_interactions, target_rank
from interactions.linguistic import (
    JiebaSegmenter,
    TargetPolicy,
    convert_full_pinyin,
    derived_context,
    exclusion_reason,
)


class FakeCandidateGenerator:
    name = "fake"
    version = "test"
    schema_id = "fake_pinyin"
    max_candidates = 3

    def __init__(self) -> None:
        self.responses = {
            "women": [Candidate("我们", 1), Candidate("我门", 2)],
            "xuyao": [Candidate("需要", 1), Candidate("须要", 2)],
            "shiyong": [Candidate("实用", 1), Candidate("使用", 2)],
            "zhezhong": [Candidate("这种", 1)],
            "fangfa": [Candidate("方法", 1)],
        }

    def candidates(self, pinyin_input: str) -> list[Candidate]:
        return list(self.responses.get(pinyin_input, []))


WORK = {
    "author_id": "zhu_ziqing",
    "author_name": "朱自清",
    "work_id": "fixture",
    "work_title": "测试作品",
    "chronology": {"value": "1925-10", "precision": "month", "certainty": "certain"},
    "processed_file": "fixture.txt",
    "source_page_url": "https://example.test/work",
    "source_revision_id": 123,
}


class LinguisticProcessingTest(unittest.TestCase):
    def test_segmentation_produces_lexical_units_with_offsets(self) -> None:
        text = "我们需要使用这种方法。"
        tokens = JiebaSegmenter().segment(text)
        lexical = [(item.text, item.start, item.end) for item in tokens if len(item.text) == 2]

        self.assertIn(("我们", 0, 2), lexical)
        self.assertIn(("需要", 2, 4), lexical)
        self.assertIn(("使用", 4, 6), lexical)
        self.assertIn(("方法", 8, 10), lexical)

    def test_tone_free_pinyin_is_normalized_and_polyphony_is_flagged(self) -> None:
        use = convert_full_pinyin("使用")
        woman = convert_full_pinyin("女人")
        important = convert_full_pinyin("重要")

        self.assertEqual(use.normalized, "shiyong")
        self.assertEqual(use.syllables, ("shi", "yong"))
        self.assertEqual(woman.normalized, "nvren")
        self.assertEqual(important.normalized, "zhongyao")
        self.assertTrue(
            any(item["character"] == "重" for item in important.polyphonic_characters)
        )

    def test_filtering_policy_reports_punctuation_non_chinese_and_lengths(self) -> None:
        policy = TargetPolicy(min_characters=2, max_characters=4)

        self.assertIsNone(exclusion_reason("使用", policy))
        self.assertEqual(exclusion_reason("。", policy), "non_chinese_or_punctuation")
        self.assertEqual(exclusion_reason("abc", policy), "non_chinese_or_punctuation")
        self.assertEqual(exclusion_reason("1号", policy), "non_chinese_or_punctuation")
        self.assertEqual(exclusion_reason("我", policy), "below_minimum_length")
        self.assertEqual(exclusion_reason("中华人民共和国", policy), "above_maximum_length")

    def test_context_suffix_is_transparent_and_preserves_raw_context_separately(self) -> None:
        raw = "开头 ABC。我们确实需要"

        self.assertEqual(derived_context(raw, 5), "们确实需要")


class InteractionConstructionTest(unittest.TestCase):
    def test_interactions_preserve_chronology_offsets_context_and_target_rank(self) -> None:
        text = "我们需要使用这种方法。"
        result = construct_work_interactions(
            text,
            WORK,
            JiebaSegmenter(),
            FakeCandidateGenerator(),
            TargetPolicy(derived_context_characters=4),
        )
        use = next(item for item in result.interactions if item["target_candidate"] == "使用")

        self.assertEqual(use["work_date"], "1925-10")
        self.assertEqual(use["work_date_precision"], "month")
        self.assertEqual((use["source_start_offset"], use["source_end_offset"]), (4, 6))
        self.assertEqual(use["raw_context"], "我们需要")
        self.assertEqual(use["derived_context"], "我们需要")
        self.assertEqual(use["target_rank"], 2)
        self.assertTrue(use["target_present"])
        self.assertIsNone(use["candidates"][0]["base_score"])

    def test_missing_target_is_retained_explicitly(self) -> None:
        result = construct_work_interactions(
            "我们",
            WORK,
            JiebaSegmenter(),
            FakeCandidateGenerator(),
            TargetPolicy(),
        )
        interaction = result.interactions[0]

        self.assertTrue(interaction["target_present"])  # fake ranks 我们 first
        self.assertIsNone(target_rank("不存在", interaction["candidates"]))

        generator = FakeCandidateGenerator()
        generator.responses["women"] = [Candidate("我门", 1)]
        missing = construct_work_interactions(
            "我们", WORK, JiebaSegmenter(), generator, TargetPolicy()
        ).interactions[0]
        self.assertFalse(missing["target_present"])
        self.assertIsNone(missing["target_rank"])

    def test_fixed_input_and_configuration_produce_deterministic_records(self) -> None:
        arguments = ("我们需要使用。", WORK)
        first = construct_work_interactions(
            *arguments,
            JiebaSegmenter(),
            FakeCandidateGenerator(),
            TargetPolicy(),
        )
        second = construct_work_interactions(
            *arguments,
            JiebaSegmenter(),
            FakeCandidateGenerator(),
            TargetPolicy(),
        )

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
