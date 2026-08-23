from __future__ import annotations

import unittest

from experiments.model_level.run_adapter_dev_evaluation_v1 import (
    prediction_row,
)


ROW = {
    "row_id": "unsupported-pinyin-test-row",
    "author": "Etinjat",
    "context": "test context",
    "pinyin_segments": ["chua"],
    "gold": "欻",
}


class UnsupportedBackend:
    device = "cpu"

    def generate(
        self,
        context,
        pinyin,
        *,
        top_k,
        beam_size,
    ):
        raise ValueError(
            "no tokenizer candidates for Pinyin 'chua'"
        )


class UnrelatedFailureBackend:
    device = "cpu"

    def generate(
        self,
        context,
        pinyin,
        *,
        top_k,
        beam_size,
    ):
        raise ValueError(
            "some unrelated evaluation failure"
        )


class FakeCandidate:
    def __init__(
        self,
        text: str,
        log_probability: float,
    ) -> None:
        self.text = text
        self.log_probability = log_probability


class FakeGenerated:
    runtime_device = "cpu"

    def __init__(self) -> None:
        self.candidates = (
            FakeCandidate("欻", -1.0),
            FakeCandidate("歘", -2.0),
        )


class SupportedBackend:
    device = "cpu"

    def generate(
        self,
        context,
        pinyin,
        *,
        top_k,
        beam_size,
    ):
        return FakeGenerated()


class UnsupportedPinyinEvaluationTest(unittest.TestCase):
    def test_unsupported_pinyin_is_explicit_miss(self) -> None:
        result = prediction_row(
            ROW,
            UnsupportedBackend(),
            condition="test",
            top_k=10,
            beam_size=16,
        )

        self.assertEqual(
            result["top10_candidates"],
            [],
        )
        self.assertEqual(
            result["top10_candidate_scores"],
            [],
        )
        self.assertIsNone(
            result["gold_top10_rank"]
        )
        self.assertFalse(
            result["top1_correct"]
        )
        self.assertFalse(
            result["top3_correct"]
        )
        self.assertFalse(
            result["top5_correct"]
        )
        self.assertFalse(
            result["top10_present"]
        )
        self.assertEqual(
            result["reciprocal_rank_at_10"],
            0.0,
        )
        self.assertTrue(
            result["unsupported_input"]
        )
        self.assertEqual(
            result["unsupported_reason"],
            "no tokenizer candidates for Pinyin 'chua'",
        )

    def test_unrelated_value_error_still_raises(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "some unrelated evaluation failure",
        ):
            prediction_row(
                ROW,
                UnrelatedFailureBackend(),
                condition="test",
                top_k=10,
                beam_size=16,
            )

    def test_supported_generation_is_unchanged(self) -> None:
        result = prediction_row(
            ROW,
            SupportedBackend(),
            condition="test",
            top_k=10,
            beam_size=16,
        )

        self.assertEqual(
            result["gold_top10_rank"],
            1,
        )
        self.assertTrue(
            result["top1_correct"]
        )
        self.assertTrue(
            result["top10_present"]
        )
        self.assertEqual(
            result["reciprocal_rank_at_10"],
            1.0,
        )
        self.assertFalse(
            result["unsupported_input"]
        )
        self.assertIsNone(
            result["unsupported_reason"]
        )


if __name__ == "__main__":
    unittest.main()
