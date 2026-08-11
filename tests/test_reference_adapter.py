import unittest

from phase_04f_fixtures import benchmark_record
from src.reference_backend.benchmark_adapter import Phase04FBenchmarkAdapter


class ReferenceAdapterTests(unittest.TestCase):
    def test_request_uses_only_strict_preceding_text_and_traces_pinyin(self):
        record = benchmark_record()
        request = Phase04FBenchmarkAdapter().request(
            record, personal_state_user_id="zhu_ziqing", top_k=10
        )
        self.assertEqual(request.preceding_text, record["raw_context"][-100:])
        self.assertEqual(request.pinyin_or_keystrokes, record["pinyin"])
        self.assertNotIn(record["target_candidate"], request.preceding_text)
        self.assertIsNone(request.external_context)

    def test_benchmark_rejects_external_interlocutor_context(self):
        with self.assertRaises(ValueError):
            Phase04FBenchmarkAdapter().request(
                benchmark_record(),
                personal_state_user_id="zhu_ziqing",
                external_context="fabricated partner message",
            )

    def test_training_trajectories_reject_mixed_users(self):
        first = benchmark_record(interaction_id="zhu-train", author_id="zhu_ziqing")
        second = benchmark_record(interaction_id="lu-train", author_id="lu_xun")
        with self.assertRaises(ValueError):
            Phase04FBenchmarkAdapter().training_trajectories(
                (first, second), user_id="zhu_ziqing"
            )

    def test_training_trajectory_is_deterministic_and_source_traceable(self):
        first = benchmark_record(interaction_id="train-1")
        second = benchmark_record(interaction_id="train-2")
        second["source_start_offset"] = 105
        second["source_end_offset"] = 107
        first_result = Phase04FBenchmarkAdapter().training_trajectories(
            (second, first), user_id="zhu_ziqing"
        )
        second_result = Phase04FBenchmarkAdapter().training_trajectories(
            (first, second), user_id="zhu_ziqing"
        )
        self.assertEqual(first_result, second_result)
        self.assertEqual(first_result[0].source_interaction_ids, ("train-1", "train-2"))


if __name__ == "__main__":
    unittest.main()
