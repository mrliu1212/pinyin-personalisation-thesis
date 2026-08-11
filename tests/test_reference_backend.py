import tempfile
import unittest
from pathlib import Path

from phase_04f_fixtures import make_backend


class ReferenceBackendTests(unittest.TestCase):
    def test_official_trigger_retrieves_same_user_memory_and_grounds_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            backend, generation, _ = make_backend(
                Path(directory),
                user_id="zhu_ziqing",
                trigger=True,
                with_memory=True,
            )
            result = backend.predict(
                "zhu_ziqing",
                "客户张总来访时我应该",
                "woyinggai",
                2,
                external_context=None,
                seed_base=10,
                record_trace=False,
            )
            self.assertTrue(result.memory_trigger.should_retrieve)
            self.assertEqual(result.memory_trigger.method, "OFFICIAL_POLICY")
            self.assertEqual(len(result.supplied_memory_ids), 1)
            self.assertEqual(result.supplied_memory_ids[0], result.retrieved_memories[0].memory_id)
            self.assertEqual(result.retrieved_memories[0].user_id, "zhu_ziqing")
            self.assertEqual(result.candidates[0].text, "准备红茶")
            self.assertEqual(result.provenance["path"], "memory_grounded_generation")
            self.assertTrue(result.provenance["direct_raw_outputs"])
            self.assertTrue(result.provenance["final_raw_outputs"])
            self.assertIn(result.supplied_memory_plaintext[0], generation.calls[-1]["prompt"])

    def test_prediction_does_not_rebuild_personal_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            backend, _, _ = make_backend(
                Path(directory),
                user_id="zhu_ziqing",
                trigger=False,
                with_memory=True,
            )
            memory_ids_before = tuple(item.memory_id for item in backend.memory_store.list())
            index_count_before = len(backend.memory_index)
            result = backend.predict(
                "zhu_ziqing", "春天来了", "chuntian", 2, seed_base=20, record_trace=False
            )
            self.assertEqual(memory_ids_before, tuple(item.memory_id for item in backend.memory_store.list()))
            self.assertEqual(index_count_before, len(backend.memory_index))
            self.assertFalse(result.cache_status["prediction_rebuilt_memory"])

    def test_backend_rejects_another_user(self):
        with tempfile.TemporaryDirectory() as directory:
            backend, _, _ = make_backend(Path(directory), user_id="zhu_ziqing")
            with self.assertRaises(ValueError):
                backend.predict("lu_xun", "前文", "pinyin", 1, record_trace=False)

    def test_input_only_prediction_has_no_fabricated_external_context(self):
        with tempfile.TemporaryDirectory() as directory:
            backend, generation, _ = make_backend(Path(directory), user_id="zhu_ziqing")
            result = backend.predict(
                "zhu_ziqing", "自己的前文", "ziji", 1, external_context=None,
                seed_base=30, record_trace=False,
            )
            self.assertIsNone(result.external_context)
            self.assertTrue(result.provenance["input_only"])
            self.assertIn("<last_msg>\n无\n</last_msg>", generation.calls[0]["prompt"])
            self.assertNotIn("[对方]", generation.calls[0]["prompt"])

    def test_foreground_trace_is_separate_and_reconstructable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend, _, _ = make_backend(root, user_id="zhu_ziqing", trace=True)
            result = backend.predict(
                "zhu_ziqing",
                "自己的前文",
                "ziji",
                1,
                seed_base=40,
                chronological_position="1933-01-01|0001",
                selected_text="候选40",
                record_trace=True,
            )
            traces = backend.interaction_store.list()
            self.assertEqual(len(traces), 1)
            self.assertEqual(traces[0].prediction["query_id"], result.query_id)
            self.assertEqual(traces[0].generated_candidates, tuple(c.text for c in result.candidates))

    def test_backend_source_has_no_phase_04e_or_luna_dependency(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "src/reference_backend/backend.py").read_text(encoding="utf-8")
        generator = (root / "src/reference_backend/candidate_generator.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("phase_04e", source.lower())
        self.assertNotIn("luna", source.lower())
        self.assertNotIn("phase_04e", generator.lower())
        self.assertNotIn("from src.learned_reranker", generator)
        self.assertNotIn("from src.personal_features", generator)


if __name__ == "__main__":
    unittest.main()
