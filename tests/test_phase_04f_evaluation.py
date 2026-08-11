import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from phase_04f_fixtures import DeterministicPinyinDecoder, benchmark_record, make_backend
from src.phase_04f_evaluation import (
    CONDITIONS,
    PHASE_04B6_CHECKSUM,
    PHASE_04C_CHECKSUM,
    PHASE_04D_CHECKSUM,
    PHASE_04E_MANIFEST_CHECKSUM,
    deterministic_seed,
    evaluate_phase_04f,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Phase04FEvaluationTests(unittest.TestCase):
    def test_evaluation_uses_exact_conditions_and_fixed_generic_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generic, _, _ = make_backend(root / "generic", user_id="generic_no_memory")
            correct, _, _ = make_backend(
                root / "correct", user_id="zhu_ziqing", with_memory=True
            )
            wrong, _, _ = make_backend(root / "wrong", user_id="lu_xun", with_memory=True)
            record = benchmark_record(target="not-generated")
            result = evaluate_phase_04f(
                (record,),
                backends={
                    "generic_no_memory": generic,
                    "correct_user_memory": correct,
                    "wrong_user_memory": wrong,
                },
                pinyin_decoder=DeterministicPinyinDecoder(),
            )
            self.assertEqual(tuple(result["conditions"]), CONDITIONS)
            row = result["rows"][0]
            self.assertIsNone(row["external_context"])
            self.assertEqual(row["seed_base"], deterministic_seed(record["interaction_id"]))
            prompts = [
                row["huoziime_conditions"][condition]["prediction"]["provenance"]["direct_prompt_sha256"]
                for condition in CONDITIONS
            ]
            seeds = [
                row["huoziime_conditions"][condition]["prediction"]["provenance"]["direct_seeds"]
                for condition in CONDITIONS
            ]
            self.assertEqual(len(set(prompts)), 1)
            self.assertEqual(seeds[0], seeds[1])
            self.assertEqual(seeds[1], seeds[2])
            self.assertFalse(result["test_time_memory_updates"])
            self.assertTrue(result["frozen_history"])

    def test_test_interactions_cannot_enter_correct_or_wrong_user_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generic, _, _ = make_backend(root / "generic", user_id="generic_no_memory")
            correct, _, _ = make_backend(
                root / "correct", user_id="zhu_ziqing", with_memory=False
            )
            wrong, _, _ = make_backend(root / "wrong", user_id="lu_xun", with_memory=False)
            record = benchmark_record(interaction_id="leaked-test")
            from src.reference_backend.memory_store import MemoryRecord

            leaked = MemoryRecord.create(
                user_id="zhu_ziqing",
                plaintext="测试数据泄漏",
                creation_position="1933-07|0001",
                source_interaction_ids=("leaked-test",),
            )
            # Use a writable fixture store to construct the invalid frozen state.
            correct.memory_store.read_only = False
            correct.memory_store.add(leaked)
            with self.assertRaises(ValueError):
                evaluate_phase_04f(
                    (record,),
                    backends={
                        "generic_no_memory": generic,
                        "correct_user_memory": correct,
                        "wrong_user_memory": wrong,
                    },
                    pinyin_decoder=DeterministicPinyinDecoder(),
                )

    def test_evaluation_does_not_mutate_memory_state_or_retrain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backends = {}
            for condition, user_id, memory in (
                ("generic_no_memory", "generic_no_memory", False),
                ("correct_user_memory", "zhu_ziqing", True),
                ("wrong_user_memory", "lu_xun", True),
            ):
                backends[condition], _, _ = make_backend(
                    root / condition, user_id=user_id, with_memory=memory
                )
            before = {
                condition: tuple(item.memory_id for item in backend.memory_store.list())
                for condition, backend in backends.items()
            }
            evaluate_phase_04f(
                (benchmark_record(),),
                backends=backends,
                pinyin_decoder=DeterministicPinyinDecoder(),
            )
            after = {
                condition: tuple(item.memory_id for item in backend.memory_store.list())
                for condition, backend in backends.items()
            }
            self.assertEqual(before, after)

    def test_previous_phase_artifacts_remain_checksum_identical(self):
        expected = {
            ROOT / "data/processed/interactions/zhu_ziqing_simplified_rime/interactions.jsonl": PHASE_04B6_CHECKSUM,
            ROOT / "results/experiments/phase_04c/evaluation.json": PHASE_04C_CHECKSUM,
            ROOT / "results/experiments/phase_04d/evaluation.json": PHASE_04D_CHECKSUM,
            ROOT / "results/experiments/phase_04e/model_manifest.json": PHASE_04E_MANIFEST_CHECKSUM,
        }
        self.assertEqual({path: sha256_file(path) for path in expected}, expected)

    def test_final_evaluation_has_not_been_run_automatically(self):
        self.assertFalse((ROOT / "results/experiments/phase_04f/evaluation.json").exists())


class Phase04FAuditArtifactTests(unittest.TestCase):
    def test_manifest_and_matrix_pin_upstream_and_have_deterministic_fields(self):
        manifest = json.loads(
            (ROOT / "results/experiments/phase_04f/backend_manifest.json").read_text()
        )
        matrix = json.loads(
            (ROOT / "results/audits/phase_04f/reproduction_matrix.json").read_text()
        )
        sha = "63f249e711f6501169e6baafec7e12318b3c765b"
        self.assertEqual(manifest["audit"]["repository_commit"], sha)
        self.assertEqual(matrix["upstream_commit"], sha)
        self.assertEqual(
            manifest["final_reproduction_label"],
            "B. Faithful HuoziIME reference-backend adaptation",
        )
        self.assertEqual(manifest["memory_trigger"]["type"], "OFFICIAL_POLICY")

    def test_reproduction_matrix_is_complete_and_classified(self):
        matrix = json.loads(
            (ROOT / "results/audits/phase_04f/reproduction_matrix.json").read_text()
        )
        required = {
            "LLM base model", "post-trained IME model", "prompt/template",
            "candidate generation", "special action tokens", "memory trigger",
            "memory extraction", "L1", "L2", "L3", "plaintext memory",
            "embedding model", "HNSW", "memory-grounded generation",
            "asynchronous memory update", "quantization", "KV/prefix caching",
            "mobile scheduling", "Android UI", "MCP/chat context", "evaluation assets",
        }
        components = {row["component"] for row in matrix["components"]}
        self.assertTrue(required <= components)
        self.assertTrue(
            all(row["classification"] in {"A", "B", "C", "D", "E"}
                for row in matrix["components"])
        )


if __name__ == "__main__":
    unittest.main()
