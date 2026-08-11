import json
import tempfile
import unittest
from pathlib import Path

from phase_04f_fixtures import (
    DeterministicEmbeddingRuntime,
    DeterministicGenerationRuntime,
    make_memory,
)
from src.reference_backend.benchmark_adapter import TrainingTrajectory
from src.reference_backend.hierarchical_memory import BackgroundMemoryProcessor
from src.reference_backend.interaction_store import InteractionTrace, InteractionTraceStore
from src.reference_backend.memory_extractor import HuoziIMEMemoryExtractor
from src.reference_backend.memory_store import MemoryStore
from src.reference_backend.vector_index import HNSWMemoryIndex


class ReferenceMemoryTests(unittest.TestCase):
    def test_memory_id_is_stable_and_provenance_is_preserved(self):
        first = make_memory("zhu_ziqing", source_id="source-a")
        second = make_memory("zhu_ziqing", source_id="source-a")
        self.assertEqual(first.memory_id, second.memory_id)
        self.assertEqual(first.source_interaction_ids, ("source-a",))
        self.assertEqual(first.provenance["work_id"], "train-work")

    def test_plaintext_memories_are_individually_addressable_after_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = MemoryStore(root, user_id="zhu_ziqing")
            record = make_memory("zhu_ziqing")
            store.add(record)
            reloaded = MemoryStore(root, user_id="zhu_ziqing")
            self.assertEqual(reloaded.get(record.memory_id), record)
            self.assertEqual(reloaded.list(), (record,))

    def test_memory_store_rejects_cross_user_state(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory), user_id="zhu_ziqing")
            with self.assertRaises(ValueError):
                store.add(make_memory("lu_xun"))

    def test_interaction_ids_are_stable_and_l3_is_chronological(self):
        values = dict(
            user_id="zhu_ziqing",
            chronological_position="1933-01-01|0001",
            preceding_text="春天来了",
            pinyin_or_keystrokes="chuntian",
            selected_text="春天",
            prediction={"candidates": ["春天"]},
            memory_triggered=False,
            retrieved_memory_ids=(),
            generated_candidates=("春天",),
        )
        first = InteractionTrace.create(**values)
        self.assertEqual(first.interaction_id, InteractionTrace.create(**values).interaction_id)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = InteractionTraceStore(root, user_id="zhu_ziqing")
            store.append(first)
            earlier = InteractionTrace.create(
                **{**values, "chronological_position": "1932-01-01|0001"}
            )
            with self.assertRaises(ValueError):
                store.append(earlier)
            self.assertEqual(InteractionTraceStore(root, user_id="zhu_ziqing").list(), (first,))

    def test_background_memory_processing_is_explicit_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = DeterministicGenerationRuntime(extraction=True)
            embedding = DeterministicEmbeddingRuntime()
            store = MemoryStore(root / "l2", user_id="zhu_ziqing")
            index = HNSWMemoryIndex(
                root / "hnsw", user_id="zhu_ziqing", dimension=embedding.dimension
            )
            processor = BackgroundMemoryProcessor(
                user_id="zhu_ziqing",
                store=store,
                index=index,
                extractor=HuoziIMEMemoryExtractor(generation),
                embedding_runtime=embedding,
                trace_path=root / "l3/background.jsonl",
            )
            trajectory = TrainingTrajectory(
                user_id="zhu_ziqing",
                work_id="train-work",
                chronological_position="1925-01-01|000000000100|train-work",
                text="客户张总本周五来访，明确约定准备红茶。",
                source_interaction_ids=("zhu-train-1", "zhu-train-2"),
            )
            result = processor.process((trajectory,))
            self.assertEqual(result[0].status, "extracted")
            self.assertEqual(store.list()[0].source_interaction_ids, trajectory.source_interaction_ids)
            self.assertEqual(len(index), 1)
            self.assertTrue((root / "l3/background.jsonl").is_file())
            trace = json.loads((root / "l3/background.jsonl").read_text().splitlines()[0])
            self.assertEqual(trace["source_interaction_ids"], list(trajectory.source_interaction_ids))

    def test_non_array_participants_are_ignored_like_upstream_jsonobject(self):
        text, what = HuoziIMEMemoryExtractor._index_text(
            {"summary": "客户本周五来访", "participants": "张总", "item": "接待"}
        )
        self.assertEqual(text, "客户本周五来访 | 事项: 接待")
        self.assertEqual(what, "接待")


if __name__ == "__main__":
    unittest.main()
