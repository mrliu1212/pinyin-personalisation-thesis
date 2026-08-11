import tempfile
import unittest
from pathlib import Path

from phase_04f_fixtures import DeterministicEmbeddingRuntime, make_memory
from src.reference_backend.memory_store import MemoryStore
from src.reference_backend.vector_index import HNSWMemoryIndex, l2_normalize


class ReferenceRetrievalTests(unittest.TestCase):
    def test_hnsw_result_maps_to_authoritative_plaintext_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embedding = DeterministicEmbeddingRuntime()
            store = MemoryStore(root / "l2", user_id="zhu_ziqing")
            index = HNSWMemoryIndex(
                root / "hnsw", user_id="zhu_ziqing", dimension=embedding.dimension
            )
            memory = make_memory("zhu_ziqing")
            indexed = index.add(memory, embedding.embed(memory.plaintext))
            store.add(indexed)
            result = index.search(
                user_id="zhu_ziqing", query_vector=embedding.embed("客户来访"), k=20
            )[0]
            self.assertEqual(result.memory_id, indexed.memory_id)
            self.assertEqual(store.get(result.memory_id).plaintext, indexed.plaintext)
            self.assertEqual(result.vector_label, indexed.vector_label)

    def test_reload_preserves_index_mapping_without_orphans(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embedding = DeterministicEmbeddingRuntime()
            store = MemoryStore(root / "l2", user_id="zhu_ziqing")
            index = HNSWMemoryIndex(
                root / "hnsw", user_id="zhu_ziqing", dimension=embedding.dimension
            )
            indexed = index.add(make_memory("zhu_ziqing"), embedding.embed("客户来访"))
            store.add(indexed)
            reloaded_store = MemoryStore(root / "l2", user_id="zhu_ziqing")
            reloaded_index = HNSWMemoryIndex(
                root / "hnsw", user_id="zhu_ziqing", dimension=embedding.dimension
            )
            reloaded_index.validate_against(reloaded_store)
            self.assertEqual(
                reloaded_index.search(
                    user_id="zhu_ziqing", query_vector=embedding.embed("客户来访"), k=1
                )[0].memory_id,
                indexed.memory_id,
            )

    def test_orphan_vector_mapping_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embedding = DeterministicEmbeddingRuntime()
            index = HNSWMemoryIndex(
                root / "hnsw", user_id="zhu_ziqing", dimension=embedding.dimension
            )
            index.add(make_memory("zhu_ziqing"), embedding.embed("客户来访"))
            empty_store = MemoryStore(root / "l2", user_id="zhu_ziqing")
            with self.assertRaises(ValueError):
                index.validate_against(empty_store)

    def test_hnsw_rejects_cross_user_queries_and_records(self):
        with tempfile.TemporaryDirectory() as directory:
            embedding = DeterministicEmbeddingRuntime()
            index = HNSWMemoryIndex(
                Path(directory), user_id="zhu_ziqing", dimension=embedding.dimension
            )
            with self.assertRaises(ValueError):
                index.add(make_memory("lu_xun"), embedding.embed("客户来访"))
            with self.assertRaises(ValueError):
                index.search(user_id="lu_xun", query_vector=embedding.embed("客户来访"))

    def test_embedding_normalization_is_unit_length_and_deterministic(self):
        first = l2_normalize((3.0, 4.0))
        second = l2_normalize((3.0, 4.0))
        self.assertEqual(first.tolist(), second.tolist())
        self.assertAlmostEqual(float(first @ first), 1.0)


if __name__ == "__main__":
    unittest.main()
