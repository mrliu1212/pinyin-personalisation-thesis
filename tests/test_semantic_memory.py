import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.semantic_memory import (
    CachedEmbeddingModel,
    SemanticMemoryInteraction,
    SemanticPersonalMemory,
    SemanticRetrievedInteraction,
    memory_features,
)


class FakeEmbeddingBackend:
    def __init__(self):
        self.query_calls = 0
        self.document_calls = 0

    @staticmethod
    def vector(text):
        return [float(text.count("甲") + 1), float(text.count("乙") + 1), 1.0]

    def encode_query(self, context):
        self.query_calls += 1
        return self.vector(context)

    def encode_document(self, context):
        self.document_calls += 1
        return self.vector(context)


def item(index, context, pinyin="ceshi", candidate="甲", user="zhu_ziqing"):
    return SemanticMemoryInteraction(
        interaction_id=f"item-{index}",
        user_id=user,
        timestamp=datetime(1920, 1, index + 1),
        context=context,
        pinyin=pinyin,
        selected_candidate=candidate,
        work_id="history",
    )


class SemanticMemoryTests(unittest.TestCase):
    def build(self, interactions, directory):
        embedding = CachedEmbeddingModel(
            FakeEmbeddingBackend(), revision="r", cache_dir=Path(directory)
        )
        return SemanticPersonalMemory(
            interactions, embedding, user_id="zhu_ziqing"
        )

    def test_same_pinyin_top5_and_determinism(self):
        with tempfile.TemporaryDirectory() as directory:
            interactions = [item(i, "甲" * (i + 1)) for i in range(6)]
            interactions.append(item(8, "甲", pinyin="qita"))
            memory = self.build(interactions, directory)
            first = memory.retrieve("甲甲", "ceshi")
            second = memory.retrieve("甲甲", "ceshi")
            self.assertEqual(first, second)
            self.assertEqual(len(first), 5)
            self.assertTrue(all(row.interaction.pinyin == "ceshi" for row in first))

    def test_user_isolation_and_zero_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            embedding = CachedEmbeddingModel(
                FakeEmbeddingBackend(), revision="r", cache_dir=Path(directory)
            )
            with self.assertRaisesRegex(ValueError, "cannot mix users"):
                SemanticPersonalMemory(
                    [item(0, "甲"), item(1, "乙", user="lu_xun")],
                    embedding,
                    user_id="zhu_ziqing",
                )
            features = memory_features([], "甲")
            self.assertEqual(features.memory_weighted_share, 0.0)
            self.assertEqual(features.memory_max_similarity, 0.0)
            self.assertEqual(features.memory_support_count, 0)

            nonpositive = memory_features(
                [
                    SemanticRetrievedInteraction(
                        interaction=item(0, "甲", candidate="甲"),
                        similarity=-0.25,
                    )
                ],
                "甲",
            )
            self.assertEqual(nonpositive.memory_weighted_share, 0.0)
            self.assertEqual(nonpositive.memory_max_similarity, 0.0)
            self.assertEqual(nonpositive.memory_support_count, 0)
            self.assertEqual(nonpositive.memory_any_support, 0.0)
            self.assertEqual(nonpositive.memory_total_support, 1)

    def test_memory_evidence_is_deterministic_and_nonnegative_weighted(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = self.build(
                [item(0, "甲甲", candidate="甲"), item(1, "甲乙", candidate="乙")],
                directory,
            )
            retrieved = memory.retrieve("甲甲", "ceshi")
            first = memory_features(retrieved, "甲")
            second = memory_features(retrieved, "甲")
            self.assertEqual(first, second)
            self.assertGreater(first.memory_weighted_share, 0.0)
            self.assertEqual(first.memory_support_count, 1)


if __name__ == "__main__":
    unittest.main()
