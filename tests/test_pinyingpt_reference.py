import tempfile
import unittest
from pathlib import Path

from src.reference_backend_pinyingpt.backend import PinyinGPTConcatBackend


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / ".build/pinyingpt2-concat"


class PinyinGPTReferenceUnitTests(unittest.TestCase):
    def test_normalize_pinyin_removes_tone_and_normalizes_umlaut(self):
        self.assertEqual(PinyinGPTConcatBackend._normalize_pinyin("nǐ3 u:e"), "ni ve")

    def test_missing_checkpoint_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "checkpoint is incomplete"):
                PinyinGPTConcatBackend(Path(directory), device="cpu")


@unittest.skipUnless(
    (CHECKPOINT / "pytorch_model.bin").is_file(),
    "project-linked PinyinGPT2-Concat checkpoint is not downloaded",
)
class PinyinGPTReferenceRealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = PinyinGPTConcatBackend(CHECKPOINT)

    def test_real_generation_is_ranked_and_pinyin_compatible(self):
        result = self.backend.generate("这个工具真的很", "shi yong", top_k=5, beam_size=8)
        self.assertEqual(result.segmented_pinyin, ("shi", "yong"))
        self.assertEqual(len(result.candidates), 5)
        self.assertEqual([item.rank for item in result.candidates], [1, 2, 3, 4, 5])
        self.assertEqual(
            [item.log_probability for item in result.candidates],
            sorted((item.log_probability for item in result.candidates), reverse=True),
        )
        for candidate in result.candidates:
            self.assertEqual(len(candidate.text), 2)
            for character, syllable in zip(candidate.text, result.segmented_pinyin):
                self.assertIn(character, self.backend.pinyin2char[syllable])

    def test_fixed_compatible_candidates_receive_comparable_exact_scores(self):
        candidates = ("使用", "实用", "适用", "试用")
        scored = self.backend.score_candidates(
            "这个工具真的很", "shi yong", candidates
        )
        self.assertEqual({item.text for item in scored}, set(candidates))
        self.assertEqual([item.rank for item in scored], [1, 2, 3, 4])
        self.assertTrue(all(item.compatible for item in scored))

    def test_incompatible_fixed_candidate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.backend.score_candidates("这个工具真的很", "shi yong", ["北京"])


if __name__ == "__main__":
    unittest.main()
