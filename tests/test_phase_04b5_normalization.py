import hashlib
import json
import shutil
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from interactions.candidates import Candidate
from interactions.construction import construct_work_interactions
from interactions.linguistic import JiebaSegmenter, TargetPolicy
from normalization.phase_04b5 import (
    OpenCCCliNormalizer,
    augment_normalized_interactions,
    compare_interactions,
    prepare_normalized_corpus,
)


class FixedNormalizer:
    name = "test-normalizer"
    version = "1"
    config = "test-t2s"

    def convert(self, text: str) -> str:
        return text.translate(str.maketrans({"們": "们", "傳": "传", "統": "统"}))


class EchoCandidateGenerator:
    name = "fake"
    version = "1"
    schema_id = "fake"
    max_candidates = 10

    def candidates(self, pinyin_input: str) -> list[Candidate]:
        candidates = {
            "women": "我们",
            "xuyao": "需要",
            "shiyong": "使用",
            "fangfa": "方法",
        }
        return [Candidate(candidates.get(pinyin_input, "其他"), 1)]


def source_manifest(root: Path) -> Path:
    text = "我們需要使用方法"
    source_file = root / "work.txt"
    source_file.write_text(text, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "author_id": "author",
        "author_name": "作者",
        "source_api": "fixture",
        "source_name": "fixture",
        "works": [
            {
                "author_id": "author",
                "author_name": "作者",
                "work_id": "work",
                "work_title": "作品",
                "included": True,
                "chronology": {
                    "value": "1930",
                    "precision": "year",
                    "certainty": "certain",
                },
                "source_page_url": "https://example.invalid/work",
                "source_revision_id": 1,
                "processed_file": "work.txt",
                "processed_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "content_variant": "canonical",
            }
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return manifest_path


class Phase04B5NormalizationTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("opencc"), "OpenCC CLI is not installed")
    def test_opencc_t2s_conversion_is_deterministic(self):
        normalizer = OpenCCCliNormalizer()
        source = "傳統與後來"
        self.assertEqual(normalizer.convert(source), "传统与后来")
        self.assertEqual(normalizer.convert(source), normalizer.convert(source))

    def test_normalized_corpus_is_separate_and_raw_source_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = source_manifest(root)
            source_path = root / "work.txt"
            before = source_path.read_bytes()
            output_dir = root / "normalized"

            first = prepare_normalized_corpus(
                manifest_path, output_dir, FixedNormalizer()
            )
            first_bytes = (output_dir / "work.txt").read_bytes()
            second = prepare_normalized_corpus(
                manifest_path, output_dir, FixedNormalizer()
            )

            self.assertEqual(source_path.read_bytes(), before)
            self.assertEqual(first, second)
            self.assertEqual((output_dir / "work.txt").read_bytes(), first_bytes)
            self.assertEqual(first_bytes.decode("utf-8"), "我们需要使用方法")
            provenance = first["works"][0]["normalization_provenance"]
            self.assertEqual(provenance["source_processed_sha256"], hashlib.sha256(before).hexdigest())
            self.assertEqual(provenance["offset_mapping"], "identity; equal code-point length verified")

    def test_normalized_interactions_and_provenance_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = source_manifest(root)
            normalized_dir = root / "normalized"
            normalized_manifest = prepare_normalized_corpus(
                manifest_path, normalized_dir, FixedNormalizer()
            )
            normalized_manifest_path = normalized_dir / "manifest.json"
            text = (normalized_dir / "work.txt").read_text(encoding="utf-8")
            work = normalized_manifest["works"][0]

            outputs = []
            for index in (1, 2):
                result = construct_work_interactions(
                    text,
                    work,
                    JiebaSegmenter(),
                    EchoCandidateGenerator(),
                    TargetPolicy(),
                )
                output = root / f"interactions-{index}.jsonl"
                output.write_text(
                    "".join(
                        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                        for item in result.interactions
                    ),
                    encoding="utf-8",
                )
                augment_normalized_interactions(
                    output, manifest_path, normalized_manifest_path
                )
                outputs.append(output.read_bytes())

            self.assertEqual(outputs[0], outputs[1])
            record = json.loads(outputs[0].splitlines()[0])
            self.assertEqual(record["text_representation"], "opencc_t2s")
            self.assertEqual(record["source_original_target"], "我們")
            self.assertEqual(
                record["normalization_provenance"][
                    "source_original_processed_file"
                ],
                "work.txt",
            )

    def test_coverage_comparison_matches_interactions_by_source_span(self):
        baseline = [
            {
                "work_id": "work",
                "work_title": "作品",
                "source_start_offset": 0,
                "source_end_offset": 2,
                "target_candidate": "傳統",
                "pinyin": "chuantong",
                "candidates": [],
                "target_rank": None,
                "target_present": False,
            }
        ]
        normalized = [
            {
                "work_id": "work",
                "work_title": "作品",
                "source_start_offset": 0,
                "source_end_offset": 2,
                "target_candidate": "传统",
                "pinyin": "chuantong",
                "candidates": [asdict(Candidate("传统", 1))],
                "target_rank": 1,
                "target_present": True,
            }
        ]
        comparison = compare_interactions(baseline, normalized)
        spans = comparison["baseline_missing_span_analysis"]
        self.assertEqual(spans["recovered_count"], 1)
        self.assertEqual(spans["remaining_missing_count"], 0)
        self.assertEqual(comparison["recovered_examples"][0]["source_target"], "傳統")
        self.assertEqual(comparison["recovered_examples"][0]["normalized_target"], "传统")


if __name__ == "__main__":
    unittest.main()
