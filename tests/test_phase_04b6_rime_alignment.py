import json
import tempfile
import unittest
from pathlib import Path

from interactions.candidates import RimeCliCandidateGenerator
from normalization.phase_04b5 import sha256_file
from normalization.phase_04b6 import (
    load_simplified_rime_config,
    write_augmented_candidate_provenance,
)


ROOT = Path(__file__).resolve().parents[1]


class Phase04B6RimeAlignmentTest(unittest.TestCase):
    def test_simplified_rime_configuration_loads_required_engine_filter(self):
        config = load_simplified_rime_config(
            ROOT / "config/rime/simplified_candidate_mode.json",
            ROOT / "data/rime/shared/luna_pinyin.schema.yaml",
        )
        self.assertEqual(config["enabled_schema_options"], ["zh_hans"])
        self.assertEqual(config["required_engine_filter"], "simplifier@zh_hans")
        self.assertEqual(config["required_opencc_config"], "t2s.json")
        self.assertEqual(
            config["candidate_conversion_location"], "inside librime engine filter"
        )

    @unittest.skipUnless(
        (ROOT / ".build/rime_candidate_cli").exists()
        and (ROOT / "data/rime/setup_manifest.json").exists(),
        "local librime adapter/data are not deployed",
    )
    def test_engine_outputs_simplified_candidates_deterministically(self):
        manifest = json.loads(
            (ROOT / "data/rime/setup_manifest.json").read_text(encoding="utf-8")
        )
        arguments = {
            "executable": ROOT / ".build/rime_candidate_cli",
            "shared_data": ROOT / manifest["shared_data_dir"],
            "prebuilt_data": ROOT / manifest["prebuilt_data_dir"],
            "version": manifest["librime"],
            "schema_id": "luna_pinyin",
            "max_candidates": 10,
            "enabled_options": ("zh_hans",),
        }
        outputs = []
        for _ in range(2):
            with RimeCliCandidateGenerator(**arguments) as generator:
                outputs.append(
                    [item.text for item in generator.candidates("weishenme")]
                )
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0][0], "为什么")
        self.assertNotIn("爲什麼", outputs[0])

    def test_phase_04a_processed_corpus_matches_recorded_checksums(self):
        manifest_path = ROOT / "data/processed/authors/zhu_ziqing/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for work in manifest["works"]:
            if not work["included"]:
                continue
            path = manifest_path.parent / work["processed_file"]
            self.assertEqual(sha256_file(path), work["processed_sha256"])

    def test_candidate_provenance_is_preserved_and_deterministic(self):
        config = json.loads(
            (ROOT / "config/rime/simplified_candidate_mode.json").read_text(
                encoding="utf-8"
            )
        )
        record = {
            "interaction_id": "id",
            "work_id": "work",
            "target_candidate": "为什么",
            "candidates": [
                {"text": "为什么", "base_rank": 1, "base_score": None}
            ],
            "source_original_target": "爲什麼",
            "normalization_provenance": {"configuration": "t2s.json"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            outputs = []
            for index in (1, 2):
                path = Path(temporary) / f"interactions-{index}.jsonl"
                path.write_text(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                write_augmented_candidate_provenance(path, config)
                outputs.append(path.read_bytes())
            self.assertEqual(outputs[0], outputs[1])
            augmented = json.loads(outputs[0])
            self.assertEqual(augmented["source_original_target"], "爲什麼")
            self.assertEqual(augmented["target_candidate"], "为什么")
            self.assertEqual(augmented["candidates"][0]["text"], "为什么")
            self.assertFalse(
                augmented["rime_script_provenance"][
                    "post_retrieval_candidate_conversion"
                ]
            )


if __name__ == "__main__":
    unittest.main()
