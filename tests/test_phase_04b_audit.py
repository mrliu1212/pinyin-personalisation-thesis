import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from audits.phase_04b import missing_diagnostics, run_audit, seeded_sample


def interaction(index: int, *, flagged: bool, present: bool) -> dict:
    target = "缺詞" if not present and index % 2 == 0 else f"詞{index}"
    return {
        "interaction_id": f"interaction-{index}",
        "work_id": "work_a" if index % 2 == 0 else "work_b",
        "work_title": "作品",
        "work_date": "1930",
        "source_processed_file": "work.txt",
        "source_page_url": "https://example.invalid/work",
        "source_revision_id": 1,
        "source_start_offset": index,
        "source_end_offset": index + 2,
        "raw_context": "完整上文",
        "derived_context": "上文",
        "target_candidate": target,
        "target_length": 2,
        "pinyin": "queci",
        "pinyin_syllables": ["que", "ci"],
        "polyphonic_review_required": flagged,
        "polyphonic_characters": [],
        "candidates": [{"text": "卻辭", "base_rank": 1, "base_score": None}],
        "candidate_list_size": 1,
        "target_rank": 1 if present else None,
        "target_present": present,
    }


class Phase04BAuditTest(unittest.TestCase):
    def test_seeded_sampling_is_stable_across_input_order(self):
        records = [interaction(i, flagged=True, present=True) for i in range(10)]
        first = seeded_sample(records, seed=40402, category="flagged", sample_size=4)
        second = seeded_sample(
            reversed(records), seed=40402, category="flagged", sample_size=4
        )
        self.assertEqual(
            [item["interaction_id"] for item in first],
            [item["interaction_id"] for item in second],
        )

    def test_missing_diagnostics_are_factual_and_include_repetitions(self):
        missing = [
            interaction(0, flagged=True, present=False),
            interaction(2, flagged=False, present=False),
            interaction(3, flagged=False, present=False),
        ]
        diagnostics = missing_diagnostics(missing, total_interactions=5)
        self.assertEqual(diagnostics["top_10_missing_count"], 3)
        self.assertEqual(diagnostics["target_length_distribution"], {"2": 3})
        self.assertEqual(diagnostics["polyphonic_review"]["flagged"], 1)
        repeated = diagnostics["repeated_missing_targets"]
        self.assertEqual(repeated[0]["target_candidate"], "缺詞")
        self.assertEqual(repeated[0]["count"], 2)

    def test_audit_preserves_input_and_writes_three_review_samples(self):
        records = []
        for index in range(12):
            records.append(
                interaction(
                    index,
                    flagged=index < 6,
                    present=index not in {0, 1, 2, 6, 7, 8},
                )
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "interactions.jsonl"
            input_path.write_text(
                "".join(
                    json.dumps(item, ensure_ascii=False) + "\n" for item in records
                ),
                encoding="utf-8",
            )
            before = hashlib.sha256(input_path.read_bytes()).hexdigest()
            output_dir = root / "audit"
            manifest = run_audit(input_path, output_dir, seed=7, sample_size=2)
            after = hashlib.sha256(input_path.read_bytes()).hexdigest()

            self.assertEqual(before, after)
            self.assertFalse(manifest["automatic_error_classification"])
            for filename in manifest["output_files"].values():
                lines = (output_dir / filename).read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(lines), 2)
                record = json.loads(lines[0])
                self.assertIn("raw_context", record)
                self.assertIn("ordered_base_candidates", record)
                self.assertIn("target_present", record)


if __name__ == "__main__":
    unittest.main()
