import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from audits.phase_04b7_manual_review import (
    MISSING_FIELDS,
    POLYPHONIC_FIELDS,
    deterministic_sample,
    prepare,
    summarize,
)


ROOT = Path(__file__).resolve().parents[1]


def interaction(index: int, *, flagged: bool, present: bool) -> dict:
    return {
        "interaction_id": f"interaction-{index}",
        "work_id": "work",
        "work_title": "作品",
        "source_start_offset": index,
        "source_end_offset": index + 2,
        "source_original_target": "時候",
        "normalization_provenance": {
            "source_original_processed_file": "work.txt"
        },
        "derived_context": "我们等待",
        "target_candidate": "时候",
        "pinyin": "shihou",
        "pinyin_syllables": ["shi", "hou"],
        "polyphonic_review_required": flagged,
        "candidates": [
            {"text": "时候", "base_rank": 1, "base_score": None},
            {"text": "事后", "base_rank": 2, "base_score": None},
        ],
        "target_rank": 1 if present else None,
        "target_present": present,
    }


class Phase04B7ManualReviewTest(unittest.TestCase):
    def test_sampling_is_deterministic_and_input_order_independent(self):
        records = [interaction(i, flagged=True, present=True) for i in range(10)]
        first = deterministic_sample(
            records, seed=40407, sample_name="polyphonic", sample_size=4
        )
        second = deterministic_sample(
            reversed(records), seed=40407, sample_name="polyphonic", sample_size=4
        )
        self.assertEqual(
            [item["interaction_id"] for item in first],
            [item["interaction_id"] for item in second],
        )

    def test_prepare_writes_blank_csvs_without_changing_source(self):
        records = [
            interaction(i, flagged=i < 5, present=i not in {0, 1, 5, 6, 7})
            for i in range(10)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "interactions.jsonl"
            input_path.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
                encoding="utf-8",
            )
            checksum = hashlib.sha256(input_path.read_bytes()).hexdigest()
            comparison_path = root / "comparison.json"
            comparison_path.write_text(
                json.dumps({"output_interactions_sha256": checksum}), encoding="utf-8"
            )
            output_dir = root / "audit"
            manifest = prepare(
                input_path,
                comparison_path,
                output_dir,
                seed=7,
                sample_size=2,
            )

            self.assertEqual(hashlib.sha256(input_path.read_bytes()).hexdigest(), checksum)
            self.assertFalse(manifest["automatic_labels_assigned"])
            with (output_dir / "polyphonic_review_sample.csv").open(
                encoding="utf-8-sig", newline=""
            ) as source:
                polyphonic = list(csv.DictReader(source))
            with (output_dir / "missing_review_sample.csv").open(
                encoding="utf-8-sig", newline=""
            ) as source:
                missing = list(csv.DictReader(source))
            self.assertEqual(len(polyphonic), 2)
            self.assertEqual(len(missing), 2)
            self.assertTrue(all(row["pinyin_judgement"] == "" for row in polyphonic))
            self.assertTrue(all(row["missing_cause"] == "" for row in missing))
            self.assertEqual(polyphonic[0]["source_original_target"], "時候")

    def test_summary_aggregates_only_manual_labels_and_reports_blanks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            polyphonic_path = root / "polyphonic_review_sample.csv"
            missing_path = root / "missing_review_sample.csv"
            polyphonic_rows = [
                {field: "" for field in POLYPHONIC_FIELDS} for _ in range(3)
            ]
            polyphonic_rows[0]["pinyin_judgement"] = "correct"
            polyphonic_rows[1]["pinyin_judgement"] = "incorrect"
            missing_rows = [{field: "" for field in MISSING_FIELDS} for _ in range(3)]
            missing_rows[0]["missing_cause"] = "proper_name"
            missing_rows[1]["missing_cause"] = "candidate_coverage_problem"
            for path, fields, rows in (
                (polyphonic_path, POLYPHONIC_FIELDS, polyphonic_rows),
                (missing_path, MISSING_FIELDS, missing_rows),
            ):
                with path.open("w", encoding="utf-8-sig", newline="") as output:
                    writer = csv.DictWriter(output, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)

            result = summarize(root)
            pinyin = result["polyphonic_pinyin_judgement"]
            self.assertEqual(pinyin["labelled_rows"], 2)
            self.assertEqual(pinyin["blank_rows"], 1)
            self.assertEqual(pinyin["values"]["correct"]["percentage_of_labelled"], 0.5)
            causes = result["missing_cause"]
            self.assertEqual(causes["labelled_rows"], 2)
            self.assertEqual(causes["values"]["proper_name"]["count"], 1)

    def test_summary_rejects_undocumented_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            polyphonic = {field: "" for field in POLYPHONIC_FIELDS}
            polyphonic["pinyin_judgement"] = "probably"
            missing = {field: "" for field in MISSING_FIELDS}
            for path, fields, row in (
                (root / "polyphonic_review_sample.csv", POLYPHONIC_FIELDS, polyphonic),
                (root / "missing_review_sample.csv", MISSING_FIELDS, missing),
            ):
                with path.open("w", encoding="utf-8-sig", newline="") as output:
                    writer = csv.DictWriter(output, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "invalid pinyin_judgement"):
                summarize(root)

    def test_phase_04b6_interactions_match_recorded_checksum(self):
        interaction_path = ROOT / (
            "data/processed/interactions/zhu_ziqing_simplified_rime/"
            "interactions.jsonl"
        )
        comparison = json.loads(
            (
                ROOT
                / "data/processed/interactions/zhu_ziqing_simplified_rime/"
                "phase_04b6_comparison.json"
            ).read_text(encoding="utf-8")
        )
        actual = hashlib.sha256(interaction_path.read_bytes()).hexdigest()
        self.assertEqual(actual, comparison["output_interactions_sha256"])


if __name__ == "__main__":
    unittest.main()
