import csv
import json
import tempfile
import unittest
from pathlib import Path

from audits.phase_04b_manual_review import (
    FIELDNAMES,
    prepare_review_csv,
    summarize_review,
)


def audit_record(sample_type: str, index: int) -> dict:
    return {
        "audit_sample_category": sample_type,
        "interaction_id": f"id-{sample_type}-{index}",
        "work_title": "作品",
        "source_start_offset": index,
        "source_end_offset": index + 2,
        "raw_context": "第一行\n第二行",
        "derived_context": "第二行",
        "target_candidate": "使用",
        "pinyin": "shiyong",
        "pinyin_syllables": ["shi", "yong"],
        "polyphonic_review_required": sample_type == "polyphonic_flagged",
        "ordered_base_candidates": [
            {"text": "實用", "base_rank": 1, "base_score": None},
            {"text": "使用", "base_rank": 2, "base_score": None},
        ],
        "target_present": sample_type != "top10_missing",
        "target_rank": None if sample_type == "top10_missing" else 2,
    }


class ManualReviewTest(unittest.TestCase):
    def create_audit_files(self, root: Path, count: int = 2) -> None:
        for sample_type in (
            "polyphonic_flagged",
            "polyphonic_unflagged",
            "top10_missing",
        ):
            path = root / f"{sample_type}_sample.jsonl"
            path.write_text(
                "".join(
                    json.dumps(audit_record(sample_type, index), ensure_ascii=False)
                    + "\n"
                    for index in range(count)
                ),
                encoding="utf-8",
            )

    def test_prepare_uses_existing_samples_and_leaves_human_labels_empty(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_audit_files(root)
            output = root / "manual.csv"
            count = prepare_review_csv(root, output)
            with output.open(encoding="utf-8-sig", newline="") as source:
                rows = list(csv.DictReader(source))

            self.assertEqual(count, 6)
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[0]["raw_context"], "第一行\n第二行")
            self.assertEqual(rows[0]["base_candidates"], "1:實用 | 2:使用")
            for row in rows:
                self.assertEqual(row["pinyin_judgement"], "")
                self.assertEqual(row["segmentation_judgement"], "")
                self.assertEqual(row["missing_cause"], "")
                self.assertEqual(row["notes"], "")

    def test_summary_uses_only_manual_labels_and_reports_blanks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_audit_files(root)
            output = root / "manual.csv"
            prepare_review_csv(root, output)
            with output.open(encoding="utf-8-sig", newline="") as source:
                rows = list(csv.DictReader(source))
            rows[0]["pinyin_judgement"] = "correct"
            rows[2]["pinyin_judgement"] = "incorrect"
            rows[4]["segmentation_judgement"] = "reasonable"
            rows[4]["missing_cause"] = "traditional_or_variant_form"
            with output.open("w", encoding="utf-8-sig", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)

            summary = summarize_review(output)
            flagged = summary["polyphonic_flagged_pinyin"]
            self.assertEqual(flagged["values"]["correct"]["count"], 1)
            self.assertEqual(flagged["unlabelled_rows"], 1)
            unflagged = summary["polyphonic_unflagged_pinyin"]
            self.assertEqual(unflagged["values"]["incorrect"]["count"], 1)
            cause = summary["top10_missing_cause"]
            self.assertEqual(
                cause["values"]["traditional_or_variant_form"]["count"], 1
            )
            self.assertEqual(cause["unlabelled_rows"], 1)

    def test_summary_rejects_an_undocumented_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_audit_files(root, count=1)
            output = root / "manual.csv"
            prepare_review_csv(root, output)
            with output.open(encoding="utf-8-sig", newline="") as source:
                rows = list(csv.DictReader(source))
            rows[0]["pinyin_judgement"] = "probably_correct"
            with output.open("w", encoding="utf-8-sig", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(ValueError, "invalid pinyin_judgement"):
                summarize_review(output)


if __name__ == "__main__":
    unittest.main()
