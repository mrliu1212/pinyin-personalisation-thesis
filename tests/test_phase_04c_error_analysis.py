import json
import tempfile
import unittest
from pathlib import Path

from audits.phase_04c_error_analysis import extract_samples, prepare


def candidate(text, rank, evidence):
    return {
        "candidate": text,
        "base_score": float(3 - rank),
        "global_evidence": evidence,
        "pinyin_evidence": evidence,
        "context_evidence": evidence,
        "personal_score": evidence,
        "final_score": 1.0 / rank,
        "final_rank": rank,
    }


def example(interaction_id, change, base_rank, personal_rank, evidence):
    return {
        "interaction_id": interaction_id,
        "work_id": "work",
        "context": "上下文",
        "pinyin": "ceshi",
        "target": "测试",
        "base_rank": base_rank,
        "personalised_rank": personal_rank,
        "change": change,
        "base_candidates": [
            {"candidate": "候选", "base_rank": 1, "base_score": 2.0},
            {"candidate": "测试", "base_rank": 2, "base_score": 1.0},
        ],
        "personalised_candidates": [
            candidate("测试", 1, evidence),
            candidate("候选", 2, 0.0),
        ],
    }


def fixture():
    return {
        "transparency_examples": {
            "correct_user": [
                example("b", "improved", 3, 1, 3.0),
                example("a", "improved", 2, 1, 2.0),
                example("c", "harmed", 1, 2, 1.0),
                example("shared", "unchanged", 1, 1, 4.0),
            ],
            "wrong_user": [
                example("d", "improved", 2, 1, 2.0),
                example("e", "harmed", 1, 3, 1.0),
                example("shared", "harmed", 1, 2, 0.5),
            ],
        }
    }


class Phase04CErrorAnalysisTests(unittest.TestCase):
    def test_extraction_is_deterministic_and_magnitude_ordered(self):
        first = extract_samples(fixture())
        second = extract_samples(fixture())
        self.assertEqual(first, second)
        correct_improved = [
            row for row in first if row["sample_type"] == "correct_user_improved"
        ]
        self.assertEqual([row["interaction_id"] for row in correct_improved], ["b", "a"])

    def test_prepare_does_not_modify_evaluation_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "evaluation.json"
            output_path = root / "samples.jsonl"
            input_path.write_text(
                json.dumps(fixture(), ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            before = input_path.read_bytes()
            prepare(input_path, output_path)
            self.assertEqual(input_path.read_bytes(), before)
            self.assertTrue(output_path.is_file())

    def test_output_schema_exposes_evidence_and_blank_review_fields(self):
        rows = extract_samples(fixture())
        required = {
            "sample_type",
            "condition",
            "interaction_id",
            "context",
            "pinyin",
            "target",
            "base_rank",
            "personalised_rank",
            "base_candidates",
            "personalised_candidates",
            "correct_user_candidates",
            "wrong_user_candidates",
            "human_category",
            "notes",
        }
        self.assertTrue(all(required <= set(row) for row in rows))
        evidence_fields = {
            "candidate",
            "base_score",
            "global_evidence",
            "pinyin_evidence",
            "context_evidence",
            "personal_score",
            "final_rank",
        }
        condition_row = next(row for row in rows if row["condition"] == "correct_user")
        self.assertTrue(evidence_fields <= set(condition_row["personalised_candidates"][0]))
        self.assertEqual(condition_row["human_category"], "")
        self.assertEqual(condition_row["notes"], "")
        comparison = next(row for row in rows if row["condition"] == "comparison")
        self.assertEqual(comparison["correct_user_rank"], 1)
        self.assertEqual(comparison["wrong_user_rank"], 2)


if __name__ == "__main__":
    unittest.main()
