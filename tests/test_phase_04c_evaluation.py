import copy
import json
import unittest

from src.phase_04c_evaluation import (
    LU_TEST_WORK_IDS,
    LU_TRAIN_WORK_IDS,
    ZHU_TEST_WORK_IDS,
    ZHU_TRAIN_WORK_IDS,
    build_personal_model,
    compute_metrics,
    evaluate_phase_04c,
    frozen_split,
    parse_work_date,
)


WORK_DATES = {
    "congcong": "1922-03-28",
    "qinhuai_river": "1924-01-25",
    "beiying": "1925-10",
    "ahe": "1926-01-11",
    "moonlight_over_lotus_pond": "1927-07",
    "to_my_late_wife": "1932-10-11",
    "spring": "1933-07",
    "madmans_diary": "1918-04",
    "kong_yiji": "1919-03",
    "medicine": "1919-04",
    "hometown": "1921-01",
    "new_years_sacrifice": "1924-02-07",
    "takeism": "1934-06-04",
    "have_chinese_lost_self_confidence": "1934-09-25",
}


def record(work_id, interaction_id, target, candidates=("乙", "甲", "丙")):
    return {
        "work_id": work_id,
        "work_date": WORK_DATES[work_id],
        "interaction_id": interaction_id,
        "source_start_offset": int(interaction_id.rsplit("-", 1)[-1]),
        "derived_context": "语境",
        "pinyin": "ceshi",
        "target_candidate": target,
        "candidates": [
            {"text": text, "base_rank": index, "base_score": None}
            for index, text in enumerate(candidates, start=1)
        ],
    }


def synthetic_datasets():
    zhu = [
        record(work_id, f"zhu-{index}", "甲")
        for index, work_id in enumerate(ZHU_TRAIN_WORK_IDS)
    ]
    zhu.extend(
        record(work_id, f"zhu-{index + 10}", "甲")
        for index, work_id in enumerate(ZHU_TEST_WORK_IDS)
    )
    lu = [
        record(work_id, f"lu-{index}", "乙")
        for index, work_id in enumerate(LU_TRAIN_WORK_IDS)
    ]
    lu.extend(
        record(work_id, f"lu-{index + 10}", "甲")
        for index, work_id in enumerate(LU_TEST_WORK_IDS)
    )
    return zhu, lu


class Phase04CEvaluationTests(unittest.TestCase):
    def test_extended_metrics_include_top5_and_top10(self):
        metrics = compute_metrics([1, 3, 5, 10, None])
        self.assertEqual(metrics.top1_accuracy, 1 / 5)
        self.assertEqual(metrics.top3_accuracy, 2 / 5)
        self.assertEqual(metrics.top5_accuracy, 3 / 5)
        self.assertEqual(metrics.top10_accuracy, 4 / 5)
        self.assertAlmostEqual(metrics.mrr, (1 + 1 / 3 + 1 / 5 + 1 / 10) / 5)
        self.assertEqual(metrics.mean_target_rank, 19 / 4)
        self.assertEqual(metrics.missing_target_count, 1)

    def test_chronological_split_is_frozen_and_strict(self):
        zhu, _ = synthetic_datasets()
        split = frozen_split(zhu, ZHU_TRAIN_WORK_IDS, ZHU_TEST_WORK_IDS)
        self.assertEqual({row["work_id"] for row in split.train}, set(ZHU_TRAIN_WORK_IDS))
        self.assertEqual({row["work_id"] for row in split.test}, set(ZHU_TEST_WORK_IDS))
        self.assertLess(
            max(parse_work_date(row["work_date"]) for row in split.train),
            min(parse_work_date(row["work_date"]) for row in split.test),
        )

    def test_correct_and_wrong_user_histories_are_isolated(self):
        zhu, lu = synthetic_datasets()
        cutoff = parse_work_date("1932-10-11")
        correct = build_personal_model(zhu[:5], "zhu_ziqing", before=cutoff)
        wrong = build_personal_model(lu[:5], "lu_xun", before=cutoff)
        self.assertGreater(correct.score("ceshi", "甲", "语境"), 0)
        self.assertEqual(correct.score("ceshi", "乙", "语境"), 0)
        self.assertGreater(wrong.score("ceshi", "乙", "语境"), 0)
        self.assertEqual(wrong.score("ceshi", "甲", "语境"), 0)

    def test_wrong_user_control_does_not_use_lu_test_partition(self):
        zhu, lu = synthetic_datasets()
        first = evaluate_phase_04c(zhu, lu)
        altered = copy.deepcopy(lu)
        for row in altered:
            if row["work_id"] in LU_TEST_WORK_IDS:
                row["target_candidate"] = "乙"
        second = evaluate_phase_04c(zhu, altered)
        self.assertEqual(first, second)
        self.assertFalse(
            first["splits"]["lu_xun"]["test_partition_used_for_wrong_user_history"]
        )

    def test_future_or_test_time_history_is_rejected(self):
        zhu, _ = synthetic_datasets()
        future = copy.deepcopy(zhu[:5])
        future[0]["work_date"] = "2035-01-01"
        with self.assertRaisesRegex(ValueError, "future interaction"):
            build_personal_model(
                future,
                "zhu_ziqing",
                before=parse_work_date("1932-10-11"),
            )

    def test_evaluation_is_deterministic_and_user_specific(self):
        zhu, lu = synthetic_datasets()
        first = evaluate_phase_04c(zhu, lu)
        second = evaluate_phase_04c(list(reversed(zhu)), list(reversed(lu)))
        self.assertEqual(
            json.dumps(first, ensure_ascii=False, sort_keys=True),
            json.dumps(second, ensure_ascii=False, sort_keys=True),
        )
        full = first["subsets"]["full_benchmark"]
        self.assertEqual(full["base"]["metrics"]["top1_accuracy"], 0.0)
        self.assertEqual(full["correct_user"]["metrics"]["top1_accuracy"], 1.0)
        self.assertEqual(full["wrong_user"]["metrics"]["top1_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
