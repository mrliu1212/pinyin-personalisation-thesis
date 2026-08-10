import hashlib
import json
import unittest
from datetime import datetime
from pathlib import Path

from src.context_similarity import (
    CharacterTfidf,
    ContextualMemory,
    MemoryInteraction,
    contextual_candidate_evidence,
)
from src.phase_04d_evaluation import (
    FULL,
    ContextualPersonalizer,
    evaluate_phase_04d,
    interpolate_personal_evidence,
    normalize_frequency_counts,
)


ROOT = Path(__file__).resolve().parents[1]
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


def memory_item(index, context, pinyin="ceshi", candidate="测试", user="zhu_ziqing"):
    return MemoryInteraction(
        interaction_id=f"memory-{index}",
        user_id=user,
        timestamp=datetime(1920, 1, index + 1),
        context=context,
        pinyin=pinyin,
        selected_candidate=candidate,
        work_id="history",
    )


def history_record(index, candidate="测试", user="zhu_ziqing", context="共同语境"):
    return {
        "author_id": user,
        "interaction_id": f"history-{user}-{index}",
        "work_id": "history",
        "work_date": "1920-01-01",
        "derived_context": context,
        "pinyin": "ceshi",
        "target_candidate": candidate,
    }


def query_record(context="完全无关", candidates=("候选", "测试")):
    return {
        "interaction_id": "query",
        "work_id": "test",
        "work_date": "1932-01-01",
        "derived_context": context,
        "pinyin": "ceshi",
        "target_candidate": "测试",
        "candidates": [
            {"text": candidate, "base_rank": index, "base_score": None}
            for index, candidate in enumerate(candidates, start=1)
        ],
    }


def benchmark_record(work_id, index, user, target):
    return {
        "author_id": user,
        "interaction_id": f"{user}-{index}",
        "work_id": work_id,
        "work_date": WORK_DATES[work_id],
        "source_start_offset": index,
        "derived_context": "共同语境",
        "pinyin": "ceshi",
        "target_candidate": target,
        "candidates": [
            {"text": "候选", "base_rank": 1, "base_score": None},
            {"text": "测试", "base_rank": 2, "base_score": None},
            {"text": "试验", "base_rank": 3, "base_score": None},
        ],
    }


class ContextSimilarityTests(unittest.TestCase):
    def test_tfidf_is_deterministic_and_identical_context_is_maximal(self):
        first = CharacterTfidf().fit(["我们可以使用", "这个软件实用"])
        second = CharacterTfidf().fit(["我们可以使用", "这个软件实用"])
        vector = first.transform("我们可以使用")
        self.assertEqual(first.idf, second.idf)
        self.assertEqual(vector, second.transform("我们可以使用"))
        self.assertAlmostEqual(first.cosine(vector, vector), 1.0)

    def test_unrelated_context_has_lower_similarity(self):
        model = CharacterTfidf().fit(["我们可以使用", "春风吹过田野"])
        query = model.transform("我们可以使用")
        related = model.cosine(query, model.transform("我们可以采用"))
        unrelated = model.cosine(query, model.transform("春风吹过田野"))
        self.assertGreater(related, unrelated)

    def test_retrieval_is_same_pinyin_positive_only_and_top5(self):
        interactions = [
            memory_item(index, f"共同语境{chr(0x4E00 + index)}") for index in range(6)
        ]
        interactions.append(memory_item(8, "共同语境", pinyin="qita"))
        memory = ContextualMemory(interactions, user_id="zhu_ziqing")
        results = memory.retrieve("共同语境", "ceshi")
        self.assertEqual(len(results), 5)
        self.assertTrue(all(item.interaction.pinyin == "ceshi" for item in results))
        self.assertTrue(all(item.similarity > 0.0 for item in results))
        self.assertEqual(memory.retrieve("毫无重合", "ceshi"), ())

    def test_memory_rejects_mixed_users(self):
        with self.assertRaisesRegex(ValueError, "cannot mix users"):
            ContextualMemory(
                [memory_item(0, "语境"), memory_item(1, "语境", user="lu_xun")],
                user_id="zhu_ziqing",
            )

    def test_context_evidence_is_similarity_normalized(self):
        memory = ContextualMemory(
            [
                memory_item(0, "相同语境", candidate="甲"),
                memory_item(1, "相同语境", candidate="乙"),
            ],
            user_id="zhu_ziqing",
        )
        retrieved = memory.retrieve("相同语境", "ceshi")
        evidence = contextual_candidate_evidence(retrieved, ["甲", "乙", "丙"])
        self.assertAlmostEqual(evidence["甲"], 0.5)
        self.assertAlmostEqual(evidence["乙"], 0.5)
        self.assertEqual(evidence["丙"], 0.0)
        self.assertAlmostEqual(sum(evidence.values()), 1.0)


class ContextualScoringTests(unittest.TestCase):
    def test_frequency_normalization_uses_candidate_list_maximum(self):
        normalized = normalize_frequency_counts(["甲", "乙", "丙"], {"甲": 4, "乙": 2})
        self.assertEqual(normalized, {"甲": 1.0, "乙": 0.5, "丙": 0.0})
        self.assertEqual(
            normalize_frequency_counts(["甲", "乙"], {}),
            {"甲": 0.0, "乙": 0.0},
        )

    def test_confidence_interpolation_and_no_gate(self):
        self.assertAlmostEqual(
            interpolate_personal_evidence(0.8, 0.2, 0.25, condition=FULL),
            0.65,
        )
        self.assertEqual(
            interpolate_personal_evidence(
                0.8, 0.2, 0.25, condition="phase_04d_no_gate"
            ),
            0.2,
        )

    def test_correct_and_wrong_user_memories_remain_isolated(self):
        before = datetime(1930, 1, 1)
        correct = ContextualPersonalizer(
            [history_record(0, candidate="测试", user="zhu_ziqing")],
            user_id="zhu_ziqing",
            before=before,
        )
        wrong = ContextualPersonalizer(
            [history_record(0, candidate="候选", user="lu_xun")],
            user_id="lu_xun",
            before=before,
        )
        correct_trace = correct.score(query_record(), condition=FULL)
        wrong_trace = wrong.score(query_record(), condition=FULL)
        correct_by_candidate = {item["candidate"]: item for item in correct_trace["candidates"]}
        wrong_by_candidate = {item["candidate"]: item for item in wrong_trace["candidates"]}
        self.assertEqual(correct_by_candidate["测试"]["pinyin_count"], 1)
        self.assertEqual(correct_by_candidate["候选"]["pinyin_count"], 0)
        self.assertEqual(wrong_by_candidate["候选"]["pinyin_count"], 1)
        self.assertEqual(wrong_by_candidate["测试"]["pinyin_count"], 0)

    def test_author_mismatch_and_future_history_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            ContextualPersonalizer(
                [history_record(0, user="lu_xun")],
                user_id="zhu_ziqing",
                before=datetime(1930, 1, 1),
            )
        future = history_record(0)
        future["work_date"] = "1935-01-01"
        with self.assertRaisesRegex(ValueError, "future interaction"):
            ContextualPersonalizer(
                [future], user_id="zhu_ziqing", before=datetime(1930, 1, 1)
            )

    def test_final_score_ties_preserve_base_order(self):
        personalizer = ContextualPersonalizer(
            [history_record(0, candidate="测试", context="历史语境")],
            user_id="zhu_ziqing",
            before=datetime(1930, 1, 1),
        )
        trace = personalizer.score(query_record(context="完全无关"), condition=FULL)
        scores = {item["candidate"]: item["final_score"] for item in trace["candidates"]}
        self.assertEqual(scores["候选"], scores["测试"])
        self.assertEqual(trace["ranking"], ["候选", "测试"])

    def test_transparency_trace_reconstructs_full_formula(self):
        personalizer = ContextualPersonalizer(
            [history_record(0, candidate="测试", context="共同语境")],
            user_id="zhu_ziqing",
            before=datetime(1930, 1, 1),
        )
        trace = personalizer.score(query_record(context="共同语境"), condition=FULL)
        target = next(item for item in trace["candidates"] if item["candidate"] == "测试")
        expected_frequency = (
            0.25 * target["normalized_global_evidence"]
            + 0.75 * target["normalized_pinyin_evidence"]
        )
        expected_personal = (
            (1.0 - target["context_confidence"]) * expected_frequency
            + target["context_confidence"] * target["context_evidence"]
        )
        expected_final = (
            0.5 * target["normalized_base_utility"] + 0.5 * expected_personal
        )
        self.assertAlmostEqual(target["frequency_evidence"], expected_frequency)
        self.assertAlmostEqual(target["personal_evidence"], expected_personal)
        self.assertAlmostEqual(target["final_score"], expected_final)
        self.assertEqual(len(trace["retrieved_contexts"]), 1)


class FrozenArtifactTests(unittest.TestCase):
    def test_phase_04c_and_phase_04b6_artifacts_remain_unchanged(self):
        paths = {
            "results/experiments/phase_04c/evaluation.json": (
                "c9a03ae4cdc18bba0facff7bcdd4ec9a0221906859cd001781719d8d646456ff"
            ),
            "data/processed/interactions/zhu_ziqing_simplified_rime/interactions.jsonl": (
                "2d0df837fed3cf6b1a141b9f43677733671cf1f08cb72ca3b9e2f0f2f13f5077"
            ),
        }
        for relative, expected in paths.items():
            with self.subTest(path=relative):
                actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)

    def test_five_condition_evaluation_is_deterministic(self):
        zhu_works = (
            "congcong",
            "qinhuai_river",
            "beiying",
            "ahe",
            "moonlight_over_lotus_pond",
            "to_my_late_wife",
            "spring",
        )
        lu_works = (
            "madmans_diary",
            "kong_yiji",
            "medicine",
            "hometown",
            "new_years_sacrifice",
            "takeism",
            "have_chinese_lost_self_confidence",
        )
        zhu = [
            benchmark_record(work, index, "zhu_ziqing", "测试")
            for index, work in enumerate(zhu_works)
        ]
        lu = [
            benchmark_record(work, index, "lu_xun", "候选")
            for index, work in enumerate(lu_works)
        ]
        first = evaluate_phase_04d(zhu, lu)
        second = evaluate_phase_04d(list(reversed(zhu)), list(reversed(lu)))
        self.assertEqual(
            json.dumps(first, ensure_ascii=False, sort_keys=True),
            json.dumps(second, ensure_ascii=False, sort_keys=True),
        )
        self.assertEqual(
            set(first["subsets"]["full_benchmark"]),
            {
                "base",
                "phase_04c_frequency_personalisation",
                "phase_04d_no_gate_correct_user",
                "phase_04d_full_correct_user",
                "phase_04d_full_wrong_user",
            },
        )


if __name__ == "__main__":
    unittest.main()
