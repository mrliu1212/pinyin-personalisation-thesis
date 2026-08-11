import tempfile
import unittest
import hashlib
from pathlib import Path

from src.learned_reranker import GENERIC_CONTEXT_FEATURES, HYBRID_PERSONAL_FEATURES
from src.phase_04c_evaluation import compute_metrics, count_rank_changes
from src.phase_04e_evaluation import (
    Phase04EFeatureExtractor,
    _memory_deletion_counterfactuals,
    build_training_sets,
    evaluate_phase_04e,
)
from src.learned_reranker import CandidateTrainingSet, PairwiseLinearReranker
from src.semantic_lm import CandidateSemanticScore, min_max_normalize
from src.semantic_memory import CachedEmbeddingModel


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


class FakeLM:
    def score_candidates(self, context, candidates):
        conditional = [float((sum(map(ord, candidate)) + len(context)) % 17) for candidate in candidates]
        prior = [float(sum(map(ord, candidate)) % 11) for candidate in candidates]
        gains = [a - b for a, b in zip(conditional, prior)]
        normalized_conditional = min_max_normalize(conditional)
        normalized_gain = min_max_normalize(gains)
        return tuple(
            CandidateSemanticScore(
                candidate=candidate,
                candidate_token_count=len(candidate),
                lm_conditional_logprob=cond,
                lm_prior_logprob=base,
                lm_context_gain=gain,
                normalized_lm_conditional=norm_cond,
                normalized_lm_context_gain=norm_gain,
            )
            for candidate, cond, base, gain, norm_cond, norm_gain in zip(
                candidates,
                conditional,
                prior,
                gains,
                normalized_conditional,
                normalized_gain,
            )
        )


class FakeEmbeddingBackend:
    def vector(self, text):
        return [float(text.count("甲") + 1), float(text.count("乙") + 1), 1.0]

    def encode_query(self, context):
        return self.vector(context)

    def encode_document(self, context):
        return self.vector(context)


def record(work_id, index, user, target, candidates):
    return {
        "author_id": user,
        "interaction_id": f"{user}-{index}",
        "work_id": work_id,
        "work_date": WORK_DATES[work_id],
        "source_start_offset": index,
        "derived_context": "十二字语境",
        "raw_context": "前文" * 40 + ("甲" if user == "zhu_ziqing" else "乙"),
        "pinyin": "ceshi",
        "target_candidate": target,
        "candidates": [
            {"text": candidate, "base_rank": rank, "base_score": None}
            for rank, candidate in enumerate(candidates, start=1)
        ],
    }


def datasets():
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
    zhu = []
    for index, work in enumerate(zhu_works):
        target = "丁" if index in (0, 5) else "甲"
        candidates = ("甲", "乙", "丁") if index < 5 else ("甲", "乙", "丙")
        zhu.append(record(work, index, "zhu_ziqing", target, candidates))
    lu = [
        record(work, index, "lu_xun", "乙", ("甲", "乙", "丙"))
        for index, work in enumerate(lu_works)
    ]
    return zhu, lu


def phase_04d_fixture(zhu):
    tests = zhu[-2:]
    base_ranks = [None, 1]
    no_gate_ranks = [None, 1]
    base_metrics = compute_metrics(base_ranks).__dict__
    no_gate_metrics = compute_metrics(no_gate_ranks).__dict__
    full_base = {"metrics": base_metrics}
    full_no_gate = {
        "metrics": no_gate_metrics,
        "rank_changes": count_rank_changes(base_ranks, no_gate_ranks).__dict__,
        "diagnostics": {},
    }
    rerank_base = {"metrics": compute_metrics([1]).__dict__}
    rerank_no_gate = {
        "metrics": compute_metrics([1]).__dict__,
        "rank_changes": count_rank_changes([1], [1]).__dict__,
        "diagnostics": {},
    }
    return {
        "splits": {},
        "subsets": {
            "full_benchmark": {
                "base": full_base,
                "phase_04d_no_gate_correct_user": full_no_gate,
            },
            "rerankable": {
                "base": rerank_base,
                "phase_04d_no_gate_correct_user": rerank_no_gate,
            },
        },
        "evaluation_rows": {
            "phase_04d_no_gate_correct_user": [
                {
                    "interaction_id": test["interaction_id"],
                    "work_id": test["work_id"],
                    "base_rank": base,
                    "personalised_rank": rank,
                }
                for test, base, rank in zip(tests, base_ranks, no_gate_ranks)
            ]
        },
    }


class SpyExtractor:
    def __init__(self):
        self.personal_histories = []

    def extract(self, record, history, *, user_id, augmented, include_personal):
        if include_personal:
            self.personal_histories.append(
                (user_id, record["interaction_id"], tuple(item["interaction_id"] for item in history))
            )
        names = HYBRID_PERSONAL_FEATURES if include_personal else GENERIC_CONTEXT_FEATURES
        row = {name: 0.0 for name in names}
        return {
            "history_size": len(history) if include_personal else 0,
            "candidates": [
                {"candidate": candidate, "ranking_features": row}
                for candidate in (record["target_candidate"], "其他")
            ],
        }


class Phase04EEvaluationTests(unittest.TestCase):
    def test_memory_deletion_is_deterministic_and_does_not_retrain(self):
        def feature_row(memory_share, memory_max, support_count, any_support):
            row = {name: 0.0 for name in HYBRID_PERSONAL_FEATURES}
            row.update(
                normalized_base_utility=1.0 - memory_share,
                candidate_char_length=1.0,
                memory_weighted_share=memory_share,
                memory_max_similarity=memory_max,
                memory_support_count=support_count,
                memory_any_support=any_support,
            )
            return row

        training = [
            CandidateTrainingSet(
                target="甲",
                candidate_features={
                    "甲": feature_row(1.0, 0.9, 1.0, 1.0),
                    "乙": feature_row(0.0, 0.0, 0.0, 0.0),
                },
            ),
            CandidateTrainingSet(
                target="乙",
                candidate_features={
                    "甲": feature_row(0.0, 0.0, 0.0, 0.0),
                    "乙": feature_row(1.0, 0.8, 1.0, 1.0),
                },
            ),
        ]
        reranker = PairwiseLinearReranker(HYBRID_PERSONAL_FEATURES).fit(training)
        retrieved = [
            {
                "interaction_id": "memory-1",
                "historical_selected_candidate": "甲",
                "nonnegative_weight": 0.9,
            },
            {
                "interaction_id": "memory-2",
                "historical_selected_candidate": "乙",
                "nonnegative_weight": 0.1,
            },
        ]
        candidates = []
        for index, (candidate, values) in enumerate(
            (
                ("甲", feature_row(0.9, 0.9, 1.0, 1.0)),
                ("乙", feature_row(0.1, 0.1, 1.0, 1.0)),
            )
        ):
            candidates.append(
                {
                    "candidate": candidate,
                    "pool_index": index,
                    "ranking_features": values,
                    **reranker.score(values),
                }
            )
        row = {
            "target": "甲",
            "personalised_rank": 1,
            "retrieved_memory": retrieved,
            "candidates": candidates,
        }
        coefficients_before = reranker.coefficients.copy()
        first = _memory_deletion_counterfactuals(row, reranker)
        second = _memory_deletion_counterfactuals(row, reranker)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(reranker.coefficients, coefficients_before)

    def test_frozen_artifacts_and_model_manifest(self):
        root = Path(__file__).resolve().parents[1]
        expected = {
            "results/experiments/phase_04c/evaluation.json": (
                "c9a03ae4cdc18bba0facff7bcdd4ec9a0221906859cd001781719d8d646456ff"
            ),
            "results/experiments/phase_04d/evaluation.json": (
                "17c3ef37a416afba87b01de3741cd0c2131b50ad59ef737bdd136c10316d9620"
            ),
            "data/processed/interactions/zhu_ziqing_simplified_rime/interactions.jsonl": (
                "2d0df837fed3cf6b1a141b9f43677733671cf1f08cb72ca3b9e2f0f2f13f5077"
            ),
        }
        for relative, checksum in expected.items():
            with self.subTest(path=relative):
                self.assertEqual(
                    hashlib.sha256((root / relative).read_bytes()).hexdigest(), checksum
                )
        manifest = (root / "results/experiments/phase_04e/model_manifest.json").read_text(
            encoding="utf-8"
        )
        self.assertIn("da87bfb608c14b7cf20ba1ce41287e8de496c0cd", manifest)
        self.assertIn("97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", manifest)
        self.assertFalse((root / "results/experiments/phase_04e/evaluation.json").exists())

    def test_training_personal_features_use_strict_author_prefixes(self):
        zhu, lu = datasets()
        spy = SpyExtractor()
        build_training_sets(
            {"zhu_ziqing": zhu[:5], "lu_xun": lu[:5]}, spy
        )
        by_user = {"zhu_ziqing": [], "lu_xun": []}
        for user, _, history in spy.personal_histories:
            by_user[user].append(history)
            self.assertTrue(all(item.startswith(user) for item in history))
        self.assertEqual([len(item) for item in by_user["zhu_ziqing"]], [0, 1, 2, 3, 4])
        self.assertEqual([len(item) for item in by_user["lu_xun"]], [0, 1, 2, 3, 4])

    def test_fixed_augmented_wrong_user_and_imported_phase4d(self):
        zhu, lu = datasets()
        with tempfile.TemporaryDirectory() as directory:
            embedding = CachedEmbeddingModel(
                FakeEmbeddingBackend(), revision="r", cache_dir=Path(directory)
            )
            result = evaluate_phase_04e(
                zhu,
                lu,
                phase_04d_fixture(zhu),
                Phase04EFeatureExtractor(FakeLM(), embedding),
            )
        full = result["subsets"]["full_benchmark"]
        self.assertEqual(
            set(full),
            {
                "base",
                "phase_04d_no_gate_correct_user",
                "phase_04e_generic_context",
                "phase_04e_hybrid_fixed_correct_user",
                "phase_04e_hybrid_fixed_wrong_user",
                "phase_04e_hybrid_augmented_correct_user",
                "phase_04e_hybrid_augmented_wrong_user",
            },
        )
        self.assertEqual(set(result["evaluation_rows"]), set(full))
        self.assertEqual(full["base"]["metrics"]["top1_accuracy"], 0.5)
        self.assertEqual(
            full["phase_04d_no_gate_correct_user"]["metrics"]["top1_accuracy"],
            0.5,
        )
        augmented = result["evaluation_rows"][
            "phase_04e_hybrid_augmented_correct_user"
        ][0]
        fixed = result["evaluation_rows"]["phase_04e_hybrid_fixed_correct_user"][0]
        wrong = result["evaluation_rows"][
            "phase_04e_hybrid_augmented_wrong_user"
        ][0]
        correct = result["evaluation_rows"][
            "phase_04e_hybrid_augmented_correct_user"
        ]
        self.assertIsNone(fixed["personalised_rank"])
        self.assertIsNotNone(augmented["personalised_rank"])
        self.assertIsNone(wrong["personalised_rank"])
        self.assertEqual(augmented["history_size"], 5)
        self.assertEqual(wrong["personal_user_id"], "lu_xun")
        self.assertTrue(
            all(
                memory["interaction_id"].startswith("zhu_ziqing")
                for row in correct
                for memory in row["retrieved_memory"]
            )
        )
        self.assertTrue(
            all(
                memory["interaction_id"].startswith("lu_xun")
                for memory in wrong["retrieved_memory"]
            )
        )
        self.assertEqual([row["history_size"] for row in correct], [5, 5])
        test_ids = {row["interaction_id"] for row in correct}
        self.assertTrue(
            test_ids.isdisjoint(
                memory["interaction_id"]
                for row in correct
                for memory in row["retrieved_memory"]
            )
        )


if __name__ == "__main__":
    unittest.main()
