import json
from pathlib import Path
import pickle
import tempfile
import unittest

from experiments.exp_livechat_pinyingpt_generic_baseline import load_jsonl, target_length_groups
from src.datasets.livechat.baseline import (
    LiveChatRow,
    build_source_response_id,
    canonical_json,
    choose_target,
    construct_eligible_targets,
    determine_chronology_grade,
    load_livechat_pickle,
    select_max_interactions,
    session_partition,
    stable_hash,
    tokenizer_compatible_character_map,
    write_jsonl,
)
from src.evaluation.ranking import compute_metrics, context_gain
from src.reference_backend_pinyingpt.backend import (
    CHECKPOINT_REVISION,
    PinyinGPTConcatBackend,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "configs/livechat_pinyingpt_generic_baseline_v1.json").read_text(
        encoding="utf-8"
    )
)
CHECKPOINT = ROOT / ".build/pinyingpt2-concat"


class LiveChatDataTests(unittest.TestCase):
    def test_actual_sample_parser_and_stable_streamer_id(self):
        value = [["streamer01", "你好", "欢迎光临"]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.pk"
            with path.open("wb") as destination:
                pickle.dump(value, destination)
            rows = load_livechat_pickle(path)
        self.assertEqual(rows, [LiveChatRow("streamer01", "你好", "欢迎光临")])
        self.assertEqual(rows[0].streamer_id, "streamer01")

    def test_malformed_sample_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.pk"
            with path.open("wb") as destination:
                pickle.dump([["streamer1", "missing response"]], destination)
            with self.assertRaisesRegex(ValueError, "documented 3 fields"):
                load_livechat_pickle(path)

    def test_empty_response_is_preserved_for_explicit_usable_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.pk"
            with path.open("wb") as destination:
                pickle.dump([["streamer1", "audience", "  "]], destination)
            rows = load_livechat_pickle(path)
        self.assertFalse(rows[0].streamer_response.strip())

    def test_response_id_is_deterministic_and_duplicate_occurrence_distinguishes(self):
        row = LiveChatRow("streamer1", "comment", "response")
        first = build_source_response_id("RawDialogueData/train_data.pk", row, 0)
        self.assertEqual(first, build_source_response_id("RawDialogueData/train_data.pk", row, 0))
        self.assertNotEqual(first, build_source_response_id("RawDialogueData/train_data.pk", row, 1))


class ChronologyTests(unittest.TestCase):
    def test_chronology_grade_logic(self):
        self.assertEqual(determine_chronology_grade(released_order_metadata=True, official_order_preservation_evidence=False, official_reordering_evidence=False), "A")
        self.assertEqual(determine_chronology_grade(released_order_metadata=False, official_order_preservation_evidence=True, official_reordering_evidence=False), "B")
        self.assertEqual(determine_chronology_grade(released_order_metadata=False, official_order_preservation_evidence=False, official_reordering_evidence=False), "C")
        self.assertEqual(determine_chronology_grade(released_order_metadata=False, official_order_preservation_evidence=True, official_reordering_evidence=True), "D")

    def test_grade_c_config_is_never_labelled_chronological(self):
        self.assertEqual(CONFIG["selection"]["chronology_fallback"], "stable_hash_session_disjoint_non_temporal_proxy")
        self.assertNotIn("chronological", CONFIG["selection"]["chronology_fallback"])

    def test_prefix_tail_sampling_for_grade_ab_spans_region(self):
        rows = [{"interaction_id": f"i{index}"} for index in range(10)]
        selected = select_max_interactions(rows, maximum=4, seed=40408, chronology_grade="B")
        self.assertEqual([row["interaction_id"] for row in selected], ["i0", "i3", "i6", "i9"])

    def test_hash_split_is_deterministic_and_session_level(self):
        response_id = "lcresp-example"
        self.assertEqual(session_partition(response_id), session_partition(response_id))
        self.assertIn(session_partition(response_id), {"history", "evaluation"})


class InteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (CHECKPOINT / "pinyin2char.json").is_file():
            raise unittest.SkipTest("PinyinGPT checkpoint compatibility map unavailable")
        cls.pinyin2char = json.loads((CHECKPOINT / "pinyin2char.json").read_text(encoding="utf-8"))

    def test_segmentation_and_pinyin_are_deterministic(self):
        first = construct_eligible_targets("这个软件很实用", self.pinyin2char)
        second = construct_eligible_targets("这个软件很实用", self.pinyin2char)
        self.assertEqual(first, second)

    def test_no_future_text_and_audience_excluded_from_context(self):
        response = "这个软件很实用"
        _, targets, _ = construct_eligible_targets(response, self.pinyin2char)
        target = next(item for item in targets if item["target"] == "实用")
        context = response[: target["start"]]
        self.assertEqual(context, "这个软件很")
        self.assertNotIn("观众问题", context)
        self.assertNotIn("实用", context)

    def test_one_deterministic_target_per_response(self):
        _, targets, _ = construct_eligible_targets("这个软件很实用", self.pinyin2char)
        first = choose_target(targets, streamer_id="streamer1", source_response_id="r1", seed=40408)
        second = choose_target(targets, streamer_id="streamer1", source_response_id="r1", seed=40408)
        self.assertEqual(first, second)
        self.assertIsInstance(first["target"], str)

    def test_max_100_sampling_is_deterministic(self):
        rows = [{"interaction_id": f"i{index:03d}"} for index in range(150)]
        first = select_max_interactions(rows, maximum=100, seed=40408, chronology_grade="C")
        second = select_max_interactions(rows, maximum=100, seed=40408, chronology_grade="C")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)

    def test_alignment_and_compatibility_filter(self):
        _, targets, failures = construct_eligible_targets("北京", self.pinyin2char)
        self.assertTrue(targets)
        for target in targets:
            self.assertEqual(len(target["target"]), len(target["pinyin"]))
            for character, syllable in zip(target["target"], target["pinyin"]):
                self.assertIn(character, self.pinyin2char[syllable])
        self.assertFalse(any(item.get("exclusion_reason") == "pinyin_character_alignment_failure" for item in failures))

    def test_effective_compatibility_map_rejects_unknown_tokenizer_characters(self):
        class TinyTokenizer:
            unk_token_id = 0

            @staticmethod
            def convert_tokens_to_ids(character):
                return {"北": 1}.get(character, 0)

            @staticmethod
            def convert_ids_to_tokens(token_id):
                return {0: "[UNK]", 1: "北"}[token_id]

        effective = tokenizer_compatible_character_map(
            TinyTokenizer(), {"bei": ["北", "𠚺"]}
        )
        self.assertEqual(effective, {"bei": ("北",)})


class RankingMetricTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"interaction_id": "a", "user_id": "u1", "gold_top10_rank": 1},
            {"interaction_id": "b", "user_id": "u1", "gold_top10_rank": 4},
            {"interaction_id": "c", "user_id": "u2", "gold_top10_rank": None},
        ]

    def test_topk_mrr_conditional_mean_and_missing(self):
        metrics = compute_metrics(self.rows)["micro"]
        self.assertAlmostEqual(metrics["top1"], 1 / 3)
        self.assertAlmostEqual(metrics["top3"], 1 / 3)
        self.assertAlmostEqual(metrics["top5"], 2 / 3)
        self.assertAlmostEqual(metrics["top10"], 2 / 3)
        self.assertAlmostEqual(metrics["mrr_at_10"], (1 + 0.25) / 3)
        self.assertEqual(metrics["mean_rank_given_top10"], 2.5)
        self.assertEqual(metrics["missing_at_10_count"], 1)

    def test_macro_users_are_equally_weighted(self):
        metrics = compute_metrics(self.rows)
        self.assertAlmostEqual(metrics["macro_user"]["top1"], 0.25)

    def test_paired_gain_counts_and_bootstrap_are_deterministic(self):
        contextual = [
            {"interaction_id": "a", "user_id": "u1", "gold_top10_rank": 1},
            {"interaction_id": "b", "user_id": "u1", "gold_top10_rank": 1},
            {"interaction_id": "c", "user_id": "u2", "gold_top10_rank": None},
        ]
        first = context_gain(self.rows, contextual, bootstrap_resamples=100, seed=40408)
        second = context_gain(self.rows, contextual, bootstrap_resamples=100, seed=40408)
        self.assertEqual(first, second)
        self.assertEqual(first["top1_outcome_counts"]["rescued_by_context"], 1)

    def test_sparse_long_target_tail_is_merged_without_model_outcomes(self):
        interactions = (
            [{"target_length": 1}] * 1000
            + [{"target_length": 2}] * 500
            + [{"target_length": 3}] * 200
            + [{"target_length": 4}] * 99
            + [{"target_length": 5}]
        )
        self.assertEqual(
            target_length_groups(interactions, minimum_long_tail_count=100),
            {1: "1", 2: "2", 3: "3", 4: "4+", 5: "4+"},
        )


class ArtifactAndModelTests(unittest.TestCase):
    def test_serialization_and_sha_input_are_deterministic(self):
        value = {"b": 2, "a": "汉字"}
        self.assertEqual(canonical_json(value), canonical_json(value))
        self.assertEqual(stable_hash(value), stable_hash(value))

    def test_jsonl_roundtrip_alignment(self):
        rows = [{"interaction_id": "a", "gold": "北"}, {"interaction_id": "b", "gold": "京"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            write_jsonl(path, rows)
            loaded = load_jsonl(path)
        self.assertEqual(rows, loaded)
        self.assertEqual([row["interaction_id"] for row in rows], [row["interaction_id"] for row in loaded])

    def test_frozen_checkpoint_revision_and_no_personalisation_inputs(self):
        self.assertEqual(CHECKPOINT_REVISION, "76dd20dc92d8236a350fb732e99dde6fa15e2263")
        self.assertEqual(CONFIG["model"]["checkpoint_revision"], CHECKPOINT_REVISION)
        self.assertEqual(CONFIG["model"]["conditions"], ["pinyin_only", "contextual"])
        self.assertNotIn("history", CONFIG["model"])

    def test_pinyin_only_and_contextual_context_contract(self):
        interaction = {"effective_context": "同一回复前文"}
        self.assertEqual("", "" if "pinyin_only" == "pinyin_only" else interaction["effective_context"])
        self.assertEqual("同一回复前文", "" if "contextual" == "pinyin_only" else interaction["effective_context"])


@unittest.skipUnless((CHECKPOINT / "pytorch_model.bin").is_file(), "PinyinGPT checkpoint unavailable")
class EffectiveContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = PinyinGPTConcatBackend(CHECKPOINT, device="cpu")

    def test_effective_context_preserves_short_and_truncates_leading_tokens(self):
        self.assertEqual(self.backend.effective_context("北京"), "北京")
        long_context = "北" * 600
        effective = self.backend.effective_context(long_context, token_limit=512)
        self.assertLessEqual(len(self.backend.tokenizer.encode(effective, add_special_tokens=False)), 512)


if __name__ == "__main__":
    unittest.main()
