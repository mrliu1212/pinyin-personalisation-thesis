import json
import sys
import tempfile
import unittest
from pathlib import Path

from phase_04f_fixtures import (
    DeterministicPinyinDecoder,
    benchmark_record,
    make_backend,
)
from src.phase_04f_evaluation import evaluate_phase_04f
from src.reference_backend.pinyin_decoder import (
    LibrimeLunaPinyinDecoder,
    PinyinDecoderCandidate,
    PinyinDecoderResult,
    normalize_pinyin,
)
from src.reference_backend.pinyin_integration import (
    HUOZIIME_MEMORY_GROUNDED_SOURCE,
    PinyinIntegratedReferenceBackend,
    integrate_separate_channels,
)


ROOT = Path(__file__).resolve().parents[1]
RIME_EXECUTABLE = ROOT / ".build" / (
    "rime_candidate_cli.exe" if sys.platform == "win32" else "rime_candidate_cli"
)


def evaluation_fixture(root: Path, decoder: DeterministicPinyinDecoder):
    generic, _, _ = make_backend(root / "generic", user_id="generic_no_memory")
    correct, _, _ = make_backend(root / "correct", user_id="zhu_ziqing", with_memory=True)
    wrong, _, _ = make_backend(root / "wrong", user_id="lu_xun", with_memory=True)
    return evaluate_phase_04f(
        (benchmark_record(target="侧视"),),
        backends={
            "generic_no_memory": generic,
            "correct_user_memory": correct,
            "wrong_user_memory": wrong,
        },
        pinyin_decoder=decoder,
    )


class Phase04F1PinyinIntegrationTests(unittest.TestCase):
    def test_normalized_pinyin_is_consumed_by_decoder(self):
        decoder = DeterministicPinyinDecoder()
        result = decoder.decode(" Bei Jing ", top_k=3)
        self.assertEqual(result.normalized_pinyin, "beijing")
        self.assertEqual(result.consumed_input, "beijing")
        self.assertEqual(decoder.calls[0]["normalized"], "beijing")

    def test_normalization_does_not_guess_toned_or_non_pinyin_input(self):
        self.assertEqual(normalize_pinyin("LU:E"), "lve")
        with self.assertRaises(ValueError):
            normalize_pinyin("běijīng")

    @unittest.skipUnless(
        RIME_EXECUTABLE.exists()
        and (ROOT / "data/rime/setup_manifest.json").exists(),
        "pinned desktop Rime decoder is not prepared",
    )
    def test_real_decoder_returns_query_compatible_chinese_candidate(self):
        manifest = json.loads((ROOT / "data/rime/setup_manifest.json").read_text())
        with LibrimeLunaPinyinDecoder(
            executable=RIME_EXECUTABLE,
            shared_data=ROOT / "data/rime/shared",
            prebuilt_data=ROOT / "data/rime/build",
            version=manifest["librime"],
        ) as decoder:
            result = decoder.decode("beijing", top_k=10)
        self.assertIn("北京", [candidate.text for candidate in result.candidates])
        self.assertEqual(result.consumed_input, "beijing")
        self.assertTrue(all(candidate.source == "PINYIN_DECODER" for candidate in result.candidates))

    def test_target_rank_comes_only_from_pinyin_channel(self):
        with tempfile.TemporaryDirectory() as directory:
            result = evaluation_fixture(Path(directory), DeterministicPinyinDecoder())
        row = result["rows"][0]
        self.assertEqual(row["pinyin_conversion"]["target_rank"], 2)
        self.assertTrue(result["evaluation_layers"]["huoziime_personalisation"]["target_ranking_computed"] is False)
        self.assertTrue(all("target_rank" not in item for item in row["huoziime_conditions"].values()))

    def test_huoziime_prompt_is_unchanged_when_only_pinyin_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            backend, generation, _ = make_backend(Path(directory), user_id="zhu_ziqing")
            integrated = PinyinIntegratedReferenceBackend(
                pinyin_decoder=DeterministicPinyinDecoder(),
                huoziime_backend=backend,
            )
            integrated.predict("zhu_ziqing", "相同前文", "ceshi", 2, seed_base=77, record_trace=False)
            integrated.predict("zhu_ziqing", "相同前文", "beijing", 2, seed_base=77, record_trace=False)
        self.assertEqual(generation.calls[0]["prompt"], generation.calls[2]["prompt"])
        self.assertNotIn("ceshi", generation.calls[0]["prompt"])
        self.assertNotIn("beijing", generation.calls[2]["prompt"])

    def test_candidate_provenance_identifies_each_subsystem(self):
        with tempfile.TemporaryDirectory() as directory:
            backend, _, _ = make_backend(Path(directory), user_id="zhu_ziqing")
            decoded = DeterministicPinyinDecoder().decode("ceshi", top_k=3)
            huoziime = backend.predict(
                "zhu_ziqing", "前文", "ceshi", 2, seed_base=8, record_trace=False
            )
            integrated = integrate_separate_channels(decoded, huoziime)
        all_sources = {source for item in integrated.candidate_provenance for source in item.sources}
        self.assertIn("PINYIN_DECODER", all_sources)
        self.assertIn("HUOZIIME_DIRECT", all_sources)

    def test_grounded_candidates_preserve_memory_ids_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            backend, _, _ = make_backend(
                Path(directory), user_id="zhu_ziqing", trigger=True, with_memory=True
            )
            decoded = DeterministicPinyinDecoder().decode("ceshi", top_k=3)
            huoziime = backend.predict(
                "zhu_ziqing", "客户张总来访", "ceshi", 2, seed_base=9, record_trace=False
            )
            integrated = integrate_separate_channels(decoded, huoziime)
        grounded = [
            item for item in integrated.candidate_provenance
            if HUOZIIME_MEMORY_GROUNDED_SOURCE in item.sources
        ]
        self.assertTrue(grounded)
        self.assertEqual(grounded[0].grounded_memory_ids, huoziime.supplied_memory_ids)

    def test_same_text_keeps_multi_source_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            backend, _, _ = make_backend(
                Path(directory), user_id="zhu_ziqing", trigger=True, with_memory=True
            )
            huoziime = backend.predict(
                "zhu_ziqing", "客户张总来访", "ceshi", 1, seed_base=10, record_trace=False
            )
            decoded = PinyinDecoderResult(
                raw_input="ceshi",
                normalized_pinyin="ceshi",
                consumed_input="ceshi",
                candidates=(PinyinDecoderCandidate("准备红茶", 1),),
                latency_ms=0.1,
                decoder={"status": "TEST_STUB"},
            )
            integrated = integrate_separate_channels(decoded, huoziime)
        item = next(item for item in integrated.candidate_provenance if item.text == "准备红茶")
        self.assertEqual(set(item.sources), {"PINYIN_DECODER", "HUOZIIME_MEMORY_GROUNDED"})

    def test_all_memory_conditions_share_one_decoder_call(self):
        decoder = DeterministicPinyinDecoder()
        with tempfile.TemporaryDirectory() as directory:
            evaluation_fixture(Path(directory), decoder)
        self.assertEqual(len(decoder.calls), 1)

    def test_pinyin_output_is_identical_across_integrated_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            result = evaluation_fixture(Path(directory), DeterministicPinyinDecoder())
        integrated = result["rows"][0]["integrated_conditions"]
        candidate_lists = [tuple(item["pinyin_candidate_texts"]) for item in integrated.values()]
        self.assertEqual(len(set(candidate_lists)), 1)

    def test_only_huoziime_personal_state_differs_across_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            result = evaluation_fixture(Path(directory), DeterministicPinyinDecoder())
        conditions = result["rows"][0]["huoziime_conditions"]
        users = [conditions[name]["personal_state_user_id"] for name in conditions]
        prompts = [conditions[name]["prediction"]["provenance"]["direct_prompt_sha256"] for name in conditions]
        seeds = [conditions[name]["prediction"]["provenance"]["direct_seeds"] for name in conditions]
        self.assertEqual(len(set(users)), 3)
        self.assertEqual(len(set(prompts)), 1)
        self.assertEqual(seeds[0], seeds[1])
        self.assertEqual(seeds[1], seeds[2])

    def test_external_context_remains_none(self):
        with tempfile.TemporaryDirectory() as directory:
            result = evaluation_fixture(Path(directory), DeterministicPinyinDecoder())
        self.assertIsNone(result["external_context"])
        self.assertIsNone(result["rows"][0]["external_context"])
        for condition in result["rows"][0]["huoziime_conditions"].values():
            self.assertIsNone(condition["prediction"]["external_context"])

    def test_no_phase_04e_reranker_is_called_or_imported(self):
        sources = "\n".join(
            (ROOT / path).read_text(encoding="utf-8")
            for path in (
                "src/reference_backend/pinyin_decoder.py",
                "src/reference_backend/pinyin_integration.py",
                "src/phase_04f_evaluation.py",
            )
        ).lower()
        self.assertNotIn("learned_reranker", sources)
        self.assertNotIn("personal_features", sources)

    def test_separate_channel_mode_has_no_invented_fusion(self):
        with tempfile.TemporaryDirectory() as directory:
            result = evaluation_fixture(Path(directory), DeterministicPinyinDecoder())
        self.assertFalse(result["integration"]["channels_unified"])
        self.assertIsNone(result["integration"]["official_numerical_fusion_rule"])
        self.assertIsNone(result["evaluation_layers"]["integrated_backend"]["unified_top_k_metrics"])
        self.assertTrue(all(
            item["channels_unified"] is False
            for item in result["rows"][0]["integrated_conditions"].values()
        ))

    def test_direct_and_grounded_suggestions_are_both_exposed(self):
        with tempfile.TemporaryDirectory() as directory:
            backend, _, _ = make_backend(
                Path(directory), user_id="zhu_ziqing", trigger=True, with_memory=True
            )
            result = backend.predict(
                "zhu_ziqing", "客户张总来访", "ceshi", 1, seed_base=11, record_trace=False
            )
        self.assertIsInstance(result.direct_candidates, tuple)
        self.assertTrue(result.grounded_candidates)
        self.assertEqual(result.grounded_candidates, result.candidates)


if __name__ == "__main__":
    unittest.main()
