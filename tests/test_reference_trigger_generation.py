import unittest

from phase_04f_fixtures import DeterministicGenerationRuntime
from src.reference_backend.candidate_generator import (
    HuoziIMECandidateGenerator,
    clean_candidate,
)
from src.reference_backend.memory_trigger import (
    OfficialTokenMemoryTrigger,
    ReferenceMemoryTriggerFallback,
    parse_memory_query,
)


class ReferenceTriggerGenerationTests(unittest.TestCase):
    def test_official_action_token_is_parsed_and_recorded(self):
        raw = '<MEM_RETRIEVAL> query="客户来访准备什么" </MEM_RETRIEVAL>'
        self.assertEqual(parse_memory_query((raw,)), ("客户来访准备什么", raw))
        decision = OfficialTokenMemoryTrigger(
            official_checkpoint_policy=True
        ).should_retrieve((raw,))
        self.assertTrue(decision.should_retrieve)
        self.assertEqual(decision.method, "OFFICIAL_POLICY")
        self.assertEqual(decision.raw_evidence, raw)

    def test_fallback_is_explicit_and_never_infers_a_threshold(self):
        decision = ReferenceMemoryTriggerFallback().should_retrieve(
            ('<MEM_RETRIEVAL> query="ignored" </MEM_RETRIEVAL>',)
        )
        self.assertFalse(decision.should_retrieve)
        self.assertEqual(decision.method, "ARCHITECTURAL_FALLBACK")

    def test_non_official_query_syntax_is_not_silently_accepted(self):
        self.assertIsNone(parse_memory_query(("query: '客户来访'",)))

    def test_candidate_generation_preserves_frozen_decoding_and_provenance(self):
        runtime = DeterministicGenerationRuntime()
        batch = HuoziIMECandidateGenerator(runtime).generate(
            "我们正在讨论",
            top_k=3,
            external_context=None,
            seed_base=100,
        )
        self.assertEqual(len(batch.candidates), 3)
        self.assertEqual(batch.seeds, (100, 101, 102))
        self.assertTrue(all(item.source == "direct_generation" for item in batch.candidates))
        call = runtime.calls[0]
        self.assertEqual(call["top_k"], 20)
        self.assertEqual(call["top_p"], 0.8)
        self.assertEqual(call["temperature"], 0.7)
        self.assertEqual(call["repeat_penalty"], 1.2)
        self.assertEqual(call["repeat_last_n"], 16)
        self.assertEqual(call["max_tokens"], 8)

    def test_memory_grounded_generation_receives_plaintext_in_prompt(self):
        runtime = DeterministicGenerationRuntime()
        batch = HuoziIMECandidateGenerator(runtime).generate(
            "客户张总本周五来访，我应该",
            top_k=1,
            external_context=None,
            memory_plaintext=("客户张总喜欢红茶",),
            seed_base=200,
        )
        self.assertEqual(batch.source, "memory_grounded_generation")
        self.assertIn("客户张总喜欢红茶", batch.prompt)
        self.assertEqual(batch.candidates[0].text, "准备红茶")
        self.assertEqual(batch.candidates[0].source, "memory_grounded_generation")

    def test_control_actions_are_not_exposed_as_text_candidates(self):
        raw = '<MEM_RETRIEVAL> query="客户" </MEM_RETRIEVAL>'
        self.assertIsNone(clean_candidate(raw, instruction_prefix=""))


if __name__ == "__main__":
    unittest.main()
