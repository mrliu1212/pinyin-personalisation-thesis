import tempfile
import unittest
from pathlib import Path

from src.semantic_lm import (
    CausalLMCandidateScorer,
    min_max_normalize,
    semantic_context_64,
)


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self):
        self.calls = []

    def __call__(self, text, *, add_special_tokens):
        self.calls.append((text, add_special_tokens))
        return {"input_ids": [ord(character) % 101 + 3 for character in text]}


class FakeBackend:
    def __init__(self):
        self.calls = []

    def mean_log_probability(self, prefix_ids, candidate_ids):
        self.calls.append((tuple(prefix_ids), tuple(candidate_ids)))
        return (sum(candidate_ids) + 0.01 * sum(prefix_ids)) / len(candidate_ids)


class SemanticLMTests(unittest.TestCase):
    def test_scoring_is_deterministic_and_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = FakeTokenizer()
            backend = FakeBackend()
            scorer = CausalLMCandidateScorer(
                tokenizer, backend, revision="revision", cache_dir=Path(directory)
            )
            first = scorer.score("上下文", "候选")
            second = scorer.score("上下文", "候选")
            self.assertEqual(first, second)
            self.assertEqual(len(backend.calls), 2)  # conditional plus cached prior

    def test_context_and_candidate_are_tokenized_separately_without_special_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = FakeTokenizer()
            backend = FakeBackend()
            scorer = CausalLMCandidateScorer(
                tokenizer, backend, revision="revision", cache_dir=Path(directory)
            )
            scorer.score("上下文", "候选")
            self.assertEqual(tokenizer.calls, [("上下文", False), ("候选", False)])
            context_ids = tuple(ord(character) % 101 + 3 for character in "上下文")
            candidate_ids = tuple(ord(character) % 101 + 3 for character in "候选")
            self.assertEqual(backend.calls[0], (context_ids, candidate_ids))

    def test_candidate_token_length_normalization_and_context_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            scorer = CausalLMCandidateScorer(
                FakeTokenizer(), FakeBackend(), revision="r", cache_dir=Path(directory)
            )
            score = scorer.score("语境", "三个字")
            self.assertEqual(score.candidate_token_count, 3)
            self.assertAlmostEqual(
                score.lm_context_gain,
                score.lm_conditional_logprob - score.lm_prior_logprob,
            )

    def test_candidate_list_normalization_and_constant_policy(self):
        self.assertEqual(min_max_normalize([2.0, 4.0, 3.0]), [0.0, 1.0, 0.5])
        self.assertEqual(min_max_normalize([2.0, 2.0]), [0.0, 0.0])

    def test_semantic_context_uses_only_final_64_preceding_chinese_characters(self):
        preceding = "前" * 70 + " abc123"
        self.assertEqual(semantic_context_64(preceding), "前" * 64)
        self.assertNotIn("目标", semantic_context_64(preceding))


if __name__ == "__main__":
    unittest.main()
