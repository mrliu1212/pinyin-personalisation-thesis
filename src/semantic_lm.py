"""Frozen causal-LM candidate scoring with deterministic local caching."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Protocol, Sequence


SEMANTIC_LM_REPO = "Qwen/Qwen3-0.6B-Base"
SEMANTIC_CONTEXT_CHARACTERS = 64
CACHE_SCHEMA_VERSION = 1


def is_chinese_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
    )


def semantic_context_64(raw_preceding_context: str) -> str:
    chinese = "".join(
        character
        for character in raw_preceding_context
        if is_chinese_character(character)
    )
    return chinese[-SEMANTIC_CONTEXT_CHARACTERS:]


def min_max_normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if low == high:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def select_device(torch_module: Any) -> str:
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def set_deterministic_seeds(seed: int = 40408) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class MeanLogProbabilityBackend(Protocol):
    def mean_log_probability(
        self, prefix_ids: Sequence[int], candidate_ids: Sequence[int]
    ) -> float: ...


class JsonCache:
    def __init__(self, directory: Path, namespace: str) -> None:
        self.directory = directory / namespace
        self._memory: dict[Path, Any] = {}

    def path(self, payload: dict[str, Any]) -> Path:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return self.directory / f"{hashlib.sha256(encoded).hexdigest()}.json"

    def get(self, payload: dict[str, Any]) -> Any | None:
        path = self.path(payload)
        if path in self._memory:
            return self._memory[path]
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))["value"]
        self._memory[path] = value
        return value

    def set(self, payload: dict[str, Any], value: Any) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path(payload)
        path.write_text(
            json.dumps(
                {"schema_version": CACHE_SCHEMA_VERSION, "key": payload, "value": value},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self._memory[path] = value


class HuggingFaceCausalBackend:
    def __init__(self, model: Any, *, device: str, torch_module: Any) -> None:
        self.model = model
        self.device = device
        self.torch = torch_module
        self.model.eval()

    def mean_log_probability(
        self, prefix_ids: Sequence[int], candidate_ids: Sequence[int]
    ) -> float:
        if not prefix_ids:
            raise ValueError("causal scoring requires a non-empty prefix")
        if not candidate_ids:
            raise ValueError("candidate must contain at least one token")
        combined = list(prefix_ids) + list(candidate_ids)
        input_ids = self.torch.tensor([combined], dtype=self.torch.long, device=self.device)
        with self.torch.no_grad():
            logits = self.model(input_ids=input_ids).logits[0]
            log_probs = self.torch.log_softmax(logits, dim=-1)
            start = len(prefix_ids) - 1
            values = [
                log_probs[start + offset, token_id]
                for offset, token_id in enumerate(candidate_ids)
            ]
            result = self.torch.stack(values).mean().item()
        if not math.isfinite(result):
            raise ValueError("causal LM returned a non-finite log probability")
        return float(result)


@dataclass(frozen=True)
class CandidateSemanticScore:
    candidate: str
    candidate_token_count: int
    lm_conditional_logprob: float
    lm_prior_logprob: float
    lm_context_gain: float
    normalized_lm_conditional: float = 0.0
    normalized_lm_context_gain: float = 0.0


class CausalLMCandidateScorer:
    CONFIGURATION = {
        "context_candidate_tokenization": "separate; add_special_tokens=False",
        "score": "mean candidate-token autoregressive log probability",
        "prior": "minimal BOS-compatible prefix",
    }

    def __init__(
        self,
        tokenizer: Any,
        backend: MeanLogProbabilityBackend,
        *,
        revision: str,
        cache_dir: Path = Path(".cache/phase_04e"),
        runtime_configuration: dict[str, str] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.backend = backend
        self.revision = revision
        self.cache = JsonCache(cache_dir, "semantic_lm")
        self.runtime_configuration = runtime_configuration or {
            "device": "stub_or_unspecified",
            "dtype": "stub_or_unspecified",
        }
        start_token = tokenizer.bos_token_id
        if start_token is None:
            start_token = tokenizer.eos_token_id
        if start_token is None:
            raise ValueError("tokenizer has no BOS- or EOS-compatible start token")
        self.start_ids = [int(start_token)]

    @classmethod
    def from_pretrained(
        cls,
        *,
        revision: str,
        cache_dir: Path = Path(".cache/phase_04e"),
        device: str | None = None,
    ) -> tuple["CausalLMCandidateScorer", dict[str, str]]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        set_deterministic_seeds()
        resolved_device = device or select_device(torch)
        dtype = torch.float32 if resolved_device == "cpu" else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(SEMANTIC_LM_REPO, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            SEMANTIC_LM_REPO, revision=revision, dtype=dtype
        ).to(resolved_device)
        backend = HuggingFaceCausalBackend(
            model, device=resolved_device, torch_module=torch
        )
        return (
            cls(
                tokenizer,
                backend,
                revision=revision,
                cache_dir=cache_dir,
                runtime_configuration={
                    "device": resolved_device,
                    "dtype": str(dtype),
                },
            ),
            {"device": resolved_device, "dtype": str(dtype)},
        )

    def _ids(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)
        return [int(value) for value in encoded["input_ids"]]

    def _cached_score(
        self,
        *,
        kind: str,
        context: str,
        candidate: str,
        prefix_ids: Sequence[int],
        candidate_ids: Sequence[int],
    ) -> float:
        key = {
            "model": SEMANTIC_LM_REPO,
            "revision": self.revision,
            "kind": kind,
            "context": context,
            "candidate": candidate,
            "configuration": self.CONFIGURATION,
            "runtime_configuration": self.runtime_configuration,
        }
        cached = self.cache.get(key)
        if cached is not None:
            return float(cached)
        value = self.backend.mean_log_probability(prefix_ids, candidate_ids)
        self.cache.set(key, value)
        return value

    def score(self, context: str, candidate: str) -> CandidateSemanticScore:
        context_ids = self._ids(context)
        candidate_ids = self._ids(candidate)
        if not candidate_ids:
            raise ValueError("candidate tokenization is empty")
        conditional_prefix = context_ids or self.start_ids
        conditional = self._cached_score(
            kind="conditional",
            context=context,
            candidate=candidate,
            prefix_ids=conditional_prefix,
            candidate_ids=candidate_ids,
        )
        prior = self._cached_score(
            kind="prior",
            context="",
            candidate=candidate,
            prefix_ids=self.start_ids,
            candidate_ids=candidate_ids,
        )
        return CandidateSemanticScore(
            candidate=candidate,
            candidate_token_count=len(candidate_ids),
            lm_conditional_logprob=conditional,
            lm_prior_logprob=prior,
            lm_context_gain=conditional - prior,
        )

    def score_candidates(
        self, context: str, candidates: Sequence[str]
    ) -> tuple[CandidateSemanticScore, ...]:
        raw = [self.score(context, candidate) for candidate in candidates]
        conditional = min_max_normalize(
            [item.lm_conditional_logprob for item in raw]
        )
        gain = min_max_normalize([item.lm_context_gain for item in raw])
        return tuple(
            CandidateSemanticScore(
                **{
                    **asdict(item),
                    "normalized_lm_conditional": normalized_conditional,
                    "normalized_lm_context_gain": normalized_gain,
                }
            )
            for item, normalized_conditional, normalized_gain in zip(
                raw, conditional, gain
            )
        )
