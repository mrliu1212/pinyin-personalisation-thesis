"""HuoziIME-style direct and memory-grounded LLM completion."""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Sequence
import unicodedata

from .model_runtime import GenerationRuntime
from .official_prompts import build_typing_prompt
from .provenance import GeneratedCandidate


INVISIBLE = {
    "\u200b", "\u200c", "\u200d", "\u200e", "\u200f", "\u202a",
    "\u202b", "\u202c", "\u202d", "\u202e", "\u2060", "\u2061",
    "\u2062", "\u2063", "\u2064", "\u2066", "\u2067", "\u2068",
    "\u2069", "\ufeff",
}
CONTROL_MARKERS = ("<MEM_RETRIEVAL>", "</MEM_RETRIEVAL>", "<NO_MEM>")


@dataclass(frozen=True)
class GenerationBatch:
    candidates: tuple[GeneratedCandidate, ...]
    raw_outputs: tuple[str, ...]
    prompt: str
    elapsed_ms: float
    seeds: tuple[int, ...]
    source: str


def _strip_invisible(value: str) -> str:
    return "".join(
        ch
        for ch in value
        if ch not in INVISIBLE and not unicodedata.category(ch).startswith("C")
    )


def _strip_thinking(value: str) -> str:
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in value:
        value = value.split("</think>", 1)[1]
    return value.replace("<think>", "")


def _truncate_like_native(value: str) -> str:
    positions = [value.find(mark) for mark in ("。", "！", "？", "\n")]
    valid = [position for position in positions if position >= 0]
    return value[: min(valid)] if valid else value


def clean_candidate(raw: str, *, instruction_prefix: str) -> str | None:
    clean = _strip_thinking(_strip_invisible(raw))
    clean = clean.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    if not clean:
        return None
    upper = clean.upper()
    if any(marker in upper for marker in CONTROL_MARKERS):
        return None
    if re.search(r'\bquery\s*[:=]', clean, re.IGNORECASE):
        return None
    if instruction_prefix and clean.startswith(instruction_prefix):
        clean = clean[len(instruction_prefix):].lstrip()
    clean = _truncate_like_native(clean).strip()
    if not clean or clean.startswith("_") or clean in {"ensions", "neider"}:
        return None
    if "[对方]" in clean or "[我]" in clean:
        return None
    return clean


class HuoziIMECandidateGenerator:
    """Direct generative path; never consumes Luna candidates or Phase 4E code."""

    TOP_K = 20
    TOP_P = 0.8
    TEMPERATURE = 0.7
    REPEAT_PENALTY = 1.2
    REPEAT_LAST_N = 16
    MAX_TOKENS = 8
    DEFAULT_CANDIDATES = 4

    def __init__(self, runtime: GenerationRuntime) -> None:
        self.runtime = runtime

    def generate(
        self,
        preceding_text: str,
        *,
        top_k: int = DEFAULT_CANDIDATES,
        external_context: str | None = None,
        memory_plaintext: Sequence[str] = (),
        seed_base: int,
    ) -> GenerationBatch:
        if not 1 <= top_k <= 20:
            raise ValueError("top_k candidate count must be between 1 and 20")
        if tuple(memory_plaintext) == ("<NO_MEM>",):
            memory = "<NO_MEM>"
            source = "memory_rerun_no_hit"
        elif memory_plaintext:
            memory = "\n".join(
                f"- {item.replace(chr(10), ' ').strip()}" for item in memory_plaintext
            )
            source = "memory_grounded_generation"
        else:
            memory = None
            source = "direct_generation"
        prompt = build_typing_prompt(
            preceding_text,
            memory=memory,
            external_context=external_context,
        )
        start = time.perf_counter()
        outputs = []
        seeds = tuple((seed_base + index) & 0x7FFF_FFFF for index in range(top_k))
        for seed in seeds:
            outputs.append(
                self.runtime.generate(
                    prompt,
                    max_tokens=self.MAX_TOKENS,
                    seed=seed,
                    top_k=self.TOP_K,
                    top_p=self.TOP_P,
                    temperature=self.TEMPERATURE,
                    repeat_penalty=self.REPEAT_PENALTY,
                    repeat_last_n=self.REPEAT_LAST_N,
                )
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        deduped: list[GeneratedCandidate] = []
        seen = set()
        prefix = preceding_text[-100:]
        for output in outputs:
            clean = clean_candidate(output.text, instruction_prefix=prefix)
            if clean is None or clean in seen:
                continue
            seen.add(clean)
            deduped.append(
                GeneratedCandidate(
                    text=clean,
                    rank=len(deduped) + 1,
                    generation_score=output.score,
                    seed=output.seed,
                    source=source,
                    raw_text=output.text,
                )
            )
        return GenerationBatch(
            candidates=tuple(deduped),
            raw_outputs=tuple(output.text for output in outputs),
            prompt=prompt,
            elapsed_ms=elapsed_ms,
            seeds=seeds,
            source=source,
        )
