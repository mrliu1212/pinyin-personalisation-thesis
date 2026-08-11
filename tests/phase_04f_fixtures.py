"""Small deterministic Phase 4F fixtures; no model downloads are required."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.reference_backend.backend import ReferencePersonalisedIMEBackend
from src.reference_backend.candidate_generator import HuoziIMECandidateGenerator
from src.reference_backend.interaction_store import InteractionTraceStore
from src.reference_backend.memory_store import MemoryRecord, MemoryStore
from src.reference_backend.model_runtime import RuntimeGeneration
from src.reference_backend.pinyin_decoder import (
    PinyinDecoderCandidate,
    PinyinDecoderResult,
    normalize_pinyin,
)
from src.reference_backend.vector_index import HNSWMemoryIndex


class DeterministicEmbeddingRuntime:
    dimension = 4

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> tuple[float, ...]:
        self.calls.append(text)
        if any(marker in text for marker in ("客户", "红茶", "来访", "接待")):
            return (1.0, 0.0, 0.0, 0.0)
        if any(marker in text for marker in ("会议", "报告", "工作")):
            return (0.0, 1.0, 0.0, 0.0)
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return tuple((digest[index] + 1) / 256.0 for index in range(self.dimension))

    def info(self) -> dict[str, Any]:
        return {"runtime": "deterministic-test-embedding", "dimension": self.dimension}


class DeterministicGenerationRuntime:
    supports_official_trigger_policy = True

    def __init__(self, *, trigger: bool = False, extraction: bool = False) -> None:
        self.trigger = trigger
        self.extraction = extraction
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        seed: int,
        top_k: int,
        top_p: float,
        temperature: float,
        repeat_penalty: float,
        repeat_last_n: int,
    ) -> RuntimeGeneration:
        self.calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "seed": seed,
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "repeat_penalty": repeat_penalty,
                "repeat_last_n": repeat_last_n,
            }
        )
        if self.extraction and "[MEMORY_WORKER]" in prompt:
            text = (
                '{"summary":"客户张总本周五来访，准备红茶",'
                '"participants":["张总"],"item":"接待","detail":"准备红茶"}'
            )
        elif "<memory>\n- " in prompt:
            text = "准备红茶"
        elif "<memory>\n<NO_MEM>" in prompt:
            text = "稍后确认"
        elif self.trigger:
            text = '<MEM_RETRIEVAL> query="客户张总来访准备什么" </MEM_RETRIEVAL>'
        else:
            text = f"候选{seed % 97}"
        return RuntimeGeneration(text=text, score=-0.25, seed=seed, elapsed_ms=0.1)

    def info(self) -> dict[str, Any]:
        return {
            "runtime": "deterministic-test-generation",
            "official_checkpoint_verified": self.supports_official_trigger_policy,
            "calls": len(self.calls),
        }


class DeterministicPinyinDecoder:
    max_candidates = 10

    def __init__(self, outputs: dict[str, tuple[str, ...]] | None = None) -> None:
        self.outputs = outputs or {
            "ceshi": ("测试", "侧视", "候选1"),
            "beijing": ("北京", "背景", "北境"),
        }
        self.calls: list[dict[str, Any]] = []

    def decode(self, pinyin_or_keystrokes: str, *, top_k: int) -> PinyinDecoderResult:
        normalized = normalize_pinyin(pinyin_or_keystrokes)
        self.calls.append({"raw": pinyin_or_keystrokes, "normalized": normalized, "top_k": top_k})
        candidates = tuple(
            PinyinDecoderCandidate(text=text, rank=index)
            for index, text in enumerate(self.outputs.get(normalized, ()), start=1)
            if index <= top_k
        )
        return PinyinDecoderResult(
            raw_input=pinyin_or_keystrokes,
            normalized_pinyin=normalized,
            consumed_input=normalized,
            candidates=candidates,
            latency_ms=0.25,
            decoder={
                "implementation": "deterministic-test-pinyin-decoder",
                "schema": "test_full_pinyin",
                "candidate_count": self.max_candidates,
                "status": "TEST_STUB",
            },
        )


def make_memory(
    user_id: str,
    *,
    source_id: str = "train-1",
    text: str = "客户张总本周五来访，准备红茶",
    position: str = "1925-01-01|000000000001|work",
) -> MemoryRecord:
    return MemoryRecord.create(
        user_id=user_id,
        plaintext=text,
        creation_position=position,
        source_interaction_ids=(source_id,),
        provenance={"work_id": "train-work"},
    )


def make_backend(
    root: Path,
    *,
    user_id: str,
    trigger: bool = False,
    with_memory: bool = False,
    trace: bool = False,
) -> tuple[
    ReferencePersonalisedIMEBackend,
    DeterministicGenerationRuntime,
    DeterministicEmbeddingRuntime,
]:
    generation = DeterministicGenerationRuntime(trigger=trigger)
    embedding = DeterministicEmbeddingRuntime()
    store = MemoryStore(root / "l2", user_id=user_id)
    index = HNSWMemoryIndex(root / "hnsw", user_id=user_id, dimension=embedding.dimension)
    if with_memory:
        memory = make_memory(user_id, source_id=f"{user_id}-train-1")
        indexed = index.add(memory, embedding.embed(memory.plaintext))
        store.add(indexed)
    index.persist()
    traces = InteractionTraceStore(root / "l3", user_id=user_id) if trace else None
    backend = ReferencePersonalisedIMEBackend(
        user_id=user_id,
        candidate_generator=HuoziIMECandidateGenerator(generation),
        embedding_runtime=embedding,
        memory_store=store,
        memory_index=index,
        interaction_store=traces,
        official_trigger_policy=True,
    )
    return backend, generation, embedding


def benchmark_record(
    *,
    interaction_id: str = "zhu-test-1",
    author_id: str = "zhu_ziqing",
    target: str = "候选1",
) -> dict[str, Any]:
    return {
        "interaction_id": interaction_id,
        "author_id": author_id,
        "work_id": "spring",
        "work_date": "1933-07",
        "source_start_offset": 101,
        "source_end_offset": 103,
        "raw_context": "甲" * 120,
        "derived_context": "甲" * 12,
        "pinyin": "ceshi",
        "target_candidate": target,
    }
