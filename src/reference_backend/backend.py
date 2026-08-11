"""Inspectable HuoziIME reference-personalisation backend."""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any

from .candidate_generator import HuoziIMECandidateGenerator
from .interaction_store import InteractionTrace, InteractionTraceStore
from .memory_store import MemoryRecord, MemoryStore
from .memory_trigger import OfficialTokenMemoryTrigger, ReferenceMemoryTriggerFallback
from .model_runtime import EmbeddingRuntime
from .provenance import (
    PredictionResult,
    RetrievedMemory,
    TimingBreakdown,
    stable_query_id,
)
from .vector_index import HNSWMemoryIndex


def _cjk_bigrams(text: str) -> set[str]:
    clean = "".join(char for char in text if "\u4e00" <= char <= "\u9fff")
    return {clean[index : index + 2] for index in range(max(0, len(clean) - 1))}


def _latin_tokens(text: str) -> set[str]:
    return {item for item in re.split(r"[^a-z0-9]+", text.lower()) if len(item) >= 2}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def lexical_score(query: str, candidate: str) -> float:
    return 0.85 * _jaccard(_cjk_bigrams(query), _cjk_bigrams(candidate)) + 0.15 * _jaccard(
        _latin_tokens(query), _latin_tokens(candidate)
    )


def normalize_memory_query(query: str) -> str:
    # Audited source-specific normalizations retained verbatim as runtime behaviour.
    return (
        query.replace("高层例会", "管理层会议")
        .replace("高层", "管理层")
        .replace("例会", "会议")
        .strip()
    )


class ReferencePersonalisedIMEBackend:
    VECTOR_TOP_K = 20
    VECTOR_COSINE_THRESHOLD = 0.4
    VECTOR_WEIGHT = 0.88
    LEXICAL_WEIGHT = 0.12

    def __init__(
        self,
        *,
        user_id: str,
        candidate_generator: HuoziIMECandidateGenerator,
        embedding_runtime: EmbeddingRuntime,
        memory_store: MemoryStore,
        memory_index: HNSWMemoryIndex,
        interaction_store: InteractionTraceStore | None = None,
        official_trigger_policy: bool | None = None,
    ) -> None:
        if memory_store.user_id != user_id or memory_index.user_id != user_id:
            raise ValueError("backend memory components must belong to user_id")
        self.user_id = user_id
        self.candidate_generator = candidate_generator
        self.embedding_runtime = embedding_runtime
        self.memory_store = memory_store
        self.memory_index = memory_index
        self.interaction_store = interaction_store
        runtime_policy = bool(
            getattr(candidate_generator.runtime, "supports_official_trigger_policy", False)
        )
        if official_trigger_policy is None:
            official_trigger_policy = runtime_policy
        if official_trigger_policy and not runtime_policy:
            raise ValueError("cannot claim official policy without the pinned checkpoint")
        self.memory_trigger = (
            OfficialTokenMemoryTrigger(official_checkpoint_policy=True)
            if official_trigger_policy
            else ReferenceMemoryTriggerFallback()
        )
        self._prediction_count = 0

    def predict(
        self,
        user_id: str,
        preceding_text: str,
        pinyin_or_keystrokes: str,
        top_k: int,
        external_context: str | None = None,
        *,
        seed_base: int | None = None,
        chronological_position: str | None = None,
        selected_text: str | None = None,
        request_nonce: str = "",
        record_trace: bool = True,
    ) -> PredictionResult:
        if user_id != self.user_id:
            raise ValueError("backend user isolation violation")
        if not isinstance(preceding_text, str) or not isinstance(pinyin_or_keystrokes, str):
            raise TypeError("preceding text and keystrokes must be strings")
        query_id = stable_query_id(
            user_id, preceding_text, pinyin_or_keystrokes, external_context, request_nonce
        )
        if seed_base is None:
            seed_base = int(hashlib.sha256(query_id.encode()).hexdigest()[:8], 16) & 0x7FFF_FFFF
        start_total = time.perf_counter()
        prediction_count_before = self._prediction_count
        direct = self.candidate_generator.generate(
            preceding_text,
            top_k=top_k,
            external_context=external_context,
            seed_base=seed_base,
        )
        trigger = self.memory_trigger.should_retrieve(direct.raw_outputs)
        retrieved: tuple[RetrievedMemory, ...] = ()
        supplied_records: tuple[MemoryRecord, ...] = ()
        query_embedding_ms = hnsw_ms = grounded_ms = 0.0
        final_batch = direct
        retrieval_diagnostic: dict[str, Any] = {
            "retrieval_attempted": trigger.should_retrieve,
            "flat_vector_scores": False,
            "selected_memory_id": None,
        }

        if trigger.should_retrieve:
            normalized_query = normalize_memory_query(trigger.query or "")
            embed_start = time.perf_counter()
            query_vector = self.embedding_runtime.embed(normalized_query)
            query_embedding_ms = (time.perf_counter() - embed_start) * 1000.0
            search_start = time.perf_counter()
            vector_results = self.memory_index.search(
                user_id=user_id,
                query_vector=query_vector,
                k=self.VECTOR_TOP_K,
            )
            hnsw_ms = (time.perf_counter() - search_start) * 1000.0
            active = {record.memory_id: record for record in self.memory_store.list(active_only=True)}
            scored = []
            for result in vector_results:
                record = active.get(result.memory_id)
                if record is None:
                    continue
                sim01 = max(0.0, min(1.0, (result.raw_cosine + 1.0) / 2.0))
                lex = lexical_score(normalized_query, record.plaintext)
                combined = max(
                    0.0,
                    min(1.0, self.VECTOR_WEIGHT * sim01 + self.LEXICAL_WEIGHT * lex),
                )
                scored.append((combined, sim01, result.raw_cosine, result.vector_label, lex, record))
            scored.sort(key=lambda item: (-item[0], -item[1], item[3]))
            raw_scores = [item[2] for item in scored]
            flat = len(raw_scores) >= 3 and max(raw_scores) - min(raw_scores) < 0.001
            retrieval_diagnostic["flat_vector_scores"] = flat
            retrieved = tuple(
                RetrievedMemory(
                    memory_id=record.memory_id,
                    user_id=record.user_id,
                    vector_label=label,
                    plaintext=record.plaintext,
                    raw_cosine=raw_cosine,
                    similarity_01=sim01,
                    lexical_score=lex,
                    combined_score=combined,
                    source_interaction_ids=record.source_interaction_ids,
                )
                for combined, sim01, raw_cosine, label, lex, record in scored
            )
            selected = None if flat else next(
                (item[-1] for item in scored if item[2] >= self.VECTOR_COSINE_THRESHOLD),
                None,
            )
            supplied_records = (selected,) if selected is not None else ()
            retrieval_diagnostic["selected_memory_id"] = (
                selected.memory_id if selected is not None else None
            )
            grounded_start = time.perf_counter()
            final_batch = self.candidate_generator.generate(
                preceding_text,
                top_k=top_k,
                external_context=external_context,
                memory_plaintext=(
                    tuple(record.plaintext for record in supplied_records)
                    if supplied_records
                    else ("<NO_MEM>",)
                ),
                seed_base=(seed_base + 100_000) & 0x7FFF_FFFF,
            )
            grounded_ms = (time.perf_counter() - grounded_start) * 1000.0

        self._prediction_count += 1
        total_ms = (time.perf_counter() - start_total) * 1000.0
        result = PredictionResult(
            query_id=query_id,
            user_id=user_id,
            preceding_text=preceding_text,
            pinyin_or_keystrokes=pinyin_or_keystrokes,
            external_context=external_context,
            candidates=final_batch.candidates,
            memory_trigger=trigger,
            retrieved_memories=retrieved,
            supplied_memory_ids=tuple(record.memory_id for record in supplied_records),
            supplied_memory_plaintext=tuple(record.plaintext for record in supplied_records),
            model_runtime=self.candidate_generator.runtime.info(),
            timing=TimingBreakdown(
                direct_generation_ms=direct.elapsed_ms,
                query_embedding_ms=query_embedding_ms,
                hnsw_search_ms=hnsw_ms,
                grounded_generation_ms=grounded_ms,
                total_ms=total_ms,
            ),
            cache_status={
                "backend_prediction_count_before": prediction_count_before,
                "resident_backend_reused": prediction_count_before > 0,
                "prediction_rebuilt_memory": False,
            },
            direct_candidates=direct.candidates,
            grounded_candidates=(final_batch.candidates if supplied_records else ()),
            provenance={
                "phase": "phase_04f",
                "path": final_batch.source,
                "direct_prompt_sha256": hashlib.sha256(direct.prompt.encode()).hexdigest(),
                "final_prompt_sha256": hashlib.sha256(final_batch.prompt.encode()).hexdigest(),
                "direct_raw_outputs": direct.raw_outputs,
                "final_raw_outputs": final_batch.raw_outputs,
                "direct_seeds": direct.seeds,
                "final_seeds": final_batch.seeds,
                "input_only": external_context is None,
                "pinyin_consumed_by_llm_prompt": False,
                "retrieval": retrieval_diagnostic,
                "upstream_commit": "63f249e711f6501169e6baafec7e12318b3c765b",
            },
        )
        if record_trace and self.interaction_store is not None:
            position = chronological_position or query_id
            trace = InteractionTrace.create(
                user_id=user_id,
                chronological_position=position,
                preceding_text=preceding_text,
                pinyin_or_keystrokes=pinyin_or_keystrokes,
                selected_text=selected_text,
                prediction=result.to_dict(),
                memory_triggered=trigger.should_retrieve,
                retrieved_memory_ids=tuple(item.memory_id for item in retrieved),
                generated_candidates=tuple(item.text for item in final_batch.candidates),
            )
            self.interaction_store.append(trace)
        return result

    @property
    def prediction_count(self) -> int:
        return self._prediction_count
