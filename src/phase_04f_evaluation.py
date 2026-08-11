"""Frozen, separate-layer evaluation for Phase 4F.1 Pinyin integration."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from statistics import mean
from typing import Any, Mapping, Sequence

from .phase_04c_evaluation import compute_metrics
from .reference_backend.backend import ReferencePersonalisedIMEBackend
from .reference_backend.benchmark_adapter import Phase04FBenchmarkAdapter, target_rank
from .reference_backend.pinyin_decoder import PinyinDecoder, normalize_pinyin
from .reference_backend.pinyin_integration import integrate_separate_channels


PHASE_04B6_CHECKSUM = "2d0df837fed3cf6b1a141b9f43677733671cf1f08cb72ca3b9e2f0f2f13f5077"
PHASE_04C_CHECKSUM = "c9a03ae4cdc18bba0facff7bcdd4ec9a0221906859cd001781719d8d646456ff"
PHASE_04D_CHECKSUM = "17c3ef37a416afba87b01de3741cd0c2131b50ad59ef737bdd136c10316d9620"
PHASE_04E_MANIFEST_CHECKSUM = "e0d9bfe875458d7b1fc40001df3017b868cef25b5456aa03c647fe4e2b38d21b"

CONDITIONS = ("generic_no_memory", "correct_user_memory", "wrong_user_memory")
INTEGRATED_CONDITIONS = (
    "pinyin_decoder_only",
    "pinyin_decoder_plus_generic_huoziime",
    "pinyin_decoder_plus_correct_user_huoziime",
    "pinyin_decoder_plus_wrong_user_huoziime",
)


def deterministic_seed(interaction_id: str) -> int:
    return int(hashlib.sha256(f"phase_04f\n{interaction_id}".encode()).hexdigest()[:8], 16) & 0x7FFF_FFFF


def _texts(candidates: Sequence[Any]) -> list[str]:
    return [candidate.text for candidate in candidates]


def _changed(left: Sequence[str], right: Sequence[str]) -> bool:
    return tuple(left) != tuple(right)


def _huoziime_summary(rows: Sequence[dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = [row["huoziime_conditions"][condition] for row in rows]
    triggered = sum(item["memory_triggered"] for item in selected)
    hits = sum(item["retrieval_hit"] for item in selected)
    grounded = sum(item["memory_grounded_generation"] for item in selected)
    direct = sum(item["output_path"] == "direct_generation" for item in selected)
    no_hit_reruns = sum(item["output_path"] == "memory_rerun_no_hit" for item in selected)
    changed = sum(item["personal_memory_changed_output"] for item in selected)
    return {
        "evaluated_count": len(selected),
        "memory_trigger_count": triggered,
        "memory_trigger_rate": triggered / len(selected) if selected else 0.0,
        "successful_retrieval_count": hits,
        "successful_retrieval_rate": hits / len(selected) if selected else 0.0,
        "memory_grounded_generation_count": grounded,
        "memory_grounded_generation_rate": grounded / len(selected) if selected else 0.0,
        "direct_output_count": direct,
        "direct_output_rate": direct / len(selected) if selected else 0.0,
        "no_memory_rerun_output_count": no_hit_reruns,
        "no_memory_rerun_output_rate": no_hit_reruns / len(selected) if selected else 0.0,
        "predictions_materially_changed_by_personal_memory": changed,
        "mean_total_latency_ms": mean(
            item["prediction"]["timing"]["total_ms"] for item in selected
        ) if selected else 0.0,
        "mean_direct_generation_latency_ms": mean(
            item["prediction"]["timing"]["direct_generation_ms"] for item in selected
        ) if selected else 0.0,
        "mean_query_embedding_latency_ms": mean(
            item["prediction"]["timing"]["query_embedding_ms"] for item in selected
        ) if selected else 0.0,
        "mean_hnsw_search_latency_ms": mean(
            item["prediction"]["timing"]["hnsw_search_ms"] for item in selected
        ) if selected else 0.0,
        "mean_grounded_generation_latency_ms": mean(
            item["prediction"]["timing"]["grounded_generation_ms"] for item in selected
        ) if selected else 0.0,
        "retrieved_memory_provenance": [
            {
                "interaction_id": row["interaction_id"],
                "memory_ids": item["prediction"]["supplied_memory_ids"],
                "memories": item["prediction"]["retrieved_memories"],
            }
            for row, item in zip(rows, selected)
            if item["memory_triggered"]
        ],
    }


def _overlap_summary(rows: Sequence[dict[str, Any]], condition: str | None) -> dict[str, Any]:
    overlaps = []
    for row in rows:
        pinyin = set(row["pinyin_conversion"]["candidate_texts"])
        if condition is None:
            ai: set[str] = set()
        else:
            ai = set(row["huoziime_conditions"][condition]["final_suggestion_texts"])
        overlaps.append(len(pinyin & ai))
    return {
        "queries_with_channel_overlap": sum(value > 0 for value in overlaps),
        "mean_overlapping_text_count": mean(overlaps) if overlaps else 0.0,
    }


def evaluate_phase_04f(
    test_records: Sequence[dict[str, Any]],
    *,
    backends: Mapping[str, ReferencePersonalisedIMEBackend],
    pinyin_decoder: PinyinDecoder,
    adapter: Phase04FBenchmarkAdapter | None = None,
) -> dict[str, Any]:
    """Evaluate Pinyin conversion and HuoziIME personalisation as distinct tasks."""
    if set(backends) != set(CONDITIONS):
        raise ValueError(f"exact Phase 4F conditions required: {CONDITIONS}")
    adapter = adapter or Phase04FBenchmarkAdapter()
    ordered = sorted(
        test_records,
        key=lambda record: (
            record["work_date"],
            int(record["source_start_offset"]),
            record["interaction_id"],
        ),
    )
    rows: list[dict[str, Any]] = []
    for record in ordered:
        seed = deterministic_seed(record["interaction_id"])
        canonical_request = adapter.request(
            record,
            personal_state_user_id="generic_no_memory",
            top_k=10,
        )
        decoded = pinyin_decoder.decode(
            canonical_request.pinyin_or_keystrokes,
            top_k=canonical_request.top_k,
        )
        pinyin_texts = _texts(decoded.candidates)
        pinyin_row = {
            "normalized_pinyin": decoded.normalized_pinyin,
            "consumed_input": decoded.consumed_input,
            "candidates": [asdict(candidate) for candidate in decoded.candidates],
            "candidate_texts": pinyin_texts,
            "target_rank": target_rank(canonical_request.target, pinyin_texts),
            "target_present": canonical_request.target in pinyin_texts,
            "latency_ms": decoded.latency_ms,
            "decoder": decoded.decoder,
        }

        condition_rows: dict[str, Any] = {}
        integrated_rows: dict[str, Any] = {
            "pinyin_decoder_only": {
                "channels_unified": False,
                "pinyin_candidate_texts": pinyin_texts,
                "ai_suggestion_texts": [],
                "overlap_texts": [],
            }
        }
        integrated_names = {
            "generic_no_memory": "pinyin_decoder_plus_generic_huoziime",
            "correct_user_memory": "pinyin_decoder_plus_correct_user_huoziime",
            "wrong_user_memory": "pinyin_decoder_plus_wrong_user_huoziime",
        }
        for condition in CONDITIONS:
            backend = backends[condition]
            request = adapter.request(
                record,
                personal_state_user_id=backend.user_id,
                top_k=10,
            )
            if normalize_pinyin(request.pinyin_or_keystrokes) != decoded.consumed_input:
                raise ValueError("Pinyin query changed between personal-memory conditions")
            result = backend.predict(
                request.personal_state_user_id,
                request.preceding_text,
                request.pinyin_or_keystrokes,
                request.top_k,
                external_context=request.external_context,
                seed_base=seed,
                chronological_position=request.chronological_position,
                selected_text=request.target,
                request_nonce=request.benchmark_interaction_id,
                record_trace=False,
            )
            direct_texts = _texts(result.direct_candidates)
            final_texts = _texts(result.candidates)
            grounded_texts = _texts(result.grounded_candidates)
            condition_rows[condition] = {
                "personal_state_user_id": request.personal_state_user_id,
                "memory_triggered": result.memory_trigger.should_retrieve,
                "retrieval_hit": bool(result.supplied_memory_ids),
                "memory_grounded_generation": bool(result.grounded_candidates),
                "output_path": result.provenance["path"],
                "personal_memory_changed_output": (
                    bool(result.supplied_memory_ids) and _changed(direct_texts, final_texts)
                ),
                "direct_suggestion_texts": direct_texts,
                "grounded_suggestion_texts": grounded_texts,
                "final_suggestion_texts": final_texts,
                "prediction": result.to_dict(),
            }
            integrated = integrate_separate_channels(decoded, result)
            integrated_rows[integrated_names[condition]] = {
                "channels_unified": False,
                "integration_mode": integrated.integration_mode,
                "pinyin_candidate_texts": pinyin_texts,
                "ai_suggestion_texts": final_texts,
                "overlap_texts": sorted(set(pinyin_texts) & set(final_texts)),
                "candidate_provenance": [asdict(item) for item in integrated.candidate_provenance],
            }

        rows.append(
            {
                "interaction_id": record["interaction_id"],
                "target_user_id": record["author_id"],
                "work_id": record["work_id"],
                "chronological_position": f"{record['work_date']}|{int(record['source_start_offset']):012d}",
                "preceding_text": canonical_request.preceding_text,
                "pinyin_or_keystrokes": canonical_request.pinyin_or_keystrokes,
                "target": canonical_request.target,
                "external_context": None,
                "seed_base": seed,
                "pinyin_conversion": pinyin_row,
                "huoziime_conditions": condition_rows,
                "integrated_conditions": integrated_rows,
            }
        )

    correct_ids = set(backends["correct_user_memory"].memory_store.source_interaction_ids())
    wrong_ids = set(backends["wrong_user_memory"].memory_store.source_interaction_ids())
    test_ids = {record["interaction_id"] for record in ordered}
    if correct_ids & test_ids or wrong_ids & test_ids:
        raise ValueError("test/future interaction leaked into frozen personal memory")
    if any(
        memory.user_id != backends[condition].user_id
        for condition in CONDITIONS
        for memory in backends[condition].memory_store.list()
    ):
        raise ValueError("cross-user memory contamination")

    pinyin_ranks = [row["pinyin_conversion"]["target_rank"] for row in rows]
    pinyin_metrics = asdict(compute_metrics(pinyin_ranks))
    pinyin_metrics["candidate_coverage"] = (
        sum(rank is not None for rank in pinyin_ranks) / len(pinyin_ranks)
        if pinyin_ranks else 0.0
    )
    pinyin_metrics["mean_decoder_latency_ms"] = mean(
        row["pinyin_conversion"]["latency_ms"] for row in rows
    ) if rows else 0.0

    generic_final = [
        row["huoziime_conditions"]["generic_no_memory"]["final_suggestion_texts"]
        for row in rows
    ]
    correct_final = [
        row["huoziime_conditions"]["correct_user_memory"]["final_suggestion_texts"]
        for row in rows
    ]
    wrong_final = [
        row["huoziime_conditions"]["wrong_user_memory"]["final_suggestion_texts"]
        for row in rows
    ]
    condition_to_huozi = {
        "pinyin_decoder_only": None,
        "pinyin_decoder_plus_generic_huoziime": "generic_no_memory",
        "pinyin_decoder_plus_correct_user_huoziime": "correct_user_memory",
        "pinyin_decoder_plus_wrong_user_huoziime": "wrong_user_memory",
    }
    return {
        "schema_version": 2,
        "phase": "Phase 4F.1 — Pinyin Integration Correction",
        "reproduction_label": "B. Faithful HuoziIME reference-backend adaptation",
        "integration": {
            "mode": "FAITHFUL_DESKTOP_ADAPTATION_SEPARATE_CHANNELS",
            "channels_unified": False,
            "official_numerical_fusion_rule": None,
            "raw_pinyin_supplied_to_huoziime_llm": False,
        },
        "evaluation_layers": {
            "pinyin_conversion": {
                "task": "normalized Pinyin to Chinese candidate ranking",
                "metrics": pinyin_metrics,
            },
            "huoziime_personalisation": {
                "task": "preceding text plus frozen personal memory to contextual suggestions",
                "target_ranking_computed": False,
                "conditions": {
                    condition: _huoziime_summary(rows, condition)
                    for condition in CONDITIONS
                },
                "generic_vs_correct_user_output_difference_count": sum(
                    _changed(generic, correct)
                    for generic, correct in zip(generic_final, correct_final)
                ),
                "correct_user_vs_wrong_user_output_difference_count": sum(
                    _changed(correct, wrong)
                    for correct, wrong in zip(correct_final, wrong_final)
                ),
            },
            "integrated_backend": {
                "conditions": list(INTEGRATED_CONDITIONS),
                "channels_unified": False,
                "unified_top_k_metrics": None,
                "reason": "Official ordinary candidates and AI/GhostText suggestions use separate surfaces; no supported shared score exists.",
                "overlap_diagnostics": {
                    name: _overlap_summary(rows, condition)
                    for name, condition in condition_to_huozi.items()
                },
            },
        },
        "conditions": list(CONDITIONS),
        "integrated_conditions": list(INTEGRATED_CONDITIONS),
        "rows": rows,
        "user_isolation": {
            "correct_user_id": backends["correct_user_memory"].user_id,
            "wrong_user_id": backends["wrong_user_memory"].user_id,
            "generic_user_id": backends["generic_no_memory"].user_id,
            "correct_memory_count": len(backends["correct_user_memory"].memory_store),
            "wrong_memory_count": len(backends["wrong_user_memory"].memory_store),
            "generic_memory_count": len(backends["generic_no_memory"].memory_store),
            "test_memory_overlap_count": len((correct_ids | wrong_ids) & test_ids),
        },
        "frozen_history": True,
        "test_time_memory_updates": False,
        "external_context": None,
        "upstream_experiment_reproduction": {
            "memory_trigger": "NOT REPRODUCIBLE FROM PUBLIC ARTIFACTS: evaluation dataset/results absent",
            "memory_processing": "NOT REPRODUCIBLE FROM PUBLIC ARTIFACTS: evaluation dataset/results absent",
            "retrieval": "NOT REPRODUCIBLE FROM PUBLIC ARTIFACTS: paper memory benchmark absent",
            "memory_grounded_generation": "NOT REPRODUCIBLE FROM PUBLIC ARTIFACTS: paper test examples absent",
            "mobile_latency": "NOT REPRODUCIBLE on non-equivalent desktop hardware",
        },
    }
