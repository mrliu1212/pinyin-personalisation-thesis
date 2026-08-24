from __future__ import annotations

"""PV1 final-list reranking: BGE vs BGE+Recency, with NGramRecency.

This is a focused extension of
``run_initial_pv1_bge_ngram_context_reranking_v1``.  It does not change the
candidate set and it does not rerun recovery.  It reuses the completed V1 BGE
historical embedding cache and plain BGE row scores as read-only inputs, then
adds a semantic-temporal BGERecency scorer.

Methods selected on ALL Train-Val by Macro-author Top1:

    PV1
        S(c) = S_PV1(c)

    BGE
        S(c) = S_PV1(c) + lambda_B * P_BGE(c)

    BGERecency
        S(c) = S_PV1(c) + lambda_B * P_BGR(c)

        For each candidate c, first select the same semantic Top-5 histories by
        cosine similarity (negative cosine is later clamped to zero).  Recency
        is applied only during aggregation, not during Top-5 retrieval:

            raw_BGR(c) = sum_{h in Top5_cos(c)}
                         max(0, cos(E(q), E(h))) * exp(-age(h) / tau_B)

        Then normalize raw non-negative mass across the frozen PV1 Top-10.

    NGramRecency
        S(c) = S_PV1(c) + lambda_N * P_NG(c)

    BGE + NGramRecency
        S(c) = S_PV1(c) + lambda_B * P_BGE(c) + lambda_N * P_NG(c)

    BGERecency + NGramRecency
        S(c) = S_PV1(c) + lambda_B * P_BGR(c) + lambda_N * P_NG(c)

Protocol invariants are inherited from V1:
- same author only;
- strictly prior causal history only;
- H5000 before exact-Pinyin filtering;
- current/future Gold never enters features/scoring;
- Dev3000/Test are untouched;
- frozen PV1 final scores are the base;
- frozen PV1 Top-10 candidate set is unchanged;
- Missing@10 and Recovery@10 must remain invariant.

BGERecency tau defaults to 2048 same-author interactions, matching the selected
NGramRecency temporal scale.  The default weight grids are expanded because
both BGE and NGramRecency previously selected lambda=4 at the old grid edge.

Required prior artifact
-----------------------
Pass --base-bge-root pointing to the completed V1 output directory containing:
    bge_scores.jsonl
    bge_history_embedding_cache.sqlite3
    comparison.json
    run_manifest.json

The prior directory is read-only.  This runner writes only to --output-root.
"""

import argparse
import hashlib
import json
import math
import os
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiments.initial_personalisation import (
    run_initial_pv1_bge_ngram_context_reranking_v1 as base,
)


EXPERIMENT = "initial_pv1_bge_recency_ngram_context_reranking_v2"
SCHEMA_VERSION = 2
DEFAULT_BGE_RECENCY_TAU = 2048.0
DEFAULT_BGE_LAMBDAS = (0.0, 0.25, 0.50, 1.0, 2.0, 4.0, 6.0, 8.0)
DEFAULT_NGRAM_LAMBDAS = (0.0, 0.25, 0.50, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as dst:
        dst.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as dst:
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    base.write_csv(path, rows)


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return base.load_jsonl_by_id(path)


def percentile(values: Sequence[float], q: float) -> float | None:
    return base.percentile(values, q)


def latency_summary(values: Sequence[float]) -> dict[str, Any]:
    return base.latency_summary(values)


def rank_of(ranking: Sequence[str], gold: str) -> int | None:
    return base.rank_of(ranking, gold)


def selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        float(row["macro_author_top1"]),
        float(row["mrr_at_10"]),
        float(row["top3"]),
        -float(row["lambda_bge"] + row["lambda_ngram"]),
        -float(row["lambda_bge"]),
        -float(row["lambda_ngram"]),
        1 if str(row["semantic_mode"]) == "plain" else 0,
    )


def load_prior_bge(args: argparse.Namespace, inputs: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    score_path = args.base_bge_root / "bge_scores.jsonl"
    cache_path = args.base_bge_root / "bge_history_embedding_cache.sqlite3"
    comparison_path = args.base_bge_root / "comparison.json"
    manifest_path = args.base_bge_root / "run_manifest.json"
    for path in (score_path, cache_path, comparison_path, manifest_path):
        if not path.is_file():
            raise RuntimeError(f"Required prior BGE artifact missing: {path}")

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison.get("status") != "complete":
        raise RuntimeError("--base-bge-root is not a completed V1 experiment")
    if comparison.get("experiment") != "initial_pv1_bge_ngram_context_reranking_v1":
        raise RuntimeError(
            "--base-bge-root comparison.json is not initial_pv1_bge_ngram_context_reranking_v1"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior_hashes = manifest.get("input_hashes", {})
    if prior_hashes != inputs["input_hashes"]:
        raise RuntimeError("Prior V1 BGE input hashes differ from current frozen inputs")
    if int(manifest.get("bge_context_chars", -1)) != base.BGE_CONTEXT_CHARS:
        raise RuntimeError("Prior V1 BGE context length differs")
    if int(manifest.get("bge_top_n", -1)) != base.BGE_TOP_N:
        raise RuntimeError("Prior V1 BGE Top-N differs")

    prior_scores = load_jsonl_by_id(score_path)
    if len(prior_scores) != base.EXPECTED_VAL_ROWS:
        raise RuntimeError(
            f"Prior V1 BGE score cache incomplete: {len(prior_scores)}/{base.EXPECTED_VAL_ROWS}"
        )
    for i, rid in enumerate(inputs["row_ids"]):
        row = prior_scores[rid]
        expected_candidates = list(inputs["baseline_rankings"][i])
        if row.get("candidates") != expected_candidates:
            raise RuntimeError(f"Prior V1 BGE candidate mismatch at {rid}")
        if len(row.get("support", [])) != len(expected_candidates):
            raise RuntimeError(f"Prior V1 BGE support length mismatch at {rid}")

    provenance = {
        "root": str(args.base_bge_root.resolve()),
        "bge_scores_path": str(score_path.resolve()),
        "bge_scores_sha256": base.sha256_file(score_path),
        "history_cache_path": str(cache_path.resolve()),
        "comparison_path": str(comparison_path.resolve()),
        "comparison_sha256": base.sha256_file(comparison_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": base.sha256_file(manifest_path),
    }
    return prior_scores, provenance


def load_all_history_vectors(cache_path: Path) -> dict[str, np.ndarray]:
    connection = sqlite3.connect(cache_path)
    vectors: dict[str, np.ndarray] = {}
    try:
        for context, dim, blob in connection.execute("SELECT context, dim, vector FROM embeddings"):
            vector = np.frombuffer(blob, dtype=np.float32).copy()
            if vector.size != int(dim):
                raise RuntimeError(f"Corrupt BGE vector cache row: {hashlib.sha256(str(context).encode()).hexdigest()}")
            vectors[str(context)] = vector
    finally:
        connection.close()
    return vectors


def bge_recency_candidate_support(
    *,
    query_vector: np.ndarray,
    candidates: Sequence[str],
    visible: Sequence[base.VisibleHistory],
    history_vectors: Mapping[str, np.ndarray],
    tau: float,
) -> tuple[list[float], dict[str, Any]]:
    """Top-5 is selected by cosine only; recency changes aggregation only."""

    grouped: dict[str, list[tuple[np.ndarray, int, str]]] = {
        candidate: [] for candidate in candidates
    }
    counts = {candidate: 0 for candidate in candidates}
    for item in visible:
        target = item.record.target
        if target not in grouped:
            continue
        key = item.record.context[-base.BGE_CONTEXT_CHARS :]
        vector = history_vectors.get(key)
        if vector is None:
            raise KeyError(
                "Missing prior V1 BGE history vector for context hash="
                + hashlib.sha256(key.encode("utf-8")).hexdigest()
            )
        grouped[target].append((vector, int(item.age), str(item.record.row_id)))
        counts[target] += 1

    raw_recency: list[float] = []
    raw_plain_check: list[float] = []
    touched = 0
    candidates_with_history = 0
    selected_history_rows = 0
    selected_ages: list[int] = []

    for candidate in candidates:
        values = grouped[candidate]
        touched += len(values)
        if not values:
            raw_recency.append(0.0)
            raw_plain_check.append(0.0)
            continue
        candidates_with_history += 1
        matrix = np.vstack([value[0] for value in values])
        similarities = matrix @ query_vector

        # Reconstruct old plain BGE semantics exactly enough for an audit:
        # Top-N values by cosine, then positive clamp and sum.
        if similarities.size > base.BGE_TOP_N:
            plain_top = np.partition(similarities, -base.BGE_TOP_N)[-base.BGE_TOP_N :]
        else:
            plain_top = similarities
        raw_plain_check.append(float(np.maximum(plain_top, 0.0).sum()))

        # New method: retrieval stays semantic-only.  Sorting uses row_id only as
        # a deterministic tie break, so recency does not decide Top-5 membership.
        order = sorted(
            range(len(values)),
            key=lambda i: (-float(similarities[i]), values[i][2]),
        )[: base.BGE_TOP_N]
        selected_history_rows += len(order)
        score = 0.0
        for i in order:
            similarity = max(0.0, float(similarities[i]))
            age = int(values[i][1])
            selected_ages.append(age)
            score += similarity * base.recency_weight(age, tau)
        raw_recency.append(score)

    return base.normalize_nonnegative(raw_recency), {
        "plain_support_check": base.normalize_nonnegative(raw_plain_check),
        "history_vectors_touched": touched,
        "candidates_with_history": candidates_with_history,
        "candidate_history_counts": [int(counts[c]) for c in candidates],
        "selected_history_rows": selected_history_rows,
        "selected_age_mean": statistics.fmean(selected_ages) if selected_ages else None,
        "selected_age_p50": percentile(selected_ages, 0.50),
        "selected_age_p95": percentile(selected_ages, 0.95),
        "raw_recency_mass": float(sum(max(0.0, value) for value in raw_recency)),
    }


def ensure_manifest(
    args: argparse.Namespace,
    inputs: Mapping[str, Any],
    prior_provenance: Mapping[str, Any],
) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "run_manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "input_hashes": dict(inputs["input_hashes"]),
        "base_bge_provenance": dict(prior_provenance),
        "bge_model": str(args.bge_model.resolve()),
        "bge_model_sha256": base.sha256_file(args.bge_model),
        "bge_context_chars": base.BGE_CONTEXT_CHARS,
        "bge_top_n": base.BGE_TOP_N,
        "bge_recency_tau": args.bge_recency_tau,
        "bge_recency_retrieval": "Top-5 by cosine only; recency applied only during aggregation",
        "hard_max_n": args.hard_max_n,
        "hard_tau": args.hard_tau,
        "lambda_bge_grid": list(args.lambda_bge),
        "lambda_ngram_grid": list(args.lambda_ngram),
        "gold_used_for_scoring_features": False,
        "dev3000_used": False,
        "test_used": False,
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(
                "Existing output directory has a different run_manifest.json; use a new versioned output path."
            )
    else:
        write_json(manifest_path, manifest)
    comparison_path = args.output_root / "comparison.json"
    if comparison_path.is_file():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        if comparison.get("status") == "complete":
            raise RuntimeError("This output directory already contains a completed experiment.")


def run_bge_recency_scoring(
    args: argparse.Namespace,
    inputs: Mapping[str, Any],
    prior_bge_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    base.ensure_cuda_path(args.cuda_path)
    from src.personalisation.pilot_a import BGEContextEmbedder

    prior_cache = args.base_bge_root / "bge_history_embedding_cache.sqlite3"
    print("\n=== LOAD READ-ONLY V1 BGE HISTORY CACHE ===", flush=True)
    history_vectors = load_all_history_vectors(prior_cache)
    print(f"History vectors loaded: {len(history_vectors)}", flush=True)

    score_path = args.output_root / "bge_recency_scores.jsonl"
    completed = load_jsonl_by_id(score_path)
    pending = [rid for rid in inputs["row_ids"] if rid not in completed]
    print("\n=== BGE64 + HISTORY RECENCY ONLINE SCORING ===", flush=True)
    print(f"tau_B: {args.bge_recency_tau:g}", flush=True)
    print("Top-5 retrieval: cosine only", flush=True)
    print("Aggregation: max(0, cosine) * exp(-age/tau_B)", flush=True)
    print(f"completed/resumable rows: {len(completed)}", flush=True)
    print(f"pending rows: {len(pending)}", flush=True)

    embedder = BGEContextEmbedder(args.bge_model)
    warm = base.bge_context_of(inputs["val_by_id"][inputs["row_ids"][0]]) or "测试"
    _ = embedder.embed(warm)

    query_embed_ms: list[float] = [float(row["query_embed_ms"]) for row in completed.values()]
    support_ms: list[float] = [float(row["support_ms"]) for row in completed.values()]
    online_ms: list[float] = [float(row["online_total_ms"]) for row in completed.values()]
    plain_audit_max_abs = 0.0
    run_started = time.perf_counter()

    with score_path.open("a", encoding="utf-8", newline="\n") as destination:
        with base.NativeStderrSilencer():
            for number, rid in enumerate(pending, 1):
                i = row_index_by_id[rid]
                vrow = inputs["val_by_id"][rid]
                candidates = inputs["baseline_rankings"][i]
                visible = inputs["history_index"].visible(
                    author=str(vrow["author"]),
                    position=int(vrow["chronological_position"]),
                    pinyin=base.pinyin_of(vrow),
                )
                qcontext = base.bge_context_of(vrow)
                total_t0 = time.perf_counter()
                embed_t0 = time.perf_counter()
                qvector = base.normalized_vector(embedder.embed(qcontext))
                embed_elapsed = (time.perf_counter() - embed_t0) * 1000.0
                support_t0 = time.perf_counter()
                support, diag = bge_recency_candidate_support(
                    query_vector=qvector,
                    candidates=candidates,
                    visible=visible,
                    history_vectors=history_vectors,
                    tau=args.bge_recency_tau,
                )
                support_elapsed = (time.perf_counter() - support_t0) * 1000.0
                total_elapsed = (time.perf_counter() - total_t0) * 1000.0

                old_plain = [float(v) for v in prior_bge_rows[rid]["support"]]
                plain_check = [float(v) for v in diag.pop("plain_support_check")]
                if len(old_plain) != len(plain_check):
                    raise RuntimeError(f"Plain BGE audit length mismatch at {rid}")
                max_abs = max((abs(a - b) for a, b in zip(old_plain, plain_check)), default=0.0)
                plain_audit_max_abs = max(plain_audit_max_abs, max_abs)
                if max_abs > 1e-5:
                    raise RuntimeError(
                        f"Plain BGE reconstruction drift at {rid}: max_abs={max_abs:.8g}"
                    )

                result = {
                    "schema_version": SCHEMA_VERSION,
                    "row_id": rid,
                    "author": str(vrow["author"]),
                    "context_chars": base.BGE_CONTEXT_CHARS,
                    "actual_context_chars": len(qcontext),
                    "top_n": base.BGE_TOP_N,
                    "tau": args.bge_recency_tau,
                    "candidates": list(candidates),
                    "support": [float(v) for v in support],
                    "plain_bge_audit_max_abs": max_abs,
                    "visible_same_pinyin_history_count": len(visible),
                    "query_embed_ms": embed_elapsed,
                    "support_ms": support_elapsed,
                    "online_total_ms": total_elapsed,
                    **diag,
                    "gold_used_for_scoring": False,
                    "dev3000_used": False,
                    "test_used": False,
                }
                destination.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                query_embed_ms.append(embed_elapsed)
                support_ms.append(support_elapsed)
                online_ms.append(total_elapsed)
                if number % args.progress_every == 0 or number == len(pending):
                    destination.flush()
                    wall = time.perf_counter() - run_started
                    recent = max(1, min(len(online_ms), 1000))
                    print(
                        f"BGERecency score: {number}/{len(pending)} pending; "
                        f"rate={number / wall:.2f} rows/s; "
                        f"embed={statistics.fmean(query_embed_ms[-recent:]):.3f} ms; "
                        f"online={statistics.fmean(online_ms[-recent:]):.3f} ms",
                        flush=True,
                    )

    final = load_jsonl_by_id(score_path)
    if len(final) != base.EXPECTED_VAL_ROWS:
        raise RuntimeError(f"Incomplete BGERecency score cache: {len(final)}/{base.EXPECTED_VAL_ROWS}")
    for i, rid in enumerate(inputs["row_ids"]):
        row = final[rid]
        expected_candidates = list(inputs["baseline_rankings"][i])
        if row.get("candidates") != expected_candidates:
            raise RuntimeError(f"BGERecency candidate mismatch at {rid}")
        if len(row.get("support", [])) != len(expected_candidates):
            raise RuntimeError(f"BGERecency support length mismatch at {rid}")

    all_plain_audit = [float(row.get("plain_bge_audit_max_abs", 0.0)) for row in final.values()]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "rows": len(final),
        "context_chars": base.BGE_CONTEXT_CHARS,
        "top_n_per_candidate": base.BGE_TOP_N,
        "tau": args.bge_recency_tau,
        "retrieval": "Top-5 by cosine only",
        "aggregation": "max(0, cosine) * exp(-age/tau)",
        "plain_bge_reconstruction_max_abs": max(all_plain_audit, default=0.0),
        "query_embedding_latency_ms": latency_summary([float(row["query_embed_ms"]) for row in final.values()]),
        "support_latency_ms": latency_summary([float(row["support_ms"]) for row in final.values()]),
        "online_total_latency_ms": latency_summary([float(row["online_total_ms"]) for row in final.values()]),
        "score_cache_sha256": base.sha256_file(score_path),
        "note": "Latency is valid only when the machine is not under competing experimental load.",
        "gold_used_for_scoring": False,
        "dev3000_used": False,
        "test_used": False,
    }
    write_json(args.output_root / "bge_recency_scoring_summary.json", summary)
    return final


def semantic_rerank(
    candidates: Sequence[str],
    pv1_scores: Sequence[float],
    semantic_support: Sequence[float],
    ngram_support: Sequence[float],
    *,
    lambda_bge: float,
    lambda_ngram: float,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    return base.rerank_joint(
        candidates,
        pv1_scores,
        semantic_support,
        ngram_support,
        lambda_bge=lambda_bge,
        lambda_ngram=lambda_ngram,
    )


def build_grid_row(
    *,
    semantic_mode: str,
    lb: float,
    ln: float,
    ranks: Sequence[int | None],
    inputs: Mapping[str, Any],
    opportunity_mask: Sequence[bool],
    pv1_rescue_vs_f_mask: Sequence[bool],
    pv1_harm_vs_f_mask: Sequence[bool],
    baseline_rec10: float,
) -> dict[str, Any]:
    authors = inputs["authors"]
    baseline_ranks = inputs["baseline_ranks"]
    conflict_mask = inputs["conflict_mask"]
    recovery_mask = inputs["recovery_mask"]
    metrics = base.author_metrics(authors, ranks)
    transition = base.transition_counts(baseline_ranks, ranks)
    conflict_transition = base.transition_counts(baseline_ranks, ranks, conflict_mask)
    rec = base.recovery_metrics(ranks, recovery_mask)
    opportunity = base.rank_metrics([rank for rank, include in zip(ranks, opportunity_mask) if include])
    movement = base.rank_movement(baseline_ranks, ranks, opportunity_mask)
    rescue_den = sum(pv1_rescue_vs_f_mask)
    harm_den = sum(pv1_harm_vs_f_mask)
    rescue_retained = sum(include and rank == 1 for include, rank in zip(pv1_rescue_vs_f_mask, ranks))
    harm_repaired = sum(include and rank == 1 for include, rank in zip(pv1_harm_vs_f_mask, ranks))
    semantic_label = "BGE" if semantic_mode == "plain" else "BGERecency"
    family = (
        "PV1" if lb == 0 and ln == 0 else
        semantic_label if lb > 0 and ln == 0 else
        "NGramRecency" if lb == 0 and ln > 0 else
        f"{semantic_label}+NGramRecency"
    )
    row = {
        "semantic_mode": semantic_mode,
        "family": family,
        "lambda_bge": lb,
        "lambda_ngram": ln,
        "macro_author_top1": metrics["macro_author"]["top1"],
        "micro_top1": metrics["micro"]["top1"],
        "top3": metrics["micro"]["top3"],
        "top5": metrics["micro"]["top5"],
        "mrr_at_10": metrics["micro"]["mrr_at_10"],
        "missing10": metrics["micro"]["missing10"],
        "mean_rank_given_top10": metrics["micro"]["mean_rank_given_top10"],
        "rescue": transition.get("rescue", 0),
        "harm": transition.get("harm", 0),
        "net": transition.get("net", 0),
        "conflict_rescue": conflict_transition.get("rescue", 0),
        "conflict_harm": conflict_transition.get("harm", 0),
        "conflict_net": conflict_transition.get("net", 0),
        "pv1_rescue_vs_f_n": rescue_den,
        "pv1_rescue_retained_n": rescue_retained,
        "pv1_rescue_retention_rate": rescue_retained / rescue_den if rescue_den else 0.0,
        "pv1_harm_vs_f_n": harm_den,
        "pv1_harm_repaired_n": harm_repaired,
        "pv1_harm_repair_rate": harm_repaired / harm_den if harm_den else 0.0,
        "opportunity_top1": opportunity["top1"],
        "opportunity_top3": opportunity["top3"],
        "opportunity_top5": opportunity["top5"],
        "opportunity_mrr_at_10": opportunity["mrr_at_10"],
        "rank_improved_n": movement["rank_improved_n"],
        "rank_worsened_n": movement["rank_worsened_n"],
        "rank_unchanged_n": movement["rank_unchanged_n"],
        "mean_rank_gain": movement["mean_rank_gain"],
        "median_rank_gain": movement["median_rank_gain"],
        "rec1": rec["rec1"],
        "rec3": rec["rec3"],
        "rec5": rec["rec5"],
        "rec10": rec["rec10"],
        "recovery_mrr_at_10": rec["recovery_mrr_at_10"],
    }
    if abs(float(row["missing10"]) - float(inputs["baseline_metrics"]["micro"]["missing10"])) > 1e-15:
        raise RuntimeError(f"Missing@10 changed: mode={semantic_mode}, BGE={lb}, NGram={ln}")
    if abs(float(row["rec10"]) - float(baseline_rec10)) > 1e-15:
        raise RuntimeError(f"Recovery@10 changed: mode={semantic_mode}, BGE={lb}, NGram={ln}")
    return row


def evaluate(
    args: argparse.Namespace,
    inputs: Mapping[str, Any],
    prior_bge_rows: Mapping[str, Mapping[str, Any]],
    bge_recency_rows: Mapping[str, Mapping[str, Any]],
    prior_provenance: Mapping[str, Any],
) -> None:
    started = time.perf_counter()
    row_ids = inputs["row_ids"]
    authors = inputs["authors"]
    golds = inputs["golds"]
    baseline_rankings = inputs["baseline_rankings"]
    baseline_scores = inputs["baseline_score_vectors"]
    baseline_ranks = inputs["baseline_ranks"]
    frequency_ranks = inputs["frequency_ranks"]
    recovery_mask = inputs["recovery_mask"]
    conflict_mask = inputs["conflict_mask"]
    ambiguous_mask = inputs["ambiguous_mask"]
    history_available_mask = inputs["history_available_mask"]
    generic_missing_mask = inputs["generic_missing_mask"]

    modes = ("plain", "recency")
    rank_surface: dict[tuple[str, float, float], list[int | None]] = {
        (mode, lb, ln): []
        for mode in modes
        for lb in args.lambda_bge
        for ln in args.lambda_ngram
    }
    plain_supports: list[list[float]] = []
    recency_supports: list[list[float]] = []
    ngram_supports: list[list[float]] = []
    ngram_effective_n = Counter()
    ngram_matched_rows: list[int] = []
    visible_history_rows: list[int] = []
    ngram_latency_ms: list[float] = []

    for i, rid in enumerate(row_ids):
        vrow = inputs["val_by_id"][rid]
        candidates = baseline_rankings[i]
        visible = inputs["history_index"].visible(
            author=str(vrow["author"]),
            position=int(vrow["chronological_position"]),
            pinyin=base.pinyin_of(vrow),
        )
        visible_history_rows.append(len(visible))
        t0 = time.perf_counter()
        ng_support, ng_diag = base.hard_backoff_ngram_recency(
            candidates=candidates,
            query_context=base.context_of(vrow),
            visible=visible,
            max_n=args.hard_max_n,
            tau=args.hard_tau,
        )
        ngram_latency_ms.append((time.perf_counter() - t0) * 1000.0)
        ngram_effective_n[int(ng_diag["effective_n"])] += 1
        ngram_matched_rows.append(int(ng_diag["matched_history_rows"]))

        plain = [float(v) for v in prior_bge_rows[rid]["support"]]
        recency = [float(v) for v in bge_recency_rows[rid]["support"]]
        if not (len(plain) == len(recency) == len(candidates)):
            raise RuntimeError(f"Support length mismatch at {rid}")
        plain_supports.append(plain)
        recency_supports.append(recency)
        ngram_supports.append(ng_support)

        semantic_by_mode = {"plain": plain, "recency": recency}
        for mode in modes:
            semantic = semantic_by_mode[mode]
            for lb in args.lambda_bge:
                for ln in args.lambda_ngram:
                    ranking, _ = semantic_rerank(
                        candidates,
                        baseline_scores[i],
                        semantic,
                        ng_support,
                        lambda_bge=lb,
                        lambda_ngram=ln,
                    )
                    if set(ranking) != set(candidates) or len(ranking) != len(candidates):
                        raise RuntimeError(f"Candidate-set invariant failed at {rid}")
                    rank = rank_of(ranking, golds[i])
                    if (rank is None) != (baseline_ranks[i] is None):
                        raise RuntimeError(f"Missing invariant failed at {rid}")
                    rank_surface[(mode, lb, ln)].append(rank)

        if (i + 1) % args.progress_every == 0 or i + 1 == len(row_ids):
            print(f"Expanded BGE/BGERecency + NGram grid: {i + 1}/{len(row_ids)}", flush=True)

    for mode in modes:
        if rank_surface[(mode, 0.0, 0.0)] != baseline_ranks:
            raise RuntimeError(f"Zero-weight control failed for semantic mode={mode}")

    opportunity_mask = [rank is not None for rank in baseline_ranks]
    pv1_wrong_opportunity_mask = [rank is not None and rank != 1 for rank in baseline_ranks]
    pv1_correct_mask = [rank == 1 for rank in baseline_ranks]
    non_conflict_mask = [not value for value in conflict_mask]
    pv1_rescue_vs_f_mask = [f != 1 and p == 1 for f, p in zip(frequency_ranks, baseline_ranks)]
    pv1_harm_vs_f_mask = [f == 1 and p != 1 for f, p in zip(frequency_ranks, baseline_ranks)]
    baseline_rec10 = base.recovery_metrics(baseline_ranks, recovery_mask)["rec10"]

    grid_rows: list[dict[str, Any]] = []
    for mode in modes:
        for lb in args.lambda_bge:
            for ln in args.lambda_ngram:
                grid_rows.append(
                    build_grid_row(
                        semantic_mode=mode,
                        lb=lb,
                        ln=ln,
                        ranks=rank_surface[(mode, lb, ln)],
                        inputs=inputs,
                        opportunity_mask=opportunity_mask,
                        pv1_rescue_vs_f_mask=pv1_rescue_vs_f_mask,
                        pv1_harm_vs_f_mask=pv1_harm_vs_f_mask,
                        baseline_rec10=float(baseline_rec10),
                    )
                )

    plain_rows = [row for row in grid_rows if row["semantic_mode"] == "plain"]
    recency_rows = [row for row in grid_rows if row["semantic_mode"] == "recency"]
    pv1 = next(row for row in plain_rows if float(row["lambda_bge"]) == 0 and float(row["lambda_ngram"]) == 0)
    bge = max(
        [row for row in plain_rows if float(row["lambda_bge"]) > 0 and float(row["lambda_ngram"]) == 0],
        key=selection_key,
    )
    bge_recency = max(
        [row for row in recency_rows if float(row["lambda_bge"]) > 0 and float(row["lambda_ngram"]) == 0],
        key=selection_key,
    )
    ngram = max(
        [row for row in plain_rows if float(row["lambda_bge"]) == 0 and float(row["lambda_ngram"]) > 0],
        key=selection_key,
    )
    joint_plain = max(
        [row for row in plain_rows if float(row["lambda_bge"]) > 0 and float(row["lambda_ngram"]) > 0],
        key=selection_key,
    )
    joint_recency = max(
        [row for row in recency_rows if float(row["lambda_bge"]) > 0 and float(row["lambda_ngram"]) > 0],
        key=selection_key,
    )
    global_best = max(grid_rows, key=selection_key)

    selections = {
        "PV1": pv1,
        "BGE": bge,
        "BGERecency": bge_recency,
        "NGramRecency": ngram,
        "BGE+NGramRecency": joint_plain,
        "BGERecency+NGramRecency": joint_recency,
        "GlobalBest": global_best,
    }

    selected_specs = {
        "PV1": ("plain", 0.0, 0.0),
        "BGE": ("plain", float(bge["lambda_bge"]), 0.0),
        "BGERecency": ("recency", float(bge_recency["lambda_bge"]), 0.0),
        "NGramRecency": ("plain", 0.0, float(ngram["lambda_ngram"])),
        "BGE+NGramRecency": (
            "plain", float(joint_plain["lambda_bge"]), float(joint_plain["lambda_ngram"])
        ),
        "BGERecency+NGramRecency": (
            "recency", float(joint_recency["lambda_bge"]), float(joint_recency["lambda_ngram"])
        ),
    }
    selected_rankings = {
        method: rank_surface[(mode, lb, ln)]
        for method, (mode, lb, ln) in selected_specs.items()
    }

    subset_masks = {
        "overall": [True] * len(row_ids),
        "history_available": history_available_mask,
        "ambiguous": ambiguous_mask,
        "conflict": conflict_mask,
        "non_conflict": non_conflict_mask,
        "generic_missing": generic_missing_mask,
        "context_opportunity_gold_in_pv1_top10": opportunity_mask,
        "pv1_wrong_but_gold_in_top10": pv1_wrong_opportunity_mask,
        "pv1_correct_top1": pv1_correct_mask,
        "pv1_rescue_vs_frequency": pv1_rescue_vs_f_mask,
        "pv1_harm_vs_frequency": pv1_harm_vs_f_mask,
        "recovery_k5": recovery_mask,
    }

    selected_method_rows: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []
    per_author_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []

    for method, ranks in selected_rankings.items():
        mode, lb, ln = selected_specs[method]
        metrics = base.author_metrics(authors, ranks)
        selection_row = selections[method]
        selected_method_rows.append({
            "method": method,
            "semantic_mode": mode,
            "lambda_bge": lb,
            "lambda_ngram": ln,
            "macro_author_top1": metrics["macro_author"]["top1"],
            "macro_author_top3": metrics["macro_author"]["top3"],
            "macro_author_top5": metrics["macro_author"]["top5"],
            "macro_author_mrr_at_10": metrics["macro_author"]["mrr_at_10"],
            "macro_author_missing10": metrics["macro_author"]["missing10"],
            "micro_top1": metrics["micro"]["top1"],
            "top3": metrics["micro"]["top3"],
            "top5": metrics["micro"]["top5"],
            "mrr_at_10": metrics["micro"]["mrr_at_10"],
            "missing10": metrics["micro"]["missing10"],
            "pv1_rescue_vs_f_n": selection_row["pv1_rescue_vs_f_n"],
            "pv1_rescue_retained_n": selection_row["pv1_rescue_retained_n"],
            "pv1_rescue_retention_rate": selection_row["pv1_rescue_retention_rate"],
            "pv1_harm_vs_f_n": selection_row["pv1_harm_vs_f_n"],
            "pv1_harm_repaired_n": selection_row["pv1_harm_repaired_n"],
            "pv1_harm_repair_rate": selection_row["pv1_harm_repair_rate"],
            **base.transition_counts(baseline_ranks, ranks),
        })
        recovery_rows.append({
            "method": method,
            "semantic_mode": mode,
            "lambda_bge": lb,
            "lambda_ngram": ln,
            **base.recovery_metrics(ranks, recovery_mask),
        })
        transition_rows.append({
            "method": method,
            "vs": "PV1",
            **base.transition_counts(baseline_ranks, ranks),
            **base.rank_movement(baseline_ranks, ranks, opportunity_mask),
        })
        for subset, mask in subset_masks.items():
            chosen_ranks = [rank for rank, include in zip(ranks, mask) if include]
            chosen_authors = [author for author, include in zip(authors, mask) if include]
            m = base.author_metrics(chosen_authors, chosen_ranks)
            subset_rows.append({
                "method": method,
                "subset": subset,
                "n": len(chosen_ranks),
                "macro_author_top1": m["macro_author"]["top1"],
                "micro_top1": m["micro"]["top1"],
                "top3": m["micro"]["top3"],
                "top5": m["micro"]["top5"],
                "mrr_at_10": m["micro"]["mrr_at_10"],
                "missing10": m["micro"]["missing10"],
            })
        for author, m in metrics["per_author"].items():
            per_author_rows.append({
                "method": method,
                "author": author,
                "n": m["n"],
                "top1": m["top1"],
                "top3": m["top3"],
                "top5": m["top5"],
                "mrr_at_10": m["mrr_at_10"],
                "missing10": m["missing10"],
            })

    # Direct ablations that matter for interpretation.
    transition_rows.append({
        "method": "BGERecency",
        "vs": "BGE",
        **base.transition_counts(selected_rankings["BGE"], selected_rankings["BGERecency"]),
        **base.rank_movement(selected_rankings["BGE"], selected_rankings["BGERecency"], opportunity_mask),
    })
    transition_rows.append({
        "method": "BGERecency+NGramRecency",
        "vs": "BGE+NGramRecency",
        **base.transition_counts(
            selected_rankings["BGE+NGramRecency"],
            selected_rankings["BGERecency+NGramRecency"],
        ),
        **base.rank_movement(
            selected_rankings["BGE+NGramRecency"],
            selected_rankings["BGERecency+NGramRecency"],
            opportunity_mask,
        ),
    })
    transition_rows.append({
        "method": "BGERecency+NGramRecency",
        "vs": "NGramRecency",
        **base.transition_counts(
            selected_rankings["NGramRecency"],
            selected_rankings["BGERecency+NGramRecency"],
        ),
        **base.rank_movement(
            selected_rankings["NGramRecency"],
            selected_rankings["BGERecency+NGramRecency"],
            opportunity_mask,
        ),
    })

    # Selected row-level traces.
    prediction_rows: list[dict[str, Any]] = []
    for i, rid in enumerate(row_ids):
        candidates = baseline_rankings[i]
        pv1_scores = baseline_scores[i]
        semantic_by_mode = {"plain": plain_supports[i], "recency": recency_supports[i]}
        ng = ngram_supports[i]
        methods: dict[str, Any] = {}
        for method, (mode, lb, ln) in selected_specs.items():
            sem = semantic_by_mode[mode]
            ranking, final_scores = semantic_rerank(
                candidates,
                pv1_scores,
                sem,
                ng,
                lambda_bge=lb,
                lambda_ngram=ln,
            )
            final_by_candidate = {candidate: float(score) for candidate, score in zip(ranking, final_scores)}
            candidate_rows = []
            for new_rank, candidate in enumerate(ranking, 1):
                old_i = candidates.index(candidate)
                candidate_rows.append({
                    "candidate": candidate,
                    "pv1_rank": old_i + 1,
                    "reranked_rank": new_rank,
                    "pv1_final_score": float(pv1_scores[old_i]),
                    "plain_bge_support": float(plain_supports[i][old_i]),
                    "bge_recency_support": float(recency_supports[i][old_i]),
                    "ngram_recency_support": float(ng[old_i]),
                    "semantic_contribution": lb * float(sem[old_i]),
                    "ngram_contribution": ln * float(ng[old_i]),
                    "reranked_final_score": final_by_candidate[candidate],
                    "source": str(inputs["baseline_candidate_rows"][i][old_i].get("source", "")),
                })
            methods[method] = {
                "semantic_mode": mode,
                "lambda_bge": lb,
                "lambda_ngram": ln,
                "gold_rank": rank_of(ranking, golds[i]),
                "candidates": candidate_rows,
            }
        prediction_rows.append({
            "schema_version": SCHEMA_VERSION,
            "row_id": rid,
            "author": authors[i],
            "gold": golds[i],
            "pv1_gold_rank": baseline_ranks[i],
            "frequency_gold_rank": frequency_ranks[i],
            "conflict": conflict_mask[i],
            "ambiguous": ambiguous_mask[i],
            "generic_missing": generic_missing_mask[i],
            "recovery_k5": recovery_mask[i],
            "methods": methods,
            "gold_used_for_scoring_features": False,
            "dev3000_used": False,
            "test_used": False,
        })

    bgr_summary = json.loads(
        (args.output_root / "bge_recency_scoring_summary.json").read_text(encoding="utf-8")
    )
    bgr_touched = [int(bge_recency_rows[rid].get("history_vectors_touched", 0)) for rid in row_ids]
    bgr_with_history = [int(bge_recency_rows[rid].get("candidates_with_history", 0)) for rid in row_ids]
    context_diagnostics = {
        "same_pinyin_history": {
            "mean_rows": statistics.fmean(visible_history_rows),
            "p50_rows": percentile(visible_history_rows, 0.50),
            "p95_rows": percentile(visible_history_rows, 0.95),
            "max_rows": max(visible_history_rows),
        },
        "bge64_plain": {
            "source": "read-only completed V1 bge_scores.jsonl",
            "context_chars": base.BGE_CONTEXT_CHARS,
            "top_n_history_per_candidate": base.BGE_TOP_N,
        },
        "bge64_recency": {
            "context_chars": base.BGE_CONTEXT_CHARS,
            "top_n_history_per_candidate": base.BGE_TOP_N,
            "tau": args.bge_recency_tau,
            "retrieval": "Top-5 by cosine only",
            "aggregation": "max(0, cosine) * exp(-age/tau)",
            "mean_history_vectors_touched": statistics.fmean(bgr_touched),
            "p95_history_vectors_touched": percentile(bgr_touched, 0.95),
            "mean_candidates_with_history": statistics.fmean(bgr_with_history),
            "rows_with_any_candidate_history": sum(v > 0 for v in bgr_with_history),
            "plain_bge_reconstruction_max_abs": bgr_summary["plain_bge_reconstruction_max_abs"],
        },
        "ngram_recency": {
            "max_n": args.hard_max_n,
            "tau": args.hard_tau,
            "effective_n_distribution": dict(sorted(ngram_effective_n.items())),
            "mean_matched_history_rows": statistics.fmean(ngram_matched_rows),
            "p50_matched_history_rows": percentile(ngram_matched_rows, 0.50),
            "p95_matched_history_rows": percentile(ngram_matched_rows, 0.95),
        },
    }

    fusion_latency: dict[str, Any] = {}
    for method, (mode, lb, ln) in selected_specs.items():
        sem_supports = plain_supports if mode == "plain" else recency_supports
        values: list[float] = []
        for i, candidates in enumerate(baseline_rankings):
            t0 = time.perf_counter()
            semantic_rerank(
                candidates,
                baseline_scores[i],
                sem_supports[i],
                ngram_supports[i],
                lambda_bge=lb,
                lambda_ngram=ln,
            )
            values.append((time.perf_counter() - t0) * 1000.0)
        fusion_latency[method] = latency_summary(values)

    prior_bge_online = [float(prior_bge_rows[rid]["online_total_ms"]) for rid in row_ids]
    latency = {
        "BGE64_plain_from_V1": {
            "online_total": latency_summary(prior_bge_online),
            "note": "Measured in the completed V1 run, not re-measured here.",
        },
        "BGE64Recency": {
            "query_embedding": bgr_summary["query_embedding_latency_ms"],
            "support": bgr_summary["support_latency_ms"],
            "online_total": bgr_summary["online_total_latency_ms"],
        },
        "NGramRecency": {
            "scorer_only_history_lookup_excluded": latency_summary(ngram_latency_ms),
        },
        "fusion_sort_only": fusion_latency,
        "warning": "Do not use latency from a run performed concurrently with another CPU/GPU-heavy experiment for thesis comparison.",
    }

    grid_path = args.output_root / "grid_results.csv"
    selection_path = args.output_root / "family_selection.json"
    method_path = args.output_root / "selected_method_metrics.csv"
    subset_path = args.output_root / "selected_subset_metrics.csv"
    author_path = args.output_root / "per_author_metrics.csv"
    recovery_path = args.output_root / "recovery_metrics.csv"
    transition_path = args.output_root / "rank_transition_metrics.csv"
    prediction_path = args.output_root / "selected_predictions.jsonl"
    diagnostic_path = args.output_root / "context_diagnostics.json"
    latency_path = args.output_root / "latency.json"
    comparison_path = args.output_root / "comparison.json"

    write_csv(grid_path, grid_rows)
    write_json(selection_path, {
        "selection_metric": "Macro-author Top1 on ALL Train-Val",
        "tie_break": "MRR@10, Top3, lower total context weight, lower BGE, lower NGram",
        **selections,
        "bge_recency_ablation": "Top-5 retrieval remains cosine-only; recency affects aggregation only",
        "gold_used_for_scoring_features": False,
        "dev3000_used": False,
        "test_used": False,
    })
    write_csv(method_path, selected_method_rows)
    write_csv(subset_path, subset_rows)
    write_csv(author_path, per_author_rows)
    write_csv(recovery_path, recovery_rows)
    write_csv(transition_path, transition_rows)
    write_jsonl(prediction_path, prediction_rows)
    write_json(diagnostic_path, context_diagnostics)
    write_json(latency_path, latency)

    comparison = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "experiment": EXPERIMENT,
        "purpose": "Test whether same-user history recency strengthens BGE semantic context, alone and combined with NGramRecency, while expanding boundary-hit context weights.",
        "evaluation_population": {
            "train_val_rows": len(row_ids),
            "authors": sorted(set(authors)),
            "context_opportunity_n": sum(opportunity_mask),
            "recovery_population_definition": "Generic Missing AND Gold in frozen Personal K5",
            "recovery_population_n": sum(recovery_mask),
            "pv1_rescue_vs_frequency_n": sum(pv1_rescue_vs_f_mask),
            "pv1_harm_vs_frequency_n": sum(pv1_harm_vs_f_mask),
        },
        "formulas": {
            "PV1": "S_PV1",
            "BGE": "S_PV1 + lambda_B * P_BGE",
            "BGERecency": "S_PV1 + lambda_B * P_BGR",
            "NGramRecency": "S_PV1 + lambda_N * P_NG",
            "BGE+NGramRecency": "S_PV1 + lambda_B * P_BGE + lambda_N * P_NG",
            "BGERecency+NGramRecency": "S_PV1 + lambda_B * P_BGR + lambda_N * P_NG",
            "P_BGR_raw": "sum_{h in Top5_cos(candidate)} max(0, cosine(q,h)) * exp(-age(h)/tau_B)",
        },
        "bge": {
            "context_chars": base.BGE_CONTEXT_CHARS,
            "top_n_per_candidate": base.BGE_TOP_N,
        },
        "bge_recency": {
            "tau": args.bge_recency_tau,
            "age_unit": "same-author interactions",
            "retrieval": "Top-5 by cosine only; no recency in retrieval",
            "aggregation": "positive cosine multiplied by exp(-age/tau)",
        },
        "ngram_recency": {
            "max_n": args.hard_max_n,
            "tau": args.hard_tau,
        },
        "weight_grids": {
            "lambda_bge": list(args.lambda_bge),
            "lambda_ngram": list(args.lambda_ngram),
        },
        "selected": selections,
        "selected_method_metrics": selected_method_rows,
        "recovery_metrics": recovery_rows,
        "prior_bge_provenance": dict(prior_provenance),
        "invariants": {
            "candidate_set_changed": False,
            "missing10_must_equal_pv1": True,
            "recovery10_must_equal_pv1": True,
            "zero_weights_exactly_reproduce_pv1": True,
            "frozen_pv1_final_scores_preserved_as_base": True,
            "h5000_before_exact_pinyin_filter": True,
            "strictly_prior_same_author_history": True,
            "bge_top5_retrieval_unchanged_by_recency": True,
        },
        "provenance": {
            name: {"path": str(path.resolve()), "sha256": inputs["input_hashes"][name]}
            for name, path in inputs["input_paths"].items()
        },
        "bge_model": {
            "path": str(args.bge_model.resolve()),
            "sha256": base.sha256_file(args.bge_model),
        },
        "gold_used_for_scoring_features": False,
        "gold_used_for_train_val_selection_evaluation": True,
        "dev3000_used": False,
        "test_used": False,
        "evaluation_runtime_seconds": time.perf_counter() - started,
    }
    write_json(comparison_path, comparison)

    output_files = [
        args.output_root / "run_manifest.json",
        args.output_root / "bge_recency_scores.jsonl",
        args.output_root / "bge_recency_scoring_summary.json",
        grid_path,
        selection_path,
        method_path,
        subset_path,
        author_path,
        recovery_path,
        transition_path,
        prediction_path,
        diagnostic_path,
        latency_path,
        comparison_path,
    ]
    checksums = {path.name: base.sha256_file(path) for path in output_files if path.is_file()}
    write_json(args.output_root / "artifact_checksums.json", {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "artifacts": checksums,
        "note": "Prior V1 BGE cache is read-only external provenance and is not copied into this result directory.",
    })

    print("\n=== SELECTED CONFIGURATIONS ===", flush=True)
    for name in (
        "BGE",
        "BGERecency",
        "NGramRecency",
        "BGE+NGramRecency",
        "BGERecency+NGramRecency",
        "GlobalBest",
    ):
        row = selections[name]
        print(
            f"{name:30s} mode={str(row['semantic_mode']):7s} "
            f"lambda_B={float(row['lambda_bge']):g} lambda_N={float(row['lambda_ngram']):g}",
            flush=True,
        )

    print("\n=== COMPLETE ===", flush=True)
    for row in selected_method_rows:
        print(
            f"{row['method']:28s} "
            f"Macro={float(row['macro_author_top1']):.6f} "
            f"Micro={float(row['micro_top1']):.6f} "
            f"Top3={float(row['top3']):.6f} "
            f"Top5={float(row['top5']):.6f} "
            f"MRR={float(row['mrr_at_10']):.6f} "
            f"Missing={float(row['missing10']):.6f}",
            flush=True,
        )
    print("\nRecovery population:", flush=True)
    for row in recovery_rows:
        print(
            f"{row['method']:28s} Rec1={float(row['rec1']):.4f} "
            f"Rec3={float(row['rec3']):.4f} Rec5={float(row['rec5']):.4f} "
            f"Rec10={float(row['rec10']):.4f} RecMRR={float(row['recovery_mrr_at_10']):.4f}",
            flush=True,
        )
    print(f"\nOutputs: {args.output_root.resolve()}", flush=True)
    print("Gold used for scoring/features: false", flush=True)
    print("Dev3000 used: false", flush=True)
    print("Test used: false", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--candidate-surface", type=Path, required=True)
    parser.add_argument("--frequency-pv1-predictions", type=Path, required=True)
    parser.add_argument("--bge-model", type=Path, required=True)
    parser.add_argument("--base-bge-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cuda-path", type=Path, default=None)
    parser.add_argument("--bge-recency-tau", type=float, default=DEFAULT_BGE_RECENCY_TAU)
    parser.add_argument("--lambda-bge", nargs="+", type=float, default=list(DEFAULT_BGE_LAMBDAS))
    parser.add_argument("--lambda-ngram", nargs="+", type=float, default=list(DEFAULT_NGRAM_LAMBDAS))
    parser.add_argument("--hard-max-n", type=int, default=base.DEFAULT_HARD_MAX_N)
    parser.add_argument("--hard-tau", type=float, default=base.DEFAULT_HARD_TAU)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--allow-unfrozen-input-hashes", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace) -> None:
    for path_name in (
        "fit",
        "val",
        "candidate_surface",
        "frequency_pv1_predictions",
        "bge_model",
    ):
        path = getattr(args, path_name)
        if not path.is_file():
            raise RuntimeError(f"Input file does not exist: --{path_name.replace('_', '-')} {path}")
    if not args.base_bge_root.is_dir():
        raise RuntimeError(f"--base-bge-root does not exist: {args.base_bge_root}")
    if args.output_root.resolve() == args.base_bge_root.resolve():
        raise RuntimeError("--output-root must differ from --base-bge-root; prior artifacts are read-only")
    if args.bge_recency_tau <= 0 or args.hard_tau <= 0:
        raise ValueError("Recency tau values must be positive")
    if args.hard_max_n <= 0:
        raise ValueError("--hard-max-n must be positive")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    args.lambda_bge = tuple(sorted(set(float(v) for v in args.lambda_bge)))
    args.lambda_ngram = tuple(sorted(set(float(v) for v in args.lambda_ngram)))
    if 0.0 not in args.lambda_bge or 0.0 not in args.lambda_ngram:
        raise ValueError("Both lambda grids must include 0 for exact PV1 control")
    if not any(v > 0 for v in args.lambda_bge) or not any(v > 0 for v in args.lambda_ngram):
        raise ValueError("Both lambda grids need at least one positive value")
    if any(v < 0 for v in args.lambda_bge + args.lambda_ngram):
        raise ValueError("Context weights must be non-negative")


row_index_by_id: dict[str, int] = {}


def main() -> None:
    args = build_parser().parse_args()
    normalize_args(args)
    total_started = time.perf_counter()

    inputs = base.prepare_inputs(args)
    global row_index_by_id
    row_index_by_id = {rid: i for i, rid in enumerate(inputs["row_ids"])}

    prior_bge_rows, prior_provenance = load_prior_bge(args, inputs)
    ensure_manifest(args, inputs, prior_provenance)

    print("=== INITIAL PV1 BGE RECENCY + NGRAM CONTEXT RERANKING V2 ===", flush=True)
    print(f"Train-Val rows: {len(inputs['row_ids'])}", flush=True)
    print(f"Authors: {sorted(set(inputs['authors']))}", flush=True)
    print("Baseline: frozen PV1 K=1, lambda_F=4, lambda_PV=4", flush=True)
    print(f"BGE64: last {base.BGE_CONTEXT_CHARS} chars, candidate-conditioned Top-{base.BGE_TOP_N} positive cosine", flush=True)
    print(f"BGERecency: Top-{base.BGE_TOP_N} chosen by cosine; tau_B={args.bge_recency_tau:g}; recency only in aggregation", flush=True)
    print(f"NGramRecency: HardBackoff maxN={args.hard_max_n}, tau={args.hard_tau:g}", flush=True)
    print(f"lambda_B grid: {list(args.lambda_bge)}", flush=True)
    print(f"lambda_N grid: {list(args.lambda_ngram)}", flush=True)
    print("Candidate set: frozen PV1 Top10 only; pure reranking", flush=True)
    print("Prior V1 BGE artifacts: read-only", flush=True)
    print("Gold used for scoring/features: false", flush=True)
    print("Dev3000 used: false", flush=True)
    print("Test used: false", flush=True)

    bge_recency_rows = run_bge_recency_scoring(args, inputs, prior_bge_rows)
    evaluate(args, inputs, prior_bge_rows, bge_recency_rows, prior_provenance)
    print(f"Total wall time: {time.perf_counter() - total_started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
