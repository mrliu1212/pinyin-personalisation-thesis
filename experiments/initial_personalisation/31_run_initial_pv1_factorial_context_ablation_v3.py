from __future__ import annotations

"""Factorial PV1 final-list context ablation: BGE, NGram, and Recency as independent factors.

Research question
-----------------
The previous final-list experiments showed that BGERecency and NGramRecency are
useful, but both methods multiply context relevance by the SAME same-author
interaction-recency signal.  This runner separates the factors so that their
main effects and additive complementarity can be measured directly.

Frozen candidate surface
------------------------
All methods rerank the frozen PV1 final Top-10 only.  No candidate is added or
removed and PV1 final_score remains the base score.

Independent factors
-------------------
For candidate c in the frozen PV1 Top-10:

    S(c) = S_PV1(c)
           + lambda_B * P_BGE(c)
           + lambda_N * P_NG(c)
           + lambda_R * P_R(c)

P_BGE(c)
    Plain BGE64 support from the completed V1 experiment: last 64 context
    characters; candidate-conditioned Top-5 positive cosine; normalized over
    the frozen PV1 Top-10.

P_NG(c)
    Plain HardBackoff NGram support WITHOUT recency.  Choose the largest exact
    suffix order k <= maxN with candidate-target evidence; if no k>0 exists,
    back off to k=0.  Count matched historical interactions for each candidate,
    then normalize over the frozen PV1 Top-10.

P_R(c)
    Recency-only support WITHOUT context matching:

        raw_R(c) = sum_{h: target(h)=c} exp(-age(h)/tau_R)

    over legal same-Pinyin history, normalized over the frozen PV1 Top-10.

This creates the clean additive ablations:
    PV1
    PV1 + Recency
    PV1 + NGram
    PV1 + NGram + Recency
    PV1 + BGE
    PV1 + BGE + Recency
    PV1 + BGE + NGram
    PV1 + BGE + NGram + Recency

Interaction controls
--------------------
For comparison only, the runner also reconstructs the already-selected V2
interaction methods from read-only artifacts:

    NGramRecency:
        suffix gating first, then matched interactions weighted exp(-age/tau)

    BGERecency:
        BGE Top-5 retrieval first, then cosine weighted exp(-age/tau)

    BGERecency + NGramRecency:
        the completed V2 selected interaction model.

This permits direct comparisons such as:
    NGram + Recency (additive)  vs  NGramRecency (interaction)
    BGE + Recency   (additive)  vs  BGERecency   (interaction)

Protocol invariants
-------------------
- Clean3 Train-Fit + Train-Val only.
- Same-author strictly-prior causal history.
- H5000 is selected BEFORE exact-Pinyin filtering.
- Current/future Gold never enters candidate construction or scoring.
- Dev3000 and Test are untouched.
- Frozen PV1 final scores are preserved as the base.
- Frozen PV1 Top-10 candidate set is unchanged.
- Missing@10 and Recovery@10 must remain invariant.
- Completed V1/V2 BGE artifacts are read-only.

The full 3-factor grid is evaluated with vectorized NumPy rank computation, so
this runner should be much faster than naively sorting Top-10 for every grid
point and row.  BGE embeddings are NOT recomputed.
"""

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiments.initial_personalisation import (
    run_initial_pv1_bge_ngram_context_reranking_v1 as base,
)


EXPERIMENT = "initial_pv1_factorial_context_ablation_v3"
SCHEMA_VERSION = 3
DEFAULT_MAX_N = 2
DEFAULT_TAU_R = 2048.0
DEFAULT_TAU_NR = 2048.0
DEFAULT_LAMBDA_BGE = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0)
DEFAULT_LAMBDA_NGRAM = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0)
DEFAULT_LAMBDA_RECENCY = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as dst:
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    base.write_csv(path, rows)


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return base.load_jsonl_by_id(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(values: Sequence[float]) -> list[float]:
    return base.normalize_nonnegative(values)


def support_factors(
    *,
    candidates: Sequence[str],
    query_context: str,
    visible: Sequence[base.VisibleHistory],
    max_n: int,
    tau_r: float,
    tau_nr: float,
) -> tuple[list[float], list[float], list[float], dict[str, Any]]:
    """Compute plain NGram, Recency-only, and NGramRecency in one history pass."""

    candidate_set = set(candidates)
    effective_n = 0
    for n in range(max_n, 0, -1):
        if any(
            item.record.target in candidate_set
            and base.suffix_matches(query_context, item.record.context, n)
            for item in visible
        ):
            effective_n = n
            break

    raw_ng = {candidate: 0.0 for candidate in candidates}
    raw_r = {candidate: 0.0 for candidate in candidates}
    raw_nr = {candidate: 0.0 for candidate in candidates}
    candidate_history_rows = 0
    matched_rows = 0

    for item in visible:
        target = item.record.target
        if target not in raw_ng:
            continue
        candidate_history_rows += 1
        rw_r = base.recency_weight(item.age, tau_r)
        raw_r[target] += rw_r
        if not base.suffix_matches(query_context, item.record.context, effective_n):
            continue
        matched_rows += 1
        raw_ng[target] += 1.0
        raw_nr[target] += base.recency_weight(item.age, tau_nr)

    ng = normalize([raw_ng[c] for c in candidates])
    rec = normalize([raw_r[c] for c in candidates])
    ngr = normalize([raw_nr[c] for c in candidates])
    return ng, rec, ngr, {
        "effective_n": effective_n,
        "visible_same_pinyin_history_rows": len(visible),
        "candidate_target_history_rows": candidate_history_rows,
        "ngram_matched_history_rows": matched_rows,
        "plain_ngram_raw_mass": float(sum(raw_ng.values())),
        "recency_raw_mass": float(sum(raw_r.values())),
        "ngram_recency_raw_mass": float(sum(raw_nr.values())),
    }


def load_prior_artifacts(
    args: argparse.Namespace,
    inputs: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    v1_scores_path = args.base_bge_root / "bge_scores.jsonl"
    v1_manifest_path = args.base_bge_root / "run_manifest.json"
    v1_comparison_path = args.base_bge_root / "comparison.json"
    v2_scores_path = args.base_bge_recency_root / "bge_recency_scores.jsonl"
    v2_manifest_path = args.base_bge_recency_root / "run_manifest.json"
    v2_selection_path = args.base_bge_recency_root / "family_selection.json"
    v2_comparison_path = args.base_bge_recency_root / "comparison.json"

    required = (
        v1_scores_path,
        v1_manifest_path,
        v1_comparison_path,
        v2_scores_path,
        v2_manifest_path,
        v2_selection_path,
        v2_comparison_path,
    )
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"Required prior artifact missing: {path}")

    v1_manifest = load_json(v1_manifest_path)
    v2_manifest = load_json(v2_manifest_path)
    v1_comparison = load_json(v1_comparison_path)
    v2_comparison = load_json(v2_comparison_path)
    v2_selection = load_json(v2_selection_path)

    if v1_comparison.get("status") != "complete":
        raise RuntimeError("--base-bge-root is not complete")
    if v2_comparison.get("status") != "complete":
        raise RuntimeError("--base-bge-recency-root is not complete")
    if v1_manifest.get("input_hashes") != inputs["input_hashes"]:
        raise RuntimeError("V1 BGE input hashes differ from current frozen inputs")
    if v2_manifest.get("input_hashes") != inputs["input_hashes"]:
        raise RuntimeError("V2 BGERecency input hashes differ from current frozen inputs")
    if int(v1_manifest.get("bge_context_chars", -1)) != base.BGE_CONTEXT_CHARS:
        raise RuntimeError("V1 BGE context length differs")
    if int(v1_manifest.get("bge_top_n", -1)) != base.BGE_TOP_N:
        raise RuntimeError("V1 BGE Top-N differs")

    v1_rows = load_jsonl_by_id(v1_scores_path)
    v2_rows = load_jsonl_by_id(v2_scores_path)
    if len(v1_rows) != base.EXPECTED_VAL_ROWS or len(v2_rows) != base.EXPECTED_VAL_ROWS:
        raise RuntimeError("Prior BGE score cache is incomplete")

    for i, rid in enumerate(inputs["row_ids"]):
        expected = list(inputs["baseline_rankings"][i])
        if v1_rows[rid].get("candidates") != expected:
            raise RuntimeError(f"V1 BGE candidate mismatch at {rid}")
        if v2_rows[rid].get("candidates") != expected:
            raise RuntimeError(f"V2 BGERecency candidate mismatch at {rid}")
        if len(v1_rows[rid].get("support", [])) != len(expected):
            raise RuntimeError(f"V1 BGE support length mismatch at {rid}")
        if len(v2_rows[rid].get("support", [])) != len(expected):
            raise RuntimeError(f"V2 BGERecency support length mismatch at {rid}")

    selected_v2: dict[str, dict[str, Any]] = {}
    for key in ("BGERecency", "NGramRecency", "BGERecency+NGramRecency"):
        if key not in v2_selection:
            raise RuntimeError(f"V2 family_selection.json lacks {key}")
        selected_v2[key] = dict(v2_selection[key])

    provenance = {
        "v1_root": str(args.base_bge_root.resolve()),
        "v1_bge_scores_sha256": base.sha256_file(v1_scores_path),
        "v1_manifest_sha256": base.sha256_file(v1_manifest_path),
        "v1_comparison_sha256": base.sha256_file(v1_comparison_path),
        "v2_root": str(args.base_bge_recency_root.resolve()),
        "v2_bge_recency_scores_sha256": base.sha256_file(v2_scores_path),
        "v2_manifest_sha256": base.sha256_file(v2_manifest_path),
        "v2_selection_sha256": base.sha256_file(v2_selection_path),
        "v2_comparison_sha256": base.sha256_file(v2_comparison_path),
        "v2_selected": selected_v2,
    }
    return v1_rows, v2_rows, provenance


def ensure_output_root(args: argparse.Namespace, inputs: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {args.output_root}. Use a new versioned path; historical artifacts are never overwritten."
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_root / "run_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "input_hashes": dict(inputs["input_hashes"]),
            "prior_artifacts": dict(provenance),
            "bge_context_chars": base.BGE_CONTEXT_CHARS,
            "bge_top_n": base.BGE_TOP_N,
            "plain_ngram_max_n": args.max_n,
            "recency_tau": args.recency_tau,
            "ngram_recency_tau": args.ngram_recency_tau,
            "lambda_bge_grid": list(args.lambda_bge),
            "lambda_ngram_grid": list(args.lambda_ngram),
            "lambda_recency_grid": list(args.lambda_recency),
            "gold_used_for_scoring_features": False,
            "dev3000_used": False,
            "test_used": False,
        },
    )


def rank_array_to_list(ranks: np.ndarray) -> list[int | None]:
    return [None if int(v) == 0 else int(v) for v in ranks.tolist()]


def rank_gold_matrix(
    scores: np.ndarray,
    valid_mask: np.ndarray,
    gold_indices: np.ndarray,
) -> np.ndarray:
    """Exact rank under tie-break (-score, original PV1 index). 0 means missing."""

    n_rows, width = scores.shape
    result = np.zeros(n_rows, dtype=np.int16)
    present = gold_indices >= 0
    if not np.any(present):
        return result
    rows = np.nonzero(present)[0]
    gi = gold_indices[rows]
    gold_scores = scores[rows, gi]
    sub = scores[rows]
    valid = valid_mask[rows]
    indices = np.arange(width, dtype=np.int16)[None, :]
    better = valid & (
        (sub > gold_scores[:, None])
        | ((sub == gold_scores[:, None]) & (indices < gi[:, None]))
    )
    result[rows] = 1 + better.sum(axis=1).astype(np.int16)
    return result


def fast_metrics(
    ranks: np.ndarray,
    author_index: np.ndarray,
    n_authors: int,
) -> dict[str, float]:
    present = ranks > 0
    top1 = ranks == 1
    macro_parts: list[float] = []
    for author_id in range(n_authors):
        mask = author_index == author_id
        macro_parts.append(float(np.mean(top1[mask])))
    reciprocal = np.zeros_like(ranks, dtype=np.float64)
    reciprocal[present] = 1.0 / ranks[present].astype(np.float64)
    return {
        "macro_author_top1": float(statistics.fmean(macro_parts)),
        "micro_top1": float(np.mean(top1)),
        "top3": float(np.mean((ranks > 0) & (ranks <= 3))),
        "top5": float(np.mean((ranks > 0) & (ranks <= 5))),
        "mrr_at_10": float(np.mean(reciprocal)),
        "missing10": float(np.mean(ranks == 0)),
    }


def fast_recovery(ranks: np.ndarray, recovery_mask: np.ndarray) -> dict[str, float]:
    rr = ranks[recovery_mask]
    present = rr > 0
    reciprocal = np.zeros_like(rr, dtype=np.float64)
    reciprocal[present] = 1.0 / rr[present].astype(np.float64)
    return {
        "rec1": float(np.mean(rr == 1)),
        "rec3": float(np.mean((rr > 0) & (rr <= 3))),
        "rec5": float(np.mean((rr > 0) & (rr <= 5))),
        "rec10": float(np.mean(rr > 0)),
        "recovery_mrr_at_10": float(np.mean(reciprocal)),
    }


def grid_selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        float(row["macro_author_top1"]),
        float(row["mrr_at_10"]),
        float(row["top3"]),
        -float(row["lambda_bge"] + row["lambda_ngram"] + row["lambda_recency"]),
        -float(row["lambda_bge"]),
        -float(row["lambda_ngram"]),
        -float(row["lambda_recency"]),
    )


def family_of(lb: float, ln: float, lr: float) -> str:
    active = []
    if lb > 0:
        active.append("BGE")
    if ln > 0:
        active.append("NGram")
    if lr > 0:
        active.append("Recency")
    return "PV1" if not active else "+".join(active)


def select_family(grid_rows: Sequence[Mapping[str, Any]], family: str) -> dict[str, Any]:
    candidates = [dict(row) for row in grid_rows if str(row["family"]) == family]
    if not candidates:
        raise RuntimeError(f"No grid rows for family {family}")
    return max(candidates, key=grid_selection_key)


def rerank_selected(
    candidates: Sequence[str],
    pv1_scores: Sequence[float],
    bge: Sequence[float],
    ng: Sequence[float],
    rec: Sequence[float],
    *,
    lb: float,
    ln: float,
    lr: float,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    scores = [
        float(s) + lb * float(b) + ln * float(n) + lr * float(r)
        for s, b, n, r in zip(pv1_scores, bge, ng, rec)
    ]
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i, candidates[i]))
    return tuple(candidates[i] for i in order), tuple(float(scores[i]) for i in order)


def rerank_interaction(
    candidates: Sequence[str],
    pv1_scores: Sequence[float],
    bge_recency: Sequence[float],
    ngram_recency: Sequence[float],
    *,
    lb: float,
    ln: float,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    return base.rerank_joint(
        candidates,
        pv1_scores,
        bge_recency,
        ngram_recency,
        lambda_bge=lb,
        lambda_ngram=ln,
    )


def evaluate(args: argparse.Namespace) -> None:
    total_started = time.perf_counter()
    inputs = base.prepare_inputs(args)
    v1_bge, v2_bge_recency, provenance = load_prior_artifacts(args, inputs)
    ensure_output_root(args, inputs, provenance)

    row_ids = inputs["row_ids"]
    n_rows = len(row_ids)
    authors = inputs["authors"]
    golds = inputs["golds"]
    candidates_all = inputs["baseline_rankings"]
    pv1_scores_all = inputs["baseline_score_vectors"]
    baseline_ranks = inputs["baseline_ranks"]
    recovery_mask_list = inputs["recovery_mask"]
    max_width = max(len(c) for c in candidates_all)

    base_matrix = np.full((n_rows, max_width), -np.inf, dtype=np.float64)
    bge_matrix = np.zeros((n_rows, max_width), dtype=np.float64)
    ng_matrix = np.zeros((n_rows, max_width), dtype=np.float64)
    rec_matrix = np.zeros((n_rows, max_width), dtype=np.float64)
    ngr_matrix = np.zeros((n_rows, max_width), dtype=np.float64)
    bger_matrix = np.zeros((n_rows, max_width), dtype=np.float64)
    valid_mask = np.zeros((n_rows, max_width), dtype=bool)
    gold_indices = np.full(n_rows, -1, dtype=np.int16)

    support_rows: list[dict[str, Any]] = []
    effective_n_counter: Counter[int] = Counter()
    visible_counts: list[int] = []
    candidate_history_counts: list[int] = []
    matched_counts: list[int] = []
    support_latency_ms: list[float] = []

    print("=== INITIAL PV1 FACTORIAL CONTEXT ABLATION V3 ===", flush=True)
    print(f"Train-Val rows: {n_rows}", flush=True)
    print(f"Authors: {sorted(set(authors))}", flush=True)
    print("Candidate set: frozen PV1 Top10 only; pure reranking", flush=True)
    print(f"BGE: read-only V1, last {base.BGE_CONTEXT_CHARS} chars, Top-{base.BGE_TOP_N} positive cosine", flush=True)
    print(f"Plain NGram: HardBackoff maxN={args.max_n}, NO recency", flush=True)
    print(f"Recency-only: exp(-age/{args.recency_tau:g}), NO context matching", flush=True)
    print(f"NGramRecency control: HardBackoff maxN={args.max_n}, tau={args.ngram_recency_tau:g}", flush=True)
    print("BGERecency control: read-only V2 support", flush=True)
    print(f"lambda_B: {list(args.lambda_bge)}", flush=True)
    print(f"lambda_N: {list(args.lambda_ngram)}", flush=True)
    print(f"lambda_R: {list(args.lambda_recency)}", flush=True)
    print("Gold used for scoring/features: false", flush=True)
    print("Dev3000 used: false", flush=True)
    print("Test used: false", flush=True)

    for i, rid in enumerate(row_ids):
        candidates = candidates_all[i]
        k = len(candidates)
        valid_mask[i, :k] = True
        base_matrix[i, :k] = np.asarray(pv1_scores_all[i], dtype=np.float64)
        bge = [float(v) for v in v1_bge[rid]["support"]]
        bger = [float(v) for v in v2_bge_recency[rid]["support"]]
        bge_matrix[i, :k] = bge
        bger_matrix[i, :k] = bger
        try:
            gold_indices[i] = candidates.index(golds[i])
        except ValueError:
            gold_indices[i] = -1

        vrow = inputs["val_by_id"][rid]
        visible = inputs["history_index"].visible(
            author=str(vrow["author"]),
            position=int(vrow["chronological_position"]),
            pinyin=base.pinyin_of(vrow),
        )
        t0 = time.perf_counter()
        ng, rec, ngr, diag = support_factors(
            candidates=candidates,
            query_context=base.context_of(vrow),
            visible=visible,
            max_n=args.max_n,
            tau_r=args.recency_tau,
            tau_nr=args.ngram_recency_tau,
        )
        support_latency_ms.append((time.perf_counter() - t0) * 1000.0)
        ng_matrix[i, :k] = ng
        rec_matrix[i, :k] = rec
        ngr_matrix[i, :k] = ngr
        effective_n_counter[int(diag["effective_n"])] += 1
        visible_counts.append(int(diag["visible_same_pinyin_history_rows"]))
        candidate_history_counts.append(int(diag["candidate_target_history_rows"]))
        matched_counts.append(int(diag["ngram_matched_history_rows"]))
        support_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "row_id": rid,
                "author": authors[i],
                "candidates": list(candidates),
                "plain_bge_support": bge,
                "plain_ngram_support": ng,
                "recency_only_support": rec,
                "ngram_recency_support": ngr,
                "bge_recency_support": bger,
                **diag,
                "gold_used_for_scoring": False,
            }
        )
        if (i + 1) % args.progress_every == 0 or i + 1 == n_rows:
            print(f"Factor support construction: {i + 1}/{n_rows}", flush=True)

    # Exact control: base matrix alone must reproduce frozen PV1 ranks.
    baseline_np = np.asarray([0 if r is None else int(r) for r in baseline_ranks], dtype=np.int16)
    pv1_rank_check = rank_gold_matrix(base_matrix, valid_mask, gold_indices)
    if not np.array_equal(pv1_rank_check, baseline_np):
        bad = int(np.nonzero(pv1_rank_check != baseline_np)[0][0])
        raise RuntimeError(f"PV1 rank reconstruction failed at row {row_ids[bad]}")

    author_names = sorted(set(authors))
    author_to_index = {name: i for i, name in enumerate(author_names)}
    author_index = np.asarray([author_to_index[a] for a in authors], dtype=np.int16)
    recovery_mask = np.asarray(recovery_mask_list, dtype=bool)
    baseline_rec10 = fast_recovery(baseline_np, recovery_mask)["rec10"]

    grid_rows: list[dict[str, Any]] = []
    grid_started = time.perf_counter()
    total_grid = len(args.lambda_bge) * len(args.lambda_ngram) * len(args.lambda_recency)
    done = 0
    for lb in args.lambda_bge:
        for ln in args.lambda_ngram:
            partial = base_matrix + float(lb) * bge_matrix + float(ln) * ng_matrix
            for lr in args.lambda_recency:
                scores = partial + float(lr) * rec_matrix
                ranks = rank_gold_matrix(scores, valid_mask, gold_indices)
                metrics = fast_metrics(ranks, author_index, len(author_names))
                rec_metrics = fast_recovery(ranks, recovery_mask)
                if abs(metrics["missing10"] - float(inputs["baseline_metrics"]["micro"]["missing10"])) > 1e-15:
                    raise RuntimeError(f"Missing@10 changed at B={lb}, N={ln}, R={lr}")
                if abs(rec_metrics["rec10"] - baseline_rec10) > 1e-15:
                    raise RuntimeError(f"Recovery@10 changed at B={lb}, N={ln}, R={lr}")
                grid_rows.append(
                    {
                        "family": family_of(float(lb), float(ln), float(lr)),
                        "lambda_bge": float(lb),
                        "lambda_ngram": float(ln),
                        "lambda_recency": float(lr),
                        **metrics,
                        **rec_metrics,
                    }
                )
                done += 1
                if done % args.grid_progress_every == 0 or done == total_grid:
                    elapsed = time.perf_counter() - grid_started
                    print(
                        f"Additive factorial grid: {done}/{total_grid}; rate={done / max(elapsed, 1e-9):.2f} configs/s",
                        flush=True,
                    )

    families = (
        "PV1",
        "Recency",
        "NGram",
        "NGram+Recency",
        "BGE",
        "BGE+Recency",
        "BGE+NGram",
        "BGE+NGram+Recency",
    )
    selected: dict[str, dict[str, Any]] = {family: select_family(grid_rows, family) for family in families}
    selected["GlobalBestAdditiveIncludingZeros"] = max(grid_rows, key=grid_selection_key)

    # Selected additive rank surfaces.
    selected_ranks: dict[str, list[int | None]] = {}
    selected_specs: dict[str, dict[str, Any]] = {}
    for family in families:
        row = selected[family]
        lb = float(row["lambda_bge"])
        ln = float(row["lambda_ngram"])
        lr = float(row["lambda_recency"])
        ranks = rank_gold_matrix(
            base_matrix + lb * bge_matrix + ln * ng_matrix + lr * rec_matrix,
            valid_mask,
            gold_indices,
        )
        selected_ranks[family] = rank_array_to_list(ranks)
        selected_specs[family] = {
            "type": "additive_independent_factors",
            "lambda_bge": lb,
            "lambda_ngram": ln,
            "lambda_recency": lr,
        }

    # Reconstruct the selected interaction controls from completed V2.
    v2_selected = provenance["v2_selected"]
    interaction_specs = {
        "BGERecency_interaction": {
            "lambda_bge": float(v2_selected["BGERecency"]["lambda_bge"]),
            "lambda_ngram": 0.0,
        },
        "NGramRecency_interaction": {
            "lambda_bge": 0.0,
            "lambda_ngram": float(v2_selected["NGramRecency"]["lambda_ngram"]),
        },
        "BGERecency+NGramRecency_interaction": {
            "lambda_bge": float(v2_selected["BGERecency+NGramRecency"]["lambda_bge"]),
            "lambda_ngram": float(v2_selected["BGERecency+NGramRecency"]["lambda_ngram"]),
        },
    }
    for method, spec in interaction_specs.items():
        # Use the same Python scoring/sort path as V2 for an exact reconstruction
        # guardrail; this is only three selected methods, so the cost is tiny.
        ranks_list: list[int | None] = []
        for i, rid in enumerate(row_ids):
            k = len(candidates_all[i])
            ranking, _ = rerank_interaction(
                candidates_all[i],
                pv1_scores_all[i],
                bger_matrix[i, :k],
                ngr_matrix[i, :k],
                lb=float(spec["lambda_bge"]),
                ln=float(spec["lambda_ngram"]),
            )
            ranks_list.append(base.rank_of(ranking, golds[i]))
        selected_ranks[method] = ranks_list
        selected_specs[method] = {
            "type": "context_x_recency_interaction",
            **spec,
            "lambda_recency": None,
        }

    # Guardrail: interaction controls must reproduce completed V2 headline values.
    v2_expected_rows = {
        str(row["method"]): row
        for row in v2_comparison_metrics(args.base_bge_recency_root)
    }
    method_to_v2_name = {
        "BGERecency_interaction": "BGERecency",
        "NGramRecency_interaction": "NGramRecency",
        "BGERecency+NGramRecency_interaction": "BGERecency+NGramRecency",
    }
    for method, v2_name in method_to_v2_name.items():
        if v2_name not in v2_expected_rows:
            raise RuntimeError(f"V2 comparison lacks selected metric row {v2_name}")
        m = base.author_metrics(authors, selected_ranks[method])
        actual = float(m["macro_author"]["top1"])
        expected = float(v2_expected_rows[v2_name]["macro_author_top1"])
        if abs(actual - expected) > 1e-12:
            raise RuntimeError(
                f"V2 interaction reconstruction drift for {method}: {actual} != {expected}"
            )

    opportunity_mask = [rank is not None for rank in baseline_ranks]
    conflict_mask = inputs["conflict_mask"]
    non_conflict_mask = [not v for v in conflict_mask]
    generic_missing_mask = inputs["generic_missing_mask"]
    ambiguous_mask = inputs["ambiguous_mask"]
    history_available_mask = inputs["history_available_mask"]
    frequency_ranks = inputs["frequency_ranks"]
    pv1_rescue_vs_f = [f != 1 and p == 1 for f, p in zip(frequency_ranks, baseline_ranks)]
    pv1_harm_vs_f = [f == 1 and p != 1 for f, p in zip(frequency_ranks, baseline_ranks)]

    subset_masks = {
        "overall": [True] * n_rows,
        "history_available": history_available_mask,
        "ambiguous": ambiguous_mask,
        "conflict": conflict_mask,
        "non_conflict": non_conflict_mask,
        "generic_missing": generic_missing_mask,
        "context_opportunity_gold_in_pv1_top10": opportunity_mask,
        "pv1_rescue_vs_frequency": pv1_rescue_vs_f,
        "pv1_harm_vs_frequency": pv1_harm_vs_f,
        "recovery_k5": recovery_mask_list,
    }

    selected_method_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []
    per_author_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []

    output_method_order = list(families) + list(interaction_specs)
    for method in output_method_order:
        ranks = selected_ranks[method]
        spec = selected_specs[method]
        metrics = base.author_metrics(authors, ranks)
        transition = base.transition_counts(baseline_ranks, ranks)
        movement = base.rank_movement(baseline_ranks, ranks, opportunity_mask)
        rescue_den = sum(pv1_rescue_vs_f)
        harm_den = sum(pv1_harm_vs_f)
        rescue_retained = sum(include and rank == 1 for include, rank in zip(pv1_rescue_vs_f, ranks))
        harm_repaired = sum(include and rank == 1 for include, rank in zip(pv1_harm_vs_f, ranks))
        selected_method_rows.append(
            {
                "method": method,
                **spec,
                "macro_author_top1": metrics["macro_author"]["top1"],
                "macro_author_top3": metrics["macro_author"]["top3"],
                "macro_author_top5": metrics["macro_author"]["top5"],
                "macro_author_mrr_at_10": metrics["macro_author"]["mrr_at_10"],
                "micro_top1": metrics["micro"]["top1"],
                "top3": metrics["micro"]["top3"],
                "top5": metrics["micro"]["top5"],
                "mrr_at_10": metrics["micro"]["mrr_at_10"],
                "missing10": metrics["micro"]["missing10"],
                "pv1_rescue_vs_f_n": rescue_den,
                "pv1_rescue_retained_n": rescue_retained,
                "pv1_rescue_retention_rate": rescue_retained / rescue_den if rescue_den else 0.0,
                "pv1_harm_vs_f_n": harm_den,
                "pv1_harm_repaired_n": harm_repaired,
                "pv1_harm_repair_rate": harm_repaired / harm_den if harm_den else 0.0,
                **transition,
            }
        )
        recovery_rows.append({"method": method, **spec, **base.recovery_metrics(ranks, recovery_mask_list)})
        transition_rows.append(
            {
                "method": method,
                "vs": "PV1",
                **transition,
                **movement,
            }
        )
        for subset_name, mask in subset_masks.items():
            subset_ranks = [rank for rank, include in zip(ranks, mask) if include]
            subset_authors = [author for author, include in zip(authors, mask) if include]
            m = base.author_metrics(subset_authors, subset_ranks)
            subset_rows.append(
                {
                    "method": method,
                    "subset": subset_name,
                    "n": len(subset_ranks),
                    "macro_author_top1": m["macro_author"]["top1"],
                    "micro_top1": m["micro"]["top1"],
                    "top3": m["micro"]["top3"],
                    "top5": m["micro"]["top5"],
                    "mrr_at_10": m["micro"]["mrr_at_10"],
                    "missing10": m["micro"]["missing10"],
                }
            )
        for author, m in metrics["per_author"].items():
            per_author_rows.append(
                {
                    "method": method,
                    "author": author,
                    "n": m["n"],
                    "top1": m["top1"],
                    "top3": m["top3"],
                    "top5": m["top5"],
                    "mrr_at_10": m["mrr_at_10"],
                    "missing10": m["missing10"],
                }
            )

    # Direct additive-vs-interaction comparisons.
    direct_pairs = (
        ("NGram+Recency", "NGramRecency_interaction"),
        ("BGE+Recency", "BGERecency_interaction"),
        ("BGE+NGram+Recency", "BGERecency+NGramRecency_interaction"),
    )
    for additive, interaction in direct_pairs:
        transition_rows.append(
            {
                "method": additive,
                "vs": interaction,
                **base.transition_counts(selected_ranks[interaction], selected_ranks[additive]),
                **base.rank_movement(selected_ranks[interaction], selected_ranks[additive], opportunity_mask),
            }
        )

    # Row-level traces only for selected methods.
    prediction_rows: list[dict[str, Any]] = []
    for i, rid in enumerate(row_ids):
        candidates = candidates_all[i]
        pv1_scores = pv1_scores_all[i]
        k = len(candidates)
        bge = bge_matrix[i, :k].tolist()
        ng = ng_matrix[i, :k].tolist()
        rec = rec_matrix[i, :k].tolist()
        ngr = ngr_matrix[i, :k].tolist()
        bger = bger_matrix[i, :k].tolist()
        methods: dict[str, Any] = {}
        for method in output_method_order:
            spec = selected_specs[method]
            if spec["type"] == "additive_independent_factors":
                ranking, final_scores = rerank_selected(
                    candidates,
                    pv1_scores,
                    bge,
                    ng,
                    rec,
                    lb=float(spec["lambda_bge"]),
                    ln=float(spec["lambda_ngram"]),
                    lr=float(spec["lambda_recency"]),
                )
            else:
                ranking, final_scores = rerank_interaction(
                    candidates,
                    pv1_scores,
                    bger,
                    ngr,
                    lb=float(spec["lambda_bge"]),
                    ln=float(spec["lambda_ngram"]),
                )
            methods[method] = {
                **spec,
                "gold_rank": base.rank_of(ranking, golds[i]),
                "ranking": list(ranking),
                "final_scores_in_ranking_order": list(final_scores),
            }
        prediction_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "row_id": rid,
                "author": authors[i],
                "gold": golds[i],
                "pv1_gold_rank": baseline_ranks[i],
                "supports_in_pv1_order": {
                    "candidates": list(candidates),
                    "pv1_final_score": list(pv1_scores),
                    "plain_bge": bge,
                    "plain_ngram": ng,
                    "recency_only": rec,
                    "ngram_recency": ngr,
                    "bge_recency": bger,
                },
                "methods": methods,
                "gold_used_for_scoring_features": False,
            }
        )

    # Latency: factor support and final selected rerank arithmetic only.
    rerank_latency: dict[str, Any] = {}
    for method in output_method_order:
        spec = selected_specs[method]
        values: list[float] = []
        for i in range(n_rows):
            k = len(candidates_all[i])
            t0 = time.perf_counter()
            if spec["type"] == "additive_independent_factors":
                rerank_selected(
                    candidates_all[i],
                    pv1_scores_all[i],
                    bge_matrix[i, :k],
                    ng_matrix[i, :k],
                    rec_matrix[i, :k],
                    lb=float(spec["lambda_bge"]),
                    ln=float(spec["lambda_ngram"]),
                    lr=float(spec["lambda_recency"]),
                )
            else:
                rerank_interaction(
                    candidates_all[i],
                    pv1_scores_all[i],
                    bger_matrix[i, :k],
                    ngr_matrix[i, :k],
                    lb=float(spec["lambda_bge"]),
                    ln=float(spec["lambda_ngram"]),
                )
            values.append((time.perf_counter() - t0) * 1000.0)
        rerank_latency[method] = base.latency_summary(values)

    context_diagnostics = {
        "same_pinyin_history": {
            "mean_rows": statistics.fmean(visible_counts),
            "p50_rows": base.percentile(visible_counts, 0.50),
            "p95_rows": base.percentile(visible_counts, 0.95),
            "max_rows": max(visible_counts),
        },
        "candidate_target_history": {
            "mean_rows": statistics.fmean(candidate_history_counts),
            "p95_rows": base.percentile(candidate_history_counts, 0.95),
        },
        "plain_ngram": {
            "max_n": args.max_n,
            "effective_n_distribution": dict(sorted(effective_n_counter.items())),
            "mean_matched_rows": statistics.fmean(matched_counts),
            "p95_matched_rows": base.percentile(matched_counts, 0.95),
        },
        "recency_only": {
            "tau": args.recency_tau,
            "age_unit": "same-author interactions; age=0 is immediately previous same-author interaction within H5000-before-Pinyin history",
            "context_gating": False,
        },
        "interaction_controls": {
            "ngram_recency_tau": args.ngram_recency_tau,
            "bge_recency_source": str(args.base_bge_recency_root.resolve()),
        },
    }

    latency = {
        "factor_support_scorer_history_lookup_excluded": base.latency_summary(support_latency_ms),
        "selected_rerank_sort_only": rerank_latency,
        "full_grid_vectorized_seconds": time.perf_counter() - grid_started,
        "note": "BGE/BGERecency embedding latency is not re-measured; their supports are loaded from completed read-only V1/V2 artifacts.",
    }

    grid_path = args.output_root / "grid_results.csv"
    write_csv(grid_path, grid_rows)
    write_json(
        args.output_root / "family_selection.json",
        {
            "selection_population": "ALL Train-Val",
            "selection_metric": "Macro-author Top1",
            "tie_break": "MRR@10, Top3, lower total weight, lower BGE, lower NGram, lower Recency",
            "additive_selected": selected,
            "interaction_controls_from_v2": interaction_specs,
            "gold_used_for_scoring_features": False,
            "dev3000_used": False,
            "test_used": False,
        },
    )
    write_csv(args.output_root / "selected_method_metrics.csv", selected_method_rows)
    write_csv(args.output_root / "selected_subset_metrics.csv", subset_rows)
    write_csv(args.output_root / "per_author_metrics.csv", per_author_rows)
    write_csv(args.output_root / "recovery_metrics.csv", recovery_rows)
    write_csv(args.output_root / "rank_transition_metrics.csv", transition_rows)
    write_jsonl(args.output_root / "factor_supports.jsonl", support_rows)
    write_jsonl(args.output_root / "selected_predictions.jsonl", prediction_rows)
    write_json(args.output_root / "context_diagnostics.json", context_diagnostics)
    write_json(args.output_root / "latency.json", latency)

    comparison = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "experiment": EXPERIMENT,
        "purpose": "Separate semantic context, lexical context, and same-user interaction recency into independent additive factors and compare them with the previously selected context-by-recency interaction scorers.",
        "evaluation_population": {
            "train_val_rows": n_rows,
            "authors": author_names,
            "recovery_population_definition": "Generic Missing AND Gold in frozen Personal K5",
            "recovery_population_n": int(sum(recovery_mask_list)),
        },
        "formulas": {
            "additive_full": "S = S_PV1 + lambda_B*P_BGE + lambda_N*P_NG + lambda_R*P_R",
            "plain_ngram": "largest exact suffix k<=maxN with candidate-target evidence; count matched histories; no recency",
            "recency_only": "raw_R(c)=sum_{h:target=c} exp(-age(h)/tau_R); no context matching",
            "ngram_recency_interaction": "suffix gating then sum exp(-age/tau_NR) over matched target histories",
            "bge_recency_interaction": "Top-5 by cosine then sum max(0,cosine)*exp(-age/tau_B)",
        },
        "weight_grids": {
            "lambda_bge": list(args.lambda_bge),
            "lambda_ngram": list(args.lambda_ngram),
            "lambda_recency": list(args.lambda_recency),
        },
        "selected_additive": selected,
        "selected_method_metrics": selected_method_rows,
        "recovery_metrics": recovery_rows,
        "interaction_controls_from_v2": interaction_specs,
        "invariants": {
            "candidate_set_changed": False,
            "missing10_invariant": True,
            "recovery10_invariant": True,
            "pv1_zero_weight_control_exact": True,
            "frozen_pv1_final_scores_preserved_as_base": True,
            "h5000_before_exact_pinyin_filter": True,
            "strictly_prior_same_author_history": True,
            "gold_used_for_scoring_features": False,
            "dev3000_used": False,
            "test_used": False,
        },
        "prior_provenance": provenance,
        "runtime_seconds": time.perf_counter() - total_started,
    }
    write_json(args.output_root / "comparison.json", comparison)

    files = [
        args.output_root / "run_manifest.json",
        grid_path,
        args.output_root / "family_selection.json",
        args.output_root / "selected_method_metrics.csv",
        args.output_root / "selected_subset_metrics.csv",
        args.output_root / "per_author_metrics.csv",
        args.output_root / "recovery_metrics.csv",
        args.output_root / "rank_transition_metrics.csv",
        args.output_root / "factor_supports.jsonl",
        args.output_root / "selected_predictions.jsonl",
        args.output_root / "context_diagnostics.json",
        args.output_root / "latency.json",
        args.output_root / "comparison.json",
    ]
    write_json(
        args.output_root / "artifact_checksums.json",
        {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "artifacts": {path.name: base.sha256_file(path) for path in files},
        },
    )

    print("\n=== SELECTED ADDITIVE FACTORIAL CONFIGURATIONS ===", flush=True)
    for family in families:
        row = selected[family]
        print(
            f"{family:24s} B={float(row['lambda_bge']):g} "
            f"N={float(row['lambda_ngram']):g} R={float(row['lambda_recency']):g}",
            flush=True,
        )
    gb = selected["GlobalBestAdditiveIncludingZeros"]
    print(
        f"{'GlobalBestAdditive':24s} family={gb['family']} B={float(gb['lambda_bge']):g} "
        f"N={float(gb['lambda_ngram']):g} R={float(gb['lambda_recency']):g}",
        flush=True,
    )

    print("\n=== COMPLETE ===", flush=True)
    for row in selected_method_rows:
        print(
            f"{row['method']:38s} "
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
            f"{row['method']:38s} "
            f"Rec1={float(row['rec1']):.4f} Rec3={float(row['rec3']):.4f} "
            f"Rec5={float(row['rec5']):.4f} Rec10={float(row['rec10']):.4f} "
            f"RecMRR={float(row['recovery_mrr_at_10']):.4f}",
            flush=True,
        )
    print(f"\nOutputs: {args.output_root.resolve()}", flush=True)
    print("Gold used for scoring/features: false", flush=True)
    print("Dev3000 used: false", flush=True)
    print("Test used: false", flush=True)
    print(f"Total wall time: {time.perf_counter() - total_started:.1f} s", flush=True)


def v2_comparison_metrics(root: Path) -> list[dict[str, Any]]:
    comparison = load_json(root / "comparison.json")
    rows = comparison.get("selected_method_metrics")
    if not isinstance(rows, list):
        raise RuntimeError("V2 comparison.json lacks selected_method_metrics")
    return [dict(row) for row in rows]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--candidate-surface", type=Path, required=True)
    parser.add_argument("--frequency-pv1-predictions", type=Path, required=True)
    parser.add_argument("--base-bge-root", type=Path, required=True)
    parser.add_argument("--base-bge-recency-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-n", type=int, default=DEFAULT_MAX_N)
    parser.add_argument("--recency-tau", type=float, default=DEFAULT_TAU_R)
    parser.add_argument("--ngram-recency-tau", type=float, default=DEFAULT_TAU_NR)
    parser.add_argument("--lambda-bge", nargs="+", type=float, default=list(DEFAULT_LAMBDA_BGE))
    parser.add_argument("--lambda-ngram", nargs="+", type=float, default=list(DEFAULT_LAMBDA_NGRAM))
    parser.add_argument("--lambda-recency", nargs="+", type=float, default=list(DEFAULT_LAMBDA_RECENCY))
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--grid-progress-every", type=int, default=50)
    parser.add_argument("--allow-unfrozen-input-hashes", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace) -> None:
    for name in ("fit", "val", "candidate_surface", "frequency_pv1_predictions"):
        path = getattr(args, name)
        if not path.is_file():
            raise RuntimeError(f"Input file does not exist: --{name.replace('_', '-')} {path}")
    for name in ("base_bge_root", "base_bge_recency_root"):
        path = getattr(args, name)
        if not path.is_dir():
            raise RuntimeError(f"Required prior result directory does not exist: --{name.replace('_', '-')} {path}")
    if args.output_root.resolve() in {
        args.base_bge_root.resolve(),
        args.base_bge_recency_root.resolve(),
    }:
        raise RuntimeError("--output-root must differ from read-only prior result roots")
    if args.max_n <= 0:
        raise ValueError("--max-n must be positive")
    if args.recency_tau <= 0 or args.ngram_recency_tau <= 0:
        raise ValueError("Recency tau values must be positive")
    if args.progress_every <= 0 or args.grid_progress_every <= 0:
        raise ValueError("Progress intervals must be positive")
    args.lambda_bge = tuple(sorted(set(float(v) for v in args.lambda_bge)))
    args.lambda_ngram = tuple(sorted(set(float(v) for v in args.lambda_ngram)))
    args.lambda_recency = tuple(sorted(set(float(v) for v in args.lambda_recency)))
    for grid_name, grid in (
        ("lambda-bge", args.lambda_bge),
        ("lambda-ngram", args.lambda_ngram),
        ("lambda-recency", args.lambda_recency),
    ):
        if 0.0 not in grid or not any(v > 0 for v in grid):
            raise ValueError(f"--{grid_name} must include 0 and at least one positive value")
        if any(v < 0 for v in grid):
            raise ValueError(f"--{grid_name} must be non-negative")


def main() -> None:
    args = build_parser().parse_args()
    normalize_args(args)
    evaluate(args)


if __name__ == "__main__":
    main()
