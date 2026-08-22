"""Full-specific retuning of the transferred Initial-final architecture, then Dev3000 comparison.

Scientific role
---------------
This is a NEW post-Dev follow-up. It does not modify the already-completed
seven-system standardized comparison. It gives the transferred architecture a
Full-specific Train-Val selection opportunity, then freezes the selected point
before evaluating it on the existing Dev3000 development surface.

The script is deliberately two-phase:

    --phase tune   Train-Val ONLY. Select and freeze one Full-retuned config.
    --phase dev    Require the frozen selection; evaluate on Dev3000; no tuning.

Test is never accepted or read.

Fixed architecture / non-tuned choices
--------------------------------------
- Frequency lambda = 4
- Personal K = 5
- P_NG: InterpolatedNGramRecency, maxN=2, kappa=1, tau=2048
- Stage2 NGramRecency: maxN=2, tau=2048
- Stage2 BGERecency: last64 context, candidate-conditioned, cosine Top5,
  tau=2048, recency only in aggregation
- causal history: same author -> strictly prior -> latest H5000 raw -> exact Pinyin
- empty Generic surface: conservative no-op (no personal injection / no Stage2)

Pre-specified sequential search
-------------------------------
Stage1 search (48 points):
    w_P  in {0,2,4,6}
    w_CS in {0,2,4,6}
    w_E  in {0,2,4}
Select by Macro-author Top1, then Micro Top1, MRR@10, then closest to the
transferred (4,4,2) point, then lexicographic order.

Stage2 search after Stage1 is frozen (25 points):
    lambda_N in {0,2,4,6,8}
    lambda_B in {0,2,4,6,8}
Select by the same metric ordering, then closest to transferred (4,6), then
lexicographic order.

The Dev phase evaluates both:
- zero-shot transferred point: (4,4,2) + (4,6)
- Full-retuned frozen point selected by --phase tune
and compares them against the existing standardized Dev methods.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Frozen provenance / protocol constants
# ---------------------------------------------------------------------------

EXPECTED_BASE_RUNNER_SHA256 = "f75d40f381e966f85cd4b20647ba7dc6a95df9116ad8657ca9a07505949a37b0"
EXPECTED_FROZEN_DEV_SHA256 = "9181f895eb19d0c36852e511263bfaefb34459dcd44efa6f45a44252e6b03f93"
EXPECTED_STANDARDIZED_DEV_PREDICTIONS_SHA256 = "dd219bfcb28fcad6a65f31eb14ddb16fc03c80f54a8b62a1cfe2504113c84233"
EXPECTED_PILOT_HISTORY_SHA256 = "7c85c38728d03985856d742f452992b3b3072af5f1c07845e099d9d07854da68"
EXPECTED_PILOT_DEV_SHA256 = "cf072d9323328b77e3d47d8a0c1beed8c40edc8767e075fb58593d6b72120606"
EXPECTED_DEV_ROWS = 3000
EXPECTED_DEV_AUTHORS = {"Agent Phage": 1000, "Etinjat": 1000, "breaddddd": 1000}

STAGE1_P_GRID = (0.0, 2.0, 4.0, 6.0)
STAGE1_CS_GRID = (0.0, 2.0, 4.0, 6.0)
STAGE1_E_GRID = (0.0, 2.0, 4.0)
LAMBDA_N_GRID = (0.0, 2.0, 4.0, 6.0, 8.0)
LAMBDA_B_GRID = (0.0, 2.0, 4.0, 6.0, 8.0)

ZERO_STAGE1 = (4.0, 4.0, 2.0)
ZERO_STAGE2 = (4.0, 6.0)

EXPECTED_ZERO_TRAINVAL_STAGE1_MACRO = 0.7885230322894308
EXPECTED_ZERO_TRAINVAL_FINAL_MACRO = 0.7953665798307687

SELECTION_RULE = (
    "maximize Macro-author Top1; tie-break maximize Micro Top1; then maximize MRR@10; "
    "then minimize L1 distance to transferred point; then lexicographic parameter order"
)


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl_plain(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                result.append(json.loads(line))
    return result


def index_rows(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for value in rows:
        row = dict(value)
        row_id = str(row["row_id"])
        if row_id in out:
            raise RuntimeError(f"Duplicate row_id in {label}: {row_id}")
        out[row_id] = row
    return out


def reject_test_rows(rows: Sequence[Mapping[str, Any]], label: str) -> None:
    for row in rows:
        if str(row.get("source_split", "")).lower() == "test":
            raise RuntimeError(f"STOP: Test row detected in {label}: {row.get('row_id')}")
        if bool(row.get("used_test", False)):
            raise RuntimeError(f"STOP: used_test=true detected in {label}: {row.get('row_id')}")


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def load_base_runner(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != EXPECTED_BASE_RUNNER_SHA256:
        raise RuntimeError(
            "Base Full-transfer runner SHA changed. This retune runner intentionally reuses the exact "
            "frozen implementation semantics.\n"
            f"expected={EXPECTED_BASE_RUNNER_SHA256}\nactual={actual}\npath={path}"
        )
    repo_root = path.resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    name = "_full_transfer_initial_final_frozen_base"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def metric_summary(rows: Sequence[Mapping[str, Any]], rank_key: str, method: str) -> dict[str, Any]:
    if not rows:
        return {"method": method, "n": 0}
    by_author: dict[str, list[int | None]] = defaultdict(list)
    ranks: list[int | None] = []
    for row in rows:
        value = row.get(rank_key)
        rank = None if value is None else int(value)
        ranks.append(rank)
        by_author[str(row["author"])].append(rank)

    def top_at(values: Sequence[int | None], k: int) -> float:
        return sum(rank is not None and rank <= k for rank in values) / len(values)

    found = [rank for rank in ranks if rank is not None]
    per_author = {author: top_at(values, 1) for author, values in sorted(by_author.items())}
    return {
        "method": method,
        "n": len(rows),
        "macro_author_top1": statistics.fmean(per_author.values()),
        "micro_top1": top_at(ranks, 1),
        "top3": top_at(ranks, 3),
        "top5": top_at(ranks, 5),
        "mrr_at_10": sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / len(ranks),
        "missing10": sum(rank is None for rank in ranks) / len(ranks),
        "mean_rank_given_top10": statistics.fmean(found) if found else None,
        "per_author_top1": per_author,
    }


def transition_counts(rows: Sequence[Mapping[str, Any]], before_key: str, after_key: str) -> dict[str, int]:
    result = {"n": len(rows), "rescue": 0, "harm": 0, "unchanged_correct": 0, "unchanged_wrong": 0}
    for row in rows:
        before = row.get(before_key)
        after = row.get(after_key)
        before_correct = before is not None and int(before) == 1
        after_correct = after is not None and int(after) == 1
        if (not before_correct) and after_correct:
            result["rescue"] += 1
        elif before_correct and (not after_correct):
            result["harm"] += 1
        elif before_correct and after_correct:
            result["unchanged_correct"] += 1
        else:
            result["unchanged_wrong"] += 1
    result["net"] = result["rescue"] - result["harm"]
    return result


def recovery_summary(rows: Sequence[Mapping[str, Any]], rank_key: str) -> dict[str, Any]:
    eligible = [row for row in rows if bool(row.get("generic_missing")) and bool(row.get("gold_in_personal_k5"))]
    ranks = [None if row.get(rank_key) is None else int(row[rank_key]) for row in eligible]
    n = len(ranks)
    def rec(k: int) -> int:
        return sum(rank is not None and rank <= k for rank in ranks)
    return {
        "eligible_n": n,
        "rec1_n": rec(1), "rec1": rec(1) / n if n else None,
        "rec3_n": rec(3), "rec3": rec(3) / n if n else None,
        "rec5_n": rec(5), "rec5": rec(5) / n if n else None,
        "rec10_n": rec(10), "rec10": rec(10) / n if n else None,
        "mrr_at_10": (sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / n if n else None),
    }


def subset(rows: Sequence[Mapping[str, Any]], name: str) -> list[Mapping[str, Any]]:
    if name == "overall":
        return list(rows)
    return [row for row in rows if bool(row.get(name))]


def selection_sort_key(metrics: Mapping[str, Any], params: Sequence[float], reference: Sequence[float]) -> tuple[Any, ...]:
    distance = sum(abs(float(a) - float(b)) for a, b in zip(params, reference))
    return (
        -float(metrics["macro_author_top1"]),
        -float(metrics["micro_top1"]),
        -float(metrics["mrr_at_10"]),
        float(distance),
        tuple(float(value) for value in params),
    )


# ---------------------------------------------------------------------------
# Parameterized architecture pieces (same semantics as frozen base runner)
# ---------------------------------------------------------------------------


def merge_stage1(
    base: Any,
    *,
    generic_rows: Sequence[Mapping[str, Any]],
    personal_k5: Sequence[str],
    p_ng: Mapping[str, float],
    choice_share: Mapping[str, float],
    entropy: float,
    w_p: float,
    w_cs: float,
    w_e: float,
) -> list[dict[str, Any]]:
    if not generic_rows:
        return []
    generic_texts = {base.candidate_text(row) for row in generic_rows}
    if generic_texts.intersection(personal_k5):
        raise RuntimeError("Personal K5 overlaps Generic surface")
    boundary = min(float(row["normalized_generic_score"]) for row in generic_rows)
    tiebreak = base.ngram_rank_map(personal_k5, p_ng)
    rows: list[dict[str, Any]] = [dict(row) for row in generic_rows]
    for original_rank, candidate in enumerate(personal_k5, start=1):
        score = (
            boundary
            + float(w_p) * float(p_ng[candidate])
            + float(w_cs) * float(choice_share[candidate])
            + float(w_e) * float(entropy)
        )
        rows.append({
            "candidate": candidate,
            "source": "personal_recovery",
            "generic_rank": None,
            "personal_candidate_rank": int(tiebreak[candidate]),
            "original_personal_frequency_rank": original_rank,
            "ngram_rank": int(tiebreak[candidate]),
            "final_score": score,
            "base_tiebreak_rank": int(tiebreak[candidate]),
            "p_ng": float(p_ng[candidate]),
            "choice_share": float(choice_share[candidate]),
            "entropy_concentration": float(entropy),
            "w_p": float(w_p), "w_cs": float(w_cs), "w_e": float(w_e),
        })
    rows.sort(key=lambda row: (
        -float(row["final_score"]),
        0 if row["source"] == "generic_frequency" else 1,
        int(row.get("generic_rank") or row.get("personal_candidate_rank") or row.get("rank") or 0),
        str(row["candidate"]),
    ))
    rows = rows[:10]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["base_rank"] = rank
        row["base_score"] = float(row["final_score"])
    return rows


def final_rerank(
    base: Any,
    *,
    stage1: Sequence[Mapping[str, Any]],
    ngram_support: Mapping[str, float],
    bge_support: Mapping[str, float],
    lambda_n: float,
    lambda_b: float,
) -> list[dict[str, Any]]:
    if not stage1:
        return []
    base_texts = [base.candidate_text(item) for item in stage1]
    if set(base_texts) != set(ngram_support) or set(base_texts) != set(bge_support):
        raise RuntimeError("Stage-2 support candidate set differs from Stage-1 Top10")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(stage1, start=1):
        row = dict(item)
        text = base.candidate_text(item)
        base_rank = int(item.get("base_rank", item.get("rank", index)))
        base_score = float(item["final_score"])
        n_score = float(ngram_support[text])
        b_score = float(bge_support[text])
        row["base_rank"] = base_rank
        row["base_score"] = base_score
        row["ngram_recency_support"] = n_score
        row["bge_recency_support"] = b_score
        row["lambda_n"] = float(lambda_n)
        row["lambda_b"] = float(lambda_b)
        row["final_score"] = base_score + float(lambda_n) * n_score + float(lambda_b) * b_score
        rows.append(row)
    rows.sort(key=lambda row: (-float(row["final_score"]), int(row["base_rank"]), str(row["candidate"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


# ---------------------------------------------------------------------------
# BGE cache helper
# ---------------------------------------------------------------------------


def fill_bge_vectors(
    base: Any,
    *,
    contexts: set[str],
    cache_path: Path,
    seed_cache: Path | None,
    bge_model: Path,
    cuda_path: Path | None,
    progress_every: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists() and seed_cache is not None and seed_cache.is_file():
        shutil.copy2(seed_cache, cache_path)
    base.ensure_cuda_path(cuda_path)
    from src.personalisation.pilot_a import BGEContextEmbedder

    cache = base.VectorCache(cache_path)
    try:
        missing = [context for context in sorted(contexts) if cache.get(context) is None]
        before = cache.count()
        print(f"BGE contexts required={len(contexts)} cache_before={before} missing={len(missing)}", flush=True)
        if missing:
            embedder = BGEContextEmbedder(bge_model)
            _ = embedder.embed(missing[0] if missing else "测试")
            started = time.perf_counter()
            for number, context in enumerate(missing, start=1):
                cache.put(context, embedder.embed(context))
                if number % 100 == 0 or number == len(missing):
                    cache.commit()
                if progress_every > 0 and (number % progress_every == 0 or number == len(missing)):
                    print(f"BGE embed {number}/{len(missing)} rate={number/max(time.perf_counter()-started,1e-9):.2f}/s", flush=True)
        vectors: dict[str, np.ndarray] = {}
        for context in contexts:
            value = cache.get(context)
            if value is None:
                raise RuntimeError("BGE cache fill incomplete")
            vectors[context] = base.normalized_vector(value)
        info = {"path": str(cache_path.resolve()), "required": len(contexts), "cache_rows": cache.count(), "generated": len(missing)}
        return vectors, info
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# Train-Val feature preparation / tuning
# ---------------------------------------------------------------------------


def verify_tune_args(args: argparse.Namespace, base: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    required = (args.fit, args.val, args.generic, args.checkpoint, args.bge_model, args.standardized_stage1)
    if any(path is None for path in required):
        raise RuntimeError("--phase tune requires --fit --val --generic --checkpoint --bge-model --standardized-stage1")
    # Reuse the frozen base verifier exactly.
    shadow = argparse.Namespace(
        fit=args.fit, val=args.val, generic=args.generic, checkpoint=args.checkpoint,
        bge_model=args.bge_model, standardized_stage1=args.standardized_stage1,
    )
    return base.verify_inputs(shadow)


def prepare_trainval_features(
    args: argparse.Namespace,
    base: Any,
    fit_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    generic: Mapping[str, Mapping[str, Any]],
    m1_state: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], Any]:
    path = args.output_root / "tune" / "train_val_stage1_features.jsonl"
    history = base.CausalHistoryIndex([*fit_rows, *val_rows])
    if path.is_file():
        rows = base.read_jsonl(path)
        if len(rows) != base.EXPECTED_VAL_ROWS:
            raise RuntimeError("Existing Train-Val Stage1 feature artifact is incomplete")
        print(f"Reusing Train-Val Stage1 features: {len(rows)} rows", flush=True)
        return rows, history

    print("Loading PinyinGPT compatibility backend for Train-Val Personal-K5 ...", flush=True)
    backend = base.PinyinGPTConcatBackend(args.checkpoint, device=args.compatibility_device)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for number, val_row in enumerate(val_rows, start=1):
        row_id = str(val_row["row_id"])
        grow = generic[row_id]
        query = base.query_of(val_row)
        generic_candidates = base.candidates_of(grow)
        visible = history.visible_same_pinyin(
            author=str(val_row["author"]), position=int(val_row["chronological_position"]), pinyin=base.pinyin_of(val_row)
        )
        history_rows = [{
            "row_id": item.record.row_id,
            "author": item.record.author,
            "chronological_position": item.record.position,
            "pinyin_segments": list(item.record.pinyin),
            "target": item.record.target,
            "context": item.record.context,
        } for item in visible]
        flags = base.subset_membership(query, base.target_of(val_row), history_rows)
        for flag in ("ambiguous", "conflict"):
            if val_row.get(flag) is not None and bool(val_row[flag]) != bool(flags[flag]):
                raise RuntimeError(f"Fresh {flag} differs from standardized manifest at {row_id}")
        f_rows = base.frequency_rows(query=query, generic_candidates=generic_candidates, history_rows=history_rows)
        g_texts = {candidate.text for candidate in generic_candidates}
        personal_k5 = base.build_personal_k5(
            visible=visible, generic_texts=g_texts, pinyin=base.pinyin_of(val_row), backend=backend
        )
        counts = Counter(item.record.target for item in visible)
        total = sum(counts.values())
        choice_share = {candidate: (counts.get(candidate, 0) / total if total else 0.0) for candidate in personal_k5}
        p_ng = base.interpolated_ngram_recency(
            candidates=personal_k5, query_context=base.scoring_context(val_row, grow), visible=visible
        )
        entropy = base.entropy_concentration(counts)
        gold = base.target_of(val_row)
        generic_rank = grow.get("gold_rank")
        if generic_rank is None:
            generic_rank = base.context_rank_of(
                tuple({"candidate": candidate.text, "rank": candidate.generic_rank} for candidate in generic_candidates), gold
            )
        m1_rank = None
        if m1_state is not None and generic_candidates:
            retrieved = m1_state[row_id]["bge_top20"][:base.M1_TOP_N]
            m1_rank = base.context_rank_of(
                base.rank_from_retrieved(generic_candidates, retrieved, lambda_memory=base.M1_LAMBDA), gold
            )
        rows.append({
            "schema_version": 1,
            "row_id": row_id,
            "author": str(val_row["author"]),
            "gold": gold,
            "ambiguous": bool(flags["ambiguous"]),
            "conflict": bool(flags["conflict"]),
            "raw_history_count": history.raw_visible_count(author=str(val_row["author"]), position=int(val_row["chronological_position"])),
            "same_pinyin_history_count": len(visible),
            "generic_surface_empty": not bool(generic_candidates),
            "generic_missing": generic_rank is None,
            "gold_in_personal_k5": gold in set(personal_k5),
            "personal_k5": list(personal_k5),
            "choice_share": choice_share,
            "p_ng": p_ng,
            "entropy_concentration": entropy,
            "generic_frequency_candidates": f_rows,
            "Generic_rank": generic_rank,
            "Frequency_rank": base.rank_of(f_rows, gold),
            "M1_rank": m1_rank,
            "gold_used_for_scoring": False,
            "used_dev3000": False,
            "used_test": False,
        })
        if args.progress_every > 0 and (number % args.progress_every == 0 or number == len(val_rows)):
            print(f"Train-Val features {number}/{len(val_rows)} rate={number/max(time.perf_counter()-started,1e-9):.1f}/s", flush=True)
    write_jsonl(path, rows)
    return rows, history


def evaluate_stage1_grid(base: Any, features: Sequence[Mapping[str, Any]], output_root: Path) -> tuple[dict[str, float], list[dict[str, Any]]]:
    grid_results: list[dict[str, Any]] = []
    total = len(STAGE1_P_GRID) * len(STAGE1_CS_GRID) * len(STAGE1_E_GRID)
    done = 0
    for w_p in STAGE1_P_GRID:
        for w_cs in STAGE1_CS_GRID:
            for w_e in STAGE1_E_GRID:
                done += 1
                eval_rows: list[dict[str, Any]] = []
                for row in features:
                    ranking = merge_stage1(
                        base,
                        generic_rows=row["generic_frequency_candidates"],
                        personal_k5=row["personal_k5"], p_ng=row["p_ng"], choice_share=row["choice_share"],
                        entropy=float(row["entropy_concentration"]), w_p=w_p, w_cs=w_cs, w_e=w_e,
                    )
                    eval_rows.append({"author": row["author"], "rank": base.rank_of(ranking, str(row["gold"]))})
                metrics = metric_summary(eval_rows, "rank", "Stage1")
                grid_results.append({"w_p": w_p, "w_cs": w_cs, "w_e": w_e, "metrics": metrics})
                print(f"Stage1 grid {done}/{total}: P={w_p:g} CS={w_cs:g} E={w_e:g} Macro={metrics['macro_author_top1']:.6f}", flush=True)
    grid_results.sort(key=lambda item: selection_sort_key(item["metrics"], (item["w_p"], item["w_cs"], item["w_e"]), ZERO_STAGE1))
    best = grid_results[0]
    write_jsonl(output_root / "tune" / "stage1_grid.jsonl", grid_results)
    return {"w_p": best["w_p"], "w_cs": best["w_cs"], "w_e": best["w_e"]}, grid_results


def build_trainval_supports(
    args: argparse.Namespace,
    base: Any,
    features: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    history: Any,
    selected_stage1: Mapping[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    support_path = args.output_root / "tune" / "train_val_stage2_supports.jsonl"
    if support_path.is_file():
        rows = base.read_jsonl(support_path)
        if len(rows) != base.EXPECTED_VAL_ROWS:
            raise RuntimeError("Existing Train-Val Stage2 support artifact is incomplete")
        info_path = args.output_root / "tune" / "train_val_bge_cache_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
        print(f"Reusing Train-Val Stage2 supports: {len(rows)} rows", flush=True)
        return rows, info

    val_by_id = index_rows(val_rows, "Train-Val")
    selected_surfaces: dict[str, list[dict[str, Any]]] = {}
    zero_surfaces: dict[str, list[dict[str, Any]]] = {}
    required_contexts: set[str] = set()
    for row in features:
        row_id = str(row["row_id"])
        selected = merge_stage1(
            base, generic_rows=row["generic_frequency_candidates"], personal_k5=row["personal_k5"],
            p_ng=row["p_ng"], choice_share=row["choice_share"], entropy=float(row["entropy_concentration"]),
            w_p=float(selected_stage1["w_p"]), w_cs=float(selected_stage1["w_cs"]), w_e=float(selected_stage1["w_e"]),
        )
        zero = merge_stage1(
            base, generic_rows=row["generic_frequency_candidates"], personal_k5=row["personal_k5"],
            p_ng=row["p_ng"], choice_share=row["choice_share"], entropy=float(row["entropy_concentration"]),
            w_p=ZERO_STAGE1[0], w_cs=ZERO_STAGE1[1], w_e=ZERO_STAGE1[2],
        )
        selected_surfaces[row_id] = selected
        zero_surfaces[row_id] = zero
        union = {base.candidate_text(item) for item in selected} | {base.candidate_text(item) for item in zero}
        if not union:
            continue
        vrow = val_by_id[row_id]
        required_contexts.add(base.context_of(vrow)[-base.BGE_CONTEXT_CHARS:])
        visible = history.visible_same_pinyin(
            author=str(vrow["author"]), position=int(vrow["chronological_position"]), pinyin=base.pinyin_of(vrow)
        )
        for item in visible:
            if item.record.target in union:
                required_contexts.add(item.record.context[-base.BGE_CONTEXT_CHARS:])

    cache_path = args.output_root / "tune" / "bge_context_cache.sqlite3"
    vectors, cache_info = fill_bge_vectors(
        base, contexts=required_contexts, cache_path=cache_path, seed_cache=args.seed_bge_cache,
        bge_model=args.bge_model, cuda_path=args.cuda_path, progress_every=args.progress_every,
    )
    write_json(args.output_root / "tune" / "train_val_bge_cache_info.json", cache_info)

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for number, feature in enumerate(features, start=1):
        row_id = str(feature["row_id"])
        vrow = val_by_id[row_id]
        visible = history.visible_same_pinyin(
            author=str(vrow["author"]), position=int(vrow["chronological_position"]), pinyin=base.pinyin_of(vrow)
        )

        def supports(surface: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float], dict[str, float], int, int, dict[str, int]]:
            candidates = [base.candidate_text(item) for item in surface]
            if not candidates:
                return {}, {}, 0, 0, {}
            ng, effective_n, matched = base.ngram_recency_support(
                query_context=base.context_of(vrow), candidates=candidates, visible=visible
            )
            q_context = base.context_of(vrow)[-base.BGE_CONTEXT_CHARS:]
            bge, counts = base.bge_recency_support(
                query_vector=vectors[q_context], candidates=candidates, visible=visible, vectors=vectors
            )
            return ng, bge, effective_n, matched, counts

        selected_surface = selected_surfaces[row_id]
        zero_surface = zero_surfaces[row_id]
        s_ng, s_bge, effective_n, matched, b_counts = supports(selected_surface)
        if [base.candidate_text(x) for x in zero_surface] == [base.candidate_text(x) for x in selected_surface]:
            z_ng, z_bge = s_ng, s_bge
        else:
            z_ng, z_bge, _, _, _ = supports(zero_surface)
        gold = str(feature["gold"])
        rows.append({
            "row_id": row_id, "author": feature["author"], "gold": gold,
            "ambiguous": bool(feature["ambiguous"]), "conflict": bool(feature["conflict"]),
            "generic_missing": bool(feature["generic_missing"]), "gold_in_personal_k5": bool(feature["gold_in_personal_k5"]),
            "personal_k5": feature["personal_k5"],
            "Generic_rank": feature["Generic_rank"], "Frequency_rank": feature["Frequency_rank"], "M1_rank": feature["M1_rank"],
            "ZeroShotStage1_rank": base.rank_of(zero_surface, gold),
            "RetunedStage1_rank": base.rank_of(selected_surface, gold),
            "zero_stage1_candidates": zero_surface,
            "retuned_stage1_candidates": selected_surface,
            "zero_ngram_support": z_ng, "zero_bge_support": z_bge,
            "retuned_ngram_support": s_ng, "retuned_bge_support": s_bge,
            "ngram_effective_n": effective_n, "ngram_matched_history_rows": matched,
            "bge_history_counts": b_counts,
            "gold_used_for_scoring": False, "used_dev3000": False, "used_test": False,
        })
        if args.progress_every > 0 and (number % args.progress_every == 0 or number == len(features)):
            print(f"Train-Val supports {number}/{len(features)} rate={number/max(time.perf_counter()-started,1e-9):.1f}/s", flush=True)
    write_jsonl(support_path, rows)
    return rows, cache_info


def evaluate_stage2_grid(base: Any, supports: Sequence[Mapping[str, Any]], output_root: Path) -> tuple[dict[str, float], list[dict[str, Any]]]:
    grid_results: list[dict[str, Any]] = []
    total = len(LAMBDA_N_GRID) * len(LAMBDA_B_GRID)
    done = 0
    for lambda_n in LAMBDA_N_GRID:
        for lambda_b in LAMBDA_B_GRID:
            done += 1
            eval_rows: list[dict[str, Any]] = []
            for row in supports:
                ranking = final_rerank(
                    base, stage1=row["retuned_stage1_candidates"],
                    ngram_support=row["retuned_ngram_support"], bge_support=row["retuned_bge_support"],
                    lambda_n=lambda_n, lambda_b=lambda_b,
                )
                eval_rows.append({"author": row["author"], "rank": base.rank_of(ranking, str(row["gold"]))})
            metrics = metric_summary(eval_rows, "rank", "Final")
            grid_results.append({"lambda_n": lambda_n, "lambda_b": lambda_b, "metrics": metrics})
            print(f"Stage2 grid {done}/{total}: N={lambda_n:g} B={lambda_b:g} Macro={metrics['macro_author_top1']:.6f}", flush=True)
    grid_results.sort(key=lambda item: selection_sort_key(item["metrics"], (item["lambda_n"], item["lambda_b"]), ZERO_STAGE2))
    best = grid_results[0]
    write_jsonl(output_root / "tune" / "stage2_grid.jsonl", grid_results)
    return {"lambda_n": best["lambda_n"], "lambda_b": best["lambda_b"]}, grid_results


def finalize_trainval(
    args: argparse.Namespace,
    base: Any,
    supports: Sequence[Mapping[str, Any]],
    selected_stage1: Mapping[str, float],
    selected_stage2: Mapping[str, float],
    stage1_grid: Sequence[Mapping[str, Any]],
    stage2_grid: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    cache_info: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in supports:
        gold = str(row["gold"])
        zero_final = final_rerank(
            base, stage1=row["zero_stage1_candidates"], ngram_support=row["zero_ngram_support"],
            bge_support=row["zero_bge_support"], lambda_n=ZERO_STAGE2[0], lambda_b=ZERO_STAGE2[1],
        )
        retuned_final = final_rerank(
            base, stage1=row["retuned_stage1_candidates"], ngram_support=row["retuned_ngram_support"],
            bge_support=row["retuned_bge_support"], lambda_n=float(selected_stage2["lambda_n"]), lambda_b=float(selected_stage2["lambda_b"]),
        )
        rows.append({
            "row_id": row["row_id"], "author": row["author"], "gold": gold,
            "ambiguous": row["ambiguous"], "conflict": row["conflict"],
            "generic_missing": row["generic_missing"], "gold_in_personal_k5": row["gold_in_personal_k5"],
            "personal_k5": row["personal_k5"],
            "Generic_rank": row["Generic_rank"], "Frequency_rank": row["Frequency_rank"], "M1_rank": row["M1_rank"],
            "ZeroShotStage1_rank": row["ZeroShotStage1_rank"],
            "ZeroShotFinal_rank": base.rank_of(zero_final, gold),
            "RetunedStage1_rank": row["RetunedStage1_rank"],
            "RetunedFinal_rank": base.rank_of(retuned_final, gold),
            "RetunedStage1_top10": [base.candidate_text(item) for item in row["retuned_stage1_candidates"]],
            "RetunedFinal_top10": [base.candidate_text(item) for item in retuned_final],
            "ngram_effective_n": row["ngram_effective_n"],
            "gold_used_for_scoring": False, "used_dev3000": False, "used_test": False,
        })
    pred_path = args.output_root / "tune" / "train_val_selected_predictions.jsonl"
    write_jsonl(pred_path, rows)

    methods = ["Generic", "Frequency", "M1", "ZeroShotStage1", "ZeroShotFinal", "RetunedStage1", "RetunedFinal"]
    metrics: dict[str, Any] = {}
    for method in methods:
        if method == "M1" and all(row.get("M1_rank") is None for row in rows):
            continue
        rank_key = f"{method}_rank"
        metrics[method] = {
            name: metric_summary(subset(rows, name), rank_key, method)
            for name in ("overall", "ambiguous", "conflict")
        }

    # Exact regression against the already-completed zero-shot Train-Val run.
    z_s1 = metrics["ZeroShotStage1"]["overall"]["macro_author_top1"]
    z_final = metrics["ZeroShotFinal"]["overall"]["macro_author_top1"]
    if abs(z_s1 - EXPECTED_ZERO_TRAINVAL_STAGE1_MACRO) > 1e-9:
        raise RuntimeError(f"Zero-shot Stage1 regression mismatch: {z_s1} vs {EXPECTED_ZERO_TRAINVAL_STAGE1_MACRO}")
    if abs(z_final - EXPECTED_ZERO_TRAINVAL_FINAL_MACRO) > 1e-9:
        raise RuntimeError(f"Zero-shot Final regression mismatch: {z_final} vs {EXPECTED_ZERO_TRAINVAL_FINAL_MACRO}")

    selected = {
        "schema_version": 1,
        "status": "FULL_TRAINVAL_SELECTED",
        "experiment": "full_retune_final_trainval_dev_v1",
        "scientific_status": "Full-specific Train-Val selection for a post-Dev follow-up; Dev3000 not used for parameter selection; Test closed",
        "selection_population": "standardized Full+Short Clean3 Train-Val",
        "selection_rule": SELECTION_RULE,
        "search_space": {
            "stage1": {"w_p": list(STAGE1_P_GRID), "w_cs": list(STAGE1_CS_GRID), "w_e": list(STAGE1_E_GRID), "points": len(STAGE1_P_GRID)*len(STAGE1_CS_GRID)*len(STAGE1_E_GRID)},
            "stage2": {"lambda_n": list(LAMBDA_N_GRID), "lambda_b": list(LAMBDA_B_GRID), "points": len(LAMBDA_N_GRID)*len(LAMBDA_B_GRID)},
            "sequential": True,
        },
        "fixed": {
            "frequency_lambda": base.FREQUENCY_LAMBDA,
            "personal_k": base.PERSONAL_K,
            "p_ng": {"max_n": base.P_NG_MAX_N, "kappa": base.P_NG_KAPPA, "tau": base.P_NG_TAU},
            "ngram_recency": {"max_n": base.NGRAM_MAX_N, "tau": base.NGRAM_TAU},
            "bge_recency": {"context_chars": base.BGE_CONTEXT_CHARS, "top_n_per_candidate": base.BGE_TOP_N, "tau": base.BGE_TAU},
            "history": "same author -> strictly prior -> latest H5000 raw -> exact segmented-Pinyin",
            "empty_generic_surface_policy": "conservative no-op",
        },
        "zero_shot_reference": {"stage1": {"w_p": 4.0, "w_cs": 4.0, "w_e": 2.0}, "stage2": {"lambda_n": 4.0, "lambda_b": 6.0}},
        "selected_stage1": dict(selected_stage1),
        "selected_stage2": dict(selected_stage2),
        "selected_train_val_metrics": metrics["RetunedFinal"]["overall"],
        "zero_shot_train_val_metrics": metrics["ZeroShotFinal"]["overall"],
        "stage1_grid_best": stage1_grid[0],
        "stage2_grid_best": stage2_grid[0],
        "provenance": provenance,
        "bge_cache": dict(cache_info),
        "gold_used_for_candidate_construction_or_scoring": False,
        "gold_used_for_train_val_selection": True,
        "used_dev3000_for_selection": False,
        "used_test": False,
    }
    selected_path = args.output_root / "selected_config.json"
    write_json(selected_path, selected)

    result = {
        "schema_version": 1, "status": "complete", "phase": "tune",
        "scientific_status": selected["scientific_status"],
        "rows": len(rows), "metrics": metrics,
        "transitions": {
            "Frequency_to_RetunedFinal": transition_counts(rows, "Frequency_rank", "RetunedFinal_rank"),
            "M1_to_RetunedFinal": transition_counts(rows, "M1_rank", "RetunedFinal_rank") if "M1" in metrics else None,
            "RetunedStage1_to_RetunedFinal": transition_counts(rows, "RetunedStage1_rank", "RetunedFinal_rank"),
            "ZeroShotFinal_to_RetunedFinal": transition_counts(rows, "ZeroShotFinal_rank", "RetunedFinal_rank"),
        },
        "recovery": {
            "ZeroShotFinal": recovery_summary(rows, "ZeroShotFinal_rank"),
            "RetunedFinal": recovery_summary(rows, "RetunedFinal_rank"),
        },
        "selected_config": selected,
        "predictions": str(pred_path.resolve()),
        "predictions_sha256": sha256_file(pred_path),
        "used_dev3000_for_selection": False,
        "used_test": False,
    }
    result_path = args.output_root / "tune" / "train_val_result.json"
    write_json(result_path, result)
    checksums = {
        "runner": sha256_file(Path(__file__)),
        "base_runner": sha256_file(args.base_runner),
        "selected_config.json": sha256_file(selected_path),
        "train_val_result.json": sha256_file(result_path),
        "train_val_selected_predictions.jsonl": sha256_file(pred_path),
        "used_dev3000_for_selection": False,
        "used_test": False,
    }
    write_json(args.output_root / "tune" / "artifact_checksums.json", checksums)
    return result


def run_tune(args: argparse.Namespace, base: Any) -> None:
    started = time.perf_counter()
    fit_rows, val_rows, generic, extra = verify_tune_args(args, base)
    args.output_root.mkdir(parents=True, exist_ok=True)
    setup_path = args.output_root / "run_setup.json"
    setup = {
        "schema_version": 1,
        "experiment": "full_retune_final_trainval_dev_v1",
        "selection_rule": SELECTION_RULE,
        "search_space": {
            "stage1": {"w_p": list(STAGE1_P_GRID), "w_cs": list(STAGE1_CS_GRID), "w_e": list(STAGE1_E_GRID)},
            "stage2": {"lambda_n": list(LAMBDA_N_GRID), "lambda_b": list(LAMBDA_B_GRID)},
        },
        "fixed_architecture": "Initial-final architecture; Full-specific weight calibration only",
        "base_runner": {"path": str(args.base_runner.resolve()), "sha256": sha256_file(args.base_runner)},
        "provenance": extra["provenance"],
        "used_dev3000_for_selection": False,
        "used_test": False,
    }
    if setup_path.is_file():
        previous = json.loads(setup_path.read_text(encoding="utf-8"))
        if previous != setup:
            raise RuntimeError("Existing output root belongs to a different tune setup; use a new versioned output root")
    else:
        if any(args.output_root.iterdir()):
            raise RuntimeError("Refusing non-empty retune output root without matching run_setup.json")
        write_json(setup_path, setup)

    print("=== FULL RETUNE V1: TRAIN-VAL SELECTION ONLY ===", flush=True)
    print("Primary selection metric: Macro-author Top1", flush=True)
    print("Dev3000 NOT read for selection. Test NOT read.", flush=True)
    features, history = prepare_trainval_features(args, base, fit_rows, val_rows, generic, extra["m1"])
    selected_stage1, stage1_grid = evaluate_stage1_grid(base, features, args.output_root)
    print(f"Selected Stage1: {selected_stage1}", flush=True)
    supports, cache_info = build_trainval_supports(args, base, features, val_rows, history, selected_stage1)
    selected_stage2, stage2_grid = evaluate_stage2_grid(base, supports, args.output_root)
    print(f"Selected Stage2: {selected_stage2}", flush=True)
    result = finalize_trainval(
        args, base, supports, selected_stage1, selected_stage2, stage1_grid, stage2_grid,
        extra["provenance"], cache_info,
    )
    result["runtime_seconds"] = time.perf_counter() - started
    tune_result_path = args.output_root / "tune" / "train_val_result.json"
    write_json(tune_result_path, result)
    # Refresh the checksum after adding runtime_seconds so the recorded digest
    # always matches the final on-disk result.
    checksum_path = args.output_root / "tune" / "artifact_checksums.json"
    checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
    checksums["train_val_result.json"] = sha256_file(tune_result_path)
    write_json(checksum_path, checksums)
    print("\n=== TRAIN-VAL SELECTION COMPLETE / FROZEN ===", flush=True)
    print(f"Selected config: {args.output_root / 'selected_config.json'}", flush=True)
    print(f"Retuned Final Macro={result['metrics']['RetunedFinal']['overall']['macro_author_top1']:.6f}", flush=True)
    print(f"Zero-shot Final Macro={result['metrics']['ZeroShotFinal']['overall']['macro_author_top1']:.6f}", flush=True)
    print(f"Runner SHA256: {sha256_file(Path(__file__))}", flush=True)


# ---------------------------------------------------------------------------
# Dev3000 frozen-config evaluation
# ---------------------------------------------------------------------------


def verify_dev_args(args: argparse.Namespace, base: Any) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    required = (args.pilot_history, args.pilot_dev, args.frozen_dev, args.standardized_dev_root, args.checkpoint, args.bge_model)
    if any(path is None for path in required):
        raise RuntimeError("--phase dev requires --pilot-history --pilot-dev --frozen-dev --standardized-dev-root --checkpoint --bge-model")
    selected_path = args.output_root / "selected_config.json"
    if not selected_path.is_file():
        raise RuntimeError("No frozen selected_config.json. Run --phase tune first.")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected.get("status") != "FULL_TRAINVAL_SELECTED" or selected.get("used_dev3000_for_selection") is not False or selected.get("used_test") is not False:
        raise RuntimeError("Frozen Train-Val selection record is invalid")
    if sha256_file(args.frozen_dev) != EXPECTED_FROZEN_DEV_SHA256:
        raise RuntimeError("Frozen Dev3000 manifest SHA changed")
    if sha256_file(args.pilot_history) != EXPECTED_PILOT_HISTORY_SHA256:
        raise RuntimeError("Pilot history manifest SHA changed")
    if sha256_file(args.pilot_dev) != EXPECTED_PILOT_DEV_SHA256:
        raise RuntimeError("Pilot Dev manifest SHA changed")
    if sha256_file(args.bge_model) != base.EXPECTED_BGE_SHA256:
        raise RuntimeError("Frozen BGE model SHA changed")

    standard_predictions_path = args.standardized_dev_root / "predictions.jsonl"
    generic_path = args.standardized_dev_root / "generic_predictions.jsonl"
    standard_result_path = args.standardized_dev_root / "standardized_dev3000_result.json"
    for path in (standard_predictions_path, generic_path, standard_result_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(standard_predictions_path) != EXPECTED_STANDARDIZED_DEV_PREDICTIONS_SHA256:
        raise RuntimeError("Standardized Dev comparator predictions SHA changed")
    standard_result = json.loads(standard_result_path.read_text(encoding="utf-8"))
    if standard_result.get("used_test") is not False or standard_result.get("frozen_dev_sha256") != EXPECTED_FROZEN_DEV_SHA256:
        raise RuntimeError("Standardized Dev result provenance changed")

    pilot_history = base.read_jsonl(args.pilot_history)
    pilot_dev = base.read_jsonl(args.pilot_dev)
    frozen_dev = base.read_jsonl(args.frozen_dev)
    reject_test_rows(pilot_history, "pilot history")
    reject_test_rows(pilot_dev, "pilot Dev")
    reject_test_rows(frozen_dev, "frozen Dev")
    if len(frozen_dev) != EXPECTED_DEV_ROWS:
        raise RuntimeError(f"Frozen Dev row count changed: {len(frozen_dev)}")
    author_counts = Counter(str(row["author"]) for row in frozen_dev)
    if dict(author_counts) != EXPECTED_DEV_AUTHORS:
        raise RuntimeError(f"Frozen Dev author balance changed: {author_counts}")

    pilot_by_id = index_rows(pilot_dev, "Pilot Dev")
    queries: list[dict[str, Any]] = []
    frozen_by_pilot: dict[str, dict[str, Any]] = {}
    for manifest_row in frozen_dev:
        pilot_id = str(manifest_row["pilot_row_id"])
        if pilot_id not in pilot_by_id:
            raise RuntimeError(f"Frozen Dev pilot_row_id missing from Pilot Dev: {pilot_id}")
        query = dict(pilot_by_id[pilot_id])
        query["canonical_row_id"] = str(manifest_row["canonical_row_id"])
        queries.append(query)
        frozen_by_pilot[pilot_id] = dict(manifest_row)

    generic = index_rows(base.read_jsonl(generic_path), "standardized Dev Generic")
    comparators = index_rows(base.read_jsonl(standard_predictions_path), "standardized Dev predictions")
    expected_ids = {str(row["row_id"]) for row in queries}
    if set(generic) != expected_ids or set(comparators) != expected_ids:
        raise RuntimeError(f"Standardized Dev row identity mismatch: generic={len(generic)} comparator={len(comparators)} expected={len(expected_ids)}")
    return selected, pilot_history, pilot_dev, queries, generic, comparators


def run_dev(args: argparse.Namespace, base: Any) -> None:
    started = time.perf_counter()
    selected, pilot_history, pilot_dev, queries, generic, comparators = verify_dev_args(args, base)
    selected_stage1 = selected["selected_stage1"]
    selected_stage2 = selected["selected_stage2"]
    frozen_manifest = {str(row["pilot_row_id"]): row for row in base.read_jsonl(args.frozen_dev)}
    history = base.CausalHistoryIndex([*pilot_history, *pilot_dev])

    dev_root = args.output_root / "dev"
    dev_root.mkdir(parents=True, exist_ok=True)
    dev_setup = {
        "schema_version": 1,
        "experiment": "full_retune_final_trainval_dev_v1",
        "phase": "dev",
        "scientific_status": "post-Dev Full-retuned follow-up; selected on Train-Val only; Dev3000 is a development comparison; Test closed",
        "selected_config": {"path": str((args.output_root / 'selected_config.json').resolve()), "sha256": sha256_file(args.output_root / "selected_config.json")},
        "selected_stage1": selected_stage1,
        "selected_stage2": selected_stage2,
        "zero_shot_reference": {"stage1": {"w_p": 4.0, "w_cs": 4.0, "w_e": 2.0}, "stage2": {"lambda_n": 4.0, "lambda_b": 6.0}},
        "frozen_dev": {"path": str(args.frozen_dev.resolve()), "sha256": sha256_file(args.frozen_dev)},
        "pilot_history": {"path": str(args.pilot_history.resolve()), "sha256": sha256_file(args.pilot_history)},
        "pilot_dev": {"path": str(args.pilot_dev.resolve()), "sha256": sha256_file(args.pilot_dev)},
        "standardized_dev_root": str(args.standardized_dev_root.resolve()),
        "standardized_predictions_sha256": sha256_file(args.standardized_dev_root / "predictions.jsonl"),
        "bge_model": {"path": str(args.bge_model.resolve()), "sha256": sha256_file(args.bge_model)},
        "gold_used_for_scoring": False,
        "hyperparameter_search_on_dev": False,
        "used_dev3000": True,
        "used_test": False,
    }
    setup_path = dev_root / "run_setup.json"
    if setup_path.is_file():
        previous = json.loads(setup_path.read_text(encoding="utf-8"))
        if previous != dev_setup:
            raise RuntimeError("Existing Dev output root belongs to a different frozen setup")
    else:
        residual = [path for path in dev_root.iterdir() if path.name != "run_setup.json"]
        if residual:
            raise RuntimeError("Refusing non-empty Dev output root without matching setup")
        write_json(setup_path, dev_setup)

    print("=== FULL RETUNE V1: FROZEN DEV3000 COMPARISON ===", flush=True)
    print(f"Frozen selected Stage1={selected_stage1} Stage2={selected_stage2}", flush=True)
    print("No Dev tuning. Test NOT read.", flush=True)

    print("Loading PinyinGPT compatibility backend for Dev Personal-K5 ...", flush=True)
    backend = base.PinyinGPTConcatBackend(args.checkpoint, device=args.compatibility_device)
    dev_features: list[dict[str, Any]] = []
    required_contexts: set[str] = set()
    started_features = time.perf_counter()
    for number, query_row in enumerate(queries, start=1):
        row_id = str(query_row["row_id"])
        grow = generic[row_id]
        comparator = comparators[row_id]
        query = base.query_of(query_row)
        generic_candidates = base.candidates_of(grow)
        visible = history.visible_same_pinyin(
            author=str(query_row["author"]), position=int(query_row["chronological_position"]), pinyin=base.pinyin_of(query_row)
        )
        frozen_row = frozen_manifest[row_id]
        if len(visible) != int(frozen_row["same_pinyin_history_count"]):
            raise RuntimeError(f"Dev same-Pinyin history count mismatch at {row_id}: {len(visible)} vs {frozen_row['same_pinyin_history_count']}")
        history_rows = [{
            "row_id": item.record.row_id, "author": item.record.author,
            "chronological_position": item.record.position, "pinyin_segments": list(item.record.pinyin),
            "target": item.record.target, "context": item.record.context,
        } for item in visible]
        gold = base.target_of(query_row)
        flags = base.subset_membership(query, gold, history_rows)
        for flag in ("ambiguous", "conflict"):
            if bool(flags[flag]) != bool(frozen_row[flag]):
                raise RuntimeError(f"Fresh Dev {flag} differs from frozen manifest at {row_id}")
        f_rows = base.frequency_rows(query=query, generic_candidates=generic_candidates, history_rows=history_rows)
        g_texts = {candidate.text for candidate in generic_candidates}
        personal_k5 = base.build_personal_k5(
            visible=visible, generic_texts=g_texts, pinyin=base.pinyin_of(query_row), backend=backend
        )
        counts = Counter(item.record.target for item in visible)
        total = sum(counts.values())
        choice_share = {candidate: (counts.get(candidate, 0) / total if total else 0.0) for candidate in personal_k5}
        p_ng = base.interpolated_ngram_recency(
            candidates=personal_k5, query_context=base.scoring_context(query_row, grow), visible=visible
        )
        entropy = base.entropy_concentration(counts)
        zero_s1 = merge_stage1(
            base, generic_rows=f_rows, personal_k5=personal_k5, p_ng=p_ng, choice_share=choice_share, entropy=entropy,
            w_p=ZERO_STAGE1[0], w_cs=ZERO_STAGE1[1], w_e=ZERO_STAGE1[2],
        )
        retuned_s1 = merge_stage1(
            base, generic_rows=f_rows, personal_k5=personal_k5, p_ng=p_ng, choice_share=choice_share, entropy=entropy,
            w_p=float(selected_stage1["w_p"]), w_cs=float(selected_stage1["w_cs"]), w_e=float(selected_stage1["w_e"]),
        )

        generic_rank = grow.get("gold_rank")
        if generic_rank is None:
            generic_rank = base.context_rank_of(
                tuple({"candidate": candidate.text, "rank": candidate.generic_rank} for candidate in generic_candidates), gold
            )
        frequency_rank = base.rank_of(f_rows, gold)
        # Exact comparator consistency check.
        if generic_rank != comparator.get("Generic_rank"):
            raise RuntimeError(f"Dev Generic rank mismatch at {row_id}: fresh={generic_rank} comparator={comparator.get('Generic_rank')}")
        if frequency_rank != comparator.get("Frequency_rank"):
            raise RuntimeError(f"Dev Frequency rank mismatch at {row_id}: fresh={frequency_rank} comparator={comparator.get('Frequency_rank')}")

        union = {base.candidate_text(item) for item in zero_s1} | {base.candidate_text(item) for item in retuned_s1}
        if union:
            required_contexts.add(base.context_of(query_row)[-base.BGE_CONTEXT_CHARS:])
            for item in visible:
                if item.record.target in union:
                    required_contexts.add(item.record.context[-base.BGE_CONTEXT_CHARS:])

        dev_features.append({
            "row_id": row_id, "canonical_row_id": query_row["canonical_row_id"],
            "author": str(query_row["author"]), "gold": gold,
            "ambiguous": bool(flags["ambiguous"]), "conflict": bool(flags["conflict"]),
            "generic_missing": generic_rank is None,
            "gold_in_personal_k5": gold in set(personal_k5),
            "personal_k5": list(personal_k5),
            "ZeroShotStage1_rank": base.rank_of(zero_s1, gold),
            "RetunedStage1_rank": base.rank_of(retuned_s1, gold),
            "zero_stage1_candidates": zero_s1,
            "retuned_stage1_candidates": retuned_s1,
            "comparator": {key: value for key, value in comparator.items() if key.endswith("_rank") or key in ("row_id", "author", "gold")},
        })
        if args.progress_every > 0 and (number % args.progress_every == 0 or number == len(queries)):
            print(f"Dev Stage1 features {number}/{len(queries)} rate={number/max(time.perf_counter()-started_features,1e-9):.1f}/s", flush=True)

    tune_cache = args.output_root / "tune" / "bge_context_cache.sqlite3"
    seed = args.seed_bge_cache if args.seed_bge_cache is not None else (tune_cache if tune_cache.is_file() else None)
    vectors, cache_info = fill_bge_vectors(
        base, contexts=required_contexts, cache_path=dev_root / "bge_context_cache.sqlite3", seed_cache=seed,
        bge_model=args.bge_model, cuda_path=args.cuda_path, progress_every=args.progress_every,
    )

    output_rows: list[dict[str, Any]] = []
    query_by_id = index_rows(queries, "selected Dev queries")
    started_final = time.perf_counter()
    standard_methods = ("Generic", "Frequency", "M1", "M2", "Hidden-M1", "Hidden-M2", "EM3")
    for number, feature in enumerate(dev_features, start=1):
        row_id = str(feature["row_id"])
        query_row = query_by_id[row_id]
        visible = history.visible_same_pinyin(
            author=str(query_row["author"]), position=int(query_row["chronological_position"]), pinyin=base.pinyin_of(query_row)
        )

        def supports(surface: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float], dict[str, float], int]:
            candidates = [base.candidate_text(item) for item in surface]
            if not candidates:
                return {}, {}, 0
            ng, effective_n, _ = base.ngram_recency_support(
                query_context=base.context_of(query_row), candidates=candidates, visible=visible
            )
            q_context = base.context_of(query_row)[-base.BGE_CONTEXT_CHARS:]
            bge, _ = base.bge_recency_support(
                query_vector=vectors[q_context], candidates=candidates, visible=visible, vectors=vectors
            )
            return ng, bge, effective_n

        zero_s1 = feature["zero_stage1_candidates"]
        retuned_s1 = feature["retuned_stage1_candidates"]
        z_ng, z_bge, z_effective_n = supports(zero_s1)
        if [base.candidate_text(x) for x in zero_s1] == [base.candidate_text(x) for x in retuned_s1]:
            r_ng, r_bge, effective_n = z_ng, z_bge, z_effective_n
        else:
            r_ng, r_bge, effective_n = supports(retuned_s1)
        zero_final = final_rerank(
            base, stage1=zero_s1, ngram_support=z_ng, bge_support=z_bge, lambda_n=ZERO_STAGE2[0], lambda_b=ZERO_STAGE2[1]
        )
        retuned_final = final_rerank(
            base, stage1=retuned_s1, ngram_support=r_ng, bge_support=r_bge,
            lambda_n=float(selected_stage2["lambda_n"]), lambda_b=float(selected_stage2["lambda_b"]),
        )
        gold = str(feature["gold"])
        row = {
            "schema_version": 1,
            "row_id": row_id, "canonical_row_id": feature["canonical_row_id"],
            "author": feature["author"], "gold": gold,
            "ambiguous": feature["ambiguous"], "conflict": feature["conflict"],
            "generic_missing": feature["generic_missing"], "gold_in_personal_k5": feature["gold_in_personal_k5"],
            "personal_k5": feature["personal_k5"],
            "ZeroShotStage1_rank": feature["ZeroShotStage1_rank"],
            "ZeroShotFinal_rank": base.rank_of(zero_final, gold),
            "RetunedStage1_rank": feature["RetunedStage1_rank"],
            "RetunedFinal_rank": base.rank_of(retuned_final, gold),
            "ZeroShotFinal_top10": [base.candidate_text(item) for item in zero_final],
            "RetunedFinal_top10": [base.candidate_text(item) for item in retuned_final],
            "ngram_effective_n": effective_n,
            "gold_used_for_scoring": False,
            "hyperparameter_search_on_dev": False,
            "used_dev3000": True,
            "used_test": False,
        }
        comparator = comparators[row_id]
        for method in standard_methods:
            row[f"{method}_rank"] = comparator.get(f"{method}_rank")
        output_rows.append(row)
        if args.progress_every > 0 and (number % args.progress_every == 0 or number == len(dev_features)):
            print(f"Dev Final {number}/{len(dev_features)} rate={number/max(time.perf_counter()-started_final,1e-9):.1f}/s", flush=True)

    pred_path = dev_root / "dev_predictions.jsonl"
    write_jsonl(pred_path, output_rows)
    methods = [*standard_methods, "ZeroShotStage1", "ZeroShotFinal", "RetunedStage1", "RetunedFinal"]
    metrics: dict[str, Any] = {}
    for method in methods:
        rank_key = f"{method}_rank"
        metrics[method] = {
            name: metric_summary(subset(output_rows, name), rank_key, method)
            for name in ("overall", "ambiguous", "conflict")
        }

    result = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "full_retune_final_trainval_dev_v1",
        "phase": "dev",
        "scientific_status": dev_setup["scientific_status"],
        "rows": len(output_rows),
        "selected_config": {"stage1": selected_stage1, "stage2": selected_stage2},
        "metrics": metrics,
        "transitions": {
            "Frequency_to_RetunedFinal": transition_counts(output_rows, "Frequency_rank", "RetunedFinal_rank"),
            "M1_to_RetunedFinal": transition_counts(output_rows, "M1_rank", "RetunedFinal_rank"),
            "RetunedStage1_to_RetunedFinal": transition_counts(output_rows, "RetunedStage1_rank", "RetunedFinal_rank"),
            "ZeroShotFinal_to_RetunedFinal": transition_counts(output_rows, "ZeroShotFinal_rank", "RetunedFinal_rank"),
        },
        "recovery": {
            "ZeroShotFinal": recovery_summary(output_rows, "ZeroShotFinal_rank"),
            "RetunedFinal": recovery_summary(output_rows, "RetunedFinal_rank"),
        },
        "bge_cache": cache_info,
        "predictions": str(pred_path.resolve()),
        "predictions_sha256": sha256_file(pred_path),
        "selected_config_sha256": sha256_file(args.output_root / "selected_config.json"),
        "gold_used_for_scoring": False,
        "hyperparameter_search_on_dev": False,
        "used_dev3000": True,
        "used_test": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    result_path = dev_root / "dev_result.json"
    write_json(result_path, result)
    checksums = {
        "runner": sha256_file(Path(__file__)),
        "base_runner": sha256_file(args.base_runner),
        "selected_config.json": sha256_file(args.output_root / "selected_config.json"),
        "dev_predictions.jsonl": sha256_file(pred_path),
        "dev_result.json": sha256_file(result_path),
        "used_dev3000": True,
        "used_test": False,
    }
    write_json(dev_root / "artifact_checksums.json", checksums)

    print("\n=== DEV3000 COMPARISON COMPLETE ===", flush=True)
    for method in methods:
        value = metrics[method]["overall"]
        print(
            f"{method:18s} Macro={value['macro_author_top1']:.6f} Micro={value['micro_top1']:.6f} "
            f"Top3={value['top3']:.6f} MRR={value['mrr_at_10']:.6f} Missing={value['missing10']:.6f}",
            flush=True,
        )
    print("Frequency -> RetunedFinal:", result["transitions"]["Frequency_to_RetunedFinal"], flush=True)
    print("M1 -> RetunedFinal:", result["transitions"]["M1_to_RetunedFinal"], flush=True)
    print("RetunedStage1 -> RetunedFinal:", result["transitions"]["RetunedStage1_to_RetunedFinal"], flush=True)
    print(f"Result: {result_path}", flush=True)
    print(f"Runner SHA256: {checksums['runner']}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("tune", "dev"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-runner", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bge-model", type=Path, required=True)
    parser.add_argument("--seed-bge-cache", type=Path, default=None)
    parser.add_argument("--compatibility-device", default="cpu")
    parser.add_argument("--cuda-path", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=500)

    # Train-Val inputs.
    parser.add_argument("--fit", type=Path, default=None)
    parser.add_argument("--val", type=Path, default=None)
    parser.add_argument("--generic", type=Path, default=None)
    parser.add_argument("--standardized-stage1", type=Path, default=None)

    # Dev-only inputs.
    parser.add_argument("--pilot-history", type=Path, default=None)
    parser.add_argument("--pilot-dev", type=Path, default=None)
    parser.add_argument("--frozen-dev", type=Path, default=None)
    parser.add_argument("--standardized-dev-root", type=Path, default=None)

    args = parser.parse_args()
    if args.base_runner is None:
        args.base_runner = Path(__file__).resolve().with_name("run_full_transfer_initial_final_v1.py")
    base = load_base_runner(args.base_runner)
    if args.phase == "tune":
        run_tune(args, base)
    else:
        run_dev(args, base)


if __name__ == "__main__":
    main()
