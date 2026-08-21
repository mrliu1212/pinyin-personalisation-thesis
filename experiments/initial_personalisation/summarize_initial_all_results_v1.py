#!/usr/bin/env python3
"""Summarize completed Initial-Pinyin personalisation experiments without inference.

This is a READ-ONLY reporting/audit utility. It recursively scans an existing
Initial-Pinyin result root, extracts known comparison schemas, recomputes metrics
from row-level rank files where useful, and writes unified CSV/JSON reports.

It NEVER runs PinyinGPT, Q8, BGE, NGram inference, and it does not read Dev3000
or Test. Missing fields are reported as NA rather than guessed.

Default result root (when run from thesis-initial-research):
    results/personalisation/initial_recovery_comparison_v1

Outputs:
    <output-dir>/overall_end_to_end.csv
    <output-dir>/recovery_metrics.csv
    <output-dir>/candidate_only.csv
    <output-dir>/missing_metrics_report.txt
    <output-dir>/all_results_summary.json
    <output-dir>/discovered_artifacts.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
EXPECTED_TRAIN_VAL_ROWS = 34416
EXPECTED_GENERIC_MISSING = 12565
EXPECTED_RECOVERABLE_K5 = 4910

OVERALL_FIELDS = (
    "macro_author_top1",
    "micro_top1",
    "top3",
    "top5",
    "mrr_at_10",
    "missing10",
    "mean_rank_given_top10",
)
RECOVERY_FIELDS = (
    "recovered_at_1",
    "recovered_at_3",
    "recovered_at_5",
    "recovered_at_10",
    "recovery_mrr_at_10",
    "mean_recovered_rank",
)
CANDIDATE_FIELDS = (
    "macro_author_top1",
    "micro_top1",
    "top3",
    "top5",
    "mrr",
    "mean_gold_rank",
    "mean_latency_ms",
    "p95_latency_ms",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_no, line in enumerate(source, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected object at {path}:{line_no}")
            rows.append(value)
    return rows


def safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def safe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def canonical_overall_metrics(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    metrics = metrics or {}
    return {
        "n": safe_int(first(metrics, "n", "rows")),
        "authors": safe_int(first(metrics, "authors", "authors_with_rows")),
        "macro_author_top1": safe_float(first(metrics, "macro_author_top1", "macro_top1")),
        "micro_top1": safe_float(first(metrics, "micro_top1", "top1")),
        "top3": safe_float(first(metrics, "top3", "micro_top3")),
        "top5": safe_float(first(metrics, "top5", "micro_top5")),
        "mrr_at_10": safe_float(first(metrics, "mrr_at_10", "mrr10", "micro_mrr10")),
        "missing10": safe_float(first(metrics, "missing10", "missing_at_10", "micro_missing10")),
        "mean_rank_given_top10": safe_float(first(metrics, "mean_rank_given_top10", "mean_gold_rank", "mean_rank")),
        "per_author_top1": metrics.get("per_author_top1") or metrics.get("per_author"),
    }


def canonical_recovery(recovery: Mapping[str, Any] | None) -> dict[str, Any]:
    recovery = recovery or {}
    return {
        "generic_missing_n": safe_int(recovery.get("generic_missing_n")),
        "recoverable_n": safe_int(recovery.get("recoverable_n")),
        "recovered_at_1": safe_float(first(recovery, "recovered_at_1", "rec_at_1")),
        "recovered_at_3": safe_float(first(recovery, "recovered_at_3", "rec_at_3")),
        "recovered_at_5": safe_float(first(recovery, "recovered_at_5", "rec_at_5")),
        "recovered_at_10": safe_float(first(recovery, "recovered_at_10", "rec_at_10")),
        "recovery_mrr_at_10": safe_float(first(recovery, "recovery_mrr_at_10", "recovery_mrr", "mrr_at_10")),
        "mean_recovered_rank": safe_float(first(recovery, "mean_recovered_rank", "mean_rank")),
    }


def canonical_transition(value: Mapping[str, Any] | None) -> dict[str, Any]:
    value = value or {}
    return {
        "rescue": safe_int(value.get("rescue")),
        "harm": safe_int(value.get("harm")),
        "net": safe_int(value.get("net")),
        "unchanged_correct": safe_int(value.get("unchanged_correct")),
        "unchanged_wrong": safe_int(value.get("unchanged_wrong")),
    }


def metric_from_ranks(rows: Sequence[Mapping[str, Any]], ranks: Sequence[int | None]) -> dict[str, Any]:
    if len(rows) != len(ranks):
        raise RuntimeError("rows/ranks length mismatch")
    author_total: Counter[str] = Counter()
    author_top1: Counter[str] = Counter()
    found: list[int] = []
    top1 = top3 = top5 = missing = 0
    reciprocal = 0.0
    for row, rank in zip(rows, ranks):
        author = str(row.get("author", ""))
        author_total[author] += 1
        if rank is None:
            missing += 1
            continue
        rank_i = int(rank)
        found.append(rank_i)
        reciprocal += 1.0 / rank_i
        if rank_i == 1:
            top1 += 1
            author_top1[author] += 1
        if rank_i <= 3:
            top3 += 1
        if rank_i <= 5:
            top5 += 1
    n = len(rows)
    per_author = {
        author: author_top1[author] / count
        for author, count in sorted(author_total.items())
        if count
    }
    return {
        "n": n,
        "authors": len(per_author),
        "macro_author_top1": statistics.fmean(per_author.values()) if per_author else None,
        "micro_top1": top1 / n if n else None,
        "top3": top3 / n if n else None,
        "top5": top5 / n if n else None,
        "mrr_at_10": reciprocal / n if n else None,
        "missing10": missing / n if n else None,
        "mean_rank_given_top10": statistics.fmean(found) if found else None,
        "per_author_top1": per_author,
    }


def recovery_from_ranks(
    rows: Sequence[Mapping[str, Any]],
    ranks: Sequence[int | None],
) -> dict[str, Any]:
    pairs: list[int | None] = []
    generic_missing_n = 0
    for row, rank in zip(rows, ranks):
        generic_missing = bool(row.get("generic_missing", False))
        if generic_missing:
            generic_missing_n += 1
        if generic_missing and bool(row.get("gold_in_personal_k5", False)):
            pairs.append(rank)
    denom = len(pairs)
    def at(k: int) -> int:
        return sum(rank is not None and int(rank) <= k for rank in pairs)
    present = [int(rank) for rank in pairs if rank is not None and int(rank) <= 10]
    reciprocal = sum(0.0 if rank is None else 1.0 / int(rank) for rank in pairs)
    return {
        "generic_missing_n": generic_missing_n,
        "recoverable_n": denom,
        "recovered_at_1": at(1) / denom if denom else None,
        "recovered_at_3": at(3) / denom if denom else None,
        "recovered_at_5": at(5) / denom if denom else None,
        "recovered_at_10": at(10) / denom if denom else None,
        "recovery_mrr_at_10": reciprocal / denom if denom else None,
        "mean_recovered_rank": statistics.fmean(present) if present else None,
    }


def transition_from_ranks(left: Sequence[int | None], right: Sequence[int | None]) -> dict[str, int]:
    if len(left) != len(right):
        raise RuntimeError("transition rank length mismatch")
    rescue = harm = unchanged_correct = unchanged_wrong = 0
    for a, b in zip(left, right):
        a_ok = a == 1
        b_ok = b == 1
        if (not a_ok) and b_ok:
            rescue += 1
        elif a_ok and (not b_ok):
            harm += 1
        elif a_ok and b_ok:
            unchanged_correct += 1
        else:
            unchanged_wrong += 1
    return {
        "rescue": rescue,
        "harm": harm,
        "net": rescue - harm,
        "unchanged_correct": unchanged_correct,
        "unchanged_wrong": unchanged_wrong,
    }


def completeness(record: Mapping[str, Any], fields: Sequence[str]) -> int:
    return sum(record.get(field) is not None for field in fields)


def merge_record(store: dict[str, dict[str, Any]], record: dict[str, Any], fields: Sequence[str]) -> None:
    key = str(record["method"])
    old = store.get(key)
    if old is None:
        store[key] = record
        return
    # Prefer the fuller record. On equal completeness, preserve the earlier one and
    # fill only missing values, while recording all sources.
    if completeness(record, fields) > completeness(old, fields):
        winner, other = record, old
    else:
        winner, other = old, record
    sources = list(dict.fromkeys([*(winner.get("sources") or [winner.get("source")]), *(other.get("sources") or [other.get("source")])]))
    for field in fields:
        if winner.get(field) is None and other.get(field) is not None:
            winner[field] = other[field]
    extra_fields = (
        "n", "authors", "rescue", "harm", "net", "unchanged_correct", "unchanged_wrong",
        "generic_missing_n", "recoverable_n", "mean_latency_ms", "p95_latency_ms",
        "configuration", "notes", *RECOVERY_FIELDS,
    )
    for field in extra_fields:
        if winner.get(field) is None and other.get(field) is not None:
            winner[field] = other[field]
    winner["sources"] = [s for s in sources if s]
    store[key] = winner


def overall_record(
    method: str,
    metrics: Mapping[str, Any] | None,
    *,
    source: Path,
    category: str,
    transition: Mapping[str, Any] | None = None,
    recovery: Mapping[str, Any] | None = None,
    configuration: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    m = canonical_overall_metrics(metrics)
    t = canonical_transition(transition)
    r = canonical_recovery(recovery)
    return {
        "scope": "end_to_end_train_val",
        "category": category,
        "method": method,
        **m,
        **t,
        **r,
        "configuration": configuration,
        "notes": notes,
        "source": str(source),
        "sources": [str(source)],
    }


def candidate_record(
    method: str,
    metrics: Mapping[str, Any] | None,
    *,
    source: Path,
    population: str,
    configuration: str | None = None,
    latency: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = metrics or {}
    latency = latency or {}
    return {
        "scope": "candidate_only",
        "method": method,
        "population": population,
        "n": safe_int(metrics.get("n")),
        "authors": safe_int(metrics.get("authors")),
        "macro_author_top1": safe_float(metrics.get("macro_author_top1")),
        "micro_top1": safe_float(metrics.get("micro_top1")),
        "top3": safe_float(metrics.get("top3")),
        "top5": safe_float(metrics.get("top5")),
        "mrr": safe_float(first(metrics, "mrr_at_10", "mrr_at_5", "mrr")),
        "mean_gold_rank": safe_float(first(metrics, "mean_gold_rank", "mean_rank_given_top10")),
        "mean_latency_ms": safe_float(first(metrics, "mean_latency_ms", "mean_ms", "online_mean_ms", *([] if not latency else ["__none__"]))),
        "p95_latency_ms": safe_float(first(metrics, "p95_latency_ms", "p95_ms")),
        "configuration": configuration,
        "source": str(source),
        "sources": [str(source)],
    }


def add_baseline_map(store: dict[str, dict[str, Any]], baseline: Mapping[str, Any], source: Path) -> None:
    for name, raw in baseline.items():
        if not isinstance(raw, Mapping):
            continue
        metrics = raw.get("metrics") if isinstance(raw.get("metrics"), Mapping) else raw
        if not isinstance(metrics, Mapping):
            continue
        rec = raw.get("recovery") if isinstance(raw.get("recovery"), Mapping) else None
        trans = raw.get("transition") if isinstance(raw.get("transition"), Mapping) else None
        rec_name = str(name)
        rec_name = {"G": "Generic G", "F": "Frequency F", "PV1-Reproduced": "PV1"}.get(rec_name, rec_name)
        if rec_name == "K5_gamma4_lambda0":
            rec_name = "PV1+NGramSelector K5"
        merge_record(store, overall_record(rec_name, metrics, source=source, category="baseline", transition=trans, recovery=rec), OVERALL_FIELDS)


def extract_end_to_end_json(path: Path, data: Mapping[str, Any], store: dict[str, dict[str, Any]]) -> None:
    experiment = str(data.get("experiment", ""))

    # Common baseline containers.
    for key in ("baseline_metrics", "baselines"):
        value = data.get(key)
        if isinstance(value, Mapping):
            add_baseline_map(store, value, path)

    # K1/K3/K5 selector ablation.
    if experiment == "initial_pv1_ngram_selector_k135_v1":
        baseline_recovery = data.get("baseline_recovery", {})
        baseline_metrics = data.get("baseline_metrics", {})
        if isinstance(baseline_metrics, Mapping) and isinstance(baseline_metrics.get("PV1"), Mapping):
            rec = baseline_recovery.get("PV1") if isinstance(baseline_recovery, Mapping) else None
            merge_record(store, overall_record("PV1", baseline_metrics["PV1"], source=path, category="baseline", recovery=rec), OVERALL_FIELDS)
        methods = data.get("methods", {})
        if isinstance(methods, Mapping):
            for raw_name, report in methods.items():
                if not isinstance(report, Mapping):
                    continue
                name = str(raw_name).replace("PV1+NGramSelector@K", "PV1+NGramSelector K")
                merge_record(
                    store,
                    overall_record(
                        name,
                        report.get("metrics"),
                        source=path,
                        category="main_selector_ablation",
                        transition=report.get("pv1_to_method"),
                        recovery=report.get("recovery"),
                        configuration=f"selector_k={report.get('selector_k', name.rsplit('K', 1)[-1])}",
                    ),
                    OVERALL_FIELDS,
                )
        return

    # NGram-CS interpolation: include best point for every alpha, because these are
    # the actual measured main ablation points discussed in the project.
    if experiment == "initial_k5_ngram_cs_interpolation_v1":
        pv1_rec = data.get("pv1_recovery")
        baseline = data.get("baseline_metrics", {})
        if isinstance(baseline, Mapping) and isinstance(baseline.get("PV1"), Mapping):
            merge_record(store, overall_record("PV1", baseline["PV1"], source=path, category="baseline", recovery=pv1_rec), OVERALL_FIELDS)
        best_by_alpha = data.get("best_by_alpha", {})
        if isinstance(best_by_alpha, Mapping):
            for alpha, report in best_by_alpha.items():
                if not isinstance(report, Mapping):
                    continue
                a = str(alpha)
                name = "Pure NGram recovery" if a in ("0", "0.0") else ("NGram-CS / CS-only" if a in ("1", "1.0") else f"NGram-CS Interp alpha={a}")
                config = f"alpha={a}, lambda={report.get('lambda_personal')}"
                merge_record(store, overall_record(name, report.get("metrics"), source=path, category="ngram_cs_interpolation", transition=report.get("pv1_to_method"), recovery=report.get("recovery"), configuration=config), OVERALL_FIELDS)
        selected_methods = data.get("selected_methods", {})
        if isinstance(selected_methods, Mapping):
            for family, report in selected_methods.items():
                if not isinstance(report, Mapping):
                    continue
                cfg = report.get("selected_config", {}) if isinstance(report.get("selected_config"), Mapping) else {}
                name = str(family)
                if name == "NGram-CS-Interpolation":
                    name = "NGram-CS Interpolation selected"
                merge_record(store, overall_record(name, report.get("metrics"), source=path, category="ngram_cs_interpolation", transition=report.get("pv1_to_method"), recovery=report.get("recovery"), configuration=json.dumps(cfg, ensure_ascii=False, sort_keys=True)), OVERALL_FIELDS)
        return

    # Direct NGram / NGram*CS / old concentration families.
    if experiment == "initial_k5_ngram_cs_concentration_recovery_v1":
        baseline = data.get("baseline_metrics", {})
        if isinstance(baseline, Mapping) and isinstance(baseline.get("PV1"), Mapping):
            merge_record(store, overall_record("PV1", baseline["PV1"], source=path, category="baseline", recovery=data.get("pv1_recovery")), OVERALL_FIELDS)
        selected_methods = data.get("selected_methods", {})
        if isinstance(selected_methods, Mapping):
            rename = {
                "NGram": "Pure NGram recovery",
                "NGram+CS": "NGram x CS",
                "NGram+CS+Entropy": "NGram x CS x Entropy",
                "NGram+CS+Margin": "NGram x CS x Margin",
                "NGram+CS+Dual": "NGram x CS x Dual",
            }
            for family, report in selected_methods.items():
                if not isinstance(report, Mapping):
                    continue
                cfg = report.get("selected_config", {}) if isinstance(report.get("selected_config"), Mapping) else {}
                merge_record(store, overall_record(rename.get(str(family), str(family)), report.get("metrics"), source=path, category="legacy_recovery_ablation", transition=report.get("pv1_to_method"), recovery=report.get("recovery"), configuration=json.dumps(cfg, ensure_ascii=False, sort_keys=True)), OVERALL_FIELDS)
        return

    # Joint gamma_F + concentration calibration.
    if experiment == "initial_pv1_ngram_k5_joint_frequency_concentration_v1":
        baselines = data.get("baselines", {})
        if isinstance(baselines, Mapping):
            if isinstance(baselines.get("PV1"), Mapping):
                x = baselines["PV1"]
                merge_record(store, overall_record("PV1", x.get("metrics"), source=path, category="baseline", recovery=x.get("recovery")), OVERALL_FIELDS)
            if isinstance(baselines.get("K5_gamma4_lambda0"), Mapping):
                x = baselines["K5_gamma4_lambda0"]
                merge_record(store, overall_record("PV1+NGramSelector K5", x.get("metrics"), source=path, category="main_selector_ablation", transition=x.get("pv1_to_k5"), recovery=x.get("recovery"), configuration="gamma_F=4, lambda_C=0"), OVERALL_FIELDS)
        best_f = data.get("best_f_only")
        if isinstance(best_f, Mapping):
            merge_record(store, overall_record(f"K5 F-only gamma={best_f.get('gamma_f')}", best_f.get("metrics"), source=path, category="joint_calibration", transition=best_f.get("pv1_to_method"), recovery=best_f.get("recovery"), configuration=f"gamma_F={best_f.get('gamma_f')}, lambda_C=0"), OVERALL_FIELDS)
        selected = data.get("selected_by_family", {})
        if isinstance(selected, Mapping):
            for family, report in selected.items():
                if not isinstance(report, Mapping):
                    continue
                name = f"K5+{family} gamma={report.get('gamma_f')} lambda={report.get('lambda_c')}"
                merge_record(store, overall_record(name, report.get("metrics"), source=path, category="joint_calibration", transition=report.get("pv1_to_method"), recovery=report.get("recovery"), configuration=f"gamma_F={report.get('gamma_f')}, lambda_C={report.get('lambda_c')}"), OVERALL_FIELDS)
        return

    # Additive concentration fixed gamma=4, if present. This is useful as an audit
    # but joint calibration supersedes it scientifically.
    if experiment == "initial_pv1_ngram_k5_additive_concentration_v1":
        baselines = data.get("baselines", {})
        if isinstance(baselines, Mapping):
            add_baseline_map(store, baselines, path)
        selected = data.get("selected_by_family", {})
        if isinstance(selected, Mapping):
            for family, report in selected.items():
                if not isinstance(report, Mapping):
                    continue
                lam = first(report, "lambda_c", "lambda_concentration", "lambda")
                name = f"K5+{family} fixed-gamma4 lambda={lam}"
                merge_record(store, overall_record(name, report.get("metrics"), source=path, category="fixed_gamma_concentration", transition=report.get("k5_to_new"), recovery=report.get("recovery"), configuration=f"gamma_F=4, lambda_C={lam}"), OVERALL_FIELDS)
        return

    # Generic fallback for result JSONs carrying selected_methods.
    selected_methods = data.get("selected_methods")
    if isinstance(selected_methods, Mapping) and "train_val" in json.dumps(data.get("population", {})).lower():
        for family, report in selected_methods.items():
            if isinstance(report, Mapping) and isinstance(report.get("metrics"), Mapping):
                merge_record(store, overall_record(str(family), report.get("metrics"), source=path, category="other", transition=report.get("pv1_to_method"), recovery=report.get("recovery")), OVERALL_FIELDS)


def extract_candidate_json(path: Path, data: Mapping[str, Any], store: dict[str, dict[str, Any]]) -> None:
    experiment = str(data.get("experiment", ""))

    if experiment == "initial_candidate_scoring_q8_vs_bge64_v1":
        sections = data.get("sections", {})
        k2 = sections.get("gold_in_personal_k2plus", {}) if isinstance(sections, Mapping) else {}
        selected = data.get("selected_fusions", {})
        latency = data.get("latency", {}) if isinstance(data.get("latency"), Mapping) else {}
        q8_latency = latency.get("Q8_exact_k5_score_call_ms", {}) if isinstance(latency, Mapping) else {}
        bge_latency = latency.get("BGE64_online_total_ms", {}) if isinstance(latency, Mapping) else {}
        if isinstance(k2, Mapping):
            for method in ("F", "Q8", "BGE64"):
                if isinstance(k2.get(method), Mapping):
                    rec = candidate_record(method, k2[method], source=path, population="Gold in Personal K5 and K>=2", latency=q8_latency if method == "Q8" else bge_latency if method == "BGE64" else {})
                    if method == "Q8":
                        rec["mean_latency_ms"] = safe_float(q8_latency.get("mean"))
                        rec["p95_latency_ms"] = safe_float(q8_latency.get("p95"))
                    elif method == "BGE64":
                        rec["mean_latency_ms"] = safe_float(bge_latency.get("mean"))
                        rec["p95_latency_ms"] = safe_float(bge_latency.get("p95"))
                    merge_record(store, rec, CANDIDATE_FIELDS)
            if isinstance(selected, Mapping):
                for family, payload in selected.items():
                    if not isinstance(payload, Mapping):
                        continue
                    alpha = payload.get("selected_alpha")
                    method_key = f"{family}@{float(alpha):g}" if alpha is not None else str(family)
                    metrics = k2.get(method_key) if isinstance(k2.get(method_key), Mapping) else payload.get("metrics")
                    if isinstance(metrics, Mapping):
                        rec = candidate_record(f"{family} alpha={alpha}", metrics, source=path, population="Gold in Personal K5 and K>=2", configuration=f"alpha={alpha}")
                        if family == "Q8+F":
                            rec["mean_latency_ms"] = safe_float(q8_latency.get("mean"))
                            rec["p95_latency_ms"] = safe_float(q8_latency.get("p95"))
                        elif family == "BGE64+F":
                            rec["mean_latency_ms"] = safe_float(bge_latency.get("mean"))
                            rec["p95_latency_ms"] = safe_float(bge_latency.get("p95"))
                        merge_record(store, rec, CANDIDATE_FIELDS)
        return

    if experiment == "initial_candidate_scoring_adaptive_ngram_top10_v1":
        selected_by_k = data.get("selected_by_k", {})
        k5 = selected_by_k.get("K5") if isinstance(selected_by_k, Mapping) else None
        if isinstance(k5, Mapping):
            metrics_map = k5.get("selected_metrics_k_specific", {})
            key_map = {
                "Frequency": k5.get("frequency"),
                "Hard NGramRecency": k5.get("best_hard_backoff"),
                "SoftSuffix NGram": k5.get("best_soft_suffix"),
                "Interpolated NGramRecency": k5.get("best_interpolated"),
            }
            for name, key in key_map.items():
                if key and isinstance(metrics_map, Mapping) and isinstance(metrics_map.get(key), Mapping):
                    merge_record(store, candidate_record(name, metrics_map[key], source=path, population="K5: Gold in Personal K5 and K>=2", configuration=str(key)), CANDIDATE_FIELDS)
        return

    if experiment == "initial_ngram_frequency_fusion_v1":
        results = data.get("results", {})
        k5 = results.get("K5") if isinstance(results, Mapping) else None
        if isinstance(k5, Mapping):
            for scorer_label, report in k5.items():
                if not isinstance(report, Mapping):
                    continue
                metrics = report.get("selected_metrics_k_specific")
                if not isinstance(metrics, Mapping):
                    continue
                name = "Hard NGram + F" if "Hard" in str(scorer_label) else "Interpolated NGram + F" if "Interpolated" in str(scorer_label) else str(scorer_label)
                merge_record(store, candidate_record(name, metrics, source=path, population="K5: Gold in Personal K5 and K>=2", configuration=f"alpha={report.get('selected_alpha')}; {report.get('ngram_method')}"), CANDIDATE_FIELDS)
        return


def enrich_from_k135_predictions(root: Path, overall: dict[str, dict[str, Any]]) -> None:
    # Find row-level K1/K3/K5 predictions and recompute all standard metrics/recovery.
    for path in root.rglob("predictions.jsonl"):
        parent_cmp = path.parent / "comparison.json"
        if not parent_cmp.is_file():
            continue
        try:
            cmp = read_json(parent_cmp)
        except Exception:
            continue
        if cmp.get("experiment") != "initial_pv1_ngram_selector_k135_v1":
            continue
        rows = read_jsonl(path)
        if not rows:
            continue
        pv1_ranks = [safe_int(row.get("pv1_rank")) for row in rows]
        merge_record(overall, overall_record("PV1", metric_from_ranks(rows, pv1_ranks), source=path, category="baseline", recovery=recovery_from_ranks(rows, pv1_ranks)), OVERALL_FIELDS)
        for k in (1, 3, 5):
            rank_key = f"ngram_selector_k{k}_rank"
            if not any(rank_key in row for row in rows):
                continue
            ranks = [safe_int(row.get(rank_key)) for row in rows]
            transition = transition_from_ranks(pv1_ranks, ranks)
            merge_record(overall, overall_record(f"PV1+NGramSelector K{k}", metric_from_ranks(rows, ranks), source=path, category="main_selector_ablation", transition=transition, recovery=recovery_from_ranks(rows, ranks), configuration=f"selector_k={k}"), OVERALL_FIELDS)
        return


def discover_json_files(root: Path) -> list[Path]:
    names = {
        "comparison.json",
        "candidate_scoring_comparison.json",
        "ngram_frequency_fusion_comparison.json",
    }
    paths = [path for path in root.rglob("*.json") if path.name in names]
    return sorted(set(paths))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


def sort_overall(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    category_order = {
        "baseline": 0,
        "legacy_recovery_ablation": 1,
        "ngram_cs_interpolation": 2,
        "main_selector_ablation": 3,
        "fixed_gamma_concentration": 4,
        "joint_calibration": 5,
        "other": 9,
    }
    return sorted(rows, key=lambda r: (category_order.get(str(r.get("category")), 99), -(r.get("macro_author_top1") or -1), str(r.get("method"))))


def sort_candidate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: (-(r.get("macro_author_top1") or -1), str(r.get("method"))))


def build_missing_report(overall_rows: Sequence[Mapping[str, Any]], candidate_rows: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("INITIAL ALL-RESULTS MISSING-METRIC AUDIT")
    lines.append("========================================")
    lines.append("")
    lines.append("End-to-end required Overall fields:")
    lines.append("  " + ", ".join(OVERALL_FIELDS))
    lines.append("Recovery required fields for recovery methods:")
    lines.append("  " + ", ".join(RECOVERY_FIELDS))
    lines.append("")

    lines.append("END-TO-END")
    lines.append("----------")
    for row in overall_rows:
        missing_overall = [field for field in OVERALL_FIELDS if row.get(field) is None]
        is_recovery_method = str(row.get("method")) not in {"Generic G", "G", "Frequency F", "F", "EM1-R", "EM1-R+F"}
        missing_recovery = [field for field in RECOVERY_FIELDS if row.get(field) is None] if is_recovery_method else []
        if missing_overall or missing_recovery:
            lines.append(f"{row['method']}")
            if missing_overall:
                lines.append("  missing Overall: " + ", ".join(missing_overall))
            if missing_recovery:
                lines.append("  missing Recovery: " + ", ".join(missing_recovery))
            lines.append("  source: " + "; ".join(row.get("sources") or [str(row.get("source"))]))
    lines.append("")
    lines.append("CANDIDATE-ONLY")
    lines.append("--------------")
    for row in candidate_rows:
        missing = [field for field in CANDIDATE_FIELDS if row.get(field) is None]
        if missing:
            lines.append(f"{row['method']}: missing " + ", ".join(missing))
            lines.append("  source: " + "; ".join(row.get("sources") or [str(row.get("source"))]))

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("--------------")
    lines.append("NA means the metric was not found in the available artifact schema and was not guessed.")
    lines.append("Most missing fields can usually be backfilled from existing rank/prediction artifacts without new model inference.")
    lines.append("If a row-level rank artifact is absent, rerun only the CPU evaluation/reporting stage if the original cached scores still exist.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only unified Initial-Pinyin experiment summarizer")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/personalisation/initial_recovery_comparison_v1"),
        help="Initial result root to scan recursively",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; default <root>/all_results_summary_v1",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Result root does not exist: {root}")
    output_dir = (args.output_dir or (root / "all_results_summary_v1")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    overall: dict[str, dict[str, Any]] = {}
    candidate: dict[str, dict[str, Any]] = {}
    discovered: list[str] = []
    errors: list[str] = []

    json_paths = discover_json_files(root)
    for path in json_paths:
        try:
            data = read_json(path)
        except Exception as exc:
            errors.append(f"READ_ERROR {path}: {exc}")
            continue
        if not isinstance(data, Mapping):
            continue
        experiment = str(data.get("experiment", ""))
        discovered.append(f"{path.relative_to(root)}\texperiment={experiment or 'unknown'}")
        try:
            extract_end_to_end_json(path, data, overall)
            extract_candidate_json(path, data, candidate)
        except Exception as exc:
            errors.append(f"EXTRACT_ERROR {path}: {type(exc).__name__}: {exc}")

    # Row-level enrichment for the current main K1/K3/K5 experiment.
    try:
        enrich_from_k135_predictions(root, overall)
    except Exception as exc:
        errors.append(f"K135_ENRICH_ERROR: {type(exc).__name__}: {exc}")

    overall_rows = sort_overall(overall.values())
    candidate_rows = sort_candidate(candidate.values())

    # Recovery table is one row per end-to-end method; keep even partial rows so
    # gaps are immediately visible.
    recovery_rows = [
        {
            "method": row["method"],
            "category": row.get("category"),
            "generic_missing_n": row.get("generic_missing_n"),
            "recoverable_n": row.get("recoverable_n"),
            **{field: row.get(field) for field in RECOVERY_FIELDS},
            "rescue": row.get("rescue"),
            "harm": row.get("harm"),
            "net": row.get("net"),
            "configuration": row.get("configuration"),
            "source": "; ".join(row.get("sources") or [str(row.get("source"))]),
        }
        for row in overall_rows
    ]

    overall_fields = (
        "category", "method", "n", "authors",
        *OVERALL_FIELDS,
        "rescue", "harm", "net", "unchanged_correct", "unchanged_wrong",
        "configuration", "notes", "source",
    )
    recovery_fields = (
        "category", "method", "generic_missing_n", "recoverable_n",
        *RECOVERY_FIELDS,
        "rescue", "harm", "net", "configuration", "source",
    )
    candidate_fields = (
        "method", "population", "n", "authors",
        *CANDIDATE_FIELDS,
        "configuration", "source",
    )

    # Flatten source lists for CSV.
    csv_overall = []
    for row in overall_rows:
        copy = dict(row)
        copy["source"] = "; ".join(row.get("sources") or [str(row.get("source"))])
        csv_overall.append(copy)
    csv_candidate = []
    for row in candidate_rows:
        copy = dict(row)
        copy["source"] = "; ".join(row.get("sources") or [str(row.get("source"))])
        csv_candidate.append(copy)

    write_csv(output_dir / "overall_end_to_end.csv", csv_overall, overall_fields)
    write_csv(output_dir / "recovery_metrics.csv", recovery_rows, recovery_fields)
    write_csv(output_dir / "candidate_only.csv", csv_candidate, candidate_fields)

    missing_report = build_missing_report(overall_rows, candidate_rows)
    (output_dir / "missing_metrics_report.txt").write_text(missing_report, encoding="utf-8")
    (output_dir / "discovered_artifacts.txt").write_text(
        "\n".join(discovered + (["", "ERRORS:", *errors] if errors else [])) + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_with_warnings" if errors else "complete",
        "mode": "read_only_summary_no_inference",
        "root": str(root),
        "output_dir": str(output_dir),
        "dev3000_used": False,
        "test_used": False,
        "expected_reference_counts": {
            "train_val_rows": EXPECTED_TRAIN_VAL_ROWS,
            "generic_missing_n": EXPECTED_GENERIC_MISSING,
            "recoverable_generic_missing_k5_n": EXPECTED_RECOVERABLE_K5,
        },
        "overall_end_to_end": overall_rows,
        "recovery_metrics": recovery_rows,
        "candidate_only": candidate_rows,
        "discovered_artifacts": discovered,
        "errors": errors,
    }
    (output_dir / "all_results_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== INITIAL ALL-RESULTS SUMMARY COMPLETE ===")
    print(f"root: {root}")
    print(f"comparison artifacts discovered: {len(json_paths)}")
    print(f"end-to-end methods summarized: {len(overall_rows)}")
    print(f"candidate-only methods summarized: {len(candidate_rows)}")
    print(f"warnings/errors: {len(errors)}")
    print()
    print("Top end-to-end methods by Macro-author Top1 (available values):")
    ranked = sorted(
        [row for row in overall_rows if row.get("macro_author_top1") is not None],
        key=lambda row: float(row["macro_author_top1"]),
        reverse=True,
    )[:12]
    for row in ranked:
        print(
            f"  {row['method'][:42]:42s} "
            f"Macro={float(row['macro_author_top1']):.6f} "
            f"Top3={fmt(row.get('top3')):>10s} "
            f"MRR={fmt(row.get('mrr_at_10')):>10s} "
            f"Missing={fmt(row.get('missing10')):>10s}"
        )
    print()
    print(f"saved: {output_dir / 'overall_end_to_end.csv'}")
    print(f"saved: {output_dir / 'recovery_metrics.csv'}")
    print(f"saved: {output_dir / 'candidate_only.csv'}")
    print(f"saved: {output_dir / 'missing_metrics_report.txt'}")
    print(f"saved: {output_dir / 'all_results_summary.json'}")
    print("Gold used for scoring/selection: false (this script performs no scoring/selection)")
    print("Dev3000 used: false")
    print("Test used: false")


if __name__ == "__main__":
    main()
