from __future__ import annotations

"""Frozen Train-Val diagnostics for the Initial-Pinyin Recovery -> Context pipeline.

This runner is analysis-only. It does NOT tune coefficients, rescore candidates, embed
BGE contexts, read Dev3000, or read Test. It consumes already-frozen Train-Val
artifacts from:

  - recovery_ngram_context_fusion_v1
  - recovery_bge_ngram_context_fusion_v3

and explains how the three frozen recovery philosophies behave through Stage 1,
NGramRecency, and full NGramRecency+BGERecency reranking.

Main outputs
------------
  diagnostic_summary.json
  headline_comparison.csv
  per_author.csv
  subset_metrics.csv
  top1_transitions.csv
  rank_movement.csv
  recovery_diagnostics.csv
  context_increment.csv
  base_disagreement.csv
  margin_diagnostics.csv
  error_examples.jsonl
  diagnostic_report.md
  run_manifest.json
  artifact_checksums.json

Scientific safeguards
---------------------
  - Train-Val diagnostics only.
  - Dev3000/Test are never read.
  - Frozen selected lambdas are read from completed V1/V3 comparison artifacts.
  - No parameter selection is performed here.
  - Gold is used only for post-hoc evaluation/diagnosis.
  - Existing experiment artifacts are read-only.
  - A new versioned output directory is required.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

EXPECTED_VAL_SHA256 = "d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4"
EXPECTED_ROWS = 34_416
EXPECTED_GENERIC_MISSING = 12_565
EXPECTED_RECOVERABLE_K5 = 4_910

BASE_ORDER = (
    "K5+Entropy",
    "4P+4CS+2E",
    "6P+2CS+.25E",
)

STAGES = ("Recovery", "Recovery+NG-R", "Recovery+NG-R+BGE-R")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}") from exc
            if "row_id" not in row:
                raise RuntimeError(f"Missing row_id at {path}:{line_no}")
            rows.append(row)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return [dict(row) for row in csv.DictReader(source)]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def index_rows(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row["row_id"])
        if row_id in result:
            raise RuntimeError(f"Duplicate row_id in {label}: {row_id}")
        result[row_id] = dict(row)
    return result


def index_selected_predictions(
    rows: Sequence[Mapping[str, Any]], label: str
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {base: {} for base in BASE_ORDER}
    for row in rows:
        base = str(row.get("base", ""))
        if base not in result:
            raise RuntimeError(f"Unexpected base in {label}: {base!r}")
        row_id = str(row["row_id"])
        if row_id in result[base]:
            raise RuntimeError(f"Duplicate {label} row: {base} {row_id}")
        result[base][row_id] = dict(row)
    for base in BASE_ORDER:
        if len(result[base]) != EXPECTED_ROWS:
            raise RuntimeError(
                f"Unexpected {label} rows for {base}: {len(result[base])}/{EXPECTED_ROWS}"
            )
    return result


def rank_metrics(ranks: Sequence[int | None]) -> dict[str, Any]:
    n = len(ranks)
    if not n:
        return {
            "n": 0,
            "top1": None,
            "top3": None,
            "top5": None,
            "mrr_at_10": None,
            "missing10": None,
            "mean_rank_given_top10": None,
        }
    found = [rank for rank in ranks if rank is not None]
    return {
        "n": n,
        "top1": sum(rank == 1 for rank in ranks) / n,
        "top3": sum(rank is not None and rank <= 3 for rank in ranks) / n,
        "top5": sum(rank is not None and rank <= 5 for rank in ranks) / n,
        "mrr_at_10": sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / n,
        "missing10": sum(rank is None for rank in ranks) / n,
        "mean_rank_given_top10": statistics.fmean(found) if found else None,
    }


def macro_top1(rows: Sequence[Mapping[str, Any]], rank_key: str) -> float | None:
    by_author: dict[str, list[int | None]] = defaultdict(list)
    for row in rows:
        rank = row.get(rank_key)
        by_author[str(row["author"])].append(None if rank is None else int(rank))
    if not by_author:
        return None
    values = []
    for ranks in by_author.values():
        values.append(sum(rank == 1 for rank in ranks) / len(ranks))
    return statistics.fmean(values)


def summary_metrics(rows: Sequence[Mapping[str, Any]], rank_key: str) -> dict[str, Any]:
    ranks = [None if row.get(rank_key) is None else int(row[rank_key]) for row in rows]
    result = rank_metrics(ranks)
    result["macro_author_top1"] = macro_top1(rows, rank_key)
    return result


def transition_counts(
    rows: Sequence[Mapping[str, Any]], before_key: str, after_key: str
) -> dict[str, int]:
    out = {
        "n": len(rows),
        "rescue": 0,
        "harm": 0,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
    }
    for row in rows:
        before = row.get(before_key) == 1
        after = row.get(after_key) == 1
        if not before and after:
            out["rescue"] += 1
        elif before and not after:
            out["harm"] += 1
        elif before and after:
            out["unchanged_correct"] += 1
        else:
            out["unchanged_wrong"] += 1
    out["net"] = out["rescue"] - out["harm"]
    return out


def rank_movement(
    rows: Sequence[Mapping[str, Any]], before_key: str, after_key: str
) -> dict[str, Any]:
    comparable = improved = worsened = same = 0
    rr_delta: list[float] = []
    signed_rank_delta: list[int] = []
    for row in rows:
        before = row.get(before_key)
        after = row.get(after_key)
        before_rr = 0.0 if before is None else 1.0 / int(before)
        after_rr = 0.0 if after is None else 1.0 / int(after)
        rr_delta.append(after_rr - before_rr)
        if before is None or after is None:
            continue
        b, a = int(before), int(after)
        comparable += 1
        signed_rank_delta.append(b - a)
        if a < b:
            improved += 1
        elif a > b:
            worsened += 1
        else:
            same += 1
    return {
        "n": len(rows),
        "comparable": comparable,
        "improved": improved,
        "worsened": worsened,
        "same": same,
        "net_improved_minus_worsened": improved - worsened,
        "mean_reciprocal_rank_delta": statistics.fmean(rr_delta) if rr_delta else None,
        "mean_signed_rank_gain_given_both_present": (
            statistics.fmean(signed_rank_delta) if signed_rank_delta else None
        ),
    }


def candidate_text(row: Mapping[str, Any]) -> str:
    value = row.get("candidate", row.get("text"))
    if value is None:
        raise RuntimeError(f"Candidate row has no text: {row}")
    return str(value)


def top1_text(pred: Mapping[str, Any]) -> str | None:
    top10 = pred.get("top10")
    if isinstance(top10, list) and top10:
        return str(top10[0])
    candidates = pred.get("candidates")
    if isinstance(candidates, list) and candidates:
        ordered = sorted(candidates, key=lambda row: int(row.get("rank", 10**9)))
        return candidate_text(ordered[0])
    return None


def score_margin(pred: Mapping[str, Any]) -> float | None:
    candidates = pred.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        return None
    ordered = sorted(candidates, key=lambda row: int(row.get("rank", 10**9)))
    if "final_score" not in ordered[0] or "final_score" not in ordered[1]:
        return None
    return float(ordered[0]["final_score"]) - float(ordered[1]["final_score"])


def margin_bin(value: float | None) -> str:
    if value is None:
        return "NA"
    if value < 0.01:
        return "[0,.01)"
    if value < 0.05:
        return "[.01,.05)"
    if value < 0.10:
        return "[.05,.10)"
    if value < 0.25:
        return "[.10,.25)"
    if value < 0.50:
        return "[.25,.50)"
    if value < 1.00:
        return "[.50,1)"
    return "[1,+inf)"


def recovery_metrics(rows: Sequence[Mapping[str, Any]], rank_key: str) -> dict[str, Any]:
    r = [
        row
        for row in rows
        if bool(row["generic_missing"]) and bool(row["gold_in_personal_k5"])
    ]
    if len(r) != EXPECTED_RECOVERABLE_K5:
        raise RuntimeError(f"Recoverable population mismatch: {len(r)}")
    ranks = [None if row.get(rank_key) is None else int(row[rank_key]) for row in r]
    base = rank_metrics(ranks)
    return {
        "recoverable_n": len(r),
        "rec1": base["top1"],
        "rec3": base["top3"],
        "rec5": base["top5"],
        "rec10": 1.0 - float(base["missing10"]),
        "recovery_mrr_at_10": base["mrr_at_10"],
        "mean_rank_given_recovered": base["mean_rank_given_top10"],
        "rank1_n": sum(rank == 1 for rank in ranks),
        "rank2_3_n": sum(rank is not None and 2 <= rank <= 3 for rank in ranks),
        "rank4_5_n": sum(rank is not None and 4 <= rank <= 5 for rank in ranks),
        "rank6_10_n": sum(rank is not None and 6 <= rank <= 10 for rank in ranks),
        "missing_n": sum(rank is None for rank in ranks),
    }


def subset_name(row: Mapping[str, Any]) -> list[str]:
    names = ["overall"]
    if bool(row["generic_missing"]):
        names.append("generic_missing")
        if bool(row["gold_in_personal_k5"]):
            names.append("recoverable_R")
        else:
            names.append("generic_missing_not_in_K5")
    else:
        names.append("generic_covered")
    if bool(row.get("ambiguous", False)):
        names.append("ambiguous")
    if bool(row.get("conflict", False)):
        names.append("conflict")
    if bool(row.get("ambiguous", False)) and bool(row.get("conflict", False)):
        names.append("ambiguous_and_conflict")
    return names


def selected_lambdas(comparison: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    selected = comparison.get("selected_by_base")
    if not isinstance(selected, Mapping):
        raise RuntimeError("comparison.json missing selected_by_base")
    result = {}
    for base in BASE_ORDER:
        row = selected.get(base)
        if not isinstance(row, Mapping):
            raise RuntimeError(f"comparison.json missing selected base {base}")
        result[base] = (float(row.get("lambda_n", 0.0)), float(row.get("lambda_b", 0.0)))
    return result


def ensure_new_output_root(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite non-empty diagnostic directory: {path}\n"
            "Use a new versioned --output-root."
        )
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--base-ngram-root", type=Path, required=True)
    parser.add_argument("--base-v3-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=300)
    args = parser.parse_args()

    ensure_new_output_root(args.output_root)
    if sha256_file(args.val) != EXPECTED_VAL_SHA256:
        raise RuntimeError("Train-Val SHA256 mismatch")

    stage1_path = args.base_ngram_root / "stage1_frozen.jsonl"
    ngram_pred_path = args.base_ngram_root / "selected_predictions.jsonl"
    ngram_comparison_path = args.base_ngram_root / "comparison.json"
    full_pred_path = args.base_v3_root / "selected_predictions.jsonl"
    full_comparison_path = args.base_v3_root / "comparison.json"
    full_comparison_csv_path = args.base_v3_root / "full_comparison.csv"

    for path in (
        stage1_path,
        ngram_pred_path,
        ngram_comparison_path,
        full_pred_path,
        full_comparison_path,
        full_comparison_csv_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    val = index_rows(read_jsonl(args.val), "Train-Val")
    stage1 = index_rows(read_jsonl(stage1_path), "Stage1")
    ng = index_selected_predictions(read_jsonl(ngram_pred_path), "V1 NGram selected predictions")
    full = index_selected_predictions(read_jsonl(full_pred_path), "V3 full selected predictions")
    v1_comparison = read_json(ngram_comparison_path)
    v3_comparison = read_json(full_comparison_path)
    headline = read_csv(full_comparison_csv_path)

    if len(val) != EXPECTED_ROWS or len(stage1) != EXPECTED_ROWS:
        raise RuntimeError(f"Unexpected row counts: val={len(val)} stage1={len(stage1)}")
    if set(val) != set(stage1):
        raise RuntimeError("Train-Val / Stage1 row IDs differ")

    v1_lambdas = selected_lambdas(v1_comparison)
    v3_lambdas = selected_lambdas(v3_comparison)

    rows_by_base: dict[str, list[dict[str, Any]]] = {base: [] for base in BASE_ORDER}
    for base in BASE_ORDER:
        for row_id in sorted(val):
            vrow = val[row_id]
            srow = stage1[row_id]
            npred = ng[base][row_id]
            fpred = full[base][row_id]
            if str(npred.get("base")) != base or str(fpred.get("base")) != base:
                raise RuntimeError(f"Base mismatch at {row_id} {base}")
            stage1_top10 = [str(x) for x in srow["bases"][base]["top10"]]
            ng_top10 = [str(x) for x in npred["top10"]]
            full_top10 = [str(x) for x in fpred["top10"]]
            if set(stage1_top10) != set(ng_top10) or set(stage1_top10) != set(full_top10):
                raise RuntimeError(f"Candidate-set invariant failed at {row_id} {base}")

            generic_missing = bool(srow["generic_missing"])
            gold_in_k5 = bool(srow["gold_in_personal_k5"])
            row = {
                "row_id": row_id,
                "author": str(srow["author"]),
                "gold": str(srow["gold"]),
                "pinyin_segments": list(srow["pinyin_segments"]),
                "context": str(vrow.get("context", "")),
                "generic_missing": generic_missing,
                "gold_in_personal_k5": gold_in_k5,
                "ambiguous": bool(vrow.get("ambiguous", False)),
                "conflict": bool(vrow.get("conflict", False)),
                "same_pinyin_history_count": int(srow.get("same_pinyin_history_count", 0)),
                "entropy_concentration": float(srow.get("entropy_concentration", 0.0)),
                "recovery_rank": srow["bases"][base]["gold_rank"],
                "ng_rank": npred.get("gold_rank"),
                "full_rank": fpred.get("gold_rank"),
                "recovery_top1": stage1_top10[0] if stage1_top10 else None,
                "ng_top1": top1_text(npred),
                "full_top1": top1_text(fpred),
                "full_margin": score_margin(fpred),
            }
            rows_by_base[base].append(row)

    generic_missing_n = sum(row["generic_missing"] for row in rows_by_base[BASE_ORDER[0]])
    recoverable_n = sum(
        row["generic_missing"] and row["gold_in_personal_k5"]
        for row in rows_by_base[BASE_ORDER[0]]
    )
    if generic_missing_n != EXPECTED_GENERIC_MISSING:
        raise RuntimeError(f"Generic Missing mismatch: {generic_missing_n}")
    if recoverable_n != EXPECTED_RECOVERABLE_K5:
        raise RuntimeError(f"Recoverable K5 mismatch: {recoverable_n}")

    # 1) Headline comparison copied from the already-frozen V3 artifact.
    headline_out: list[dict[str, Any]] = []
    for row in headline:
        headline_out.append(dict(row))
    write_csv(args.output_root / "headline_comparison.csv", headline_out)

    # 2) Per-author metrics across three stages.
    per_author_rows: list[dict[str, Any]] = []
    for base in BASE_ORDER:
        rows = rows_by_base[base]
        for author in sorted({str(row["author"]) for row in rows}):
            chosen = [row for row in rows if str(row["author"]) == author]
            for stage, key in (
                ("Recovery", "recovery_rank"),
                ("Recovery+NG-R", "ng_rank"),
                ("Recovery+NG-R+BGE-R", "full_rank"),
            ):
                m = summary_metrics(chosen, key)
                per_author_rows.append({
                    "base": base,
                    "author": author,
                    "stage": stage,
                    **m,
                })
    write_csv(args.output_root / "per_author.csv", per_author_rows)

    # 3) Subset metrics.
    subset_rows: list[dict[str, Any]] = []
    subset_order = (
        "overall",
        "generic_covered",
        "generic_missing",
        "recoverable_R",
        "generic_missing_not_in_K5",
        "ambiguous",
        "conflict",
        "ambiguous_and_conflict",
    )
    for base in BASE_ORDER:
        rows = rows_by_base[base]
        membership: dict[str, list[dict[str, Any]]] = {name: [] for name in subset_order}
        for row in rows:
            for name in subset_name(row):
                membership[name].append(row)
        for subset in subset_order:
            chosen = membership[subset]
            if not chosen:
                continue
            for stage, key in (
                ("Recovery", "recovery_rank"),
                ("Recovery+NG-R", "ng_rank"),
                ("Recovery+NG-R+BGE-R", "full_rank"),
            ):
                m = summary_metrics(chosen, key)
                subset_rows.append({
                    "base": base,
                    "subset": subset,
                    "stage": stage,
                    **m,
                })
    write_csv(args.output_root / "subset_metrics.csv", subset_rows)

    # 4) Top1 rescue/harm and rank movement, overall and key subsets.
    transition_rows: list[dict[str, Any]] = []
    movement_rows: list[dict[str, Any]] = []
    transitions = (
        ("Recovery->NG-R", "recovery_rank", "ng_rank"),
        ("NG-R->Full", "ng_rank", "full_rank"),
        ("Recovery->Full", "recovery_rank", "full_rank"),
    )
    diag_subsets = ("overall", "generic_covered", "generic_missing", "recoverable_R", "ambiguous", "conflict")
    for base in BASE_ORDER:
        rows = rows_by_base[base]
        for subset in diag_subsets:
            if subset == "overall":
                chosen = rows
            else:
                chosen = [row for row in rows if subset in subset_name(row)]
            for label, before, after in transitions:
                t = transition_counts(chosen, before, after)
                transition_rows.append({"base": base, "subset": subset, "transition": label, **t})
                movement_rows.append({
                    "base": base,
                    "subset": subset,
                    "transition": label,
                    **rank_movement(chosen, before, after),
                })
    write_csv(args.output_root / "top1_transitions.csv", transition_rows)
    write_csv(args.output_root / "rank_movement.csv", movement_rows)

    # 5) Recovery-only diagnostics on fixed R.
    recovery_rows: list[dict[str, Any]] = []
    for base in BASE_ORDER:
        rows = rows_by_base[base]
        for stage, key in (
            ("Recovery", "recovery_rank"),
            ("Recovery+NG-R", "ng_rank"),
            ("Recovery+NG-R+BGE-R", "full_rank"),
        ):
            recovery_rows.append({"base": base, "stage": stage, **recovery_metrics(rows, key)})
    write_csv(args.output_root / "recovery_diagnostics.csv", recovery_rows)

    # 6) Explicit NG and BGE incremental contribution tables.
    increment_rows: list[dict[str, Any]] = []
    for base in BASE_ORDER:
        rows = rows_by_base[base]
        for label, before, after in (
            ("NGram incremental", "recovery_rank", "ng_rank"),
            ("BGE incremental given NGram", "ng_rank", "full_rank"),
        ):
            for subset in ("overall", "generic_covered", "recoverable_R", "ambiguous", "conflict"):
                chosen = rows if subset == "overall" else [row for row in rows if subset in subset_name(row)]
                before_m = summary_metrics(chosen, before)
                after_m = summary_metrics(chosen, after)
                t = transition_counts(chosen, before, after)
                rm = rank_movement(chosen, before, after)
                increment_rows.append({
                    "base": base,
                    "increment": label,
                    "subset": subset,
                    "n": len(chosen),
                    "delta_macro_top1": (
                        None if before_m["macro_author_top1"] is None or after_m["macro_author_top1"] is None
                        else float(after_m["macro_author_top1"]) - float(before_m["macro_author_top1"])
                    ),
                    "delta_micro_top1": (
                        None if before_m["top1"] is None or after_m["top1"] is None
                        else float(after_m["top1"]) - float(before_m["top1"])
                    ),
                    "delta_top3": (
                        None if before_m["top3"] is None or after_m["top3"] is None
                        else float(after_m["top3"]) - float(before_m["top3"])
                    ),
                    "delta_top5": (
                        None if before_m["top5"] is None or after_m["top5"] is None
                        else float(after_m["top5"]) - float(before_m["top5"])
                    ),
                    "delta_mrr_at_10": (
                        None if before_m["mrr_at_10"] is None or after_m["mrr_at_10"] is None
                        else float(after_m["mrr_at_10"]) - float(before_m["mrr_at_10"])
                    ),
                    "rescue": t["rescue"],
                    "harm": t["harm"],
                    "top1_net": t["net"],
                    "rank_improved": rm["improved"],
                    "rank_worsened": rm["worsened"],
                    "rank_net": rm["net_improved_minus_worsened"],
                    "mean_rr_delta": rm["mean_reciprocal_rank_delta"],
                })
    write_csv(args.output_root / "context_increment.csv", increment_rows)

    # 7) Final-model disagreement among the three frozen philosophies.
    base_disagreement_rows: list[dict[str, Any]] = []
    rows_by_id = {base: {row["row_id"]: row for row in rows_by_base[base]} for base in BASE_ORDER}
    pattern_counter: Counter[tuple[bool, bool, bool]] = Counter()
    all_same = 0
    any_disagree = 0
    for row_id in sorted(val):
        top1s = [rows_by_id[base][row_id]["full_top1"] for base in BASE_ORDER]
        gold = str(stage1[row_id]["gold"])
        correct = tuple(top1 == gold for top1 in top1s)
        pattern_counter[correct] += 1
        if len(set(top1s)) == 1:
            all_same += 1
        else:
            any_disagree += 1
    base_disagreement_rows.append({
        "category": "top1_candidate_agreement",
        "pattern": "all_same",
        "n": all_same,
        "rate": all_same / EXPECTED_ROWS,
    })
    base_disagreement_rows.append({
        "category": "top1_candidate_agreement",
        "pattern": "any_disagreement",
        "n": any_disagree,
        "rate": any_disagree / EXPECTED_ROWS,
    })
    for pattern, n in sorted(pattern_counter.items()):
        label = ",".join(
            f"{base}={'correct' if ok else 'wrong'}" for base, ok in zip(BASE_ORDER, pattern)
        )
        base_disagreement_rows.append({
            "category": "correctness_pattern",
            "pattern": label,
            "n": n,
            "rate": n / EXPECTED_ROWS,
        })
    write_csv(args.output_root / "base_disagreement.csv", base_disagreement_rows)

    # 8) Score-margin diagnostics for the final full-context selected models.
    margin_rows: list[dict[str, Any]] = []
    for base in BASE_ORDER:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows_by_base[base]:
            groups[margin_bin(row["full_margin"])].append(row)
        for bin_name in ("[0,.01)", "[.01,.05)", "[.05,.10)", "[.10,.25)", "[.25,.50)", "[.50,1)", "[1,+inf)", "NA"):
            chosen = groups.get(bin_name, [])
            if not chosen:
                continue
            m = summary_metrics(chosen, "full_rank")
            margin_rows.append({
                "base": base,
                "margin_bin": bin_name,
                **m,
            })
    write_csv(args.output_root / "margin_diagnostics.csv", margin_rows)

    # 9) Deterministic diagnostic examples. These are analysis-only and may include Gold/context.
    examples: list[dict[str, Any]] = []
    categories = (
        ("BGE_rescue", lambda row: row["ng_rank"] != 1 and row["full_rank"] == 1),
        ("BGE_harm", lambda row: row["ng_rank"] == 1 and row["full_rank"] != 1),
        ("NG_rescue", lambda row: row["recovery_rank"] != 1 and row["ng_rank"] == 1),
        ("NG_harm", lambda row: row["recovery_rank"] == 1 and row["ng_rank"] != 1),
        ("Recovery_case_full_correct", lambda row: row["generic_missing"] and row["gold_in_personal_k5"] and row["full_rank"] == 1),
        ("Recovery_case_full_wrong", lambda row: row["generic_missing"] and row["gold_in_personal_k5"] and row["full_rank"] != 1),
    )
    per_category_cap = max(1, args.max_examples // (len(BASE_ORDER) * len(categories)))
    for base in BASE_ORDER:
        source = rows_by_base[base]
        for category, predicate in categories:
            chosen = [row for row in source if predicate(row)][:per_category_cap]
            for row in chosen:
                examples.append({
                    "schema_version": 1,
                    "base": base,
                    "category": category,
                    **row,
                    "gold_used_for_scoring": False,
                    "analysis_only": True,
                })
    write_jsonl(args.output_root / "error_examples.jsonl", examples)

    # 10) Compact machine-readable summary + markdown report.
    primary = "4P+4CS+2E"
    coverage = "K5+Entropy"
    front = "6P+2CS+.25E"

    def find_increment(base: str, increment: str, subset: str = "overall") -> dict[str, Any]:
        return next(
            row for row in increment_rows
            if row["base"] == base and row["increment"] == increment and row["subset"] == subset
        )

    summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_recovery_context_diagnostics_v1",
        "scope": "frozen Train-Val post-hoc diagnostics only",
        "rows": EXPECTED_ROWS,
        "generic_missing_n": generic_missing_n,
        "recoverable_k5_n": recoverable_n,
        "frozen_selected_lambdas": {
            "ngram_only_v1": {base: {"lambda_n": v1_lambdas[base][0]} for base in BASE_ORDER},
            "full_context_v3": {
                base: {"lambda_n": v3_lambdas[base][0], "lambda_b": v3_lambdas[base][1]}
                for base in BASE_ORDER
            },
        },
        "primary_overall_model": primary,
        "coverage_model": coverage,
        "front_rank_model": front,
        "primary_ng_increment_overall": find_increment(primary, "NGram incremental"),
        "primary_bge_increment_overall": find_increment(primary, "BGE incremental given NGram"),
        "diagnosis_is_non_tuning": True,
        "dev3000_used": False,
        "test_used": False,
    }
    write_json(args.output_root / "diagnostic_summary.json", summary)

    # Pick concise values for the human-readable report from computed tables.
    recovery_lookup = {(row["base"], row["stage"]): row for row in recovery_rows}
    primary_full = summary_metrics(rows_by_base[primary], "full_rank")
    coverage_full_r = recovery_lookup[(coverage, "Recovery+NG-R+BGE-R")]
    primary_full_r = recovery_lookup[(primary, "Recovery+NG-R+BGE-R")]
    front_full_r = recovery_lookup[(front, "Recovery+NG-R+BGE-R")]
    p_ng = find_increment(primary, "NGram incremental")
    p_bge = find_increment(primary, "BGE incremental given NGram")

    report = "# Initial Recovery -> Context Diagnostics V1\n\n"
    report += "## Scope\n\n"
    report += (
        "This report is a **post-hoc Train-Val diagnosis of already-frozen development models**. "
        "It does not tune coefficients, recompute context scores, access Dev3000, or access Test.\n\n"
    )
    report += "Frozen full-context operating points:\n\n"
    for base in BASE_ORDER:
        ln, lb = v3_lambdas[base]
        report += f"- `{base}`: `lambda_N={ln:g}`, `lambda_B={lb:g}`\n"
    report += "\n## Headline interpretation\n\n"
    report += (
        f"- **Primary overall model:** `{primary}` + NGramRecency + BGERecency, "
        f"Macro Top1={primary_full['macro_author_top1']:.6f}, Micro Top1={primary_full['top1']:.6f}, "
        f"Top3={primary_full['top3']:.6f}, Top5={primary_full['top5']:.6f}, "
        f"MRR@10={primary_full['mrr_at_10']:.6f}.\n"
    )
    report += (
        f"- **Coverage-oriented model:** `{coverage}` has final Recovery Rec@10="
        f"{coverage_full_r['rec10']:.4f}, Rec@3={coverage_full_r['rec3']:.4f}, "
        f"Recovery MRR={coverage_full_r['recovery_mrr_at_10']:.4f}.\n"
    )
    report += (
        f"- **Balanced primary recovery:** `{primary}` has final Recovery Rec@10="
        f"{primary_full_r['rec10']:.4f}, Rec@3={primary_full_r['rec3']:.4f}, "
        f"Recovery MRR={primary_full_r['recovery_mrr_at_10']:.4f}.\n"
    )
    report += (
        f"- **Front-rank comparison:** `{front}` has final Recovery Rec@1="
        f"{front_full_r['rec1']:.4f} and Recovery MRR={front_full_r['recovery_mrr_at_10']:.4f}.\n\n"
    )
    report += "## Context contribution for the primary model\n\n"
    report += (
        f"NGramRecency increment: Delta Macro Top1={p_ng['delta_macro_top1']:+.6f}, "
        f"Top1 rescue={p_ng['rescue']}, harm={p_ng['harm']}, net={p_ng['top1_net']:+d}.\n\n"
    )
    report += (
        f"BGERecency increment given NGramRecency: Delta Macro Top1={p_bge['delta_macro_top1']:+.6f}, "
        f"Top1 rescue={p_bge['rescue']}, harm={p_bge['harm']}, net={p_bge['top1_net']:+d}.\n\n"
    )
    report += "## Files to inspect\n\n"
    report += (
        "- `per_author.csv`: whether gains are consistent across authors.\n"
        "- `subset_metrics.csv`: Generic-covered, Generic-missing, recoverable R, Ambiguous, Conflict.\n"
        "- `top1_transitions.csv`: rescue/harm/net at each stage transition.\n"
        "- `rank_movement.csv`: Gold rank improved/worsened and reciprocal-rank movement.\n"
        "- `recovery_diagnostics.csv`: fixed-R recovery behavior at all three stages.\n"
        "- `context_increment.csv`: explicit NGram and BGE incremental contribution.\n"
        "- `base_disagreement.csv`: where the three recovery philosophies disagree.\n"
        "- `margin_diagnostics.csv`: error rate versus final winner/runner-up score margin.\n"
        "- `error_examples.jsonl`: deterministic rows for qualitative inspection.\n\n"
    )
    report += "## Interpretation boundary\n\n"
    report += (
        "These diagnostics explain the frozen Train-Val development behavior. They must not be used "
        "to reopen lambda/recovery tuning before Dev3000. No significance or generalization claim is "
        "made here.\n"
    )
    (args.output_root / "diagnostic_report.md").write_text(report, encoding="utf-8")

    output_files = [
        args.output_root / "diagnostic_summary.json",
        args.output_root / "headline_comparison.csv",
        args.output_root / "per_author.csv",
        args.output_root / "subset_metrics.csv",
        args.output_root / "top1_transitions.csv",
        args.output_root / "rank_movement.csv",
        args.output_root / "recovery_diagnostics.csv",
        args.output_root / "context_increment.csv",
        args.output_root / "base_disagreement.csv",
        args.output_root / "margin_diagnostics.csv",
        args.output_root / "error_examples.jsonl",
        args.output_root / "diagnostic_report.md",
    ]

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_recovery_context_diagnostics_v1",
        "analysis_only": True,
        "no_tuning": True,
        "inputs": {
            "val": str(args.val.resolve()),
            "stage1_frozen": str(stage1_path.resolve()),
            "ngram_selected_predictions": str(ngram_pred_path.resolve()),
            "v3_selected_predictions": str(full_pred_path.resolve()),
            "v3_full_comparison": str(full_comparison_csv_path.resolve()),
        },
        "input_sha256": {
            "val": sha256_file(args.val),
            "stage1_frozen": sha256_file(stage1_path),
            "ngram_selected_predictions": sha256_file(ngram_pred_path),
            "v3_selected_predictions": sha256_file(full_pred_path),
            "v3_full_comparison": sha256_file(full_comparison_csv_path),
        },
        "dev3000_used": False,
        "test_used": False,
        "gold_used_for_scoring": False,
        "gold_used_for_posthoc_diagnosis": True,
    }
    manifest_path = args.output_root / "run_manifest.json"
    write_json(manifest_path, manifest)
    output_files.append(manifest_path)

    checksums = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in output_files
    }
    write_json(args.output_root / "artifact_checksums.json", checksums)

    print("=== INITIAL RECOVERY -> CONTEXT DIAGNOSTICS V1 COMPLETE ===")
    print(f"Rows: {EXPECTED_ROWS}")
    print(f"Generic Missing: {generic_missing_n}")
    print(f"Recoverable R: {recoverable_n}")
    print("Frozen V3 lambdas:")
    for base in BASE_ORDER:
        ln, lb = v3_lambdas[base]
        print(f"  {base:16s} lambda_N={ln:g} lambda_B={lb:g}")
    print()
    print("Primary 4P+4CS+2E context increments:")
    print(
        f"  NGram: DeltaMacro={p_ng['delta_macro_top1']:+.6f} "
        f"rescue={p_ng['rescue']} harm={p_ng['harm']} net={p_ng['top1_net']:+d}"
    )
    print(
        f"  BGE  : DeltaMacro={p_bge['delta_macro_top1']:+.6f} "
        f"rescue={p_bge['rescue']} harm={p_bge['harm']} net={p_bge['top1_net']:+d}"
    )
    print()
    print("Outputs:")
    for path in output_files:
        print(f"  {path}")
    print(f"  {args.output_root / 'artifact_checksums.json'}")
    print("Dev3000 used: false")
    print("Test used: false")
    print("Diagnosis only: no tuning performed")


if __name__ == "__main__":
    main()
