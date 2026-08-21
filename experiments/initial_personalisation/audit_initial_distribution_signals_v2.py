from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


EXPECTED = {
    "fit_sha256": "162f5c98daa86cc69947571e6d8f20fc401f0a82cdd3fd6e517eb7be2addbdb4",
    "val_sha256": "d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4",
    "pred_sha256": "7fd8aa158d8cd50bced36b55610f8d932bc65e3aae1dbbd5bd65907ff1707ea7",
    "fit_rows": 144526,
    "val_rows": 34416,
    "pred_rows": 34416,
    "formal_conflict_n": 15353,
    "legacy_winner_mismatch_n": 16192,
    "f_to_pv1_rescue": 933,
    "f_to_pv1_harm": 304,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as src:
        return [json.loads(line) for line in src if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as dst:
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pinyin_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(x) for x in row["pinyin_segments"])


def target_text(row: dict[str, Any]) -> str:
    return str(row.get("target", row.get("gold")))


def candidate_text(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return None
    for key in ("candidate", "text", "target"):
        if item.get(key) is not None:
            return str(item[key])
    return None


def top_ranked_candidate(items: Any) -> str | None:
    if not isinstance(items, list) or not items:
        return None
    ranked = [x for x in items if isinstance(x, dict) and x.get("rank") == 1]
    if ranked:
        return candidate_text(ranked[0])
    return candidate_text(items[0])


def distribution_stats(counts: Counter[str]) -> dict[str, Any]:
    evidence_count = sum(counts.values())
    distinct = len(counts)
    if evidence_count == 0:
        return {
            "evidence_count": 0,
            "distinct_targets": 0,
            "winner": None,
            "winner_count": 0,
            "winner_share": 0.0,
            "second_count": 0,
            "second_share": 0.0,
            "margin": 0.0,
            "entropy": 0.0,
            "entropy_norm": 0.0,
            "winner_tied": False,
        }

    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_count = ordered[0][1]
    tied = sum(1 for _, n in ordered if n == top_count) > 1
    winner = None if tied else ordered[0][0]
    second_count = ordered[1][1] if len(ordered) > 1 else 0
    probs = [n / evidence_count for _, n in ordered]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    entropy_norm = entropy / math.log(distinct) if distinct > 1 else 0.0
    return {
        "evidence_count": evidence_count,
        "distinct_targets": distinct,
        "winner": winner,
        "winner_count": top_count,
        "winner_share": top_count / evidence_count,
        "second_count": second_count,
        "second_share": second_count / evidence_count,
        "margin": (top_count - second_count) / evidence_count,
        "entropy": entropy,
        "entropy_norm": entropy_norm,
        "winner_tied": tied,
    }


def mean_or_none(values: list[float | int | None]) -> float | None:
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return sum(xs) / len(xs) if xs else None


def median_or_none(values: list[float | int | None]) -> float | None:
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.median(xs) if xs else None


def pct(n: int, d: int) -> float | None:
    return n / d if d else None


def evidence_bin(n: int) -> str:
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if n <= 4:
        return "3-4"
    if n <= 9:
        return "5-9"
    if n <= 19:
        return "10-19"
    if n <= 49:
        return "20-49"
    if n <= 99:
        return "50-99"
    return "100+"


def exposure_bin(x: float) -> str:
    # x is a fraction of the raw visible causal history, not a percentage.
    if x == 0:
        return "0"
    if x < 0.001:
        return "(0,0.1%)"
    if x < 0.0025:
        return "[0.1%,0.25%)"
    if x < 0.005:
        return "[0.25%,0.5%)"
    if x < 0.01:
        return "[0.5%,1%)"
    if x < 0.02:
        return "[1%,2%)"
    if x < 0.05:
        return "[2%,5%)"
    return ">=5%"


def choice_share_bin(x: float) -> str:
    if x < 0.10:
        return "[0,.10)"
    if x < 0.25:
        return "[.10,.25)"
    if x < 0.50:
        return "[.25,.50)"
    if x < 0.75:
        return "[.50,.75)"
    if x < 1.0:
        return "[.75,1)"
    return "1"


def margin_bin(x: float) -> str:
    if x < 0.10:
        return "[0,.10)"
    if x < 0.25:
        return "[.10,.25)"
    if x < 0.50:
        return "[.25,.50)"
    if x < 0.75:
        return "[.50,.75)"
    return "[.75,1]"


def entropy_bin(x: float) -> str:
    if x == 0:
        return "0"
    if x <= 0.25:
        return "(0,.25]"
    if x <= 0.50:
        return "(.25,.50]"
    if x <= 0.75:
        return "(.50,.75]"
    return "(.75,1]"


def lift_bin(x: float | None) -> str:
    if x is None:
        return "unavailable"
    if x < -2:
        return "<-2"
    if x < -0.5:
        return "[-2,-.5)"
    if x < 0.5:
        return "[-.5,.5)"
    if x < 2:
        return "[.5,2)"
    return ">=2"


def excess_bin(x: float | None) -> str:
    if x is None:
        return "unavailable"
    if x < -0.25:
        return "<-.25"
    if x < -0.10:
        return "[-.25,-.10)"
    if x < 0.10:
        return "[-.10,.10)"
    if x < 0.25:
        return "[.10,.25)"
    if x < 0.50:
        return "[.25,.50)"
    return ">=.50"


def bg_evidence_bin(n: int) -> str:
    if n == 0:
        return "0"
    if n <= 4:
        return "1-4"
    if n <= 9:
        return "5-9"
    if n <= 24:
        return "10-24"
    if n <= 49:
        return "25-49"
    if n <= 99:
        return "50-99"
    return "100+"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--history-budget", type=int, default=5000)
    parser.add_argument("--log-epsilon", type=float, default=1e-6)
    args = parser.parse_args()

    root = args.root
    out = args.output_root
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output root: {out}")
    out.mkdir(parents=True, exist_ok=True)

    fit_path = root / "initial_train_fit_v1.jsonl"
    val_path = root / "initial_train_val_v1.jsonl"
    pred_path = root / "frequency_pv1" / "predictions.jsonl"

    actual_sha = {
        "fit_sha256": sha256_file(fit_path),
        "val_sha256": sha256_file(val_path),
        "pred_sha256": sha256_file(pred_path),
    }
    for key, value in actual_sha.items():
        if value != EXPECTED[key]:
            raise RuntimeError(f"{key} mismatch: expected={EXPECTED[key]} actual={value}")

    fit = load_jsonl(fit_path)
    val = load_jsonl(val_path)
    pred = load_jsonl(pred_path)
    if len(fit) != EXPECTED["fit_rows"]:
        raise RuntimeError("Frozen Train-Fit row count mismatch")
    if len(val) != EXPECTED["val_rows"]:
        raise RuntimeError("Frozen Train-Val row count mismatch")
    if len(pred) != EXPECTED["pred_rows"]:
        raise RuntimeError("Frozen prediction row count mismatch")

    pred_by_id = {str(r["row_id"]): r for r in pred}
    val_ids = {str(r["row_id"]) for r in val}
    if val_ids != set(pred_by_id):
        raise RuntimeError("Val/pred row-id surfaces differ")

    # ------------------------------------------------------------------
    # Train-Fit-only leave-one-author-out background.
    # We preserve the v1 macro semantics, but v2 also records the amount
    # of evidence behind the background estimate and a pooled comparison.
    # ------------------------------------------------------------------
    bg_counts: dict[str, dict[tuple[str, ...], Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for row in fit:
        bg_counts[str(row["author"])][pinyin_key(row)][target_text(row)] += 1
    authors = sorted(bg_counts)

    def background_stats(
        author: str,
        p: tuple[str, ...],
        candidate: str | None,
    ) -> dict[str, Any]:
        other_authors = [x for x in authors if x != author]
        if candidate is None:
            return {
                "macro_prob_seen_users": None,
                "pooled_prob": None,
                "users_total": len(other_authors),
                "users_seen_initial": 0,
                "user_coverage_rate": 0.0,
                "initial_evidence_total": 0,
                "candidate_evidence_total": 0,
                "candidate_zero_given_initial_seen": False,
                "initial_unseen_in_background": True,
            }

        probs: list[float] = []
        total_p = 0
        total_c = 0
        users_seen = 0

        for other in other_authors:
            cnt = bg_counts[other].get(p)
            if not cnt:
                continue
            denom = sum(cnt.values())
            if denom <= 0:
                continue
            users_seen += 1
            cand_n = int(cnt.get(candidate, 0))
            probs.append(cand_n / denom)
            total_p += denom
            total_c += cand_n

        macro = sum(probs) / len(probs) if probs else None
        pooled = total_c / total_p if total_p else None
        return {
            "macro_prob_seen_users": macro,
            "pooled_prob": pooled,
            "users_total": len(other_authors),
            "users_seen_initial": users_seen,
            "user_coverage_rate": users_seen / len(other_authors) if other_authors else None,
            "initial_evidence_total": total_p,
            "candidate_evidence_total": total_c,
            "candidate_zero_given_initial_seen": bool(total_p > 0 and total_c == 0),
            "initial_unseen_in_background": bool(total_p == 0),
        }

    # ------------------------------------------------------------------
    # Reconstruct the exact causal H5000 history used for Initial.
    # Equal chronological positions do not see one another.
    # ------------------------------------------------------------------
    combined_by_author: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fit:
        x = dict(row)
        x["__partition"] = "fit"
        combined_by_author[str(row["author"])].append(x)
    for row in val:
        x = dict(row)
        x["__partition"] = "val"
        combined_by_author[str(row["author"])].append(x)

    diagnostic_rows: list[dict[str, Any]] = []
    support_mismatch = 0
    raw_history_mismatch = 0
    ambiguous_mismatch = 0
    winner_tie_mismatch = 0
    winner_mismatch = 0
    formal_conflict_mismatch = 0
    legacy_mismatch = 0
    eps = args.log_epsilon

    for author, seq in sorted(combined_by_author.items()):
        seq.sort(key=lambda r: (int(r["chronological_position"]), str(r["row_id"])))
        history: deque[tuple[tuple[str, ...], str]] = deque()
        counts_by_p: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)

        idx = 0
        while idx < len(seq):
            pos = int(seq[idx]["chronological_position"])
            end = idx + 1
            while end < len(seq) and int(seq[end]["chronological_position"]) == pos:
                end += 1
            group = seq[idx:end]

            for row in group:
                if row["__partition"] != "val":
                    continue

                rid = str(row["row_id"])
                p = pinyin_key(row)
                gold = str(row["gold"])
                counts = Counter(counts_by_p.get(p, Counter()))
                ds = distribution_stats(counts)

                evidence_count = int(ds["evidence_count"])
                raw_history_n = len(history)
                exposure_rate = evidence_count / raw_history_n if raw_history_n else 0.0

                if evidence_count != int(row.get("same_initial_history_count", evidence_count)):
                    support_mismatch += 1
                if raw_history_n != int(row.get("visible_raw_history_count", raw_history_n)):
                    raw_history_mismatch += 1
                if (ds["distinct_targets"] >= 2) != bool(row.get("ambiguous")):
                    ambiguous_mismatch += 1
                if bool(ds["winner_tied"]) != bool(row.get("frequency_winner_tied")):
                    winner_tie_mismatch += 1
                stored_winner = row.get("frequency_winner")
                if (
                    not ds["winner_tied"]
                    and ds["winner"] is not None
                    and str(stored_winner) != str(ds["winner"])
                ):
                    winner_mismatch += 1

                unique_winner_mismatch = (
                    (not ds["winner_tied"])
                    and ds["winner"] is not None
                    and str(ds["winner"]) != gold
                )
                formal_conflict = bool(row.get("ambiguous")) and unique_winner_mismatch
                if formal_conflict != bool(row.get("conflict")):
                    formal_conflict_mismatch += 1

                pr = pred_by_id[rid]
                if unique_winner_mismatch != bool(pr.get("conflict")):
                    legacy_mismatch += 1

                f_correct = pr.get("frequency_rank") == 1
                pv1_correct = pr.get("pv1_rank") == 1
                if not f_correct and pv1_correct:
                    transition = "rescue"
                elif f_correct and not pv1_correct:
                    transition = "harm"
                elif f_correct and pv1_correct:
                    transition = "unchanged_correct"
                else:
                    transition = "unchanged_wrong"

                if formal_conflict:
                    taxonomy = "formal_conflict"
                elif bool(row.get("ambiguous")):
                    taxonomy = "ambiguous_non_conflict"
                elif unique_winner_mismatch:
                    taxonomy = "nonambiguous_winner_mismatch"
                else:
                    taxonomy = "nonambiguous_other"

                pv1_top1 = top_ranked_candidate(pr.get("pv1_candidates"))
                if pv1_top1 is None:
                    raise RuntimeError(f"Cannot resolve PV1 Top1 candidate for row {rid}")

                gold_count = int(counts.get(gold, 0))
                pv1_count = int(counts.get(pv1_top1, 0))
                gold_share = gold_count / evidence_count if evidence_count else 0.0
                pv1_share = pv1_count / evidence_count if evidence_count else 0.0

                bg_gold = background_stats(author, p, gold)
                bg_pv1 = background_stats(author, p, pv1_top1)

                bg_gold_macro = bg_gold["macro_prob_seen_users"]
                bg_pv1_macro = bg_pv1["macro_prob_seen_users"]

                gold_lift = (
                    math.log((gold_share + eps) / (float(bg_gold_macro) + eps))
                    if bg_gold_macro is not None
                    else None
                )
                pv1_lift = (
                    math.log((pv1_share + eps) / (float(bg_pv1_macro) + eps))
                    if bg_pv1_macro is not None
                    else None
                )
                gold_excess = (
                    gold_share - float(bg_gold_macro)
                    if bg_gold_macro is not None
                    else None
                )
                pv1_excess = (
                    pv1_share - float(bg_pv1_macro)
                    if bg_pv1_macro is not None
                    else None
                )

                diagnostic_rows.append(
                    {
                        "row_id": rid,
                        "author": author,
                        "chronological_position": pos,
                        "pinyin_segments": list(p),
                        "gold": gold,
                        "pv1_top1": pv1_top1,
                        "transition": transition,
                        "taxonomy": taxonomy,
                        "ambiguous": bool(row.get("ambiguous")),
                        "formal_conflict": formal_conflict,
                        "winner_mismatch": unique_winner_mismatch,
                        # New terminology: Exposure Rate is primary prevalence;
                        # Evidence Count is retained as the amount of evidence.
                        "exposure_rate": exposure_rate,
                        "evidence_count": evidence_count,
                        "raw_history_n": raw_history_n,
                        # Legacy alias only for traceability with v1 output.
                        "support_n_legacy_alias": evidence_count,
                        "distinct_targets": int(ds["distinct_targets"]),
                        "winner": ds["winner"],
                        "winner_count": int(ds["winner_count"]),
                        "winner_share": float(ds["winner_share"]),
                        "second_count": int(ds["second_count"]),
                        "second_share": float(ds["second_share"]),
                        "margin": float(ds["margin"]),
                        "entropy": float(ds["entropy"]),
                        "entropy_norm": float(ds["entropy_norm"]),
                        "gold_count": gold_count,
                        "gold_choice_share": gold_share,
                        "pv1_top1_count": pv1_count,
                        "pv1_top1_choice_share": pv1_share,
                        # Background measurement audit.
                        "background_gold_prob_fit_macro": bg_gold_macro,
                        "background_gold_prob_fit_pooled": bg_gold["pooled_prob"],
                        "background_gold_initial_evidence_total": bg_gold["initial_evidence_total"],
                        "background_gold_candidate_evidence_total": bg_gold["candidate_evidence_total"],
                        "background_gold_users_seen_initial": bg_gold["users_seen_initial"],
                        "background_gold_user_coverage_rate": bg_gold["user_coverage_rate"],
                        "background_gold_candidate_zero_given_initial_seen": bg_gold["candidate_zero_given_initial_seen"],
                        "background_gold_initial_unseen": bg_gold["initial_unseen_in_background"],
                        "gold_raw_log_lift_fit_macro": gold_lift,
                        "gold_excess_share_fit_macro": gold_excess,
                        "background_pv1_prob_fit_macro": bg_pv1_macro,
                        "background_pv1_prob_fit_pooled": bg_pv1["pooled_prob"],
                        "background_pv1_initial_evidence_total": bg_pv1["initial_evidence_total"],
                        "background_pv1_candidate_evidence_total": bg_pv1["candidate_evidence_total"],
                        "background_pv1_users_seen_initial": bg_pv1["users_seen_initial"],
                        "background_pv1_user_coverage_rate": bg_pv1["user_coverage_rate"],
                        "background_pv1_candidate_zero_given_initial_seen": bg_pv1["candidate_zero_given_initial_seen"],
                        "background_pv1_initial_unseen": bg_pv1["initial_unseen_in_background"],
                        "pv1_top1_raw_log_lift_fit_macro": pv1_lift,
                        "pv1_top1_excess_share_fit_macro": pv1_excess,
                        # Bins used only for descriptive diagnostics, never ranking.
                        "exposure_bin": exposure_bin(exposure_rate),
                        "evidence_bin": evidence_bin(evidence_count),
                        "choice_share_bin": choice_share_bin(pv1_share),
                        "margin_bin": margin_bin(float(ds["margin"])),
                        "entropy_bin": entropy_bin(float(ds["entropy_norm"])),
                        "pv1_lift_bin": lift_bin(pv1_lift),
                        "pv1_excess_bin": excess_bin(pv1_excess),
                        "background_evidence_bin": bg_evidence_bin(int(bg_pv1["initial_evidence_total"])),
                        "f_correct": f_correct,
                        "pv1_correct": pv1_correct,
                    }
                )

            for row in group:
                item = (pinyin_key(row), target_text(row))
                if len(history) >= args.history_budget:
                    old_p, old_target = history.popleft()
                    old_counter = counts_by_p[old_p]
                    old_counter[old_target] -= 1
                    if old_counter[old_target] <= 0:
                        del old_counter[old_target]
                    if not old_counter:
                        del counts_by_p[old_p]
                history.append(item)
                counts_by_p[item[0]][item[1]] += 1

            idx = end

    if len(diagnostic_rows) != EXPECTED["val_rows"]:
        raise RuntimeError(
            f"Expected {EXPECTED['val_rows']} diagnostic rows; got {len(diagnostic_rows)}"
        )

    invariant_counts = {
        "evidence_count_mismatch": support_mismatch,
        "raw_history_mismatch": raw_history_mismatch,
        "ambiguous_mismatch": ambiguous_mismatch,
        "winner_tie_mismatch": winner_tie_mismatch,
        "winner_mismatch": winner_mismatch,
        "formal_conflict_mismatch": formal_conflict_mismatch,
        "legacy_winner_mismatch_flag_mismatch": legacy_mismatch,
    }
    if any(invariant_counts.values()):
        raise RuntimeError(f"Causal reconstruction invariant failed: {invariant_counts}")

    transition_counts = Counter(r["transition"] for r in diagnostic_rows)
    taxonomy_counts = Counter(r["taxonomy"] for r in diagnostic_rows)
    formal_n = sum(r["formal_conflict"] for r in diagnostic_rows)
    legacy_n = sum(r["winner_mismatch"] for r in diagnostic_rows)
    if formal_n != EXPECTED["formal_conflict_n"]:
        raise RuntimeError("Formal Conflict count regression failed")
    if legacy_n != EXPECTED["legacy_winner_mismatch_n"]:
        raise RuntimeError("Legacy winner-mismatch count regression failed")
    if transition_counts["rescue"] != EXPECTED["f_to_pv1_rescue"]:
        raise RuntimeError("F->PV1 rescue regression failed")
    if transition_counts["harm"] != EXPECTED["f_to_pv1_harm"]:
        raise RuntimeError("F->PV1 harm regression failed")

    # ------------------------------------------------------------------
    # Group summaries: the main comparison is rescue vs harm.
    # Gold-derived fields are analysis-only and are not candidate features.
    # ------------------------------------------------------------------
    signal_fields = [
        "exposure_rate",
        "evidence_count",
        "winner_share",
        "margin",
        "entropy_norm",
        "gold_choice_share",
        "pv1_top1_choice_share",
        "gold_raw_log_lift_fit_macro",
        "pv1_top1_raw_log_lift_fit_macro",
        "gold_excess_share_fit_macro",
        "pv1_top1_excess_share_fit_macro",
        "background_pv1_initial_evidence_total",
        "background_pv1_candidate_evidence_total",
        "background_pv1_users_seen_initial",
        "background_pv1_user_coverage_rate",
    ]

    group_rows: list[dict[str, Any]] = []
    for group_kind, field in (("transition", "transition"), ("taxonomy", "taxonomy")):
        for name in sorted({str(r[field]) for r in diagnostic_rows}):
            selected = [r for r in diagnostic_rows if str(r[field]) == name]
            row_out: dict[str, Any] = {
                "group_kind": group_kind,
                "group": name,
                "n": len(selected),
                "rescue": sum(r["transition"] == "rescue" for r in selected),
                "harm": sum(r["transition"] == "harm" for r in selected),
            }
            row_out["net"] = row_out["rescue"] - row_out["harm"]
            for signal in signal_fields:
                row_out[f"{signal}_mean"] = mean_or_none([r.get(signal) for r in selected])
                row_out[f"{signal}_median"] = median_or_none([r.get(signal) for r in selected])

            n = len(selected)
            row_out["background_initial_unseen_rate"] = (
                sum(bool(r["background_pv1_initial_unseen"]) for r in selected) / n
            )
            row_out["background_candidate_zero_given_seen_rate"] = (
                sum(bool(r["background_pv1_candidate_zero_given_initial_seen"]) for r in selected) / n
            )
            row_out["background_any_candidate_evidence_rate"] = (
                sum(int(r["background_pv1_candidate_evidence_total"]) > 0 for r in selected) / n
            )
            group_rows.append(row_out)

    def bin_table(field: str, order: list[str]) -> list[dict[str, Any]]:
        out_rows: list[dict[str, Any]] = []
        for name in order:
            selected = [r for r in diagnostic_rows if r[field] == name]
            if not selected:
                continue
            rescue = sum(r["transition"] == "rescue" for r in selected)
            harm = sum(r["transition"] == "harm" for r in selected)
            out_rows.append(
                {
                    "bin": name,
                    "n": len(selected),
                    "rescue": rescue,
                    "harm": harm,
                    "net": rescue - harm,
                    "rescue_rate_within_bin": rescue / len(selected),
                    "harm_rate_within_bin": harm / len(selected),
                    "formal_conflict_n": sum(r["formal_conflict"] for r in selected),
                    "winner_mismatch_n": sum(r["winner_mismatch"] for r in selected),
                }
            )
        return out_rows

    exposure_rows = bin_table(
        "exposure_bin",
        [
            "0",
            "(0,0.1%)",
            "[0.1%,0.25%)",
            "[0.25%,0.5%)",
            "[0.5%,1%)",
            "[1%,2%)",
            "[2%,5%)",
            ">=5%",
        ],
    )
    evidence_rows = bin_table(
        "evidence_bin",
        ["0", "1", "2", "3-4", "5-9", "10-19", "20-49", "50-99", "100+"],
    )
    choice_rows = bin_table(
        "choice_share_bin", ["[0,.10)", "[.10,.25)", "[.25,.50)", "[.50,.75)", "[.75,1)", "1"]
    )
    margin_rows = bin_table(
        "margin_bin", ["[0,.10)", "[.10,.25)", "[.25,.50)", "[.50,.75)", "[.75,1]"]
    )
    entropy_rows = bin_table(
        "entropy_bin", ["0", "(0,.25]", "(.25,.50]", "(.50,.75]", "(.75,1]"]
    )
    lift_rows = bin_table(
        "pv1_lift_bin", ["unavailable", "<-2", "[-2,-.5)", "[-.5,.5)", "[.5,2)", ">=2"]
    )
    excess_rows = bin_table(
        "pv1_excess_bin", ["unavailable", "<-.25", "[-.25,-.10)", "[-.10,.10)", "[.10,.25)", "[.25,.50)", ">=.50"]
    )
    bg_evidence_rows = bin_table(
        "background_evidence_bin", ["0", "1-4", "5-9", "10-24", "25-49", "50-99", "100+"]
    )

    # Two-dimensional grids answer the user's key question:
    # does low exposure/evidence become reliable when concentration is high?
    def joint_grid(row_field: str, col_field: str) -> list[dict[str, Any]]:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for r in diagnostic_rows:
            groups[(str(r[row_field]), str(r[col_field]))].append(r)
        out_rows: list[dict[str, Any]] = []
        for (a, b), selected in sorted(groups.items()):
            rescue = sum(r["transition"] == "rescue" for r in selected)
            harm = sum(r["transition"] == "harm" for r in selected)
            out_rows.append(
                {
                    row_field: a,
                    col_field: b,
                    "n": len(selected),
                    "rescue": rescue,
                    "harm": harm,
                    "net": rescue - harm,
                    "rescue_rate_within_cell": rescue / len(selected),
                    "harm_rate_within_cell": harm / len(selected),
                    "formal_conflict_n": sum(r["formal_conflict"] for r in selected),
                }
            )
        return out_rows

    exposure_margin = joint_grid("exposure_bin", "margin_bin")
    exposure_entropy = joint_grid("exposure_bin", "entropy_bin")
    evidence_margin = joint_grid("evidence_bin", "margin_bin")
    evidence_entropy = joint_grid("evidence_bin", "entropy_bin")

    # Background sparsity audit directly tests why raw log-lift was larger for harm.
    sparsity_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in diagnostic_rows:
        status = (
            "initial_unseen"
            if r["background_pv1_initial_unseen"]
            else "candidate_zero"
            if r["background_pv1_candidate_zero_given_initial_seen"]
            else "candidate_seen"
        )
        coverage = f"users_seen={r['background_pv1_users_seen_initial']}"
        sparsity_groups[(r["transition"], status, coverage)].append(r)

    sparsity_rows: list[dict[str, Any]] = []
    for (transition, status, coverage), selected in sorted(sparsity_groups.items()):
        sparsity_rows.append(
            {
                "transition": transition,
                "background_status": status,
                "background_user_coverage": coverage,
                "n": len(selected),
                "pv1_choice_share_median": median_or_none(
                    [r["pv1_top1_choice_share"] for r in selected]
                ),
                "pv1_raw_log_lift_median": median_or_none(
                    [r["pv1_top1_raw_log_lift_fit_macro"] for r in selected]
                ),
                "pv1_excess_share_median": median_or_none(
                    [r["pv1_top1_excess_share_fit_macro"] for r in selected]
                ),
                "background_initial_evidence_median": median_or_none(
                    [r["background_pv1_initial_evidence_total"] for r in selected]
                ),
                "background_candidate_evidence_median": median_or_none(
                    [r["background_pv1_candidate_evidence_total"] for r in selected]
                ),
            }
        )

    summary = {
        "schema_version": 2,
        "experiment": "initial_distribution_diagnostic_v2",
        "status": "complete",
        "history_budget": args.history_budget,
        "history_semantics": (
            "latest strictly-prior same-author H5000 before Initial matching; "
            "equal chronological positions do not see one another"
        ),
        "terminology": {
            "exposure_rate": "N_u(p) / |H_t|; prevalence of the Initial in visible raw causal history",
            "evidence_count": "N_u(p); amount of same-Initial evidence, retained as a reliability quantity",
            "choice_share": "N_u(c,p) / N_u(p) = P_u(c|p)",
            "raw_log_lift": "log((P_u(c|p)+eps)/(P_bg,-u(c|p)+eps)); diagnostic only",
            "excess_share": "P_u(c|p) - P_bg,-u(c|p); absolute background-adjusted difference",
        },
        "background_semantics": (
            "Train-Fit-only leave-one-author-out. Macro probability averages per-author P(c|p) "
            "over other authors that observed p; candidate absence contributes 0 for an author that observed p. "
            "V2 additionally records background user coverage, total Initial evidence, candidate evidence, "
            "and pooled probability for measurement diagnostics."
        ),
        "background_log_epsilon": eps,
        "no_smoothing_or_reliability_weighting_applied": True,
        "reason_no_smoothing_yet": (
            "V2 is diagnostic only: first measure whether high raw lift is concentrated in sparse/zero background cells "
            "before choosing alpha/beta/kappa or any model feature."
        ),
        "rows": len(diagnostic_rows),
        "authors": authors,
        "dev3000_used": False,
        "test_used": False,
        "parameter_tuning": False,
        "input_sha256": actual_sha,
        "invariants": invariant_counts,
        "formal_conflict_n": formal_n,
        "winner_mismatch_n": legacy_n,
        "transition_counts": dict(transition_counts),
        "taxonomy_counts": dict(taxonomy_counts),
        "derived_four_way_accounting": {
            name: {
                "n": sum(r["taxonomy"] == name for r in diagnostic_rows),
                "rescue": sum(
                    r["taxonomy"] == name and r["transition"] == "rescue"
                    for r in diagnostic_rows
                ),
                "harm": sum(
                    r["taxonomy"] == name and r["transition"] == "harm"
                    for r in diagnostic_rows
                ),
            }
            for name in sorted(taxonomy_counts)
        },
    }
    for value in summary["derived_four_way_accounting"].values():
        value["net"] = value["rescue"] - value["harm"]

    write_jsonl(out / "rows.jsonl", diagnostic_rows)
    write_json(out / "summary.json", summary)
    write_csv(out / "signal_group_summary.csv", group_rows)
    write_csv(out / "exposure_rate_bins.csv", exposure_rows)
    write_csv(out / "evidence_count_bins.csv", evidence_rows)
    write_csv(out / "choice_share_bins.csv", choice_rows)
    write_csv(out / "margin_bins.csv", margin_rows)
    write_csv(out / "entropy_bins.csv", entropy_rows)
    write_csv(out / "background_raw_lift_bins.csv", lift_rows)
    write_csv(out / "background_excess_share_bins.csv", excess_rows)
    write_csv(out / "background_evidence_bins.csv", bg_evidence_rows)
    write_csv(out / "exposure_x_margin.csv", exposure_margin)
    write_csv(out / "exposure_x_entropy.csv", exposure_entropy)
    write_csv(out / "evidence_x_margin.csv", evidence_margin)
    write_csv(out / "evidence_x_entropy.csv", evidence_entropy)
    write_csv(out / "background_sparsity_audit.csv", sparsity_rows)
    write_json(
        out / "artifact_checksums.json",
        {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(out.iterdir())
            if path.is_file() and path.name != "artifact_checksums.json"
        },
    )

    print("\n=== INITIAL DISTRIBUTION DIAGNOSTIC V2 COMPLETE ===")
    print("rows:", len(diagnostic_rows))
    print("formal conflict:", formal_n)
    print("winner mismatch (legacy B2 semantics):", legacy_n)
    print("transitions:", dict(transition_counts))

    print("\ntransition signal summary:")
    for row in group_rows:
        if row["group_kind"] != "transition":
            continue
        print(
            f"  {row['group']}: n={row['n']} "
            f"exposure_med={row['exposure_rate_median']:.6f} "
            f"evidence_med={row['evidence_count_median']:.3f} "
            f"margin_med={row['margin_median']:.4f} "
            f"entropy_med={row['entropy_norm_median']:.4f} "
            f"choice_share_med={row['pv1_top1_choice_share_median']:.4f} "
            f"raw_lift_med={row['pv1_top1_raw_log_lift_fit_macro_median']} "
            f"excess_med={row['pv1_top1_excess_share_fit_macro_median']} "
            f"bg_initial_unseen_rate={row['background_initial_unseen_rate']:.4f} "
            f"bg_candidate_zero_rate={row['background_candidate_zero_given_seen_rate']:.4f}"
        )

    print("\nPrimary questions for V2:")
    print("  1) Does high concentration rescue low Exposure Rate / low Evidence Count?")
    print("  2) Is high raw log-lift in harm concentrated where background evidence is sparse or zero?")
    print("  3) Does Excess Share separate rescue/harm more cleanly than raw log-lift?")
    print("\noutputs:", out)
    print("Dev3000 used: false")
    print("Test used: false")
    print("Parameter tuning: false")
    print("Smoothing/reliability weighting applied: false")


if __name__ == "__main__":
    main()
