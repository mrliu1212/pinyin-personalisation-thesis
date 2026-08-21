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
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    support = sum(counts.values())
    distinct = len(counts)
    if support == 0:
        return {
            "support_n": 0,
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
    probs = [n / support for _, n in ordered]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    entropy_norm = entropy / math.log(distinct) if distinct > 1 else 0.0
    return {
        "support_n": support,
        "distinct_targets": distinct,
        "winner": winner,
        "winner_count": top_count,
        "winner_share": top_count / support,
        "second_count": second_count,
        "second_share": second_count / support,
        "margin": (top_count - second_count) / support,
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


def support_bin(n: int) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--history-budget", type=int, default=5000)
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
    if len(fit) != EXPECTED["fit_rows"] or len(val) != EXPECTED["val_rows"] or len(pred) != EXPECTED["pred_rows"]:
        raise RuntimeError("Frozen row count mismatch")

    pred_by_id = {str(r["row_id"]): r for r in pred}
    val_ids = {str(r["row_id"]) for r in val}
    if val_ids != set(pred_by_id):
        raise RuntimeError("Val/pred row-id surfaces differ")

    # Train-Fit-only, per-user conditional distributions for a leakage-safe background prior.
    bg_counts: dict[str, dict[tuple[str, ...], Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for row in fit:
        bg_counts[str(row["author"])][pinyin_key(row)][target_text(row)] += 1
    authors = sorted(bg_counts)

    def background_prob(author: str, p: tuple[str, ...], candidate: str | None) -> tuple[float | None, int]:
        if candidate is None:
            return None, 0
        probs: list[float] = []
        for other in authors:
            if other == author:
                continue
            cnt = bg_counts[other].get(p)
            if not cnt:
                continue
            denom = sum(cnt.values())
            if denom:
                probs.append(cnt.get(candidate, 0) / denom)
        if not probs:
            return None, 0
        return sum(probs) / len(probs), len(probs)

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
    eps = 1e-6

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

            # All rows at equal position see exactly the same strictly-prior history.
            for row in group:
                if row["__partition"] != "val":
                    continue
                rid = str(row["row_id"])
                p = pinyin_key(row)
                gold = str(row["gold"])
                counts = Counter(counts_by_p.get(p, Counter()))
                ds = distribution_stats(counts)

                if ds["support_n"] != int(row.get("same_initial_history_count", ds["support_n"])):
                    support_mismatch += 1
                if len(history) != int(row.get("visible_raw_history_count", len(history))):
                    raw_history_mismatch += 1
                if (ds["distinct_targets"] >= 2) != bool(row.get("ambiguous")):
                    ambiguous_mismatch += 1
                if bool(ds["winner_tied"]) != bool(row.get("frequency_winner_tied")):
                    winner_tie_mismatch += 1
                stored_winner = row.get("frequency_winner")
                if not ds["winner_tied"] and ds["winner"] is not None and str(stored_winner) != str(ds["winner"]):
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

                gold_count = counts.get(gold, 0)
                pv1_count = counts.get(pv1_top1, 0)
                support = int(ds["support_n"])
                gold_share = gold_count / support if support else 0.0
                pv1_share = pv1_count / support if support else 0.0

                bg_gold, bg_gold_users = background_prob(author, p, gold)
                bg_pv1, bg_pv1_users = background_prob(author, p, pv1_top1)
                gold_lift = math.log((gold_share + eps) / (bg_gold + eps)) if bg_gold is not None else None
                pv1_lift = math.log((pv1_share + eps) / (bg_pv1 + eps)) if bg_pv1 is not None else None

                diagnostic_rows.append({
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
                    "support_n": support,
                    "raw_history_n": len(history),
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
                    "gold_share": gold_share,
                    "pv1_top1_count": pv1_count,
                    "pv1_top1_share": pv1_share,
                    "background_gold_prob_fit_macro": bg_gold,
                    "background_gold_users": bg_gold_users,
                    "gold_log_lift_fit_macro": gold_lift,
                    "background_pv1_prob_fit_macro": bg_pv1,
                    "background_pv1_users": bg_pv1_users,
                    "pv1_top1_log_lift_fit_macro": pv1_lift,
                    "support_bin": support_bin(support),
                    "margin_bin": margin_bin(float(ds["margin"])),
                    "entropy_bin": entropy_bin(float(ds["entropy_norm"])),
                    "pv1_lift_bin": lift_bin(pv1_lift),
                    "f_correct": f_correct,
                    "pv1_correct": pv1_correct,
                })

            # Add this position only after all queries at the position have been evaluated.
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
        raise RuntimeError(f"Expected {EXPECTED['val_rows']} diagnostic rows; got {len(diagnostic_rows)}")

    invariant_counts = {
        "support_mismatch": support_mismatch,
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
    if formal_n != EXPECTED["formal_conflict_n"] or legacy_n != EXPECTED["legacy_winner_mismatch_n"]:
        raise RuntimeError("Conflict/winner-mismatch count regression failed")
    if transition_counts["rescue"] != EXPECTED["f_to_pv1_rescue"] or transition_counts["harm"] != EXPECTED["f_to_pv1_harm"]:
        raise RuntimeError("F->PV1 transition regression failed")

    signal_fields = [
        "support_n", "winner_share", "margin", "entropy_norm",
        "gold_share", "pv1_top1_share", "gold_log_lift_fit_macro",
        "pv1_top1_log_lift_fit_macro",
    ]

    group_rows: list[dict[str, Any]] = []
    for group_kind, field in (("transition", "transition"), ("taxonomy", "taxonomy")):
        for name in sorted({str(r[field]) for r in diagnostic_rows}):
            selected = [r for r in diagnostic_rows if str(r[field]) == name]
            row: dict[str, Any] = {"group_kind": group_kind, "group": name, "n": len(selected)}
            row["rescue"] = sum(r["transition"] == "rescue" for r in selected)
            row["harm"] = sum(r["transition"] == "harm" for r in selected)
            row["net"] = row["rescue"] - row["harm"]
            for signal in signal_fields:
                row[f"{signal}_mean"] = mean_or_none([r.get(signal) for r in selected])
                row[f"{signal}_median"] = median_or_none([r.get(signal) for r in selected])
            row["background_pv1_available_rate"] = sum(r["background_pv1_users"] > 0 for r in selected) / len(selected)
            group_rows.append(row)

    def bin_table(field: str, order: list[str]) -> list[dict[str, Any]]:
        out_rows = []
        for name in order:
            selected = [r for r in diagnostic_rows if r[field] == name]
            if not selected:
                continue
            rescue = sum(r["transition"] == "rescue" for r in selected)
            harm = sum(r["transition"] == "harm" for r in selected)
            out_rows.append({
                "bin": name,
                "n": len(selected),
                "rescue": rescue,
                "harm": harm,
                "net": rescue - harm,
                "formal_conflict_n": sum(r["formal_conflict"] for r in selected),
                "winner_mismatch_n": sum(r["winner_mismatch"] for r in selected),
            })
        return out_rows

    support_rows = bin_table("support_bin", ["0", "1", "2", "3-4", "5-9", "10-19", "20-49", "50-99", "100+"])
    margin_rows = bin_table("margin_bin", ["[0,.10)", "[.10,.25)", "[.25,.50)", "[.50,.75)", "[.75,1]"])
    entropy_rows = bin_table("entropy_bin", ["0", "(0,.25]", "(.25,.50]", "(.50,.75]", "(.75,1]"])
    lift_rows = bin_table("pv1_lift_bin", ["unavailable", "<-2", "[-2,-.5)", "[-.5,.5)", "[.5,2)", ">=2"])

    summary = {
        "schema_version": 1,
        "experiment": "initial_distribution_diagnostic_v1",
        "status": "complete",
        "history_budget": args.history_budget,
        "history_semantics": "latest strictly-prior same-author H5000 before Initial matching; equal chronological positions do not see one another",
        "background_semantics": "Train-Fit-only leave-one-author-out macro average of per-author P(c|p); candidate absence contributes 0 when p is observed; authors with no p are excluded",
        "background_epsilon_for_log_lift": eps,
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
                "rescue": sum(r["taxonomy"] == name and r["transition"] == "rescue" for r in diagnostic_rows),
                "harm": sum(r["taxonomy"] == name and r["transition"] == "harm" for r in diagnostic_rows),
            }
            for name in sorted(taxonomy_counts)
        },
    }
    for value in summary["derived_four_way_accounting"].values():
        value["net"] = value["rescue"] - value["harm"]

    write_jsonl(out / "rows.jsonl", diagnostic_rows)
    write_json(out / "summary.json", summary)
    write_csv(out / "signal_group_summary.csv", group_rows)
    write_csv(out / "support_bins.csv", support_rows)
    write_csv(out / "margin_bins.csv", margin_rows)
    write_csv(out / "entropy_bins.csv", entropy_rows)
    write_csv(out / "background_lift_bins.csv", lift_rows)
    write_json(out / "artifact_checksums.json", {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "artifact_checksums.json"
    })

    print("\n=== INITIAL DISTRIBUTION DIAGNOSTIC V1 COMPLETE ===")
    print("rows:", len(diagnostic_rows))
    print("formal conflict:", formal_n)
    print("winner mismatch (legacy B2 flag semantics):", legacy_n)
    print("transitions:", dict(transition_counts))
    print("taxonomy:")
    for name, values in summary["derived_four_way_accounting"].items():
        print(f"  {name}: n={values['n']} rescue={values['rescue']} harm={values['harm']} net={values['net']}")

    print("\ntransition signal summary:")
    for row in group_rows:
        if row["group_kind"] != "transition":
            continue
        print(
            f"  {row['group']}: n={row['n']} "
            f"support_med={row['support_n_median']:.3f} "
            f"margin_med={row['margin_median']:.4f} "
            f"entropy_med={row['entropy_norm_median']:.4f} "
            f"pv1_share_med={row['pv1_top1_share_median']:.4f} "
            f"pv1_lift_med={row['pv1_top1_log_lift_fit_macro_median']}"
        )

    print("\noutputs:", out)
    print("Dev3000 used: false")
    print("Test used: false")
    print("Parameter tuning: false")


if __name__ == "__main__":
    main()
