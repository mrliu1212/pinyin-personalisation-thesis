from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

H = 5000
DEFAULT_AUTHORS = ["Etinjat", "Re_spectators", "breaddddd"]


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def get(row: dict[str, Any], *names: str, default=None):
    for name in names:
        if name in row:
            return row[name]
    return default


def rid(row):
    return str(get(row, "row_id", "condition_id", "id"))


def author(row):
    return str(get(row, "author", "user_id"))


def position(row, fallback):
    value = get(
        row,
        "chronological_position",
        "position",
        "interaction_position",
        "query_position",
        default=None,
    )
    return fallback if value is None else int(value)


def pinyin(row):
    value = get(row, "pinyin_segments", "segmented_pinyin", "pinyin")
    if isinstance(value, (list, tuple)):
        return tuple(str(x) for x in value)
    return (str(value),)


def gold(row):
    return str(get(row, "gold", "target", "target_candidate", "current_gold"))


def context(row):
    return str(
        get(
            row,
            "context",
            "preceding_context",
            "current_context",
            default="",
        )
        or ""
    )


def document_id(row):
    value = get(
        row,
        "work_id",
        "document_id",
        "source_work_id",
        "page_id",
        "source_id",
        "work_key",
        "article_id",
        default=None,
    )
    return None if value is None else str(value)


def is_top1(value):
    if value is None:
        return False
    try:
        return int(value) == 1
    except (TypeError, ValueError):
        return False


def rank_candidates(cands, lf, lc):
    return sorted(
        cands,
        key=lambda c: (
            -(
                float(c["normalized_generic_score"])
                + lf * float(c["frequency_support"])
                + lc * float(c["context_support"])
            ),
            int(c["generic_rank"]),
        ),
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pilot-root",
        type=Path,
        default=Path(
            r"C:\Users\chiar\Desktop\LBH\thesis-personalisation"
            r"\results\personalisation\pilot_a_context_memory"
        ),
    )
    p.add_argument(
        "--four-way-rows",
        type=Path,
        default=Path(
            r"C:\Users\chiar\Desktop\LBH\thesis-context-lab"
            r"\results\personalisation\external_memory"
            r"\em2_four_way_dev_compare\rows.jsonl"
        ),
    )
    p.add_argument(
        "--surface-rows",
        type=Path,
        default=Path(
            r"C:\Users\chiar\Desktop\LBH\thesis-context-lab"
            r"\results\personalisation\external_memory"
            r"\em2_fixed_gfc_dev\selected_rows.jsonl"
        ),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            r"results\personalisation\external_memory"
            r"\em3_context_outcome_examples"
        ),
    )
    p.add_argument("--authors", nargs="+", default=DEFAULT_AUTHORS)
    p.add_argument("--history-budget", type=int, default=H)
    p.add_argument("--examples-per-group", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    authors = list(dict.fromkeys(args.authors))
    author_set = set(authors)

    history_path = args.pilot_root / "history_manifest.jsonl"
    dev_path = args.pilot_root / "dev_manifest.jsonl"

    four = {rid(r): r for r in read_jsonl(args.four_way_rows)}
    surface = {rid(r): r for r in read_jsonl(args.surface_rows)}

    dev_rows = [
        r
        for r in read_jsonl(dev_path)
        if author(r) in author_set and rid(r) in four
    ]

    all_by_author = defaultdict(list)
    idx = 0

    for row in read_jsonl(history_path):
        if author(row) in author_set:
            all_by_author[author(row)].append(
                (position(row, idx), idx, False, row)
            )
        idx += 1

    for row in dev_rows:
        all_by_author[author(row)].append(
            (position(row, idx), idx, True, row)
        )
        idx += 1

    groups = {
        "G_OK_F_OK_H_BAD": [],
        "G_OK_F_BAD_H_BAD": [],
        "G_BAD_F_OK_H_BAD": [],
        "G_BAD_F_OK_H_OK": [],
    }

    for a in authors:
        data = sorted(all_by_author[a], key=lambda x: (x[0], x[1]))
        visible = deque(maxlen=args.history_budget)

        i = 0
        while i < len(data):
            cur_pos = data[i][0]
            j = i
            group = []
            while j < len(data) and data[j][0] == cur_pos:
                group.append(data[j])
                j += 1

            for qpos, _, is_dev, q in group:
                qid = rid(q)
                if not is_dev or qid not in four:
                    continue

                ff = four[qid]
                g_ok = is_top1(ff.get("G_rank"))
                f_ok = is_top1(ff.get("F_rank"))
                h_ok = is_top1(ff.get("Hidden_M1_rank"))

                label = None
                if g_ok and f_ok and not h_ok:
                    label = "G_OK_F_OK_H_BAD"
                elif g_ok and not f_ok and not h_ok:
                    label = "G_OK_F_BAD_H_BAD"
                elif not g_ok and f_ok and not h_ok:
                    label = "G_BAD_F_OK_H_BAD"
                elif not g_ok and f_ok and h_ok:
                    label = "G_BAD_F_OK_H_OK"

                if label is None:
                    continue

                qpy = pinyin(q)
                qgold = gold(q)
                same = [h for _, h in visible if pinyin(h) == qpy]
                counts = Counter(gold(h) for h in same)
                top_counts = counts.most_common()

                freq_winner = top_counts[0][0] if top_counts else None
                freq_winner_count = top_counts[0][1] if top_counts else 0
                second_count = top_counts[1][1] if len(top_counts) > 1 else 0
                freq_share = (
                    freq_winner_count / len(same) if same else 0.0
                )
                freq_margin = freq_winner_count - second_count

                g_pred = f_pred = h_pred = None
                candidate_evidence = []
                srow = surface.get(qid)
                if srow and "ranking" in srow:
                    cands = srow["ranking"]
                    g_pred = rank_candidates(cands, 0, 0)[0]["candidate"]
                    f_pred = rank_candidates(cands, 4, 0)[0]["candidate"]
                    h_pred = rank_candidates(cands, 0, 4)[0]["candidate"]

                    ranked_context = sorted(
                        cands,
                        key=lambda c: (
                            -float(c["context_support"]),
                            int(c["generic_rank"]),
                        ),
                    )
                    for c in ranked_context[:5]:
                        candidate_evidence.append(
                            {
                                "candidate": c["candidate"],
                                "generic_rank": int(c["generic_rank"]),
                                "frequency_support": float(c["frequency_support"]),
                                "context_support": float(c["context_support"]),
                                "normalized_generic_score": float(c["normalized_generic_score"]),
                            }
                        )

                rec = {
                    "group": label,
                    "row_id": qid,
                    "author": a,
                    "document_id": document_id(q),
                    "query_position": qpos,
                    "pinyin_segments": list(qpy),
                    "pinyin": " ".join(qpy),
                    "gold": qgold,  # analysis-only oracle label
                    "G": g_pred,
                    "F": f_pred,
                    "Hidden_M1": h_pred,
                    "visible_history_count": len(visible),
                    "same_pinyin_history_count": len(same),
                    "distinct_targets": len(counts),
                    "frequency_winner": freq_winner,
                    "frequency_winner_count": freq_winner_count,
                    "frequency_winner_share": freq_share,
                    "frequency_second_count": second_count,
                    "frequency_margin_count": freq_margin,
                    "target_distribution": top_counts[:10],
                    "current_context": context(q)[-180:].replace("\n", " "),
                    "top_context_candidate_evidence": candidate_evidence,
                }
                groups[label].append(rec)

            # Strictly prior: add current position only after evaluation.
            for p0, _, _, row in group:
                visible.append((p0, row))

            i = j

    args.output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "schema_version": 1,
        "experiment": "em3_context_outcome_examples",
        "authors": authors,
        "history_budget": args.history_budget,
        "test_used": False,
        "groups": {},
    }

    def choose_diverse(records, n):
        # Prefer strong-history examples while avoiding repeated author/Pinyin/Gold patterns.
        ordered = sorted(
            records,
            key=lambda r: (
                -r["same_pinyin_history_count"],
                -r["frequency_winner_share"],
                r["author"],
                r["row_id"],
            ),
        )
        chosen = []
        seen = set()
        for r in ordered:
            key = (r["author"], r["pinyin"], r["gold"])
            if key in seen:
                continue
            chosen.append(r)
            seen.add(key)
            if len(chosen) >= n:
                return chosen
        for r in ordered:
            if r not in chosen:
                chosen.append(r)
                if len(chosen) >= n:
                    break
        return chosen

    for name, records in groups.items():
        path = args.output_root / f"{name.lower()}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        summary["groups"][name] = {
            "rows": len(records),
            "authors": dict(Counter(r["author"] for r in records)),
        }

        print("\n" + "#" * 110)
        print(name)
        print("TOTAL:", len(records))

        for r in choose_diverse(records, args.examples_per_group):
            print("\n" + "=" * 110)
            print("ROW:", r["row_id"])
            print("AUTHOR:", r["author"])
            print("DOCUMENT:", r["document_id"])
            print("PINYIN:", r["pinyin"])
            print("GOLD [analysis only]:", r["gold"])
            print("G:", r["G"], "| F:", r["F"], "| Hidden-M1:", r["Hidden_M1"])
            print("VISIBLE HISTORY:", r["visible_history_count"])
            print("SAME-PINYIN HISTORY:", r["same_pinyin_history_count"])
            print("DISTINCT TARGETS:", r["distinct_targets"])
            print(
                "FREQUENCY WINNER:",
                r["frequency_winner"],
                f"count={r['frequency_winner_count']}",
                f"share={r['frequency_winner_share']:.3f}",
                f"margin={r['frequency_margin_count']}",
            )
            print("TARGET DISTRIBUTION:", r["target_distribution"])
            print("CURRENT CONTEXT:")
            print(r["current_context"])
            print("\nTOP CONTEXT-EVIDENCE CANDIDATES:")
            for c in r["top_context_candidate_evidence"]:
                print(
                    " ",
                    c["candidate"],
                    f"| G-rank={c['generic_rank']}",
                    f"| freq={c['frequency_support']:.4f}",
                    f"| ctx={c['context_support']:.4f}",
                    f"| Gscore={c['normalized_generic_score']:.4f}",
                )

    summary_path = args.output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
