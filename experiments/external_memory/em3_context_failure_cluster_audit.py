from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import mean, median
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


def numeric_summary(values):
    values = list(values)
    if not values:
        return None
    values_sorted = sorted(values)
    n = len(values_sorted)

    def q(p):
        if n == 1:
            return values_sorted[0]
        x = (n - 1) * p
        lo = math.floor(x)
        hi = math.ceil(x)
        if lo == hi:
            return values_sorted[lo]
        return values_sorted[lo] * (hi - x) + values_sorted[hi] * (x - lo)

    return {
        "n": n,
        "mean": mean(values_sorted),
        "median": median(values_sorted),
        "min": min(values_sorted),
        "p25": q(0.25),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": max(values_sorted),
    }


def count_bins(values, edges):
    out = Counter()
    for value in values:
        placed = False
        for lo, hi, label in edges:
            if lo <= value <= hi:
                out[label] += 1
                placed = True
                break
        if not placed:
            out[f">{edges[-1][1]}"] += 1
    return dict(out)


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
        "--output-root",
        type=Path,
        default=Path(
            r"results\personalisation\external_memory"
            r"\em3_context_failure_cluster_audit"
        ),
    )
    p.add_argument("--authors", nargs="+", default=DEFAULT_AUTHORS)
    p.add_argument("--history-budget", type=int, default=H)
    return p.parse_args()


def main():
    args = parse_args()
    authors = list(dict.fromkeys(args.authors))
    author_set = set(authors)

    history_path = args.pilot_root / "history_manifest.jsonl"
    dev_path = args.pilot_root / "dev_manifest.jsonl"

    four = {rid(r): r for r in read_jsonl(args.four_way_rows)}
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

    subsets = {
        "g_right_hidden_wrong": [],
        "f_right_hidden_wrong": [],
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

                if not ((g_ok and not h_ok) or (f_ok and not h_ok)):
                    continue

                qpy = pinyin(q)
                qgold = gold(q)

                same = [h for _, h in visible if pinyin(h) == qpy]
                positives = [h for h in same if gold(h) == qgold]
                negatives = [h for h in same if gold(h) != qgold]
                counts = Counter(gold(h) for h in same)

                rec = {
                    "row_id": qid,
                    "author": a,
                    "query_position": qpos,
                    "document_id": document_id(q),
                    "pinyin_segments": list(qpy),
                    "pinyin": " ".join(qpy),
                    "gold": qgold,
                    "positive_count": len(positives),
                    "negative_count": len(negatives),
                    "same_pinyin_history_count": len(same),
                    "distinct_targets": len(counts),
                    "gold_share": (
                        len(positives) / len(same) if same else 0.0
                    ),
                    "target_counts": counts.most_common(),
                    "G_rank": ff.get("G_rank"),
                    "F_rank": ff.get("F_rank"),
                    "Hidden_M1_rank": ff.get("Hidden_M1_rank"),
                }

                if g_ok and not h_ok:
                    subsets["g_right_hidden_wrong"].append(rec)
                if f_ok and not h_ok:
                    subsets["f_right_hidden_wrong"].append(rec)

            for p0, _, _, row in group:
                visible.append((p0, row))
            i = j

    args.output_root.mkdir(parents=True, exist_ok=True)

    bin_edges = [
        (0, 0, "0"),
        (1, 1, "1"),
        (2, 2, "2"),
        (3, 5, "3-5"),
        (6, 10, "6-10"),
        (11, 20, "11-20"),
        (21, 50, "21-50"),
        (51, 10**9, "51+"),
    ]

    summary = {
        "schema_version": 1,
        "experiment": "em3_context_failure_cluster_audit",
        "history_budget": args.history_budget,
        "authors": authors,
        "test_used": False,
        "subsets": {},
    }

    for subset_name, records in subsets.items():
        rows_path = args.output_root / f"{subset_name}.jsonl"
        with rows_path.open("w", encoding="utf-8", newline="\n") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        pattern_counter = Counter(
            (r["author"], r["pinyin"], r["gold"]) for r in records
        )
        pinyin_gold_counter = Counter(
            (r["pinyin"], r["gold"]) for r in records
        )
        author_counter = Counter(r["author"] for r in records)
        doc_counter = Counter(
            (r["author"], r["document_id"])
            for r in records
            if r["document_id"] is not None
        )

        # Position clustering within the same author/Pinyin/Gold pattern.
        gaps = []
        per_pattern_positions = defaultdict(list)
        for r in records:
            per_pattern_positions[
                (r["author"], r["pinyin"], r["gold"])
            ].append(r["query_position"])
        for positions in per_pattern_positions.values():
            positions = sorted(positions)
            gaps.extend(
                b - a for a, b in zip(positions, positions[1:])
            )

        top_patterns = [
            {
                "author": key[0],
                "pinyin": key[1],
                "gold": key[2],
                "count": count,
            }
            for key, count in pattern_counter.most_common(20)
        ]

        top_pinyin_gold = [
            {
                "pinyin": key[0],
                "gold": key[1],
                "count": count,
            }
            for key, count in pinyin_gold_counter.most_common(20)
        ]

        top_documents = [
            {
                "author": key[0],
                "document_id": key[1],
                "count": count,
            }
            for key, count in doc_counter.most_common(20)
        ]

        positive_counts = [r["positive_count"] for r in records]
        negative_counts = [r["negative_count"] for r in records]
        gold_shares = [r["gold_share"] for r in records]
        distinct_targets = [r["distinct_targets"] for r in records]

        unique_patterns = len(pattern_counter)
        repeated_pattern_rows = sum(
            count for count in pattern_counter.values() if count > 1
        )

        summary["subsets"][subset_name] = {
            "rows": len(records),
            "authors": dict(author_counter),
            "unique_author_pinyin_gold_patterns": unique_patterns,
            "unique_pinyin_gold_patterns": len(pinyin_gold_counter),
            "rows_in_repeated_author_pinyin_gold_patterns": repeated_pattern_rows,
            "repeated_pattern_row_share": (
                repeated_pattern_rows / len(records) if records else 0.0
            ),
            "largest_pattern_size": max(pattern_counter.values(), default=0),
            "top_author_pinyin_gold_patterns": top_patterns,
            "top_pinyin_gold_patterns": top_pinyin_gold,
            "positive_count": numeric_summary(positive_counts),
            "positive_count_bins": count_bins(positive_counts, bin_edges),
            "negative_count": numeric_summary(negative_counts),
            "gold_share": numeric_summary(gold_shares),
            "distinct_targets": numeric_summary(distinct_targets),
            "document_identity_available_rows": sum(
                r["document_id"] is not None for r in records
            ),
            "unique_author_documents": len(doc_counter),
            "top_documents": top_documents,
            "same_pattern_position_gap": numeric_summary(gaps),
        }

    summary_path = args.output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
