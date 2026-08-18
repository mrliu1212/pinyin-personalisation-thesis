from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


A2 = Path(
    r"results\personalisation\context_lab"
    r"\diagnostic_a2_decision\rows.jsonl"
)

F = Path(
    r"C:\Users\chiar\Desktop\LBH\thesis-personalisation"
    r"\results\personalisation\pilot_a_context_memory"
    r"\h5000\frequency_predictions.jsonl"
)

M1 = Path(
    r"C:\Users\chiar\Desktop\LBH\thesis-personalisation"
    r"\results\personalisation\pilot_a_context_memory"
    r"\h5000\memory_predictions.jsonl"
)

M2 = Path(
    r"C:\Users\chiar\Desktop\LBH\thesis-personalisation"
    r"\results\personalisation\m2_h5000"
    r"\m2_predictions.jsonl"
)

OUT = Path(
    r"results\personalisation\context_lab"
    r"\diagnostic_a2b_evidence_competition_v2"
)

AUTHORS = {
    "Etinjat",
    "Re_spectators",
    "breaddddd",
}


def load(path: Path, key: str):
    out = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            if row.get("author") not in AUTHORS:
                continue

            k = str(row[key])

            if k in out:
                raise RuntimeError(f"duplicate key: {k}")

            out[k] = row

    return out


def candidate(row, text):
    for c in row["candidates"]:
        if c["candidate"] == text:
            return c
    return None


def evidence_for(row, target):
    return [
        e
        for e in row.get("retrieved_evidence", [])
        if str(e["historical_target"]) == target
    ]


def get_support(row, text, field):
    c = candidate(row, text)
    if c is None:
        return None
    return c.get(field)


def compare(a, b):
    if a is None or b is None:
        return "unavailable"

    if a > b:
        return "gold_gt_other"

    if a < b:
        return "other_gt_gold"

    return "equal"


def build_regression_case(a, frow, method_row, method):
    gold = a["gold"]

    if method == "m1":
        wrong_winner = a["m1_top1"]
        support_field = "memory_score"
    else:
        wrong_winner = a["m2_top1"]
        support_field = "m2_support"

    gold_support = get_support(
        method_row,
        gold,
        support_field,
    )

    winner_support = get_support(
        method_row,
        wrong_winner,
        support_field,
    )

    return {
        "row_id": a["row_id"],
        "author": a["author"],
        "gold": gold,
        "generic_top1": a["generic_top1"],
        "frequency_top1": a["frequency_top1"],
        "method_top1": wrong_winner,

        "gold_retrieval_rank": (
            a["gold_retrieval_rank"]
        ),

        "gold_support": gold_support,
        "wrong_winner_support": winner_support,

        "support_comparison": compare(
            gold_support,
            winner_support,
        ),

        "gold_evidence": evidence_for(
            method_row,
            gold,
        ),

        "wrong_winner_evidence": evidence_for(
            method_row,
            wrong_winner,
        ),

        "context": method_row["context"],
        "pinyin_segments": a["pinyin_segments"],
    }


def build_rescue_case(a, frow, method_row, method):
    gold = a["gold"]
    generic_wrong = a["generic_top1"]
    frequency_wrong = a["frequency_top1"]

    if method == "m1":
        support_field = "memory_score"
    else:
        support_field = "m2_support"

    gold_support = get_support(
        method_row,
        gold,
        support_field,
    )

    generic_wrong_support = get_support(
        method_row,
        generic_wrong,
        support_field,
    )

    frequency_wrong_support = get_support(
        method_row,
        frequency_wrong,
        support_field,
    )

    return {
        "row_id": a["row_id"],
        "author": a["author"],
        "gold": gold,

        "generic_wrong": generic_wrong,
        "frequency_wrong": frequency_wrong,
        "method_top1": (
            a["m1_top1"]
            if method == "m1"
            else a["m2_top1"]
        ),

        "generic_and_frequency_same": (
            generic_wrong == frequency_wrong
        ),

        "gold_retrieval_rank": (
            a["gold_retrieval_rank"]
        ),

        "gold_support": gold_support,

        "generic_wrong_support": (
            generic_wrong_support
        ),

        "frequency_wrong_support": (
            frequency_wrong_support
        ),

        "gold_vs_generic": compare(
            gold_support,
            generic_wrong_support,
        ),

        "gold_vs_frequency": compare(
            gold_support,
            frequency_wrong_support,
        ),

        "gold_evidence": evidence_for(
            method_row,
            gold,
        ),

        "generic_wrong_evidence": evidence_for(
            method_row,
            generic_wrong,
        ),

        "frequency_wrong_evidence": evidence_for(
            method_row,
            frequency_wrong,
        ),

        "context": method_row["context"],
        "pinyin_segments": a["pinyin_segments"],
    }


def summarize_regression(cases):
    return {
        "cases": len(cases),

        "gold_evidence_present": sum(
            bool(c["gold_evidence"])
            for c in cases
        ),

        "wrong_winner_evidence_present": sum(
            bool(c["wrong_winner_evidence"])
            for c in cases
        ),

        "support_comparison": dict(
            Counter(
                c["support_comparison"]
                for c in cases
            )
        ),
    }


def summarize_rescue(cases):
    return {
        "cases": len(cases),

        "generic_and_frequency_same": sum(
            c["generic_and_frequency_same"]
            for c in cases
        ),

        "gold_evidence_present": sum(
            bool(c["gold_evidence"])
            for c in cases
        ),

        "gold_vs_generic": dict(
            Counter(
                c["gold_vs_generic"]
                for c in cases
            )
        ),

        "gold_vs_frequency": dict(
            Counter(
                c["gold_vs_frequency"]
                for c in cases
            )
        ),

        "gold_retrieval_rank": dict(
            Counter(
                "none"
                if c["gold_retrieval_rank"] is None
                else str(c["gold_retrieval_rank"])
                for c in cases
            )
        ),
    }


def write_jsonl(path, rows):
    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def main():
    a2 = load(A2, "row_id")
    f = load(F, "condition_id")
    m1 = load(M1, "condition_id")
    m2 = load(M2, "condition_id")

    categories = {
        "m1_strong_regression": [],
        "m1_unique_rescue": [],
        "m2_strong_regression": [],
        "m2_unique_rescue": [],
    }

    for row_id, a in a2.items():
        if a["m1_strong_regression"]:
            categories[
                "m1_strong_regression"
            ].append(
                build_regression_case(
                    a,
                    f[row_id],
                    m1[row_id],
                    "m1",
                )
            )

        if a["m1_unique_context_rescue"]:
            categories[
                "m1_unique_rescue"
            ].append(
                build_rescue_case(
                    a,
                    f[row_id],
                    m1[row_id],
                    "m1",
                )
            )

        if a["m2_strong_regression"]:
            categories[
                "m2_strong_regression"
            ].append(
                build_regression_case(
                    a,
                    f[row_id],
                    m2[row_id],
                    "m2",
                )
            )

        if a["m2_unique_context_rescue"]:
            categories[
                "m2_unique_rescue"
            ].append(
                build_rescue_case(
                    a,
                    f[row_id],
                    m2[row_id],
                    "m2",
                )
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, rows in categories.items():
        write_jsonl(
            OUT / f"{name}.jsonl",
            rows,
        )

    summary = {
        "m1_strong_regression": (
            summarize_regression(
                categories["m1_strong_regression"]
            )
        ),

        "m1_unique_rescue": (
            summarize_rescue(
                categories["m1_unique_rescue"]
            )
        ),

        "m2_strong_regression": (
            summarize_regression(
                categories["m2_strong_regression"]
            )
        ),

        "m2_unique_rescue": (
            summarize_rescue(
                categories["m2_unique_rescue"]
            )
        ),
    }

    (OUT / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "=== A2b v2 EVIDENCE COMPETITION ==="
    )

    for name, s in summary.items():
        print()
        print(f"[{name}]")

        for key, value in s.items():
            print(key, "=", value)


if __name__ == "__main__":
    main()
