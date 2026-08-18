from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


AUTHORS = {
    "Etinjat",
    "Re_spectators",
    "breaddddd",
}

A1_PATH = Path(
    r"results\personalisation\context_lab"
    r"\diagnostic_a1_retrieval\rows_full_short.jsonl"
)

F_PATH = Path(
    r"C:\Users\chiar\Desktop\LBH\thesis-personalisation"
    r"\results\personalisation\pilot_a_context_memory"
    r"\h5000\frequency_predictions.jsonl"
)

M1_PATH = Path(
    r"C:\Users\chiar\Desktop\LBH\thesis-personalisation"
    r"\results\personalisation\pilot_a_context_memory"
    r"\h5000\memory_predictions.jsonl"
)

M2_PATH = Path(
    r"C:\Users\chiar\Desktop\LBH\thesis-personalisation"
    r"\results\personalisation\m2_h5000"
    r"\m2_predictions.jsonl"
)

OUT = Path(
    r"results\personalisation\context_lab"
    r"\diagnostic_a2_decision"
)


def load_index(
    path: Path,
    key: str,
) -> dict[str, dict[str, Any]]:
    result = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            if row.get("author") not in AUTHORS:
                continue

            k = str(row[key])

            if k in result:
                raise RuntimeError(
                    f"duplicate key in {path}: {k}"
                )

            result[k] = row

    return result


def final_top1(row: Mapping[str, Any]) -> str:
    hits = [
        candidate["candidate"]
        for candidate in row["candidates"]
        if int(candidate["rank"]) == 1
    ]

    if len(hits) != 1:
        raise RuntimeError(
            f"expected exactly one final rank-1: {hits}"
        )

    return str(hits[0])


def generic_top1(row: Mapping[str, Any]) -> str:
    hits = [
        candidate["candidate"]
        for candidate in row["candidates"]
        if int(candidate["generic_rank"]) == 1
    ]

    if len(hits) != 1:
        raise RuntimeError(
            f"expected exactly one generic rank-1: {hits}"
        )

    return str(hits[0])


def candidate_record(
    row: Mapping[str, Any],
    candidate_text: str,
) -> Mapping[str, Any] | None:
    for candidate in row["candidates"]:
        if candidate["candidate"] == candidate_text:
            return candidate

    return None


def evidence_has_gold(
    row: Mapping[str, Any],
    gold: str,
) -> bool:
    return any(
        str(evidence["historical_target"]) == gold
        for evidence in row.get(
            "retrieved_evidence",
            [],
        )
    )


def rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "not_retrieved"

    if rank == 1:
        return "rank_1"

    if rank <= 3:
        return "rank_2_3"

    if rank <= 5:
        return "rank_4_5"

    if rank <= 10:
        return "rank_6_10"

    if rank <= 20:
        return "rank_11_20"

    return "rank_gt_20"


def build_rows() -> list[dict[str, Any]]:
    a1 = load_index(A1_PATH, "row_id")
    f = load_index(F_PATH, "condition_id")
    m1 = load_index(M1_PATH, "condition_id")
    m2 = load_index(M2_PATH, "condition_id")

    shared = set(a1) & set(f) & set(m1) & set(m2)

    if len(shared) != 3000:
        raise RuntimeError(
            f"expected 3000 shared rows, got {len(shared)}"
        )

    output = []

    for row_id in sorted(shared):
        a = a1[row_id]
        fr = f[row_id]
        mr1 = m1[row_id]
        mr2 = m2[row_id]

        gold = str(a["gold"])

        if not (
            gold == fr["gold"]
            == mr1["gold"]
            == mr2["gold"]
        ):
            raise RuntimeError(
                f"Gold mismatch: {row_id}"
            )

        g_top = generic_top1(fr)
        f_top = final_top1(fr)
        m1_top = final_top1(mr1)
        m2_top = final_top1(mr2)

        generic_correct = g_top == gold
        f_correct = f_top == gold
        m1_correct = m1_top == gold
        m2_correct = m2_top == gold

        gold_rank = a["gold_retrieval_rank"]

        m1_gold_evidence = evidence_has_gold(
            mr1,
            gold,
        )

        m2_gold_evidence = evidence_has_gold(
            mr2,
            gold,
        )

        f_gold_candidate = candidate_record(
            fr,
            gold,
        )

        m1_gold_candidate = candidate_record(
            mr1,
            gold,
        )

        m2_gold_candidate = candidate_record(
            mr2,
            gold,
        )

        output.append(
            {
                "row_id": row_id,
                "anchor_id": a["anchor_id"],
                "author": a["author"],
                "gold": gold,
                "pinyin_segments": a[
                    "pinyin_segments"
                ],

                "history_available": bool(
                    a["history_available"]
                ),
                "gold_history_exists": bool(
                    a["gold_history_exists"]
                ),
                "ambiguous": bool(a["ambiguous"]),
                "conflict": bool(a["conflict"]),
                "visible_history_count": int(
                    a["visible_history_count"]
                ),
                "gold_history_count": int(
                    a["gold_history_count"]
                ),
                "frequency_winner": a[
                    "frequency_winner"
                ],

                "gold_retrieval_rank": gold_rank,
                "gold_retrieval_bucket": (
                    rank_bucket(gold_rank)
                ),

                "generic_top1": g_top,
                "frequency_top1": f_top,
                "m1_top1": m1_top,
                "m2_top1": m2_top,

                "generic_correct": generic_correct,
                "frequency_correct": f_correct,
                "m1_correct": m1_correct,
                "m2_correct": m2_correct,

                "m1_gold_evidence_present": (
                    m1_gold_evidence
                ),
                "m2_gold_evidence_present": (
                    m2_gold_evidence
                ),

                "f_gold_frequency_count": (
                    f_gold_candidate.get(
                        "frequency_count"
                    )
                    if f_gold_candidate
                    else None
                ),

                "m1_gold_memory_score": (
                    m1_gold_candidate.get(
                        "memory_score"
                    )
                    if m1_gold_candidate
                    else None
                ),

                "m2_gold_support": (
                    m2_gold_candidate.get(
                        "m2_support"
                    )
                    if m2_gold_candidate
                    else None
                ),

                # ------------------------------
                # F <-> M1 transitions
                # ------------------------------

                "f_wrong_m1_correct": (
                    (not f_correct)
                    and m1_correct
                ),

                "f_correct_m1_wrong": (
                    f_correct
                    and (not m1_correct)
                ),

                "m1_unique_context_rescue": (
                    (not generic_correct)
                    and (not f_correct)
                    and m1_correct
                ),

                "m1_protects_generic_from_f": (
                    generic_correct
                    and (not f_correct)
                    and m1_correct
                ),

                "m1_strong_regression": (
                    generic_correct
                    and f_correct
                    and (not m1_correct)
                ),

                "m1_evidence_available_but_wrong": (
                    bool(a["gold_history_exists"])
                    and m1_gold_evidence
                    and (not m1_correct)
                ),

                # ------------------------------
                # F <-> M2 transitions
                # ------------------------------

                "f_wrong_m2_correct": (
                    (not f_correct)
                    and m2_correct
                ),

                "f_correct_m2_wrong": (
                    f_correct
                    and (not m2_correct)
                ),

                "m2_unique_context_rescue": (
                    (not generic_correct)
                    and (not f_correct)
                    and m2_correct
                ),

                "m2_protects_generic_from_f": (
                    generic_correct
                    and (not f_correct)
                    and m2_correct
                ),

                "m2_strong_regression": (
                    generic_correct
                    and f_correct
                    and (not m2_correct)
                ),

                "m2_evidence_available_but_wrong": (
                    bool(a["gold_history_exists"])
                    and m2_gold_evidence
                    and (not m2_correct)
                ),
            }
        )

    return output


def select_subset(
    rows: list[dict[str, Any]],
    subset: str,
) -> list[dict[str, Any]]:
    if subset == "overall":
        return rows

    if subset == "history_available":
        return [
            r for r in rows
            if r["history_available"]
        ]

    if subset == "gold_history_exists":
        return [
            r for r in rows
            if r["gold_history_exists"]
        ]

    if subset == "ambiguous":
        return [
            r for r in rows
            if r["ambiguous"]
        ]

    if subset == "conflict":
        return [
            r for r in rows
            if r["conflict"]
        ]

    raise ValueError(subset)


def summarize(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    n = len(rows)

    def count(name: str) -> int:
        return sum(bool(r[name]) for r in rows)

    result = {
        "rows": n,

        "generic_correct": count(
            "generic_correct"
        ),
        "frequency_correct": count(
            "frequency_correct"
        ),
        "m1_correct": count("m1_correct"),
        "m2_correct": count("m2_correct"),

        "f_wrong_m1_correct": count(
            "f_wrong_m1_correct"
        ),
        "f_correct_m1_wrong": count(
            "f_correct_m1_wrong"
        ),

        "m1_net_vs_f": (
            count("f_wrong_m1_correct")
            - count("f_correct_m1_wrong")
        ),

        "f_wrong_m2_correct": count(
            "f_wrong_m2_correct"
        ),
        "f_correct_m2_wrong": count(
            "f_correct_m2_wrong"
        ),

        "m2_net_vs_f": (
            count("f_wrong_m2_correct")
            - count("f_correct_m2_wrong")
        ),

        "m1_unique_context_rescue": count(
            "m1_unique_context_rescue"
        ),
        "m1_protects_generic_from_f": count(
            "m1_protects_generic_from_f"
        ),
        "m1_strong_regression": count(
            "m1_strong_regression"
        ),
        "m1_evidence_available_but_wrong": count(
            "m1_evidence_available_but_wrong"
        ),

        "m2_unique_context_rescue": count(
            "m2_unique_context_rescue"
        ),
        "m2_protects_generic_from_f": count(
            "m2_protects_generic_from_f"
        ),
        "m2_strong_regression": count(
            "m2_strong_regression"
        ),
        "m2_evidence_available_but_wrong": count(
            "m2_evidence_available_but_wrong"
        ),

        "gold_retrieval_buckets": dict(
            Counter(
                r["gold_retrieval_bucket"]
                for r in rows
                if r["gold_history_exists"]
            )
        ),
    }

    if n:
        for method in (
            "generic",
            "frequency",
            "m1",
            "m2",
        ):
            result[f"{method}_accuracy"] = (
                result[f"{method}_correct"] / n
            )

    return result


def transition_by_rank(
    rows: list[dict[str, Any]],
    transition: str,
) -> dict[str, int]:
    return dict(
        Counter(
            row["gold_retrieval_bucket"]
            for row in rows
            if row[transition]
        )
    )


def main() -> None:
    rows = build_rows()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    row_path = OUT / "rows.jsonl"

    with row_path.open(
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

    subsets = {}

    for subset in (
        "overall",
        "history_available",
        "gold_history_exists",
        "ambiguous",
        "conflict",
    ):
        selected = select_subset(
            rows,
            subset,
        )

        subsets[subset] = {
            "summary": summarize(selected),

            "m1_rescue_by_gold_rank": (
                transition_by_rank(
                    selected,
                    "f_wrong_m1_correct",
                )
            ),

            "m1_harm_by_gold_rank": (
                transition_by_rank(
                    selected,
                    "f_correct_m1_wrong",
                )
            ),

            "m2_rescue_by_gold_rank": (
                transition_by_rank(
                    selected,
                    "f_wrong_m2_correct",
                )
            ),

            "m2_harm_by_gold_rank": (
                transition_by_rank(
                    selected,
                    "f_correct_m2_wrong",
                )
            ),
        }

    per_author = {}

    for author in sorted(AUTHORS):
        author_rows = [
            row
            for row in rows
            if row["author"] == author
        ]

        per_author[author] = {
            "overall": summarize(
                author_rows
            ),
            "conflict": summarize(
                [
                    row
                    for row in author_rows
                    if row["conflict"]
                ]
            ),
        }

    summary = {
        "schema_version": 1,
        "experiment": (
            "context_lab_diagnostic_a2_decision"
        ),
        "condition": "full_short",
        "history_budget": 5000,
        "authors": sorted(AUTHORS),

        "m1_evidence_depth": 5,
        "m2_evidence_depth": 20,

        "important_note": (
            "M1 evidence availability is taken "
            "from its actual retrieved_evidence "
            "(Top5), while M2 uses its actual "
            "Top20 retrieved_evidence."
        ),

        "subsets": subsets,
        "per_author": per_author,
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
        "=== DIAGNOSTIC A2 — FULL+SHORT ==="
    )

    for subset in (
        "overall",
        "gold_history_exists",
        "ambiguous",
        "conflict",
    ):
        s = subsets[subset]["summary"]

        print("\n[" + subset + "]")

        print(
            "rows =",
            s["rows"],
        )

        print(
            "accuracy:",
            "Generic=",
            f"{s.get('generic_accuracy', 0):.4%}",
            "F=",
            f"{s.get('frequency_accuracy', 0):.4%}",
            "M1=",
            f"{s.get('m1_accuracy', 0):.4%}",
            "M2=",
            f"{s.get('m2_accuracy', 0):.4%}",
        )

        print(
            "M1:",
            "rescue=",
            s["f_wrong_m1_correct"],
            "harm=",
            s["f_correct_m1_wrong"],
            "net=",
            s["m1_net_vs_f"],
            "unique_context_rescue=",
            s["m1_unique_context_rescue"],
            "protect_generic=",
            s["m1_protects_generic_from_f"],
            "strong_regression=",
            s["m1_strong_regression"],
            "evidence_available_but_wrong=",
            s["m1_evidence_available_but_wrong"],
        )

        print(
            "M2:",
            "rescue=",
            s["f_wrong_m2_correct"],
            "harm=",
            s["f_correct_m2_wrong"],
            "net=",
            s["m2_net_vs_f"],
            "unique_context_rescue=",
            s["m2_unique_context_rescue"],
            "protect_generic=",
            s["m2_protects_generic_from_f"],
            "strong_regression=",
            s["m2_strong_regression"],
            "evidence_available_but_wrong=",
            s["m2_evidence_available_but_wrong"],
        )

        print(
            "M1 rescue by rank =",
            subsets[subset][
                "m1_rescue_by_gold_rank"
            ],
        )

        print(
            "M1 harm by rank =",
            subsets[subset][
                "m1_harm_by_gold_rank"
            ],
        )

        print(
            "M2 rescue by rank =",
            subsets[subset][
                "m2_rescue_by_gold_rank"
            ],
        )

        print(
            "M2 harm by rank =",
            subsets[subset][
                "m2_harm_by_gold_rank"
            ],
        )


if __name__ == "__main__":
    main()
