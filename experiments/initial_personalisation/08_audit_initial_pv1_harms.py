from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

EXPECTED_ROWS = 34416
EXPECTED_PREDICTIONS_SHA256 = (
    "7fd8aa158d8cd50bced36b55610f8d932bc65e3aae1dbbd5bd65907ff1707ea7"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def top_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return min(candidates, key=lambda row: int(row["rank"]))


def personal_candidate(row: dict[str, Any]) -> dict[str, Any] | None:
    for candidate in row.get("pv1_candidates", []):
        if candidate.get("source") == "personal_vocabulary":
            return candidate
    return None


def enrich(row: dict[str, Any], transition: str) -> dict[str, Any]:
    f_top = top_candidate(row.get("frequency_candidates", []))
    pv1_top = top_candidate(row.get("pv1_candidates", []))
    personal = personal_candidate(row)

    return {
        "row_id": row["row_id"],
        "anchor_id": row.get("anchor_id"),
        "author": row["author"],
        "work_id": row.get("work_id"),
        "chronological_position": row.get("chronological_position"),
        "transition": transition,
        "gold": row["gold"],
        "pinyin_segments": row.get("pinyin_segments"),
        "generic_rank": row.get("generic_rank"),
        "frequency_rank": row.get("frequency_rank"),
        "pv1_rank": row.get("pv1_rank"),
        "generic_missing": bool(row.get("generic_missing")),
        "history_available": bool(row.get("history_available")),
        "ambiguous": bool(row.get("ambiguous")),
        "conflict": bool(row.get("conflict")),
        "compatible_recoverable_missing": bool(
            row.get("compatible_recoverable_missing")
        ),
        "selected_surface_contains_gold": bool(
            row.get("selected_surface_contains_gold")
        ),
        "f_top1": None if f_top is None else f_top.get("candidate"),
        "f_top1_score": None if f_top is None else f_top.get("final_score"),
        "pv1_top1": None if pv1_top is None else pv1_top.get("candidate"),
        "pv1_top1_source": None if pv1_top is None else pv1_top.get("source"),
        "pv1_top1_score": None if pv1_top is None else pv1_top.get("final_score"),
        "selected_personal_candidate": (
            None if personal is None else personal.get("candidate")
        ),
        "selected_personal_rank": (
            None if personal is None else personal.get("rank")
        ),
        "selected_personal_final_score": (
            None if personal is None else personal.get("final_score")
        ),
        "selected_personal_frequency_count": (
            None if personal is None else personal.get("frequency_count")
        ),
        "selected_personal_frequency_support": (
            None if personal is None else personal.get("frequency_support")
        ),
        "selected_personal_lexicon_provenance": (
            None if personal is None else personal.get("lexicon_provenance")
        ),
        "selected_k_pv": row.get("selected_k_pv"),
        "selected_lambda_frequency": row.get("selected_lambda_frequency"),
        "selected_lambda_pv": row.get("selected_lambda_pv"),
        "dev3000_used": bool(row.get("dev3000_used")),
        "test_used": bool(row.get("test_used")),
    }


def transition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    authors = Counter(str(row["author"]) for row in rows)

    def count_flag(name: str) -> int:
        return sum(bool(row.get(name)) for row in rows)

    return {
        "n": n,
        "authors": dict(sorted(authors.items())),
        "ambiguous_n": count_flag("ambiguous"),
        "ambiguous_rate": count_flag("ambiguous") / n if n else 0.0,
        "conflict_n": count_flag("conflict"),
        "conflict_rate": count_flag("conflict") / n if n else 0.0,
        "history_available_n": count_flag("history_available"),
        "generic_missing_n": count_flag("generic_missing"),
        "selected_surface_contains_gold_n": count_flag(
            "selected_surface_contains_gold"
        ),
        "pv1_top1_personal_n": sum(
            row.get("pv1_top1_source") == "personal_vocabulary"
            for row in rows
        ),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU-only F -> PV1 harm/rescue audit for Initial B2."
    )
    parser.add_argument("--b2-predictions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    actual_sha = sha256_file(args.b2_predictions)
    if actual_sha != EXPECTED_PREDICTIONS_SHA256:
        raise RuntimeError(
            "B2 predictions SHA mismatch:\n"
            f"expected={EXPECTED_PREDICTIONS_SHA256}\n"
            f"actual={actual_sha}"
        )

    rows = read_jsonl(args.b2_predictions)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, found {len(rows)}"
        )

    # Protocol and frozen-parameter guards.
    for row in rows:
        if row.get("dev3000_used") is not False:
            raise RuntimeError("Dev3000-marked row found")
        if row.get("test_used") is not False:
            raise RuntimeError("Test-marked row found")
        if int(row.get("selected_k_pv", -1)) != 1:
            raise RuntimeError("Unexpected selected_k_pv")
        if float(row.get("selected_lambda_frequency", -1)) != 4.0:
            raise RuntimeError("Unexpected selected_lambda_frequency")
        if float(row.get("selected_lambda_pv", -1)) != 4.0:
            raise RuntimeError("Unexpected selected_lambda_pv")

    harms = []
    rescues = []
    unchanged_correct = 0
    unchanged_wrong = 0

    for row in rows:
        f_correct = row.get("frequency_rank") == 1
        pv1_correct = row.get("pv1_rank") == 1

        if f_correct and not pv1_correct:
            harms.append(enrich(row, "harm"))
        elif not f_correct and pv1_correct:
            rescues.append(enrich(row, "rescue"))
        elif f_correct and pv1_correct:
            unchanged_correct += 1
        else:
            unchanged_wrong += 1

    if (len(rescues), len(harms), len(rescues) - len(harms)) != (933, 304, 629):
        raise RuntimeError(
            "F -> PV1 transition regression failed: "
            f"rescue={len(rescues)} harm={len(harms)} "
            f"net={len(rescues) - len(harms)}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)

    write_jsonl(args.output_root / "pv1_harm_rows.jsonl", harms)
    write_jsonl(args.output_root / "pv1_rescue_rows.jsonl", rescues)

    authors = sorted({str(row["author"]) for row in rows})
    by_author = []
    for author in authors:
        author_harms = [row for row in harms if row["author"] == author]
        author_rescues = [row for row in rescues if row["author"] == author]
        by_author.append(
            {
                "author": author,
                "harm": len(author_harms),
                "rescue": len(author_rescues),
                "net": len(author_rescues) - len(author_harms),
                "harm_conflict": sum(row["conflict"] for row in author_harms),
                "rescue_conflict": sum(row["conflict"] for row in author_rescues),
                "harm_pv1_top1_personal": sum(
                    row["pv1_top1_source"] == "personal_vocabulary"
                    for row in author_harms
                ),
                "rescue_pv1_top1_personal": sum(
                    row["pv1_top1_source"] == "personal_vocabulary"
                    for row in author_rescues
                ),
            }
        )

    with (args.output_root / "pv1_harm_rescue_by_author.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=list(by_author[0]))
        writer.writeheader()
        writer.writerows(by_author)

    summary = {
        "schema_version": 1,
        "experiment": "initial_short_b2_pv1_harm_rescue_audit_v1",
        "status": "complete",
        "rows": len(rows),
        "b2_predictions_sha256": actual_sha,
        "f_to_pv1": {
            "rescue": len(rescues),
            "harm": len(harms),
            "net": len(rescues) - len(harms),
            "unchanged_correct": unchanged_correct,
            "unchanged_wrong": unchanged_wrong,
        },
        "harm": transition_summary(harms),
        "rescue": transition_summary(rescues),
        "by_author": by_author,
        "selected_hyperparameters": {
            "k_pv": 1,
            "lambda_frequency": 4.0,
            "lambda_pv": 4.0,
        },
        "dev3000_used": False,
        "test_used": False,
        "new_pinyingpt_inference": False,
        "parameter_tuning": False,
    }

    summary_path = args.output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== INITIAL B2 PV1 HARM / RESCUE AUDIT ===")
    print(f"Rows: {len(rows)}")
    print(
        f"F -> PV1 rescue={len(rescues)} "
        f"harm={len(harms)} net={len(rescues) - len(harms)}"
    )
    print(
        f"Harm conflict: {summary['harm']['conflict_n']}/"
        f"{summary['harm']['n']} "
        f"({summary['harm']['conflict_rate']:.3%})"
    )
    print(
        f"Rescue conflict: {summary['rescue']['conflict_n']}/"
        f"{summary['rescue']['n']} "
        f"({summary['rescue']['conflict_rate']:.3%})"
    )
    print(
        f"Harm with personal candidate at PV1 Top1: "
        f"{summary['harm']['pv1_top1_personal_n']}/{summary['harm']['n']}"
    )
    print(f"Summary: {summary_path}")
    print("Dev3000 used: false")
    print("Test used: false")


if __name__ == "__main__":
    main()
