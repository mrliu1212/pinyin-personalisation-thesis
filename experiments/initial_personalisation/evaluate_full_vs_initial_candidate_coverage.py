from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

EXPECTED_ROWS = 34416
K_VALUES = (1, 3, 5)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def gold_of(row: dict[str, Any]) -> str:
    value = row.get("gold", row.get("target"))
    if value is None:
        raise RuntimeError("Row has neither gold nor target")
    return str(value)


def rank_of_generic(row: dict[str, Any], gold: str) -> int | None:
    candidates = row.get("top10_candidates", [])
    if not isinstance(candidates, list):
        raise RuntimeError("top10_candidates is not a list")
    for candidate in candidates:
        if str(candidate["text"]) == gold:
            return int(candidate["rank"])
    return None


def candidate_texts(row: dict[str, Any]) -> set[str]:
    candidates = row.get("top10_candidates", [])
    if not isinstance(candidates, list):
        raise RuntimeError("top10_candidates is not a list")
    return {str(candidate["text"]) for candidate in candidates}


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def metrics_from_ranks(ranks: Iterable[int | None]) -> dict[str, Any]:
    values = list(ranks)
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "top1": None,
            "top3": None,
            "mrr_at_10": None,
            "missing_at_10": None,
            "recall_at_10": None,
        }
    missing = sum(rank is None for rank in values)
    return {
        "n": n,
        "top1": sum(rank == 1 for rank in values) / n,
        "top3": sum(rank is not None and rank <= 3 for rank in values) / n,
        "mrr_at_10": sum(0.0 if rank is None else 1.0 / rank for rank in values) / n,
        "missing_at_10": missing / n,
        "recall_at_10": 1.0 - missing / n,
    }


def compatibility(backend: Any, target: str, pinyin: tuple[str, ...]) -> tuple[bool, str]:
    characters = list(target)
    if len(characters) != len(pinyin):
        return False, "character_count_mismatch"
    token_ids = backend.tokenizer.convert_tokens_to_ids(characters)
    for index, (token_id, segment) in enumerate(zip(token_ids, pinyin)):
        if token_id == backend.tokenizer.unk_token_id:
            return False, f"tokenizer_unknown_at_{index}"
        if token_id not in backend.allowed_token_ids.get(segment, ()):
            return False, f"pinyin_incompatible_at_{index}"
    return True, "compatible"


def unique_index(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, ""))
        if not value:
            raise RuntimeError(f"{label}: missing {key}")
        if value in output:
            raise RuntimeError(f"{label}: duplicate {key}: {value}")
        output[value] = row
    return output


def resolve_generic_by_full_row(
    full_val: list[dict[str, Any]],
    generic_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], str]:
    val_row_ids = {str(row["row_id"]) for row in full_val}
    generic_row_ids = {str(row.get("row_id", "")) for row in generic_rows}
    if generic_row_ids == val_row_ids:
        return unique_index(generic_rows, "row_id", "Full Generic"), "row_id"

    val_anchors = {str(row["anchor_id"]) for row in full_val}
    generic_anchors = {str(row.get("anchor_id", "")) for row in generic_rows}
    if generic_anchors == val_anchors:
        return unique_index(generic_rows, "anchor_id", "Full Generic"), "anchor_id"

    raise RuntimeError(
        "Full Generic predictions match neither the Full Train-Val row_id surface "
        "nor the anchor_id surface"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only standardized Full-vs-Initial candidate-coverage and "
            "recoverability decomposition on the same 34,416 Train-Val anchors."
        )
    )
    parser.add_argument("--full-train-fit", type=Path, required=True)
    parser.add_argument("--full-train-val", type=Path, required=True)
    parser.add_argument("--full-generic-predictions", type=Path, required=True)
    parser.add_argument("--initial-train-val", type=Path, required=True)
    parser.add_argument("--initial-recoverability-rows", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    full_fit = load_jsonl(args.full_train_fit)
    full_val = load_jsonl(args.full_train_val)
    full_generic = load_jsonl(args.full_generic_predictions)
    initial_val = load_jsonl(args.initial_train_val)
    initial_recovery = load_jsonl(args.initial_recoverability_rows)

    if len(full_val) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} Full Train-Val rows; found {len(full_val)}")
    if len(full_generic) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} Full Generic rows; found {len(full_generic)}")
    if len(initial_val) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} Initial Train-Val rows; found {len(initial_val)}")
    if len(initial_recovery) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} Initial A2 rows; found {len(initial_recovery)}")

    full_by_anchor = unique_index(full_val, "anchor_id", "Full Train-Val")
    initial_by_anchor = unique_index(initial_val, "anchor_id", "Initial Train-Val")
    if set(full_by_anchor) != set(initial_by_anchor):
        raise RuntimeError("Full and Initial Train-Val anchor surfaces differ")

    # Strong same-surface audit: Initial must be only a deterministic pinyin transform.
    for anchor_id in sorted(full_by_anchor):
        full_row = full_by_anchor[anchor_id]
        initial_row = initial_by_anchor[anchor_id]
        for field in ("author", "work_id", "chronological_position", "context"):
            if full_row.get(field) != initial_row.get(field):
                raise RuntimeError(f"Full/Initial semantic mismatch for {anchor_id}: {field}")
        if gold_of(full_row) != gold_of(initial_row):
            raise RuntimeError(f"Full/Initial Gold mismatch for {anchor_id}")
        expected_initial = [str(segment)[0].lower() for segment in full_row["pinyin_segments"]]
        actual_initial = [str(segment).lower() for segment in initial_row["pinyin_segments"]]
        if expected_initial != actual_initial:
            raise RuntimeError(
                f"Initial transform mismatch for {anchor_id}: "
                f"expected={expected_initial} actual={actual_initial}"
            )

    initial_recovery_by_row = unique_index(initial_recovery, "row_id", "Initial A2")
    if set(initial_recovery_by_row) != {str(row["row_id"]) for row in initial_val}:
        raise RuntimeError("Initial A2 row_id surface differs from Initial Train-Val")

    full_generic_index, generic_key = resolve_generic_by_full_row(full_val, full_generic)

    from src.personalisation.context_memory import PredictionQuery
    from src.personalisation.pilot_a import HistoryIndex
    from src.reference_backend_pinyingpt import PinyinGPTConcatBackend

    history_index = HistoryIndex(full_fit + full_val, 5000)
    backend = PinyinGPTConcatBackend(args.checkpoint, device=args.device)

    incompatibility_reasons: Counter[str] = Counter()
    full_rows_output: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []

    for number, full_row in enumerate(full_val, start=1):
        row_id = str(full_row["row_id"])
        anchor_id = str(full_row["anchor_id"])
        initial_row = initial_by_anchor[anchor_id]
        initial_a2 = initial_recovery_by_row[str(initial_row["row_id"])]

        generic_lookup_key = row_id if generic_key == "row_id" else anchor_id
        prediction = full_generic_index[generic_lookup_key]

        gold = gold_of(full_row)
        pinyin = tuple(str(value) for value in full_row["pinyin_segments"])
        query = PredictionQuery(
            row_id=row_id,
            author=str(full_row["author"]),
            work_id=str(full_row["work_id"]),
            chronological_position=int(full_row["chronological_position"]),
            context=str(full_row["context"]),
            pinyin=pinyin,
        )
        visible = history_index.visible(query)
        counts = Counter(str(row.get("target", row.get("gold"))) for row in visible)
        lexicon = sorted(counts, key=lambda target: (-counts[target], target))

        generic_surface = candidate_texts(prediction)
        personal_only = [target for target in lexicon if target not in generic_surface]

        compatible_personal: list[str] = []
        for target in personal_only:
            ok, reason = compatibility(backend, target, pinyin)
            if ok:
                compatible_personal.append(target)
            else:
                incompatibility_reasons[reason] += 1

        full_rank = rank_of_generic(prediction, gold)
        full_missing = full_rank is None
        raw_any = bool(full_missing and gold in personal_only)
        compatible_any = bool(full_missing and gold in compatible_personal)
        raw_at = {k: bool(full_missing and gold in personal_only[:k]) for k in K_VALUES}
        compatible_at = {k: bool(full_missing and gold in compatible_personal[:k]) for k in K_VALUES}

        full_result = {
            "row_id": row_id,
            "anchor_id": anchor_id,
            "author": str(full_row["author"]),
            "gold": gold,
            "full_pinyin_segments": list(pinyin),
            "generic_rank": full_rank,
            "generic_missing": full_missing,
            "history_available": bool(visible),
            "same_full_pinyin_history_count": len(visible),
            "distinct_full_pinyin_targets": len(counts),
            "raw_personal_only_count": len(personal_only),
            "compatible_personal_only_count": len(compatible_personal),
            "gold_in_history": gold in counts,
            "raw_recoverable_any": raw_any,
            "compatible_recoverable_any": compatible_any,
            **{f"raw_recoverable_at_{k}": raw_at[k] for k in K_VALUES},
            **{f"compatible_recoverable_at_{k}": compatible_at[k] for k in K_VALUES},
        }
        full_rows_output.append(full_result)

        initial_rank = initial_a2["generic_rank"]
        initial_missing = bool(initial_a2["generic_missing"])
        paired_rows.append(
            {
                "anchor_id": anchor_id,
                "full_row_id": row_id,
                "initial_row_id": str(initial_row["row_id"]),
                "author": str(full_row["author"]),
                "gold": gold,
                "full_pinyin_segments": list(pinyin),
                "initial_pinyin_segments": list(initial_row["pinyin_segments"]),
                "full_generic_rank": full_rank,
                "initial_generic_rank": initial_rank,
                "full_generic_missing": full_missing,
                "initial_generic_missing": initial_missing,
                "full_covered_initial_missing": (not full_missing) and initial_missing,
                "full_missing_initial_covered": full_missing and (not initial_missing),
                "both_missing": full_missing and initial_missing,
                "both_covered": (not full_missing) and (not initial_missing),
                "full_top1": full_rank == 1,
                "initial_top1": initial_rank == 1,
                "initial_compatible_recoverable_any": bool(initial_a2["compatible_recoverable_any"]),
                **{
                    f"initial_compatible_recoverable_at_{k}": bool(
                        initial_a2[f"compatible_recoverable_at_{k}"]
                    )
                    for k in K_VALUES
                },
                "full_compatible_recoverable_any": compatible_any,
                **{f"full_compatible_recoverable_at_{k}": compatible_at[k] for k in K_VALUES},
            }
        )

        if number % 1000 == 0 or number == len(full_val):
            print(f"Full-vs-Initial audit: {number}/{len(full_val)}", flush=True)

    full_metrics = metrics_from_ranks(row["generic_rank"] for row in full_rows_output)
    initial_metrics = metrics_from_ranks(row["initial_generic_rank"] for row in paired_rows)

    full_missing_n = sum(row["generic_missing"] for row in full_rows_output)
    initial_missing_n = sum(row["initial_generic_missing"] for row in paired_rows)

    full_recoverable = sum(row["compatible_recoverable_any"] for row in full_rows_output)
    initial_recoverable = sum(row["initial_compatible_recoverable_any"] for row in paired_rows)

    transitions = {
        "full_covered_initial_missing": sum(row["full_covered_initial_missing"] for row in paired_rows),
        "full_missing_initial_covered": sum(row["full_missing_initial_covered"] for row in paired_rows),
        "both_missing": sum(row["both_missing"] for row in paired_rows),
        "both_covered": sum(row["both_covered"] for row in paired_rows),
    }
    transitions.update({key + "_rate": value / EXPECTED_ROWS for key, value in list(transitions.items())})

    initial_loss_rows = [row for row in paired_rows if row["full_covered_initial_missing"]]
    both_missing_rows = [row for row in paired_rows if row["both_missing"]]

    initial_loss_recovery = {
        "n": len(initial_loss_rows),
        "compatible_recoverable_any": sum(
            row["initial_compatible_recoverable_any"] for row in initial_loss_rows
        ),
    }
    initial_loss_recovery["compatible_recoverable_any_rate"] = safe_rate(
        initial_loss_recovery["compatible_recoverable_any"], initial_loss_recovery["n"]
    )
    for k in K_VALUES:
        count = sum(row[f"initial_compatible_recoverable_at_{k}"] for row in initial_loss_rows)
        initial_loss_recovery[f"compatible_recoverable_at_{k}"] = count
        initial_loss_recovery[f"compatible_recoverable_at_{k}_rate"] = safe_rate(
            count, initial_loss_recovery["n"]
        )

    both_missing_recovery = {
        "n": len(both_missing_rows),
        "compatible_recoverable_any": sum(
            row["initial_compatible_recoverable_any"] for row in both_missing_rows
        ),
    }
    both_missing_recovery["compatible_recoverable_any_rate"] = safe_rate(
        both_missing_recovery["compatible_recoverable_any"], both_missing_recovery["n"]
    )

    authors = sorted({row["author"] for row in paired_rows})
    per_author: dict[str, Any] = {}
    for author in authors:
        rows = [row for row in paired_rows if row["author"] == author]
        full_author = metrics_from_ranks(row["full_generic_rank"] for row in rows)
        initial_author = metrics_from_ranks(row["initial_generic_rank"] for row in rows)
        per_author[author] = {
            "rows": len(rows),
            "full": full_author,
            "initial": initial_author,
            "top1_delta_initial_minus_full": initial_author["top1"] - full_author["top1"],
            "missing_delta_initial_minus_full": (
                initial_author["missing_at_10"] - full_author["missing_at_10"]
            ),
        }

    macro_full_top1 = statistics.fmean(values["full"]["top1"] for values in per_author.values())
    macro_initial_top1 = statistics.fmean(values["initial"]["top1"] for values in per_author.values())

    summary = {
        "schema_version": 1,
        "experiment": "standardized_full_vs_initial_candidate_coverage_train_val_v1",
        "status": "complete",
        "rows": EXPECTED_ROWS,
        "authors": authors,
        "paired_surface": {
            "identity": "same frozen 34,416 Train-Val anchor IDs",
            "assertions": [
                "same author",
                "same work_id",
                "same chronological_position",
                "same context",
                "same Gold",
                "Initial pinyin_segments == first letter of each Full pinyin segment",
            ],
        },
        "full": {
            "generic": full_metrics,
            "macro_author_top1": macro_full_top1,
            "compatible_recoverable_any_n": full_recoverable,
            "compatible_recoverable_given_missing": safe_rate(full_recoverable, full_missing_n),
            **{
                f"compatible_recoverable_at_{k}_given_missing": safe_rate(
                    sum(row[f"compatible_recoverable_at_{k}"] for row in full_rows_output),
                    full_missing_n,
                )
                for k in K_VALUES
            },
        },
        "initial": {
            "generic": initial_metrics,
            "macro_author_top1": macro_initial_top1,
            "compatible_recoverable_any_n": initial_recoverable,
            "compatible_recoverable_given_missing": safe_rate(initial_recoverable, initial_missing_n),
            **{
                f"compatible_recoverable_at_{k}_given_missing": safe_rate(
                    sum(row[f"initial_compatible_recoverable_at_{k}"] for row in paired_rows),
                    initial_missing_n,
                )
                for k in K_VALUES
            },
        },
        "paired_deltas": {
            "top1_initial_minus_full": initial_metrics["top1"] - full_metrics["top1"],
            "missing_at_10_initial_minus_full": (
                initial_metrics["missing_at_10"] - full_metrics["missing_at_10"]
            ),
            "missing_at_10_ratio_initial_over_full": (
                initial_metrics["missing_at_10"] / full_metrics["missing_at_10"]
                if full_metrics["missing_at_10"]
                else None
            ),
        },
        "coverage_transitions": transitions,
        "initial_only_candidate_loss": {
            "definition": "Full Top10 contains Gold but Initial Top10 misses Gold on the same anchor",
            **initial_loss_recovery,
        },
        "both_conditions_missing": both_missing_recovery,
        "per_author": per_author,
        "full_backend_incompatibility_reasons": dict(incompatibility_reasons),
        "provenance": {
            "full_train_fit_sha256": sha256_file(args.full_train_fit),
            "full_train_val_sha256": sha256_file(args.full_train_val),
            "full_generic_predictions_sha256": sha256_file(args.full_generic_predictions),
            "initial_train_val_sha256": sha256_file(args.initial_train_val),
            "initial_recoverability_rows_sha256": sha256_file(args.initial_recoverability_rows),
            "checkpoint_path": str(args.checkpoint.resolve()),
            "full_generic_join_key": generic_key,
        },
        "dev3000_used": False,
        "test_used": False,
        "interpretation_boundary": (
            "This paired Train-Val audit can localize candidate-coverage loss associated with "
            "the deterministic Full-to-Initial abbreviation on the same anchors. It does not "
            "by itself prove that candidate coverage is the only cause of the Top1 gap."
        ),
    }

    args.output_root.mkdir(parents=True, exist_ok=True)

    full_rows_path = args.output_root / "full_recoverability_rows.jsonl"
    with full_rows_path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in full_rows_output:
            destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    paired_rows_path = args.output_root / "full_vs_initial_paired_rows.jsonl"
    with paired_rows_path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in paired_rows:
            destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary["provenance"]["full_recoverability_rows_sha256"] = sha256_file(full_rows_path)
    summary["provenance"]["paired_rows_sha256"] = sha256_file(paired_rows_path)

    summary_path = args.output_root / "full_vs_initial_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    csv_path = args.output_root / "full_vs_initial_headline.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=[
                "condition",
                "rows",
                "macro_author_top1",
                "micro_top1",
                "top3",
                "mrr_at_10",
                "missing_at_10",
                "recall_at_10",
                "compatible_recoverable_given_missing",
                "compatible_recoverable_at_1_given_missing",
                "compatible_recoverable_at_3_given_missing",
                "compatible_recoverable_at_5_given_missing",
            ],
        )
        writer.writeheader()
        for condition in ("full", "initial"):
            values = summary[condition]
            generic_values = values["generic"]
            writer.writerow(
                {
                    "condition": condition,
                    "rows": EXPECTED_ROWS,
                    "macro_author_top1": values["macro_author_top1"],
                    "micro_top1": generic_values["top1"],
                    "top3": generic_values["top3"],
                    "mrr_at_10": generic_values["mrr_at_10"],
                    "missing_at_10": generic_values["missing_at_10"],
                    "recall_at_10": generic_values["recall_at_10"],
                    "compatible_recoverable_given_missing": values[
                        "compatible_recoverable_given_missing"
                    ],
                    "compatible_recoverable_at_1_given_missing": values[
                        "compatible_recoverable_at_1_given_missing"
                    ],
                    "compatible_recoverable_at_3_given_missing": values[
                        "compatible_recoverable_at_3_given_missing"
                    ],
                    "compatible_recoverable_at_5_given_missing": values[
                        "compatible_recoverable_at_5_given_missing"
                    ],
                }
            )

    print("\n=== STANDARDIZED FULL vs INITIAL CANDIDATE COVERAGE ===")
    print(f"Rows: {EXPECTED_ROWS}")
    print(f"Full Generic Top1: {full_metrics['top1']:.6f}")
    print(f"Initial Generic Top1: {initial_metrics['top1']:.6f}")
    print(f"Full Missing@10: {full_metrics['missing_at_10']:.6f}")
    print(f"Initial Missing@10: {initial_metrics['missing_at_10']:.6f}")
    print(
        "Missing delta (Initial-Full): "
        f"{summary['paired_deltas']['missing_at_10_initial_minus_full']:.6f}"
    )
    ratio = summary["paired_deltas"]["missing_at_10_ratio_initial_over_full"]
    print(f"Missing ratio (Initial/Full): {ratio:.3f}" if ratio is not None else "Missing ratio: n/a")
    print(
        "Full-covered -> Initial-missing: "
        f"{transitions['full_covered_initial_missing']} "
        f"({transitions['full_covered_initial_missing_rate']:.3%})"
    )
    print(
        "Initial-loss recoverable from compatible personal history: "
        f"{initial_loss_recovery['compatible_recoverable_any_rate']}"
    )
    print(f"Summary: {summary_path}")
    print("Dev3000 used: false")
    print("Test used: false")


if __name__ == "__main__":
    main()
