from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

EXPECTED_ROWS = 34416
HISTORY_BUDGET = 5000
K_VALUES = (1, 3, 5)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "row_id" not in row:
                raise RuntimeError(f"Missing row_id at line {line_no}: {path}")
            rows.append(row)
    return rows


def target_of(row: Mapping[str, Any]) -> str:
    value = row.get("target", row.get("gold"))
    if value is None:
        raise RuntimeError(f"History row has no target/gold: {row.get('row_id')}")
    return str(value)


def gold_of(row: Mapping[str, Any]) -> str:
    value = row.get("gold", row.get("target"))
    if value is None:
        raise RuntimeError(f"Query row has no gold/target: {row.get('row_id')}")
    return str(value)


def generic_candidates(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = row.get("top10_candidates")
    if not isinstance(values, list) or not values:
        raise RuntimeError(f"Missing Generic candidates for {row.get('row_id')}")
    output = []
    seen: set[str] = set()
    for expected_rank, item in enumerate(values, 1):
        text = str(item["text"])
        rank = int(item["rank"])
        if rank != expected_rank:
            raise RuntimeError(f"Non-contiguous Generic rank for {row.get('row_id')}")
        if text in seen:
            raise RuntimeError(f"Duplicate Generic candidate for {row.get('row_id')}: {text}")
        seen.add(text)
        output.append(dict(item))
    return output


def generic_rank(row: Mapping[str, Any], gold: str) -> int | None:
    for item in generic_candidates(row):
        if str(item["text"]) == gold:
            return int(item["rank"])
    return None


def compatibility(backend: Any, target: str, pinyin: tuple[str, ...]) -> tuple[bool, str]:
    chars = list(target)
    if len(chars) != len(pinyin):
        return False, "character_count_mismatch"
    token_ids = backend.tokenizer.convert_tokens_to_ids(chars)
    for idx, (token_id, segment) in enumerate(zip(token_ids, pinyin)):
        if token_id == backend.tokenizer.unk_token_id:
            return False, f"tokenizer_unknown_at_{idx}"
        if token_id not in backend.allowed_token_ids.get(segment, ()):
            return False, f"pinyin_incompatible_at_{idx}"
    return True, "compatible"


def frequency_state(counts: Counter[str], gold: str) -> dict[str, Any]:
    if not counts:
        return {
            "ambiguous": False,
            "frequency_winner": None,
            "frequency_winner_count": 0,
            "frequency_winner_tied": False,
            "conflict": False,
        }
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    max_count = ordered[0][1]
    tied = sum(count == max_count for _, count in ordered) > 1
    winner = None if tied else ordered[0][0]
    return {
        "ambiguous": len(counts) >= 2,
        "frequency_winner": winner,
        "frequency_winner_count": max_count,
        "frequency_winner_tied": tied,
        "conflict": bool(winner is not None and winner != gold),
    }


def candidate_record(
    target: str,
    rank: int,
    rows: Sequence[Mapping[str, Any]],
    total_visible: int,
) -> dict[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (int(row["chronological_position"]), str(row["row_id"])),
    )
    return {
        "text": target,
        "personal_rank": rank,
        "frequency_count": len(ordered),
        "frequency_share": len(ordered) / total_visible if total_visible else 0.0,
        "first_history_row_id": str(ordered[0]["row_id"]),
        "last_history_row_id": str(ordered[-1]["row_id"]),
        "first_chronological_position": int(ordered[0]["chronological_position"]),
        "last_chronological_position": int(ordered[-1]["chronological_position"]),
    }


def safe_rate(n: int, d: int) -> float | None:
    return n / d if d else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B1: freeze one unified backend-compatible Initial personal candidate surface for PV1 and EM1."
    )
    parser.add_argument("--initial-train-fit", type=Path, required=True)
    parser.add_argument("--initial-train-val", type=Path, required=True)
    parser.add_argument("--generic-predictions", type=Path, required=True)
    parser.add_argument("--a2-recoverability-rows", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    fit = load_jsonl(args.initial_train_fit)
    val = load_jsonl(args.initial_train_val)
    generic = load_jsonl(args.generic_predictions)
    a2 = load_jsonl(args.a2_recoverability_rows)

    if len(val) != EXPECTED_ROWS or len(generic) != EXPECTED_ROWS or len(a2) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} Val/Generic/A2 rows; found {len(val)}/{len(generic)}/{len(a2)}"
        )

    val_by_id = {str(row["row_id"]): row for row in val}
    generic_by_id = {str(row["row_id"]): row for row in generic}
    a2_by_id = {str(row["row_id"]): row for row in a2}
    if len(val_by_id) != EXPECTED_ROWS:
        raise RuntimeError("Duplicate Train-Val row_id")
    if set(val_by_id) != set(generic_by_id) or set(val_by_id) != set(a2_by_id):
        raise RuntimeError("Train-Val, Generic, and A2 row_id surfaces differ")

    from src.personalisation.context_memory import PredictionQuery
    from src.personalisation.pilot_a import HistoryIndex
    from src.reference_backend_pinyingpt import PinyinGPTConcatBackend

    # Query-specific rolling H5000: same author -> strictly prior -> latest 5000 RAW rows -> Initial match.
    history_index = HistoryIndex(fit + val, HISTORY_BUDGET)
    backend = PinyinGPTConcatBackend(args.checkpoint, device=args.device)

    args.output_root.mkdir(parents=True, exist_ok=True)
    surface_path = args.output_root / "train_val_candidate_surface.jsonl"

    incompatibility_reasons: Counter[str] = Counter()
    author_stats: dict[str, Counter[str]] = defaultdict(Counter)
    personal_sizes: list[int] = []
    crosscheck_failures: list[dict[str, Any]] = []
    headline = Counter()

    with surface_path.open("w", encoding="utf-8", newline="\n") as out:
        for number, row in enumerate(val, 1):
            rid = str(row["row_id"])
            pred = generic_by_id[rid]
            a2_row = a2_by_id[rid]
            author = str(row["author"])
            gold = gold_of(row)
            pinyin = tuple(str(value) for value in row["pinyin_segments"])

            query = PredictionQuery(
                row_id=rid,
                author=author,
                work_id=str(row["work_id"]),
                chronological_position=int(row["chronological_position"]),
                context=str(row["context"]),
                pinyin=pinyin,
            )
            visible = history_index.visible(query)

            grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for history_row in visible:
                grouped[target_of(history_row)].append(history_row)
            counts = Counter({target: len(rows) for target, rows in grouped.items()})
            lexicon = sorted(counts, key=lambda target: (-counts[target], target))

            generic_list = generic_candidates(pred)
            generic_texts = {str(item["text"]) for item in generic_list}
            raw_personal_only = [target for target in lexicon if target not in generic_texts]

            compatible_personal_only: list[str] = []
            for target in raw_personal_only:
                ok, reason = compatibility(backend, target, pinyin)
                if ok:
                    compatible_personal_only.append(target)
                else:
                    incompatibility_reasons[reason] += 1

            top5_targets = compatible_personal_only[: max(K_VALUES)]
            personal_records = [
                candidate_record(target, rank, grouped[target], len(visible))
                for rank, target in enumerate(top5_targets, 1)
            ]
            personal_sizes.append(len(compatible_personal_only))

            g_rank = generic_rank(pred, gold)
            missing = g_rank is None
            freq = frequency_state(counts, gold)

            # Independent regression against A2. Candidate selection must not drift.
            expected_top5 = [str(value) for value in a2_row.get("compatible_personal_only_targets_top5", [])]
            observed_top5 = [item["text"] for item in personal_records]
            expected_missing = bool(a2_row["generic_missing"])
            mismatch_fields: list[str] = []
            if observed_top5 != expected_top5:
                mismatch_fields.append("compatible_personal_only_targets_top5")
            if missing != expected_missing:
                mismatch_fields.append("generic_missing")
            for k in K_VALUES:
                observed_recoverable = bool(missing and gold in observed_top5[:k])
                if observed_recoverable != bool(a2_row[f"compatible_recoverable_at_{k}"]):
                    mismatch_fields.append(f"compatible_recoverable_at_{k}")
            if mismatch_fields:
                crosscheck_failures.append({"row_id": rid, "fields": mismatch_fields})
                if len(crosscheck_failures) >= 20:
                    raise RuntimeError(
                        "B1 diverged from A2 on >=20 rows; first failures: "
                        + json.dumps(crosscheck_failures[:5], ensure_ascii=False)
                    )

            pools: dict[str, list[str]] = {}
            for k in K_VALUES:
                injected = observed_top5[:k]
                pools[str(k)] = [str(item["text"]) for item in generic_list] + injected

                injected_n = len(injected)
                headline[f"injected_candidates_k{k}"] += injected_n
                if injected_n:
                    headline[f"rows_with_personal_candidate_k{k}"] += 1
                recovered = bool(missing and gold in injected)
                if recovered:
                    headline[f"gold_recovered_k{k}"] += 1
                    author_stats[author][f"gold_recovered_k{k}"] += 1

            if missing:
                headline["generic_missing"] += 1
                author_stats[author]["generic_missing"] += 1
            if compatible_personal_only:
                headline["rows_with_any_compatible_personal_only"] += 1
            if bool(missing and gold in compatible_personal_only):
                headline["compatible_recoverable_any"] += 1
                author_stats[author]["compatible_recoverable_any"] += 1

            author_stats[author]["rows"] += 1
            author_stats[author]["ambiguous"] += int(freq["ambiguous"])
            author_stats[author]["conflict"] += int(freq["conflict"])

            output = {
                "schema_version": 1,
                "experiment": "initial_short_standardized_b1_candidate_surface_v1",
                "row_id": rid,
                "anchor_id": row.get("anchor_id"),
                "condition_id": row.get("condition_id"),
                "author": author,
                "work_id": str(row["work_id"]),
                "chronological_position": int(row["chronological_position"]),
                "context": str(row["context"]),
                "pinyin_input": str(row.get("pinyin_input", " ".join(pinyin))),
                "pinyin_segments": list(pinyin),
                "gold": gold,
                "history_budget": HISTORY_BUDGET,
                "same_initial_history_count": len(visible),
                "distinct_initial_targets": len(counts),
                **freq,
                "generic_rank": g_rank,
                "generic_missing": missing,
                "generic_candidates": generic_list,
                "raw_personal_only_count": len(raw_personal_only),
                "compatible_personal_only_count": len(compatible_personal_only),
                "personal_candidates_top5": personal_records,
                "personal_candidate_texts_top5": observed_top5,
                "candidate_pool_texts_by_k": pools,
                "gold_compatible_recoverable_any": bool(missing and gold in compatible_personal_only),
                **{
                    f"gold_compatible_recoverable_at_{k}": bool(missing and gold in observed_top5[:k])
                    for k in K_VALUES
                },
                "candidate_selection_used_gold": False,
                "dev3000_used": False,
                "test_used": False,
            }
            out.write(canonical_json(output) + "\n")

            if number % 1000 == 0 or number == len(val):
                print(f"B1 surface: {number}/{len(val)}", flush=True)

    if crosscheck_failures:
        raise RuntimeError(
            "B1/A2 cross-check failed: " + json.dumps(crosscheck_failures[:10], ensure_ascii=False)
        )

    missing_n = headline["generic_missing"]
    per_k: dict[str, Any] = {}
    for k in K_VALUES:
        recovered = headline[f"gold_recovered_k{k}"]
        per_k[str(k)] = {
            "rows_with_at_least_one_personal_candidate": headline[f"rows_with_personal_candidate_k{k}"],
            "injected_candidates_total": headline[f"injected_candidates_k{k}"],
            "gold_recovered_from_generic_missing_n": recovered,
            "gold_recovered_given_generic_missing": safe_rate(recovered, missing_n),
        }

    per_author: dict[str, Any] = {}
    for author in sorted(author_stats):
        stats = author_stats[author]
        author_missing = stats["generic_missing"]
        per_author[author] = {
            "rows": stats["rows"],
            "generic_missing_n": author_missing,
            "ambiguous_n": stats["ambiguous"],
            "conflict_n": stats["conflict"],
            "compatible_recoverable_any_n": stats["compatible_recoverable_any"],
            "compatible_recoverable_any_given_missing": safe_rate(
                stats["compatible_recoverable_any"], author_missing
            ),
            **{
                f"gold_recovered_at_{k}_given_missing": safe_rate(
                    stats[f"gold_recovered_k{k}"], author_missing
                )
                for k in K_VALUES
            },
        }

    summary = {
        "schema_version": 1,
        "experiment": "initial_short_standardized_b1_candidate_surface_v1",
        "status": "complete",
        "rows": len(val),
        "history_budget": HISTORY_BUDGET,
        "history_semantics": (
            "same author -> strictly prior -> latest up-to-5000 RAW interactions -> "
            "exact Initial-segment match -> frequency lexicon -> remove Generic Top10 -> backend compatibility"
        ),
        "personal_candidate_order": "frequency descending, lexical tie-break",
        "k_values": list(K_VALUES),
        "candidate_surface_rule": (
            "PV1 and EM1 must use the same backend-compatible personal candidate list; "
            "K takes the prefix of personal_candidates_top5."
        ),
        "candidate_selection_used_gold": False,
        "generic_missing_n": missing_n,
        "generic_missing_rate": safe_rate(missing_n, len(val)),
        "rows_with_any_compatible_personal_only": headline["rows_with_any_compatible_personal_only"],
        "compatible_recoverable_any_n": headline["compatible_recoverable_any"],
        "compatible_recoverable_any_given_missing": safe_rate(
            headline["compatible_recoverable_any"], missing_n
        ),
        "personal_compatible_candidate_count": {
            "mean": statistics.fmean(personal_sizes) if personal_sizes else 0.0,
            "median": statistics.median(personal_sizes) if personal_sizes else 0.0,
            "max": max(personal_sizes) if personal_sizes else 0,
        },
        "per_k": per_k,
        "per_author": per_author,
        "backend_incompatibility_reasons": dict(incompatibility_reasons),
        "a2_crosscheck": {
            "rows_checked": len(val),
            "failures": 0,
            "status": "passed",
        },
        "provenance": {
            "initial_train_fit_path": str(args.initial_train_fit.resolve()),
            "initial_train_fit_sha256": sha256_file(args.initial_train_fit),
            "initial_train_val_path": str(args.initial_train_val.resolve()),
            "initial_train_val_sha256": sha256_file(args.initial_train_val),
            "generic_predictions_path": str(args.generic_predictions.resolve()),
            "generic_predictions_sha256": sha256_file(args.generic_predictions),
            "a2_recoverability_rows_path": str(args.a2_recoverability_rows.resolve()),
            "a2_recoverability_rows_sha256": sha256_file(args.a2_recoverability_rows),
            "checkpoint_path": str(args.checkpoint.resolve()),
            "candidate_surface_path": str(surface_path.resolve()),
            "candidate_surface_sha256": sha256_file(surface_path),
        },
        "dev3000_used": False,
        "test_used": False,
    }

    summary_path = args.output_root / "candidate_surface_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "artifact": "Initial+Short standardized unified candidate surface",
        "status": "frozen_for_train_val_method_comparison",
        "surface_file": str(surface_path.resolve()),
        "surface_sha256": summary["provenance"]["candidate_surface_sha256"],
        "summary_file": str(summary_path.resolve()),
        "summary_sha256": sha256_file(summary_path),
        "history_budget": HISTORY_BUDGET,
        "k_values": list(K_VALUES),
        "gold_used_for_candidate_selection": False,
        "intended_consumers": ["PV1", "EM1-R", "EM1-R+F", "same-surface scoring ablation"],
        "dev3000_used": False,
        "test_used": False,
    }
    manifest_path = args.output_root / "candidate_surface_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\n=== B1 UNIFIED INITIAL CANDIDATE SURFACE COMPLETE ===")
    print("Rows:", len(val))
    print("Generic Missing@10:", summary["generic_missing_rate"])
    print("Compatible recoverable | missing:", summary["compatible_recoverable_any_given_missing"])
    for k in K_VALUES:
        print(
            f"Recovered@{k} | missing:",
            summary["per_k"][str(k)]["gold_recovered_given_generic_missing"],
        )
    print("A2 cross-check: PASSED", len(val), "rows")
    print("Surface:", surface_path)
    print("Surface SHA256:", summary["provenance"]["candidate_surface_sha256"])
    print("Manifest:", manifest_path)
    print("Dev3000 used: false")
    print("Test used: false")


if __name__ == "__main__":
    main()
