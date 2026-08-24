from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

EXPECTED_ROWS = 34416
EXPECTED_SURFACE_SHA256 = "205c0ba01cd0678d7a4341c503fa2e74cf126a70182ff582687025e4946764b2"
EXPECTED_GENERIC_SHA256 = "bd0fb4dc304e0b266b90fae6fe3ac65424d2f52b23fedfa881212706ba2c2873"
EXPECTED_B2_SELECTION_SHA256 = "39e4756e805a525ae1c7a1be577741629daaf1f322f3eea402e1469fe51a00d2"
FROZEN_K = 1
FROZEN_LAMBDA_FREQUENCY = 4.0
FROZEN_LAMBDA_PV1 = 4.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "row_id" not in row:
                raise RuntimeError(f"Missing row_id at {path}:{line_no}")
            rows.append(row)
    return rows


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    values = {str(row["row_id"]): row for row in rows}
    if len(values) != len(rows):
        raise RuntimeError(f"Duplicate row_id in existing cache: {path}")
    return values


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generic_candidates(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = row.get("top10_candidates")
    if not isinstance(values, list) or len(values) != 10:
        raise RuntimeError(f"Expected Generic Top10 for {row.get('row_id')}")
    return [dict(item) for item in values]


def generic_log_probability(item: Mapping[str, Any]) -> float:
    for key in ("log_probability", "generic_score", "score"):
        if key in item and item[key] is not None:
            return float(item[key])
    raise RuntimeError(f"Generic candidate has no log-probability-like score: {item}")


def validate_selection(path: Path) -> dict[str, Any]:
    actual = sha256_file(path)
    if actual != EXPECTED_B2_SELECTION_SHA256:
        raise RuntimeError(
            "B2 selection SHA256 differs. "
            f"Expected {EXPECTED_B2_SELECTION_SHA256}, got {actual}"
        )
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("candidate_surface_sha256") != EXPECTED_SURFACE_SHA256:
        raise RuntimeError("B2 selection points to a different B1 candidate surface")
    if selection.get("candidate_surface_used_gold") is not False:
        raise RuntimeError("B2 selection says candidate construction used Gold")
    if selection.get("dev3000_used") is not False or selection.get("test_used") is not False:
        raise RuntimeError("B2 selection crossed the Dev3000/Test boundary")
    frequency = selection.get("frequency", {})
    pv1 = selection.get("pv1", {})
    if float(frequency.get("selected_lambda_frequency")) != FROZEN_LAMBDA_FREQUENCY:
        raise RuntimeError("Frozen Frequency lambda is not 4.0")
    if int(pv1.get("selected_k_pv")) != FROZEN_K:
        raise RuntimeError("Frozen PV1 K is not 1")
    if float(pv1.get("selected_lambda_pv")) != FROZEN_LAMBDA_PV1:
        raise RuntimeError("Frozen PV1 lambda is not 4.0")
    if float(pv1.get("frozen_lambda_frequency")) != FROZEN_LAMBDA_FREQUENCY:
        raise RuntimeError("PV1 did not freeze Frequency lambda 4.0")
    return selection


def load_inputs(
    candidate_surface: Path,
    generic_predictions: Path,
    b2_selection: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    surface_sha = sha256_file(candidate_surface)
    if surface_sha != EXPECTED_SURFACE_SHA256:
        raise RuntimeError(
            "Frozen B1 surface SHA256 differs. "
            f"Expected {EXPECTED_SURFACE_SHA256}, got {surface_sha}"
        )
    generic_sha = sha256_file(generic_predictions)
    if generic_sha != EXPECTED_GENERIC_SHA256:
        raise RuntimeError(
            "Frozen Initial Generic predictions SHA256 differs. "
            f"Expected {EXPECTED_GENERIC_SHA256}, got {generic_sha}"
        )
    selection = validate_selection(b2_selection)

    surface = load_jsonl(candidate_surface)
    generic = load_jsonl(generic_predictions)
    if len(surface) != EXPECTED_ROWS or len(generic) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} surface/generic rows; got {len(surface)}/{len(generic)}"
        )
    surface_by_id = {str(row["row_id"]): row for row in surface}
    generic_by_id = {str(row["row_id"]): row for row in generic}
    if len(surface_by_id) != EXPECTED_ROWS or len(generic_by_id) != EXPECTED_ROWS:
        raise RuntimeError("Duplicate row IDs in B1 surface or Initial Generic predictions")
    if set(surface_by_id) != set(generic_by_id):
        raise RuntimeError("B1 surface and Initial Generic prediction row-ID surfaces differ")

    for row in surface:
        if row.get("candidate_selection_used_gold") is not False:
            raise RuntimeError(f"B1 row used Gold for candidate selection: {row['row_id']}")
        if row.get("dev3000_used") is not False or row.get("test_used") is not False:
            raise RuntimeError(f"B1 row crossed Dev3000/Test boundary: {row['row_id']}")

    return surface, generic_by_id, selection


def deterministic_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def choose_preflight_rows(
    surface: Sequence[Mapping[str, Any]],
    rows_per_author: int,
) -> list[Mapping[str, Any]]:
    by_author: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in surface:
        by_author[str(row["author"])].append(row)
    selected: list[Mapping[str, Any]] = []
    for author in sorted(by_author):
        ordered = sorted(by_author[author], key=lambda row: deterministic_key(str(row["row_id"])))
        if len(ordered) < rows_per_author:
            raise RuntimeError(f"{author}: not enough rows for preflight sample")
        selected.extend(ordered[:rows_per_author])
    return selected


def run_preflight(args: argparse.Namespace) -> None:
    surface, generic_by_id, selection = load_inputs(
        args.candidate_surface,
        args.generic_predictions,
        args.b2_selection,
    )
    from src.reference_backend_pinyingpt import PinyinGPTConcatBackend

    backend = PinyinGPTConcatBackend(args.checkpoint, device=args.device)
    selected = choose_preflight_rows(surface, args.rows_per_author)
    comparisons: list[dict[str, Any]] = []
    differences: list[float] = []

    for number, srow in enumerate(selected, 1):
        rid = str(srow["row_id"])
        grow = generic_by_id[rid]
        if "model_used_context" not in grow:
            raise RuntimeError(f"Generic prediction lacks model_used_context: {rid}")
        cached = generic_candidates(grow)
        positions = sorted({0, len(cached) // 2, len(cached) - 1})
        chosen = [cached[index] for index in positions]
        texts = [str(item["text"]) for item in chosen]
        fresh = backend.score_candidates(
            context=str(grow["model_used_context"]),
            typed_pinyin=tuple(str(value) for value in srow["pinyin_segments"]),
            candidates=tuple(texts),
        )
        fresh_by_text = {value.text: value for value in fresh}
        if set(fresh_by_text) != set(texts):
            raise RuntimeError(f"Fixed-score preflight candidate identity mismatch: {rid}")
        for item in chosen:
            text = str(item["text"])
            cached_score = generic_log_probability(item)
            fixed_score = float(fresh_by_text[text].log_probability)
            difference = abs(cached_score - fixed_score)
            differences.append(difference)
            comparisons.append(
                {
                    "row_id": rid,
                    "author": str(srow["author"]),
                    "candidate": text,
                    "generic_rank": int(item["rank"]),
                    "cached_log_probability": cached_score,
                    "fixed_log_probability": fixed_score,
                    "absolute_difference": difference,
                }
            )
        if number % 3 == 0 or number == len(selected):
            print(f"B3 preflight: {number}/{len(selected)} rows", flush=True)

    maximum = max(differences) if differences else math.inf
    passed = maximum <= args.compatibility_tolerance
    summary = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "experiment": "initial_short_standardized_b3_em1_exact_score_compatibility_v1",
        "rows_sampled": len(selected),
        "candidate_comparisons": len(comparisons),
        "rows_per_author": args.rows_per_author,
        "authors": sorted({str(row["author"]) for row in selected}),
        "compatibility_tolerance": args.compatibility_tolerance,
        "maximum_absolute_difference": maximum,
        "mean_absolute_difference": statistics.fmean(differences) if differences else None,
        "median_absolute_difference": statistics.median(differences) if differences else None,
        "compatibility_passed": passed,
        "candidate_surface_sha256": EXPECTED_SURFACE_SHA256,
        "generic_predictions_sha256": EXPECTED_GENERIC_SHA256,
        "b2_selection_sha256": EXPECTED_B2_SELECTION_SHA256,
        "frozen_k": FROZEN_K,
        "frozen_lambda_frequency": FROZEN_LAMBDA_FREQUENCY,
        "frozen_lambda_pv1": FROZEN_LAMBDA_PV1,
        "selection_population": selection["selection_population"],
        "gold_used_for_candidate_selection": False,
        "gold_used_for_scoring": False,
        "dev3000_used": False,
        "test_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "preflight_score_compatibility.json", summary)
    with (args.output_root / "preflight_comparisons.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        for row in comparisons:
            destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print("\n=== B3 EM1 EXACT-SCORE PREFLIGHT ===")
    print("Rows sampled:", len(selected))
    print("Candidate comparisons:", len(comparisons))
    print("Max absolute difference:", f"{maximum:.12g}")
    print("Tolerance:", f"{args.compatibility_tolerance:.1e}")
    print("Gate:", "PASS" if passed else "FAIL")
    print("Dev3000 used: false")
    print("Test used: false")
    if not passed:
        raise RuntimeError("B3 fixed-candidate score compatibility gate failed")


def run_score(args: argparse.Namespace) -> None:
    surface, generic_by_id, selection = load_inputs(
        args.candidate_surface,
        args.generic_predictions,
        args.b2_selection,
    )
    preflight_path = args.output_root / "preflight_score_compatibility.json"
    if not preflight_path.exists():
        raise RuntimeError("B3 preflight is missing; run --phase preflight first")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("compatibility_passed") is not True:
        raise RuntimeError("B3 preflight did not pass")
    if preflight.get("candidate_surface_sha256") != EXPECTED_SURFACE_SHA256:
        raise RuntimeError("B3 preflight belongs to a different candidate surface")
    if preflight.get("generic_predictions_sha256") != EXPECTED_GENERIC_SHA256:
        raise RuntimeError("B3 preflight belongs to different Generic predictions")

    from src.reference_backend_pinyingpt import PinyinGPTConcatBackend

    eligible: list[tuple[str, Mapping[str, Any], Mapping[str, Any], str]] = []
    for srow in surface:
        rid = str(srow["row_id"])
        targets = [str(value) for value in srow.get("personal_candidate_texts_top5", [])]
        if not targets:
            continue
        candidate = targets[0]  # frozen K=1 prefix; Gold is never consulted.
        grow = generic_by_id[rid]
        generic_texts = {str(item["text"]) for item in generic_candidates(grow)}
        if candidate in generic_texts:
            raise RuntimeError(f"Frozen personal-only candidate overlaps Generic Top10: {rid}")
        if "model_used_context" not in grow:
            raise RuntimeError(f"Generic prediction lacks model_used_context: {rid}")
        eligible.append((rid, srow, grow, candidate))

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / "recovered_candidate_scores.jsonl"
    existing = load_existing(output_path)
    eligible_by_id = {row[0]: row for row in eligible}
    stale = set(existing) - set(eligible_by_id)
    if stale:
        raise RuntimeError(f"Existing exact-score cache has {len(stale)} stale row IDs")
    for rid, cached in existing.items():
        scores = cached.get("scores", [])
        if len(scores) != 1:
            raise RuntimeError(f"Existing exact-score row does not contain exactly one score: {rid}")
        expected_candidate = eligible_by_id[rid][3]
        if str(scores[0].get("candidate")) != expected_candidate:
            raise RuntimeError(f"Existing exact-score candidate differs from frozen K=1 surface: {rid}")
        if cached.get("candidate_surface_sha256") != EXPECTED_SURFACE_SHA256:
            raise RuntimeError(f"Existing exact-score row belongs to another surface: {rid}")

    pending = [item for item in eligible if item[0] not in existing]
    print("\n=== B3 EM1 EXACT SCORING ===")
    print("Frozen K:", FROZEN_K)
    print("Frozen Frequency lambda:", FROZEN_LAMBDA_FREQUENCY)
    print("Candidate surface SHA256:", EXPECTED_SURFACE_SHA256)
    print("Eligible rows:", len(eligible))
    print("Already cached:", len(existing))
    print("Pending:", len(pending))
    print("Device:", args.device)

    if pending:
        backend = PinyinGPTConcatBackend(args.checkpoint, device=args.device)
        started = time.perf_counter()
        with output_path.open("a", encoding="utf-8", newline="\n") as destination:
            for index, (rid, srow, grow, candidate) in enumerate(pending, 1):
                scored = backend.score_candidates(
                    context=str(grow["model_used_context"]),
                    typed_pinyin=tuple(str(value) for value in srow["pinyin_segments"]),
                    candidates=(candidate,),
                )
                if len(scored) != 1 or scored[0].text != candidate:
                    raise RuntimeError(f"Unexpected fixed-candidate scorer output: {rid}")
                value = scored[0]
                record = {
                    "schema_version": 1,
                    "experiment": "initial_short_standardized_b3_em1_exact_scoring_v1",
                    "partition": "standardized_train_val",
                    "row_id": rid,
                    "anchor_id": srow.get("anchor_id"),
                    "author": str(srow["author"]),
                    "pinyin_segments": [str(value) for value in srow["pinyin_segments"]],
                    "model_used_context_tokens": grow.get("model_used_context_tokens"),
                    "frozen_recovery_k": FROZEN_K,
                    "frozen_frequency_lambda": FROZEN_LAMBDA_FREQUENCY,
                    "selected_personal_candidate_rank": 1,
                    "scored_candidate_count": 1,
                    "scores": [
                        {
                            "candidate": value.text,
                            "personal_candidate_rank": 1,
                            "fixed_log_probability": float(value.log_probability),
                            "fixed_mean_log_probability": float(value.mean_log_probability),
                        }
                    ],
                    "candidate_surface_sha256": EXPECTED_SURFACE_SHA256,
                    "generic_predictions_sha256": EXPECTED_GENERIC_SHA256,
                    "b2_selection_sha256": EXPECTED_B2_SELECTION_SHA256,
                    "gold_used_for_candidate_selection": False,
                    "gold_used_for_scoring": False,
                    "dev3000_used": False,
                    "test_used": False,
                }
                destination.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                destination.flush()
                if index % 100 == 0 or index == len(pending):
                    elapsed = time.perf_counter() - started
                    rate = index / elapsed if elapsed else 0.0
                    print(f"B3 exact score: {index}/{len(pending)} rate={rate:.2f} rows/s", flush=True)

    final_cache = load_existing(output_path)
    if set(final_cache) != set(eligible_by_id):
        raise RuntimeError(
            f"Final exact-score cache is incomplete: {len(final_cache)}/{len(eligible_by_id)}"
        )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_short_standardized_b3_em1_exact_scoring_v1",
        "partition": "standardized Clean3 Train-Val only",
        "train_val_rows": EXPECTED_ROWS,
        "eligible_rows": len(eligible),
        "cached_rows_total": len(final_cache),
        "candidate_scores_total": sum(int(row["scored_candidate_count"]) for row in final_cache.values()),
        "fixed_candidate_scoring_used": True,
        "frozen_recovery_k": FROZEN_K,
        "frozen_frequency_lambda": FROZEN_LAMBDA_FREQUENCY,
        "frozen_pv1_lambda": FROZEN_LAMBDA_PV1,
        "candidate_surface_sha256": EXPECTED_SURFACE_SHA256,
        "generic_predictions_sha256": EXPECTED_GENERIC_SHA256,
        "b2_selection_sha256": EXPECTED_B2_SELECTION_SHA256,
        "exact_scores_sha256": sha256_file(output_path),
        "preflight_sha256": sha256_file(preflight_path),
        "gold_used_for_candidate_selection": False,
        "gold_used_for_scoring": False,
        "parameter_tuning": False,
        "dev3000_used": False,
        "test_used": False,
        "selection_population": selection["selection_population"],
    }
    write_json(args.output_root / "scoring_summary.json", summary)
    print("\n=== B3 EXACT SCORING COMPLETE ===")
    print("Eligible rows:", len(eligible))
    print("Exact scores SHA256:", summary["exact_scores_sha256"])
    print("Gold used for scoring: false")
    print("Parameter tuning: false")
    print("Dev3000 used: false")
    print("Test used: false")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B3a: preflight and resumable exact PinyinGPT scoring of the frozen Initial K=1 personal candidate surface."
    )
    parser.add_argument("--phase", choices=("preflight", "score"), required=True)
    parser.add_argument("--candidate-surface", type=Path, required=True)
    parser.add_argument("--generic-predictions", type=Path, required=True)
    parser.add_argument("--b2-selection", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows-per-author", type=int, default=3)
    parser.add_argument("--compatibility-tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    if args.rows_per_author <= 0:
        raise ValueError("--rows-per-author must be positive")
    if args.compatibility_tolerance <= 0:
        raise ValueError("--compatibility-tolerance must be positive")
    if args.phase == "preflight":
        run_preflight(args)
    else:
        run_score(args)


if __name__ == "__main__":
    main()
