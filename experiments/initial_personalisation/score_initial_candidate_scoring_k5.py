"""Build/resume a K=5 exact-score cache for Initial personal candidates.

Purpose
-------
This is a Train-Val development utility for Candidate Scoring V1.
It reuses the historical K=1 exact-score cache by (row_id, candidate)
and calls the frozen PinyinGPT fixed-candidate scorer only for missing
K=2..K=5 candidate pairs.

Safety / research boundaries
----------------------------
- Candidate selection comes only from the frozen candidate-surface artifact.
- Gold is never consulted for candidate selection or scoring.
- Dev3000 is not read.
- Test is not read.
- Historical K=1 artifacts are read-only and never overwritten.
- Output goes to a new candidate_scoring_v1 directory.
- The output cache is resumable at row granularity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

from src.reference_backend_pinyingpt.backend import PinyinGPTConcatBackend


EXPECTED_SURFACE_SHA256 = (
    "205c0ba01cd0678d7a4341c503fa2e74cf126a70182ff582687025e4946764b2"
)
EXPECTED_OLD_K1_SHA256 = (
    "f74a50971d4ca2cf9d34998ae9b8946645141e2deec88ee7c633212b3e374400"
)
EXPECTED_ROWS = 34416
EXPECTED_K5_PAIRS = 123738
EXPECTED_OLD_K1_PAIRS = 30509
EXPECTED_NEW_PAIRS = 93229
MAX_K = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def candidate_texts_top5(row: dict[str, Any]) -> tuple[str, ...]:
    values = row.get("personal_candidate_texts_top5")
    if isinstance(values, list):
        texts = [str(value) for value in values[:MAX_K]]
    else:
        values = row.get("personal_candidates_top5")
        if not isinstance(values, list):
            raise RuntimeError(
                f"{row.get('row_id')}: no usable personal top-5 field"
            )
        texts = []
        for item in values[:MAX_K]:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("candidate") or item.get("target")
                if text is None:
                    raise RuntimeError(
                        f"{row.get('row_id')}: candidate dict has no text field: {item}"
                    )
                texts.append(str(text))
            else:
                raise RuntimeError(
                    f"{row.get('row_id')}: unsupported candidate value: {item!r}"
                )

    # Preserve surface order while removing accidental duplicates.
    return tuple(dict.fromkeys(texts))


def index_rows(
    rows: Iterable[dict[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row["row_id"])
        if row_id in output:
            raise RuntimeError(f"Duplicate {label} row_id: {row_id}")
        output[row_id] = row
    return output


def score_pairs_from_old_cache(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        row_id = str(row["row_id"])
        for score in row.get("scores", []):
            candidate = str(score["candidate"])
            key = (row_id, candidate)
            if key in output:
                raise RuntimeError(f"Duplicate old K1 score pair: {key}")
            output[key] = {
                "candidate": candidate,
                "fixed_log_probability": float(score["fixed_log_probability"]),
                "fixed_mean_log_probability": float(
                    score["fixed_mean_log_probability"]
                ),
                "score_source": "reused_historical_k1_exact_cache",
            }
    return output


def load_completed_output(
    path: Path,
    surface: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    completed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(read_jsonl(path), start=1):
        row_id = str(row["row_id"])
        if row_id in completed:
            raise RuntimeError(
                f"Duplicate output row_id at line {line_number}: {row_id}"
            )
        if row_id not in surface:
            raise RuntimeError(f"Stale output row outside frozen surface: {row_id}")

        expected = candidate_texts_top5(surface[row_id])
        actual = tuple(str(item["candidate"]) for item in row.get("scores", []))
        if actual != expected:
            raise RuntimeError(
                f"Existing output candidate identity/order mismatch for {row_id}:\n"
                f"expected={expected}\nactual={actual}"
            )
        completed[row_id] = row
    return completed


def audit(
    surface_rows: list[dict[str, Any]],
    old_rows: list[dict[str, Any]],
    generic_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    surface = index_rows(surface_rows, label="surface")
    generic = index_rows(generic_rows, label="generic")

    if len(surface) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Unexpected surface rows: expected={EXPECTED_ROWS} actual={len(surface)}"
        )
    if set(surface) != set(generic):
        raise RuntimeError(
            "Frozen candidate surface and Generic prediction row IDs differ: "
            f"surface_only={len(set(surface) - set(generic))} "
            f"generic_only={len(set(generic) - set(surface))}"
        )

    old_pairs = score_pairs_from_old_cache(old_rows)
    required_pairs: set[tuple[str, str]] = set()
    rows_with_personal = 0
    count_by_k: dict[int, int] = {k: 0 for k in range(MAX_K + 1)}

    for row_id, row in surface.items():
        candidates = candidate_texts_top5(row)
        count_by_k[len(candidates)] = count_by_k.get(len(candidates), 0) + 1
        if candidates:
            rows_with_personal += 1
        for candidate in candidates:
            required_pairs.add((row_id, candidate))

    reusable_pairs = required_pairs & set(old_pairs)
    missing_pairs = required_pairs - set(old_pairs)
    rows_needing_new_scores = {row_id for row_id, _ in missing_pairs}

    summary = {
        "surface_rows": len(surface),
        "generic_rows": len(generic),
        "rows_with_personal_candidates": rows_with_personal,
        "required_k5_pairs": len(required_pairs),
        "old_k1_pairs": len(old_pairs),
        "reusable_old_k1_pairs": len(reusable_pairs),
        "new_pairs_required": len(missing_pairs),
        "rows_needing_new_scores": len(rows_needing_new_scores),
        "candidate_count_distribution": count_by_k,
    }

    # These invariants intentionally freeze the workload discovered before GPU work.
    expected = {
        "required_k5_pairs": EXPECTED_K5_PAIRS,
        "old_k1_pairs": EXPECTED_OLD_K1_PAIRS,
        "reusable_old_k1_pairs": EXPECTED_OLD_K1_PAIRS,
        "new_pairs_required": EXPECTED_NEW_PAIRS,
    }
    for key, value in expected.items():
        if summary[key] != value:
            raise RuntimeError(
                f"Candidate-scoring workload changed for {key}: "
                f"expected={value} actual={summary[key]}"
            )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "score"), required=True)
    parser.add_argument("--candidate-surface", type=Path, required=True)
    parser.add_argument("--generic-predictions", type=Path, required=True)
    parser.add_argument("--old-k1-scores", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")

    for path in (
        args.candidate_surface,
        args.generic_predictions,
        args.old_k1_scores,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    surface_sha = sha256_file(args.candidate_surface)
    old_k1_sha = sha256_file(args.old_k1_scores)
    generic_sha = sha256_file(args.generic_predictions)

    if surface_sha != EXPECTED_SURFACE_SHA256:
        raise RuntimeError(
            "Frozen candidate surface SHA mismatch:\n"
            f"expected={EXPECTED_SURFACE_SHA256}\nactual={surface_sha}"
        )
    if old_k1_sha != EXPECTED_OLD_K1_SHA256:
        raise RuntimeError(
            "Historical K1 exact-score SHA mismatch:\n"
            f"expected={EXPECTED_OLD_K1_SHA256}\nactual={old_k1_sha}"
        )

    surface_rows = read_jsonl(args.candidate_surface)
    generic_rows = read_jsonl(args.generic_predictions)
    old_rows = read_jsonl(args.old_k1_scores)

    workload = audit(surface_rows, old_rows, generic_rows)

    print()
    print("=== INITIAL CANDIDATE SCORING K5 AUDIT ===")
    for key, value in workload.items():
        print(f"{key}: {value}")
    print(f"candidate_surface_sha256: {surface_sha}")
    print(f"generic_predictions_sha256: {generic_sha}")
    print(f"old_k1_scores_sha256: {old_k1_sha}")
    print("Gold used for candidate selection: false")
    print("Gold used for scoring: false")
    print("Dev3000 used: false")
    print("Test used: false")

    if args.phase == "audit":
        print("GPU inference: false")
        return

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for --phase score")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / "k5_exact_scores.jsonl"
    summary_path = args.output_root / "summary.json"

    surface = index_rows(surface_rows, label="surface")
    generic = index_rows(generic_rows, label="generic")
    old_pairs = score_pairs_from_old_cache(old_rows)
    completed = load_completed_output(output_path, surface)

    eligible_row_ids = [
        row_id
        for row_id in sorted(surface)
        if candidate_texts_top5(surface[row_id])
    ]

    pending_row_ids = [row_id for row_id in eligible_row_ids if row_id not in completed]

    missing_pairs_before = 0
    reused_pairs_before = 0
    for row_id in pending_row_ids:
        for candidate in candidate_texts_top5(surface[row_id]):
            if (row_id, candidate) in old_pairs:
                reused_pairs_before += 1
            else:
                missing_pairs_before += 1

    print()
    print("=== INITIAL CANDIDATE SCORING K5 SCORE ===")
    print(f"eligible rows: {len(eligible_row_ids)}")
    print(f"already completed rows: {len(completed)}")
    print(f"pending rows: {len(pending_row_ids)}")
    print(f"historical K1 pairs reused in pending rows: {reused_pairs_before}")
    print(f"new pairs to score in pending rows: {missing_pairs_before}")
    print(f"device: {args.device}")
    print(f"output: {output_path}")
    print()

    if not pending_row_ids:
        print("Nothing to score; cache is already complete.")
    else:
        print(f"Loading Frozen PinyinGPT: {args.checkpoint}")
        backend = PinyinGPTConcatBackend(args.checkpoint, device=args.device)
        started = time.perf_counter()
        new_pairs_done = 0
        rows_done = 0

        mode = "a" if output_path.is_file() and output_path.stat().st_size else "w"
        with output_path.open(mode, encoding="utf-8", newline="\n") as destination:
            for row_number, row_id in enumerate(pending_row_ids, start=1):
                surface_row = surface[row_id]
                generic_row = generic[row_id]
                candidates = candidate_texts_top5(surface_row)

                if tuple(str(x) for x in surface_row.get("pinyin_segments", [])) != tuple(
                    str(x) for x in generic_row.get("pinyin_segments", [])
                ):
                    raise RuntimeError(f"Pinyin mismatch between surface/generic: {row_id}")

                missing = [
                    candidate
                    for candidate in candidates
                    if (row_id, candidate) not in old_pairs
                ]

                fresh_by_text: dict[str, Any] = {}
                if missing:
                    scored = backend.score_candidates(
                        context=str(generic_row["model_used_context"]),
                        typed_pinyin=tuple(str(x) for x in surface_row["pinyin_segments"]),
                        candidates=tuple(missing),
                    )
                    fresh_by_text = {str(value.text): value for value in scored}
                    if set(fresh_by_text) != set(missing):
                        raise RuntimeError(
                            f"Scorer returned different candidate identities for {row_id}:\n"
                            f"requested={missing}\nreturned={sorted(fresh_by_text)}"
                        )
                    new_pairs_done += len(missing)

                scores: list[dict[str, Any]] = []
                for personal_rank, candidate in enumerate(candidates, start=1):
                    key = (row_id, candidate)
                    if key in old_pairs:
                        value = old_pairs[key]
                        score_record = {
                            "candidate": candidate,
                            "personal_candidate_rank": personal_rank,
                            "fixed_log_probability": value["fixed_log_probability"],
                            "fixed_mean_log_probability": value[
                                "fixed_mean_log_probability"
                            ],
                            "score_source": value["score_source"],
                        }
                    else:
                        value = fresh_by_text[candidate]
                        score_record = {
                            "candidate": candidate,
                            "personal_candidate_rank": personal_rank,
                            "fixed_log_probability": float(value.log_probability),
                            "fixed_mean_log_probability": float(
                                value.mean_log_probability
                            ),
                            "score_source": "new_k5_exact_inference",
                        }
                    scores.append(score_record)

                result = {
                    "schema_version": 1,
                    "experiment": "initial_candidate_scoring_k5_exact_v1",
                    "partition": "standardized_train_val",
                    "row_id": row_id,
                    "condition_id": str(surface_row.get("condition_id", "")),
                    "anchor_id": str(surface_row.get("anchor_id", "")),
                    "author": str(surface_row["author"]),
                    "pinyin_segments": list(surface_row["pinyin_segments"]),
                    "max_recovery_k": MAX_K,
                    "personal_candidate_count": len(candidates),
                    "model_used_context_tokens": generic_row.get(
                        "model_used_context_tokens"
                    ),
                    "scores": scores,
                    "gold_used_for_candidate_selection": False,
                    "gold_used_for_scoring": False,
                    "dev3000_used": False,
                    "test_used": False,
                }

                destination.write(
                    json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"
                )
                rows_done += 1

                if (
                    row_number % args.progress_every == 0
                    or row_number == len(pending_row_ids)
                ):
                    destination.flush()
                    elapsed = time.perf_counter() - started
                    rate = rows_done / elapsed if elapsed else 0.0
                    print(
                        f"K5 exact: {row_number}/{len(pending_row_ids)} pending rows; "
                        f"new_pairs={new_pairs_done}; rate={rate:.2f} rows/s",
                        flush=True,
                    )

    final_rows = load_completed_output(output_path, surface)
    if set(final_rows) != set(eligible_row_ids):
        missing_rows = set(eligible_row_ids) - set(final_rows)
        extra_rows = set(final_rows) - set(eligible_row_ids)
        raise RuntimeError(
            "Final K5 score cache is incomplete or stale: "
            f"missing_rows={len(missing_rows)} extra_rows={len(extra_rows)}"
        )

    total_pairs = sum(len(row["scores"]) for row in final_rows.values())
    if total_pairs != EXPECTED_K5_PAIRS:
        raise RuntimeError(
            f"Final K5 score-pair count mismatch: "
            f"expected={EXPECTED_K5_PAIRS} actual={total_pairs}"
        )

    reused_total = sum(
        1
        for row in final_rows.values()
        for score in row["scores"]
        if score["score_source"] == "reused_historical_k1_exact_cache"
    )
    new_total = total_pairs - reused_total

    summary = {
        "schema_version": 1,
        "experiment": "initial_candidate_scoring_k5_exact_v1",
        "status": "complete",
        "partition": "standardized_train_val",
        "max_recovery_k": MAX_K,
        "surface_rows": EXPECTED_ROWS,
        "eligible_rows": len(eligible_row_ids),
        "candidate_scores_total": total_pairs,
        "reused_historical_k1_pairs": reused_total,
        "new_k5_exact_pairs": new_total,
        "score_fields": [
            "fixed_log_probability",
            "fixed_mean_log_probability",
        ],
        "primary_candidate_scoring_field_for_next_audit": (
            "fixed_mean_log_probability"
        ),
        "gold_used_for_candidate_selection": False,
        "gold_used_for_scoring": False,
        "parameter_tuning": False,
        "dev3000_used": False,
        "test_used": False,
        "provenance": {
            "candidate_surface": str(args.candidate_surface.resolve()),
            "candidate_surface_sha256": surface_sha,
            "generic_predictions": str(args.generic_predictions.resolve()),
            "generic_predictions_sha256": generic_sha,
            "old_k1_scores": str(args.old_k1_scores.resolve()),
            "old_k1_scores_sha256": old_k1_sha,
            "checkpoint": str(args.checkpoint.resolve()),
            "k5_exact_scores": str(output_path.resolve()),
            "k5_exact_scores_sha256": sha256_file(output_path),
        },
        "frozen_workload": workload,
    }
    write_json(summary_path, summary)

    print()
    print("=== INITIAL CANDIDATE SCORING K5 COMPLETE ===")
    print(f"eligible rows: {summary['eligible_rows']}")
    print(f"candidate scores total: {total_pairs}")
    print(f"reused historical K1 pairs: {reused_total}")
    print(f"new K5 exact pairs: {new_total}")
    print(f"cache SHA256: {summary['provenance']['k5_exact_scores_sha256']}")
    print("Gold used for scoring: false")
    print("Parameter tuning: false")
    print("Dev3000 used: false")
    print("Test used: false")
    print(f"outputs: {args.output_root}")


if __name__ == "__main__":
    main()
