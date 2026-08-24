"""Build/resume a K=5 Q64 exact-score cache for Initial personal candidates.

Q64 definition
--------------
For each Train-Val query, take the last 64 Python Unicode code points from the
frozen Generic artifact's ``model_used_context`` and use that short local
context for fixed-candidate PinyinGPT scoring of all personal K<=5 candidates.

Important consequence
---------------------
Historical K=1 exact scores are NOT reusable here because they were produced
with a different (long) context. Q64 therefore rescored every personal
candidate under the same 64-character context definition.

Safety / research boundaries
----------------------------
- Candidate selection comes only from the frozen candidate-surface artifact.
- Gold is never consulted for candidate selection or scoring.
- Dev3000 is not read.
- Test is not read.
- Existing long-context/full-context score caches are not read or modified.
- Output goes to a new candidate_scoring_q64_v1 directory.
- The output cache is resumable at row granularity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

from src.reference_backend_pinyingpt.backend import PinyinGPTConcatBackend


EXPECTED_SURFACE_SHA256 = (
    "205c0ba01cd0678d7a4341c503fa2e74cf126a70182ff582687025e4946764b2"
)
EXPECTED_GENERIC_SHA256 = (
    "bd0fb4dc304e0b266b90fae6fe3ac65424d2f52b23fedfa881212706ba2c2873"
)
EXPECTED_ROWS = 34416
EXPECTED_ELIGIBLE_ROWS = 30509
EXPECTED_K5_PAIRS = 123738
MAX_K = 5
CONTEXT_CHARS = 64


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


def q64_context(generic_row: dict[str, Any]) -> str:
    if "model_used_context" not in generic_row:
        raise RuntimeError(
            f"{generic_row.get('row_id')}: Generic row lacks model_used_context"
        )
    return str(generic_row["model_used_context"])[-CONTEXT_CHARS:]


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * fraction)]


def audit(
    surface_rows: list[dict[str, Any]],
    generic_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    surface = index_rows(surface_rows, label="surface")
    generic = index_rows(generic_rows, label="generic")

    if len(surface) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Unexpected surface rows: expected={EXPECTED_ROWS} actual={len(surface)}"
        )
    if len(generic) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Unexpected Generic rows: expected={EXPECTED_ROWS} actual={len(generic)}"
        )
    if set(surface) != set(generic):
        raise RuntimeError(
            "Frozen candidate surface and Generic prediction row IDs differ: "
            f"surface_only={len(set(surface) - set(generic))} "
            f"generic_only={len(set(generic) - set(surface))}"
        )

    pair_count = 0
    eligible_rows = 0
    count_by_k: dict[int, int] = {k: 0 for k in range(MAX_K + 1)}
    q64_char_lengths: list[int] = []
    original_context_token_counts: list[int] = []

    for row_id, surface_row in surface.items():
        generic_row = generic[row_id]
        candidates = candidate_texts_top5(surface_row)
        count_by_k[len(candidates)] = count_by_k.get(len(candidates), 0) + 1
        pair_count += len(candidates)

        if candidates:
            eligible_rows += 1
            q64_char_lengths.append(len(q64_context(generic_row)))
            token_count = generic_row.get("model_used_context_tokens")
            if token_count is not None:
                original_context_token_counts.append(int(token_count))

        surface_pinyin = tuple(str(x) for x in surface_row.get("pinyin_segments", []))
        generic_pinyin = tuple(str(x) for x in generic_row.get("pinyin_segments", []))
        if surface_pinyin != generic_pinyin:
            raise RuntimeError(f"Pinyin mismatch between surface/generic: {row_id}")

    if eligible_rows != EXPECTED_ELIGIBLE_ROWS:
        raise RuntimeError(
            "Eligible-row workload changed: "
            f"expected={EXPECTED_ELIGIBLE_ROWS} actual={eligible_rows}"
        )
    if pair_count != EXPECTED_K5_PAIRS:
        raise RuntimeError(
            "K5 pair workload changed: "
            f"expected={EXPECTED_K5_PAIRS} actual={pair_count}"
        )

    original_stats: dict[str, Any] = {}
    if original_context_token_counts:
        original_stats = {
            "n": len(original_context_token_counts),
            "mean": statistics.fmean(original_context_token_counts),
            "median": percentile(original_context_token_counts, 0.50),
            "p90": percentile(original_context_token_counts, 0.90),
            "p95": percentile(original_context_token_counts, 0.95),
            "max": max(original_context_token_counts),
        }

    return {
        "surface_rows": len(surface),
        "generic_rows": len(generic),
        "eligible_rows": eligible_rows,
        "required_q64_k5_pairs": pair_count,
        "candidate_count_distribution": count_by_k,
        "q64_context_unit": "last_64_python_unicode_codepoints",
        "q64_context_chars": {
            "min": min(q64_char_lengths) if q64_char_lengths else 0,
            "median": percentile(q64_char_lengths, 0.50),
            "max": max(q64_char_lengths) if q64_char_lengths else 0,
        },
        "original_model_used_context_tokens": original_stats,
    }


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
        if int(row.get("context_chars", -1)) != CONTEXT_CHARS:
            raise RuntimeError(
                f"Existing output has wrong Q64 definition for {row_id}"
            )

        expected = candidate_texts_top5(surface[row_id])
        actual = tuple(str(item["candidate"]) for item in row.get("scores", []))
        if actual != expected:
            raise RuntimeError(
                f"Existing output candidate identity/order mismatch for {row_id}:\n"
                f"expected={expected}\nactual={actual}"
            )
        completed[row_id] = row
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "score"), required=True)
    parser.add_argument("--candidate-surface", type=Path, required=True)
    parser.add_argument("--generic-predictions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--max-new-rows",
        type=int,
        default=None,
        help=(
            "Optional engineering benchmark cap. Completed rows remain resumable; "
            "rerun without this option to finish the full Train-Val cache."
        ),
    )
    args = parser.parse_args()

    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    if args.max_new_rows is not None and args.max_new_rows <= 0:
        raise ValueError("--max-new-rows must be positive")

    for path in (args.candidate_surface, args.generic_predictions):
        if not path.is_file():
            raise FileNotFoundError(path)

    surface_sha = sha256_file(args.candidate_surface)
    generic_sha = sha256_file(args.generic_predictions)

    if surface_sha != EXPECTED_SURFACE_SHA256:
        raise RuntimeError(
            "Frozen candidate surface SHA mismatch:\n"
            f"expected={EXPECTED_SURFACE_SHA256}\nactual={surface_sha}"
        )
    if generic_sha != EXPECTED_GENERIC_SHA256:
        raise RuntimeError(
            "Frozen Generic prediction SHA mismatch:\n"
            f"expected={EXPECTED_GENERIC_SHA256}\nactual={generic_sha}"
        )

    surface_rows = read_jsonl(args.candidate_surface)
    generic_rows = read_jsonl(args.generic_predictions)
    workload = audit(surface_rows, generic_rows)

    print()
    print("=== INITIAL CANDIDATE SCORING Q64 K5 AUDIT ===")
    for key, value in workload.items():
        print(f"{key}: {value}")
    print(f"candidate_surface_sha256: {surface_sha}")
    print(f"generic_predictions_sha256: {generic_sha}")
    print("Context definition: last 64 Python Unicode code points")
    print("Historical long-context K1 score reuse: false")
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
    output_path = args.output_root / "q64_k5_exact_scores.jsonl"
    summary_path = args.output_root / "summary.json"
    progress_path = args.output_root / "progress.json"

    surface = index_rows(surface_rows, label="surface")
    generic = index_rows(generic_rows, label="generic")
    completed = load_completed_output(output_path, surface)

    eligible_row_ids = [
        row_id
        for row_id in sorted(surface)
        if candidate_texts_top5(surface[row_id])
    ]
    pending_all = [row_id for row_id in eligible_row_ids if row_id not in completed]
    pending_row_ids = pending_all
    if args.max_new_rows is not None:
        pending_row_ids = pending_all[: args.max_new_rows]

    print()
    print("=== INITIAL CANDIDATE SCORING Q64 K5 SCORE ===")
    print(f"eligible rows: {len(eligible_row_ids)}")
    print(f"already completed rows: {len(completed)}")
    print(f"pending rows total: {len(pending_all)}")
    print(f"rows scheduled this run: {len(pending_row_ids)}")
    print(f"candidate pairs scheduled this run: {sum(len(candidate_texts_top5(surface[r])) for r in pending_row_ids)}")
    print(f"context chars: {CONTEXT_CHARS}")
    print(f"device: {args.device}")
    print(f"output: {output_path}")
    print()

    run_started = time.perf_counter()
    inference_seconds = 0.0
    rows_done = 0
    pairs_done = 0

    if pending_row_ids:
        print(f"Loading Frozen PinyinGPT: {args.checkpoint}")
        backend = PinyinGPTConcatBackend(args.checkpoint, device=args.device)

        mode = "a" if output_path.is_file() and output_path.stat().st_size else "w"
        with output_path.open(mode, encoding="utf-8", newline="\n") as destination:
            for row_number, row_id in enumerate(pending_row_ids, start=1):
                surface_row = surface[row_id]
                generic_row = generic[row_id]
                candidates = candidate_texts_top5(surface_row)
                context = q64_context(generic_row)

                call_started = time.perf_counter()
                scored = backend.score_candidates(
                    context=context,
                    typed_pinyin=tuple(str(x) for x in surface_row["pinyin_segments"]),
                    candidates=candidates,
                )
                call_elapsed = time.perf_counter() - call_started
                inference_seconds += call_elapsed

                score_by_text = {str(value.text): value for value in scored}
                if set(score_by_text) != set(candidates):
                    raise RuntimeError(
                        f"Scorer returned different candidate identities for {row_id}:\n"
                        f"requested={candidates}\nreturned={sorted(score_by_text)}"
                    )

                scores: list[dict[str, Any]] = []
                for personal_rank, candidate in enumerate(candidates, start=1):
                    value = score_by_text[candidate]
                    scores.append(
                        {
                            "candidate": candidate,
                            "personal_candidate_rank": personal_rank,
                            "fixed_log_probability": float(value.log_probability),
                            "fixed_mean_log_probability": float(
                                value.mean_log_probability
                            ),
                            "score_source": "q64_exact_inference",
                        }
                    )

                result = {
                    "schema_version": 1,
                    "experiment": "initial_candidate_scoring_q64_k5_exact_v1",
                    "partition": "standardized_train_val",
                    "row_id": row_id,
                    "condition_id": str(surface_row.get("condition_id", "")),
                    "anchor_id": str(surface_row.get("anchor_id", "")),
                    "author": str(surface_row["author"]),
                    "pinyin_segments": list(surface_row["pinyin_segments"]),
                    "max_recovery_k": MAX_K,
                    "personal_candidate_count": len(candidates),
                    "context_chars": CONTEXT_CHARS,
                    "q64_context_char_count": len(context),
                    "original_model_used_context_tokens": generic_row.get(
                        "model_used_context_tokens"
                    ),
                    "row_inference_seconds": call_elapsed,
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
                pairs_done += len(candidates)

                if (
                    row_number % args.progress_every == 0
                    or row_number == len(pending_row_ids)
                ):
                    destination.flush()
                    elapsed = time.perf_counter() - run_started
                    rows_per_second = rows_done / elapsed if elapsed else 0.0
                    mean_inference_ms = (
                        1000.0 * inference_seconds / rows_done if rows_done else 0.0
                    )
                    print(
                        f"Q64 K5: {row_number}/{len(pending_row_ids)} scheduled rows; "
                        f"pairs={pairs_done}; rate={rows_per_second:.2f} rows/s; "
                        f"mean_score_call={mean_inference_ms:.2f} ms/row",
                        flush=True,
                    )
    else:
        print("Nothing to score in this run.")

    final_rows = load_completed_output(output_path, surface)
    remaining_rows = len(eligible_row_ids) - len(final_rows)
    progress = {
        "schema_version": 1,
        "experiment": "initial_candidate_scoring_q64_k5_exact_v1",
        "status": "complete" if remaining_rows == 0 else "partial",
        "eligible_rows": len(eligible_row_ids),
        "completed_rows": len(final_rows),
        "remaining_rows": remaining_rows,
        "context_chars": CONTEXT_CHARS,
        "last_run": {
            "rows_scored": rows_done,
            "candidate_pairs_scored": pairs_done,
            "wall_seconds": time.perf_counter() - run_started,
            "score_call_seconds": inference_seconds,
            "mean_score_call_ms_per_row": (
                1000.0 * inference_seconds / rows_done if rows_done else None
            ),
        },
        "gold_used_for_candidate_selection": False,
        "gold_used_for_scoring": False,
        "dev3000_used": False,
        "test_used": False,
    }
    write_json(progress_path, progress)

    if remaining_rows:
        print()
        print("=== Q64 K5 PARTIAL / RESUMABLE ===")
        print(f"completed rows: {len(final_rows)}")
        print(f"remaining rows: {remaining_rows}")
        print(f"progress: {progress_path}")
        print("Rerun the same score command without --max-new-rows to finish.")
        return

    total_pairs = sum(len(row["scores"]) for row in final_rows.values())
    if total_pairs != EXPECTED_K5_PAIRS:
        raise RuntimeError(
            f"Final Q64 K5 pair count mismatch: "
            f"expected={EXPECTED_K5_PAIRS} actual={total_pairs}"
        )

    row_latencies_ms = [
        1000.0 * float(row["row_inference_seconds"])
        for row in final_rows.values()
        if row.get("row_inference_seconds") is not None
    ]
    row_latencies_ms.sort()

    def latency_percentile(fraction: float) -> float | None:
        if not row_latencies_ms:
            return None
        return row_latencies_ms[int((len(row_latencies_ms) - 1) * fraction)]

    summary = {
        "schema_version": 1,
        "experiment": "initial_candidate_scoring_q64_k5_exact_v1",
        "status": "complete",
        "partition": "standardized_train_val",
        "context_definition": {
            "chars": CONTEXT_CHARS,
            "unit": "last_python_unicode_codepoints",
            "source": "frozen_generic_model_used_context",
        },
        "max_recovery_k": MAX_K,
        "surface_rows": EXPECTED_ROWS,
        "eligible_rows": len(eligible_row_ids),
        "candidate_scores_total": total_pairs,
        "historical_long_context_score_reuse": False,
        "score_fields": [
            "fixed_log_probability",
            "fixed_mean_log_probability",
        ],
        "primary_candidate_scoring_field_for_next_audit": (
            "fixed_mean_log_probability"
        ),
        "latency_from_row_score_calls_ms": {
            "n": len(row_latencies_ms),
            "mean": statistics.fmean(row_latencies_ms) if row_latencies_ms else None,
            "p50": latency_percentile(0.50),
            "p95": latency_percentile(0.95),
            "p99": latency_percentile(0.99),
        },
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
            "checkpoint": str(args.checkpoint.resolve()),
            "q64_k5_exact_scores": str(output_path.resolve()),
            "q64_k5_exact_scores_sha256": sha256_file(output_path),
        },
        "frozen_workload": workload,
    }
    write_json(summary_path, summary)

    print()
    print("=== INITIAL CANDIDATE SCORING Q64 K5 COMPLETE ===")
    print(f"eligible rows: {summary['eligible_rows']}")
    print(f"candidate scores total: {total_pairs}")
    print(f"cache SHA256: {summary['provenance']['q64_k5_exact_scores_sha256']}")
    print(f"latency: {summary['latency_from_row_score_calls_ms']}")
    print("Historical long-context K1 score reuse: false")
    print("Gold used for scoring: false")
    print("Parameter tuning: false")
    print("Dev3000 used: false")
    print("Test used: false")
    print(f"outputs: {args.output_root}")


if __name__ == "__main__":
    main()
