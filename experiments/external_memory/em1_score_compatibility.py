from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from src.reference_backend_pinyingpt.backend import PinyinGPTConcatBackend


DEV_GENERIC_SHA256 = (
    "588aa84c6397e8cb1a13576c0d5dfecd9dd2c4305b45be351328dd83ef62007d"
)

AUTHORS = (
    "Etinjat",
    "Re_spectators",
    "breaddddd",
)

HISTORY_BUDGET = 5000
CONDITION = "Full+Short"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_key(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        str(row["row_id"]).encode("utf-8")
    ).hexdigest()


def load_dev_generic(
    path: Path,
    rows_per_author: int,
) -> list[dict[str, Any]]:
    actual_hash = sha256_file(path)
    if actual_hash != DEV_GENERIC_SHA256:
        raise RuntimeError(
            "Frozen Dev Generic cache SHA-256 mismatch:\n"
            f"expected={DEV_GENERIC_SHA256}\n"
            f"actual={actual_hash}"
        )

    by_author: dict[str, list[dict[str, Any]]] = {
        author: [] for author in AUTHORS
    }

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            row = json.loads(line)

            if row.get("pilot_partition") != "tune":
                continue

            author = str(row.get("author", ""))
            if author not in by_author:
                continue

            pinyin_segments = row.get("pinyin_segments")
            if not isinstance(pinyin_segments, list) or not pinyin_segments:
                raise RuntimeError(
                    f"Invalid pinyin_segments at line {line_number}"
                )

            candidates = row.get("top10_candidates")
            if not isinstance(candidates, list) or not candidates:
                continue

            by_author[author].append(row)

    selected: list[dict[str, Any]] = []

    for author in AUTHORS:
        rows = sorted(
            by_author[author],
            key=deterministic_key,
        )

        if len(rows) < rows_per_author:
            raise RuntimeError(
                f"{author}: only {len(rows)} eligible Dev tune rows; "
                f"need {rows_per_author}"
            )

        selected.extend(rows[:rows_per_author])

    return selected


def choose_candidates(
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = row["top10_candidates"]

    # Cover beam head, middle, and tail.
    positions = sorted(
        {
            0,
            len(candidates) // 2,
            len(candidates) - 1,
        }
    )

    return [candidates[index] for index in positions]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "EM-1 score compatibility gate: compare cached Dev Generic "
            "beam-search log-probabilities against fresh fixed-candidate "
            "PinyinGPT scores."
        )
    )
    parser.add_argument(
        "--generic-cache",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--rows-per-author",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    args = parser.parse_args()

    if args.rows_per_author <= 0:
        raise ValueError("--rows-per-author must be positive")

    rows = load_dev_generic(
        args.generic_cache,
        args.rows_per_author,
    )

    author_counts = Counter(
        str(row["author"]) for row in rows
    )

    print("EM-1 score compatibility audit")
    print(f"Condition: {CONDITION}")
    print(f"History budget semantics: H{HISTORY_BUDGET}")
    print("Partition: Dev tune")
    print(f"Authors: {', '.join(AUTHORS)}")
    print(f"Sampled rows: {len(rows)}")
    print(f"Per-author rows: {dict(author_counts)}")
    print()

    print(f"Loading checkpoint: {args.checkpoint}")
    backend = PinyinGPTConcatBackend(
        args.checkpoint,
        device=args.device,
    )

    comparisons: list[dict[str, Any]] = []

    for row_number, row in enumerate(rows, start=1):
        cached_candidates = choose_candidates(row)
        candidate_texts = [
            str(candidate["text"])
            for candidate in cached_candidates
        ]

        fresh_scores = backend.score_candidates(
            context=str(row["model_used_context"]),
            typed_pinyin=tuple(row["pinyin_segments"]),
            candidates=candidate_texts,
        )

        fresh_by_text = {
            score.text: score
            for score in fresh_scores
        }

        if set(fresh_by_text) != set(candidate_texts):
            raise RuntimeError(
                f"Candidate identity mismatch for {row['row_id']}"
            )

        for cached in cached_candidates:
            text = str(cached["text"])
            fresh = fresh_by_text[text]

            cached_score = float(
                cached["log_probability"]
            )
            fixed_score = float(
                fresh.log_probability
            )
            absolute_difference = abs(
                cached_score - fixed_score
            )

            comparisons.append(
                {
                    "row_id": str(row["row_id"]),
                    "condition_id": str(
                        row["condition_id"]
                    ),
                    "anchor_id": str(
                        row["anchor_id"]
                    ),
                    "author": str(row["author"]),
                    "pilot_partition": str(
                        row["pilot_partition"]
                    ),
                    "pinyin_segments": list(
                        row["pinyin_segments"]
                    ),
                    "candidate": text,
                    "cached_rank": int(
                        cached["rank"]
                    ),
                    "cached_log_probability": (
                        cached_score
                    ),
                    "fixed_log_probability": (
                        fixed_score
                    ),
                    "absolute_difference": (
                        absolute_difference
                    ),
                    "model_used_context_tokens": (
                        row.get(
                            "model_used_context_tokens"
                        )
                    ),
                    "checkpoint_revision": row.get(
                        "checkpoint_revision"
                    ),
                }
            )

        print(
            f"{row_number:02d}/{len(rows)} "
            f"{row['author']} "
            f"{row['row_id']} "
            f"checked={len(cached_candidates)}",
            flush=True,
        )

    differences = [
        row["absolute_difference"]
        for row in comparisons
    ]

    if not differences:
        raise RuntimeError(
            "No candidate comparisons were produced"
        )

    exact_matches = sum(
        difference == 0.0
        for difference in differences
    )

    summary = {
        "schema_version": 1,
        "experiment": (
            "em1_generic_score_compatibility"
        ),
        "status": "audit_complete",
        "condition": CONDITION,
        "history_budget": HISTORY_BUDGET,
        "history_budget_note": (
            "H5000 defines the surrounding EM-1 "
            "personal-history protocol. This score "
            "compatibility gate itself does not "
            "consult history."
        ),
        "partition": "dev_tune",
        "authors": list(AUTHORS),
        "rows_per_author": (
            args.rows_per_author
        ),
        "sampled_rows": len(rows),
        "per_author_rows": dict(
            author_counts
        ),
        "candidate_selection": (
            "Generic head, middle, and tail "
            "candidate from each sampled row"
        ),
        "candidate_comparisons": len(
            comparisons
        ),
        "exact_score_matches": (
            exact_matches
        ),
        "maximum_absolute_difference": (
            max(differences)
        ),
        "mean_absolute_difference": (
            statistics.fmean(differences)
        ),
        "median_absolute_difference": (
            statistics.median(differences)
        ),
        "generic_cache_path": str(
            args.generic_cache.resolve()
        ),
        "generic_cache_sha256": (
            sha256_file(args.generic_cache)
        ),
        "checkpoint_path": str(
            args.checkpoint.resolve()
        ),
        "test_rows_used": 0,
        "history_rows_used": 0,
        "gold_used_for_selection": False,
        "compatibility_tolerance": 1e-4,
        "compatibility_tolerance_basis": (
            "Existing PinyinGPT regression test uses "
            "assertAlmostEqual(..., places=4) when comparing "
            "batched and single generation log-probabilities."
        ),
        "compatibility_passed": max(differences) <= 1e-4,
        "historical_backend_modified": (
            False
        ),
    }

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_path = (
        args.output_root
        / "comparisons.jsonl"
    )

    with comparison_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as destination:
        for comparison in comparisons:
            destination.write(
                json.dumps(
                    comparison,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    summary_path = (
        args.output_root
        / "summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as destination:
        json.dump(
            summary,
            destination,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        destination.write("\n")

    print()
    print(
        "=== Compatibility Summary ==="
    )
    print(
        f"Sampled rows: "
        f"{summary['sampled_rows']}"
    )
    print(
        f"Candidate comparisons: "
        f"{summary['candidate_comparisons']}"
    )
    print(
        f"Exact score matches: "
        f"{summary['exact_score_matches']}"
    )
    print(
        "Max absolute difference: "
        f"{summary['maximum_absolute_difference']:.12g}"
    )
    print(
        "Mean absolute difference: "
        f"{summary['mean_absolute_difference']:.12g}"
    )
    print(
        "Median absolute difference: "
        f"{summary['median_absolute_difference']:.12g}"
    )
    tolerance = 1e-4
    passed = max(differences) <= tolerance

    print()
    print(
        f"Compatibility tolerance: {tolerance:.1e}"
    )
    print(
        f"Gate result: {'PASS' if passed else 'FAIL'}"
    )

    if not passed:
        raise RuntimeError(
            "EM-1 score compatibility gate failed"
        )


if __name__ == "__main__":
    main()

