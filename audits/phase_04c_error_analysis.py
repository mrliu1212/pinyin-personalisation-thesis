"""Prepare deterministic human-review samples from Phase 4C transparency data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("results/experiments/phase_04c/evaluation.json")
DEFAULT_OUTPUT = Path("results/audits/phase_04c/error_analysis_samples.jsonl")
CONDITIONS = ("correct_user", "wrong_user")
CHANGES = ("improved", "harmed", "unchanged")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    required = (
        "candidate",
        "base_score",
        "global_evidence",
        "pinyin_evidence",
        "context_evidence",
        "personal_score",
        "final_rank",
    )
    missing = [field for field in required if field not in candidate]
    if missing:
        raise ValueError(f"personalised candidate is missing fields: {missing}")
    return {field: candidate[field] for field in required}


def _base_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    required = ("candidate", "base_rank", "base_score")
    missing = [field for field in required if field not in candidate]
    if missing:
        raise ValueError(f"base candidate is missing fields: {missing}")
    return {field: candidate[field] for field in required}


def _validate_example(example: dict[str, Any]) -> None:
    required = (
        "interaction_id",
        "work_id",
        "context",
        "pinyin",
        "target",
        "base_rank",
        "personalised_rank",
        "change",
        "base_candidates",
        "personalised_candidates",
    )
    missing = [field for field in required if field not in example]
    if missing:
        raise ValueError(f"transparency example is missing fields: {missing}")
    if example["change"] not in CHANGES:
        raise ValueError(f"unsupported rank-change label: {example['change']!r}")


def _rank_delta(example: dict[str, Any]) -> int | None:
    base = example["base_rank"]
    personal = example["personalised_rank"]
    if base is None or personal is None:
        return None
    return int(base) - int(personal)


def _review_row(condition: str, example: dict[str, Any]) -> dict[str, Any]:
    _validate_example(example)
    return {
        "schema_version": 1,
        "sample_type": f"{condition}_{example['change']}",
        "condition": condition,
        "change": example["change"],
        "interaction_id": example["interaction_id"],
        "work_id": example["work_id"],
        "context": example["context"],
        "pinyin": example["pinyin"],
        "target": example["target"],
        "base_rank": example["base_rank"],
        "personalised_rank": example["personalised_rank"],
        "rank_delta_base_minus_personalised": _rank_delta(example),
        "base_candidates": [_base_candidate(item) for item in example["base_candidates"]],
        "personalised_candidates": [
            _candidate(item) for item in example["personalised_candidates"]
        ],
        "correct_user_rank": (
            example["personalised_rank"] if condition == "correct_user" else None
        ),
        "wrong_user_rank": (
            example["personalised_rank"] if condition == "wrong_user" else None
        ),
        "correct_user_candidates": (
            [_candidate(item) for item in example["personalised_candidates"]]
            if condition == "correct_user"
            else []
        ),
        "wrong_user_candidates": (
            [_candidate(item) for item in example["personalised_candidates"]]
            if condition == "wrong_user"
            else []
        ),
        "human_category": "",
        "notes": "",
    }


def _comparison_row(
    correct: dict[str, Any], wrong: dict[str, Any]
) -> dict[str, Any]:
    _validate_example(correct)
    _validate_example(wrong)
    identity_fields = ("interaction_id", "work_id", "context", "pinyin", "target", "base_rank")
    if any(correct[field] != wrong[field] for field in identity_fields):
        raise ValueError("correct-user and wrong-user examples do not describe the same interaction")
    correct_candidates = [_candidate(item) for item in correct["personalised_candidates"]]
    wrong_candidates = [_candidate(item) for item in wrong["personalised_candidates"]]
    return {
        "schema_version": 1,
        "sample_type": "correct_user_vs_wrong_user",
        "condition": "comparison",
        "change": "comparison",
        "interaction_id": correct["interaction_id"],
        "work_id": correct["work_id"],
        "context": correct["context"],
        "pinyin": correct["pinyin"],
        "target": correct["target"],
        "base_rank": correct["base_rank"],
        "personalised_rank": None,
        "rank_delta_base_minus_personalised": None,
        "base_candidates": [_base_candidate(item) for item in correct["base_candidates"]],
        "personalised_candidates": [],
        "correct_user_rank": correct["personalised_rank"],
        "wrong_user_rank": wrong["personalised_rank"],
        "correct_user_candidates": correct_candidates,
        "wrong_user_candidates": wrong_candidates,
        "human_category": "",
        "notes": "",
    }


def _sort_examples(examples: Iterable[dict[str, Any]], change: str) -> list[dict[str, Any]]:
    def key(example: dict[str, Any]) -> tuple[int, str]:
        delta = _rank_delta(example)
        magnitude = abs(delta) if delta is not None else 0
        return (-magnitude, example["interaction_id"])

    return sorted((item for item in examples if item["change"] == change), key=key)


def extract_samples(evaluation: dict[str, Any], limit: int = 20) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit must be at least one")
    transparency = evaluation.get("transparency_examples")
    if not isinstance(transparency, dict):
        raise ValueError("evaluation is missing transparency_examples")

    rows: list[dict[str, Any]] = []
    by_condition: dict[str, dict[str, dict[str, Any]]] = {}
    for condition in CONDITIONS:
        examples = transparency.get(condition)
        if not isinstance(examples, list):
            raise ValueError(f"evaluation is missing {condition} transparency examples")
        by_condition[condition] = {item["interaction_id"]: item for item in examples}
        for change in CHANGES:
            rows.extend(
                _review_row(condition, example)
                for example in _sort_examples(examples, change)[:limit]
            )

    correct = by_condition["correct_user"]
    wrong = by_condition["wrong_user"]
    for interaction_id in sorted(set(correct) & set(wrong))[:limit]:
        correct_example = correct[interaction_id]
        wrong_example = wrong[interaction_id]
        if (
            correct_example["personalised_rank"] != wrong_example["personalised_rank"]
            or correct_example["personalised_candidates"]
            != wrong_example["personalised_candidates"]
        ):
            rows.append(_comparison_row(correct_example, wrong_example))
    return rows


def prepare(input_path: Path, output_path: Path, limit: int = 20) -> list[dict[str, Any]]:
    source_bytes = input_path.read_bytes()
    source_checksum = sha256_bytes(source_bytes)
    evaluation = json.loads(source_bytes.decode("utf-8"))
    rows = extract_samples(evaluation, limit=limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    if sha256_bytes(input_path.read_bytes()) != source_checksum:
        raise RuntimeError("evaluation input changed during extraction")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="maximum stored examples per condition/category and comparison section",
    )
    args = parser.parse_args()
    before = sha256_bytes(args.input.read_bytes())
    rows = prepare(args.input, args.output, limit=args.limit)
    print(f"Evaluation SHA-256: {before}")
    print(f"Review rows: {len(rows)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
