"""Explain Phase 4B versus Phase 4B.5 interaction-set differences read-only."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from interactions.linguistic import JiebaSegmenter, TargetPolicy, exclusion_reason


DEFAULT_BASELINE = Path("data/processed/interactions/zhu_ziqing/interactions.jsonl")
DEFAULT_NORMALIZED = Path(
    "data/processed/interactions/zhu_ziqing_t2s/interactions.jsonl"
)
DEFAULT_SOURCE_MANIFEST = Path("data/processed/authors/zhu_ziqing/manifest.json")
DEFAULT_NORMALIZED_MANIFEST = Path(
    "data/processed/normalized/authors/zhu_ziqing_t2s/manifest.json"
)
DEFAULT_OUTPUT = Path(
    "results/audits/phase_04b/script_normalization_interaction_delta.json"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def interaction_key(record: dict[str, Any]) -> tuple[str, int, int]:
    return (
        record["work_id"],
        int(record["source_start_offset"]),
        int(record["source_end_offset"]),
    )


def token_inventory(
    text: str, policy: TargetPolicy, segmenter: JiebaSegmenter
) -> list[dict[str, Any]]:
    return [
        {
            "start": token.start,
            "end": token.end,
            "text": token.text,
            "eligible": exclusion_reason(token.text, policy) is None,
            "exclusion_reason": exclusion_reason(token.text, policy),
        }
        for token in segmenter.segment(text)
    ]


def overlapping_tokens(
    tokens: list[dict[str, Any]], start: int, end: int
) -> list[dict[str, Any]]:
    return [
        token
        for token in tokens
        if int(token["start"]) < end and int(token["end"]) > start
    ]


def classify_delta(
    *,
    record: dict[str, Any],
    raw_text: str,
    normalized_text: str,
    other_tokens: list[dict[str, Any]],
    delta_type: str,
) -> dict[str, Any]:
    start = int(record["source_start_offset"])
    end = int(record["source_end_offset"])
    exact_other = next(
        (
            token
            for token in other_tokens
            if token["start"] == start and token["end"] == end
        ),
        None,
    )
    overlaps = overlapping_tokens(other_tokens, start, end)
    if exact_other is None:
        primary_reason = "changed_jieba_segmentation"
    elif not exact_other["eligible"]:
        primary_reason = "filtering_differences"
    else:
        primary_reason = "other"

    raw_target = raw_text[start:end]
    normalized_target = normalized_text[start:end]
    categories = [primary_reason]
    if raw_target != normalized_target:
        categories.append("chinese_character_normalization")
    if any(not token["eligible"] for token in overlaps):
        if "filtering_differences" not in categories:
            categories.append("filtering_differences")
    return {
        "delta_type": delta_type,
        "interaction_id": record["interaction_id"],
        "work_id": record["work_id"],
        "work_title": record["work_title"],
        "source_start_offset": start,
        "source_end_offset": end,
        "raw_target": raw_target,
        "normalized_target": normalized_target,
        "interaction_target": record["target_candidate"],
        "primary_reason": primary_reason,
        "categories": categories,
        "other_representation_exact_span_token": exact_other,
        "other_representation_overlapping_tokens": overlaps,
    }


def summarize_delta(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_work = Counter(record["work_id"] for record in records)
    primary = Counter(record["primary_reason"] for record in records)
    categories = Counter(
        category for record in records for category in record["categories"]
    )
    examples_by_reason: dict[str, list[dict[str, Any]]] = {}
    for reason in (
        "changed_jieba_segmentation",
        "filtering_differences",
        "other",
    ):
        examples_by_reason[reason] = [
            record for record in records if record["primary_reason"] == reason
        ][:10]
    return {
        "count": len(records),
        "by_work": dict(sorted(by_work.items())),
        "primary_reason_counts": dict(sorted(primary.items())),
        "category_counts_nonexclusive": dict(sorted(categories.items())),
        "examples": records[:20],
        "examples_by_primary_reason": examples_by_reason,
        "interactions": records,
    }


def analyze(
    baseline_path: Path,
    normalized_path: Path,
    source_manifest_path: Path,
    normalized_manifest_path: Path,
) -> dict[str, Any]:
    baseline = read_jsonl(baseline_path)
    normalized = read_jsonl(normalized_path)
    baseline_by_key = {interaction_key(record): record for record in baseline}
    normalized_by_key = {interaction_key(record): record for record in normalized}
    baseline_keys = set(baseline_by_key)
    normalized_keys = set(normalized_by_key)

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    normalized_manifest = json.loads(
        normalized_manifest_path.read_text(encoding="utf-8")
    )
    source_work_metadata = {
        work["work_id"]: work for work in source_manifest["works"] if work["included"]
    }
    normalized_work_metadata = {
        work["work_id"]: work
        for work in normalized_manifest["works"]
        if work["included"]
    }
    source_texts = {
        work_id: (source_manifest_path.parent / work["processed_file"]).read_text(
            encoding="utf-8"
        )
        for work_id, work in source_work_metadata.items()
    }
    normalized_texts = {
        work_id: (
            normalized_manifest_path.parent / work["processed_file"]
        ).read_text(encoding="utf-8")
        for work_id, work in normalized_work_metadata.items()
    }
    policy = TargetPolicy()
    segmenter = JiebaSegmenter()
    source_tokens = {
        work_id: token_inventory(text, policy, segmenter)
        for work_id, text in source_texts.items()
    }
    normalized_tokens = {
        work_id: token_inventory(text, policy, segmenter)
        for work_id, text in normalized_texts.items()
    }

    added = [
        classify_delta(
            record=normalized_by_key[key],
            raw_text=source_texts[key[0]],
            normalized_text=normalized_texts[key[0]],
            other_tokens=source_tokens[key[0]],
            delta_type="added_after_normalization",
        )
        for key in sorted(normalized_keys - baseline_keys)
    ]
    removed = [
        classify_delta(
            record=baseline_by_key[key],
            raw_text=source_texts[key[0]],
            normalized_text=normalized_texts[key[0]],
            other_tokens=normalized_tokens[key[0]],
            delta_type="removed_after_normalization",
        )
        for key in sorted(baseline_keys - normalized_keys)
    ]

    result = {
        "schema_version": 1,
        "comparison_key": [
            "work_id",
            "source_start_offset",
            "source_end_offset",
        ],
        "inputs": {
            "baseline_interactions": str(baseline_path),
            "baseline_sha256": sha256_file(baseline_path),
            "normalized_interactions": str(normalized_path),
            "normalized_sha256": sha256_file(normalized_path),
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "normalized_manifest": str(normalized_manifest_path),
            "normalized_manifest_sha256": sha256_file(normalized_manifest_path),
        },
        "counts": {
            "baseline_interactions": len(baseline),
            "normalized_interactions": len(normalized),
            "retained_same_span_interactions": len(baseline_keys & normalized_keys),
            "added_interactions": len(added),
            "removed_interactions": len(removed),
            "net_change": len(normalized) - len(baseline),
            "delta_identity_check": len(added) - len(removed),
        },
        "reason_definitions": {
            "changed_jieba_segmentation": (
                "No token with the same start/end span exists in the other "
                "representation's Jieba token stream."
            ),
            "chinese_character_normalization": (
                "Non-exclusive attribute: source and normalized substrings at "
                "the interaction span differ."
            ),
            "filtering_differences": (
                "Non-exclusive category: at least one same-span or overlapping "
                "token in the other representation is ineligible under the "
                "unchanged 2–4 all-Chinese target policy. It is a consequence "
                "of the changed segmentation, not a changed filter rule."
            ),
            "other": (
                "A same-span eligible token exists in the other representation "
                "but no interaction exists; this could include conversion failure "
                "or an unexpected pipeline discrepancy."
            ),
        },
        "added": summarize_delta(added),
        "removed": summarize_delta(removed),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST
    )
    parser.add_argument(
        "--normalized-manifest", type=Path, default=DEFAULT_NORMALIZED_MANIFEST
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze(
        args.baseline,
        args.normalized,
        args.source_manifest,
        args.normalized_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["counts"], indent=2, sort_keys=True))
    print(f"Added primary reasons: {result['added']['primary_reason_counts']}")
    print(f"Removed primary reasons: {result['removed']['primary_reason_counts']}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
