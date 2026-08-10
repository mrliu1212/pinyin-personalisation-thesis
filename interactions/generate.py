"""Generate Phase 4B interactions and Base coverage from the Phase 4A corpus."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .candidates import RimeCliCandidateGenerator
from .construction import construct_work_interactions
from .linguistic import JiebaSegmenter, TargetPolicy


def coverage_summary(interactions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(interactions)
    ranks = [item["target_rank"] for item in interactions]
    sizes = [item["candidate_list_size"] for item in interactions]
    counts = {
        f"top_{k}": sum(rank is not None and rank <= k for rank in ranks)
        for k in (1, 3, 5, 10)
    }
    return {
        "interaction_count": total,
        "target_present_top_1": counts["top_1"],
        "target_present_top_3": counts["top_3"],
        "target_present_top_5": counts["top_5"],
        "target_present_top_10": counts["top_10"],
        "target_absent": sum(rank is None for rank in ranks),
        "coverage_top_1": counts["top_1"] / total if total else 0.0,
        "coverage_top_3": counts["top_3"] / total if total else 0.0,
        "coverage_top_5": counts["top_5"] / total if total else 0.0,
        "coverage_top_10": counts["top_10"] / total if total else 0.0,
        "missing_target_rate": sum(rank is None for rank in ranks) / total if total else 0.0,
        "mean_candidate_list_size": statistics.mean(sizes) if sizes else 0.0,
        "median_candidate_list_size": statistics.median(sizes) if sizes else 0.0,
        "polyphonic_review_count": sum(
            item["polyphonic_review_required"] for item in interactions
        ),
        "interactions_by_work": dict(
            sorted(Counter(item["work_id"] for item in interactions).items())
        ),
        "interactions_by_target_length": {
            str(key): value
            for key, value in sorted(
                Counter(item["target_length"] for item in interactions).items()
            )
        },
    }


def generate(args: argparse.Namespace) -> dict[str, Any]:
    corpus_manifest = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
    rime_manifest = json.loads(args.rime_manifest.read_text(encoding="utf-8"))
    works = [item for item in corpus_manifest["works"] if item["included"]]
    if args.work_id:
        works = [item for item in works if item["work_id"] == args.work_id]
        if not works:
            raise ValueError(f"included work not found: {args.work_id}")

    policy = TargetPolicy(
        min_characters=args.min_target_length,
        max_characters=args.max_target_length,
        derived_context_characters=args.context_characters,
    )
    segmenter = JiebaSegmenter()
    all_interactions: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    eligible_before_limit = 0
    pinyin_failures = 0
    with RimeCliCandidateGenerator(
        args.rime_executable,
        Path(rime_manifest["shared_data_dir"]),
        Path(rime_manifest["prebuilt_data_dir"]),
        version=rime_manifest["librime"],
        schema_id=rime_manifest["schema_id"],
        max_candidates=args.max_candidates,
        enabled_options=tuple(getattr(args, "rime_options", ())),
    ) as generator:
        for work in works:
            text_path = args.corpus_manifest.parent / work["processed_file"]
            result = construct_work_interactions(
                text_path.read_text(encoding="utf-8"),
                work,
                segmenter,
                generator,
                policy,
                max_interactions=args.max_interactions,
            )
            all_interactions.extend(result.interactions)
            exclusions.update(result.exclusions)
            eligible_before_limit += result.eligible_before_limit
            pinyin_failures += result.pinyin_failures

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "interactions.jsonl").open("w", encoding="utf-8") as output:
        for interaction in all_interactions:
            output.write(json.dumps(interaction, ensure_ascii=False, sort_keys=True) + "\n")

    coverage = coverage_summary(all_interactions)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "author_id": corpus_manifest["author_id"],
        "author_name": corpus_manifest["author_name"],
        "source_corpus_manifest": str(args.corpus_manifest),
        "work_filter": args.work_id,
        "preprocessing": {
            "segmentation_method": "jieba.Tokenizer.tokenize(mode=default)",
            "jieba_version": importlib.metadata.version("jieba"),
            "pinyin_method": "pypinyin Style.NORMAL, strict, tone-free, concatenated",
            "pypinyin_version": importlib.metadata.version("pypinyin"),
            "target_policy": {
                "all_chinese_only": True,
                "minimum_characters": policy.min_characters,
                "maximum_characters": policy.max_characters,
                "exclude_punctuation_latin_numbers": True,
            },
            "context_policy": {
                "raw_context": "complete preceding source text",
                "derived_context": "last N Chinese characters from raw context",
                "derived_context_characters": policy.derived_context_characters,
            },
        },
        "candidate_generator": {
            "name": "librime",
            "version": rime_manifest["librime"],
            "schema_id": rime_manifest["schema_id"],
            "maximum_k": args.max_candidates,
            "numeric_score_available": False,
            "ordering": "candidate iterator order returned by librime",
            "enabled_schema_options": list(getattr(args, "rime_options", ())),
            "rime_source_lock": rime_manifest["source_lock"],
        },
        "eligible_targets_before_optional_limit": eligible_before_limit,
        "successfully_converted_pinyin_interactions": len(all_interactions),
        "pinyin_conversion_failures": pinyin_failures,
        "excluded_token_counts": dict(sorted(exclusions.items())),
        "coverage": coverage,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=Path("data/processed/authors/zhu_ziqing/manifest.json"),
    )
    parser.add_argument(
        "--rime-manifest", type=Path, default=Path("data/rime/setup_manifest.json")
    )
    parser.add_argument(
        "--rime-executable", type=Path, default=Path(".build/rime_candidate_cli")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/interactions/zhu_ziqing"),
    )
    parser.add_argument("--work-id")
    parser.add_argument("--max-interactions", type=int)
    parser.add_argument("--min-target-length", type=int, default=2)
    parser.add_argument("--max-target-length", type=int, default=4)
    parser.add_argument("--context-characters", type=int, default=12)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument(
        "--rime-option",
        dest="rime_options",
        action="append",
        default=[],
        help="enable a Rime schema option in the isolated candidate session",
    )
    args = parser.parse_args()
    manifest = generate(args)
    coverage = manifest["coverage"]
    print(f"Interactions: {coverage['interaction_count']}")
    print(
        "Coverage: "
        f"Top-1={coverage['coverage_top_1']:.3f}, "
        f"Top-3={coverage['coverage_top_3']:.3f}, "
        f"Top-5={coverage['coverage_top_5']:.3f}, "
        f"Top-10={coverage['coverage_top_10']:.3f}"
    )
    print(
        f"Missing targets: {coverage['target_absent']} "
        f"({coverage['missing_target_rate']:.3f})"
    )
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
