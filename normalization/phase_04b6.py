"""Run the aligned Simplified-corpus plus Simplified-Rime experiment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from interactions.candidates import RimeCliCandidateGenerator
from interactions.generate import generate
from normalization.phase_04b5 import (
    OpenCCCliNormalizer,
    augment_normalized_interactions,
    coverage_from_records,
    interaction_key,
    read_jsonl,
    sha256_file,
)


DEFAULT_CONFIG = Path("config/rime/simplified_candidate_mode.json")
DEFAULT_SCHEMA = Path("data/rime/shared/luna_pinyin.schema.yaml")
DEFAULT_SOURCE_MANIFEST = Path("data/processed/authors/zhu_ziqing/manifest.json")
DEFAULT_NORMALIZED_MANIFEST = Path(
    "data/processed/normalized/authors/zhu_ziqing_t2s/manifest.json"
)
DEFAULT_BASELINE = Path("data/processed/interactions/zhu_ziqing/interactions.jsonl")
DEFAULT_CORPUS_ONLY = Path(
    "data/processed/interactions/zhu_ziqing_t2s/interactions.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/processed/interactions/zhu_ziqing_simplified_rime"
)


def load_simplified_rime_config(
    config_path: Path, schema_path: Path
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    schema = schema_path.read_text(encoding="utf-8")
    required = (
        config["required_engine_filter"],
        f"option_name: {config['enabled_schema_options'][0]}",
        f"opencc_config: {config['required_opencc_config']}",
    )
    missing = [item for item in required if item not in schema]
    if missing:
        raise ValueError(
            "Luna schema does not satisfy simplified mode: " + ", ".join(missing)
        )
    if config["candidate_conversion_location"] != "inside librime engine filter":
        raise ValueError("candidate conversion must occur inside librime")
    return config


def write_augmented_candidate_provenance(
    interaction_path: Path, rime_config: dict[str, Any]
) -> None:
    records = read_jsonl(interaction_path)
    with interaction_path.open("w", encoding="utf-8") as output:
        for record in records:
            record["candidate_representation"] = "rime_engine_zh_hans"
            record["rime_script_provenance"] = {
                "schema_id": rime_config["schema_id"],
                "enabled_schema_options": rime_config["enabled_schema_options"],
                "engine_filter": rime_config["required_engine_filter"],
                "opencc_config": rime_config["required_opencc_config"],
                "candidate_conversion_location": rime_config[
                    "candidate_conversion_location"
                ],
                "user_directory_policy": rime_config["user_directory_policy"],
                "post_retrieval_candidate_conversion": False,
            }
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def batch_convert(values: list[str], config: str) -> dict[str, str]:
    unique = sorted(set(values))
    if not unique:
        return {}
    normalizer = OpenCCCliNormalizer(config=config)
    converted = normalizer.convert("\n".join(unique)).split("\n")
    if converted and converted[-1] == "":
        converted.pop()
    if len(converted) != len(unique):
        raise ValueError(f"OpenCC {config} did not preserve candidate line count")
    return dict(zip(unique, converted, strict=True))


def candidate_script_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    occurrences = Counter(
        candidate["text"] for record in records for candidate in record["candidates"]
    )
    values = list(occurrences)
    t2s = batch_convert(values, "t2s.json")
    s2t = batch_convert(values, "s2t.json")
    classified: dict[str, str] = {}
    for value in values:
        changes_under_t2s = t2s[value] != value
        changes_under_s2t = s2t[value] != value
        if not changes_under_t2s and changes_under_s2t:
            category = "simplified_only"
        elif changes_under_t2s and not changes_under_s2t:
            category = "traditional_only"
        elif changes_under_t2s and changes_under_s2t:
            category = "mixed"
        else:
            category = "script_invariant"
        classified[value] = category

    occurrence_counts = Counter(
        {category: 0 for category in (
            "simplified_only", "traditional_only", "mixed", "script_invariant"
        )}
    )
    unique_counts = Counter(occurrence_counts)
    for value, count in occurrences.items():
        occurrence_counts[classified[value]] += count
        unique_counts[classified[value]] += 1
    total = sum(occurrence_counts.values())
    return {
        "classification_method": (
            "Compare each complete candidate with OpenCC t2s and s2t outputs. "
            "Invariant candidates are reported separately so counts are exhaustive."
        ),
        "candidate_occurrence_count": total,
        "occurrence_counts": dict(occurrence_counts),
        "occurrence_rates": {
            category: count / total if total else 0.0
            for category, count in occurrence_counts.items()
        },
        "unique_candidate_counts": dict(unique_counts),
        "examples": {
            category: [
                {"candidate": value, "occurrences": occurrences[value]}
                for value in sorted(
                    (
                        item
                        for item in values
                        if classified[item] == category
                    ),
                    key=lambda item: (-occurrences[item], item),
                )[:20]
            ]
            for category in occurrence_counts
        },
    }


def coverage_table(
    baseline: list[dict[str, Any]],
    corpus_only: list[dict[str, Any]],
    aligned: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    settings = (
        (
            "Phase 4B",
            "Original Traditional/mixed",
            "Default Luna Traditional/mixed",
            baseline,
        ),
        (
            "Phase 4B.5",
            "OpenCC T2S",
            "Default Luna Traditional/mixed",
            corpus_only,
        ),
        (
            "Phase 4B.6",
            "OpenCC T2S",
            "Luna zh_hans engine option",
            aligned,
        ),
    )
    return [
        {
            "setting": name,
            "corpus": corpus,
            "rime_output": rime_output,
            **coverage_from_records(records),
        }
        for name, corpus, rime_output, records in settings
    ]


def changed_segmentation(
    baseline: list[dict[str, Any]], aligned: list[dict[str, Any]]
) -> dict[str, Any]:
    baseline_by_key = {interaction_key(item): item for item in baseline}
    aligned_by_key = {interaction_key(item): item for item in aligned}
    baseline_keys = set(baseline_by_key)
    aligned_keys = set(aligned_by_key)
    added = [aligned_by_key[key] for key in sorted(aligned_keys - baseline_keys)]
    removed = [baseline_by_key[key] for key in sorted(baseline_keys - aligned_keys)]
    return {
        "comparison_key": ["work_id", "source_start_offset", "source_end_offset"],
        "added_count": len(added),
        "removed_count": len(removed),
        "net_change": len(added) - len(removed),
        "added_by_work": dict(sorted(Counter(item["work_id"] for item in added).items())),
        "removed_by_work": dict(
            sorted(Counter(item["work_id"] for item in removed).items())
        ),
        "added_examples": [
            {
                "work_id": item["work_id"],
                "work_title": item["work_title"],
                "source_start_offset": item["source_start_offset"],
                "source_end_offset": item["source_end_offset"],
                "raw_target": item["source_original_target"],
                "simplified_target": item["target_candidate"],
            }
            for item in added[:20]
        ],
        "removed_examples": [
            {
                "work_id": item["work_id"],
                "work_title": item["work_title"],
                "source_start_offset": item["source_start_offset"],
                "source_end_offset": item["source_end_offset"],
                "raw_target": item["target_candidate"],
            }
            for item in removed[:20]
        ],
    }


def recovery_analysis(
    reference: list[dict[str, Any]],
    aligned: list[dict[str, Any]],
    reference_name: str,
) -> dict[str, Any]:
    aligned_by_key = {interaction_key(item): item for item in aligned}
    missing = [item for item in reference if not item["target_present"]]
    recovered = []
    remaining = []
    unmatched = []
    for item in missing:
        candidate = aligned_by_key.get(interaction_key(item))
        if candidate is None:
            unmatched.append(item)
        elif candidate["target_present"]:
            recovered.append((item, candidate))
        else:
            remaining.append((item, candidate))

    def example(pair: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
        before, after = pair
        return {
            "work_id": before["work_id"],
            "work_title": before["work_title"],
            "source_start_offset": before["source_start_offset"],
            "source_target": before["target_candidate"],
            "aligned_target": after["target_candidate"],
            "pinyin": after["pinyin"],
            "reference_rank": before["target_rank"],
            "aligned_rank": after["target_rank"],
            "aligned_candidates": [item["text"] for item in after["candidates"]],
        }

    return {
        "reference": reference_name,
        "reference_missing_count": len(missing),
        "matching_aligned_span_count": len(recovered) + len(remaining),
        "recovered_count": len(recovered),
        "remaining_missing_count": len(remaining),
        "no_matching_span_count": len(unmatched),
        "recovered_examples": [example(pair) for pair in recovered[:20]],
        "remaining_missing_examples": [example(pair) for pair in remaining[:20]],
    }


def candidate_change_examples(
    baseline: list[dict[str, Any]], aligned: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    baseline_by_pinyin: dict[str, list[str]] = {}
    aligned_by_pinyin: dict[str, list[str]] = {}
    for record in baseline:
        baseline_by_pinyin.setdefault(
            record["pinyin"], [item["text"] for item in record["candidates"]]
        )
    for record in aligned:
        aligned_by_pinyin.setdefault(
            record["pinyin"], [item["text"] for item in record["candidates"]]
        )
    shared = set(baseline_by_pinyin) & set(aligned_by_pinyin)
    ordered = sorted(shared, key=lambda item: (item != "weishenme", item))
    return [
        {
            "pinyin": pinyin,
            "baseline_candidates": baseline_by_pinyin[pinyin],
            "simplified_candidates": aligned_by_pinyin[pinyin],
        }
        for pinyin in ordered
        if baseline_by_pinyin[pinyin] != aligned_by_pinyin[pinyin]
    ][:30]


def engine_verification_examples(
    rime_manifest_path: Path,
    rime_executable: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rime_manifest = json.loads(rime_manifest_path.read_text(encoding="utf-8"))
    common = {
        "executable": rime_executable,
        "shared_data": Path(rime_manifest["shared_data_dir"]),
        "prebuilt_data": Path(rime_manifest["prebuilt_data_dir"]),
        "version": rime_manifest["librime"],
        "schema_id": config["schema_id"],
        "max_candidates": 10,
    }
    examples = []
    with RimeCliCandidateGenerator(**common) as baseline_generator:
        with RimeCliCandidateGenerator(
            **common,
            enabled_options=tuple(config["enabled_schema_options"]),
        ) as simplified_generator:
            for pinyin in ("weishenme", "women", "shihou"):
                examples.append(
                    {
                        "pinyin": pinyin,
                        "baseline_candidates": [
                            item.text for item in baseline_generator.candidates(pinyin)
                        ],
                        "simplified_candidates": [
                            item.text for item in simplified_generator.candidates(pinyin)
                        ],
                        "post_retrieval_conversion": False,
                    }
                )
    return examples


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_simplified_rime_config(args.config, args.schema)
    source_hashes_before = {
        path.name: sha256_file(path)
        for path in sorted(args.source_manifest.parent.glob("*.txt"))
    }
    generation_args = SimpleNamespace(
        corpus_manifest=args.normalized_manifest,
        rime_manifest=args.rime_manifest,
        rime_executable=args.rime_executable,
        output_dir=args.output_dir,
        work_id=None,
        max_interactions=None,
        min_target_length=2,
        max_target_length=4,
        context_characters=12,
        max_candidates=10,
        rime_options=config["enabled_schema_options"],
    )
    manifest = generate(generation_args)
    interaction_path = args.output_dir / "interactions.jsonl"
    augment_normalized_interactions(
        interaction_path, args.source_manifest, args.normalized_manifest
    )
    write_augmented_candidate_provenance(interaction_path, config)
    manifest["candidate_script_mode"] = config
    manifest["source_original_corpus_manifest"] = str(args.source_manifest)
    manifest["normalized_corpus_manifest"] = str(args.normalized_manifest)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    source_hashes_after = {
        path.name: sha256_file(path)
        for path in sorted(args.source_manifest.parent.glob("*.txt"))
    }
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("Phase 4A corpus changed during Phase 4B.6")

    baseline = read_jsonl(args.baseline)
    corpus_only = read_jsonl(args.corpus_only)
    aligned = read_jsonl(interaction_path)
    comparison = {
        "schema_version": 1,
        "research_question": (
            "How does candidate coverage change when corpus and Rime output "
            "use the same Simplified Chinese convention?"
        ),
        "rime_simplified_configuration": config,
        "source_integrity": {
            "phase_04a_unchanged": True,
            "phase_04a_file_sha256": source_hashes_after,
        },
        "coverage_table": coverage_table(baseline, corpus_only, aligned),
        "candidate_script_distribution": {
            "phase_04b_default_rime": candidate_script_distribution(baseline),
            "phase_04b6_simplified_rime": candidate_script_distribution(aligned),
        },
        "candidate_change_examples": candidate_change_examples(baseline, aligned),
        "engine_verification_examples": engine_verification_examples(
            args.rime_manifest, args.rime_executable, config
        ),
        "interaction_delta_vs_phase_04b": changed_segmentation(baseline, aligned),
        "coverage_recovery_from_phase_04b5": recovery_analysis(
            corpus_only, aligned, "Phase 4B.5 corpus-only T2S"
        ),
        "coverage_recovery_from_phase_04b": recovery_analysis(
            baseline, aligned, "Phase 4B baseline"
        ),
        "input_sha256": {
            "phase_04b_interactions": sha256_file(args.baseline),
            "phase_04b5_interactions": sha256_file(args.corpus_only),
            "phase_04b5_normalized_manifest": sha256_file(args.normalized_manifest),
        },
        "output_interactions_sha256": sha256_file(interaction_path),
    }
    comparison_path = args.output_dir / "phase_04b6_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return comparison


def print_summary(comparison: dict[str, Any]) -> None:
    print("Coverage comparison")
    for row in comparison["coverage_table"]:
        print(
            f"{row['setting']}: n={row['interaction_count']}, "
            f"Top-1={row['top_1_rate']:.2%}, Top-3={row['top_3_rate']:.2%}, "
            f"Top-5={row['top_5_rate']:.2%}, Top-10={row['top_10_rate']:.2%}, "
            f"missing={row['missing_rate']:.2%}"
        )
    distribution = comparison["candidate_script_distribution"][
        "phase_04b6_simplified_rime"
    ]
    print(f"Simplified Rime candidate scripts: {distribution['occurrence_counts']}")
    recovery = comparison["coverage_recovery_from_phase_04b5"]
    print(
        "Phase 4B.5 misses recovered by engine-aligned Rime: "
        f"{recovery['recovered_count']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument(
        "--normalized-manifest", type=Path, default=DEFAULT_NORMALIZED_MANIFEST
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--corpus-only", type=Path, default=DEFAULT_CORPUS_ONLY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--rime-manifest", type=Path, default=Path("data/rime/setup_manifest.json")
    )
    parser.add_argument(
        "--rime-executable", type=Path, default=Path(".build/rime_candidate_cli")
    )
    args = parser.parse_args()
    comparison = run(args)
    print_summary(comparison)
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
