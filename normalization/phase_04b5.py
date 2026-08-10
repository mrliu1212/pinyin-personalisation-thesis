"""Build and compare OpenCC-normalized Phase 4B interactions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from interactions.generate import generate


DEFAULT_SOURCE_MANIFEST = Path("data/processed/authors/zhu_ziqing/manifest.json")
DEFAULT_NORMALIZED_CORPUS_DIR = Path(
    "data/processed/normalized/authors/zhu_ziqing_t2s"
)
DEFAULT_BASELINE_INTERACTIONS = Path(
    "data/processed/interactions/zhu_ziqing/interactions.jsonl"
)
DEFAULT_NORMALIZED_INTERACTION_DIR = Path(
    "data/processed/interactions/zhu_ziqing_t2s"
)
DEFAULT_RIME_MANIFEST = Path("data/rime/setup_manifest.json")
DEFAULT_RIME_EXECUTABLE = Path(".build/rime_candidate_cli")
OPENCC_CONFIG = "t2s.json"


class TextNormalizer(Protocol):
    name: str
    version: str
    config: str

    def convert(self, text: str) -> str: ...


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


class OpenCCCliNormalizer:
    name = "OpenCC"

    def __init__(self, executable: str = "opencc", config: str = OPENCC_CONFIG):
        resolved = shutil.which(executable)
        if resolved is None:
            raise RuntimeError(
                "OpenCC executable not found. Install it with: brew install opencc"
            )
        self.executable = resolved
        self.config = config
        version_result = subprocess.run(
            [self.executable, "--version"],
            check=True,
            text=True,
            capture_output=True,
        )
        version_output = (version_result.stdout + version_result.stderr).strip()
        self.version = " ".join(version_output.split())

    def convert(self, text: str) -> str:
        result = subprocess.run(
            [self.executable, "-c", self.config],
            input=text,
            check=True,
            text=True,
            capture_output=True,
            encoding="utf-8",
        )
        return result.stdout


def prepare_normalized_corpus(
    source_manifest_path: Path,
    output_dir: Path,
    normalizer: TextNormalizer,
) -> dict[str, Any]:
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_works: list[dict[str, Any]] = []

    for source_work in source_manifest["works"]:
        work = dict(source_work)
        if not work["included"]:
            normalized_works.append(work)
            continue

        source_file = source_manifest_path.parent / source_work["processed_file"]
        source_bytes = source_file.read_bytes()
        source_sha256 = sha256_bytes(source_bytes)
        if source_sha256 != source_work["processed_sha256"]:
            raise ValueError(f"source checksum mismatch: {source_file}")
        source_text = source_bytes.decode("utf-8")
        normalized_text = normalizer.convert(source_text)
        if len(source_text) != len(normalized_text):
            raise ValueError(
                f"normalization changed text length for {source_work['work_id']}; "
                "exact source offsets cannot be preserved"
            )

        normalized_file = output_dir / source_work["processed_file"]
        normalized_bytes = normalized_text.encode("utf-8")
        normalized_file.write_bytes(normalized_bytes)
        normalized_sha256 = sha256_bytes(normalized_bytes)
        changed_positions = sum(
            source_character != normalized_character
            for source_character, normalized_character in zip(
                source_text, normalized_text, strict=True
            )
        )
        work["content_variant"] = "derived_opencc_t2s"
        work["processed_sha256"] = normalized_sha256
        work["normalization_provenance"] = {
            "source_processed_file": str(source_file),
            "source_processed_sha256": source_sha256,
            "normalized_processed_file": source_work["processed_file"],
            "normalized_processed_sha256": normalized_sha256,
            "normalizer": normalizer.name,
            "normalizer_version": normalizer.version,
            "configuration": normalizer.config,
            "changed_codepoint_positions": changed_positions,
            "source_codepoint_length": len(source_text),
            "normalized_codepoint_length": len(normalized_text),
            "offset_mapping": "identity; equal code-point length verified",
        }
        normalized_works.append(work)

    manifest = {
        "schema_version": 1,
        "author_id": source_manifest["author_id"],
        "author_name": source_manifest["author_name"],
        "source_corpus_manifest": str(source_manifest_path),
        "source_corpus_manifest_sha256": sha256_file(source_manifest_path),
        "normalization": {
            "normalizer": normalizer.name,
            "normalizer_version": normalizer.version,
            "configuration": normalizer.config,
            "direction": "Traditional Chinese to Simplified Chinese",
            "writes_source_files": False,
            "offset_policy": (
                "require equal Unicode code-point length; fail otherwise"
            ),
        },
        "works": normalized_works,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def augment_normalized_interactions(
    interaction_path: Path,
    source_manifest_path: Path,
    normalized_manifest_path: Path,
) -> None:
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    normalized_manifest = json.loads(
        normalized_manifest_path.read_text(encoding="utf-8")
    )
    source_works = {
        work["work_id"]: work for work in source_manifest["works"] if work["included"]
    }
    normalized_works = {
        work["work_id"]: work
        for work in normalized_manifest["works"]
        if work["included"]
    }
    source_texts = {
        work_id: (
            source_manifest_path.parent / work["processed_file"]
        ).read_text(encoding="utf-8")
        for work_id, work in source_works.items()
    }

    records: list[dict[str, Any]] = []
    for line in interaction_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        work_id = record["work_id"]
        source_work = source_works[work_id]
        normalized_work = normalized_works[work_id]
        start = record["source_start_offset"]
        end = record["source_end_offset"]
        source_target = source_texts[work_id][start:end]
        provenance = normalized_work["normalization_provenance"]
        record["text_representation"] = "opencc_t2s"
        record["source_original_target"] = source_target
        record["normalization_provenance"] = {
            "source_corpus_manifest": str(source_manifest_path),
            "source_original_processed_file": source_work["processed_file"],
            "source_original_processed_sha256": provenance[
                "source_processed_sha256"
            ],
            "normalized_corpus_manifest": str(normalized_manifest_path),
            "normalized_processed_file": normalized_work["processed_file"],
            "normalized_processed_sha256": provenance[
                "normalized_processed_sha256"
            ],
            "normalizer": provenance["normalizer"],
            "normalizer_version": provenance["normalizer_version"],
            "configuration": provenance["configuration"],
            "offset_mapping": provenance["offset_mapping"],
        }
        records.append(record)

    with interaction_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def interaction_key(item: dict[str, Any]) -> tuple[str, int, int]:
    return (
        item["work_id"],
        int(item["source_start_offset"]),
        int(item["source_end_offset"]),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def coverage_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    result: dict[str, Any] = {"interaction_count": total}
    for k in (1, 3, 5, 10):
        count = sum(
            item["target_rank"] is not None and item["target_rank"] <= k
            for item in records
        )
        result[f"top_{k}_count"] = count
        result[f"top_{k}_rate"] = count / total if total else 0.0
    missing = sum(not item["target_present"] for item in records)
    result["missing_count"] = missing
    result["missing_rate"] = missing / total if total else 0.0
    return result


def comparison_example(
    baseline: dict[str, Any], normalized: dict[str, Any]
) -> dict[str, Any]:
    return {
        "work_id": baseline["work_id"],
        "work_title": baseline["work_title"],
        "source_start_offset": baseline["source_start_offset"],
        "source_end_offset": baseline["source_end_offset"],
        "source_target": baseline["target_candidate"],
        "normalized_target": normalized["target_candidate"],
        "baseline_pinyin": baseline["pinyin"],
        "normalized_pinyin": normalized["pinyin"],
        "baseline_rank": baseline["target_rank"],
        "normalized_rank": normalized["target_rank"],
        "normalized_base_candidates": [
            candidate["text"] for candidate in normalized["candidates"]
        ],
    }


def compare_interactions(
    baseline_records: list[dict[str, Any]],
    normalized_records: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_by_span = {
        interaction_key(item): item for item in normalized_records
    }
    baseline_missing = [
        item for item in baseline_records if not item["target_present"]
    ]
    recovered: list[tuple[dict[str, Any], dict[str, Any]]] = []
    remaining: list[tuple[dict[str, Any], dict[str, Any]]] = []
    no_matching_span: list[dict[str, Any]] = []
    for baseline in baseline_missing:
        normalized = normalized_by_span.get(interaction_key(baseline))
        if normalized is None:
            no_matching_span.append(baseline)
        elif normalized["target_present"]:
            recovered.append((baseline, normalized))
        else:
            remaining.append((baseline, normalized))

    baseline_present = [item for item in baseline_records if item["target_present"]]
    previously_present_now_missing = []
    for baseline in baseline_present:
        normalized = normalized_by_span.get(interaction_key(baseline))
        if normalized is not None and not normalized["target_present"]:
            previously_present_now_missing.append((baseline, normalized))

    baseline_coverage = coverage_from_records(baseline_records)
    normalized_coverage = coverage_from_records(normalized_records)
    return {
        "comparison_definition": (
            "Recovery requires an identical work/start/end source span; "
            "spans changed by re-segmentation are reported separately."
        ),
        "baseline": baseline_coverage,
        "normalized": normalized_coverage,
        "rate_delta_normalized_minus_baseline": {
            f"top_{k}": normalized_coverage[f"top_{k}_rate"]
            - baseline_coverage[f"top_{k}_rate"]
            for k in (1, 3, 5, 10)
        }
        | {
            "missing": normalized_coverage["missing_rate"]
            - baseline_coverage["missing_rate"]
        },
        "baseline_missing_span_analysis": {
            "baseline_missing_count": len(baseline_missing),
            "matching_normalized_span_count": len(recovered) + len(remaining),
            "recovered_count": len(recovered),
            "remaining_missing_count": len(remaining),
            "no_matching_span_after_normalized_resegmentation": len(
                no_matching_span
            ),
            "previously_present_now_missing_same_span_count": len(
                previously_present_now_missing
            ),
        },
        "recovered_examples": [
            comparison_example(baseline, normalized)
            for baseline, normalized in recovered[:20]
        ],
        "remaining_missing_examples": [
            comparison_example(baseline, normalized)
            for baseline, normalized in remaining[:20]
        ],
        "previously_present_now_missing_examples": [
            comparison_example(baseline, normalized)
            for baseline, normalized in previously_present_now_missing[:20]
        ],
        "no_matching_span_examples": [
            {
                "work_id": item["work_id"],
                "work_title": item["work_title"],
                "source_start_offset": item["source_start_offset"],
                "source_end_offset": item["source_end_offset"],
                "source_target": item["target_candidate"],
                "baseline_pinyin": item["pinyin"],
            }
            for item in no_matching_span[:20]
        ],
    }


def run_phase_04b5(args: argparse.Namespace) -> dict[str, Any]:
    normalizer = OpenCCCliNormalizer(args.opencc_executable, args.opencc_config)
    source_hashes_before = {
        path.name: sha256_file(path)
        for path in sorted(args.source_manifest.parent.glob("*.txt"))
    }
    normalized_manifest = prepare_normalized_corpus(
        args.source_manifest, args.normalized_corpus_dir, normalizer
    )
    normalized_manifest_path = args.normalized_corpus_dir / "manifest.json"

    generation_args = SimpleNamespace(
        corpus_manifest=normalized_manifest_path,
        rime_manifest=args.rime_manifest,
        rime_executable=args.rime_executable,
        output_dir=args.normalized_interaction_dir,
        work_id=None,
        max_interactions=None,
        min_target_length=2,
        max_target_length=4,
        context_characters=12,
        max_candidates=10,
    )
    interaction_manifest = generate(generation_args)
    normalized_interaction_path = (
        args.normalized_interaction_dir / "interactions.jsonl"
    )
    augment_normalized_interactions(
        normalized_interaction_path,
        args.source_manifest,
        normalized_manifest_path,
    )
    interaction_manifest["normalization"] = normalized_manifest["normalization"]
    interaction_manifest["source_original_corpus_manifest"] = str(
        args.source_manifest
    )
    interaction_manifest["normalized_corpus_manifest"] = str(
        normalized_manifest_path
    )
    (args.normalized_interaction_dir / "manifest.json").write_text(
        json.dumps(interaction_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    source_hashes_after = {
        path.name: sha256_file(path)
        for path in sorted(args.source_manifest.parent.glob("*.txt"))
    }
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("Phase 4A source corpus changed during normalization")

    comparison = compare_interactions(
        read_jsonl(args.baseline_interactions),
        read_jsonl(normalized_interaction_path),
    )
    comparison["source_integrity"] = {
        "unchanged": True,
        "phase_04a_file_sha256": source_hashes_after,
    }
    comparison["normalization"] = normalized_manifest["normalization"]
    comparison_path = args.normalized_interaction_dir / "coverage_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return comparison


def print_comparison(comparison: dict[str, Any]) -> None:
    baseline = comparison["baseline"]
    normalized = comparison["normalized"]
    print("Coverage comparison")
    print(f"{'Metric':<10} {'Before':>10} {'After T2S':>10}")
    for k in (1, 3, 5, 10):
        print(
            f"Top-{k:<5} {baseline[f'top_{k}_rate']:>9.2%} "
            f"{normalized[f'top_{k}_rate']:>10.2%}"
        )
    print(
        f"{'Missing':<10} {baseline['missing_rate']:>9.2%} "
        f"{normalized['missing_rate']:>10.2%}"
    )
    spans = comparison["baseline_missing_span_analysis"]
    print(f"Previously missing recovered: {spans['recovered_count']}")
    print(f"Previously missing and still missing: {spans['remaining_missing_count']}")
    print(
        "No matching span after normalized re-segmentation: "
        f"{spans['no_matching_span_after_normalized_resegmentation']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST
    )
    parser.add_argument(
        "--normalized-corpus-dir", type=Path, default=DEFAULT_NORMALIZED_CORPUS_DIR
    )
    parser.add_argument(
        "--baseline-interactions", type=Path, default=DEFAULT_BASELINE_INTERACTIONS
    )
    parser.add_argument(
        "--normalized-interaction-dir",
        type=Path,
        default=DEFAULT_NORMALIZED_INTERACTION_DIR,
    )
    parser.add_argument("--rime-manifest", type=Path, default=DEFAULT_RIME_MANIFEST)
    parser.add_argument(
        "--rime-executable", type=Path, default=DEFAULT_RIME_EXECUTABLE
    )
    parser.add_argument("--opencc-executable", default="opencc")
    parser.add_argument("--opencc-config", default=OPENCC_CONFIG)
    args = parser.parse_args()
    comparison = run_phase_04b5(args)
    print_comparison(comparison)
    print(f"Normalized corpus: {args.normalized_corpus_dir}")
    print(f"Normalized interactions: {args.normalized_interaction_dir}")


if __name__ == "__main__":
    main()
