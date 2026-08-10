"""Audit the existing Phase 4B interaction dataset without changing it."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("data/processed/interactions/zhu_ziqing/interactions.jsonl")
DEFAULT_OUTPUT = Path("results/audits/phase_04b")
DEFAULT_SEED = 40402
DEFAULT_SAMPLE_SIZE = 100


def read_interactions(path: Path) -> list[dict[str, Any]]:
    interactions: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                interactions.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on {path}:{line_number}") from error
    return interactions


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seeded_sample(
    interactions: Iterable[dict[str, Any]],
    *,
    seed: int,
    category: str,
    sample_size: int,
) -> list[dict[str, Any]]:
    """Select a stable hash-ranked pseudorandom sample.

    Including the category in the seeded hash gives each stratum an independent
    deterministic ordering. Selection is stable across input ordering and
    Python versions.
    """

    pool = list(interactions)
    if len(pool) < sample_size:
        raise ValueError(
            f"{category} contains {len(pool)} interactions; "
            f"cannot sample {sample_size} without replacement"
        )

    def sample_key(item: dict[str, Any]) -> tuple[str, str]:
        interaction_id = str(item["interaction_id"])
        digest = hashlib.sha256(
            f"{seed}:{category}:{interaction_id}".encode("utf-8")
        ).hexdigest()
        return digest, interaction_id

    return sorted(pool, key=sample_key)[:sample_size]


def review_record(item: dict[str, Any], category: str) -> dict[str, Any]:
    return {
        "audit_sample_category": category,
        "interaction_id": item["interaction_id"],
        "work_id": item["work_id"],
        "work_title": item["work_title"],
        "work_date": item["work_date"],
        "source_processed_file": item.get("source_processed_file"),
        "source_page_url": item.get("source_page_url"),
        "source_revision_id": item.get("source_revision_id"),
        "source_start_offset": item.get("source_start_offset"),
        "source_end_offset": item.get("source_end_offset"),
        "raw_context": item["raw_context"],
        "derived_context": item["derived_context"],
        "target_candidate": item["target_candidate"],
        "target_length": item["target_length"],
        "pinyin": item.get("pinyin"),
        "pinyin_syllables": item.get("pinyin_syllables"),
        "polyphonic_review_required": item["polyphonic_review_required"],
        "polyphonic_characters": item.get("polyphonic_characters", []),
        "ordered_base_candidates": item["candidates"],
        "target_rank": item["target_rank"],
        "target_present": item["target_present"],
    }


def missing_diagnostics(
    missing: list[dict[str, Any]], total_interactions: int, maximum_k: int = 10
) -> dict[str, Any]:
    target_counts = Counter(item["target_candidate"] for item in missing)
    repeated: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in missing:
        grouped[item["target_candidate"]].append(item)
    for target, count in sorted(
        target_counts.items(), key=lambda pair: (-pair[1], pair[0])
    ):
        if count < 2:
            continue
        examples = grouped[target]
        repeated.append(
            {
                "target_candidate": target,
                "count": count,
                "pinyin_values": sorted(
                    {
                        str(item["pinyin"])
                        for item in examples
                        if item.get("pinyin")
                    }
                ),
                "work_ids": sorted({item["work_id"] for item in examples}),
                "example_interaction_ids": [
                    item["interaction_id"] for item in examples[:3]
                ],
            }
        )

    flagged = sum(item["polyphonic_review_required"] for item in missing)
    pinyin_stored = sum(bool(item.get("pinyin")) for item in missing)
    syllables_stored = sum(bool(item.get("pinyin_syllables")) for item in missing)
    return {
        "maximum_candidate_k": maximum_k,
        "source_interaction_count": total_interactions,
        "top_10_missing_count": len(missing),
        "top_10_missing_rate": len(missing) / total_interactions,
        "target_length_distribution": dict(
            sorted(Counter(str(item["target_length"]) for item in missing).items())
        ),
        "work_distribution": dict(
            sorted(Counter(item["work_id"] for item in missing).items())
        ),
        "pinyin_conversion_status": {
            "stored_pinyin": pinyin_stored,
            "missing_stored_pinyin": len(missing) - pinyin_stored,
            "stored_pinyin_syllables": syllables_stored,
            "missing_stored_pinyin_syllables": len(missing) - syllables_stored,
        },
        "polyphonic_review": {
            "flagged": flagged,
            "unflagged": len(missing) - flagged,
            "flagged_rate": flagged / len(missing) if missing else 0.0,
        },
        "candidate_list_size_distribution": dict(
            sorted(Counter(str(item["candidate_list_size"]) for item in missing).items())
        ),
        "repeated_missing_targets": repeated,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def render_summary(diagnostics: dict[str, Any]) -> str:
    polyphonic = diagnostics["polyphonic_review"]
    pinyin = diagnostics["pinyin_conversion_status"]
    repeated = diagnostics["repeated_missing_targets"][:20]
    repeated_rows = "\n".join(
        f"| {item['target_candidate']} | {item['count']} | "
        f"{', '.join(item['pinyin_values'])} | {', '.join(item['work_ids'])} |"
        for item in repeated
    )
    if not repeated_rows:
        repeated_rows = "| *(none)* | 0 | — | — |"
    return f"""# Phase 4B Data-Quality Audit Summary

This is a factual audit of the existing interaction dataset. It does not infer
error causes, alter eligibility, or report personalisation performance.

## Top-10 Missing Diagnostics

- Source interactions: {diagnostics['source_interaction_count']}
- Target absent from Top-10: {diagnostics['top_10_missing_count']}
- Missing rate: {diagnostics['top_10_missing_rate']:.2%}
- Stored Pinyin: {pinyin['stored_pinyin']}
- Stored syllable lists: {pinyin['stored_pinyin_syllables']}
- Polyphonic-review flagged: {polyphonic['flagged']} ({polyphonic['flagged_rate']:.2%})
- Polyphonic-review unflagged: {polyphonic['unflagged']}

Target-length distribution: `{json.dumps(diagnostics['target_length_distribution'], ensure_ascii=False, sort_keys=True)}`

Work distribution: `{json.dumps(diagnostics['work_distribution'], ensure_ascii=False, sort_keys=True)}`

Candidate-list-size distribution: `{json.dumps(diagnostics['candidate_list_size_distribution'], ensure_ascii=False, sort_keys=True)}`

## Most Frequent Repeated Missing Targets

| Target | Count | Stored Pinyin | Works |
| --- | ---: | --- | --- |
{repeated_rows}

The complete repeated-target list and aggregate counts are in
`top10_missing_diagnostics.json`. Manual reviewers may consider Pinyin,
segmentation, character variants, vocabulary rarity, candidate truncation, and
other explanations, but this audit assigns none of those labels automatically.
"""


def run_audit(
    input_path: Path,
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> dict[str, Any]:
    interactions = read_interactions(input_path)
    flagged = [item for item in interactions if item["polyphonic_review_required"]]
    unflagged = [
        item for item in interactions if not item["polyphonic_review_required"]
    ]
    missing = [item for item in interactions if not item["target_present"]]
    strata = {
        "polyphonic_flagged": flagged,
        "polyphonic_unflagged": unflagged,
        "top10_missing": missing,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: dict[str, str] = {}
    for category, pool in strata.items():
        sample = seeded_sample(
            pool, seed=seed, category=category, sample_size=sample_size
        )
        filename = f"{category}_sample.jsonl"
        write_jsonl(
            output_dir / filename,
            (review_record(item, category) for item in sample),
        )
        output_files[category] = filename

    diagnostics = missing_diagnostics(missing, len(interactions))
    write_json(output_dir / "top10_missing_diagnostics.json", diagnostics)
    (output_dir / "audit_summary.md").write_text(
        render_summary(diagnostics), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "input_path": str(input_path),
        "input_sha256": file_sha256(input_path),
        "fixed_seed": seed,
        "sampling_method": (
            "SHA-256 rank of seed, stratum name, and interaction_id; "
            "sampling without replacement"
        ),
        "sample_size_per_stratum": sample_size,
        "source_interaction_count": len(interactions),
        "eligible_population_counts": {
            category: len(pool) for category, pool in strata.items()
        },
        "output_files": output_files,
        "diagnostics_file": "top10_missing_diagnostics.json",
        "summary_file": "audit_summary.md",
        "automatic_error_classification": False,
    }
    write_json(output_dir / "audit_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    args = parser.parse_args()
    manifest = run_audit(
        args.input, args.output_dir, seed=args.seed, sample_size=args.sample_size
    )
    print(f"Source interactions: {manifest['source_interaction_count']}")
    for category, count in manifest["eligible_population_counts"].items():
        print(f"{category}: sampled {args.sample_size} from {count}")
    print(f"Fixed seed: {args.seed}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
