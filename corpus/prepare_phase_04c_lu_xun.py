"""Reproduce the Phase 4C Lu Xun corpus and aligned interaction dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from interactions.generate import generate
from normalization.phase_04b5 import (
    OpenCCCliNormalizer,
    augment_normalized_interactions,
    prepare_normalized_corpus,
)
from normalization.phase_04b6 import (
    load_simplified_rime_config,
    write_augmented_candidate_provenance,
)

from .acquire import acquire
from .prepare import prepare


DEFAULT_CATALOG = Path("data/manifests/lu_xun_works.json")
DEFAULT_RAW_DIR = Path("data/raw/authors/lu_xun")
DEFAULT_PROCESSED_DIR = Path("data/processed/authors/lu_xun")
DEFAULT_NORMALIZED_DIR = Path("data/processed/normalized/authors/lu_xun_t2s")
DEFAULT_INTERACTION_DIR = Path(
    "data/processed/interactions/lu_xun_simplified_rime"
)


def build(args: argparse.Namespace) -> dict[str, Any]:
    """Run the accepted Zhu processing stages with separate Lu Xun paths."""
    if args.acquire:
        acquire(args.catalog, args.raw_dir)
    if not (args.raw_dir / "acquisition_manifest.json").exists():
        raise FileNotFoundError(
            f"missing raw acquisition manifest: {args.raw_dir}; "
            "run with --acquire or restore the revision-pinned raw corpus"
        )

    processed_manifest = prepare(args.catalog, args.raw_dir, args.processed_dir)
    normalized_manifest = prepare_normalized_corpus(
        args.processed_dir / "manifest.json",
        args.normalized_dir,
        OpenCCCliNormalizer(config="t2s.json"),
    )
    interaction_manifest = generate(
        SimpleNamespace(
            corpus_manifest=args.normalized_dir / "manifest.json",
            rime_manifest=args.rime_manifest,
            rime_executable=args.rime_executable,
            output_dir=args.interaction_dir,
            work_id=None,
            max_interactions=None,
            min_target_length=2,
            max_target_length=4,
            context_characters=12,
            max_candidates=10,
            rime_options=["zh_hans"],
        )
    )
    interaction_path = args.interaction_dir / "interactions.jsonl"
    augment_normalized_interactions(
        interaction_path,
        args.processed_dir / "manifest.json",
        args.normalized_dir / "manifest.json",
    )
    rime_config = load_simplified_rime_config(args.rime_config, args.rime_schema)
    write_augmented_candidate_provenance(interaction_path, rime_config)
    return {
        "processed_manifest": processed_manifest,
        "normalized_manifest": normalized_manifest,
        "interaction_manifest": interaction_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--interaction-dir", type=Path, default=DEFAULT_INTERACTION_DIR)
    parser.add_argument(
        "--rime-manifest",
        type=Path,
        default=Path("data/rime/setup_manifest.json"),
    )
    parser.add_argument(
        "--rime-executable", type=Path, default=Path(".build/rime_candidate_cli")
    )
    parser.add_argument(
        "--rime-config",
        type=Path,
        default=Path("config/rime/simplified_candidate_mode.json"),
    )
    parser.add_argument(
        "--rime-schema",
        type=Path,
        default=Path("data/rime/shared/luna_pinyin.schema.yaml"),
    )
    parser.add_argument(
        "--acquire",
        action="store_true",
        help="refresh all catalog pages from Wikisource before processing",
    )
    args = parser.parse_args()
    result = build(args)
    coverage = result["interaction_manifest"]["coverage"]
    print(f"Included works: {sum(w['included'] for w in result['processed_manifest']['works'])}")
    print(f"Interactions: {coverage['interaction_count']}")
    print(f"Output: {args.interaction_dir}")


if __name__ == "__main__":
    main()
