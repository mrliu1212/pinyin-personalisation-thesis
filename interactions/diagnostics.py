"""Print Phase 4B interaction-construction and Base coverage diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/interactions/zhu_ziqing/manifest.json"),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    coverage = manifest["coverage"]
    total = coverage["interaction_count"]
    print(f"Author: {manifest['author_name']} ({manifest['author_id']})")
    print(f"Eligible targets before optional limit: {manifest['eligible_targets_before_optional_limit']}")
    print(f"Successfully converted interactions: {total}")
    print(f"Pinyin conversion failures: {manifest['pinyin_conversion_failures']}")
    for k in (1, 3, 5, 10):
        print(
            f"Base Top-{k}: {coverage[f'target_present_top_{k}']} "
            f"({coverage[f'coverage_top_{k}']:.2%})"
        )
    print(
        f"Target absent from Top-{manifest['candidate_generator']['maximum_k']}: "
        f"{coverage['target_absent']} ({coverage['missing_target_rate']:.2%})"
    )
    print(f"Mean candidate list size: {coverage['mean_candidate_list_size']:.2f}")
    print(f"Median candidate list size: {coverage['median_candidate_list_size']:.2f}")
    print(f"Polyphonic review flags: {coverage['polyphonic_review_count']}")
    print(f"Interactions by work: {coverage['interactions_by_work']}")
    print(f"Interactions by target length: {coverage['interactions_by_target_length']}")
    print(f"Excluded tokens: {manifest['excluded_token_counts']}")


if __name__ == "__main__":
    main()

