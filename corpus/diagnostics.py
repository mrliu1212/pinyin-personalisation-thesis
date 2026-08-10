"""Print corpus-level Phase 4A diagnostics from the processed manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def diagnostic_lines(manifest: dict[str, object]) -> list[str]:
    works = manifest["works"]
    included = [item for item in works if item["included"]]
    excluded = [item for item in works if not item["included"]]
    total_characters = sum(item["chinese_character_count"] for item in included)
    review = [
        item
        for item in included
        if item["chronology_needs_review"] or item["publication_needs_review"]
    ]
    lines = [
        f"Author: {manifest['author_name']} ({manifest['author_id']})",
        f"Included works: {len(included)}",
        f"Excluded works: {len(excluded)}",
        f"Total Chinese characters: {total_characters}",
        "Chronological included works:",
    ]
    for index, item in enumerate(included, start=1):
        chronology = item["chronology"]
        lines.append(
            f"  {index}. {chronology['value'] or 'unknown'} "
            f"[{chronology['precision']}, {chronology['certainty']}] "
            f"{item['work_title']} — {item['chinese_character_count']} chars"
        )
    lines.append("Excluded works:")
    for item in excluded:
        lines.append(f"  - {item['work_title']}: {item['exclusion_reason']}")
    lines.append("Missing/uncertain date metadata:")
    if review:
        for item in review:
            fields = []
            if item["chronology_needs_review"]:
                fields.append(f"chronology={item['chronology']}")
            if item["publication_needs_review"]:
                fields.append(f"publication={item['publication']}")
            lines.append(f"  - {item['work_title']}: {'; '.join(fields)}")
    else:
        lines.append("  none")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/authors/zhu_ziqing/manifest.json"),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    print("\n".join(diagnostic_lines(manifest)))


if __name__ == "__main__":
    main()
