"""Prepare cleaned prose and a chronological manifest without changing raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .cleaning import clean_wikisource_html, count_chinese_characters
from .common import chronology_sort_key, date_needs_review, load_catalog


def prepare(catalog_path: Path, raw_dir: Path, processed_dir: Path) -> dict[str, object]:
    catalog = load_catalog(catalog_path)
    acquisition_path = raw_dir / "acquisition_manifest.json"
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    acquired_by_id = {item["work_id"]: item for item in acquisition["works"]}
    processed_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for work in sorted(catalog.works, key=chronology_sort_key):
        acquired = acquired_by_id[work.work_id]
        raw_path = raw_dir / acquired["raw_file"]
        raw_before = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        if raw_before != acquired["raw_sha256"]:
            raise RuntimeError(f"raw source checksum mismatch for {work.work_id}")
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        record: dict[str, object] = {
            "author_id": work.author_id,
            "author_name": work.author_name,
            "work_id": work.work_id,
            "work_title": work.work_title,
            "genre": work.genre,
            "included": work.included,
            "exclusion_reason": work.exclusion_reason,
            "chronology": {
                "value": work.chronology.value,
                "precision": work.chronology.precision,
                "certainty": work.chronology.certainty,
                "basis": work.chronology.basis,
            },
            "publication": {
                "value": work.publication.value,
                "precision": work.publication.precision,
                "certainty": work.publication.certainty,
                "basis": work.publication.basis,
            },
            "chronology_needs_review": date_needs_review(work.chronology),
            "publication_needs_review": date_needs_review(work.publication),
            "source_page_title": acquired["source_page_title"],
            "source_page_url": acquired["source_page_url"],
            "source_page_id": acquired["page_id"],
            "source_revision_id": acquired["revision_id"],
            "source_request_url": acquired["request_url"],
            "retrieved_at": acquired["retrieved_at"],
            "content_variant": acquired["content_variant"],
            "raw_file": acquired["raw_file"],
            "raw_sha256": acquired["raw_sha256"],
            "processed_file": None,
            "processed_sha256": None,
            "chinese_character_count": 0,
        }
        if work.included:
            cleaned = clean_wikisource_html(payload["parse"]["text"])
            processed_filename = f"{work.work_id}.txt"
            (processed_dir / processed_filename).write_text(cleaned, encoding="utf-8")
            record["processed_file"] = processed_filename
            record["processed_sha256"] = hashlib.sha256(
                cleaned.encode("utf-8")
            ).hexdigest()
            record["chinese_character_count"] = count_chinese_characters(cleaned)
        raw_after = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        if raw_before != raw_after:
            raise RuntimeError(f"raw source changed while processing {work.work_id}")
        records.append(record)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "author_id": catalog.author_id,
        "author_name": catalog.author_name,
        "source_name": catalog.source_name,
        "source_api": catalog.source_api,
        "works": records,
    }
    (processed_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/manifests/zhu_ziqing_works.json"),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/authors/zhu_ziqing"),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/authors/zhu_ziqing"),
    )
    args = parser.parse_args()
    manifest = prepare(args.catalog, args.raw_dir, args.processed_dir)
    included = sum(item["included"] for item in manifest["works"])
    print(f"Prepared {included} included works.")
    print(f"Processed directory: {args.processed_dir}")


if __name__ == "__main__":
    main()
