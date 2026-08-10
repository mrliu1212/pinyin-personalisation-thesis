"""Acquire versioned Zhu Ziqing pages through the MediaWiki Action API."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .common import load_catalog


USER_AGENT = "PinyinPersonalisationThesis/0.1 (Phase 4A corpus research)"


def build_api_url(api_url: str, page_title: str) -> str:
    query = urlencode(
        {
            "action": "parse",
            "page": page_title,
            "prop": "text|wikitext|revid",
            "format": "json",
            "formatversion": "2",
        }
    )
    return f"{api_url}?{query}"


def fetch_page(api_url: str, page_title: str, timeout: float = 30.0) -> tuple[bytes, str]:
    request_url = build_api_url(api_url, page_title)
    request = Request(request_url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read(), request_url


def _save_payloads(
    catalog_path: Path,
    raw_dir: Path,
    payloads: list[tuple[bytes, str]],
) -> dict[str, object]:
    catalog = load_catalog(catalog_path)
    if len(payloads) != len(catalog.works):
        raise ValueError("one API payload is required for every catalog work")
    raw_dir.mkdir(parents=True, exist_ok=True)
    retrieval_time = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, object]] = []

    for work, (payload_bytes, request_url) in zip(catalog.works, payloads):
        payload = json.loads(payload_bytes.decode("utf-8"))
        if "error" in payload:
            raise RuntimeError(f"MediaWiki error for {work.work_id}: {payload['error']}")
        parsed = payload["parse"]
        revision_id = int(parsed["revid"])
        raw_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        raw_filename = (
            f"{work.work_id}__rev_{revision_id}__sha256_{raw_sha256[:16]}.json"
        )
        raw_path = raw_dir / raw_filename
        if not raw_path.exists():
            raw_path.write_bytes(payload_bytes)
        records.append(
            {
                "work_id": work.work_id,
                "work_title": work.work_title,
                "included": work.included,
                "source_page_title": parsed["title"],
                "source_page_url": work.source_page_url,
                "request_url": request_url,
                "page_id": int(parsed["pageid"]),
                "revision_id": revision_id,
                "retrieved_at": retrieval_time,
                "content_variant": "canonical",
                "raw_file": raw_filename,
                "raw_sha256": raw_sha256,
            }
        )

    manifest: dict[str, object] = {
        "schema_version": 1,
        "author_id": catalog.author_id,
        "author_name": catalog.author_name,
        "source_name": catalog.source_name,
        "source_api": catalog.source_api,
        "retrieved_at": retrieval_time,
        "works": records,
    }
    (raw_dir / "acquisition_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def acquire(catalog_path: Path, raw_dir: Path) -> dict[str, object]:
    catalog = load_catalog(catalog_path)
    payloads = [
        fetch_page(catalog.source_api, work.source_page_title)
        for work in catalog.works
    ]
    return _save_payloads(catalog_path, raw_dir, payloads)


def import_downloads(
    catalog_path: Path, raw_dir: Path, download_dir: Path
) -> dict[str, object]:
    """Import API responses named ``<work_id>.json`` from an offline download."""
    catalog = load_catalog(catalog_path)
    payloads = [
        (
            (download_dir / f"{work.work_id}.json").read_bytes(),
            build_api_url(catalog.source_api, work.source_page_title),
        )
        for work in catalog.works
    ]
    return _save_payloads(catalog_path, raw_dir, payloads)


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
        "--download-dir",
        type=Path,
        help="Import pre-downloaded <work_id>.json API responses instead of using the network.",
    )
    args = parser.parse_args()
    manifest = (
        import_downloads(args.catalog, args.raw_dir, args.download_dir)
        if args.download_dir
        else acquire(args.catalog, args.raw_dir)
    )
    print(f"Acquired {len(manifest['works'])} versioned Wikisource pages.")
    print(f"Raw directory: {args.raw_dir}")


if __name__ == "__main__":
    main()
