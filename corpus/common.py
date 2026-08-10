"""Shared metadata and chronology helpers for the author corpus pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_PRECISIONS = {"day", "month", "year", "unknown"}
VALID_CERTAINTIES = {"certain", "uncertain", "unknown"}


@dataclass(frozen=True)
class DateMetadata:
    value: str | None
    precision: str
    certainty: str
    basis: str


@dataclass(frozen=True)
class WorkMetadata:
    author_id: str
    author_name: str
    work_id: str
    work_title: str
    source_page_title: str
    source_page_url: str
    genre: str
    included: bool
    exclusion_reason: str | None
    chronology: DateMetadata
    publication: DateMetadata


@dataclass(frozen=True)
class CorpusCatalog:
    schema_version: int
    author_id: str
    author_name: str
    source_name: str
    source_api: str
    source_author_page: str
    works: tuple[WorkMetadata, ...]


def _parse_date_metadata(value: Any, field_name: str) -> DateMetadata:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    metadata = DateMetadata(
        value=value.get("value"),
        precision=value.get("precision", "unknown"),
        certainty=value.get("certainty", "unknown"),
        basis=value.get("basis", ""),
    )
    if metadata.precision not in VALID_PRECISIONS:
        raise ValueError(f"invalid {field_name} precision: {metadata.precision}")
    if metadata.certainty not in VALID_CERTAINTIES:
        raise ValueError(f"invalid {field_name} certainty: {metadata.certainty}")
    if metadata.value is None and metadata.precision != "unknown":
        raise ValueError(f"missing {field_name} value must have unknown precision")
    if metadata.value is not None:
        parts = metadata.value.split("-")
        expected = {"year": 1, "month": 2, "day": 3}.get(metadata.precision)
        if expected is None or len(parts) != expected or not all(part.isdigit() for part in parts):
            raise ValueError(
                f"{field_name} value {metadata.value!r} does not match "
                f"precision {metadata.precision!r}"
            )
    return metadata


def load_catalog(path: Path) -> CorpusCatalog:
    payload = json.loads(path.read_text(encoding="utf-8"))
    author_id = payload["author_id"]
    author_name = payload["author_name"]
    works: list[WorkMetadata] = []
    seen_ids: set[str] = set()
    for raw in payload["works"]:
        work_id = raw["work_id"]
        if work_id in seen_ids:
            raise ValueError(f"duplicate work_id: {work_id}")
        seen_ids.add(work_id)
        included = raw["included"]
        exclusion_reason = raw.get("exclusion_reason")
        if included and exclusion_reason is not None:
            raise ValueError(f"included work {work_id} must not have an exclusion reason")
        if not included and not exclusion_reason:
            raise ValueError(f"excluded work {work_id} requires an exclusion reason")
        works.append(
            WorkMetadata(
                author_id=author_id,
                author_name=author_name,
                work_id=work_id,
                work_title=raw["work_title"],
                source_page_title=raw["source_page_title"],
                source_page_url=raw["source_page_url"],
                genre=raw["genre"],
                included=included,
                exclusion_reason=exclusion_reason,
                chronology=_parse_date_metadata(raw["chronology"], "chronology"),
                publication=_parse_date_metadata(raw["publication"], "publication"),
            )
        )
    return CorpusCatalog(
        schema_version=payload["schema_version"],
        author_id=author_id,
        author_name=author_name,
        source_name=payload["source_name"],
        source_api=payload["source_api"],
        source_author_page=payload["source_author_page"],
        works=tuple(works),
    )


def chronology_sort_key(work: WorkMetadata) -> tuple[int, int, int, int, str]:
    date = work.chronology
    if date.value is None:
        return (1, 9999, 12, 31, work.work_id)
    parts = [int(part) for part in date.value.split("-")]
    year = parts[0]
    month = parts[1] if len(parts) > 1 else 0
    day = parts[2] if len(parts) > 2 else 0
    uncertain = int(date.certainty != "certain")
    return (uncertain, year, month, day, work.work_id)


def date_needs_review(date: DateMetadata) -> bool:
    return date.value is None or date.certainty != "certain"

