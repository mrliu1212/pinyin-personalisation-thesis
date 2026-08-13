"""Deterministic preparation of the Deep Author contextual Pinyin corpus."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

import hanzidentifier
import jieba
from pypinyin import Style, lazy_pinyin
import requests


ROOT = Path(__file__).resolve().parents[3]
AUTHOR_CONFIG = ROOT / "config/deep_author/authors_v1.json"
RAW_ROOT = ROOT / "data/raw/deep_author"
PROCESSED_ROOT = ROOT / "data/processed/deep_author"
MANIFEST_ROOT = ROOT / "data/manifests"
AUDIT_ROOT = ROOT / "results/audits/deep_author_dataset_v1"
API_BASE = "https://scpper.mer.run/api"
SEED = 40408
CONTEXT_CHARACTER_LIMIT = 512

HARD_SENTENCE_END = frozenset("。！？!?；;")
EXCLUDED_TAGS = frozenset(
    {
        "中心",
        "作者",
        "版式",
        "管理",
        "指导",
        "段落",
        "合作",
        "合著",
        "艺术作品",
        "图像",
    }
)
TRANSLATION_TAGS = frozenset({"翻译", "外语", "译文"})
BOILERPLATE_LINES = frozenset(
    {
        "Loading...",
        "进行模因疫苗接种",
        "评分模块",
        "页面版本",
    }
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def author_id(name: str, wikidot_id: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")
    return f"da-author-{slug}-{stable_hash(wikidot_id, name)[:8]}"


def work_id(wikidot_id: int) -> str:
    return f"da-work-{int(wikidot_id)}"


def interaction_id(work: str, start: int, end: int, composition_type: str) -> str:
    return "da-int-" + stable_hash(work, start, end, composition_type)[:24]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_han(character: str) -> bool:
    code = ord(character)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x3134F
    )


def han_count(text: str) -> int:
    return sum(is_han(character) for character in text)


def clean_text(text: str) -> str:
    """Apply conservative formatting cleanup without rewriting prose."""

    value = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", value):
        lines = []
        for line in block.splitlines():
            line = re.sub(r"[\t\u00a0\u3000 ]+", " ", line).strip()
            if not line or line in BOILERPLATE_LINES or re.fullmatch(r"[.·•]{3,}", line):
                continue
            lines.append(line)
        paragraph = "\n".join(lines).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs).strip() + ("\n" if paragraphs else "")


def full_pinyin(text: str) -> list[str]:
    if not text or not all(is_han(character) for character in text):
        raise ValueError("target must contain Han characters only")
    syllables = [
        value.lower()
        for value in lazy_pinyin(
            text,
            style=Style.NORMAL,
            strict=True,
            errors=lambda value: list(value),
        )
    ]
    if len(syllables) != len(text) or any(not re.fullmatch(r"[a-z]+", item) for item in syllables):
        raise ValueError("Chinese-character/Pinyin-syllable alignment failure")
    return syllables


def initial_pinyin(syllables: Sequence[str]) -> list[str]:
    if not syllables or any(not value for value in syllables):
        raise ValueError("empty Pinyin syllable")
    return [value[0].lower() for value in syllables]


def sentence_count(text: str) -> int:
    return sum(1 for part in re.split(r"(?<=[。！？!?；;])", text) if part.strip())


def script_label(text: str) -> str:
    return {
        hanzidentifier.SIMPLIFIED: "simplified",
        hanzidentifier.TRADITIONAL: "traditional",
        hanzidentifier.BOTH: "shared_or_ambiguous",
        hanzidentifier.MIXED: "mixed",
        hanzidentifier.UNKNOWN: "no_identifiable_chinese",
    }[hanzidentifier.identify(text)]


def segment_text(text: str) -> list[dict[str, Any]]:
    return [
        {
            "text": token,
            "token_index": index,
            "start": start,
            "end": end,
            "is_han": bool(token) and all(is_han(character) for character in token),
        }
        for index, (token, start, end) in enumerate(jieba.tokenize(text, mode="default"))
    ]


def preliminary_exclusion_reason(
    discovered: Mapping[str, Any], window_start: str, window_end: str
) -> str:
    tags = set(discovered.get("tags") or [])
    creation_date = str(discovered.get("firstRevisionAt") or "")[:10]
    slug = str(discovered.get("url", "")).split("/")[-1]
    if not (window_start <= creation_date <= window_end):
        return "out_of_window"
    if "原创" not in tags:
        return "not_marked_chinese_original"
    if tags & TRANSLATION_TAGS:
        return "translation"
    if tags & EXCLUDED_TAGS:
        return "structural_or_non_work_page"
    if slug.startswith("fragment:"):
        return "fragment_not_independent_work"
    if slug.startswith("component:"):
        return "structural_or_non_work_page"
    return ""


def attribution_exclusion_reason(
    attributions: Sequence[Mapping[str, Any]],
    selected_author_names: set[str],
    discovered_for: Sequence[str],
) -> str:
    submitters = [item for item in attributions if item.get("type") == "SUBMITTER"]
    selected = [item for item in submitters if item.get("displayName") in selected_author_names]
    if len(submitters) != 1 or len(selected) != 1:
        return "coauthored_or_unclear_attribution"
    if selected[0].get("displayName") not in discovered_for:
        return "attribution_mismatch"
    return ""


def _hard_boundary_between(text: str, start: int, end: int) -> bool:
    value = text[start:end]
    return "\n" in value or any(character in HARD_SENTENCE_END for character in value)


def make_interactions(work: Mapping[str, Any], tokens: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    text = work["cleaned_text"]
    interactions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        if not token["is_han"]:
            continue
        start, end = int(token["start"]), int(token["end"])
        if start <= 0:
            continue
        try:
            syllables = full_pinyin(token["text"])
        except ValueError as error:
            failures.append({"work_id": work["work_id"], "token_index": index, "text": token["text"], "reason": str(error)})
            continue
        base = {
            "author_id": work["author_id"],
            "author_name": work["author_name"],
            "work_id": work["work_id"],
            "work_title": work["page_title"],
            "context_source_position_start": max(0, start - CONTEXT_CHARACTER_LIMIT),
            "context": text[max(0, start - CONTEXT_CHARACTER_LIMIT) : start],
            "source_creation_date": work["creation_date"],
            "source_hash": work["SHA256"],
        }
        short = {
            **base,
            "interaction_id": interaction_id(work["work_id"], start, end, "short"),
            "source_position_start": start,
            "source_position_end": end,
            "gold": token["text"],
            "full_pinyin": " ".join(syllables),
            "initial_pinyin": " ".join(initial_pinyin(syllables)),
            "composition_type": "short",
            "token_count": 1,
            "gold_char_length": len(token["text"]),
        }
        interactions.append(short)

        merged = [token]
        previous_end = end
        for next_token in tokens[index + 1 :]:
            if len(merged) >= 4:
                break
            next_start, next_end = int(next_token["start"]), int(next_token["end"])
            if _hard_boundary_between(text, previous_end, next_start) or not next_token["is_han"]:
                break
            merged.append(next_token)
            previous_end = next_end
        if len(merged) < 2:
            continue
        multi_end = int(merged[-1]["end"])
        gold = text[start:multi_end]
        if not all(is_han(character) for character in gold):
            continue
        try:
            multi_syllables = full_pinyin(gold)
        except ValueError as error:
            failures.append({"work_id": work["work_id"], "token_index": index, "text": gold, "reason": str(error)})
            continue
        interactions.append(
            {
                **base,
                "interaction_id": interaction_id(work["work_id"], start, multi_end, "multi"),
                "source_position_start": start,
                "source_position_end": multi_end,
                "gold": gold,
                "full_pinyin": " ".join(multi_syllables),
                "initial_pinyin": " ".join(initial_pinyin(multi_syllables)),
                "composition_type": "multi",
                "token_count": len(merged),
                "merged_token_count": len(merged),
                "gold_char_length": len(gold),
            }
        )
    return interactions, failures


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(canonical_json(row) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class DeepAuthorBuilder:
    root: Path = ROOT
    api_base: str = API_BASE
    access_date: str = date.today().isoformat()

    def __post_init__(self) -> None:
        self.raw_root = self.root / "data/raw/deep_author"
        self.processed_root = self.root / "data/processed/deep_author"
        self.manifest_root = self.root / "data/manifests"
        self.audit_root = self.root / "results/audits/deep_author_dataset_v1"
        self.config = json.loads((self.root / "config/deep_author/authors_v1.json").read_text(encoding="utf-8"))
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "DeepAuthorDatasetV1/1.0 (academic reproducibility)"

    def _get(self, endpoint: str, **params: Any) -> Any:
        response = self.session.get(f"{self.api_base}/{endpoint.lstrip('/')}", params=params, timeout=60)
        response.raise_for_status()
        return response.json()

    def _preserve_raw(self, relative: Path, value: Any) -> tuple[Path, str, int]:
        path = self.raw_root / relative
        payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if path.exists():
            current = path.read_bytes()
            if current != payload:
                raise RuntimeError(f"immutable raw file differs: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        return path, sha256_bytes(payload), len(payload)

    def discover(self) -> list[dict[str, Any]]:
        discoveries: dict[int, dict[str, Any]] = {}
        for author in self.config["authors"]:
            relative = Path("discoveries") / f"{author_id(author['name'], author['wikidot_id'])}.json"
            snapshot = self.raw_root / relative
            if snapshot.exists():
                author_rows = json.loads(snapshot.read_text(encoding="utf-8"))
            else:
                offset = 0
                author_rows = []
                while True:
                    result = self._get(
                        "search/pages",
                        authorIds=str(author["wikidot_id"]),
                        dateMin=self.config["window_start"],
                        dateMax=self.config["window_end"],
                        deletedFilter="exclude",
                        orderBy="recent_asc",
                        limit=100,
                        offset=offset,
                        includeTotal="true",
                        includeSnippet="true",
                        includeDate="true",
                    )
                    rows = result.get("results", [])
                    author_rows.extend(rows)
                    offset += len(rows)
                    if not rows or offset >= int(result.get("total", offset)):
                        break
                self._preserve_raw(relative, author_rows)
            for row in author_rows:
                page_key = int(row["wikidotId"])
                record = discoveries.setdefault(page_key, {**row, "discovered_for": []})
                record["discovered_for"].append(author["name"])
        return [discoveries[key] for key in sorted(discoveries)]

    def acquire(self, discoveries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        author_by_name = {item["name"]: item for item in self.config["authors"]}
        for index, discovered in enumerate(discoveries, 1):
            wid = int(discovered["wikidotId"])
            raw_relative = Path("works") / f"{work_id(wid)}.json"
            cached_raw_path = self.raw_root / raw_relative
            tags = set(discovered.get("tags") or [])
            creation_date = str(discovered.get("firstRevisionAt") or "")[:10]
            preliminary_reason = preliminary_exclusion_reason(
                discovered, self.config["window_start"], self.config["window_end"]
            )

            try:
                if cached_raw_path.exists():
                    raw_path = cached_raw_path
                    raw_bytes = raw_path.read_bytes()
                    payload = json.loads(raw_bytes.decode("utf-8"))
                    raw_sha, raw_size = sha256_bytes(raw_bytes), len(raw_bytes)
                else:
                    payload = {"search_metadata": discovered}
                    if not preliminary_reason:
                        payload.update(
                            {
                                "page": self._get("pages/by-id", wikidotId=wid),
                                "attributions": self._get(f"pages/{wid}/attributions"),
                            }
                        )
                    raw_path, raw_sha, raw_size = self._preserve_raw(raw_relative, payload)
                error = ""
            except Exception as exc:  # acquisition failures must survive in the manifest
                payload = None
                raw_path = self.raw_root / raw_relative
                raw_sha, raw_size = "", 0
                error = f"{type(exc).__name__}: {exc}"

            target_names = list(discovered.get("discovered_for") or [])
            inclusion = "included"
            reason = ""
            attributions: list[Mapping[str, Any]] = []
            if error:
                inclusion, reason = "excluded", "acquisition_failed"
            elif preliminary_reason:
                inclusion, reason = "excluded", preliminary_reason
            else:
                raw_attributions = payload["attributions"]
                attributions = raw_attributions if isinstance(raw_attributions, list) else [raw_attributions]
                reason = attribution_exclusion_reason(
                    attributions, set(author_by_name), target_names
                )
                if reason:
                    inclusion = "excluded"

            author_name = ""
            if payload and inclusion == "included":
                author_name = next(item["displayName"] for item in attributions if item.get("type") == "SUBMITTER")
            elif target_names:
                author_name = target_names[0]
            author = author_by_name.get(author_name, {"name": author_name, "wikidot_id": 0})
            records.append(
                {
                    "author_id": author_id(author["name"], author["wikidot_id"]) if author["name"] else "",
                    "author_name": author["name"],
                    "work_id": work_id(wid),
                    "wikidot_id": wid,
                    "page_title": discovered.get("alternateTitle") or discovered.get("title") or "",
                    "source_url": str(discovered.get("url", "")).replace("http://", "https://"),
                    "original_source_url": str(discovered.get("url", "")).replace("http://", "https://"),
                    "source_site": "SCP-CN via SCPPER-CN",
                    "creation_date": creation_date,
                    "modification_date": "",
                    "eligible_date_flag": self.config["window_start"] <= creation_date <= self.config["window_end"],
                    "authorship_status": "single_selected_author" if inclusion == "included" else "excluded_or_unverified",
                    "language": "zh",
                    "translation_flag": bool(tags & TRANSLATION_TAGS) or "原创" not in tags,
                    "coauthor_flag": bool(payload and len([item for item in attributions if item.get("type") == "SUBMITTER"]) != 1),
                    "inclusion_status": inclusion,
                    "exclusion_reason": reason,
                    "raw_filename": raw_path.relative_to(self.root).as_posix(),
                    "raw_format": "SCPPER-CN JSON bundle",
                    "raw_byte_size": raw_size,
                    "SHA256": raw_sha,
                    "access_date": self.access_date,
                    "license_note": self.config["license"],
                    "tags": "|".join(sorted(tags)),
                    "acquisition_error": error,
                }
            )
            if index % 50 == 0:
                print(f"acquired {index}/{len(discoveries)}", flush=True)
        return records

    def process(self, records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        works: list[dict[str, Any]] = []
        interactions: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for record in records:
            if record["inclusion_status"] != "included":
                continue
            raw_path = self.root / record["raw_filename"]
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            extracted = str(payload["page"].get("textContent") or "")
            cleaned = clean_text(extracted)
            if not cleaned or han_count(cleaned) < 20:
                record["inclusion_status"] = "excluded"
                record["exclusion_reason"] = "insufficient_extracted_chinese_text"
                continue
            tokens = segment_text(cleaned)
            work = {
                **record,
                "cleaned_text": cleaned,
                "raw_character_count": len(extracted),
                "cleaned_character_count": len(cleaned),
                "han_character_count": han_count(cleaned),
                "removed_character_count": max(0, len(extracted) - len(cleaned)),
                "paragraph_count": len([value for value in cleaned.split("\n\n") if value.strip()]),
                "sentence_count": sentence_count(cleaned),
                "segmented_token_count": len(tokens),
                "script_distribution": script_label(cleaned),
                "extraction_method": "SCPPER-CN text-content endpoint; NFC/whitespace/known UI cleanup",
                "cleaned_sha256": sha256_bytes(cleaned.encode("utf-8")),
            }
            work_interactions, work_failures = make_interactions(work, tokens)
            short_count = sum(row["composition_type"] == "short" for row in work_interactions)
            multi_count = sum(row["composition_type"] == "multi" for row in work_interactions)
            work["short_interaction_count"] = short_count
            work["multi_interaction_count"] = multi_count
            work["full_pinyin_success_count"] = len(work_interactions)
            work["initial_pinyin_success_count"] = len(work_interactions)
            work["alignment_failure_count"] = len(work_failures)
            work_path = self.processed_root / "works" / f"{work['work_id']}.json"
            token_path = self.processed_root / "works" / f"{work['work_id']}.tokens.jsonl"
            serializable_work = {key: value for key, value in work.items() if key != "cleaned_text"}
            serializable_work["cleaned_text"] = cleaned
            _write_json(work_path, serializable_work)
            _write_jsonl(token_path, tokens)
            works.append(work)
            interactions.extend(work_interactions)
            failures.extend(work_failures)
        interactions.sort(key=lambda row: (row["author_id"], row["work_id"], row["source_position_start"], row["composition_type"]))
        retained = {work["work_id"] for work in works}
        work_root = self.processed_root / "works"
        if work_root.exists():
            for path in work_root.iterdir():
                stem = path.name.split(".tokens.jsonl", 1)[0].split(".json", 1)[0]
                if path.is_file() and stem.startswith("da-work-") and stem not in retained:
                    path.unlink()
        _write_jsonl(self.processed_root / "interactions_t1_ready.jsonl", interactions)
        return works, interactions, failures

    def _manifest_rows(self, records: Sequence[Mapping[str, Any]]) -> None:
        columns = [
            "author_id", "author_name", "work_id", "wikidot_id", "page_title", "source_url",
            "original_source_url", "source_site", "creation_date", "modification_date", "eligible_date_flag",
            "authorship_status", "language", "translation_flag", "coauthor_flag", "inclusion_status",
            "exclusion_reason", "raw_filename", "raw_format", "raw_byte_size", "SHA256", "access_date",
            "license_note", "tags", "acquisition_error",
        ]
        _write_csv(self.manifest_root / "deep_author_works_manifest.csv", records, columns)
        author_rows = []
        for author in self.config["authors"]:
            author_rows.append(
                {
                    "author_id": author_id(author["name"], author["wikidot_id"]),
                    "author_name": author["name"],
                    "wikidot_id": author["wikidot_id"],
                    "role": "primary",
                    "source_profile": f"https://scpper.mer.run/user/{author['wikidot_id']}",
                    "access_date": self.access_date,
                }
            )
        for author in self.config["reserve_authors"]:
            author_rows.append({"author_id": "", "author_name": author["name"], "wikidot_id": "", "role": "reserve_not_used", "source_profile": "", "access_date": self.access_date})
        _write_csv(self.manifest_root / "deep_author_authors_manifest.csv", author_rows, list(author_rows[0]))

    def audit(self, records: Sequence[Mapping[str, Any]], works: Sequence[Mapping[str, Any]], interactions: Sequence[Mapping[str, Any]], failures: Sequence[Mapping[str, Any]]) -> None:
        self._manifest_rows(records)
        per_work = []
        for work in works:
            per_work.append({key: work[key] for key in (
                "author_id", "author_name", "work_id", "page_title", "creation_date", "raw_character_count",
                "cleaned_character_count", "han_character_count", "removed_character_count", "paragraph_count",
                "sentence_count", "segmented_token_count", "short_interaction_count", "multi_interaction_count",
                "full_pinyin_success_count", "initial_pinyin_success_count", "alignment_failure_count", "script_distribution",
            )})
        _write_csv(self.audit_root / "corpus_statistics.csv", per_work, list(per_work[0]) if per_work else [])

        work_by_author: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for work in works:
            work_by_author[work["author_name"]].append(work)
        author_stats = []
        for author in self.config["authors"]:
            name = author["name"]
            selected = work_by_author[name]
            discovered = [row for row in records if row["author_name"] == name]
            row = {
                "author_id": author_id(name, author["wikidot_id"]),
                "author_name": name,
                "included_work_count": len(selected),
                "excluded_work_count": len(discovered) - len(selected),
                "total_cleaned_characters": sum(item["cleaned_character_count"] for item in selected),
                "han_characters": sum(item["han_character_count"] for item in selected),
                "paragraphs": sum(item["paragraph_count"] for item in selected),
                "sentences": sum(item["sentence_count"] for item in selected),
                "segmented_tokens": sum(item["segmented_token_count"] for item in selected),
                "short_interactions": sum(item["short_interaction_count"] for item in selected),
                "multi_interactions": sum(item["multi_interaction_count"] for item in selected),
                "full_pinyin_success": sum(item["full_pinyin_success_count"] for item in selected),
                "initial_pinyin_success": sum(item["initial_pinyin_success_count"] for item in selected),
                "alignment_failures": sum(item["alignment_failure_count"] for item in selected),
                "depth_concern": "review_required" if len(selected) < 5 or sum(item["han_character_count"] for item in selected) < 10000 else "none_observed",
            }
            author_stats.append(row)
        _write_csv(self.audit_root / "author_statistics.csv", author_stats, list(author_stats[0]))
        _write_json(self.audit_root / "interaction_statistics.json", {
            "total": len(interactions),
            "short": sum(row["composition_type"] == "short" for row in interactions),
            "multi": sum(row["composition_type"] == "multi" for row in interactions),
            "authors": len({row["author_id"] for row in interactions}),
            "works": len({row["work_id"] for row in interactions}),
        })
        _write_json(self.audit_root / "acquisition_summary.json", {
            "discovered": len(records), "included": len(works), "excluded": len(records) - len(works),
            "failed": sum(bool(row["acquisition_error"]) for row in records),
            "exclusion_reasons": Counter(row["exclusion_reason"] for row in records if row["exclusion_reason"]),
        })
        _write_json(self.audit_root / "cleaning_summary.json", {
            "raw_characters": sum(row["raw_character_count"] for row in works),
            "cleaned_characters": sum(row["cleaned_character_count"] for row in works),
            "han_characters": sum(row["han_character_count"] for row in works),
            "removed_characters": sum(row["removed_character_count"] for row in works),
            "script_distribution": Counter(row["script_distribution"] for row in works),
            "traditional_conversion_applied": False,
        })
        raw_groups: dict[str, list[str]] = defaultdict(list)
        clean_groups: dict[str, list[str]] = defaultdict(list)
        for work in works:
            raw_groups[work["SHA256"]].append(work["work_id"])
            clean_groups[work["cleaned_sha256"]].append(work["work_id"])
        _write_json(self.audit_root / "duplicate_audit.json", {
            "exact_raw_duplicates": [value for value in raw_groups.values() if len(value) > 1],
            "exact_cleaned_text_duplicates": [value for value in clean_groups.values() if len(value) > 1],
            "duplicate_work_ids": [key for key, count in Counter(row["work_id"] for row in works).items() if count > 1],
            "repeated_sections_removed": False,
            "note": "Normal literary repetition was not removed.",
        })
        failure_columns = ["work_id", "token_index", "text", "reason"]
        _write_csv(self.audit_root / "alignment_failures.csv", failures, failure_columns)

        samples = []
        for author in self.config["authors"]:
            name = author["name"]
            author_works = sorted(work_by_author[name], key=lambda row: row["work_id"])
            author_interactions = [row for row in interactions if row["author_name"] == name]
            for work in author_works[:2]:
                samples.append({"author": name, "work": work["page_title"], "source_excerpt": work["cleaned_text"][:180], "context": "", "gold": "", "full_pinyin": "", "initial_pinyin": "", "composition_type": "cleaned_text", "text_clean": "", "segmentation_ok": "", "pinyin_ok": "", "composition_ok": "", "notes": ""})
            for kind in ("short", "multi"):
                choices = [row for row in author_interactions if row["composition_type"] == kind][:3]
                for row in choices:
                    samples.append({"author": name, "work": row["work_title"], "source_excerpt": (row["context"][-80:] + row["gold"])[:180], "context": row["context"][-120:], "gold": row["gold"], "full_pinyin": row["full_pinyin"], "initial_pinyin": row["initial_pinyin"], "composition_type": kind, "text_clean": "", "segmentation_ok": "", "pinyin_ok": "", "composition_ok": "", "notes": ""})
        _write_csv(self.audit_root / "manual_review.csv", samples, list(samples[0]) if samples else [])

        checksummed = [
            self.manifest_root / "deep_author_authors_manifest.csv",
            self.manifest_root / "deep_author_works_manifest.csv",
            self.processed_root / "interactions_t1_ready.jsonl",
            self.audit_root / "author_statistics.csv",
            self.audit_root / "corpus_statistics.csv",
            self.audit_root / "interaction_statistics.json",
        ]
        checksums = {path.relative_to(self.root).as_posix(): {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in checksummed}
        _write_json(self.audit_root / "checksums.json", checksums)
        _write_json(self.audit_root / "manifest.json", {
            "dataset": "Deep Author Dataset V1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "authors": [item["name"] for item in self.config["authors"]],
            "window": [self.config["window_start"], self.config["window_end"]],
            "source": self.config["source"],
            "license": self.config["license"],
            "model_inference": False,
            "personalisation": False,
            "random_seed": SEED,
            "context_character_limit": CONTEXT_CHARACTER_LIMIT,
            "source_git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.root, text=True
            ).strip(),
            "run_command": ".\\.venv\\Scripts\\python.exe -m experiments.prepare_deep_author_dataset",
            "dependencies": {name: importlib.metadata.version(name) for name in ("requests", "jieba", "pypinyin", "hanzidentifier")},
            "outputs": checksums,
        })

    def run(self) -> dict[str, Any]:
        jieba.initialize()
        discoveries = self.discover()
        records = self.acquire(discoveries)
        works, interactions, failures = self.process(records)
        self.audit(records, works, interactions, failures)
        return {"discovered": len(records), "included": len(works), "interactions": len(interactions), "failures": len(failures)}
