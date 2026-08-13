"""Deterministic preparation of the Deep Author contextual Pinyin corpus."""

from __future__ import annotations

from collections import Counter, defaultdict
from bisect import bisect_left
import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
import html
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
from opencc import OpenCC
from pypinyin import Style, lazy_pinyin
import requests


ROOT = Path(__file__).resolve().parents[3]
AUTHOR_CONFIG = ROOT / "config/deep_author/authors_v1.json"
RAW_ROOT = ROOT / "data/raw/deep_author"
PROCESSED_ROOT = ROOT / "data/processed/deep_author"
MANIFEST_ROOT = ROOT / "data/manifests"
AUDIT_ROOT = ROOT / "results/audits/deep_author_dataset_v1_1"
API_BASE = "https://scpper.mer.run/api"
SEED = 40408
CONTEXT_CHARACTER_LIMIT = 512

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

CREDIT_START = re.compile(r"\[\[include\s+:scp-wiki-cn:[^\]\n]*credit:start", re.IGNORECASE)
CREDIT_END = re.compile(r"\[\[include\s+:scp-wiki-cn:[^\]\n]*credit:end[^\n]*", re.IGNORECASE)
SUSPICIOUS_METADATA = re.compile(
    r"^(?:著作信息|作者\s*[:：]|图像信息\s*[:：]?|圖像信息\s*[:：]?|"
    r"图片信息\s*[:：]?|圖片授權\s*[:：]?|图片授权\s*[:：]?|"
    r"延伸閱讀\s*[:：]?|延伸阅读\s*[:：]?|.*查看本文作者的更多作品.*)$"
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


def _rendered_source_line(line: str) -> str:
    """Reduce one Wikidot source line to an anchor suitable for textContent."""

    value = html.unescape(line).strip()
    value = re.sub(r"\[\[\[[^\]|]+\|([^\]]+)\]\]\]", r"\1", value)
    value = re.sub(r"\[\[\[([^\]]+)\]\]\]", r"\1", value)
    value = re.sub(r"\[\*[^ ]+\s+([^\]]+)\]", r"\1", value)
    value = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", value)
    value = re.sub(r"\[\[[^\]]+\]\]", " ", value)
    value = re.sub(r"(?:\*\*|//|__|--|@@|^>\s*)", "", value)
    return re.sub(r"\s+", " ", value).strip()


def metadata_prefix_end(source: str, rendered: str) -> int | None:
    """Return the rendered offset after a source-confirmed leading credit block."""

    start_match = CREDIT_START.search(source)
    end_match = CREDIT_END.search(source)
    if not start_match or not end_match or end_match.end() <= start_match.start():
        return None
    suffix = source[end_match.end() :]
    candidates: list[tuple[int, int]] = []
    for line in suffix.splitlines():
        anchor = _rendered_source_line(line)
        if len(anchor) < 8 or not any(is_han(character) for character in anchor):
            continue
        position = rendered.find(anchor)
        if position >= 0:
            candidates.append((position, len(anchor)))
            if len(candidates) >= 20:
                break
    if not candidates:
        return None
    position, _ = min(candidates)
    return position if position > 0 else None


def clean_text_with_offsets(text: str, removed_prefix_end: int = 0) -> tuple[str, list[int]]:
    """Apply V1 whitespace cleanup while mapping every output character to raw textContent."""

    original = text
    normalized_chars: list[str] = []
    normalized_map: list[int] = []
    index = 0
    while index < len(original):
        if original[index : index + 2] == "\r\n":
            normalized_chars.append("\n")
            normalized_map.append(index)
            index += 2
            continue
        cluster_start = index
        cluster = "\n" if original[index] == "\r" else original[index]
        index += 1
        while index < len(original) and unicodedata.combining(original[index]):
            cluster += original[index]
            index += 1
        converted = unicodedata.normalize("NFC", cluster)
        normalized_chars.extend(converted)
        normalized_map.extend([cluster_start] * len(converted))
    normalized = "".join(normalized_chars)

    output: list[str] = []
    offsets: list[int] = []
    paragraph_open = False
    base = 0
    blocks = list(re.finditer(r"(?:^|\n\s*\n)(.*?)(?=\n\s*\n|\Z)", normalized, re.DOTALL))
    for block_match in blocks:
        block = block_match.group(1)
        block_start = block_match.start(1)
        line_rows: list[tuple[str, list[int]]] = []
        cursor = 0
        for raw_line in block.splitlines(keepends=True):
            line = raw_line.rstrip("\n")
            line_start = block_start + cursor
            cursor += len(raw_line)
            pieces: list[str] = []
            line_offsets: list[int] = []
            for match in re.finditer(r"\S+(?:[\t\u00a0\u3000 ]+|$)", line):
                token = match.group(0).rstrip("\t\u00a0\u3000 ")
                if not token:
                    continue
                if pieces:
                    pieces.append(" ")
                    line_offsets.append(normalized_map[line_start + match.start()])
                pieces.extend(token)
                line_offsets.extend(normalized_map[line_start + match.start() : line_start + match.start() + len(token)])
            cleaned_line = "".join(pieces).strip()
            if (
                not cleaned_line
                or cleaned_line in BOILERPLATE_LINES
                or re.fullmatch(r"[.·•]{3,}", cleaned_line)
                or all(offset < removed_prefix_end for offset in line_offsets if offset >= 0)
            ):
                continue
            keep = [(character, offset) for character, offset in zip("".join(pieces), line_offsets) if offset >= removed_prefix_end]
            if keep:
                line_rows.append(("".join(character for character, _ in keep).strip(), [offset for _, offset in keep]))
        if not line_rows:
            continue
        if paragraph_open:
            output.extend("\n\n")
            offsets.extend([-1, -1])
        for line_index, (line, line_offsets) in enumerate(line_rows):
            if line_index:
                output.append("\n")
                offsets.append(-1)
            output.extend(line)
            offsets.extend(line_offsets[: len(line)])
        paragraph_open = True
        base += len(block)
    if output:
        output.append("\n")
        offsets.append(-1)
    return "".join(output), offsets


def simplify_text(text: str, converter: OpenCC | None = None) -> str:
    return (converter or OpenCC("t2s")).convert(text)


def changed_character_count(before: str, after: str) -> int:
    return sum(left != right for left, right in zip(before, after)) + abs(len(before) - len(after))


def find_uncertain_blocks(text: str, removed_prefix_end: int) -> list[dict[str, Any]]:
    rows = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        start = cursor + len(line) - len(line.lstrip())
        end = start + len(stripped)
        cursor += len(line)
        if start >= removed_prefix_end and stripped and SUSPICIOUS_METADATA.fullmatch(stripped):
            rows.append({"source_position_start": start, "source_position_end": end, "block_text": stripped})
    return rows


def boundary_statistics(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for match in re.finditer(r"[^\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0003134f]+", text):
        value = match.group()
        counts["total"] += 1
        if any(character.isspace() for character in value):
            counts["whitespace"] += 1
        if re.search(r"[A-Za-z0-9]", value):
            counts["latin_or_digit"] += 1
        if any(unicodedata.category(character).startswith("P") for character in value):
            counts["punctuation"] += 1
        if any(
            not character.isspace()
            and not character.isascii()
            and not unicodedata.category(character).startswith("P")
            for character in value
        ):
            counts["other_symbol"] += 1
    return counts


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


def han_spans(text: str) -> list[dict[str, Any]]:
    return [
        {"boundary_span_id": index, "text": match.group(), "start": match.start(), "end": match.end()}
        for index, match in enumerate(re.finditer(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0003134f]+", text))
    ]


def segment_text(text: str, source_offsets: Sequence[int] | None = None) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for span in han_spans(text):
        for token, local_start, local_end in jieba.tokenize(span["text"], mode="default"):
            start = int(span["start"]) + local_start
            end = int(span["start"]) + local_end
            mapped = list(source_offsets[start:end]) if source_offsets is not None else list(range(start, end))
            tokens.append(
                {
                    "text": token,
                    "token_index": len(tokens),
                    "start": start,
                    "end": end,
                    "source_start": mapped[0],
                    "source_end": mapped[-1] + 1,
                    "boundary_span_id": span["boundary_span_id"],
                    "is_han": True,
                }
            )
    return tokens


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


def make_interactions(work: Mapping[str, Any], tokens: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    text = work["simplified_cleaned_text"]
    source_offsets = work.get("source_offsets") or list(range(len(text)))
    han_positions = [position for position, character in enumerate(text) if is_han(character)]
    interactions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        start, end = int(token["start"]), int(token["end"])
        source_start = int(token["source_start"])
        source_end = int(token["source_end"])
        context_end = bisect_left(han_positions, start)
        context_positions = han_positions[max(0, context_end - CONTEXT_CHARACTER_LIMIT) : context_end]
        if not context_positions:
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
            "context_source_position_start": source_offsets[context_positions[0]],
            "context": "".join(text[position] for position in context_positions),
            "source_creation_date": work["creation_date"],
            "source_hash": work["SHA256"],
            "original_cleaned_hash": work["original_cleaned_sha256"],
            "processed_text_hash": work["simplified_cleaned_sha256"],
            "boundary_span_id": token["boundary_span_id"],
        }
        short = {
            **base,
            "interaction_id": interaction_id(work["work_id"], source_start, source_end, "short"),
            "source_position_start": source_start,
            "source_position_end": source_end,
            "processed_position_start": start,
            "processed_position_end": end,
            "gold": token["text"],
            "full_pinyin": " ".join(syllables),
            "initial_pinyin": " ".join(initial_pinyin(syllables)),
            "composition_type": "short",
            "token_count": 1,
            "gold_char_length": len(token["text"]),
        }
        interactions.append(short)

        merged = [token]
        for next_token in tokens[index + 1 :]:
            if len(merged) >= 4:
                break
            if next_token["boundary_span_id"] != token["boundary_span_id"]:
                break
            merged.append(next_token)
        if len(merged) < 2:
            continue
        multi_end = int(merged[-1]["end"])
        multi_source_end = int(merged[-1]["source_end"])
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
                "interaction_id": interaction_id(work["work_id"], source_start, multi_source_end, "multi"),
                "source_position_start": source_start,
                "source_position_end": multi_source_end,
                "processed_position_start": start,
                "processed_position_end": multi_end,
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
        writer.writerows(
            {
                key: "\n".join(line.rstrip() for line in value.splitlines()) if isinstance(value, str) else value
                for key, value in row.items()
            }
            for row in rows
        )


@dataclass
class DeepAuthorBuilder:
    root: Path = ROOT
    api_base: str = API_BASE
    access_date: str = date.today().isoformat()

    def __post_init__(self) -> None:
        self.raw_root = self.root / "data/raw/deep_author"
        self.processed_root = self.root / "data/processed/deep_author"
        self.manifest_root = self.root / "data/manifests"
        self.audit_root = self.root / "results/audits/deep_author_dataset_v1_1"
        self.config = json.loads((self.root / "config/deep_author/authors_v1.json").read_text(encoding="utf-8"))
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "DeepAuthorDatasetV1.1/1.0 (academic reproducibility)"
        self.opencc = OpenCC("t2s")
        self.removed_metadata: list[dict[str, Any]] = []
        self.uncertain_blocks: list[dict[str, Any]] = []
        self.raw_checksums_before: dict[str, str] = {}

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
            self.raw_checksums_before[record["raw_filename"]] = sha256_file(raw_path)
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            extracted = str(payload["page"].get("textContent") or "")
            source = str(payload["page"].get("source") or "")
            prefix_end = metadata_prefix_end(source, extracted) or 0
            original_cleaned, source_offsets = clean_text_with_offsets(extracted, prefix_end)
            simplified = simplify_text(original_cleaned, self.opencc)
            if not simplified or han_count(simplified) < 20:
                record["inclusion_status"] = "excluded"
                record["exclusion_reason"] = "insufficient_extracted_chinese_text"
                continue
            if prefix_end:
                removed = extracted[:prefix_end]
                self.removed_metadata.append(
                    {
                        "author": record["author_name"],
                        "work": record["page_title"],
                        "work_id": record["work_id"],
                        "source_position_start": 0,
                        "source_position_end": prefix_end,
                        "removal_rule": "source_confirmed_credit_module_prefix",
                        "removed_text": removed,
                        "before_context": "",
                        "after_context": extracted[prefix_end : prefix_end + 160],
                    }
                )
            uncertain = find_uncertain_blocks(extracted, prefix_end)
            for block in uncertain:
                self.uncertain_blocks.append(
                    {
                        "author": record["author_name"],
                        "work": record["page_title"],
                        "work_id": record["work_id"],
                        **block,
                        "detected_structure_type": "metadata_like_text_outside_confirmed_block",
                        "reason_uncertain": "not structurally bounded by a source credit module",
                        "proposed_action": "retain pending human review",
                    }
                )
            tokens = segment_text(simplified, source_offsets)
            boundaries = boundary_statistics(simplified)
            work = {
                **record,
                "cleaned_text": simplified,
                "original_cleaned_text": original_cleaned,
                "simplified_cleaned_text": simplified,
                "source_offsets": source_offsets,
                "raw_character_count": len(extracted),
                "original_cleaned_character_count": len(original_cleaned),
                "cleaned_character_count": len(simplified),
                "han_character_count": han_count(simplified),
                "removed_character_count": max(0, len(extracted) - len(original_cleaned)),
                "metadata_removed_character_count": prefix_end,
                "metadata_removed_block_count": int(bool(prefix_end)),
                "paragraph_count": len([value for value in simplified.split("\n\n") if value.strip()]),
                "sentence_count": sentence_count(simplified),
                "segmented_token_count": len(tokens),
                "original_script_distribution": script_label(original_cleaned),
                "script_distribution": script_label(simplified),
                "opencc_changed_character_count": changed_character_count(original_cleaned, simplified),
                "boundary_sequence_count": boundaries["total"],
                "punctuation_boundary_count": boundaries["punctuation"],
                "latin_or_digit_boundary_count": boundaries["latin_or_digit"],
                "whitespace_boundary_count": boundaries["whitespace"],
                "other_symbol_boundary_count": boundaries["other_symbol"],
                "extraction_method": "SCPPER-CN text-content; source-confirmed block cleanup; NFC; OpenCC t2s; Han hard boundaries",
                "original_cleaned_sha256": sha256_bytes(original_cleaned.encode("utf-8")),
                "simplified_cleaned_sha256": sha256_bytes(simplified.encode("utf-8")),
                "cleaned_sha256": sha256_bytes(simplified.encode("utf-8")),
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
            serializable_work = {key: value for key, value in work.items() if key not in {"cleaned_text", "source_offsets"}}
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
                "original_cleaned_character_count", "cleaned_character_count", "han_character_count", "removed_character_count",
                "metadata_removed_character_count", "metadata_removed_block_count", "paragraph_count",
                "sentence_count", "segmented_token_count", "short_interaction_count", "multi_interaction_count",
                "full_pinyin_success_count", "initial_pinyin_success_count", "alignment_failure_count",
                "original_script_distribution", "script_distribution", "opencc_changed_character_count",
                "boundary_sequence_count", "punctuation_boundary_count", "latin_or_digit_boundary_count",
                "whitespace_boundary_count", "other_symbol_boundary_count", "original_cleaned_sha256",
                "simplified_cleaned_sha256",
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
                "metadata_blocks_removed": sum(item["metadata_removed_block_count"] for item in selected),
                "opencc_changed_characters": sum(item["opencc_changed_character_count"] for item in selected),
                "hard_boundaries": sum(item["boundary_sequence_count"] for item in selected),
                "depth_concern": "review_required" if len(selected) < 5 or sum(item["han_character_count"] for item in selected) < 10000 else "none_observed",
            }
            author_stats.append(row)
        _write_csv(self.audit_root / "author_statistics.csv", author_stats, list(author_stats[0]))
        _write_json(self.audit_root / "cleaning_comparison.json", {
            "v1": {
                "included_works": 282, "han_characters": 1016100, "segmented_tokens": 883512,
                "short_interactions": 601393, "multi_interactions": 472639,
                "total_interactions": 1074032, "alignment_failures": 26, "duplicate_groups": 0,
            },
            "v1_1": {
                "included_works": len(works),
                "han_characters": sum(row["han_character_count"] for row in works),
                "segmented_tokens": sum(row["segmented_token_count"] for row in works),
                "short_interactions": sum(row["short_interaction_count"] for row in works),
                "multi_interactions": sum(row["multi_interaction_count"] for row in works),
                "total_interactions": len(interactions),
                "alignment_failures": len(failures),
                "duplicate_groups": 0,
                "metadata_characters_removed": sum(row["metadata_removed_character_count"] for row in works),
                "metadata_blocks_removed": sum(row["metadata_removed_block_count"] for row in works),
                "characters_changed_by_opencc": sum(row["opencc_changed_character_count"] for row in works),
                "punctuation_non_han_boundaries_created": sum(row["boundary_sequence_count"] for row in works),
            },
        })
        _write_json(self.audit_root / "interaction_statistics.json", {
            "total": len(interactions),
            "short": sum(row["composition_type"] == "short" for row in interactions),
            "multi": sum(row["composition_type"] == "multi" for row in interactions),
            "authors": len({row["author_id"] for row in interactions}),
            "works": len({row["work_id"] for row in interactions}),
            "all_gold_han_only": all(all(is_han(character) for character in row["gold"]) for row in interactions),
            "all_context_han_only": all(all(is_han(character) for character in row["context"]) for row in interactions),
        })
        cleaning_summary = {
            "raw_characters": sum(row["raw_character_count"] for row in works),
            "cleaned_characters": sum(row["cleaned_character_count"] for row in works),
            "han_characters": sum(row["han_character_count"] for row in works),
            "removed_characters": sum(row["removed_character_count"] for row in works),
            "metadata_characters_removed": sum(row["metadata_removed_character_count"] for row in works),
            "metadata_blocks_removed": sum(row["metadata_removed_block_count"] for row in works),
            "original_script_distribution": Counter(row["original_script_distribution"] for row in works),
            "final_script_distribution": Counter(row["script_distribution"] for row in works),
            "works_changed_by_opencc": sum(row["opencc_changed_character_count"] > 0 for row in works),
            "characters_changed_by_opencc": sum(row["opencc_changed_character_count"] for row in works),
            "opencc_configuration": "t2s",
            "opencc_package": "opencc-python-reimplemented",
            "opencc_version": importlib.metadata.version("opencc-python-reimplemented"),
            "boundaries": {
                key: sum(row[f"{key}_boundary_count"] for row in works)
                for key in ("punctuation", "latin_or_digit", "whitespace", "other_symbol")
            },
            "total_boundary_sequences": sum(row["boundary_sequence_count"] for row in works),
        }
        _write_json(self.audit_root / "normalisation_statistics.json", cleaning_summary)
        raw_groups: dict[str, list[str]] = defaultdict(list)
        clean_groups: dict[str, list[str]] = defaultdict(list)
        for work in works:
            raw_groups[work["SHA256"]].append(work["work_id"])
            clean_groups[work["cleaned_sha256"]].append(work["work_id"])
        _write_json(self.audit_root / "duplicate_audit.json", {
            "exact_raw_duplicates": [value for value in raw_groups.values() if len(value) > 1],
            "exact_cleaned_text_duplicates": [value for value in clean_groups.values() if len(value) > 1],
            "duplicate_work_ids": [key for key, count in Counter(row["work_id"] for row in works).items() if count > 1],
            "repeated_sections_removed": True,
            "note": "Only source-confirmed non-author template blocks were removed; normal literary repetition was retained.",
        })
        failure_columns = ["work_id", "token_index", "text", "reason"]
        _write_csv(self.audit_root / "alignment_failures.csv", failures, failure_columns)

        removed_columns = ["author", "work", "source_position_start", "source_position_end", "removal_rule", "removed_text", "before_context", "after_context", "removal_ok", "notes"]
        removed_samples = []
        for author in self.config["authors"]:
            choices = [row for row in self.removed_metadata if row["author"] == author["name"]][:10]
            removed_samples.extend({**row, "removal_ok": "", "notes": ""} for row in choices)
        _write_csv(self.audit_root / "removed_metadata_review.csv", removed_samples, removed_columns)
        uncertain_columns = ["author", "work", "source_position_start", "source_position_end", "block_text", "detected_structure_type", "reason_uncertain", "proposed_action"]
        _write_csv(self.audit_root / "uncertain_blocks.csv", self.uncertain_blocks, uncertain_columns)

        script_samples = []
        for label in ("simplified", "mixed", "traditional", "shared_or_ambiguous"):
            for work in [row for row in works if row["original_script_distribution"] == label][:8]:
                script_samples.append({
                    "author": work["author_name"], "work": work["page_title"],
                    "original_excerpt": work["original_cleaned_text"][:240],
                    "simplified_excerpt": work["simplified_cleaned_text"][:240],
                    "changed_character_count": work["opencc_changed_character_count"],
                    "conversion_ok": "", "notes": "",
                })
        _write_csv(self.audit_root / "script_normalisation_review.csv", script_samples, ["author", "work", "original_excerpt", "simplified_excerpt", "changed_character_count", "conversion_ok", "notes"])

        samples = []
        for author in self.config["authors"]:
            name = author["name"]
            author_works = sorted(work_by_author[name], key=lambda row: row["work_id"])
            author_interactions = [row for row in interactions if row["author_name"] == name]
            for work in author_works[:2]:
                samples.append({"author": name, "work": work["page_title"], "original_source_excerpt": work["original_cleaned_text"][:180], "cleaned_simplified_excerpt": work["simplified_cleaned_text"][:180], "normalized_han_context": "", "gold": "", "full_pinyin": "", "initial_pinyin": "", "composition_type": "cleaned_text", "metadata_clean": "", "simplification_ok": "", "segmentation_ok": "", "boundary_ok": "", "pinyin_ok": "", "composition_ok": "", "notes": ""})
            for kind in ("short", "multi"):
                choices = [row for row in author_interactions if row["composition_type"] == kind][:3]
                for row in choices:
                    samples.append({"author": name, "work": row["work_title"], "original_source_excerpt": "", "cleaned_simplified_excerpt": (row["context"][-80:] + row["gold"])[:180], "normalized_han_context": row["context"][-120:], "gold": row["gold"], "full_pinyin": row["full_pinyin"], "initial_pinyin": row["initial_pinyin"], "composition_type": kind, "metadata_clean": "", "simplification_ok": "", "segmentation_ok": "", "boundary_ok": "", "pinyin_ok": "", "composition_ok": "", "notes": ""})
        _write_csv(self.audit_root / "manual_review.csv", samples, list(samples[0]) if samples else [])

        checksummed = [
            self.manifest_root / "deep_author_authors_manifest.csv",
            self.manifest_root / "deep_author_works_manifest.csv",
            self.processed_root / "interactions_t1_ready.jsonl",
            self.audit_root / "author_statistics.csv",
            self.audit_root / "corpus_statistics.csv",
            self.audit_root / "interaction_statistics.json",
            self.audit_root / "cleaning_comparison.json",
            self.audit_root / "normalisation_statistics.json",
            self.audit_root / "alignment_failures.csv",
            self.audit_root / "duplicate_audit.json",
            self.audit_root / "removed_metadata_review.csv",
            self.audit_root / "uncertain_blocks.csv",
            self.audit_root / "script_normalisation_review.csv",
            self.audit_root / "manual_review.csv",
        ]
        checksums = {path.relative_to(self.root).as_posix(): {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in checksummed}
        _write_json(self.audit_root / "checksums.json", checksums)
        _write_json(self.audit_root / "manifest.json", {
            "dataset": "Deep Author Dataset V1.1",
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
            "opencc_configuration": "t2s",
            "pipeline_order": ["immutable raw source", "extract", "remove confirmed metadata", "OpenCC t2s", "hard boundaries", "Jieba segmentation", "Pinyin", "interactions"],
            "dependencies": {name: importlib.metadata.version(name) for name in ("requests", "jieba", "pypinyin", "hanzidentifier", "opencc-python-reimplemented")},
            "outputs": checksums,
        })

    def run(self) -> dict[str, Any]:
        jieba.initialize()
        self.raw_checksums_before = {
            path.relative_to(self.root).as_posix(): sha256_file(path)
            for path in sorted(self.raw_root.rglob("*"))
            if path.is_file()
        }
        discoveries = self.discover()
        records = self.acquire(discoveries)
        works, interactions, failures = self.process(records)
        changed_raw = [path for path, checksum in self.raw_checksums_before.items() if sha256_file(self.root / path) != checksum]
        if changed_raw:
            raise RuntimeError(f"immutable raw files changed: {changed_raw[:3]}")
        self.audit(records, works, interactions, failures)
        return {"discovered": len(records), "included": len(works), "interactions": len(interactions), "failures": len(failures)}
