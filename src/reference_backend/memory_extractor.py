"""Faithful adaptation of HuoziIME's background memory worker."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
from typing import Mapping, Sequence

from .memory_store import MemoryRecord
from .model_runtime import GenerationRuntime
from .official_prompts import build_memory_worker_prompt


ALLOWED_FIELDS = ("summary", "datetime", "location", "participants", "item", "detail")


@dataclass(frozen=True)
class ExtractionResult:
    memory: MemoryRecord | None
    raw_output: str
    status: str
    elapsed_ms: float


class HuoziIMEMemoryExtractor:
    MAX_OUTPUT_TOKENS = 192
    MIN_INPUT_CHARS = 6

    def __init__(self, runtime: GenerationRuntime) -> None:
        self.runtime = runtime

    @staticmethod
    def _parse_json(raw: str) -> dict | None:
        clean = raw.replace("<think>", "")
        if "</think>" in clean:
            clean = clean.split("</think>", 1)[-1]
        clean = clean.replace("<|im_end|>", "").strip()
        if clean.upper() == "<NO_MEM>":
            return None
        try:
            value = json.loads(clean)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
            if match is None:
                raise ValueError("memory worker output is neither JSON nor <NO_MEM>")
            value = json.loads(match.group(0))
        if not isinstance(value, dict):
            raise ValueError("memory worker JSON must be an object")
        return {field: value[field] for field in ALLOWED_FIELDS if field in value}

    @staticmethod
    def _index_text(value: Mapping[str, object]) -> tuple[str, str]:
        summary = str(value.get("summary", "")).strip()
        if len(summary) < 3:
            raise ValueError("memory summary is too short")
        participants_value = value.get("participants", ())
        if participants_value is None:
            participants: list[str] = []
        elif isinstance(participants_value, list):
            participants = [str(item).strip() for item in participants_value if str(item).strip()]
        else:
            # JSONObject.optJSONArray returns null for another JSON type and
            # upstream silently omits that optional field.
            participants = []
        pieces = [summary]
        for key, label in (
            ("datetime", "时间"),
            ("location", "地点"),
        ):
            text = str(value.get(key, "")).strip()
            if text:
                pieces.append(f"{label}: {text}")
        if participants:
            pieces.append(f"参与者: {', '.join(participants)}")
        item = str(value.get("item", "")).strip()
        if item:
            pieces.append(f"事项: {item}")
        detail = str(value.get("detail", "")).strip()
        if detail and detail != summary:
            pieces.append(f"细节: {detail}")
        return " | ".join(pieces), item or "memory_worker"

    def extract(
        self,
        *,
        user_id: str,
        trajectory_text: str,
        creation_position: str,
        source_interaction_ids: Sequence[str],
        source_line_index: int,
        seed: int,
        provenance: Mapping[str, object] | None = None,
    ) -> ExtractionResult:
        text = trajectory_text.strip()
        if len(text) < self.MIN_INPUT_CHARS:
            return ExtractionResult(None, "", "skipped_short", 0.0)
        if text.upper() == "<NO_MEM>":
            return ExtractionResult(None, text, "skipped_no_mem_marker", 0.0)
        start = time.perf_counter()
        output = self.runtime.generate(
            build_memory_worker_prompt(text),
            max_tokens=self.MAX_OUTPUT_TOKENS,
            seed=seed,
            top_k=1,
            top_p=1.0,
            temperature=0.0,
            repeat_penalty=1.0,
            repeat_last_n=0,
        )
        elapsed = (time.perf_counter() - start) * 1000.0
        try:
            parsed = self._parse_json(output.text)
        except (ValueError, json.JSONDecodeError):
            return ExtractionResult(None, output.text, "extract_invalid_json", elapsed)
        if parsed is None:
            return ExtractionResult(None, output.text, "no_mem", elapsed)
        try:
            index_text, what = self._index_text(parsed)
        except ValueError:
            return ExtractionResult(None, output.text, "extract_empty_summary", elapsed)
        if len(index_text) < self.MIN_INPUT_CHARS:
            return ExtractionResult(None, output.text, "index_skipped_short", elapsed)
        record = MemoryRecord.create(
            user_id=user_id,
            plaintext=index_text,
            creation_position=creation_position,
            source_interaction_ids=source_interaction_ids,
            what=what,
            processed_at=creation_position,
            indexed_ok=False,
            source_line_index=source_line_index,
            provenance={
                "extractor": "HuoziIMEMemoryExtractor",
                "raw_extraction_output": output.text,
                "structured_fields": parsed,
                **dict(provenance or {}),
            },
        )
        return ExtractionResult(record, output.text, "extracted", elapsed)
