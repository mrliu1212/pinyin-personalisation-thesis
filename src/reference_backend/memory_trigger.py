"""Selective memory-action parsing from the official HuoziIME token protocol."""

from __future__ import annotations

import re
from typing import Iterable

from .provenance import TriggerDecision


QUERY_RE = re.compile(r'query\s*=\s*"([^"\n]{1,512})"', re.IGNORECASE)
def parse_memory_query(outputs: Iterable[str]) -> tuple[str, str] | None:
    candidates = tuple(outputs)
    for raw in candidates:
        if "<MEM_RETRIEVAL>" in raw.upper() or "</MEM_RETRIEVAL>" in raw.upper():
            match = QUERY_RE.search(raw)
            if match and match.group(1).strip():
                return match.group(1).strip(), raw
    for raw in candidates:
        if 'query="' in raw.lower():
            match = QUERY_RE.search(raw)
            if match and match.group(1).strip():
                return match.group(1).strip(), raw
    return None


class OfficialTokenMemoryTrigger:
    def __init__(self, *, official_checkpoint_policy: bool) -> None:
        self.official_checkpoint_policy = official_checkpoint_policy

    @property
    def method(self) -> str:
        return (
            "OFFICIAL_POLICY"
            if self.official_checkpoint_policy
            else "OFFICIAL_RUNTIME_LOGIC"
        )

    def should_retrieve(self, generated_outputs: Iterable[str]) -> TriggerDecision:
        parsed = parse_memory_query(generated_outputs)
        if parsed is None:
            return TriggerDecision(False, None, self.method)
        query, evidence = parsed
        return TriggerDecision(True, query, self.method, evidence)


class ReferenceMemoryTriggerFallback:
    """Honest no-retrieval fallback when the learned policy is unavailable.

    It intentionally does not invent a lexical threshold or heuristic.
    """

    method = "ARCHITECTURAL_FALLBACK"

    def should_retrieve(self, generated_outputs: Iterable[str]) -> TriggerDecision:
        del generated_outputs
        return TriggerDecision(False, None, self.method)
