"""Separate chronological L3 interaction and decision traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .provenance import stable_interaction_id


@dataclass(frozen=True)
class InteractionTrace:
    interaction_id: str
    user_id: str
    chronological_position: str
    preceding_text: str
    pinyin_or_keystrokes: str
    selected_text: str | None
    prediction: Mapping[str, Any]
    memory_triggered: bool
    retrieved_memory_ids: tuple[str, ...]
    generated_candidates: tuple[str, ...]
    trace_type: str = "foreground_prediction"

    @classmethod
    def create(
        cls,
        *,
        user_id: str,
        chronological_position: str,
        preceding_text: str,
        pinyin_or_keystrokes: str,
        selected_text: str | None,
        prediction: Mapping[str, Any],
        memory_triggered: bool,
        retrieved_memory_ids: tuple[str, ...],
        generated_candidates: tuple[str, ...],
        trace_type: str = "foreground_prediction",
    ) -> "InteractionTrace":
        return cls(
            interaction_id=stable_interaction_id(
                user_id,
                chronological_position,
                preceding_text,
                pinyin_or_keystrokes,
                selected_text,
            ),
            user_id=user_id,
            chronological_position=chronological_position,
            preceding_text=preceding_text,
            pinyin_or_keystrokes=pinyin_or_keystrokes,
            selected_text=selected_text,
            prediction=dict(prediction),
            memory_triggered=memory_triggered,
            retrieved_memory_ids=retrieved_memory_ids,
            generated_candidates=generated_candidates,
            trace_type=trace_type,
        )


class InteractionTraceStore:
    def __init__(self, root: Path, *, user_id: str, read_only: bool = False) -> None:
        self.user_id = user_id
        self.root = Path(root) / user_id
        self.path = self.root / "interactions.jsonl"
        self.read_only = read_only
        self._traces: list[InteractionTrace] = []
        self.reload()

    def reload(self) -> None:
        traces = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                value["retrieved_memory_ids"] = tuple(value.get("retrieved_memory_ids", ()))
                value["generated_candidates"] = tuple(value.get("generated_candidates", ()))
                trace = InteractionTrace(**value)
                if trace.user_id != self.user_id:
                    raise ValueError("cross-user L3 trace")
                traces.append(trace)
        positions = [item.chronological_position for item in traces]
        if positions != sorted(positions):
            raise ValueError("L3 traces are not chronological")
        self._traces = traces

    def append(self, trace: InteractionTrace) -> None:
        if self.read_only:
            raise PermissionError("interaction trace store is read-only")
        if trace.user_id != self.user_id:
            raise ValueError("cannot add another user's trace")
        if self._traces and trace.chronological_position < self._traces[-1].chronological_position:
            raise ValueError("L3 traces must be appended chronologically")
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(trace), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        self._traces.append(trace)

    def list(self) -> tuple[InteractionTrace, ...]:
        return tuple(self._traces)
