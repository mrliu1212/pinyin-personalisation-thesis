"""Authoritative, user-isolated plaintext L2 memory store."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .provenance import stable_digest


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    user_id: str
    plaintext: str
    creation_position: str
    source_interaction_ids: tuple[str, ...]
    active: bool = True
    vector_label: int | None = None
    who: str = "user"
    what: str = "memory_worker"
    source: str = "memory_worker"
    processed_at: str | None = None
    indexed_ok: bool | None = None
    source_line_index: int | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        user_id: str,
        plaintext: str,
        creation_position: str,
        source_interaction_ids: Sequence[str],
        vector_label: int | None = None,
        what: str = "memory_worker",
        processed_at: str | None = None,
        indexed_ok: bool | None = None,
        source_line_index: int | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> "MemoryRecord":
        text = plaintext.strip()
        if not user_id or not text or not creation_position:
            raise ValueError("memory requires user_id, plaintext, and creation_position")
        # Mirrors upstream stableId(timestamp + newline + indexText).
        memory_id = stable_digest(creation_position, text)
        return cls(
            memory_id=memory_id,
            user_id=user_id,
            plaintext=text,
            creation_position=creation_position,
            source_interaction_ids=tuple(source_interaction_ids),
            vector_label=vector_label,
            what=what or "memory_worker",
            processed_at=processed_at,
            indexed_ok=indexed_ok,
            source_line_index=source_line_index,
            provenance=dict(provenance or {}),
        )

    def with_vector(self, label: int, *, indexed_ok: bool = True) -> "MemoryRecord":
        values = asdict(self)
        values["source_interaction_ids"] = self.source_interaction_ids
        values["vector_label"] = label
        values["indexed_ok"] = indexed_ok
        return MemoryRecord(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MemoryRecord":
        copied = dict(value)
        copied["source_interaction_ids"] = tuple(copied.get("source_interaction_ids", ()))
        copied["provenance"] = dict(copied.get("provenance", {}))
        return cls(**copied)


class MemoryStore:
    """Append-only JSONL store scoped to exactly one user."""

    def __init__(self, root: Path, *, user_id: str, read_only: bool = False) -> None:
        if not user_id:
            raise ValueError("user_id must be non-empty")
        self.root = Path(root) / user_id
        self.user_id = user_id
        self.read_only = read_only
        self.path = self.root / "memories.jsonl"
        self._records: dict[str, MemoryRecord] = {}
        self.reload()

    def reload(self) -> None:
        records: dict[str, MemoryRecord] = {}
        if self.path.exists():
            for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                record = MemoryRecord.from_dict(json.loads(line))
                if record.user_id != self.user_id:
                    raise ValueError(f"cross-user memory at line {line_number}")
                existing = records.get(record.memory_id)
                if existing is not None and existing != record:
                    raise ValueError(f"conflicting duplicate memory {record.memory_id}")
                records[record.memory_id] = record
        self._records = records

    def add(self, record: MemoryRecord) -> MemoryRecord:
        if self.read_only:
            raise PermissionError("memory store is read-only")
        if record.user_id != self.user_id:
            raise ValueError("cannot add another user's memory")
        existing = self._records.get(record.memory_id)
        if existing is not None:
            if existing != record:
                raise ValueError("stable memory ID collision")
            return existing
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        self._records[record.memory_id] = record
        return record

    def get(self, memory_id: str) -> MemoryRecord:
        try:
            return self._records[memory_id]
        except KeyError as exc:
            raise KeyError(f"unknown memory for user {self.user_id}: {memory_id}") from exc

    def list(self, *, active_only: bool = False) -> tuple[MemoryRecord, ...]:
        records = self._records.values()
        if active_only:
            records = (record for record in records if record.active)
        return tuple(sorted(records, key=lambda item: (item.creation_position, item.memory_id)))

    def __len__(self) -> int:
        return len(self._records)

    def source_interaction_ids(self) -> frozenset[str]:
        return frozenset(
            interaction_id
            for record in self._records.values()
            for interaction_id in record.source_interaction_ids
        )
