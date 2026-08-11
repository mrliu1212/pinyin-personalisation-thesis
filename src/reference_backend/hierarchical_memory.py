"""Desktop L1/L2/L3 role separation and background memory evolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Sequence

from .benchmark_adapter import TrainingTrajectory
from .memory_extractor import HuoziIMEMemoryExtractor
from .memory_store import MemoryRecord, MemoryStore
from .model_runtime import EmbeddingRuntime
from .vector_index import HNSWMemoryIndex


@dataclass(frozen=True)
class BackgroundProcessRecord:
    user_id: str
    work_id: str
    chronological_position: str
    source_interaction_ids: tuple[str, ...]
    status: str
    memory_id: str | None
    extraction_ms: float
    embedding_index_ms: float


class BackgroundMemoryProcessor:
    """Explicit background path; foreground prediction never calls this class."""

    MAX_LINES = 12

    def __init__(
        self,
        *,
        user_id: str,
        store: MemoryStore,
        index: HNSWMemoryIndex,
        extractor: HuoziIMEMemoryExtractor,
        embedding_runtime: EmbeddingRuntime,
        trace_path: Path,
    ) -> None:
        if store.user_id != user_id or index.user_id != user_id:
            raise ValueError("background components must belong to one user")
        self.user_id = user_id
        self.store = store
        self.index = index
        self.extractor = extractor
        self.embedding_runtime = embedding_runtime
        self.trace_path = Path(trace_path)
        self._processing = False

    def process(
        self, trajectories: Sequence[TrainingTrajectory]
    ) -> tuple[BackgroundProcessRecord, ...]:
        if self._processing:
            raise RuntimeError("background memory processing is already active")
        if any(item.user_id != self.user_id for item in trajectories):
            raise ValueError("background batch cannot mix users")
        ordered = tuple(sorted(trajectories, key=lambda item: item.chronological_position))
        if len(ordered) > self.MAX_LINES:
            raise ValueError("batch exceeds the audited IdleMemoryWorker maxLines=12")
        self._processing = True
        results: list[BackgroundProcessRecord] = []
        try:
            for line_index, trajectory in enumerate(ordered):
                seed = int(
                    hashlib.sha256(
                        f"{self.user_id}\n{trajectory.chronological_position}".encode("utf-8")
                    ).hexdigest()[:8],
                    16,
                ) & 0x7FFF_FFFF
                extraction = self.extractor.extract(
                    user_id=self.user_id,
                    trajectory_text=trajectory.text,
                    creation_position=trajectory.chronological_position,
                    source_interaction_ids=trajectory.source_interaction_ids,
                    source_line_index=line_index,
                    seed=seed,
                    provenance={
                        "work_id": trajectory.work_id,
                        "trajectory_characters": len(trajectory.text),
                        "trajectory_policy": "upstream-style aggregated input capped at 4000 characters",
                    },
                )
                indexed_ms = 0.0
                memory = extraction.memory
                if memory is not None:
                    start = time.perf_counter()
                    vector = self.embedding_runtime.embed(memory.plaintext)
                    indexed = self.index.add(memory, vector)
                    self.store.add(indexed)
                    indexed_ms = (time.perf_counter() - start) * 1000.0
                    memory = indexed
                record = BackgroundProcessRecord(
                    user_id=self.user_id,
                    work_id=trajectory.work_id,
                    chronological_position=trajectory.chronological_position,
                    source_interaction_ids=trajectory.source_interaction_ids,
                    status=extraction.status,
                    memory_id=memory.memory_id if memory else None,
                    extraction_ms=extraction.elapsed_ms,
                    embedding_index_ms=indexed_ms,
                )
                self._append_trace(record)
                results.append(record)
            self.index.persist()
            self.index.validate_against(self.store)
            return tuple(results)
        finally:
            self._processing = False

    def _append_trace(self, record: BackgroundProcessRecord) -> None:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


@dataclass(frozen=True)
class HierarchicalMemoryStatus:
    l1: str
    l2_memory_count: int
    l2_index_count: int
    l3_trace_count: int


def status(*, store: MemoryStore, index: HNSWMemoryIndex, l3_trace_count: int) -> HierarchicalMemoryStatus:
    return HierarchicalMemoryStatus(
        l1="mobile KV memory omitted; resident runtime cache telemetry only",
        l2_memory_count=len(store),
        l2_index_count=len(index),
        l3_trace_count=l3_trace_count,
    )
