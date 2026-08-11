"""Persisted, per-user HNSW index mapped to authoritative plaintext memory."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from .memory_store import MemoryRecord, MemoryStore


@dataclass(frozen=True)
class VectorSearchResult:
    vector_label: int
    memory_id: str
    raw_cosine: float


def l2_normalize(vector: Sequence[float]) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("embedding must be a non-empty one-dimensional vector")
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("embedding norm must be finite and positive")
    return values / norm


class HNSWMemoryIndex:
    MAX_ELEMENTS = 2048
    M = 16
    EF_CONSTRUCTION = 200
    EF_SEARCH = 64
    SPACE = "ip"

    def __init__(self, root: Path, *, user_id: str, dimension: int) -> None:
        if not user_id or dimension <= 0:
            raise ValueError("index requires user_id and positive dimension")
        try:
            import hnswlib
        except ImportError as exc:
            raise RuntimeError("hnswlib is required; install requirements-phase4f.txt") from exc
        self._hnswlib = hnswlib
        self.root = Path(root) / user_id
        self.user_id = user_id
        self.dimension = dimension
        self.index_path = self.root / "hnsw.index"
        self.mapping_path = self.root / "mapping.json"
        self.config_path = self.root / "config.json"
        self._mapping: dict[int, str] = {}
        self._index = hnswlib.Index(space=self.SPACE, dim=dimension)
        if self.mapping_path.exists() or self.index_path.exists():
            if not self.mapping_path.exists() or not self.index_path.exists():
                raise ValueError("HNSW persistence is incomplete")
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
            if config["user_id"] != user_id or int(config["dimension"]) != dimension:
                raise ValueError("HNSW configuration does not match requested user/dimension")
            self._mapping = {
                int(label): memory_id
                for label, memory_id in json.loads(
                    self.mapping_path.read_text(encoding="utf-8")
                ).items()
            }
            self._index.load_index(str(self.index_path), max_elements=self.MAX_ELEMENTS)
            self._index.set_ef(self.EF_SEARCH)
        else:
            self._index.init_index(
                max_elements=self.MAX_ELEMENTS,
                ef_construction=self.EF_CONSTRUCTION,
                M=self.M,
                random_seed=100,
                allow_replace_deleted=False,
            )
            self._index.set_ef(self.EF_SEARCH)

    def next_label(self) -> int:
        return max(self._mapping, default=0) + 1

    def add(self, memory: MemoryRecord, vector: Sequence[float]) -> MemoryRecord:
        if memory.user_id != self.user_id:
            raise ValueError("cannot index another user's memory")
        if memory.memory_id in self._mapping.values():
            label = next(label for label, value in self._mapping.items() if value == memory.memory_id)
            return memory.with_vector(label)
        if len(self._mapping) >= self.MAX_ELEMENTS:
            raise ValueError("official HNSW max_elements capacity reached")
        label = self.next_label()
        normalized = l2_normalize(vector)
        if normalized.size != self.dimension:
            raise ValueError("embedding dimension mismatch")
        self._index.add_items(normalized.reshape(1, -1), np.asarray([label], dtype=np.int64))
        self._mapping[label] = memory.memory_id
        self.persist()
        return memory.with_vector(label)

    def persist(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._index.save_index(str(self.index_path))
        self.mapping_path.write_text(
            json.dumps(
                {str(label): memory_id for label, memory_id in sorted(self._mapping.items())},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.config_path.write_text(
            json.dumps(
                {
                    "dimension": self.dimension,
                    "ef_construction": self.EF_CONSTRUCTION,
                    "ef_search": self.EF_SEARCH,
                    "m": self.M,
                    "max_elements": self.MAX_ELEMENTS,
                    "metric": "inner product over L2-normalized vectors",
                    "schema_version": 1,
                    "user_id": self.user_id,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def search(
        self,
        *,
        user_id: str,
        query_vector: Sequence[float],
        k: int = 20,
    ) -> tuple[VectorSearchResult, ...]:
        if user_id != self.user_id:
            raise ValueError("cross-user HNSW query rejected")
        if not self._mapping:
            return ()
        normalized = l2_normalize(query_vector)
        if normalized.size != self.dimension:
            raise ValueError("query embedding dimension mismatch")
        count = min(max(1, k), len(self._mapping))
        labels, distances = self._index.knn_query(normalized.reshape(1, -1), k=count)
        results = [
            VectorSearchResult(
                vector_label=int(label),
                memory_id=self._mapping[int(label)],
                raw_cosine=max(-1.0, min(1.0, 1.0 - float(distance))),
            )
            for label, distance in zip(labels[0], distances[0])
        ]
        return tuple(sorted(results, key=lambda item: (-item.raw_cosine, item.vector_label)))

    def validate_against(self, store: MemoryStore) -> None:
        if store.user_id != self.user_id:
            raise ValueError("store/index user mismatch")
        records = {record.memory_id: record for record in store.list()}
        orphans = set(self._mapping.values()) - set(records)
        if orphans:
            raise ValueError(f"orphan HNSW mappings: {sorted(orphans)}")
        for label, memory_id in self._mapping.items():
            record = records[memory_id]
            if record.vector_label != label or not record.indexed_ok:
                raise ValueError(f"memory/index mapping mismatch: {memory_id}")

    def __len__(self) -> int:
        return len(self._mapping)
