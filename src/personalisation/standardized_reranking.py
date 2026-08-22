"""Reusable orchestration helpers for the sealed standardized comparison.

The functions here do not choose an evaluation population.  They preserve the
frozen ranking implementations while making Stage-1 retrieval and cache-backed
cross-encoder scoring usable on Train-Val and, after freeze, Dev3000.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from src.personalisation.context_memory import Candidate, PredictionQuery, metric_values


def load_hidden_vectors(path: Path, expected_size: int = 768) -> dict[str, np.ndarray]:
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    vectors: dict[str, np.ndarray] = {}
    try:
        for row_id, hidden_size, blob in connection.execute(
            "SELECT row_id, hidden_size, vector FROM hidden_states"
        ):
            if int(hidden_size) != expected_size:
                raise RuntimeError(f"hidden size differs for {row_id}")
            vector = np.frombuffer(blob, dtype="<f4").astype(np.float32, copy=True)
            if vector.shape != (expected_size,):
                raise RuntimeError(f"bad hidden vector shape for {row_id}: {vector.shape}")
            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                raise RuntimeError(f"zero hidden vector for {row_id}")
            vectors[str(row_id)] = vector / norm
    finally:
        connection.close()
    return vectors


def retrieve_hidden(
    query: PredictionQuery,
    visible: Sequence[Mapping[str, Any]],
    vectors: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], ...]:
    query_vector = vectors.get(query.row_id)
    if query_vector is None:
        raise RuntimeError(f"missing query hidden vector: {query.row_id}")
    values = []
    for history in visible:
        history_id = str(history["row_id"])
        history_vector = vectors.get(history_id)
        if history_vector is None:
            raise RuntimeError(f"missing historical hidden vector: {history_id}")
        similarity = float(np.dot(query_vector, history_vector))
        values.append({
            "historical_interaction_id": history_id,
            "historical_target": str(history["target"]),
            "similarity": similarity,
            "weight": max(similarity, 0.0),
            "chronological_position": int(history["chronological_position"]),
        })
    values.sort(key=lambda row: (
        -float(row["similarity"]), int(row["chronological_position"]),
        str(row["historical_interaction_id"]),
    ))
    return tuple(values)


def query_of(row: Mapping[str, Any]) -> PredictionQuery:
    return PredictionQuery(
        row_id=str(row["row_id"]), author=str(row["author"]),
        work_id=str(row["work_id"]),
        chronological_position=int(row["chronological_position"]),
        context=str(row["context"]), pinyin=tuple(map(str, row["pinyin_segments"])),
    )


def candidates_of(row: Mapping[str, Any]) -> tuple[Candidate, ...]:
    return tuple(
        Candidate(str(item["text"]), int(item["rank"]), float(item["log_probability"]))
        for item in row.get("top10_candidates", ())
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )


def selection_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[int | None]] = defaultdict(list)
    for row in rows:
        grouped[str(row["author"])].append(row.get("rank"))
    per_author = {author: metric_values(ranks) for author, ranks in sorted(grouped.items())}
    return {
        "micro": metric_values([row.get("rank") for row in rows]),
        "macro_author": {
            field: statistics.fmean(float(values[field]) for values in per_author.values())
            for field in ("top1", "top3", "mrr_at_10", "missing_at_10")
        },
        "per_author": per_author,
    }


def choose_grid(
    rows_by_grid: Mapping[tuple[int, float], Sequence[Mapping[str, Any]]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grid = []
    for (retrieval_k, weight), rows in sorted(rows_by_grid.items()):
        metrics = selection_metrics(rows)
        grid.append({
            "retrieval_k": retrieval_k,
            "lambda": weight,
            "metrics": metrics,
        })
    selected = max(
        grid,
        key=lambda row: (
            float(row["metrics"]["macro_author"]["top1"]),
            -float(row["lambda"]), -int(row["retrieval_k"]),
        ),
    )
    return selected, grid
