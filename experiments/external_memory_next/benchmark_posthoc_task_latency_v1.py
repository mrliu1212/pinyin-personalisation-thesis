"""Benchmark warm Task-BiEncoder query and frozen Top-5 support latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any, Mapping

import numpy as np

from experiments.context_comparison import run_full_transfer_initial_final_v1 as base
from experiments.external_memory_next.prepare_posthoc_context_support_v1 import VectorStore
from src.personalisation.posthoc_context_calibration import cosine_top5_support
from src.personalisation.task_specific_biencoder import (
    SharedContextEncoder,
    refuse_closed_path,
    sha256_file,
    sha256_tree,
    write_json,
)


CHECKPOINT_SHA256 = "f9b87af11fcff692ad7c25fb6330f44f9f23ffedb480af9aec36af0e7cd08a8e"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def summarize(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]

    return {
        "n": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(.50),
        "p95_ms": percentile(.95),
        "p99_ms": percentile(.99),
    }


class CachedVectors(Mapping[str, np.ndarray]):
    def __init__(self, store: VectorStore) -> None:
        self.store = store

    def __getitem__(self, key: str) -> np.ndarray:
        value = self.store.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __iter__(self):
        raise TypeError("iteration is not supported")

    def __len__(self) -> int:
        return self.store.count()

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.store.get(key) is not None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--task-checkpoint", type=Path, required=True)
    parser.add_argument("--task-vectors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    for path in vars(args).values():
        if isinstance(path, Path):
            refuse_closed_path(path)
    checkpoint_sha, _files = sha256_tree(args.task_checkpoint)
    if checkpoint_sha != CHECKPOINT_SHA256:
        raise ValueError("task checkpoint hash changed")

    fit = read_jsonl(args.fit)
    val = read_jsonl(args.val)
    supports = {str(row["row_id"]): row for row in read_jsonl(args.support)}
    if len(val) != 34_416 or len(supports) != len(val):
        raise ValueError("benchmark population changed")
    if any(row.get("used_dev3000") or row.get("used_test") for row in [*val, *supports.values()]):
        raise ValueError("closed-resource marker found")

    history = base.CausalHistoryIndex([*fit, *val])
    sample = val[: args.warmup + args.queries]
    store = VectorStore(args.task_vectors, read_only=True)
    vectors = CachedVectors(store)
    encoder = SharedContextEncoder(args.task_checkpoint, device="cuda")
    embedding_ms: list[float] = []
    retrieval_ms: list[float] = []
    total_ms: list[float] = []
    for number, row in enumerate(sample):
        row_id = str(row["row_id"])
        support = supports[row_id]
        candidates = list(map(str, support["candidate_union"]))
        candidate_set = set(candidates)
        visible = history.visible_same_pinyin(
            author=str(row["author"]),
            position=int(row["chronological_position"]),
            pinyin=base.pinyin_of(row),
        )
        visible = tuple(item for item in visible if item.record.target in candidate_set)
        started = time.perf_counter()
        query_vector = encoder.embed([base.context_of(row)[-64:]], batch_size=1)[0]
        embedded = time.perf_counter()
        cosine_top5_support(
            query_vector=query_vector,
            candidates=candidates,
            visible=visible,
            vectors=vectors,
            tau=2048.0,
            normalize=False,
        )
        finished = time.perf_counter()
        if number >= args.warmup:
            embedding_ms.append((embedded - started) * 1000.0)
            retrieval_ms.append((finished - embedded) * 1000.0)
            total_ms.append((finished - started) * 1000.0)

    import torch

    result = {
        "schema_version": 1,
        "status": "complete",
        "sample": "first 500 canonical Initial Train-Val rows after 20 warmup rows",
        "batch_size": 1,
        "history_embeddings_precomputed": True,
        "query_embeddings_precomputed": False,
        "online_query_embedding": summarize(embedding_ms),
        "online_cached_top5_retrieval_scoring": summarize(retrieval_ms),
        "online_total": summarize(total_ms),
        "hardware": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "inputs": {
            "fit_sha256": sha256_file(args.fit),
            "val_sha256": sha256_file(args.val),
            "support_sha256": sha256_file(args.support),
            "checkpoint_sha256": checkpoint_sha,
        },
        "used_dev3000": False,
        "used_test": False,
    }
    store.close()
    write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
