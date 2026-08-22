"""Resumable frozen Generic generation for standardized comparison manifests."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_completed(paths: Sequence[Path], valid_ids: set[str]) -> dict[str, dict[str, Any]]:
    completed = {}
    for path in paths:
        for row in read_jsonl(path):
            row_id = str(row.get("row_id", ""))
            if row_id not in valid_ids:
                raise RuntimeError(f"Generic cache contains an unknown row: {row_id}")
            if row_id in completed and completed[row_id] != row:
                raise RuntimeError(f"Generic cache contains conflicting rows: {row_id}")
            completed[row_id] = row
    return completed


def generate_resumable(
    rows: Sequence[Mapping[str, Any]], backend: Any, output_path: Path, *, batch_size: int = 2,
    checkpoint_revision: str, official_code_revision: str,
    backend_source_revision: str, backend_integration_revision: str,
    context_semantics: str,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if any(str(row.get("source_split", "")).lower() == "test" for row in rows):
        raise RuntimeError("STOP: Test row detected in Generic input")
    ids = [str(row["row_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Generic input row IDs are not unique")
    partial_path = output_path.with_suffix(".partial.jsonl")
    completed = load_completed((output_path, partial_path), set(ids))
    initial = len(completed)
    pending = [row for row in rows if str(row["row_id"]) not in completed]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with partial_path.open("a", encoding="utf-8", newline="\n") as destination:
        for window_start in range(0, len(pending), 256):
            window = pending[window_start:window_start + 256]
            buckets: dict[tuple[int, int], list[tuple[Mapping[str, Any], tuple[str, ...], str, int, int, bool]]] = defaultdict(list)
            for row in window:
                segments = tuple(map(str, row["pinyin_segments"]))
                used, original_tokens, used_tokens, truncated = backend.truncate_context_for_generation(str(row["context"]), segments)
                prompt, _ = backend._prompt(used, segments)
                buckets[(len(prompt), len(segments))].append((row, segments, used, original_tokens, used_tokens, truncated))
            for bucket_key in sorted(buckets):
                bucket = buckets[bucket_key]
                for start in range(0, len(bucket), batch_size):
                    batch = bucket[start:start + batch_size]
                    results = backend.generate_batch([(item[2], item[1]) for item in batch], top_k=10, beam_size=16)
                    if len(results) != len(batch):
                        raise RuntimeError("Generic backend returned an unexpected result count")
                    for item, result in zip(batch, results):
                        row, _, used, original_tokens, used_tokens, truncated = item
                        candidates = [candidate.to_dict() for candidate in result.candidates]
                        output = {
                            **row, "model_used_context": used,
                            "original_stored_context_tokens": original_tokens,
                            "model_used_context_tokens": used_tokens, "context_truncated": truncated,
                            "top10_candidates": candidates,
                            "gold_rank": next((c["rank"] for c in candidates if c["text"] == row["gold"]), None),
                            "beam_size": 16, "top_k": 10, "runtime_device": result.runtime_device,
                            "checkpoint_revision": checkpoint_revision, "official_code_revision": official_code_revision,
                            "backend_source_revision": backend_source_revision,
                            "backend_integration_revision": backend_integration_revision,
                            "context_semantics": context_semantics, "used_test": False,
                        }
                        destination.write(canonical_json(output) + "\n")
                        completed[str(row["row_id"])] = output
                    destination.flush()
            done = len(completed)
            elapsed = time.perf_counter() - started
            rate = (done - initial) / elapsed if elapsed else 0.0
            print(f"generic {done}/{len(rows)} reused={initial} rate={rate:.3f}/s", flush=True)
    if len(completed) != len(rows):
        raise RuntimeError(f"Generic generation incomplete: {len(completed)}/{len(rows)}")
    temporary = output_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as destination:
        for row_id in ids:
            destination.write(canonical_json(completed[row_id]) + "\n")
    temporary.replace(output_path)
    return {
        "status": "complete", "rows": len(rows), "reused": initial,
        "generated": len(rows) - initial, "output": str(output_path),
        "elapsed_seconds": time.perf_counter() - started, "used_test": False,
    }
