"""Audit/preflight and materialize frozen Generic/Task context support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiments.context_comparison import run_full_transfer_initial_final_v1 as base
from src.personalisation.posthoc_context_calibration import (
    cosine_top5_support,
    restrict_and_normalize,
)
from src.personalisation.task_specific_biencoder import (
    SharedContextEncoder,
    refuse_closed_path,
    sha256_file,
    sha256_tree,
    write_json,
    write_jsonl,
)


EXPECTED = {
    "initial_fit": "162f5c98daa86cc69947571e6d8f20fc401f0a82cdd3fd6e517eb7be2addbdb4",
    "initial_val": "d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4",
    "initial_stage1": "54e60073daabb14bb7cf43136a335216888ea03c06d078a4eec56e5775a0cfbc",
    "initial_ngram": "03858de42c41a26c4134d4b069b61ab2a5468c24cbd70a71958d600e448a97e1",
    "initial_frequency": "7fd8aa158d8cd50bced36b55610f8d932bc65e3aae1dbbd5bd65907ff1707ea7",
    "full_fit": "547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6",
    "full_val": "d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220",
    "full_stage1": "e311261cb0c5ea252ce593fdaa43ff87826d19eef440f39e2efc67ddc4310b13",
    "full_stage2": "d413d02650a759c0a759e1845212e68b2d948c1a1d20fc88abfd89ea7973bc64",
}
EXPECTED_ROWS = 34_416
TASK_CHECKPOINT_SHA256 = "f9b87af11fcff692ad7c25fb6330f44f9f23ffedb480af9aec36af0e7cd08a8e"
INITIAL_BASES = ("K5+Entropy", "4P+4CS+2E", "6P+2CS+.25E")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def index_rows(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    result = {str(row["row_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate row IDs in {label}")
    return result


class VectorStore:
    def __init__(self, path: Path, *, read_only: bool = False, checkpoint: str | None = None) -> None:
        self.path = path
        self.read_only = read_only
        if read_only:
            if not path.is_file():
                raise FileNotFoundError(path)
            self.connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(path)
            self.connection.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
            self.connection.execute("CREATE TABLE IF NOT EXISTS embeddings(context TEXT PRIMARY KEY,dim INTEGER NOT NULL,vector BLOB NOT NULL)")
            if checkpoint is not None:
                previous = self.connection.execute("SELECT value FROM metadata WHERE key='checkpoint_sha256'").fetchone()
                if previous is not None and previous[0] != checkpoint:
                    raise ValueError("vector cache checkpoint mismatch")
                self.connection.execute("INSERT OR REPLACE INTO metadata VALUES('checkpoint_sha256',?)", (checkpoint,))
            self.connection.commit()

    def get(self, context: str) -> np.ndarray | None:
        row = self.connection.execute("SELECT dim,vector FROM embeddings WHERE context=?", (context,)).fetchone()
        if row is None:
            return None
        value = np.frombuffer(row[1], dtype=np.float32).copy()
        if value.size != int(row[0]):
            raise ValueError(f"corrupt vector in {self.path}")
        norm = float(np.linalg.norm(value))
        return value / norm if norm > 0 else value

    def put_many(self, contexts: Sequence[str], vectors: Sequence[np.ndarray]) -> None:
        if self.read_only:
            raise ValueError("read-only vector store")
        self.connection.executemany(
            "INSERT OR REPLACE INTO embeddings(context,dim,vector) VALUES(?,?,?)",
            [(context, int(value.size), np.asarray(value, dtype=np.float32).tobytes()) for context, value in zip(contexts, vectors)],
        )
        self.connection.commit()

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def candidate_text(item: Mapping[str, Any]) -> str:
    return str(item["candidate"])


def verify_path(path: Path, expected: str, label: str) -> None:
    refuse_closed_path(path)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: expected={expected} actual={actual}")


def load_track(args: argparse.Namespace) -> dict[str, Any]:
    if args.track == "initial":
        verify_path(args.fit, EXPECTED["initial_fit"], "Initial Train-Fit")
        verify_path(args.val, EXPECTED["initial_val"], "Initial Train-Val")
        verify_path(args.stage1, EXPECTED["initial_stage1"], "Initial Stage-1")
        feature_rows = read_jsonl(args.stage1)
        support_rows = read_jsonl(args.existing_support)
        if args.frequency_predictions is None:
            raise ValueError("Initial support preparation requires --frequency-predictions")
        verify_path(args.frequency_predictions, EXPECTED["initial_frequency"], "Initial Frequency/PV1")
        frequency_rows = read_jsonl(args.frequency_predictions)
    else:
        verify_path(args.fit, EXPECTED["full_fit"], "Full Train-Fit")
        verify_path(args.val, EXPECTED["full_val"], "Full Train-Val")
        verify_path(args.stage1, EXPECTED["full_stage1"], "Full Stage-1")
        verify_path(args.existing_support, EXPECTED["full_stage2"], "Full Stage-2")
        feature_rows = read_jsonl(args.stage1)
        support_rows = read_jsonl(args.existing_support)
        frequency_rows = []
    fit_rows = read_jsonl(args.fit)
    val_rows = read_jsonl(args.val)
    if len(val_rows) != EXPECTED_ROWS or len(feature_rows) != EXPECTED_ROWS or len(support_rows) != EXPECTED_ROWS:
        raise ValueError("track row count differs from 34,416")
    if any(row.get("used_dev3000") or row.get("dev3000_used") or row.get("used_test") or row.get("test_used") for row in [*fit_rows, *val_rows, *feature_rows, *support_rows]):
        raise ValueError("closed-resource marker found")
    order = [str(row["row_id"]) for row in val_rows]
    feature_order = [str(row["row_id"]) for row in feature_rows]
    support_order = [str(row["row_id"]) for row in support_rows]
    if set(order) != set(feature_order) or set(order) != set(support_order):
        raise ValueError("track row-ID population mismatch")
    # Historical Initial support files are deliberately row-ID sorted, while
    # the manifest retains canonical interaction order. Reindex both immutable
    # inputs to manifest order; Full artifacts are already manifest ordered.
    source_order_restored = order != feature_order or order != support_order
    if args.track == "full" and source_order_restored:
        raise ValueError("Full row/order mismatch")
    return {
        "fit_rows": fit_rows,
        "val_rows": val_rows,
        "features": index_rows(feature_rows, "features"),
        "supports": index_rows(support_rows, "supports"),
        "frequency": index_rows(frequency_rows, "frequency") if frequency_rows else {},
        "order": order,
        "source_order_restored": source_order_restored,
    }


def union_candidates(
    track: str,
    feature: Mapping[str, Any],
    support: Mapping[str, Any],
    frequency: Mapping[str, Any] | None = None,
) -> list[str]:
    values: list[str] = []
    if track == "initial":
        values.extend(map(str, feature["personal_k5"]))
        if frequency is None:
            raise ValueError("Initial frozen Generic surface is missing")
        values.extend(candidate_text(item) for item in frequency["frequency_candidates"])
        for name in INITIAL_BASES:
            values.extend(map(str, feature["bases"][name]["top10"]))
    else:
        values.extend(map(str, feature["personal_k5"]))
        values.extend(candidate_text(item) for item in feature["generic_frequency_candidates"])
        values.extend(candidate_text(item) for item in support["retuned_stage1_candidates"])
    return list(dict.fromkeys(values))


def audit_contexts(track: str, data: Mapping[str, Any]) -> tuple[Any, dict[str, Sequence[Any]], dict[str, list[str]], set[str]]:
    history = base.CausalHistoryIndex([*data["fit_rows"], *data["val_rows"]])
    visible_by_id: dict[str, Sequence[Any]] = {}
    candidates_by_id: dict[str, list[str]] = {}
    contexts: set[str] = set()
    val = index_rows(data["val_rows"], "val")
    for row_id in data["order"]:
        row = val[row_id]
        candidates = union_candidates(
            track,
            data["features"][row_id],
            data["supports"][row_id],
            data["frequency"].get(row_id),
        )
        visible = history.visible_same_pinyin(
            author=str(row["author"]),
            position=int(row["chronological_position"]),
            pinyin=base.pinyin_of(row),
        )
        visible = tuple(item for item in visible if item.record.target in set(candidates))
        visible_by_id[row_id] = visible
        candidates_by_id[row_id] = candidates
        contexts.add(base.context_of(row)[-64:])
        contexts.update(item.record.context[-64:] for item in visible)
    return history, visible_by_id, candidates_by_id, contexts


def cache_coverage(contexts: Iterable[str], stores: Sequence[VectorStore]) -> tuple[int, int]:
    reusable = 0
    missing = 0
    for context in contexts:
        if any(store.get(context) is not None for store in stores):
            reusable += 1
        else:
            missing += 1
    return reusable, missing


def preflight(args: argparse.Namespace, data: Mapping[str, Any], contexts: set[str]) -> None:
    generic_seed = VectorStore(args.generic_seed_cache, read_only=True)
    task_seed = VectorStore(args.task_seed_cache, read_only=True)
    generic_reused, generic_missing = cache_coverage(contexts, [generic_seed])
    task_reused, task_missing = cache_coverage(contexts, [task_seed])
    generic_seed.close()
    task_seed.close()

    sample_n = min(args.sample_contexts, task_missing)
    sample_seconds = 0.0
    if sample_n:
        seed = VectorStore(args.task_seed_cache, read_only=True)
        sample = [context for context in sorted(contexts) if seed.get(context) is None][:sample_n]
        seed.close()
        started = time.perf_counter()
        encoder = SharedContextEncoder(args.task_checkpoint, device="cuda")
        encoder.embed(sample, batch_size=args.task_batch_size)
        sample_seconds = time.perf_counter() - started
    task_estimate = (sample_seconds / sample_n * task_missing) if sample_n else 0.0
    generic_rate = float(args.generic_ms_per_context) / 1000.0
    payload = {
        "schema_version": 1,
        "status": "complete",
        "phase": "preflight",
        "track": args.track,
        "rows": len(data["order"]),
        "source_order_restored": bool(data["source_order_restored"]),
        "unique_required_contexts": len(contexts),
        "generic_cache_reusable": generic_reused,
        "generic_cache_missing": generic_missing,
        "task_cache_reusable": task_reused,
        "task_cache_missing": task_missing,
        "task_sample_contexts": sample_n,
        "task_sample_seconds_including_model_load": sample_seconds,
        "estimated_task_seconds_conservative": task_estimate,
        "estimated_generic_seconds": generic_missing * generic_rate,
        "estimated_incremental_seconds": task_estimate + generic_missing * generic_rate,
        "estimated_vector_disk_bytes": task_missing * (512 * 4 + 160) + generic_missing * (384 * 4 + 160),
        "expected_support_jsonl_bytes": len(data["order"]) * 3500,
        "used_dev3000": False,
        "used_test": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / f"preflight_{args.track}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def fill_task_vectors(args: argparse.Namespace, contexts: set[str]) -> tuple[VectorStore, dict[str, Any]]:
    seed = VectorStore(args.task_seed_cache, read_only=True)
    cache = VectorStore(args.output_root / f"{args.track}_task_vectors.sqlite3", checkpoint=TASK_CHECKPOINT_SHA256)
    copied_contexts = []
    copied_vectors = []
    for context in sorted(contexts):
        if cache.get(context) is None:
            value = seed.get(context)
            if value is not None:
                copied_contexts.append(context)
                copied_vectors.append(value)
    if copied_contexts:
        cache.put_many(copied_contexts, copied_vectors)
    seed.close()
    missing = [context for context in sorted(contexts) if cache.get(context) is None]
    started = time.perf_counter()
    if missing:
        encoder = SharedContextEncoder(args.task_checkpoint, device="cuda")
        for start in range(0, len(missing), args.task_batch_size):
            batch = missing[start : start + args.task_batch_size]
            cache.put_many(batch, encoder.embed(batch, batch_size=args.task_batch_size))
            if args.progress_every and (start // args.task_batch_size + 1) % args.progress_every == 0:
                print(f"Task embeddings {min(start+len(batch),len(missing)):,}/{len(missing):,}", flush=True)
    return cache, {"seed_reused": len(copied_contexts), "fresh": len(missing), "seconds": time.perf_counter() - started}


def fill_generic_vectors(args: argparse.Namespace, contexts: set[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    seed = VectorStore(args.generic_seed_cache, read_only=True)
    cache = VectorStore(args.output_root / f"{args.track}_generic_query_vectors.sqlite3")
    values: dict[str, np.ndarray] = {}
    missing = []
    for context in sorted(contexts):
        value = seed.get(context)
        if value is None:
            value = cache.get(context)
        if value is None:
            missing.append(context)
        else:
            values[context] = value
    started = time.perf_counter()
    if missing:
        if args.generic_bge_model is None:
            raise ValueError(f"generic cache lacks {len(missing)} contexts and --generic-bge-model was omitted")
        from src.personalisation.pilot_a import BGEContextEmbedder

        embedder = BGEContextEmbedder(args.generic_bge_model)
        for number, context in enumerate(missing, start=1):
            value = np.asarray(embedder.embed(context), dtype=np.float32)
            norm = float(np.linalg.norm(value))
            value = value / norm if norm > 0 else value
            cache.put_many([context], [value])
            values[context] = value
            if args.progress_every and number % (args.progress_every * 50) == 0:
                print(f"Generic embeddings {number:,}/{len(missing):,}", flush=True)
    seed.close()
    cache.close()
    return values, {"seed_reused": len(contexts) - len(missing), "fresh": len(missing), "seconds": time.perf_counter() - started}


def score(args: argparse.Namespace, data: Mapping[str, Any], visible: Mapping[str, Sequence[Any]], candidates: Mapping[str, list[str]], contexts: set[str]) -> None:
    checkpoint_sha, _files = sha256_tree(args.task_checkpoint)
    if checkpoint_sha != TASK_CHECKPOINT_SHA256:
        raise ValueError("task checkpoint hash changed")
    started = time.perf_counter()
    task_cache, task_runtime = fill_task_vectors(args, contexts)
    task_vectors = {context: task_cache.get(context) for context in contexts}
    if any(value is None for value in task_vectors.values()):
        raise RuntimeError("task vector cache incomplete")
    generic_vectors, generic_runtime = fill_generic_vectors(args, contexts)
    val = index_rows(data["val_rows"], "val")
    output = []
    for number, row_id in enumerate(data["order"], start=1):
        row = val[row_id]
        names = candidates[row_id]
        query_context = base.context_of(row)[-64:]
        raw = {}
        for label, vectors in (("generic", generic_vectors), ("task", task_vectors)):
            raw[f"{label}_plain"] = cosine_top5_support(
                query_vector=vectors[query_context], candidates=names, visible=visible[row_id], vectors=vectors,
                tau=None, normalize=False,
            )
            raw[f"{label}_recency"] = cosine_top5_support(
                query_vector=vectors[query_context], candidates=names, visible=visible[row_id], vectors=vectors,
                tau=2048.0, normalize=False,
            )
        output.append({
            "schema_version": 1,
            "track": args.track,
            "row_id": row_id,
            "author": str(row["author"]),
            "candidate_union": names,
            "raw_support": raw,
            "gold_used_for_scoring": False,
            "used_dev3000": False,
            "used_test": False,
        })
        if args.progress_every and (number % (args.progress_every * 10) == 0 or number == len(data["order"])):
            print(f"Support {number:,}/{len(data['order']):,}", flush=True)

    # Reconstruct the exact existing recency support before accepting output.
    max_difference = 0.0
    max_detail: dict[str, Any] = {}
    for item in output:
        row_id = str(item["row_id"])
        support = data["supports"][row_id]
        if args.track == "initial":
            for name in INITIAL_BASES:
                surface = list(map(str, data["features"][row_id]["bases"][name]["top10"]))
                actual = restrict_and_normalize(item["raw_support"]["generic_recency"], surface)
                expected = {str(key): float(value) for key, value in support["bases"][name]["support"].items()}
                for key in surface:
                    difference = abs(actual[key] - expected[key])
                    if difference > max_difference:
                        max_difference = difference
                        max_detail = {"row_id": row_id, "base": name, "candidate": key, "actual": actual[key], "expected": expected[key]}
        else:
            surface = [candidate_text(value) for value in support["retuned_stage1_candidates"]]
            actual = restrict_and_normalize(item["raw_support"]["generic_recency"], surface)
            expected = {str(key): float(value) for key, value in support["retuned_bge_support"].items()}
            for key in surface:
                difference = abs(actual[key] - expected[key])
                if difference > max_difference:
                    max_difference = difference
                    max_detail = {"row_id": row_id, "candidate": key, "actual": actual[key], "expected": expected[key]}
    if max_difference > 1e-6:
        row_id = str(max_detail["row_id"])
        row = val[row_id]
        surface = (
            list(map(str, data["features"][row_id]["bases"][str(max_detail["base"])]["top10"]))
            if args.track == "initial"
            else [candidate_text(value) for value in data["supports"][row_id]["retuned_stage1_candidates"]]
        )
        direct, _counts = base.bge_recency_support(
            query_vector=generic_vectors[base.context_of(row)[-64:]],
            candidates=surface,
            visible=visible[row_id],
            vectors=generic_vectors,
        )
        max_detail["direct_frozen_function"] = float(direct[str(max_detail["candidate"])])
        raise RuntimeError(f"frozen Generic-BGE recency reconstruction differs: {max_difference}; {max_detail}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / f"{args.track}_support.jsonl"
    write_jsonl(output_path, output)
    result = {
        "schema_version": 1,
        "status": "complete",
        "track": args.track,
        "rows": len(output),
        "source_order_restored": bool(data["source_order_restored"]),
        "unique_contexts": len(contexts),
        "generic_runtime": generic_runtime,
        "task_runtime": task_runtime,
        "generic_recency_reconstruction_max_abs_difference": max_difference,
        "support_sha256": sha256_file(output_path),
        "runtime_seconds": time.perf_counter() - started,
        "used_dev3000": False,
        "used_test": False,
    }
    write_json(args.output_root / f"support_{args.track}_result.json", result)
    task_cache.close()
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "score"), required=True)
    parser.add_argument("--track", choices=("initial", "full"), required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--existing-support", type=Path, required=True)
    parser.add_argument("--frequency-predictions", type=Path)
    parser.add_argument("--generic-seed-cache", type=Path, required=True)
    parser.add_argument("--task-seed-cache", type=Path, required=True)
    parser.add_argument("--task-checkpoint", type=Path, required=True)
    parser.add_argument("--generic-bge-model", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-batch-size", type=int, default=128)
    parser.add_argument("--sample-contexts", type=int, default=256)
    parser.add_argument("--generic-ms-per-context", type=float, default=2.0)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.fit, args.val, args.stage1, args.existing_support, args.frequency_predictions, args.generic_seed_cache, args.task_seed_cache, args.task_checkpoint, args.output_root):
        if path is not None:
            refuse_closed_path(path)
    data = load_track(args)
    _history, visible, candidates, contexts = audit_contexts(args.track, data)
    if args.phase == "preflight":
        preflight(args, data, contexts)
    else:
        score(args, data, visible, candidates, contexts)


if __name__ == "__main__":
    main()
