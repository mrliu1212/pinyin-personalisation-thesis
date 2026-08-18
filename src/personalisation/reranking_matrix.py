"""Resumable F/M1/M2 ranking-personalisation matrix over frozen T1 conditions."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

from src.datasets.deep_author.pipeline import stable_hash
from src.evaluation.deep_author_v2 import (
    AUTHORS,
    BACKEND_INTEGRATION_REVISION,
    BACKEND_SOURCE_REVISION,
    CHECKPOINT_REVISION,
    CONDITIONS,
    OFFICIAL_CODE_REVISION,
    T1Runner,
    canonical_json,
    conditions_for_anchor,
    load_tokens,
    sha256_file,
    valid_anchors_for_work,
    write_csv,
    write_json,
    write_jsonl,
)
from src.personalisation.candidate_memory_m2 import PairIdentity, rank_m2
from src.personalisation.context_memory import (
    Candidate,
    PredictionQuery,
    assert_candidate_pool,
    macro_author_metrics,
    rank_frequency,
    rank_from_retrieved,
    rank_of,
    retrieve_memory,
    subset_membership,
)
from src.personalisation.h5000 import H5000Runner, T1_MANIFEST_SHA256, T1_PREDICTIONS_SHA256
from src.personalisation.m2_h5000 import M2H5000Runner, M2_LAMBDAS, M2_RETRIEVAL_KS
from src.personalisation.pilot_a import (
    DATASET_V1_BYTES,
    DATASET_V1_SHA256,
    EmbeddingCache,
    EmbeddingLookup,
    FREQUENCY_LAMBDAS,
    HistoryIndex,
    MEMORY_LAMBDAS,
    MEMORY_TOP_NS,
    PilotManifestBuilder,
    split_dev_works,
)


CONDITION_LABELS = {
    "full_short": "Full + Short",
    "initial_short": "Initial + Short",
    "full_multi3": "Full + Multi3",
    "initial_multi3": "Initial + Multi3",
}
HISTORY_BUDGETS: dict[str, int | None] = {"H500": 500, "H5000": 5000, "HFull": None}
METHODS = ("F", "M1", "M2")
REUSED_CELL = ("full_short", "H5000")
SCHEMA_VERSION = 1
EXPERIMENT = "reranking_personalisation_matrix"

FROZEN_ARTIFACTS = {
    "T1": {"predictions.jsonl": T1_PREDICTIONS_SHA256},
    "M1": {
        "frequency_predictions.jsonl": "71c5626b8318e125776c235dd3cccf45677c884deb2699fd8c1f82e907e0abf6",
        "memory_predictions.jsonl": "75907c88f1d099c3dedc6dc71ee4811ca6258d5267c071bf28ef94d6b95e128b",
        "metrics_summary.json": "e35fb9efbe3bdd31d7f8354c227efbed2aa178855061955b3ac16a70137e424d",
    },
    "M2": {
        "m2_predictions.jsonl": "0a199c31e9fc7b9a35c39aef1cdf48f8a8514b1663fb37416844657eacac79fb",
        "metrics_summary.json": "9ad6acecf41b9f36aa1a1bf1bd702cfc729322c4226a4a6a9e3fde4082c6f6d8",
        "selected_hyperparameters.json": "e47e765b950804ceaed2d2fff5a4d2d1dba0ddeb652ac9bf20ccd89a42a182f4",
    },
    "PV": {
        "predictions.jsonl": "cb39d210c2c35453aa40ac250188f237742f0a3c7837c5945bc86721765ff3d7",
        "metrics_summary.json": "ab5566991f85474ecc1dc6f3f6ab7216ee25a02515cad76fb7958218c0e6f29c",
        "selected_hyperparameters.json": "56caf3842d2b0a843f53b4bcb35daaa00fb2125fe26413a4b41541812b5400e3",
    },
}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _release_torch_cuda_cache() -> None:
    """Release PyTorch's inactive CUDA blocks before entering llama.cpp."""

    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass(frozen=True)
class PreparedDevGenericRequest:
    row: Mapping[str, Any]
    segments: tuple[str, ...]
    context: str
    original_context_tokens: int
    used_context_tokens: int
    context_truncated: bool
    prompt_token_length: int


def _generate_compatible_dev_batches(
    backend: Any,
    prepared: Sequence[PreparedDevGenericRequest],
    *,
    batch_size: int = 2,
    on_batch: Any | None = None,
) -> list[tuple[PreparedDevGenericRequest, Any]]:
    """Batch equal model shapes, then restore the caller's stable row order."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    buckets: dict[tuple[int, int], list[tuple[int, PreparedDevGenericRequest]]] = defaultdict(list)
    for index, request in enumerate(prepared):
        buckets[(request.prompt_token_length, len(request.segments))].append((index, request))
    restored: list[tuple[PreparedDevGenericRequest, Any] | None] = [None] * len(prepared)
    for bucket in buckets.values():
        for start in range(0, len(bucket), batch_size):
            batch = bucket[start : start + batch_size]
            results = backend.generate_batch(
                [(request.context, request.segments) for _, request in batch],
                top_k=10,
                beam_size=16,
            )
            if len(results) != len(batch):
                raise RuntimeError("Dev Generic backend returned an unexpected result count")
            completed_batch = []
            for (index, request), result in zip(batch, results):
                restored[index] = (request, result)
                completed_batch.append((request, result))
            if on_batch is not None:
                on_batch(completed_batch)
    if any(value is None for value in restored):
        raise AssertionError("Dev Generic batch restoration is incomplete")
    return [value for value in restored if value is not None]


def deterministic_wrong_user_mapping(authors: Sequence[str] = AUTHORS) -> dict[str, str]:
    ordered = tuple(authors)
    if len(ordered) < 2 or len(set(ordered)) != len(ordered):
        raise ValueError("wrong-user mapping requires at least two distinct users")
    mapping = {author: ordered[(index + 1) % len(ordered)] for index, author in enumerate(ordered)}
    if any(author == wrong for author, wrong in mapping.items()):
        raise AssertionError("wrong-user mapping contains a self mapping")
    return mapping


@dataclass
class RerankingMatrixRunner:
    root: Path
    dataset_root: Path
    pinyingpt_model: Path
    embedding_model: Path
    reranker_model: Path
    t1_predictions: Path
    output_root: Path
    m1_root: Path
    m2_root: Path
    batch_size: int = 32
    max_length: int = 512

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.output_root = Path(self.output_root)
        self.m1 = H5000Runner(self.root, self.dataset_root, self.pinyingpt_model, self.embedding_model, self.m1_root, self.t1_predictions)
        self.m2 = M2H5000Runner(self.m1, self.reranker_model, self.m2_root, batch_size=self.batch_size, max_length=self.max_length)

    @property
    def manifest_path(self) -> Path:
        return self.output_root / "matrix_manifest.json"

    @property
    def embedding_cache_path(self) -> Path:
        return self.m1.embedding_cache_path

    @property
    def pair_cache_path(self) -> Path:
        return self.m2.pair_cache_path

    def _artifact_paths(self) -> dict[str, dict[str, Path]]:
        return {
            "T1": {"predictions.jsonl": self.t1_predictions},
            "M1": {name: self.m1.output_root / name for name in FROZEN_ARTIFACTS["M1"]},
            "M2": {name: self.m2_root / name for name in FROZEN_ARTIFACTS["M2"]},
            "PV": {name: self.root / "results/personalisation/personal_vocabulary_h5000" / name for name in FROZEN_ARTIFACTS["PV"]},
        }

    def verify_prior_artifacts(self) -> dict[str, dict[str, str]]:
        actual: dict[str, dict[str, str]] = {}
        for group, paths in self._artifact_paths().items():
            actual[group] = {}
            for name, path in paths.items():
                if not path.is_file():
                    raise RuntimeError(f"frozen {group} artifact is absent: {path}")
                digest = sha256_file(path)
                if digest != FROZEN_ARTIFACTS[group][name]:
                    raise RuntimeError(f"frozen {group} artifact changed: {name}")
                actual[group][name] = digest
        return actual

    def _work_rows(self) -> list[dict[str, str]]:
        with (self.root / "results/evaluation/deep_author_v2/design/work_split_manifest.csv").open(encoding="utf-8-sig", newline="") as source:
            return list(csv.DictReader(source))

    def _conditions(self) -> list[dict[str, Any]]:
        path = self.root / "results/evaluation/deep_author_v2/design/t1_condition_manifest.jsonl"
        text = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()
        if digest != T1_MANIFEST_SHA256:
            raise RuntimeError("frozen T1 condition manifest SHA-256 mismatch")
        rows = [json.loads(line) for line in text.splitlines()]
        if len(rows) != 24_000 or Counter(row["condition"] for row in rows) != Counter({name: 6000 for name in CONDITIONS}):
            raise RuntimeError("frozen T1 condition population differs")
        return rows

    def _t1_generic(self) -> dict[str, dict[str, Any]]:
        expected = {str(row["condition_id"]): row for row in self._conditions()}
        if sha256_file(self.t1_predictions) != T1_PREDICTIONS_SHA256:
            raise RuntimeError("frozen T1 prediction SHA-256 mismatch")
        completed: dict[str, dict[str, Any]] = {}
        with self.t1_predictions.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                key = str(row["condition_id"])
                if key in completed or key not in expected:
                    raise RuntimeError(f"invalid frozen T1 prediction ID: {key}")
                T1Runner.validate_cached_prediction(row, expected[key])
                completed[key] = row
        if len(completed) != 24_000:
            raise RuntimeError("frozen T1 cache is incomplete")
        return completed

    def _build_manifests(self) -> dict[str, Any]:
        manifest_root = self.output_root / "manifests"
        expected_paths = [manifest_root / f"history_{condition}.jsonl" for condition in CONDITIONS] + [manifest_root / f"dev_{condition}.jsonl" for condition in CONDITIONS]
        if all(path.is_file() for path in expected_paths):
            counts = {path.stem: sum(1 for _ in path.open(encoding="utf-8")) for path in expected_paths}
            return {"status": "reused", "counts": counts}
        dataset = self.dataset_root / "data/processed/deep_author/interactions_t1_ready.jsonl"
        if dataset.stat().st_size != DATASET_V1_BYTES or sha256_file(dataset) != DATASET_V1_SHA256:
            raise RuntimeError("Dataset V1 source differs from the frozen source")
        compatibility = json.loads((self.pinyingpt_model / "pinyin2char.json").read_text(encoding="utf-8"))
        builder = PilotManifestBuilder(self.root, self.dataset_root, self.pinyingpt_model, self.m1_root)
        split_rows = self._work_rows()
        by_condition_history: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
        by_condition_dev: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
        for author in AUTHORS:
            author_works = sorted((row for row in split_rows if row["author"] == author and row["split"] in {"history", "dev"}), key=lambda row: int(row["chronological_index"]))
            dev_works = [row["work_id"] for row in author_works if row["split"] == "dev"]
            tune_works, _ = split_dev_works(dev_works)
            tune_set = set(tune_works)
            for split_row in author_works:
                work_path = self.dataset_root / "data/processed/deep_author/works" / f"{split_row['work_id']}.json"
                work = json.loads(work_path.read_text(encoding="utf-8"))
                anchors = valid_anchors_for_work(work, load_tokens(work_path.with_name(work_path.stem + ".tokens.jsonl")), compatibility)
                for anchor in sorted(anchors, key=lambda row: (int(row["source_position_start"]), str(row["anchor_id"]))):
                    for condition_row in conditions_for_anchor(anchor):
                        condition = str(condition_row["condition"])
                        row = {
                            **condition_row,
                            "row_id": "matrix-dev-" + stable_hash(str(condition_row["condition_id"]), condition)[:24],
                            "work_creation_date": str(split_row["creation_date"]),
                            "work_chronological_index": int(split_row["chronological_index"]),
                            "source_split": str(split_row["split"]),
                            "pinyin_segments": str(condition_row["pinyin_input"]).split(),
                            "target": condition_row["gold"],
                            "chronological_position": int(split_row["chronological_index"]) * 1_000_000_000 + int(condition_row["source_position_start"]),
                        }
                        if split_row["split"] == "history":
                            row["pilot_partition"] = "history"
                            by_condition_history[condition].append(row)
                        else:
                            row["pilot_partition"] = "tune" if row["work_id"] in tune_set else "evaluation"
                            by_condition_dev[condition].append(row)
        manifest_root.mkdir(parents=True, exist_ok=True)
        counts = {}
        for condition in CONDITIONS:
            history = sorted(by_condition_history[condition], key=lambda row: (int(row["chronological_position"]), str(row["row_id"])))
            dev = sorted(by_condition_dev[condition], key=lambda row: (int(row["chronological_position"]), str(row["row_id"])))
            write_jsonl(manifest_root / f"history_{condition}.jsonl", history)
            write_jsonl(manifest_root / f"dev_{condition}.jsonl", dev)
            counts[f"history_{condition}"] = len(history)
            counts[f"dev_{condition}"] = len(dev)
        return {"status": "created", "counts": counts}

    def _history(self, condition: str) -> list[dict[str, Any]]:
        return _read_jsonl(self.output_root / "manifests" / f"history_{condition}.jsonl")

    def _dev(self, condition: str) -> list[dict[str, Any]]:
        return _read_jsonl(self.output_root / "manifests" / f"dev_{condition}.jsonl")

    def _test(self, condition: str) -> list[dict[str, Any]]:
        work = {str(row["work_id"]): row for row in self._work_rows()}
        rows = []
        for value in self._conditions():
            if value["condition"] != condition:
                continue
            split = work[str(value["work_id"])]
            if split["split"] != "test":
                raise RuntimeError("T1 Test condition belongs to a non-Test work")
            rows.append({**value, "row_id": value["condition_id"], "pinyin_segments": str(value["pinyin_input"]).split(), "target": value["gold"], "source_split": "test", "pilot_partition": "test", "chronological_position": int(split["chronological_index"]) * 1_000_000_000 + int(value["source_position_start"])})
        if len(rows) != 6000 or Counter(row["author"] for row in rows) != Counter({author: 1000 for author in AUTHORS}):
            raise RuntimeError(f"invalid Test population for {condition}")
        return rows

    @staticmethod
    def _query(row: Mapping[str, Any], *, author: str | None = None) -> PredictionQuery:
        return PredictionQuery(str(row["row_id"]), author or str(row["author"]), str(row["work_id"]), int(row["chronological_position"]), str(row["context"]), tuple(row["pinyin_segments"]))

    @staticmethod
    def _candidates(row: Mapping[str, Any]) -> tuple[Candidate, ...]:
        return tuple(Candidate(str(value["text"]), int(value["rank"]), float(value["log_probability"])) for value in row["top10_candidates"])

    def _generic_dev_path(self, condition: str) -> Path:
        return self.output_root / "cache/dev_generic" / f"{condition}.jsonl"

    def _generic_dev_partial_path(self, condition: str) -> Path:
        return self.output_root / "cache/dev_generic" / f"{condition}.partial.jsonl"

    def _load_dev_generic(self, condition: str) -> dict[str, dict[str, Any]]:
        result = {}
        expected = {str(row["row_id"]): row for row in self._dev(condition) if row["pilot_partition"] == "tune"}
        for path in (self._generic_dev_path(condition), self._generic_dev_partial_path(condition)):
            if not path.is_file():
                continue
            for value in _read_jsonl(path):
                key = str(value["row_id"])
                if key not in expected:
                    raise RuntimeError(f"invalid matrix Dev Generic row: {key}")
                if key in result:
                    if result[key] != value:
                        raise RuntimeError(f"conflicting matrix Dev Generic row: {key}")
                    continue
                result[key] = value
            if len(result) == len(expected):
                break
        return result

    def _seed_full_short_dev(self) -> int:
        path = self._generic_dev_path("full_short")
        if path.is_file():
            return len(self._load_dev_generic("full_short"))
        existing_path = self.m1_root / "cache/generic_predictions.jsonl"
        existing = {str(row["condition_id"]): row for row in _read_jsonl(existing_path)}
        seeded = []
        for row in self._dev("full_short"):
            if row["pilot_partition"] != "tune":
                continue
            source = existing.get(str(row["condition_id"]))
            if source is None:
                continue
            seeded.append({**source, **{key: row[key] for key in ("row_id", "condition_id", "anchor_id", "author", "work_id", "chronological_position", "context", "pinyin_input", "pinyin_segments", "gold", "target", "pilot_partition")}})
        write_jsonl(path, seeded)
        return len(seeded)

    def ensure_dev_generic(self, condition: str, *, backend: Any | None = None) -> dict[str, Any]:
        if condition == "full_short":
            self._seed_full_short_dev()
        rows = [row for row in self._dev(condition) if row["pilot_partition"] == "tune"]
        completed = self._load_dev_generic(condition)
        primary_path = self._generic_dev_path(condition)
        primary_rows = _read_jsonl(primary_path) if primary_path.is_file() else []
        primary_ids = [str(row["row_id"]) for row in primary_rows]
        if primary_ids != [str(row["row_id"]) for row in rows[: len(primary_ids)]]:
            raise RuntimeError("Dev Generic cache rows are not in frozen Dev order")
        pending = [row for row in rows if row["row_id"] not in completed]
        owns_backend = False
        print(f"matrix Dev Generic {condition}: required={len(rows)} reused={len(completed)} missing={len(pending)}", flush=True)
        if pending:
            if backend is None:
                from src.reference_backend_pinyingpt import PinyinGPTConcatBackend
                backend = PinyinGPTConcatBackend(self.pinyingpt_model, device="cuda")
                owns_backend = True
            primary_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path = self._generic_dev_partial_path(condition)
            prepared: list[PreparedDevGenericRequest] = []
            for row in pending:
                segments = list(row["pinyin_segments"])
                context, original, used, truncated = backend.truncate_context_for_generation(row["context"], segments)
                prompt, _ = backend._prompt(context, segments)
                prepared.append(PreparedDevGenericRequest(row, tuple(segments), context, original, used, truncated, len(prompt)))

            with partial_path.open("a", encoding="utf-8", newline="\n") as destination:
                def persist_batch(batch: Sequence[tuple[PreparedDevGenericRequest, Any]]) -> None:
                    for value, result in batch:
                        row = value.row
                        candidates = [candidate.to_dict() for candidate in result.candidates]
                        output = {**row, "model_used_context": value.context, "original_stored_context_tokens": value.original_context_tokens, "model_used_context_tokens": value.used_context_tokens, "context_truncated": value.context_truncated, "top10_candidates": candidates, "gold_rank": next((candidate["rank"] for candidate in candidates if candidate["text"] == row["gold"]), None), "beam_size": 16, "top_k": 10, "runtime_device": result.runtime_device, "checkpoint_revision": CHECKPOINT_REVISION, "official_code_revision": OFFICIAL_CODE_REVISION, "backend_source_revision": BACKEND_SOURCE_REVISION, "backend_integration_revision": BACKEND_INTEGRATION_REVISION}
                        destination.write(canonical_json(output) + "\n")
                        completed[str(row["row_id"])] = output
                    destination.flush()
                    if len(completed) % 100 < len(batch) or len(completed) == len(rows):
                        print(f"matrix Dev Generic {condition}: {len(completed)}/{len(rows)}", flush=True)
                _generate_compatible_dev_batches(backend, prepared, on_batch=persist_batch)
        if len(completed) == len(rows) and (pending or len(primary_rows) != len(rows)):
            temporary = primary_path.with_suffix(primary_path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8", newline="\n") as destination:
                for row in rows:
                    destination.write(canonical_json(completed[str(row["row_id"])]) + "\n")
            temporary.replace(primary_path)
        if owns_backend:
            backend = None
            _release_torch_cuda_cache()
        return {"required": len(rows), "reused_at_start": len(rows) - len(pending), "added": len(pending), "complete": len(completed) == len(rows)}

    def _condition_required_contexts(self, condition: str) -> set[str]:
        cache_path = self.output_root / "cache" / f"required_contexts_{condition}.json"
        if cache_path.is_file():
            values = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise RuntimeError(f"invalid required-context cache: {cache_path}")
            return set(values)
        contexts: set[str] = set()
        history = self._history(condition)
        dev = self._dev(condition)
        tune = [row for row in dev if row["pilot_partition"] == "tune"]
        test = self._test(condition)
        dev_index = HistoryIndex(history + dev, None)
        test_index = HistoryIndex(history, None)
        for row in tune:
            contexts.add(str(row["context"]))
            contexts.update(str(value["context"]) for value in dev_index.visible(self._query(row)))
        for row in test:
            contexts.add(str(row["context"]))
            contexts.update(str(value["context"]) for value in test_index.visible(self._query(row)))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(sorted(contexts), ensure_ascii=False, separators=(",", ":")), encoding="utf-8", newline="\n")
        temporary.replace(cache_path)
        return contexts

    def _required_contexts(self) -> set[str]:
        contexts: set[str] = set()
        for condition in CONDITIONS:
            contexts.update(self._condition_required_contexts(condition))
        return contexts

    def _pair_work_upper_bound(self) -> int:
        """Conservative pre-embedding bound; actual semantic keys deduplicate further."""

        total = 0
        for condition in CONDITIONS:
            history = self._history(condition)
            dev = self._dev(condition)
            tune = [row for row in dev if row["pilot_partition"] == "tune"]
            test = self._test(condition)
            for budget, limit in HISTORY_BUDGETS.items():
                if (condition, budget) == REUSED_CELL:
                    continue
                dev_index = HistoryIndex(history + dev, limit)
                test_index = HistoryIndex(history, limit)
                total += sum(min(max(M2_RETRIEVAL_KS), len(dev_index.visible(self._query(row)))) for row in tune)
                total += sum(min(max(M2_RETRIEVAL_KS), len(test_index.visible(self._query(row)))) for row in test)
        wrong_index = HistoryIndex(self._history("full_short"), None)
        mapping = deterministic_wrong_user_mapping()
        total += sum(min(max(M2_RETRIEVAL_KS), len(wrong_index.visible(self._query(row, author=mapping[str(row["author"])])))) for row in self._test("full_short"))
        return total

    def _ensure_embedding_values(self, required: Iterable[str]) -> dict[str, Any]:
        contexts = sorted(set(required), key=lambda value: hashlib.sha256(value.encode()).hexdigest())
        cache = EmbeddingCache(self.embedding_cache_path)
        hits = sum(cache.get(value) is not None for value in contexts)
        missing = [value for value in contexts if cache.get(value) is None]
        added = 0
        try:
            if missing:
                from src.personalisation.pilot_a import BGEContextEmbedder
                embedder = BGEContextEmbedder(self.embedding_model)
                for index, context in enumerate(missing, 1):
                    cache.put(context, embedder.embed(context))
                    added += 1
                    if index % 100 == 0:
                        cache.commit()
                        print(f"matrix BGE added={index}/{len(missing)}", flush=True)
                cache.commit()
            return {"required_unique_contexts": len(contexts), "cache_hits": hits, "cache_misses": len(missing), "new_embeddings_computed": added, "final_cache_rows": cache.count(), "cache_path": str(cache.path)}
        finally:
            cache.close()

    def ensure_embeddings(self, condition: str | None = None) -> dict[str, Any]:
        required = self._condition_required_contexts(condition) if condition else self._required_contexts()
        return self._ensure_embedding_values(required)

    def _initial_manifest(self) -> dict[str, Any]:
        cells = []
        for condition in CONDITIONS:
            for budget in HISTORY_BUDGETS:
                for method in METHODS:
                    reused = (condition, budget) == REUSED_CELL
                    cells.append({"condition": condition, "history_budget": budget, "method": method, "state": "reused_complete" if reused else "pending", "dev_selection_required": not reused, "test_required": not reused, "output_path": str(self._cell_root(condition, budget, method)), "existing_artifact_path": str(self.m2_root if method == "M2" else self.m1.output_root) if reused else None, "selected_hyperparameters": None, "error": None})
        return {"schema_version": SCHEMA_VERSION, "experiment": EXPERIMENT, "status": "prepared", "cells": cells, "generic_test_inference_rows": 0, "prior_artifact_sha256": self.verify_prior_artifacts(), "updated_at": datetime.now(timezone.utc).isoformat()}

    def _manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            value = self._initial_manifest()
            _atomic_json(self.manifest_path, value)
            return value
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(self.manifest_path, manifest)

    def _cell(self, manifest: Mapping[str, Any], condition: str, budget: str, method: str) -> dict[str, Any]:
        return next(value for value in manifest["cells"] if (value["condition"], value["history_budget"], value["method"]) == (condition, budget, method))

    def _cell_root(self, condition: str, budget: str, method: str) -> Path:
        return self.output_root / "cells" / condition / budget / method

    def audit(self, *, estimate_pairs: bool = True) -> dict[str, Any]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        prior = self.verify_prior_artifacts()
        manifests = self._build_manifests()
        self._seed_full_short_dev()
        generic = self._t1_generic()
        required_contexts = self._required_contexts()
        cache = EmbeddingCache(self.embedding_cache_path)
        try:
            embedding_hits = sum(cache.get(value) is not None for value in required_contexts)
        finally:
            cache.close()
        pair_cache = self.m2._new_pair_cache()
        pair_rows = pair_cache.count()
        pair_cache.close()
        pair_upper_bound = self._pair_work_upper_bound() if estimate_pairs else None
        manifest = self._manifest()
        result = {"schema_version": SCHEMA_VERSION, "status": "passed", "prior_artifact_sha256": prior, "manifests": manifests, "generic": {"required_test_rows": 24000, "cached_test_rows": len(generic), "missing_test_rows": 24000 - len(generic), "estimated_new_test_inference_rows": 0, "dev_tune_rows_by_condition": {condition: sum(row["pilot_partition"] == "tune" for row in self._dev(condition)) for condition in CONDITIONS}, "dev_cached_by_condition": {condition: len(self._load_dev_generic(condition)) for condition in CONDITIONS}}, "bge": {"required_unique_contexts": len(required_contexts), "cache_hits": embedding_hits, "cache_misses": len(required_contexts) - embedding_hits, "estimated_new_embeddings": len(required_contexts) - embedding_hits, "cache_path": str(self.embedding_cache_path)}, "m2": {"existing_pair_cache_rows": pair_rows, "estimated_reusable_pair_scores_available": pair_rows, "estimated_total_pair_requests_upper_bound": pair_upper_bound, "estimated_new_pair_scores_upper_bound": None if pair_upper_bound is None else max(0, pair_upper_bound - pair_rows), "estimate_note": "conservative pre-BGE bound; actual Stage-1 semantic keys deduplicate across budgets", "cache_path": str(self.pair_cache_path)}, "cells": {"total": 36, "reused_complete": sum(row["state"] == "reused_complete" for row in manifest["cells"]), "new_required": sum(row["state"] != "reused_complete" for row in manifest["cells"])}, "generic_test_inference_rows": 0, "completed_full_short_h5000_recomputed": False}
        # New conditions can legitimately require new semantic contexts. The
        # audit blocks only missing frozen Test Generic rows or failure to reuse
        # the complete known BGE cache, not newly required content.
        result["bge"]["existing_cache_rows_reused"] = embedding_hits
        result["bge"]["existing_content_recomputed"] = 0
        if result["generic"]["missing_test_rows"] or embedding_hits < 39_680:
            result["status"] = "blocked"
        _atomic_json(self.output_root / "audit_summary.json", result)
        print(f"matrix audit: status={result['status']} generic_test={len(generic)}/24000 bge={embedding_hits}/{len(required_contexts)} reused_cells=3 new_cells=33 pair_cache_rows={pair_rows}", flush=True)
        return result

    def _ensure_pairs(self, pairs: Sequence[PairIdentity], label: str) -> dict[str, int]:
        cache = self.m2._new_pair_cache()
        unique = {cache.key(pair): pair for pair in pairs}
        hits = sum(cache.get(pair) is not None for pair in unique.values())
        pending = [pair for pair in unique.values() if cache.get(pair) is None]
        added = 0
        try:
            if pending:
                reranker = self.m2._new_reranker()
                reranker.load()
                for start in range(0, len(pending), self.batch_size):
                    batch = pending[start : start + self.batch_size]
                    prepared = [reranker.prepare(pair) for pair in batch]
                    for pair, prepared_pair, score in zip(batch, prepared, reranker.score_prepared(prepared)):
                        cache.put(pair, prepared_pair, score)
                        added += 1
                    if added % (self.batch_size * 10) == 0 or added == len(pending):
                        cache.commit()
                        print(f"matrix M2 {label}: {added}/{len(pending)} new pairs", flush=True)
            return {"requested_unique_pairs": len(unique), "cache_hits": hits, "cache_misses": len(pending), "pairs_added": added}
        finally:
            cache.close()

    def _prepare_rows(self, condition: str, budget: str, rows: Sequence[Mapping[str, Any]], history: Sequence[Mapping[str, Any]], generic: Mapping[str, Mapping[str, Any]], lookup: EmbeddingLookup, *, wrong_mapping: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
        index = HistoryIndex(history, HISTORY_BUDGETS[budget])
        states = []
        for row in rows:
            query = self._query(row, author=wrong_mapping.get(str(row["author"])) if wrong_mapping else None)
            visible = index.visible(query)
            candidates = self._candidates(generic[str(row["row_id"])])
            retrieved = retrieve_memory(query, visible, lookup) if visible else ()
            states.append({"row": row, "query": query, "visible": visible, "candidates": candidates, "retrieved": retrieved, "flags": subset_membership(query, str(row["gold"]), visible)})
        return states

    @staticmethod
    def _grid_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return macro_author_metrics(rows, "rank")["macro_author"]

    def _tune(self, condition: str, budget: str, states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        f_rows = {value: [] for value in FREQUENCY_LAMBDAS}
        m1_rows = {(top_n, value): [] for top_n in MEMORY_TOP_NS for value in MEMORY_LAMBDAS}
        pairs = []
        for state in states:
            by_id = {str(value["row_id"]): value for value in state["visible"]}
            pairs.extend(
                self.m2._pair(state["query"], by_id[str(value["historical_interaction_id"])])
                for value in state["retrieved"][: max(M2_RETRIEVAL_KS)]
            )
        pair_stats = self._ensure_pairs(pairs, f"dev-{condition}-{budget}")
        pair_cache = self.m2._new_pair_cache()
        m2_rows = {(k, value): [] for k in M2_RETRIEVAL_KS for value in M2_LAMBDAS}
        try:
            for state in states:
                row, query, candidates, visible, retrieved = state["row"], state["query"], state["candidates"], state["visible"], state["retrieved"]
                for value in FREQUENCY_LAMBDAS:
                    f_rows[value].append({"author": row["author"], "rank": rank_of(rank_frequency(query, candidates, visible, lambda_frequency=value), row["gold"])})
                for top_n in MEMORY_TOP_NS:
                    for value in MEMORY_LAMBDAS:
                        m1_rows[(top_n, value)].append({"author": row["author"], "rank": rank_of(rank_from_retrieved(candidates, retrieved[:top_n], lambda_memory=value), row["gold"])})
                by_id = {str(value["row_id"]): value for value in visible}
                stage1 = [by_id[str(value["historical_interaction_id"])] for value in retrieved[:max(M2_RETRIEVAL_KS)]]
                for k in M2_RETRIEVAL_KS:
                    evidence = self.m2._evidence(query, stage1[:k], pair_cache)
                    for value in M2_LAMBDAS:
                        m2_rows[(k, value)].append({"author": row["author"], "rank": rank_of(rank_m2(candidates, evidence, lambda_m2=value), row["gold"])})
        finally:
            pair_cache.close()
        f_search = [{"lambda_frequency": value, **self._grid_metrics(f_rows[value])} for value in FREQUENCY_LAMBDAS]
        m1_search = [{"top_n": top_n, "lambda_memory": value, **self._grid_metrics(m1_rows[(top_n, value)])} for top_n in MEMORY_TOP_NS for value in MEMORY_LAMBDAS]
        m2_search = [{"retrieval_k": k, "lambda_m2": value, **self._grid_metrics(m2_rows[(k, value)])} for k in M2_RETRIEVAL_KS for value in M2_LAMBDAS]
        f = max(f_search, key=lambda row: (row["top1"], -row["lambda_frequency"]))
        m1 = max(m1_search, key=lambda row: (row["top1"], -row["lambda_memory"], -row["top_n"]))
        m2 = max(m2_search, key=lambda row: (row["top1"], -row["lambda_m2"], -row["retrieval_k"]))
        selection = {"status": "complete", "condition": condition, "history_budget": budget, "selection_population": "chronologically earlier whole-work Dev tune partition", "selection_metric": "Macro-author Top-1", "test_rows_seen_during_selection": 0, "test_gold_used_for_selection": False, "tune_rows": len(states), "F": {"lambda_frequency": f["lambda_frequency"]}, "M1": {"top_n": m1["top_n"], "lambda_memory": m1["lambda_memory"]}, "M2": {"retrieval_k": m2["retrieval_k"], "lambda_m2": m2["lambda_m2"]}, "pair_cache": pair_stats}
        selection_root = self.output_root / "selections" / condition / budget
        write_csv(selection_root / "f_search.csv", f_search, list(f_search[0]))
        write_csv(selection_root / "m1_search.csv", m1_search, list(m1_search[0]))
        write_csv(selection_root / "m2_search.csv", m2_search, list(m2_search[0]))
        _atomic_json(selection_root / "selected.json", selection)
        return selection

    def _evaluate_method(self, method: str, states: Sequence[Mapping[str, Any]], parameters: Mapping[str, Any], condition: str, budget: str, *, wrong_user: bool = False) -> dict[str, Any]:
        pair_cache = self.m2._new_pair_cache() if method == "M2" else None
        predictions = []
        metric_rows = []
        try:
            if method == "M2":
                pairs = []
                for state in states:
                    by_id = {str(value["row_id"]): value for value in state["visible"]}
                    pairs.extend(self.m2._pair(state["query"], by_id[str(value["historical_interaction_id"])]) for value in state["retrieved"][:int(parameters["retrieval_k"])])
                if pair_cache:
                    pair_cache.close()
                pair_stats = self._ensure_pairs(pairs, f"test-{condition}-{budget}{'-wrong' if wrong_user else ''}")
                pair_cache = self.m2._new_pair_cache()
            else:
                pair_stats = None
            for state in states:
                row, query, candidates, visible, retrieved = state["row"], state["query"], state["candidates"], state["visible"], state["retrieved"]
                if method == "F":
                    ranked = rank_frequency(query, candidates, visible, lambda_frequency=float(parameters["lambda_frequency"]))
                elif method == "M1":
                    ranked = rank_from_retrieved(candidates, retrieved[:int(parameters["top_n"])], lambda_memory=float(parameters["lambda_memory"]))
                else:
                    by_id = {str(value["row_id"]): value for value in visible}
                    histories = [by_id[str(value["historical_interaction_id"])] for value in retrieved[:int(parameters["retrieval_k"])]]
                    ranked = rank_m2(candidates, self.m2._evidence(query, histories, pair_cache), lambda_m2=float(parameters["lambda_m2"]))
                assert_candidate_pool(candidates, ranked, ranked)
                gold_rank = rank_of(ranked, str(row["gold"]))
                generic_rank = next((candidate.generic_rank for candidate in candidates if candidate.text == str(row["gold"])), None)
                if (generic_rank is None) != (gold_rank is None):
                    raise AssertionError("Missing@10 changed despite the frozen Generic candidate surface")
                common = {"condition_id": row["condition_id"], "anchor_id": row["anchor_id"], "author": row["author"], "work_id": row["work_id"], "gold": row["gold"], **state["flags"]}
                metric_rows.append({**common, "rank": gold_rank})
                predictions.append({**common, "condition": condition, "history_budget": budget, "method": method, "selected_hyperparameters": dict(parameters), "candidates": ranked, "gold_rank": gold_rank, "wrong_user_control": wrong_user})
        finally:
            if pair_cache is not None:
                pair_cache.close()
        subsets = {"overall": metric_rows, "history_available": [row for row in metric_rows if row["history_available"]], "ambiguous": [row for row in metric_rows if row["ambiguous"]], "conflict": [row for row in metric_rows if row["conflict"]]}
        metrics = {name: macro_author_metrics(values, "rank") for name, values in subsets.items()}
        root = self._cell_root(condition, budget, method) if not wrong_user else self.output_root / "wrong_user" / method
        write_jsonl(root / "predictions.jsonl", predictions)
        result = {"schema_version": SCHEMA_VERSION, "status": "complete", "condition": condition, "history_budget": budget, "method": method, "rows": len(metric_rows), "per_author_rows": dict(Counter(row["author"] for row in metric_rows)), "selected_hyperparameters": dict(parameters), "metrics": metrics, "subset_rows": {name: len(values) for name, values in subsets.items()}, "candidate_pool_invariant": True, "generic_rows_reused": len(metric_rows), "generic_test_inference_rows": 0, "test_gold_used_for_tuning": False, "pair_cache": pair_stats, "wrong_user_control": wrong_user}
        _atomic_json(root / "result.json", result)
        return result

    def _reused_results(self) -> dict[str, dict[str, Any]]:
        m1 = json.loads((self.m1.output_root / "metrics_summary.json").read_text(encoding="utf-8"))
        m2 = json.loads((self.m2_root / "metrics_summary.json").read_text(encoding="utf-8"))
        return {"F": {"parameters": {"lambda_frequency": 4.0}, "metrics": {subset: m1["metrics"][subset]["F-H5000"] for subset in m1["metrics"]}, "subset_rows": m1["subset_rows"]}, "M1": {"parameters": {"top_n": 5, "lambda_memory": 4.0}, "metrics": {subset: m1["metrics"][subset]["M1-H5000"] for subset in m1["metrics"]}, "subset_rows": m1["subset_rows"]}, "M2": {"parameters": {"retrieval_k": 20, "lambda_m2": 4.0}, "metrics": {subset: m2["metrics"][subset]["M2-H5000"] for subset in m2["metrics"]}, "subset_rows": m2["subset_rows"]}}

    def run_cell_group(self, condition: str, budget: str) -> dict[str, Any]:
        if (condition, budget) == REUSED_CELL:
            return {"status": "reused_complete", "methods": list(METHODS)}
        manifest = self._manifest()
        pending = [method for method in METHODS if self._cell(manifest, condition, budget, method)["state"] != "complete"]
        if not pending:
            return {"status": "complete", "methods": []}
        for method in pending:
            self._cell(manifest, condition, budget, method)["state"] = "running"
        self._save_manifest(manifest)
        try:
            self.ensure_dev_generic(condition)
            self.ensure_embeddings(condition)
            dev_rows = [row for row in self._dev(condition) if row["pilot_partition"] == "tune"]
            test_rows = self._test(condition)
            dev_generic = self._load_dev_generic(condition)
            test_generic_all = self._t1_generic()
            test_generic = {str(row["row_id"]): test_generic_all[str(row["condition_id"])] for row in test_rows}
            history = self._history(condition)
            embedding_cache = EmbeddingCache(self.embedding_cache_path)
            lookup = EmbeddingLookup(embedding_cache)
            try:
                dev_states = self._prepare_rows(condition, budget, dev_rows, history + self._dev(condition), dev_generic, lookup)
                selection = self._tune(condition, budget, dev_states)
                test_states = self._prepare_rows(condition, budget, test_rows, history, test_generic, lookup)
                results = {method: self._evaluate_method(method, test_states, selection[method], condition, budget) for method in pending}
            finally:
                embedding_cache.close()
            manifest = self._manifest()
            for method, result in results.items():
                cell = self._cell(manifest, condition, budget, method)
                cell.update({"state": "complete", "selected_hyperparameters": result["selected_hyperparameters"], "error": None})
            self._save_manifest(manifest)
            return {"status": "complete", "methods": list(results)}
        except Exception:
            error = traceback.format_exc()
            manifest = self._manifest()
            for method in pending:
                cell = self._cell(manifest, condition, budget, method)
                if cell["state"] == "running":
                    cell.update({"state": "failed", "error": error})
            self._save_manifest(manifest)
            raise

    def smoke(self) -> dict[str, Any]:
        audit_path = self.output_root / "audit_summary.json"
        if not audit_path.is_file():
            raise RuntimeError("run --phase audit before smoke")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit["status"] != "passed":
            raise RuntimeError("matrix audit must pass before smoke")
        self.verify_prior_artifacts()
        self.ensure_dev_generic("full_short")
        dev = self._dev("full_short")
        history = self._history("full_short")
        smoke_index = HistoryIndex(history + dev, 500)
        rows = [
            next(
                row
                for row in dev
                if row["pilot_partition"] == "tune"
                and row["author"] == author
                and smoke_index.visible(self._query(row))
            )
            for author in AUTHORS
        ]
        smoke_contexts = {str(row["context"]) for row in rows}
        for row in rows:
            smoke_contexts.update(str(value["context"]) for value in smoke_index.visible(self._query(row)))
        embedding_reuse = self._ensure_embedding_values(smoke_contexts)
        generic = self._load_dev_generic("full_short")
        cache = EmbeddingCache(self.embedding_cache_path)
        try:
            states = self._prepare_rows("full_short", "H500", rows, history + dev, generic, EmbeddingLookup(cache))
            selection = {"F": {"lambda_frequency": 4.0}, "M1": {"top_n": 5, "lambda_memory": 4.0}, "M2": {"retrieval_k": 10, "lambda_m2": 1.0}}
            results = {method: self._evaluate_method(method, states, selection[method], "smoke_full_short", "H500") for method in METHODS}
        finally:
            cache.close()
        summary = {"status": "passed", "research_result": False, "rows": len(rows), "authors": list(AUTHORS), "methods": list(results), "generic_inference_rows": 0, "embedding_reuse": embedding_reuse, "current_gold_in_prediction_input": False, "candidate_pool_invariant": True}
        _atomic_json(self.output_root / "smoke_summary.json", summary)
        return summary

    def wrong_user_control(self) -> dict[str, Any]:
        condition, budget = "full_short", "HFull"
        selection = json.loads((self.output_root / "selections" / condition / budget / "selected.json").read_text(encoding="utf-8"))
        rows = self._test(condition)
        generic_all = self._t1_generic()
        generic = {str(row["row_id"]): generic_all[str(row["condition_id"])] for row in rows}
        history = self._history(condition)
        mapping = deterministic_wrong_user_mapping()
        cache = EmbeddingCache(self.embedding_cache_path)
        try:
            states = self._prepare_rows(condition, budget, rows, history, generic, EmbeddingLookup(cache), wrong_mapping=mapping)
            wrong = {method: self._evaluate_method(method, states, selection[method], condition, budget, wrong_user=True) for method in METHODS}
        finally:
            cache.close()
        correct = {method: json.loads((self._cell_root(condition, budget, method) / "result.json").read_text(encoding="utf-8")) for method in METHODS}
        result = {"status": "complete", "condition": condition, "history_budget": budget, "mapping": mapping, "mapping_rule": "cyclic frozen AUTHORS order", "methods": {method: {"correct": correct[method]["metrics"]["overall"], "wrong": wrong[method]["metrics"]["overall"], "top1_delta_correct_minus_wrong": correct[method]["metrics"]["overall"]["macro_author"]["top1"] - wrong[method]["metrics"]["overall"]["macro_author"]["top1"]} for method in METHODS}}
        _atomic_json(self.output_root / "wrong_user_summary.json", result)
        return result

    def finalize(self) -> dict[str, Any]:
        manifest = self._manifest()
        failed = [cell for cell in manifest["cells"] if cell["state"] not in {"complete", "reused_complete"}]
        if failed:
            completion = {"status": "incomplete", "completed_cell_count": 36 - len(failed), "failed_cell_count": len(failed), "failed_cells": [{key: cell[key] for key in ("condition", "history_budget", "method", "state", "error")} for cell in failed]}
            _atomic_json(self.output_root / "COMPLETE.json", completion)
            return completion
        reused = self._reused_results()
        cell_results = []
        selections = {}
        for condition in CONDITIONS:
            for budget in HISTORY_BUDGETS:
                for method in METHODS:
                    if (condition, budget) == REUSED_CELL:
                        value = reused[method]
                        result = {"condition": condition, "history_budget": budget, "method": method, "selected_hyperparameters": value["parameters"], "metrics": value["metrics"], "subset_rows": value["subset_rows"]}
                    else:
                        result = json.loads((self._cell_root(condition, budget, method) / "result.json").read_text(encoding="utf-8"))
                    cell_results.append(result)
                    selections[f"{condition}|{budget}|{method}"] = result["selected_hyperparameters"]
        generic = self._t1_generic()
        matrix_rows = []
        learning = []
        diagnostics = []
        for result in cell_results:
            overall = result["metrics"]["overall"]["macro_author"]
            matrix_rows.append({"condition": result["condition"], "history_budget": result["history_budget"], "method": result["method"], "selected_hyperparameters": canonical_json(result["selected_hyperparameters"]), **overall, **{f"{subset}_top1": result["metrics"][subset]["macro_author"]["top1"] for subset in ("history_available", "ambiguous", "conflict")}, **{f"{subset}_rows": result["subset_rows"][subset] for subset in ("history_available", "ambiguous", "conflict")}})
            learning.append({"condition": result["condition"], "method": result["method"], "history_budget": result["history_budget"], **overall})
        for condition in CONDITIONS:
            g_rows = [row for row in generic.values() if row["condition"] == condition]
            g_metrics = macro_author_metrics([{"author": row["author"], "rank": row["gold_rank"]} for row in g_rows], "rank")["macro_author"]
            for method in METHODS:
                learning.append({"condition": condition, "method": method, "history_budget": "H0", **g_metrics})
            for budget in HISTORY_BUDGETS:
                values = {row["method"]: row for row in matrix_rows if row["condition"] == condition and row["history_budget"] == budget}
                diagnostics.append({"condition": condition, "history_budget": budget, "ambiguous_rows": values["F"]["ambiguous_rows"], "conflict_rows": values["F"]["conflict_rows"], **{f"{method.lower()}_ambiguous_top1": values[method]["ambiguous_top1"] for method in METHODS}, **{f"{method.lower()}_conflict_top1": values[method]["conflict_top1"] for method in METHODS}, "f_minus_m1_ambiguous_top1": values["F"]["ambiguous_top1"] - values["M1"]["ambiguous_top1"], "f_minus_m2_ambiguous_top1": values["F"]["ambiguous_top1"] - values["M2"]["ambiguous_top1"], "m1_minus_m2_ambiguous_top1": values["M1"]["ambiguous_top1"] - values["M2"]["ambiguous_top1"], "f_minus_m1_conflict_top1": values["F"]["conflict_top1"] - values["M1"]["conflict_top1"], "f_minus_m2_conflict_top1": values["F"]["conflict_top1"] - values["M2"]["conflict_top1"], "m1_minus_m2_conflict_top1": values["M1"]["conflict_top1"] - values["M2"]["conflict_top1"]})
        write_csv(self.output_root / "condition_matrix.csv", matrix_rows, list(matrix_rows[0]))
        write_csv(self.output_root / "learning_curves.csv", sorted(learning, key=lambda row: (row["condition"], row["method"], ("H0", "H500", "H5000", "HFull").index(row["history_budget"]))), list(learning[0]))
        write_csv(self.output_root / "context_diagnostics.csv", diagnostics, list(diagnostics[0]))
        _atomic_json(self.output_root / "selected_hyperparameters.json", selections)
        wrong = self.wrong_user_control()
        audit = json.loads((self.output_root / "audit_summary.json").read_text(encoding="utf-8"))
        prior = self.verify_prior_artifacts()
        summary = {"status": "complete", "rows_per_cell": 6000, "cells": 36, "reused_cells": 3, "new_cells": 33, "generic_test_rows_reused": 24000, "generic_test_inference_rows": 0, "test_gold_used_for_tuning": False, "prior_artifacts_unchanged": True, "prior_artifact_sha256": prior, "wrong_user_control": wrong, "audit": audit}
        _atomic_json(self.output_root / "metrics_summary.json", summary)
        _atomic_json(self.output_root / "cache_reuse_summary.json", {"generic": audit["generic"], "bge": audit["bge"], "m2": audit["m2"]})
        _atomic_json(self.output_root / "runtime_summary.json", {"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat()})
        artifacts = [path for path in self.output_root.iterdir() if path.is_file() and path.name not in {"artifact_checksums.json", "COMPLETE.json"} and path.suffix in {".json", ".csv"}]
        _atomic_json(self.output_root / "artifact_checksums.json", {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(artifacts)})
        complete = {"status": "complete", "completion_timestamp": datetime.now(timezone.utc).isoformat(), "completed_cell_count": 36, "failed_cell_count": 0, "generic_inference_rows": 0, "embedding_cache_hits": audit["bge"]["cache_hits"], "embedding_cache_misses": audit["bge"]["cache_misses"], "m2_pair_cache_rows_at_audit": audit["m2"]["existing_pair_cache_rows"], "prior_artifacts_unchanged": True, "test_gold_used_for_tuning": False}
        _atomic_json(self.output_root / "COMPLETE.json", complete)
        manifest["status"] = "complete"
        self._save_manifest(manifest)
        return complete

    def all(self) -> dict[str, Any]:
        audit = self.audit()
        if audit["status"] != "passed":
            raise RuntimeError("matrix audit blocked the long run")
        order = (("full_short", "H500"), ("full_short", "HFull")) + tuple((condition, budget) for condition in CONDITIONS[1:] for budget in HISTORY_BUDGETS)
        failures = []
        for condition, budget in order:
            print(f"matrix group start: condition={condition} budget={budget}", flush=True)
            try:
                self.run_cell_group(condition, budget)
            except Exception as error:
                failures.append({"condition": condition, "history_budget": budget, "error": repr(error)})
                print(f"matrix group failed: condition={condition} budget={budget} error={error!r}; continuing", flush=True)
        if failures:
            print(f"matrix required groups failed={len(failures)}; final status will be incomplete", flush=True)
        return self.finalize()
