"""Dev-only Full+Short Context-Aware Personal Memory pilot orchestration."""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import sqlite3
import statistics
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.datasets.deep_author.pipeline import stable_hash
from src.evaluation.deep_author_v2 import (
    AUTHORS,
    CHECKPOINT_REVISION,
    DATASET_V1_BYTES,
    DATASET_V1_SHA256,
    OFFICIAL_CODE_REVISION,
    canonical_json,
    conditions_for_anchor,
    load_tokens,
    sha256_file,
    valid_anchors_for_work,
    write_csv,
    write_json,
    write_jsonl,
)
from src.personalisation.context_memory import (
    Candidate,
    PredictionQuery,
    assert_candidate_pool,
    macro_author_metrics,
    rank_frequency,
    rank_from_retrieved,
    rank_memory,
    rank_of,
    retrieve_memory,
    subset_membership,
)


PILOT_NAME = "personalisation_pilot_a_context_memory"
FREQUENCY_LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
MEMORY_TOP_NS = (1, 3, 5, 10, 20)
MEMORY_LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
EMBEDDING_MODEL_ID = "bge-small-zh-v1.5"
EMBEDDING_MODEL_SHA256 = "5a88d266870fbd27c6f329df60de80e2d4cf3bbd5e6f080bd5c1b2e5abb12039"
EMBEDDING_DIMENSION = 512
BACKEND_SOURCE_REVISION = "07a79f301a094d3db88780f00fcf85a4abf80d7f"
BACKEND_INTEGRATION_REVISION = "8c608f106ee7bb49ca5573e72de3da5eeb2290af"
GENERIC_CONTEXT_SEMANTICS = "evaluation-v2-full-preceding-context-v1"
EMBEDDING_PREPROCESSING_VERSION = "bge-gguf-left-truncate-most-recent-510-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_or_validate_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    expected = "".join(canonical_json(row) + "\n" for row in rows)
    if path.exists():
        if path.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"existing frozen pilot manifest differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8", newline="\n")


def _write_or_validate_json(path: Path, value: Mapping[str, Any]) -> None:
    expected = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"existing frozen pilot summary differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8", newline="\n")


def percentile90(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]


def timing_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean_ms": statistics.fmean(values) if values else None,
        "median_ms": statistics.median(values) if values else None,
        "p90_ms": percentile90(values),
    }


def split_dev_works(work_ids: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Freeze an earlier-work tune / later-work evaluation boundary."""

    if len(work_ids) < 2:
        raise ValueError("at least two chronological Dev works are required")
    tune_count = max(1, len(work_ids) // 2)
    return tuple(work_ids[:tune_count]), tuple(work_ids[tune_count:])


@dataclass
class PilotManifestBuilder:
    root: Path
    dataset_root: Path
    pinyingpt_model: Path
    output_root: Path

    @property
    def split_manifest(self) -> Path:
        return self.root / "results/evaluation/deep_author_v2/design/work_split_manifest.csv"

    def _split_rows(self) -> list[dict[str, str]]:
        with self.split_manifest.open(encoding="utf-8-sig", newline="") as source:
            return list(csv.DictReader(source))

    def _work_interactions(
        self,
        split_row: Mapping[str, str],
        compatibility: Mapping[str, Sequence[str]],
    ) -> list[dict[str, Any]]:
        work_id = str(split_row["work_id"])
        work_path = self.dataset_root / "data/processed/deep_author/works" / f"{work_id}.json"
        if not work_path.is_file():
            raise FileNotFoundError(work_path)
        work = json.loads(work_path.read_text(encoding="utf-8"))
        tokens = load_tokens(work_path.with_name(work_path.stem + ".tokens.jsonl"))
        anchors = valid_anchors_for_work(work, tokens, compatibility)
        rows = []
        for anchor in sorted(anchors, key=lambda row: (int(row["source_position_start"]), str(row["anchor_id"]))):
            condition = conditions_for_anchor(anchor)[0]
            rows.append(
                {
                    "row_id": "pilot-a-" + stable_hash(condition["condition_id"], "dev-full-short")[:24],
                    "condition_id": condition["condition_id"],
                    "anchor_id": condition["anchor_id"],
                    "author": condition["author"],
                    "work_id": condition["work_id"],
                    "work_creation_date": str(split_row["creation_date"]),
                    "work_chronological_index": int(split_row["chronological_index"]),
                    "source_split": str(split_row["split"]),
                    "source_position_start": int(condition["source_position_start"]),
                    "source_position_end": int(condition["source_position_end"]),
                    "context": condition["context"],
                    "pinyin_input": condition["pinyin_input"],
                    "pinyin_segments": str(condition["pinyin_input"]).split(),
                    "gold": condition["gold"],
                    "target": condition["gold"],
                    "gold_char_length": condition["gold_char_length"],
                    "source_hash": condition["source_hash"],
                    "cleaned_text_hash": condition["cleaned_text_hash"],
                }
            )
        return rows

    def run(self) -> dict[str, Any]:
        history_path = self.output_root / "history_manifest.jsonl"
        dev_path = self.output_root / "dev_manifest.jsonl"
        summary_path = self.output_root / "dev_split_summary.json"
        checksums_path = self.output_root / "manifest_checksums.json"
        if all(path.exists() for path in (history_path, dev_path, summary_path, checksums_path)):
            checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
            actual = {
                "history_manifest_sha256": sha256_file(history_path),
                "dev_manifest_sha256": sha256_file(dev_path),
                "summary_sha256": sha256_file(summary_path),
            }
            if any(checksums.get(key) != value for key, value in actual.items()):
                raise RuntimeError("existing Pilot A manifest checksum mismatch")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("test_rows") != 0:
                raise RuntimeError("existing Pilot A manifest includes Test data")
            return summary
        dataset = self.dataset_root / "data/processed/deep_author/interactions_t1_ready.jsonl"
        if dataset.stat().st_size != DATASET_V1_BYTES or sha256_file(dataset) != DATASET_V1_SHA256:
            raise RuntimeError("Dataset V1 source does not match the frozen size and SHA-256")
        compatibility_path = self.pinyingpt_model / "pinyin2char.json"
        compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
        split_rows = self._split_rows()
        if {row["split"] for row in split_rows} != {"history", "dev", "test"}:
            raise RuntimeError("frozen chronological split is incomplete")

        history_rows: list[dict[str, Any]] = []
        dev_rows: list[dict[str, Any]] = []
        split_rule: dict[str, Any] = {}
        position = 0
        for author in AUTHORS:
            author_rows = sorted(
                (row for row in split_rows if row["author"] == author and row["split"] in {"history", "dev"}),
                key=lambda row: int(row["chronological_index"]),
            )
            dev_works = [row["work_id"] for row in author_rows if row["split"] == "dev"]
            if len(dev_works) < 2:
                raise RuntimeError(f"Dev-internal whole-work split is impossible for {author}")
            tune_work_ids, evaluation_work_ids = split_dev_works(dev_works)
            tune_works = set(tune_work_ids)
            evaluation_works = set(evaluation_work_ids)
            split_rule[author] = {
                "tune_work_ids": sorted(tune_works),
                "evaluation_work_ids": sorted(evaluation_works),
            }
            for split_row in author_rows:
                for interaction in self._work_interactions(split_row, compatibility):
                    position += 1
                    interaction["chronological_position"] = position
                    if interaction["source_split"] == "history":
                        interaction["pilot_partition"] = "history"
                        history_rows.append(interaction)
                    else:
                        interaction["pilot_partition"] = (
                            "tune" if interaction["work_id"] in tune_works else "evaluation"
                        )
                        dev_rows.append(interaction)

        if any(row["source_split"] == "test" for row in history_rows + dev_rows):
            raise AssertionError("Test data entered Pilot A")
        combined = sorted(history_rows + dev_rows, key=lambda row: int(row["chronological_position"]))
        visible_by_author: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        history_available = same_pinyin_available = ambiguous = 0
        for row in combined:
            prior = visible_by_author[row["author"]]
            if row["source_split"] == "dev":
                history_available += bool(prior)
                matching = [item for item in prior if item["pinyin_segments"] == row["pinyin_segments"]]
                same_pinyin_available += bool(matching)
                ambiguous += len({item["gold"] for item in matching}) >= 2
            visible_by_author[row["author"]].append(row)

        by_author = Counter(row["author"] for row in dev_rows)
        by_work = Counter(row["work_id"] for row in dev_rows)
        summary = {
            "schema_version": 1,
            "pilot": PILOT_NAME,
            "dataset": "Deep Author Dataset V1 development source",
            "dataset_sha256": DATASET_V1_SHA256,
            "condition": "Full + Short only",
            "history_rows": len(history_rows),
            "dev_rows": len(dev_rows),
            "tune_rows": sum(row["pilot_partition"] == "tune" for row in dev_rows),
            "evaluation_rows": sum(row["pilot_partition"] == "evaluation" for row in dev_rows),
            "per_author_dev_rows": dict(by_author),
            "per_work_dev_rows": dict(sorted(by_work.items())),
            "unique_pinyin": len({tuple(row["pinyin_segments"]) for row in dev_rows}),
            "history_available_rows": history_available,
            "same_pinyin_history_rows": same_pinyin_available,
            "ambiguous_history_rows": ambiguous,
            "split_rule": "Within each author, earlier floor(Dev works / 2), minimum one, are tune; all later Dev works are evaluation.",
            "per_author_work_split": split_rule,
            "test_rows": 0,
        }
        _write_or_validate_jsonl(self.output_root / "history_manifest.jsonl", history_rows)
        _write_or_validate_jsonl(self.output_root / "dev_manifest.jsonl", dev_rows)
        _write_or_validate_json(self.output_root / "dev_split_summary.json", summary)
        manifest = {
            "history_manifest_sha256": sha256_file(self.output_root / "history_manifest.jsonl"),
            "dev_manifest_sha256": sha256_file(self.output_root / "dev_manifest.jsonl"),
            "summary_sha256": sha256_file(self.output_root / "dev_split_summary.json"),
            "checkpoint_revision": CHECKPOINT_REVISION,
            "official_code_revision": OFFICIAL_CODE_REVISION,
        }
        _write_or_validate_json(self.output_root / "manifest_checksums.json", manifest)
        return summary


class BGEContextEmbedder:
    """Pinned mean-pooled BGE GGUF runtime with tokenizer-aware left truncation."""

    def __init__(self, model_path: Path, *, n_gpu_layers: int = -1) -> None:
        self.model_path = Path(model_path)
        if sha256_file(self.model_path) != EMBEDDING_MODEL_SHA256:
            raise ValueError("embedding model does not match the pinned SHA-256")
        self.n_gpu_layers = n_gpu_layers
        self._model = None
        self.load_ms: float | None = None
        self.truncation_count = 0

    def _load(self):
        if self._model is None:
            from llama_cpp import LLAMA_POOLING_TYPE_MEAN, Llama

            started = time.perf_counter()
            self._model = Llama(
                model_path=str(self.model_path),
                n_ctx=512,
                n_gpu_layers=self.n_gpu_layers,
                embedding=True,
                pooling_type=LLAMA_POOLING_TYPE_MEAN,
                verbose=False,
            )
            self.load_ms = (time.perf_counter() - started) * 1000.0
        return self._model

    def _model_text(self, text: str) -> str:
        model = self._load()
        tokens = model.tokenize(text.encode("utf-8"), add_bos=False)
        if len(tokens) <= 510:
            return text
        self.truncation_count += 1
        return model.detokenize(tokens[-510:]).decode("utf-8", errors="ignore")

    def embed(self, text: str) -> tuple[float, ...]:
        result = self._load().create_embedding(self._model_text(text))
        vector = np.asarray(result["data"][0]["embedding"], dtype=np.float32)
        if vector.size != EMBEDDING_DIMENSION:
            raise ValueError(f"unexpected embedding dimension: {vector.size}")
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("zero embedding")
        return tuple(float(value) for value in vector / norm)

    def info(self) -> dict[str, Any]:
        import llama_cpp

        return {
            "model_id": EMBEDDING_MODEL_ID,
            "model_revision": None,
            "model_sha256": EMBEDDING_MODEL_SHA256,
            "model_path": str(self.model_path),
            "dimension": EMBEDDING_DIMENSION,
            "pooling": "mean",
            "normalization": "L2",
            "maximum_model_tokens": 512,
            "context_truncation": "tokenizer-aware left truncation to the most recent 510 content tokens",
            "preprocessing_version": EMBEDDING_PREPROCESSING_VERSION,
            "truncation_count": self.truncation_count,
            "runtime": "llama-cpp-python",
            "runtime_version": getattr(llama_cpp, "__version__", "unknown"),
            "n_gpu_layers": self.n_gpu_layers,
            "model_load_ms": self.load_ms,
        }


class EmbeddingCache:
    """Durable normalized float32 context embeddings in SQLite."""

    def __init__(self, path: Path, model_sha256: str = EMBEDDING_MODEL_SHA256) -> None:
        self.path = path
        self.model_sha256 = model_sha256
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (cache_key TEXT PRIMARY KEY, context_sha256 TEXT NOT NULL, vector BLOB NOT NULL)"
        )
        try:
            runtime_version = version("llama-cpp-python")
        except PackageNotFoundError:
            runtime_version = "unavailable"
        expected = {
            "model_id": EMBEDDING_MODEL_ID,
            "model_sha256": model_sha256,
            "dimension": str(EMBEDDING_DIMENSION),
            "pooling": "mean",
            "normalization": "L2",
            "preprocessing_version": EMBEDDING_PREPROCESSING_VERSION,
            "runtime": "llama-cpp-python",
            "runtime_version": runtime_version,
        }
        existing = dict(self.connection.execute("SELECT key, value FROM metadata"))
        if existing and existing != expected:
            raise RuntimeError("embedding cache provenance differs from the frozen configuration")
        if not existing:
            self.connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", expected.items())
            self.connection.commit()

    def key(self, context: str) -> str:
        identity = "\0".join(
            (
                self.model_sha256,
                EMBEDDING_PREPROCESSING_VERSION,
                "mean",
                "L2",
                context,
            )
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def get(self, context: str) -> np.ndarray | None:
        row = self.connection.execute("SELECT vector FROM embeddings WHERE cache_key = ?", (self.key(context),)).fetchone()
        if row is None:
            return None
        vector = np.frombuffer(row[0], dtype="<f4")
        if vector.size != EMBEDDING_DIMENSION:
            raise RuntimeError("cached embedding has the wrong dimension")
        return vector.copy()

    def put(self, context: str, vector: Sequence[float]) -> None:
        values = np.asarray(vector, dtype="<f4")
        if values.size != EMBEDDING_DIMENSION:
            raise ValueError("embedding has the wrong dimension")
        self.connection.execute(
            "INSERT OR IGNORE INTO embeddings(cache_key, context_sha256, vector) VALUES (?, ?, ?)",
            (self.key(context), hashlib.sha256(context.encode("utf-8")).hexdigest(), values.tobytes()),
        )

    def __getitem__(self, context: str) -> np.ndarray:
        vector = self.get(context)
        if vector is None:
            raise KeyError(context)
        return vector

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


class EmbeddingLookup:
    def __init__(self, cache: EmbeddingCache) -> None:
        self.cache = cache

    @lru_cache(maxsize=20_000)
    def __getitem__(self, context: str) -> np.ndarray:
        return self.cache[context]


class HistoryIndex:
    """Strictly-prior same-user history, optionally capped before Pinyin filtering."""

    def __init__(self, records: Sequence[Mapping[str, Any]], history_budget: int | None = None) -> None:
        if history_budget is not None and history_budget <= 0:
            raise ValueError("history_budget must be positive")
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[str(record["author"])].append(record)
        self.records = {key: tuple(sorted(values, key=lambda row: int(row["chronological_position"]))) for key, values in grouped.items()}
        self.positions = {key: tuple(int(row["chronological_position"]) for row in values) for key, values in self.records.items()}
        self.history_budget = history_budget

    def visible(self, query: PredictionQuery) -> tuple[Mapping[str, Any], ...]:
        key = query.author
        values = self.records.get(key, ())
        stop = bisect_left(self.positions.get(key, ()), query.chronological_position)
        start = max(0, stop - self.history_budget) if self.history_budget is not None else 0
        budgeted = values[start:stop]
        return tuple(row for row in budgeted if tuple(row["pinyin_segments"]) == query.pinyin)


@dataclass
class PilotRunner:
    root: Path
    dataset_root: Path
    pinyingpt_model: Path
    embedding_model: Path
    output_root: Path
    history_budget: int | None = None
    prediction_partition: str | None = None

    def __post_init__(self) -> None:
        if self.prediction_partition not in {None, "tune", "evaluation"}:
            raise ValueError("prediction_partition must be tune, evaluation, or None")

    @property
    def cache_root(self) -> Path:
        return self.output_root / "cache"

    @property
    def generic_cache_path(self) -> Path:
        legacy = self.output_root / "generic_predictions.jsonl"
        canonical = self.cache_root / "generic_predictions.jsonl"
        return legacy if legacy.exists() and not canonical.exists() else canonical

    @property
    def embedding_cache_path(self) -> Path:
        legacy = self.output_root / "embedding_cache.sqlite3"
        canonical = self.cache_root / "embedding_cache.sqlite3"
        return legacy if legacy.exists() and not canonical.exists() else canonical

    def prepare(self) -> dict[str, Any]:
        return PilotManifestBuilder(self.root, self.dataset_root, self.pinyingpt_model, self.output_root).run()

    def _manifests(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        history_path = self.output_root / "history_manifest.jsonl"
        dev_path = self.output_root / "dev_manifest.jsonl"
        if not history_path.exists() or not dev_path.exists():
            raise RuntimeError("Pilot manifests are absent; run --phase prepare first")
        history = read_jsonl(history_path)
        dev = read_jsonl(dev_path)
        if any(row["source_split"] == "test" for row in history + dev):
            raise RuntimeError("Test rows are forbidden in Pilot A")
        if any(row["source_split"] != "dev" for row in dev):
            raise RuntimeError("Pilot prediction manifest must contain Dev only")
        return history, dev

    @staticmethod
    def _query(row: Mapping[str, Any]) -> PredictionQuery:
        return PredictionQuery(
            row_id=str(row["row_id"]),
            author=str(row["author"]),
            work_id=str(row["work_id"]),
            chronological_position=int(row["chronological_position"]),
            context=str(row["context"]),
            pinyin=tuple(row["pinyin_segments"]),
        )

    @staticmethod
    def _candidates(generic_row: Mapping[str, Any]) -> tuple[Candidate, ...]:
        return tuple(
            Candidate(str(row["text"]), int(row["rank"]), float(row["log_probability"]))
            for row in generic_row["top10_candidates"]
        )

    def _load_generic(self, dev: Sequence[Mapping[str, Any]], *, require_complete: bool) -> dict[str, dict[str, Any]]:
        path = self.generic_cache_path
        if not path.exists():
            if require_complete:
                raise RuntimeError("Generic Dev cache is absent; run --phase generic")
            return {}
        _, full_dev = self._manifests()
        expected = {str(row["row_id"]): row for row in full_dev}
        requested = {str(row["row_id"]) for row in dev}
        all_completed: dict[str, dict[str, Any]] = {}
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            row = json.loads(line)
            row_id = str(row.get("row_id", ""))
            if row_id in all_completed or row_id not in expected:
                raise RuntimeError(f"invalid Generic cache ID at line {line_number}: {row_id}")
            frozen = expected[row_id]
            for key in ("author", "work_id", "chronological_position", "context", "pinyin_input", "pinyin_segments", "gold", "pilot_partition"):
                if row.get(key) != frozen[key]:
                    raise RuntimeError(f"Generic cache differs from Dev manifest: {row_id} {key}")
            if row.get("checkpoint_revision") != CHECKPOINT_REVISION or row.get("official_code_revision") != OFFICIAL_CODE_REVISION:
                raise RuntimeError("Generic cache model provenance mismatch")
            if row.get("beam_size") != 16 or row.get("top_k") != 10 or row.get("runtime_device") != "cuda":
                raise RuntimeError("Generic cache decoding/runtime provenance mismatch")
            if row.get("backend_source_revision") != BACKEND_SOURCE_REVISION or row.get("backend_integration_revision") != BACKEND_INTEGRATION_REVISION:
                raise RuntimeError("Generic cache backend provenance mismatch")
            if row.get("context_semantics") != GENERIC_CONTEXT_SEMANTICS:
                raise RuntimeError("Generic cache context-semantics provenance mismatch")
            candidates = row.get("top10_candidates")
            if not isinstance(candidates, list) or not 1 <= len(candidates) <= 10:
                raise RuntimeError(f"invalid Generic candidate surface: {row_id}")
            if [candidate["rank"] for candidate in candidates] != list(range(1, len(candidates) + 1)):
                raise RuntimeError(f"invalid Generic ranks: {row_id}")
            if len({candidate["text"] for candidate in candidates}) != len(candidates):
                raise RuntimeError(f"duplicate Generic candidates: {row_id}")
            all_completed[row_id] = row
        completed = {row_id: all_completed[row_id] for row_id in requested if row_id in all_completed}
        if require_complete and set(completed) != requested:
            raise RuntimeError(f"Generic cache is incomplete: {len(completed)}/{len(requested)}")
        return completed

    def generic(self) -> dict[str, Any]:
        from src.reference_backend_pinyingpt import PinyinGPTConcatBackend

        _, dev = self._manifests()
        if self.prediction_partition is not None:
            dev = [row for row in dev if row["pilot_partition"] == self.prediction_partition]
        self.cache_root.mkdir(parents=True, exist_ok=True)
        completed = self._load_generic(dev, require_complete=False)
        existing = len(completed)
        backend = None
        load_seconds = 0.0
        prediction_path = self.generic_cache_path
        pending = [row for row in dev if row["row_id"] not in completed]
        print(
            f"Generic Dev cache: requested={len(dev)} reused={existing} missing={len(pending)}",
            flush=True,
        )
        if pending:
            load_started = time.perf_counter()
            backend = PinyinGPTConcatBackend(self.pinyingpt_model, device="cuda")
            load_seconds = time.perf_counter() - load_started
        started = time.perf_counter()
        latencies = []
        mode = "a" if prediction_path.exists() and prediction_path.stat().st_size else "w"
        with prediction_path.open(mode, encoding="utf-8", newline="\n") as destination:
            for window_start in range(0, len(pending), 16):
                window = pending[window_start : window_start + 16]
                prepared = []
                for row in window:
                    segments = list(row["pinyin_segments"])
                    assert backend is not None
                    used_context, original_tokens, used_tokens, truncated = backend.truncate_context_for_generation(row["context"], segments)
                    prompt, _ = backend._prompt(used_context, segments)
                    prepared.append((row, segments, used_context, original_tokens, used_tokens, truncated, len(prompt)))
                groups: dict[tuple[int, int], list[tuple[Any, ...]]] = defaultdict(list)
                for item in prepared:
                    groups[(len(item[1]), item[6])].append(item)
                for equal_shape_group in groups.values():
                    for group_start in range(0, len(equal_shape_group), 2):
                        group = equal_shape_group[group_start : group_start + 2]
                        inference_started = time.perf_counter()
                        results = backend.generate_batch([(item[2], item[1]) for item in group], top_k=10, beam_size=16)
                        elapsed_ms = (time.perf_counter() - inference_started) * 1000.0
                        latencies.extend([elapsed_ms / len(group)] * len(group))
                        for item, result in zip(group, results):
                            row, _, used_context, original_tokens, used_tokens, truncated, _ = item
                            candidates = [candidate.to_dict() for candidate in result.candidates]
                            gold_rank = next((candidate["rank"] for candidate in candidates if candidate["text"] == row["gold"]), None)
                            output = {
                                **row,
                                "model_used_context": used_context,
                                "original_stored_context_tokens": original_tokens,
                                "model_used_context_tokens": used_tokens,
                                "context_truncated": truncated,
                                "top10_candidates": candidates,
                                "gold_rank": gold_rank,
                                "beam_size": 16,
                                "top_k": 10,
                                "runtime_device": result.runtime_device,
                                "checkpoint_revision": CHECKPOINT_REVISION,
                                "official_code_revision": OFFICIAL_CODE_REVISION,
                                "backend_source_revision": BACKEND_SOURCE_REVISION,
                                "backend_integration_revision": BACKEND_INTEGRATION_REVISION,
                                "context_semantics": GENERIC_CONTEXT_SEMANTICS,
                            }
                            destination.write(canonical_json(output) + "\n")
                            completed[str(row["row_id"])] = output
                destination.flush()
                count = len(completed)
                if count % 100 < len(window) or count == len(dev):
                    elapsed = time.perf_counter() - started
                    added = count - existing
                    rate = added / elapsed if elapsed else 0.0
                    eta = (len(dev) - count) / rate if rate else None
                    print(f"generic {count}/{len(dev)}; rate={rate:.3f}/s; eta={eta:.1f}s", flush=True)
        elapsed = time.perf_counter() - started
        summary = {
            "status": "complete",
            "requested_rows": len(dev),
            "rows": len(completed),
            "cache_hits": existing,
            "resume_existing_rows": existing,
            "missing_rows_at_start": len(pending),
            "rows_added": len(completed) - existing,
            "model_load_seconds": load_seconds,
            "inference_seconds": elapsed,
            "conditions_per_second": (len(completed) - existing) / elapsed if len(completed) > existing else 0.0,
            "per_condition_latency": timing_summary(latencies),
            "runtime": backend.runtime_info() if backend is not None else None,
            "cache_path": str(prediction_path),
            "cache_sha256": sha256_file(prediction_path),
        }
        write_json(self.output_root / "generic_runtime.json", summary)
        return summary

    def _required_embedding_contexts(self, records: Sequence[Mapping[str, Any]], dev: Sequence[Mapping[str, Any]]) -> list[str]:
        index = HistoryIndex(records, self.history_budget)
        contexts = {str(row["context"]) for row in dev}
        for row in dev:
            contexts.update(str(item["context"]) for item in index.visible(self._query(row)))
        return sorted(contexts, key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())

    def embeddings(self) -> dict[str, Any]:
        history, dev = self._manifests()
        if self.prediction_partition is not None:
            dev = [row for row in dev if row["pilot_partition"] == self.prediction_partition]
        contexts = self._required_embedding_contexts(history + dev, dev)
        cache = EmbeddingCache(self.embedding_cache_path)
        embedder = BGEContextEmbedder(self.embedding_model)
        existing = cache.count()
        requested_hits = sum(cache.get(context) is not None for context in contexts)
        requested_missing = len(contexts) - requested_hits
        print(
            f"Embedding cache: requested={len(contexts)} cache_hits={requested_hits} missing={requested_missing}",
            flush=True,
        )
        latencies = []
        started = time.perf_counter()
        added = 0
        try:
            for index, context in enumerate(contexts, start=1):
                if cache.get(context) is None:
                    call_started = time.perf_counter()
                    cache.put(context, embedder.embed(context))
                    latencies.append((time.perf_counter() - call_started) * 1000.0)
                    added += 1
                    if added % 100 == 0:
                        cache.commit()
                if index % 100 == 0 or index == len(contexts):
                    elapsed = time.perf_counter() - started
                    rate = index / elapsed if elapsed else 0.0
                    eta = (len(contexts) - index) / rate if rate else None
                    print(f"embeddings checked={index}/{len(contexts)} added={added}; rate={rate:.3f}/s; eta={eta:.1f}s", flush=True)
            cache.commit()
            summary = {
                "status": "complete",
                "history_budget": self.history_budget,
                "required_unique_contexts": len(contexts),
                "cache_hits": requested_hits,
                "missing_at_start": requested_missing,
                "existing_embeddings_at_start": existing,
                "embeddings_added": added,
                "final_cache_rows": cache.count(),
                "elapsed_seconds": time.perf_counter() - started,
                "embedding_latency": timing_summary(latencies),
                "embedding_model": embedder.info(),
                "cache_path": str(cache.path),
            }
            write_json(self.output_root / "embedding_runtime.json", summary)
            return summary
        finally:
            cache.close()

    def _indexed_inputs(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], HistoryIndex]:
        history, dev = self._manifests()
        return history, dev, HistoryIndex(history + dev, self.history_budget)

    def tune(self) -> dict[str, Any]:
        history, dev, index = self._indexed_inputs()
        del history
        tune_rows = [row for row in dev if row["pilot_partition"] == "tune"]
        if not tune_rows or any(row["pilot_partition"] != "tune" for row in tune_rows):
            raise RuntimeError("Dev tune partition is absent or invalid")
        generic = self._load_generic(tune_rows, require_complete=True)
        cache = EmbeddingCache(self.embedding_cache_path)
        lookup = EmbeddingLookup(cache)
        frequency_ranks: dict[float, list[dict[str, Any]]] = {value: [] for value in FREQUENCY_LAMBDAS}
        memory_ranks: dict[tuple[int, float], list[dict[str, Any]]] = {
            (top_n, value): [] for top_n in MEMORY_TOP_NS for value in MEMORY_LAMBDAS
        }
        retrieval_times = []
        rerank_times = []
        started = time.perf_counter()
        try:
            for index_value, row in enumerate(tune_rows, start=1):
                query = self._query(row)
                visible = index.visible(query)
                candidates = self._candidates(generic[row["row_id"]])
                for value in FREQUENCY_LAMBDAS:
                    ranked = rank_frequency(query, candidates, visible, lambda_frequency=value)
                    frequency_ranks[value].append({"author": row["author"], "rank": rank_of(ranked, row["gold"])})
                retrieval_started = time.perf_counter()
                retrieved = retrieve_memory(query, visible, lookup) if visible else ()
                retrieval_times.append((time.perf_counter() - retrieval_started) * 1000.0)
                rerank_started = time.perf_counter()
                for top_n in MEMORY_TOP_NS:
                    selected = retrieved[:top_n]
                    for value in MEMORY_LAMBDAS:
                        ranked = rank_from_retrieved(candidates, selected, lambda_memory=value)
                        memory_ranks[(top_n, value)].append({"author": row["author"], "rank": rank_of(ranked, row["gold"])})
                rerank_times.append((time.perf_counter() - rerank_started) * 1000.0)
                if index_value % 100 == 0 or index_value == len(tune_rows):
                    elapsed = time.perf_counter() - started
                    print(f"tune {index_value}/{len(tune_rows)}; rate={index_value / elapsed:.3f}/s", flush=True)

            frequency_search = []
            for value in FREQUENCY_LAMBDAS:
                metrics = macro_author_metrics(frequency_ranks[value], "rank")["macro_author"]
                frequency_search.append({"lambda_frequency": value, **metrics})
            memory_search = []
            for top_n in MEMORY_TOP_NS:
                for value in MEMORY_LAMBDAS:
                    metrics = macro_author_metrics(memory_ranks[(top_n, value)], "rank")["macro_author"]
                    memory_search.append({"top_n": top_n, "lambda_memory": value, **metrics})
            selected_frequency = max(frequency_search, key=lambda row: (float(row["top1"]), -float(row["lambda_frequency"])))
            selected_memory = max(
                memory_search,
                key=lambda row: (float(row["top1"]), -float(row["lambda_memory"]), -int(row["top_n"])),
            )
            write_csv(self.output_root / "frequency_hyperparameter_search.csv", frequency_search, list(frequency_search[0]))
            write_csv(self.output_root / "memory_hyperparameter_search.csv", memory_search, list(memory_search[0]))
            selection = {
                "status": "complete",
                "history_budget": self.history_budget,
                "selection_population": "chronologically earlier whole-work Dev tune partition",
                "reported_population": "chronologically later whole-work Dev evaluation partition",
                "selection_metric": "Macro-author Top-1",
                "tie_break_frequency": "lower lambda_frequency",
                "tie_break_memory": "lower lambda_memory, then lower top_n",
                "frequency": {"lambda_frequency": selected_frequency["lambda_frequency"]},
                "memory": {"top_n": selected_memory["top_n"], "lambda_memory": selected_memory["lambda_memory"]},
                "frequency_grid": list(FREQUENCY_LAMBDAS),
                "memory_top_n_grid": list(MEMORY_TOP_NS),
                "memory_lambda_grid": list(MEMORY_LAMBDAS),
                "tune_rows": len(tune_rows),
                "tune_work_ids": sorted({row["work_id"] for row in tune_rows}),
                "evaluation_work_ids": sorted({row["work_id"] for row in dev if row["pilot_partition"] == "evaluation"}),
                "runtime_seconds": time.perf_counter() - started,
                "retrieval_timing": timing_summary(retrieval_times),
                "reranking_grid_timing": timing_summary(rerank_times),
            }
            write_json(self.output_root / "selected_hyperparameters.json", selection)
            return selection
        finally:
            cache.close()

    def evaluate(self) -> dict[str, Any]:
        history, dev, index = self._indexed_inputs()
        del history
        generic = self._load_generic(dev, require_complete=True)
        selection_path = self.output_root / "selected_hyperparameters.json"
        if not selection_path.exists():
            raise RuntimeError("selected hyperparameters are absent; run --phase tune")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        lambda_frequency = float(selection["frequency"]["lambda_frequency"])
        top_n = int(selection["memory"]["top_n"])
        lambda_memory = float(selection["memory"]["lambda_memory"])
        evaluation_rows = [row for row in dev if row["pilot_partition"] == "evaluation"]
        if set(selection["evaluation_work_ids"]) != {row["work_id"] for row in evaluation_rows}:
            raise RuntimeError("evaluation work set differs from frozen hyperparameter selection")
        cache = EmbeddingCache(self.embedding_cache_path)
        lookup = EmbeddingLookup(cache)
        frequency_outputs = []
        memory_outputs = []
        metric_rows = []
        ambiguous_rows = []
        conflict_rows = []
        frequency_times = []
        retrieval_times = []
        rerank_times = []
        started = time.perf_counter()
        try:
            for index_value, row in enumerate(evaluation_rows, start=1):
                query = self._query(row)
                visible = index.visible(query)
                generic_row = generic[row["row_id"]]
                candidates = self._candidates(generic_row)
                frequency_started = time.perf_counter()
                frequency_ranked = rank_frequency(query, candidates, visible, lambda_frequency=lambda_frequency)
                frequency_times.append((time.perf_counter() - frequency_started) * 1000.0)
                retrieval_started = time.perf_counter()
                retrieved = retrieve_memory(query, visible, lookup) if visible else ()
                retrieval_times.append((time.perf_counter() - retrieval_started) * 1000.0)
                memory_started = time.perf_counter()
                memory_ranked = rank_from_retrieved(candidates, retrieved[:top_n], lambda_memory=lambda_memory)
                rerank_times.append((time.perf_counter() - memory_started) * 1000.0)
                assert_candidate_pool(candidates, frequency_ranked, memory_ranked)
                generic_rank = generic_row["gold_rank"]
                frequency_rank = rank_of(frequency_ranked, row["gold"])
                memory_rank = rank_of(memory_ranked, row["gold"])
                if (generic_rank is None) != (frequency_rank is None) or (generic_rank is None) != (memory_rank is None):
                    raise AssertionError("Missing@10 changed despite the frozen candidate pool")
                flags = subset_membership(query, row["gold"], visible)
                common = {
                    "row_id": row["row_id"],
                    "author": row["author"],
                    "work_id": row["work_id"],
                    "chronological_position": row["chronological_position"],
                    "context": row["context"],
                    "pinyin_segments": row["pinyin_segments"],
                    "gold": row["gold"],
                    **flags,
                }
                frequency_outputs.append({**common, "lambda_frequency": lambda_frequency, "candidates": frequency_ranked, "gold_rank": frequency_rank})
                memory_candidates = []
                for candidate in memory_ranked:
                    candidate = dict(candidate)
                    candidate["memory_score"] = candidate.pop("personal_score")
                    memory_candidates.append(candidate)
                memory_outputs.append(
                    {
                        **common,
                        "top_n": top_n,
                        "lambda_memory": lambda_memory,
                        "candidates": memory_candidates,
                        "retrieved_evidence": retrieved[:top_n],
                        "gold_rank": memory_rank,
                    }
                )
                metrics_row = {**common, "generic_rank": generic_rank, "frequency_rank": frequency_rank, "memory_rank": memory_rank}
                metric_rows.append(metrics_row)
                if flags["ambiguous"]:
                    ambiguous_rows.append(metrics_row)
                if flags["conflict"]:
                    conflict_rows.append(metrics_row)
                if index_value % 100 == 0 or index_value == len(evaluation_rows):
                    elapsed = time.perf_counter() - started
                    print(f"evaluate {index_value}/{len(evaluation_rows)}; rate={index_value / elapsed:.3f}/s", flush=True)

            write_jsonl(self.output_root / "frequency_predictions.jsonl", frequency_outputs)
            write_jsonl(self.output_root / "memory_predictions.jsonl", memory_outputs)
            write_jsonl(self.output_root / "ambiguous_subset.jsonl", ambiguous_rows)
            write_jsonl(self.output_root / "conflict_subset.jsonl", conflict_rows)
            subsets = {
                "overall": metric_rows,
                "history_available": [row for row in metric_rows if row["history_available"]],
                "ambiguous": ambiguous_rows,
                "conflict": conflict_rows,
            }
            models = {"G0": "generic_rank", "F": "frequency_rank", "M": "memory_rank"}
            metrics = {
                subset: {model: macro_author_metrics(rows, rank_key) for model, rank_key in models.items()}
                for subset, rows in subsets.items()
            }
            missing_values = {
                model: sum(row[rank_key] is None for row in metric_rows)
                for model, rank_key in models.items()
            }
            if len(set(missing_values.values())) != 1:
                raise AssertionError("Missing@10 is not invariant across the frozen candidate pool")
            summary = {
                "schema_version": 1,
                "status": "complete",
                "history_budget": self.history_budget,
                "population": "Dev-evaluation Full+Short",
                "rows": len(metric_rows),
                "selected_hyperparameters": selection,
                "metrics": metrics,
                "subset_rows": {name: len(rows) for name, rows in subsets.items()},
                "missing_counts": missing_values,
                "candidate_pool_invariant": True,
                "test_rows_used": 0,
                "author_identity_used_only_for_history_isolation": True,
            }
            write_json(self.output_root / "metrics_summary.json", summary)
            author_output = []
            for model, rank_key in models.items():
                per_author = macro_author_metrics(metric_rows, rank_key)["per_author"]
                author_output.extend({"model": model, "author": author, **values} for author, values in per_author.items())
            write_csv(self.output_root / "metrics_by_author.csv", author_output, list(author_output[0]))
            subset_output = []
            for subset, model_values in metrics.items():
                for model, values in model_values.items():
                    subset_output.append({"subset": subset, "model": model, **values["macro_author"]})
            write_csv(self.output_root / "metrics_by_subset.csv", subset_output, list(subset_output[0]))
            runtime = {
                "status": "complete",
                "evaluation_rows": len(metric_rows),
                "elapsed_seconds": time.perf_counter() - started,
                "frequency_reranking": timing_summary(frequency_times),
                "memory_retrieval": timing_summary(retrieval_times),
                "memory_reranking": timing_summary(rerank_times),
                "history_budget": self.history_budget,
                "generic": json.loads((self.output_root / "generic_runtime.json").read_text(encoding="utf-8")),
                "embeddings": json.loads((self.output_root / "embedding_runtime.json").read_text(encoding="utf-8")),
            }
            write_json(self.output_root / "runtime_summary.json", runtime)
            artifacts = [
                path for path in sorted(self.output_root.iterdir())
                if path.is_file() and path.name != "artifact_checksums.json" and path.suffix in {".json", ".jsonl", ".csv", ".sqlite3"}
            ]
            write_json(
                self.output_root / "artifact_checksums.json",
                {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in artifacts},
            )
            return summary
        finally:
            cache.close()

    def smoke(self, rows_per_author: int = 2) -> dict[str, Any]:
        """Isolated engineering smoke; synthetic prior records are not research data."""

        from src.reference_backend_pinyingpt import PinyinGPTConcatBackend

        _, dev = self._manifests()
        selected = [row for author in AUTHORS for row in [item for item in dev if item["author"] == author][:rows_per_author]]
        smoke_root = self.output_root / "smoke"
        smoke_root.mkdir(parents=True, exist_ok=True)
        backend = PinyinGPTConcatBackend(self.pinyingpt_model, device="cuda")
        embedder = BGEContextEmbedder(self.embedding_model)
        cache = EmbeddingCache(smoke_root / "embedding_cache_v2.sqlite3")
        outputs = []
        generic_outputs = []
        started = time.perf_counter()
        try:
            for row in selected:
                query = self._query(row)
                result = backend.generate(row["context"], row["pinyin_segments"], top_k=10, beam_size=16)
                candidates = tuple(Candidate(candidate.text, candidate.rank, candidate.log_probability) for candidate in result.candidates)
                synthetic_history = [
                    {"row_id": f"smoke-{row['row_id']}-a", "author": row["author"], "work_id": "smoke-history", "chronological_position": row["chronological_position"] - 2, "context": row["context"] + "\n[smoke-prior-a]", "pinyin_segments": row["pinyin_segments"], "target": candidates[0].text},
                    {"row_id": f"smoke-{row['row_id']}-b", "author": row["author"], "work_id": "smoke-history", "chronological_position": row["chronological_position"] - 1, "context": row["context"] + "\n[smoke-prior-b]", "pinyin_segments": row["pinyin_segments"], "target": candidates[1].text},
                ]
                for context in [row["context"], *(item["context"] for item in synthetic_history)]:
                    if cache.get(context) is None:
                        cache.put(context, embedder.embed(context))
                cache.commit()
                frequency = rank_frequency(query, candidates, synthetic_history, lambda_frequency=0.5)
                memory, evidence = rank_memory(query, candidates, synthetic_history, EmbeddingLookup(cache), top_n=2, lambda_memory=0.5)
                assert_candidate_pool(candidates, frequency, memory)
                outputs.append({"row_id": row["row_id"], "author": row["author"], "generic_rank": rank_of([{"candidate": c.text, "rank": c.generic_rank} for c in candidates], row["gold"]), "frequency_rank": rank_of(frequency, row["gold"]), "memory_rank": rank_of(memory, row["gold"]), "retrieved": evidence})
                generic_outputs.append({"row_id": row["row_id"], "candidates": [candidate.__dict__ for candidate in candidates]})
            write_jsonl(smoke_root / "generic_predictions_v2.jsonl", generic_outputs)
            write_jsonl(smoke_root / "smoke_predictions_v2.jsonl", outputs)
            summary = {"status": "passed", "research_result": False, "rows": len(outputs), "cuda_device": backend.runtime_info()["device_name"], "embedding_model": embedder.info(), "embedding_cache_rows": cache.count(), "candidate_pool_invariant": True, "metrics_pipeline": macro_author_metrics(outputs, "memory_rank"), "elapsed_seconds": time.perf_counter() - started}
            write_json(smoke_root / "smoke_summary_v2.json", summary)
            return summary
        finally:
            cache.close()

    def all(self) -> dict[str, Any]:
        return {
            "prepare": self.prepare(),
            "generic": self.generic(),
            "embeddings": self.embeddings(),
            "tune": self.tune(),
            "evaluate": self.evaluate(),
        }
