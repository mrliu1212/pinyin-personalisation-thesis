"""T1-aligned Personalisation Pilot A: M1 with a 5,000-record history budget."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from src.evaluation.deep_author_v2 import (
    AUTHORS,
    T1Runner,
    sha256_file,
    write_csv,
    write_json,
    write_jsonl,
)
from src.personalisation.context_memory import (
    Candidate,
    assert_candidate_pool,
    macro_author_metrics,
    rank_frequency,
    rank_from_retrieved,
    rank_of,
    retrieve_memory,
    subset_membership,
)
from src.personalisation.pilot_a import (
    BGEContextEmbedder,
    EmbeddingCache,
    EmbeddingLookup,
    HistoryIndex,
    PilotRunner,
    PredictionQuery,
    _write_or_validate_json,
    _write_or_validate_jsonl,
    read_jsonl,
    timing_summary,
)


HISTORY_BUDGET = 5000
T1_MANIFEST_SHA256 = "45b9cafedd7a8269d1f0b66d3f7f135ee990140e4b5b3668c67645863ab00d39"
T1_PREDICTIONS_SHA256 = "764db39887f3db04b913d1739d9dbd46295f0e46e5a2bffa649f1563b56ee4e2"
EXPERIMENT_NAME = "personalisation_pilot_a_m1_h5000"


@dataclass
class H5000Runner:
    root: Path
    dataset_root: Path
    pinyingpt_model: Path
    embedding_model: Path
    pilot_root: Path
    t1_predictions: Path

    @property
    def output_root(self) -> Path:
        return self.pilot_root / "h5000"

    @property
    def cache_root(self) -> Path:
        return self.pilot_root / "cache"

    @property
    def embedding_cache_path(self) -> Path:
        return self.cache_root / "embedding_cache.sqlite3"

    @property
    def t1_manifest_path(self) -> Path:
        return self.root / "results/evaluation/deep_author_v2/design/t1_condition_manifest.jsonl"

    @property
    def work_split_path(self) -> Path:
        return self.root / "results/evaluation/deep_author_v2/design/work_split_manifest.csv"

    @property
    def dev_runner(self) -> PilotRunner:
        return PilotRunner(
            self.root,
            self.dataset_root,
            self.pinyingpt_model,
            self.embedding_model,
            self.pilot_root,
            history_budget=HISTORY_BUDGET,
            prediction_partition="tune",
        )

    def _conditions(self) -> list[dict[str, Any]]:
        text = self.t1_manifest_path.read_text(encoding="utf-8")
        normalized_sha256 = hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()
        if normalized_sha256 != T1_MANIFEST_SHA256:
            raise RuntimeError("frozen T1 condition manifest SHA-256 mismatch")
        conditions = [json.loads(line) for line in text.splitlines()]
        if len(conditions) != 24_000:
            raise RuntimeError("frozen T1 manifest must contain exactly 24,000 conditions")
        return conditions

    def _work_indices(self) -> dict[str, tuple[str, int]]:
        with self.work_split_path.open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        return {str(row["work_id"]): (str(row["split"]), int(row["chronological_index"])) for row in rows}

    def _load_t1_generic(
        self, *, expected_total: int = 24_000, expected_full_short: int = 6_000
    ) -> dict[str, dict[str, Any]]:
        if not self.t1_predictions.is_file():
            raise FileNotFoundError(f"completed T1 prediction cache is absent: {self.t1_predictions}")
        if sha256_file(self.t1_predictions) != T1_PREDICTIONS_SHA256:
            raise RuntimeError("completed T1 prediction cache SHA-256 mismatch")
        conditions = self._conditions()
        expected = {str(row["condition_id"]): row for row in conditions}
        completed: dict[str, dict[str, Any]] = {}
        full_short: dict[str, dict[str, Any]] = {}
        with self.t1_predictions.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                row = json.loads(line)
                condition_id = str(row.get("condition_id", ""))
                if condition_id in completed or condition_id not in expected:
                    raise RuntimeError(f"invalid T1 prediction ID at line {line_number}: {condition_id}")
                T1Runner.validate_cached_prediction(row, expected[condition_id])
                completed[condition_id] = row
                if row["condition"] == "full_short":
                    full_short[str(row["anchor_id"])] = row
        if len(completed) != expected_total or len(full_short) != expected_full_short:
            raise RuntimeError("T1 prediction cache is not complete at the frozen total/Full+Short counts")
        return full_short

    def _test_rows(self) -> list[dict[str, Any]]:
        path = self.output_root / "test_manifest.jsonl"
        if not path.is_file():
            raise RuntimeError("H5000 Test manifest is absent; run --phase prepare")
        rows = read_jsonl(path)
        if len(rows) != 6_000 or len({row["anchor_id"] for row in rows}) != 6_000:
            raise RuntimeError("H5000 Test manifest must contain exactly 6,000 unique anchors")
        if Counter(row["author"] for row in rows) != Counter({author: 1_000 for author in AUTHORS}):
            raise RuntimeError("H5000 Test manifest must contain 1,000 anchors per author")
        return rows

    def _history_rows(self) -> list[dict[str, Any]]:
        history = read_jsonl(self.pilot_root / "history_manifest.jsonl")
        work_indices = self._work_indices()
        normalized = []
        for row in history:
            split, work_index = work_indices[str(row["work_id"])]
            if split != "history" or row["source_split"] != "history":
                raise RuntimeError("H5000 legal history pool contains a non-History row")
            normalized.append(
                {
                    **row,
                    "chronological_position": work_index * 1_000_000_000 + int(row["source_position_start"]),
                }
            )
        return normalized

    @staticmethod
    def _query(row: Mapping[str, Any]) -> PredictionQuery:
        return PredictionQuery(
            row_id=str(row["condition_id"]),
            author=str(row["author"]),
            work_id=str(row["work_id"]),
            chronological_position=int(row["chronological_position"]),
            context=str(row["context"]),
            pinyin=tuple(row["pinyin_segments"]),
        )

    @staticmethod
    def _candidates(generic: Mapping[str, Any]) -> tuple[Candidate, ...]:
        return tuple(
            Candidate(str(row["text"]), int(row["rank"]), float(row["log_probability"]))
            for row in generic["top10_candidates"]
        )

    def prepare(self) -> dict[str, Any]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.dev_runner.prepare()
        conditions = [row for row in self._conditions() if row["condition"] == "full_short"]
        work_indices = self._work_indices()
        rows = []
        for condition in conditions:
            split, work_index = work_indices[str(condition["work_id"])]
            if split != "test":
                raise RuntimeError("frozen T1 Full+Short anchor is not assigned to Test")
            rows.append(
                {
                    **condition,
                    "row_id": condition["condition_id"],
                    "pinyin_segments": str(condition["pinyin_input"]).split(),
                    "target": condition["gold"],
                    "source_split": "test",
                    "pilot_partition": "test",
                    "work_chronological_index": work_index,
                    "chronological_position": work_index * 1_000_000_000 + int(condition["source_position_start"]),
                }
            )
        if len(rows) != 6_000 or Counter(row["author"] for row in rows) != Counter({author: 1_000 for author in AUTHORS}):
            raise RuntimeError("H5000 query population differs from frozen T1 Full+Short")
        expected_anchor_ids = {row["anchor_id"] for row in conditions}
        if {row["anchor_id"] for row in rows} != expected_anchor_ids:
            raise RuntimeError("H5000 anchor IDs differ from frozen T1 Full+Short")
        _write_or_validate_jsonl(self.output_root / "test_manifest.jsonl", rows)
        generic = self._load_t1_generic()
        if set(generic) != expected_anchor_ids:
            raise RuntimeError("T1 Generic Full+Short cache does not match all H5000 anchors")
        summary = {
            "schema_version": 1,
            "experiment": EXPERIMENT_NAME,
            "status": "prepared",
            "condition": "Full + Short",
            "history_budget": HISTORY_BUDGET,
            "history_budget_rule": "5000 most recent strictly-prior same-author legal History-split interactions, before exact segmented-Pinyin filtering",
            "anchors": len(rows),
            "per_author_anchors": dict(Counter(row["author"] for row in rows)),
            "t1_anchor_ids_exact": True,
            "t1_manifest_path": str(self.t1_manifest_path),
            "t1_manifest_sha256": T1_MANIFEST_SHA256,
            "t1_manifest_local_bytes_sha256": sha256_file(self.t1_manifest_path),
            "t1_manifest_hash_normalization": "CRLF/LF normalized to LF before frozen SHA-256 validation",
            "t1_predictions_path": str(self.t1_predictions),
            "t1_predictions_sha256": T1_PREDICTIONS_SHA256,
            "generic_test_inference_required": 0,
            "test_manifest_sha256": sha256_file(self.output_root / "test_manifest.jsonl"),
            "test_gold_used_for_tuning": False,
        }
        _write_or_validate_json(self.output_root / "manifest_summary.json", summary)
        return summary

    def _required_contexts(self, rows: Sequence[Mapping[str, Any]], index: HistoryIndex) -> list[str]:
        contexts = {str(row["context"]) for row in rows}
        for row in rows:
            contexts.update(str(item["context"]) for item in index.visible(self._query(row)))
        return sorted(contexts, key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())

    def embeddings(self) -> dict[str, Any]:
        rows = self._test_rows()
        index = HistoryIndex(self._history_rows(), HISTORY_BUDGET)
        contexts = self._required_contexts(rows, index)
        cache = EmbeddingCache(self.embedding_cache_path)
        embedder = BGEContextEmbedder(self.embedding_model)
        hits = sum(cache.get(context) is not None for context in contexts)
        missing = len(contexts) - hits
        print(f"H5000 embeddings: requested={len(contexts)} cache_hits={hits} missing={missing}", flush=True)
        latencies = []
        started = time.perf_counter()
        added = 0
        try:
            for number, context in enumerate(contexts, start=1):
                if cache.get(context) is None:
                    call_started = time.perf_counter()
                    cache.put(context, embedder.embed(context))
                    latencies.append((time.perf_counter() - call_started) * 1000.0)
                    added += 1
                    if added % 100 == 0:
                        cache.commit()
                if number % 100 == 0 or number == len(contexts):
                    print(f"H5000 embeddings checked={number}/{len(contexts)} added={added}", flush=True)
            cache.commit()
            summary = {
                "status": "complete",
                "history_budget": HISTORY_BUDGET,
                "requested_unique_contexts": len(contexts),
                "cache_hits": hits,
                "missing_at_start": missing,
                "embeddings_added": added,
                "final_cache_rows": cache.count(),
                "cache_path": str(cache.path),
                "embedding_model": embedder.info(),
                "embedding_latency": timing_summary(latencies),
                "elapsed_seconds": time.perf_counter() - started,
            }
            write_json(self.output_root / "embedding_runtime.json", summary)
            return summary
        finally:
            cache.close()

    def tune(self) -> dict[str, Any]:
        selection = self.dev_runner.tune()
        if selection.get("history_budget") != HISTORY_BUDGET:
            raise RuntimeError("Dev hyperparameters were not selected under H5000")
        frozen = {
            **selection,
            "frozen_for": EXPERIMENT_NAME,
            "dev_holdout_population": selection["reported_population"],
            "reported_population": "6000 frozen T1 Test Full+Short anchors after parameter freeze",
            "test_gold_used_for_selection": False,
            "source_path": str(self.pilot_root / "selected_hyperparameters.json"),
            "source_sha256": sha256_file(self.pilot_root / "selected_hyperparameters.json"),
        }
        write_json(self.output_root / "frozen_hyperparameters.json", frozen)
        return frozen

    def evaluate(self) -> dict[str, Any]:
        rows = self._test_rows()
        generic = self._load_t1_generic()
        selection_path = self.output_root / "frozen_hyperparameters.json"
        if not selection_path.is_file():
            raise RuntimeError("H5000 hyperparameters are not frozen; run --phase tune")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("test_gold_used_for_selection") is not False:
            raise RuntimeError("invalid H5000 tuning provenance")
        lambda_frequency = float(selection["frequency"]["lambda_frequency"])
        top_n = int(selection["memory"]["top_n"])
        lambda_memory = float(selection["memory"]["lambda_memory"])
        index = HistoryIndex(self._history_rows(), HISTORY_BUDGET)
        cache = EmbeddingCache(self.embedding_cache_path)
        lookup = EmbeddingLookup(cache)
        frequency_outputs = []
        memory_outputs = []
        metric_rows = []
        frequency_times = []
        retrieval_times = []
        rerank_times = []
        started = time.perf_counter()
        try:
            for number, row in enumerate(rows, start=1):
                query = self._query(row)
                visible = index.visible(query)
                generic_row = generic[str(row["anchor_id"])]
                candidates = self._candidates(generic_row)
                call_started = time.perf_counter()
                frequency = rank_frequency(query, candidates, visible, lambda_frequency=lambda_frequency)
                frequency_times.append((time.perf_counter() - call_started) * 1000.0)
                call_started = time.perf_counter()
                retrieved = retrieve_memory(query, visible, lookup) if visible else ()
                retrieval_times.append((time.perf_counter() - call_started) * 1000.0)
                call_started = time.perf_counter()
                memory = rank_from_retrieved(candidates, retrieved[:top_n], lambda_memory=lambda_memory)
                rerank_times.append((time.perf_counter() - call_started) * 1000.0)
                assert_candidate_pool(candidates, frequency, memory)
                generic_rank = generic_row["gold_rank"]
                frequency_rank = rank_of(frequency, row["gold"])
                memory_rank = rank_of(memory, row["gold"])
                if len({generic_rank is None, frequency_rank is None, memory_rank is None}) != 1:
                    raise AssertionError("Missing@10 changed despite the frozen T1 candidate pool")
                flags = subset_membership(query, row["gold"], visible)
                common = {key: row[key] for key in ("condition_id", "anchor_id", "author", "work_id", "chronological_position", "context", "pinyin_segments", "gold")}
                common.update(flags)
                frequency_outputs.append({**common, "history_budget": HISTORY_BUDGET, "lambda_frequency": lambda_frequency, "candidates": frequency, "gold_rank": frequency_rank})
                memory_candidates = []
                for candidate in memory:
                    value = dict(candidate)
                    value["memory_score"] = value.pop("personal_score")
                    memory_candidates.append(value)
                memory_outputs.append({**common, "history_budget": HISTORY_BUDGET, "top_n": top_n, "lambda_memory": lambda_memory, "candidates": memory_candidates, "retrieved_evidence": retrieved[:top_n], "gold_rank": memory_rank})
                metric_rows.append({**common, "generic_rank": generic_rank, "frequency_rank": frequency_rank, "memory_rank": memory_rank})
                if number % 100 == 0 or number == len(rows):
                    print(f"H5000 evaluate {number}/{len(rows)}", flush=True)
            ambiguous = [row for row in metric_rows if row["ambiguous"]]
            conflict = [row for row in metric_rows if row["conflict"]]
            write_jsonl(self.output_root / "frequency_predictions.jsonl", frequency_outputs)
            write_jsonl(self.output_root / "memory_predictions.jsonl", memory_outputs)
            write_jsonl(self.output_root / "ambiguous_subset.jsonl", ambiguous)
            write_jsonl(self.output_root / "conflict_subset.jsonl", conflict)
            subsets = {"overall": metric_rows, "history_available": [row for row in metric_rows if row["history_available"]], "ambiguous": ambiguous, "conflict": conflict}
            models = {"G0": "generic_rank", "F-H5000": "frequency_rank", "M1-H5000": "memory_rank"}
            metrics = {name: {model: macro_author_metrics(values, key) for model, key in models.items()} for name, values in subsets.items()}
            missing = {model: sum(row[key] is None for row in metric_rows) for model, key in models.items()}
            if len(set(missing.values())) != 1:
                raise AssertionError("Missing@10 is not invariant")
            summary = {
                "schema_version": 1,
                "status": "complete",
                "experiment": EXPERIMENT_NAME,
                "population": "6000 frozen T1 Test anchors, Full + Short",
                "history_budget": HISTORY_BUDGET,
                "history_budget_applied_before_pinyin_filter": True,
                "rows": len(metric_rows),
                "per_author_rows": dict(Counter(row["author"] for row in metric_rows)),
                "selected_hyperparameters": selection,
                "metrics": metrics,
                "subset_rows": {name: len(values) for name, values in subsets.items()},
                "missing_counts": missing,
                "candidate_pool_invariant": True,
                "t1_generic_rows_reused": len(generic),
                "generic_test_inference_rows": 0,
                "test_gold_used_for_tuning": False,
            }
            write_json(self.output_root / "metrics_summary.json", summary)
            author_rows = []
            for model, key in models.items():
                author_rows.extend({"model": model, "author": author, **values} for author, values in macro_author_metrics(metric_rows, key)["per_author"].items())
            write_csv(self.output_root / "metrics_by_author.csv", author_rows, list(author_rows[0]))
            subset_rows = [{"subset": subset, "model": model, **values["macro_author"]} for subset, model_values in metrics.items() for model, values in model_values.items()]
            write_csv(self.output_root / "metrics_by_subset.csv", subset_rows, list(subset_rows[0]))
            runtime = {"status": "complete", "rows": len(metric_rows), "elapsed_seconds": time.perf_counter() - started, "frequency_reranking": timing_summary(frequency_times), "memory_retrieval": timing_summary(retrieval_times), "memory_reranking": timing_summary(rerank_times)}
            write_json(self.output_root / "runtime_summary.json", runtime)
            artifacts = [path for path in sorted(self.output_root.iterdir()) if path.is_file() and path.name != "artifact_checksums.json" and path.suffix in {".json", ".jsonl", ".csv"}]
            write_json(self.output_root / "artifact_checksums.json", {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in artifacts})
            return summary
        finally:
            cache.close()

    def smoke(self) -> dict[str, Any]:
        manifest = self._test_rows()
        rows = [next(row for row in manifest if row["author"] == author) for author in AUTHORS]
        generic = self._load_t1_generic()
        index = HistoryIndex(self._history_rows(), HISTORY_BUDGET)
        checks = []
        for row in rows:
            visible = index.visible(self._query(row))
            checks.append({"anchor_id": row["anchor_id"], "t1_generic_reused": row["anchor_id"] in generic, "visible_same_pinyin": len(visible)})
        summary = {"status": "passed", "research_result": False, "rows": len(rows), "t1_generic_cache_hits": sum(row["t1_generic_reused"] for row in checks), "generic_inference_rows": 0, "history_budget": HISTORY_BUDGET, "checks": checks}
        write_json(self.output_root / "smoke_summary.json", summary)
        return summary

    def all(self) -> dict[str, Any]:
        return {
            "prepare": self.prepare(),
            "dev_generic": self.dev_runner.generic(),
            "dev_embeddings": self.dev_runner.embeddings(),
            "tune": self.tune(),
            "test_embeddings": self.embeddings(),
            "evaluate": self.evaluate(),
        }
