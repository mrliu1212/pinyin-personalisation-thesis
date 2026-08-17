"""M2: candidate-aware Cross-Encoder support over frozen M1 H5000 retrieval."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from src.evaluation.deep_author_v2 import AUTHORS, sha256_file, write_csv, write_json, write_jsonl
from src.personalisation.candidate_memory_m2 import (
    BGEReranker,
    INPUT_TEMPLATE_VERSION,
    PairIdentity,
    PairScoreCache,
    RERANKER_LICENSE,
    RERANKER_MODEL_SHA256,
    RERANKER_REPOSITORY,
    RERANKER_REVISION,
    RERANKER_TOKENIZER_SHA256,
    TRUNCATION_VERSION,
    rank_m2,
)
from src.personalisation.context_memory import (
    Candidate,
    assert_candidate_pool,
    macro_author_metrics,
    rank_of,
    retrieve_memory,
    subset_membership,
)
from src.personalisation.h5000 import H5000Runner, HISTORY_BUDGET
from src.personalisation.pilot_a import EmbeddingCache, EmbeddingLookup, HistoryIndex, timing_summary


M2_RETRIEVAL_KS = (10, 20)
M2_LAMBDAS = (0.5, 1.0, 2.0, 4.0)
EXPERIMENT_NAME = "personalisation_m2_h5000"
SUPPORT_AGGREGATION = "sigmoid(raw_logit), sum by historical target, divide by total retrieved sigmoid support"


class M2H5000Runner:
    """Resume-safe Dev selection and one-shot frozen Test evaluation for M2."""

    def __init__(
        self,
        m1_runner: H5000Runner,
        reranker_model: Path,
        output_root: Path,
        *,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self.m1 = m1_runner
        self.reranker_model = Path(reranker_model)
        self.output_root = Path(output_root)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)

    @property
    def pair_cache_path(self) -> Path:
        return self.output_root / "cache/pair_scores.sqlite3"

    @property
    def selection_path(self) -> Path:
        return self.output_root / "selected_hyperparameters.json"

    def _new_reranker(self) -> BGEReranker:
        return BGEReranker(
            self.reranker_model,
            revision=RERANKER_REVISION,
            model_sha256=RERANKER_MODEL_SHA256,
            tokenizer_sha256=RERANKER_TOKENIZER_SHA256,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )

    def _new_pair_cache(self) -> PairScoreCache:
        return PairScoreCache(
            self.pair_cache_path,
            model_revision=RERANKER_REVISION,
            model_sha256=RERANKER_MODEL_SHA256,
            tokenizer_sha256=RERANKER_TOKENIZER_SHA256,
            max_length=self.max_length,
            dtype="float16",
        )

    def _validate_model_files(self) -> dict[str, Any]:
        expected = {
            "model.safetensors": RERANKER_MODEL_SHA256,
            "tokenizer.json": RERANKER_TOKENIZER_SHA256,
            "config.json": "289adf7ada1eb6b4afa7589a48a032d45a076cf2e46dcdb3b4cabc33be14f708",
            "sentencepiece.bpe.model": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
            "special_tokens_map.json": "d5469a60db23249c7f8945013d78df30b44b6bf686c6bb4740f4223f77b1b535",
            "tokenizer_config.json": "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b",
        }
        artifacts = {}
        for name, digest in expected.items():
            path = self.reranker_model / name
            if not path.is_file() or sha256_file(path) != digest:
                raise RuntimeError(f"frozen M2 reranker artifact mismatch: {name}")
            artifacts[name] = {"bytes": path.stat().st_size, "sha256": digest}
        return artifacts

    def _m1_hashes(self) -> dict[str, str]:
        paths = {
            "metrics_summary.json": self.m1.output_root / "metrics_summary.json",
            "frequency_predictions.jsonl": self.m1.output_root / "frequency_predictions.jsonl",
            "memory_predictions.jsonl": self.m1.output_root / "memory_predictions.jsonl",
        }
        if any(not path.is_file() for path in paths.values()):
            raise RuntimeError("completed M1-H5000 artifacts are absent")
        metrics = json.loads(paths["metrics_summary.json"].read_text(encoding="utf-8"))
        if metrics.get("status") != "complete" or metrics.get("rows") != 6000:
            raise RuntimeError("M1-H5000 is not a complete frozen 6,000-row result")
        return {name: sha256_file(path) for name, path in paths.items()}

    def prepare(self) -> dict[str, Any]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        m1_manifest = self.m1.prepare()
        if m1_manifest["anchors"] != 6000:
            raise RuntimeError("M2 population differs from frozen M1/T1 anchors")
        artifacts = self._validate_model_files()
        summary = {
            "schema_version": 1,
            "status": "prepared",
            "experiment": EXPERIMENT_NAME,
            "population": "exact frozen T1 Full+Short Test anchors",
            "rows": 6000,
            "per_author_rows": {author: 1000 for author in AUTHORS},
            "history_budget": HISTORY_BUDGET,
            "history_budget_applied_before_pinyin_filter": True,
            "stage1": "pinned BGE context cosine retrieval from exact same-Pinyin H5000-visible history",
            "retrieval_k_grid": list(M2_RETRIEVAL_KS),
            "lambda_m2_grid": list(M2_LAMBDAS),
            "support_aggregation": SUPPORT_AGGREGATION,
            "generic_score_normalization": "unchanged within-query population z-score",
            "candidate_pool": "unchanged frozen Generic Top-10 surface",
            "reranker": {
                "repository": RERANKER_REPOSITORY,
                "revision": RERANKER_REVISION,
                "license": RERANKER_LICENSE,
                "model_path": str(self.reranker_model),
                "artifacts": artifacts,
                "input_template_version": INPUT_TEMPLATE_VERSION,
                "truncation_version": TRUNCATION_VERSION,
                "max_length": self.max_length,
                "dtype": "float16",
                "device": "cuda",
            },
            "m1_artifact_sha256_before_m2": self._m1_hashes(),
            "test_gold_used_for_tuning": False,
        }
        write_json(self.output_root / "manifest_summary.json", summary)
        embedding_cache = EmbeddingCache(self.m1.embedding_cache_path)
        try:
            print(
                "M2 prepare: frozen_test_rows=6000 t1_generic_rows_reused=6000 "
                f"bge_embedding_cache_rows={embedding_cache.count()} m1_artifacts_hash_checked=true",
                flush=True,
            )
        finally:
            embedding_cache.close()
        return summary

    @staticmethod
    def _pair(query: Any, history: Mapping[str, Any]) -> PairIdentity:
        target = str(history["target"])
        return PairIdentity.from_query_history(query, history, target)

    @staticmethod
    def _stage1(
        query: Any,
        visible: Sequence[Mapping[str, Any]],
        embeddings: Any,
        k: int,
    ) -> tuple[Mapping[str, Any], ...]:
        if not visible:
            return ()
        retrieved = retrieve_memory(query, visible, embeddings)[:k]
        by_id = {str(row["row_id"]): row for row in visible}
        return tuple(by_id[str(row["historical_interaction_id"])] for row in retrieved)

    def _score_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        index: HistoryIndex,
        embeddings: EmbeddingLookup,
        *,
        k: int,
        label: str,
    ) -> dict[str, Any]:
        reranker = self._new_reranker()
        reranker.load()
        cache = self._new_pair_cache()
        requested: list[PairIdentity] = []
        seen: set[str] = set()
        for row in rows:
            query = self.m1.dev_runner._query(row) if "pilot_partition" in row else self.m1._query(row)
            for history in self._stage1(query, index.visible(query), embeddings, k):
                pair = self._pair(query, history)
                key = cache.key(pair)
                if key not in seen:
                    seen.add(key)
                    requested.append(pair)
        hits = sum(cache.get(pair) is not None for pair in requested)
        pending = [pair for pair in requested if cache.get(pair) is None]
        print(f"M2 {label} pair cache: requested={len(requested)} reused={hits} missing={len(pending)}", flush=True)
        started = time.perf_counter()
        latencies: list[float] = []
        truncated_current = 0
        truncated_history = 0
        added = 0
        try:
            for start in range(0, len(pending), self.batch_size):
                pairs = pending[start : start + self.batch_size]
                prepared = [reranker.prepare(pair) for pair in pairs]
                call_started = time.perf_counter()
                scores = reranker.score_prepared(prepared)
                elapsed_ms = (time.perf_counter() - call_started) * 1000.0
                latencies.extend([elapsed_ms / len(pairs)] * len(pairs))
                for pair, prepared_pair, score in zip(pairs, prepared, scores):
                    cache.put(pair, prepared_pair, score)
                    truncated_current += prepared_pair.current_context_truncated
                    truncated_history += prepared_pair.historical_context_truncated
                    added += 1
                if added % (self.batch_size * 10) == 0 or start + len(pairs) == len(pending):
                    cache.commit()
                    elapsed = time.perf_counter() - started
                    rate = added / elapsed if elapsed else 0.0
                    eta = (len(pending) - added) / rate if rate else 0.0
                    print(f"M2 {label} pairs {added}/{len(pending)}; rate={rate:.2f}/s; eta={eta:.1f}s", flush=True)
            cache.commit()
            summary = {
                "status": "complete",
                "label": label,
                "queries": len(rows),
                "retrieval_k": k,
                "requested_unique_pairs": len(requested),
                "cache_hits": hits,
                "missing_at_start": len(pending),
                "pairs_added": added,
                "final_cache_rows": cache.count(),
                "current_context_truncations_added": truncated_current,
                "history_context_truncations_added": truncated_history,
                "pair_latency_ms": timing_summary(latencies),
                "elapsed_seconds": time.perf_counter() - started,
                "reranker": reranker.info(),
                "cache_path": str(self.pair_cache_path),
                "bge_embedding_cache_path": str(embeddings.cache.path),
                "bge_embedding_cache_rows_reused": embeddings.cache.count(),
            }
            write_json(self.output_root / f"{label}_pair_runtime.json", summary)
            return summary
        finally:
            cache.close()

    def _dev_inputs(self) -> tuple[list[dict[str, Any]], HistoryIndex, EmbeddingCache, EmbeddingLookup]:
        history, dev = self.m1.dev_runner._manifests()
        tune = [row for row in dev if row["pilot_partition"] == "tune"]
        if len(tune) != 16171 or set(row["author"] for row in tune) != set(AUTHORS):
            raise RuntimeError("frozen M1 Dev tune population differs")
        index = HistoryIndex(history + dev, HISTORY_BUDGET)
        embeddings = EmbeddingCache(self.m1.embedding_cache_path)
        return tune, index, embeddings, EmbeddingLookup(embeddings)

    def dev_scores(self) -> dict[str, Any]:
        tune, index, cache, lookup = self._dev_inputs()
        try:
            return self._score_rows(tune, index, lookup, k=max(M2_RETRIEVAL_KS), label="dev")
        finally:
            cache.close()

    def _evidence(
        self,
        query: Any,
        histories: Sequence[Mapping[str, Any]],
        cache: PairScoreCache,
    ) -> tuple[dict[str, Any], ...]:
        values = []
        for history in histories:
            pair = self._pair(query, history)
            score = cache.get(pair)
            if score is None:
                raise RuntimeError("required M2 pair score is absent; resume pair scoring")
            values.append(
                {
                    "historical_interaction_id": pair.historical_id,
                    "historical_target": pair.historical_target,
                    "raw_score": score["raw_score"],
                    "input_tokens": score["input_tokens"],
                    "current_context_truncated": score["current_context_truncated"],
                    "historical_context_truncated": score["historical_context_truncated"],
                }
            )
        return tuple(values)

    def tune(self) -> dict[str, Any]:
        tune, index, embedding_cache, embeddings = self._dev_inputs()
        generic = self.m1.dev_runner._load_generic(tune, require_complete=True)
        pair_cache = self._new_pair_cache()
        rows_by_grid: dict[tuple[int, float], list[dict[str, Any]]] = {
            (k, value): [] for k in M2_RETRIEVAL_KS for value in M2_LAMBDAS
        }
        started = time.perf_counter()
        try:
            for number, row in enumerate(tune, start=1):
                query = self.m1.dev_runner._query(row)
                visible = index.visible(query)
                stage1 = self._stage1(query, visible, embeddings, max(M2_RETRIEVAL_KS))
                candidates = self.m1.dev_runner._candidates(generic[str(row["row_id"])])
                for k in M2_RETRIEVAL_KS:
                    evidence = self._evidence(query, stage1[:k], pair_cache)
                    for value in M2_LAMBDAS:
                        ranked = rank_m2(candidates, evidence, lambda_m2=value)
                        rows_by_grid[(k, value)].append({"author": row["author"], "rank": rank_of(ranked, row["gold"])})
                if number % 500 == 0 or number == len(tune):
                    print(f"M2 tune metrics {number}/{len(tune)}", flush=True)
            search = []
            for k in M2_RETRIEVAL_KS:
                for value in M2_LAMBDAS:
                    metrics = macro_author_metrics(rows_by_grid[(k, value)], "rank")["macro_author"]
                    search.append({"retrieval_k": k, "lambda_m2": value, **metrics})
            selected = max(search, key=lambda row: (float(row["top1"]), -float(row["lambda_m2"]), -int(row["retrieval_k"])))
            write_csv(self.output_root / "hyperparameter_search.csv", search, list(search[0]))
            result = {
                "status": "complete",
                "experiment": EXPERIMENT_NAME,
                "selection_population": "chronologically earlier whole-work Dev tune partition",
                "selection_metric": "Macro-author Top-1",
                "tie_break": "lower lambda_m2, then lower retrieval_k",
                "tune_rows": len(tune),
                "tune_work_ids": sorted({row["work_id"] for row in tune}),
                "retrieval_k_grid": list(M2_RETRIEVAL_KS),
                "lambda_m2_grid": list(M2_LAMBDAS),
                "selected": {"retrieval_k": int(selected["retrieval_k"]), "lambda_m2": float(selected["lambda_m2"])},
                "support_aggregation": SUPPORT_AGGREGATION,
                "generic_score_normalization": "unchanged within-query population z-score",
                "test_gold_used_for_selection": False,
                "test_rows_seen_during_selection": 0,
                "runtime_seconds": time.perf_counter() - started,
                "pair_cache_path": str(self.pair_cache_path),
            }
            write_json(self.selection_path, result)
            return result
        finally:
            pair_cache.close()
            embedding_cache.close()

    def _selection(self) -> dict[str, Any]:
        if not self.selection_path.is_file():
            raise RuntimeError("M2 Dev selection is absent; run --phase tune")
        value = json.loads(self.selection_path.read_text(encoding="utf-8"))
        if value.get("test_gold_used_for_selection") is not False or value.get("test_rows_seen_during_selection") != 0:
            raise RuntimeError("invalid M2 selection provenance")
        if value.get("retrieval_k_grid") != list(M2_RETRIEVAL_KS) or value.get("lambda_m2_grid") != list(M2_LAMBDAS):
            raise RuntimeError("M2 search grid differs from the frozen design")
        return value

    def _test_inputs(self) -> tuple[list[dict[str, Any]], HistoryIndex, EmbeddingCache, EmbeddingLookup]:
        rows = self.m1._test_rows()
        if len(rows) != 6000 or Counter(row["author"] for row in rows) != Counter({author: 1000 for author in AUTHORS}):
            raise RuntimeError("M2 Test population differs from frozen T1/M1")
        index = HistoryIndex(self.m1._history_rows(), HISTORY_BUDGET)
        embeddings = EmbeddingCache(self.m1.embedding_cache_path)
        return rows, index, embeddings, EmbeddingLookup(embeddings)

    def test_scores(self) -> dict[str, Any]:
        selection = self._selection()
        rows, index, cache, lookup = self._test_inputs()
        try:
            return self._score_rows(rows, index, lookup, k=int(selection["selected"]["retrieval_k"]), label="test")
        finally:
            cache.close()

    @staticmethod
    def _read_by(path: Path, key: str) -> dict[str, dict[str, Any]]:
        with path.open(encoding="utf-8") as source:
            rows = [json.loads(line) for line in source]
        result = {str(row[key]): row for row in rows}
        if len(result) != len(rows):
            raise RuntimeError(f"duplicate IDs in {path}")
        return result

    def evaluate(self) -> dict[str, Any]:
        selection = self._selection()
        k = int(selection["selected"]["retrieval_k"])
        lambda_m2 = float(selection["selected"]["lambda_m2"])
        rows, index, embedding_cache, embeddings = self._test_inputs()
        generic = self.m1._load_t1_generic()
        frequency = self._read_by(self.m1.output_root / "frequency_predictions.jsonl", "anchor_id")
        memory = self._read_by(self.m1.output_root / "memory_predictions.jsonl", "anchor_id")
        if set(generic) != set(frequency) or set(generic) != set(memory):
            raise RuntimeError("M1/T1 Test IDs differ")
        pair_cache = self._new_pair_cache()
        metric_rows = []
        predictions = []
        started = time.perf_counter()
        try:
            for number, row in enumerate(rows, start=1):
                anchor_id = str(row["anchor_id"])
                query = self.m1._query(row)
                visible = index.visible(query)
                stage1 = self._stage1(query, visible, embeddings, k)
                evidence = self._evidence(query, stage1, pair_cache)
                candidates = self.m1._candidates(generic[anchor_id])
                ranked = rank_m2(candidates, evidence, lambda_m2=lambda_m2)
                assert_candidate_pool(candidates, frequency[anchor_id]["candidates"], ranked)
                if {candidate.text for candidate in candidates} != {str(value["candidate"]) for value in memory[anchor_id]["candidates"]}:
                    raise AssertionError("M1 candidate pool differs from frozen G0")
                flags = subset_membership(query, str(row["gold"]), visible)
                for name in ("history_available", "ambiguous", "conflict"):
                    if bool(flags[name]) != bool(frequency[anchor_id][name]) or bool(flags[name]) != bool(memory[anchor_id][name]):
                        raise AssertionError(f"M2 diagnostic subset differs from M1: {name}")
                ranks = {
                    "generic_rank": generic[anchor_id]["gold_rank"],
                    "frequency_rank": frequency[anchor_id]["gold_rank"],
                    "memory_rank": memory[anchor_id]["gold_rank"],
                    "m2_rank": rank_of(ranked, str(row["gold"])),
                }
                if len({value is None for value in ranks.values()}) != 1:
                    raise AssertionError("Missing@10 changed despite the frozen candidate pool")
                common = {
                    "condition_id": row["condition_id"],
                    "anchor_id": anchor_id,
                    "author": row["author"],
                    "work_id": row["work_id"],
                    "chronological_position": row["chronological_position"],
                    "context": row["context"],
                    "pinyin_segments": row["pinyin_segments"],
                    "gold": row["gold"],
                    **flags,
                }
                metric_rows.append({**common, **ranks})
                predictions.append(
                    {
                        **common,
                        "history_budget": HISTORY_BUDGET,
                        "retrieval_k": k,
                        "lambda_m2": lambda_m2,
                        "candidates": ranked,
                        "retrieved_evidence": evidence,
                        "gold_rank": ranks["m2_rank"],
                    }
                )
                if number % 100 == 0 or number == len(rows):
                    print(f"M2 evaluate {number}/{len(rows)}", flush=True)
            write_jsonl(self.output_root / "m2_predictions.jsonl", predictions)
            subsets = {
                "overall": metric_rows,
                "history_available": [row for row in metric_rows if row["history_available"]],
                "ambiguous": [row for row in metric_rows if row["ambiguous"]],
                "conflict": [row for row in metric_rows if row["conflict"]],
            }
            expected_counts = {"overall": 6000, "history_available": 3904, "ambiguous": 1661, "conflict": 377}
            if {name: len(values) for name, values in subsets.items()} != expected_counts:
                raise AssertionError("M2 diagnostic row IDs/counts differ from completed M1 H5000")
            models = {"G0": "generic_rank", "F-H5000": "frequency_rank", "M1-H5000": "memory_rank", "M2-H5000": "m2_rank"}
            metrics = {
                name: {model: macro_author_metrics(values, key) for model, key in models.items()}
                for name, values in subsets.items()
            }
            missing = {model: sum(row[key] is None for row in metric_rows) for model, key in models.items()}
            if len(set(missing.values())) != 1:
                raise AssertionError("Missing@10 is not invariant across G0/F/M1/M2")
            m1_hashes_after = self._m1_hashes()
            before = json.loads((self.output_root / "manifest_summary.json").read_text(encoding="utf-8"))["m1_artifact_sha256_before_m2"]
            if before != m1_hashes_after:
                raise AssertionError("completed M1 artifacts changed during M2")
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
                "subset_rows": expected_counts,
                "missing_counts": missing,
                "candidate_pool_invariant": True,
                "t1_generic_rows_reused": len(generic),
                "generic_test_inference_rows": 0,
                "test_gold_used_for_tuning": False,
                "m1_artifacts_unchanged": True,
                "m1_artifact_sha256": m1_hashes_after,
            }
            write_json(self.output_root / "metrics_summary.json", summary)
            author_rows = []
            for model, key in models.items():
                author_rows.extend({"model": model, "author": author, **values} for author, values in macro_author_metrics(metric_rows, key)["per_author"].items())
            write_csv(self.output_root / "metrics_by_author.csv", author_rows, list(author_rows[0]))
            subset_metrics = [
                {"subset": subset, "model": model, **values["macro_author"]}
                for subset, model_values in metrics.items()
                for model, values in model_values.items()
            ]
            write_csv(self.output_root / "metrics_by_subset.csv", subset_metrics, list(subset_metrics[0]))
            runtime = {"status": "complete", "rows": len(metric_rows), "elapsed_seconds": time.perf_counter() - started, "pair_cache_rows": pair_cache.count()}
            write_json(self.output_root / "runtime_summary.json", runtime)
            artifacts = [path for path in sorted(self.output_root.iterdir()) if path.is_file() and path.name != "artifact_checksums.json" and path.suffix in {".json", ".jsonl", ".csv"}]
            write_json(self.output_root / "artifact_checksums.json", {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in artifacts})
            return summary
        finally:
            pair_cache.close()
            embedding_cache.close()

    def benchmark(self, sample_queries: int = 64) -> dict[str, Any]:
        tune, index, embedding_cache, embeddings = self._dev_inputs()
        rows = []
        for row in tune:
            query = self.m1.dev_runner._query(row)
            if len(index.visible(query)) >= max(M2_RETRIEVAL_KS):
                rows.append(row)
            if len(rows) == sample_queries:
                break
        if not rows:
            raise RuntimeError("no DEV rows have enough real history for the M2 benchmark")
        reranker = self._new_reranker()
        reranker.load()
        pairs = []
        try:
            for row in rows:
                query = self.m1.dev_runner._query(row)
                pairs.extend(self._pair(query, history) for history in self._stage1(query, index.visible(query), embeddings, 20))
        finally:
            embedding_cache.close()
        import torch

        prepared_k20 = [reranker.prepare(pair) for pair in pairs]
        prepared_k10 = [pair for start in range(0, len(prepared_k20), 20) for pair in prepared_k20[start : start + 10]]
        reranker.score_prepared(prepared_k20[: min(len(prepared_k20), self.batch_size)])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started_k10 = time.perf_counter()
        reranker.score_prepared(prepared_k10)
        torch.cuda.synchronize()
        elapsed_k10 = time.perf_counter() - started_k10
        started_k20 = time.perf_counter()
        reranker.score_prepared(prepared_k20)
        torch.cuda.synchronize()
        elapsed_k20 = time.perf_counter() - started_k20
        result = {
            "status": "passed",
            "research_result": False,
            "selection_signal_used": False,
            "dev_sample_queries": len(rows),
            "pairs": len(pairs),
            "batch_size": self.batch_size,
            "pairs_per_second_k10": len(prepared_k10) / elapsed_k10,
            "pairs_per_second_k20": len(prepared_k20) / elapsed_k20,
            "queries_per_second_k10": len(rows) / elapsed_k10,
            "queries_per_second_k20": len(rows) / elapsed_k20,
            "seconds_per_query_k10": elapsed_k10 / len(rows),
            "seconds_per_query_k20": elapsed_k20 / len(rows),
            "elapsed_seconds_k10": elapsed_k10,
            "elapsed_seconds_k20": elapsed_k20,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "input_tokens_max": max(row.input_tokens for row in prepared_k20),
            "current_context_truncations": sum(row.current_context_truncated for row in prepared_k20),
            "history_context_truncations": sum(row.historical_context_truncated for row in prepared_k20),
            "reranker": reranker.info(),
        }
        write_json(self.output_root / "dev_runtime_benchmark.json", result)
        return result

    def smoke(self) -> dict[str, Any]:
        tune, index, embedding_cache, embeddings = self._dev_inputs()
        rows = [next(row for row in tune if row["author"] == author and index.visible(self.m1.dev_runner._query(row))) for author in AUTHORS]
        try:
            score_summary = self._score_rows(rows, index, embeddings, k=10, label="smoke")
            generic = self.m1.dev_runner._load_generic(rows, require_complete=True)
            pair_cache = self._new_pair_cache()
            try:
                for row in rows:
                    query = self.m1.dev_runner._query(row)
                    stage1 = self._stage1(query, index.visible(query), embeddings, 10)
                    evidence = self._evidence(query, stage1, pair_cache)
                    candidates = self.m1.dev_runner._candidates(generic[str(row["row_id"])])
                    ranked = rank_m2(candidates, evidence, lambda_m2=1.0)
                    if {candidate.text for candidate in candidates} != {str(value["candidate"]) for value in ranked}:
                        raise AssertionError("M2 smoke changed the candidate pool")
            finally:
                pair_cache.close()
        finally:
            embedding_cache.close()
        result = {
            "status": "passed",
            "research_result": False,
            "rows": len(rows),
            "authors": sorted(row["author"] for row in rows),
            "candidate_pool_invariant": True,
            "current_gold_in_model_input": False,
            "future_text_in_model_input": False,
            "pair_scores": score_summary["requested_unique_pairs"],
            "cuda_device": score_summary["reranker"]["device_name"],
            "pair_cache_resume_ready": True,
        }
        write_json(self.output_root / "smoke_summary.json", result)
        return result

    def all(self) -> dict[str, Any]:
        return {
            "prepare": self.prepare(),
            "dev_scores": self.dev_scores(),
            "tune": self.tune(),
            "test_scores": self.test_scores(),
            "evaluate": self.evaluate(),
        }
