"""Frozen H5000 Personal Vocabulary evaluation: PV0, PV1, and PV2."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from src.evaluation.deep_author_v2 import AUTHORS, canonical_json, sha256_file, write_csv, write_json, write_jsonl
from src.personalisation.context_memory import macro_author_metrics, rank_of
from src.personalisation.h5000 import H5000Runner, HISTORY_BUDGET, T1_PREDICTIONS_SHA256
from src.personalisation.personal_vocabulary import (
    FROZEN_FREQUENCY_LAMBDA,
    FROZEN_M1_TOP_N,
    PERSONAL_VOCABULARY_VERSION,
    PV1_K_GRID,
    PV1_LAMBDA_GRID,
    PV2_CONTEXT_LAMBDA_GRID,
    PV2_CONTEXT_VERSION,
    PersonalVocabularyState,
    lexicon_size_statistics,
    prepare_personal_vocabulary_state,
    rank_pv1,
    rank_pv2,
    transition_counts,
)
from src.personalisation.pilot_a import BGEContextEmbedder, EmbeddingCache, EmbeddingLookup, HistoryIndex


EXPERIMENT_NAME = "personal_vocabulary_h5000"
GENERIC_MISSING = 538
M1_EXPECTED = {
    "metrics_summary.json": "e35fb9efbe3bdd31d7f8354c227efbed2aa178855061955b3ac16a70137e424d",
    "frequency_predictions.jsonl": "71c5626b8318e125776c235dd3cccf45677c884deb2699fd8c1f82e907e0abf6",
    "memory_predictions.jsonl": "75907c88f1d099c3dedc6dc71ee4811ca6258d5267c071bf28ef94d6b95e128b",
}
M2_EXPECTED = {
    "metrics_summary.json": "9ad6acecf41b9f36aa1a1bf1bd702cfc729322c4226a4a6a9e3fde4082c6f6d8",
    "m2_predictions.jsonl": "0a199c31e9fc7b9a35c39aef1cdf48f8a8514b1663fb37416844657eacac79fb",
    "selected_hyperparameters.json": "e47e765b950804ceaed2d2fff5a4d2d1dba0ddeb652ac9bf20ccd89a42a182f4",
}


class PersonalVocabularyH5000Runner:
    def __init__(self, m1_runner: H5000Runner, m2_root: Path, output_root: Path) -> None:
        self.m1 = m1_runner
        self.m2_root = Path(m2_root)
        self.output_root = Path(output_root)

    @property
    def dev_state_path(self) -> Path:
        return self.output_root / "cache/dev_states.jsonl"

    @property
    def test_state_path(self) -> Path:
        return self.output_root / "cache/test_states.jsonl"

    @property
    def selection_path(self) -> Path:
        return self.output_root / "selected_hyperparameters.json"

    @property
    def embedding_cache_path(self) -> Path:
        return self.m1.embedding_cache_path

    def _previous_hashes(self) -> dict[str, Any]:
        values: dict[str, Any] = {"M1": {}, "M2": {}, "T1": {}}
        for name, expected in M1_EXPECTED.items():
            path = self.m1.output_root / name
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(f"frozen M1 artifact changed: {name}")
            values["M1"][name] = actual
        for name, expected in M2_EXPECTED.items():
            path = self.m2_root / name
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(f"frozen M2 artifact changed: {name}")
            values["M2"][name] = actual
        actual_t1 = sha256_file(self.m1.t1_predictions)
        if actual_t1 != T1_PREDICTIONS_SHA256:
            raise RuntimeError("frozen T1 Generic predictions changed")
        values["T1"]["predictions.jsonl"] = actual_t1
        return values

    def _dev_inputs(self) -> tuple[list[dict[str, Any]], HistoryIndex]:
        history, dev = self.m1.dev_runner._manifests()
        tune = [row for row in dev if row["pilot_partition"] == "tune"]
        if len(tune) != 16171 or set(row["author"] for row in tune) != set(AUTHORS):
            raise RuntimeError("frozen Dev tune population differs")
        return tune, HistoryIndex(history + dev, HISTORY_BUDGET)

    def _test_inputs(self) -> tuple[list[dict[str, Any]], HistoryIndex]:
        rows = self.m1._test_rows()
        if len(rows) != 6000 or Counter(row["author"] for row in rows) != Counter({author: 1000 for author in AUTHORS}):
            raise RuntimeError("frozen Test population differs")
        return rows, HistoryIndex(self.m1._history_rows(), HISTORY_BUDGET)

    def _required_contexts(
        self,
        rows: Sequence[Mapping[str, Any]],
        index: HistoryIndex,
        *,
        test: bool,
    ) -> set[str]:
        contexts = set()
        query_fn = self.m1._query if test else self.m1.dev_runner._query
        for row in rows:
            query = query_fn(row)
            contexts.add(query.context)
            contexts.update(str(item["context"]) for item in index.visible(query))
        return contexts

    def ensure_embeddings(self) -> dict[str, Any]:
        dev, dev_index = self._dev_inputs()
        test, test_index = self._test_inputs()
        required = self._required_contexts(dev, dev_index, test=False) | self._required_contexts(test, test_index, test=True)
        cache = EmbeddingCache(self.embedding_cache_path)
        hits = sum(cache.get(context) is not None for context in required)
        # The sort has no scientific meaning; it only fixes resumable write order.
        missing_contexts = sorted(
            (context for context in required if cache.get(context) is None),
            key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
        )
        embedder = BGEContextEmbedder(self.m1.embedding_model)
        started = time.perf_counter()
        added = 0
        try:
            for number, context in enumerate(missing_contexts, start=1):
                if cache.get(context) is None:
                    cache.put(context, embedder.embed(context))
                    added += 1
                if number % 100 == 0 or number == len(missing_contexts):
                    cache.commit()
                    print(f"PV embeddings {number}/{len(missing_contexts)} added={added}", flush=True)
            cache.commit()
            summary = {
                "status": "complete",
                "required_unique_contexts": len(required),
                "embedding_cache_hits": hits,
                "embedding_cache_misses": len(missing_contexts),
                "new_embeddings_computed": added,
                "final_embedding_cache_rows": cache.count(),
                "embedding_cache_path": str(self.embedding_cache_path),
                "elapsed_seconds": time.perf_counter() - started,
                "embedding_model_loaded": embedder.load_ms is not None,
            }
            write_json(self.output_root / "embedding_reuse.json", summary)
            return summary
        finally:
            cache.close()

    def prepare(self) -> dict[str, Any]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        dev, _ = self._dev_inputs()
        test, _ = self._test_inputs()
        generic = self.m1._load_t1_generic()
        if len(generic) != 6000:
            raise RuntimeError("frozen T1 Generic reuse is incomplete")
        embedding = self.ensure_embeddings()
        summary = {
            "schema_version": 1,
            "status": "prepared",
            "experiment": EXPERIMENT_NAME,
            "history_budget": HISTORY_BUDGET,
            "history_budget_applied_before_pinyin_filter": True,
            "exact_segmented_pinyin": True,
            "dev_tune_rows": len(dev),
            "test_rows": len(test),
            "per_author_test_rows": dict(Counter(row["author"] for row in test)),
            "generic_missing_denominator": GENERIC_MISSING,
            "generic_test_inference_rows": 0,
            "t1_generic_rows_reused": len(generic),
            "frequency_lambda_reused": FROZEN_FREQUENCY_LAMBDA,
            "m1_top_n_reused": FROZEN_M1_TOP_N,
            "personal_vocabulary_version": PERSONAL_VOCABULARY_VERSION,
            "pv2_context_version": PV2_CONTEXT_VERSION,
            "pv1_k_grid": list(PV1_K_GRID),
            "pv1_lambda_grid": list(PV1_LAMBDA_GRID),
            "pv2_context_lambda_grid": list(PV2_CONTEXT_LAMBDA_GRID),
            "embedding_reuse": embedding,
            "previous_artifact_sha256": self._previous_hashes(),
            "test_gold_used_for_tuning": False,
            "gold_used_for_vocabulary_construction": False,
            "m2_cross_encoder_used": False,
        }
        write_json(self.output_root / "manifest_summary.json", summary)
        print(
            f"PV prepare: dev={len(dev)} test={len(test)} generic_reused={len(generic)} "
            f"embedding_hits={embedding['embedding_cache_hits']} embedding_misses={embedding['embedding_cache_misses']} "
            f"new_embeddings={embedding['new_embeddings_computed']} previous_hashes_valid=true",
            flush=True,
        )
        return summary

    def _load_or_build_states(
        self,
        rows: Sequence[Mapping[str, Any]],
        index: HistoryIndex,
        generic: Mapping[str, Mapping[str, Any]],
        path: Path,
        *,
        test: bool,
        label: str,
    ) -> dict[str, PersonalVocabularyState]:
        expected = {str(row["condition_id"] if test else row["row_id"]): row for row in rows}
        states: dict[str, PersonalVocabularyState] = {}
        if path.is_file():
            with path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    value = json.loads(line)
                    state = PersonalVocabularyState.from_dict(value)
                    if state.row_id in states or state.row_id not in expected:
                        raise RuntimeError(f"invalid {label} state cache ID at line {line_number}")
                    row = expected[state.row_id]
                    if state.author != row["author"] or state.pinyin != tuple(row["pinyin_segments"]):
                        raise RuntimeError(f"{label} state cache differs from frozen manifest")
                    states[state.row_id] = state
        pending = [row for key, row in expected.items() if key not in states]
        print(f"PV {label} states: requested={len(rows)} reused={len(states)} missing={len(pending)}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        embedding_cache = EmbeddingCache(self.embedding_cache_path)
        embeddings = EmbeddingLookup(embedding_cache)
        query_fn = self.m1._query if test else self.m1.dev_runner._query
        candidate_fn = self.m1._candidates if test else self.m1.dev_runner._candidates
        started = time.perf_counter()
        mode = "a" if path.is_file() and path.stat().st_size else "w"
        try:
            with path.open(mode, encoding="utf-8", newline="\n") as destination:
                for number, row in enumerate(pending, start=1):
                    row_id = str(row["condition_id"] if test else row["row_id"])
                    query = query_fn(row)
                    visible = index.visible(query)
                    generic_key = str(row["anchor_id"] if test else row["row_id"])
                    state = prepare_personal_vocabulary_state(
                        query,
                        candidate_fn(generic[generic_key]),
                        visible,
                        embeddings,
                    )
                    destination.write(canonical_json(state.to_dict()) + "\n")
                    states[row_id] = state
                    if number % 250 == 0 or number == len(pending):
                        destination.flush()
                        elapsed = time.perf_counter() - started
                        rate = number / elapsed if elapsed else 0.0
                        print(f"PV {label} states {number}/{len(pending)}; rate={rate:.2f}/s", flush=True)
            return states
        finally:
            embedding_cache.close()

    def dev_states(self) -> tuple[list[dict[str, Any]], dict[str, PersonalVocabularyState]]:
        rows, index = self._dev_inputs()
        generic = self.m1.dev_runner._load_generic(rows, require_complete=True)
        return rows, self._load_or_build_states(rows, index, generic, self.dev_state_path, test=False, label="dev")

    def test_states(self) -> tuple[list[dict[str, Any]], dict[str, PersonalVocabularyState]]:
        rows, index = self._test_inputs()
        generic = self.m1._load_t1_generic()
        return rows, self._load_or_build_states(rows, index, generic, self.test_state_path, test=True, label="test")

    def pv0(self) -> dict[str, Any]:
        rows, states = self.test_states()
        generic = self.m1._load_t1_generic()
        outputs = []
        missing_rows = []
        sizes = []
        occurrences = Counter()
        for row in rows:
            state = states[str(row["condition_id"])]
            lexicon_targets = {entry.target for entry in state.lexicon}
            sizes.append(len(lexicon_targets))
            original_missing = generic[str(row["anchor_id"])]["gold_rank"] is None
            recoverable = original_missing and str(row["gold"]) in lexicon_targets
            entry = next((entry for entry in state.lexicon if entry.target == row["gold"]), None)
            output = {
                "condition_id": row["condition_id"],
                "anchor_id": row["anchor_id"],
                "author": row["author"],
                "lexicon_size": len(lexicon_targets),
                "generic_missing": original_missing,
                "recoverable_missing": recoverable,
                "gold_historical_occurrences": entry.count if recoverable and entry else 0,
            }
            outputs.append(output)
            if original_missing:
                missing_rows.append(output)
                if recoverable and entry:
                    occurrences[entry.count] += 1
        if len(missing_rows) != GENERIC_MISSING:
            raise AssertionError("original Generic Missing denominator differs from 538")
        recoverable_count = sum(row["recoverable_missing"] for row in missing_rows)
        unrecoverable = GENERIC_MISSING - recoverable_count
        per_author = {}
        for author in AUTHORS:
            values = [row for row in missing_rows if row["author"] == author]
            recovered = sum(row["recoverable_missing"] for row in values)
            per_author[author] = {
                "generic_missing": len(values),
                "recoverable_missing": recovered,
                "unrecoverable_missing": len(values) - recovered,
                "recoverability_rate": recovered / len(values) if values else None,
            }
        summary = {
            "schema_version": 1,
            "status": "complete",
            "variant": "PV0",
            "research_type": "candidate availability / recoverability, not ranking",
            "rows": len(rows),
            "generic_missing": GENERIC_MISSING,
            "recoverable_missing": recoverable_count,
            "unrecoverable_missing": unrecoverable,
            "recoverability_rate": recoverable_count / GENERIC_MISSING,
            "per_author": per_author,
            "recoverable_gold_occurrence_count_distribution": {str(key): value for key, value in sorted(occurrences.items())},
            "personal_lexicon_size": lexicon_size_statistics(sizes),
            "gold_used_after_vocabulary_construction_only": True,
            "gold_used_for_vocabulary_construction": False,
        }
        write_jsonl(self.output_root / "pv0_rows.jsonl", outputs)
        write_json(self.output_root / "pv0_recoverability.json", summary)
        return summary

    def tune(self) -> dict[str, Any]:
        rows, states = self.dev_states()
        pv1_grid: dict[tuple[int, float], list[dict[str, Any]]] = {
            (k, value): [] for k in PV1_K_GRID for value in PV1_LAMBDA_GRID
        }
        started = time.perf_counter()
        for number, row in enumerate(rows, start=1):
            state = states[str(row["row_id"])]
            for k in PV1_K_GRID:
                for value in PV1_LAMBDA_GRID:
                    ranked = rank_pv1(state, k_pv=k, lambda_pv=value)
                    pv1_grid[(k, value)].append({"author": row["author"], "rank": rank_of(ranked, row["gold"])})
            if number % 1000 == 0 or number == len(rows):
                print(f"PV1 tune arithmetic {number}/{len(rows)}", flush=True)
        pv1_search = []
        for k in PV1_K_GRID:
            for value in PV1_LAMBDA_GRID:
                metrics = macro_author_metrics(pv1_grid[(k, value)], "rank")["macro_author"]
                pv1_search.append({"k_pv": k, "lambda_pv": value, **metrics})
        selected_pv1 = max(pv1_search, key=lambda row: (float(row["top1"]), -float(row["lambda_pv"]), -int(row["k_pv"])))
        k_pv = int(selected_pv1["k_pv"])
        lambda_pv = float(selected_pv1["lambda_pv"])
        pv2_grid: dict[float, list[dict[str, Any]]] = {value: [] for value in PV2_CONTEXT_LAMBDA_GRID}
        for number, row in enumerate(rows, start=1):
            state = states[str(row["row_id"])]
            for value in PV2_CONTEXT_LAMBDA_GRID:
                ranked = rank_pv2(state, k_pv=k_pv, lambda_pv=lambda_pv, lambda_ctx=value)
                pv2_grid[value].append({"author": row["author"], "rank": rank_of(ranked, row["gold"])})
            if number % 1000 == 0 or number == len(rows):
                print(f"PV2 tune arithmetic {number}/{len(rows)}", flush=True)
        pv2_search = []
        for value in PV2_CONTEXT_LAMBDA_GRID:
            metrics = macro_author_metrics(pv2_grid[value], "rank")["macro_author"]
            pv2_search.append({"lambda_ctx": value, "frozen_k_pv": k_pv, "frozen_lambda_pv": lambda_pv, **metrics})
        selected_pv2 = max(pv2_search, key=lambda row: (float(row["top1"]), -float(row["lambda_ctx"])))
        write_csv(self.output_root / "pv1_hyperparameter_search.csv", pv1_search, list(pv1_search[0]))
        write_csv(self.output_root / "pv2_hyperparameter_search.csv", pv2_search, list(pv2_search[0]))
        selection = {
            "schema_version": 1,
            "status": "complete",
            "experiment": EXPERIMENT_NAME,
            "selection_population": "chronologically earlier whole-work Dev tune partition",
            "selection_metric": "Macro-author Top-1",
            "tune_rows": len(rows),
            "test_rows_seen_during_selection": 0,
            "test_gold_used_for_selection": False,
            "pv1": {
                "k_grid": list(PV1_K_GRID),
                "lambda_pv_grid": list(PV1_LAMBDA_GRID),
                "selected_k_pv": k_pv,
                "selected_lambda_pv": lambda_pv,
                "tie_break": "lower lambda_pv, then lower k_pv",
            },
            "pv2": {
                "lambda_ctx_grid": list(PV2_CONTEXT_LAMBDA_GRID),
                "selected_lambda_ctx": float(selected_pv2["lambda_ctx"]),
                "frozen_pv1_k_pv": k_pv,
                "frozen_pv1_lambda_pv": lambda_pv,
                "frozen_m1_top_n": FROZEN_M1_TOP_N,
                "tie_break": "lower lambda_ctx",
            },
            "runtime_seconds": time.perf_counter() - started,
        }
        write_json(self.selection_path, selection)
        return selection

    def _selection(self) -> dict[str, Any]:
        if not self.selection_path.is_file():
            raise RuntimeError("Personal Vocabulary Dev selection is absent")
        value = json.loads(self.selection_path.read_text(encoding="utf-8"))
        if value.get("test_gold_used_for_selection") is not False or value.get("test_rows_seen_during_selection") != 0:
            raise RuntimeError("invalid Personal Vocabulary selection provenance")
        if value["pv1"]["k_grid"] != list(PV1_K_GRID) or value["pv1"]["lambda_pv_grid"] != list(PV1_LAMBDA_GRID):
            raise RuntimeError("PV1 grid differs from frozen design")
        if value["pv2"]["lambda_ctx_grid"] != list(PV2_CONTEXT_LAMBDA_GRID):
            raise RuntimeError("PV2 grid differs from frozen design")
        if value["pv2"]["frozen_pv1_k_pv"] != value["pv1"]["selected_k_pv"] or value["pv2"]["frozen_pv1_lambda_pv"] != value["pv1"]["selected_lambda_pv"]:
            raise RuntimeError("PV2 did not freeze PV1 selection")
        return value

    @staticmethod
    def _read_by(path: Path, key: str) -> dict[str, dict[str, Any]]:
        with path.open(encoding="utf-8") as source:
            rows = [json.loads(line) for line in source]
        values = {str(row[key]): row for row in rows}
        if len(values) != len(rows):
            raise RuntimeError(f"duplicate IDs in {path}")
        return values

    def evaluate(self) -> dict[str, Any]:
        selection = self._selection()
        pv0 = json.loads((self.output_root / "pv0_recoverability.json").read_text(encoding="utf-8"))
        rows, states = self.test_states()
        generic = self.m1._load_t1_generic()
        frequency = self._read_by(self.m1.output_root / "frequency_predictions.jsonl", "anchor_id")
        memory = self._read_by(self.m1.output_root / "memory_predictions.jsonl", "anchor_id")
        m2 = self._read_by(self.m2_root / "m2_predictions.jsonl", "anchor_id")
        k_pv = int(selection["pv1"]["selected_k_pv"])
        lambda_pv = float(selection["pv1"]["selected_lambda_pv"])
        lambda_ctx = float(selection["pv2"]["selected_lambda_ctx"])
        predictions = []
        metric_rows = []
        started = time.perf_counter()
        for number, row in enumerate(rows, start=1):
            anchor_id = str(row["anchor_id"])
            state = states[str(row["condition_id"])]
            pv1 = rank_pv1(state, k_pv=k_pv, lambda_pv=lambda_pv)
            pv2 = rank_pv2(state, k_pv=k_pv, lambda_pv=lambda_pv, lambda_ctx=lambda_ctx)
            if len({value["candidate"] for value in pv1}) != len(pv1) or len({value["candidate"] for value in pv2}) != len(pv2):
                raise AssertionError("Personal Vocabulary merge produced duplicate surfaces")
            frozen_frequency = frequency[anchor_id]["candidates"]
            generic_part = [value for value in pv1 if value["source"] == "generic"]
            frozen_by_candidate = {value["candidate"]: value for value in frozen_frequency}
            for value in generic_part:
                frozen = frozen_by_candidate[value["candidate"]]
                if value["final_score"] != frozen["final_score"] or value["frequency_support"] != frozen["personal_score"]:
                    raise AssertionError("PV Generic scoring differs from frozen F-H5000")
            ranks = {
                "generic_rank": generic[anchor_id]["gold_rank"],
                "frequency_rank": frequency[anchor_id]["gold_rank"],
                "memory_rank": memory[anchor_id]["gold_rank"],
                "m2_rank": m2[anchor_id]["gold_rank"],
                "pv1_rank": rank_of(pv1, str(row["gold"])),
                "pv2_rank": rank_of(pv2, str(row["gold"])),
            }
            common = {
                "condition_id": row["condition_id"],
                "anchor_id": anchor_id,
                "author": row["author"],
                "work_id": row["work_id"],
                "gold": row["gold"],
                "generic_missing": ranks["generic_rank"] is None,
                "pv0_recoverable": ranks["generic_rank"] is None and any(entry.target == row["gold"] for entry in state.lexicon),
                **ranks,
            }
            metric_rows.append(common)
            predictions.append({**common, "k_pv": k_pv, "lambda_pv": lambda_pv, "lambda_ctx": lambda_ctx, "pv1_candidates": pv1, "pv2_candidates": pv2})
            if number % 500 == 0 or number == len(rows):
                print(f"PV final shared pass {number}/{len(rows)}", flush=True)
        write_jsonl(self.output_root / "predictions.jsonl", predictions)
        models = {
            "G0": "generic_rank",
            "F-H5000": "frequency_rank",
            "M1-H5000": "memory_rank",
            "M2-H5000": "m2_rank",
            "PV1-H5000": "pv1_rank",
            "PV2-H5000": "pv2_rank",
        }
        metrics = {model: macro_author_metrics(metric_rows, key) for model, key in models.items()}
        missing_rows = [row for row in metric_rows if row["generic_missing"]]
        if len(missing_rows) != GENERIC_MISSING:
            raise AssertionError("original Generic Missing denominator differs from 538")
        recoverable = int(pv0["recoverable_missing"])

        def recovery(key: str) -> dict[str, Any]:
            top10 = sum(row[key] is not None for row in missing_rows)
            top3 = sum(row[key] is not None and row[key] <= 3 for row in missing_rows)
            top1 = sum(row[key] == 1 for row in missing_rows)
            by_author = {}
            for author in AUTHORS:
                values = [row for row in missing_rows if row["author"] == author]
                recovered_author = sum(row[key] is not None for row in values)
                by_author[author] = {
                    "generic_missing": len(values),
                    "recovered_to_top10": recovered_author,
                    "missing_recovery_rate": recovered_author / len(values) if values else None,
                }
            return {
                "recovered_to_top10": top10,
                "recovered_to_top3": top3,
                "recovered_to_top1": top1,
                "missing_recovery_rate": top10 / GENERIC_MISSING,
                "recoverable_recovery_rate": top10 / recoverable if recoverable else None,
                "per_author": by_author,
            }

        pv1_recovery = recovery("pv1_rank")
        pv2_recovery = recovery("pv2_rank")
        f_to_pv1 = transition_counts([row["frequency_rank"] for row in metric_rows], [row["pv1_rank"] for row in metric_rows])
        f_to_pv1["net_help"] = f_to_pv1.get("helped", 0) - f_to_pv1.get("harmed", 0)
        pv1_to_pv2 = transition_counts(
            [row["pv1_rank"] for row in metric_rows],
            [row["pv2_rank"] for row in metric_rows],
            helped_name="helped_by_context",
            harmed_name="harmed_by_context",
        )
        pv1_to_pv2["net_context_help"] = pv1_to_pv2.get("helped_by_context", 0) - pv1_to_pv2.get("harmed_by_context", 0)
        if sum(f_to_pv1.get(key, 0) for key in ("helped", "harmed", "unchanged_correct", "unchanged_wrong")) != 6000:
            raise AssertionError("F to PV1 accounting does not sum to 6000")
        if sum(pv1_to_pv2.get(key, 0) for key in ("helped_by_context", "harmed_by_context", "unchanged_correct", "unchanged_wrong")) != 6000:
            raise AssertionError("PV1 to PV2 accounting does not sum to 6000")
        generic_covered = [row for row in metric_rows if not row["generic_missing"]]
        covered_harm = sum(row["frequency_rank"] == 1 and row["pv1_rank"] != 1 for row in generic_covered)
        previous_after = self._previous_hashes()
        previous_before = json.loads((self.output_root / "manifest_summary.json").read_text(encoding="utf-8"))["previous_artifact_sha256"]
        if previous_before != previous_after:
            raise AssertionError("previous completed artifacts changed during Personal Vocabulary")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "experiment": EXPERIMENT_NAME,
            "rows": 6000,
            "per_author_rows": dict(Counter(row["author"] for row in metric_rows)),
            "selection": selection,
            "metrics": metrics,
            "pv0": pv0,
            "pv1_recovery": pv1_recovery,
            "pv2_recovery": pv2_recovery,
            "f_to_pv1_transitions": f_to_pv1,
            "pv1_to_pv2_transitions": pv1_to_pv2,
            "generic_covered_rows": len(generic_covered),
            "covered_case_harmed": covered_harm,
            "covered_case_harm_rate": covered_harm / len(generic_covered),
            "generic_test_inference_rows": 0,
            "t1_generic_rows_reused": 6000,
            "embedding_reuse": json.loads((self.output_root / "embedding_reuse.json").read_text(encoding="utf-8")),
            "frequency_semantics_reused": True,
            "m1_top_n_reused": FROZEN_M1_TOP_N,
            "m2_cross_encoder_used": False,
            "test_gold_used_for_tuning": False,
            "gold_used_for_vocabulary_construction": False,
            "previous_artifacts_unchanged": True,
            "previous_artifact_sha256": previous_after,
            "runtime_seconds": time.perf_counter() - started,
        }
        write_json(self.output_root / "metrics_summary.json", summary)
        author_rows = []
        for model, values in metrics.items():
            author_rows.extend({"model": model, "author": author, **row} for author, row in values["per_author"].items())
        write_csv(self.output_root / "metrics_by_author.csv", author_rows, list(author_rows[0]))
        write_json(self.output_root / "artifact_checksums.json", {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(self.output_root.iterdir())
            if path.is_file() and path.name != "artifact_checksums.json" and path.suffix in {".json", ".jsonl", ".csv"}
        })
        return summary

    def smoke(self) -> dict[str, Any]:
        rows, index = self._dev_inputs()
        sample = [next(row for row in rows if row["author"] == author) for author in AUTHORS]
        generic = self.m1.dev_runner._load_generic(sample, require_complete=True)
        cache = EmbeddingCache(self.embedding_cache_path)
        lookup = EmbeddingLookup(cache)
        started = time.perf_counter()
        try:
            outputs = []
            for row in sample:
                query = self.m1.dev_runner._query(row)
                state = prepare_personal_vocabulary_state(query, self.m1.dev_runner._candidates(generic[row["row_id"]]), index.visible(query), lookup)
                pv1 = rank_pv1(state, k_pv=1, lambda_pv=0.5)
                pv2 = rank_pv2(state, k_pv=1, lambda_pv=0.5, lambda_ctx=0.5)
                outputs.append({"row_id": row["row_id"], "author": row["author"], "lexicon_size": len(state.lexicon), "pv1_candidates": len(pv1), "pv2_candidates": len(pv2)})
            summary = {
                "status": "passed",
                "research_result": False,
                "rows": len(sample),
                "authors": sorted(row["author"] for row in sample),
                "generic_cache_rows_reused": len(sample),
                "generic_inference_rows": 0,
                "bge_cache_rows": cache.count(),
                "new_embeddings_computed": 0,
                "frequency_semantics_reused": True,
                "m2_cross_encoder_used": False,
                "current_gold_in_state": False,
                "elapsed_seconds": time.perf_counter() - started,
                "outputs": outputs,
            }
            write_json(self.output_root / "smoke_summary.json", summary)
            return summary
        finally:
            cache.close()

    def all(self) -> dict[str, Any]:
        return {
            "prepare": self.prepare(),
            "pv0": self.pv0(),
            "tune": self.tune(),
            "evaluate": self.evaluate(),
        }
