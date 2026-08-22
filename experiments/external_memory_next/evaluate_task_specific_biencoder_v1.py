"""One-shot Train-Val evaluation of the frozen task-specific bi-encoder."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import statistics
import time
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.context_comparison import run_full_retune_final_trainval_dev_v1 as retune
from experiments.context_comparison import run_full_transfer_initial_final_v1 as base
from src.personalisation.task_specific_biencoder import (
    SharedContextEncoder,
    group_retrieval_metrics,
    ranking_metrics,
    read_jsonl,
    refuse_closed_path,
    sha256_file,
    sha256_tree,
    transition_counts,
    write_json,
    write_jsonl,
)


EXPECTED_FIT_SHA256 = "547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6"
EXPECTED_VAL_SHA256 = "d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220"
EXPECTED_STAGE2_SHA256 = "d413d02650a759c0a759e1845212e68b2d948c1a1d20fc88abfd89ea7973bc64"
EXPECTED_PREDICTIONS_SHA256 = "f3e902e5a9e7d25e62799b9abb719026c336381eacc42999d1e7edccf2731b22"


class ReadOnlyVectors:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection = sqlite3.connect(str(path))

    def get(self, context: str) -> np.ndarray | None:
        row = self.connection.execute("SELECT dim,vector FROM embeddings WHERE context=?", (context,)).fetchone()
        if row is None:
            return None
        dim, blob = row
        value = np.frombuffer(blob, dtype=np.float32).copy()
        if value.size != int(dim):
            raise RuntimeError("corrupt generic BGE cache")
        return base.normalized_vector(value)

    def close(self) -> None:
        self.connection.close()


class TaskVectorCache:
    def __init__(self, path: Path, checkpoint_sha256: str) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS embeddings(context TEXT PRIMARY KEY,dim INTEGER NOT NULL,vector BLOB NOT NULL)")
        previous = self.connection.execute("SELECT value FROM metadata WHERE key='checkpoint_sha256'").fetchone()
        if previous is not None and previous[0] != checkpoint_sha256:
            raise RuntimeError("task vector cache belongs to a different checkpoint")
        self.connection.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('checkpoint_sha256',?)", (checkpoint_sha256,))
        self.connection.commit()

    def get(self, context: str) -> np.ndarray | None:
        row = self.connection.execute("SELECT dim,vector FROM embeddings WHERE context=?", (context,)).fetchone()
        if row is None:
            return None
        dim, blob = row
        value = np.frombuffer(blob, dtype=np.float32).copy()
        if value.size != int(dim):
            raise RuntimeError("corrupt task vector cache")
        return base.normalized_vector(value)

    def put_many(self, contexts: Sequence[str], values: np.ndarray) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO embeddings(context,dim,vector) VALUES(?,?,?)",
            [(context, int(value.size), np.asarray(value, dtype=np.float32).tobytes()) for context, value in zip(contexts, values)],
        )
        self.connection.commit()

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def index_rows(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Mapping[str, Any]]:
    values = {str(row["row_id"]): row for row in rows}
    if len(values) != len(rows):
        raise ValueError(f"duplicate row ID in {name}")
    return values


def candidate_text(item: Mapping[str, Any]) -> str:
    return str(item["candidate"])


def intrinsic_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    retrieval = group_retrieval_metrics(rows)
    by_author: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_author[str(row["author"])].append(row)
    retrieval["overall"]["target_support_top1"] = statistics.fmean(float(row["target_support_rank"] == 1) for row in rows)
    retrieval["overall"]["mean_gold_support_margin"] = statistics.fmean(float(row["gold_support_margin"]) for row in rows)
    retrieval["overall"]["macro_author_target_support_top1"] = statistics.fmean(
        statistics.fmean(float(row["target_support_rank"] == 1) for row in values)
        for values in by_author.values()
    )
    for author, values in by_author.items():
        retrieval["per_author"][author]["target_support_top1"] = statistics.fmean(
            float(row["target_support_rank"] == 1) for row in values
        )
    return retrieval


def intrinsic_row(
    *,
    author: str,
    gold: str,
    candidates: Sequence[str],
    visible: Sequence[Any],
    query_vector: np.ndarray,
    vectors: Mapping[str, np.ndarray],
    support: Mapping[str, float],
) -> dict[str, Any] | None:
    histories = [item for item in visible if item.record.target in set(candidates)]
    if not any(item.record.target == gold for item in histories) or not any(item.record.target != gold for item in histories):
        return None
    scored = []
    for item in histories:
        vector = vectors[item.record.context[-base.BGE_CONTEXT_CHARS :]]
        scored.append((float(vector @ query_vector), int(item.record.position), str(item.record.row_id), item.record.target))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    rank = next(index for index, item in enumerate(scored, start=1) if item[3] == gold)
    support_order = sorted(candidates, key=lambda text: (-float(support[text]), candidates.index(text), text))
    target_rank = support_order.index(gold) + 1
    best_wrong = max(float(support[text]) for text in candidates if text != gold)
    return {
        "author": author,
        "rank": rank,
        "target_support_rank": target_rank,
        "gold_support_margin": float(support[gold]) - best_wrong,
    }


def metric_bundle(rows: Sequence[Mapping[str, Any]], rank_key: str) -> dict[str, Any]:
    return {
        "overall": ranking_metrics(rows, rank_key),
        "ambiguous": ranking_metrics([row for row in rows if row["ambiguous"]], rank_key),
        "conflict": ranking_metrics([row for row in rows if row["conflict"]], rank_key),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--frozen-predictions", type=Path, required=True)
    parser.add_argument("--generic-bge-cache", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--lambdamart-result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()
    for path in (args.fit, args.val, args.stage2, args.frozen_predictions, args.generic_bge_cache, args.training_result, args.lambdamart_result, args.checkpoint):
        refuse_closed_path(path)
    expected_hashes = {
        args.fit: EXPECTED_FIT_SHA256,
        args.val: EXPECTED_VAL_SHA256,
        args.stage2: EXPECTED_STAGE2_SHA256,
        args.frozen_predictions: EXPECTED_PREDICTIONS_SHA256,
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen input hash changed: {path}")
    training = json.loads(args.training_result.read_text(encoding="utf-8"))
    if training.get("status") != "complete" or training.get("used_dev3000") or training.get("used_test"):
        raise ValueError("training result is incomplete or used closed data")
    checkpoint_sha, checkpoint_files = sha256_tree(args.checkpoint)
    if checkpoint_sha != training["final_checkpoint_sha256"]:
        raise ValueError("checkpoint hash differs from training freeze")

    started = time.perf_counter()
    fit_rows = read_jsonl(args.fit)
    val_rows = read_jsonl(args.val)
    if len(fit_rows) != 144_526 or len(val_rows) != 34_416:
        raise ValueError("Clean3 population changed")
    if any(row.get("used_dev3000") or row.get("used_test") or row.get("pilot_partition") == "test" for row in [*fit_rows, *val_rows]):
        raise ValueError("closed-data row found")
    stage2_rows = read_jsonl(args.stage2)
    frozen_rows = read_jsonl(args.frozen_predictions)
    stage2 = index_rows(stage2_rows, "Stage2")
    frozen = index_rows(frozen_rows, "frozen predictions")
    if [str(row["row_id"]) for row in val_rows] != [str(row["row_id"]) for row in stage2_rows] or set(stage2) != set(frozen):
        raise ValueError("Train-Val row order or population differs")
    history = base.CausalHistoryIndex([*fit_rows, *val_rows])

    required_contexts: set[str] = set()
    per_row_visible: dict[str, Sequence[Any]] = {}
    for number, val_row in enumerate(val_rows, start=1):
        row_id = str(val_row["row_id"])
        candidates = {candidate_text(item) for item in stage2[row_id]["retuned_stage1_candidates"]}
        if not candidates:
            per_row_visible[row_id] = ()
            continue
        required_contexts.add(base.context_of(val_row)[-base.BGE_CONTEXT_CHARS :])
        visible = history.visible_same_pinyin(
            author=str(val_row["author"]),
            position=int(val_row["chronological_position"]),
            pinyin=base.pinyin_of(val_row),
        )
        visible = tuple(item for item in visible if item.record.target in candidates)
        per_row_visible[row_id] = visible
        required_contexts.update(item.record.context[-base.BGE_CONTEXT_CHARS :] for item in visible)
        if args.progress_every and number % (args.progress_every * 5) == 0:
            print(f"context audit {number:,}/{len(val_rows):,} unique={len(required_contexts):,}", flush=True)

    task_cache = TaskVectorCache(args.output_root / "task_vectors.sqlite3", checkpoint_sha)
    missing = [context for context in sorted(required_contexts) if task_cache.get(context) is None]
    print(f"task contexts required={len(required_contexts):,} cache={task_cache.count():,} missing={len(missing):,}", flush=True)
    if missing:
        encoder = SharedContextEncoder(args.checkpoint, device="cuda")
        for start in range(0, len(missing), 128):
            batch = missing[start : start + 128]
            task_cache.put_many(batch, encoder.embed(batch, batch_size=128))
            if args.progress_every and (start // 128 + 1) % 20 == 0:
                print(f"task embed {min(start+128, len(missing)):,}/{len(missing):,}", flush=True)
    task_vectors = {context: task_cache.get(context) for context in required_contexts}
    if any(value is None for value in task_vectors.values()):
        raise RuntimeError("task vector cache incomplete")

    generic_cache = ReadOnlyVectors(args.generic_bge_cache)
    generic_vectors = {context: generic_cache.get(context) for context in required_contexts}
    missing_generic = [context for context, value in generic_vectors.items() if value is None]
    if missing_generic:
        raise RuntimeError(f"generic BGE cache lacks {len(missing_generic)} identical-population contexts")

    prediction_rows = []
    generic_intrinsic = []
    task_intrinsic = []
    support_max_abs = 0.0
    for number, val_row in enumerate(val_rows, start=1):
        row_id = str(val_row["row_id"])
        srow = stage2[row_id]
        frow = frozen[row_id]
        stage1 = srow["retuned_stage1_candidates"]
        candidates = [candidate_text(item) for item in stage1]
        visible = per_row_visible[row_id]
        gold = str(frow["gold"])
        if candidates:
            query_context = base.context_of(val_row)[-base.BGE_CONTEXT_CHARS :]
            generic_support, _ = base.bge_recency_support(
                query_vector=generic_vectors[query_context],
                candidates=candidates,
                visible=visible,
                vectors=generic_vectors,
            )
            task_support, _ = base.bge_recency_support(
                query_vector=task_vectors[query_context],
                candidates=candidates,
                visible=visible,
                vectors=task_vectors,
            )
            support_max_abs = max(
                support_max_abs,
                max(abs(float(generic_support[text]) - float(srow["retuned_bge_support"][text])) for text in candidates),
            )
            generic_final = retune.final_rerank(
                base,
                stage1=stage1,
                ngram_support=srow["retuned_ngram_support"],
                bge_support=generic_support,
                lambda_n=6.0,
                lambda_b=6.0,
            )
            task_final = retune.final_rerank(
                base,
                stage1=stage1,
                ngram_support=srow["retuned_ngram_support"],
                bge_support=task_support,
                lambda_n=6.0,
                lambda_b=6.0,
            )
            generic_rank = base.rank_of(generic_final, gold)
            task_rank = base.rank_of(task_final, gold)
            generic_top10 = [candidate_text(item) for item in generic_final]
            task_top10 = [candidate_text(item) for item in task_final]
            generic_item = intrinsic_row(
                author=str(frow["author"]), gold=gold, candidates=candidates, visible=visible,
                query_vector=generic_vectors[query_context], vectors=generic_vectors, support=generic_support,
            )
            task_item = intrinsic_row(
                author=str(frow["author"]), gold=gold, candidates=candidates, visible=visible,
                query_vector=task_vectors[query_context], vectors=task_vectors, support=task_support,
            )
            if generic_item is not None:
                generic_intrinsic.append(generic_item)
                task_intrinsic.append(task_item)
        else:
            generic_rank = task_rank = None
            generic_top10 = task_top10 = []
            task_support = {}
        if generic_rank != frow.get("RetunedFinal_rank") or generic_top10 != frow.get("RetunedFinal_top10"):
            raise RuntimeError(f"generic-BGE frozen reconstruction failed at {row_id}")
        prediction_rows.append(
            {
                "schema_version": 1,
                "row_id": row_id,
                "author": str(frow["author"]),
                "gold": gold,
                "ambiguous": bool(frow["ambiguous"]),
                "conflict": bool(frow["conflict"]),
                "FrozenGenericBGE_rank": generic_rank,
                "TaskBiEncoderFixed_rank": task_rank,
                "TaskBiEncoderFixed_top10": task_top10,
                "task_bge_support": task_support,
                "gold_used_for_scoring": False,
                "used_dev3000": False,
                "used_test": False,
            }
        )
        if args.progress_every and (number % args.progress_every == 0 or number == len(val_rows)):
            print(f"evaluate {number:,}/{len(val_rows):,}", flush=True)

    task_cache_rows = task_cache.count()
    task_cache.close()
    generic_cache.close()
    if support_max_abs > 1e-6:
        raise RuntimeError(f"generic BGE support reconstruction differs: {support_max_abs}")
    if len(generic_intrinsic) != len(task_intrinsic):
        raise RuntimeError("intrinsic populations differ")

    fixed_metrics = {
        "frozen_generic_bge": metric_bundle(prediction_rows, "FrozenGenericBGE_rank"),
        "task_biencoder_fixed_fusion": metric_bundle(prediction_rows, "TaskBiEncoderFixed_rank"),
    }
    intrinsic = {
        "population": "candidate-conditioned legal histories; query has both Gold-target and wrong-target history",
        "generic_bge": intrinsic_summary(generic_intrinsic),
        "task_biencoder": intrinsic_summary(task_intrinsic),
    }
    transitions = transition_counts(prediction_rows, "FrozenGenericBGE_rank", "TaskBiEncoderFixed_rank")
    lambda_result = json.loads(args.lambdamart_result.read_text(encoding="utf-8"))
    if lambda_result.get("used_dev3000") or lambda_result.get("used_test"):
        raise ValueError("LambdaMART comparator used closed data")
    gate = {
        "intrinsic_macro_recall1_improved": (
            intrinsic["task_biencoder"]["overall"]["macro_author_recall_at_1"]
            > intrinsic["generic_bge"]["overall"]["macro_author_recall_at_1"]
        ),
        "fixed_fusion_macro_top1_improved": (
            fixed_metrics["task_biencoder_fixed_fusion"]["overall"]["macro_author_top1"]
            > fixed_metrics["frozen_generic_bge"]["overall"]["macro_author_top1"]
        ),
    }
    gate["task_specific_lambdamart_refit_authorized"] = all(gate.values())
    args.output_root.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_root / "predictions.jsonl"
    write_jsonl(prediction_path, prediction_rows)
    result = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "task_specific_biencoder_v1",
        "rows": len(prediction_rows),
        "intrinsic": intrinsic,
        "fixed_fusion_metrics": fixed_metrics,
        "transition_from_frozen": transitions,
        "completed_lambdamart_comparator": lambda_result["metrics"],
        "nonlinear_refit_gate": gate,
        "generic_support_reconstruction_max_abs_difference": support_max_abs,
        "task_vector_cache_rows": task_cache_rows,
        "training_checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": checkpoint_sha, "files": checkpoint_files},
        "runtime_seconds": time.perf_counter() - started,
        "used_dev3000": False,
        "used_test": False,
    }
    result_path = args.output_root / "result.json"
    write_json(result_path, result)
    checksums = {
        "runner": sha256_file(Path(__file__)),
        "predictions.jsonl": sha256_file(prediction_path),
        "result.json": sha256_file(result_path),
        "checkpoint": checkpoint_sha,
        "used_dev3000": False,
        "used_test": False,
    }
    write_json(args.output_root / "artifact_checksums.json", checksums)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
