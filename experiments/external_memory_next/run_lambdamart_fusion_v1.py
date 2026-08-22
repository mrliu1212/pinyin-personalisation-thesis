"""Train and select the predeclared author-free LambdaMART fusion model."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EXPECTED_LIGHTGBM = "4.7.0"
EXPECTED_SMOOTHING_SHA256 = "41863119c16590bc67a8d39892cd8e45dceb783044814288c313f30313d8c2c2"
SEED = 1729


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                if bool(row.get("used_test", False)):
                    raise RuntimeError(f"Test dependency in {path}")
                yield row


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def metric_summary(meta: Sequence[Mapping[str, Any]], ranks: Sequence[int | None],
                   subset: str = "overall") -> dict[str, Any]:
    selected = [(row, rank) for row, rank in zip(meta, ranks)
                if subset == "overall" or bool(row[subset])]
    by_author: dict[str, list[int | None]] = {}
    values = []
    for row, rank in selected:
        values.append(rank)
        by_author.setdefault(str(row["author"]), []).append(rank)

    def top(items: Sequence[int | None], k: int) -> float:
        return sum(rank is not None and rank <= k for rank in items) / len(items)

    per_author = {author: top(items, 1) for author, items in sorted(by_author.items())}
    return {"n": len(values), "macro_author_top1": statistics.fmean(per_author.values()),
            "micro_top1": top(values, 1), "top3": top(values, 3), "top5": top(values, 5),
            "mrr_at_10": sum(0 if rank is None else 1 / rank for rank in values) / len(values),
            "missing10": sum(rank is None for rank in values) / len(values),
            "per_author_top1": per_author}


def transition_summary(before: Sequence[int | None], after: Sequence[int | None]) -> dict[str, int]:
    result = {"n": len(before), "rescue": 0, "harm": 0,
              "unchanged_correct": 0, "unchanged_wrong": 0}
    for old, new in zip(before, after):
        old_correct, new_correct = old == 1, new == 1
        key = "rescue" if new_correct and not old_correct else "harm" if old_correct and not new_correct else "unchanged_correct" if old_correct else "unchanged_wrong"
        result[key] += 1
    result["net"] = result["rescue"] - result["harm"]
    return result


def ranks_from_scores(scores: Sequence[float], meta: Sequence[Mapping[str, Any]]) -> tuple[list[int | None], list[list[str]]]:
    ranks: list[int | None] = []
    top10s: list[list[str]] = []
    for row in meta:
        offset, count = int(row["offset"]), int(row["candidate_count"])
        candidates = list(map(str, row["candidates"]))
        if count != len(candidates):
            raise RuntimeError(f"Candidate count mismatch: {row['row_id']}")
        baseline_order = {candidate: rank for rank, candidate in enumerate(map(str, row["baseline_top10"]), start=1)}
        if set(baseline_order) != set(candidates):
            raise RuntimeError(f"Candidate surface changed: {row['row_id']}")
        order = sorted(range(count), key=lambda index: (
            -float(scores[offset + index]), baseline_order[candidates[index]], candidates[index]))
        top10 = [candidates[index] for index in order]
        gold = str(row["gold"])
        rank = next((index for index, candidate in enumerate(top10, start=1) if candidate == gold), None)
        ranks.append(rank)
        top10s.append(top10)
    return ranks, top10s


def selection_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = record["metrics"]["overall"]
    config = record["config"]
    return (-float(metrics["macro_author_top1"]), -float(metrics["micro_top1"]),
            -float(metrics["mrr_at_10"]), int(config["max_depth"]),
            int(config["rounds"]), -int(config["min_data_in_leaf"]), str(record["config_id"]))


def common_params() -> dict[str, Any]:
    return {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1, 3, 5, 10],
            "label_gain": [0, 1], "lambdarank_truncation_level": 10,
            "learning_rate": 0.05, "feature_pre_filter": False,
            "deterministic": True, "force_col_wise": True, "num_threads": 8,
            "seed": SEED, "feature_fraction_seed": SEED,
            "bagging_seed": SEED, "data_random_seed": SEED, "verbosity": -1}


def configurations() -> list[dict[str, Any]]:
    values = [{"config_id": "additive_stumps", "kind": "additive_control",
               "max_depth": 1, "num_leaves": 2, "min_data_in_leaf": 500, "rounds": 100}]
    for depth in (2, 3, 5):
        for minimum in (100, 500):
            for rounds in (50, 100):
                values.append({"config_id": f"d{depth}_m{minimum}_r{rounds}", "kind": "nonlinear",
                               "max_depth": depth, "num_leaves": 2 ** depth - 1,
                               "min_data_in_leaf": minimum, "rounds": rounds})
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--deps-root", type=Path, required=True)
    parser.add_argument("--smoothing-predictions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.matrix_root / "matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or audit.get("status") != "complete":
        raise RuntimeError("Matrix/audit gate is incomplete")
    if any(value.get("used_dev3000") is not False or value.get("used_test") is not False
           for value in (manifest, audit)):
        raise RuntimeError("Closed-data gate changed")
    if not audit.get("frozen_val_baseline_exact"):
        raise RuntimeError("Frozen Val reconstruction gate did not pass")
    if manifest.get("author_identity_feature") is not False or manifest.get("gold_labels_are_separate_from_runtime_features") is not True:
        raise RuntimeError("Matrix feature/label boundary changed")
    for name, value in manifest["artifacts"].items():
        if sha256_file(Path(value["path"])) != value["sha256"]:
            raise RuntimeError(f"Matrix artifact changed: {name}")
    if sha256_file(args.smoothing_predictions) != EXPECTED_SMOOTHING_SHA256:
        raise RuntimeError("Smoothing comparison artifact changed")
    sys.path.insert(0, str(args.deps_root.resolve()))
    import lightgbm as lgb
    if lgb.__version__ != EXPECTED_LIGHTGBM:
        raise RuntimeError(f"LightGBM version changed: {lgb.__version__}")

    feature_names = list(manifest["feature_names"])
    if "author" in feature_names or "gold" in feature_names:
        raise RuntimeError("Forbidden feature in matrix")
    fit_x = np.load(args.matrix_root / "fit_X.npy", mmap_mode="r")
    fit_y = np.load(args.matrix_root / "fit_y.npy", mmap_mode="r")
    fit_group = np.load(args.matrix_root / "fit_group.npy", mmap_mode="r")
    val_x = np.load(args.matrix_root / "val_X.npy", mmap_mode="r")
    meta = list(iter_jsonl(args.matrix_root / "val_query_meta.jsonl"))
    smoothing = list(iter_jsonl(args.smoothing_predictions))
    if len(meta) != len(smoothing) or any(str(a["row_id"]) != str(b["row_id"]) for a, b in zip(meta, smoothing)):
        raise RuntimeError("Smoothing/Val order differs")
    if int(fit_group.sum()) != len(fit_x) or int(fit_y.sum()) != len(fit_group):
        raise RuntimeError("Fit query-group labels changed")
    baseline_ranks = [None if row.get("baseline_rank") is None else int(row["baseline_rank"]) for row in meta]
    smoothing_ranks = [None if row.get("rank") is None else int(row["rank"]) for row in smoothing]
    train = lgb.Dataset(fit_x, label=fit_y, group=fit_group,
                        feature_name=feature_names, free_raw_data=False)
    args.output_root.mkdir(parents=True, exist_ok=True)
    models_root = args.output_root / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    matrix_manifest_sha256 = sha256_file(manifest_path)
    records = []
    for config in configurations():
        model_path = models_root / f"{config['config_id']}.txt"
        record_path = models_root / f"{config['config_id']}.json"
        if model_path.is_file() and record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if (record.get("config") != config
                    or record.get("model_sha256") != sha256_file(model_path)
                    or record.get("matrix_manifest_sha256") != matrix_manifest_sha256):
                raise RuntimeError(f"Existing model record conflicts: {config['config_id']}")
            print(f"Reusing {config['config_id']}", flush=True)
            records.append(record)
            continue
        params = {**common_params(), "max_depth": config["max_depth"],
                  "num_leaves": config["num_leaves"],
                  "min_data_in_leaf": config["min_data_in_leaf"]}
        started = time.perf_counter()
        booster = lgb.train(params, train, num_boost_round=config["rounds"])
        elapsed = time.perf_counter() - started
        booster.save_model(str(model_path))
        scores = booster.predict(val_x, num_iteration=booster.current_iteration())
        ranks, _ = ranks_from_scores(scores, meta)
        record = {"config_id": config["config_id"], "kind": config["kind"], "config": config,
                  "params": params, "elapsed_seconds": elapsed,
                  "metrics": {name: metric_summary(meta, ranks, name)
                              for name in ("overall", "ambiguous", "conflict")},
                  "transition_from_frozen": transition_summary(baseline_ranks, ranks),
                  "model_sha256": sha256_file(model_path),
                  "matrix_manifest_sha256": matrix_manifest_sha256,
                  "used_dev3000": False, "used_test": False}
        write_json(record_path, record)
        records.append(record)
        print(f"{config['config_id']} Macro={record['metrics']['overall']['macro_author_top1']:.9f} "
              f"elapsed={elapsed:.1f}s", flush=True)
    nonlinear = sorted((record for record in records if record["kind"] == "nonlinear"), key=selection_key)
    selected_record = nonlinear[0]
    selected_model_path = models_root / f"{selected_record['config_id']}.txt"
    booster = lgb.Booster(model_file=str(selected_model_path))
    scores = booster.predict(val_x, num_iteration=booster.current_iteration())
    selected_ranks, selected_top10 = ranks_from_scores(scores, meta)
    stump_record = next(record for record in records if record["kind"] == "additive_control")
    stump_booster = lgb.Booster(model_file=str(models_root / "additive_stumps.txt"))
    stump_scores = stump_booster.predict(val_x, num_iteration=stump_booster.current_iteration())
    stump_ranks, _ = ranks_from_scores(stump_scores, meta)

    shap_abs = np.zeros(len(feature_names), dtype=np.float64)
    shap_signed = np.zeros(len(feature_names), dtype=np.float64)
    shap_n = 0
    for start in range(0, len(val_x), 50_000):
        contributions = np.asarray(booster.predict(val_x[start:start + 50_000], pred_contrib=True))[:, :-1]
        shap_abs += np.abs(contributions).sum(axis=0)
        shap_signed += contributions.sum(axis=0)
        shap_n += len(contributions)
    importance = []
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    for index, name in enumerate(feature_names):
        importance.append({"feature": name, "gain": float(gain[index]), "split": int(split[index]),
                           "mean_abs_shap": float(shap_abs[index] / shap_n),
                           "mean_signed_shap": float(shap_signed[index] / shap_n)})
    importance.sort(key=lambda row: (-row["mean_abs_shap"], -row["gain"], row["feature"]))

    predictions = []
    for index, row in enumerate(meta):
        predictions.append({"row_id": row["row_id"], "author": row["author"], "gold": row["gold"],
                            "ambiguous": row["ambiguous"], "conflict": row["conflict"],
                            "Frozen_rank": baseline_ranks[index], "Smoothing_rank": smoothing_ranks[index],
                            "AdditiveStumps_rank": stump_ranks[index], "LambdaMART_rank": selected_ranks[index],
                            "LambdaMART_top10": selected_top10[index],
                            "used_dev3000": False, "used_test": False})
    prediction_path = args.output_root / "selected_predictions.jsonl"
    write_jsonl(prediction_path, predictions)
    selected_metrics = {name: metric_summary(meta, selected_ranks, name)
                        for name in ("overall", "ambiguous", "conflict")}
    result = {"schema_version": 1, "status": "complete",
              "experiment": "lambdamart_external_memory_fusion_v1",
              "library": {"name": "lightgbm", "version": lgb.__version__},
              "selection_rule": "Macro-author Top1, Micro Top1, MRR@10, lower depth, fewer rounds, larger min leaf, config ID",
              "selected": selected_record, "metrics": selected_metrics,
              "controls": {"frozen": {name: metric_summary(meta, baseline_ranks, name) for name in ("overall", "ambiguous", "conflict")},
                           "smoothing_alpha128": {name: metric_summary(meta, smoothing_ranks, name) for name in ("overall", "ambiguous", "conflict")},
                           "additive_stumps": stump_record["metrics"]},
              "transitions": {"from_frozen": transition_summary(baseline_ranks, selected_ranks),
                              "from_smoothing": transition_summary(smoothing_ranks, selected_ranks),
                              "from_additive_stumps": transition_summary(stump_ranks, selected_ranks)},
              "grid": records, "feature_importance": importance,
              "fit_zero_positive_groups_excluded": manifest["fit"]["zero_positive_groups_excluded"],
              "fixed_candidate_surface": True,
              "provenance": {"matrix_manifest": {"path": str(manifest_path.resolve()), "sha256": matrix_manifest_sha256},
                             "audit": {"path": str(args.audit.resolve()), "sha256": sha256_file(args.audit)},
                             "smoothing_predictions": EXPECTED_SMOOTHING_SHA256},
              "used_dev3000": False, "used_test": False}
    result_path = args.output_root / "result.json"
    write_json(result_path, result)
    write_json(args.output_root / "artifact_checksums.json", {
        "runner": sha256_file(Path(__file__)), "result.json": sha256_file(result_path),
        "selected_model.txt": sha256_file(selected_model_path),
        "selected_predictions.jsonl": sha256_file(prediction_path),
        "used_dev3000": False, "used_test": False})
    print(json.dumps({"status": "complete", "selected": selected_record["config"],
                      "metrics": selected_metrics["overall"], "transitions": result["transitions"],
                      "output": str(result_path)}, indent=2))


if __name__ == "__main__":
    main()
