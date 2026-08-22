"""Reconstruct the frozen Full RetunedFinal Train-Val ranking arithmetically.

The runner consumes only hash-pinned Train-Val feature/support artifacts. It
does not accept Dev3000 or Test inputs and performs no model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_ROWS = 34416
EXPECTED = {
    "stage1": "e311261cb0c5ea252ce593fdaa43ff87826d19eef440f39e2efc67ddc4310b13",
    "stage2": "d413d02650a759c0a759e1845212e68b2d948c1a1d20fc88abfd89ea7973bc64",
    "predictions": "f3e902e5a9e7d25e62799b9abb719026c336381eacc42999d1e7edccf2731b22",
    "config": "3dc3fb908aeeaa853526ad71cf85de7400f47d261ed7c09acdd8197446f5fa3d",
}
EXPECTED_WEIGHTS = {"w_p": 2.0, "w_cs": 6.0, "w_e": 4.0,
                    "lambda_n": 6.0, "lambda_b": 6.0}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("source_split", "")).lower() == "test" or bool(row.get("used_test", False)):
                raise RuntimeError(f"Test row in {path}:{number}")
            yield row


def candidate_text(row: Mapping[str, Any]) -> str:
    value = row.get("candidate", row.get("text", row.get("target")))
    if value is None:
        raise RuntimeError(f"Candidate text missing: {row}")
    return str(value)


def ngram_rank_map(personal_k5: Sequence[str], p_ng: Mapping[str, float]) -> dict[str, int]:
    order = sorted(range(len(personal_k5)),
                   key=lambda index: (-float(p_ng[personal_k5[index]]), index, str(personal_k5[index])))
    return {personal_k5[index]: rank for rank, index in enumerate(order, start=1)}


def merge_stage1(feature: Mapping[str, Any], weights: Mapping[str, float]) -> list[dict[str, Any]]:
    generic = [dict(row) for row in feature["generic_frequency_candidates"]]
    if not generic:
        return []
    personal = list(map(str, feature["personal_k5"]))
    p_ng = {str(key): float(value) for key, value in feature["p_ng"].items()}
    choice = {str(key): float(value) for key, value in feature["choice_share"].items()}
    generic_texts = {candidate_text(row) for row in generic}
    if generic_texts.intersection(personal):
        raise RuntimeError("Personal K5 overlaps Generic surface")
    boundary = min(float(row["normalized_generic_score"]) for row in generic)
    tiebreak = ngram_rank_map(personal, p_ng)
    rows = generic
    for original_rank, candidate in enumerate(personal, start=1):
        score = (boundary + weights["w_p"] * p_ng[candidate]
                 + weights["w_cs"] * choice[candidate]
                 + weights["w_e"] * float(feature["entropy_concentration"]))
        rows.append({"candidate": candidate, "source": "personal_recovery",
                     "generic_rank": None, "personal_candidate_rank": tiebreak[candidate],
                     "original_personal_frequency_rank": original_rank,
                     "ngram_rank": tiebreak[candidate], "final_score": score})
    rows.sort(key=lambda row: (-float(row["final_score"]),
                               0 if row["source"] == "generic_frequency" else 1,
                               int(row.get("generic_rank") or row.get("personal_candidate_rank") or row.get("rank") or 0),
                               candidate_text(row)))
    for rank, row in enumerate(rows[:10], start=1):
        row["rank"] = rank
        row["base_rank"] = rank
        row["base_score"] = float(row["final_score"])
    return rows[:10]


def final_rerank(stage1: Sequence[Mapping[str, Any]], ngram: Mapping[str, Any],
                 bge: Mapping[str, Any], weights: Mapping[str, float]) -> list[dict[str, Any]]:
    names = [candidate_text(row) for row in stage1]
    if set(names) != set(ngram) or set(names) != set(bge):
        raise RuntimeError("Stage-2 support candidate set differs from reconstructed Stage-1 Top10")
    rows = []
    for index, item in enumerate(stage1, start=1):
        row = dict(item)
        text = candidate_text(row)
        row["final_score"] = (float(row["final_score"])
                              + weights["lambda_n"] * float(ngram[text])
                              + weights["lambda_b"] * float(bge[text]))
        row["base_rank"] = int(item.get("base_rank", item.get("rank", index)))
        rows.append(row)
    rows.sort(key=lambda row: (-float(row["final_score"]), int(row["base_rank"]), candidate_text(row)))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_author: dict[str, list[int | None]] = defaultdict(list)
    ranks: list[int | None] = []
    for row in rows:
        rank = None if row.get("rank") is None else int(row["rank"])
        ranks.append(rank)
        by_author[str(row["author"])].append(rank)

    def top(values: Sequence[int | None], k: int) -> float:
        return sum(value is not None and value <= k for value in values) / len(values)

    per_author = {author: top(values, 1) for author, values in sorted(by_author.items())}
    return {"n": len(rows), "macro_author_top1": sum(per_author.values()) / len(per_author),
            "micro_top1": top(ranks, 1), "top3": top(ranks, 3), "top5": top(ranks, 5),
            "mrr_at_10": sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / len(ranks),
            "missing10": sum(rank is None for rank in ranks) / len(ranks),
            "per_author_top1": per_author}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in EXPECTED:
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in EXPECTED}
    provenance = {}
    for name, path in paths.items():
        actual = sha256_file(path)
        if actual != EXPECTED[name]:
            raise RuntimeError(f"{name} SHA mismatch: {actual}")
        provenance[name] = {"path": str(path.resolve()), "sha256": actual, "bytes": path.stat().st_size}

    config = json.loads(args.config.read_text(encoding="utf-8"))
    weights = {**{key: float(config["selected_stage1"][key]) for key in ("w_p", "w_cs", "w_e")},
               **{key: float(config["selected_stage2"][key]) for key in ("lambda_n", "lambda_b")}}
    if weights != EXPECTED_WEIGHTS:
        raise RuntimeError(f"Frozen weights changed: {weights}")
    if config.get("used_dev3000_for_selection") is not False or config.get("used_test") is not False:
        raise RuntimeError("Frozen selection boundary changed")

    reconstructed: list[dict[str, Any]] = []
    iterators = (iter_jsonl(args.stage1), iter_jsonl(args.stage2), iter_jsonl(args.predictions))
    for number, group in enumerate(itertools.zip_longest(*iterators), start=1):
        if any(row is None for row in group):
            raise RuntimeError("Input row counts differ")
        feature, support, prediction = group
        row_ids = {str(row["row_id"]) for row in group}
        if len(row_ids) != 1:
            raise RuntimeError(f"Input row order differs at {number}: {row_ids}")
        stage1 = merge_stage1(feature, weights)
        expected_stage1 = support["retuned_stage1_candidates"]
        if [candidate_text(row) for row in stage1] != [candidate_text(row) for row in expected_stage1]:
            raise RuntimeError(f"Stage-1 reconstruction mismatch: {feature['row_id']}")
        for actual, expected in zip(stage1, expected_stage1):
            if not math.isclose(float(actual["final_score"]), float(expected["final_score"]), rel_tol=0, abs_tol=1e-12):
                raise RuntimeError(f"Stage-1 score mismatch: {feature['row_id']}")
        final = final_rerank(stage1, support["retuned_ngram_support"], support["retuned_bge_support"], weights)
        top10 = [candidate_text(row) for row in final]
        if top10 != list(map(str, prediction["RetunedFinal_top10"])):
            raise RuntimeError(f"Final Top10 mismatch: {feature['row_id']}")
        gold = str(prediction["gold"])
        rank = next((index for index, candidate in enumerate(top10, start=1) if candidate == gold), None)
        if rank != prediction.get("RetunedFinal_rank"):
            raise RuntimeError(f"Final rank mismatch: {feature['row_id']}")
        reconstructed.append({"row_id": feature["row_id"], "author": feature["author"],
                              "ambiguous": bool(feature["ambiguous"]), "conflict": bool(feature["conflict"]),
                              "rank": rank})
    if len(reconstructed) != EXPECTED_ROWS:
        raise RuntimeError(f"Row count changed: {len(reconstructed)}")

    metrics = {name: metric_summary([row for row in reconstructed if name == "overall" or row[name]])
               for name in ("overall", "ambiguous", "conflict")}
    expected = config["selected_train_val_metrics"]
    for key in ("macro_author_top1", "micro_top1", "top3", "top5", "mrr_at_10", "missing10"):
        if not math.isclose(float(metrics["overall"][key]), float(expected[key]), rel_tol=0, abs_tol=1e-15):
            raise RuntimeError(f"Metric mismatch: {key}")

    result = {"schema_version": 1, "status": "exact_reproduction",
              "experiment": "full_retuned_baseline_reproduction_v1", "weights": weights,
              "metrics": metrics, "reconstructed_rows": len(reconstructed),
              "candidate_orders_exact": True, "ranks_exact": True,
              "provenance": provenance, "used_dev3000": False, "used_test": False}
    output = args.output_root / "baseline_reproduction.json"
    write_json(output, result)
    write_json(args.output_root / "artifact_checksums.json",
               {"runner": sha256_file(Path(__file__)), "baseline_reproduction.json": sha256_file(output),
                "used_dev3000": False, "used_test": False})
    print(json.dumps({"status": result["status"], "rows": len(reconstructed),
                      "metrics": metrics["overall"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
