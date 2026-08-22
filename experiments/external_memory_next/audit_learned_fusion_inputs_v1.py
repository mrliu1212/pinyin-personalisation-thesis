"""Audit causal Train-Fit/Train-Val candidate tables for learned fusion."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FEATURE_NAMES = (
    "base_score", "base_rank", "source_personal", "has_generic",
    "generic_score", "normalized_generic_score", "generic_rank",
    "frequency_count", "personal_score", "has_personal",
    "personal_candidate_rank", "p_ng", "choice_share",
    "entropy_concentration", "log1p_same_pinyin_history",
    "log1p_raw_history", "ngram_support", "bge_support",
    "log1p_bge_history_count", "ngram_effective_n",
    "log1p_ngram_matched_history", "base_gap_to_top",
    "ngram_gap_to_top", "bge_gap_to_top", "frozen_linear_score",
)
EXPECTED_VAL_STAGE1_SHA256 = "e311261cb0c5ea252ce593fdaa43ff87826d19eef440f39e2efc67ddc4310b13"
EXPECTED_VAL_STAGE2_SHA256 = "d413d02650a759c0a759e1845212e68b2d948c1a1d20fc88abfd89ea7973bc64"
EXPECTED_VAL_PREDICTIONS_SHA256 = "f3e902e5a9e7d25e62799b9abb719026c336381eacc42999d1e7edccf2731b22"
EXPECTED_FIT_ROWS = 144526
EXPECTED_VAL_ROWS = 34416


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
    return str(row.get("candidate", row.get("text", row.get("target"))))


def extract_group(feature: Mapping[str, Any], support: Mapping[str, Any]) -> tuple[list[list[float]], list[int], list[str]]:
    candidates = list(support["retuned_stage1_candidates"])
    names = [candidate_text(row) for row in candidates]
    ngram = {str(key): float(value) for key, value in support["retuned_ngram_support"].items()}
    bge = {str(key): float(value) for key, value in support["retuned_bge_support"].items()}
    bge_counts = {str(key): int(value) for key, value in support["bge_history_counts"].items()}
    if set(names) != set(ngram) or set(names) != set(bge) or set(names) != set(bge_counts):
        raise RuntimeError(f"Support candidate mismatch: {feature['row_id']}")
    base_top = max((float(row["final_score"]) for row in candidates), default=0.0)
    ngram_top = max(ngram.values(), default=0.0)
    bge_top = max(bge.values(), default=0.0)
    entropy = float(feature["entropy_concentration"])
    same_n = int(feature["same_pinyin_history_count"])
    raw_n = int(feature["raw_history_count"])
    effective_n = int(support["ngram_effective_n"])
    matched = int(support["ngram_matched_history_rows"])
    vectors: list[list[float]] = []
    labels = []
    gold = str(feature["gold"])
    for row, name in zip(candidates, names):
        personal = row["source"] == "personal_recovery"
        has_generic = row.get("generic_rank") is not None
        has_personal = personal
        base_score = float(row["final_score"])
        n_score, b_score = ngram[name], bge[name]
        values = [
            base_score, float(row.get("base_rank", row.get("rank", 0))), float(personal), float(has_generic),
            float(row.get("generic_score", 0.0)), float(row.get("normalized_generic_score", 0.0)),
            float(row.get("generic_rank") or 0), float(row.get("frequency_count", 0)),
            float(row.get("personal_score", 0.0)), float(has_personal),
            float(row.get("personal_candidate_rank") or 0), float(row.get("p_ng", 0.0)),
            float(row.get("choice_share", 0.0)), entropy, math.log1p(same_n), math.log1p(raw_n),
            n_score, b_score, math.log1p(bge_counts[name]), float(effective_n), math.log1p(matched),
            base_score - base_top, n_score - ngram_top, b_score - bge_top,
            base_score + 6.0 * n_score + 6.0 * b_score,
        ]
        if len(values) != len(FEATURE_NAMES) or not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"Invalid runtime feature vector: {feature['row_id']}")
        vectors.append(values)
        labels.append(int(name == gold))
    if sum(labels) > 1:
        raise RuntimeError(f"Multiple gold candidates: {feature['row_id']}")
    return vectors, labels, names


def quantiles(values: Sequence[float | int]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "p25", "p50", "p75", "p90", "p99", "max", "mean")}
    ordered = sorted(map(float, values))
    def at(q: float) -> float:
        position = q * (len(ordered) - 1)
        left, right = math.floor(position), math.ceil(position)
        return ordered[left] if left == right else ordered[left] * (right-position) + ordered[right] * (position-left)
    return {"min": ordered[0], "p25": at(.25), "p50": at(.5), "p75": at(.75),
            "p90": at(.9), "p99": at(.99), "max": ordered[-1], "mean": statistics.fmean(ordered)}


def audit_groups(groups: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]], expected: int) -> dict[str, Any]:
    rows = candidates = empty_groups = positive_groups = positive_candidates = positive_group_candidates = 0
    group_sizes: list[int] = []
    source_counts: Counter[str] = Counter()
    source_gold: Counter[str] = Counter()
    feature_values: list[list[float]] = [[] for _ in FEATURE_NAMES]
    per_author: Counter[str] = Counter()
    positive_per_author: Counter[str] = Counter()
    for feature, support in groups:
        if str(feature["row_id"]) != str(support["row_id"]):
            raise RuntimeError("Feature/support row order differs")
        vectors, labels, names = extract_group(feature, support)
        rows += 1
        candidates += len(vectors)
        empty_groups += not vectors
        group_sizes.append(len(vectors))
        author = str(feature["author"])
        per_author[author] += 1
        if sum(labels):
            positive_groups += 1
            positive_group_candidates += len(vectors)
            positive_per_author[author] += 1
        positive_candidates += sum(labels)
        for vector, label, candidate in zip(vectors, labels, names):
            source = next(row["source"] for row in support["retuned_stage1_candidates"] if candidate_text(row) == candidate)
            source_counts[str(source)] += 1
            if label:
                source_gold[str(source)] += 1
            for values, value in zip(feature_values, vector):
                values.append(value)
    if rows != expected:
        raise RuntimeError(f"Group count changed: {rows}")
    return {"groups": rows, "candidates": candidates, "empty_groups": empty_groups,
            "candidate_count": quantiles(group_sizes),
            "positive_groups": positive_groups, "zero_positive_groups": rows - positive_groups,
            "positive_group_candidates": positive_group_candidates,
            "positive_group_rate": positive_groups / rows,
            "positive_candidates": positive_candidates,
            "per_author_groups": dict(sorted(per_author.items())),
            "per_author_positive_groups": dict(sorted(positive_per_author.items())),
            "source_candidates": dict(sorted(source_counts.items())),
            "source_gold": dict(sorted(source_gold.items())),
            "features": {name: quantiles(values) for name, values in zip(FEATURE_NAMES, feature_values)}}


def validate_val_baseline(stage2_path: Path, predictions_path: Path) -> int:
    count = 0
    for count, pair in enumerate(itertools.zip_longest(iter_jsonl(stage2_path), iter_jsonl(predictions_path)), start=1):
        support, prediction = pair
        if support is None or prediction is None:
            raise RuntimeError("Val support/prediction counts differ")
        if str(support["row_id"]) != str(prediction["row_id"]):
            raise RuntimeError(f"Val support/prediction order differs at {count}")
        rows = []
        for item in support["retuned_stage1_candidates"]:
            candidate = candidate_text(item)
            score = (float(item["final_score"])
                     + 6.0 * float(support["retuned_ngram_support"][candidate])
                     + 6.0 * float(support["retuned_bge_support"][candidate]))
            rows.append((score, int(item.get("base_rank", item.get("rank", 0))), candidate))
        rows.sort(key=lambda value: (-value[0], value[1], value[2]))
        if [row[2] for row in rows] != list(map(str, prediction["RetunedFinal_top10"])):
            raise RuntimeError(f"Frozen Val ranking mismatch: {support['row_id']}")
    if count != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"Val baseline count changed: {count}")
    return count


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-supports", type=Path, required=True)
    parser.add_argument("--val-stage1", type=Path, required=True)
    parser.add_argument("--val-stage2", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    expected_hashes = ((args.val_stage1, EXPECTED_VAL_STAGE1_SHA256),
                       (args.val_stage2, EXPECTED_VAL_STAGE2_SHA256),
                       (args.val_predictions, EXPECTED_VAL_PREDICTIONS_SHA256))
    for path, expected in expected_hashes:
        if sha256_file(path) != expected:
            raise RuntimeError(f"Frozen Val artifact changed: {path}")
    fit = audit_groups(((row, row) for row in iter_jsonl(args.fit_supports)), EXPECTED_FIT_ROWS)
    val = audit_groups(zip(iter_jsonl(args.val_stage1), iter_jsonl(args.val_stage2)), EXPECTED_VAL_ROWS)
    baseline_rows = validate_val_baseline(args.val_stage2, args.val_predictions)
    result = {"schema_version": 1, "status": "complete", "feature_names": list(FEATURE_NAMES),
              "author_identity_feature": False, "gold_correctness_runtime_feature": False,
              "fit": fit, "val": val, "frozen_val_baseline_exact": True,
              "frozen_val_baseline_rows": baseline_rows,
              "zero_positive_fit_policy": "exclude from fitting; retain all groups for evaluation",
              "provenance": {"fit_supports": {"path": str(args.fit_supports.resolve()), "sha256": sha256_file(args.fit_supports)},
                             "val_stage1": EXPECTED_VAL_STAGE1_SHA256,
                             "val_stage2": EXPECTED_VAL_STAGE2_SHA256,
                             "val_predictions": EXPECTED_VAL_PREDICTIONS_SHA256},
              "used_dev3000": False, "used_test": False}
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "audit.json"
    write_json(output, result)
    write_json(args.output_root / "artifact_checksums.json", {
        "runner": sha256_file(Path(__file__)), "audit.json": sha256_file(output),
        "used_dev3000": False, "used_test": False})
    print(json.dumps({"status": "complete", "fit": {key: fit[key] for key in ("groups", "candidates", "positive_groups", "zero_positive_groups")},
                      "val": {key: val[key] for key in ("groups", "candidates", "positive_groups", "zero_positive_groups")},
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
