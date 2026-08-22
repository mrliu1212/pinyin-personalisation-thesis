"""Audit the frozen causal pair registry and prepare compact bi-encoder groups."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from src.personalisation.task_specific_biencoder import (
    assign_inner_split,
    canonical_json,
    context64,
    refuse_closed_path,
    sha256_file,
    split_position_cutoffs,
    write_json,
    write_jsonl,
)


EXPECTED_FIT_SHA256 = "547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6"
EXPECTED_PAIRS_SHA256 = "bf36b4e6f5b67867a5d2daf1010ebbafed3becdbdac51684bf1ca09e53ecedf8"
EXPECTED = {
    "eligible_queries": 35_290,
    "groups": 99_671,
    "positive_pairs": 99_671,
    "negative_pairs": 169_400,
    "total_pairs": 269_071,
    "trainable_groups": 66_672,
}


def load_fit(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("standardized_partition") != "train_fit":
                raise ValueError("non-Train-Fit row in frozen fit manifest")
            if row.get("used_dev3000") or row.get("used_test") or row.get("pilot_partition") == "test":
                raise ValueError("closed-data marker in Train-Fit manifest")
            row_id = str(row["row_id"])
            if row_id in rows:
                raise ValueError(f"duplicate Train-Fit row: {row_id}")
            context = str(row["context"])
            rows[row_id] = {
                "author": str(row["author"]),
                "position": int(row["chronological_position"]),
                "pinyin": tuple(str(item) for item in row["pinyin_segments"]),
                "target": str(row["target"]),
                "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
                "context64": context64(context),
            }
    return rows


def prepare(fit_path: Path, pair_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sha256_file(fit_path) != EXPECTED_FIT_SHA256:
        raise ValueError("Train-Fit SHA256 differs from the frozen Clean3 manifest")
    if sha256_file(pair_path) != EXPECTED_PAIRS_SHA256:
        raise ValueError("pair registry SHA256 differs from the audited EM3 artifact")
    fit = load_fit(fit_path)

    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    pair_keys: set[tuple[str, str]] = set()
    counts = Counter()
    violations = Counter()
    with pair_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            pair = json.loads(line)
            query_id = str(pair["query_row_id"])
            history_id = str(pair["history_row_id"])
            query = fit.get(query_id)
            history = fit.get(history_id)
            if query is None or history is None:
                violations["unknown_row"] += 1
                continue
            pair_key = (query_id, history_id)
            if pair_key in pair_keys:
                violations["duplicate_query_history"] += 1
            pair_keys.add(pair_key)
            author = str(pair["author"])
            label = int(pair["label"])
            if author != query["author"] or author != history["author"]:
                violations["cross_author"] += 1
            if query["pinyin"] != history["pinyin"] or tuple(pair["pinyin_segments"]) != query["pinyin"]:
                violations["cross_pinyin"] += 1
            if history["position"] >= query["position"]:
                violations["non_prior"] += 1
            if int(pair["query_position"]) != query["position"] or int(pair["history_position"]) != history["position"]:
                violations["position_mismatch"] += 1
            if str(pair["current_gold"]) != query["target"] or str(pair["history_target"]) != history["target"]:
                violations["target_mismatch"] += 1
            expected_label = int(query["target"] == history["target"])
            if label != expected_label:
                violations["label_mismatch"] += 1
            if hashlib.sha256(str(pair["current_context"]).encode("utf-8")).hexdigest() != query["context_sha256"]:
                violations["query_context_mismatch"] += 1
            if hashlib.sha256(str(pair["history_context"]).encode("utf-8")).hexdigest() != history["context_sha256"]:
                violations["history_context_mismatch"] += 1

            group_key = (query_id, int(pair["round"]))
            group = grouped.setdefault(
                group_key,
                {
                    "query_row_id": query_id,
                    "round": int(pair["round"]),
                    "author": author,
                    "query_position": query["position"],
                    "query_context": query["context64"],
                    "positive": [],
                    "negative": [],
                },
            )
            if any(
                group[key] != value
                for key, value in (
                    ("author", author),
                    ("query_position", query["position"]),
                    ("query_context", query["context64"]),
                )
            ):
                violations["inconsistent_group"] += 1
            bucket = "positive" if label else "negative"
            group[bucket].append({"row_id": history_id, "context": history["context64"]})
            counts["total_pairs"] += 1
            counts["positive_pairs" if label else "negative_pairs"] += 1
            if line_number % 50_000 == 0:
                print(f"pair audit {line_number:,}", flush=True)

    query_ids = {key[0] for key in grouped}
    cutoffs = split_position_cutoffs(
        [(query_id, str(fit[query_id]["author"]), int(fit[query_id]["position"])) for query_id in query_ids]
    )
    output_groups = []
    group_sizes = Counter()
    split_counts = Counter()
    split_author_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for (query_id, round_number), group in sorted(
        grouped.items(), key=lambda item: (item[1]["author"], item[1]["query_position"], item[0])
    ):
        if len(group["positive"]) != 1:
            violations["positive_count"] += 1
        if not 0 <= len(group["negative"]) <= 3:
            violations["negative_count"] += 1
        split = assign_inner_split(group["author"], group["query_position"], cutoffs)
        histories = [*group["positive"], *group["negative"]]
        group_id = hashlib.sha256(f"{query_id}\0{round_number}".encode("utf-8")).hexdigest()[:24]
        output_groups.append(
            {
                "schema_version": 1,
                "group_id": group_id,
                "query_row_id": query_id,
                "round": round_number,
                "author": group["author"],
                "query_position": group["query_position"],
                "query_context": group["query_context"],
                "history_row_ids": [item["row_id"] for item in histories],
                "history_contexts": [item["context"] for item in histories],
                "labels": [1, *([0] * len(group["negative"]))],
                "trainable": bool(group["negative"]),
                "split": split,
                "used_dev3000": False,
                "used_test": False,
            }
        )
        group_sizes[len(histories)] += 1
        split_counts[split] += 1
        split_author_counts[group["author"]][split] += 1

    counts["groups"] = len(output_groups)
    counts["eligible_queries"] = len(query_ids)
    counts["trainable_groups"] = sum(bool(row["trainable"]) for row in output_groups)
    if dict(counts) != EXPECTED:
        violations["frozen_count_mismatch"] += 1

    split_positions: dict[str, dict[str, int]] = {}
    for author in cutoffs:
        fit_positions = [row["query_position"] for row in output_groups if row["author"] == author and row["split"] == "inner_fit"]
        gate_positions = [row["query_position"] for row in output_groups if row["author"] == author and row["split"] == "inner_gate"]
        split_positions[author] = {
            "max_inner_fit_position": max(fit_positions),
            "min_inner_gate_position": min(gate_positions),
            "strictly_chronological": max(fit_positions) < min(gate_positions),
        }
        if not split_positions[author]["strictly_chronological"]:
            violations["split_chronology"] += 1

    audit = {
        "schema_version": 1,
        "status": "passed" if not violations else "failed",
        "counts": dict(counts),
        "expected_counts": EXPECTED,
        "violations": dict(violations),
        "group_size_distribution": {str(key): value for key, value in sorted(group_sizes.items())},
        "inner_split": {
            "rule": "per-author earliest whole position blocks up to approximately 90%; later blocks gate",
            "cutoff_positions": cutoffs,
            "group_counts": dict(split_counts),
            "trainable_group_counts": dict(Counter(row["split"] for row in output_groups if row["trainable"])),
            "per_author_group_counts": {author: dict(value) for author, value in sorted(split_author_counts.items())},
            "chronology": split_positions,
        },
        "supervision": "one matching-target positive plus query-local same-Pinyin wrong-target negatives",
        "author_serialized": False,
        "gold_serialized": False,
        "used_dev3000": False,
        "used_test": False,
    }
    if violations:
        raise RuntimeError(canonical_json(audit))
    return output_groups, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    refuse_closed_path(args.fit)
    refuse_closed_path(args.pairs)
    started = time.perf_counter()
    groups, audit = prepare(args.fit, args.pairs)
    args.output_root.mkdir(parents=True, exist_ok=True)
    group_path = args.output_root / "groups.jsonl"
    write_jsonl(group_path, groups)
    audit["runtime_seconds"] = time.perf_counter() - started
    audit["inputs"] = {
        "fit": {"path": str(args.fit.resolve()), "sha256": sha256_file(args.fit)},
        "pairs": {"path": str(args.pairs.resolve()), "sha256": sha256_file(args.pairs)},
    }
    audit["outputs"] = {"groups": {"path": str(group_path.resolve()), "sha256": sha256_file(group_path)}}
    audit["runner_sha256"] = sha256_file(Path(__file__))
    write_json(args.output_root / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
