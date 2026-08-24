from __future__ import annotations

"""Expanded-grid arithmetic-only follow-up for Recovery -> NGramRecency + BGERecency.

Purpose
-------
The completed V2 experiment selected lambda_B=8 for K5+Entropy, exactly the
upper edge of the BGE grid. This V3 follow-up expands the BGE weight grid while
reusing the already-computed frozen Stage-1, NGramRecency, and BGERecency
supports. No BGE history/query embedding is recomputed.

Frozen recovery bases
---------------------
1. K5+Entropy          : coverage-first
2. 4P+4CS+2E           : balanced
3. 6P+2CS+.25E         : front-rank / Top3-oriented

Stage-2 score
-------------
    S_final(c) = S_REC(c)
               + lambda_N * P_NG-R(c)
               + lambda_B * P_BGE-R(c)

Default expanded grid
---------------------
    lambda_N in {0,.25,.5,1,2,4,6,8,12}
    lambda_B in {0,.25,.5,1,2,4,6,8,12,16}

Scientific safeguards
---------------------
- Reads only completed Train-Val development artifacts from V1/V2.
- Dev3000/Test are never read.
- Gold is never used for scoring/features; only evaluation / lambda selection.
- Stage-2 remains pure reranking of each frozen Stage-1 Top10.
- Candidate set, Missing@10 and Recovery Rec@10 must remain invariant.
- (lambda_N,lambda_B)=(0,0) must exactly reproduce Stage 1.
- lambda_B=0 must reproduce the completed V1 NGram-only selected point.
- The old V2 selected full-context point must reproduce exactly inside V3.
- V2 BGERecency support is read-only and its SHA is checked against V2 provenance.
- Output is written to a new versioned directory; historical artifacts are not overwritten.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_FIT_SHA256 = "162f5c98daa86cc69947571e6d8f20fc401f0a82cdd3fd6e517eb7be2addbdb4"
EXPECTED_VAL_SHA256 = "d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4"
EXPECTED_STAGE1_FROZEN_SHA256 = "54e60073daabb14bb7cf43136a335216888ea03c06d078a4eec56e5775a0cfbc"
EXPECTED_NGRAM_SUPPORT_SHA256 = "03858de42c41a26c4134d4b069b61ab2a5468c24cbd70a71958d600e448a97e1"

EXPECTED_FIT_ROWS = 144_526
EXPECTED_VAL_ROWS = 34_416
EXPECTED_GENERIC_MISSING = 12_565
EXPECTED_RECOVERABLE_K5 = 4_910

DEFAULT_LAMBDA_N = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0)
DEFAULT_LAMBDA_B = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0)

BASE_ORDER = (
    "K5+Entropy",
    "4P+4CS+2E",
    "6P+2CS+.25E",
)

HISTORICAL_CONTROLS: tuple[dict[str, Any], ...] = (
    {
        "group": "Generic control",
        "method": "G",
        "macro_author_top1": 0.307099,
        "micro_top1": 0.330573,
        "top3": 0.491341,
        "top5": 0.557996,
        "mrr_at_10": 0.426472,
        "missing10": 0.365092,
        "recovery_rec1": 0.0,
        "recovery_rec3": 0.0,
        "recovery_rec5": 0.0,
        "recovery_rec10": 0.0,
        "recovery_mrr_at_10": 0.0,
        "source_precision": "historical_reported",
    },
    {
        "group": "Frequency control",
        "method": "F",
        "macro_author_top1": 0.382495,
        "micro_top1": 0.408473,
        "top3": 0.555120,
        "top5": 0.601000,
        "mrr_at_10": 0.489432,
        "missing10": 0.365092,
        "recovery_rec1": 0.0,
        "recovery_rec3": 0.0,
        "recovery_rec5": 0.0,
        "recovery_rec10": 0.0,
        "recovery_mrr_at_10": 0.0,
        "source_precision": "historical_reported",
    },
    {
        "group": "Historical Recovery control",
        "method": "PV1",
        "macro_author_top1": 0.401872,
        "micro_top1": 0.426749,
        "top3": 0.598907,
        "top5": 0.663093,
        "mrr_at_10": 0.524450,
        "missing10": 0.291144,
        "recovery_rec1": 0.1900,
        "recovery_rec3": 0.4923,
        "recovery_rec5": 0.5363,
        "recovery_rec10": 0.5401,
        "recovery_mrr_at_10": 0.3363,
        "source_precision": "historical_reported_rounded_recovery",
    },
    {
        "group": "Historical Context control",
        "method": "PV1 + NGramRecency",
        "macro_author_top1": 0.429091,
        "micro_top1": 0.452755,
        "top3": 0.611285,
        "top5": 0.665940,
        "mrr_at_10": 0.541944,
        "missing10": 0.291144,
        "recovery_rec1": 0.3601,
        "recovery_rec3": 0.5149,
        "recovery_rec5": 0.5379,
        "recovery_rec10": 0.5401,
        "recovery_mrr_at_10": 0.4361,
        "source_precision": "historical_reported_rounded_recovery",
    },
    {
        "group": "Full historical Context control",
        "method": "PV1 + NGramRecency + BGERecency",
        "macro_author_top1": 0.429506,
        "micro_top1": 0.453423,
        "top3": 0.612099,
        "top5": 0.667335,
        "mrr_at_10": 0.542766,
        "missing10": 0.291144,
        "recovery_rec1": 0.3582,
        "recovery_rec3": 0.5165,
        "recovery_rec5": 0.5381,
        "recovery_rec10": 0.5401,
        "recovery_mrr_at_10": 0.4356,
        "source_precision": "historical_reported_rounded_recovery",
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}") from exc
            if "row_id" not in row:
                raise RuntimeError(f"Missing row_id at {path}:{line_no}")
            rows.append(row)
    return rows


def index_rows(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row["row_id"])
        if row_id in result:
            raise RuntimeError(f"Duplicate row_id in {label}: {row_id}")
        result[row_id] = dict(row)
    return result


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def parse_grid(text: str, *, label: str) -> tuple[float, ...]:
    values = sorted({float(value.strip()) for value in text.split(",") if value.strip()})
    if not values:
        raise ValueError(f"{label} grid is empty")
    if any(value < 0 for value in values):
        raise ValueError(f"{label} values must be non-negative")
    if 0.0 not in values:
        raise ValueError(f"{label} grid must include 0")
    return tuple(values)


def candidate_text(item: Mapping[str, Any]) -> str:
    value = item.get("candidate", item.get("text", item.get("target")))
    if value is None:
        raise RuntimeError(f"Cannot identify candidate text: {item}")
    return str(value)


def rank_of(ranking: Sequence[Mapping[str, Any]], gold: str) -> int | None:
    for index, item in enumerate(ranking, 1):
        if candidate_text(item) == gold:
            return int(item.get("rank", index))
    return None


def close_enough(a: float, b: float, tol: float = 2e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def rerank_fixed_candidate_set(
    base_candidates: Sequence[Mapping[str, Any]],
    ngram_support: Mapping[str, float],
    bge_support: Mapping[str, float],
    *,
    lambda_n: float,
    lambda_b: float,
) -> list[dict[str, Any]]:
    base_texts = [candidate_text(item) for item in base_candidates]
    if set(base_texts) != set(ngram_support) or set(base_texts) != set(bge_support):
        raise RuntimeError("Context support candidate set differs from frozen Stage-1 Top10")

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(base_candidates, 1):
        row = dict(item)
        text = candidate_text(item)
        base_rank = int(item.get("base_rank", item.get("rank", index)))
        base_score = float(item["final_score"])
        nscore = float(ngram_support[text])
        bscore = float(bge_support[text])
        row["base_rank"] = base_rank
        row["base_score"] = base_score
        row["ngram_recency_support"] = nscore
        row["bge_recency_support"] = bscore
        row["context_lambda_n"] = float(lambda_n)
        row["context_lambda_b"] = float(lambda_b)
        row["final_score"] = base_score + float(lambda_n) * nscore + float(lambda_b) * bscore
        rows.append(row)

    rows.sort(
        key=lambda row: (
            -float(row["final_score"]),
            int(row["base_rank"]),
            str(row["candidate"]),
        )
    )
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def metric_summary(rows: Sequence[Mapping[str, Any]], rank_key: str, method: str) -> dict[str, Any]:
    by_author: dict[str, list[int | None]] = defaultdict(list)
    ranks: list[int | None] = []
    for row in rows:
        rank = row.get(rank_key)
        rank = None if rank is None else int(rank)
        ranks.append(rank)
        by_author[str(row["author"])].append(rank)
    if not ranks:
        raise RuntimeError("Cannot score empty rows")

    def top_at(values: Sequence[int | None], k: int) -> float:
        return sum(rank is not None and rank <= k for rank in values) / len(values)

    per_author_top1 = {author: top_at(values, 1) for author, values in sorted(by_author.items())}
    found = [rank for rank in ranks if rank is not None]
    return {
        "method": method,
        "n": len(ranks),
        "authors": len(per_author_top1),
        "macro_author_top1": statistics.fmean(per_author_top1.values()),
        "micro_top1": top_at(ranks, 1),
        "top3": top_at(ranks, 3),
        "top5": top_at(ranks, 5),
        "mrr_at_10": sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / len(ranks),
        "missing10": sum(rank is None for rank in ranks) / len(ranks),
        "mean_rank_given_top10": statistics.fmean(found) if found else None,
        "per_author_top1": per_author_top1,
    }


def recovery_summary(rows: Sequence[Mapping[str, Any]], rank_key: str) -> dict[str, Any]:
    missing = [row for row in rows if bool(row["generic_missing"])]
    available = [row for row in missing if bool(row["gold_in_personal_k5"])]
    if len(missing) != EXPECTED_GENERIC_MISSING:
        raise RuntimeError(f"Generic Missing regression failed: {len(missing)}")
    if len(available) != EXPECTED_RECOVERABLE_K5:
        raise RuntimeError(f"K5 recoverable regression failed: {len(available)}")
    ranks = [None if row.get(rank_key) is None else int(row[rank_key]) for row in available]

    def count_at(k: int) -> int:
        return sum(rank is not None and rank <= k for rank in ranks)

    r1, r3, r5, r10 = (count_at(k) for k in (1, 3, 5, 10))
    found = [rank for rank in ranks if rank is not None]
    denom = len(ranks)
    return {
        "generic_missing_n": len(missing),
        "recoverable_n": denom,
        "recovered_at_1_n": r1,
        "recovered_at_3_n": r3,
        "recovered_at_5_n": r5,
        "recovered_at_10_n": r10,
        "recovered_at_1": r1 / denom,
        "recovered_at_3": r3 / denom,
        "recovered_at_5": r5 / denom,
        "recovered_at_10": r10 / denom,
        "recovery_mrr_at_10": sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / denom,
        "mean_recovered_rank": statistics.fmean(found) if found else None,
    }


def transition_counts(rows: Sequence[Mapping[str, Any]], base_key: str, new_key: str) -> dict[str, int]:
    result = {"n": 0, "rescue": 0, "harm": 0, "unchanged_correct": 0, "unchanged_wrong": 0}
    for row in rows:
        result["n"] += 1
        before = row.get(base_key) == 1
        after = row.get(new_key) == 1
        if not before and after:
            result["rescue"] += 1
        elif before and not after:
            result["harm"] += 1
        elif before and after:
            result["unchanged_correct"] += 1
        else:
            result["unchanged_wrong"] += 1
    result["net"] = result["rescue"] - result["harm"]
    return result


def rank_transition_counts(rows: Sequence[Mapping[str, Any]], base_key: str, new_key: str) -> dict[str, int]:
    result = {"comparable": 0, "improved": 0, "worsened": 0, "same": 0}
    for row in rows:
        before = row.get(base_key)
        after = row.get(new_key)
        if before is None or after is None:
            continue
        before_i, after_i = int(before), int(after)
        result["comparable"] += 1
        if after_i < before_i:
            result["improved"] += 1
        elif after_i > before_i:
            result["worsened"] += 1
        else:
            result["same"] += 1
    result["net"] = result["improved"] - result["worsened"]
    return result


def evaluate_config(
    *,
    base: str,
    lambda_n: float,
    lambda_b: float,
    row_ids: Sequence[str],
    stage1: Mapping[str, Mapping[str, Any]],
    ngram: Mapping[str, Mapping[str, Any]],
    bge: Mapping[str, Mapping[str, Any]],
    include_predictions: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eval_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for row_id in row_ids:
        srow = stage1[row_id]
        base_payload = srow["bases"][base]
        base_candidates = base_payload["candidates"]
        candidates = tuple(str(x) for x in base_payload["top10"])

        n_payload = ngram[row_id]["bases"][base]
        b_payload = bge[row_id]["bases"][base]
        n_candidates = tuple(str(x) for x in n_payload["candidates"])
        b_candidates = tuple(str(x) for x in b_payload["candidates"])
        if candidates != n_candidates or candidates != b_candidates:
            raise RuntimeError(f"Support candidate-order mismatch at {row_id} {base}")

        n_support = {str(k): float(v) for k, v in n_payload["support"].items()}
        b_support = {str(k): float(v) for k, v in b_payload["support"].items()}
        ranked = rerank_fixed_candidate_set(
            base_candidates,
            n_support,
            b_support,
            lambda_n=lambda_n,
            lambda_b=lambda_b,
        )
        ranked_texts = [candidate_text(item) for item in ranked]
        if set(ranked_texts) != set(candidates):
            raise RuntimeError(f"Candidate-set invariant failed at {row_id} {base}")
        if lambda_n == 0.0 and lambda_b == 0.0 and ranked_texts != list(candidates):
            raise RuntimeError(f"(0,0) did not reproduce Stage1 order at {row_id} {base}")

        gold = str(srow["gold"])
        new_rank = rank_of(ranked, gold)
        base_rank = base_payload["gold_rank"]
        eval_rows.append(
            {
                "row_id": row_id,
                "author": srow["author"],
                "generic_missing": bool(srow["generic_missing"]),
                "gold_in_personal_k5": bool(srow["gold_in_personal_k5"]),
                "base_rank": base_rank,
                "rank": new_rank,
            }
        )
        if include_predictions:
            prediction_rows.append(
                {
                    "schema_version": 1,
                    "experiment": "initial_recovery_bge_ngram_context_fusion_v3",
                    "row_id": row_id,
                    "author": srow["author"],
                    "gold": gold,
                    "base": base,
                    "lambda_n": lambda_n,
                    "lambda_b": lambda_b,
                    "base_gold_rank": base_rank,
                    "gold_rank": new_rank,
                    "top10": ranked_texts,
                    "candidates": ranked,
                    "gold_used_for_scoring": False,
                    "dev3000_used": False,
                    "test_used": False,
                }
            )

    method = f"{base}+NG-R+BGE-R|lambda_N={lambda_n:g}|lambda_B={lambda_b:g}"
    metrics = metric_summary(eval_rows, "rank", method)
    recovery = recovery_summary(eval_rows, "rank")
    top1 = transition_counts(eval_rows, "base_rank", "rank")
    rank_trans = rank_transition_counts(eval_rows, "base_rank", "rank")
    grid_row = {
        "base": base,
        "lambda_n": lambda_n,
        "lambda_b": lambda_b,
        "macro_author_top1": metrics["macro_author_top1"],
        "micro_top1": metrics["micro_top1"],
        "top3": metrics["top3"],
        "top5": metrics["top5"],
        "mrr_at_10": metrics["mrr_at_10"],
        "missing10": metrics["missing10"],
        "recovery_rec1": recovery["recovered_at_1"],
        "recovery_rec3": recovery["recovered_at_3"],
        "recovery_rec5": recovery["recovered_at_5"],
        "recovery_rec10": recovery["recovered_at_10"],
        "recovery_mrr_at_10": recovery["recovery_mrr_at_10"],
        "top1_rescue": top1["rescue"],
        "top1_harm": top1["harm"],
        "top1_net": top1["net"],
        "rank_improved": rank_trans["improved"],
        "rank_worsened": rank_trans["worsened"],
        "rank_same": rank_trans["same"],
        "rank_net": rank_trans["net"],
    }
    return grid_row, prediction_rows


def prepare_output(output_root: Path, setup: Mapping[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    setup_path = output_root / "run_setup.json"
    existing = list(output_root.iterdir())
    if not existing:
        write_json(setup_path, setup)
        return
    if not setup_path.is_file():
        raise RuntimeError(
            f"Refusing non-empty output directory without run_setup.json: {output_root}\n"
            "Choose a new versioned --output-root."
        )
    previous = json.loads(setup_path.read_text(encoding="utf-8"))
    if previous != dict(setup):
        raise RuntimeError(
            f"Existing output setup differs from this run: {output_root}\n"
            "Choose a new versioned --output-root."
        )
    print(f"Matching output root already exists: {output_root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument(
        "--base-ngram-root",
        type=Path,
        required=True,
        help="Completed recovery_ngram_context_fusion_v1 directory",
    )
    parser.add_argument(
        "--base-bge-root",
        type=Path,
        required=True,
        help="Completed recovery_bge_ngram_context_fusion_v2 directory; read-only",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lambda-n", default=",".join(str(x) for x in DEFAULT_LAMBDA_N))
    parser.add_argument("--lambda-b", default=",".join(str(x) for x in DEFAULT_LAMBDA_B))
    parser.add_argument("--progress-every-configs", type=int, default=10)
    args = parser.parse_args()

    lambda_n_grid = parse_grid(args.lambda_n, label="lambda_N")
    lambda_b_grid = parse_grid(args.lambda_b, label="lambda_B")

    stage1_path = args.base_ngram_root / "stage1_frozen.jsonl"
    ngram_path = args.base_ngram_root / "ngram_recency_support.jsonl"
    v1_comparison_path = args.base_ngram_root / "comparison.json"
    bge_path = args.base_bge_root / "bge_recency_support.jsonl"
    v2_comparison_path = args.base_bge_root / "comparison.json"
    v2_checksums_path = args.base_bge_root / "artifact_checksums.json"

    required = {
        "fit": args.fit,
        "val": args.val,
        "stage1_frozen": stage1_path,
        "ngram_recency_support": ngram_path,
        "v1_comparison": v1_comparison_path,
        "bge_recency_support": bge_path,
        "v2_comparison": v2_comparison_path,
        "v2_checksums": v2_checksums_path,
    }
    for label, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")

    input_hashes = {label: sha256_file(path) for label, path in required.items()}
    if input_hashes["fit"] != EXPECTED_FIT_SHA256:
        raise RuntimeError(f"Fit SHA mismatch: {input_hashes['fit']}")
    if input_hashes["val"] != EXPECTED_VAL_SHA256:
        raise RuntimeError(f"Val SHA mismatch: {input_hashes['val']}")
    if input_hashes["stage1_frozen"] != EXPECTED_STAGE1_FROZEN_SHA256:
        raise RuntimeError(f"Stage1 SHA mismatch: {input_hashes['stage1_frozen']}")
    if input_hashes["ngram_recency_support"] != EXPECTED_NGRAM_SUPPORT_SHA256:
        raise RuntimeError(f"NGram support SHA mismatch: {input_hashes['ngram_recency_support']}")

    v1_comparison = json.loads(v1_comparison_path.read_text(encoding="utf-8"))
    v2_comparison = json.loads(v2_comparison_path.read_text(encoding="utf-8"))
    v2_checksums = json.loads(v2_checksums_path.read_text(encoding="utf-8"))

    if v1_comparison.get("status") != "complete" or v2_comparison.get("status") != "complete":
        raise RuntimeError("V1/V2 base comparison is not complete")
    for label, comparison in (("V1", v1_comparison), ("V2", v2_comparison)):
        protocol = comparison.get("protocol", {})
        if protocol.get("dev3000_used") is not False or protocol.get("test_used") is not False:
            raise RuntimeError(f"{label} provenance crossed Dev3000/Test boundary")

    recorded_bge_sha = (
        v2_comparison.get("bge_scoring_summary", {}).get("bge_support_sha256")
        or v2_checksums.get("bge_recency_support.jsonl", {}).get("sha256")
    )
    if not recorded_bge_sha:
        raise RuntimeError("V2 does not expose a recorded BGE support SHA")
    if input_hashes["bge_recency_support"] != str(recorded_bge_sha):
        raise RuntimeError(
            "V2 BGERecency support SHA mismatch:\n"
            f"recorded={recorded_bge_sha}\nactual={input_hashes['bge_recency_support']}"
        )

    setup = {
        "schema_version": 1,
        "experiment": "initial_recovery_bge_ngram_context_fusion_v3",
        "status": "setup",
        "purpose": "expanded lambda_B boundary follow-up; arithmetic only; no embedding recomputation",
        "input_sha256": input_hashes,
        "base_ngram_root": str(args.base_ngram_root.resolve()),
        "base_bge_root": str(args.base_bge_root.resolve()),
        "lambda_n_grid": list(lambda_n_grid),
        "lambda_b_grid": list(lambda_b_grid),
        "gold_used_for_scoring": False,
        "dev3000_used": False,
        "test_used": False,
    }
    prepare_output(args.output_root, setup)

    started = time.perf_counter()
    print("=== RECOVERY -> NGRAMRECENCY + BGERECENCY EXPANDED GRID V3 ===")
    print(f"Recovery bases: {list(BASE_ORDER)}")
    print(f"lambda_N grid: {list(lambda_n_grid)}")
    print(f"lambda_B grid: {list(lambda_b_grid)}")
    print("BGERecency support: read-only from completed V2")
    print("BGE embeddings recomputed: false")
    print("Candidate set: frozen separately for each Recovery base; pure Stage-2 reranking")
    print("Gold used for scoring/features: false")
    print("Gold used for Train-Val lambda selection/evaluation only: true")
    print("Dev3000 used: false")
    print("Test used: false\n")

    fit_rows = read_jsonl(args.fit)
    val_rows = read_jsonl(args.val)
    stage1_rows = read_jsonl(stage1_path)
    ngram_rows = read_jsonl(ngram_path)
    bge_rows = read_jsonl(bge_path)
    if len(fit_rows) != EXPECTED_FIT_ROWS:
        raise RuntimeError(f"Unexpected Train-Fit rows: {len(fit_rows)}")
    if not (len(val_rows) == len(stage1_rows) == len(ngram_rows) == len(bge_rows) == EXPECTED_VAL_ROWS):
        raise RuntimeError("Unexpected Train-Val / support row count")

    val = index_rows(val_rows, "Train-Val")
    stage1 = index_rows(stage1_rows, "Stage1 frozen")
    ngram = index_rows(ngram_rows, "NGramRecency support")
    bge = index_rows(bge_rows, "BGERecency support")
    if not (set(val) == set(stage1) == set(ngram) == set(bge)):
        raise RuntimeError("Row-ID mismatch among Train-Val / Stage1 / NGram / BGE")
    row_ids = sorted(stage1)

    # Full support candidate-order audit once before the grid.
    print("Auditing frozen candidate-order alignment across Stage1 / NGram / BGE ...")
    for number, row_id in enumerate(row_ids, 1):
        for base in BASE_ORDER:
            expected = tuple(str(x) for x in stage1[row_id]["bases"][base]["top10"])
            n_actual = tuple(str(x) for x in ngram[row_id]["bases"][base]["candidates"])
            b_actual = tuple(str(x) for x in bge[row_id]["bases"][base]["candidates"])
            if expected != n_actual or expected != b_actual:
                raise RuntimeError(f"Candidate-order alignment failed at {row_id} {base}")
        if number % 5000 == 0 or number == len(row_ids):
            print(f"  audit {number}/{len(row_ids)}", flush=True)

    stage1_metrics = v1_comparison["stage1_metrics"]
    stage1_recovery = v1_comparison["stage1_recovery"]
    v1_selected = v1_comparison["selected_by_base"]
    v2_selected = v2_comparison["selected_by_base"]

    print("\nEvaluating expanded lambda_N x lambda_B grid ...")
    grid_rows: list[dict[str, Any]] = []
    total_configs = len(BASE_ORDER) * len(lambda_n_grid) * len(lambda_b_grid)
    config_no = 0

    for base in BASE_ORDER:
        base_missing = float(stage1_metrics[base]["missing10"])
        base_rec10 = float(stage1_recovery[base]["recovered_at_10"])
        for lambda_n in lambda_n_grid:
            for lambda_b in lambda_b_grid:
                config_no += 1
                grid_row, _ = evaluate_config(
                    base=base,
                    lambda_n=lambda_n,
                    lambda_b=lambda_b,
                    row_ids=row_ids,
                    stage1=stage1,
                    ngram=ngram,
                    bge=bge,
                    include_predictions=False,
                )
                if not close_enough(float(grid_row["missing10"]), base_missing):
                    raise RuntimeError(
                        f"Missing@10 changed under pure reranking for {base} N={lambda_n} B={lambda_b}"
                    )
                if not close_enough(float(grid_row["recovery_rec10"]), base_rec10):
                    raise RuntimeError(
                        f"Rec@10 changed under pure reranking for {base} N={lambda_n} B={lambda_b}"
                    )
                grid_rows.append(grid_row)
                if args.progress_every_configs > 0 and (
                    config_no % args.progress_every_configs == 0 or config_no == total_configs
                ):
                    elapsed = time.perf_counter() - started
                    print(
                        f"  configs {config_no}/{total_configs}  rate={config_no / elapsed:.2f} configs/s",
                        flush=True,
                    )

    grid_path = args.output_root / "grid_results.csv"
    write_csv(grid_path, grid_rows)

    # Regression 1: V1 NGram-only selected point must be identical at lambda_B=0.
    print("\nV1 NGram-only regression checks (lambda_B=0):")
    for base in BASE_ORDER:
        expected = v1_selected[base]
        match = next(
            row for row in grid_rows
            if row["base"] == base
            and float(row["lambda_n"]) == float(expected["lambda_n"])
            and float(row["lambda_b"]) == 0.0
        )
        for key in (
            "macro_author_top1", "micro_top1", "top3", "top5", "mrr_at_10", "missing10",
            "recovery_rec1", "recovery_rec3", "recovery_rec5", "recovery_rec10", "recovery_mrr_at_10",
        ):
            if not close_enough(float(match[key]), float(expected[key])):
                raise RuntimeError(f"V1 regression failed {base} {key}: {match[key]} != {expected[key]}")
        print(
            f"  PASS {base:16s} lambda_N={float(expected['lambda_n']):g}  "
            f"Macro={float(match['macro_author_top1']):.6f} MRR={float(match['mrr_at_10']):.6f}"
        )

    # Regression 2: the old V2 selected point must be reproduced inside the expanded grid.
    print("\nV2 full-context selected-point regression checks:")
    for base in BASE_ORDER:
        expected = v2_selected[base]
        match = next(
            row for row in grid_rows
            if row["base"] == base
            and float(row["lambda_n"]) == float(expected["lambda_n"])
            and float(row["lambda_b"]) == float(expected["lambda_b"])
        )
        for key in (
            "macro_author_top1", "micro_top1", "top3", "top5", "mrr_at_10", "missing10",
            "recovery_rec1", "recovery_rec3", "recovery_rec5", "recovery_rec10", "recovery_mrr_at_10",
            "top1_rescue", "top1_harm", "top1_net", "rank_improved", "rank_worsened", "rank_same", "rank_net",
        ):
            if not close_enough(float(match[key]), float(expected[key])):
                raise RuntimeError(f"V2 regression failed {base} {key}: {match[key]} != {expected[key]}")
        print(
            f"  PASS {base:16s} lambda_N={float(expected['lambda_n']):g} "
            f"lambda_B={float(expected['lambda_b']):g}  Macro={float(match['macro_author_top1']):.6f}"
        )

    selected_by_base: dict[str, dict[str, Any]] = {}
    selected_rows: list[dict[str, Any]] = []
    for base in BASE_ORDER:
        candidates = [row for row in grid_rows if row["base"] == base]
        best = max(
            candidates,
            key=lambda row: (
                float(row["macro_author_top1"]),
                float(row["mrr_at_10"]),
                -(float(row["lambda_n"]) + float(row["lambda_b"])),
                -float(row["lambda_b"]),
                -float(row["lambda_n"]),
            ),
        )
        selected_by_base[base] = dict(best)
        selected_rows.append(dict(best))

    selected_metrics_path = args.output_root / "selected_metrics.csv"
    write_csv(selected_metrics_path, selected_rows)

    # Materialize predictions only for the three selected configurations.
    selected_predictions: list[dict[str, Any]] = []
    for base in BASE_ORDER:
        best = selected_by_base[base]
        check_row, predictions = evaluate_config(
            base=base,
            lambda_n=float(best["lambda_n"]),
            lambda_b=float(best["lambda_b"]),
            row_ids=row_ids,
            stage1=stage1,
            ngram=ngram,
            bge=bge,
            include_predictions=True,
        )
        for key in best:
            if key == "base":
                continue
            if isinstance(best[key], (int, float)) and isinstance(check_row.get(key), (int, float)):
                if not close_enough(float(best[key]), float(check_row[key])):
                    raise RuntimeError(f"Selected re-materialization mismatch {base} {key}")
        selected_predictions.extend(predictions)

    selected_predictions_path = args.output_root / "selected_predictions.jsonl"
    write_jsonl(selected_predictions_path, selected_predictions)

    # Full comparison table.
    full_comparison_rows: list[dict[str, Any]] = [dict(row) for row in HISTORICAL_CONTROLS]
    for base in BASE_ORDER:
        m1 = stage1_metrics[base]
        r1 = stage1_recovery[base]
        full_comparison_rows.append(
            {
                "group": "Recovery only",
                "method": base,
                "macro_author_top1": m1["macro_author_top1"],
                "micro_top1": m1["micro_top1"],
                "top3": m1["top3"],
                "top5": m1["top5"],
                "mrr_at_10": m1["mrr_at_10"],
                "missing10": m1["missing10"],
                "recovery_rec1": r1["recovered_at_1"],
                "recovery_rec3": r1["recovered_at_3"],
                "recovery_rec5": r1["recovered_at_5"],
                "recovery_rec10": r1["recovered_at_10"],
                "recovery_mrr_at_10": r1["recovery_mrr_at_10"],
                "source_precision": "exact_v1_artifact",
            }
        )
        ng = v1_selected[base]
        full_comparison_rows.append(
            {
                "group": "+ NGram Context",
                "method": f"{base} + NGramRecency",
                "macro_author_top1": ng["macro_author_top1"],
                "micro_top1": ng["micro_top1"],
                "top3": ng["top3"],
                "top5": ng["top5"],
                "mrr_at_10": ng["mrr_at_10"],
                "missing10": ng["missing10"],
                "recovery_rec1": ng["recovery_rec1"],
                "recovery_rec3": ng["recovery_rec3"],
                "recovery_rec5": ng["recovery_rec5"],
                "recovery_rec10": ng["recovery_rec10"],
                "recovery_mrr_at_10": ng["recovery_mrr_at_10"],
                "lambda_n": ng["lambda_n"],
                "lambda_b": 0.0,
                "source_precision": "exact_v1_artifact",
            }
        )
        full = selected_by_base[base]
        full_comparison_rows.append(
            {
                "group": "+ NGram + BGE Context (expanded grid)",
                "method": f"{base} + NGramRecency + BGERecency",
                **{k: v for k, v in full.items() if k != "base"},
                "source_precision": "exact_v3_artifact",
            }
        )

    full_comparison_path = args.output_root / "full_comparison.csv"
    write_csv(full_comparison_path, full_comparison_rows)

    max_lambda_b = max(lambda_b_grid)
    max_lambda_n = max(lambda_n_grid)
    selection_deltas: dict[str, dict[str, Any]] = {}
    boundary_status: dict[str, dict[str, Any]] = {}
    for base in BASE_ORDER:
        old = v2_selected[base]
        new = selected_by_base[base]
        selection_deltas[base] = {
            "old_v2_lambda_n": old["lambda_n"],
            "old_v2_lambda_b": old["lambda_b"],
            "new_v3_lambda_n": new["lambda_n"],
            "new_v3_lambda_b": new["lambda_b"],
            "macro_delta": float(new["macro_author_top1"]) - float(old["macro_author_top1"]),
            "micro_delta": float(new["micro_top1"]) - float(old["micro_top1"]),
            "top3_delta": float(new["top3"]) - float(old["top3"]),
            "top5_delta": float(new["top5"]) - float(old["top5"]),
            "mrr_delta": float(new["mrr_at_10"]) - float(old["mrr_at_10"]),
            "rec1_delta": float(new["recovery_rec1"]) - float(old["recovery_rec1"]),
            "rec3_delta": float(new["recovery_rec3"]) - float(old["recovery_rec3"]),
            "rec5_delta": float(new["recovery_rec5"]) - float(old["recovery_rec5"]),
            "recmrr_delta": float(new["recovery_mrr_at_10"]) - float(old["recovery_mrr_at_10"]),
        }
        boundary_status[base] = {
            "selected_lambda_n": new["lambda_n"],
            "selected_lambda_b": new["lambda_b"],
            "lambda_n_hits_upper_boundary": float(new["lambda_n"]) == float(max_lambda_n),
            "lambda_b_hits_upper_boundary": float(new["lambda_b"]) == float(max_lambda_b),
        }

    comparison = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_recovery_bge_ngram_context_fusion_v3",
        "research_question": (
            "Does the V2 K5+Entropy BGE boundary result persist when lambda_B is expanded, "
            "and which Recovery philosophy is best under the common expanded context grid?"
        ),
        "purpose": "expanded lambda_B boundary follow-up; arithmetic only; no embedding recomputation",
        "rows": EXPECTED_VAL_ROWS,
        "stage2_formula": "S_final(c)=S_REC(c)+lambda_N*P_NG-R(c)+lambda_B*P_BGE-R(c)",
        "lambda_n_grid": list(lambda_n_grid),
        "lambda_b_grid": list(lambda_b_grid),
        "selection_rule": (
            "per base: maximize Macro-author Top1; tie MRR@10; then smaller total context weight; "
            "then smaller lambda_B; then smaller lambda_N"
        ),
        "stage1_metrics": stage1_metrics,
        "stage1_recovery": stage1_recovery,
        "v1_ngram_selected_by_base": v1_selected,
        "v2_full_selected_by_base": v2_selected,
        "selected_by_base": selected_by_base,
        "selection_delta_vs_v2": selection_deltas,
        "boundary_status": boundary_status,
        "historical_controls_reporting_only": list(HISTORICAL_CONTROLS),
        "invariants": {
            "candidate_set_changed": False,
            "lambda_0_0_must_exactly_reproduce_stage1_order": True,
            "missing10_must_equal_stage1": True,
            "recovery_rec10_must_equal_stage1_on_fixed_R": True,
            "lambda_b_0_reproduces_v1_ngram_selected": True,
            "old_v2_selected_points_reproduced": True,
            "bge_support_recomputed": False,
        },
        "protocol": {
            "train_fit_rows": EXPECTED_FIT_ROWS,
            "train_val_rows": EXPECTED_VAL_ROWS,
            "generic_missing_n": EXPECTED_GENERIC_MISSING,
            "recoverable_k5_n": EXPECTED_RECOVERABLE_K5,
            "gold_used_for_feature_construction": False,
            "gold_used_for_scoring": False,
            "gold_used_for_train_val_selection_and_evaluation_only": True,
            "dev3000_used": False,
            "test_used": False,
        },
        "provenance": {
            "input_sha256": input_hashes,
            "bge_recency_support_read_only": str(bge_path.resolve()),
            "bge_recency_support_sha256": input_hashes["bge_recency_support"],
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    comparison_path = args.output_root / "comparison.json"
    write_json(comparison_path, comparison)

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_recovery_bge_ngram_context_fusion_v3",
        "input_sha256": input_hashes,
        "outputs": {
            "grid_results": str(grid_path.resolve()),
            "selected_metrics": str(selected_metrics_path.resolve()),
            "selected_predictions": str(selected_predictions_path.resolve()),
            "full_comparison": str(full_comparison_path.resolve()),
            "comparison": str(comparison_path.resolve()),
        },
        "bge_support_recomputed": False,
        "gold_used_for_scoring": False,
        "dev3000_used": False,
        "test_used": False,
    }
    manifest_path = args.output_root / "run_manifest.json"
    write_json(manifest_path, manifest)

    artifact_paths = [
        grid_path,
        selected_metrics_path,
        selected_predictions_path,
        full_comparison_path,
        comparison_path,
        manifest_path,
        args.output_root / "run_setup.json",
    ]
    checksums = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in artifact_paths
    }
    write_json(args.output_root / "artifact_checksums.json", checksums)

    print("\n=== SELECTED EXPANDED-GRID RESULT PER RECOVERY BASE ===")
    for base in BASE_ORDER:
        row = selected_by_base[base]
        print(
            f"{base:16s} lambda_N={float(row['lambda_n']):g} lambda_B={float(row['lambda_b']):g}  "
            f"Macro={float(row['macro_author_top1']):.6f}  "
            f"Micro={float(row['micro_top1']):.6f}  "
            f"Top3={float(row['top3']):.6f}  Top5={float(row['top5']):.6f}  "
            f"MRR={float(row['mrr_at_10']):.6f}  Missing={float(row['missing10']):.6f}  "
            f"Rec1={float(row['recovery_rec1']):.4f}  Rec3={float(row['recovery_rec3']):.4f}  "
            f"Rec5={float(row['recovery_rec5']):.4f}  Rec10={float(row['recovery_rec10']):.4f}  "
            f"RecMRR={float(row['recovery_mrr_at_10']):.4f}  Top1Net={int(row['top1_net']):+d}"
        )

    print("\n=== DELTA VS PREVIOUS V2 SELECTED POINT ===")
    for base in BASE_ORDER:
        delta = selection_deltas[base]
        print(
            f"{base:16s} "
            f"({float(delta['old_v2_lambda_n']):g},{float(delta['old_v2_lambda_b']):g}) -> "
            f"({float(delta['new_v3_lambda_n']):g},{float(delta['new_v3_lambda_b']):g})  "
            f"DeltaMacro={float(delta['macro_delta']):+.6f}  DeltaMRR={float(delta['mrr_delta']):+.6f}"
        )

    print("\n=== BOUNDARY STATUS ===")
    any_boundary = False
    for base in BASE_ORDER:
        status = boundary_status[base]
        hit_b = bool(status["lambda_b_hits_upper_boundary"])
        any_boundary = any_boundary or hit_b
        print(
            f"{base:16s} lambda_B={float(status['selected_lambda_b']):g}  "
            f"upper={max_lambda_b:g}  hit_upper_boundary={str(hit_b).lower()}"
        )
    if any_boundary:
        print("WARNING: at least one selected lambda_B still hits the expanded upper boundary.")
    else:
        print("PASS: no selected lambda_B hits the expanded upper boundary.")

    print("\nOutputs:")
    for path in (
        grid_path,
        selected_metrics_path,
        selected_predictions_path,
        full_comparison_path,
        comparison_path,
        manifest_path,
        args.output_root / "artifact_checksums.json",
    ):
        print(f"  {path}")
    print(f"\nRuntime: {time.perf_counter() - started:.1f}s")
    print("BGE embeddings recomputed: false")
    print("Dev3000 used: false")
    print("Test used: false")


if __name__ == "__main__":
    main()
