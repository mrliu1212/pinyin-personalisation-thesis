"""Pre-Dev paired statistical validation for Initial-Pinyin personalisation.

Purpose
-------
This runner compares the frozen Train-Val primary method

    Balanced Joint = B + 4 P_NG + 4 CS + 2 C_E

against four pre-declared comparators:

    1. PV1
    2. Balanced Joint without Entropy = B + 4 P_NG + 4 CS
    3. NGramSelector@K5  (same K5 breadth architectural comparator)
    4. NGramSelector@K3  (best balanced selector operating point)

It is analysis-only. It performs no model inference, no candidate construction,
no parameter tuning, and does not read Dev3000 or Test.

Statistics
----------
- Micro Top1: exact two-sided McNemar test.
- Macro-author Top1: paired author-stratified bootstrap.
- Micro Top1 / Top3 / Top5 / MRR@10 / Missing@10: paired
  author-stratified bootstrap using the same resampled rows for both methods.
- Per-author metrics and rescue/harm/net are also written.

The bootstrap resamples rows independently *within each author* while preserving
each author's original row count. Macro-author Top1 is recomputed by equally
averaging the per-author Top1 values on every replicate.

Rank-source resolution
----------------------
PV1 is read directly from ``pv1_rank`` in the frozen Frequency/PV1 artifact.
For the two later prediction artifacts, exact field schemas varied across
experimental runners. To avoid silently guessing a rank field, this script
scans scalar rank-like leaves and resolves the desired rank vector by matching
its full Train-Val headline metric signature against frozen expected metrics.
If no unique compatible field is found, it fails loudly and writes no result.

Outputs
-------
<output-root>/
    method_metrics.csv
    per_author_metrics.csv
    pairwise_top1_mcnemar.csv
    rescue_harm.csv
    paired_bootstrap_metrics.csv
    rank_source_resolution.json
    summary.json
    artifact_checksums.json

Safety
------
- Frozen Train-Val SHA256 is verified.
- Frozen Frequency/PV1 prediction SHA256 is verified.
- Row IDs must align exactly across all inputs.
- Gold is used only for already-materialized rank evaluation, never scoring.
- Dev3000 used: false.
- Test used: false.
- Existing non-empty output directories are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
EXPERIMENT = "initial_predev_paired_validation_v1"
EXPECTED_ROWS = 34_416
EXPECTED_VAL_SHA256 = "d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4"
EXPECTED_FREQUENCY_PV1_SHA256 = "7fd8aa158d8cd50bced36b55610f8d932bc65e3aae1dbbd5bd65907ff1707ea7"
DEFAULT_BOOTSTRAP_REPS = 10_000
DEFAULT_SEED = 20_260_821
DEFAULT_BOOTSTRAP_BATCH = 128
AUTO_MATCH_TOLERANCE = 5e-7

PRIMARY = "Joint_B4_CS4_E2"
COMPARATORS = (
    "PV1",
    "Joint_B4_CS4_E0",
    "Selector_K5",
    "Selector_K3",
)
METHOD_ORDER = (PRIMARY, *COMPARATORS)

# These signatures are not used to choose a method or tune a parameter. They
# only identify the already-selected row-level rank vector inside an artifact
# whose nested field names may differ between historical runner versions.
EXPECTED_SIGNATURES: dict[str, dict[str, float]] = {
    "Joint_B4_CS4_E2": {
        "macro_author_top1": 0.404807,
        "micro_top1": 0.429364,
        "top3": 0.614801,
        "top5": 0.685815,
        "mrr_at_10": 0.537433,
        "missing10": 0.243172,
    },
    "Joint_B4_CS4_E0": {
        "macro_author_top1": 0.402628,
        "micro_top1": 0.427272,
        "top3": 0.612593,
        "top5": 0.683461,
        "mrr_at_10": 0.535609,
        "missing10": 0.245613,
    },
    "Selector_K5": {
        "macro_author_top1": 0.403772,
        "micro_top1": 0.428376,
        "top3": 0.603266,
        "top5": 0.677708,
        "mrr_at_10": 0.533908,
        "missing10": 0.243172,
    },
    "Selector_K3": {
        "macro_author_top1": 0.403964,
        "micro_top1": 0.428522,
        "top3": 0.604544,
        "top5": 0.681079,
        "mrr_at_10": 0.534120,
        "missing10": 0.247007,
    },
}


# ---------------------------------------------------------------------------
# IO / provenance
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def index_unique_rows(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if "row_id" not in row:
            raise RuntimeError(f"{label} row lacks row_id")
        row_id = str(row["row_id"])
        if row_id in output:
            raise RuntimeError(
                f"{label} contains duplicate row_id={row_id}. "
                "This validator expects one durable prediction record per Train-Val row."
            )
        output[row_id] = row
    return output


# ---------------------------------------------------------------------------
# Rank extraction / metric signatures
# ---------------------------------------------------------------------------


def flatten_scalar_leaves(value: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    output: dict[tuple[str, ...], Any] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            output.update(flatten_scalar_leaves(child, (*prefix, str(key))))
    elif isinstance(value, (list, tuple)):
        # Candidate lists / Top10 materializations are intentionally ignored.
        pass
    else:
        output[prefix] = value
    return output


def get_path(row: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = row
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(path)
        current = current[key]
    return current


def normalize_rank(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("bool is not a rank")
    if isinstance(value, (int, np.integer)):
        rank = int(value)
    elif isinstance(value, float) and math.isfinite(value) and float(value).is_integer():
        rank = int(value)
    else:
        raise ValueError(f"not an integer rank: {value!r}")
    if not 1 <= rank <= 10:
        raise ValueError(f"rank outside Top10: {rank}")
    return rank


def candidate_scalar_paths(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, ...]]:
    # Union across a deterministic small sample, then validate every candidate
    # over the full artifact. We prefer rank-named paths but also allow other
    # scalar leaves so historical schemas can still be resolved by signature.
    if not rows:
        return []
    sample_indices = sorted(set(np.linspace(0, len(rows) - 1, num=min(64, len(rows)), dtype=int)))
    paths: set[tuple[str, ...]] = set()
    for index in sample_indices:
        for path, value in flatten_scalar_leaves(rows[index]).items():
            if not path or path == ("row_id",):
                continue
            if value is None or (isinstance(value, (int, float, np.integer)) and not isinstance(value, bool)):
                paths.add(path)
    return sorted(paths, key=lambda p: (0 if any("rank" in x.lower() for x in p) else 1, len(p), p))


def ranks_from_path(
    rows_by_id: Mapping[str, Mapping[str, Any]],
    row_ids: Sequence[str],
    path: tuple[str, ...],
) -> list[int | None] | None:
    result: list[int | None] = []
    seen_non_null = False
    seen_missing = False
    for row_id in row_ids:
        row = rows_by_id[row_id]
        try:
            raw = get_path(row, path)
            rank = normalize_rank(raw)
        except (KeyError, ValueError, TypeError):
            return None
        if rank is None:
            seen_missing = True
        else:
            seen_non_null = True
        result.append(rank)
    if not seen_non_null:
        return None
    # A genuine Top10 rank vector in this workload should normally have misses.
    # Do not require it, but the metric signature will disambiguate constants.
    _ = seen_missing
    return result


def metric_summary(authors: Sequence[str], ranks: Sequence[int | None]) -> dict[str, Any]:
    if len(authors) != len(ranks):
        raise RuntimeError("author/rank length mismatch")
    n = len(ranks)
    by_author: dict[str, list[int | None]] = defaultdict(list)
    for author, rank in zip(authors, ranks):
        by_author[str(author)].append(rank)

    per_author_top1 = {
        author: sum(rank == 1 for rank in values) / len(values)
        for author, values in sorted(by_author.items())
    }
    reciprocal = sum(0.0 if rank is None else 1.0 / rank for rank in ranks)
    return {
        "n": n,
        "authors": len(per_author_top1),
        "macro_author_top1": float(np.mean(list(per_author_top1.values()))),
        "micro_top1": sum(rank == 1 for rank in ranks) / n,
        "top3": sum(rank is not None and rank <= 3 for rank in ranks) / n,
        "top5": sum(rank is not None and rank <= 5 for rank in ranks) / n,
        "mrr_at_10": reciprocal / n,
        "missing10": sum(rank is None for rank in ranks) / n,
        "per_author_top1": per_author_top1,
    }


def signature_distance(metrics: Mapping[str, Any], expected: Mapping[str, float]) -> float:
    return max(abs(float(metrics[key]) - float(target)) for key, target in expected.items())


def resolve_rank_path(
    *,
    label: str,
    rows: Sequence[Mapping[str, Any]],
    row_ids: Sequence[str],
    authors: Sequence[str],
    expected: Mapping[str, float],
    tolerance: float,
) -> tuple[tuple[str, ...], list[int | None], dict[str, Any], list[dict[str, Any]]]:
    by_id = index_unique_rows(rows, label)
    if set(by_id) != set(row_ids):
        missing = sorted(set(row_ids) - set(by_id))[:10]
        extra = sorted(set(by_id) - set(row_ids))[:10]
        raise RuntimeError(f"{label} row-id surface mismatch; missing={missing} extra={extra}")

    candidates: list[dict[str, Any]] = []
    for path in candidate_scalar_paths(rows):
        ranks = ranks_from_path(by_id, row_ids, path)
        if ranks is None:
            continue
        metrics = metric_summary(authors, ranks)
        distance = signature_distance(metrics, expected)
        candidates.append(
            {
                "path_tuple": path,
                "path": ".".join(path),
                "distance": distance,
                "metrics": metrics,
                "ranks": ranks,
                "rank_named": any("rank" in key.lower() for key in path),
            }
        )

    candidates.sort(
        key=lambda item: (
            float(item["distance"]),
            0 if item["rank_named"] else 1,
            len(item["path_tuple"]),
            item["path"],
        )
    )
    if not candidates:
        raise RuntimeError(f"No usable scalar Top10 rank fields found in {label}")

    best = candidates[0]
    if float(best["distance"]) > tolerance:
        preview = [
            {
                "path": item["path"],
                "distance": item["distance"],
                "metrics": {key: item["metrics"][key] for key in expected},
            }
            for item in candidates[:12]
        ]
        raise RuntimeError(
            f"Could not resolve {label}: best metric-signature distance "
            f"{best['distance']:.9g} exceeds tolerance {tolerance}.\n"
            f"Top candidate fields:\n{json.dumps(preview, ensure_ascii=False, indent=2)}"
        )

    # If multiple paths are essentially exact and yield different rank vectors,
    # fail rather than silently choosing one. Identical aliases are harmless.
    near = [item for item in candidates if float(item["distance"]) <= tolerance]
    best_ranks = best["ranks"]
    conflicting = [item for item in near[1:] if item["ranks"] != best_ranks]
    if conflicting:
        preview = [best, *conflicting[:10]]
        raise RuntimeError(
            f"Ambiguous rank resolution for {label}: multiple metric-compatible "
            "fields have different row-level ranks.\n"
            + json.dumps(
                [
                    {
                        "path": item["path"],
                        "distance": item["distance"],
                        "metrics": {key: item["metrics"][key] for key in expected},
                    }
                    for item in preview
                ],
                ensure_ascii=False,
                indent=2,
            )
        )

    aliases = [
        {
            "path": item["path"],
            "distance": item["distance"],
        }
        for item in near
        if item["ranks"] == best_ranks
    ]
    return best["path_tuple"], best_ranks, best["metrics"], aliases


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def ranks_to_arrays(ranks: Sequence[int | None]) -> dict[str, np.ndarray]:
    numeric = np.array([0 if rank is None else int(rank) for rank in ranks], dtype=np.int16)
    present = numeric > 0
    return {
        "top1": (numeric == 1).astype(np.float64),
        "top3": ((numeric >= 1) & (numeric <= 3)).astype(np.float64),
        "top5": ((numeric >= 1) & (numeric <= 5)).astype(np.float64),
        "mrr_at_10": np.where(present, 1.0 / np.maximum(numeric, 1), 0.0).astype(np.float64),
        "missing10": (~present).astype(np.float64),
    }


def exact_mcnemar_two_sided(b: int, c: int) -> float:
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    m = min(int(b), int(c))
    log_two = math.log(2.0)
    logs = [
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * log_two
        for k in range(m + 1)
    ]
    anchor = max(logs)
    tail = math.exp(anchor) * sum(math.exp(value - anchor) for value in logs)
    return min(1.0, 2.0 * tail)


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    low, high = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear")
    return float(low), float(high)


def bootstrap_two_sided_p(values: np.ndarray) -> float:
    if values.size == 0:
        return 1.0
    lower = (np.sum(values <= 0.0) + 1.0) / (values.size + 1.0)
    upper = (np.sum(values >= 0.0) + 1.0) / (values.size + 1.0)
    return float(min(1.0, 2.0 * min(lower, upper)))


def paired_author_stratified_bootstrap(
    *,
    authors: Sequence[str],
    primary_ranks: Sequence[int | None],
    comparator_ranks: Sequence[int | None],
    reps: int,
    seed: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    if reps <= 0:
        raise ValueError("bootstrap reps must be positive")
    if batch_size <= 0:
        raise ValueError("bootstrap batch size must be positive")

    primary = ranks_to_arrays(primary_ranks)
    comparator = ranks_to_arrays(comparator_ranks)
    differences = {key: primary[key] - comparator[key] for key in primary}
    author_values = sorted(set(str(author) for author in authors))
    author_indices = {
        author: np.array([i for i, value in enumerate(authors) if str(value) == author], dtype=np.int64)
        for author in author_values
    }
    if any(index.size == 0 for index in author_indices.values()):
        raise RuntimeError("empty author stratum")

    rng = np.random.default_rng(seed)
    outputs = {
        "macro_author_top1": np.empty(reps, dtype=np.float64),
        "micro_top1": np.empty(reps, dtype=np.float64),
        "top3": np.empty(reps, dtype=np.float64),
        "top5": np.empty(reps, dtype=np.float64),
        "mrr_at_10": np.empty(reps, dtype=np.float64),
        "missing10": np.empty(reps, dtype=np.float64),
    }
    total_n = len(authors)

    offset = 0
    while offset < reps:
        width = min(batch_size, reps - offset)
        metric_sums = {
            key: np.zeros(width, dtype=np.float64)
            for key in ("top1", "top3", "top5", "mrr_at_10", "missing10")
        }
        macro_top1 = np.zeros(width, dtype=np.float64)

        for author in author_values:
            idx = author_indices[author]
            draws = rng.integers(0, idx.size, size=(width, idx.size), endpoint=False)
            sampled_global = idx[draws]

            author_top1_delta = differences["top1"][sampled_global].mean(axis=1)
            macro_top1 += author_top1_delta

            for key in metric_sums:
                metric_sums[key] += differences[key][sampled_global].sum(axis=1)

        outputs["macro_author_top1"][offset : offset + width] = macro_top1 / len(author_values)
        outputs["micro_top1"][offset : offset + width] = metric_sums["top1"] / total_n
        outputs["top3"][offset : offset + width] = metric_sums["top3"] / total_n
        outputs["top5"][offset : offset + width] = metric_sums["top5"] / total_n
        outputs["mrr_at_10"][offset : offset + width] = metric_sums["mrr_at_10"] / total_n
        outputs["missing10"][offset : offset + width] = metric_sums["missing10"] / total_n
        offset += width

    return outputs


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def per_author_metrics(authors: Sequence[str], ranks: Sequence[int | None]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[int | None]] = defaultdict(list)
    for author, rank in zip(authors, ranks):
        grouped[str(author)].append(rank)
    result: dict[str, dict[str, float | int]] = {}
    for author, values in sorted(grouped.items()):
        n = len(values)
        result[author] = {
            "n": n,
            "top1": sum(rank == 1 for rank in values) / n,
            "top3": sum(rank is not None and rank <= 3 for rank in values) / n,
            "top5": sum(rank is not None and rank <= 5 for rank in values) / n,
            "mrr_at_10": sum(0.0 if rank is None else 1.0 / rank for rank in values) / n,
            "missing10": sum(rank is None for rank in values) / n,
        }
    return result


def transition_counts(base: Sequence[int | None], new: Sequence[int | None]) -> dict[str, int]:
    if len(base) != len(new):
        raise RuntimeError("transition rank length mismatch")
    counts = {
        "n": len(base),
        "rescue": 0,
        "harm": 0,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
    }
    for before_rank, after_rank in zip(base, new):
        before = before_rank == 1
        after = after_rank == 1
        if not before and after:
            counts["rescue"] += 1
        elif before and not after:
            counts["harm"] += 1
        elif before and after:
            counts["unchanged_correct"] += 1
        else:
            counts["unchanged_wrong"] += 1
    counts["net"] = counts["rescue"] - counts["harm"]
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--frequency-pv1-predictions", type=Path, required=True)
    parser.add_argument(
        "--joint-predictions",
        type=Path,
        required=True,
        help="ngram_cs_entropy_two_anchors_v1/predictions.jsonl",
    )
    parser.add_argument(
        "--selector-predictions",
        type=Path,
        required=True,
        help="pv1_ngram_selector_k135_v1/predictions.jsonl",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-batch", type=int, default=DEFAULT_BOOTSTRAP_BATCH)
    parser.add_argument("--match-tolerance", type=float, default=AUTO_MATCH_TOLERANCE)
    parser.add_argument(
        "--allow-unfrozen-input-hashes",
        action="store_true",
        help=(
            "Only bypasses the frozen SHA checks for Train-Val/Frequency-PV1. "
            "Not recommended for the canonical run."
        ),
    )
    args = parser.parse_args()

    for path in (
        args.val,
        args.frequency_pv1_predictions,
        args.joint_predictions,
        args.selector_predictions,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    provenance = {
        "val": str(args.val.resolve()),
        "val_sha256": sha256_file(args.val),
        "frequency_pv1_predictions": str(args.frequency_pv1_predictions.resolve()),
        "frequency_pv1_predictions_sha256": sha256_file(args.frequency_pv1_predictions),
        "joint_predictions": str(args.joint_predictions.resolve()),
        "joint_predictions_sha256": sha256_file(args.joint_predictions),
        "selector_predictions": str(args.selector_predictions.resolve()),
        "selector_predictions_sha256": sha256_file(args.selector_predictions),
    }
    if not args.allow_unfrozen_input_hashes:
        if provenance["val_sha256"] != EXPECTED_VAL_SHA256:
            raise RuntimeError(
                "Frozen Train-Val SHA mismatch:\n"
                f"expected={EXPECTED_VAL_SHA256}\nactual={provenance['val_sha256']}"
            )
        if provenance["frequency_pv1_predictions_sha256"] != EXPECTED_FREQUENCY_PV1_SHA256:
            raise RuntimeError(
                "Frozen Frequency/PV1 prediction SHA mismatch:\n"
                f"expected={EXPECTED_FREQUENCY_PV1_SHA256}\n"
                f"actual={provenance['frequency_pv1_predictions_sha256']}"
            )

    val_rows = read_jsonl(args.val)
    freq_rows = read_jsonl(args.frequency_pv1_predictions)
    joint_rows = read_jsonl(args.joint_predictions)
    selector_rows = read_jsonl(args.selector_predictions)

    if len(val_rows) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} Train-Val rows, got {len(val_rows)}")

    val = index_unique_rows(val_rows, "Train-Val")
    freq = index_unique_rows(freq_rows, "Frequency/PV1")
    if set(val) != set(freq):
        raise RuntimeError("Train-Val / Frequency-PV1 row IDs differ")

    # Preserve frozen Train-Val row order for all paired calculations.
    row_ids = [str(row["row_id"]) for row in val_rows]
    authors = [str(row["author"]) for row in val_rows]
    if len(set(row_ids)) != EXPECTED_ROWS:
        raise RuntimeError("Duplicate Train-Val row IDs")
    unique_authors = sorted(set(authors))

    # PV1 is a frozen explicit field, not auto-resolved.
    pv1_ranks = [normalize_rank(freq[row_id].get("pv1_rank")) for row_id in row_ids]
    pv1_metrics = metric_summary(authors, pv1_ranks)

    print("=== INITIAL PRE-DEV PAIRED VALIDATION ===")
    print(f"Train-Val rows: {len(row_ids)}")
    print(f"Authors: {unique_authors}")
    print(f"Bootstrap reps: {args.bootstrap_reps}")
    print(f"Bootstrap seed: {args.seed}")
    print("Primary: B + 4 P_NG + 4 CS + 2 C_E")
    print("Comparators: PV1, B+4P+4CS, Selector K5, Selector K3")
    print("Gold used for scoring/features: false")
    print("Gold/ranks used for Train-Val evaluation only: true")
    print("Dev3000 used: false")
    print("Test used: false")
    print()

    # Auto-resolve the row-level vectors from already-materialized prediction files.
    resolved: dict[str, dict[str, Any]] = {}
    method_ranks: dict[str, list[int | None]] = {"PV1": pv1_ranks}
    method_metrics: dict[str, dict[str, Any]] = {"PV1": pv1_metrics}

    source_jobs = (
        (PRIMARY, joint_rows, EXPECTED_SIGNATURES[PRIMARY], "joint_predictions"),
        ("Joint_B4_CS4_E0", joint_rows, EXPECTED_SIGNATURES["Joint_B4_CS4_E0"], "joint_predictions"),
        ("Selector_K5", selector_rows, EXPECTED_SIGNATURES["Selector_K5"], "selector_predictions"),
        ("Selector_K3", selector_rows, EXPECTED_SIGNATURES["Selector_K3"], "selector_predictions"),
    )

    for method, rows, signature, source_name in source_jobs:
        path, ranks, metrics, aliases = resolve_rank_path(
            label=method,
            rows=rows,
            row_ids=row_ids,
            authors=authors,
            expected=signature,
            tolerance=args.match_tolerance,
        )
        method_ranks[method] = ranks
        method_metrics[method] = metrics
        resolved[method] = {
            "source": source_name,
            "resolved_rank_path": ".".join(path),
            "metric_signature_distance": signature_distance(metrics, signature),
            "expected_signature": signature,
            "observed_signature": {key: metrics[key] for key in signature},
            "compatible_aliases": aliases,
        }
        print(
            f"Resolved {method:>20}: {'.'.join(path)}  "
            f"max_abs_metric_error={resolved[method]['metric_signature_distance']:.3g}"
        )

    # Verify PV1 headline values too, but do not use them for field resolution.
    expected_pv1 = {
        "macro_author_top1": 0.401872474944,
        "micro_top1": 0.426749186425,
        "top3": 0.598907484891,
        "top5": 0.663092747559,
        "mrr_at_10": 0.524449799097,
        "missing10": 0.291143654114,
    }
    pv1_error = signature_distance(pv1_metrics, expected_pv1)
    if pv1_error > args.match_tolerance:
        raise RuntimeError(
            f"PV1 metric regression failed: max abs error={pv1_error}; metrics={pv1_metrics}"
        )
    resolved["PV1"] = {
        "source": "frequency_pv1_predictions",
        "resolved_rank_path": "pv1_rank",
        "metric_signature_distance": pv1_error,
        "expected_signature": expected_pv1,
        "observed_signature": {key: pv1_metrics[key] for key in expected_pv1},
        "compatible_aliases": [{"path": "pv1_rank", "distance": pv1_error}],
    }

    resolution_path = args.output_root / "rank_source_resolution.json"
    write_json(
        resolution_path,
        {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "status": "complete",
            "match_tolerance": args.match_tolerance,
            "methods": resolved,
        },
    )

    # Method headline metrics.
    method_metric_rows: list[dict[str, Any]] = []
    per_author_rows: list[dict[str, Any]] = []
    per_author_by_method: dict[str, dict[str, dict[str, float | int]]] = {}
    for method in METHOD_ORDER:
        metrics = method_metrics[method]
        method_metric_rows.append(
            {
                "method": method,
                "n": metrics["n"],
                "authors": metrics["authors"],
                "macro_author_top1": metrics["macro_author_top1"],
                "micro_top1": metrics["micro_top1"],
                "top3": metrics["top3"],
                "top5": metrics["top5"],
                "mrr_at_10": metrics["mrr_at_10"],
                "missing10": metrics["missing10"],
            }
        )
        author_metrics = per_author_metrics(authors, method_ranks[method])
        per_author_by_method[method] = author_metrics
        for author, values in author_metrics.items():
            per_author_rows.append({"method": method, "author": author, **values})

    write_csv(
        args.output_root / "method_metrics.csv",
        (
            "method",
            "n",
            "authors",
            "macro_author_top1",
            "micro_top1",
            "top3",
            "top5",
            "mrr_at_10",
            "missing10",
        ),
        method_metric_rows,
    )
    write_csv(
        args.output_root / "per_author_metrics.csv",
        ("method", "author", "n", "top1", "top3", "top5", "mrr_at_10", "missing10"),
        per_author_rows,
    )

    # Pairwise Top1 transitions + McNemar.
    mcnemar_rows: list[dict[str, Any]] = []
    rescue_rows: list[dict[str, Any]] = []
    for comparator in COMPARATORS:
        transitions = transition_counts(method_ranks[comparator], method_ranks[PRIMARY])
        b = transitions["rescue"]  # primary correct, comparator wrong
        c = transitions["harm"]    # comparator correct, primary wrong
        p_value = exact_mcnemar_two_sided(b, c)
        mcnemar_rows.append(
            {
                "primary": PRIMARY,
                "comparator": comparator,
                "n": EXPECTED_ROWS,
                "primary_correct_comparator_wrong": b,
                "primary_wrong_comparator_correct": c,
                "discordant_n": b + c,
                "exact_two_sided_p": p_value,
            }
        )
        rescue_rows.append(
            {
                "primary": PRIMARY,
                "comparator": comparator,
                **transitions,
            }
        )

    write_csv(
        args.output_root / "pairwise_top1_mcnemar.csv",
        (
            "primary",
            "comparator",
            "n",
            "primary_correct_comparator_wrong",
            "primary_wrong_comparator_correct",
            "discordant_n",
            "exact_two_sided_p",
        ),
        mcnemar_rows,
    )
    write_csv(
        args.output_root / "rescue_harm.csv",
        (
            "primary",
            "comparator",
            "n",
            "rescue",
            "harm",
            "net",
            "unchanged_correct",
            "unchanged_wrong",
        ),
        rescue_rows,
    )

    # Paired author-stratified bootstrap.
    bootstrap_rows: list[dict[str, Any]] = []
    metric_directions = {
        "macro_author_top1": "higher_better",
        "micro_top1": "higher_better",
        "top3": "higher_better",
        "top5": "higher_better",
        "mrr_at_10": "higher_better",
        "missing10": "lower_better",
    }
    for pair_index, comparator in enumerate(COMPARATORS):
        print(f"Bootstrap {PRIMARY} vs {comparator} ...", flush=True)
        distributions = paired_author_stratified_bootstrap(
            authors=authors,
            primary_ranks=method_ranks[PRIMARY],
            comparator_ranks=method_ranks[comparator],
            reps=args.bootstrap_reps,
            seed=args.seed + pair_index,
            batch_size=args.bootstrap_batch,
        )
        for metric, values in distributions.items():
            observed = float(method_metrics[PRIMARY][metric]) - float(method_metrics[comparator][metric])
            ci_low, ci_high = percentile_ci(values)
            p_value = bootstrap_two_sided_p(values)
            direction = metric_directions[metric]
            if direction == "higher_better":
                favorable = observed > 0
            else:
                favorable = observed < 0
            excludes_zero = bool(ci_low > 0 or ci_high < 0)
            bootstrap_rows.append(
                {
                    "primary": PRIMARY,
                    "comparator": comparator,
                    "metric": metric,
                    "direction": direction,
                    "observed_delta_primary_minus_comparator": observed,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "bootstrap_two_sided_p": p_value,
                    "ci95_excludes_zero": excludes_zero,
                    "observed_delta_favors_primary": favorable,
                    "bootstrap_reps": args.bootstrap_reps,
                    "seed": args.seed + pair_index,
                    "resampling": "paired_rows_within_author; author counts fixed",
                }
            )

    write_csv(
        args.output_root / "paired_bootstrap_metrics.csv",
        (
            "primary",
            "comparator",
            "metric",
            "direction",
            "observed_delta_primary_minus_comparator",
            "ci95_low",
            "ci95_high",
            "bootstrap_two_sided_p",
            "ci95_excludes_zero",
            "observed_delta_favors_primary",
            "bootstrap_reps",
            "seed",
            "resampling",
        ),
        bootstrap_rows,
    )

    # Author consistency of primary-vs-comparator Top1 deltas.
    author_consistency: dict[str, Any] = {}
    for comparator in COMPARATORS:
        deltas = {
            author: float(per_author_by_method[PRIMARY][author]["top1"])
            - float(per_author_by_method[comparator][author]["top1"])
            for author in unique_authors
        }
        author_consistency[comparator] = {
            "per_author_top1_delta_primary_minus_comparator": deltas,
            "authors_primary_better": sum(value > 0 for value in deltas.values()),
            "authors_tied": sum(value == 0 for value in deltas.values()),
            "authors_primary_worse": sum(value < 0 for value in deltas.values()),
        }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "experiment": EXPERIMENT,
        "population": {
            "partition": "Train-Val",
            "rows": EXPECTED_ROWS,
            "authors": unique_authors,
            "primary_metric": "macro_author_top1",
        },
        "primary_method": {
            "name": PRIMARY,
            "formula": "B + 4*P_NG + 4*CS + 2*C_E",
            "role": "pre-declared balanced development-best",
        },
        "comparators": {
            "PV1": "historical personal-recovery baseline",
            "Joint_B4_CS4_E0": "Entropy ablation: same balanced joint scorer with lambda_E=0",
            "Selector_K5": "same-K5-breadth architectural comparator",
            "Selector_K3": "best balanced selector-family operating point",
        },
        "method_metrics": {row["method"]: row for row in method_metric_rows},
        "mcnemar_top1": {row["comparator"]: row for row in mcnemar_rows},
        "rescue_harm": {row["comparator"]: row for row in rescue_rows},
        "paired_bootstrap": {
            comparator: [row for row in bootstrap_rows if row["comparator"] == comparator]
            for comparator in COMPARATORS
        },
        "author_consistency": author_consistency,
        "statistics_contract": {
            "mcnemar": "exact two-sided on discordant Top1 outcomes",
            "bootstrap": "paired row resampling independently within each author",
            "macro_author_top1": "equal mean of resampled per-author Top1 values",
            "ci": "95% percentile bootstrap CI of primary-minus-comparator delta",
            "bootstrap_p": "two-sided sign-tail bootstrap p-value with +1 correction",
            "bootstrap_reps": args.bootstrap_reps,
            "seed_base": args.seed,
        },
        "rank_source_resolution": resolved,
        "provenance": provenance,
        "research_boundaries": {
            "parameter_tuning_in_this_runner": False,
            "model_inference_in_this_runner": False,
            "candidate_construction_in_this_runner": False,
            "gold_used_for_scoring_or_features": False,
            "gold_or_gold_derived_ranks_used_for_train_val_evaluation_only": True,
            "dev3000_used": False,
            "test_used": False,
        },
    }
    summary_path = args.output_root / "summary.json"
    write_json(summary_path, summary)

    output_files = [
        args.output_root / "method_metrics.csv",
        args.output_root / "per_author_metrics.csv",
        args.output_root / "pairwise_top1_mcnemar.csv",
        args.output_root / "rescue_harm.csv",
        args.output_root / "paired_bootstrap_metrics.csv",
        resolution_path,
        summary_path,
    ]
    checksums = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in output_files
    }
    write_json(args.output_root / "artifact_checksums.json", checksums)

    print()
    print("=== COMPLETE ===")
    for method in METHOD_ORDER:
        metrics = method_metrics[method]
        print(
            f"{method:>20}  Macro={metrics['macro_author_top1']:.6f}  "
            f"Micro={metrics['micro_top1']:.6f}  Top3={metrics['top3']:.6f}  "
            f"Top5={metrics['top5']:.6f}  MRR={metrics['mrr_at_10']:.6f}  "
            f"Missing={metrics['missing10']:.6f}"
        )
    print()
    for row in mcnemar_rows:
        print(
            f"McNemar {PRIMARY} vs {row['comparator']}: "
            f"rescue={row['primary_correct_comparator_wrong']} "
            f"harm={row['primary_wrong_comparator_correct']} "
            f"p={row['exact_two_sided_p']:.6g}"
        )
    print()
    print(f"Outputs: {args.output_root.resolve()}")
    print("Dev3000 used: false")
    print("Test used: false")


if __name__ == "__main__":
    main()
