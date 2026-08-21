"""PV1 reproduction with an Interpolated-NGram K5 selector ablation.

Research question
-----------------
If the historical Initial-Pinyin PV1 recovery mechanism is kept fixed, does
replacing its frequency-ordered Personal-K5 Top1 selector with the frozen
Interpolated-NGram K5 Top1 selector improve end-to-end ranking?

This is intentionally a *selector-only* ablation.

Frozen PV1 semantics reproduced here
------------------------------------
- Frozen Personal K5 surface, frequency ordered.
- Generic side is the frozen Frequency-F ranking.
- PV1 uses Kpv=1 and lambda_pv=4.
- Personal frequency support is exactly:

      raw(c) = log(1 + count(c))
      support(c) = raw(c) / max_{j in Personal K5} raw(j)

- Personal recovery score is exactly:

      boundary + lambda_pv * support(c)

  where boundary is the minimum normalized Generic score.
- Generic wins exact score ties; final list is truncated to Top10.

Two methods are materialized:

1) PV1-Reproduced
      selector = Personal-K5 frequency Top1 (original order)

2) PV1+NGramSelector
      selector = frozen K5 Interpolated-NGram Top1
      recovery support / lambda / boundary / Generic-F merge are unchanged.

Before reporting the ablation, the script requires PV1-Reproduced to match the
frozen PV1 artifact exactly on every row's Top10 candidate ordering and Gold
rank. This prevents an accidental change to the PV1 mechanism from being
mistaken for a selector effect.

Protocol
--------
- Train-Val only (34,416 rows).
- Gold is used only for evaluation/diagnostics, never selection/scoring.
- Dev3000 is not read. Test is not read.
- No PinyinGPT/BGE/NGram inference is performed; frozen adaptive scores are reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_VAL_SHA256 = "d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4"
EXPECTED_SURFACE_SHA256 = "205c0ba01cd0678d7a4341c503fa2e74cf126a70182ff582687025e4946764b2"
EXPECTED_FREQUENCY_PV1_SHA256 = "7fd8aa158d8cd50bced36b55610f8d932bc65e3aae1dbbd5bd65907ff1707ea7"
EXPECTED_VAL_ROWS = 34_416
EXPECTED_GENERIC_MISSING = 12_565
EXPECTED_K5_RECOVERABLE_MISSING = 4_910

FROZEN_LAMBDA_FREQUENCY = 4.0
FROZEN_K_PV = 1
FROZEN_LAMBDA_PV = 4.0


# ---------------------------------------------------------------------------
# IO / utility
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


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


def index_rows(rows: Iterable[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for value in rows:
        row = dict(value)
        row_id = str(row["row_id"])
        if row_id in output:
            raise RuntimeError(f"Duplicate row_id in {label}: {row_id}")
        output[row_id] = row
    return output


def safe_rate(n: float | int, d: int) -> float:
    return float(n) / d if d else 0.0


def candidate_text(row: Mapping[str, Any]) -> str:
    return str(row.get("candidate", row.get("text", "")))


def rank_of(rows: Sequence[Mapping[str, Any]], gold: str) -> int | None:
    for rank, row in enumerate(rows, start=1):
        if candidate_text(row) == gold:
            return rank
    return None


# ---------------------------------------------------------------------------
# Frozen surfaces / supports
# ---------------------------------------------------------------------------


def extract_personal_records(surface: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = surface.get("personal_candidates_top5", [])
    if not isinstance(raw, list):
        raise RuntimeError(f"Invalid personal_candidates_top5 at {surface.get('row_id')}")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, value in enumerate(raw, start=1):
        if not isinstance(value, Mapping):
            raise RuntimeError(f"Invalid Personal-K5 record at {surface.get('row_id')}: {value!r}")
        text = str(value.get("text", ""))
        if not text:
            raise RuntimeError(f"Empty Personal-K5 target at {surface.get('row_id')}")
        if text in seen:
            raise RuntimeError(f"Duplicate Personal-K5 target at {surface.get('row_id')}: {text}")
        seen.add(text)
        count = int(value.get("frequency_count", 0))
        if count <= 0:
            raise RuntimeError(f"Non-positive frozen frequency_count at {surface.get('row_id')}: {text}")
        recorded_rank = int(value.get("personal_rank", position))
        if recorded_rank != position:
            raise RuntimeError(
                f"Frozen Personal-K5 rank mismatch at {surface.get('row_id')}: "
                f"target={text} expected={position} got={recorded_rank}"
            )
        records.append({"text": text, "frequency_count": count, "frequency_rank": position})

    texts = [record["text"] for record in records]
    frozen_texts = [str(value) for value in surface.get("personal_candidate_texts_top5", [])]
    if texts != frozen_texts:
        raise RuntimeError(
            f"Personal-K5 record/text fields differ at {surface.get('row_id')}: "
            f"records={texts} texts={frozen_texts}"
        )
    return tuple(records)


def pv1_frequency_support(personal_records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Exact context_memory.frequency_support semantics on frozen Personal K5 counts."""
    raw = {
        str(record["text"]): math.log1p(int(record["frequency_count"]))
        for record in personal_records
    }
    maximum = max(raw.values(), default=0.0)
    return {text: value / maximum for text, value in raw.items()} if maximum else {}


def ngram_rank_order(candidates: Sequence[str], support: Sequence[float]) -> list[int]:
    """Exact adaptive-runner rank_candidates tie policy: support, original index, text."""
    if len(candidates) != len(support):
        raise RuntimeError("NGram candidate/support length mismatch")
    return sorted(
        range(len(candidates)),
        key=lambda i: (-float(support[i]), i, str(candidates[i])),
    )


# ---------------------------------------------------------------------------
# Exact PV1-style merge
# ---------------------------------------------------------------------------


def pv1_style_merge(
    *,
    frequency_candidates: Sequence[Mapping[str, Any]],
    selected_personal_target: str | None,
    selected_frequency_support: float,
    selected_frequency_count: int | None,
    selected_original_frequency_rank: int | None,
    selector_label: str,
) -> list[dict[str, Any]]:
    """Reproduce personal_vocabulary._merge for k_pv=1, lambda_ctx=0."""

    rows: list[dict[str, Any]] = []
    generic_texts: set[str] = set()
    normalized_generic: list[float] = []

    for position, frozen in enumerate(frequency_candidates, start=1):
        target = candidate_text(frozen)
        if not target:
            raise RuntimeError("Cannot identify frozen Frequency candidate")
        if target in generic_texts:
            raise RuntimeError(f"Duplicate frozen Generic candidate: {target!r}")
        generic_texts.add(target)

        normalized = float(frozen["normalized_generic_score"])
        normalized_generic.append(normalized)

        # Keep the fields that matter scientifically, while preserving frozen F score.
        row = dict(frozen)
        row["candidate"] = target
        row["source"] = "generic"
        row["frequency_support"] = float(frozen.get("personal_score", 0.0))
        row["context_support"] = 0.0
        row["final_score"] = float(frozen["final_score"])
        row["generic_rank"] = int(frozen.get("generic_rank") or position)
        rows.append(row)

    if not rows:
        raise RuntimeError("Frozen Frequency candidate list is empty")

    boundary = min(normalized_generic)

    if selected_personal_target is not None:
        if selected_personal_target in generic_texts:
            raise RuntimeError(
                f"Selected Personal candidate overlaps frozen Generic: {selected_personal_target!r}"
            )
        rows.append(
            {
                "candidate": selected_personal_target,
                "generic_rank": None,
                "generic_score": None,
                "normalized_generic_score": boundary,
                "personal_score": float(selected_frequency_support),
                "frequency_support": float(selected_frequency_support),
                "context_support": 0.0,
                "frequency_count": int(selected_frequency_count or 0),
                "final_score": boundary + FROZEN_LAMBDA_PV * float(selected_frequency_support),
                "source": "personal_vocabulary",
                # Original _merge uses personal_candidate_rank only as a tie-break.
                # k_pv=1 means this is always rank 1 even when selection came from NGram.
                "personal_candidate_rank": 1,
                "selector": selector_label,
                "original_personal_frequency_rank": selected_original_frequency_rank,
            }
        )

    # Exact personal_vocabulary._merge order for k_pv=1:
    # final_score desc; Generic wins ties; then generic/personal rank; lexical final tie.
    rows.sort(
        key=lambda row: (
            -float(row["final_score"]),
            0 if row["source"] == "generic" else 1,
            int(row.get("generic_rank") or row.get("personal_candidate_rank", 0)),
            str(row["candidate"]),
        )
    )
    values = rows[:10]
    for rank, row in enumerate(values, start=1):
        row["rank"] = rank
    return values


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metric_summary(rows: Sequence[Mapping[str, Any]], rank_key: str, method: str) -> dict[str, Any]:
    author_total: Counter[str] = Counter()
    author_top1: Counter[str] = Counter()
    top1 = top3 = top5 = missing = 0
    reciprocal = 0.0
    observed_ranks: list[int] = []

    for row in rows:
        author = str(row["author"])
        author_total[author] += 1
        rank = row.get(rank_key)
        if rank is None:
            missing += 1
            continue
        rank_i = int(rank)
        observed_ranks.append(rank_i)
        reciprocal += 1.0 / rank_i
        if rank_i == 1:
            top1 += 1
            author_top1[author] += 1
        if rank_i <= 3:
            top3 += 1
        if rank_i <= 5:
            top5 += 1

    n = len(rows)
    per_author = {
        author: safe_rate(author_top1[author], count)
        for author, count in sorted(author_total.items())
    }
    return {
        "method": method,
        "n": n,
        "authors": len(per_author),
        "macro_author_top1": statistics.fmean(per_author.values()) if per_author else 0.0,
        "micro_top1": safe_rate(top1, n),
        "top3": safe_rate(top3, n),
        "top5": safe_rate(top5, n),
        "mrr_at_10": safe_rate(reciprocal, n),
        "missing10": safe_rate(missing, n),
        "mean_rank_given_top10": statistics.fmean(observed_ranks) if observed_ranks else None,
        "per_author_top1": per_author,
    }


def transition_counts(
    rows: Sequence[Mapping[str, Any]],
    base_key: str,
    new_key: str,
) -> dict[str, int]:
    counts = {
        "n": 0,
        "rescue": 0,
        "harm": 0,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
    }
    for row in rows:
        counts["n"] += 1
        before = row.get(base_key) == 1
        after = row.get(new_key) == 1
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


def recovery_summary(rows: Sequence[Mapping[str, Any]], rank_key: str) -> dict[str, Any]:
    generic_missing = [row for row in rows if bool(row["generic_missing"])]
    recoverable = [row for row in generic_missing if bool(row["gold_in_personal_k5"])]
    ranks = [None if row.get(rank_key) is None else int(row[rank_key]) for row in recoverable]

    def count_at(limit: int) -> int:
        return sum(rank is not None and rank <= limit for rank in ranks)

    recovered_ranks = [rank for rank in ranks if rank is not None and rank <= 10]
    reciprocal = sum(0.0 if rank is None else 1.0 / rank for rank in ranks)
    denom = len(recoverable)

    return {
        "generic_missing_n": len(generic_missing),
        "recoverable_n": denom,
        "recovered_at_1_n": count_at(1),
        "recovered_at_3_n": count_at(3),
        "recovered_at_5_n": count_at(5),
        "recovered_at_10_n": count_at(10),
        "recovered_at_1": safe_rate(count_at(1), denom),
        "recovered_at_3": safe_rate(count_at(3), denom),
        "recovered_at_5": safe_rate(count_at(5), denom),
        "recovered_at_10": safe_rate(count_at(10), denom),
        "recovery_mrr_at_10": safe_rate(reciprocal, denom),
        "mean_recovered_rank": statistics.fmean(recovered_ranks) if recovered_ranks else None,
    }


def delta_metrics(method: Mapping[str, Any], pv1: Mapping[str, Any]) -> dict[str, float]:
    keys = ("macro_author_top1", "micro_top1", "top3", "top5", "mrr_at_10", "missing10")
    return {key: float(method[key]) - float(pv1[key]) for key in keys}


# ---------------------------------------------------------------------------
# Frozen input verification
# ---------------------------------------------------------------------------


def verify_inputs(args: argparse.Namespace) -> dict[str, Any]:
    frozen = {
        "val": (args.val, EXPECTED_VAL_SHA256),
        "candidate_surface": (args.candidate_surface, EXPECTED_SURFACE_SHA256),
        "frequency_pv1_predictions": (args.frequency_pv1_predictions, EXPECTED_FREQUENCY_PV1_SHA256),
    }
    hashes: dict[str, str] = {}
    for label, (path, expected) in frozen.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        hashes[label] = actual
        if actual != expected:
            raise RuntimeError(
                f"SHA mismatch for {label}: expected={expected} actual={actual} path={path}"
            )

    scores = args.adaptive_dir / "scores.jsonl"
    comparison_path = args.adaptive_dir / "comparison.json"
    if not scores.is_file():
        raise FileNotFoundError(scores)
    if not comparison_path.is_file():
        raise FileNotFoundError(comparison_path)

    comparison = read_json(comparison_path)
    provenance = comparison.get("provenance", {})
    if provenance.get("val_sha256") != EXPECTED_VAL_SHA256:
        raise RuntimeError("Adaptive NGram comparison uses another Train-Val artifact")
    if provenance.get("frozen_candidate_surface_sha256") != EXPECTED_SURFACE_SHA256:
        raise RuntimeError("Adaptive NGram comparison uses another Personal-K5 surface")
    if provenance.get("candidate_selection_used_gold") is not False:
        raise RuntimeError("Adaptive NGram candidate selection is not Gold-free")
    if provenance.get("gold_used_for_scoring") is not False:
        raise RuntimeError("Adaptive NGram scoring is not Gold-free")
    if provenance.get("dev3000_used") is not False or provenance.get("test_used") is not False:
        raise RuntimeError("Adaptive NGram artifact crossed Dev3000/Test boundary")

    selected_by_k = comparison.get("selected_by_k", {})
    selected_key = str(selected_by_k.get("K5", {}).get("best_interpolated", ""))
    if not selected_key.startswith("K5|Interpolated|"):
        raise RuntimeError(f"Unexpected selected K5 Interpolated key: {selected_key!r}")

    actual_scores_sha = sha256_file(scores)
    recorded_scores_sha = provenance.get("scores_sha256")
    if recorded_scores_sha and recorded_scores_sha != actual_scores_sha:
        raise RuntimeError("Adaptive scores.jsonl SHA differs from comparison provenance")

    return {
        "paths": {
            "val": str(args.val),
            "candidate_surface": str(args.candidate_surface),
            "frequency_pv1_predictions": str(args.frequency_pv1_predictions),
            "adaptive_scores": str(scores),
            "adaptive_comparison": str(comparison_path),
        },
        "sha256": {
            **hashes,
            "adaptive_scores": actual_scores_sha,
            "adaptive_comparison": sha256_file(comparison_path),
        },
        "selected_k5_interpolated_key": selected_key,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact Initial PV1 reproduction with frozen K5 Interpolated-NGram Top1 selector ablation"
    )
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--candidate-surface", type=Path, required=True)
    parser.add_argument("--frequency-pv1-predictions", type=Path, required=True)
    parser.add_argument(
        "--adaptive-dir",
        type=Path,
        required=True,
        help="adaptive_ngram_top10_v1 directory containing scores.jsonl and comparison.json",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    provenance = verify_inputs(args)
    selected_ngram_key = str(provenance["selected_k5_interpolated_key"])

    print("=== PV1 NGRAM-SELECTOR K5 ABLATION ===")
    print("Frozen PV1: Kpv=1, lambda_F=4, lambda_PV=4")
    print(f"Frozen K5 Interpolated method: {selected_ngram_key}")
    print("Only changed component: Personal candidate selector")
    print("PV1 selector: frequency Top1")
    print("Ablation selector: Interpolated-NGram K5 Top1")
    print("Gold used for scoring/selection: false")
    print("Dev3000 used: false")
    print("Test used: false")
    print()

    val_rows = read_jsonl(args.val)
    surface_by_id = index_rows(read_jsonl(args.candidate_surface), "candidate surface")
    pred_by_id = index_rows(read_jsonl(args.frequency_pv1_predictions), "Frequency/PV1 predictions")
    ngram_by_id = index_rows(read_jsonl(args.adaptive_dir / "scores.jsonl"), "adaptive NGram scores")
    val_by_id = index_rows(val_rows, "Train-Val")

    if len(val_rows) != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"Unexpected Train-Val rows: {len(val_rows)}")
    if not (set(val_by_id) == set(surface_by_id) == set(pred_by_id) == set(ngram_by_id)):
        raise RuntimeError("Train-Val / surface / PV1 / NGram row IDs differ")

    outputs: list[dict[str, Any]] = []
    reproduction_mismatch_examples: list[dict[str, Any]] = []
    reproduced_rank_mismatch_n = 0
    reproduced_top10_mismatch_n = 0

    selector_rows_with_candidates = 0
    selector_changed_n = 0
    selected_frequency_rank_distribution: Counter[int] = Counter()
    selected_support_values: list[float] = []

    candidate_only = {
        "eligible_gold_in_k5_and_k_ge_2": 0,
        "frequency_top1_gold": 0,
        "ngram_top1_gold": 0,
        "ngram_rescue_vs_frequency": 0,
        "ngram_harm_vs_frequency": 0,
    }

    started = time.perf_counter()
    for number, row_id in enumerate(sorted(val_by_id), start=1):
        row = val_by_id[row_id]
        surface = surface_by_id[row_id]
        frozen = pred_by_id[row_id]
        ngram_row = ngram_by_id[row_id]

        gold = str(row.get("gold", row.get("target")))
        author = str(row["author"])

        # Ensure this really is the frozen Initial PV1 selection we intend to reproduce.
        if int(frozen.get("selected_k_pv", -1)) != FROZEN_K_PV:
            raise RuntimeError(f"Unexpected frozen selected_k_pv at {row_id}: {frozen.get('selected_k_pv')}")
        if not math.isclose(float(frozen.get("selected_lambda_frequency", -1)), FROZEN_LAMBDA_FREQUENCY):
            raise RuntimeError(
                f"Unexpected frozen selected_lambda_frequency at {row_id}: "
                f"{frozen.get('selected_lambda_frequency')}"
            )
        if not math.isclose(float(frozen.get("selected_lambda_pv", -1)), FROZEN_LAMBDA_PV):
            raise RuntimeError(
                f"Unexpected frozen selected_lambda_pv at {row_id}: {frozen.get('selected_lambda_pv')}"
            )

        frequency_candidates = frozen.get("frequency_candidates")
        frozen_pv1_candidates = frozen.get("pv1_candidates")
        if not isinstance(frequency_candidates, list) or not frequency_candidates:
            raise RuntimeError(f"Missing frozen frequency_candidates at {row_id}")
        if not isinstance(frozen_pv1_candidates, list) or not frozen_pv1_candidates:
            raise RuntimeError(f"Missing frozen pv1_candidates at {row_id}")

        personal_records = extract_personal_records(surface)
        personal_k5 = tuple(str(record["text"]) for record in personal_records)
        frequency_support = pv1_frequency_support(personal_records)

        methods = ngram_row.get("methods")
        if not isinstance(methods, Mapping) or selected_ngram_key not in methods:
            raise RuntimeError(f"Missing selected NGram method at {row_id}: {selected_ngram_key}")
        method = methods[selected_ngram_key]
        ngram_candidates = tuple(str(value) for value in method.get("candidates", []))
        ngram_support = tuple(float(value) for value in method.get("support", []))
        if ngram_candidates != personal_k5:
            raise RuntimeError(
                f"Frozen Personal K5 != NGram K5 at {row_id}: surface={personal_k5} ngram={ngram_candidates}"
            )
        if len(ngram_support) != len(personal_k5):
            raise RuntimeError(f"NGram support length mismatch at {row_id}")
        if ngram_support and not math.isclose(sum(ngram_support), 1.0, rel_tol=1e-7, abs_tol=1e-7):
            raise RuntimeError(f"NGram support does not sum to 1 at {row_id}: {sum(ngram_support)}")
        if ngram_row.get("gold_used_for_scoring") is not False:
            raise RuntimeError(f"NGram row has invalid Gold provenance at {row_id}")

        # Original selector = frequency-sorted Personal K5 first item.
        original_target = personal_k5[0] if personal_k5 else None
        original_count = int(personal_records[0]["frequency_count"]) if personal_records else None
        original_support = float(frequency_support.get(original_target, 0.0)) if original_target else 0.0

        reproduced = pv1_style_merge(
            frequency_candidates=frequency_candidates,
            selected_personal_target=original_target,
            selected_frequency_support=original_support,
            selected_frequency_count=original_count,
            selected_original_frequency_rank=1 if original_target else None,
            selector_label="frequency_top1",
        )

        frozen_texts = [candidate_text(value) for value in frozen_pv1_candidates]
        reproduced_texts = [candidate_text(value) for value in reproduced]
        frozen_rank = frozen.get("pv1_rank")
        reproduced_rank = rank_of(reproduced, gold)

        if reproduced_texts != frozen_texts:
            reproduced_top10_mismatch_n += 1
        if reproduced_rank != frozen_rank:
            reproduced_rank_mismatch_n += 1

        # Stronger cross-check: candidate score agreement wherever candidate is shared.
        score_mismatches: list[dict[str, Any]] = []
        frozen_by_target = {candidate_text(value): value for value in frozen_pv1_candidates}
        for value in reproduced:
            target = candidate_text(value)
            frozen_value = frozen_by_target.get(target)
            if frozen_value is None:
                score_mismatches.append({"candidate": target, "reason": "missing_from_frozen"})
                continue
            if not math.isclose(
                float(value["final_score"]),
                float(frozen_value["final_score"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                score_mismatches.append(
                    {
                        "candidate": target,
                        "reproduced": float(value["final_score"]),
                        "frozen": float(frozen_value["final_score"]),
                    }
                )

        if (reproduced_texts != frozen_texts or reproduced_rank != frozen_rank or score_mismatches) and len(reproduction_mismatch_examples) < 20:
            reproduction_mismatch_examples.append(
                {
                    "row_id": row_id,
                    "gold": gold,
                    "personal_k5": list(personal_k5),
                    "reproduced_texts": reproduced_texts,
                    "frozen_texts": frozen_texts,
                    "reproduced_rank": reproduced_rank,
                    "frozen_rank": frozen_rank,
                    "score_mismatches": score_mismatches[:5],
                }
            )

        # New selector = frozen Interpolated-NGram Top1 over the exact same K5.
        if personal_k5:
            selector_rows_with_candidates += 1
            order = ngram_rank_order(personal_k5, ngram_support)
            selected_index = order[0]
            ngram_target = personal_k5[selected_index]
            ngram_original_frequency_rank = selected_index + 1
            ngram_count = int(personal_records[selected_index]["frequency_count"])
            ngram_frequency_support = float(frequency_support[ngram_target])
            selected_frequency_rank_distribution[ngram_original_frequency_rank] += 1
            selected_support_values.append(ngram_frequency_support)
            selector_changed = ngram_target != original_target
            selector_changed_n += int(selector_changed)
        else:
            ngram_target = None
            ngram_original_frequency_rank = None
            ngram_count = None
            ngram_frequency_support = 0.0
            selector_changed = False

        new_ranking = pv1_style_merge(
            frequency_candidates=frequency_candidates,
            selected_personal_target=ngram_target,
            selected_frequency_support=ngram_frequency_support,
            selected_frequency_count=ngram_count,
            selected_original_frequency_rank=ngram_original_frequency_rank,
            selector_label="interpolated_ngram_top1",
        )
        new_rank = rank_of(new_ranking, gold)

        if len(personal_k5) >= 2 and gold in set(personal_k5):
            candidate_only["eligible_gold_in_k5_and_k_ge_2"] += 1
            freq_correct = original_target == gold
            ngram_correct = ngram_target == gold
            candidate_only["frequency_top1_gold"] += int(freq_correct)
            candidate_only["ngram_top1_gold"] += int(ngram_correct)
            candidate_only["ngram_rescue_vs_frequency"] += int((not freq_correct) and ngram_correct)
            candidate_only["ngram_harm_vs_frequency"] += int(freq_correct and (not ngram_correct))

        generic_missing = bool(frozen.get("generic_missing", surface.get("generic_missing", False)))
        outputs.append(
            {
                "schema_version": 1,
                "row_id": row_id,
                "author": author,
                "gold": gold,
                "generic_missing": generic_missing,
                "gold_in_personal_k5": gold in set(personal_k5),
                "generic_rank": frozen.get("generic_rank"),
                "frequency_rank": frozen.get("frequency_rank"),
                "pv1_rank": frozen.get("pv1_rank"),
                "pv1_reproduced_rank": reproduced_rank,
                "ngram_selector_pv1_rank": new_rank,
                "personal_k5": list(personal_k5),
                "personal_frequency_support": frequency_support,
                "frequency_selector_target": original_target,
                "ngram_selector_target": ngram_target,
                "ngram_selector_original_frequency_rank": ngram_original_frequency_rank,
                "ngram_selector_frequency_support": ngram_frequency_support,
                "ngram_selector_changed": selector_changed,
                "ngram_support": list(ngram_support),
                "pv1_reproduced_candidates": reproduced,
                "ngram_selector_pv1_candidates": new_ranking,
                "gold_used_for_scoring": False,
                "dev3000_used": False,
                "test_used": False,
            }
        )

        if args.progress_every > 0 and (number % args.progress_every == 0 or number == len(val_by_id)):
            elapsed = time.perf_counter() - started
            print(f"ROWS {number}/{len(val_by_id)} elapsed={elapsed:.1f}s", flush=True)

    # Exact reproduction is a precondition for interpreting the selector ablation.
    reproduction_ok = (
        reproduced_rank_mismatch_n == 0
        and reproduced_top10_mismatch_n == 0
        and not reproduction_mismatch_examples
    )
    reproduction_audit = {
        "status": "passed" if reproduction_ok else "failed",
        "rows": len(outputs),
        "pv1_rank_mismatch_n": reproduced_rank_mismatch_n,
        "pv1_top10_candidate_order_mismatch_n": reproduced_top10_mismatch_n,
        "examples": reproduction_mismatch_examples,
        "frozen_k_pv": FROZEN_K_PV,
        "frozen_lambda_frequency": FROZEN_LAMBDA_FREQUENCY,
        "frozen_lambda_pv": FROZEN_LAMBDA_PV,
    }
    write_json(args.output_root / "pv1_reproduction_audit.json", reproduction_audit)
    if not reproduction_ok:
        raise RuntimeError(
            "PV1 reproduction failed; selector ablation is not scientifically interpretable. "
            f"See {args.output_root / 'pv1_reproduction_audit.json'}"
        )

    generic_missing_n = sum(bool(row["generic_missing"]) for row in outputs)
    recoverable_n = sum(
        bool(row["generic_missing"]) and bool(row["gold_in_personal_k5"])
        for row in outputs
    )
    if generic_missing_n != EXPECTED_GENERIC_MISSING:
        raise RuntimeError(f"Unexpected Generic-Missing count: {generic_missing_n}")
    if recoverable_n != EXPECTED_K5_RECOVERABLE_MISSING:
        raise RuntimeError(f"Unexpected K5-recoverable Generic-Missing count: {recoverable_n}")

    pv1_metrics = metric_summary(outputs, "pv1_rank", "PV1")
    reproduced_metrics = metric_summary(outputs, "pv1_reproduced_rank", "PV1-Reproduced")
    new_metrics = metric_summary(outputs, "ngram_selector_pv1_rank", "PV1+NGramSelector")

    # The metric layer should also exactly reproduce frozen PV1.
    if pv1_metrics != {**reproduced_metrics, "method": "PV1"}:
        # Avoid brittle dict-order / label issues: check all scientific fields explicitly.
        for key in (
            "n", "authors", "macro_author_top1", "micro_top1", "top3", "top5",
            "mrr_at_10", "missing10", "mean_rank_given_top10", "per_author_top1",
        ):
            if pv1_metrics[key] != reproduced_metrics[key]:
                raise RuntimeError(f"PV1 reproduced metric mismatch for {key}")

    transition = transition_counts(outputs, "pv1_rank", "ngram_selector_pv1_rank")
    pv1_recovery = recovery_summary(outputs, "pv1_rank")
    new_recovery = recovery_summary(outputs, "ngram_selector_pv1_rank")

    if pv1_recovery["generic_missing_n"] != EXPECTED_GENERIC_MISSING:
        raise RuntimeError("PV1 recovery denominator drift")
    if pv1_recovery["recoverable_n"] != EXPECTED_K5_RECOVERABLE_MISSING:
        raise RuntimeError("PV1 recoverable denominator drift")

    candidate_only["net"] = (
        candidate_only["ngram_rescue_vs_frequency"]
        - candidate_only["ngram_harm_vs_frequency"]
    )
    candidate_only["frequency_top1_accuracy"] = safe_rate(
        candidate_only["frequency_top1_gold"],
        candidate_only["eligible_gold_in_k5_and_k_ge_2"],
    )
    candidate_only["ngram_top1_accuracy"] = safe_rate(
        candidate_only["ngram_top1_gold"],
        candidate_only["eligible_gold_in_k5_and_k_ge_2"],
    )

    selector_summary = {
        "rows_with_personal_candidates": selector_rows_with_candidates,
        "selector_changed_n": selector_changed_n,
        "selector_changed_rate": safe_rate(selector_changed_n, selector_rows_with_candidates),
        "ngram_selected_original_frequency_rank_distribution": {
            str(key): value for key, value in sorted(selected_frequency_rank_distribution.items())
        },
        "ngram_selected_frequency_support_mean": (
            statistics.fmean(selected_support_values) if selected_support_values else None
        ),
        "ngram_selected_frequency_support_median": (
            statistics.median(selected_support_values) if selected_support_values else None
        ),
        "candidate_only_gold_in_k5_k_ge_2": candidate_only,
    }

    comparison = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_pv1_ngram_selector_k5_v1",
        "research_question": (
            "Does replacing only PV1's frequency Top1 Personal-K5 selector with the frozen "
            "Interpolated-NGram K5 Top1 selector improve end-to-end PV1 recovery?"
        ),
        "primary_comparator": "PV1",
        "rows": len(outputs),
        "frozen_pv1": {
            "k_pv": FROZEN_K_PV,
            "lambda_frequency": FROZEN_LAMBDA_FREQUENCY,
            "lambda_pv": FROZEN_LAMBDA_PV,
            "frequency_support": "log1p(count) / max_log1p_count_over_frozen_PersonalK5",
            "merge": "PV1 boundary injection; Generic F unchanged; Generic wins exact ties",
        },
        "changed_component_only": {
            "from": "frequency-ordered Personal-K5 Top1",
            "to": "frozen Interpolated-NGram Personal-K5 Top1",
        },
        "pv1_reproduction_audit": reproduction_audit,
        "metrics": {
            "PV1": pv1_metrics,
            "PV1-Reproduced": reproduced_metrics,
            "PV1+NGramSelector": new_metrics,
        },
        "delta_vs_pv1": delta_metrics(new_metrics, pv1_metrics),
        "pv1_to_ngram_selector": transition,
        "recovery": {
            "PV1": pv1_recovery,
            "PV1+NGramSelector": new_recovery,
        },
        "selector_diagnostics": selector_summary,
        "generic_missing_n": generic_missing_n,
        "recoverable_generic_missing_k5_n": recoverable_n,
        "selected_k5_interpolated_key": selected_ngram_key,
        "gold_used_for_scoring_or_selection": False,
        "gold_used_for_train_val_evaluation_only": True,
        "dev3000_used": False,
        "test_used": False,
        "provenance": provenance,
    }

    predictions_path = args.output_root / "predictions.jsonl"
    write_jsonl(predictions_path, outputs)
    comparison_path = args.output_root / "comparison.json"
    write_json(comparison_path, comparison)

    artifact_paths = [
        args.output_root / "pv1_reproduction_audit.json",
        predictions_path,
        comparison_path,
    ]
    checksums = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in artifact_paths
    }
    write_json(args.output_root / "artifact_checksums.json", checksums)

    print("\n=== RESULTS (PRIMARY COMPARATOR = PV1) ===")
    print(
        "PV1                 "
        f"Macro={pv1_metrics['macro_author_top1']:.6f} "
        f"MRR10={pv1_metrics['mrr_at_10']:.6f} "
        f"Missing10={pv1_metrics['missing10']:.6f} "
        f"Rec@1={pv1_recovery['recovered_at_1']:.4f} "
        f"Rec@10={pv1_recovery['recovered_at_10']:.4f}"
    )
    print(
        "PV1+NGramSelector   "
        f"Macro={new_metrics['macro_author_top1']:.6f} "
        f"dPV1={new_metrics['macro_author_top1'] - pv1_metrics['macro_author_top1']:+.6f} "
        f"rescue={transition['rescue']} harm={transition['harm']} net={transition['net']:+d} "
        f"Rec@1={new_recovery['recovered_at_1']:.4f} "
        f"Rec@3={new_recovery['recovered_at_3']:.4f} "
        f"Rec@5={new_recovery['recovered_at_5']:.4f} "
        f"Rec@10={new_recovery['recovered_at_10']:.4f} "
        f"RecMRR={new_recovery['recovery_mrr_at_10']:.4f} "
        f"MeanRecRank={new_recovery['mean_recovered_rank']}"
    )
    print(
        "Selector diagnostic: "
        f"changed={selector_changed_n}/{selector_rows_with_candidates} "
        f"({safe_rate(selector_changed_n, selector_rows_with_candidates):.4f}); "
        f"candidate-only F-top1={candidate_only['frequency_top1_accuracy']:.6f} "
        f"NGram-top1={candidate_only['ngram_top1_accuracy']:.6f} "
        f"net={candidate_only['net']:+d}"
    )
    print("PV1 reproduction: PASSED exact Top10/rank cross-check")
    print(f"Saved: {comparison_path}")
    print("Gold used for scoring/selection: false")
    print("Dev3000 used: false")
    print("Test used: false")


if __name__ == "__main__":
    main()
