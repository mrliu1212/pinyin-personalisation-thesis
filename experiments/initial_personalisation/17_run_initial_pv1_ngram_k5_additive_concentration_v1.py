"""Additive Margin / Entropy / Dual concentration on frozen PV1+NGramSelector K5.

Research question
-----------------
Starting from the validated Initial-Pinyin development system

    Personal K5 -> frozen Interpolated-NGram ordering -> admit all K5
                -> exact PV1 candidate frequency support -> Generic-F boundary merge

keep the selector, candidate pool, Generic-F ranking, and PV1 frequency support fixed,
and test whether a query-level history concentration term adds useful calibration.

For every admitted Personal-K5 candidate c:

    score(c) = boundary + gamma_F * F_PV(c) + lambda_C * C(q)

where gamma_F is frozen to 4 and

    F_PV(c) = log(1 + count(c)) / max_j log(1 + count(j))

is the exact original PV1 candidate support over the frozen Personal K5.

Concentration families
----------------------
Margin:
    C_margin = p_(1) - p_(2)

Entropy:
    C_entropy = 1 - H_norm
    H_norm = -sum_i p_i log(p_i) / log(M)

Dual / Both:
    C_dual = sqrt(C_margin * C_entropy)

The distribution is the COMPLETE legal same-Pinyin history after applying the
strict same-author rolling H5000 budget BEFORE exact-Pinyin filtering. These are
the same definitions used in the earlier concentration experiment.

Important interpretation
------------------------
C(q) is shared by all Personal-K5 candidates of the current query. Therefore the
additive concentration term does NOT change their mutual ordering; it shifts the
whole admitted Personal-K5 group relative to Generic-F candidates.

Protocol
--------
- Clean3 Train-Fit + Train-Val only.
- Strictly-prior same-author H5000; H5000 BEFORE exact-Pinyin filtering.
- Earlier Train-Val rows may become legal history for later Train-Val rows.
- Frozen Personal-only K5 candidate surface.
- Frozen Generic Frequency-F ranking, lambda_F=4.
- Frozen Interpolated-NGram K5 ordering; no new NGram inference.
- Frozen gamma_F=4 unless explicitly overridden for diagnostic use.
- Gold is used only for Train-Val evaluation/model selection, never scoring.
- Dev3000 and Test are not read.

Controls
--------
1. Exact frozen PV1 Top10/rank/score reproduction is required.
2. lambda_C=0 must reproduce the validated K5 baseline metrics:
       Macro Top1 = 0.403772 (full precision constant below)
       Top3       = 0.603266
       MRR@10     = 0.533908
       Missing@10 = 0.243172
   The check uses full stored metric constants from the validated run.
3. All three families at lambda_C=0 must be identical row-by-row.

Default lambda grid
-------------------
    {0, .25, .5, 1, 2, 4}

Selection
---------
Each family selects lambda_C by Macro-author Top1, then MRR@10, then lower
lambda_C. This is Train-Val development selection only. All grid points are kept.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Frozen provenance / validated controls
# ---------------------------------------------------------------------------

EXPECTED_FIT_SHA256 = "162f5c98daa86cc69947571e6d8f20fc401f0a82cdd3fd6e517eb7be2addbdb4"
EXPECTED_VAL_SHA256 = "d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4"
EXPECTED_SURFACE_SHA256 = "205c0ba01cd0678d7a4341c503fa2e74cf126a70182ff582687025e4946764b2"
EXPECTED_FREQUENCY_PV1_SHA256 = "7fd8aa158d8cd50bced36b55610f8d932bc65e3aae1dbbd5bd65907ff1707ea7"

EXPECTED_FIT_ROWS = 144_526
EXPECTED_VAL_ROWS = 34_416
EXPECTED_GENERIC_MISSING = 12_565
EXPECTED_K5_RECOVERABLE_MISSING = 4_910
EXPECTED_K5_ELIGIBLE = 30_509
EXPECTED_K5_PAIRS = 123_738

FROZEN_LAMBDA_FREQUENCY = 4.0
FROZEN_K_PV = 1
FROZEN_LAMBDA_PV = 4.0
HISTORY_BUDGET = 5000
DEFAULT_GAMMA_F = 4.0
DEFAULT_LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
FAMILIES = ("Margin", "Entropy", "Dual")

# Full values from the validated PV1+NGramSelector@K5 run in this protocol.
# Rounded values printed by the validated K135 run. The cross-run control checks
# these to the displayed 6-decimal precision; row-wise lambda=0 identity is exact.
EXPECTED_K5_BASELINE = {
    "macro_author_top1": 0.403772,
    "top3": 0.603266,
    "mrr_at_10": 0.533908,
    "missing10": 0.243172,
}


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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


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


def parse_float_grid(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated floats") from exc
    if not values:
        raise argparse.ArgumentTypeError("Lambda grid must not be empty")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("Lambda values must be non-negative")
    values = tuple(dict.fromkeys(values))
    return values


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
# Exact causal H5000 concentration features (same definitions as old experiment)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoryRecord:
    row_id: str
    author: str
    position: int
    pinyin: tuple[str, ...]
    target: str


class CausalHistoryIndex:
    """Strictly-prior same-author H5000 before exact-Pinyin filtering."""

    def __init__(self, records: Sequence[HistoryRecord]) -> None:
        grouped: dict[str, list[HistoryRecord]] = defaultdict(list)
        for record in records:
            grouped[record.author].append(record)

        self.positions: dict[str, tuple[int, ...]] = {}
        self.pinyin_records: dict[tuple[str, tuple[str, ...]], tuple[HistoryRecord, ...]] = {}
        self.pinyin_ordinals: dict[tuple[str, tuple[str, ...]], tuple[int, ...]] = {}

        for author, values in grouped.items():
            ordered = tuple(sorted(values, key=lambda row: (row.position, row.row_id)))
            self.positions[author] = tuple(row.position for row in ordered)
            by_pinyin: dict[tuple[str, ...], list[tuple[int, HistoryRecord]]] = defaultdict(list)
            for ordinal, record in enumerate(ordered):
                by_pinyin[record.pinyin].append((ordinal, record))
            for pinyin, pairs in by_pinyin.items():
                key = (author, pinyin)
                self.pinyin_ordinals[key] = tuple(ordinal for ordinal, _ in pairs)
                self.pinyin_records[key] = tuple(record for _, record in pairs)

    def visible(
        self,
        *,
        author: str,
        position: int,
        pinyin: tuple[str, ...],
    ) -> tuple[HistoryRecord, ...]:
        positions = self.positions.get(author, ())
        stop = bisect.bisect_left(positions, position)
        start = max(0, stop - HISTORY_BUDGET)

        key = (author, pinyin)
        ordinals = self.pinyin_ordinals.get(key, ())
        records = self.pinyin_records.get(key, ())
        left = bisect.bisect_left(ordinals, start)
        right = bisect.bisect_left(ordinals, stop)
        return records[left:right]


def to_history_record(row: Mapping[str, Any]) -> HistoryRecord:
    target = row.get("target", row.get("gold"))
    if target is None:
        raise RuntimeError(f"History row has no target/gold: {row.get('row_id')}")
    return HistoryRecord(
        row_id=str(row["row_id"]),
        author=str(row["author"]),
        position=int(row["chronological_position"]),
        pinyin=tuple(str(value) for value in row["pinyin_segments"]),
        target=str(target),
    )


def distribution_features(counts: Mapping[str, int]) -> dict[str, float | int]:
    positive = sorted((int(value) for value in counts.values() if int(value) > 0), reverse=True)
    total = sum(positive)
    distinct = len(positive)

    if total <= 0 or distinct == 0:
        return {
            "same_pinyin_history_count": 0,
            "distinct_targets": 0,
            "entropy_norm": 1.0,
            "conc_entropy": 0.0,
            "winner_share": 0.0,
            "runner_up_share": 0.0,
            "conc_margin": 0.0,
            "conc_dual": 0.0,
        }

    shares = [value / total for value in positive]
    winner_share = shares[0]
    runner_up_share = shares[1] if distinct >= 2 else 0.0

    if distinct == 1:
        entropy_norm = 0.0
        conc_entropy = 1.0
        conc_margin = 1.0
    else:
        entropy = -sum(p * math.log(p) for p in shares if p > 0)
        entropy_norm = entropy / math.log(distinct)
        entropy_norm = max(0.0, min(1.0, entropy_norm))
        conc_entropy = 1.0 - entropy_norm
        conc_margin = max(0.0, min(1.0, winner_share - runner_up_share))

    conc_dual = math.sqrt(max(0.0, conc_entropy) * max(0.0, conc_margin))

    return {
        "same_pinyin_history_count": total,
        "distinct_targets": distinct,
        "entropy_norm": entropy_norm,
        "conc_entropy": conc_entropy,
        "winner_share": winner_share,
        "runner_up_share": runner_up_share,
        "conc_margin": conc_margin,
        "conc_dual": conc_dual,
    }


def concentration_for_family(feature: Mapping[str, Any], family: str) -> float:
    if family == "Margin":
        value = float(feature["conc_margin"])
    elif family == "Entropy":
        value = float(feature["conc_entropy"])
    elif family == "Dual":
        value = float(feature["conc_dual"])
    else:
        raise ValueError(f"Unknown concentration family: {family}")
    if value < -1e-12 or value > 1.0 + 1e-12:
        raise RuntimeError(f"Concentration outside [0,1]: {family}={value}")
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Frozen Personal K5 / exact PV1 support / frozen NGram ordering
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
        text = str(value.get("text", value.get("target", value.get("candidate", ""))))
        if not text:
            raise RuntimeError(f"Empty Personal-K5 target at {surface.get('row_id')}")
        if text in seen:
            raise RuntimeError(f"Duplicate Personal-K5 target at {surface.get('row_id')}: {text}")
        seen.add(text)

        count = int(value.get("frequency_count", 0))
        if count <= 0:
            raise RuntimeError(f"Non-positive frequency_count at {surface.get('row_id')}: {text}")
        recorded_rank = int(value.get("personal_rank", position))
        if recorded_rank != position:
            raise RuntimeError(
                f"Frozen Personal-K5 rank mismatch at {surface.get('row_id')}: "
                f"target={text} expected={position} got={recorded_rank}"
            )
        records.append({"text": text, "frequency_count": count, "frequency_rank": position})

    frozen_texts = [str(value) for value in surface.get("personal_candidate_texts_top5", [])]
    texts = [record["text"] for record in records]
    if frozen_texts and texts != frozen_texts:
        raise RuntimeError(
            f"Personal-K5 record/text fields differ at {surface.get('row_id')}: "
            f"records={texts} texts={frozen_texts}"
        )
    return tuple(records)


def pv1_frequency_support(personal_records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    raw = {
        str(record["text"]): math.log1p(int(record["frequency_count"]))
        for record in personal_records
    }
    maximum = max(raw.values(), default=0.0)
    return {text: value / maximum for text, value in raw.items()} if maximum else {}


def ngram_rank_order(candidates: Sequence[str], support: Sequence[float]) -> list[int]:
    if len(candidates) != len(support):
        raise RuntimeError("NGram candidate/support length mismatch")
    return sorted(
        range(len(candidates)),
        key=lambda i: (-float(support[i]), i, str(candidates[i])),
    )


# ---------------------------------------------------------------------------
# PV1-style merge with additive query concentration
# ---------------------------------------------------------------------------


def additive_merge(
    *,
    frequency_candidates: Sequence[Mapping[str, Any]],
    selected_personal: Sequence[Mapping[str, Any]],
    gamma_f: float,
    lambda_c: float,
    concentration: float,
    selector_label: str,
) -> list[dict[str, Any]]:
    """Generic-F unchanged; Personal score = boundary + gamma_F*F + lambda_C*C."""

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
        row = dict(frozen)
        row["candidate"] = target
        row["source"] = "generic"
        row["final_score"] = float(frozen["final_score"])
        row["generic_rank"] = int(frozen.get("generic_rank") or position)
        rows.append(row)

    if not rows:
        raise RuntimeError("Frozen Frequency candidate list is empty")
    boundary = min(normalized_generic)

    seen_personal: set[str] = set()
    for admitted_rank, item in enumerate(selected_personal, start=1):
        target = str(item["target"])
        if target in generic_texts:
            raise RuntimeError(f"Selected Personal candidate overlaps Generic: {target!r}")
        if target in seen_personal:
            raise RuntimeError(f"Duplicate selected Personal candidate: {target!r}")
        seen_personal.add(target)

        support = float(item["frequency_support"])
        count = int(item["frequency_count"])
        original_frequency_rank = int(item["original_frequency_rank"])
        ngram_rank = int(item.get("ngram_rank", admitted_rank))
        final_score = boundary + float(gamma_f) * support + float(lambda_c) * float(concentration)

        rows.append(
            {
                "candidate": target,
                "generic_rank": None,
                "generic_score": None,
                "normalized_generic_score": boundary,
                "personal_score": support,
                "frequency_support": support,
                "frequency_count": count,
                "concentration": float(concentration),
                "gamma_f": float(gamma_f),
                "lambda_c": float(lambda_c),
                "concentration_boost": float(lambda_c) * float(concentration),
                "final_score": final_score,
                "source": "personal_vocabulary",
                "personal_candidate_rank": admitted_rank,
                "selector": selector_label,
                "ngram_rank": ngram_rank,
                "original_personal_frequency_rank": original_frequency_rank,
            }
        )

    # Exact K135 tie policy: Generic wins exact ties; Personal-personal ties follow
    # NGram admission order, then text.
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
# Metrics / diagnostics
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


def transition_counts(rows: Sequence[Mapping[str, Any]], base_key: str, new_key: str) -> dict[str, int]:
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


def delta_metrics(method: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, float]:
    keys = ("macro_author_top1", "micro_top1", "top3", "top5", "mrr_at_10", "missing10")
    return {key: float(method[key]) - float(base[key]) for key in keys}


def conc_bin(value: float) -> str:
    if value < 0.10:
        return "[0,.10)"
    if value < 0.25:
        return "[.10,.25)"
    if value < 0.50:
        return "[.25,.50)"
    if value < 0.75:
        return "[.50,.75)"
    return "[.75,1]"


def bin_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    rank_key: str,
    concentration_key: str,
    base_key: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[conc_bin(float(row[concentration_key]))].append(row)

    output: list[dict[str, Any]] = []
    for label in ("[0,.10)", "[.10,.25)", "[.25,.50)", "[.50,.75)", "[.75,1]"):
        values = grouped.get(label, [])
        trans = transition_counts(values, base_key, rank_key)
        output.append({"bin": label, "n": len(values), **trans})
    return output


def choose_lambda(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        raise RuntimeError("Cannot select from empty lambda grid")
    return dict(
        max(
            results,
            key=lambda row: (
                float(row["metrics"]["macro_author_top1"]),
                float(row["metrics"]["mrr_at_10"]),
                -float(row["lambda_c"]),
            ),
        )
    )


# ---------------------------------------------------------------------------
# Input verification
# ---------------------------------------------------------------------------


def verify_inputs(args: argparse.Namespace) -> dict[str, Any]:
    frozen = {
        "fit": (args.fit, EXPECTED_FIT_SHA256),
        "val": (args.val, EXPECTED_VAL_SHA256),
        "candidate_surface": (args.candidate_surface, EXPECTED_SURFACE_SHA256),
        "frequency_pv1_predictions": (
            args.frequency_pv1_predictions,
            EXPECTED_FREQUENCY_PV1_SHA256,
        ),
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

    adaptive = read_json(comparison_path)
    adaptive_provenance = adaptive.get("provenance", {})
    if adaptive_provenance.get("val_sha256") != EXPECTED_VAL_SHA256:
        raise RuntimeError("Adaptive NGram comparison uses another Train-Val artifact")
    if adaptive_provenance.get("frozen_candidate_surface_sha256") != EXPECTED_SURFACE_SHA256:
        raise RuntimeError("Adaptive NGram comparison uses another Personal-K5 surface")
    if adaptive_provenance.get("candidate_selection_used_gold") is not False:
        raise RuntimeError("Adaptive NGram candidate selection is not Gold-free")
    if adaptive_provenance.get("gold_used_for_scoring") is not False:
        raise RuntimeError("Adaptive NGram scoring is not Gold-free")
    if adaptive_provenance.get("dev3000_used") is not False or adaptive_provenance.get("test_used") is not False:
        raise RuntimeError("Adaptive NGram artifact crossed Dev3000/Test boundary")

    selected_by_k = adaptive.get("selected_by_k", {})
    selected_key = str(selected_by_k.get("K5", {}).get("best_interpolated", ""))
    if not selected_key.startswith("K5|Interpolated|"):
        raise RuntimeError(f"Unexpected selected K5 Interpolated key: {selected_key!r}")

    actual_scores_sha = sha256_file(scores)
    recorded_scores_sha = adaptive_provenance.get("scores_sha256")
    if recorded_scores_sha and recorded_scores_sha != actual_scores_sha:
        raise RuntimeError("Adaptive scores.jsonl SHA differs from comparison provenance")

    return {
        "paths": {
            "fit": str(args.fit),
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
        description=(
            "K5 PV1+Interpolated-NGram additive concentration: "
            "boundary + gamma_F*F_PV + lambda_C*C for Margin/Entropy/Dual"
        )
    )
    parser.add_argument("--fit", type=Path, required=True)
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
    parser.add_argument(
        "--lambdas",
        type=parse_float_grid,
        default=DEFAULT_LAMBDAS,
        help="Comma-separated lambda_C grid; default: 0,.25,.5,1,2,4",
    )
    parser.add_argument(
        "--gamma-f",
        type=float,
        default=DEFAULT_GAMMA_F,
        help="Weight on exact PV1 frequency support. Frozen development control is 4.",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    lambdas: tuple[float, ...] = args.lambdas
    if 0.0 not in lambdas:
        raise RuntimeError("Lambda grid must include 0 for the exact K5 control")
    if args.gamma_f < 0:
        raise RuntimeError("--gamma-f must be non-negative")
    # Scientific control: this experiment is concentration-only. Deliberate overrides
    # remain possible for diagnostics, but the validated K5 metric check is only valid at 4.
    strict_k5_control = math.isclose(float(args.gamma_f), DEFAULT_GAMMA_F, rel_tol=0.0, abs_tol=0.0)

    args.output_root.mkdir(parents=True, exist_ok=True)
    provenance = verify_inputs(args)
    selected_ngram_key = str(provenance["selected_k5_interpolated_key"])

    print("=== K5 ADDITIVE CONCENTRATION ON PV1 + NGRAM SELECTOR ===")
    print(f"formula: boundary + {args.gamma_f:g} * F_PV(c) + lambda_C * C(q)")
    print(f"lambda_C grid: {lambdas}")
    print("families: Margin, Entropy, Dual=sqrt(Margin*Entropy)")
    print(f"frozen K5 Interpolated method: {selected_ngram_key}")
    print("K: 5 only; all available Personal K5 candidates admitted")
    print("Gold used for scoring/selection: false")
    print("Dev3000 used: false")
    print("Test used: false")
    print()

    fit_rows = read_jsonl(args.fit)
    val_rows = read_jsonl(args.val)
    surface_by_id = index_rows(read_jsonl(args.candidate_surface), "candidate surface")
    pred_by_id = index_rows(read_jsonl(args.frequency_pv1_predictions), "Frequency/PV1 predictions")
    ngram_by_id = index_rows(read_jsonl(args.adaptive_dir / "scores.jsonl"), "adaptive NGram scores")
    val_by_id = index_rows(val_rows, "Train-Val")

    if len(fit_rows) != EXPECTED_FIT_ROWS:
        raise RuntimeError(f"Unexpected Train-Fit rows: {len(fit_rows)}")
    if len(val_rows) != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"Unexpected Train-Val rows: {len(val_rows)}")
    if not (set(val_by_id) == set(surface_by_id) == set(pred_by_id) == set(ngram_by_id)):
        raise RuntimeError("Train-Val / surface / PV1 / NGram row IDs differ")

    records = [to_history_record(row) for row in fit_rows]
    records.extend(to_history_record(row) for row in val_rows)
    history = CausalHistoryIndex(records)

    outputs: list[dict[str, Any]] = []
    pv1_mismatch_examples: list[dict[str, Any]] = []
    pv1_rank_mismatch_n = 0
    pv1_top10_mismatch_n = 0
    candidate_count_distribution: Counter[int] = Counter()

    started = time.perf_counter()
    for number, row_id in enumerate(sorted(val_by_id), start=1):
        row = val_by_id[row_id]
        surface = surface_by_id[row_id]
        frozen = pred_by_id[row_id]
        ngram_row = ngram_by_id[row_id]

        gold = str(row.get("gold", row.get("target")))
        author = str(row["author"])
        pinyin = tuple(str(value) for value in row["pinyin_segments"])

        if int(frozen.get("selected_k_pv", -1)) != FROZEN_K_PV:
            raise RuntimeError(f"Unexpected frozen selected_k_pv at {row_id}")
        if not math.isclose(float(frozen.get("selected_lambda_frequency", -1)), FROZEN_LAMBDA_FREQUENCY):
            raise RuntimeError(f"Unexpected frozen Frequency lambda at {row_id}")
        if not math.isclose(float(frozen.get("selected_lambda_pv", -1)), FROZEN_LAMBDA_PV):
            raise RuntimeError(f"Unexpected frozen PV1 lambda at {row_id}")

        frequency_candidates = frozen.get("frequency_candidates")
        frozen_pv1_candidates = frozen.get("pv1_candidates")
        if not isinstance(frequency_candidates, list) or not frequency_candidates:
            raise RuntimeError(f"Missing frozen frequency_candidates at {row_id}")
        if not isinstance(frozen_pv1_candidates, list) or not frozen_pv1_candidates:
            raise RuntimeError(f"Missing frozen pv1_candidates at {row_id}")

        personal_records = extract_personal_records(surface)
        personal_k5 = tuple(str(record["text"]) for record in personal_records)
        personal_k5_set = set(personal_k5)
        candidate_count_distribution[len(personal_k5)] += 1
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

        # Exact concentration history using same semantics/definitions as old experiment.
        visible = history.visible(
            author=author,
            position=int(row["chronological_position"]),
            pinyin=pinyin,
        )
        counts = Counter(record.target for record in visible)
        dist = distribution_features(counts)

        # Cross-check frozen candidate counts against the legal visible same-Pinyin history.
        for record in personal_records:
            target = str(record["text"])
            visible_count = int(counts.get(target, 0))
            frozen_count = int(record["frequency_count"])
            if visible_count != frozen_count:
                raise RuntimeError(
                    f"Personal frequency count drift at {row_id} target={target!r}: "
                    f"surface={frozen_count} recomputed={visible_count}"
                )

        ngram_order = ngram_rank_order(personal_k5, ngram_support) if personal_k5 else []
        selected_personal: list[dict[str, Any]] = []
        for ngram_rank, index in enumerate(ngram_order, start=1):
            target = personal_k5[index]
            selected_personal.append(
                {
                    "target": target,
                    "frequency_support": float(frequency_support[target]),
                    "frequency_count": int(personal_records[index]["frequency_count"]),
                    "original_frequency_rank": index + 1,
                    "ngram_rank": ngram_rank,
                }
            )

        # Exact original PV1 reproduction (frequency Top1 only).
        original_selected: list[dict[str, Any]] = []
        if personal_k5:
            target = personal_k5[0]
            original_selected.append(
                {
                    "target": target,
                    "frequency_support": float(frequency_support[target]),
                    "frequency_count": int(personal_records[0]["frequency_count"]),
                    "original_frequency_rank": 1,
                    "ngram_rank": 1,
                }
            )
        reproduced_pv1 = additive_merge(
            frequency_candidates=frequency_candidates,
            selected_personal=original_selected,
            gamma_f=FROZEN_LAMBDA_PV,
            lambda_c=0.0,
            concentration=0.0,
            selector_label="frequency_top1_pv1_control",
        )
        frozen_texts = [candidate_text(value) for value in frozen_pv1_candidates]
        reproduced_texts = [candidate_text(value) for value in reproduced_pv1]
        reproduced_pv1_rank = rank_of(reproduced_pv1, gold)
        if reproduced_texts != frozen_texts:
            pv1_top10_mismatch_n += 1
        if reproduced_pv1_rank != frozen.get("pv1_rank"):
            pv1_rank_mismatch_n += 1

        score_mismatch = False
        frozen_by_target = {candidate_text(value): value for value in frozen_pv1_candidates}
        for value in reproduced_pv1:
            target = candidate_text(value)
            fv = frozen_by_target.get(target)
            if fv is None or not math.isclose(
                float(value["final_score"]), float(fv["final_score"]), rel_tol=1e-12, abs_tol=1e-12
            ):
                score_mismatch = True
                break
        if (reproduced_texts != frozen_texts or reproduced_pv1_rank != frozen.get("pv1_rank") or score_mismatch) and len(pv1_mismatch_examples) < 20:
            pv1_mismatch_examples.append(
                {
                    "row_id": row_id,
                    "gold": gold,
                    "reproduced_texts": reproduced_texts,
                    "frozen_texts": frozen_texts,
                    "reproduced_rank": reproduced_pv1_rank,
                    "frozen_rank": frozen.get("pv1_rank"),
                    "score_mismatch": score_mismatch,
                }
            )

        row_output: dict[str, Any] = {
            "schema_version": 1,
            "row_id": row_id,
            "author": author,
            "gold": gold,
            "pinyin_segments": list(pinyin),
            "generic_missing": bool(frozen.get("generic_missing", surface.get("generic_missing", False))),
            "gold_in_personal_k5": gold in personal_k5_set,
            "generic_rank": frozen.get("generic_rank"),
            "frequency_rank": frozen.get("frequency_rank"),
            "pv1_rank": frozen.get("pv1_rank"),
            "pv1_reproduced_rank": reproduced_pv1_rank,
            "personal_k5": list(personal_k5),
            "personal_frequency_support": frequency_support,
            "ngram_ordered_personal_k5": [personal_k5[i] for i in ngram_order],
            "selected_personal_k5": selected_personal,
            **dist,
            "gold_used_for_scoring": False,
            "dev3000_used": False,
            "test_used": False,
        }

        # K5 baseline: exactly boundary + 4F, concentration disabled.
        k5_baseline = additive_merge(
            frequency_candidates=frequency_candidates,
            selected_personal=selected_personal,
            gamma_f=float(args.gamma_f),
            lambda_c=0.0,
            concentration=0.0,
            selector_label="interpolated_ngram_k5_baseline",
        )
        row_output["k5_baseline_rank"] = rank_of(k5_baseline, gold)
        row_output["k5_baseline_candidates"] = k5_baseline

        for family in FAMILIES:
            concentration = concentration_for_family(row_output, family)
            for lambda_c in lambdas:
                label = f"{family}@lambda={lambda_c:g}"
                ranking = additive_merge(
                    frequency_candidates=frequency_candidates,
                    selected_personal=selected_personal,
                    gamma_f=float(args.gamma_f),
                    lambda_c=float(lambda_c),
                    concentration=concentration,
                    selector_label=f"k5_{family.lower()}_lambda_{lambda_c:g}",
                )
                row_output[f"rank__{label}"] = rank_of(ranking, gold)

        outputs.append(row_output)

        if args.progress_every > 0 and (number % args.progress_every == 0 or number == len(val_by_id)):
            elapsed = time.perf_counter() - started
            print(f"ROWS {number}/{len(val_by_id)} elapsed={elapsed:.1f}s", flush=True)

    # PV1 exact reproduction is a hard precondition.
    pv1_reproduction_ok = pv1_rank_mismatch_n == 0 and pv1_top10_mismatch_n == 0 and not pv1_mismatch_examples
    pv1_audit = {
        "status": "passed" if pv1_reproduction_ok else "failed",
        "rows": len(outputs),
        "pv1_rank_mismatch_n": pv1_rank_mismatch_n,
        "pv1_top10_candidate_order_mismatch_n": pv1_top10_mismatch_n,
        "examples": pv1_mismatch_examples,
    }
    write_json(args.output_root / "pv1_reproduction_audit.json", pv1_audit)
    if not pv1_reproduction_ok:
        raise RuntimeError("PV1 exact reproduction failed; aborting concentration experiment")

    eligible_n = sum(bool(row["personal_k5"]) for row in outputs)
    pair_n = sum(len(row["personal_k5"]) for row in outputs)
    if eligible_n != EXPECTED_K5_ELIGIBLE:
        raise RuntimeError(f"K5 eligible-row drift: expected={EXPECTED_K5_ELIGIBLE} actual={eligible_n}")
    if pair_n != EXPECTED_K5_PAIRS:
        raise RuntimeError(f"K5 pair-count drift: expected={EXPECTED_K5_PAIRS} actual={pair_n}")

    generic_missing_n = sum(bool(row["generic_missing"]) for row in outputs)
    recoverable_n = sum(bool(row["generic_missing"]) and bool(row["gold_in_personal_k5"]) for row in outputs)
    if generic_missing_n != EXPECTED_GENERIC_MISSING:
        raise RuntimeError(f"Unexpected Generic-Missing count: {generic_missing_n}")
    if recoverable_n != EXPECTED_K5_RECOVERABLE_MISSING:
        raise RuntimeError(f"Unexpected K5-recoverable Generic-Missing count: {recoverable_n}")

    pv1_metrics = metric_summary(outputs, "pv1_rank", "PV1")
    k5_metrics = metric_summary(outputs, "k5_baseline_rank", "PV1+NGramSelector@K5")
    k5_recovery = recovery_summary(outputs, "k5_baseline_rank")
    k5_transition = transition_counts(outputs, "pv1_rank", "k5_baseline_rank")

    # The scientific concentration-only experiment requires the validated gamma_F=4 baseline.
    k5_metric_check: dict[str, Any] = {
        "strict_control_enabled": strict_k5_control,
        "expected": EXPECTED_K5_BASELINE,
        "actual": {key: k5_metrics[key] for key in EXPECTED_K5_BASELINE},
        "passed": None,
    }
    if strict_k5_control:
        mismatches: dict[str, Any] = {}
        for key, expected in EXPECTED_K5_BASELINE.items():
            actual = float(k5_metrics[key])
            # These control constants come from the prior runner's displayed 6-decimal
            # output. Require agreement at that displayed precision.
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=5e-7):
                mismatches[key] = {"expected": expected, "actual": actual}
        k5_metric_check["mismatches"] = mismatches
        k5_metric_check["passed"] = not mismatches
        if mismatches:
            write_json(args.output_root / "k5_baseline_metric_check.json", k5_metric_check)
            raise RuntimeError(
                "K5 lambda=0 control does not reproduce the validated K5 baseline; "
                f"see {args.output_root / 'k5_baseline_metric_check.json'}"
            )
    else:
        k5_metric_check["passed"] = False
        k5_metric_check["note"] = "gamma_F was overridden, so validated K5 control is intentionally disabled"
    write_json(args.output_root / "k5_baseline_metric_check.json", k5_metric_check)

    # All families at lambda=0 must be row-wise identical to K5 baseline.
    lambda0_row_mismatches: dict[str, int] = {}
    for family in FAMILIES:
        key = f"rank__{family}@lambda=0"
        mismatch = sum(row.get(key) != row.get("k5_baseline_rank") for row in outputs)
        lambda0_row_mismatches[family] = mismatch
        if mismatch:
            raise RuntimeError(f"{family} lambda=0 is not row-wise identical to K5 baseline: {mismatch}")

    grid_results: dict[str, list[dict[str, Any]]] = {family: [] for family in FAMILIES}
    selected_by_family: dict[str, Any] = {}

    for family in FAMILIES:
        concentration_key = {
            "Margin": "conc_margin",
            "Entropy": "conc_entropy",
            "Dual": "conc_dual",
        }[family]
        family_results: list[dict[str, Any]] = []
        for lambda_c in lambdas:
            label = f"{family}@lambda={lambda_c:g}"
            rank_key = f"rank__{label}"
            metrics = metric_summary(outputs, rank_key, label)
            transition = transition_counts(outputs, "k5_baseline_rank", rank_key)
            transition_vs_pv1 = transition_counts(outputs, "pv1_rank", rank_key)
            recovery = recovery_summary(outputs, rank_key)
            report = {
                "family": family,
                "lambda_c": float(lambda_c),
                "gamma_f": float(args.gamma_f),
                "metrics": metrics,
                "delta_vs_k5": delta_metrics(metrics, k5_metrics),
                "delta_vs_pv1": delta_metrics(metrics, pv1_metrics),
                "k5_to_method": transition,
                "pv1_to_method": transition_vs_pv1,
                "recovery": recovery,
                "concentration_bins_vs_k5": bin_diagnostics(
                    outputs, rank_key, concentration_key, "k5_baseline_rank"
                ),
            }
            family_results.append(report)
            print(
                f"{family:<8} lambda={lambda_c:>4g}  "
                f"Macro={metrics['macro_author_top1']:.6f} "
                f"dK5={report['delta_vs_k5']['macro_author_top1']:+.6f} "
                f"Top3={metrics['top3']:.6f} "
                f"MRR10={metrics['mrr_at_10']:.6f} "
                f"Missing10={metrics['missing10']:.6f} "
                f"K5->new rescue={transition['rescue']} harm={transition['harm']} net={transition['net']:+d}",
                flush=True,
            )
        grid_results[family] = family_results
        selected_by_family[family] = choose_lambda(family_results)

    # Cross-family development-best. Exact ties prefer simpler concentration definition
    # in the declared order Margin -> Entropy -> Dual, then lower lambda.
    family_complexity = {family: index for index, family in enumerate(FAMILIES)}
    development_best = dict(
        max(
            selected_by_family.values(),
            key=lambda row: (
                float(row["metrics"]["macro_author_top1"]),
                float(row["metrics"]["mrr_at_10"]),
                -family_complexity[str(row["family"])],
                -float(row["lambda_c"]),
            ),
        )
    )

    feature_summary = {
        "rows": len(outputs),
        "history_budget": HISTORY_BUDGET,
        "history_semantics": "strictly-prior same-author H5000 before exact-Pinyin filtering",
        "candidate_count_distribution": {
            str(key): value for key, value in sorted(candidate_count_distribution.items())
        },
        "mean_same_pinyin_history_count": statistics.fmean(
            int(row["same_pinyin_history_count"]) for row in outputs
        ),
        "mean_distinct_targets": statistics.fmean(int(row["distinct_targets"]) for row in outputs),
        "mean_conc_margin": statistics.fmean(float(row["conc_margin"]) for row in outputs),
        "mean_conc_entropy": statistics.fmean(float(row["conc_entropy"]) for row in outputs),
        "mean_conc_dual": statistics.fmean(float(row["conc_dual"]) for row in outputs),
    }

    predictions_path = args.output_root / "predictions.jsonl"
    write_jsonl(predictions_path, outputs)

    comparison = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_pv1_ngram_k5_additive_concentration_v1",
        "research_question": (
            "Does additive query-level concentration improve frozen PV1+Interpolated-NGram K5 "
            "when candidate-specific PV1 frequency support remains fixed?"
        ),
        "formula": "boundary + gamma_F * F_PV(candidate) + lambda_C * C(query)",
        "gamma_f": float(args.gamma_f),
        "lambda_grid": list(lambdas),
        "families": {
            "Margin": "p1 - p2 over full legal same-Pinyin history distribution",
            "Entropy": "1 - normalized entropy over full legal same-Pinyin history distribution",
            "Dual": "sqrt(Margin * Entropy)",
        },
        "concentration_effect": (
            "query-shared additive offset to all admitted Personal-K5 candidates; "
            "does not alter their mutual order"
        ),
        "selection_policy": "Macro-author Top1, then MRR@10, then lower lambda_C",
        "rows": len(outputs),
        "feature_summary": feature_summary,
        "pv1_reproduction_audit": pv1_audit,
        "k5_baseline_metric_check": k5_metric_check,
        "lambda0_row_mismatches": lambda0_row_mismatches,
        "baselines": {
            "PV1": {
                "metrics": pv1_metrics,
                "recovery": recovery_summary(outputs, "pv1_rank"),
            },
            "PV1+NGramSelector@K5": {
                "metrics": k5_metrics,
                "delta_vs_pv1": delta_metrics(k5_metrics, pv1_metrics),
                "pv1_to_k5": k5_transition,
                "recovery": k5_recovery,
            },
        },
        "grid_results": grid_results,
        "selected_by_family": selected_by_family,
        "development_best": development_best,
        "gold_used_for_scoring_or_selection": False,
        "gold_used_for_train_val_selection_evaluation_only": True,
        "dev3000_used": False,
        "test_used": False,
        "provenance": provenance,
    }
    comparison_path = args.output_root / "comparison.json"
    write_json(comparison_path, comparison)

    artifacts = [
        args.output_root / "pv1_reproduction_audit.json",
        args.output_root / "k5_baseline_metric_check.json",
        predictions_path,
        comparison_path,
    ]
    checksums = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in artifacts
    }
    write_json(args.output_root / "artifact_checksums.json", checksums)

    print("\n=== BASELINES ===")
    print(
        "PV1                      "
        f"Macro={pv1_metrics['macro_author_top1']:.6f} "
        f"Top3={pv1_metrics['top3']:.6f} MRR10={pv1_metrics['mrr_at_10']:.6f} "
        f"Missing10={pv1_metrics['missing10']:.6f}"
    )
    print(
        "PV1+NGramSelector@K5     "
        f"Macro={k5_metrics['macro_author_top1']:.6f} "
        f"Top3={k5_metrics['top3']:.6f} MRR10={k5_metrics['mrr_at_10']:.6f} "
        f"Missing10={k5_metrics['missing10']:.6f} "
        f"PV1->K5 rescue={k5_transition['rescue']} harm={k5_transition['harm']} net={k5_transition['net']:+d} "
        f"Rec@10={k5_recovery['recovered_at_10']:.4f}"
    )

    print("\n=== SELECTED BY FAMILY ===")
    for family in FAMILIES:
        report = selected_by_family[family]
        metrics = report["metrics"]
        delta = report["delta_vs_k5"]
        transition = report["k5_to_method"]
        recovery = report["recovery"]
        print(
            f"{family:<8} lambda={report['lambda_c']:g}  "
            f"Macro={metrics['macro_author_top1']:.6f} dK5Top1={delta['macro_author_top1']:+.6f} "
            f"Top3={metrics['top3']:.6f} dK5Top3={delta['top3']:+.6f} "
            f"MRR10={metrics['mrr_at_10']:.6f} Missing10={metrics['missing10']:.6f} "
            f"K5->new rescue={transition['rescue']} harm={transition['harm']} net={transition['net']:+d} "
            f"Rec@1={recovery['recovered_at_1']:.4f} Rec@3={recovery['recovered_at_3']:.4f} "
            f"Rec@5={recovery['recovered_at_5']:.4f} Rec@10={recovery['recovered_at_10']:.4f}"
        )

    best = development_best
    print("\nDEVELOPMENT BEST:")
    print(
        f"{best['family']} lambda={best['lambda_c']:g} "
        f"Macro={best['metrics']['macro_author_top1']:.6f} "
        f"dK5Top1={best['delta_vs_k5']['macro_author_top1']:+.6f} "
        f"Top3={best['metrics']['top3']:.6f} "
        f"MRR10={best['metrics']['mrr_at_10']:.6f} "
        f"Missing10={best['metrics']['missing10']:.6f}"
    )
    print("PV1 reproduction: PASSED exact Top10/rank/score cross-check")
    if strict_k5_control:
        print("K5 lambda=0 control: PASSED validated metric cross-check")
    else:
        print("K5 lambda=0 control: NOT STRICT (gamma_F override)")
    print(f"Saved: {comparison_path}")
    print("Gold used for scoring/selection: false")
    print("Dev3000 used: false")
    print("Test used: false")


if __name__ == "__main__":
    main()
