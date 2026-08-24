"""Joint calibration of PV1 frequency weight and additive concentration on K5.

System
------
Personal K5 -> frozen Interpolated-NGram ordering -> admit all K5
            -> PV1 candidate support F_PV(c) -> Generic-F boundary merge

For each admitted personal candidate c:

    score(c) = boundary + gamma_F * F_PV(c) + lambda_C * C(q)

Families:
- F-only: lambda_C = 0
- Margin: C = p1 - p2
- Entropy: C = 1 - normalized entropy
- Dual/Both: C = sqrt(Margin * Entropy)

Default grids:
    gamma_F in {1,2,3,4,5,6,8}
    lambda_C in {0,.25,.5,1,2,4}

The validated K5 point (gamma_F=4, lambda_C=0) is a hard control and must
reproduce the prior K5 metrics. The script separately reports the best F-only
point and the best joint point in each concentration family so that any gain
from concentration can be distinguished from simply re-tuning gamma_F.

Protocol is unchanged: Clean3 Train-Fit/Train-Val only; same-author causal
H5000 before exact-Pinyin filtering; frozen Personal K5 / Generic-F / NGram
artifacts; Gold is evaluation-only; Dev3000/Test are not read.
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
DEFAULT_GAMMAS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
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
# Joint-grid helpers / main
# ---------------------------------------------------------------------------


def metric_summary_from_ranks(
    rows: Sequence[Mapping[str, Any]],
    ranks: Sequence[int | None],
    method: str,
) -> dict[str, Any]:
    if len(rows) != len(ranks):
        raise RuntimeError("Row/rank length mismatch")
    author_total: Counter[str] = Counter()
    author_top1: Counter[str] = Counter()
    top1 = top3 = top5 = missing = 0
    reciprocal = 0.0
    observed: list[int] = []
    for row, rank in zip(rows, ranks):
        author = str(row["author"])
        author_total[author] += 1
        if rank is None:
            missing += 1
            continue
        r = int(rank)
        observed.append(r)
        reciprocal += 1.0 / r
        if r == 1:
            top1 += 1
            author_top1[author] += 1
        if r <= 3:
            top3 += 1
        if r <= 5:
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
        "mean_rank_given_top10": statistics.fmean(observed) if observed else None,
        "per_author_top1": per_author,
    }


def transition_counts_from_ranks(
    base: Sequence[int | None], new: Sequence[int | None]
) -> dict[str, int]:
    if len(base) != len(new):
        raise RuntimeError("Transition rank length mismatch")
    out = {"n": 0, "rescue": 0, "harm": 0, "unchanged_correct": 0, "unchanged_wrong": 0}
    for before_rank, after_rank in zip(base, new):
        out["n"] += 1
        before = before_rank == 1
        after = after_rank == 1
        if not before and after:
            out["rescue"] += 1
        elif before and not after:
            out["harm"] += 1
        elif before and after:
            out["unchanged_correct"] += 1
        else:
            out["unchanged_wrong"] += 1
    out["net"] = out["rescue"] - out["harm"]
    return out


def recovery_summary_from_ranks(
    rows: Sequence[Mapping[str, Any]], ranks: Sequence[int | None]
) -> dict[str, Any]:
    if len(rows) != len(ranks):
        raise RuntimeError("Recovery rank length mismatch")
    selected: list[int | None] = []
    generic_missing_n = 0
    for row, rank in zip(rows, ranks):
        if not bool(row["generic_missing"]):
            continue
        generic_missing_n += 1
        if bool(row["gold_in_personal_k5"]):
            selected.append(None if rank is None else int(rank))
    denom = len(selected)
    def count_at(k: int) -> int:
        return sum(r is not None and r <= k for r in selected)
    recovered = [r for r in selected if r is not None and r <= 10]
    reciprocal = sum(0.0 if r is None else 1.0 / r for r in selected)
    return {
        "generic_missing_n": generic_missing_n,
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
        "mean_recovered_rank": statistics.fmean(recovered) if recovered else None,
    }


def choose_joint(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        raise RuntimeError("Cannot select from empty joint grid")
    return dict(max(
        results,
        key=lambda row: (
            float(row["metrics"]["macro_author_top1"]),
            float(row["metrics"]["mrr_at_10"]),
            -float(row["lambda_c"]),
            -float(row["gamma_f"]),
        ),
    ))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "K5 joint calibration: boundary + gamma_F*F_PV + lambda_C*C "
            "for F-only, Margin, Entropy, Dual"
        )
    )
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--candidate-surface", type=Path, required=True)
    parser.add_argument("--frequency-pv1-predictions", type=Path, required=True)
    parser.add_argument("--adaptive-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--gammas",
        type=parse_float_grid,
        default=DEFAULT_GAMMAS,
        help="Comma-separated gamma_F grid; default: 1,2,3,4,5,6,8",
    )
    parser.add_argument(
        "--lambdas",
        type=parse_float_grid,
        default=DEFAULT_LAMBDAS,
        help="Comma-separated lambda_C grid; default: 0,.25,.5,1,2,4",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    gammas: tuple[float, ...] = args.gammas
    lambdas: tuple[float, ...] = args.lambdas
    if DEFAULT_GAMMA_F not in gammas:
        raise RuntimeError("Gamma grid must include 4 for the validated K5 control")
    if 0.0 not in lambdas:
        raise RuntimeError("Lambda grid must include 0 for F-only and K5 control")

    args.output_root.mkdir(parents=True, exist_ok=True)
    provenance = verify_inputs(args)
    selected_ngram_key = str(provenance["selected_k5_interpolated_key"])

    print("=== K5 JOINT FREQUENCY + CONCENTRATION CALIBRATION ===")
    print("formula: boundary + gamma_F * F_PV(c) + lambda_C * C(q)")
    print(f"gamma_F grid: {gammas}")
    print(f"lambda_C grid: {lambdas}")
    print("families: F-only, Margin, Entropy, Dual=sqrt(Margin*Entropy)")
    print(f"frozen K5 Interpolated method: {selected_ngram_key}")
    print("K: 5 only; all available Personal K5 candidates admitted")
    print("Gold used for scoring/selection: false")
    print("Dev3000 used: false")
    print("Test used: false\n")

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

    f_labels = {g: f"Fonly@gamma={g:g}" for g in gammas}
    joint_labels = {
        (family, g, l): f"{family}@gamma={g:g}@lambda={l:g}"
        for family in FAMILIES for g in gammas for l in lambdas if l > 0
    }
    rank_arrays: dict[str, list[int | None]] = {
        label: [] for label in list(f_labels.values()) + list(joint_labels.values())
    }

    base_rows: list[dict[str, Any]] = []
    pv1_ranks: list[int | None] = []
    control_k5_ranks: list[int | None] = []
    pv1_rank_mismatch_n = 0
    pv1_top10_mismatch_n = 0
    pv1_mismatch_examples: list[dict[str, Any]] = []
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
            raise RuntimeError(f"Frozen Personal K5 != NGram K5 at {row_id}")
        if len(ngram_support) != len(personal_k5):
            raise RuntimeError(f"NGram support length mismatch at {row_id}")
        if ngram_support and not math.isclose(sum(ngram_support), 1.0, rel_tol=1e-7, abs_tol=1e-7):
            raise RuntimeError(f"NGram support does not sum to 1 at {row_id}")
        if ngram_row.get("gold_used_for_scoring") is not False:
            raise RuntimeError(f"NGram row has invalid Gold provenance at {row_id}")

        visible = history.visible(
            author=author,
            position=int(row["chronological_position"]),
            pinyin=pinyin,
        )
        counts = Counter(record.target for record in visible)
        dist = distribution_features(counts)
        for record in personal_records:
            target = str(record["text"])
            if int(counts.get(target, 0)) != int(record["frequency_count"]):
                raise RuntimeError(f"Personal frequency count drift at {row_id} target={target!r}")

        order = ngram_rank_order(personal_k5, ngram_support) if personal_k5 else []
        selected_personal: list[dict[str, Any]] = []
        for ngram_rank, index in enumerate(order, start=1):
            target = personal_k5[index]
            selected_personal.append({
                "target": target,
                "frequency_support": float(frequency_support[target]),
                "frequency_count": int(personal_records[index]["frequency_count"]),
                "original_frequency_rank": index + 1,
                "ngram_rank": ngram_rank,
            })

        # Exact original PV1 reproduction.
        original_selected: list[dict[str, Any]] = []
        if personal_k5:
            target = personal_k5[0]
            original_selected.append({
                "target": target,
                "frequency_support": float(frequency_support[target]),
                "frequency_count": int(personal_records[0]["frequency_count"]),
                "original_frequency_rank": 1,
                "ngram_rank": 1,
            })
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
        reproduced_rank = rank_of(reproduced_pv1, gold)
        if reproduced_texts != frozen_texts:
            pv1_top10_mismatch_n += 1
        if reproduced_rank != frozen.get("pv1_rank"):
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
        if (reproduced_texts != frozen_texts or reproduced_rank != frozen.get("pv1_rank") or score_mismatch) and len(pv1_mismatch_examples) < 20:
            pv1_mismatch_examples.append({
                "row_id": row_id,
                "gold": gold,
                "reproduced_texts": reproduced_texts,
                "frozen_texts": frozen_texts,
                "reproduced_rank": reproduced_rank,
                "frozen_rank": frozen.get("pv1_rank"),
                "score_mismatch": score_mismatch,
            })

        base_rows.append({
            "row_id": row_id,
            "author": author,
            "gold": gold,
            "generic_missing": bool(frozen.get("generic_missing", surface.get("generic_missing", False))),
            "gold_in_personal_k5": gold in personal_k5_set,
            **dist,
        })
        pv1_ranks.append(frozen.get("pv1_rank"))

        for gamma in gammas:
            ranking = additive_merge(
                frequency_candidates=frequency_candidates,
                selected_personal=selected_personal,
                gamma_f=float(gamma),
                lambda_c=0.0,
                concentration=0.0,
                selector_label=f"k5_fonly_gamma_{gamma:g}",
            )
            rank = rank_of(ranking, gold)
            rank_arrays[f_labels[gamma]].append(rank)
            if gamma == DEFAULT_GAMMA_F:
                control_k5_ranks.append(rank)

        for family in FAMILIES:
            concentration = concentration_for_family(dist, family)
            for gamma in gammas:
                for lambda_c in lambdas:
                    if lambda_c <= 0:
                        continue
                    label = joint_labels[(family, gamma, lambda_c)]
                    ranking = additive_merge(
                        frequency_candidates=frequency_candidates,
                        selected_personal=selected_personal,
                        gamma_f=float(gamma),
                        lambda_c=float(lambda_c),
                        concentration=float(concentration),
                        selector_label=(
                            f"k5_{family.lower()}_gamma_{gamma:g}_lambda_{lambda_c:g}"
                        ),
                    )
                    rank_arrays[label].append(rank_of(ranking, gold))

        if args.progress_every > 0 and (number % args.progress_every == 0 or number == len(val_by_id)):
            print(f"ROWS {number}/{len(val_by_id)} elapsed={time.perf_counter()-started:.1f}s", flush=True)

    pv1_ok = pv1_rank_mismatch_n == 0 and pv1_top10_mismatch_n == 0 and not pv1_mismatch_examples
    pv1_audit = {
        "status": "passed" if pv1_ok else "failed",
        "rows": len(base_rows),
        "pv1_rank_mismatch_n": pv1_rank_mismatch_n,
        "pv1_top10_candidate_order_mismatch_n": pv1_top10_mismatch_n,
        "examples": pv1_mismatch_examples,
    }
    write_json(args.output_root / "pv1_reproduction_audit.json", pv1_audit)
    if not pv1_ok:
        raise RuntimeError("PV1 exact reproduction failed")

    actual_eligible = sum(count for k, count in candidate_count_distribution.items() if k > 0)
    actual_pairs = sum(k * count for k, count in candidate_count_distribution.items())
    if actual_eligible != EXPECTED_K5_ELIGIBLE or actual_pairs != EXPECTED_K5_PAIRS:
        raise RuntimeError(
            f"K5 surface drift: eligible={actual_eligible}/{EXPECTED_K5_ELIGIBLE} "
            f"pairs={actual_pairs}/{EXPECTED_K5_PAIRS}"
        )
    generic_missing_n = sum(bool(row["generic_missing"]) for row in base_rows)
    recoverable_n = sum(bool(row["generic_missing"]) and bool(row["gold_in_personal_k5"]) for row in base_rows)
    if generic_missing_n != EXPECTED_GENERIC_MISSING or recoverable_n != EXPECTED_K5_RECOVERABLE_MISSING:
        raise RuntimeError(
            f"Recovery population drift: generic_missing={generic_missing_n} recoverable={recoverable_n}"
        )

    pv1_metrics = metric_summary_from_ranks(base_rows, pv1_ranks, "PV1")
    k5_control_metrics = metric_summary_from_ranks(base_rows, control_k5_ranks, "K5@gamma4,lambda0")
    mismatches: dict[str, Any] = {}
    for key, expected in EXPECTED_K5_BASELINE.items():
        actual = float(k5_control_metrics[key])
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=5e-7):
            mismatches[key] = {"expected": expected, "actual": actual}
    control_audit = {
        "expected": EXPECTED_K5_BASELINE,
        "actual": {key: k5_control_metrics[key] for key in EXPECTED_K5_BASELINE},
        "mismatches": mismatches,
        "passed": not mismatches,
    }
    write_json(args.output_root / "k5_gamma4_lambda0_control.json", control_audit)
    if mismatches:
        raise RuntimeError("Validated K5 gamma=4, lambda=0 control failed")

    # Build F-only reports first.
    f_only_results: list[dict[str, Any]] = []
    control_trans = transition_counts_from_ranks(pv1_ranks, control_k5_ranks)
    for gamma in gammas:
        ranks = rank_arrays[f_labels[gamma]]
        metrics = metric_summary_from_ranks(base_rows, ranks, f_labels[gamma])
        report = {
            "family": "F-only",
            "gamma_f": float(gamma),
            "lambda_c": 0.0,
            "metrics": metrics,
            "delta_vs_gamma4_k5": delta_metrics(metrics, k5_control_metrics),
            "delta_vs_pv1": delta_metrics(metrics, pv1_metrics),
            "gamma4_k5_to_method": transition_counts_from_ranks(control_k5_ranks, ranks),
            "pv1_to_method": transition_counts_from_ranks(pv1_ranks, ranks),
            "recovery": recovery_summary_from_ranks(base_rows, ranks),
        }
        f_only_results.append(report)
    best_f_only = choose_joint(f_only_results)
    best_f_ranks = rank_arrays[f_labels[float(best_f_only["gamma_f"])]]
    best_f_metrics = best_f_only["metrics"]

    grid_results: dict[str, list[dict[str, Any]]] = {"F-only": f_only_results}
    selected_by_family: dict[str, Any] = {}
    selected_positive_lambda_by_family: dict[str, Any] = {}

    print("\n=== F-ONLY GAMMA GRID ===")
    for report in f_only_results:
        m = report["metrics"]
        tr = report["gamma4_k5_to_method"]
        print(
            f"gamma={report['gamma_f']:>4g}  Macro={m['macro_author_top1']:.6f} "
            f"dK5={report['delta_vs_gamma4_k5']['macro_author_top1']:+.6f} "
            f"Top3={m['top3']:.6f} MRR10={m['mrr_at_10']:.6f} "
            f"Missing10={m['missing10']:.6f} K5->new net={tr['net']:+d}",
            flush=True,
        )

    for family in FAMILIES:
        family_results: list[dict[str, Any]] = []
        for gamma in gammas:
            # lambda=0 is exactly F-only for this gamma.
            f_report = next(r for r in f_only_results if float(r["gamma_f"]) == float(gamma))
            zero_report = dict(f_report)
            zero_report["family"] = family
            zero_report["concentration_active"] = False
            zero_report["delta_vs_best_f_only"] = delta_metrics(zero_report["metrics"], best_f_metrics)
            zero_report["best_f_only_to_method"] = transition_counts_from_ranks(
                best_f_ranks, rank_arrays[f_labels[gamma]]
            )
            family_results.append(zero_report)

            for lambda_c in lambdas:
                if lambda_c <= 0:
                    continue
                label = joint_labels[(family, gamma, lambda_c)]
                ranks = rank_arrays[label]
                metrics = metric_summary_from_ranks(base_rows, ranks, label)
                family_results.append({
                    "family": family,
                    "gamma_f": float(gamma),
                    "lambda_c": float(lambda_c),
                    "concentration_active": True,
                    "metrics": metrics,
                    "delta_vs_gamma4_k5": delta_metrics(metrics, k5_control_metrics),
                    "delta_vs_best_f_only": delta_metrics(metrics, best_f_metrics),
                    "delta_vs_pv1": delta_metrics(metrics, pv1_metrics),
                    "gamma4_k5_to_method": transition_counts_from_ranks(control_k5_ranks, ranks),
                    "best_f_only_to_method": transition_counts_from_ranks(best_f_ranks, ranks),
                    "pv1_to_method": transition_counts_from_ranks(pv1_ranks, ranks),
                    "recovery": recovery_summary_from_ranks(base_rows, ranks),
                })
        grid_results[family] = family_results
        selected_by_family[family] = choose_joint(family_results)
        positive = [r for r in family_results if float(r["lambda_c"]) > 0]
        selected_positive_lambda_by_family[family] = choose_joint(positive)

    family_order = {"F-only": 0, "Margin": 1, "Entropy": 2, "Dual": 3}
    candidate_best = [best_f_only, *selected_by_family.values()]
    development_best = dict(max(
        candidate_best,
        key=lambda row: (
            float(row["metrics"]["macro_author_top1"]),
            float(row["metrics"]["mrr_at_10"]),
            -family_order[str(row["family"])],
            -float(row["lambda_c"]),
            -float(row["gamma_f"]),
        ),
    ))

    # Locate selected rank arrays for compact row-level output.
    def ranks_for_report(report: Mapping[str, Any]) -> Sequence[int | None]:
        fam = str(report["family"])
        g = float(report["gamma_f"])
        l = float(report["lambda_c"])
        if fam == "F-only" or l == 0.0:
            return rank_arrays[f_labels[g]]
        return rank_arrays[joint_labels[(fam, g, l)]]

    selected_predictions = []
    dev_ranks = ranks_for_report(development_best)
    selected_ranks_by_family = {fam: ranks_for_report(rep) for fam, rep in selected_by_family.items()}
    for i, row in enumerate(base_rows):
        item = {
            "row_id": row["row_id"],
            "author": row["author"],
            "gold": row["gold"],
            "pv1_rank": pv1_ranks[i],
            "k5_gamma4_lambda0_rank": control_k5_ranks[i],
            "best_f_only_rank": best_f_ranks[i],
            "development_best_rank": dev_ranks[i],
            "conc_margin": row["conc_margin"],
            "conc_entropy": row["conc_entropy"],
            "conc_dual": row["conc_dual"],
        }
        for fam, ranks in selected_ranks_by_family.items():
            item[f"selected_{fam.lower()}_rank"] = ranks[i]
        selected_predictions.append(item)
    selected_predictions_path = args.output_root / "selected_predictions.jsonl"
    write_jsonl(selected_predictions_path, selected_predictions)

    comparison = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_pv1_ngram_k5_joint_frequency_concentration_v1",
        "formula": "boundary + gamma_F * F_PV(candidate) + lambda_C * C(query)",
        "gamma_grid": list(gammas),
        "lambda_grid": list(lambdas),
        "families": {
            "F-only": "lambda_C=0",
            "Margin": "p1-p2",
            "Entropy": "1-normalized entropy",
            "Dual": "sqrt(Margin*Entropy)",
        },
        "selection_policy": (
            "Macro-author Top1, then MRR@10, then lower lambda_C, then lower gamma_F; "
            "cross-family exact ties prefer F-only then Margin then Entropy then Dual"
        ),
        "rows": len(base_rows),
        "candidate_count_distribution": {
            str(k): v for k, v in sorted(candidate_count_distribution.items())
        },
        "pv1_reproduction_audit": pv1_audit,
        "k5_gamma4_lambda0_control": control_audit,
        "baselines": {
            "PV1": {
                "metrics": pv1_metrics,
                "recovery": recovery_summary_from_ranks(base_rows, pv1_ranks),
            },
            "K5_gamma4_lambda0": {
                "metrics": k5_control_metrics,
                "pv1_to_k5": control_trans,
                "recovery": recovery_summary_from_ranks(base_rows, control_k5_ranks),
            },
        },
        "best_f_only": best_f_only,
        "grid_results": grid_results,
        "selected_by_family": selected_by_family,
        "selected_positive_lambda_by_family": selected_positive_lambda_by_family,
        "development_best": development_best,
        "gold_used_for_scoring_or_candidate_selection": False,
        "gold_used_for_train_val_selection_evaluation_only": True,
        "dev3000_used": False,
        "test_used": False,
        "provenance": provenance,
    }
    comparison_path = args.output_root / "comparison.json"
    write_json(comparison_path, comparison)
    write_json(args.output_root / "grid_results.json", grid_results)

    artifacts = [
        args.output_root / "pv1_reproduction_audit.json",
        args.output_root / "k5_gamma4_lambda0_control.json",
        args.output_root / "grid_results.json",
        selected_predictions_path,
        comparison_path,
    ]
    write_json(
        args.output_root / "artifact_checksums.json",
        {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in artifacts},
    )

    print("\n=== BASELINES ===")
    print(
        f"PV1                    Macro={pv1_metrics['macro_author_top1']:.6f} "
        f"Top3={pv1_metrics['top3']:.6f} MRR10={pv1_metrics['mrr_at_10']:.6f} "
        f"Missing10={pv1_metrics['missing10']:.6f}"
    )
    print(
        f"K5 gamma=4 lambda=0    Macro={k5_control_metrics['macro_author_top1']:.6f} "
        f"Top3={k5_control_metrics['top3']:.6f} MRR10={k5_control_metrics['mrr_at_10']:.6f} "
        f"Missing10={k5_control_metrics['missing10']:.6f} "
        f"PV1->K5 rescue={control_trans['rescue']} harm={control_trans['harm']} net={control_trans['net']:+d}"
    )

    print("\n=== BEST F-ONLY ===")
    bf = best_f_only
    bm = bf["metrics"]
    btr = bf["gamma4_k5_to_method"]
    br = bf["recovery"]
    print(
        f"gamma={bf['gamma_f']:g} lambda=0  Macro={bm['macro_author_top1']:.6f} "
        f"dK5={bf['delta_vs_gamma4_k5']['macro_author_top1']:+.6f} "
        f"Top3={bm['top3']:.6f} MRR10={bm['mrr_at_10']:.6f} Missing10={bm['missing10']:.6f} "
        f"K5->new rescue={btr['rescue']} harm={btr['harm']} net={btr['net']:+d} "
        f"Rec@10={br['recovered_at_10']:.4f}"
    )

    print("\n=== SELECTED BY CONCENTRATION FAMILY (lambda may be 0) ===")
    for family in FAMILIES:
        rep = selected_by_family[family]
        m = rep["metrics"]
        tr = rep["best_f_only_to_method"]
        rec = rep["recovery"]
        print(
            f"{family:<8} gamma={rep['gamma_f']:g} lambda={rep['lambda_c']:g}  "
            f"Macro={m['macro_author_top1']:.6f} "
            f"dBestF={rep['delta_vs_best_f_only']['macro_author_top1']:+.6f} "
            f"Top3={m['top3']:.6f} MRR10={m['mrr_at_10']:.6f} Missing10={m['missing10']:.6f} "
            f"BestF->new rescue={tr['rescue']} harm={tr['harm']} net={tr['net']:+d} "
            f"Rec@10={rec['recovered_at_10']:.4f}"
        )

    print("\n=== BEST POSITIVE-LAMBDA POINT BY FAMILY ===")
    for family in FAMILIES:
        rep = selected_positive_lambda_by_family[family]
        m = rep["metrics"]
        print(
            f"{family:<8} gamma={rep['gamma_f']:g} lambda={rep['lambda_c']:g}  "
            f"Macro={m['macro_author_top1']:.6f} "
            f"dBestF={rep['delta_vs_best_f_only']['macro_author_top1']:+.6f} "
            f"Top3={m['top3']:.6f} MRR10={m['mrr_at_10']:.6f} Missing10={m['missing10']:.6f}"
        )

    dev = development_best
    dm = dev["metrics"]
    print("\nDEVELOPMENT BEST:")
    print(
        f"{dev['family']} gamma={dev['gamma_f']:g} lambda={dev['lambda_c']:g} "
        f"Macro={dm['macro_author_top1']:.6f} Top3={dm['top3']:.6f} "
        f"MRR10={dm['mrr_at_10']:.6f} Missing10={dm['missing10']:.6f}"
    )
    print("PV1 reproduction: PASSED exact Top10/rank/score cross-check")
    print("K5 gamma=4 lambda=0 control: PASSED validated metric cross-check")
    print(f"Saved: {comparison_path}")
    print("Gold used for scoring/selection: false")
    print("Dev3000 used: false")
    print("Test used: false")


if __name__ == "__main__":
    main()
