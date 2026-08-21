"""Two fixed NGram+Choice-Share anchors with additive Entropy concentration.

Research question
-----------------
Starting from the two already-observed NGram-CS operating points, does a query-level
Entropy concentration term add useful recovery strength without sacrificing final
ranked-list quality?

The two frozen anchors are exact algebraic rewrites of the prior interpolation run:

    Top3Anchor    = Boundary + 6 * P_N(c) + 2 * CS(c)
                  = Boundary + 8 * (0.75 * P_N(c) + 0.25 * CS(c))

    BalancedAnchor= Boundary + 4 * P_N(c) + 4 * CS(c)
                  = Boundary + 8 * (0.50 * P_N(c) + 0.50 * CS(c))

This experiment adds only:

    + lambda_E * C_E(q)

where C_E(q)=1-H_norm is shared by every personal candidate in the query. Therefore
it changes the Personal block's strength relative to Generic but does not change
within-Personal ordering.

Frozen semantics
----------------
- Same standardized Clean3 Train-Fit / Train-Val inputs.
- Same strictly-prior same-author H5000-before-exact-Pinyin filtering.
- Same frozen Personal-only K5 surface.
- Same frozen Frequency-reranked Generic Top10.
- Same already-selected K5 Interpolated-NGram score cache; no NGram recomputation or retuning.
- Gold is used only for Train-Val evaluation/diagnostics, never scoring.
- Dev3000 and Test are not read.

Controls
--------
lambda_E=0 MUST reproduce the previous alpha=.25, lambda_D=8 and alpha=.5,
lambda_D=8 end-to-end metrics. The runner aborts if either control differs.

Metrics
-------
For every lambda_E point and both anchors, report:
- Macro-author Top1, Micro Top1, Top3, Top5, MRR@10, Missing@10, mean rank when present
- per-author Top1/Top3/Top5/MRR/Missing
- PV1->method and anchor(lambda_E=0)->method rescue/harm/net
- Rec@1/3/5/10, Recovery MRR@10, mean recovered rank on the fixed R=4910 population
- Entropy-bin diagnostics

No single objective is silently privileged: the output explicitly reports best points by
Macro Top1, Top3, Top5, MRR, Missing, and Rec@3.
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


EXPECTED_FIT_SHA256 = "162f5c98daa86cc69947571e6d8f20fc401f0a82cdd3fd6e517eb7be2addbdb4"
EXPECTED_VAL_SHA256 = "d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4"
EXPECTED_SURFACE_SHA256 = "205c0ba01cd0678d7a4341c503fa2e74cf126a70182ff582687025e4946764b2"
EXPECTED_FREQUENCY_PV1_SHA256 = "7fd8aa158d8cd50bced36b55610f8d932bc65e3aae1dbbd5bd65907ff1707ea7"
EXPECTED_FIT_ROWS = 144_526
EXPECTED_VAL_ROWS = 34_416
EXPECTED_GENERIC_MISSING = 12_565
EXPECTED_K5_RECOVERABLE_MISSING = 4_910
HISTORY_BUDGET = 5000

DEFAULT_ENTROPY_LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)

# Exact rewrites of prior interpolation operating points at lambda_D=8.
ANCHORS: dict[str, dict[str, float]] = {
    "Top3Anchor": {"beta_ngram": 6.0, "gamma_cs": 2.0, "source_alpha": 0.25, "source_lambda_D": 8.0},
    "BalancedAnchor": {"beta_ngram": 4.0, "gamma_cs": 4.0, "source_alpha": 0.50, "source_lambda_D": 8.0},
}
ANCHOR_ORDER = ("Top3Anchor", "BalancedAnchor")

# Exact historical controls from recovery_ngram_cs_interpolation_k5_v1.
EXPECTED_ANCHOR0_METRICS: dict[str, dict[str, float]] = {
    "Top3Anchor": {
        "macro_author_top1": 0.400212322119,
        "micro_top1": 0.423552998605,
        "top3": 0.615731055323,
        "top5": 0.685669456067,
        "mrr_at_10": 0.534314004929,
        "missing10": 0.244392143189,
    },
    "BalancedAnchor": {
        "macro_author_top1": 0.402627651190,
        "micro_top1": 0.427272198977,
        "top3": 0.612592980009,
        "top5": 0.683461180846,
        "mrr_at_10": 0.535608645481,
        "missing10": 0.245612505811,
    },
}


# ---------------------------------------------------------------------------
# IO / utilities
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
                raise RuntimeError(f"Expected object at {path}:{line_number}")
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
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Empty numeric grid")
    if any(value < 0 for value in values):
        raise ValueError("Grid values must be non-negative")
    return tuple(dict.fromkeys(values))


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


# ---------------------------------------------------------------------------
# Frozen causal history semantics
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

    def visible(self, *, author: str, position: int, pinyin: tuple[str, ...]) -> tuple[HistoryRecord, ...]:
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


# ---------------------------------------------------------------------------
# Candidate / history features
# ---------------------------------------------------------------------------


def extract_personal_k5(row: Mapping[str, Any]) -> tuple[str, ...]:
    for key in ("personal_candidate_texts_top5", "personal_k5"):
        value = row.get(key)
        if isinstance(value, list):
            result = tuple(str(item) for item in value)
            if len(result) > 5:
                raise RuntimeError(f"{key} contains >5 candidates at {row.get('row_id')}")
            return result

    value = row.get("personal_candidates_top5")
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if isinstance(item, str):
                result.append(item)
                continue
            if not isinstance(item, Mapping):
                raise RuntimeError(f"Unexpected Personal-K5 item: {item!r}")
            target = item.get("target", item.get("candidate", item.get("text")))
            if target is None:
                raise RuntimeError(f"Cannot identify Personal-K5 target at {row.get('row_id')}")
            result.append(str(target))
        if len(result) > 5:
            raise RuntimeError(f"personal_candidates_top5 contains >5 at {row.get('row_id')}")
        return tuple(result)

    raise RuntimeError(f"No recognized Personal-K5 field at {row.get('row_id')}")


def distribution_features(counts: Mapping[str, int]) -> dict[str, float | int]:
    positive = sorted((int(v) for v in counts.values() if int(v) > 0), reverse=True)
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
        entropy_norm = max(0.0, min(1.0, entropy / math.log(distinct)))
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


def ngram_rank_order(candidates: Sequence[str], support: Sequence[float]) -> list[int]:
    if len(candidates) != len(support):
        raise RuntimeError("NGram candidate/support length mismatch")
    return sorted(
        range(len(candidates)),
        key=lambda i: (-float(support[i]), i, str(candidates[i])),
    )


def personal_components(*, anchor: str, ngram_support: float, choice_share: float, conc_entropy: float, lambda_entropy: float) -> dict[str, float]:
    if anchor not in ANCHORS:
        raise ValueError(f"Unknown anchor: {anchor}")
    cfg = ANCHORS[anchor]
    ngram_term = float(cfg["beta_ngram"]) * float(ngram_support)
    cs_term = float(cfg["gamma_cs"]) * float(choice_share)
    entropy_term = float(lambda_entropy) * float(conc_entropy)
    return {
        "ngram_term": ngram_term,
        "cs_term": cs_term,
        "entropy_term": entropy_term,
        "recovery_evidence": ngram_term + cs_term + entropy_term,
    }


# ---------------------------------------------------------------------------
# Ranking / metrics
# ---------------------------------------------------------------------------


def rank_merged(
    *,
    frequency_candidates: Sequence[Mapping[str, Any]],
    personal_candidates: Sequence[str],
    ngram_supports: Sequence[float],
    choice_shares: Sequence[float],
    conc_entropy: float,
    anchor: str,
    lambda_entropy: float,
) -> list[dict[str, Any]]:
    if not (len(personal_candidates) == len(ngram_supports) == len(choice_shares)):
        raise RuntimeError("Personal candidate / NGram / CS length mismatch")

    generic_rows: list[dict[str, Any]] = []
    generic_texts: set[str] = set()
    normalized_generic: list[float] = []

    for position, value in enumerate(frequency_candidates, start=1):
        candidate = str(value.get("candidate", value.get("text")))
        if not candidate:
            raise RuntimeError("Cannot identify frozen Frequency Generic candidate")
        if candidate in generic_texts:
            raise RuntimeError(f"Duplicate frozen Generic candidate: {candidate!r}")
        generic_texts.add(candidate)
        normalized_generic.append(float(value["normalized_generic_score"]))
        generic_rows.append({
            "candidate": candidate,
            "source": "generic_frequency",
            "final_score": float(value["final_score"]),
            "generic_rank": int(value.get("generic_rank") or value.get("rank") or position),
            "ngram_rank": None,
        })

    if not generic_rows:
        raise RuntimeError("Frozen Frequency Generic candidate list is empty")

    boundary = min(normalized_generic)
    order = ngram_rank_order(personal_candidates, ngram_supports)
    rank_by_index = {candidate_index: rank for rank, candidate_index in enumerate(order, start=1)}
    rows = list(generic_rows)

    for i, candidate in enumerate(personal_candidates):
        if candidate in generic_texts:
            raise RuntimeError(f"Frozen Personal K5 overlaps Generic: {candidate!r}")
        parts = personal_components(
            anchor=anchor,
            ngram_support=float(ngram_supports[i]),
            choice_share=float(choice_shares[i]),
            conc_entropy=float(conc_entropy),
            lambda_entropy=float(lambda_entropy),
        )
        rows.append({
            "candidate": candidate,
            "source": "personal_k5",
            "final_score": boundary + parts["recovery_evidence"],
            "generic_rank": None,
            "ngram_rank": int(rank_by_index[i]),
            "ngram_support": float(ngram_supports[i]),
            "choice_share": float(choice_shares[i]),
            "conc_entropy": float(conc_entropy),
            "lambda_entropy": float(lambda_entropy),
            **parts,
        })

    rows.sort(key=lambda row: (
        -float(row["final_score"]),
        0 if row["source"] == "generic_frequency" else 1,
        int(row["generic_rank"] or row["ngram_rank"] or 0),
        str(row["candidate"]),
    ))
    return rows[:10]


def rank_of(ranking: Sequence[Mapping[str, Any]], gold: str) -> int | None:
    for rank, row in enumerate(ranking, start=1):
        if str(row["candidate"]) == gold:
            return rank
    return None


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
    predicate=lambda row: True,
) -> dict[str, int]:
    counts = {
        "n": 0,
        "rescue": 0,
        "harm": 0,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
    }
    for row in rows:
        if not predicate(row):
            continue
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

    ranks = [
        None if row.get(rank_key) is None else int(row[rank_key])
        for row in recoverable
    ]

    def count_at(limit: int) -> int:
        return sum(rank is not None and rank <= limit for rank in ranks)

    recovered_ranks = [rank for rank in ranks if rank is not None and rank <= 10]
    recovery_mrr = sum(0.0 if rank is None else 1.0 / rank for rank in ranks)
    denom = len(recoverable)

    return {
        "generic_missing_n": len(generic_missing),
        "recoverable_n": denom,
        "recoverable_rate_given_generic_missing": safe_rate(denom, len(generic_missing)),
        "recovered_at_1_n": count_at(1),
        "recovered_at_3_n": count_at(3),
        "recovered_at_5_n": count_at(5),
        "recovered_at_10_n": count_at(10),
        "recovered_at_1": safe_rate(count_at(1), denom),
        "recovered_at_3": safe_rate(count_at(3), denom),
        "recovered_at_5": safe_rate(count_at(5), denom),
        "recovered_at_10": safe_rate(count_at(10), denom),
        "recovery_mrr_at_10": safe_rate(recovery_mrr, denom),
        "mean_recovered_rank": statistics.fmean(recovered_ranks) if recovered_ranks else None,
    }


def delta_metrics(method: Mapping[str, Any], pv1: Mapping[str, Any]) -> dict[str, float]:
    keys = ("macro_author_top1", "micro_top1", "top3", "top5", "mrr_at_10", "missing10")
    return {key: float(method[key]) - float(pv1[key]) for key in keys}


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
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[conc_bin(float(row[concentration_key]))].append(row)

    output: list[dict[str, Any]] = []
    for label in ("[0,.10)", "[.10,.25)", "[.25,.50)", "[.50,.75)", "[.75,1]"):
        values = grouped.get(label, [])
        trans = transition_counts(values, "pv1_rank", rank_key)
        recovery = recovery_summary(values, rank_key)
        output.append(
            {
                "bin": label,
                "n": len(values),
                "pv1_to_method_rescue": trans["rescue"],
                "pv1_to_method_harm": trans["harm"],
                "pv1_to_method_net": trans["net"],
                "recoverable_n": recovery["recoverable_n"],
                "recovered_at_1": recovery["recovered_at_1"],
                "recovered_at_10": recovery["recovered_at_10"],
            }
        )
    return output


# ---------------------------------------------------------------------------
# Input verification / frozen NGram selection
# ---------------------------------------------------------------------------


def verify_frozen_inputs(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "fit": args.fit,
        "val": args.val,
        "candidate_surface": args.candidate_surface,
        "frequency_pv1_predictions": args.frequency_pv1_predictions,
    }
    expected = {
        "fit": EXPECTED_FIT_SHA256,
        "val": EXPECTED_VAL_SHA256,
        "candidate_surface": EXPECTED_SURFACE_SHA256,
        "frequency_pv1_predictions": EXPECTED_FREQUENCY_PV1_SHA256,
    }
    hashes: dict[str, str] = {}
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        hashes[label] = digest
        if digest != expected[label]:
            raise RuntimeError(
                f"SHA mismatch for {label}:\nexpected={expected[label]}\nactual={digest}\npath={path}"
            )

    adaptive_score_path = args.adaptive_dir / "scores.jsonl"
    adaptive_comparison_path = args.adaptive_dir / "comparison.json"
    if not adaptive_score_path.is_file():
        raise FileNotFoundError(adaptive_score_path)
    if not adaptive_comparison_path.is_file():
        raise FileNotFoundError(adaptive_comparison_path)

    comparison = read_json(adaptive_comparison_path)
    provenance = comparison.get("provenance", {})
    if provenance.get("fit_sha256") != EXPECTED_FIT_SHA256:
        raise RuntimeError("Adaptive NGram comparison uses another Train-Fit artifact")
    if provenance.get("val_sha256") != EXPECTED_VAL_SHA256:
        raise RuntimeError("Adaptive NGram comparison uses another Train-Val artifact")
    if provenance.get("frozen_candidate_surface_sha256") != EXPECTED_SURFACE_SHA256:
        raise RuntimeError("Adaptive NGram comparison uses another frozen Personal-K5 surface")
    if provenance.get("candidate_selection_used_gold") is not False:
        raise RuntimeError("Adaptive NGram candidate selection provenance is not Gold-free")
    if provenance.get("gold_used_for_scoring") is not False:
        raise RuntimeError("Adaptive NGram scoring provenance is not Gold-free")
    if provenance.get("dev3000_used") is not False or provenance.get("test_used") is not False:
        raise RuntimeError("Adaptive NGram artifact crossed Dev3000/Test boundary")

    selected_by_k = comparison.get("selected_by_k", {})
    if "K5" not in selected_by_k:
        raise RuntimeError("Adaptive comparison has no K5 selected methods")
    selected_key = str(selected_by_k["K5"].get("best_interpolated", ""))
    if not selected_key.startswith("K5|Interpolated|"):
        raise RuntimeError(f"Unexpected selected K5 Interpolated key: {selected_key!r}")

    recorded_scores_sha = provenance.get("scores_sha256")
    actual_scores_sha = sha256_file(adaptive_score_path)
    if recorded_scores_sha and recorded_scores_sha != actual_scores_sha:
        raise RuntimeError(
            "Adaptive scores.jsonl SHA differs from comparison provenance:\n"
            f"recorded={recorded_scores_sha}\nactual={actual_scores_sha}"
        )

    return {
        "paths": {
            **{key: str(path) for key, path in paths.items()},
            "adaptive_scores": str(adaptive_score_path),
            "adaptive_comparison": str(adaptive_comparison_path),
        },
        "sha256": {
            **hashes,
            "adaptive_scores": actual_scores_sha,
            "adaptive_comparison": sha256_file(adaptive_comparison_path),
        },
        "selected_k5_interpolated_key": selected_key,
        "adaptive_selection_population": comparison.get("selection_population"),
        "adaptive_selection_metric": comparison.get("selection_metric"),
    }



def per_author_rank_metrics(rows: Sequence[Mapping[str, Any]], rank_key: str) -> dict[str, Any]:
    grouped: dict[str, list[int | None]] = defaultdict(list)
    for row in rows:
        rank = row.get(rank_key)
        grouped[str(row["author"])].append(None if rank is None else int(rank))

    output: dict[str, Any] = {}
    for author, ranks in sorted(grouped.items()):
        n = len(ranks)
        present = [r for r in ranks if r is not None]
        output[author] = {
            "n": n,
            "top1": safe_rate(sum(r == 1 for r in ranks), n),
            "top3": safe_rate(sum(r is not None and r <= 3 for r in ranks), n),
            "top5": safe_rate(sum(r is not None and r <= 5 for r in ranks), n),
            "mrr_at_10": safe_rate(sum(0.0 if r is None else 1.0 / r for r in ranks), n),
            "missing10": safe_rate(sum(r is None for r in ranks), n),
            "mean_rank_given_top10": statistics.fmean(present) if present else None,
        }
    return output


def verify_anchor_control(anchor: str, metrics: Mapping[str, Any]) -> None:
    expected = EXPECTED_ANCHOR0_METRICS[anchor]
    failures: list[str] = []
    for key, target in expected.items():
        actual = float(metrics[key])
        if not math.isclose(actual, float(target), rel_tol=0.0, abs_tol=5e-12):
            failures.append(f"{key}: expected={target:.12f} actual={actual:.12f}")
    if failures:
        raise RuntimeError(
            f"lambda_E=0 control failed for {anchor}; the new runner is not comparable to the frozen interpolation experiment:\n"
            + "\n".join(failures)
        )


def objective_best(rows: Sequence[Mapping[str, Any]], objective: str) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("No rows for objective selection")

    def key(row: Mapping[str, Any]) -> tuple[float, ...]:
        m = row["metrics"]
        r = row["recovery"]
        lam = float(row["lambda_entropy"])
        if objective == "macro_top1":
            return (float(m["macro_author_top1"]), float(m["mrr_at_10"]), -lam)
        if objective == "top3":
            return (float(m["top3"]), float(m["mrr_at_10"]), float(m["macro_author_top1"]), -lam)
        if objective == "top5":
            return (float(m["top5"]), float(m["mrr_at_10"]), float(m["top3"]), -lam)
        if objective == "mrr":
            return (float(m["mrr_at_10"]), float(m["top3"]), float(m["macro_author_top1"]), -lam)
        if objective == "missing":
            return (-float(m["missing10"]), float(m["top3"]), float(m["mrr_at_10"]), -lam)
        if objective == "recovery_top3":
            return (float(r["recovered_at_3"]), float(r["recovery_mrr_at_10"]), float(m["top3"]), -lam)
        raise ValueError(objective)

    return dict(max(rows, key=key))


def delta_against(metrics: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, float]:
    keys = ("macro_author_top1", "micro_top1", "top3", "top5", "mrr_at_10", "missing10", "mean_rank_given_top10")
    out: dict[str, float] = {}
    for key in keys:
        if metrics.get(key) is None or base.get(key) is None:
            continue
        out[key] = float(metrics[key]) - float(base[key])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Two fixed NGram+CS anchors with additive Entropy concentration")
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--candidate-surface", type=Path, required=True)
    parser.add_argument("--frequency-pv1-predictions", type=Path, required=True)
    parser.add_argument(
        "--adaptive-dir", type=Path, required=True,
        help="adaptive_ngram_top10_v1 directory containing scores.jsonl and comparison.json",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--entropy-lambdas",
        default=",".join(str(v) for v in DEFAULT_ENTROPY_LAMBDAS),
        help="comma-separated lambda_E grid; MUST include 0",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    entropy_lambdas = parse_float_grid(args.entropy_lambdas)
    if 0.0 not in entropy_lambdas:
        raise RuntimeError("Entropy lambda grid must include 0 for exact anchor controls")

    args.output_root.mkdir(parents=True, exist_ok=True)
    provenance = verify_frozen_inputs(args)
    selected_ngram_key = str(provenance["selected_k5_interpolated_key"])

    print("=== NGRAM + CS + ENTROPY: TWO FROZEN ANCHORS ===")
    print(f"Frozen Interpolated NGram: {selected_ngram_key}")
    print(f"lambda_E grid: {entropy_lambdas}")
    for name in ANCHOR_ORDER:
        cfg = ANCHORS[name]
        print(f"{name}: Boundary + {cfg['beta_ngram']:g}*P_N + {cfg['gamma_cs']:g}*CS + lambda_E*C_E")
    print("Gold used for scoring/features: false")
    print("Dev3000 used: false")
    print("Test used: false")
    print()

    fit_rows = read_jsonl(args.fit)
    val_rows = read_jsonl(args.val)
    surface_rows = index_rows(read_jsonl(args.candidate_surface), "candidate surface")
    pred_rows = index_rows(read_jsonl(args.frequency_pv1_predictions), "Frequency/PV1 predictions")
    ngram_rows = index_rows(read_jsonl(args.adaptive_dir / "scores.jsonl"), "adaptive NGram scores")
    val = index_rows(val_rows, "Train-Val")

    if len(fit_rows) != EXPECTED_FIT_ROWS:
        raise RuntimeError(f"Unexpected Train-Fit rows: {len(fit_rows)}")
    if len(val_rows) != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"Unexpected Train-Val rows: {len(val_rows)}")
    if not (set(val) == set(surface_rows) == set(pred_rows) == set(ngram_rows)):
        raise RuntimeError("Train-Val / surface / PV1 / adaptive-NGram row IDs differ")

    history_records = [to_history_record(row) for row in fit_rows]
    history_records.extend(to_history_record(row) for row in val_rows)
    history = CausalHistoryIndex(history_records)

    feature_rows: list[dict[str, Any]] = []
    candidate_count_distribution: Counter[int] = Counter()
    started = time.perf_counter()

    for number, row_id in enumerate(sorted(val), start=1):
        row = val[row_id]
        pred = pred_rows[row_id]
        surface = surface_rows[row_id]
        ngram_row = ngram_rows[row_id]

        author = str(row["author"])
        gold = str(row.get("target", row.get("gold")))
        pinyin = tuple(str(v) for v in row["pinyin_segments"])
        visible = history.visible(author=author, position=int(row["chronological_position"]), pinyin=pinyin)
        counts = Counter(record.target for record in visible)
        dist = distribution_features(counts)

        personal_k5 = extract_personal_k5(surface)
        candidate_count_distribution[len(personal_k5)] += 1
        frequency_candidates = list(pred.get("frequency_candidates", []))
        if not frequency_candidates:
            raise RuntimeError(f"Missing frequency_candidates at {row_id}")
        generic_texts = {str(v.get("candidate", v.get("text"))) for v in frequency_candidates}
        overlap = generic_texts.intersection(personal_k5)
        if overlap:
            raise RuntimeError(f"Personal K5 overlaps Generic at {row_id}: {sorted(overlap)}")

        methods = ngram_row.get("methods")
        if not isinstance(methods, Mapping) or selected_ngram_key not in methods:
            raise RuntimeError(f"Missing selected NGram method at {row_id}: {selected_ngram_key}")
        selected_method = methods[selected_ngram_key]
        ngram_candidates = tuple(str(v) for v in selected_method.get("candidates", []))
        ngram_supports = tuple(float(v) for v in selected_method.get("support", []))
        if ngram_candidates != personal_k5:
            raise RuntimeError(f"Frozen K5 != selected NGram candidate order at {row_id}")
        if len(ngram_supports) != len(personal_k5):
            raise RuntimeError(f"NGram support length mismatch at {row_id}")
        if ngram_supports and not math.isclose(sum(ngram_supports), 1.0, rel_tol=1e-7, abs_tol=1e-7):
            raise RuntimeError(f"Selected NGram support does not sum to 1 at {row_id}")
        if ngram_row.get("gold_used_for_scoring") is not False:
            raise RuntimeError(f"Adaptive NGram row has invalid Gold provenance at {row_id}")

        history_total = int(dist["same_pinyin_history_count"])
        personal_counts = [int(counts.get(c, 0)) for c in personal_k5]
        choice_shares = [count / history_total if history_total > 0 else 0.0 for count in personal_counts]
        if any(count <= 0 for count in personal_counts):
            raise RuntimeError(f"Frozen Personal K5 lacks legal visible same-Pinyin support at {row_id}")

        generic_missing = bool(row.get("generic_missing", pred.get("generic_missing", False)))
        feature_rows.append({
            "row_id": row_id,
            "author": author,
            "gold": gold,
            "generic_rank": pred.get("generic_rank", row.get("generic_rank")),
            "frequency_rank": pred.get("frequency_rank"),
            "pv1_rank": pred.get("pv1_rank"),
            "generic_missing": generic_missing,
            "gold_in_personal_k5": gold in set(personal_k5),
            "personal_k5": list(personal_k5),
            "choice_shares": choice_shares,
            "ngram_supports": list(ngram_supports),
            "conc_entropy": float(dist["conc_entropy"]),
            "entropy_norm": float(dist["entropy_norm"]),
            "same_pinyin_history_count": history_total,
            "distinct_targets": int(dist["distinct_targets"]),
            "frequency_candidates": frequency_candidates,
        })

        if args.progress_every > 0 and (number % args.progress_every == 0 or number == len(val)):
            print(f"FEATURES {number}/{len(val)} elapsed={time.perf_counter()-started:.1f}s", flush=True)

    generic_missing_n = sum(bool(r["generic_missing"]) for r in feature_rows)
    recoverable_n = sum(bool(r["generic_missing"]) and bool(r["gold_in_personal_k5"]) for r in feature_rows)
    if generic_missing_n != EXPECTED_GENERIC_MISSING:
        raise RuntimeError(f"Unexpected Generic-Missing count: {generic_missing_n}")
    if recoverable_n != EXPECTED_K5_RECOVERABLE_MISSING:
        raise RuntimeError(f"Unexpected frozen K5 recoverable count: {recoverable_n}")

    features_path = args.output_root / "features.jsonl"
    write_jsonl(features_path, feature_rows)

    baseline_rows = [{
        "row_id": r["row_id"], "author": r["author"],
        "generic_rank": r["generic_rank"], "frequency_rank": r["frequency_rank"], "pv1_rank": r["pv1_rank"],
        "generic_missing": r["generic_missing"], "gold_in_personal_k5": r["gold_in_personal_k5"],
    } for r in feature_rows]
    baseline_metrics = {
        "G": metric_summary(baseline_rows, "generic_rank", "G"),
        "F": metric_summary(baseline_rows, "frequency_rank", "F"),
        "PV1": metric_summary(baseline_rows, "pv1_rank", "PV1"),
    }
    baseline_recovery = {"PV1": recovery_summary(baseline_rows, "pv1_rank")}

    prediction_rows: list[dict[str, Any]] = [{
        "schema_version": 1,
        "row_id": f["row_id"], "author": f["author"], "gold": f["gold"],
        "generic_missing": f["generic_missing"], "gold_in_personal_k5": f["gold_in_personal_k5"],
        "conc_entropy": f["conc_entropy"],
        "ranks": {"G": f["generic_rank"], "F": f["frequency_rank"], "PV1": f["pv1_rank"]},
    } for f in feature_rows]

    grid_results: list[dict[str, Any]] = []
    print("\n=== FULL GRID ===")
    for anchor in ANCHOR_ORDER:
        anchor_results: list[dict[str, Any]] = []
        for lam in entropy_lambdas:
            config_key = f"{anchor}|lambda_E={lam:g}"
            eval_rows: list[dict[str, Any]] = []
            for i, feature in enumerate(feature_rows):
                ranking = rank_merged(
                    frequency_candidates=feature["frequency_candidates"],
                    personal_candidates=feature["personal_k5"],
                    ngram_supports=feature["ngram_supports"],
                    choice_shares=feature["choice_shares"],
                    conc_entropy=float(feature["conc_entropy"]),
                    anchor=anchor,
                    lambda_entropy=float(lam),
                )
                rank = rank_of(ranking, str(feature["gold"]))
                prediction_rows[i]["ranks"][config_key] = rank
                eval_rows.append({
                    "row_id": feature["row_id"], "author": feature["author"], "rank": rank,
                    "pv1_rank": feature["pv1_rank"], "generic_missing": feature["generic_missing"],
                    "gold_in_personal_k5": feature["gold_in_personal_k5"],
                })

            metrics = metric_summary(eval_rows, "rank", config_key)
            metrics["per_author_all"] = per_author_rank_metrics(eval_rows, "rank")
            recovery = recovery_summary(eval_rows, "rank")
            pv1_trans = transition_counts(eval_rows, "pv1_rank", "rank")
            result = {
                "anchor": anchor,
                "beta_ngram": ANCHORS[anchor]["beta_ngram"],
                "gamma_cs": ANCHORS[anchor]["gamma_cs"],
                "source_alpha": ANCHORS[anchor]["source_alpha"],
                "source_lambda_D": ANCHORS[anchor]["source_lambda_D"],
                "lambda_entropy": float(lam),
                "metrics": metrics,
                "recovery": recovery,
                "pv1_to_method": pv1_trans,
                "delta_vs_pv1": delta_metrics(metrics, baseline_metrics["PV1"]),
            }
            if math.isclose(float(lam), 0.0):
                verify_anchor_control(anchor, metrics)
            anchor_results.append(result)
            grid_results.append(result)
            print(
                f"{anchor:15s} lambda_E={lam:g}  "
                f"Macro={metrics['macro_author_top1']:.6f} Micro={metrics['micro_top1']:.6f} "
                f"Top3={metrics['top3']:.6f} Top5={metrics['top5']:.6f} "
                f"MRR={metrics['mrr_at_10']:.6f} Missing={metrics['missing10']:.6f} "
                f"PV1net={pv1_trans['net']:+d} Rec@3={recovery['recovered_at_3']:.4f} "
                f"Rec@10={recovery['recovered_at_10']:.4f}", flush=True,
            )

    # Add transitions and deltas relative to each anchor's lambda_E=0 control.
    by_key = {f"{r['anchor']}|lambda_E={float(r['lambda_entropy']):g}": r for r in grid_results}
    for anchor in ANCHOR_ORDER:
        base_key = f"{anchor}|lambda_E=0"
        base_result = by_key[base_key]
        for result in [r for r in grid_results if r["anchor"] == anchor]:
            key = f"{anchor}|lambda_E={float(result['lambda_entropy']):g}"
            trans_rows = [{
                "base_rank": row["ranks"][base_key], "new_rank": row["ranks"][key]
            } for row in prediction_rows]
            result["anchor0_to_method"] = transition_counts(trans_rows, "base_rank", "new_rank")
            result["delta_vs_anchor0"] = delta_against(result["metrics"], base_result["metrics"])
            result["recovery_delta_vs_anchor0"] = {
                k: float(result["recovery"][k]) - float(base_result["recovery"][k])
                for k in ("recovered_at_1", "recovered_at_3", "recovered_at_5", "recovered_at_10", "recovery_mrr_at_10")
            }

    best_by_anchor: dict[str, Any] = {}
    objectives = ("macro_top1", "top3", "top5", "mrr", "missing", "recovery_top3")
    for anchor in ANCHOR_ORDER:
        rows = [r for r in grid_results if r["anchor"] == anchor]
        best_by_anchor[anchor] = {obj: objective_best(rows, obj) for obj in objectives}

    # Entropy-bin diagnostics for every grid point.
    for result in grid_results:
        config_key = f"{result['anchor']}|lambda_E={float(result['lambda_entropy']):g}"
        flat = [{
            "author": p["author"], "pv1_rank": p["ranks"]["PV1"], "rank": p["ranks"][config_key],
            "generic_missing": p["generic_missing"], "gold_in_personal_k5": p["gold_in_personal_k5"],
            "conc_entropy": p["conc_entropy"],
        } for p in prediction_rows]
        result["entropy_bins"] = bin_diagnostics(flat, "rank", "conc_entropy")

    predictions_path = args.output_root / "predictions.jsonl"
    write_jsonl(predictions_path, prediction_rows)

    # CSV is convenient for immediate inspection.
    import csv
    csv_path = args.output_root / "grid_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as sink:
        fields = [
            "anchor", "beta_ngram", "gamma_cs", "lambda_entropy",
            "macro_author_top1", "micro_top1", "top3", "top5", "mrr_at_10", "missing10", "mean_rank_given_top10",
            "pv1_rescue", "pv1_harm", "pv1_net", "anchor_rescue", "anchor_harm", "anchor_net",
            "rec1", "rec3", "rec5", "rec10", "recovery_mrr", "mean_recovered_rank",
            "d_anchor_macro", "d_anchor_top3", "d_anchor_top5", "d_anchor_mrr", "d_anchor_missing", "d_anchor_rec3",
        ]
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        for r in grid_results:
            m, rec, pt, at = r["metrics"], r["recovery"], r["pv1_to_method"], r["anchor0_to_method"]
            d, dr = r["delta_vs_anchor0"], r["recovery_delta_vs_anchor0"]
            writer.writerow({
                "anchor": r["anchor"], "beta_ngram": r["beta_ngram"], "gamma_cs": r["gamma_cs"], "lambda_entropy": r["lambda_entropy"],
                "macro_author_top1": m["macro_author_top1"], "micro_top1": m["micro_top1"], "top3": m["top3"], "top5": m["top5"],
                "mrr_at_10": m["mrr_at_10"], "missing10": m["missing10"], "mean_rank_given_top10": m["mean_rank_given_top10"],
                "pv1_rescue": pt["rescue"], "pv1_harm": pt["harm"], "pv1_net": pt["net"],
                "anchor_rescue": at["rescue"], "anchor_harm": at["harm"], "anchor_net": at["net"],
                "rec1": rec["recovered_at_1"], "rec3": rec["recovered_at_3"], "rec5": rec["recovered_at_5"], "rec10": rec["recovered_at_10"],
                "recovery_mrr": rec["recovery_mrr_at_10"], "mean_recovered_rank": rec["mean_recovered_rank"],
                "d_anchor_macro": d["macro_author_top1"], "d_anchor_top3": d["top3"], "d_anchor_top5": d["top5"],
                "d_anchor_mrr": d["mrr_at_10"], "d_anchor_missing": d["missing10"], "d_anchor_rec3": dr["recovered_at_3"],
            })

    feature_summary = {
        "schema_version": 1, "status": "complete", "experiment": "initial_ngram_cs_entropy_two_anchors_v1",
        "rows": len(feature_rows), "candidate_count_distribution": dict(sorted(candidate_count_distribution.items())),
        "generic_missing_n": generic_missing_n, "recoverable_generic_missing_k5_n": recoverable_n,
        "selected_k5_interpolated_key": selected_ngram_key,
        "mean_entropy_concentration": statistics.fmean(float(r["conc_entropy"]) for r in feature_rows),
        "gold_used_for_feature_construction": False, "dev3000_used": False, "test_used": False,
        "provenance": provenance,
    }
    write_json(args.output_root / "feature_summary.json", feature_summary)

    comparison = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_ngram_cs_entropy_two_anchors_v1",
        "population": {"train_val_rows": len(feature_rows), "generic_missing_n": generic_missing_n, "recoverable_n": recoverable_n},
        "formula": "Boundary + beta*P_N(c) + gamma*CS(c) + lambda_E*C_E(q)",
        "entropy": "C_E(q)=1-normalized entropy over all legal visible same-Pinyin historical targets",
        "anchors": ANCHORS,
        "entropy_lambda_grid": list(entropy_lambdas),
        "baseline_metrics": baseline_metrics,
        "baseline_recovery": baseline_recovery,
        "grid_results": grid_results,
        "best_by_anchor_and_objective": best_by_anchor,
        "control_validation": {
            "Top3Anchor_lambda0_reproduces_alpha025_lambdaD8": True,
            "BalancedAnchor_lambda0_reproduces_alpha05_lambdaD8": True,
        },
        "invariants": {
            "personal_candidate_pool": "frozen K5",
            "generic_base": "frozen Frequency-reranked Generic Top10",
            "ngram_recomputed": False,
            "ngram_retuned": False,
            "gold_used_for_scoring": False,
            "dev3000_used": False,
            "test_used": False,
        },
        "provenance": provenance,
    }
    comparison_path = args.output_root / "comparison.json"
    write_json(comparison_path, comparison)

    checksums = {
        "schema_version": 1,
        "features.jsonl": sha256_file(features_path),
        "predictions.jsonl": sha256_file(predictions_path),
        "grid_results.csv": sha256_file(csv_path),
        "feature_summary.json": sha256_file(args.output_root / "feature_summary.json"),
        "comparison.json": sha256_file(comparison_path),
    }
    write_json(args.output_root / "artifact_checksums.json", checksums)

    print("\n=== CONTROLS ===")
    for anchor in ANCHOR_ORDER:
        base = by_key[f"{anchor}|lambda_E=0"]
        m, r = base["metrics"], base["recovery"]
        print(
            f"{anchor:15s} lambda_E=0 PASSED  Macro={m['macro_author_top1']:.6f} "
            f"Top3={m['top3']:.6f} Top5={m['top5']:.6f} MRR={m['mrr_at_10']:.6f} "
            f"Missing={m['missing10']:.6f} Rec@3={r['recovered_at_3']:.4f}"
        )

    print("\n=== BEST BY OBJECTIVE ===")
    labels = {
        "macro_top1": "MacroTop1", "top3": "Top3", "top5": "Top5", "mrr": "MRR",
        "missing": "Missing", "recovery_top3": "Rec@3",
    }
    for anchor in ANCHOR_ORDER:
        print(f"\n{anchor}:")
        for obj in objectives:
            best = best_by_anchor[anchor][obj]
            m, r, d, t = best["metrics"], best["recovery"], best["delta_vs_anchor0"], best["anchor0_to_method"]
            print(
                f"  {labels[obj]:9s} lambda_E={best['lambda_entropy']:g} "
                f"Macro={m['macro_author_top1']:.6f} Top3={m['top3']:.6f} Top5={m['top5']:.6f} "
                f"MRR={m['mrr_at_10']:.6f} Missing={m['missing10']:.6f} Rec@3={r['recovered_at_3']:.4f} "
                f"dTop3={d['top3']:+.6f} dMRR={d['mrr_at_10']:+.6f} anchorNet={t['net']:+d}"
            )

    print(f"\nSaved: {comparison_path}")
    print(f"CSV:   {csv_path}")
    print("Gold used for scoring/selection: false")
    print("Dev3000 used: false")
    print("Test used: false")


if __name__ == "__main__":
    main()
