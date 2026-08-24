"""Unified K5 Interpolated-NGram + CS + Concentration recovery for Initial Pinyin.

Research question
-----------------
Can frozen K5 Interpolated-NGram candidate scoring improve end-to-end Personal
Vocabulary Recovery over the historical PV1 baseline, and do transparent
Choice Share / concentration signals further improve recovery strength?

Frozen inputs / semantics
-------------------------
- Clean3 Train-Fit + Train-Val only.
- Strictly-prior same-author H5000 selected BEFORE exact Initial-Pinyin filtering.
- Earlier Train-Val rows may enter history for later Train-Val rows.
- Frozen compatible Personal K5 surface; Gold is never used to construct candidates.
- Frozen Frequency-reranked Generic Top10 and frozen PV1 predictions are reused.
- Frozen adaptive K5 Interpolated-NGram scores are reused; NGram is NOT retuned here.
- Dev3000 is not read. Test is not read.
- Gold is used only for Train-Val grid selection, evaluation, and diagnostics.

Primary comparator
------------------
PV1 is the primary baseline. For every selected proposed method we report:
- delta vs PV1 on system metrics;
- PV1 -> Method rescue / harm / net;
- Personal-vocabulary recovery quality on the fixed recoverable population:
    Generic-Missing AND Gold in frozen Personal K5.

Proposed recovery families
--------------------------
Let P_N(c) be the already-normalized frozen K5 Interpolated-NGram support.
Let CS(c)=n_c/N use ALL legal visible same-Pinyin history in the denominator.
Let:
    C_entropy = 1 - normalized_entropy
    C_margin  = p_(1) - p_(2)
    C_dual    = sqrt(C_entropy * C_margin)

For a Personal-K5 candidate c:

    NGram:
        D(c) = P_N(c)

    NGram+CS:
        D(c) = P_N(c) * CS(c)

    NGram+CS+Entropy:
        D(c) = P_N(c) * CS(c) * C_entropy^gamma

    NGram+CS+Margin:
        D(c) = P_N(c) * CS(c) * C_margin^gamma

    NGram+CS+Dual:
        D(c) = P_N(c) * CS(c) * C_dual^gamma

Personal-only candidates use the same PV-style Generic boundary injection:

    final_personal_score = generic_boundary + lambda_D * D(c)

Generic candidates keep their frozen Frequency final_score. Generic wins exact
score ties. Among Personal candidates, the frozen NGram rank is the tie-break.
The final list is truncated to Top10.

Outputs
-------
<output-root>/
    features.jsonl
    feature_summary.json
    grid_results.json
    selected_predictions.jsonl
    comparison.json
    artifact_checksums.json

The script is intentionally CPU-only and performs no PinyinGPT/BGE inference.
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

# Keep lambda=0 as an audit point and otherwise preserve the existing recovery grid.
# If the selected lambda lands on 16, the output flags the grid edge so a later
# extension can be justified explicitly rather than silently changing the protocol.
DEFAULT_LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
DEFAULT_GAMMAS = (0.0, 0.5, 1.0, 2.0)

FAMILY_ORDER = (
    "NGram",
    "NGram+CS",
    "NGram+CS+Entropy",
    "NGram+CS+Margin",
    "NGram+CS+Dual",
)
FAMILY_COMPLEXITY = {name: index for index, name in enumerate(FAMILY_ORDER)}


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


def evidence_for_family(
    *,
    family: str,
    ngram_support: float,
    choice_share: float,
    conc_entropy: float,
    conc_margin: float,
    conc_dual: float,
    gamma: float,
) -> float:
    if family == "NGram":
        return ngram_support

    base = ngram_support * choice_share
    if family == "NGram+CS":
        return base
    if gamma == 0.0:
        return base
    if family == "NGram+CS+Entropy":
        return base * (conc_entropy ** gamma)
    if family == "NGram+CS+Margin":
        return base * (conc_margin ** gamma)
    if family == "NGram+CS+Dual":
        return base * (conc_dual ** gamma)
    raise ValueError(f"Unknown family: {family}")


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
    conc_margin: float,
    conc_dual: float,
    family: str,
    gamma: float,
    lambda_personal: float,
) -> list[dict[str, Any]]:
    if not (
        len(personal_candidates)
        == len(ngram_supports)
        == len(choice_shares)
    ):
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
        generic_rows.append(
            {
                "candidate": candidate,
                "source": "generic_frequency",
                "final_score": float(value["final_score"]),
                "generic_rank": int(value.get("generic_rank") or value.get("rank") or position),
                "ngram_rank": None,
            }
        )

    if not generic_rows:
        raise RuntimeError("Frozen Frequency Generic candidate list is empty")

    boundary = min(normalized_generic)
    order = ngram_rank_order(personal_candidates, ngram_supports)
    rank_by_index = {candidate_index: rank for rank, candidate_index in enumerate(order, start=1)}
    rows = list(generic_rows)

    for i, candidate in enumerate(personal_candidates):
        if candidate in generic_texts:
            raise RuntimeError(f"Frozen Personal K5 overlaps Generic: {candidate!r}")
        evidence = evidence_for_family(
            family=family,
            ngram_support=float(ngram_supports[i]),
            choice_share=float(choice_shares[i]),
            conc_entropy=float(conc_entropy),
            conc_margin=float(conc_margin),
            conc_dual=float(conc_dual),
            gamma=float(gamma),
        )
        rows.append(
            {
                "candidate": candidate,
                "source": "personal_k5",
                "final_score": boundary + float(lambda_personal) * evidence,
                "generic_rank": None,
                "ngram_rank": int(rank_by_index[i]),
                "ngram_support": float(ngram_supports[i]),
                "choice_share": float(choice_shares[i]),
                "recovery_evidence": float(evidence),
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row["final_score"]),
            0 if row["source"] == "generic_frequency" else 1,
            int(row["generic_rank"] or row["ngram_rank"] or 0),
            str(row["candidate"]),
        )
    )
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


def family_configs(family: str, lambdas: Sequence[float], gammas: Sequence[float]) -> list[tuple[float, float]]:
    if family in ("NGram", "NGram+CS"):
        return [(float(lam), 0.0) for lam in lambdas]
    return [(float(lam), float(gamma)) for gamma in gammas for lam in lambdas]


def choose_config(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        raise RuntimeError("Cannot select from empty grid")
    return dict(
        max(
            results,
            key=lambda row: (
                float(row["metrics"]["macro_author_top1"]),
                float(row["metrics"]["mrr_at_10"]),
                -float(row["gamma"]),
                -float(row["lambda_personal"]),
            ),
        )
    )


def choose_development_best(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(
        max(
            selected,
            key=lambda row: (
                float(row["metrics"]["macro_author_top1"]),
                float(row["metrics"]["mrr_at_10"]),
                -FAMILY_COMPLEXITY[str(row["family"])],
                -float(row["gamma"]),
                -float(row["lambda_personal"]),
            ),
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="K5 frozen Interpolated-NGram + CS + concentration recovery vs PV1"
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
        default=",".join(str(value) for value in DEFAULT_LAMBDAS),
        help="comma-separated recovery lambda_D grid",
    )
    parser.add_argument(
        "--gammas",
        default=",".join(str(value) for value in DEFAULT_GAMMAS),
        help="comma-separated concentration gamma grid",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    lambdas = parse_float_grid(args.lambdas)
    gammas = parse_float_grid(args.gammas)
    if 0.0 not in lambdas:
        raise RuntimeError("Lambda grid must include 0 for the no-injection audit point")
    if 0.0 not in gammas:
        raise RuntimeError("Gamma grid must include 0 so concentration may reduce to NGram+CS")

    args.output_root.mkdir(parents=True, exist_ok=True)
    provenance = verify_frozen_inputs(args)
    selected_ngram_key = str(provenance["selected_k5_interpolated_key"])

    print("=== INITIAL K5 NGRAM + CS + CONCENTRATION RECOVERY ===")
    print(f"Primary comparator: PV1")
    print(f"Frozen K5 Interpolated method: {selected_ngram_key}")
    print(f"lambda_D grid: {lambdas}")
    print(f"gamma grid: {gammas}")
    print("families:")
    for family in FAMILY_ORDER:
        print(f"  - {family}")
    print("Gold used for scoring/features: false")
    print("Gold used for Train-Val selection/evaluation only: true")
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
    ngram_top1_changed = 0
    ngram_frequency_disagreement_eligible = 0
    ngram_top1_gold = 0
    frequency_top1_gold = 0
    disagreement_ngram_gold = 0
    disagreement_frequency_gold = 0

    started = time.perf_counter()
    for number, row_id in enumerate(sorted(val), start=1):
        row = val[row_id]
        pred = pred_rows[row_id]
        surface = surface_rows[row_id]
        ngram_row = ngram_rows[row_id]

        author = str(row["author"])
        gold = str(row.get("target", row.get("gold")))
        pinyin = tuple(str(value) for value in row["pinyin_segments"])
        visible = history.visible(
            author=author,
            position=int(row["chronological_position"]),
            pinyin=pinyin,
        )
        counts = Counter(record.target for record in visible)
        dist = distribution_features(counts)

        personal_k5 = extract_personal_k5(surface)
        candidate_count_distribution[len(personal_k5)] += 1
        frequency_candidates = list(pred.get("frequency_candidates", []))
        if not frequency_candidates:
            raise RuntimeError(f"Missing frequency_candidates at {row_id}")
        generic_texts = {
            str(value.get("candidate", value.get("text")))
            for value in frequency_candidates
        }
        overlap = generic_texts.intersection(personal_k5)
        if overlap:
            raise RuntimeError(f"Personal K5 overlaps Generic at {row_id}: {sorted(overlap)}")

        methods = ngram_row.get("methods")
        if not isinstance(methods, Mapping) or selected_ngram_key not in methods:
            raise RuntimeError(f"Missing selected NGram method at {row_id}: {selected_ngram_key}")
        selected_method = methods[selected_ngram_key]
        ngram_candidates = tuple(str(value) for value in selected_method.get("candidates", []))
        ngram_supports = tuple(float(value) for value in selected_method.get("support", []))
        if ngram_candidates != personal_k5:
            raise RuntimeError(
                f"Frozen K5 != selected NGram candidate order at {row_id}:\n"
                f"surface={personal_k5}\nngram={ngram_candidates}"
            )
        if len(ngram_supports) != len(personal_k5):
            raise RuntimeError(f"NGram support length mismatch at {row_id}")
        if ngram_supports and not math.isclose(sum(ngram_supports), 1.0, rel_tol=1e-7, abs_tol=1e-7):
            raise RuntimeError(f"Selected NGram support does not sum to 1 at {row_id}: {sum(ngram_supports)}")
        if ngram_row.get("gold_used_for_scoring") is not False:
            raise RuntimeError(f"Adaptive NGram row has invalid Gold provenance at {row_id}")

        history_total = int(dist["same_pinyin_history_count"])
        personal_counts = [int(counts.get(candidate, 0)) for candidate in personal_k5]
        choice_shares = [
            count / history_total if history_total > 0 else 0.0
            for count in personal_counts
        ]
        if any(count <= 0 for count in personal_counts):
            raise RuntimeError(
                f"Frozen Personal K5 lacks legal visible same-Pinyin support at {row_id}: "
                f"{list(zip(personal_k5, personal_counts))}"
            )

        order = ngram_rank_order(personal_k5, ngram_supports)
        ngram_top1 = personal_k5[order[0]] if order else None
        frequency_personal_top1 = personal_k5[0] if personal_k5 else None
        changed = bool(ngram_top1 is not None and frequency_personal_top1 is not None and ngram_top1 != frequency_personal_top1)
        if changed:
            ngram_top1_changed += 1

        if len(personal_k5) >= 2 and gold in set(personal_k5):
            ngram_frequency_disagreement_eligible += 1
            ngram_correct = ngram_top1 == gold
            frequency_correct = frequency_personal_top1 == gold
            ngram_top1_gold += int(ngram_correct)
            frequency_top1_gold += int(frequency_correct)
            if changed:
                disagreement_ngram_gold += int(ngram_correct)
                disagreement_frequency_gold += int(frequency_correct)

        generic_missing = bool(row.get("generic_missing", pred.get("generic_missing", False)))
        feature_rows.append(
            {
                "row_id": row_id,
                "author": author,
                "gold": gold,
                "pinyin_segments": list(pinyin),
                "generic_rank": pred.get("generic_rank", row.get("generic_rank")),
                "frequency_rank": pred.get("frequency_rank"),
                "pv1_rank": pred.get("pv1_rank"),
                "generic_missing": generic_missing,
                "ambiguous": bool(row.get("ambiguous", pred.get("ambiguous", False))),
                "formal_conflict": bool(row.get("conflict", pred.get("conflict", False))),
                "gold_in_personal_k5": gold in set(personal_k5),
                "personal_k5": list(personal_k5),
                "personal_counts": personal_counts,
                "choice_shares": choice_shares,
                "ngram_supports": list(ngram_supports),
                "ngram_top1": ngram_top1,
                "frequency_personal_top1": frequency_personal_top1,
                "ngram_top1_differs_from_frequency": changed,
                "same_pinyin_history_count": history_total,
                "distinct_targets": int(dist["distinct_targets"]),
                "entropy_norm": float(dist["entropy_norm"]),
                "conc_entropy": float(dist["conc_entropy"]),
                "winner_share": float(dist["winner_share"]),
                "runner_up_share": float(dist["runner_up_share"]),
                "conc_margin": float(dist["conc_margin"]),
                "conc_dual": float(dist["conc_dual"]),
                "frequency_candidates": frequency_candidates,
            }
        )

        if args.progress_every > 0 and (number % args.progress_every == 0 or number == len(val)):
            elapsed = time.perf_counter() - started
            print(f"FEATURES {number}/{len(val)} elapsed={elapsed:.1f}s", flush=True)

    generic_missing_n = sum(bool(row["generic_missing"]) for row in feature_rows)
    recoverable_n = sum(
        bool(row["generic_missing"]) and bool(row["gold_in_personal_k5"])
        for row in feature_rows
    )
    if generic_missing_n != EXPECTED_GENERIC_MISSING:
        raise RuntimeError(f"Unexpected Generic-Missing count: {generic_missing_n}")
    if recoverable_n != EXPECTED_K5_RECOVERABLE_MISSING:
        raise RuntimeError(f"Unexpected frozen K5 recoverable Generic-Missing count: {recoverable_n}")

    features_path = args.output_root / "features.jsonl"
    write_jsonl(features_path, feature_rows)

    feature_summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_k5_ngram_cs_concentration_recovery_v1",
        "rows": len(feature_rows),
        "candidate_count_distribution": dict(sorted(candidate_count_distribution.items())),
        "generic_missing_n": generic_missing_n,
        "recoverable_generic_missing_k5_n": recoverable_n,
        "selected_k5_interpolated_key": selected_ngram_key,
        "ngram_vs_personal_frequency": {
            "ngram_top1_differs_all_rows_n": ngram_top1_changed,
            "candidate_only_eligible_n": ngram_frequency_disagreement_eligible,
            "ngram_top1_gold_n": ngram_top1_gold,
            "personal_frequency_top1_gold_n": frequency_top1_gold,
            "on_changed_rows_ngram_top1_gold_n": disagreement_ngram_gold,
            "on_changed_rows_personal_frequency_top1_gold_n": disagreement_frequency_gold,
        },
        "gold_used_for_scoring": False,
        "dev3000_used": False,
        "test_used": False,
        "provenance": provenance,
    }
    write_json(args.output_root / "feature_summary.json", feature_summary)

    # Baselines from the frozen prediction artifact, evaluated on the same 34,416 rows.
    baseline_rows = [
        {
            "row_id": row["row_id"],
            "author": row["author"],
            "generic_rank": row["generic_rank"],
            "frequency_rank": row["frequency_rank"],
            "pv1_rank": row["pv1_rank"],
            "generic_missing": row["generic_missing"],
            "gold_in_personal_k5": row["gold_in_personal_k5"],
        }
        for row in feature_rows
    ]
    baseline_metrics = {
        "G": metric_summary(baseline_rows, "generic_rank", "G"),
        "F": metric_summary(baseline_rows, "frequency_rank", "F"),
        "PV1": metric_summary(baseline_rows, "pv1_rank", "PV1"),
    }
    pv1_recovery = recovery_summary(baseline_rows, "pv1_rank")

    grid_results: list[dict[str, Any]] = []
    selected_by_family: dict[str, dict[str, Any]] = {}

    print("\n=== GRID SEARCH ===")
    for family in FAMILY_ORDER:
        family_rows: list[dict[str, Any]] = []
        configs = family_configs(family, lambdas, gammas)
        for config_number, (lambda_personal, gamma) in enumerate(configs, start=1):
            eval_rows: list[dict[str, Any]] = []
            for feature in feature_rows:
                ranking = rank_merged(
                    frequency_candidates=feature["frequency_candidates"],
                    personal_candidates=feature["personal_k5"],
                    ngram_supports=feature["ngram_supports"],
                    choice_shares=feature["choice_shares"],
                    conc_entropy=float(feature["conc_entropy"]),
                    conc_margin=float(feature["conc_margin"]),
                    conc_dual=float(feature["conc_dual"]),
                    family=family,
                    gamma=gamma,
                    lambda_personal=lambda_personal,
                )
                eval_rows.append(
                    {
                        "row_id": feature["row_id"],
                        "author": feature["author"],
                        "rank": rank_of(ranking, str(feature["gold"])),
                        "pv1_rank": feature["pv1_rank"],
                        "generic_missing": feature["generic_missing"],
                        "gold_in_personal_k5": feature["gold_in_personal_k5"],
                    }
                )

            metrics = metric_summary(eval_rows, "rank", family)
            trans = transition_counts(eval_rows, "pv1_rank", "rank")
            recovery = recovery_summary(eval_rows, "rank")
            result = {
                "family": family,
                "lambda_personal": lambda_personal,
                "gamma": gamma,
                "metrics": metrics,
                "delta_vs_pv1": delta_metrics(metrics, baseline_metrics["PV1"]),
                "pv1_to_method": trans,
                "recovery": recovery,
            }
            family_rows.append(result)
            grid_results.append(result)

            print(
                f"{family:24s} {config_number:02d}/{len(configs):02d} "
                f"lambda={lambda_personal:g} gamma={gamma:g} "
                f"Macro={metrics['macro_author_top1']:.6f} "
                f"dPV1={result['delta_vs_pv1']['macro_author_top1']:+.6f} "
                f"PV1->M net={trans['net']:+d} "
                f"Rec@1={recovery['recovered_at_1']:.4f} Rec@10={recovery['recovered_at_10']:.4f}",
                flush=True,
            )

        selected = choose_config(family_rows)
        selected["selected_lambda_hits_upper_grid_boundary"] = math.isclose(
            float(selected["lambda_personal"]), max(lambdas)
        )
        selected_by_family[family] = selected

    development_best = choose_development_best(list(selected_by_family.values()))

    # Materialize selected predictions with final Top10 lists.
    selected_rows: list[dict[str, Any]] = []
    for feature in feature_rows:
        output_row: dict[str, Any] = {
            "schema_version": 1,
            "row_id": feature["row_id"],
            "author": feature["author"],
            "gold": feature["gold"],
            "generic_missing": feature["generic_missing"],
            "gold_in_personal_k5": feature["gold_in_personal_k5"],
            "personal_k5": feature["personal_k5"],
            "ngram_supports": feature["ngram_supports"],
            "choice_shares": feature["choice_shares"],
            "conc_entropy": feature["conc_entropy"],
            "conc_margin": feature["conc_margin"],
            "conc_dual": feature["conc_dual"],
            "ranks": {
                "G": feature["generic_rank"],
                "F": feature["frequency_rank"],
                "PV1": feature["pv1_rank"],
            },
            "selected": {},
        }
        for family in FAMILY_ORDER:
            selected = selected_by_family[family]
            ranking = rank_merged(
                frequency_candidates=feature["frequency_candidates"],
                personal_candidates=feature["personal_k5"],
                ngram_supports=feature["ngram_supports"],
                choice_shares=feature["choice_shares"],
                conc_entropy=float(feature["conc_entropy"]),
                conc_margin=float(feature["conc_margin"]),
                conc_dual=float(feature["conc_dual"]),
                family=family,
                gamma=float(selected["gamma"]),
                lambda_personal=float(selected["lambda_personal"]),
            )
            output_row["ranks"][family] = rank_of(ranking, str(feature["gold"]))
            output_row["selected"][family] = {
                "lambda_personal": float(selected["lambda_personal"]),
                "gamma": float(selected["gamma"]),
                "top10": [str(item["candidate"]) for item in ranking],
            }
        selected_rows.append(output_row)

    selected_predictions_path = args.output_root / "selected_predictions.jsonl"
    write_jsonl(selected_predictions_path, selected_rows)

    selected_reports: dict[str, Any] = {}
    for family in FAMILY_ORDER:
        flat_rows = [
            {
                **row,
                "pv1_rank": row["ranks"]["PV1"],
                "selected_rank": row["ranks"][family],
            }
            for row in selected_rows
        ]
        metrics = metric_summary(flat_rows, "selected_rank", family)
        recovery = recovery_summary(flat_rows, "selected_rank")
        selected = selected_by_family[family]
        selected_reports[family] = {
            "selected_config": {
                "lambda_personal": float(selected["lambda_personal"]),
                "gamma": float(selected["gamma"]),
                "lambda_hits_upper_grid_boundary": bool(selected["selected_lambda_hits_upper_grid_boundary"]),
            },
            "metrics": metrics,
            "delta_vs_pv1": delta_metrics(metrics, baseline_metrics["PV1"]),
            "pv1_to_method": transition_counts(flat_rows, "pv1_rank", "selected_rank"),
            "recovery": recovery,
            "formal_conflict_pv1_to_method": transition_counts(
                [
                    {**flat, "formal_conflict": feature["formal_conflict"]}
                    for flat, feature in zip(flat_rows, feature_rows)
                ],
                "pv1_rank",
                "selected_rank",
                predicate=lambda r: bool(r["formal_conflict"]),
            ),
            "concentration_bins": {
                "entropy": bin_diagnostics(
                    [
                        {**flat, "conc_entropy": feature["conc_entropy"]}
                        for flat, feature in zip(flat_rows, feature_rows)
                    ],
                    "selected_rank",
                    "conc_entropy",
                ),
                "margin": bin_diagnostics(
                    [
                        {**flat, "conc_margin": feature["conc_margin"]}
                        for flat, feature in zip(flat_rows, feature_rows)
                    ],
                    "selected_rank",
                    "conc_margin",
                ),
                "dual": bin_diagnostics(
                    [
                        {**flat, "conc_dual": feature["conc_dual"]}
                        for flat, feature in zip(flat_rows, feature_rows)
                    ],
                    "selected_rank",
                    "conc_dual",
                ),
            },
        }

    grid_payload = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_k5_ngram_cs_concentration_recovery_v1",
        "primary_comparator": "PV1",
        "selection_population": "all 34416 standardized Clean3 Train-Val rows",
        "selection_metric": "Macro-author Top1; tie MRR@10, lower gamma, lower lambda",
        "selected_k5_interpolated_key": selected_ngram_key,
        "lambda_grid": list(lambdas),
        "gamma_grid": list(gammas),
        "families": list(FAMILY_ORDER),
        "results": grid_results,
        "selected_by_family": selected_by_family,
        "development_best": development_best,
        "gold_used_for_scoring": False,
        "gold_used_for_train_val_selection": True,
        "dev3000_used": False,
        "test_used": False,
    }
    grid_path = args.output_root / "grid_results.json"
    write_json(grid_path, grid_payload)

    comparison = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_k5_ngram_cs_concentration_recovery_v1",
        "research_question": (
            "Does frozen K5 Interpolated-NGram candidate scoring, combined with Choice Share "
            "and entropy/margin/dual concentration, improve end-to-end Personal Vocabulary "
            "Recovery over frozen PV1?"
        ),
        "primary_comparator": "PV1",
        "selected_k5_interpolated_key": selected_ngram_key,
        "population": {
            "train_val_rows": len(feature_rows),
            "generic_missing_n": generic_missing_n,
            "recoverable_generic_missing_gold_in_personal_k5_n": recoverable_n,
        },
        "baseline_metrics": baseline_metrics,
        "pv1_recovery": pv1_recovery,
        "selected_methods": selected_reports,
        "development_best": {
            "family": development_best["family"],
            "lambda_personal": development_best["lambda_personal"],
            "gamma": development_best["gamma"],
            "metrics": development_best["metrics"],
            "delta_vs_pv1": development_best["delta_vs_pv1"],
            "pv1_to_method": development_best["pv1_to_method"],
            "recovery": development_best["recovery"],
        },
        "ngram_vs_personal_frequency": feature_summary["ngram_vs_personal_frequency"],
        "method_definitions": {
            "NGram": "P_N(c)",
            "NGram+CS": "P_N(c) * CS(c)",
            "NGram+CS+Entropy": "P_N(c) * CS(c) * C_entropy^gamma",
            "NGram+CS+Margin": "P_N(c) * CS(c) * C_margin^gamma",
            "NGram+CS+Dual": "P_N(c) * CS(c) * sqrt(C_entropy*C_margin)^gamma",
            "CS": "n_c / all legal visible same-Pinyin history count",
            "boundary": "minimum normalized Generic score before frozen Frequency boost",
        },
        "invariants": {
            "personal_candidate_pool": "frozen K5",
            "generic_base": "frozen Frequency-reranked Generic Top10",
            "pv1_recomputed": False,
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
        "feature_summary.json": sha256_file(args.output_root / "feature_summary.json"),
        "grid_results.json": sha256_file(grid_path),
        "selected_predictions.jsonl": sha256_file(selected_predictions_path),
        "comparison.json": sha256_file(comparison_path),
    }
    write_json(args.output_root / "artifact_checksums.json", checksums)

    print("\n=== SELECTED RESULTS (PRIMARY COMPARATOR = PV1) ===")
    pv1 = baseline_metrics["PV1"]
    print(
        f"PV1                     Macro={pv1['macro_author_top1']:.6f} "
        f"MRR10={pv1['mrr_at_10']:.6f} Missing10={pv1['missing10']:.6f} "
        f"Rec@1={pv1_recovery['recovered_at_1']:.4f} Rec@10={pv1_recovery['recovered_at_10']:.4f}"
    )
    for family in FAMILY_ORDER:
        report = selected_reports[family]
        cfg = report["selected_config"]
        m = report["metrics"]
        d = report["delta_vs_pv1"]
        t = report["pv1_to_method"]
        r = report["recovery"]
        edge = " [LAMBDA GRID EDGE]" if cfg["lambda_hits_upper_grid_boundary"] else ""
        print(
            f"{family:24s} lambda={cfg['lambda_personal']:g} gamma={cfg['gamma']:g} "
            f"Macro={m['macro_author_top1']:.6f} dPV1={d['macro_author_top1']:+.6f} "
            f"rescue={t['rescue']} harm={t['harm']} net={t['net']:+d} "
            f"Rec@1={r['recovered_at_1']:.4f} Rec@3={r['recovered_at_3']:.4f} "
            f"Rec@5={r['recovered_at_5']:.4f} Rec@10={r['recovered_at_10']:.4f} "
            f"RecMRR={r['recovery_mrr_at_10']:.4f} MeanRecRank={r['mean_recovered_rank']}"
            f"{edge}"
        )

    print(f"\nDevelopment-best family: {development_best['family']}")
    print(f"Saved: {comparison_path}")
    print("Gold used for scoring/features: false")
    print("Dev3000 used: false")
    print("Test used: false")


if __name__ == "__main__":
    main()
