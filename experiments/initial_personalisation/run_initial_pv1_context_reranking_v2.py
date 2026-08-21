from __future__ import annotations

"""PV1-only context reranking for Initial-Pinyin personalisation.

Purpose
-------
This runner isolates context reranking from the newer recovery strategies.  It
starts from the already-frozen PV1 final Top-10 candidate list and ONLY changes
candidate order.  It never adds or removes candidates.

Families
--------
1. PV1 baseline (no context).
2. Existing HardBackoffNGramRecency baseline, faithfully reused from the prior
   candidate-scoring work.  Defaults are frozen at maxN=2, tau=2048.
3. Position-only local lexical context:

       S_pos(q,h) = sum_d alpha^(d-1) * 1[q[-d] == h[-d]]

   Candidate support is the weighted same-Pinyin historical target mass divided
   by the total position-similarity mass across the legal same-Pinyin history.
4. Position + recency:

       w_h = S_pos(q,h) * exp(-age(h)/tau)

   with age=0 for the immediately previous same-author interaction.

For every context family the frozen PV1 candidate score is preserved and context
is added as a new interpretable signal:

    final_score_new(c) = final_score_PV1(c) + lambda_C * context(c)

This keeps the original score margins produced by Generic+Frequency+PV1 recovery
instead of collapsing the frozen PV1 list to rank-only values. Tie-breaking always
preserves the original PV1 order. lambda_C=0 MUST exactly reproduce PV1.

Causal history protocol
-----------------------
- Same author only.
- Strictly prior rows only.
- H5000 is applied on the same-author stream BEFORE exact-Pinyin filtering.
- Earlier Train-Val rows may be history for later Train-Val rows.
- Current/future target is never visible to scoring.
- Gold is used only after scores are computed for Train-Val evaluation/selection.
- Dev3000 is not read.
- Test is not read.

Outputs
-------
<output-root>/
    grid_results.csv
    family_selection.json
    selected_method_metrics.csv
    selected_subset_metrics.csv
    per_author_metrics.csv
    recovery_metrics.csv
    rank_transition_metrics.csv
    selected_predictions.jsonl
    context_diagnostics.json
    latency.json
    comparison.json
    artifact_checksums.json

The runner refuses to overwrite a non-empty output directory.
"""

import argparse
import bisect
import csv
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
# Frozen provenance / protocol
# ---------------------------------------------------------------------------

EXPECTED_FIT_SHA256 = "162f5c98daa86cc69947571e6d8f20fc401f0a82cdd3fd6e517eb7be2addbdb4"
EXPECTED_VAL_SHA256 = "d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4"
EXPECTED_SURFACE_SHA256 = "205c0ba01cd0678d7a4341c503fa2e74cf126a70182ff582687025e4946764b2"
EXPECTED_FREQUENCY_PV1_SHA256 = "7fd8aa158d8cd50bced36b55610f8d932bc65e3aae1dbbd5bd65907ff1707ea7"

EXPECTED_FIT_ROWS = 144526
EXPECTED_VAL_ROWS = 34416
EXPECTED_RECOVERY_K5_N = 4910
HISTORY_BUDGET = 5000

# Headline PV1 values are rounded references, used only as a guardrail after the
# frozen prediction SHA has already matched.
EXPECTED_PV1_HEADLINE = {
    "macro_author_top1": 0.401872,
    "micro_top1": 0.426749,
    "top3": 0.598907,
    "top5": 0.663093,
    "mrr_at_10": 0.524450,
    "missing10": 0.291144,
}
EXPECTED_HEADLINE_TOL = 1e-6

DEFAULT_CONTEXT_LAMBDAS = (0.0, 0.25, 0.50, 1.0, 2.0, 4.0)
DEFAULT_POSITION_ALPHAS = (0.25, 0.50, 0.75)
DEFAULT_POSITION_RECENCY_TAUS = (512.0, 2048.0, 8192.0)
DEFAULT_HARD_MAX_N = 2
DEFAULT_HARD_TAU = 2048.0


# ---------------------------------------------------------------------------
# I/O
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
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected JSON object at {path}:{line_no}")
            if "row_id" not in value:
                raise RuntimeError(f"Missing row_id at {path}:{line_no}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ensure_empty_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite non-empty output directory: {path}. "
            "Use a new versioned output path."
        )
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def target_of(row: Mapping[str, Any]) -> str:
    value = row.get("target", row.get("gold"))
    if value is None:
        raise RuntimeError(f"No target/gold for {row.get('row_id')}")
    return str(value)


def pinyin_of(row: Mapping[str, Any]) -> tuple[str, ...]:
    values = row.get("pinyin_segments")
    if not isinstance(values, list):
        raise RuntimeError(f"No pinyin_segments list for {row.get('row_id')}")
    return tuple(str(value) for value in values)


def context_of(row: Mapping[str, Any]) -> str:
    # Faithful to the earlier lexical-context scorer: missing context becomes an
    # empty prefix rather than causing accidental access to any future text.
    return str(row.get("context", ""))


def candidate_text(item: Mapping[str, Any]) -> str:
    for key in ("candidate", "text", "target"):
        value = item.get(key)
        if value is not None:
            return str(value)
    raise RuntimeError(f"Cannot identify candidate text from: {item}")


def pv1_scored_candidates(row: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return frozen PV1 candidates in frozen rank order with their real scores."""
    values = row.get("pv1_candidates")
    if not isinstance(values, list) or not values:
        raise RuntimeError(f"Missing pv1_candidates for {row.get('row_id')}")
    pairs: list[tuple[int, int, dict[str, Any]]] = []
    for index, item in enumerate(values, 1):
        if not isinstance(item, Mapping):
            raise RuntimeError(f"Unexpected pv1 candidate at {row.get('row_id')}: {item!r}")
        copied = dict(item)
        if copied.get("final_score") is None:
            raise RuntimeError(f"PV1 candidate missing final_score at {row.get('row_id')}: {item!r}")
        rank = int(copied.get("rank", index))
        copied["candidate"] = candidate_text(copied)
        copied["rank"] = rank
        copied["final_score"] = float(copied["final_score"])
        pairs.append((rank, index, copied))
    pairs.sort(key=lambda x: (x[0], x[1], str(x[2]["candidate"])))
    rows = tuple(item for _, _, item in pairs)
    texts = tuple(str(item["candidate"]) for item in rows)
    if len(texts) != len(set(texts)):
        raise RuntimeError(f"Duplicate candidate in PV1 final list: {row.get('row_id')}")
    if len(texts) > 10:
        raise RuntimeError(f"PV1 list longer than Top10: {row.get('row_id')} len={len(texts)}")
    # Verify that frozen rank order is consistent with frozen final_score and tie-breaks.
    expected = tuple(
        str(item["candidate"])
        for item in sorted(
            rows,
            key=lambda item: (
                -float(item["final_score"]),
                0 if str(item.get("source", "generic")) == "generic" else 1,
                int(item.get("generic_rank") or item.get("personal_candidate_rank", 0)),
                str(item["candidate"]),
            ),
        )
    )
    if expected != texts:
        raise RuntimeError(f"Frozen PV1 rank/final_score order mismatch at {row.get('row_id')}")
    return rows


def pv1_ranking(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(item["candidate"]) for item in pv1_scored_candidates(row))


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
            elif isinstance(item, Mapping):
                result.append(candidate_text(item))
            else:
                raise RuntimeError(f"Unexpected personal K5 item at {row.get('row_id')}: {item!r}")
        if len(result) > 5:
            raise RuntimeError(f"personal_candidates_top5 contains >5 at {row.get('row_id')}")
        return tuple(result)

    raise RuntimeError(f"Cannot identify Personal K5 field at {row.get('row_id')}")


def rank_of(ranking: Sequence[str], gold: str) -> int | None:
    for index, value in enumerate(ranking, 1):
        if value == gold:
            return index
    return None


# ---------------------------------------------------------------------------
# Causal history
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoryRecord:
    row_id: str
    author: str
    position: int
    pinyin: tuple[str, ...]
    target: str
    context: str


@dataclass(frozen=True)
class VisibleHistory:
    record: HistoryRecord
    age: int


class CausalHistoryIndex:
    """Strictly-prior same-author H5000, then exact-Pinyin filtering."""

    def __init__(self, records: Sequence[HistoryRecord]) -> None:
        grouped: dict[str, list[HistoryRecord]] = defaultdict(list)
        for record in records:
            grouped[record.author].append(record)

        self.positions: dict[str, tuple[int, ...]] = {}
        self.pinyin_records: dict[tuple[str, tuple[str, ...]], tuple[HistoryRecord, ...]] = {}
        self.pinyin_ordinals: dict[tuple[str, tuple[str, ...]], tuple[int, ...]] = {}

        for author, values in grouped.items():
            ordered = tuple(sorted(values, key=lambda r: (r.position, r.row_id)))
            positions = tuple(r.position for r in ordered)
            if len(positions) != len(set((r.position, r.row_id) for r in ordered)):
                raise RuntimeError(f"Duplicate history ordering key for author {author}")
            self.positions[author] = positions

            by_pinyin: dict[tuple[str, ...], list[tuple[int, HistoryRecord]]] = defaultdict(list)
            for ordinal, record in enumerate(ordered):
                by_pinyin[record.pinyin].append((ordinal, record))
            for pinyin, pairs in by_pinyin.items():
                key = (author, pinyin)
                self.pinyin_ordinals[key] = tuple(ordinal for ordinal, _ in pairs)
                self.pinyin_records[key] = tuple(record for _, record in pairs)

    def visible(
        self, *, author: str, position: int, pinyin: tuple[str, ...]
    ) -> tuple[VisibleHistory, ...]:
        positions = self.positions.get(author, ())
        stop = bisect.bisect_left(positions, position)
        start = max(0, stop - HISTORY_BUDGET)

        key = (author, pinyin)
        ordinals = self.pinyin_ordinals.get(key, ())
        records = self.pinyin_records.get(key, ())
        left = bisect.bisect_left(ordinals, start)
        right = bisect.bisect_left(ordinals, stop)

        return tuple(
            VisibleHistory(record=record, age=stop - 1 - ordinal)
            for ordinal, record in zip(ordinals[left:right], records[left:right])
        )


def to_history_record(row: Mapping[str, Any]) -> HistoryRecord:
    return HistoryRecord(
        row_id=str(row["row_id"]),
        author=str(row["author"]),
        position=int(row["chronological_position"]),
        pinyin=pinyin_of(row),
        target=target_of(row),
        context=context_of(row),
    )


# ---------------------------------------------------------------------------
# Context scorers
# ---------------------------------------------------------------------------


def recency_weight(age: int, tau: float) -> float:
    if tau <= 0:
        raise ValueError("tau must be positive")
    return math.exp(-float(age) / float(tau))


def suffix_matches(a: str, b: str, n: int) -> bool:
    if n <= 0:
        return True
    return len(a) >= n and len(b) >= n and a[-n:] == b[-n:]


def normalize_candidate_mass(values: Sequence[float]) -> list[float]:
    clipped = [max(0.0, float(value)) for value in values]
    total = sum(clipped)
    if total <= 0:
        # Exact behaviour of the earlier HardBackoff scorer: no candidate mass
        # becomes a uniform distribution.  Because every candidate receives the
        # same context score, the frozen PV1 order is still preserved by ties.
        return [1.0 / len(clipped) for _ in clipped] if clipped else []
    return [value / total for value in clipped]


def hard_backoff_ngram_recency(
    *,
    candidates: Sequence[str],
    query_context: str,
    visible: Sequence[VisibleHistory],
    max_n: int,
    tau: float,
) -> tuple[list[float], dict[str, Any]]:
    """Faithful reuse of the previous HardBackoffNGramRecency scorer.

    As in the earlier candidate-scoring runner, the backoff order is selected
    using only history whose target is in the CURRENT candidate pool.
    """

    candidate_set = set(candidates)
    effective_n = 0
    for n in range(max_n, 0, -1):
        if any(
            item.record.target in candidate_set
            and suffix_matches(query_context, item.record.context, n)
            for item in visible
        ):
            effective_n = n
            break

    raw = {candidate: 0.0 for candidate in candidates}
    matched = 0
    for item in visible:
        target = item.record.target
        if target not in raw:
            continue
        if not suffix_matches(query_context, item.record.context, effective_n):
            continue
        matched += 1
        raw[target] += recency_weight(item.age, tau)

    support = normalize_candidate_mass([raw[candidate] for candidate in candidates])
    return support, {
        "effective_n": effective_n,
        "matched_history_rows": matched,
        "candidate_mass": sum(raw.values()),
    }


def position_similarity(
    query_context: str,
    history_context: str,
    *,
    alpha: float,
    max_len: int | None,
) -> float:
    if not (0.0 < alpha < 1.0):
        raise ValueError("position alpha must satisfy 0 < alpha < 1")
    upper = min(len(query_context), len(history_context))
    if max_len is not None:
        upper = min(upper, max_len)
    score = 0.0
    weight = 1.0
    for d in range(1, upper + 1):
        if query_context[-d] == history_context[-d]:
            score += weight
        weight *= alpha
    return score


def position_distribution(
    *,
    candidates: Sequence[str],
    query_context: str,
    visible: Sequence[VisibleHistory],
    alpha: float,
    max_len: int | None,
    tau: float | None,
) -> tuple[list[float], dict[str, Any]]:
    """Position-only or Position+Recency local target distribution.

    The denominator follows the design note literally: total weighted mass over
    ALL legal same-Pinyin history rows, not only rows whose targets happen to be
    in the PV1 Top10.  Therefore the returned candidate scores may sum to < 1.
    This preserves a useful notion of candidate-list coverage/confidence.
    """

    candidate_set = set(candidates)
    numerator = {candidate: 0.0 for candidate in candidates}
    denominator = 0.0
    positive_rows = 0
    candidate_positive_rows = 0

    for item in visible:
        s_pos = position_similarity(
            query_context,
            item.record.context,
            alpha=alpha,
            max_len=max_len,
        )
        if s_pos <= 0:
            continue
        weight = s_pos
        if tau is not None:
            weight *= recency_weight(item.age, tau)
        if weight <= 0:
            continue
        denominator += weight
        positive_rows += 1
        if item.record.target in candidate_set:
            numerator[item.record.target] += weight
            candidate_positive_rows += 1

    if denominator <= 0:
        support = [0.0 for _ in candidates]
    else:
        support = [numerator[candidate] / denominator for candidate in candidates]

    return support, {
        "weighted_history_mass": denominator,
        "positive_history_rows": positive_rows,
        "candidate_positive_history_rows": candidate_positive_rows,
        "candidate_probability_mass": sum(support),
    }


def rerank(
    candidates: Sequence[str],
    pv1_final_scores: Sequence[float],
    context_support: Sequence[float],
    *,
    lambda_context: float,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Add context evidence to the frozen PV1 candidate scores.

    Ties preserve the original PV1 order via the original list index.
    """
    if len(candidates) != len(context_support) or len(candidates) != len(pv1_final_scores):
        raise RuntimeError("Candidate/base-score/support length mismatch")
    if lambda_context < 0.0:
        raise ValueError("lambda_context must be non-negative")
    scores = [
        float(base) + lambda_context * float(context)
        for base, context in zip(pv1_final_scores, context_support)
    ]
    order = sorted(
        range(len(candidates)),
        key=lambda i: (-scores[i], i, candidates[i]),
    )
    ranking = tuple(candidates[i] for i in order)
    ordered_scores = tuple(float(scores[i]) for i in order)
    return ranking, ordered_scores


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def rank_metrics(ranks: Sequence[int | None]) -> dict[str, float | int | None]:
    n = len(ranks)
    if n == 0:
        return {
            "n": 0,
            "top1": 0.0,
            "top3": 0.0,
            "top5": 0.0,
            "mrr_at_10": 0.0,
            "missing10": 0.0,
            "mean_rank_given_top10": None,
        }
    present = [int(rank) for rank in ranks if rank is not None]
    return {
        "n": n,
        "top1": sum(rank == 1 for rank in ranks) / n,
        "top3": sum(rank is not None and rank <= 3 for rank in ranks) / n,
        "top5": sum(rank is not None and rank <= 5 for rank in ranks) / n,
        "mrr_at_10": sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / n,
        "missing10": sum(rank is None for rank in ranks) / n,
        "mean_rank_given_top10": statistics.fmean(present) if present else None,
    }


def author_metrics(
    authors: Sequence[str], ranks: Sequence[int | None]
) -> dict[str, Any]:
    if len(authors) != len(ranks):
        raise RuntimeError("authors/ranks length mismatch")
    grouped: dict[str, list[int | None]] = defaultdict(list)
    for author, rank in zip(authors, ranks):
        grouped[str(author)].append(rank)
    per_author = {
        author: rank_metrics(values)
        for author, values in sorted(grouped.items())
    }
    macro_keys = ("top1", "top3", "top5", "mrr_at_10", "missing10")
    macro = {
        key: statistics.fmean(float(metrics[key]) for metrics in per_author.values())
        for key in macro_keys
    } if per_author else {key: 0.0 for key in macro_keys}
    return {
        "macro_author": macro,
        "micro": rank_metrics(ranks),
        "per_author": per_author,
    }


def transition_counts(
    base_ranks: Sequence[int | None],
    new_ranks: Sequence[int | None],
    mask: Sequence[bool] | None = None,
) -> dict[str, int]:
    if len(base_ranks) != len(new_ranks):
        raise RuntimeError("transition rank length mismatch")
    if mask is None:
        mask = [True] * len(base_ranks)
    counts = {
        "n": 0,
        "rescue": 0,
        "harm": 0,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
        "net": 0,
    }
    for include, before, after in zip(mask, base_ranks, new_ranks):
        if not include:
            continue
        counts["n"] += 1
        before_ok = before == 1
        after_ok = after == 1
        if not before_ok and after_ok:
            counts["rescue"] += 1
        elif before_ok and not after_ok:
            counts["harm"] += 1
        elif before_ok and after_ok:
            counts["unchanged_correct"] += 1
        else:
            counts["unchanged_wrong"] += 1
    counts["net"] = counts["rescue"] - counts["harm"]
    return counts


def rank_movement(
    base_ranks: Sequence[int | None],
    new_ranks: Sequence[int | None],
    mask: Sequence[bool],
) -> dict[str, float | int | None]:
    deltas: list[int] = []
    improved = worsened = unchanged = 0
    for include, before, after in zip(mask, base_ranks, new_ranks):
        if not include:
            continue
        # Candidate-set invariant means present-before == present-after.  Still
        # guard explicitly so this diagnostic fails loudly if that changes.
        if (before is None) != (after is None):
            raise RuntimeError("Candidate-set invariant violated during rank movement")
        if before is None:
            continue
        delta = int(before) - int(after)  # positive = improved
        deltas.append(delta)
        if delta > 0:
            improved += 1
        elif delta < 0:
            worsened += 1
        else:
            unchanged += 1
    return {
        "n_present": len(deltas),
        "rank_improved_n": improved,
        "rank_worsened_n": worsened,
        "rank_unchanged_n": unchanged,
        "mean_rank_gain": statistics.fmean(deltas) if deltas else None,
        "median_rank_gain": statistics.median(deltas) if deltas else None,
    }


def recovery_metrics(
    ranks: Sequence[int | None], recovery_mask: Sequence[bool]
) -> dict[str, float | int | None]:
    values = [rank for rank, include in zip(ranks, recovery_mask) if include]
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "rec1": 0.0,
            "rec3": 0.0,
            "rec5": 0.0,
            "rec10": 0.0,
            "recovery_mrr_at_10": 0.0,
            "mean_recovered_rank": None,
            "recovered_top10_n": 0,
        }
    present = [int(rank) for rank in values if rank is not None]
    return {
        "n": n,
        "rec1": sum(rank == 1 for rank in values) / n,
        "rec3": sum(rank is not None and rank <= 3 for rank in values) / n,
        "rec5": sum(rank is not None and rank <= 5 for rank in values) / n,
        "rec10": sum(rank is not None and rank <= 10 for rank in values) / n,
        "recovery_mrr_at_10": sum(0.0 if rank is None else 1.0 / rank for rank in values) / n,
        "mean_recovered_rank": statistics.fmean(present) if present else None,
        "recovered_top10_n": len(present),
    }


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def latency_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    return {
        "n": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
    }


# ---------------------------------------------------------------------------
# Configuration / selection helpers
# ---------------------------------------------------------------------------


def float_label(value: float) -> str:
    return f"{value:g}"


def hard_config_key(max_n: int, tau: float) -> str:
    return f"HardNGramRecency|maxN={max_n}|tau={float_label(tau)}"


def position_config_key(alpha: float, max_len: int | None) -> str:
    length = "all" if max_len is None else str(max_len)
    return f"Position|alpha={float_label(alpha)}|L={length}"


def position_recency_config_key(alpha: float, tau: float, max_len: int | None) -> str:
    length = "all" if max_len is None else str(max_len)
    return f"PositionRecency|alpha={float_label(alpha)}|tau={float_label(tau)}|L={length}"


def method_key(config_key: str, lambda_context: float) -> str:
    return f"{config_key}|lambdaC={float_label(lambda_context)}"


def select_family(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        raise RuntimeError("Cannot select from empty family")
    # Primary metric remains Macro-author Top1.  MRR and Top3 are secondary.
    # Lower lambdaC wins exact ties, then lexical config key for determinism.
    return max(
        rows,
        key=lambda row: (
            float(row["macro_author_top1"]),
            float(row["mrr_at_10"]),
            float(row["top3"]),
            -float(row["lambda_context"]),
            str(row["config_key"]),
        ),
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    ensure_empty_output_dir(args.output_root)

    input_paths = {
        "fit": args.fit,
        "val": args.val,
        "candidate_surface": args.candidate_surface,
        "frequency_pv1_predictions": args.frequency_pv1_predictions,
    }
    expected_hashes = {
        "fit": EXPECTED_FIT_SHA256,
        "val": EXPECTED_VAL_SHA256,
        "candidate_surface": EXPECTED_SURFACE_SHA256,
        "frequency_pv1_predictions": EXPECTED_FREQUENCY_PV1_SHA256,
    }
    input_hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    if not args.allow_unfrozen_input_hashes:
        for name, expected in expected_hashes.items():
            actual = input_hashes[name]
            if actual != expected:
                raise RuntimeError(f"Frozen SHA mismatch for {name}: expected {expected}, got {actual}")

    fit = read_jsonl(args.fit)
    val = read_jsonl(args.val)
    surface_rows = read_jsonl(args.candidate_surface)
    pv1_rows = read_jsonl(args.frequency_pv1_predictions)

    if len(fit) != EXPECTED_FIT_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_FIT_ROWS} Train-Fit rows, got {len(fit)}")
    if len(val) != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_VAL_ROWS} Train-Val rows, got {len(val)}")
    if len(surface_rows) != EXPECTED_VAL_ROWS or len(pv1_rows) != EXPECTED_VAL_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_VAL_ROWS} surface/PV1 rows, got {len(surface_rows)}/{len(pv1_rows)}"
        )

    val_by_id = {str(row["row_id"]): row for row in val}
    surface_by_id = {str(row["row_id"]): row for row in surface_rows}
    pv1_by_id = {str(row["row_id"]): row for row in pv1_rows}
    if len(val_by_id) != len(val) or len(surface_by_id) != len(surface_rows) or len(pv1_by_id) != len(pv1_rows):
        raise RuntimeError("Duplicate row_id in one or more inputs")
    if set(val_by_id) != set(surface_by_id) or set(val_by_id) != set(pv1_by_id):
        raise RuntimeError("Train-Val / candidate-surface / PV1 row-ID surfaces differ")

    # Preserve Train-Val file order for all outputs and metrics.
    row_ids = [str(row["row_id"]) for row in val]
    authors = [str(row["author"]) for row in val]
    golds = [target_of(row) for row in val]

    baseline_rankings: list[tuple[str, ...]] = []
    baseline_score_vectors: list[tuple[float, ...]] = []
    baseline_candidate_rows: list[tuple[dict[str, Any], ...]] = []
    baseline_ranks: list[int | None] = []
    frequency_ranks: list[int | None] = []
    k5_rows: list[tuple[str, ...]] = []
    conflict_mask: list[bool] = []
    ambiguous_mask: list[bool] = []
    history_available_mask: list[bool] = []
    generic_missing_mask: list[bool] = []
    recovery_mask: list[bool] = []

    for rid, gold in zip(row_ids, golds):
        prow = pv1_by_id[rid]
        srow = surface_by_id[rid]
        if prow.get("dev3000_used") is not False or prow.get("test_used") is not False:
            raise RuntimeError(f"PV1 row crossed Dev/Test boundary: {rid}")
        if int(prow.get("selected_k_pv", -1)) != 1:
            raise RuntimeError(f"PV1 row does not use frozen K=1: {rid}")
        if float(prow.get("selected_lambda_pv", -1.0)) != 4.0:
            raise RuntimeError(f"PV1 row does not use frozen lambda_pv=4: {rid}")
        if float(prow.get("selected_lambda_frequency", -1.0)) != 4.0:
            raise RuntimeError(f"PV1 row does not use frozen lambda_F=4: {rid}")

        scored_candidates = pv1_scored_candidates(prow)
        ranking = tuple(str(item["candidate"]) for item in scored_candidates)
        pv1_scores = tuple(float(item["final_score"]) for item in scored_candidates)
        computed_rank = rank_of(ranking, gold)
        stored_rank = prow.get("pv1_rank")
        stored_rank = None if stored_rank is None else int(stored_rank)
        if computed_rank != stored_rank:
            raise RuntimeError(f"PV1 rank/candidate list mismatch at {rid}: {computed_rank} != {stored_rank}")

        k5 = extract_personal_k5(srow)
        generic_missing = bool(prow.get("generic_missing", srow.get("generic_missing", False)))

        baseline_rankings.append(ranking)
        baseline_score_vectors.append(pv1_scores)
        baseline_candidate_rows.append(scored_candidates)
        baseline_ranks.append(stored_rank)
        frequency_rank = prow.get("frequency_rank")
        frequency_ranks.append(None if frequency_rank is None else int(frequency_rank))
        k5_rows.append(k5)
        conflict_mask.append(bool(prow.get("conflict", srow.get("conflict", False))))
        ambiguous_mask.append(bool(prow.get("ambiguous", srow.get("ambiguous", False))))
        history_available_mask.append(bool(prow.get("history_available", False)))
        generic_missing_mask.append(generic_missing)
        recovery_mask.append(generic_missing and gold in set(k5))

    recovery_n = sum(recovery_mask)
    if recovery_n != EXPECTED_RECOVERY_K5_N:
        raise RuntimeError(f"Expected fixed K5 recovery population n={EXPECTED_RECOVERY_K5_N}, got {recovery_n}")

    baseline_metrics = author_metrics(authors, baseline_ranks)
    baseline_headline = {
        "macro_author_top1": baseline_metrics["macro_author"]["top1"],
        "micro_top1": baseline_metrics["micro"]["top1"],
        "top3": baseline_metrics["micro"]["top3"],
        "top5": baseline_metrics["micro"]["top5"],
        "mrr_at_10": baseline_metrics["micro"]["mrr_at_10"],
        "missing10": baseline_metrics["micro"]["missing10"],
    }
    for key, expected in EXPECTED_PV1_HEADLINE.items():
        if abs(float(baseline_headline[key]) - expected) > EXPECTED_HEADLINE_TOL:
            raise RuntimeError(
                f"Frozen PV1 metric guardrail failed for {key}: "
                f"expected~{expected}, got {baseline_headline[key]}"
            )

    # Useful scale diagnostics for interpreting the additive context coefficient.
    pv1_score_ranges = [max(values) - min(values) for values in baseline_score_vectors if values]
    pv1_top1_margins = [
        values[0] - values[1]
        for values in baseline_score_vectors
        if len(values) >= 2
    ]

    print("=== INITIAL PV1 CONTEXT RERANKING V2 ===", flush=True)
    print(f"Train-Val rows: {len(val)}", flush=True)
    print(f"Authors: {sorted(set(authors))}", flush=True)
    print(f"Recovery population R (Generic Missing & Gold in Personal K5): {recovery_n}", flush=True)
    print("Baseline: frozen PV1 K=1, lambda_F=4, lambda_PV=4", flush=True)
    print("Fusion: frozen PV1 final_score + lambda_C * context_support", flush=True)
    print(
        f"PV1 score range mean={statistics.fmean(pv1_score_ranges):.4f}; "
        f"Top1-vs-Top2 margin mean={statistics.fmean(pv1_top1_margins):.4f}",
        flush=True,
    )
    print(f"Context lambdas: {list(args.context_lambdas)}", flush=True)
    print(f"Position alphas: {list(args.position_alphas)}", flush=True)
    print(f"Position+Recency taus: {list(args.position_recency_taus)}", flush=True)
    print(f"Hard NGramRecency: maxN={args.hard_max_n}, tau={args.hard_tau:g}", flush=True)
    print(f"Position max length: {'all available context' if args.position_max_len is None else args.position_max_len}", flush=True)
    print("Gold used for scoring/features: false", flush=True)
    print("Gold used for Train-Val evaluation/selection only: true", flush=True)
    print("Dev3000 used: false", flush=True)
    print("Test used: false", flush=True)

    history_records = [to_history_record(row) for row in fit]
    history_records.extend(to_history_record(row) for row in val)
    history_index = CausalHistoryIndex(history_records)

    hard_cfg = hard_config_key(args.hard_max_n, args.hard_tau)
    position_cfgs = [position_config_key(alpha, args.position_max_len) for alpha in args.position_alphas]
    posrec_cfgs = [
        position_recency_config_key(alpha, tau, args.position_max_len)
        for alpha in args.position_alphas
        for tau in args.position_recency_taus
    ]
    config_family: dict[str, str] = {hard_cfg: "HardNGramRecency"}
    config_params: dict[str, dict[str, Any]] = {
        hard_cfg: {"max_n": args.hard_max_n, "tau": args.hard_tau}
    }
    for alpha, cfg in zip(args.position_alphas, position_cfgs):
        config_family[cfg] = "Position"
        config_params[cfg] = {"alpha": alpha, "max_len": args.position_max_len}
    posrec_iter = iter(posrec_cfgs)
    for alpha in args.position_alphas:
        for tau in args.position_recency_taus:
            cfg = next(posrec_iter)
            config_family[cfg] = "PositionRecency"
            config_params[cfg] = {"alpha": alpha, "tau": tau, "max_len": args.position_max_len}

    config_keys = [hard_cfg] + position_cfgs + posrec_cfgs
    rank_by_method: dict[str, list[int | None]] = {
        method_key(cfg, lam): []
        for cfg in config_keys
        for lam in args.context_lambdas
    }

    hard_effective_n = Counter()
    hard_matched_history: list[int] = []
    visible_history_counts: list[int] = []
    position_diag_accum: dict[str, dict[str, Any]] = {
        cfg: {
            "rows_with_positive_mass": 0,
            "rows_with_candidate_mass": 0,
            "candidate_probability_mass": [],
            "positive_history_rows": [],
            "candidate_positive_history_rows": [],
        }
        for cfg in position_cfgs + posrec_cfgs
    }

    for index, rid in enumerate(row_ids):
        vrow = val_by_id[rid]
        ranking = baseline_rankings[index]
        pv1_scores = baseline_score_vectors[index]
        gold = golds[index]
        query_context = context_of(vrow)
        visible = history_index.visible(
            author=str(vrow["author"]),
            position=int(vrow["chronological_position"]),
            pinyin=pinyin_of(vrow),
        )
        visible_history_counts.append(len(visible))

        support_by_config: dict[str, list[float]] = {}

        hard_support, hard_diag = hard_backoff_ngram_recency(
            candidates=ranking,
            query_context=query_context,
            visible=visible,
            max_n=args.hard_max_n,
            tau=args.hard_tau,
        )
        support_by_config[hard_cfg] = hard_support
        hard_effective_n[int(hard_diag["effective_n"])] += 1
        hard_matched_history.append(int(hard_diag["matched_history_rows"]))

        for alpha, cfg in zip(args.position_alphas, position_cfgs):
            support, diag = position_distribution(
                candidates=ranking,
                query_context=query_context,
                visible=visible,
                alpha=alpha,
                max_len=args.position_max_len,
                tau=None,
            )
            support_by_config[cfg] = support
            acc = position_diag_accum[cfg]
            if float(diag["weighted_history_mass"]) > 0:
                acc["rows_with_positive_mass"] += 1
            if float(diag["candidate_probability_mass"]) > 0:
                acc["rows_with_candidate_mass"] += 1
            acc["candidate_probability_mass"].append(float(diag["candidate_probability_mass"]))
            acc["positive_history_rows"].append(int(diag["positive_history_rows"]))
            acc["candidate_positive_history_rows"].append(int(diag["candidate_positive_history_rows"]))

        for alpha in args.position_alphas:
            for tau in args.position_recency_taus:
                cfg = position_recency_config_key(alpha, tau, args.position_max_len)
                support, diag = position_distribution(
                    candidates=ranking,
                    query_context=query_context,
                    visible=visible,
                    alpha=alpha,
                    max_len=args.position_max_len,
                    tau=tau,
                )
                support_by_config[cfg] = support
                acc = position_diag_accum[cfg]
                if float(diag["weighted_history_mass"]) > 0:
                    acc["rows_with_positive_mass"] += 1
                if float(diag["candidate_probability_mass"]) > 0:
                    acc["rows_with_candidate_mass"] += 1
                acc["candidate_probability_mass"].append(float(diag["candidate_probability_mass"]))
                acc["positive_history_rows"].append(int(diag["positive_history_rows"]))
                acc["candidate_positive_history_rows"].append(int(diag["candidate_positive_history_rows"]))

        for cfg, support in support_by_config.items():
            for lam in args.context_lambdas:
                new_ranking, _new_scores = rerank(
                    ranking, pv1_scores, support, lambda_context=lam
                )
                if len(new_ranking) != len(ranking) or set(new_ranking) != set(ranking):
                    raise RuntimeError(f"Candidate-set invariant failed at {rid}, {cfg}, lambda={lam}")
                new_rank = rank_of(new_ranking, gold)
                if (new_rank is None) != (baseline_ranks[index] is None):
                    raise RuntimeError(f"Missing@10 row invariant failed at {rid}, {cfg}, lambda={lam}")
                rank_by_method[method_key(cfg, lam)].append(new_rank)

        if (index + 1) % args.progress_every == 0 or index + 1 == len(row_ids):
            print(f"Context scoring/reranking: {index + 1}/{len(row_ids)}", flush=True)

    # lambdaC=0 must be exact PV1 for every config.
    for cfg in config_keys:
        control = rank_by_method[method_key(cfg, 0.0)]
        if control != baseline_ranks:
            raise RuntimeError(f"lambdaC=0 does not exactly reproduce PV1 for {cfg}")

    # ------------------------------------------------------------------
    # Full grid evaluation
    # ------------------------------------------------------------------
    opportunity_mask = [rank is not None for rank in baseline_ranks]
    pv1_wrong_opportunity_mask = [rank is not None and rank != 1 for rank in baseline_ranks]
    pv1_correct_mask = [rank == 1 for rank in baseline_ranks]
    non_conflict_mask = [not value for value in conflict_mask]
    pv1_rescue_vs_f_mask = [
        f_rank != 1 and p_rank == 1
        for f_rank, p_rank in zip(frequency_ranks, baseline_ranks)
    ]
    pv1_harm_vs_f_mask = [
        f_rank == 1 and p_rank != 1
        for f_rank, p_rank in zip(frequency_ranks, baseline_ranks)
    ]

    grid_rows: list[dict[str, Any]] = []
    family_grid_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for cfg in config_keys:
        family = config_family[cfg]
        params = config_params[cfg]
        for lam in args.context_lambdas:
            mkey = method_key(cfg, lam)
            ranks = rank_by_method[mkey]
            metrics = author_metrics(authors, ranks)
            transition = transition_counts(baseline_ranks, ranks)
            rec = recovery_metrics(ranks, recovery_mask)
            opportunity = rank_metrics([rank for rank, include in zip(ranks, opportunity_mask) if include])
            movement = rank_movement(baseline_ranks, ranks, opportunity_mask)
            conflict_transition = transition_counts(baseline_ranks, ranks, conflict_mask)
            pv1_rescue_retained_n = sum(
                include and rank == 1
                for include, rank in zip(pv1_rescue_vs_f_mask, ranks)
            )
            pv1_harm_repaired_n = sum(
                include and rank == 1
                for include, rank in zip(pv1_harm_vs_f_mask, ranks)
            )

            row: dict[str, Any] = {
                "family": family,
                "config_key": cfg,
                "method_key": mkey,
                "lambda_context": lam,
                "macro_author_top1": metrics["macro_author"]["top1"],
                "micro_top1": metrics["micro"]["top1"],
                "top3": metrics["micro"]["top3"],
                "top5": metrics["micro"]["top5"],
                "mrr_at_10": metrics["micro"]["mrr_at_10"],
                "missing10": metrics["micro"]["missing10"],
                "mean_rank_given_top10": metrics["micro"]["mean_rank_given_top10"],
                "rescue": transition["rescue"],
                "harm": transition["harm"],
                "net": transition["net"],
                "conflict_rescue": conflict_transition["rescue"],
                "conflict_harm": conflict_transition["harm"],
                "conflict_net": conflict_transition["net"],
                "pv1_rescue_vs_f_n": sum(pv1_rescue_vs_f_mask),
                "pv1_rescue_retained_n": pv1_rescue_retained_n,
                "pv1_rescue_retention_rate": (
                    pv1_rescue_retained_n / sum(pv1_rescue_vs_f_mask)
                    if sum(pv1_rescue_vs_f_mask) else None
                ),
                "pv1_harm_vs_f_n": sum(pv1_harm_vs_f_mask),
                "pv1_harm_repaired_n": pv1_harm_repaired_n,
                "pv1_harm_repair_rate": (
                    pv1_harm_repaired_n / sum(pv1_harm_vs_f_mask)
                    if sum(pv1_harm_vs_f_mask) else None
                ),
                "opportunity_top1": opportunity["top1"],
                "opportunity_top3": opportunity["top3"],
                "opportunity_top5": opportunity["top5"],
                "opportunity_mrr_at_10": opportunity["mrr_at_10"],
                "rank_improved_n": movement["rank_improved_n"],
                "rank_worsened_n": movement["rank_worsened_n"],
                "rank_unchanged_n": movement["rank_unchanged_n"],
                "mean_rank_gain": movement["mean_rank_gain"],
                "rec1": rec["rec1"],
                "rec3": rec["rec3"],
                "rec5": rec["rec5"],
                "rec10": rec["rec10"],
                "recovery_mrr_at_10": rec["recovery_mrr_at_10"],
                **params,
            }
            # Pure reranking must not change Missing@10 or Recovery@10.
            if float(row["missing10"]) != float(baseline_metrics["micro"]["missing10"]):
                raise RuntimeError(f"Missing@10 changed under pure reranking: {mkey}")
            baseline_rec10 = recovery_metrics(baseline_ranks, recovery_mask)["rec10"]
            if float(row["rec10"]) != float(baseline_rec10):
                raise RuntimeError(f"Recovery@10 changed under pure reranking: {mkey}")
            grid_rows.append(row)
            family_grid_rows[family].append(row)

    selections = {
        family: dict(select_family(rows))
        for family, rows in sorted(family_grid_rows.items())
    }

    # ------------------------------------------------------------------
    # Selected-method detailed metrics
    # ------------------------------------------------------------------
    selected_rankings: dict[str, list[int | None]] = {"PV1": baseline_ranks}
    for family, selected in selections.items():
        selected_rankings[family] = rank_by_method[str(selected["method_key"])]

    subset_masks: dict[str, list[bool]] = {
        "overall": [True] * len(row_ids),
        "history_available": history_available_mask,
        "ambiguous": ambiguous_mask,
        "conflict": conflict_mask,
        "non_conflict": non_conflict_mask,
        "generic_missing": generic_missing_mask,
        "context_opportunity_gold_in_pv1_top10": opportunity_mask,
        "pv1_wrong_but_gold_in_top10": pv1_wrong_opportunity_mask,
        "pv1_correct_top1": pv1_correct_mask,
        "pv1_rescue_vs_frequency": pv1_rescue_vs_f_mask,
        "pv1_harm_vs_frequency": pv1_harm_vs_f_mask,
        "recovery_k5": recovery_mask,
    }

    selected_method_rows: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []
    per_author_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []

    for method_name, ranks in selected_rankings.items():
        metrics = author_metrics(authors, ranks)
        selected_method_rows.append({
            "method": method_name,
            "config": "PV1 frozen baseline" if method_name == "PV1" else selections[method_name]["method_key"],
            "macro_author_top1": metrics["macro_author"]["top1"],
            "macro_author_top3": metrics["macro_author"]["top3"],
            "macro_author_top5": metrics["macro_author"]["top5"],
            "macro_author_mrr_at_10": metrics["macro_author"]["mrr_at_10"],
            "macro_author_missing10": metrics["macro_author"]["missing10"],
            "micro_top1": metrics["micro"]["top1"],
            "top3": metrics["micro"]["top3"],
            "top5": metrics["micro"]["top5"],
            "mrr_at_10": metrics["micro"]["mrr_at_10"],
            "missing10": metrics["micro"]["missing10"],
            "mean_rank_given_top10": metrics["micro"]["mean_rank_given_top10"],
        })
        for author, author_result in metrics["per_author"].items():
            per_author_rows.append({
                "method": method_name,
                "author": author,
                **author_result,
            })

        rec = recovery_metrics(ranks, recovery_mask)
        recovery_rows.append({"method": method_name, **rec})

        if method_name != "PV1":
            for subset_name, mask in (
                ("overall", subset_masks["overall"]),
                ("conflict", conflict_mask),
                ("non_conflict", non_conflict_mask),
                ("recovery_k5", recovery_mask),
                ("context_opportunity", opportunity_mask),
                ("pv1_rescue_vs_frequency", pv1_rescue_vs_f_mask),
                ("pv1_harm_vs_frequency", pv1_harm_vs_f_mask),
            ):
                trans = transition_counts(baseline_ranks, ranks, mask)
                movement = rank_movement(baseline_ranks, ranks, mask)
                transition_rows.append({
                    "method": method_name,
                    "subset": subset_name,
                    **trans,
                    **movement,
                })

        for subset_name, mask in subset_masks.items():
            subset_authors = [author for author, include in zip(authors, mask) if include]
            subset_ranks = [rank for rank, include in zip(ranks, mask) if include]
            subset_metric = author_metrics(subset_authors, subset_ranks)
            subset_rows.append({
                "method": method_name,
                "subset": subset_name,
                "n": len(subset_ranks),
                "macro_author_top1": subset_metric["macro_author"]["top1"],
                "micro_top1": subset_metric["micro"]["top1"],
                "top3": subset_metric["micro"]["top3"],
                "top5": subset_metric["micro"]["top5"],
                "mrr_at_10": subset_metric["micro"]["mrr_at_10"],
                "missing10": subset_metric["micro"]["missing10"],
                "mean_rank_given_top10": subset_metric["micro"]["mean_rank_given_top10"],
            })

    # ------------------------------------------------------------------
    # Selected predictions + exact selected context supports
    # ------------------------------------------------------------------
    selected_prediction_rows: list[dict[str, Any]] = []
    selected_method_specs = {
        family: {
            "config_key": str(selection["config_key"]),
            "lambda_context": float(selection["lambda_context"]),
            **config_params[str(selection["config_key"])],
        }
        for family, selection in selections.items()
    }

    for index, rid in enumerate(row_ids):
        vrow = val_by_id[rid]
        baseline = baseline_rankings[index]
        pv1_scores = baseline_score_vectors[index]
        baseline_rows = baseline_candidate_rows[index]
        query_context = context_of(vrow)
        visible = history_index.visible(
            author=str(vrow["author"]),
            position=int(vrow["chronological_position"]),
            pinyin=pinyin_of(vrow),
        )
        methods_out: dict[str, Any] = {}
        for family, spec in selected_method_specs.items():
            if family == "HardNGramRecency":
                support, diag = hard_backoff_ngram_recency(
                    candidates=baseline,
                    query_context=query_context,
                    visible=visible,
                    max_n=int(spec["max_n"]),
                    tau=float(spec["tau"]),
                )
            elif family == "Position":
                support, diag = position_distribution(
                    candidates=baseline,
                    query_context=query_context,
                    visible=visible,
                    alpha=float(spec["alpha"]),
                    max_len=spec["max_len"],
                    tau=None,
                )
            elif family == "PositionRecency":
                support, diag = position_distribution(
                    candidates=baseline,
                    query_context=query_context,
                    visible=visible,
                    alpha=float(spec["alpha"]),
                    max_len=spec["max_len"],
                    tau=float(spec["tau"]),
                )
            else:
                raise RuntimeError(f"Unknown selected family: {family}")
            ranking, ordered_final_scores = rerank(
                baseline, pv1_scores, support, lambda_context=float(spec["lambda_context"])
            )
            support_by_candidate = {candidate: float(value) for candidate, value in zip(baseline, support)}
            pv1_score_by_candidate = {candidate: float(value) for candidate, value in zip(baseline, pv1_scores)}
            reranked_candidates = [
                {
                    "candidate": candidate,
                    "pv1_final_score": pv1_score_by_candidate[candidate],
                    "context_support": support_by_candidate[candidate],
                    "context_contribution": float(spec["lambda_context"]) * support_by_candidate[candidate],
                    "reranked_final_score": float(score),
                    "pv1_rank": baseline.index(candidate) + 1,
                    "reranked_rank": rank_idx,
                    "source": next(
                        str(item.get("source", ""))
                        for item in baseline_rows
                        if str(item["candidate"]) == candidate
                    ),
                }
                for rank_idx, (candidate, score) in enumerate(zip(ranking, ordered_final_scores), start=1)
            ]
            methods_out[family] = {
                "rank": rank_of(ranking, golds[index]),
                "ranking": list(ranking),
                "reranked_candidates": reranked_candidates,
                "context_support_in_pv1_order": [float(value) for value in support],
                "diagnostics": diag,
            }

        selected_prediction_rows.append({
            "schema_version": 1,
            "row_id": rid,
            "author": authors[index],
            "chronological_position": int(vrow["chronological_position"]),
            "gold": golds[index],
            "pinyin_segments": list(pinyin_of(vrow)),
            "frequency_rank": frequency_ranks[index],
            "pv1_rank": baseline_ranks[index],
            "pv1_ranking": list(baseline),
            "pv1_candidates": [dict(item) for item in baseline_rows],
            "pv1_final_scores": [float(value) for value in pv1_scores],
            "generic_missing": generic_missing_mask[index],
            "recovery_population_k5": recovery_mask[index],
            "conflict": conflict_mask[index],
            "ambiguous": ambiguous_mask[index],
            "pv1_rescue_vs_frequency": pv1_rescue_vs_f_mask[index],
            "pv1_harm_vs_frequency": pv1_harm_vs_f_mask[index],
            "selected_context_methods": methods_out,
            "gold_used_for_scoring_features": False,
            "dev3000_used": False,
            "test_used": False,
        })

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    position_diagnostics: dict[str, Any] = {}
    for cfg, acc in position_diag_accum.items():
        masses = acc["candidate_probability_mass"]
        positive_counts = acc["positive_history_rows"]
        candidate_counts = acc["candidate_positive_history_rows"]
        position_diagnostics[cfg] = {
            "rows": len(row_ids),
            "rows_with_positive_position_mass": acc["rows_with_positive_mass"],
            "rows_with_positive_position_mass_rate": acc["rows_with_positive_mass"] / len(row_ids),
            "rows_with_candidate_context_mass": acc["rows_with_candidate_mass"],
            "rows_with_candidate_context_mass_rate": acc["rows_with_candidate_mass"] / len(row_ids),
            "mean_candidate_probability_mass": statistics.fmean(masses) if masses else None,
            "mean_positive_history_rows": statistics.fmean(positive_counts) if positive_counts else None,
            "mean_candidate_positive_history_rows": statistics.fmean(candidate_counts) if candidate_counts else None,
        }

    context_diagnostics = {
        "pv1_score_scale": {
            "score_range_mean": statistics.fmean(pv1_score_ranges) if pv1_score_ranges else None,
            "score_range_median": statistics.median(pv1_score_ranges) if pv1_score_ranges else None,
            "score_range_p95": percentile(pv1_score_ranges, 0.95),
            "top1_minus_top2_margin_mean": statistics.fmean(pv1_top1_margins) if pv1_top1_margins else None,
            "top1_minus_top2_margin_median": statistics.median(pv1_top1_margins) if pv1_top1_margins else None,
            "top1_minus_top2_margin_p95": percentile(pv1_top1_margins, 0.95),
        },
        "pv1_vs_frequency_transition_populations": {
            "pv1_rescue_n": sum(pv1_rescue_vs_f_mask),
            "pv1_harm_n": sum(pv1_harm_vs_f_mask),
        },
        "history": {
            "history_budget": HISTORY_BUDGET,
            "h5000_before_pinyin_filter": True,
            "strictly_prior_same_author": True,
            "same_pinyin_after_h5000": True,
            "rows_with_same_pinyin_history": sum(count > 0 for count in visible_history_counts),
            "rows_with_same_pinyin_history_rate": sum(count > 0 for count in visible_history_counts) / len(row_ids),
            "mean_same_pinyin_history_rows": statistics.fmean(visible_history_counts),
            "p50_same_pinyin_history_rows": percentile(visible_history_counts, 0.50),
            "p95_same_pinyin_history_rows": percentile(visible_history_counts, 0.95),
        },
        "hard_ngram_recency": {
            "config": hard_cfg,
            "effective_n_distribution": dict(sorted(hard_effective_n.items())),
            "mean_matched_history_rows": statistics.fmean(hard_matched_history),
            "p50_matched_history_rows": percentile(hard_matched_history, 0.50),
            "p95_matched_history_rows": percentile(hard_matched_history, 0.95),
        },
        "position_families": position_diagnostics,
    }

    # ------------------------------------------------------------------
    # Selected online latency: context scorer + fusion/ranking, per query.
    # ------------------------------------------------------------------
    latency: dict[str, Any] = {}
    for family, spec in selected_method_specs.items():
        values: list[float] = []
        for index, rid in enumerate(row_ids):
            vrow = val_by_id[rid]
            baseline = baseline_rankings[index]
            query_context = context_of(vrow)
            visible = history_index.visible(
                author=str(vrow["author"]),
                position=int(vrow["chronological_position"]),
                pinyin=pinyin_of(vrow),
            )
            t0 = time.perf_counter()
            if family == "HardNGramRecency":
                support, _ = hard_backoff_ngram_recency(
                    candidates=baseline,
                    query_context=query_context,
                    visible=visible,
                    max_n=int(spec["max_n"]),
                    tau=float(spec["tau"]),
                )
            elif family == "Position":
                support, _ = position_distribution(
                    candidates=baseline,
                    query_context=query_context,
                    visible=visible,
                    alpha=float(spec["alpha"]),
                    max_len=spec["max_len"],
                    tau=None,
                )
            elif family == "PositionRecency":
                support, _ = position_distribution(
                    candidates=baseline,
                    query_context=query_context,
                    visible=visible,
                    alpha=float(spec["alpha"]),
                    max_len=spec["max_len"],
                    tau=float(spec["tau"]),
                )
            else:
                raise RuntimeError(f"Unknown latency family: {family}")
            _, _ = rerank(
                baseline, baseline_score_vectors[index], support,
                lambda_context=float(spec["lambda_context"])
            )
            values.append((time.perf_counter() - t0) * 1000.0)
        latency[family] = {
            "selected_spec": spec,
            "context_scorer_plus_fusion": latency_summary(values),
            "history_lookup_not_included": True,
        }

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    grid_path = args.output_root / "grid_results.csv"
    selection_path = args.output_root / "family_selection.json"
    method_metrics_path = args.output_root / "selected_method_metrics.csv"
    subset_metrics_path = args.output_root / "selected_subset_metrics.csv"
    per_author_path = args.output_root / "per_author_metrics.csv"
    recovery_path = args.output_root / "recovery_metrics.csv"
    transition_path = args.output_root / "rank_transition_metrics.csv"
    predictions_path = args.output_root / "selected_predictions.jsonl"
    diagnostics_path = args.output_root / "context_diagnostics.json"
    latency_path = args.output_root / "latency.json"
    comparison_path = args.output_root / "comparison.json"

    write_csv(grid_path, grid_rows)
    write_json(selection_path, {
        "selection_metric": "Macro-author Top1 on ALL Train-Val",
        "tie_break": "MRR@10, Top3, lower lambda_context, deterministic config key",
        "families": selections,
        "gold_used_for_scoring_features": False,
        "gold_used_for_train_val_selection_evaluation": True,
        "dev3000_used": False,
        "test_used": False,
    })
    write_csv(method_metrics_path, selected_method_rows)
    write_csv(subset_metrics_path, subset_rows)
    write_csv(per_author_path, per_author_rows)
    write_csv(recovery_path, recovery_rows)
    write_csv(transition_path, transition_rows)
    write_jsonl(predictions_path, selected_prediction_rows)
    write_json(diagnostics_path, context_diagnostics)
    write_json(latency_path, latency)

    comparison = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_pv1_context_reranking_v2",
        "purpose": "PV1-only final-list context reranking using frozen PV1 final_score as base; no new recovery strategy",
        "evaluation_population": {
            "train_val_rows": len(row_ids),
            "authors": sorted(set(authors)),
            "context_opportunity_n": sum(opportunity_mask),
            "recovery_population_definition": "Generic Missing AND Gold in frozen Personal K5",
            "recovery_population_n": recovery_n,
            "pv1_rescue_vs_frequency_n": sum(pv1_rescue_vs_f_mask),
            "pv1_harm_vs_frequency_n": sum(pv1_harm_vs_f_mask),
        },
        "baseline": {
            "method": "PV1",
            "k_pv": 1,
            "lambda_frequency": 4.0,
            "lambda_pv": 4.0,
            "metrics": baseline_headline,
            "recovery": recovery_metrics(baseline_ranks, recovery_mask),
        },
        "context_methods": {
            "HardNGramRecency": {
                "role": "existing lexical-context baseline reused from prior candidate scoring",
                "definition": "largest matching suffix order <= maxN, recency-weighted target mass",
                "fixed_default": {"max_n": args.hard_max_n, "tau": args.hard_tau},
            },
            "Position": {
                "role": "new soft position-aware local lexical context",
                "alpha_grid": list(args.position_alphas),
                "position_max_len": args.position_max_len,
            },
            "PositionRecency": {
                "role": "new position-aware local context plus same-author interaction recency",
                "alpha_grid": list(args.position_alphas),
                "tau_grid": list(args.position_recency_taus),
                "position_max_len": args.position_max_len,
            },
        },
        "fusion": {
            "formula": "frozen_PV1_final_score + lambda_C * context_support",
            "lambda_context_grid": list(args.context_lambdas),
            "pv1_score_source": "frequency_pv1/predictions.jsonl -> pv1_candidates[*].final_score",
            "tie_break": "original PV1 order",
            "interpretation": "lambda_C is additive context strength on top of frozen PV1 score margins",
        },
        "selected": selections,
        "selected_method_metrics": selected_method_rows,
        "recovery_metrics": recovery_rows,
        "invariants": {
            "candidate_set_changed": False,
            "missing10_must_equal_pv1": True,
            "recovery10_must_equal_pv1": True,
            "lambda0_exactly_reproduces_pv1": True,
            "frozen_pv1_final_scores_preserved_as_base": True,
            "h5000_before_exact_pinyin_filter": True,
            "strictly_prior_same_author_history": True,
        },
        "provenance": {
            name: {"path": str(path.resolve()), "sha256": input_hashes[name]}
            for name, path in input_paths.items()
        },
        "gold_used_for_scoring_features": False,
        "gold_used_for_train_val_selection_evaluation": True,
        "dev3000_used": False,
        "test_used": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(comparison_path, comparison)

    output_files = [
        grid_path,
        selection_path,
        method_metrics_path,
        subset_metrics_path,
        per_author_path,
        recovery_path,
        transition_path,
        predictions_path,
        diagnostics_path,
        latency_path,
        comparison_path,
    ]
    checksums = {
        path.name: sha256_file(path)
        for path in output_files
    }
    checksums_path = args.output_root / "artifact_checksums.json"
    write_json(checksums_path, {
        "schema_version": 1,
        "experiment": "initial_pv1_context_reranking_v2",
        "artifacts": checksums,
    })

    print("\n=== SELECTED CONFIGURATIONS ===", flush=True)
    for family, selection in selections.items():
        print(f"{family:20s} {selection['method_key']}", flush=True)

    print("\n=== COMPLETE ===", flush=True)
    for row in selected_method_rows:
        print(
            f"{row['method']:20s} "
            f"Macro={float(row['macro_author_top1']):.6f} "
            f"Micro={float(row['micro_top1']):.6f} "
            f"Top3={float(row['top3']):.6f} "
            f"Top5={float(row['top5']):.6f} "
            f"MRR={float(row['mrr_at_10']):.6f} "
            f"Missing={float(row['missing10']):.6f}",
            flush=True,
        )
    print("\nRecovery population:", flush=True)
    for row in recovery_rows:
        print(
            f"{row['method']:20s} "
            f"Rec1={float(row['rec1']):.4f} "
            f"Rec3={float(row['rec3']):.4f} "
            f"Rec5={float(row['rec5']):.4f} "
            f"Rec10={float(row['rec10']):.4f} "
            f"RecMRR={float(row['recovery_mrr_at_10']):.4f}",
            flush=True,
        )
    print(f"\nOutputs: {args.output_root.resolve()}", flush=True)
    print("Gold used for scoring/features: false", flush=True)
    print("Dev3000 used: false", flush=True)
    print("Test used: false", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--candidate-surface", type=Path, required=True)
    parser.add_argument("--frequency-pv1-predictions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)

    parser.add_argument(
        "--context-lambdas",
        nargs="+",
        type=float,
        default=list(DEFAULT_CONTEXT_LAMBDAS),
        help="Additive context coefficients applied to frozen PV1 final_score",
    )
    parser.add_argument(
        "--position-alphas",
        nargs="+",
        type=float,
        default=list(DEFAULT_POSITION_ALPHAS),
        help="Distance-decay alpha values for position-aware similarity",
    )
    parser.add_argument(
        "--position-recency-taus",
        nargs="+",
        type=float,
        default=list(DEFAULT_POSITION_RECENCY_TAUS),
        help="Same-author interaction-age decay constants for Position+Recency",
    )
    parser.add_argument(
        "--position-max-len",
        type=int,
        default=0,
        help="Maximum backward context characters for position score; 0 means all available",
    )
    parser.add_argument("--hard-max-n", type=int, default=DEFAULT_HARD_MAX_N)
    parser.add_argument("--hard-tau", type=float, default=DEFAULT_HARD_TAU)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--allow-unfrozen-input-hashes",
        action="store_true",
        help="Development escape hatch only; canonical run should NOT use this",
    )
    return parser


def normalize_args(args: argparse.Namespace) -> None:
    if not args.context_lambdas:
        raise ValueError("context lambda grid cannot be empty")
    if any(value < 0 for value in args.context_lambdas):
        raise ValueError("context lambdas must be non-negative")
    if 0.0 not in args.context_lambdas:
        raise ValueError("context lambda grid must include 0 for exact PV1 control")
    if any(not (0 < value < 1) for value in args.position_alphas):
        raise ValueError("all position alphas must satisfy 0 < alpha < 1")
    if any(value <= 0 for value in args.position_recency_taus):
        raise ValueError("all Position+Recency taus must be positive")
    if args.hard_max_n <= 0:
        raise ValueError("hard-max-n must be positive")
    if args.hard_tau <= 0:
        raise ValueError("hard-tau must be positive")
    if args.position_max_len < 0:
        raise ValueError("position-max-len must be >=0")
    args.position_max_len = None if args.position_max_len == 0 else int(args.position_max_len)
    if args.progress_every <= 0:
        raise ValueError("progress-every must be positive")

    # Stable deduplication while preserving CLI order.
    args.context_lambdas = tuple(dict.fromkeys(float(value) for value in args.context_lambdas))
    args.position_alphas = tuple(dict.fromkeys(float(value) for value in args.position_alphas))
    args.position_recency_taus = tuple(
        dict.fromkeys(float(value) for value in args.position_recency_taus)
    )


def main() -> None:
    args = build_parser().parse_args()
    normalize_args(args)
    run(args)


if __name__ == "__main__":
    main()
