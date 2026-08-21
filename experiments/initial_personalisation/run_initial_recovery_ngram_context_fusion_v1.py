from __future__ import annotations

"""Three-base Recovery -> NGramRecency final-list reranking experiment.

Research question
-----------------
Which Stage-1 recovery philosophy is most compatible with a common Stage-2
context reranker?

Frozen Stage-1 bases
--------------------
1. K5+Entropy          : coverage-first
2. 4P+4CS+2E           : balanced
3. 6P+2CS+.25E         : front-rank / Top3-oriented

Stage-2 reranker
----------------
HardBackoff NGramRecency, maxN=2, tau=2048, applied ONLY to each already-
frozen Stage-1 Top10 candidate set.

    S_final(c) = S_REC(c) + lambda_N * P_NG-R(c)

The candidate set is immutable during Stage 2. Therefore Missing@10 and
Recovery Rec@10 must remain exactly invariant for a fixed recovery base.

Scientific safeguards
---------------------
- Train-Fit / Train-Val only.
- Dev3000 and Test are never read.
- Gold is never used for feature construction or scoring.
- Gold is used only for Train-Val evaluation / lambda selection.
- Same-author strictly-prior history.
- H5000 is applied BEFORE exact-Pinyin filtering.
- Earlier Train-Val rows may be history for later Train-Val rows.
- A non-empty output directory is refused to avoid overwriting history.
- Stage-1 metrics are regression-checked before any context result is accepted.

Expected inputs are the already-frozen Initial-Pinyin artifacts from
initial_recovery_comparison_v1.
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
# Frozen protocol / provenance
# ---------------------------------------------------------------------------

EXPECTED_FIT_SHA256 = "162f5c98daa86cc69947571e6d8f20fc401f0a82cdd3fd6e517eb7be2addbdb4"
EXPECTED_VAL_SHA256 = "d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4"
EXPECTED_SURFACE_SHA256 = "205c0ba01cd0678d7a4341c503fa2e74cf126a70182ff582687025e4946764b2"
EXPECTED_FREQUENCY_PV1_SHA256 = "7fd8aa158d8cd50bced36b55610f8d932bc65e3aae1dbbd5bd65907ff1707ea7"

EXPECTED_FIT_ROWS = 144_526
EXPECTED_VAL_ROWS = 34_416
EXPECTED_GENERIC_MISSING = 12_565
EXPECTED_RECOVERABLE_K5 = 4_910

HISTORY_BUDGET = 5000
NGRAM_MAX_N = 2
NGRAM_TAU = 2048.0
INTERPOLATED_KEY = "K5|Interpolated|maxN=2|kappa=1|tau=2048"

DEFAULT_LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0)

BASE_ORDER = (
    "K5+Entropy",
    "4P+4CS+2E",
    "6P+2CS+.25E",
)

# Rounded historical regression targets. These are guards, not re-selection.
EXPECTED_STAGE1 = {
    "K5+Entropy": {
        "macro_author_top1": 0.403790,
        "micro_top1": 0.428405,
        "top3": 0.602336,
        "top5": 0.677069,
        "mrr_at_10": 0.533534,
        "missing10": 0.243288,
        "rec1": 0.2126,
        "rec3": 0.6126,
        "rec5": 0.8035,
        "rec10": 0.9876,
        "recovery_mrr_at_10": 0.4552,
    },
    "4P+4CS+2E": {
        "macro_author_top1": 0.4048067470537085,
        "micro_top1": 0.4293642491864249,
        "top3": 0.6148012552301255,
        "top5": 0.6858147373314737,
        "mrr_at_10": 0.5374326748171903,
        "missing10": 0.24317178056717806,
        "rec1": 0.2588594704684318,
        "rec3": 0.5492871690427699,
        "rec5": 0.7169042769857433,
        "rec10": 0.9484725050916497,
        "recovery_mrr_at_10": 0.4542168557850839,
    },
    "6P+2CS+.25E": {
        "macro_author_top1": 0.40063760349181027,
        "micro_top1": 0.42398884239888424,
        "top3": 0.6158763365876336,
        "top5": 0.6860471873547187,
        "mrr_at_10": 0.5345453019267679,
        "missing10": 0.24386913063691307,
        "rec1": 0.2769857433808554,
        "rec3": 0.5737270875763747,
        "rec5": 0.7109979633401222,
        "rec10": 0.9283095723014256,
        "recovery_mrr_at_10": 0.4678997349109365,
    },
}


# ---------------------------------------------------------------------------
# Utilities
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
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
            if "row_id" not in row:
                raise RuntimeError(f"Missing row_id at {path}:{line_number}")
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


def parse_grid(text: str) -> tuple[float, ...]:
    values = sorted({float(value.strip()) for value in text.split(",") if value.strip()})
    if not values:
        raise ValueError("Lambda grid is empty")
    if any(value < 0 for value in values):
        raise ValueError("Lambda values must be non-negative")
    if 0.0 not in values:
        raise ValueError("Lambda grid must include 0 for the exact Stage-1 control")
    return tuple(values)


def target_of(row: Mapping[str, Any]) -> str:
    value = row.get("target", row.get("gold"))
    if value is None:
        raise RuntimeError(f"No target/gold on row {row.get('row_id')}")
    return str(value)


def pinyin_of(row: Mapping[str, Any]) -> tuple[str, ...]:
    value = row.get("pinyin_segments")
    if not isinstance(value, list):
        raise RuntimeError(f"Missing pinyin_segments on row {row.get('row_id')}")
    return tuple(str(item) for item in value)


def context_of(row: Mapping[str, Any]) -> str:
    value = row.get("context", row.get("model_used_context", ""))
    return str(value)


def generic_score(item: Mapping[str, Any]) -> float:
    for key in ("log_probability", "generic_score", "score"):
        if item.get(key) is not None:
            return float(item[key])
    raise RuntimeError(f"Generic candidate has no score: {item}")


def candidate_text(item: Mapping[str, Any]) -> str:
    value = item.get("candidate", item.get("text", item.get("target")))
    if value is None:
        raise RuntimeError(f"Cannot identify candidate text: {item}")
    return str(value)


def normalize_generic_scores(items: Sequence[Mapping[str, Any]]) -> tuple[float, ...]:
    values = [generic_score(item) for item in items]
    if not values:
        raise RuntimeError("Generic candidate list is empty")
    mean = statistics.fmean(values)
    deviation = statistics.pstdev(values)
    if deviation == 0.0:
        return tuple(0.0 for _ in values)
    return tuple((value - mean) / deviation for value in values)


def extract_personal_k5(row: Mapping[str, Any]) -> tuple[str, ...]:
    for key in ("personal_candidate_texts_top5", "personal_k5"):
        value = row.get(key)
        if isinstance(value, list):
            result = tuple(str(item) for item in value)
            if len(result) > 5:
                raise RuntimeError(f"{key} > 5 at {row.get('row_id')}")
            return result
    value = row.get("personal_candidates_top5")
    if isinstance(value, list):
        result = tuple(candidate_text(item) if isinstance(item, Mapping) else str(item) for item in value)
        if len(result) > 5:
            raise RuntimeError(f"personal_candidates_top5 > 5 at {row.get('row_id')}")
        return result
    raise RuntimeError(f"No recognized Personal K5 field at {row.get('row_id')}")


def rank_of(ranking: Sequence[Mapping[str, Any]], gold: str) -> int | None:
    for index, item in enumerate(ranking, start=1):
        if candidate_text(item) == gold:
            return int(item.get("rank", index))
    return None


def normalize_nonnegative(values: Sequence[float], *, uniform_if_zero: bool = False) -> list[float]:
    clipped = [max(0.0, float(value)) for value in values]
    total = sum(clipped)
    if total <= 0.0:
        if uniform_if_zero and clipped:
            return [1.0 / len(clipped)] * len(clipped)
        return [0.0] * len(clipped)
    return [value / total for value in clipped]


# ---------------------------------------------------------------------------
# Causal H5000 history
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
    age: int  # 0 = immediately previous same-author interaction in H5000 stream


class CausalHistoryIndex:
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        grouped: dict[str, list[HistoryRecord]] = defaultdict(list)
        for row in rows:
            record = HistoryRecord(
                row_id=str(row["row_id"]),
                author=str(row["author"]),
                position=int(row["chronological_position"]),
                pinyin=pinyin_of(row),
                target=target_of(row),
                context=context_of(row),
            )
            grouped[record.author].append(record)

        self.records: dict[str, tuple[HistoryRecord, ...]] = {}
        self.positions: dict[str, tuple[int, ...]] = {}
        for author, values in grouped.items():
            values.sort(key=lambda r: (r.position, r.row_id))
            positions = tuple(r.position for r in values)
            if len(positions) != len(set((r.position, r.row_id) for r in values)):
                raise RuntimeError(f"History duplicate record identity for author={author}")
            self.records[author] = tuple(values)
            self.positions[author] = positions

    def visible_same_pinyin(
        self,
        *,
        author: str,
        position: int,
        pinyin: tuple[str, ...],
    ) -> tuple[VisibleHistory, ...]:
        records = self.records.get(author, ())
        positions = self.positions.get(author, ())
        stop = bisect.bisect_left(positions, position)
        start = max(0, stop - HISTORY_BUDGET)
        output: list[VisibleHistory] = []
        for index in range(start, stop):
            record = records[index]
            if record.pinyin != pinyin:
                continue
            age = stop - 1 - index
            output.append(VisibleHistory(record=record, age=age))
        return tuple(output)


# ---------------------------------------------------------------------------
# Stage-1 signals and ranking
# ---------------------------------------------------------------------------


def entropy_concentration(counts: Mapping[str, int]) -> float:
    positive = [int(value) for value in counts.values() if int(value) > 0]
    total = sum(positive)
    distinct = len(positive)
    if total <= 0 or distinct == 0:
        return 0.0
    if distinct == 1:
        return 1.0
    shares = [value / total for value in positive]
    entropy = -sum(p * math.log(p) for p in shares if p > 0.0)
    entropy_norm = entropy / math.log(distinct)
    return max(0.0, min(1.0, 1.0 - entropy_norm))


def frequency_support_for_personal(
    personal_k5: Sequence[str], counts: Mapping[str, int]
) -> dict[str, float]:
    raw = {candidate: math.log1p(int(counts.get(candidate, 0))) for candidate in personal_k5}
    maximum = max(raw.values(), default=0.0)
    if maximum <= 0.0:
        return {candidate: 0.0 for candidate in personal_k5}
    return {candidate: value / maximum for candidate, value in raw.items()}


def adaptive_interpolated_support(
    adaptive_row: Mapping[str, Any], personal_k5: Sequence[str]
) -> dict[str, float]:
    methods = adaptive_row.get("methods")
    if not isinstance(methods, Mapping) or INTERPOLATED_KEY not in methods:
        raise RuntimeError(
            f"Adaptive scores missing {INTERPOLATED_KEY} at row={adaptive_row.get('row_id')}"
        )
    method = methods[INTERPOLATED_KEY]
    candidates = tuple(str(value) for value in method.get("candidates", []))
    expected = tuple(personal_k5)
    if candidates != expected:
        raise RuntimeError(
            f"Adaptive K5 candidate mismatch at {adaptive_row.get('row_id')}: "
            f"expected={expected} actual={candidates}"
        )
    support = list(method.get("support", []))
    if len(support) != len(candidates):
        raise RuntimeError(f"Adaptive support length mismatch at {adaptive_row.get('row_id')}")
    values = [float(value) for value in support]
    if values and not math.isclose(sum(values), 1.0, rel_tol=1e-7, abs_tol=1e-7):
        raise RuntimeError(
            f"Adaptive Interpolated support does not sum to 1 at {adaptive_row.get('row_id')}: "
            f"sum={sum(values)}"
        )
    return dict(zip(candidates, values))


def generic_frequency_rows(pred_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = pred_row.get("frequency_candidates")
    if not isinstance(values, list) or not values:
        raise RuntimeError(f"Missing frequency_candidates at {pred_row.get('row_id')}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(values, start=1):
        if not isinstance(item, Mapping):
            raise RuntimeError(f"Bad frequency candidate at {pred_row.get('row_id')}")
        text = candidate_text(item)
        if text in seen:
            raise RuntimeError(f"Duplicate Frequency candidate {text} at {pred_row.get('row_id')}")
        seen.add(text)
        if item.get("final_score") is None:
            raise RuntimeError(f"Frequency candidate lacks final_score at {pred_row.get('row_id')}: {text}")
        row = dict(item)
        row["candidate"] = text
        row["final_score"] = float(item["final_score"])
        row["source"] = "generic"
        row["base_tiebreak_rank"] = int(item.get("rank", item.get("generic_rank", index)))
        rows.append(row)
    return rows


def generic_boundary_from_frequency(generic_rows: Sequence[Mapping[str, Any]]) -> float:
    values = []
    for row in generic_rows:
        if row.get("normalized_generic_score") is None:
            raise RuntimeError("Frozen Frequency row lacks normalized_generic_score")
        values.append(float(row["normalized_generic_score"]))
    if not values:
        raise RuntimeError("Frozen Frequency candidate list is empty")
    return min(values)


def ngram_rank_map(personal_k5: Sequence[str], p_ng: Mapping[str, float]) -> dict[str, int]:
    order = sorted(
        range(len(personal_k5)),
        key=lambda i: (-float(p_ng[personal_k5[i]]), i, str(personal_k5[i])),
    )
    return {personal_k5[index]: rank for rank, index in enumerate(order, start=1)}


def merge_stage1(
    *,
    generic_rows: Sequence[Mapping[str, Any]],
    personal_k5: Sequence[str],
    personal_scores: Mapping[str, float],
    personal_tiebreak_rank: Mapping[str, int],
) -> list[dict[str, Any]]:
    generic_texts = {candidate_text(item) for item in generic_rows}
    overlap = generic_texts.intersection(personal_k5)
    if overlap:
        raise RuntimeError(f"Personal K5 overlaps Generic: {sorted(overlap)}")

    rows: list[dict[str, Any]] = [dict(item) for item in generic_rows]
    for personal_rank, candidate in enumerate(personal_k5, start=1):
        rows.append(
            {
                "candidate": candidate,
                "source": "personal_recovery",
                "generic_rank": None,
                "personal_candidate_rank": int(personal_tiebreak_rank[candidate]),
                "original_personal_frequency_rank": personal_rank,
                "ngram_rank": int(personal_tiebreak_rank[candidate]),
                "final_score": float(personal_scores[candidate]),
                "base_tiebreak_rank": int(personal_tiebreak_rank[candidate]),
            }
        )

    # Same deterministic preference as frozen PV-style merge: generic wins exact ties.
    rows.sort(
        key=lambda row: (
            -float(row["final_score"]),
            0 if row["source"] == "generic" else 1,
            int(row.get("generic_rank") or row.get("personal_candidate_rank") or 0),
            str(row["candidate"]),
        )
    )
    rows = rows[:10]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["base_rank"] = rank
    return rows


def build_stage1_for_row(
    *,
    row_id: str,
    val_row: Mapping[str, Any],
    surface_row: Mapping[str, Any],
    pred_row: Mapping[str, Any],
    adaptive_row: Mapping[str, Any],
    history: CausalHistoryIndex,
) -> dict[str, Any]:
    author = str(val_row["author"])
    pinyin = pinyin_of(val_row)
    visible = history.visible_same_pinyin(
        author=author,
        position=int(val_row["chronological_position"]),
        pinyin=pinyin,
    )
    counts = Counter(item.record.target for item in visible)
    history_total = sum(counts.values())
    personal_k5 = extract_personal_k5(surface_row)

    if any(int(counts.get(candidate, 0)) <= 0 for candidate in personal_k5):
        raise RuntimeError(f"Personal K5 lacks legal history evidence at {row_id}")

    generic_rows = generic_frequency_rows(pred_row)
    boundary = generic_boundary_from_frequency(generic_rows)
    f_support = frequency_support_for_personal(personal_k5, counts)
    cs_support = {
        candidate: (int(counts.get(candidate, 0)) / history_total if history_total else 0.0)
        for candidate in personal_k5
    }
    p_ng = adaptive_interpolated_support(adaptive_row, personal_k5)
    personal_ngram_rank = ngram_rank_map(personal_k5, p_ng)
    c_e = entropy_concentration(counts)

    personal_scores_by_base: dict[str, dict[str, float]] = {
        "K5+Entropy": {
            candidate: boundary + 4.0 * f_support[candidate] + 0.25 * c_e
            for candidate in personal_k5
        },
        "4P+4CS+2E": {
            candidate: boundary + 4.0 * p_ng[candidate] + 4.0 * cs_support[candidate] + 2.0 * c_e
            for candidate in personal_k5
        },
        "6P+2CS+.25E": {
            candidate: boundary + 6.0 * p_ng[candidate] + 2.0 * cs_support[candidate] + 0.25 * c_e
            for candidate in personal_k5
        },
    }

    rankings = {
        base: merge_stage1(
            generic_rows=generic_rows,
            personal_k5=personal_k5,
            personal_scores=personal_scores_by_base[base],
            personal_tiebreak_rank=personal_ngram_rank,
        )
        for base in BASE_ORDER
    }

    generic_texts = {candidate_text(item) for item in generic_rows}
    gold = target_of(val_row)
    generic_missing = gold not in generic_texts
    gold_in_k5 = gold in set(personal_k5)

    return {
        "schema_version": 1,
        "row_id": row_id,
        "author": author,
        "chronological_position": int(val_row["chronological_position"]),
        "gold": gold,
        "pinyin_segments": list(pinyin),
        "generic_missing": generic_missing,
        "gold_in_personal_k5": gold_in_k5,
        "same_pinyin_history_count": history_total,
        "entropy_concentration": c_e,
        "personal_k5": list(personal_k5),
        "choice_share": cs_support,
        "personal_frequency_support": f_support,
        "interpolated_ngram_support": p_ng,
        "bases": {
            base: {
                "top10": [str(item["candidate"]) for item in rankings[base]],
                "candidates": rankings[base],
                "gold_rank": rank_of(rankings[base], gold),
            }
            for base in BASE_ORDER
        },
        "gold_used_for_scoring": False,
    }


# ---------------------------------------------------------------------------
# Stage-2 HardBackoff NGramRecency
# ---------------------------------------------------------------------------


def suffix_matches(query_context: str, history_context: str, order: int) -> bool:
    if order <= 0:
        return True
    if len(query_context) < order or len(history_context) < order:
        return False
    return history_context[-order:] == query_context[-order:]


def ngram_recency_support(
    *,
    query_context: str,
    candidates: Sequence[str],
    visible: Sequence[VisibleHistory],
    max_n: int = NGRAM_MAX_N,
    tau: float = NGRAM_TAU,
) -> tuple[dict[str, float], int, int]:
    candidate_set = set(candidates)
    candidate_history = [item for item in visible if item.record.target in candidate_set]

    effective_n = 0
    matched: list[VisibleHistory] = candidate_history
    for order in range(min(max_n, len(query_context)), 0, -1):
        current = [
            item
            for item in candidate_history
            if suffix_matches(query_context, item.record.context, order)
        ]
        if current:
            effective_n = order
            matched = current
            break

    raw = {candidate: 0.0 for candidate in candidates}
    for item in matched:
        raw[item.record.target] += math.exp(-float(item.age) / tau)

    values = [raw[candidate] for candidate in candidates]
    normalized = normalize_nonnegative(values, uniform_if_zero=True)
    return dict(zip(candidates, normalized)), effective_n, len(matched)


def rerank_fixed_candidate_set(
    base_candidates: Sequence[Mapping[str, Any]],
    support: Mapping[str, float],
    lambda_n: float,
) -> list[dict[str, Any]]:
    base_texts = [candidate_text(item) for item in base_candidates]
    if set(base_texts) != set(support):
        raise RuntimeError("Context support candidate set differs from frozen Stage-1 Top10")

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(base_candidates, start=1):
        row = dict(item)
        base_rank = int(item.get("base_rank", item.get("rank", index)))
        base_score = float(item["final_score"])
        context_score = float(support[candidate_text(item)])
        row["base_rank"] = base_rank
        row["base_score"] = base_score
        row["ngram_recency_support"] = context_score
        row["context_lambda_n"] = float(lambda_n)
        row["final_score"] = base_score + float(lambda_n) * context_score
        rows.append(row)

    # Exact lambda=0 recovery reproduction and stable tie handling via base rank.
    rows.sort(key=lambda row: (-float(row["final_score"]), int(row["base_rank"]), str(row["candidate"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metric_summary(rows: Sequence[Mapping[str, Any]], rank_key: str, method: str) -> dict[str, Any]:
    by_author: dict[str, list[int | None]] = defaultdict(list)
    ranks: list[int | None] = []
    for row in rows:
        rank = row.get(rank_key)
        rank = None if rank is None else int(rank)
        ranks.append(rank)
        by_author[str(row["author"])].append(rank)

    n = len(ranks)
    if not n:
        raise RuntimeError("Cannot score empty row set")

    def top_at(values: Sequence[int | None], k: int) -> float:
        return sum(rank is not None and rank <= k for rank in values) / len(values)

    per_author_top1 = {
        author: top_at(values, 1) for author, values in sorted(by_author.items())
    }
    found = [rank for rank in ranks if rank is not None]
    return {
        "method": method,
        "n": n,
        "authors": len(per_author_top1),
        "macro_author_top1": statistics.fmean(per_author_top1.values()),
        "micro_top1": top_at(ranks, 1),
        "top3": top_at(ranks, 3),
        "top5": top_at(ranks, 5),
        "mrr_at_10": sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / n,
        "missing10": sum(rank is None for rank in ranks) / n,
        "mean_rank_given_top10": statistics.fmean(found) if found else None,
        "per_author_top1": per_author_top1,
    }


def recovery_summary(rows: Sequence[Mapping[str, Any]], rank_key: str) -> dict[str, Any]:
    missing = [row for row in rows if bool(row["generic_missing"])]
    available = [row for row in missing if bool(row["gold_in_personal_k5"])]
    if len(missing) != EXPECTED_GENERIC_MISSING:
        raise RuntimeError(f"Generic Missing regression failed: {len(missing)}")
    if len(available) != EXPECTED_RECOVERABLE_K5:
        raise RuntimeError(f"K5 recoverable population regression failed: {len(available)}")

    ranks = [None if row.get(rank_key) is None else int(row[rank_key]) for row in available]

    def count_at(k: int) -> int:
        return sum(rank is not None and rank <= k for rank in ranks)

    r1, r3, r5, r10 = (count_at(k) for k in (1, 3, 5, 10))
    recovered = [rank for rank in ranks if rank is not None]
    denom = len(available)
    return {
        "generic_missing_n": len(missing),
        "recoverable_n": denom,
        "recoverable_rate_given_generic_missing": denom / len(missing),
        "recovered_at_1_n": r1,
        "recovered_at_3_n": r3,
        "recovered_at_5_n": r5,
        "recovered_at_10_n": r10,
        "recovered_at_1": r1 / denom,
        "recovered_at_3": r3 / denom,
        "recovered_at_5": r5 / denom,
        "recovered_at_10": r10 / denom,
        "recovery_mrr_at_10": sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / denom,
        "mean_recovered_rank": statistics.fmean(recovered) if recovered else None,
    }


def transition_counts(rows: Sequence[Mapping[str, Any]], base_key: str, new_key: str) -> dict[str, int]:
    result = {
        "n": 0,
        "rescue": 0,
        "harm": 0,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
    }
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
    result = {
        "n": len(rows),
        "present_both": 0,
        "improved": 0,
        "worsened": 0,
        "same": 0,
        "missing_both": 0,
        "availability_changed": 0,
    }
    for row in rows:
        before = row.get(base_key)
        after = row.get(new_key)
        if before is None and after is None:
            result["missing_both"] += 1
            continue
        if (before is None) != (after is None):
            result["availability_changed"] += 1
            continue
        result["present_both"] += 1
        before_i, after_i = int(before), int(after)
        if after_i < before_i:
            result["improved"] += 1
        elif after_i > before_i:
            result["worsened"] += 1
        else:
            result["same"] += 1
    result["net_improved_minus_worsened"] = result["improved"] - result["worsened"]
    return result


def assert_close(label: str, actual: float, expected: float, tolerance: float) -> None:
    if abs(float(actual) - float(expected)) > tolerance:
        raise RuntimeError(
            f"Stage-1 regression failed for {label}: actual={actual:.12f} "
            f"expected={expected:.12f} tolerance={tolerance}"
        )


def regression_check_stage1(base: str, metrics: Mapping[str, Any], recovery: Mapping[str, Any]) -> None:
    expected = EXPECTED_STAGE1[base]
    # K5+Entropy targets are archived to ~4-6 decimals; anchors have exact values.
    tol = 6e-5 if base == "K5+Entropy" else 1e-9
    for key in ("macro_author_top1", "micro_top1", "top3", "top5", "mrr_at_10", "missing10"):
        assert_close(f"{base}.{key}", float(metrics[key]), float(expected[key]), tol)
    recovery_map = {
        "rec1": "recovered_at_1",
        "rec3": "recovered_at_3",
        "rec5": "recovered_at_5",
        "rec10": "recovered_at_10",
        "recovery_mrr_at_10": "recovery_mrr_at_10",
    }
    for expected_key, actual_key in recovery_map.items():
        assert_close(
            f"{base}.{actual_key}",
            float(recovery[actual_key]),
            float(expected[expected_key]),
            tol,
        )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def verify_inputs(args: argparse.Namespace) -> dict[str, Any]:
    required = {
        "fit": args.fit,
        "val": args.val,
        "candidate_surface": args.candidate_surface,
        "frequency_pv1_predictions": args.frequency_pv1_predictions,
        "adaptive_scores": args.adaptive_scores,
    }
    for label, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")

    hashes = {
        "fit": sha256_file(args.fit),
        "val": sha256_file(args.val),
        "candidate_surface": sha256_file(args.candidate_surface),
        "frequency_pv1_predictions": sha256_file(args.frequency_pv1_predictions),
        "adaptive_scores": sha256_file(args.adaptive_scores),
    }
    frozen_expected = {
        "fit": EXPECTED_FIT_SHA256,
        "val": EXPECTED_VAL_SHA256,
        "candidate_surface": EXPECTED_SURFACE_SHA256,
        "frequency_pv1_predictions": EXPECTED_FREQUENCY_PV1_SHA256,
    }
    for key, expected in frozen_expected.items():
        if hashes[key] != expected:
            raise RuntimeError(
                f"Frozen SHA mismatch for {key}:\nexpected={expected}\nactual={hashes[key]}"
            )
    return hashes


def refuse_nonempty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite non-empty output directory: {path}\n"
            "Choose a new versioned --output-root."
        )
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--candidate-surface", type=Path, required=True)
    parser.add_argument("--frequency-pv1-predictions", type=Path, required=True)
    parser.add_argument("--adaptive-scores", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--lambda-n",
        default=",".join(str(value) for value in DEFAULT_LAMBDAS),
        help="Comma-separated NGramRecency lambda grid; must include 0",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    lambdas = parse_grid(args.lambda_n)
    refuse_nonempty_output(args.output_root)
    input_hashes = verify_inputs(args)

    started = time.perf_counter()
    print("=== RECOVERY -> NGRAMRECENCY FUSION V1 ===")
    print(f"Recovery bases: {list(BASE_ORDER)}")
    print(f"NGramRecency: HardBackoff maxN={NGRAM_MAX_N}, tau={NGRAM_TAU:g}")
    print(f"lambda_N grid: {list(lambdas)}")
    print("Candidate set: frozen separately for each Recovery base; Stage 2 is pure reranking")
    print("Gold used for scoring/features: false")
    print("Gold used for Train-Val lambda selection/evaluation only: true")
    print("Dev3000 used: false")
    print("Test used: false")
    print()

    fit_rows = read_jsonl(args.fit)
    val_rows = read_jsonl(args.val)
    surface_rows = read_jsonl(args.candidate_surface)
    pred_rows = read_jsonl(args.frequency_pv1_predictions)
    adaptive_rows = read_jsonl(args.adaptive_scores)

    if len(fit_rows) != EXPECTED_FIT_ROWS:
        raise RuntimeError(f"Unexpected Train-Fit rows: {len(fit_rows)}")
    if len(val_rows) != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"Unexpected Train-Val rows: {len(val_rows)}")

    val = index_rows(val_rows, "Train-Val")
    surface = index_rows(surface_rows, "candidate surface")
    pred = index_rows(pred_rows, "Frequency/PV1 predictions")
    adaptive = index_rows(adaptive_rows, "adaptive NGram scores")

    if not (set(val) == set(surface) == set(pred) == set(adaptive)):
        raise RuntimeError(
            "Row-ID mismatch among Train-Val / candidate surface / F-PV1 predictions / adaptive scores"
        )

    history = CausalHistoryIndex([*fit_rows, *val_rows])

    # ------------------------------------------------------------------
    # Stage 1: reconstruct and freeze the three recovery bases.
    # ------------------------------------------------------------------
    stage1_rows: list[dict[str, Any]] = []
    print("Reconstructing frozen Stage-1 bases ...")
    t_stage1 = time.perf_counter()
    for number, row_id in enumerate(sorted(val), start=1):
        stage1_rows.append(
            build_stage1_for_row(
                row_id=row_id,
                val_row=val[row_id],
                surface_row=surface[row_id],
                pred_row=pred[row_id],
                adaptive_row=adaptive[row_id],
                history=history,
            )
        )
        if args.progress_every > 0 and (number % args.progress_every == 0 or number == len(val)):
            elapsed = time.perf_counter() - t_stage1
            print(f"Stage1 {number}/{len(val)}  rate={number / elapsed:.1f} rows/s", flush=True)

    stage1_path = args.output_root / "stage1_frozen.jsonl"

    # Flatten ranks for metrics and regression checks BEFORE freezing any output.
    stage1_eval: dict[str, list[dict[str, Any]]] = {base: [] for base in BASE_ORDER}
    stage1_metrics: dict[str, dict[str, Any]] = {}
    stage1_recovery: dict[str, dict[str, Any]] = {}
    for row in stage1_rows:
        for base in BASE_ORDER:
            stage1_eval[base].append(
                {
                    "row_id": row["row_id"],
                    "author": row["author"],
                    "generic_missing": row["generic_missing"],
                    "gold_in_personal_k5": row["gold_in_personal_k5"],
                    "rank": row["bases"][base]["gold_rank"],
                }
            )
        
    print("\nStage-1 regression checks:")
    for base in BASE_ORDER:
        metrics = metric_summary(stage1_eval[base], "rank", base)
        recovery = recovery_summary(stage1_eval[base], "rank")
        regression_check_stage1(base, metrics, recovery)
        stage1_metrics[base] = metrics
        stage1_recovery[base] = recovery
        print(
            f"  PASS {base:16s} Macro={metrics['macro_author_top1']:.6f} "
            f"Missing={metrics['missing10']:.6f} "
            f"Rec1={recovery['recovered_at_1']:.4f} "
            f"Rec3={recovery['recovered_at_3']:.4f} "
            f"Rec10={recovery['recovered_at_10']:.4f}"
        )

    # Only freeze Stage 1 after every historical regression check passes.
    write_jsonl(stage1_path, stage1_rows)

    # ------------------------------------------------------------------
    # Stage 2 support: compute NGramRecency on each base's frozen Top10.
    # ------------------------------------------------------------------
    print("\nComputing NGramRecency support on each frozen Stage-1 Top10 ...")
    support_rows: list[dict[str, Any]] = []
    t_support = time.perf_counter()
    for number, stage1 in enumerate(stage1_rows, start=1):
        row_id = str(stage1["row_id"])
        val_row = val[row_id]
        visible = history.visible_same_pinyin(
            author=str(val_row["author"]),
            position=int(val_row["chronological_position"]),
            pinyin=pinyin_of(val_row),
        )
        query_context = context_of(val_row)
        base_payload: dict[str, Any] = {}
        for base in BASE_ORDER:
            candidates = tuple(str(value) for value in stage1["bases"][base]["top10"])
            support, effective_n, matched = ngram_recency_support(
                query_context=query_context,
                candidates=candidates,
                visible=visible,
            )
            base_payload[base] = {
                "candidates": list(candidates),
                "support": support,
                "effective_n": effective_n,
                "matched_history_rows": matched,
            }
        support_rows.append(
            {
                "schema_version": 1,
                "row_id": row_id,
                "author": stage1["author"],
                "bases": base_payload,
                "max_n": NGRAM_MAX_N,
                "tau": NGRAM_TAU,
                "gold_used_for_scoring": False,
            }
        )
        if args.progress_every > 0 and (number % args.progress_every == 0 or number == len(stage1_rows)):
            elapsed = time.perf_counter() - t_support
            print(f"Context {number}/{len(stage1_rows)}  rate={number / elapsed:.1f} rows/s", flush=True)

    support_path = args.output_root / "ngram_recency_support.jsonl"
    write_jsonl(support_path, support_rows)
    support_by_id = index_rows(support_rows, "NGramRecency support")

    # ------------------------------------------------------------------
    # Lambda grid: pure reranking only.
    # ------------------------------------------------------------------
    grid_rows: list[dict[str, Any]] = []
    eval_cache: dict[tuple[str, float], list[dict[str, Any]]] = {}
    prediction_cache: dict[tuple[str, float], list[dict[str, Any]]] = {}

    print("\nEvaluating lambda_N grid ...")
    for base in BASE_ORDER:
        base_missing = stage1_metrics[base]["missing10"]
        base_rec10 = stage1_recovery[base]["recovered_at_10"]
        for lambda_n in lambdas:
            eval_rows: list[dict[str, Any]] = []
            prediction_rows: list[dict[str, Any]] = []
            for stage1 in stage1_rows:
                row_id = str(stage1["row_id"])
                base_candidates = stage1["bases"][base]["candidates"]
                support_info = support_by_id[row_id]["bases"][base]
                support = {str(k): float(v) for k, v in support_info["support"].items()}
                ranked = rerank_fixed_candidate_set(base_candidates, support, lambda_n)

                before_set = {candidate_text(item) for item in base_candidates}
                after_set = {candidate_text(item) for item in ranked}
                if before_set != after_set:
                    raise RuntimeError(f"Candidate-set invariant failed at {row_id} {base} lambda={lambda_n}")
                if lambda_n == 0.0:
                    before_order = [candidate_text(item) for item in base_candidates]
                    after_order = [candidate_text(item) for item in ranked]
                    if before_order != after_order:
                        raise RuntimeError(f"lambda=0 did not reproduce Stage1 ordering at {row_id} {base}")

                gold = str(stage1["gold"])
                base_rank = stage1["bases"][base]["gold_rank"]
                new_rank = rank_of(ranked, gold)
                eval_rows.append(
                    {
                        "row_id": row_id,
                        "author": stage1["author"],
                        "generic_missing": stage1["generic_missing"],
                        "gold_in_personal_k5": stage1["gold_in_personal_k5"],
                        "base_rank": base_rank,
                        "rank": new_rank,
                    }
                )
                prediction_rows.append(
                    {
                        "schema_version": 1,
                        "row_id": row_id,
                        "author": stage1["author"],
                        "gold": gold,
                        "base": base,
                        "lambda_n": lambda_n,
                        "base_gold_rank": base_rank,
                        "gold_rank": new_rank,
                        "top10": [candidate_text(item) for item in ranked],
                        "candidates": ranked,
                    }
                )

            metrics = metric_summary(eval_rows, "rank", f"{base}+NGramRecency|lambda={lambda_n:g}")
            recovery = recovery_summary(eval_rows, "rank")
            transitions = transition_counts(eval_rows, "base_rank", "rank")
            rank_transitions = rank_transition_counts(eval_rows, "base_rank", "rank")

            # Pure-reranking invariants.
            if metrics["missing10"] != base_missing:
                raise RuntimeError(
                    f"Missing@10 changed under pure reranking for {base} lambda={lambda_n}: "
                    f"{base_missing} -> {metrics['missing10']}"
                )
            if recovery["recovered_at_10"] != base_rec10:
                raise RuntimeError(
                    f"Recovery Rec@10 changed under pure reranking for {base} lambda={lambda_n}: "
                    f"{base_rec10} -> {recovery['recovered_at_10']}"
                )
            if rank_transitions["availability_changed"] != 0:
                raise RuntimeError(f"Gold availability changed for {base} lambda={lambda_n}")

            flat = {
                "base": base,
                "lambda_n": lambda_n,
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
                "top1_rescue": transitions["rescue"],
                "top1_harm": transitions["harm"],
                "top1_net": transitions["net"],
                "rank_improved": rank_transitions["improved"],
                "rank_worsened": rank_transitions["worsened"],
                "rank_same": rank_transitions["same"],
                "rank_net": rank_transitions["net_improved_minus_worsened"],
            }
            grid_rows.append(flat)
            eval_cache[(base, lambda_n)] = eval_rows
            prediction_cache[(base, lambda_n)] = prediction_rows

    grid_path = args.output_root / "grid_results.csv"
    write_csv(grid_path, grid_rows)

    # Select independently per recovery base: Macro-author Top1, tie MRR, then smaller lambda.
    selected: dict[str, dict[str, Any]] = {}
    selected_metric_rows: list[dict[str, Any]] = []
    selected_prediction_rows: list[dict[str, Any]] = []
    for base in BASE_ORDER:
        candidates = [row for row in grid_rows if row["base"] == base]
        best = max(
            candidates,
            key=lambda row: (
                float(row["macro_author_top1"]),
                float(row["mrr_at_10"]),
                -float(row["lambda_n"]),
            ),
        )
        lambda_n = float(best["lambda_n"])
        selected[base] = dict(best)
        selected_metric_rows.append(dict(best))
        selected_prediction_rows.extend(prediction_cache[(base, lambda_n)])

    selected_metrics_path = args.output_root / "selected_metrics.csv"
    write_csv(selected_metrics_path, selected_metric_rows)
    selected_predictions_path = args.output_root / "selected_predictions.jsonl"
    write_jsonl(selected_predictions_path, selected_prediction_rows)

    comparison = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_recovery_ngram_context_fusion_v1",
        "research_question": "Which Stage-1 recovery philosophy is most compatible with Stage-2 NGramRecency reranking?",
        "rows": len(stage1_rows),
        "recovery_bases": {
            "K5+Entropy": "coverage-first",
            "4P+4CS+2E": "balanced",
            "6P+2CS+.25E": "front-rank / Top3-oriented",
        },
        "stage1_formulas": {
            "K5+Entropy": "B + 4*F_PV(c) + 0.25*C_E(q)",
            "4P+4CS+2E": "B + 4*P_NG(c) + 4*CS(c) + 2*C_E(q)",
            "6P+2CS+.25E": "B + 6*P_NG(c) + 2*CS(c) + 0.25*C_E(q)",
            "P_NG": INTERPOLATED_KEY,
        },
        "stage2_formula": "S_final(c) = S_REC(c) + lambda_N * P_NG-R(c)",
        "ngram_recency": {
            "type": "HardBackoffNGramRecency",
            "max_n": NGRAM_MAX_N,
            "tau": NGRAM_TAU,
            "age_semantics": "age=0 is immediately previous same-author interaction in the raw H5000 author stream",
            "history_semantics": "same author -> strictly prior -> latest H5000 RAW -> exact Pinyin filter",
        },
        "lambda_n_grid": list(lambdas),
        "selection_rule": "per base: maximize Macro-author Top1; tie MRR@10; then smaller lambda_N",
        "stage1_metrics": stage1_metrics,
        "stage1_recovery": stage1_recovery,
        "selected_by_base": selected,
        "invariants": {
            "candidate_set_changed": False,
            "missing10_must_equal_stage1": True,
            "recovery_rec10_must_equal_stage1_on_fixed_R": True,
            "lambda0_must_exactly_reproduce_stage1_order": True,
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
            "adaptive_scores_fixed_method": INTERPOLATED_KEY,
            "stage1_frozen_sha256": sha256_file(stage1_path),
            "ngram_recency_support_sha256": sha256_file(support_path),
            "grid_results_sha256": sha256_file(grid_path),
            "selected_metrics_sha256": sha256_file(selected_metrics_path),
            "selected_predictions_sha256": sha256_file(selected_predictions_path),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    comparison_path = args.output_root / "comparison.json"
    write_json(comparison_path, comparison)

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "experiment": comparison["experiment"],
        "output_root": str(args.output_root),
        "inputs": {
            "fit": str(args.fit),
            "val": str(args.val),
            "candidate_surface": str(args.candidate_surface),
            "frequency_pv1_predictions": str(args.frequency_pv1_predictions),
            "adaptive_scores": str(args.adaptive_scores),
        },
        "input_sha256": input_hashes,
        "dev3000_used": False,
        "test_used": False,
    }
    manifest_path = args.output_root / "run_manifest.json"
    write_json(manifest_path, manifest)

    outputs = [
        stage1_path,
        support_path,
        grid_path,
        selected_metrics_path,
        selected_predictions_path,
        comparison_path,
        manifest_path,
    ]
    checksums = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in outputs
    }
    checksums_path = args.output_root / "artifact_checksums.json"
    write_json(checksums_path, checksums)

    print("\n=== SELECTED NGRAMRECENCY RESULT PER RECOVERY BASE ===")
    for base in BASE_ORDER:
        row = selected[base]
        print(
            f"{base:16s}  lambda_N={float(row['lambda_n']):g}  "
            f"Macro={float(row['macro_author_top1']):.6f}  "
            f"Micro={float(row['micro_top1']):.6f}  "
            f"Top3={float(row['top3']):.6f}  "
            f"MRR={float(row['mrr_at_10']):.6f}  "
            f"Missing={float(row['missing10']):.6f}  "
            f"Rec1={float(row['recovery_rec1']):.4f}  "
            f"Rec3={float(row['recovery_rec3']):.4f}  "
            f"Rec10={float(row['recovery_rec10']):.4f}  "
            f"Top1Net={int(row['top1_net']):+d}"
        )

    print("\nOutputs:")
    for path in [*outputs, checksums_path]:
        print(f"  {path}")
    print(f"\nRuntime: {comparison['runtime_seconds']:.1f}s")
    print("Dev3000 used: false")
    print("Test used: false")


if __name__ == "__main__":
    main()
