"""Initial personal-candidate scorer study: suffix N-gram vs N-gram+Recency.

This is a CPU-only companion to the Q8/BGE64 candidate-scoring experiment.
It keeps the frozen Personal K<=5 candidate surface and the same causal H5000
history semantics, then asks whether a very transparent lexical-context scorer
can distinguish personal candidates.

Methods
-------
F
    Baseline normalized log(1 + historical count) over legal same-Pinyin
    history for the current Personal K<=5 candidates.

NGram(N)
    Transparent suffix backoff. For a configured maximum order N, choose the
    largest k in N..1 for which at least one legal same-Pinyin historical row
    (whose target is in the Personal K<=5 set) has the same last-k Python
    Unicode code points as the current query context. If no such lexical
    context match exists, k=0 and all legal same-Pinyin history is used.
    Candidate raw support is the number of matched rows for that target,
    normalized across the current Personal K<=5 candidates.

NGramRecency(N, tau)
    Uses the same suffix-backoff rule, but each matched historical row is
    weighted by exp(-age/tau). age is the author's interaction distance inside
    the H5000 window: age=0 means the immediately previous same-author row,
    regardless of Pinyin. This preserves "H5000 before Pinyin filtering".

Frozen default grids
--------------------
    N in {1, 2, 3, 4, 6, 8}
    tau in {32, 128, 512, 2048}

Selection / leakage policy
--------------------------
- Gold is never used to build candidates, retrieve history, choose the backoff
  level, or compute any scorer support.
- Gold is used only in the final Train-Val evaluation to select the best N and
  best (N, tau) by Macro-author Top1 on Gold-in-Personal K>=2.
- Dev3000 is not read.
- Test is not read.

Outputs
-------
<output-root>/ngram_recency/
    ngram_recency_scores.jsonl
    progress.json
    score_summary.json
    candidate_scoring_comparison.json

The script is resumable at row granularity.
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
# Frozen provenance / workload
# ---------------------------------------------------------------------------

EXPECTED_SURFACE_SHA256 = "205c0ba01cd0678d7a4341c503fa2e74cf126a70182ff582687025e4946764b2"
EXPECTED_GENERIC_SHA256 = "bd0fb4dc304e0b266b90fae6fe3ac65424d2f52b23fedfa881212706ba2c2873"
EXPECTED_FIT_SHA256 = "162f5c98daa86cc69947571e6d8f20fc401f0a82cdd3fd6e517eb7be2addbdb4"
EXPECTED_VAL_SHA256 = "d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4"

EXPECTED_SURFACE_ROWS = 34416
EXPECTED_VAL_ROWS = 34416
EXPECTED_FIT_ROWS = 144526
EXPECTED_ELIGIBLE_ROWS = 30509
EXPECTED_K5_PAIRS = 123738

MAX_K = 5
HISTORY_BUDGET = 5000
DEFAULT_N_VALUES = (1, 2, 3, 4, 6, 8)
DEFAULT_TAU_VALUES = (32.0, 128.0, 512.0, 2048.0)


# ---------------------------------------------------------------------------
# Generic utilities
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
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def index_rows(rows: Iterable[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row["row_id"])
        if row_id in output:
            raise RuntimeError(f"Duplicate {label} row_id: {row_id}")
        output[row_id] = row
    return output


def candidate_texts_top5(row: Mapping[str, Any]) -> tuple[str, ...]:
    values = row.get("personal_candidate_texts_top5")
    if isinstance(values, list):
        texts = [str(value) for value in values[:MAX_K]]
    else:
        values = row.get("personal_candidates_top5")
        if not isinstance(values, list):
            raise RuntimeError(f"{row.get('row_id')}: no usable personal top-5 field")
        texts: list[str] = []
        for item in values[:MAX_K]:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("candidate") or item.get("target")
                if text is None:
                    raise RuntimeError(f"{row.get('row_id')}: candidate dict has no text")
                texts.append(str(text))
            else:
                raise RuntimeError(f"{row.get('row_id')}: unsupported candidate {item!r}")
    return tuple(dict.fromkeys(texts))


def generic_candidate_texts(row: Mapping[str, Any]) -> tuple[str, ...]:
    values = row.get("generic_candidates")
    if not isinstance(values, list):
        return ()
    output: list[str] = []
    for item in values:
        if isinstance(item, str):
            output.append(item)
        elif isinstance(item, dict):
            text = item.get("text") or item.get("candidate") or item.get("target")
            if text is not None:
                output.append(str(text))
    return tuple(output)


def normalize_nonnegative(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    clipped = [max(0.0, float(value)) for value in values]
    total = sum(clipped)
    if total <= 0:
        return [1.0 / len(clipped)] * len(clipped)
    return [value / total for value in clipped]


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    return ordered[int((len(ordered) - 1) * fraction)]


def latency_summary(values_ms: Sequence[float]) -> dict[str, Any]:
    values = [float(v) for v in values_ms]
    mean = statistics.fmean(values) if values else None
    return {
        "n": len(values),
        "mean": mean,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "queries_per_second_from_mean": 1000.0 / mean if mean and mean > 0 else None,
    }


def rank_candidates(candidates: Sequence[str], distribution: Sequence[float]) -> list[str]:
    return [
        candidates[index]
        for index in sorted(
            range(len(candidates)),
            key=lambda i: (-float(distribution[i]), i, candidates[i]),
        )
    ]


def gold_rank(ranking: Sequence[str], gold: str) -> int | None:
    for index, candidate in enumerate(ranking, start=1):
        if candidate == gold:
            return index
    return None


# ---------------------------------------------------------------------------
# Frozen inputs
# ---------------------------------------------------------------------------


@dataclass
class Inputs:
    fit_rows: list[dict[str, Any]]
    val_rows: list[dict[str, Any]]
    surface_rows: list[dict[str, Any]]
    generic_rows: list[dict[str, Any]]
    fit: dict[str, dict[str, Any]]
    val: dict[str, dict[str, Any]]
    surface: dict[str, dict[str, Any]]
    generic: dict[str, dict[str, Any]]


def load_and_verify(args: argparse.Namespace) -> Inputs:
    expected = (
        (args.fit, EXPECTED_FIT_SHA256, "Train-Fit"),
        (args.val, EXPECTED_VAL_SHA256, "Train-Val"),
        (args.candidate_surface, EXPECTED_SURFACE_SHA256, "candidate surface"),
        (args.generic_predictions, EXPECTED_GENERIC_SHA256, "Generic predictions"),
    )
    for path, expected_sha, label in expected:
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected_sha:
            raise RuntimeError(
                f"Frozen {label} SHA mismatch:\nexpected={expected_sha}\nactual={actual}\npath={path}"
            )

    fit_rows = read_jsonl(args.fit)
    val_rows = read_jsonl(args.val)
    surface_rows = read_jsonl(args.candidate_surface)
    generic_rows = read_jsonl(args.generic_predictions)

    if len(fit_rows) != EXPECTED_FIT_ROWS:
        raise RuntimeError(f"Unexpected Train-Fit rows: {len(fit_rows)}")
    if len(val_rows) != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"Unexpected Train-Val rows: {len(val_rows)}")
    if len(surface_rows) != EXPECTED_SURFACE_ROWS:
        raise RuntimeError(f"Unexpected surface rows: {len(surface_rows)}")
    if len(generic_rows) != EXPECTED_SURFACE_ROWS:
        raise RuntimeError(f"Unexpected Generic rows: {len(generic_rows)}")

    fit = index_rows(fit_rows, label="fit")
    val = index_rows(val_rows, label="val")
    surface = index_rows(surface_rows, label="surface")
    generic = index_rows(generic_rows, label="generic")

    if set(val) != set(surface) or set(surface) != set(generic):
        raise RuntimeError("Train-Val / candidate surface / Generic row IDs differ")

    pair_count = sum(len(candidate_texts_top5(row)) for row in surface_rows)
    eligible = sum(bool(candidate_texts_top5(row)) for row in surface_rows)
    if pair_count != EXPECTED_K5_PAIRS:
        raise RuntimeError(f"K5 workload changed: expected={EXPECTED_K5_PAIRS} actual={pair_count}")
    if eligible != EXPECTED_ELIGIBLE_ROWS:
        raise RuntimeError(
            f"Eligible rows changed: expected={EXPECTED_ELIGIBLE_ROWS} actual={eligible}"
        )

    for row_id in val:
        p_val = tuple(str(x) for x in val[row_id]["pinyin_segments"])
        p_surface = tuple(str(x) for x in surface[row_id]["pinyin_segments"])
        p_generic = tuple(str(x) for x in generic[row_id]["pinyin_segments"])
        if not (p_val == p_surface == p_generic):
            raise RuntimeError(f"Pinyin mismatch at {row_id}")

    return Inputs(
        fit_rows=fit_rows,
        val_rows=val_rows,
        surface_rows=surface_rows,
        generic_rows=generic_rows,
        fit=fit,
        val=val,
        surface=surface,
        generic=generic,
    )


# ---------------------------------------------------------------------------
# Legal rolling H5000 history with author-level age
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
    age: int  # 0 = immediately previous same-author row


class CausalHistoryIndex:
    """Strictly-prior same-user H5000 selected before Pinyin filtering."""

    def __init__(self, records: Sequence[HistoryRecord]) -> None:
        grouped: dict[str, list[HistoryRecord]] = defaultdict(list)
        for record in records:
            grouped[record.author].append(record)

        self.positions: dict[str, tuple[int, ...]] = {}
        self.pinyin_records: dict[tuple[str, tuple[str, ...]], tuple[HistoryRecord, ...]] = {}
        self.pinyin_ordinals: dict[tuple[str, tuple[str, ...]], tuple[int, ...]] = {}

        for author, values in grouped.items():
            ordered = tuple(sorted(values, key=lambda r: (r.position, r.row_id)))
            self.positions[author] = tuple(record.position for record in ordered)
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
    ) -> tuple[VisibleHistory, ...]:
        positions = self.positions.get(author, ())
        stop = bisect.bisect_left(positions, position)
        start = max(0, stop - HISTORY_BUDGET)

        key = (author, pinyin)
        ordinals = self.pinyin_ordinals.get(key, ())
        matching = self.pinyin_records.get(key, ())
        left = bisect.bisect_left(ordinals, start)
        right = bisect.bisect_left(ordinals, stop)

        output: list[VisibleHistory] = []
        for ordinal, record in zip(ordinals[left:right], matching[left:right]):
            output.append(VisibleHistory(record=record, age=(stop - 1 - ordinal)))
        return tuple(output)


def to_history_record(row: Mapping[str, Any]) -> HistoryRecord:
    target = row.get("target", row.get("gold"))
    if target is None:
        raise RuntimeError(f"History row has no target/gold: {row.get('row_id')}")
    return HistoryRecord(
        row_id=str(row["row_id"]),
        author=str(row["author"]),
        position=int(row["chronological_position"]),
        pinyin=tuple(str(x) for x in row["pinyin_segments"]),
        target=str(target),
        context=str(row.get("context", "")),
    )


def build_history_index(inputs: Inputs) -> CausalHistoryIndex:
    records = [to_history_record(row) for row in inputs.fit_rows]
    records.extend(to_history_record(row) for row in inputs.val_rows)
    return CausalHistoryIndex(records)


def query_history(
    inputs: Inputs,
    history_index: CausalHistoryIndex,
    row_id: str,
) -> tuple[VisibleHistory, ...]:
    row = inputs.val[row_id]
    return history_index.visible(
        author=str(row["author"]),
        position=int(row["chronological_position"]),
        pinyin=tuple(str(x) for x in row["pinyin_segments"]),
    )


# ---------------------------------------------------------------------------
# N-gram scorers
# ---------------------------------------------------------------------------


def suffix_matches(query_context: str, history_context: str, n: int) -> bool:
    if n <= 0:
        return True
    if len(query_context) < n or len(history_context) < n:
        return False
    return query_context[-n:] == history_context[-n:]


def choose_effective_n(
    *,
    max_n: int,
    query_context: str,
    candidates: set[str],
    visible: Sequence[VisibleHistory],
) -> int:
    """Largest suffix order with at least one candidate-target history match."""
    for n in range(max_n, 0, -1):
        for item in visible:
            if item.record.target not in candidates:
                continue
            if suffix_matches(query_context, item.record.context, n):
                return n
    return 0


def supports_for_config(
    *,
    candidates: Sequence[str],
    query_context: str,
    visible: Sequence[VisibleHistory],
    max_n: int,
    tau: float | None,
) -> tuple[list[float], int, int]:
    candidate_set = set(candidates)
    effective_n = choose_effective_n(
        max_n=max_n,
        query_context=query_context,
        candidates=candidate_set,
        visible=visible,
    )
    raw = {candidate: 0.0 for candidate in candidates}
    matched_rows = 0
    for item in visible:
        target = item.record.target
        if target not in raw:
            continue
        if not suffix_matches(query_context, item.record.context, effective_n):
            continue
        matched_rows += 1
        if tau is None:
            weight = 1.0
        else:
            weight = math.exp(-float(item.age) / float(tau))
        raw[target] += weight
    return normalize_nonnegative([raw[c] for c in candidates]), effective_n, matched_rows


def frequency_support(candidates: Sequence[str], visible: Sequence[VisibleHistory]) -> list[float]:
    counts = Counter(item.record.target for item in visible)
    return normalize_nonnegative([math.log1p(counts.get(candidate, 0)) for candidate in candidates])


# ---------------------------------------------------------------------------
# Audit / scoring
# ---------------------------------------------------------------------------


def parse_n_values(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(sorted(set(int(v) for v in values)))
    if not result or any(v <= 0 for v in result):
        raise ValueError("--n-values must contain positive integers")
    return result


def parse_tau_values(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(sorted(set(float(v) for v in values)))
    if not result or any(v <= 0 for v in result):
        raise ValueError("--tau-values must contain positive values")
    return result


def run_audit(args: argparse.Namespace) -> None:
    inputs = load_and_verify(args)
    n_values = parse_n_values(args.n_values)
    tau_values = parse_tau_values(args.tau_values)
    count_by_k = Counter(len(candidate_texts_top5(row)) for row in inputs.surface_rows)
    eligible_ids = [
        row_id for row_id in sorted(inputs.surface)
        if candidate_texts_top5(inputs.surface[row_id])
    ]
    q_lengths = [len(str(inputs.generic[row_id]["model_used_context"])) for row_id in eligible_ids]

    print("\n=== INITIAL N-GRAM + RECENCY CANDIDATE SCORING AUDIT ===")
    print(f"Train-Fit rows: {len(inputs.fit_rows)}")
    print(f"Train-Val rows: {len(inputs.val_rows)}")
    print(f"eligible rows: {len(eligible_ids)}")
    print(f"K5 candidate pairs: {sum(len(candidate_texts_top5(inputs.surface[r])) for r in eligible_ids)}")
    print(f"candidate count distribution: {dict(sorted(count_by_k.items()))}")
    print(f"N grid: {n_values}")
    print(f"tau grid: {tau_values}")
    print(f"query context chars: min={min(q_lengths)} median={percentile(q_lengths, .5)} max={max(q_lengths)}")
    print("query lexical context source: Generic model_used_context")
    print("history lexical context source: Train-Fit / earlier Train-Val context")
    print("NGram semantics: longest exact suffix backoff within configured max N")
    print("Recency: exp(-author_level_age/tau), age=0 is immediately prior same-author row")
    print("H5000-before-Pinyin rolling causal history: true")
    print("Gold used for scoring/candidate selection: false")
    print("Dev3000 used: false")
    print("Test used: false")


def load_completed_rows(
    output_path: Path,
    inputs: Inputs,
    n_values: Sequence[int],
    tau_values: Sequence[float],
) -> dict[str, dict[str, Any]]:
    if not output_path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    expected_ng = {str(n) for n in n_values}
    expected_nr = {f"{n}|{tau:g}" for n in n_values for tau in tau_values}
    with output_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row["row_id"])
            if row_id in completed:
                raise RuntimeError(f"Duplicate output row {row_id} at line {line_number}")
            if row_id not in inputs.surface:
                raise RuntimeError(f"Stale output row: {row_id}")
            expected_candidates = candidate_texts_top5(inputs.surface[row_id])
            if tuple(str(x) for x in row.get("candidates", [])) != expected_candidates:
                raise RuntimeError(f"Candidate identity/order mismatch for {row_id}")
            if set(row.get("ngram", {})) != expected_ng:
                raise RuntimeError(f"N grid mismatch in cached row {row_id}")
            if set(row.get("ngram_recency", {})) != expected_nr:
                raise RuntimeError(f"N/tau grid mismatch in cached row {row_id}")
            completed[row_id] = row
    return completed


def run_score(args: argparse.Namespace) -> None:
    inputs = load_and_verify(args)
    n_values = parse_n_values(args.n_values)
    tau_values = parse_tau_values(args.tau_values)
    output_dir = args.output_root / "ngram_recency"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ngram_recency_scores.jsonl"
    progress_path = output_dir / "progress.json"
    summary_path = output_dir / "score_summary.json"

    eligible = [
        row_id for row_id in sorted(inputs.surface)
        if candidate_texts_top5(inputs.surface[row_id])
    ]
    completed = load_completed_rows(output_path, inputs, n_values, tau_values)
    pending = [row_id for row_id in eligible if row_id not in completed]
    history_index = build_history_index(inputs)

    print("\n=== N-GRAM + RECENCY SCORE ===")
    print(f"eligible rows: {len(eligible)}")
    print(f"already completed: {len(completed)}")
    print(f"pending: {len(pending)}")
    print(f"N grid: {n_values}")
    print(f"tau grid: {tau_values}")
    print("device: CPU")

    latencies_ms: list[float] = []
    processed = 0
    started = time.perf_counter()
    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as sink:
        for row_id in pending:
            candidates = candidate_texts_top5(inputs.surface[row_id])
            query_context = str(inputs.generic[row_id]["model_used_context"])
            call_started = time.perf_counter()
            visible = query_history(inputs, history_index, row_id)
            f_support = frequency_support(candidates, visible)

            ng: dict[str, Any] = {}
            nr: dict[str, Any] = {}
            for n in n_values:
                support, effective_n, matched = supports_for_config(
                    candidates=candidates,
                    query_context=query_context,
                    visible=visible,
                    max_n=n,
                    tau=None,
                )
                ng[str(n)] = {
                    "effective_n": effective_n,
                    "matched_history_rows": matched,
                    "support": support,
                }
                for tau in tau_values:
                    support_r, effective_n_r, matched_r = supports_for_config(
                        candidates=candidates,
                        query_context=query_context,
                        visible=visible,
                        max_n=n,
                        tau=tau,
                    )
                    if effective_n_r != effective_n or matched_r != matched:
                        raise AssertionError("Recency changed lexical backoff/match set")
                    nr[f"{n}|{tau:g}"] = {
                        "effective_n": effective_n_r,
                        "matched_history_rows": matched_r,
                        "support": support_r,
                    }

            latency_ms = (time.perf_counter() - call_started) * 1000.0
            latencies_ms.append(latency_ms)
            row_out = {
                "schema_version": 1,
                "row_id": row_id,
                "author": str(inputs.val[row_id]["author"]),
                "candidates": list(candidates),
                "visible_same_pinyin_history": len(visible),
                "frequency_support": f_support,
                "ngram": ng,
                "ngram_recency": nr,
                "grid_score_ms": latency_ms,
                "gold_used_for_scoring": False,
            }
            sink.write(json.dumps(row_out, ensure_ascii=False, separators=(",", ":")) + "\n")
            processed += 1

            if processed % args.progress_every == 0 or processed == len(pending):
                elapsed = time.perf_counter() - started
                rate = processed / elapsed if elapsed > 0 else 0.0
                mean_ms = statistics.fmean(latencies_ms) if latencies_ms else 0.0
                print(
                    f"NGRAM: {processed}/{len(pending)} new rows; "
                    f"overall={len(completed)+processed}/{len(eligible)}; "
                    f"rate={rate:.1f} rows/s; grid_mean={mean_ms:.3f} ms/row",
                    flush=True,
                )
                write_json(
                    progress_path,
                    {
                        "status": "running",
                        "eligible_rows": len(eligible),
                        "completed_rows": len(completed) + processed,
                        "remaining_rows": len(eligible) - len(completed) - processed,
                        "n_values": list(n_values),
                        "tau_values": list(tau_values),
                        "gold_used_for_scoring": False,
                        "dev3000_used": False,
                        "test_used": False,
                    },
                )

    final_rows = load_completed_rows(output_path, inputs, n_values, tau_values)
    if len(final_rows) != EXPECTED_ELIGIBLE_ROWS:
        print(f"Partial/resumable: {len(final_rows)}/{EXPECTED_ELIGIBLE_ROWS} complete")
        return

    all_latency = [float(row["grid_score_ms"]) for row in final_rows.values()]
    effective_counts = {
        str(n): dict(sorted(Counter(int(row["ngram"][str(n)]["effective_n"]) for row in final_rows.values()).items()))
        for n in n_values
    }
    summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_candidate_scoring_ngram_recency_v1",
        "eligible_rows": len(final_rows),
        "n_values": list(n_values),
        "tau_values": list(tau_values),
        "history_budget": HISTORY_BUDGET,
        "grid_scoring_latency_ms": latency_summary(all_latency),
        "effective_n_distribution": effective_counts,
        "output_sha256": sha256_file(output_path),
        "gold_used_for_scoring": False,
        "dev3000_used": False,
        "test_used": False,
    }
    write_json(summary_path, summary)
    write_json(progress_path, {**summary, "completed_rows": len(final_rows), "remaining_rows": 0})
    print("\n=== N-GRAM SCORE COMPLETE ===")
    print(json.dumps(summary["grid_scoring_latency_ms"], indent=2))


# ---------------------------------------------------------------------------
# Evaluation / selection
# ---------------------------------------------------------------------------


def subset_metrics(records: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    chosen = [record for record in records if record["include"]]
    by_author: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in chosen:
        by_author[str(record["author"])].append(record)
    ranks = [int(record["rank"]) for record in chosen if record["rank"] is not None]
    author_top1 = {
        author: sum(int(row["rank"] == 1) for row in rows) / len(rows)
        for author, rows in by_author.items()
        if rows
    }
    return {
        "method": method,
        "n": len(chosen),
        "authors": len(author_top1),
        "macro_author_top1": statistics.fmean(author_top1.values()) if author_top1 else None,
        "micro_top1": sum(rank == 1 for rank in ranks) / len(chosen) if chosen else None,
        "top3": sum(rank <= 3 for rank in ranks) / len(chosen) if chosen else None,
        "mrr_at_5": sum((1.0 / rank) if rank <= 5 else 0.0 for rank in ranks) / len(chosen) if chosen else None,
        "mean_gold_rank": statistics.fmean(ranks) if ranks else None,
        "per_author_top1": author_top1,
    }


def disagreement_counts(rows_a: Sequence[Mapping[str, Any]], rows_b: Sequence[Mapping[str, Any]], label_a: str, label_b: str) -> dict[str, int]:
    a_by = {str(r["row_id"]): r for r in rows_a if int(r["k"]) >= 2}
    b_by = {str(r["row_id"]): r for r in rows_b if int(r["k"]) >= 2}
    counts: Counter[str] = Counter()
    for row_id in sorted(set(a_by) & set(b_by)):
        a = a_by[row_id]
        b = b_by[row_id]
        if a["winner"] == b["winner"]:
            counts["same_winner"] += 1
        elif a["winner"] == a["gold"]:
            counts[f"disagree_{label_a}_correct"] += 1
        elif b["winner"] == b["gold"]:
            counts[f"disagree_{label_b}_correct"] += 1
        else:
            counts["disagree_neither_correct"] += 1
    return dict(counts)


def replay_selected_latency(
    *,
    inputs: Inputs,
    history_index: CausalHistoryIndex,
    n: int,
    tau: float | None,
) -> dict[str, Any]:
    latencies: list[float] = []
    for row_id in sorted(inputs.surface):
        candidates = candidate_texts_top5(inputs.surface[row_id])
        if not candidates:
            continue
        query_context = str(inputs.generic[row_id]["model_used_context"])
        started = time.perf_counter()
        visible = query_history(inputs, history_index, row_id)
        supports_for_config(
            candidates=candidates,
            query_context=query_context,
            visible=visible,
            max_n=n,
            tau=tau,
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
    return latency_summary(latencies)


def run_evaluate(args: argparse.Namespace) -> None:
    inputs = load_and_verify(args)
    n_values = parse_n_values(args.n_values)
    tau_values = parse_tau_values(args.tau_values)
    output_dir = args.output_root / "ngram_recency"
    score_path = output_dir / "ngram_recency_scores.jsonl"
    if not score_path.is_file():
        raise RuntimeError("Run --phase score first")
    scored = load_completed_rows(score_path, inputs, n_values, tau_values)
    if len(scored) != EXPECTED_ELIGIBLE_ROWS:
        raise RuntimeError(f"Incomplete score cache: {len(scored)}/{EXPECTED_ELIGIBLE_ROWS}")

    rows_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    history_index = build_history_index(inputs)
    for row_id in sorted(scored):
        score_row = scored[row_id]
        val_row = inputs.val[row_id]
        surface_row = inputs.surface[row_id]
        candidates = tuple(str(x) for x in score_row["candidates"])
        gold = str(val_row.get("target", val_row.get("gold")))
        author = str(val_row["author"])
        k = len(candidates)
        generic_set = set(generic_candidate_texts(surface_row))
        pure_recovered = gold in candidates and gold not in generic_set
        visible = query_history(inputs, history_index, row_id)
        counts = Counter(item.record.target for item in visible)
        distinct_targets = len(counts)
        ambiguous = distinct_targets >= 2
        ordered_counts = counts.most_common()
        unique_winner = bool(ordered_counts) and (
            len(ordered_counts) == 1 or ordered_counts[0][1] > ordered_counts[1][1]
        )
        frequency_winner = ordered_counts[0][0] if unique_winner else None
        formal_conflict = bool(
            ambiguous and unique_winner and frequency_winner is not None and gold != frequency_winner
        )
        ambiguous_non_conflict = ambiguous and not formal_conflict

        common = {
            "row_id": row_id,
            "author": author,
            "gold": gold,
            "k": k,
            "gold_in_personal": gold in candidates,
            "pure_recovered": pure_recovered,
            "formal_conflict": formal_conflict,
            "ambiguous_non_conflict": ambiguous_non_conflict,
        }

        method_supports: dict[str, tuple[list[float], int | None]] = {
            "F": ([float(x) for x in score_row["frequency_support"]], None)
        }
        for n in n_values:
            item = score_row["ngram"][str(n)]
            method_supports[f"NGram@{n}"] = ([float(x) for x in item["support"]], int(item["effective_n"]))
            for tau in tau_values:
                item_r = score_row["ngram_recency"][f"{n}|{tau:g}"]
                method_supports[f"NGramRecency@{n},tau={tau:g}"] = (
                    [float(x) for x in item_r["support"]],
                    int(item_r["effective_n"]),
                )

        for method, (support, effective_n) in method_supports.items():
            ranking = rank_candidates(candidates, support)
            rows_by_method[method].append(
                {
                    **common,
                    "include": gold in candidates,
                    "rank": gold_rank(ranking, gold),
                    "winner": ranking[0],
                    "effective_n": effective_n,
                    "context_hit": bool(effective_n is not None and effective_n > 0),
                }
            )

    def metrics_for_filter(method: str, predicate: Any) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for record in rows_by_method[method]:
            copy = dict(record)
            copy["include"] = bool(record["gold_in_personal"] and predicate(record))
            rows.append(copy)
        return subset_metrics(rows, method)

    filters = {
        "gold_in_personal_all": lambda r: True,
        "gold_in_personal_k2plus": lambda r: int(r["k"]) >= 2,
        "pure_recovered_gold": lambda r: bool(r["pure_recovered"]),
        "formal_conflict_gold_in_personal": lambda r: bool(r["formal_conflict"]),
        "ambiguous_non_conflict_gold_in_personal": lambda r: bool(r["ambiguous_non_conflict"]),
    }
    sections = {
        section_name: {
            method: metrics_for_filter(method, predicate)
            for method in rows_by_method
        }
        for section_name, predicate in filters.items()
    }

    selection_section = sections["gold_in_personal_k2plus"]

    n_options = []
    for n in n_values:
        method = f"NGram@{n}"
        metrics = selection_section[method]
        n_options.append((float(metrics["macro_author_top1"] or 0.0), -n, n, method, metrics))
    best_ng = max(n_options, key=lambda item: (item[0], item[1]))

    nr_options = []
    for n in n_values:
        for tau in tau_values:
            method = f"NGramRecency@{n},tau={tau:g}"
            metrics = selection_section[method]
            # exact ties prefer smaller N, then larger tau (less aggressive recency)
            nr_options.append((float(metrics["macro_author_top1"] or 0.0), -n, tau, n, method, metrics))
    best_nr = max(nr_options, key=lambda item: (item[0], item[1], item[2]))

    selected_ng_method = best_ng[3]
    selected_nr_method = best_nr[4]
    selected_ng_n = int(best_ng[2])
    selected_nr_n = int(best_nr[3])
    selected_nr_tau = float(selected_nr_method.split("tau=")[1])

    selected_ng_rows = rows_by_method[selected_ng_method]
    selected_nr_rows = rows_by_method[selected_nr_method]
    f_rows = rows_by_method["F"]

    context_coverage = {}
    for method in (selected_ng_method, selected_nr_method):
        rows = rows_by_method[method]
        context_coverage[method] = {
            "eligible_rows": len(rows),
            "context_hit_rows": sum(bool(r["context_hit"]) for r in rows),
            "context_hit_rate": sum(bool(r["context_hit"]) for r in rows) / len(rows),
            "effective_n_distribution": dict(sorted(Counter(int(r["effective_n"]) for r in rows).items())),
        }

    selected_latency = {
        selected_ng_method: replay_selected_latency(
            inputs=inputs, history_index=history_index, n=selected_ng_n, tau=None
        ),
        selected_nr_method: replay_selected_latency(
            inputs=inputs, history_index=history_index, n=selected_nr_n, tau=selected_nr_tau
        ),
    }

    result = {
        "schema_version": 1,
        "experiment": "initial_candidate_scoring_ngram_recency_v1",
        "status": "complete",
        "partition": "standardized_train_val",
        "candidate_pool": "frozen personal K<=5 only for scorer-quality evaluation",
        "methods": {
            "F": "normalized log(1+count)",
            "NGram": "longest exact suffix backoff up to N; count matched candidate-target history",
            "NGramRecency": "same suffix backoff; matched history weighted exp(-author_level_age/tau)",
            "n_grid": list(n_values),
            "tau_grid": list(tau_values),
            "age_definition": "0=immediately prior same-author interaction within H5000-before-Pinyin history",
        },
        "selection": {
            "metric": "Macro-author Top1 on Train-Val Gold-in-personal K>=2",
            "best_ngram": {
                "method": selected_ng_method,
                "n": selected_ng_n,
                "metrics": best_ng[4],
            },
            "best_ngram_recency": {
                "method": selected_nr_method,
                "n": selected_nr_n,
                "tau": selected_nr_tau,
                "metrics": best_nr[5],
            },
            "tie_break": "NGram: smaller N; NGramRecency: smaller N then larger tau",
        },
        "sections": sections,
        "context_coverage": context_coverage,
        "disagreements_k2plus": {
            f"{selected_ng_method}_vs_F": disagreement_counts(selected_ng_rows, f_rows, "ngram", "frequency"),
            f"{selected_nr_method}_vs_F": disagreement_counts(selected_nr_rows, f_rows, "ngram_recency", "frequency"),
            f"{selected_ng_method}_vs_{selected_nr_method}": disagreement_counts(
                selected_ng_rows, selected_nr_rows, "ngram", "ngram_recency"
            ),
        },
        "selected_online_latency_ms": selected_latency,
        "score_cache_sha256": sha256_file(score_path),
        "gold_used_for_scoring": False,
        "gold_used_for_selection_and_evaluation": True,
        "dev3000_used": False,
        "test_used": False,
    }
    comparison_path = output_dir / "candidate_scoring_comparison.json"
    write_json(comparison_path, result)

    print("\n=== N-GRAM + RECENCY EVALUATION COMPLETE ===")
    print(f"best NGram: {selected_ng_method}")
    print(json.dumps(best_ng[4], indent=2, ensure_ascii=False))
    print(f"best NGramRecency: {selected_nr_method}")
    print(json.dumps(best_nr[5], indent=2, ensure_ascii=False))
    print("selected online latency:")
    print(json.dumps(selected_latency, indent=2, ensure_ascii=False))
    print(f"saved: {comparison_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_all(args: argparse.Namespace) -> None:
    run_audit(args)
    run_score(args)
    run_evaluate(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "score", "evaluate", "all"), default="all")
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--candidate-surface", type=Path, required=True)
    parser.add_argument("--generic-predictions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--n-values", nargs="+", type=int, default=list(DEFAULT_N_VALUES))
    parser.add_argument("--tau-values", nargs="+", type=float, default=list(DEFAULT_TAU_VALUES))
    parser.add_argument("--progress-every", type=int, default=500)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    if args.phase == "audit":
        run_audit(args)
    elif args.phase == "score":
        run_score(args)
    elif args.phase == "evaluate":
        run_evaluate(args)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
