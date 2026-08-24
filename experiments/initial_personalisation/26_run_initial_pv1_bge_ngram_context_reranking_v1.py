from __future__ import annotations

"""PV1 final-list context reranking with semantic BGE and lexical NGramRecency.

Purpose
-------
This runner isolates final-list context reranking from recovery. It starts from
frozen PV1 Top-10 candidates and their frozen ``final_score`` values, then adds
candidate-specific context evidence without changing the candidate set.

Methods
-------
PV1 baseline
    S(c) = S_PV1(c)

PV1 + BGE64
    S(c) = S_PV1(c) + lambda_B * P_BGE(c)

    For each PV1 candidate c, collect legal same-Pinyin historical rows whose
    target is c. Embed the last 64 context characters with the existing BGE
    context embedder, compute cosine similarity to the current query context,
    keep the candidate's Top-5 similarities, clamp negative cosine to zero,
    sum them, and normalize non-negative mass across the current PV1 Top-10.

PV1 + HardBackoffNGramRecency
    S(c) = S_PV1(c) + lambda_N * P_NG(c)

    Reuse the previously selected lexical scorer at maxN=2, tau=2048. Choose
    the longest exact suffix order <= maxN that has candidate-target history;
    matched rows contribute exp(-age/tau), then candidate mass is normalized.

PV1 + BGE64 + HardBackoffNGramRecency
    S(c) = S_PV1(c) + lambda_B * P_BGE(c) + lambda_N * P_NG(c)

Protocol
--------
- Same author only.
- Strictly prior history only.
- H5000 BEFORE exact-Pinyin filtering.
- Earlier Train-Val rows may be history for later Train-Val rows.
- Current/future Gold is never used in features or scoring.
- Gold is used only for Train-Val evaluation / development selection.
- Dev3000 is not read.
- Test is not read.
- Candidate set must remain identical to frozen PV1 for every method.
- lambda_B=lambda_N=0 must reproduce PV1 exactly.

BGE cache / resume
------------------
Historical BGE vectors are stored in this experiment's own SQLite cache.
The runner is resumable: an interrupted cache/scoring pass can continue in the
same output directory if the manifest matches. A previous BGE cache may be
supplied with --seed-bge-cache; it is read only and copied into the new cache,
so historical artifacts are never modified.

Outputs
-------
<output-root>/
    run_manifest.json
    bge_history_embedding_cache.sqlite3
    bge_scores.jsonl
    bge_scoring_summary.json
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
"""

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import sqlite3
import statistics
import sys
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

EXPECTED_PV1_HEADLINE = {
    "macro_author_top1": 0.401872,
    "micro_top1": 0.426749,
    "top3": 0.598907,
    "top5": 0.663093,
    "mrr_at_10": 0.524450,
    "missing10": 0.291144,
}
EXPECTED_HEADLINE_TOL = 1e-6

BGE_CONTEXT_CHARS = 64
BGE_TOP_N = 5
DEFAULT_HARD_MAX_N = 2
DEFAULT_HARD_TAU = 2048.0
DEFAULT_BGE_LAMBDAS = (0.0, 0.25, 0.50, 1.0, 2.0, 4.0)
DEFAULT_NGRAM_LAMBDAS = (0.0, 0.25, 0.50, 1.0, 2.0, 4.0)


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


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as destination:
        destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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
        writer.writerows(rows)


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows = read_jsonl(path)
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        rid = str(row["row_id"])
        if rid in output:
            raise RuntimeError(f"Duplicate row_id in resumable cache {path}: {rid}")
        output[rid] = row
    return output


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
    return str(row.get("context", ""))


def bge_context_of(row: Mapping[str, Any]) -> str:
    return context_of(row)[-BGE_CONTEXT_CHARS:]


def candidate_text(item: Mapping[str, Any]) -> str:
    for key in ("candidate", "text", "target"):
        value = item.get(key)
        if value is not None:
            return str(value)
    raise RuntimeError(f"Cannot identify candidate text from: {item}")


def pv1_scored_candidates(row: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    values = row.get("pv1_candidates")
    if not isinstance(values, list) or not values:
        raise RuntimeError(f"Missing pv1_candidates for {row.get('row_id')}")
    copied_rows: list[dict[str, Any]] = []
    for index, item in enumerate(values, 1):
        if not isinstance(item, Mapping):
            raise RuntimeError(f"Unexpected PV1 candidate at {row.get('row_id')}: {item!r}")
        copied = dict(item)
        copied["candidate"] = candidate_text(copied)
        copied["rank"] = int(copied.get("rank", index))
        if copied.get("final_score") is None:
            raise RuntimeError(f"PV1 candidate missing final_score at {row.get('row_id')}")
        copied["final_score"] = float(copied["final_score"])
        copied_rows.append(copied)
    copied_rows.sort(key=lambda item: int(item["rank"]))
    texts = [str(item["candidate"]) for item in copied_rows]
    if len(texts) != len(set(texts)):
        raise RuntimeError(f"Duplicate candidate in frozen PV1 list: {row.get('row_id')}")
    if len(texts) > 10:
        raise RuntimeError(f"Frozen PV1 list >10 at {row.get('row_id')}")
    expected = [
        str(item["candidate"])
        for item in sorted(
            copied_rows,
            key=lambda item: (
                -float(item["final_score"]),
                0 if str(item.get("source", "generic")) == "generic" else 1,
                int(item.get("generic_rank") or item.get("personal_candidate_rank", 0)),
                str(item["candidate"]),
            ),
        )
    ]
    if expected != texts:
        raise RuntimeError(f"PV1 rank/final_score order mismatch at {row.get('row_id')}")
    return tuple(copied_rows)


def extract_personal_k5(row: Mapping[str, Any]) -> tuple[str, ...]:
    for key in ("personal_candidate_texts_top5", "personal_k5"):
        value = row.get(key)
        if isinstance(value, list):
            return tuple(str(item) for item in value[:5])
    value = row.get("personal_candidates_top5")
    if isinstance(value, list):
        out: list[str] = []
        for item in value[:5]:
            out.append(str(item) if isinstance(item, str) else candidate_text(item))
        return tuple(out)
    raise RuntimeError(f"Cannot identify Personal K5 at {row.get('row_id')}")


def rank_of(ranking: Sequence[str], gold: str) -> int | None:
    for index, candidate in enumerate(ranking, 1):
        if candidate == gold:
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
            self.positions[author] = tuple(r.position for r in ordered)
            by_pinyin: dict[tuple[str, ...], list[tuple[int, HistoryRecord]]] = defaultdict(list)
            for ordinal, record in enumerate(ordered):
                by_pinyin[record.pinyin].append((ordinal, record))
            for pinyin, pairs in by_pinyin.items():
                key = (author, pinyin)
                self.pinyin_ordinals[key] = tuple(o for o, _ in pairs)
                self.pinyin_records[key] = tuple(r for _, r in pairs)

    def visible(self, *, author: str, position: int, pinyin: tuple[str, ...]) -> tuple[VisibleHistory, ...]:
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
# Lexical scorer: existing HardBackoffNGramRecency
# ---------------------------------------------------------------------------


def recency_weight(age: int, tau: float) -> float:
    return math.exp(-float(age) / float(tau))


def suffix_matches(a: str, b: str, n: int) -> bool:
    if n <= 0:
        return True
    return len(a) >= n and len(b) >= n and a[-n:] == b[-n:]


def normalize_nonnegative(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    clipped = [max(0.0, float(value)) for value in values]
    total = sum(clipped)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in clipped]


def hard_backoff_ngram_recency(
    *,
    candidates: Sequence[str],
    query_context: str,
    visible: Sequence[VisibleHistory],
    max_n: int,
    tau: float,
) -> tuple[list[float], dict[str, Any]]:
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
    support = normalize_nonnegative([raw[candidate] for candidate in candidates])
    return support, {
        "effective_n": effective_n,
        "matched_history_rows": matched,
        "candidate_mass": sum(raw.values()),
    }


# ---------------------------------------------------------------------------
# Semantic scorer: BGE64, same semantics as prior BGE candidate scorer
# ---------------------------------------------------------------------------


class VectorCache:
    def __init__(self, path: Path) -> None:
        import numpy as np

        self.np = np
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "context TEXT PRIMARY KEY, dim INTEGER NOT NULL, vector BLOB NOT NULL)"
        )
        self.connection.commit()

    def get(self, context: str) -> Any | None:
        row = self.connection.execute(
            "SELECT dim, vector FROM embeddings WHERE context=?", (context,)
        ).fetchone()
        if row is None:
            return None
        dim, blob = row
        vector = self.np.frombuffer(blob, dtype=self.np.float32).copy()
        if vector.size != int(dim):
            raise RuntimeError("Corrupt BGE vector cache row")
        return vector

    def put(self, context: str, vector: Any) -> None:
        value = self.np.asarray(vector, dtype=self.np.float32).reshape(-1)
        self.connection.execute(
            "INSERT OR REPLACE INTO embeddings(context, dim, vector) VALUES(?,?,?)",
            (context, int(value.size), value.tobytes()),
        )

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def normalized_vector(vector: Any) -> Any:
    import numpy as np

    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 0 else value


def bge_candidate_support(
    *,
    query_vector: Any,
    candidates: Sequence[str],
    visible: Sequence[VisibleHistory],
    history_vectors: Mapping[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    import numpy as np

    grouped: dict[str, list[Any]] = {candidate: [] for candidate in candidates}
    counts = {candidate: 0 for candidate in candidates}
    for item in visible:
        record = item.record
        if record.target not in grouped:
            continue
        counts[record.target] += 1
        key = record.context[-BGE_CONTEXT_CHARS:]
        vector = history_vectors.get(key)
        if vector is None:
            raise KeyError(
                "Missing cached BGE history vector for context hash="
                + hashlib.sha256(key.encode("utf-8")).hexdigest()
            )
        grouped[record.target].append(vector)

    raw: list[float] = []
    touched = 0
    candidates_with_history = 0
    for candidate in candidates:
        values = grouped[candidate]
        touched += len(values)
        if not values:
            raw.append(0.0)
            continue
        candidates_with_history += 1
        matrix = np.vstack(values)
        similarities = matrix @ query_vector
        if similarities.size > BGE_TOP_N:
            top = np.partition(similarities, -BGE_TOP_N)[-BGE_TOP_N:]
        else:
            top = similarities
        raw.append(float(np.maximum(top, 0.0).sum()))

    support = normalize_nonnegative(raw)
    return support, {
        "history_vectors_touched": touched,
        "candidates_with_history": candidates_with_history,
        "candidate_history_counts": [int(counts[c]) for c in candidates],
        "raw_positive_mass": float(sum(max(0.0, value) for value in raw)),
    }


def ensure_cuda_path(explicit: Path | None) -> None:
    if explicit is not None:
        candidate = explicit
        if not (candidate / "bin").is_dir():
            raise RuntimeError(f"--cuda-path has no bin directory: {candidate}")
        os.environ["CUDA_PATH"] = str(candidate)
        os.environ["PATH"] = str(candidate / "bin") + os.pathsep + os.environ.get("PATH", "")
        return
    current = os.environ.get("CUDA_PATH")
    if current and (Path(current) / "bin").is_dir():
        return
    base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if base.is_dir():
        versions = sorted(
            (path for path in base.iterdir() if path.is_dir() and (path / "bin").is_dir()),
            reverse=True,
        )
        if versions:
            os.environ["CUDA_PATH"] = str(versions[0])
            os.environ["PATH"] = str(versions[0] / "bin") + os.pathsep + os.environ.get("PATH", "")
            print(f"CUDA_PATH auto-selected: {versions[0]}", flush=True)
            return
    raise RuntimeError("No valid CUDA_PATH found. Pass --cuda-path explicitly.")


class NativeStderrSilencer:
    def __enter__(self) -> "NativeStderrSilencer":
        self._stderr_fd = sys.stderr.fileno()
        self._saved_fd = os.dup(self._stderr_fd)
        self._null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._null_fd, self._stderr_fd)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        os.dup2(self._saved_fd, self._stderr_fd)
        os.close(self._saved_fd)
        os.close(self._null_fd)


def copy_seed_cache(seed_path: Path, destination: VectorCache, needed: set[str]) -> int:
    """Copy matching vectors from a prior cache without modifying that cache."""
    if not seed_path.is_file():
        raise RuntimeError(f"Seed BGE cache not found: {seed_path}")
    source = sqlite3.connect(seed_path)
    copied = 0
    try:
        for context in needed:
            if destination.get(context) is not None:
                continue
            row = source.execute(
                "SELECT dim, vector FROM embeddings WHERE context=?", (context,)
            ).fetchone()
            if row is None:
                continue
            dim, blob = row
            destination.connection.execute(
                "INSERT OR IGNORE INTO embeddings(context, dim, vector) VALUES(?,?,?)",
                (context, int(dim), blob),
            )
            copied += 1
        destination.commit()
    finally:
        source.close()
    return copied


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def rerank_joint(
    candidates: Sequence[str],
    pv1_scores: Sequence[float],
    bge_support: Sequence[float],
    ngram_support: Sequence[float],
    *,
    lambda_bge: float,
    lambda_ngram: float,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    if not (len(candidates) == len(pv1_scores) == len(bge_support) == len(ngram_support)):
        raise RuntimeError("Candidate/score/support length mismatch")
    scores = [
        float(base) + lambda_bge * float(bge) + lambda_ngram * float(ngram)
        for base, bge, ngram in zip(pv1_scores, bge_support, ngram_support)
    ]
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i, candidates[i]))
    return (
        tuple(candidates[i] for i in order),
        tuple(float(scores[i]) for i in order),
    )


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


def author_metrics(authors: Sequence[str], ranks: Sequence[int | None]) -> dict[str, Any]:
    grouped: dict[str, list[int | None]] = defaultdict(list)
    for author, rank in zip(authors, ranks):
        grouped[str(author)].append(rank)
    per_author = {author: rank_metrics(values) for author, values in sorted(grouped.items())}
    macro_keys = ("top1", "top3", "top5", "mrr_at_10", "missing10")
    macro = {
        key: statistics.fmean(float(metrics[key]) for metrics in per_author.values())
        for key in macro_keys
    } if per_author else {key: 0.0 for key in macro_keys}
    return {"macro_author": macro, "micro": rank_metrics(ranks), "per_author": per_author}


def transition_counts(
    base_ranks: Sequence[int | None],
    new_ranks: Sequence[int | None],
    mask: Sequence[bool] | None = None,
) -> dict[str, int]:
    if mask is None:
        mask = [True] * len(base_ranks)
    counts = Counter()
    for include, before, after in zip(mask, base_ranks, new_ranks):
        if not include:
            continue
        b = before == 1
        a = after == 1
        if not b and a:
            counts["rescue"] += 1
        elif b and not a:
            counts["harm"] += 1
        elif b and a:
            counts["unchanged_correct"] += 1
        else:
            counts["unchanged_wrong"] += 1
        counts["n"] += 1
    counts["net"] = counts["rescue"] - counts["harm"]
    return dict(counts)


def rank_movement(
    base_ranks: Sequence[int | None],
    new_ranks: Sequence[int | None],
    mask: Sequence[bool],
) -> dict[str, Any]:
    gains: list[int] = []
    improved = worsened = unchanged = 0
    for include, before, after in zip(mask, base_ranks, new_ranks):
        if not include or before is None or after is None:
            continue
        gain = int(before) - int(after)
        gains.append(gain)
        if gain > 0:
            improved += 1
        elif gain < 0:
            worsened += 1
        else:
            unchanged += 1
    return {
        "n": len(gains),
        "rank_improved_n": improved,
        "rank_worsened_n": worsened,
        "rank_unchanged_n": unchanged,
        "mean_rank_gain": statistics.fmean(gains) if gains else None,
        "median_rank_gain": statistics.median(gains) if gains else None,
    }


def recovery_metrics(ranks: Sequence[int | None], mask: Sequence[bool]) -> dict[str, Any]:
    chosen = [rank for rank, include in zip(ranks, mask) if include]
    n = len(chosen)
    recovered = [int(rank) for rank in chosen if rank is not None]
    return {
        "n": n,
        "rec1": sum(rank == 1 for rank in chosen) / n if n else 0.0,
        "rec3": sum(rank is not None and rank <= 3 for rank in chosen) / n if n else 0.0,
        "rec5": sum(rank is not None and rank <= 5 for rank in chosen) / n if n else 0.0,
        "rec10": sum(rank is not None and rank <= 10 for rank in chosen) / n if n else 0.0,
        "recovery_mrr_at_10": sum(0.0 if rank is None else 1.0 / rank for rank in chosen) / n if n else 0.0,
        "mean_recovered_rank": statistics.fmean(recovered) if recovered else None,
    }


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(values: Sequence[float]) -> dict[str, Any]:
    values = [float(v) for v in values]
    mean = statistics.fmean(values) if values else None
    return {
        "n": len(values),
        "mean_ms": mean,
        "p50_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "queries_per_second_from_mean": 1000.0 / mean if mean and mean > 0 else None,
    }


def selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        float(row["macro_author_top1"]),
        float(row["mrr_at_10"]),
        float(row["top3"]),
        -float(row["lambda_bge"] + row["lambda_ngram"]),
        -float(row["lambda_bge"]),
        -float(row["lambda_ngram"]),
    )


# ---------------------------------------------------------------------------
# Manifest / input loading
# ---------------------------------------------------------------------------


def prepare_inputs(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = {
        "fit": args.fit,
        "val": args.val,
        "candidate_surface": args.candidate_surface,
        "frequency_pv1_predictions": args.frequency_pv1_predictions,
    }
    input_hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    expected = {
        "fit": EXPECTED_FIT_SHA256,
        "val": EXPECTED_VAL_SHA256,
        "candidate_surface": EXPECTED_SURFACE_SHA256,
        "frequency_pv1_predictions": EXPECTED_FREQUENCY_PV1_SHA256,
    }
    if not args.allow_unfrozen_input_hashes:
        for name, expected_hash in expected.items():
            if input_hashes[name] != expected_hash:
                raise RuntimeError(
                    f"Frozen SHA mismatch for {name}: expected {expected_hash}, got {input_hashes[name]}"
                )

    fit = read_jsonl(args.fit)
    val = read_jsonl(args.val)
    surface_rows = read_jsonl(args.candidate_surface)
    pv1_rows = read_jsonl(args.frequency_pv1_predictions)
    if len(fit) != EXPECTED_FIT_ROWS or len(val) != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"Unexpected Fit/Val rows: {len(fit)}/{len(val)}")
    if len(surface_rows) != EXPECTED_VAL_ROWS or len(pv1_rows) != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"Unexpected surface/PV1 rows: {len(surface_rows)}/{len(pv1_rows)}")

    val_by_id = {str(row["row_id"]): row for row in val}
    surface_by_id = {str(row["row_id"]): row for row in surface_rows}
    pv1_by_id = {str(row["row_id"]): row for row in pv1_rows}
    if set(val_by_id) != set(surface_by_id) or set(val_by_id) != set(pv1_by_id):
        raise RuntimeError("Train-Val / candidate-surface / PV1 row-ID surfaces differ")

    row_ids = [str(row["row_id"]) for row in val]
    authors = [str(row["author"]) for row in val]
    golds = [target_of(row) for row in val]
    baseline_rankings: list[tuple[str, ...]] = []
    baseline_score_vectors: list[tuple[float, ...]] = []
    baseline_candidate_rows: list[tuple[dict[str, Any], ...]] = []
    baseline_ranks: list[int | None] = []
    frequency_ranks: list[int | None] = []
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
            raise RuntimeError(f"PV1 K is not frozen K=1 at {rid}")
        if float(prow.get("selected_lambda_pv", -1.0)) != 4.0:
            raise RuntimeError(f"PV1 lambda is not frozen 4 at {rid}")
        if float(prow.get("selected_lambda_frequency", -1.0)) != 4.0:
            raise RuntimeError(f"Frequency lambda is not frozen 4 at {rid}")
        scored = pv1_scored_candidates(prow)
        ranking = tuple(str(item["candidate"]) for item in scored)
        scores = tuple(float(item["final_score"]) for item in scored)
        computed = rank_of(ranking, gold)
        stored = prow.get("pv1_rank")
        stored = None if stored is None else int(stored)
        if computed != stored:
            raise RuntimeError(f"PV1 rank mismatch at {rid}: {computed} != {stored}")
        baseline_rankings.append(ranking)
        baseline_score_vectors.append(scores)
        baseline_candidate_rows.append(scored)
        baseline_ranks.append(stored)
        f_rank = prow.get("frequency_rank")
        frequency_ranks.append(None if f_rank is None else int(f_rank))
        conflict_mask.append(bool(prow.get("conflict", srow.get("conflict", False))))
        ambiguous_mask.append(bool(prow.get("ambiguous", srow.get("ambiguous", False))))
        history_available_mask.append(bool(prow.get("history_available", False)))
        generic_missing = bool(prow.get("generic_missing", srow.get("generic_missing", False)))
        generic_missing_mask.append(generic_missing)
        recovery_mask.append(generic_missing and gold in set(extract_personal_k5(srow)))

    if sum(recovery_mask) != EXPECTED_RECOVERY_K5_N:
        raise RuntimeError(
            f"Expected recovery population n={EXPECTED_RECOVERY_K5_N}, got {sum(recovery_mask)}"
        )

    baseline_metrics = author_metrics(authors, baseline_ranks)
    headline = {
        "macro_author_top1": baseline_metrics["macro_author"]["top1"],
        "micro_top1": baseline_metrics["micro"]["top1"],
        "top3": baseline_metrics["micro"]["top3"],
        "top5": baseline_metrics["micro"]["top5"],
        "mrr_at_10": baseline_metrics["micro"]["mrr_at_10"],
        "missing10": baseline_metrics["micro"]["missing10"],
    }
    for key, expected_value in EXPECTED_PV1_HEADLINE.items():
        if abs(float(headline[key]) - expected_value) > EXPECTED_HEADLINE_TOL:
            raise RuntimeError(f"PV1 headline guardrail failed for {key}: {headline[key]}")

    history_records = [to_history_record(row) for row in fit]
    history_records.extend(to_history_record(row) for row in val)
    history_index = CausalHistoryIndex(history_records)

    return {
        "input_paths": input_paths,
        "input_hashes": input_hashes,
        "fit": fit,
        "val": val,
        "surface_by_id": surface_by_id,
        "pv1_by_id": pv1_by_id,
        "val_by_id": val_by_id,
        "row_ids": row_ids,
        "authors": authors,
        "golds": golds,
        "baseline_rankings": baseline_rankings,
        "baseline_score_vectors": baseline_score_vectors,
        "baseline_candidate_rows": baseline_candidate_rows,
        "baseline_ranks": baseline_ranks,
        "frequency_ranks": frequency_ranks,
        "conflict_mask": conflict_mask,
        "ambiguous_mask": ambiguous_mask,
        "history_available_mask": history_available_mask,
        "generic_missing_mask": generic_missing_mask,
        "recovery_mask": recovery_mask,
        "baseline_metrics": baseline_metrics,
        "headline": headline,
        "history_index": history_index,
    }


def ensure_manifest(args: argparse.Namespace, inputs: Mapping[str, Any]) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "run_manifest.json"
    manifest = {
        "schema_version": 1,
        "experiment": "initial_pv1_bge_ngram_context_reranking_v1",
        "input_hashes": dict(inputs["input_hashes"]),
        "bge_model": str(args.bge_model.resolve()),
        "bge_model_sha256": sha256_file(args.bge_model),
        "bge_context_chars": BGE_CONTEXT_CHARS,
        "bge_top_n": BGE_TOP_N,
        "hard_max_n": args.hard_max_n,
        "hard_tau": args.hard_tau,
        "lambda_bge_grid": list(args.lambda_bge),
        "lambda_ngram_grid": list(args.lambda_ngram),
        "gold_used_for_scoring_features": False,
        "dev3000_used": False,
        "test_used": False,
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(
                "Existing output directory has a different run_manifest.json; use a new versioned output path."
            )
    else:
        write_json(manifest_path, manifest)
    if (args.output_root / "comparison.json").is_file():
        existing = json.loads((args.output_root / "comparison.json").read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            raise RuntimeError("This output directory already contains a completed experiment.")


# ---------------------------------------------------------------------------
# BGE cache + scoring
# ---------------------------------------------------------------------------


def required_history_contexts(inputs: Mapping[str, Any], progress_every: int) -> set[str]:
    contexts: set[str] = set()
    history_index: CausalHistoryIndex = inputs["history_index"]
    for number, rid in enumerate(inputs["row_ids"], 1):
        vrow = inputs["val_by_id"][rid]
        candidates = set(inputs["baseline_rankings"][number - 1])
        visible = history_index.visible(
            author=str(vrow["author"]),
            position=int(vrow["chronological_position"]),
            pinyin=pinyin_of(vrow),
        )
        for item in visible:
            if item.record.target in candidates:
                contexts.add(item.record.context[-BGE_CONTEXT_CHARS:])
        if number % max(progress_every * 5, 5000) == 0 or number == len(inputs["row_ids"]):
            print(
                f"BGE history-context audit: {number}/{len(inputs['row_ids'])}; unique_required={len(contexts)}",
                flush=True,
            )
    return contexts


def load_history_vectors(cache: VectorCache, contexts: set[str]) -> dict[str, Any]:
    vectors: dict[str, Any] = {}
    for context in contexts:
        vector = cache.get(context)
        if vector is None:
            raise RuntimeError("BGE history cache fill incomplete")
        vectors[context] = normalized_vector(vector)
    return vectors


def run_bge_scoring(args: argparse.Namespace, inputs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    ensure_cuda_path(args.cuda_path)
    from src.personalisation.pilot_a import BGEContextEmbedder

    cache_path = args.output_root / "bge_history_embedding_cache.sqlite3"
    score_path = args.output_root / "bge_scores.jsonl"
    summary_path = args.output_root / "bge_scoring_summary.json"

    print("\n=== BGE64 PREPARE HISTORICAL CONTEXT CACHE ===", flush=True)
    contexts = required_history_contexts(inputs, args.progress_every)
    cache = VectorCache(cache_path)
    if args.seed_bge_cache is not None:
        copied = copy_seed_cache(args.seed_bge_cache, cache, contexts)
        print(f"Seed cache copied vectors: {copied}", flush=True)
    missing = [context for context in sorted(contexts) if cache.get(context) is None]
    print(f"Required unique historical contexts: {len(contexts)}", flush=True)
    print(f"Cache rows before embedding: {cache.count()}", flush=True)
    print(f"Missing historical embeddings: {len(missing)}", flush=True)

    embedder = BGEContextEmbedder(args.bge_model)
    warm = next(iter(contexts), "测试")
    _ = embedder.embed(warm)

    history_embed_ms: list[float] = []
    if missing:
        with NativeStderrSilencer():
            for number, context in enumerate(missing, 1):
                t0 = time.perf_counter()
                vector = embedder.embed(context)
                history_embed_ms.append((time.perf_counter() - t0) * 1000.0)
                cache.put(context, vector)
                if number % 100 == 0 or number == len(missing):
                    cache.commit()
                if number % args.progress_every == 0 or number == len(missing):
                    mean = statistics.fmean(history_embed_ms)
                    print(
                        f"BGE history embed: {number}/{len(missing)}; mean={mean:.3f} ms/context",
                        flush=True,
                    )

    history_vectors = load_history_vectors(cache, contexts)
    cache.close()

    completed = load_jsonl_by_id(score_path)
    pending = [rid for rid in inputs["row_ids"] if rid not in completed]
    print("\n=== BGE64 PV1 TOP10 ONLINE SCORING ===", flush=True)
    print(f"completed/resumable rows: {len(completed)}", flush=True)
    print(f"pending rows: {len(pending)}", flush=True)

    query_embed_ms: list[float] = [float(row["query_embed_ms"]) for row in completed.values()]
    cosine_ms: list[float] = [float(row["lookup_cosine_ms"]) for row in completed.values()]
    online_ms: list[float] = [float(row["online_total_ms"]) for row in completed.values()]
    run_started = time.perf_counter()

    with score_path.open("a", encoding="utf-8", newline="\n") as destination:
        with NativeStderrSilencer():
            for number, rid in enumerate(pending, 1):
                # O(1) index map is prepared globally to avoid quadratic lookup.
                vrow = inputs["val_by_id"][rid]
                row_index = row_index_by_id[rid]
                candidates = inputs["baseline_rankings"][row_index]
                visible = inputs["history_index"].visible(
                    author=str(vrow["author"]),
                    position=int(vrow["chronological_position"]),
                    pinyin=pinyin_of(vrow),
                )
                qcontext = bge_context_of(vrow)
                total_t0 = time.perf_counter()
                embed_t0 = time.perf_counter()
                qvector = normalized_vector(embedder.embed(qcontext))
                embed_elapsed = (time.perf_counter() - embed_t0) * 1000.0
                cosine_t0 = time.perf_counter()
                support, diag = bge_candidate_support(
                    query_vector=qvector,
                    candidates=candidates,
                    visible=visible,
                    history_vectors=history_vectors,
                )
                cosine_elapsed = (time.perf_counter() - cosine_t0) * 1000.0
                total_elapsed = (time.perf_counter() - total_t0) * 1000.0
                result = {
                    "schema_version": 1,
                    "row_id": rid,
                    "author": str(vrow["author"]),
                    "context_chars": BGE_CONTEXT_CHARS,
                    "actual_context_chars": len(qcontext),
                    "candidates": list(candidates),
                    "support": [float(value) for value in support],
                    "visible_same_pinyin_history_count": len(visible),
                    "query_embed_ms": embed_elapsed,
                    "lookup_cosine_ms": cosine_elapsed,
                    "online_total_ms": total_elapsed,
                    **diag,
                    "gold_used_for_scoring": False,
                    "dev3000_used": False,
                    "test_used": False,
                }
                destination.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                query_embed_ms.append(embed_elapsed)
                cosine_ms.append(cosine_elapsed)
                online_ms.append(total_elapsed)
                if number % args.progress_every == 0 or number == len(pending):
                    destination.flush()
                    wall = time.perf_counter() - run_started
                    print(
                        f"BGE score: {number}/{len(pending)} pending; "
                        f"rate={number / wall:.2f} rows/s; "
                        f"embed={statistics.fmean(query_embed_ms[-max(1, min(len(query_embed_ms), 1000)):]):.3f} ms; "
                        f"online={statistics.fmean(online_ms[-max(1, min(len(online_ms), 1000)):]):.3f} ms",
                        flush=True,
                    )

    final = load_jsonl_by_id(score_path)
    if len(final) != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"Incomplete BGE score cache: {len(final)}/{EXPECTED_VAL_ROWS}")
    for rid in inputs["row_ids"]:
        row = final[rid]
        expected_candidates = list(inputs["baseline_rankings"][row_index_by_id[rid]])
        if row.get("candidates") != expected_candidates:
            raise RuntimeError(f"BGE cached candidate list mismatch at {rid}")
        if len(row.get("support", [])) != len(expected_candidates):
            raise RuntimeError(f"BGE cached support length mismatch at {rid}")

    summary = {
        "schema_version": 1,
        "status": "complete",
        "rows": len(final),
        "required_unique_historical_contexts": len(contexts),
        "history_cache_rows": len(history_vectors),
        "new_history_embedding_latency_ms_this_run": latency_summary(history_embed_ms),
        "online_query_embedding_latency_ms": latency_summary([float(row["query_embed_ms"]) for row in final.values()]),
        "online_lookup_cosine_latency_ms": latency_summary([float(row["lookup_cosine_ms"]) for row in final.values()]),
        "online_total_latency_ms": latency_summary([float(row["online_total_ms"]) for row in final.values()]),
        "score_cache_sha256": sha256_file(score_path),
        "note": "Latency is valid only when the machine is not under competing experimental load.",
        "gold_used_for_scoring": False,
        "dev3000_used": False,
        "test_used": False,
    }
    write_json(summary_path, summary)
    return final


# Global row index map is set once after input loading. It avoids copying a large
# mapping through every hot-loop call while keeping the scorer implementation simple.
row_index_by_id: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(args: argparse.Namespace, inputs: Mapping[str, Any], bge_rows: Mapping[str, Mapping[str, Any]]) -> None:
    started = time.perf_counter()
    row_ids = inputs["row_ids"]
    authors = inputs["authors"]
    golds = inputs["golds"]
    baseline_rankings = inputs["baseline_rankings"]
    baseline_scores = inputs["baseline_score_vectors"]
    baseline_ranks = inputs["baseline_ranks"]
    frequency_ranks = inputs["frequency_ranks"]
    recovery_mask = inputs["recovery_mask"]
    conflict_mask = inputs["conflict_mask"]
    ambiguous_mask = inputs["ambiguous_mask"]
    history_available_mask = inputs["history_available_mask"]
    generic_missing_mask = inputs["generic_missing_mask"]

    rank_by_pair: dict[tuple[float, float], list[int | None]] = {
        (lb, ln): [] for lb in args.lambda_bge for ln in args.lambda_ngram
    }
    ngram_supports: list[list[float]] = []
    bge_supports: list[list[float]] = []
    ngram_effective_n = Counter()
    ngram_matched_rows: list[int] = []
    visible_history_rows: list[int] = []
    ngram_latency_ms: list[float] = []

    for index, rid in enumerate(row_ids):
        vrow = inputs["val_by_id"][rid]
        candidates = baseline_rankings[index]
        visible = inputs["history_index"].visible(
            author=str(vrow["author"]),
            position=int(vrow["chronological_position"]),
            pinyin=pinyin_of(vrow),
        )
        visible_history_rows.append(len(visible))
        t0 = time.perf_counter()
        ng_support, ng_diag = hard_backoff_ngram_recency(
            candidates=candidates,
            query_context=context_of(vrow),
            visible=visible,
            max_n=args.hard_max_n,
            tau=args.hard_tau,
        )
        ngram_latency_ms.append((time.perf_counter() - t0) * 1000.0)
        ngram_effective_n[int(ng_diag["effective_n"])] += 1
        ngram_matched_rows.append(int(ng_diag["matched_history_rows"]))
        bg_support = [float(value) for value in bge_rows[rid]["support"]]
        if len(bg_support) != len(candidates):
            raise RuntimeError(f"BGE support length mismatch at {rid}")
        ngram_supports.append(ng_support)
        bge_supports.append(bg_support)

        for lb in args.lambda_bge:
            for ln in args.lambda_ngram:
                ranking, _ = rerank_joint(
                    candidates,
                    baseline_scores[index],
                    bg_support,
                    ng_support,
                    lambda_bge=lb,
                    lambda_ngram=ln,
                )
                if set(ranking) != set(candidates) or len(ranking) != len(candidates):
                    raise RuntimeError(f"Candidate-set invariant failed at {rid}")
                rank = rank_of(ranking, golds[index])
                if (rank is None) != (baseline_ranks[index] is None):
                    raise RuntimeError(f"Missing invariant failed at {rid}")
                rank_by_pair[(lb, ln)].append(rank)

        if (index + 1) % args.progress_every == 0 or index + 1 == len(row_ids):
            print(f"NGram + joint grid evaluation: {index + 1}/{len(row_ids)}", flush=True)

    if rank_by_pair[(0.0, 0.0)] != baseline_ranks:
        raise RuntimeError("lambda_B=lambda_N=0 does not exactly reproduce PV1")

    opportunity_mask = [rank is not None for rank in baseline_ranks]
    pv1_wrong_opportunity_mask = [rank is not None and rank != 1 for rank in baseline_ranks]
    pv1_correct_mask = [rank == 1 for rank in baseline_ranks]
    non_conflict_mask = [not value for value in conflict_mask]
    pv1_rescue_vs_f_mask = [f != 1 and p == 1 for f, p in zip(frequency_ranks, baseline_ranks)]
    pv1_harm_vs_f_mask = [f == 1 and p != 1 for f, p in zip(frequency_ranks, baseline_ranks)]
    baseline_rec10 = recovery_metrics(baseline_ranks, recovery_mask)["rec10"]

    grid_rows: list[dict[str, Any]] = []
    for lb in args.lambda_bge:
        for ln in args.lambda_ngram:
            ranks = rank_by_pair[(lb, ln)]
            metrics = author_metrics(authors, ranks)
            transition = transition_counts(baseline_ranks, ranks)
            conflict_transition = transition_counts(baseline_ranks, ranks, conflict_mask)
            rec = recovery_metrics(ranks, recovery_mask)
            opportunity = rank_metrics([rank for rank, include in zip(ranks, opportunity_mask) if include])
            movement = rank_movement(baseline_ranks, ranks, opportunity_mask)
            rescue_retained = sum(include and rank == 1 for include, rank in zip(pv1_rescue_vs_f_mask, ranks))
            harm_repaired = sum(include and rank == 1 for include, rank in zip(pv1_harm_vs_f_mask, ranks))
            family = (
                "PV1" if lb == 0 and ln == 0 else
                "BGE" if lb > 0 and ln == 0 else
                "NGramRecency" if lb == 0 and ln > 0 else
                "BGE+NGramRecency"
            )
            row = {
                "family": family,
                "lambda_bge": lb,
                "lambda_ngram": ln,
                "macro_author_top1": metrics["macro_author"]["top1"],
                "micro_top1": metrics["micro"]["top1"],
                "top3": metrics["micro"]["top3"],
                "top5": metrics["micro"]["top5"],
                "mrr_at_10": metrics["micro"]["mrr_at_10"],
                "missing10": metrics["micro"]["missing10"],
                "mean_rank_given_top10": metrics["micro"]["mean_rank_given_top10"],
                "rescue": transition.get("rescue", 0),
                "harm": transition.get("harm", 0),
                "net": transition.get("net", 0),
                "conflict_rescue": conflict_transition.get("rescue", 0),
                "conflict_harm": conflict_transition.get("harm", 0),
                "conflict_net": conflict_transition.get("net", 0),
                "pv1_rescue_vs_f_n": sum(pv1_rescue_vs_f_mask),
                "pv1_rescue_retained_n": rescue_retained,
                "pv1_rescue_retention_rate": rescue_retained / sum(pv1_rescue_vs_f_mask),
                "pv1_harm_vs_f_n": sum(pv1_harm_vs_f_mask),
                "pv1_harm_repaired_n": harm_repaired,
                "pv1_harm_repair_rate": harm_repaired / sum(pv1_harm_vs_f_mask),
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
            }
            if float(row["missing10"]) != float(inputs["baseline_metrics"]["micro"]["missing10"]):
                raise RuntimeError(f"Missing@10 changed at weights BGE={lb}, NGram={ln}")
            if float(row["rec10"]) != float(baseline_rec10):
                raise RuntimeError(f"Recovery@10 changed at weights BGE={lb}, NGram={ln}")
            grid_rows.append(row)

    by_pair = {(float(row["lambda_bge"]), float(row["lambda_ngram"])): row for row in grid_rows}
    bge_candidates = [row for row in grid_rows if float(row["lambda_ngram"]) == 0.0]
    ng_candidates = [row for row in grid_rows if float(row["lambda_bge"]) == 0.0]
    joint_all = list(grid_rows)
    joint_interior = [row for row in grid_rows if float(row["lambda_bge"]) > 0 and float(row["lambda_ngram"]) > 0]
    selections = {
        "PV1": by_pair[(0.0, 0.0)],
        "BGE": max(bge_candidates, key=selection_key),
        "NGramRecency": max(ng_candidates, key=selection_key),
        "BGE+NGramRecency": max(joint_interior, key=selection_key),
        "GlobalBestIncludingZeroWeights": max(joint_all, key=selection_key),
    }

    selected_pairs = {
        "PV1": (0.0, 0.0),
        "BGE": (float(selections["BGE"]["lambda_bge"]), 0.0),
        "NGramRecency": (0.0, float(selections["NGramRecency"]["lambda_ngram"])),
        "BGE+NGramRecency": (
            float(selections["BGE+NGramRecency"]["lambda_bge"]),
            float(selections["BGE+NGramRecency"]["lambda_ngram"]),
        ),
    }
    selected_rankings = {name: rank_by_pair[pair] for name, pair in selected_pairs.items()}

    subset_masks = {
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

    for method, ranks in selected_rankings.items():
        pair = selected_pairs[method]
        metrics = author_metrics(authors, ranks)
        selected_method_rows.append({
            "method": method,
            "lambda_bge": pair[0],
            "lambda_ngram": pair[1],
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
        for author, result in metrics["per_author"].items():
            per_author_rows.append({"method": method, "author": author, **result})
        recovery_rows.append({"method": method, **recovery_metrics(ranks, recovery_mask)})
        for subset_name, mask in subset_masks.items():
            sub_authors = [author for author, include in zip(authors, mask) if include]
            sub_ranks = [rank for rank, include in zip(ranks, mask) if include]
            sub = author_metrics(sub_authors, sub_ranks)
            subset_rows.append({
                "method": method,
                "subset": subset_name,
                "n": len(sub_ranks),
                "macro_author_top1": sub["macro_author"]["top1"],
                "micro_top1": sub["micro"]["top1"],
                "top3": sub["micro"]["top3"],
                "top5": sub["micro"]["top5"],
                "mrr_at_10": sub["micro"]["mrr_at_10"],
                "missing10": sub["micro"]["missing10"],
                "mean_rank_given_top10": sub["micro"]["mean_rank_given_top10"],
            })
        if method != "PV1":
            for subset_name, mask in (
                ("overall", subset_masks["overall"]),
                ("conflict", conflict_mask),
                ("non_conflict", non_conflict_mask),
                ("recovery_k5", recovery_mask),
                ("context_opportunity", opportunity_mask),
            ):
                transition_rows.append({
                    "method": method,
                    "subset": subset_name,
                    **transition_counts(baseline_ranks, ranks, mask),
                    **rank_movement(baseline_ranks, ranks, mask),
                })

    # Joint-vs-single comparisons are useful for the complementarity question.
    joint_ranks = selected_rankings["BGE+NGramRecency"]
    for comparator in ("BGE", "NGramRecency"):
        transition_rows.append({
            "method": "BGE+NGramRecency",
            "subset": f"paired_vs_{comparator}",
            **transition_counts(selected_rankings[comparator], joint_ranks),
            **rank_movement(selected_rankings[comparator], joint_ranks, opportunity_mask),
        })

    # Selected row-level predictions.
    selected_prediction_rows: list[dict[str, Any]] = []
    for index, rid in enumerate(row_ids):
        candidates = baseline_rankings[index]
        pv1_scores = baseline_scores[index]
        bg_support = bge_supports[index]
        ng_support = ngram_supports[index]
        method_out: dict[str, Any] = {}
        for method, pair in selected_pairs.items():
            ranking, ordered_scores = rerank_joint(
                candidates, pv1_scores, bg_support, ng_support,
                lambda_bge=pair[0], lambda_ngram=pair[1],
            )
            score_by_candidate = {
                candidate: float(pv1_scores[i] + pair[0] * bg_support[i] + pair[1] * ng_support[i])
                for i, candidate in enumerate(candidates)
            }
            rows = []
            for new_rank, candidate in enumerate(ranking, 1):
                i = candidates.index(candidate)
                rows.append({
                    "candidate": candidate,
                    "pv1_rank": i + 1,
                    "reranked_rank": new_rank,
                    "pv1_final_score": float(pv1_scores[i]),
                    "bge_support": float(bg_support[i]),
                    "ngram_recency_support": float(ng_support[i]),
                    "bge_contribution": pair[0] * float(bg_support[i]),
                    "ngram_contribution": pair[1] * float(ng_support[i]),
                    "reranked_final_score": score_by_candidate[candidate],
                    "source": str(inputs["baseline_candidate_rows"][index][i].get("source", "")),
                })
            method_out[method] = {
                "lambda_bge": pair[0],
                "lambda_ngram": pair[1],
                "gold_rank": rank_of(ranking, golds[index]),
                "candidates": rows,
            }
        selected_prediction_rows.append({
            "schema_version": 1,
            "row_id": rid,
            "author": authors[index],
            "gold": golds[index],
            "pv1_gold_rank": baseline_ranks[index],
            "frequency_gold_rank": frequency_ranks[index],
            "conflict": conflict_mask[index],
            "ambiguous": ambiguous_mask[index],
            "generic_missing": generic_missing_mask[index],
            "recovery_k5": recovery_mask[index],
            "methods": method_out,
            "gold_used_for_scoring_features": False,
            "dev3000_used": False,
            "test_used": False,
        })

    bge_summary = json.loads((args.output_root / "bge_scoring_summary.json").read_text(encoding="utf-8"))
    bge_history_touched = [int(bge_rows[rid].get("history_vectors_touched", 0)) for rid in row_ids]
    bge_candidates_with_history = [int(bge_rows[rid].get("candidates_with_history", 0)) for rid in row_ids]
    context_diagnostics = {
        "same_pinyin_history": {
            "mean_rows": statistics.fmean(visible_history_rows),
            "p50_rows": percentile(visible_history_rows, 0.50),
            "p95_rows": percentile(visible_history_rows, 0.95),
            "max_rows": max(visible_history_rows),
        },
        "bge64": {
            "context_chars": BGE_CONTEXT_CHARS,
            "top_n_history_per_candidate": BGE_TOP_N,
            "required_unique_historical_contexts": bge_summary["required_unique_historical_contexts"],
            "mean_history_vectors_touched": statistics.fmean(bge_history_touched),
            "p95_history_vectors_touched": percentile(bge_history_touched, 0.95),
            "mean_candidates_with_history": statistics.fmean(bge_candidates_with_history),
            "rows_with_any_candidate_history": sum(value > 0 for value in bge_candidates_with_history),
        },
        "ngram_recency": {
            "max_n": args.hard_max_n,
            "tau": args.hard_tau,
            "effective_n_distribution": dict(sorted(ngram_effective_n.items())),
            "mean_matched_history_rows": statistics.fmean(ngram_matched_rows),
            "p50_matched_history_rows": percentile(ngram_matched_rows, 0.50),
            "p95_matched_history_rows": percentile(ngram_matched_rows, 0.95),
        },
    }

    # Latency: BGE online values were measured during actual scoring; NGram here
    # measures scorer only. Joint fusion cost is negligible but measured separately.
    fusion_latency: dict[str, Any] = {}
    for method, pair in selected_pairs.items():
        values: list[float] = []
        for i, candidates in enumerate(baseline_rankings):
            t0 = time.perf_counter()
            rerank_joint(
                candidates, baseline_scores[i], bge_supports[i], ngram_supports[i],
                lambda_bge=pair[0], lambda_ngram=pair[1],
            )
            values.append((time.perf_counter() - t0) * 1000.0)
        fusion_latency[method] = latency_summary(values)

    latency = {
        "BGE64": {
            "query_embedding": bge_summary["online_query_embedding_latency_ms"],
            "lookup_cosine": bge_summary["online_lookup_cosine_latency_ms"],
            "online_total": bge_summary["online_total_latency_ms"],
            "historical_embedding_offline_this_run": bge_summary["new_history_embedding_latency_ms_this_run"],
        },
        "NGramRecency": {
            "scorer_only_history_lookup_excluded": latency_summary(ngram_latency_ms),
        },
        "fusion_sort_only": fusion_latency,
        "warning": "Do not use latency from a run performed concurrently with another CPU/GPU-heavy experiment for thesis comparison.",
    }

    # Write outputs.
    grid_path = args.output_root / "grid_results.csv"
    selection_path = args.output_root / "family_selection.json"
    method_path = args.output_root / "selected_method_metrics.csv"
    subset_path = args.output_root / "selected_subset_metrics.csv"
    author_path = args.output_root / "per_author_metrics.csv"
    recovery_path = args.output_root / "recovery_metrics.csv"
    transition_path = args.output_root / "rank_transition_metrics.csv"
    prediction_path = args.output_root / "selected_predictions.jsonl"
    diagnostic_path = args.output_root / "context_diagnostics.json"
    latency_path = args.output_root / "latency.json"
    comparison_path = args.output_root / "comparison.json"

    write_csv(grid_path, grid_rows)
    write_json(selection_path, {
        "selection_metric": "Macro-author Top1 on ALL Train-Val",
        "tie_break": "MRR@10, Top3, lower total context weight, lower BGE, lower NGram",
        "PV1": selections["PV1"],
        "BGE": selections["BGE"],
        "NGramRecency": selections["NGramRecency"],
        "BGE+NGramRecency": selections["BGE+NGramRecency"],
        "GlobalBestIncludingZeroWeights": selections["GlobalBestIncludingZeroWeights"],
        "joint_interior_definition": "lambda_bge>0 AND lambda_ngram>0",
        "gold_used_for_scoring_features": False,
        "dev3000_used": False,
        "test_used": False,
    })
    write_csv(method_path, selected_method_rows)
    write_csv(subset_path, subset_rows)
    write_csv(author_path, per_author_rows)
    write_csv(recovery_path, recovery_rows)
    write_csv(transition_path, transition_rows)
    write_jsonl(prediction_path, selected_prediction_rows)
    write_json(diagnostic_path, context_diagnostics)
    write_json(latency_path, latency)

    comparison = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_pv1_bge_ngram_context_reranking_v1",
        "purpose": "PV1-only final-list reranking: semantic BGE64 vs lexical-temporal NGramRecency and their additive combination",
        "evaluation_population": {
            "train_val_rows": len(row_ids),
            "authors": sorted(set(authors)),
            "context_opportunity_n": sum(opportunity_mask),
            "recovery_population_definition": "Generic Missing AND Gold in frozen Personal K5",
            "recovery_population_n": sum(recovery_mask),
            "pv1_rescue_vs_frequency_n": sum(pv1_rescue_vs_f_mask),
            "pv1_harm_vs_frequency_n": sum(pv1_harm_vs_f_mask),
        },
        "baseline": {
            "method": "PV1",
            "k_pv": 1,
            "lambda_frequency": 4.0,
            "lambda_pv": 4.0,
            "metrics": inputs["headline"],
            "recovery": recovery_metrics(baseline_ranks, recovery_mask),
        },
        "formulas": {
            "BGE": "S_PV1 + lambda_B * P_BGE",
            "NGramRecency": "S_PV1 + lambda_N * P_NG",
            "Joint": "S_PV1 + lambda_B * P_BGE + lambda_N * P_NG",
        },
        "bge": {
            "context_chars": BGE_CONTEXT_CHARS,
            "top_n_per_candidate": BGE_TOP_N,
            "similarity": "cosine; negative values clamped to zero; candidate Top-5 summed; normalized over current PV1 Top10",
        },
        "ngram_recency": {
            "max_n": args.hard_max_n,
            "tau": args.hard_tau,
            "definition": "largest exact suffix order with candidate-target history; exp(-age/tau) target mass; normalized over current PV1 Top10",
        },
        "weight_grids": {
            "lambda_bge": list(args.lambda_bge),
            "lambda_ngram": list(args.lambda_ngram),
        },
        "selected": selections,
        "selected_method_metrics": selected_method_rows,
        "recovery_metrics": recovery_rows,
        "invariants": {
            "candidate_set_changed": False,
            "missing10_must_equal_pv1": True,
            "recovery10_must_equal_pv1": True,
            "zero_weights_exactly_reproduce_pv1": True,
            "frozen_pv1_final_scores_preserved_as_base": True,
            "h5000_before_exact_pinyin_filter": True,
            "strictly_prior_same_author_history": True,
        },
        "provenance": {
            name: {"path": str(path.resolve()), "sha256": inputs["input_hashes"][name]}
            for name, path in inputs["input_paths"].items()
        },
        "bge_model": {
            "path": str(args.bge_model.resolve()),
            "sha256": sha256_file(args.bge_model),
        },
        "gold_used_for_scoring_features": False,
        "gold_used_for_train_val_selection_evaluation": True,
        "dev3000_used": False,
        "test_used": False,
        "evaluation_runtime_seconds": time.perf_counter() - started,
    }
    write_json(comparison_path, comparison)

    output_files = [
        args.output_root / "run_manifest.json",
        args.output_root / "bge_scores.jsonl",
        args.output_root / "bge_scoring_summary.json",
        grid_path, selection_path, method_path, subset_path, author_path,
        recovery_path, transition_path, prediction_path, diagnostic_path,
        latency_path, comparison_path,
    ]
    checksums = {path.name: sha256_file(path) for path in output_files if path.is_file()}
    write_json(args.output_root / "artifact_checksums.json", {
        "schema_version": 1,
        "experiment": "initial_pv1_bge_ngram_context_reranking_v1",
        "artifacts": checksums,
        "note": "SQLite embedding cache is intentionally excluded because it is an implementation cache, not a canonical research-result artifact.",
    })

    print("\n=== SELECTED CONFIGURATIONS ===", flush=True)
    for name in ("BGE", "NGramRecency", "BGE+NGramRecency", "GlobalBestIncludingZeroWeights"):
        row = selections[name]
        print(
            f"{name:30s} lambda_B={float(row['lambda_bge']):g} lambda_N={float(row['lambda_ngram']):g}",
            flush=True,
        )
    print("\n=== COMPLETE ===", flush=True)
    for row in selected_method_rows:
        print(
            f"{row['method']:22s} "
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
            f"{row['method']:22s} Rec1={float(row['rec1']):.4f} "
            f"Rec3={float(row['rec3']):.4f} Rec5={float(row['rec5']):.4f} "
            f"Rec10={float(row['rec10']):.4f} RecMRR={float(row['recovery_mrr_at_10']):.4f}",
            flush=True,
        )
    print(f"\nOutputs: {args.output_root.resolve()}", flush=True)
    print("Gold used for scoring/features: false", flush=True)
    print("Dev3000 used: false", flush=True)
    print("Test used: false", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--candidate-surface", type=Path, required=True)
    parser.add_argument("--frequency-pv1-predictions", type=Path, required=True)
    parser.add_argument("--bge-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed-bge-cache", type=Path, default=None)
    parser.add_argument("--cuda-path", type=Path, default=None)
    parser.add_argument("--lambda-bge", nargs="+", type=float, default=list(DEFAULT_BGE_LAMBDAS))
    parser.add_argument("--lambda-ngram", nargs="+", type=float, default=list(DEFAULT_NGRAM_LAMBDAS))
    parser.add_argument("--hard-max-n", type=int, default=DEFAULT_HARD_MAX_N)
    parser.add_argument("--hard-tau", type=float, default=DEFAULT_HARD_TAU)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--allow-unfrozen-input-hashes", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace) -> None:
    for path_name in ("fit", "val", "candidate_surface", "frequency_pv1_predictions", "bge_model"):
        path = getattr(args, path_name)
        if not path.is_file():
            raise RuntimeError(f"Input file does not exist: --{path_name.replace('_', '-')} {path}")
    if args.seed_bge_cache is not None and not args.seed_bge_cache.is_file():
        raise RuntimeError(f"Seed BGE cache does not exist: {args.seed_bge_cache}")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    if args.hard_max_n <= 0:
        raise ValueError("--hard-max-n must be positive")
    if args.hard_tau <= 0:
        raise ValueError("--hard-tau must be positive")
    args.lambda_bge = tuple(sorted(set(float(v) for v in args.lambda_bge)))
    args.lambda_ngram = tuple(sorted(set(float(v) for v in args.lambda_ngram)))
    if 0.0 not in args.lambda_bge or 0.0 not in args.lambda_ngram:
        raise ValueError("Both lambda grids must include 0 for exact PV1 control")
    if not any(v > 0 for v in args.lambda_bge) or not any(v > 0 for v in args.lambda_ngram):
        raise ValueError("Both lambda grids need at least one positive value")
    if any(v < 0 for v in args.lambda_bge + args.lambda_ngram):
        raise ValueError("Context weights must be non-negative")


def main() -> None:
    args = build_parser().parse_args()
    normalize_args(args)
    total_started = time.perf_counter()
    inputs = prepare_inputs(args)
    global row_index_by_id
    row_index_by_id = {rid: index for index, rid in enumerate(inputs["row_ids"])}
    ensure_manifest(args, inputs)

    print("=== INITIAL PV1 BGE + NGRAM CONTEXT RERANKING V1 ===", flush=True)
    print(f"Train-Val rows: {len(inputs['row_ids'])}", flush=True)
    print(f"Authors: {sorted(set(inputs['authors']))}", flush=True)
    print("Baseline: frozen PV1 K=1, lambda_F=4, lambda_PV=4", flush=True)
    print("BGE64: last 64 chars, candidate-conditioned Top-5 positive cosine", flush=True)
    print(f"NGramRecency: HardBackoff maxN={args.hard_max_n}, tau={args.hard_tau:g}", flush=True)
    print(f"lambda_B grid: {list(args.lambda_bge)}", flush=True)
    print(f"lambda_N grid: {list(args.lambda_ngram)}", flush=True)
    print("Candidate set: frozen PV1 Top10 only; pure reranking", flush=True)
    print("Gold used for scoring/features: false", flush=True)
    print("Dev3000 used: false", flush=True)
    print("Test used: false", flush=True)

    bge_rows = run_bge_scoring(args, inputs)
    evaluate(args, inputs, bge_rows)
    print(f"Total wall time: {time.perf_counter() - total_started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
