from __future__ import annotations

"""Three-base Recovery -> NGramRecency + BGERecency final-list reranking.

This experiment continues from the completed V1 Recovery -> NGramRecency run.
It does NOT reconstruct or re-select the three Stage-1 recovery bases. Instead it
reads the frozen V1 artifacts and adds a semantic-temporal BGERecency signal to
exactly the same per-base Top10 candidate sets.

Frozen recovery bases
---------------------
1. K5+Entropy          : coverage-first
2. 4P+4CS+2E           : balanced
3. 6P+2CS+.25E         : front-rank / Top3-oriented

Stage-2 score
-------------
    S_final(c) = S_REC(c)
               + lambda_N * P_NG-R(c)
               + lambda_B * P_BGE-R(c)

NGramRecency is read from V1 (HardBackoff maxN=2, tau_N=2048).
BGERecency uses:
- current context: last 64 Python Unicode code points;
- candidate-conditioned legal same-Pinyin history;
- Top-5 history retrieval PER CANDIDATE by cosine similarity ONLY;
- recency affects aggregation only:
      max(0, cosine) * exp(-age / 2048)
- support is normalized across the frozen Top10 candidate set.

Scientific safeguards
---------------------
- Train-Fit / Train-Val only; Dev3000/Test are never read.
- Gold is never used for feature construction or scoring.
- Gold is used only for Train-Val evaluation / lambda selection.
- Same-author strictly-prior history.
- H5000 is applied BEFORE exact-Pinyin filtering.
- Earlier Train-Val rows may be history for later Train-Val rows.
- Stage-2 is pure reranking: candidate set, Missing@10 and Recovery Rec@10
  must remain invariant for every configuration.
- (lambda_N, lambda_B)=(0,0) must exactly reproduce frozen Stage 1.
- lambda_B=0 must reproduce the completed V1 NGramRecency selection.
- The previous PV1 BGE cache is read-only seed evidence. New candidate surfaces
  may require additional historical context embeddings; those are written only
  to this experiment's versioned local cache.
- The output directory is resumable only when its setup manifest exactly matches
  the current inputs/configuration; otherwise the run refuses to overwrite it.
"""

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Frozen protocol / provenance
# ---------------------------------------------------------------------------

EXPECTED_FIT_SHA256 = "162f5c98daa86cc69947571e6d8f20fc401f0a82cdd3fd6e517eb7be2addbdb4"
EXPECTED_VAL_SHA256 = "d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4"
EXPECTED_STAGE1_FROZEN_SHA256 = "54e60073daabb14bb7cf43136a335216888ea03c06d078a4eec56e5775a0cfbc"
EXPECTED_NGRAM_SUPPORT_SHA256 = "03858de42c41a26c4134d4b069b61ab2a5468c24cbd70a71958d600e448a97e1"

EXPECTED_FIT_ROWS = 144_526
EXPECTED_VAL_ROWS = 34_416
EXPECTED_GENERIC_MISSING = 12_565
EXPECTED_RECOVERABLE_K5 = 4_910

HISTORY_BUDGET = 5000
BGE_CONTEXT_CHARS = 64
BGE_TOP_N = 5
BGE_TAU = 2048.0

DEFAULT_LAMBDA_N = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0)
DEFAULT_LAMBDA_B = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0)

BASE_ORDER = (
    "K5+Entropy",
    "4P+4CS+2E",
    "6P+2CS+.25E",
)

# Reporting-only historical controls. These are never used for scoring or
# hyperparameter selection. Values are the already-recorded Train-Val results.
HISTORICAL_CONTROLS: tuple[dict[str, Any], ...] = (
    {
        "group": "Generic control",
        "method": "G",
        "macro_author_top1": 0.307099,
        "micro_top1": 0.330573,
        "top3": 0.491341,
        "top5": 0.557996,
        "mrr_at_10": 0.426472,
        "missing10": 0.365092,
        "recovery_rec1": 0.0,
        "recovery_rec3": 0.0,
        "recovery_rec5": 0.0,
        "recovery_rec10": 0.0,
        "recovery_mrr_at_10": 0.0,
        "source_precision": "historical_reported",
    },
    {
        "group": "Frequency control",
        "method": "F",
        "macro_author_top1": 0.382495,
        "micro_top1": 0.408473,
        "top3": 0.555120,
        "top5": 0.601000,
        "mrr_at_10": 0.489432,
        "missing10": 0.365092,
        "recovery_rec1": 0.0,
        "recovery_rec3": 0.0,
        "recovery_rec5": 0.0,
        "recovery_rec10": 0.0,
        "recovery_mrr_at_10": 0.0,
        "source_precision": "historical_reported",
    },
    {
        "group": "Historical Recovery control",
        "method": "PV1",
        "macro_author_top1": 0.401872,
        "micro_top1": 0.426749,
        "top3": 0.598907,
        "top5": 0.663093,
        "mrr_at_10": 0.524450,
        "missing10": 0.291144,
        "recovery_rec1": 0.1900,
        "recovery_rec3": 0.4923,
        "recovery_rec5": 0.5363,
        "recovery_rec10": 0.5401,
        "recovery_mrr_at_10": 0.3363,
        "source_precision": "historical_reported_rounded_recovery",
    },
    {
        "group": "Historical Context control",
        "method": "PV1 + NGramRecency",
        "macro_author_top1": 0.429091,
        "micro_top1": 0.452755,
        "top3": 0.611285,
        "top5": 0.665940,
        "mrr_at_10": 0.541944,
        "missing10": 0.291144,
        "recovery_rec1": 0.3601,
        "recovery_rec3": 0.5149,
        "recovery_rec5": 0.5379,
        "recovery_rec10": 0.5401,
        "recovery_mrr_at_10": 0.4361,
        "source_precision": "historical_reported_rounded_recovery",
    },
    {
        "group": "Full historical Context control",
        "method": "PV1 + NGramRecency + BGERecency",
        "macro_author_top1": 0.429506,
        "micro_top1": 0.453423,
        "top3": 0.612099,
        "top5": 0.667335,
        "mrr_at_10": 0.542766,
        "missing10": 0.291144,
        "recovery_rec1": 0.3582,
        "recovery_rec3": 0.5165,
        "recovery_rec5": 0.5381,
        "recovery_rec10": 0.5401,
        "recovery_mrr_at_10": 0.4356,
        "source_precision": "historical_reported_rounded_recovery",
    },
)


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
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}") from exc
            if "row_id" not in row:
                raise RuntimeError(f"Missing row_id at {path}:{line_no}")
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


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *, mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8", newline="\n") as sink:
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


def parse_grid(text: str, *, label: str) -> tuple[float, ...]:
    values = sorted({float(value.strip()) for value in text.split(",") if value.strip()})
    if not values:
        raise ValueError(f"{label} grid is empty")
    if any(value < 0 for value in values):
        raise ValueError(f"{label} values must be non-negative")
    if 0.0 not in values:
        raise ValueError(f"{label} grid must include 0")
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
    return tuple(str(x) for x in value)


def context_of(row: Mapping[str, Any]) -> str:
    return str(row.get("context", row.get("model_used_context", "")))


def candidate_text(item: Mapping[str, Any]) -> str:
    value = item.get("candidate", item.get("text", item.get("target")))
    if value is None:
        raise RuntimeError(f"Cannot identify candidate text: {item}")
    return str(value)


def rank_of(ranking: Sequence[Mapping[str, Any]], gold: str) -> int | None:
    for index, item in enumerate(ranking, 1):
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


def normalized_vector(vector: Any) -> Any:
    import numpy as np

    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 0 else value


def latency_summary(values_ms: Sequence[float]) -> dict[str, Any]:
    values = [float(x) for x in values_ms]
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "p95": None, "p99": None}
    ordered = sorted(values)

    def p(frac: float) -> float:
        return ordered[int((len(ordered) - 1) * frac)]

    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": p(0.50),
        "p90": p(0.90),
        "p95": p(0.95),
        "p99": p(0.99),
    }


# ---------------------------------------------------------------------------
# CUDA / native stderr helpers
# ---------------------------------------------------------------------------


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
    raise RuntimeError(
        "No valid CUDA_PATH found. Pass --cuda-path, e.g. "
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    )


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
    age: int  # 0 = immediately previous same-author interaction in raw H5000 stream


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
            self.records[author] = tuple(values)
            self.positions[author] = tuple(r.position for r in values)

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
            output.append(VisibleHistory(record=record, age=stop - 1 - index))
        return tuple(output)


# ---------------------------------------------------------------------------
# BGE cache and BGERecency support
# ---------------------------------------------------------------------------


class VectorCache:
    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        import numpy as np

        self.np = np
        self.path = path
        self.read_only = read_only
        if read_only:
            if not path.is_file():
                raise FileNotFoundError(path)
            uri = f"file:{path.resolve().as_posix()}?mode=ro"
            self.connection = sqlite3.connect(uri, uri=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
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
            raise RuntimeError(f"Corrupt BGE vector cache row in {self.path}")
        return vector

    def put(self, context: str, vector: Any) -> None:
        if self.read_only:
            raise RuntimeError("Cannot write to read-only VectorCache")
        value = self.np.asarray(vector, dtype=self.np.float32).reshape(-1)
        self.connection.execute(
            "INSERT OR REPLACE INTO embeddings(context, dim, vector) VALUES(?,?,?)",
            (context, int(value.size), value.tobytes()),
        )

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])

    def commit(self) -> None:
        if not self.read_only:
            self.connection.commit()

    def close(self) -> None:
        if not self.read_only:
            self.connection.commit()
        self.connection.close()


def required_bge_history_contexts(
    *,
    stage1_rows: Sequence[Mapping[str, Any]],
    val: Mapping[str, Mapping[str, Any]],
    history: CausalHistoryIndex,
    progress_every: int,
) -> set[str]:
    contexts: set[str] = set()
    for number, stage1 in enumerate(stage1_rows, 1):
        row_id = str(stage1["row_id"])
        candidate_union: set[str] = set()
        for base in BASE_ORDER:
            candidate_union.update(str(x) for x in stage1["bases"][base]["top10"])
        vrow = val[row_id]
        visible = history.visible_same_pinyin(
            author=str(vrow["author"]),
            position=int(vrow["chronological_position"]),
            pinyin=pinyin_of(vrow),
        )
        for item in visible:
            if item.record.target in candidate_union:
                contexts.add(item.record.context[-BGE_CONTEXT_CHARS:])
        if progress_every > 0 and (number % (progress_every * 5) == 0 or number == len(stage1_rows)):
            print(
                f"BGE history-context audit {number}/{len(stage1_rows)}  unique_required={len(contexts)}",
                flush=True,
            )
    return contexts


def bge_recency_support(
    *,
    query_vector: Any,
    candidates: Sequence[str],
    visible: Sequence[VisibleHistory],
    history_vectors: Mapping[str, Any],
    tau: float = BGE_TAU,
) -> tuple[dict[str, float], dict[str, int], int]:
    """Candidate-conditioned cosine Top-5 retrieval; recency only in aggregation."""
    import numpy as np

    grouped: dict[str, list[VisibleHistory]] = {candidate: [] for candidate in candidates}
    counts = {candidate: 0 for candidate in candidates}
    for item in visible:
        target = item.record.target
        if target in grouped:
            grouped[target].append(item)
            counts[target] += 1

    raw = {candidate: 0.0 for candidate in candidates}
    vectors_touched = 0
    for candidate in candidates:
        histories = grouped[candidate]
        if not histories:
            continue
        vectors = []
        for item in histories:
            key = item.record.context[-BGE_CONTEXT_CHARS:]
            vector = history_vectors.get(key)
            if vector is None:
                raise KeyError(
                    "Missing cached BGE history vector for context sha256="
                    + hashlib.sha256(key.encode("utf-8")).hexdigest()
                )
            vectors.append(vector)
        matrix = np.vstack(vectors)
        similarities = matrix @ query_vector
        vectors_touched += len(histories)

        # Retrieval is cosine-only. Deterministic secondary keys are used only
        # for exact cosine ties and do not introduce recency into retrieval.
        order = sorted(
            range(len(histories)),
            key=lambda i: (
                -float(similarities[i]),
                int(histories[i].record.position),
                str(histories[i].record.row_id),
            ),
        )[:BGE_TOP_N]
        value = 0.0
        for i in order:
            similarity = max(0.0, float(similarities[i]))
            value += similarity * math.exp(-float(histories[i].age) / tau)
        raw[candidate] = value

    normalized = normalize_nonnegative([raw[c] for c in candidates], uniform_if_zero=True)
    return dict(zip(candidates, normalized)), counts, vectors_touched


# ---------------------------------------------------------------------------
# Metrics / reranking
# ---------------------------------------------------------------------------


def rerank_fixed_candidate_set(
    base_candidates: Sequence[Mapping[str, Any]],
    ngram_support: Mapping[str, float],
    bge_support: Mapping[str, float],
    *,
    lambda_n: float,
    lambda_b: float,
) -> list[dict[str, Any]]:
    base_texts = [candidate_text(item) for item in base_candidates]
    if set(base_texts) != set(ngram_support) or set(base_texts) != set(bge_support):
        raise RuntimeError("Context support candidate set differs from frozen Stage-1 Top10")

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(base_candidates, 1):
        row = dict(item)
        text = candidate_text(item)
        base_rank = int(item.get("base_rank", item.get("rank", index)))
        base_score = float(item["final_score"])
        nscore = float(ngram_support[text])
        bscore = float(bge_support[text])
        row["base_rank"] = base_rank
        row["base_score"] = base_score
        row["ngram_recency_support"] = nscore
        row["bge_recency_support"] = bscore
        row["context_lambda_n"] = float(lambda_n)
        row["context_lambda_b"] = float(lambda_b)
        row["final_score"] = base_score + float(lambda_n) * nscore + float(lambda_b) * bscore
        rows.append(row)

    rows.sort(
        key=lambda row: (
            -float(row["final_score"]),
            int(row["base_rank"]),
            str(row["candidate"]),
        )
    )
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def metric_summary(rows: Sequence[Mapping[str, Any]], rank_key: str, method: str) -> dict[str, Any]:
    by_author: dict[str, list[int | None]] = defaultdict(list)
    ranks: list[int | None] = []
    for row in rows:
        rank = row.get(rank_key)
        rank = None if rank is None else int(rank)
        ranks.append(rank)
        by_author[str(row["author"])].append(rank)
    if not ranks:
        raise RuntimeError("Cannot score empty rows")

    def top_at(values: Sequence[int | None], k: int) -> float:
        return sum(rank is not None and rank <= k for rank in values) / len(values)

    per_author_top1 = {author: top_at(values, 1) for author, values in sorted(by_author.items())}
    found = [rank for rank in ranks if rank is not None]
    return {
        "method": method,
        "n": len(ranks),
        "authors": len(per_author_top1),
        "macro_author_top1": statistics.fmean(per_author_top1.values()),
        "micro_top1": top_at(ranks, 1),
        "top3": top_at(ranks, 3),
        "top5": top_at(ranks, 5),
        "mrr_at_10": sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / len(ranks),
        "missing10": sum(rank is None for rank in ranks) / len(ranks),
        "mean_rank_given_top10": statistics.fmean(found) if found else None,
        "per_author_top1": per_author_top1,
    }


def recovery_summary(rows: Sequence[Mapping[str, Any]], rank_key: str) -> dict[str, Any]:
    missing = [row for row in rows if bool(row["generic_missing"])]
    available = [row for row in missing if bool(row["gold_in_personal_k5"])]
    if len(missing) != EXPECTED_GENERIC_MISSING:
        raise RuntimeError(f"Generic Missing regression failed: {len(missing)}")
    if len(available) != EXPECTED_RECOVERABLE_K5:
        raise RuntimeError(f"K5 recoverable regression failed: {len(available)}")
    ranks = [None if row.get(rank_key) is None else int(row[rank_key]) for row in available]

    def count_at(k: int) -> int:
        return sum(rank is not None and rank <= k for rank in ranks)

    r1, r3, r5, r10 = (count_at(k) for k in (1, 3, 5, 10))
    found = [rank for rank in ranks if rank is not None]
    denom = len(ranks)
    return {
        "generic_missing_n": len(missing),
        "recoverable_n": denom,
        "recovered_at_1_n": r1,
        "recovered_at_3_n": r3,
        "recovered_at_5_n": r5,
        "recovered_at_10_n": r10,
        "recovered_at_1": r1 / denom,
        "recovered_at_3": r3 / denom,
        "recovered_at_5": r5 / denom,
        "recovered_at_10": r10 / denom,
        "recovery_mrr_at_10": sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / denom,
        "mean_recovered_rank": statistics.fmean(found) if found else None,
    }


def transition_counts(rows: Sequence[Mapping[str, Any]], base_key: str, new_key: str) -> dict[str, int]:
    result = {"n": 0, "rescue": 0, "harm": 0, "unchanged_correct": 0, "unchanged_wrong": 0}
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
    result = {"comparable": 0, "improved": 0, "worsened": 0, "same": 0}
    for row in rows:
        before = row.get(base_key)
        after = row.get(new_key)
        if before is None or after is None:
            continue
        before_i, after_i = int(before), int(after)
        result["comparable"] += 1
        if after_i < before_i:
            result["improved"] += 1
        elif after_i > before_i:
            result["worsened"] += 1
        else:
            result["same"] += 1
    result["net"] = result["improved"] - result["worsened"]
    return result


def close_enough(a: float, b: float, tol: float = 2e-6) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


# ---------------------------------------------------------------------------
# Input / resumability verification
# ---------------------------------------------------------------------------


def verify_inputs(args: argparse.Namespace) -> dict[str, Any]:
    stage1_path = args.base_ngram_root / "stage1_frozen.jsonl"
    ngram_path = args.base_ngram_root / "ngram_recency_support.jsonl"
    comparison_path = args.base_ngram_root / "comparison.json"
    required = {
        "fit": args.fit,
        "val": args.val,
        "stage1_frozen": stage1_path,
        "ngram_recency_support": ngram_path,
        "v1_comparison": comparison_path,
        "bge_model": args.bge_model,
    }
    for label, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")
    if args.seed_bge_cache is not None and not args.seed_bge_cache.is_file():
        raise FileNotFoundError(f"seed_bge_cache: {args.seed_bge_cache}")

    hashes = {label: sha256_file(path) for label, path in required.items() if label != "bge_model"}
    hashes["bge_model"] = sha256_file(args.bge_model)
    if hashes["fit"] != EXPECTED_FIT_SHA256:
        raise RuntimeError(f"Fit SHA mismatch: {hashes['fit']}")
    if hashes["val"] != EXPECTED_VAL_SHA256:
        raise RuntimeError(f"Val SHA mismatch: {hashes['val']}")
    if hashes["stage1_frozen"] != EXPECTED_STAGE1_FROZEN_SHA256:
        raise RuntimeError(
            "V1 frozen Stage-1 SHA mismatch:\n"
            f"expected={EXPECTED_STAGE1_FROZEN_SHA256}\nactual={hashes['stage1_frozen']}"
        )
    if hashes["ngram_recency_support"] != EXPECTED_NGRAM_SUPPORT_SHA256:
        raise RuntimeError(
            "V1 NGramRecency support SHA mismatch:\n"
            f"expected={EXPECTED_NGRAM_SUPPORT_SHA256}\nactual={hashes['ngram_recency_support']}"
        )
    if args.seed_bge_cache is not None:
        hashes["seed_bge_cache"] = sha256_file(args.seed_bge_cache)
    return hashes


def setup_payload(
    *,
    args: argparse.Namespace,
    input_hashes: Mapping[str, Any],
    lambda_n: Sequence[float],
    lambda_b: Sequence[float],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "initial_recovery_bge_ngram_context_fusion_v2",
        "status": "setup",
        "input_sha256": dict(input_hashes),
        "base_ngram_root": str(args.base_ngram_root.resolve()),
        "bge_model": str(args.bge_model.resolve()),
        "seed_bge_cache": str(args.seed_bge_cache.resolve()) if args.seed_bge_cache else None,
        "lambda_n_grid": list(lambda_n),
        "lambda_b_grid": list(lambda_b),
        "bge_context_chars": BGE_CONTEXT_CHARS,
        "bge_top_n": BGE_TOP_N,
        "bge_tau": BGE_TAU,
        "history_budget": HISTORY_BUDGET,
        "gold_used_for_scoring": False,
        "dev3000_used": False,
        "test_used": False,
    }


def prepare_resumable_output(output_root: Path, setup: Mapping[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    setup_path = output_root / "run_setup.json"
    existing = list(output_root.iterdir())
    if not existing:
        write_json(setup_path, setup)
        return
    if not setup_path.is_file():
        raise RuntimeError(
            f"Refusing non-empty output directory without run_setup.json: {output_root}\n"
            "Choose a new versioned --output-root."
        )
    previous = json.loads(setup_path.read_text(encoding="utf-8"))
    if previous != dict(setup):
        raise RuntimeError(
            f"Existing output setup differs from this run: {output_root}\n"
            "Choose a new versioned --output-root."
        )
    print(f"Resuming matching output root: {output_root}", flush=True)


def load_completed_bge_support(
    path: Path,
    stage1: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    completed = index_rows(read_jsonl(path), "BGERecency support")
    for row_id, row in completed.items():
        if row_id not in stage1:
            raise RuntimeError(f"BGE support row not in Stage1: {row_id}")
        bases = row.get("bases")
        if not isinstance(bases, Mapping):
            raise RuntimeError(f"Malformed BGE support bases at {row_id}")
        for base in BASE_ORDER:
            expected = tuple(str(x) for x in stage1[row_id]["bases"][base]["top10"])
            actual = tuple(str(x) for x in bases[base]["candidates"])
            if actual != expected:
                raise RuntimeError(f"Resumed BGE candidate mismatch at {row_id} {base}")
    return completed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument(
        "--base-ngram-root",
        type=Path,
        required=True,
        help="Completed recovery_ngram_context_fusion_v1 directory",
    )
    parser.add_argument("--bge-model", type=Path, required=True)
    parser.add_argument(
        "--seed-bge-cache",
        type=Path,
        default=None,
        help="Optional prior BGE history cache, opened strictly read-only",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--lambda-n",
        default=",".join(str(x) for x in DEFAULT_LAMBDA_N),
    )
    parser.add_argument(
        "--lambda-b",
        default=",".join(str(x) for x in DEFAULT_LAMBDA_B),
    )
    parser.add_argument("--cuda-path", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    lambda_n_grid = parse_grid(args.lambda_n, label="lambda_N")
    lambda_b_grid = parse_grid(args.lambda_b, label="lambda_B")
    input_hashes = verify_inputs(args)
    setup = setup_payload(
        args=args,
        input_hashes=input_hashes,
        lambda_n=lambda_n_grid,
        lambda_b=lambda_b_grid,
    )
    prepare_resumable_output(args.output_root, setup)

    started = time.perf_counter()
    print("=== RECOVERY -> NGRAMRECENCY + BGERECENCY FUSION V2 ===")
    print(f"Recovery bases: {list(BASE_ORDER)}")
    print("NGramRecency: read frozen V1 support; HardBackoff maxN=2, tau_N=2048")
    print(
        f"BGERecency: last-{BGE_CONTEXT_CHARS}, candidate-conditioned Top-{BGE_TOP_N} "
        f"by cosine only, tau_B={BGE_TAU:g}, recency only in aggregation"
    )
    print(f"lambda_N grid: {list(lambda_n_grid)}")
    print(f"lambda_B grid: {list(lambda_b_grid)}")
    print("Candidate set: frozen separately for each Recovery base; pure Stage-2 reranking")
    print("Gold used for scoring/features: false")
    print("Gold used for Train-Val selection/evaluation only: true")
    print("Dev3000 used: false")
    print("Test used: false")
    print()

    fit_rows = read_jsonl(args.fit)
    val_rows = read_jsonl(args.val)
    stage1_rows = read_jsonl(args.base_ngram_root / "stage1_frozen.jsonl")
    ngram_rows = read_jsonl(args.base_ngram_root / "ngram_recency_support.jsonl")
    v1_comparison = json.loads((args.base_ngram_root / "comparison.json").read_text(encoding="utf-8"))

    if len(fit_rows) != EXPECTED_FIT_ROWS:
        raise RuntimeError(f"Unexpected Train-Fit rows: {len(fit_rows)}")
    if len(val_rows) != EXPECTED_VAL_ROWS or len(stage1_rows) != EXPECTED_VAL_ROWS:
        raise RuntimeError("Unexpected Train-Val / Stage1 row count")

    val = index_rows(val_rows, "Train-Val")
    stage1 = index_rows(stage1_rows, "Stage1 frozen")
    ngram = index_rows(ngram_rows, "V1 NGramRecency support")
    if not (set(val) == set(stage1) == set(ngram)):
        raise RuntimeError("Row-ID mismatch among Train-Val / Stage1 / NGram support")

    # Verify V1 protocol before extending it.
    if v1_comparison.get("status") != "complete":
        raise RuntimeError("Base V1 comparison is not complete")
    protocol = v1_comparison.get("protocol", {})
    if protocol.get("dev3000_used") is not False or protocol.get("test_used") is not False:
        raise RuntimeError("Base V1 provenance crossed Dev3000/Test boundary")
    if v1_comparison.get("invariants", {}).get("candidate_set_changed") is not False:
        raise RuntimeError("Base V1 candidate-set invariant not satisfied")

    history = CausalHistoryIndex([*fit_rows, *val_rows])

    # ------------------------------------------------------------------
    # Required historical BGE contexts and local cache population.
    # ------------------------------------------------------------------
    print("Auditing historical BGE contexts required by the union of all three frozen Top10 surfaces ...")
    required_contexts = required_bge_history_contexts(
        stage1_rows=stage1_rows,
        val=val,
        history=history,
        progress_every=args.progress_every,
    )
    print(f"Required unique historical BGE contexts: {len(required_contexts)}")

    ensure_cuda_path(args.cuda_path)
    from src.personalisation.pilot_a import BGEContextEmbedder

    local_cache_path = args.output_root / "bge_history_embedding_cache.sqlite3"
    local_cache = VectorCache(local_cache_path)
    seed_cache = VectorCache(args.seed_bge_cache, read_only=True) if args.seed_bge_cache else None

    local_before = local_cache.count()
    reused_from_seed = 0
    missing_contexts: list[str] = []
    for context in sorted(required_contexts):
        if local_cache.get(context) is not None:
            continue
        vector = seed_cache.get(context) if seed_cache is not None else None
        if vector is not None:
            local_cache.put(context, vector)
            reused_from_seed += 1
        else:
            missing_contexts.append(context)
    local_cache.commit()
    if seed_cache is not None:
        seed_rows = seed_cache.count()
        seed_cache.close()
    else:
        seed_rows = 0

    print(f"Local cache rows before: {local_before}")
    print(f"Seed cache rows (read-only): {seed_rows}")
    print(f"Required vectors reused from seed this run: {reused_from_seed}")
    print(f"New historical embeddings required: {len(missing_contexts)}")

    embedder = BGEContextEmbedder(args.bge_model)
    warm_context = next(iter(required_contexts), "测试")
    _ = embedder.embed(warm_context)

    history_embed_ms: list[float] = []
    if missing_contexts:
        print("Embedding missing historical contexts into the versioned local cache ...")
        with NativeStderrSilencer():
            for number, context in enumerate(missing_contexts, 1):
                t0 = time.perf_counter()
                local_cache.put(context, embedder.embed(context))
                history_embed_ms.append((time.perf_counter() - t0) * 1000.0)
                if number % 100 == 0 or number == len(missing_contexts):
                    local_cache.commit()
                if args.progress_every > 0 and (
                    number % args.progress_every == 0 or number == len(missing_contexts)
                ):
                    print(
                        f"BGE history embed {number}/{len(missing_contexts)}  "
                        f"mean={statistics.fmean(history_embed_ms):.3f} ms/context",
                        flush=True,
                    )

    history_vectors: dict[str, Any] = {}
    for context in required_contexts:
        vector = local_cache.get(context)
        if vector is None:
            raise RuntimeError("Local BGE cache fill incomplete")
        history_vectors[context] = normalized_vector(vector)
    local_cache.commit()
    local_cache.close()

    # ------------------------------------------------------------------
    # BGERecency online support, one query embedding shared across 3 bases.
    # ------------------------------------------------------------------
    bge_support_path = args.output_root / "bge_recency_support.jsonl"
    completed = load_completed_bge_support(bge_support_path, stage1)
    pending = [row_id for row_id in sorted(stage1) if row_id not in completed]
    print("\n=== BGERECENCY ONLINE SCORING ===")
    print(f"completed/resumable rows: {len(completed)}")
    print(f"pending rows: {len(pending)}")

    query_embed_ms: list[float] = []
    online_ms: list[float] = []
    scoring_started = time.perf_counter()
    mode = "a" if bge_support_path.is_file() and bge_support_path.stat().st_size else "w"
    with bge_support_path.open(mode, encoding="utf-8", newline="\n") as sink:
        with NativeStderrSilencer():
            for number, row_id in enumerate(pending, 1):
                vrow = val[row_id]
                visible = history.visible_same_pinyin(
                    author=str(vrow["author"]),
                    position=int(vrow["chronological_position"]),
                    pinyin=pinyin_of(vrow),
                )
                qcontext = context_of(vrow)[-BGE_CONTEXT_CHARS:]
                total_started = time.perf_counter()
                embed_started = time.perf_counter()
                query_vector = normalized_vector(embedder.embed(qcontext))
                embed_elapsed = (time.perf_counter() - embed_started) * 1000.0

                bases_out: dict[str, Any] = {}
                for base in BASE_ORDER:
                    candidates = tuple(str(x) for x in stage1[row_id]["bases"][base]["top10"])
                    support, counts, vectors_touched = bge_recency_support(
                        query_vector=query_vector,
                        candidates=candidates,
                        visible=visible,
                        history_vectors=history_vectors,
                    )
                    bases_out[base] = {
                        "candidates": list(candidates),
                        "support": support,
                        "history_counts": counts,
                        "history_vectors_touched": vectors_touched,
                    }

                total_elapsed = (time.perf_counter() - total_started) * 1000.0
                row_out = {
                    "schema_version": 1,
                    "experiment": "initial_recovery_bge_ngram_context_fusion_v2",
                    "row_id": row_id,
                    "author": str(vrow["author"]),
                    "actual_context_chars": len(qcontext),
                    "context_chars": BGE_CONTEXT_CHARS,
                    "history_top_n_per_candidate": BGE_TOP_N,
                    "tau_b": BGE_TAU,
                    "retrieval": "cosine_only",
                    "aggregation": "max(0,cosine)*exp(-age/tau_B)",
                    "bases": bases_out,
                    "query_embed_ms": embed_elapsed,
                    "online_total_ms": total_elapsed,
                    "gold_used_for_scoring": False,
                    "dev3000_used": False,
                    "test_used": False,
                }
                sink.write(json.dumps(row_out, ensure_ascii=False, sort_keys=True) + "\n")
                query_embed_ms.append(embed_elapsed)
                online_ms.append(total_elapsed)

                if args.progress_every > 0 and (number % args.progress_every == 0 or number == len(pending)):
                    sink.flush()
                    wall = time.perf_counter() - scoring_started
                    print(
                        f"BGERecency {number}/{len(pending)} pending  "
                        f"rate={number / wall:.2f} rows/s  "
                        f"embed={statistics.fmean(query_embed_ms):.3f} ms  "
                        f"online={statistics.fmean(online_ms):.3f} ms",
                        flush=True,
                    )

    bge_rows = read_jsonl(bge_support_path)
    if len(bge_rows) != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"Incomplete BGERecency support: {len(bge_rows)}/{EXPECTED_VAL_ROWS}")
    bge = index_rows(bge_rows, "BGERecency support")
    if set(bge) != set(stage1):
        raise RuntimeError("BGERecency support row IDs differ from Stage1")

    bge_scoring_summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_recovery_bge_ngram_context_fusion_v2",
        "rows": len(bge_rows),
        "required_unique_historical_contexts": len(required_contexts),
        "local_history_cache": str(local_cache_path.resolve()),
        "local_history_cache_rows": len(history_vectors),
        "seed_cache_rows": seed_rows,
        "seed_vectors_reused_this_run": reused_from_seed,
        "new_history_embeddings_this_run": len(history_embed_ms),
        "new_history_embedding_latency_ms": latency_summary(history_embed_ms),
        "new_query_embedding_latency_ms_this_run": latency_summary(query_embed_ms),
        "new_online_latency_ms_this_run": latency_summary(online_ms),
        "bge_support_sha256": sha256_file(bge_support_path),
        "gold_used_for_scoring": False,
        "dev3000_used": False,
        "test_used": False,
    }
    write_json(args.output_root / "bge_recency_scoring_summary.json", bge_scoring_summary)

    # ------------------------------------------------------------------
    # Full 2D lambda grid.
    # ------------------------------------------------------------------
    print("\nEvaluating full lambda_N x lambda_B grid ...")
    grid_rows: list[dict[str, Any]] = []
    eval_cache: dict[tuple[str, float, float], list[dict[str, Any]]] = {}
    pred_cache: dict[tuple[str, float, float], list[dict[str, Any]]] = {}

    # Stage1 metrics are read from V1 instead of re-estimated from rounded history.
    stage1_metrics = v1_comparison["stage1_metrics"]
    stage1_recovery = v1_comparison["stage1_recovery"]

    for base in BASE_ORDER:
        base_missing = float(stage1_metrics[base]["missing10"])
        base_rec10 = float(stage1_recovery[base]["recovered_at_10"])
        for lambda_n in lambda_n_grid:
            for lambda_b in lambda_b_grid:
                eval_rows: list[dict[str, Any]] = []
                prediction_rows: list[dict[str, Any]] = []
                for row_id in sorted(stage1):
                    srow = stage1[row_id]
                    base_candidates = srow["bases"][base]["candidates"]
                    n_payload = ngram[row_id]["bases"][base]
                    b_payload = bge[row_id]["bases"][base]
                    candidates = tuple(str(x) for x in srow["bases"][base]["top10"])

                    n_candidates = tuple(str(x) for x in n_payload["candidates"])
                    b_candidates = tuple(str(x) for x in b_payload["candidates"])
                    if candidates != n_candidates or candidates != b_candidates:
                        raise RuntimeError(f"Support candidate-order mismatch at {row_id} {base}")
                    n_support = {str(k): float(v) for k, v in n_payload["support"].items()}
                    b_support = {str(k): float(v) for k, v in b_payload["support"].items()}

                    ranked = rerank_fixed_candidate_set(
                        base_candidates,
                        n_support,
                        b_support,
                        lambda_n=lambda_n,
                        lambda_b=lambda_b,
                    )
                    ranked_set = {candidate_text(x) for x in ranked}
                    if ranked_set != set(candidates):
                        raise RuntimeError(f"Candidate-set invariant failed at {row_id} {base}")
                    if lambda_n == 0.0 and lambda_b == 0.0:
                        actual_order = [candidate_text(x) for x in ranked]
                        if actual_order != list(candidates):
                            raise RuntimeError(f"(0,0) did not reproduce Stage1 order at {row_id} {base}")

                    gold = str(srow["gold"])
                    new_rank = rank_of(ranked, gold)
                    base_rank = srow["bases"][base]["gold_rank"]
                    record = {
                        "row_id": row_id,
                        "author": srow["author"],
                        "generic_missing": bool(srow["generic_missing"]),
                        "gold_in_personal_k5": bool(srow["gold_in_personal_k5"]),
                        "base_rank": base_rank,
                        "rank": new_rank,
                    }
                    eval_rows.append(record)
                    prediction_rows.append(
                        {
                            "schema_version": 1,
                            "row_id": row_id,
                            "author": srow["author"],
                            "gold": gold,
                            "base": base,
                            "lambda_n": lambda_n,
                            "lambda_b": lambda_b,
                            "base_gold_rank": base_rank,
                            "gold_rank": new_rank,
                            "top10": [candidate_text(x) for x in ranked],
                            "candidates": ranked,
                        }
                    )

                method = f"{base}+NG-R+BGE-R|lambda_N={lambda_n:g}|lambda_B={lambda_b:g}"
                metrics = metric_summary(eval_rows, "rank", method)
                recovery = recovery_summary(eval_rows, "rank")
                top1 = transition_counts(eval_rows, "base_rank", "rank")
                rank_trans = rank_transition_counts(eval_rows, "base_rank", "rank")

                if not close_enough(metrics["missing10"], base_missing, tol=1e-12):
                    raise RuntimeError(
                        f"Missing@10 changed under pure reranking for {base} "
                        f"N={lambda_n} B={lambda_b}"
                    )
                if not close_enough(recovery["recovered_at_10"], base_rec10, tol=1e-12):
                    raise RuntimeError(
                        f"Rec@10 changed under pure reranking for {base} "
                        f"N={lambda_n} B={lambda_b}"
                    )

                grid_row = {
                    "base": base,
                    "lambda_n": lambda_n,
                    "lambda_b": lambda_b,
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
                    "top1_rescue": top1["rescue"],
                    "top1_harm": top1["harm"],
                    "top1_net": top1["net"],
                    "rank_improved": rank_trans["improved"],
                    "rank_worsened": rank_trans["worsened"],
                    "rank_same": rank_trans["same"],
                    "rank_net": rank_trans["net"],
                }
                grid_rows.append(grid_row)
                eval_cache[(base, lambda_n, lambda_b)] = eval_rows
                pred_cache[(base, lambda_n, lambda_b)] = prediction_rows

    grid_path = args.output_root / "grid_results.csv"
    write_csv(grid_path, grid_rows)

    # ------------------------------------------------------------------
    # Regression: lambda_B=0 should recover V1 selected NGram-only result.
    # ------------------------------------------------------------------
    print("\nV1 NGram-only regression checks (lambda_B=0):")
    v1_selected = v1_comparison["selected_by_base"]
    for base in BASE_ORDER:
        expected = v1_selected[base]
        expected_lambda_n = float(expected["lambda_n"])
        matches = [
            row for row in grid_rows
            if row["base"] == base
            and float(row["lambda_b"]) == 0.0
            and float(row["lambda_n"]) == expected_lambda_n
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Cannot locate V1 control grid row for {base}")
        actual = matches[0]
        for key in (
            "macro_author_top1", "micro_top1", "top3", "top5", "mrr_at_10", "missing10",
            "recovery_rec1", "recovery_rec3", "recovery_rec5", "recovery_rec10", "recovery_mrr_at_10",
        ):
            expected_key = key
            if key.startswith("recovery_rec"):
                expected_key = key
            if not close_enough(float(actual[key]), float(expected[expected_key]), tol=2e-12):
                raise RuntimeError(
                    f"V1 regression failed {base} {key}: actual={actual[key]} expected={expected[expected_key]}"
                )
        print(
            f"  PASS {base:16s} lambda_N={expected_lambda_n:g}  "
            f"Macro={actual['macro_author_top1']:.6f} MRR={actual['mrr_at_10']:.6f}"
        )

    # ------------------------------------------------------------------
    # Select best full interaction per base.
    # ------------------------------------------------------------------
    selected_rows: list[dict[str, Any]] = []
    selected_predictions: list[dict[str, Any]] = []
    selected_by_base: dict[str, dict[str, Any]] = {}

    for base in BASE_ORDER:
        candidates = [row for row in grid_rows if row["base"] == base]
        best = max(
            candidates,
            key=lambda row: (
                float(row["macro_author_top1"]),
                float(row["mrr_at_10"]),
                -(float(row["lambda_n"]) + float(row["lambda_b"])),
                -float(row["lambda_b"]),
                -float(row["lambda_n"]),
            ),
        )
        selected_rows.append(best)
        selected_by_base[base] = dict(best)
        selected_predictions.extend(
            pred_cache[(base, float(best["lambda_n"]), float(best["lambda_b"]))]
        )

    selected_metrics_path = args.output_root / "selected_metrics.csv"
    write_csv(selected_metrics_path, selected_rows)
    selected_predictions_path = args.output_root / "selected_predictions.jsonl"
    write_jsonl(selected_predictions_path, selected_predictions)

    # ------------------------------------------------------------------
    # Full comparison table: historical controls + Stage1 + NG-only + full.
    # ------------------------------------------------------------------
    full_comparison_rows: list[dict[str, Any]] = [dict(row) for row in HISTORICAL_CONTROLS]
    for base in BASE_ORDER:
        m1 = stage1_metrics[base]
        r1 = stage1_recovery[base]
        full_comparison_rows.append(
            {
                "group": "Recovery only",
                "method": base,
                "macro_author_top1": m1["macro_author_top1"],
                "micro_top1": m1["micro_top1"],
                "top3": m1["top3"],
                "top5": m1["top5"],
                "mrr_at_10": m1["mrr_at_10"],
                "missing10": m1["missing10"],
                "recovery_rec1": r1["recovered_at_1"],
                "recovery_rec3": r1["recovered_at_3"],
                "recovery_rec5": r1["recovered_at_5"],
                "recovery_rec10": r1["recovered_at_10"],
                "recovery_mrr_at_10": r1["recovery_mrr_at_10"],
                "source_precision": "exact_v1_artifact",
            }
        )
        ng = v1_selected[base]
        full_comparison_rows.append(
            {
                "group": "+ NGram Context",
                "method": f"{base} + NGramRecency",
                "macro_author_top1": ng["macro_author_top1"],
                "micro_top1": ng["micro_top1"],
                "top3": ng["top3"],
                "top5": ng["top5"],
                "mrr_at_10": ng["mrr_at_10"],
                "missing10": ng["missing10"],
                "recovery_rec1": ng["recovery_rec1"],
                "recovery_rec3": ng["recovery_rec3"],
                "recovery_rec5": ng["recovery_rec5"],
                "recovery_rec10": ng["recovery_rec10"],
                "recovery_mrr_at_10": ng["recovery_mrr_at_10"],
                "lambda_n": ng["lambda_n"],
                "lambda_b": 0.0,
                "source_precision": "exact_v1_artifact",
            }
        )
        full = selected_by_base[base]
        full_comparison_rows.append(
            {
                "group": "+ NGram + BGE Context",
                "method": f"{base} + NGramRecency + BGERecency",
                **{k: v for k, v in full.items() if k != "base"},
                "source_precision": "exact_v2_artifact",
            }
        )

    full_comparison_path = args.output_root / "full_comparison.csv"
    write_csv(full_comparison_path, full_comparison_rows)

    # ------------------------------------------------------------------
    # Comparison JSON / manifest / checksums.
    # ------------------------------------------------------------------
    comparison = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_recovery_bge_ngram_context_fusion_v2",
        "research_question": (
            "Which Stage-1 recovery philosophy is most compatible with a common "
            "NGramRecency+BGERecency Stage-2 reranker?"
        ),
        "rows": EXPECTED_VAL_ROWS,
        "recovery_bases": {
            "K5+Entropy": "coverage-first",
            "4P+4CS+2E": "balanced",
            "6P+2CS+.25E": "front-rank / Top3-oriented",
        },
        "stage2_formula": "S_final(c)=S_REC(c)+lambda_N*P_NG-R(c)+lambda_B*P_BGE-R(c)",
        "ngram_recency": v1_comparison.get("ngram_recency"),
        "bge_recency": {
            "context_chars": BGE_CONTEXT_CHARS,
            "history_top_n_per_candidate": BGE_TOP_N,
            "retrieval": "candidate-conditioned cosine only",
            "aggregation": "max(0,cosine)*exp(-age/tau_B)",
            "tau_b": BGE_TAU,
            "age_semantics": "age=0 is immediately previous same-author interaction in raw H5000 stream",
        },
        "lambda_n_grid": list(lambda_n_grid),
        "lambda_b_grid": list(lambda_b_grid),
        "selection_rule": (
            "per base: maximize Macro-author Top1; tie MRR@10; then smaller total context weight; "
            "then smaller lambda_B; then smaller lambda_N"
        ),
        "stage1_metrics": stage1_metrics,
        "stage1_recovery": stage1_recovery,
        "v1_ngram_selected_by_base": v1_selected,
        "selected_by_base": selected_by_base,
        "historical_controls_reporting_only": list(HISTORICAL_CONTROLS),
        "invariants": {
            "candidate_set_changed": False,
            "lambda_0_0_must_exactly_reproduce_stage1_order": True,
            "missing10_must_equal_stage1": True,
            "recovery_rec10_must_equal_stage1_on_fixed_R": True,
            "lambda_b_0_reproduces_v1_ngram_selected": True,
        },
        "protocol": {
            "train_fit_rows": EXPECTED_FIT_ROWS,
            "train_val_rows": EXPECTED_VAL_ROWS,
            "generic_missing_n": EXPECTED_GENERIC_MISSING,
            "recoverable_k5_n": EXPECTED_RECOVERABLE_K5,
            "history_budget": HISTORY_BUDGET,
            "gold_used_for_feature_construction": False,
            "gold_used_for_scoring": False,
            "gold_used_for_train_val_selection_and_evaluation_only": True,
            "dev3000_used": False,
            "test_used": False,
        },
        "bge_scoring_summary": bge_scoring_summary,
        "runtime_seconds": time.perf_counter() - started,
    }
    comparison_path = args.output_root / "comparison.json"
    write_json(comparison_path, comparison)

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_recovery_bge_ngram_context_fusion_v2",
        "input_sha256": input_hashes,
        "outputs": {
            "bge_recency_support": str(bge_support_path.resolve()),
            "grid_results": str(grid_path.resolve()),
            "selected_metrics": str(selected_metrics_path.resolve()),
            "selected_predictions": str(selected_predictions_path.resolve()),
            "full_comparison": str(full_comparison_path.resolve()),
            "comparison": str(comparison_path.resolve()),
        },
        "history_semantics": "same author -> strictly prior -> latest H5000 RAW -> exact Pinyin filter",
        "bge_seed_cache_read_only": args.seed_bge_cache is not None,
        "gold_used_for_scoring": False,
        "dev3000_used": False,
        "test_used": False,
    }
    manifest_path = args.output_root / "run_manifest.json"
    write_json(manifest_path, manifest)

    artifact_paths = [
        bge_support_path,
        args.output_root / "bge_recency_scoring_summary.json",
        grid_path,
        selected_metrics_path,
        selected_predictions_path,
        full_comparison_path,
        comparison_path,
        manifest_path,
        args.output_root / "run_setup.json",
    ]
    checksums = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in artifact_paths
    }
    checksums[local_cache_path.name] = {
        "bytes": local_cache_path.stat().st_size,
        "sha256": sha256_file(local_cache_path),
    }
    write_json(args.output_root / "artifact_checksums.json", checksums)

    print("\n=== SELECTED NGRAMRECENCY + BGERECENCY RESULT PER RECOVERY BASE ===")
    for base in BASE_ORDER:
        row = selected_by_base[base]
        print(
            f"{base:16s} lambda_N={float(row['lambda_n']):g} lambda_B={float(row['lambda_b']):g}  "
            f"Macro={float(row['macro_author_top1']):.6f}  "
            f"Micro={float(row['micro_top1']):.6f}  "
            f"Top3={float(row['top3']):.6f}  Top5={float(row['top5']):.6f}  "
            f"MRR={float(row['mrr_at_10']):.6f}  Missing={float(row['missing10']):.6f}  "
            f"Rec1={float(row['recovery_rec1']):.4f}  Rec3={float(row['recovery_rec3']):.4f}  "
            f"Rec5={float(row['recovery_rec5']):.4f}  Rec10={float(row['recovery_rec10']):.4f}  "
            f"RecMRR={float(row['recovery_mrr_at_10']):.4f}  "
            f"Top1Net={int(row['top1_net']):+d}"
        )

    historical_full = next(
        row for row in HISTORICAL_CONTROLS
        if row["method"] == "PV1 + NGramRecency + BGERecency"
    )
    print("\n=== HISTORICAL FULL CONTEXT CONTROL ===")
    print(
        "PV1 + NGramRecency + BGERecency  "
        f"Macro={historical_full['macro_author_top1']:.6f}  "
        f"Micro={historical_full['micro_top1']:.6f}  "
        f"Top3={historical_full['top3']:.6f}  Top5={historical_full['top5']:.6f}  "
        f"MRR={historical_full['mrr_at_10']:.6f}  Missing={historical_full['missing10']:.6f}  "
        f"Rec1={historical_full['recovery_rec1']:.4f}  Rec3={historical_full['recovery_rec3']:.4f}  "
        f"Rec5={historical_full['recovery_rec5']:.4f}  Rec10={historical_full['recovery_rec10']:.4f}  "
        f"RecMRR={historical_full['recovery_mrr_at_10']:.4f}"
    )

    print("\nOutputs:")
    for path in (
        bge_support_path,
        args.output_root / "bge_recency_scoring_summary.json",
        grid_path,
        selected_metrics_path,
        selected_predictions_path,
        full_comparison_path,
        comparison_path,
        manifest_path,
        args.output_root / "artifact_checksums.json",
    ):
        print(f"  {path}")
    print(f"\nRuntime: {time.perf_counter() - started:.1f}s")
    print("Dev3000 used: false")
    print("Test used: false")


if __name__ == "__main__":
    main()
