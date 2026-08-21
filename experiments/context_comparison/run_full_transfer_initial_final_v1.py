"""Zero-shot transfer of the frozen Initial-Pinyin final model to Full+Short Train-Val.

This is a NEW follow-up experiment.  It does not modify or reopen the frozen
seven-system standardized context comparison.

Transferred frozen model
------------------------
Stage 1 (Balanced recovery):
    S_REC(c) = boundary + 4 * P_NG(c) + 4 * CS(c) + 2 * C_entropy(q)
for Personal-only K5 candidates, merged with the frozen Full+Short Frequency
reranking of the Generic Top10.

P_NG is the frozen Initial InterpolatedNGramRecency setting:
    maxN=2, kappa=1, tau=2048.

Stage 2:
    S_FINAL(c) = S_REC(c) + 4 * NG-R(c) + 6 * BGE-R(c)

NG-R:
    candidate-conditioned hard suffix backoff, maxN=2, tau=2048.

BGE-R:
    last 64 Chinese/context Unicode code points; candidate-conditioned same-Pinyin
    history; Top5 by cosine only; recency used only in aggregation;
    tau=2048.

Scientific safeguards
---------------------
- Only standardized Full+Short Clean3 Train-Fit + Train-Val are accepted.
- Exact frozen Train-Fit / Train-Val SHA256 values are enforced.
- Same-author, strictly-prior, rolling H5000 raw history is selected BEFORE
  exact segmented-Pinyin filtering.
- Gold is never used for candidate construction, feature computation, or scoring.
- Initial-Pinyin row-level candidate surfaces/caches are NOT consumed.
- Dev3000 and Test are not accepted as inputs and are never read.
- No transferred parameter is tuned on Full+Short.
- The original seven-system comparison remains method-frozen.

Outputs
-------
<output-root>/
    run_setup.json
    stage1_predictions.jsonl
    bge_context_cache.sqlite3
    final_predictions.jsonl
    result.json
    artifact_checksums.json

The BGE context cache is resumable.  Re-running is allowed only if run_setup.json
matches exactly.
"""

from __future__ import annotations

import argparse
import bisect
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

import numpy as np

# Allow direct execution from experiments/context_comparison without requiring PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.personalisation.context_memory import (
    rank_frequency,
    rank_from_retrieved,
    rank_of as context_rank_of,
    subset_membership,
)
from src.personalisation.standardized_reranking import candidates_of, query_of, read_jsonl
from src.reference_backend_pinyingpt import PinyinGPTConcatBackend


# ---------------------------------------------------------------------------
# Frozen Full+Short population + transferred Initial-final configuration
# ---------------------------------------------------------------------------

EXPECTED_FIT_SHA256 = "547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6"
EXPECTED_VAL_SHA256 = "d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220"
EXPECTED_BGE_SHA256 = "5a88d266870fbd27c6f329df60de80e2d4cf3bbd5e6f080bd5c1b2e5abb12039"
EXPECTED_FIT_ROWS = 144_526
EXPECTED_VAL_ROWS = 34_416
HISTORY_BUDGET = 5000

# Full standardized Frequency comparator
FREQUENCY_LAMBDA = 4.0

# Initial frozen Stage-1 Balanced recovery
PERSONAL_K = 5
P_NG_MAX_N = 2
P_NG_KAPPA = 1.0
P_NG_TAU = 2048.0
RECOVERY_P_NG_WEIGHT = 4.0
RECOVERY_CHOICE_SHARE_WEIGHT = 4.0
RECOVERY_ENTROPY_WEIGHT = 2.0

# Initial frozen Stage-2 context
NGRAM_MAX_N = 2
NGRAM_TAU = 2048.0
LAMBDA_N = 4.0
BGE_CONTEXT_CHARS = 64
BGE_TOP_N = 5
BGE_TAU = 2048.0
LAMBDA_B = 6.0

# Current standardized M1 selection, used only if --standardized-stage1 is provided
M1_TOP_N = 5
M1_LAMBDA = 4.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()




def ensure_cuda_path(explicit: Path | None) -> None:
    """Match the working Windows BGE/GGUF CUDA environment used by Initial V2."""
    if os.name != "nt":
        return
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
    raise RuntimeError("No valid CUDA_PATH found. Pass --cuda-path to the installed CUDA root.")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def index_rows(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in rows:
        row = dict(value)
        row_id = str(row["row_id"])
        if row_id in result:
            raise RuntimeError(f"Duplicate row_id in {label}: {row_id}")
        result[row_id] = row
    return result


def target_of(row: Mapping[str, Any]) -> str:
    value = row.get("target", row.get("gold"))
    if value is None:
        raise RuntimeError(f"No target/gold on row {row.get('row_id')}")
    return str(value)


def pinyin_of(row: Mapping[str, Any]) -> tuple[str, ...]:
    values = row.get("pinyin_segments")
    if not isinstance(values, list):
        raise RuntimeError(f"Missing pinyin_segments on row {row.get('row_id')}")
    return tuple(str(value) for value in values)


def context_of(row: Mapping[str, Any]) -> str:
    return str(row.get("context", row.get("model_used_context", "")))


def scoring_context(val_row: Mapping[str, Any], generic_row: Mapping[str, Any]) -> str:
    # The Initial adaptive P_NG scorer used the actual Generic model-used context.
    # Prefer that exact field when the frozen Generic artifact exposes it; otherwise
    # standardized Full+Short manifest context is the authoritative query context.
    if generic_row.get("model_used_context") is not None:
        return str(generic_row["model_used_context"])
    if generic_row.get("context") is not None:
        return str(generic_row["context"])
    return context_of(val_row)


def candidate_text(item: Mapping[str, Any]) -> str:
    value = item.get("candidate", item.get("text", item.get("target")))
    if value is None:
        raise RuntimeError(f"Cannot identify candidate text: {item}")
    return str(value)


def rank_of(ranking: Sequence[Mapping[str, Any]], gold: str) -> int | None:
    for index, item in enumerate(ranking, start=1):
        if candidate_text(item) == gold:
            return int(item.get("rank", index))
    return None


def normalize(values: Sequence[float], *, uniform_if_zero: bool = True) -> list[float]:
    clipped = [max(0.0, float(value)) for value in values]
    total = sum(clipped)
    if total <= 0.0:
        if uniform_if_zero and clipped:
            return [1.0 / len(clipped)] * len(clipped)
        return [0.0] * len(clipped)
    return [value / total for value in clipped]


def normalized_vector(vector: Any) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm <= 0.0:
        return value
    return value / norm


# ---------------------------------------------------------------------------
# Exact rolling H5000-before-Pinyin history, including raw-stream age
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
    age: int  # age=0 means immediately previous same-author raw interaction


class CausalHistoryIndex:
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        grouped: dict[str, list[HistoryRecord]] = defaultdict(list)
        for row in rows:
            grouped[str(row["author"])].append(
                HistoryRecord(
                    row_id=str(row["row_id"]),
                    author=str(row["author"]),
                    position=int(row["chronological_position"]),
                    pinyin=pinyin_of(row),
                    target=target_of(row),
                    context=context_of(row),
                )
            )
        self.records: dict[str, tuple[HistoryRecord, ...]] = {}
        self.positions: dict[str, tuple[int, ...]] = {}
        for author, records in grouped.items():
            ordered = tuple(sorted(records, key=lambda row: (row.position, row.row_id)))
            self.records[author] = ordered
            self.positions[author] = tuple(row.position for row in ordered)

    def visible_same_pinyin(
        self, *, author: str, position: int, pinyin: tuple[str, ...]
    ) -> tuple[VisibleHistory, ...]:
        records = self.records.get(author, ())
        positions = self.positions.get(author, ())
        stop = bisect.bisect_left(positions, int(position))
        start = max(0, stop - HISTORY_BUDGET)
        result: list[VisibleHistory] = []
        for ordinal in range(start, stop):
            record = records[ordinal]
            if record.pinyin != pinyin:
                continue
            result.append(VisibleHistory(record=record, age=stop - 1 - ordinal))
        return tuple(result)

    def raw_visible_count(self, *, author: str, position: int) -> int:
        positions = self.positions.get(author, ())
        stop = bisect.bisect_left(positions, int(position))
        return min(stop, HISTORY_BUDGET)


# ---------------------------------------------------------------------------
# Frozen Personal-K5 construction
# ---------------------------------------------------------------------------


def compatible_personal_candidate(backend: Any, target: str, pinyin: tuple[str, ...]) -> bool:
    chars = list(target)
    if len(chars) != len(pinyin):
        return False
    token_ids = backend.tokenizer.convert_tokens_to_ids(chars)
    for token_id, segment in zip(token_ids, pinyin):
        if token_id == backend.tokenizer.unk_token_id:
            return False
        if token_id not in backend.allowed_token_ids.get(segment, ()):
            return False
    return True


def build_personal_k5(
    *,
    visible: Sequence[VisibleHistory],
    generic_texts: set[str],
    pinyin: tuple[str, ...],
    backend: Any,
) -> tuple[str, ...]:
    counts = Counter(item.record.target for item in visible)
    ranked = sorted(counts, key=lambda target: (-counts[target], target))
    personal_only = [target for target in ranked if target not in generic_texts]
    compatible = [
        target for target in personal_only
        if compatible_personal_candidate(backend, target, pinyin)
    ]
    return tuple(compatible[:PERSONAL_K])


# ---------------------------------------------------------------------------
# Initial frozen Stage-1 signals
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


def suffix_matches(a: str, b: str, n: int) -> bool:
    if n <= 0:
        return True
    return len(a) >= n and len(b) >= n and a[-n:] == b[-n:]


def recency_weight(age: int, tau: float) -> float:
    return math.exp(-float(age) / float(tau))


def interpolated_ngram_recency(
    *,
    candidates: Sequence[str],
    query_context: str,
    visible: Sequence[VisibleHistory],
    max_n: int = P_NG_MAX_N,
    kappa: float = P_NG_KAPPA,
    tau: float = P_NG_TAU,
) -> dict[str, float]:
    if not candidates:
        return {}
    base_raw = {candidate: 0.0 for candidate in candidates}
    for item in visible:
        if item.record.target in base_raw:
            base_raw[item.record.target] += recency_weight(item.age, tau)
    p = normalize([base_raw[candidate] for candidate in candidates])
    for order in range(1, max_n + 1):
        raw = {candidate: 0.0 for candidate in candidates}
        mass = 0.0
        for item in visible:
            target = item.record.target
            if target not in raw or not suffix_matches(query_context, item.record.context, order):
                continue
            weight = recency_weight(item.age, tau)
            raw[target] += weight
            mass += weight
        if mass > 0.0:
            p_hat = normalize([raw[candidate] for candidate in candidates])
            layer_lambda = mass / (mass + kappa)
            p = [
                layer_lambda * current + (1.0 - layer_lambda) * prior
                for current, prior in zip(p_hat, p)
            ]
            p = normalize(p)
    return dict(zip(candidates, p))


def ngram_rank_map(personal_k5: Sequence[str], p_ng: Mapping[str, float]) -> dict[str, int]:
    order = sorted(
        range(len(personal_k5)),
        key=lambda index: (-float(p_ng[personal_k5[index]]), index, str(personal_k5[index])),
    )
    return {personal_k5[index]: rank for rank, index in enumerate(order, start=1)}


def frequency_rows(
    *, query: Any, generic_candidates: Sequence[Any], history_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    ranked = rank_frequency(
        query,
        generic_candidates,
        history_rows,
        lambda_frequency=FREQUENCY_LAMBDA,
    )
    rows = [dict(value) for value in ranked]
    for index, row in enumerate(rows, start=1):
        row["candidate"] = candidate_text(row)
        row["source"] = "generic_frequency"
        row["rank"] = index
        if row.get("normalized_generic_score") is None:
            raise RuntimeError("Frequency row lacks normalized_generic_score")
        if row.get("final_score") is None:
            raise RuntimeError("Frequency row lacks final_score")
    return rows


def merge_balanced_stage1(
    *,
    generic_rows: Sequence[Mapping[str, Any]],
    personal_k5: Sequence[str],
    p_ng: Mapping[str, float],
    choice_share: Mapping[str, float],
    entropy: float,
) -> list[dict[str, Any]]:
    # The frozen Full Generic artifact can contain shape-safe rows with zero
    # candidates.  The transferred Initial recovery formula anchors personal
    # injection at min(normalized Generic score), which is undefined for an
    # empty Generic surface.  To avoid inventing a Full-specific boundary,
    # preserve those rows as an empty/no-op surface.
    if not generic_rows:
        return []
    generic_texts = {candidate_text(row) for row in generic_rows}
    if generic_texts.intersection(personal_k5):
        raise RuntimeError("Personal K5 overlaps Generic surface")
    boundary = min(float(row["normalized_generic_score"]) for row in generic_rows)
    tiebreak = ngram_rank_map(personal_k5, p_ng)
    rows: list[dict[str, Any]] = [dict(row) for row in generic_rows]
    for original_rank, candidate in enumerate(personal_k5, start=1):
        score = (
            boundary
            + RECOVERY_P_NG_WEIGHT * float(p_ng[candidate])
            + RECOVERY_CHOICE_SHARE_WEIGHT * float(choice_share[candidate])
            + RECOVERY_ENTROPY_WEIGHT * float(entropy)
        )
        rows.append(
            {
                "candidate": candidate,
                "source": "personal_recovery",
                "generic_rank": None,
                "personal_candidate_rank": int(tiebreak[candidate]),
                "original_personal_frequency_rank": original_rank,
                "ngram_rank": int(tiebreak[candidate]),
                "final_score": score,
                "base_tiebreak_rank": int(tiebreak[candidate]),
                "p_ng": float(p_ng[candidate]),
                "choice_share": float(choice_share[candidate]),
                "entropy_concentration": float(entropy),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["final_score"]),
            0 if row["source"] == "generic_frequency" else 1,
            int(row.get("generic_rank") or row.get("personal_candidate_rank") or row.get("rank") or 0),
            str(row["candidate"]),
        )
    )
    rows = rows[:10]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["base_rank"] = rank
        row["base_score"] = float(row["final_score"])
    return rows


# ---------------------------------------------------------------------------
# Initial frozen Stage-2 NGramRecency
# ---------------------------------------------------------------------------


def ngram_recency_support(
    *, query_context: str, candidates: Sequence[str], visible: Sequence[VisibleHistory]
) -> tuple[dict[str, float], int, int]:
    candidate_set = set(candidates)
    candidate_history = [item for item in visible if item.record.target in candidate_set]
    effective_n = 0
    matched: list[VisibleHistory] = candidate_history
    for order in range(min(NGRAM_MAX_N, len(query_context)), 0, -1):
        current = [
            item for item in candidate_history
            if suffix_matches(query_context, item.record.context, order)
        ]
        if current:
            effective_n = order
            matched = current
            break
    raw = {candidate: 0.0 for candidate in candidates}
    for item in matched:
        raw[item.record.target] += recency_weight(item.age, NGRAM_TAU)
    values = normalize([raw[candidate] for candidate in candidates], uniform_if_zero=True)
    return dict(zip(candidates, values)), effective_n, len(matched)


# ---------------------------------------------------------------------------
# Initial frozen Stage-2 BGERecency + resumable vector cache
# ---------------------------------------------------------------------------


class VectorCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "context TEXT PRIMARY KEY, dim INTEGER NOT NULL, vector BLOB NOT NULL)"
        )
        self.connection.commit()

    def get(self, context: str) -> np.ndarray | None:
        row = self.connection.execute(
            "SELECT dim, vector FROM embeddings WHERE context=?", (context,)
        ).fetchone()
        if row is None:
            return None
        dim, blob = row
        vector = np.frombuffer(blob, dtype=np.float32).copy()
        if vector.size != int(dim):
            raise RuntimeError(f"Corrupt BGE vector cache row: {self.path}")
        return vector

    def put(self, context: str, vector: Any) -> None:
        value = np.asarray(vector, dtype=np.float32).reshape(-1)
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


def bge_recency_support(
    *,
    query_vector: np.ndarray,
    candidates: Sequence[str],
    visible: Sequence[VisibleHistory],
    vectors: Mapping[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, int]]:
    grouped: dict[str, list[VisibleHistory]] = {candidate: [] for candidate in candidates}
    counts = {candidate: 0 for candidate in candidates}
    for item in visible:
        if item.record.target in grouped:
            grouped[item.record.target].append(item)
            counts[item.record.target] += 1
    raw = {candidate: 0.0 for candidate in candidates}
    for candidate in candidates:
        histories = grouped[candidate]
        if not histories:
            continue
        history_vectors: list[np.ndarray] = []
        for item in histories:
            key = item.record.context[-BGE_CONTEXT_CHARS:]
            if key not in vectors:
                raise RuntimeError("Required BGE context vector missing from local cache")
            history_vectors.append(vectors[key])
        matrix = np.vstack(history_vectors)
        similarities = matrix @ query_vector
        order = sorted(
            range(len(histories)),
            key=lambda index: (
                -float(similarities[index]),
                int(histories[index].record.position),
                str(histories[index].record.row_id),
            ),
        )[:BGE_TOP_N]
        total = 0.0
        for index in order:
            similarity = max(0.0, float(similarities[index]))
            total += similarity * recency_weight(histories[index].age, BGE_TAU)
        raw[candidate] = total
    support = normalize([raw[candidate] for candidate in candidates], uniform_if_zero=True)
    return dict(zip(candidates, support)), counts


def final_rerank(
    *,
    stage1: Sequence[Mapping[str, Any]],
    ngram_support: Mapping[str, float],
    bge_support: Mapping[str, float],
) -> list[dict[str, Any]]:
    base_texts = [candidate_text(item) for item in stage1]
    if set(base_texts) != set(ngram_support) or set(base_texts) != set(bge_support):
        raise RuntimeError("Stage-2 support candidate set differs from Stage-1 Top10")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(stage1, start=1):
        row = dict(item)
        text = candidate_text(item)
        base_rank = int(item.get("base_rank", item.get("rank", index)))
        base_score = float(item["final_score"])
        n_score = float(ngram_support[text])
        b_score = float(bge_support[text])
        row["base_rank"] = base_rank
        row["base_score"] = base_score
        row["ngram_recency_support"] = n_score
        row["bge_recency_support"] = b_score
        row["lambda_n"] = LAMBDA_N
        row["lambda_b"] = LAMBDA_B
        row["final_score"] = base_score + LAMBDA_N * n_score + LAMBDA_B * b_score
        rows.append(row)
    rows.sort(key=lambda row: (-float(row["final_score"]), int(row["base_rank"]), str(row["candidate"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def metric_summary(rows: Sequence[Mapping[str, Any]], rank_key: str, method: str) -> dict[str, Any]:
    if not rows:
        return {"method": method, "n": 0}
    by_author: dict[str, list[int | None]] = defaultdict(list)
    ranks: list[int | None] = []
    for row in rows:
        rank = row.get(rank_key)
        rank_value = None if rank is None else int(rank)
        ranks.append(rank_value)
        by_author[str(row["author"])].append(rank_value)

    def top_at(values: Sequence[int | None], k: int) -> float:
        return sum(rank is not None and rank <= k for rank in values) / len(values)

    per_author = {author: top_at(values, 1) for author, values in sorted(by_author.items())}
    found = [rank for rank in ranks if rank is not None]
    return {
        "method": method,
        "n": len(rows),
        "macro_author_top1": statistics.fmean(per_author.values()),
        "micro_top1": top_at(ranks, 1),
        "top3": top_at(ranks, 3),
        "top5": top_at(ranks, 5),
        "mrr_at_10": sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / len(ranks),
        "missing10": sum(rank is None for rank in ranks) / len(ranks),
        "mean_rank_given_top10": statistics.fmean(found) if found else None,
        "per_author_top1": per_author,
    }


def transition_counts(rows: Sequence[Mapping[str, Any]], before_key: str, after_key: str) -> dict[str, int]:
    result = {"n": len(rows), "rescue": 0, "harm": 0, "unchanged_correct": 0, "unchanged_wrong": 0}
    for row in rows:
        before = row.get(before_key) == 1
        after = row.get(after_key) == 1
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


def recovery_summary(rows: Sequence[Mapping[str, Any]], rank_key: str) -> dict[str, Any]:
    generic_missing = [row for row in rows if bool(row["generic_missing"])]
    recoverable = [row for row in generic_missing if bool(row["gold_in_personal_k5"])]
    ranks = [None if row.get(rank_key) is None else int(row[rank_key]) for row in recoverable]

    def count_at(k: int) -> int:
        return sum(rank is not None and rank <= k for rank in ranks)

    denominator = len(recoverable)
    result = {
        "generic_missing_n": len(generic_missing),
        "recoverable_personal_k5_n": denominator,
        "recoverable_rate_given_generic_missing": denominator / len(generic_missing) if generic_missing else None,
    }
    for k in (1, 3, 5, 10):
        count = count_at(k)
        result[f"recovered_at_{k}_n"] = count
        result[f"recovered_at_{k}"] = count / denominator if denominator else None
    result["recovery_mrr_at_10"] = (
        sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / denominator
        if denominator else None
    )
    return result


def select_subset(rows: Sequence[Mapping[str, Any]], flag: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if bool(row.get(flag))]


# ---------------------------------------------------------------------------
# Input verification / setup
# ---------------------------------------------------------------------------


def verify_inputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    for label, path in (("fit", args.fit), ("val", args.val), ("generic", args.generic), ("checkpoint", args.checkpoint), ("bge_model", args.bge_model)):
        if not path.exists():
            raise FileNotFoundError(f"{label}: {path}")
    fit_sha = sha256_file(args.fit)
    val_sha = sha256_file(args.val)
    if fit_sha != EXPECTED_FIT_SHA256:
        raise RuntimeError(f"Full Train-Fit SHA mismatch: {fit_sha}")
    if val_sha != EXPECTED_VAL_SHA256:
        raise RuntimeError(f"Full Train-Val SHA mismatch: {val_sha}")
    if not args.bge_model.is_file():
        raise RuntimeError("--bge-model must point to the frozen BGE GGUF file")
    bge_sha = sha256_file(args.bge_model)
    if bge_sha != EXPECTED_BGE_SHA256:
        raise RuntimeError(
            "Frozen BGE model SHA mismatch:\n"
            f"expected={EXPECTED_BGE_SHA256}\nactual={bge_sha}"
        )
    fit_rows = read_jsonl(args.fit)
    val_rows = read_jsonl(args.val)
    generic_rows = read_jsonl(args.generic)
    if len(fit_rows) != EXPECTED_FIT_ROWS or len(val_rows) != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"Unexpected Fit/Val rows: {len(fit_rows)}/{len(val_rows)}")
    if any(str(row.get("source_split", "")).lower() == "test" for row in fit_rows + val_rows):
        raise RuntimeError("STOP: Test row detected")
    if any(str(row.get("standardized_partition", "")) != "train_val" for row in val_rows):
        raise RuntimeError("Train-Val standardized partition marker changed")
    val_ids = {str(row["row_id"]) for row in val_rows}
    generic = index_rows(generic_rows, "Generic")
    if set(generic) != val_ids:
        raise RuntimeError(f"Generic Full+Short surface differs: {len(generic)}/{len(val_ids)}")
    for row in generic.values():
        if bool(row.get("used_test", False)):
            raise RuntimeError("Generic artifact reports used_test=true")
        if row.get("beam_size") is not None and int(row["beam_size"]) != 16:
            raise RuntimeError("Generic beam size differs from frozen 16")
        if row.get("top_k") is not None and int(row["top_k"]) != 10:
            raise RuntimeError("Generic top_k differs from frozen 10")
    m1_stage1 = None
    if args.standardized_stage1 is not None:
        if not args.standardized_stage1.is_file():
            raise FileNotFoundError(args.standardized_stage1)
        rows = read_jsonl(args.standardized_stage1)
        if len(rows) != EXPECTED_VAL_ROWS:
            raise RuntimeError("Standardized Stage-1 M1 artifact row count differs")
        m1_stage1 = index_rows(rows, "standardized Stage1")
        if set(m1_stage1) != val_ids:
            raise RuntimeError("Standardized Stage-1 row IDs differ from Train-Val")
    provenance = {
        "fit": {"path": str(args.fit.resolve()), "sha256": fit_sha, "rows": len(fit_rows)},
        "val": {"path": str(args.val.resolve()), "sha256": val_sha, "rows": len(val_rows)},
        "generic": {"path": str(args.generic.resolve()), "sha256": sha256_file(args.generic), "rows": len(generic_rows)},
        "checkpoint": {"path": str(args.checkpoint.resolve())},
        "bge_model": {"path": str(args.bge_model.resolve()), "sha256": bge_sha},
        "standardized_stage1": (
            {"path": str(args.standardized_stage1.resolve()), "sha256": sha256_file(args.standardized_stage1)}
            if args.standardized_stage1 else None
        ),
        "used_dev3000": False,
        "used_test": False,
    }
    return fit_rows, val_rows, generic, {"m1": m1_stage1, "provenance": provenance}


def setup_payload(args: argparse.Namespace, provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "full_transfer_initial_final_v1",
        "status": "setup",
        "scientific_status": "post-Dev follow-up; zero-shot transferred frozen Initial-final configuration; Train-Val descriptive evaluation only",
        "population": "standardized Full+Short Clean3 Train-Val",
        "history": "same author -> strictly prior -> latest H5000 raw -> exact segmented-Pinyin",
        "frequency_lambda": FREQUENCY_LAMBDA,
        "personal_k": PERSONAL_K,
        "stage1": {
            "formula": "boundary + 4*P_NG + 4*ChoiceShare + 2*EntropyConcentration",
            "p_ng": {"type": "InterpolatedNGramRecency", "max_n": P_NG_MAX_N, "kappa": P_NG_KAPPA, "tau": P_NG_TAU},
        },
        "stage2": {
            "ngram_recency": {"max_n": NGRAM_MAX_N, "tau": NGRAM_TAU, "lambda": LAMBDA_N},
            "bge_recency": {"context_chars": BGE_CONTEXT_CHARS, "top_n_per_candidate": BGE_TOP_N, "tau": BGE_TAU, "lambda": LAMBDA_B, "retrieval": "cosine_only", "aggregation": "max(0,cosine)*exp(-age/tau)"},
        },
        "m1_optional_comparator": {"top_n": M1_TOP_N, "lambda": M1_LAMBDA},
        "empty_generic_surface_policy": (
            "conservative no-op: when frozen Generic has zero candidates, the "
            "Initial generic-boundary anchor is undefined, so no Personal-K5 "
            "injection or Stage-2 reranking is performed"
        ),
        "provenance": provenance,
        "gold_used_for_scoring": False,
        "hyperparameter_search": False,
        "used_dev3000": False,
        "used_test": False,
    }


def prepare_output_root(path: Path, setup: Mapping[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    setup_path = path / "run_setup.json"
    existing = list(path.iterdir())
    if not existing:
        write_json(setup_path, setup)
        return
    if not setup_path.is_file():
        raise RuntimeError(f"Refusing non-empty output root without run_setup.json: {path}")
    previous = json.loads(setup_path.read_text(encoding="utf-8"))
    if previous != dict(setup):
        raise RuntimeError("Existing output root belongs to a different setup; use a new versioned output root")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--generic", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Frozen PinyinGPT checkpoint directory, used for Personal-K5 compatibility filtering")
    parser.add_argument("--bge-model", type=Path, required=True, help="Frozen BGE GGUF file")
    parser.add_argument("--standardized-stage1", type=Path, default=None, help="Optional existing context_comparison_v2/stage1/train_val.jsonl for paired M1 comparison")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--compatibility-device", default="cpu")
    parser.add_argument("--cuda-path", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    started = time.perf_counter()
    fit_rows, val_rows, generic, extra = verify_inputs(args)
    setup = setup_payload(args, extra["provenance"])
    prepare_output_root(args.output_root, setup)
    val = index_rows(val_rows, "Train-Val")
    history = CausalHistoryIndex([*fit_rows, *val_rows])
    m1_state = extra["m1"]

    empty_generic_ids = [row_id for row_id, row in generic.items() if not candidates_of(row)]

    print("=== FULL+SHORT ZERO-SHOT TRANSFER: INITIAL FINAL V2 ===", flush=True)
    print("Transferred point: 4P+4CS+2E + NG-R4 + BGE-R6", flush=True)
    print("No Full hyperparameter tuning. Dev3000/Test not read.", flush=True)
    print(
        f"Frozen Generic empty candidate surfaces: {len(empty_generic_ids)}"
        + (f"; first IDs={empty_generic_ids[:5]}" if empty_generic_ids else ""),
        flush=True,
    )

    # ------------------------------------------------------------------
    # Build Stage 1 exactly from Full causal history.
    # ------------------------------------------------------------------
    stage1_path = args.output_root / "stage1_predictions.jsonl"
    if stage1_path.is_file():
        stage1_rows = read_jsonl(stage1_path)
        if len(stage1_rows) != EXPECTED_VAL_ROWS:
            raise RuntimeError("Existing Stage-1 artifact is incomplete")
        stage1 = index_rows(stage1_rows, "Stage1")
        if set(stage1) != set(val):
            raise RuntimeError("Existing Stage-1 row IDs differ")
        print(f"Reusing completed Stage-1: {len(stage1_rows)} rows", flush=True)
    else:
        print("Loading PinyinGPT compatibility backend ...", flush=True)
        backend = PinyinGPTConcatBackend(args.checkpoint, device=args.compatibility_device)
        stage1_rows: list[dict[str, Any]] = []
        stage1_started = time.perf_counter()
        for number, val_row in enumerate(val_rows, start=1):
            row_id = str(val_row["row_id"])
            grow = generic[row_id]
            query = query_of(val_row)
            generic_candidates = candidates_of(grow)
            visible = history.visible_same_pinyin(
                author=str(val_row["author"]),
                position=int(val_row["chronological_position"]),
                pinyin=pinyin_of(val_row),
            )
            history_rows = [
                {
                    "row_id": item.record.row_id,
                    "author": item.record.author,
                    "chronological_position": item.record.position,
                    "pinyin_segments": list(item.record.pinyin),
                    "target": item.record.target,
                    "context": item.record.context,
                }
                for item in visible
            ]
            flags = subset_membership(query, target_of(val_row), history_rows)
            for flag in ("ambiguous", "conflict"):
                if val_row.get(flag) is not None and bool(val_row[flag]) != bool(flags[flag]):
                    raise RuntimeError(f"Fresh {flag} differs from standardized manifest at {row_id}")
            f_rows = frequency_rows(query=query, generic_candidates=generic_candidates, history_rows=history_rows)
            g_texts = {candidate.text for candidate in generic_candidates}
            personal_k5 = build_personal_k5(
                visible=visible,
                generic_texts=g_texts,
                pinyin=pinyin_of(val_row),
                backend=backend,
            )
            counts = Counter(item.record.target for item in visible)
            total = sum(counts.values())
            choice_share = {
                candidate: (counts.get(candidate, 0) / total if total else 0.0)
                for candidate in personal_k5
            }
            p_ng = interpolated_ngram_recency(
                candidates=personal_k5,
                query_context=scoring_context(val_row, grow),
                visible=visible,
            )
            entropy = entropy_concentration(counts)
            s1 = merge_balanced_stage1(
                generic_rows=f_rows,
                personal_k5=personal_k5,
                p_ng=p_ng,
                choice_share=choice_share,
                entropy=entropy,
            )
            gold = target_of(val_row)
            generic_rank = grow.get("gold_rank")
            if generic_rank is None:
                generic_rank = context_rank_of(
                    tuple({"candidate": candidate.text, "rank": candidate.generic_rank} for candidate in generic_candidates),
                    gold,
                )
            frequency_rank = rank_of(f_rows, gold)
            m1_rank = None
            if m1_state is not None and generic_candidates:
                retrieved = m1_state[row_id]["bge_top20"][:M1_TOP_N]
                m1_rank = context_rank_of(rank_from_retrieved(generic_candidates, retrieved, lambda_memory=M1_LAMBDA), gold)
            stage1_rows.append(
                {
                    "schema_version": 1,
                    "row_id": row_id,
                    "author": str(val_row["author"]),
                    "gold": gold,
                    "ambiguous": bool(flags["ambiguous"]),
                    "conflict": bool(flags["conflict"]),
                    "raw_history_count": history.raw_visible_count(author=str(val_row["author"]), position=int(val_row["chronological_position"])),
                    "same_pinyin_history_count": len(visible),
                    "generic_missing": generic_rank is None,
                    "generic_surface_empty": not bool(generic_candidates),
                    "gold_in_personal_k5": gold in set(personal_k5),
                    "personal_k5": list(personal_k5),
                    "choice_share": choice_share,
                    "p_ng": p_ng,
                    "entropy_concentration": entropy,
                    "Generic_rank": generic_rank,
                    "Frequency_rank": frequency_rank,
                    "M1_rank": m1_rank,
                    "Stage1_rank": rank_of(s1, gold),
                    "Frequency_top10": [candidate_text(item) for item in f_rows],
                    "Stage1_top10": [candidate_text(item) for item in s1],
                    "stage1_candidates": s1,
                    "gold_used_for_scoring": False,
                    "used_dev3000": False,
                    "used_test": False,
                }
            )
            if args.progress_every > 0 and (number % args.progress_every == 0 or number == len(val_rows)):
                elapsed = time.perf_counter() - stage1_started
                print(f"Stage1 {number}/{len(val_rows)} rate={number/max(elapsed,1e-9):.1f}/s", flush=True)
        write_jsonl(stage1_path, stage1_rows)
        stage1 = index_rows(stage1_rows, "Stage1")

    # ------------------------------------------------------------------
    # Determine all exact last64 contexts required by BGE-R.
    # ------------------------------------------------------------------
    required_contexts: set[str] = set()
    for number, val_row in enumerate(val_rows, start=1):
        row_id = str(val_row["row_id"])
        candidates = set(str(value) for value in stage1[row_id]["Stage1_top10"])
        if not candidates:
            continue
        required_contexts.add(context_of(val_row)[-BGE_CONTEXT_CHARS:])
        visible = history.visible_same_pinyin(
            author=str(val_row["author"]),
            position=int(val_row["chronological_position"]),
            pinyin=pinyin_of(val_row),
        )
        for item in visible:
            if item.record.target in candidates:
                required_contexts.add(item.record.context[-BGE_CONTEXT_CHARS:])
        if args.progress_every > 0 and (number % (args.progress_every * 5) == 0 or number == len(val_rows)):
            print(f"BGE context audit {number}/{len(val_rows)} unique={len(required_contexts)}", flush=True)

    # ------------------------------------------------------------------
    # Fill local BGE context cache.  This does not touch existing M1 caches.
    # ------------------------------------------------------------------
    ensure_cuda_path(args.cuda_path)
    from src.personalisation.pilot_a import BGEContextEmbedder

    cache_path = args.output_root / "bge_context_cache.sqlite3"
    cache = VectorCache(cache_path)
    missing = [context for context in sorted(required_contexts) if cache.get(context) is None]
    print(f"BGE required unique contexts: {len(required_contexts)}", flush=True)
    print(f"BGE cache rows before: {cache.count()}; missing: {len(missing)}", flush=True)
    if missing:
        embedder = BGEContextEmbedder(args.bge_model)
        _ = embedder.embed(missing[0] if missing else "测试")
        embed_started = time.perf_counter()
        for number, context in enumerate(missing, start=1):
            cache.put(context, embedder.embed(context))
            if number % 100 == 0 or number == len(missing):
                cache.commit()
            if args.progress_every > 0 and (number % args.progress_every == 0 or number == len(missing)):
                elapsed = time.perf_counter() - embed_started
                print(f"BGE embed {number}/{len(missing)} rate={number/max(elapsed,1e-9):.2f}/s", flush=True)
    vectors: dict[str, np.ndarray] = {}
    for context in required_contexts:
        value = cache.get(context)
        if value is None:
            cache.close()
            raise RuntimeError("BGE local cache fill incomplete")
        vectors[context] = normalized_vector(value)
    cache_rows = cache.count()
    cache.close()

    # ------------------------------------------------------------------
    # Frozen Stage 2, one fixed point only.
    # ------------------------------------------------------------------
    final_rows: list[dict[str, Any]] = []
    final_started = time.perf_counter()
    for number, val_row in enumerate(val_rows, start=1):
        row_id = str(val_row["row_id"])
        srow = stage1[row_id]
        s1 = srow["stage1_candidates"]
        candidates = [candidate_text(item) for item in s1]
        visible = history.visible_same_pinyin(
            author=str(val_row["author"]),
            position=int(val_row["chronological_position"]),
            pinyin=pinyin_of(val_row),
        )
        if candidates:
            ng_support, effective_n, ng_matched = ngram_recency_support(
                query_context=context_of(val_row),
                candidates=candidates,
                visible=visible,
            )
            q_context = context_of(val_row)[-BGE_CONTEXT_CHARS:]
            q_vector = vectors[q_context]
            b_support, b_counts = bge_recency_support(
                query_vector=q_vector,
                candidates=candidates,
                visible=visible,
                vectors=vectors,
            )
            final = final_rerank(stage1=s1, ngram_support=ng_support, bge_support=b_support)
        else:
            ng_support, effective_n, ng_matched = {}, 0, 0
            b_support, b_counts = {}, {}
            final = []
        gold = target_of(val_row)
        final_rows.append(
            {
                "schema_version": 1,
                "row_id": row_id,
                "author": srow["author"],
                "gold": gold,
                "ambiguous": srow["ambiguous"],
                "conflict": srow["conflict"],
                "generic_missing": srow["generic_missing"],
                "generic_surface_empty": bool(srow.get("generic_surface_empty", False)),
                "gold_in_personal_k5": srow["gold_in_personal_k5"],
                "personal_k5": srow["personal_k5"],
                "Generic_rank": srow["Generic_rank"],
                "Frequency_rank": srow["Frequency_rank"],
                "M1_rank": srow["M1_rank"],
                "Stage1_rank": srow["Stage1_rank"],
                "Final_rank": rank_of(final, gold),
                "Stage1_top10": srow["Stage1_top10"],
                "Final_top10": [candidate_text(item) for item in final],
                "ngram_effective_n": effective_n,
                "ngram_matched_history_rows": ng_matched,
                "ngram_recency_support": ng_support,
                "bge_history_counts": b_counts,
                "bge_recency_support": b_support,
                "final_candidates": final,
                "gold_used_for_scoring": False,
                "used_dev3000": False,
                "used_test": False,
            }
        )
        if args.progress_every > 0 and (number % args.progress_every == 0 or number == len(val_rows)):
            elapsed = time.perf_counter() - final_started
            print(f"Stage2 {number}/{len(val_rows)} rate={number/max(elapsed,1e-9):.1f}/s", flush=True)

    final_path = args.output_root / "final_predictions.jsonl"
    write_jsonl(final_path, final_rows)

    methods = ["Generic", "Frequency", "Stage1", "Final"]
    if m1_state is not None:
        methods.insert(2, "M1")
    metrics: dict[str, Any] = {}
    for method in methods:
        rank_key = f"{method}_rank"
        metrics[method] = {
            "overall": metric_summary(final_rows, rank_key, method),
            "ambiguous": metric_summary(select_subset(final_rows, "ambiguous"), rank_key, method),
            "conflict": metric_summary(select_subset(final_rows, "conflict"), rank_key, method),
        }

    result = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "full_transfer_initial_final_v1",
        "scientific_status": "post-Dev follow-up / zero-shot transferred Initial-final model; Full Train-Val descriptive result only",
        "rows": len(final_rows),
        "metrics": metrics,
        "transitions": {
            "Frequency_to_Stage1": transition_counts(final_rows, "Frequency_rank", "Stage1_rank"),
            "Frequency_to_Final": transition_counts(final_rows, "Frequency_rank", "Final_rank"),
            "Stage1_to_Final": transition_counts(final_rows, "Stage1_rank", "Final_rank"),
            **({"M1_to_Final": transition_counts(final_rows, "M1_rank", "Final_rank")} if m1_state is not None else {}),
        },
        "recovery": {
            "Stage1": recovery_summary(final_rows, "Stage1_rank"),
            "Final": recovery_summary(final_rows, "Final_rank"),
        },
        "candidate_surface": {
            "rows_with_personal_k5": sum(bool(row["personal_k5"]) for row in final_rows),
            "personal_k5_candidates_total": sum(len(row["personal_k5"]) for row in final_rows),
            "empty_generic_surface_n": sum(bool(row.get("generic_surface_empty")) for row in final_rows),
            "empty_generic_surface_policy": setup["empty_generic_surface_policy"],
            "generic_missing_n": sum(bool(row["generic_missing"]) for row in final_rows),
            "generic_missing_and_gold_in_personal_k5_n": sum(bool(row["generic_missing"]) and bool(row["gold_in_personal_k5"]) for row in final_rows),
        },
        "bge_cache": {"path": str(cache_path.resolve()), "required_unique_contexts": len(required_contexts), "rows": cache_rows},
        "transferred_parameters": setup["stage1"] | {"stage2": setup["stage2"]},
        "provenance": setup["provenance"],
        "gold_used_for_scoring": False,
        "hyperparameter_search": False,
        "used_dev3000": False,
        "used_test": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    result_path = args.output_root / "result.json"
    write_json(result_path, result)

    checksums = {
        "run_setup.json": sha256_file(args.output_root / "run_setup.json"),
        "stage1_predictions.jsonl": sha256_file(stage1_path),
        "final_predictions.jsonl": sha256_file(final_path),
        "result.json": sha256_file(result_path),
        "runner": sha256_file(Path(__file__)),
        "used_dev3000": False,
        "used_test": False,
    }
    write_json(args.output_root / "artifact_checksums.json", checksums)

    print("\n=== RESULT ===", flush=True)
    for method in methods:
        value = metrics[method]["overall"]
        print(
            f"{method:10s} Macro={value['macro_author_top1']:.6f} "
            f"Micro={value['micro_top1']:.6f} Top3={value['top3']:.6f} "
            f"MRR={value['mrr_at_10']:.6f} Missing={value['missing10']:.6f}",
            flush=True,
        )
    print("Frequency -> Final:", result["transitions"]["Frequency_to_Final"], flush=True)
    if m1_state is not None:
        print("M1 -> Final:", result["transitions"]["M1_to_Final"], flush=True)
    print("Recovery Final:", result["recovery"]["Final"], flush=True)
    print(f"Result: {result_path}", flush=True)
    print(f"Runner SHA256: {checksums['runner']}", flush=True)


if __name__ == "__main__":
    main()
