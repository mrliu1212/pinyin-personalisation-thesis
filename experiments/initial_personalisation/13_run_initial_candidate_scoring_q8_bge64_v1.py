"""Run the complete Initial candidate-scoring comparison: Q8 vs BGE64.

Scope
-----
This runner is intentionally limited to the *personal candidate scoring* study.
It does NOT change the frozen Generic candidate generation, does NOT use Dev3000,
and does NOT read Test.

Methods
-------
Q8
    Frozen PinyinGPT fixed-candidate exact scoring of every personal K<=5
    candidate using the last 8 Python Unicode code points of the frozen
    Generic artifact's ``model_used_context``.

BGE64
    Pinned BGE context embedding over the last 64 Python Unicode code points.
    Historical context embeddings are precomputed/cached. At query time the
    current context is embedded once, then candidate-specific support is the
    sum of the positive cosine similarities of the Top-5 strictly-prior
    same-user, same-Pinyin history rows for that candidate, normalized across
    the personal K<=5 candidates.

Frequency (F)
    Candidate-specific log(1 + count) support from the same legal history,
    normalized across the personal K<=5 candidates.

Evaluation
----------
After Q8 and BGE64 caches are complete, the CPU-only evaluation compares:
    F, Q8, BGE64, Q8+F, BGE64+F
with alpha in {0, .25, .5, .75, 1}. It reports candidate-only Top1/Top3/MRR@5,
mean Gold rank, macro-author Top1, pure recovered-Gold results, K>=2 results,
formal Conflict / ambiguous-non-conflict diagnostics, disagreement tables, and
measured latency.

Important
---------
- Gold is NOT used for candidate selection, Q8 scoring, BGE64 scoring, history
  retrieval, or cache construction. Gold is used only by the final evaluation.
- H5000 semantics: latest 5,000 strictly-prior same-author interactions are
  selected BEFORE exact segmented-Pinyin filtering.
- Train-Val uses rolling causal history: Train-Fit plus strictly earlier
  Train-Val interactions.
- Existing Q64/full-context caches are never read or modified.
- Every expensive cache is resumable.
"""

from __future__ import annotations

import argparse
import bisect
import gc
import hashlib
import json
import math
import os
import sqlite3
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Frozen provenance / workload
# ---------------------------------------------------------------------------

EXPECTED_SURFACE_SHA256 = (
    "205c0ba01cd0678d7a4341c503fa2e74cf126a70182ff582687025e4946764b2"
)
EXPECTED_GENERIC_SHA256 = (
    "bd0fb4dc304e0b266b90fae6fe3ac65424d2f52b23fedfa881212706ba2c2873"
)
EXPECTED_FIT_SHA256 = (
    "162f5c98daa86cc69947571e6d8f20fc401f0a82cdd3fd6e517eb7be2addbdb4"
)
EXPECTED_VAL_SHA256 = (
    "d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4"
)

EXPECTED_SURFACE_ROWS = 34416
EXPECTED_VAL_ROWS = 34416
EXPECTED_FIT_ROWS = 144526
EXPECTED_ELIGIBLE_ROWS = 30509
EXPECTED_K5_PAIRS = 123738

MAX_K = 5
HISTORY_BUDGET = 5000
Q_CONTEXT_CHARS = 8
BGE_CONTEXT_CHARS = 64
BGE_TOP_N = 5
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
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


def index_rows(
    rows: Iterable[dict[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
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
            raise RuntimeError(
                f"{row.get('row_id')}: no usable personal top-5 field"
            )
        texts: list[str] = []
        for item in values[:MAX_K]:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("candidate") or item.get("target")
                if text is None:
                    raise RuntimeError(
                        f"{row.get('row_id')}: candidate dict has no text field"
                    )
                texts.append(str(text))
            else:
                raise RuntimeError(
                    f"{row.get('row_id')}: unsupported personal candidate: {item!r}"
                )
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


def softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    if not total:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in exps]


def normalize_nonnegative(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    total = sum(max(0.0, float(value)) for value in values)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [max(0.0, float(value)) / total for value in values]


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    return ordered[int((len(ordered) - 1) * fraction)]


def latency_summary(values_ms: Sequence[float]) -> dict[str, Any]:
    values = [float(v) for v in values_ms]
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "queries_per_second_from_mean": (
            1000.0 / statistics.fmean(values) if values and statistics.fmean(values) > 0 else None
        ),
    }


def ensure_cuda_path(explicit: Path | None) -> None:
    """Repair a stale CUDA_PATH before llama_cpp is imported."""
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
    """Temporarily silence C/C++ stderr spam while preserving Python stdout."""

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
# Legal rolling H5000 history
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoryRecord:
    row_id: str
    author: str
    position: int
    pinyin: tuple[str, ...]
    target: str
    context: str


class CausalHistoryIndex:
    """H5000-before-Pinyin, strictly-prior, same-author rolling history."""

    def __init__(self, records: Sequence[HistoryRecord]) -> None:
        grouped: dict[str, list[HistoryRecord]] = defaultdict(list)
        for record in records:
            grouped[record.author].append(record)

        self.records: dict[str, tuple[HistoryRecord, ...]] = {}
        self.positions: dict[str, tuple[int, ...]] = {}
        self.pinyin_records: dict[
            tuple[str, tuple[str, ...]], tuple[HistoryRecord, ...]
        ] = {}
        self.pinyin_ordinals: dict[
            tuple[str, tuple[str, ...]], tuple[int, ...]
        ] = {}

        for author, values in grouped.items():
            ordered = tuple(sorted(values, key=lambda r: (r.position, r.row_id)))
            self.records[author] = ordered
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
    ) -> tuple[HistoryRecord, ...]:
        positions = self.positions.get(author, ())
        stop = bisect.bisect_left(positions, position)
        start = max(0, stop - HISTORY_BUDGET)

        key = (author, pinyin)
        ordinals = self.pinyin_ordinals.get(key, ())
        matching = self.pinyin_records.get(key, ())
        left = bisect.bisect_left(ordinals, start)
        right = bisect.bisect_left(ordinals, stop)
        return matching[left:right]


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


# ---------------------------------------------------------------------------
# Input bundle / audit
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
        raise RuntimeError(
            "Train-Val / candidate surface / Generic row IDs differ: "
            f"val={len(val)} surface={len(surface)} generic={len(generic)}"
        )

    pair_count = sum(len(candidate_texts_top5(row)) for row in surface_rows)
    eligible = sum(bool(candidate_texts_top5(row)) for row in surface_rows)
    if pair_count != EXPECTED_K5_PAIRS:
        raise RuntimeError(
            f"K5 workload changed: expected={EXPECTED_K5_PAIRS} actual={pair_count}"
        )
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


def run_audit(args: argparse.Namespace) -> None:
    inputs = load_and_verify(args)
    count_by_k = Counter(len(candidate_texts_top5(row)) for row in inputs.surface_rows)
    q_lengths = [
        len(str(inputs.generic[row_id]["model_used_context"])[-Q_CONTEXT_CHARS:])
        for row_id in inputs.surface
        if candidate_texts_top5(inputs.surface[row_id])
    ]
    b_lengths = [
        len(str(inputs.generic[row_id]["model_used_context"])[-BGE_CONTEXT_CHARS:])
        for row_id in inputs.surface
        if candidate_texts_top5(inputs.surface[row_id])
    ]
    print("\n=== INITIAL CANDIDATE SCORING Q8 + BGE64 AUDIT ===")
    print(f"Train-Fit rows: {len(inputs.fit_rows)}")
    print(f"Train-Val rows: {len(inputs.val_rows)}")
    print(f"eligible rows: {EXPECTED_ELIGIBLE_ROWS}")
    print(f"K5 candidate pairs: {EXPECTED_K5_PAIRS}")
    print(f"candidate count distribution: {dict(sorted(count_by_k.items()))}")
    print(f"Q8 context chars: min={min(q_lengths)} median={percentile(q_lengths, .5)} max={max(q_lengths)}")
    print(f"BGE64 context chars: min={min(b_lengths)} median={percentile(b_lengths, .5)} max={max(b_lengths)}")
    print("Q context: last 8 Python Unicode code points")
    print("BGE context: last 64 Python Unicode code points")
    print("H5000-before-Pinyin rolling causal history: true")
    print("Gold used for scoring/candidate selection: false")
    print("Dev3000 used: false")
    print("Test used: false")


# ---------------------------------------------------------------------------
# Q8 exact PinyinGPT scoring
# ---------------------------------------------------------------------------


def load_completed_score_rows(
    path: Path,
    surface: Mapping[str, Mapping[str, Any]],
    *,
    expected_context_chars: int,
    method: str,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row["row_id"])
            if row_id in completed:
                raise RuntimeError(f"Duplicate {method} output row: {row_id}")
            if row_id not in surface:
                raise RuntimeError(f"Stale {method} output row: {row_id}")
            if int(row.get("context_chars", -1)) != expected_context_chars:
                raise RuntimeError(f"Wrong context definition in {method}: {row_id}")
            expected_candidates = candidate_texts_top5(surface[row_id])
            actual_candidates = tuple(
                str(item["candidate"]) for item in row.get("scores", [])
            )
            if actual_candidates != expected_candidates:
                raise RuntimeError(
                    f"{method} candidate identity/order mismatch for {row_id}"
                )
            completed[row_id] = row
    return completed


def run_q8(args: argparse.Namespace) -> None:
    inputs = load_and_verify(args)
    from src.reference_backend_pinyingpt.backend import PinyinGPTConcatBackend

    output_dir = args.output_root / "q8"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "q8_k5_exact_scores.jsonl"
    progress_path = output_dir / "progress.json"
    summary_path = output_dir / "summary.json"

    completed = load_completed_score_rows(
        output_path,
        inputs.surface,
        expected_context_chars=Q_CONTEXT_CHARS,
        method="Q8",
    )
    eligible = [
        row_id
        for row_id in sorted(inputs.surface)
        if candidate_texts_top5(inputs.surface[row_id])
    ]
    pending = [row_id for row_id in eligible if row_id not in completed]

    print("\n=== Q8 K5 EXACT SCORING ===")
    print(f"eligible rows: {len(eligible)}")
    print(f"already completed: {len(completed)}")
    print(f"pending: {len(pending)}")
    print(f"pending pairs: {sum(len(candidate_texts_top5(inputs.surface[r])) for r in pending)}")
    print(f"context: last {Q_CONTEXT_CHARS} chars")
    print(f"device: {args.device}")
    print(f"output: {output_path}")

    rows_done = 0
    pairs_done = 0
    score_seconds = 0.0
    run_started = time.perf_counter()

    if pending:
        if args.pinyin_checkpoint is None:
            raise RuntimeError("--pinyin-checkpoint is required for Q8 scoring")
        backend = PinyinGPTConcatBackend(args.pinyin_checkpoint, device=args.device)
        mode = "a" if output_path.is_file() and output_path.stat().st_size else "w"
        with output_path.open(mode, encoding="utf-8", newline="\n") as destination:
            for number, row_id in enumerate(pending, start=1):
                surface_row = inputs.surface[row_id]
                generic_row = inputs.generic[row_id]
                candidates = candidate_texts_top5(surface_row)
                context = str(generic_row["model_used_context"])[-Q_CONTEXT_CHARS:]

                started = time.perf_counter()
                scored = backend.score_candidates(
                    context=context,
                    typed_pinyin=tuple(str(x) for x in surface_row["pinyin_segments"]),
                    candidates=candidates,
                )
                elapsed = time.perf_counter() - started
                score_seconds += elapsed

                by_text = {str(value.text): value for value in scored}
                if set(by_text) != set(candidates):
                    raise RuntimeError(f"Q8 scorer candidate mismatch: {row_id}")

                scores = []
                for personal_rank, candidate in enumerate(candidates, start=1):
                    value = by_text[candidate]
                    scores.append(
                        {
                            "candidate": candidate,
                            "personal_candidate_rank": personal_rank,
                            "fixed_log_probability": float(value.log_probability),
                            "fixed_mean_log_probability": float(value.mean_log_probability),
                        }
                    )

                result = {
                    "schema_version": 1,
                    "experiment": "initial_candidate_scoring_q8_k5_exact_v1",
                    "partition": "standardized_train_val",
                    "row_id": row_id,
                    "author": str(surface_row["author"]),
                    "pinyin_segments": list(surface_row["pinyin_segments"]),
                    "context_chars": Q_CONTEXT_CHARS,
                    "actual_context_chars": len(context),
                    "personal_candidate_count": len(candidates),
                    "row_inference_seconds": elapsed,
                    "scores": scores,
                    "gold_used_for_candidate_selection": False,
                    "gold_used_for_scoring": False,
                    "dev3000_used": False,
                    "test_used": False,
                }
                destination.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                rows_done += 1
                pairs_done += len(candidates)

                if number % args.progress_every == 0 or number == len(pending):
                    destination.flush()
                    wall = time.perf_counter() - run_started
                    print(
                        f"Q8 K5: {number}/{len(pending)} pending rows; "
                        f"pairs={pairs_done}; "
                        f"rate={rows_done / wall:.2f} rows/s; "
                        f"mean_score_call={1000.0 * score_seconds / rows_done:.3f} ms/row",
                        flush=True,
                    )

        del backend
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    final_rows = load_completed_score_rows(
        output_path,
        inputs.surface,
        expected_context_chars=Q_CONTEXT_CHARS,
        method="Q8",
    )
    remaining = len(eligible) - len(final_rows)
    write_json(
        progress_path,
        {
            "status": "complete" if remaining == 0 else "partial",
            "eligible_rows": len(eligible),
            "completed_rows": len(final_rows),
            "remaining_rows": remaining,
            "last_run_rows": rows_done,
            "last_run_pairs": pairs_done,
            "last_run_score_seconds": score_seconds,
            "context_chars": Q_CONTEXT_CHARS,
            "dev3000_used": False,
            "test_used": False,
        },
    )
    if remaining:
        print(f"Q8 partial/resumable: remaining={remaining}")
        return

    total_pairs = sum(len(row["scores"]) for row in final_rows.values())
    if total_pairs != EXPECTED_K5_PAIRS:
        raise RuntimeError(f"Q8 final pair count mismatch: {total_pairs}")
    latencies = [1000.0 * float(row["row_inference_seconds"]) for row in final_rows.values()]
    summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_candidate_scoring_q8_k5_exact_v1",
        "context_chars": Q_CONTEXT_CHARS,
        "eligible_rows": len(final_rows),
        "candidate_scores_total": total_pairs,
        "latency_ms": latency_summary(latencies),
        "score_field_for_evaluation": "fixed_mean_log_probability",
        "candidate_surface_sha256": EXPECTED_SURFACE_SHA256,
        "generic_predictions_sha256": EXPECTED_GENERIC_SHA256,
        "output_sha256": sha256_file(output_path),
        "gold_used_for_scoring": False,
        "dev3000_used": False,
        "test_used": False,
    }
    write_json(summary_path, summary)
    print("\n=== Q8 COMPLETE ===")
    print(json.dumps(summary["latency_ms"], indent=2))


# ---------------------------------------------------------------------------
# BGE64 historical vector cache and online candidate support
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


def build_history_index(inputs: Inputs) -> CausalHistoryIndex:
    records = [to_history_record(row) for row in inputs.fit_rows]
    records.extend(to_history_record(row) for row in inputs.val_rows)
    return CausalHistoryIndex(records)


def query_history(
    inputs: Inputs,
    history_index: CausalHistoryIndex,
    row_id: str,
) -> tuple[HistoryRecord, ...]:
    row = inputs.val[row_id]
    return history_index.visible(
        author=str(row["author"]),
        position=int(row["chronological_position"]),
        pinyin=tuple(str(x) for x in row["pinyin_segments"]),
    )


def required_bge_history_contexts(
    inputs: Inputs,
    history_index: CausalHistoryIndex,
    eligible: Sequence[str],
) -> set[str]:
    contexts: set[str] = set()
    for number, row_id in enumerate(eligible, start=1):
        candidates = set(candidate_texts_top5(inputs.surface[row_id]))
        for record in query_history(inputs, history_index, row_id):
            if record.target in candidates:
                contexts.add(record.context[-BGE_CONTEXT_CHARS:])
        if number % 5000 == 0:
            print(
                f"BGE64 history-context audit: {number}/{len(eligible)} rows; "
                f"unique_required={len(contexts)}",
                flush=True,
            )
    return contexts


def bge_candidate_support(
    *,
    query_vector: Any,
    candidates: Sequence[str],
    visible: Sequence[HistoryRecord],
    history_vectors: Mapping[str, Any],
) -> tuple[list[float], list[int], int]:
    import numpy as np

    grouped: dict[str, list[Any]] = {candidate: [] for candidate in candidates}
    counts = {candidate: 0 for candidate in candidates}
    for record in visible:
        if record.target in grouped:
            counts[record.target] += 1
            key = record.context[-BGE_CONTEXT_CHARS:]
            vector = history_vectors.get(key)
            if vector is None:
                raise KeyError(f"Missing cached BGE64 history vector for context hash={hashlib.sha256(key.encode()).hexdigest()}")
            grouped[record.target].append(vector)

    raw: list[float] = []
    vectors_touched = 0
    for candidate in candidates:
        values = grouped[candidate]
        vectors_touched += len(values)
        if not values:
            raw.append(0.0)
            continue
        matrix = np.vstack(values)
        similarities = matrix @ query_vector
        if similarities.size > BGE_TOP_N:
            top = np.partition(similarities, -BGE_TOP_N)[-BGE_TOP_N:]
        else:
            top = similarities
        raw.append(float(np.maximum(top, 0.0).sum()))

    return normalize_nonnegative(raw), [counts[c] for c in candidates], vectors_touched


def run_bge64(args: argparse.Namespace) -> None:
    inputs = load_and_verify(args)
    if args.bge_model is None:
        raise RuntimeError("--bge-model is required for BGE64 scoring")
    ensure_cuda_path(args.cuda_path)

    from src.personalisation.pilot_a import BGEContextEmbedder

    output_dir = args.output_root / "bge64"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "bge64_k5_context_scores.jsonl"
    progress_path = output_dir / "progress.json"
    summary_path = output_dir / "summary.json"
    vector_cache_path = output_dir / "history_embedding_cache.sqlite3"

    eligible = [
        row_id
        for row_id in sorted(inputs.surface)
        if candidate_texts_top5(inputs.surface[row_id])
    ]

    print("\nBuilding legal H5000 rolling history index ...", flush=True)
    history_index = build_history_index(inputs)
    print("Finding BGE64 history contexts needed by K5 candidates ...", flush=True)
    required_contexts = required_bge_history_contexts(inputs, history_index, eligible)
    print(f"Required unique historical BGE64 contexts: {len(required_contexts)}", flush=True)

    cache = VectorCache(vector_cache_path)
    missing = [context for context in sorted(required_contexts) if cache.get(context) is None]
    print(f"BGE64 vector cache rows before: {cache.count()}")
    print(f"BGE64 historical embeddings missing: {len(missing)}")

    embedder = BGEContextEmbedder(args.bge_model)
    # First call loads the model. Let model-load diagnostics remain visible.
    warm_context = next(iter(required_contexts), "测试")
    _ = embedder.embed(warm_context)

    history_embed_latencies: list[float] = []
    if missing:
        print("Precomputing/caching historical BGE64 embeddings ...", flush=True)
        with NativeStderrSilencer():
            for number, context in enumerate(missing, start=1):
                started = time.perf_counter()
                vector = embedder.embed(context)
                history_embed_latencies.append((time.perf_counter() - started) * 1000.0)
                cache.put(context, vector)
                if number % 100 == 0 or number == len(missing):
                    cache.commit()
                if number % args.progress_every == 0 or number == len(missing):
                    print(
                        f"BGE64 history embed: {number}/{len(missing)}; "
                        f"mean={statistics.fmean(history_embed_latencies):.3f} ms/context",
                        flush=True,
                    )

    history_vectors: dict[str, Any] = {}
    for context in required_contexts:
        vector = cache.get(context)
        if vector is None:
            raise RuntimeError("BGE64 vector cache fill incomplete")
        history_vectors[context] = normalized_vector(vector)
    cache.commit()
    cache.close()

    completed = load_completed_score_rows(
        output_path,
        inputs.surface,
        expected_context_chars=BGE_CONTEXT_CHARS,
        method="BGE64",
    )
    pending = [row_id for row_id in eligible if row_id not in completed]

    print("\n=== BGE64 K5 ONLINE SCORING ===")
    print(f"eligible rows: {len(eligible)}")
    print(f"already completed: {len(completed)}")
    print(f"pending: {len(pending)}")
    print(f"context: last {BGE_CONTEXT_CHARS} chars")
    print(f"history Top-N per candidate: {BGE_TOP_N}")
    print("historical embeddings: cached/offline")

    rows_done = 0
    query_embed_ms: list[float] = []
    lookup_cosine_ms: list[float] = []
    online_total_ms: list[float] = []
    run_started = time.perf_counter()

    mode = "a" if output_path.is_file() and output_path.stat().st_size else "w"
    with output_path.open(mode, encoding="utf-8", newline="\n") as destination:
        # Silence repetitive llama.cpp embedding warnings only during hot loop.
        with NativeStderrSilencer():
            for number, row_id in enumerate(pending, start=1):
                candidates = candidate_texts_top5(inputs.surface[row_id])
                visible = query_history(inputs, history_index, row_id)
                qcontext = str(inputs.generic[row_id]["model_used_context"])[
                    -BGE_CONTEXT_CHARS:
                ]

                online_started = time.perf_counter()
                embed_started = time.perf_counter()
                query_vector = normalized_vector(embedder.embed(qcontext))
                embed_elapsed_ms = (time.perf_counter() - embed_started) * 1000.0

                retrieval_started = time.perf_counter()
                support, counts, vectors_touched = bge_candidate_support(
                    query_vector=query_vector,
                    candidates=candidates,
                    visible=visible,
                    history_vectors=history_vectors,
                )
                retrieval_elapsed_ms = (time.perf_counter() - retrieval_started) * 1000.0
                total_elapsed_ms = (time.perf_counter() - online_started) * 1000.0

                scores = [
                    {
                        "candidate": candidate,
                        "personal_candidate_rank": rank,
                        "bge64_support": float(support[rank - 1]),
                        "frequency_count": int(counts[rank - 1]),
                    }
                    for rank, candidate in enumerate(candidates, start=1)
                ]
                result = {
                    "schema_version": 1,
                    "experiment": "initial_candidate_scoring_bge64_k5_v1",
                    "partition": "standardized_train_val",
                    "row_id": row_id,
                    "author": str(inputs.surface[row_id]["author"]),
                    "pinyin_segments": list(inputs.surface[row_id]["pinyin_segments"]),
                    "context_chars": BGE_CONTEXT_CHARS,
                    "actual_context_chars": len(qcontext),
                    "personal_candidate_count": len(candidates),
                    "visible_same_pinyin_history_count": len(visible),
                    "history_vectors_touched": vectors_touched,
                    "query_embed_ms": embed_elapsed_ms,
                    "lookup_cosine_aggregate_ms": retrieval_elapsed_ms,
                    "online_total_ms": total_elapsed_ms,
                    "scores": scores,
                    "gold_used_for_candidate_selection": False,
                    "gold_used_for_scoring": False,
                    "dev3000_used": False,
                    "test_used": False,
                }
                destination.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")

                rows_done += 1
                query_embed_ms.append(embed_elapsed_ms)
                lookup_cosine_ms.append(retrieval_elapsed_ms)
                online_total_ms.append(total_elapsed_ms)

                if number % args.progress_every == 0 or number == len(pending):
                    destination.flush()
                    wall = time.perf_counter() - run_started
                    print(
                        f"BGE64: {number}/{len(pending)} pending rows; "
                        f"rate={rows_done / wall:.2f} rows/s; "
                        f"embed={statistics.fmean(query_embed_ms):.3f} ms; "
                        f"lookup+cos={statistics.fmean(lookup_cosine_ms):.3f} ms; "
                        f"online={statistics.fmean(online_total_ms):.3f} ms",
                        flush=True,
                    )

    final_rows = load_completed_score_rows(
        output_path,
        inputs.surface,
        expected_context_chars=BGE_CONTEXT_CHARS,
        method="BGE64",
    )
    remaining = len(eligible) - len(final_rows)
    write_json(
        progress_path,
        {
            "status": "complete" if remaining == 0 else "partial",
            "eligible_rows": len(eligible),
            "completed_rows": len(final_rows),
            "remaining_rows": remaining,
            "history_embedding_cache_rows": len(history_vectors),
            "context_chars": BGE_CONTEXT_CHARS,
            "history_top_n": BGE_TOP_N,
            "dev3000_used": False,
            "test_used": False,
        },
    )
    if remaining:
        print(f"BGE64 partial/resumable: remaining={remaining}")
        return

    all_embed = [float(row["query_embed_ms"]) for row in final_rows.values()]
    all_lookup = [float(row["lookup_cosine_aggregate_ms"]) for row in final_rows.values()]
    all_online = [float(row["online_total_ms"]) for row in final_rows.values()]
    summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_candidate_scoring_bge64_k5_v1",
        "context_chars": BGE_CONTEXT_CHARS,
        "history_top_n_per_candidate": BGE_TOP_N,
        "eligible_rows": len(final_rows),
        "candidate_scores_total": sum(len(row["scores"]) for row in final_rows.values()),
        "required_unique_historical_contexts": len(required_contexts),
        "history_embedding_cache": str(vector_cache_path.resolve()),
        "history_embedding_cache_rows": len(history_vectors),
        "new_history_embedding_latency_ms_this_run": latency_summary(history_embed_latencies),
        "online_query_embedding_latency_ms": latency_summary(all_embed),
        "online_lookup_cosine_latency_ms": latency_summary(all_lookup),
        "online_total_latency_ms": latency_summary(all_online),
        "output_sha256": sha256_file(output_path),
        "gold_used_for_scoring": False,
        "dev3000_used": False,
        "test_used": False,
    }
    write_json(summary_path, summary)
    print("\n=== BGE64 COMPLETE ===")
    print(json.dumps(summary["online_total_latency_ms"], indent=2))


# ---------------------------------------------------------------------------
# CPU-only candidate-scoring evaluation
# ---------------------------------------------------------------------------


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


def subset_metrics(records: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    chosen = [record for record in records if record["include"]]
    by_author: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in chosen:
        by_author[str(record["author"])].append(record)

    ranks = [int(record["rank"]) for record in chosen if record["rank"] is not None]
    author_top1 = {
        author: sum(int(r["rank"] == 1) for r in rows) / len(rows)
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


def best_alpha_by_macro(rows: Sequence[Mapping[str, Any]], prefix: str) -> tuple[float, dict[str, Any]]:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for alpha in ALPHA_GRID:
        method = f"{prefix}_alpha_{alpha:g}"
        metrics = subset_metrics(rows, method)
        value = metrics["macro_author_top1"]
        candidates.append((alpha, metrics))
    # populated by caller by swapping rank field; kept for API symmetry
    return candidates[0]


def run_evaluate(args: argparse.Namespace) -> None:
    inputs = load_and_verify(args)
    q8_path = args.output_root / "q8" / "q8_k5_exact_scores.jsonl"
    bge_path = args.output_root / "bge64" / "bge64_k5_context_scores.jsonl"
    if not q8_path.is_file() or not bge_path.is_file():
        raise RuntimeError("Q8 and BGE64 score caches must both exist before evaluation")

    q8 = load_completed_score_rows(
        q8_path, inputs.surface, expected_context_chars=Q_CONTEXT_CHARS, method="Q8"
    )
    bge = load_completed_score_rows(
        bge_path, inputs.surface, expected_context_chars=BGE_CONTEXT_CHARS, method="BGE64"
    )
    if len(q8) != EXPECTED_ELIGIBLE_ROWS or len(bge) != EXPECTED_ELIGIBLE_ROWS:
        raise RuntimeError(
            f"Incomplete score caches: Q8={len(q8)} BGE64={len(bge)} expected={EXPECTED_ELIGIBLE_ROWS}"
        )

    history_index = build_history_index(inputs)
    rows_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    disagreement_q8_f = Counter()
    disagreement_bge_f = Counter()
    disagreement_q8_bge = Counter()

    for row_id in sorted(q8):
        surface_row = inputs.surface[row_id]
        val_row = inputs.val[row_id]
        candidates = candidate_texts_top5(surface_row)
        gold = str(val_row.get("target", val_row.get("gold")))
        author = str(val_row["author"])
        k = len(candidates)
        generic_set = set(generic_candidate_texts(surface_row))
        pure_recovered = gold in candidates and gold not in generic_set

        visible = query_history(inputs, history_index, row_id)
        counts = Counter(record.target for record in visible)
        count_values = [math.log1p(counts.get(candidate, 0)) for candidate in candidates]
        f_dist = normalize_nonnegative(count_values)

        q_by_text = {str(item["candidate"]): item for item in q8[row_id]["scores"]}
        q_values = [float(q_by_text[c]["fixed_mean_log_probability"]) for c in candidates]
        q_dist = softmax(q_values)

        b_by_text = {str(item["candidate"]): item for item in bge[row_id]["scores"]}
        b_dist = [float(b_by_text[c]["bge64_support"]) for c in candidates]
        b_dist = normalize_nonnegative(b_dist)

        f_ranked = rank_candidates(candidates, f_dist)
        q_ranked = rank_candidates(candidates, q_dist)
        b_ranked = rank_candidates(candidates, b_dist)

        # Gold-derived diagnostics only; never used by scorers.
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

        base_common = {
            "row_id": row_id,
            "author": author,
            "gold": gold,
            "k": k,
            "gold_in_personal": gold in candidates,
            "pure_recovered": pure_recovered,
            "formal_conflict": formal_conflict,
            "ambiguous_non_conflict": ambiguous_non_conflict,
        }

        method_rankings: dict[str, list[str]] = {
            "F": f_ranked,
            "Q8": q_ranked,
            "BGE64": b_ranked,
        }
        for alpha in ALPHA_GRID:
            qf = [(1 - alpha) * q + alpha * f for q, f in zip(q_dist, f_dist)]
            bf = [(1 - alpha) * b + alpha * f for b, f in zip(b_dist, f_dist)]
            method_rankings[f"Q8+F@{alpha:g}"] = rank_candidates(candidates, qf)
            method_rankings[f"BGE64+F@{alpha:g}"] = rank_candidates(candidates, bf)

        for method, ranking in method_rankings.items():
            rows_by_method[method].append(
                {
                    **base_common,
                    "include": gold in candidates,
                    "rank": gold_rank(ranking, gold),
                }
            )

        if k >= 2:
            f_winner = f_ranked[0]
            q_winner = q_ranked[0]
            b_winner = b_ranked[0]
            if q_winner == f_winner:
                disagreement_q8_f["same_winner"] += 1
            elif q_winner == gold:
                disagreement_q8_f["disagree_q8_correct"] += 1
            elif f_winner == gold:
                disagreement_q8_f["disagree_frequency_correct"] += 1
            else:
                disagreement_q8_f["disagree_neither_correct"] += 1

            if b_winner == f_winner:
                disagreement_bge_f["same_winner"] += 1
            elif b_winner == gold:
                disagreement_bge_f["disagree_bge64_correct"] += 1
            elif f_winner == gold:
                disagreement_bge_f["disagree_frequency_correct"] += 1
            else:
                disagreement_bge_f["disagree_neither_correct"] += 1

            if q_winner == b_winner:
                disagreement_q8_bge["same_winner"] += 1
            elif q_winner == gold:
                disagreement_q8_bge["disagree_q8_correct"] += 1
            elif b_winner == gold:
                disagreement_q8_bge["disagree_bge64_correct"] += 1
            else:
                disagreement_q8_bge["disagree_neither_correct"] += 1

    def metrics_for_filter(method: str, predicate: Any) -> dict[str, Any]:
        rows = []
        for record in rows_by_method[method]:
            copy = dict(record)
            copy["include"] = bool(record["gold_in_personal"] and predicate(record))
            rows.append(copy)
        return subset_metrics(rows, method)

    all_methods = list(rows_by_method)
    sections: dict[str, dict[str, Any]] = {}
    filters = {
        "gold_in_personal_all": lambda r: True,
        "gold_in_personal_k2plus": lambda r: int(r["k"]) >= 2,
        "pure_recovered_gold": lambda r: bool(r["pure_recovered"]),
        "formal_conflict_gold_in_personal": lambda r: bool(r["formal_conflict"]),
        "ambiguous_non_conflict_gold_in_personal": lambda r: bool(r["ambiguous_non_conflict"]),
    }
    for section_name, predicate in filters.items():
        sections[section_name] = {
            method: metrics_for_filter(method, predicate) for method in all_methods
        }

    # Select fusion alpha by Macro-author Top1 on Gold-in-personal K>=2.
    selected: dict[str, Any] = {}
    for family in ("Q8+F", "BGE64+F"):
        options = []
        for alpha in ALPHA_GRID:
            method = f"{family}@{alpha:g}"
            metrics = sections["gold_in_personal_k2plus"][method]
            options.append((
                float(metrics["macro_author_top1"] or 0.0),
                -alpha,  # lower alpha wins exact ties
                alpha,
                metrics,
            ))
        winner = max(options, key=lambda item: (item[0], item[1]))
        selected[family] = {
            "selected_alpha": winner[2],
            "selection_metric": "Macro-author Top1 on Train-Val Gold-in-personal K>=2",
            "metrics": winner[3],
        }

    q8_summary = json.loads((args.output_root / "q8" / "summary.json").read_text(encoding="utf-8"))
    bge_summary = json.loads((args.output_root / "bge64" / "summary.json").read_text(encoding="utf-8"))

    result = {
        "schema_version": 1,
        "experiment": "initial_candidate_scoring_q8_vs_bge64_v1",
        "status": "complete",
        "partition": "standardized_train_val",
        "candidate_pool": "frozen personal K<=5 only for scorer-quality evaluation",
        "methods": {
            "F": "normalized log(1+count)",
            "Q8": "softmax(fixed_mean_log_probability), last 8 chars",
            "BGE64": "target-conditioned positive cosine Top5, last 64 chars",
            "fusion": "(1-alpha)*contextual + alpha*frequency",
            "alpha_grid": list(ALPHA_GRID),
        },
        "sections": sections,
        "selected_fusions": selected,
        "disagreements_k2plus": {
            "Q8_vs_F": dict(disagreement_q8_f),
            "BGE64_vs_F": dict(disagreement_bge_f),
            "Q8_vs_BGE64": dict(disagreement_q8_bge),
        },
        "latency": {
            "Q8_exact_k5_score_call_ms": q8_summary["latency_ms"],
            "BGE64_online_total_ms": bge_summary["online_total_latency_ms"],
            "BGE64_query_embedding_only_ms": bge_summary["online_query_embedding_latency_ms"],
            "BGE64_lookup_cosine_ms": bge_summary["online_lookup_cosine_latency_ms"],
            "note": "BGE historical embedding construction is offline/cached and excluded from online_total.",
        },
        "gold_use": {
            "candidate_selection": False,
            "Q8_scoring": False,
            "BGE64_scoring": False,
            "evaluation_only": True,
        },
        "dev3000_used": False,
        "test_used": False,
        "provenance": {
            "fit_sha256": EXPECTED_FIT_SHA256,
            "val_sha256": EXPECTED_VAL_SHA256,
            "candidate_surface_sha256": EXPECTED_SURFACE_SHA256,
            "generic_predictions_sha256": EXPECTED_GENERIC_SHA256,
            "q8_scores_sha256": sha256_file(q8_path),
            "bge64_scores_sha256": sha256_file(bge_path),
        },
    }

    comparison_path = args.output_root / "candidate_scoring_comparison.json"
    write_json(comparison_path, result)

    print("\n=== CANDIDATE SCORING COMPARISON COMPLETE ===")
    print(f"output: {comparison_path}")
    print("\nGold-in-personal K>=2:")
    headline_methods = ["F", "Q8", "BGE64", f"Q8+F@{selected['Q8+F']['selected_alpha']:g}", f"BGE64+F@{selected['BGE64+F']['selected_alpha']:g}"]
    for method in headline_methods:
        metrics = sections["gold_in_personal_k2plus"][method]
        print(
            f"  {method:14s} n={metrics['n']:5d} "
            f"MacroTop1={metrics['macro_author_top1']:.6f} "
            f"MicroTop1={metrics['micro_top1']:.6f} "
            f"Top3={metrics['top3']:.6f} MRR={metrics['mrr_at_5']:.6f}"
        )
    print("\nLatency:")
    print(
        f"  Q8 mean={q8_summary['latency_ms']['mean']:.3f} ms; "
        f"p95={q8_summary['latency_ms']['p95']:.3f} ms"
    )
    print(
        f"  BGE64 online mean={bge_summary['online_total_latency_ms']['mean']:.3f} ms; "
        f"p95={bge_summary['online_total_latency_ms']['p95']:.3f} ms"
    )
    print("Gold used for evaluation only: true")
    print("Dev3000 used: false")
    print("Test used: false")


# ---------------------------------------------------------------------------
# All-phase orchestration (separate subprocesses release GPU memory cleanly)
# ---------------------------------------------------------------------------


def child_command(args: argparse.Namespace, phase: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.initial_personalisation.run_initial_candidate_scoring_q8_bge64_v1",
        "--phase",
        phase,
        "--fit",
        str(args.fit),
        "--val",
        str(args.val),
        "--candidate-surface",
        str(args.candidate_surface),
        "--generic-predictions",
        str(args.generic_predictions),
        "--output-root",
        str(args.output_root),
        "--device",
        str(args.device),
        "--progress-every",
        str(args.progress_every),
    ]
    if args.pinyin_checkpoint is not None:
        command += ["--pinyin-checkpoint", str(args.pinyin_checkpoint)]
    if args.bge_model is not None:
        command += ["--bge-model", str(args.bge_model)]
    if args.cuda_path is not None:
        command += ["--cuda-path", str(args.cuda_path)]
    return command


def run_all(args: argparse.Namespace) -> None:
    for phase in ("audit", "q8", "bge64", "evaluate"):
        print("\n" + "=" * 78)
        print(f"MASTER: starting phase {phase}")
        print("=" * 78, flush=True)
        subprocess.run(child_command(args, phase), check=True)
    print("\nALL Q8 + BGE64 CANDIDATE-SCORING TESTS COMPLETE.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("audit", "q8", "bge64", "evaluate", "all"),
        required=True,
    )
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--candidate-surface", type=Path, required=True)
    parser.add_argument("--generic-predictions", type=Path, required=True)
    parser.add_argument("--pinyin-checkpoint", type=Path)
    parser.add_argument("--bge-model", type=Path)
    parser.add_argument("--cuda-path", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.phase == "audit":
        run_audit(args)
    elif args.phase == "q8":
        run_q8(args)
    elif args.phase == "bge64":
        run_bge64(args)
    elif args.phase == "evaluate":
        run_evaluate(args)
    elif args.phase == "all":
        if args.pinyin_checkpoint is None:
            raise RuntimeError("--pinyin-checkpoint is required for --phase all")
        if args.bge_model is None:
            raise RuntimeError("--bge-model is required for --phase all")
        run_all(args)
    else:
        raise AssertionError(args.phase)


if __name__ == "__main__":
    main()
