"""Candidate-aware second-stage scoring for Personalisation M2.

The module contains no dataset labels or experiment names.  Its pair cache is
therefore reusable whenever the current/history/candidate tuple and frozen
reranker provenance are identical.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
from typing import Any, Mapping, Sequence

from src.personalisation.context_memory import Candidate, PredictionQuery, _ranked, normalize_generic_scores


RERANKER_REPOSITORY = "BAAI/bge-reranker-base"
RERANKER_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"
RERANKER_MODEL_SHA256 = "ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd"
RERANKER_TOKENIZER_SHA256 = "9eb652ac4e40cc093272bbbe0f55d521cf67570060227109b5cdc20945a4489e"
RERANKER_LICENSE = "MIT"
INPUT_TEMPLATE_VERSION = "candidate-aware-current-history-v1"
TRUNCATION_VERSION = "paired-recent-context-balanced-v1"
DEFAULT_MAX_LENGTH = 512


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PairIdentity:
    current_id: str
    current_context: str
    pinyin: tuple[str, ...]
    historical_id: str
    historical_context: str
    historical_target: str
    candidate: str

    @classmethod
    def from_query_history(
        cls,
        query: PredictionQuery,
        history: Mapping[str, Any],
        candidate: str,
    ) -> "PairIdentity":
        return cls(
            current_id=query.row_id,
            current_context=query.context,
            pinyin=query.pinyin,
            historical_id=str(history["row_id"]),
            historical_context=str(history["context"]),
            historical_target=str(history["target"]),
            candidate=candidate,
        )


@dataclass(frozen=True)
class PreparedPair:
    query_text: str
    history_text: str
    current_context_truncated: bool
    historical_context_truncated: bool
    input_tokens: int


class CandidateAwareTemplate:
    """Deterministic pair template that preserves all non-context fields."""

    def __init__(self, tokenizer: Any, max_length: int | None = None) -> None:
        self.tokenizer = tokenizer
        declared = int(getattr(tokenizer, "model_max_length", DEFAULT_MAX_LENGTH))
        if declared <= 0 or declared > 100_000:
            declared = DEFAULT_MAX_LENGTH
        self.max_length = int(max_length or declared)

    @staticmethod
    def _query(context: str, pinyin: Sequence[str], candidate: str) -> str:
        return (
            "[CURRENT]\n"
            f"context: {context}\n"
            f"pinyin: {' '.join(pinyin)}\n"
            f"candidate: {candidate}"
        )

    @staticmethod
    def _history(context: str, selected: str) -> str:
        return "[HISTORY]\n" f"context: {context}\n" f"selected: {selected}"

    def _ids(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False, verbose=False))

    def prepare(self, pair: PairIdentity) -> PreparedPair:
        empty_query = self._query("", pair.pinyin, pair.candidate)
        empty_history = self._history("", pair.historical_target)
        mandatory = len(
            self.tokenizer(
                empty_query,
                empty_history,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )
        available = self.max_length - mandatory
        if available < 0:
            raise ValueError("mandatory M2 fields exceed the reranker token limit")
        current_ids = self._ids(pair.current_context)
        history_ids = self._ids(pair.historical_context)
        current_budget = (available + 1) // 2
        history_budget = available // 2
        unused_current = max(0, current_budget - len(current_ids))
        unused_history = max(0, history_budget - len(history_ids))
        current_budget += unused_history
        history_budget += unused_current
        current_kept = current_ids[-current_budget:] if current_budget else []
        history_kept = history_ids[-history_budget:] if history_budget else []
        def render() -> tuple[str, str, int]:
            current_context = self.tokenizer.decode(current_kept, skip_special_tokens=False)
            historical_context = self.tokenizer.decode(history_kept, skip_special_tokens=False)
            query_text = self._query(current_context, pair.pinyin, pair.candidate)
            history_text = self._history(historical_context, pair.historical_target)
            tokens = len(
                self.tokenizer(
                    query_text,
                    history_text,
                    add_special_tokens=True,
                    truncation=False,
                    verbose=False,
                )["input_ids"]
            )
            return query_text, history_text, tokens

        query_text, history_text, tokens = render()
        # SentencePiece decode/encode is not perfectly idempotent for every
        # suffix. Remove additional *oldest* context tokens deterministically
        # until the serialized pair is truly within the model boundary.
        while tokens > self.max_length and (current_kept or history_kept):
            if current_kept and (not history_kept or len(current_kept) >= len(history_kept)):
                current_kept = current_kept[1:]
            else:
                history_kept = history_kept[1:]
            query_text, history_text, tokens = render()
        if tokens > self.max_length:
            raise AssertionError("mandatory M2 fields exceed the reranker token limit after safety trimming")
        return PreparedPair(
            query_text=query_text,
            history_text=history_text,
            current_context_truncated=len(current_kept) < len(current_ids),
            historical_context_truncated=len(history_kept) < len(history_ids),
            input_tokens=tokens,
        )


class PairScoreCache:
    """Provenance-checked, profile-neutral SQLite cache for raw pair logits."""

    def __init__(
        self,
        path: Path,
        *,
        model_revision: str,
        model_sha256: str,
        tokenizer_sha256: str,
        max_length: int,
        dtype: str,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model_revision = model_revision
        self.model_sha256 = model_sha256
        self.tokenizer_sha256 = tokenizer_sha256
        self.max_length = int(max_length)
        self.dtype = dtype
        self.connection = sqlite3.connect(self.path, timeout=60.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS pair_scores ("
            "cache_key TEXT PRIMARY KEY, current_sha256 TEXT NOT NULL, historical_sha256 TEXT NOT NULL, "
            "candidate_sha256 TEXT NOT NULL, raw_score REAL NOT NULL, input_tokens INTEGER NOT NULL, "
            "current_context_truncated INTEGER NOT NULL, historical_context_truncated INTEGER NOT NULL)"
        )
        expected = {
            "repository": RERANKER_REPOSITORY,
            "revision": model_revision,
            "model_sha256": model_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "input_template_version": INPUT_TEMPLATE_VERSION,
            "truncation_version": TRUNCATION_VERSION,
            "max_length": str(self.max_length),
            "dtype": dtype,
            "score": "raw sequence-classification logit",
        }
        existing = dict(self.connection.execute("SELECT key, value FROM metadata"))
        if existing and existing != expected:
            raise RuntimeError("M2 pair-score cache provenance differs from the frozen configuration")
        if not existing:
            self.connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", expected.items())
            self.connection.commit()

    def key(self, pair: PairIdentity) -> str:
        identity = {
            "candidate": pair.candidate,
            "current_context": pair.current_context,
            "current_id": pair.current_id,
            "historical_context": pair.historical_context,
            "historical_id": pair.historical_id,
            "historical_target": pair.historical_target,
            "input_template_version": INPUT_TEMPLATE_VERSION,
            "max_length": self.max_length,
            "model_revision": self.model_revision,
            "model_sha256": self.model_sha256,
            "pinyin": list(pair.pinyin),
            "tokenizer_sha256": self.tokenizer_sha256,
            "truncation_version": TRUNCATION_VERSION,
        }
        return sha256_text(json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")))

    def get(self, pair: PairIdentity) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT raw_score, input_tokens, current_context_truncated, historical_context_truncated "
            "FROM pair_scores WHERE cache_key = ?",
            (self.key(pair),),
        ).fetchone()
        if row is None:
            return None
        return {
            "raw_score": float(row[0]),
            "input_tokens": int(row[1]),
            "current_context_truncated": bool(row[2]),
            "historical_context_truncated": bool(row[3]),
        }

    def put(self, pair: PairIdentity, prepared: PreparedPair, raw_score: float) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO pair_scores VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self.key(pair),
                sha256_text(pair.current_id + "\0" + pair.current_context + "\0" + " ".join(pair.pinyin)),
                sha256_text(pair.historical_id + "\0" + pair.historical_context + "\0" + pair.historical_target),
                sha256_text(pair.candidate),
                float(raw_score),
                prepared.input_tokens,
                int(prepared.current_context_truncated),
                int(prepared.historical_context_truncated),
            ),
        )

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM pair_scores").fetchone()[0])

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


class BGEReranker:
    """Pinned Transformers runtime for the pretrained BGE Cross-Encoder."""

    def __init__(
        self,
        model_path: Path,
        *,
        revision: str,
        model_sha256: str,
        tokenizer_sha256: str,
        batch_size: int = 32,
        max_length: int | None = None,
        device: str = "cuda",
    ) -> None:
        self.model_path = Path(model_path)
        self.revision = revision
        self.model_sha256 = model_sha256
        self.tokenizer_sha256 = tokenizer_sha256
        self.batch_size = int(batch_size)
        self.device = device
        self.requested_max_length = max_length
        self.tokenizer = None
        self.template = None
        self.model = None
        self.load_seconds: float | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the frozen M2 run")
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.template = CandidateAwareTemplate(self.tokenizer, self.requested_max_length)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=dtype,
        ).to(self.device)
        self.model.eval()
        self.load_seconds = time.perf_counter() - started

    @property
    def max_length(self) -> int:
        self.load()
        assert self.template is not None
        return self.template.max_length

    @property
    def dtype_name(self) -> str:
        return "float16" if self.device == "cuda" else "float32"

    def prepare(self, pair: PairIdentity) -> PreparedPair:
        self.load()
        assert self.template is not None
        return self.template.prepare(pair)

    def score_prepared(self, prepared: Sequence[PreparedPair]) -> list[float]:
        if not prepared:
            return []
        import torch

        self.load()
        assert self.tokenizer is not None and self.model is not None
        values: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(prepared), self.batch_size):
                batch = prepared[start : start + self.batch_size]
                inputs = self.tokenizer(
                    [(row.query_text, row.history_text) for row in batch],
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                logits = self.model(**inputs, return_dict=True).logits.view(-1).float().cpu()
                values.extend(float(value) for value in logits)
        return values

    def info(self) -> dict[str, Any]:
        import torch
        import transformers

        self.load()
        return {
            "repository": RERANKER_REPOSITORY,
            "revision": self.revision,
            "model_sha256": self.model_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "model_path": str(self.model_path),
            "license": RERANKER_LICENSE,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "dtype": self.dtype_name,
            "device": self.device,
            "device_name": torch.cuda.get_device_name(0) if self.device == "cuda" else "cpu",
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "load_seconds": self.load_seconds,
            "input_template_version": INPUT_TEMPLATE_VERSION,
            "truncation_version": TRUNCATION_VERSION,
        }


def monotonic_support(raw_score: float) -> float:
    """Map a raw Cross-Encoder logit to finite non-negative support."""

    if raw_score >= 0:
        value = math.exp(-raw_score)
        return 1.0 / (1.0 + value)
    value = math.exp(raw_score)
    return value / (1.0 + value)


def rank_m2(
    candidates: Sequence[Candidate],
    evidence: Sequence[Mapping[str, Any]],
    *,
    lambda_m2: float,
) -> tuple[dict[str, Any], ...]:
    """Normalize sigmoid support within a query and combine with frozen G0 z-scores."""

    support: dict[str, float] = defaultdict(float)
    total = 0.0
    candidate_texts = {candidate.text for candidate in candidates}
    for row in evidence:
        target = str(row["historical_target"])
        value = monotonic_support(float(row["raw_score"]))
        total += value
        if target in candidate_texts:
            support[target] += value
    if total:
        support = {candidate: value / total for candidate, value in support.items()}
    ranked = _ranked(candidates, normalize_generic_scores(candidates), support, lambda_m2)
    for row in ranked:
        row["m2_support"] = row.pop("personal_score")
    return ranked
