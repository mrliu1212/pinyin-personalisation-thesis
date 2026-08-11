"""Frozen semantic same-Pinyin personal-memory retrieval for Phase 4E."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
from typing import Any, Protocol, Sequence

from .semantic_lm import JsonCache, set_deterministic_seeds


SEMANTIC_EMBEDDING_REPO = "Qwen/Qwen3-Embedding-0.6B"
RETRIEVAL_INSTRUCTION = (
    "Given a current Chinese text context, retrieve previous contexts in which "
    "the same Pinyin input was used in a semantically similar way."
)
SEMANTIC_MEMORY_K = 5


def normalize_vector(vector: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return tuple(0.0 for _ in vector)
    return tuple(float(value / norm) for value in vector)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions differ")
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right))))


class EmbeddingBackend(Protocol):
    def encode_query(self, context: str) -> Sequence[float]: ...

    def encode_document(self, context: str) -> Sequence[float]: ...


class SentenceTransformerEmbeddingBackend:
    def __init__(self, model: Any, *, torch_module: Any, device: str) -> None:
        self.model = model
        self.torch = torch_module
        self.device = device
        self.dtype = str(next(model.parameters()).dtype)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls, *, revision: str, device: str
    ) -> "SentenceTransformerEmbeddingBackend":
        import torch
        from sentence_transformers import SentenceTransformer

        set_deterministic_seeds()
        dtype = torch.float32 if device == "cpu" else torch.float16
        model = SentenceTransformer(
            SEMANTIC_EMBEDDING_REPO,
            revision=revision,
            device=device,
            model_kwargs={"dtype": dtype},
        )
        return cls(model, torch_module=torch, device=device)

    def _encode(self, text: str) -> tuple[float, ...]:
        with self.torch.no_grad():
            vector = self.model.encode(
                [text], normalize_embeddings=True, convert_to_numpy=True
            )[0]
        result = tuple(float(value) for value in vector)
        if not all(math.isfinite(value) for value in result):
            raise ValueError("embedding model returned a non-finite vector")
        return result

    def encode_query(self, context: str) -> Sequence[float]:
        text = f"Instruct: {RETRIEVAL_INSTRUCTION}\nQuery: {context}"
        return self._encode(text)

    def encode_document(self, context: str) -> Sequence[float]:
        return self._encode(context)


class CachedEmbeddingModel:
    CONFIGURATION = {
        "normalization": "L2",
        "similarity": "cosine",
        "query_instruction": RETRIEVAL_INSTRUCTION,
        "document_instruction": None,
    }

    def __init__(
        self,
        backend: EmbeddingBackend,
        *,
        revision: str,
        cache_dir: Path = Path(".cache/phase_04e"),
    ) -> None:
        self.backend = backend
        self.revision = revision
        self.cache = JsonCache(cache_dir, "semantic_embeddings")
        self.runtime_configuration = {
            "device": str(getattr(backend, "device", "stub_or_unspecified")),
            "dtype": str(getattr(backend, "dtype", "stub_or_unspecified")),
        }

    def _encode(self, context: str, kind: str) -> tuple[float, ...]:
        key = {
            "model": SEMANTIC_EMBEDDING_REPO,
            "revision": self.revision,
            "kind": kind,
            "context": context,
            "configuration": self.CONFIGURATION,
            "runtime_configuration": self.runtime_configuration,
        }
        cached = self.cache.get(key)
        if cached is not None:
            return tuple(float(value) for value in cached)
        raw = (
            self.backend.encode_query(context)
            if kind == "query"
            else self.backend.encode_document(context)
        )
        vector = normalize_vector(raw)
        self.cache.set(key, list(vector))
        return vector

    def encode_query(self, context: str) -> tuple[float, ...]:
        return self._encode(context, "query")

    def encode_document(self, context: str) -> tuple[float, ...]:
        return self._encode(context, "document")


@dataclass(frozen=True)
class SemanticMemoryInteraction:
    interaction_id: str
    user_id: str
    timestamp: datetime
    context: str
    pinyin: str
    selected_candidate: str
    work_id: str = ""


@dataclass(frozen=True)
class SemanticRetrievedInteraction:
    interaction: SemanticMemoryInteraction
    similarity: float


@dataclass(frozen=True)
class SemanticMemoryFeatures:
    memory_weighted_share: float
    memory_max_similarity: float
    memory_support_count: int
    memory_any_support: float
    memory_total_support: int


def memory_features(
    retrieved: Sequence[SemanticRetrievedInteraction], candidate: str
) -> SemanticMemoryFeatures:
    weights = [max(item.similarity, 0.0) for item in retrieved]
    denominator = sum(weights)
    if denominator == 0.0:
        return SemanticMemoryFeatures(
            memory_weighted_share=0.0,
            memory_max_similarity=0.0,
            memory_support_count=0,
            memory_any_support=0.0,
            memory_total_support=len(retrieved),
        )
    supported = [
        (item, weight)
        for item, weight in zip(retrieved, weights)
        if item.interaction.selected_candidate == candidate
    ]
    supported_weights = [weight for _, weight in supported]
    return SemanticMemoryFeatures(
        memory_weighted_share=(
            sum(supported_weights) / denominator
        ),
        memory_max_similarity=max(supported_weights, default=0.0),
        memory_support_count=len(supported),
        memory_any_support=float(bool(supported)),
        memory_total_support=len(retrieved),
    )


class SemanticPersonalMemory:
    def __init__(
        self,
        interactions: Sequence[SemanticMemoryInteraction],
        embedding_model: CachedEmbeddingModel,
        *,
        user_id: str,
    ) -> None:
        if any(item.user_id != user_id for item in interactions):
            raise ValueError("semantic memory cannot mix users")
        self.interactions = tuple(
            sorted(interactions, key=lambda item: (item.timestamp, item.interaction_id))
        )
        self.embedding_model = embedding_model
        self.user_id = user_id
        self._document_vectors = {
            item.interaction_id: embedding_model.encode_document(item.context)
            for item in self.interactions
        }

    def eligible_count(self, pinyin: str) -> int:
        return sum(item.pinyin == pinyin for item in self.interactions)

    def retrieve(
        self, context: str, pinyin: str, *, k: int = SEMANTIC_MEMORY_K
    ) -> tuple[SemanticRetrievedInteraction, ...]:
        if k != SEMANTIC_MEMORY_K:
            raise ValueError(f"Phase 4E semantic-memory K is frozen at {SEMANTIC_MEMORY_K}")
        eligible = [item for item in self.interactions if item.pinyin == pinyin]
        if not eligible:
            return ()
        query = self.embedding_model.encode_query(context)
        scored = [
            SemanticRetrievedInteraction(
                interaction=item,
                similarity=cosine(query, self._document_vectors[item.interaction_id]),
            )
            for item in eligible
        ]
        scored.sort(
            key=lambda item: (
                -item.similarity,
                item.interaction.timestamp,
                item.interaction.interaction_id,
            )
        )
        return tuple(scored[:SEMANTIC_MEMORY_K])
