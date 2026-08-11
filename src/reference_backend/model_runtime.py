"""Desktop llama.cpp-compatible runtime boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import platform
import time
from typing import Any, Protocol, Sequence


OFFICIAL_GENERATION_SHA256 = (
    "2012b7aa860674e5f2b9fc0c90cc4828b7e5f50f7be4069fa0122685956416a5"
)
OFFICIAL_EMBEDDING_SHA256 = (
    "5a88d266870fbd27c6f329df60de80e2d4cf3bbd5e6f080bd5c1b2e5abb12039"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RuntimeGeneration:
    text: str
    score: float | None
    seed: int
    elapsed_ms: float


class GenerationRuntime(Protocol):
    supports_official_trigger_policy: bool

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        seed: int,
        top_k: int,
        top_p: float,
        temperature: float,
        repeat_penalty: float,
        repeat_last_n: int,
    ) -> RuntimeGeneration: ...

    def info(self) -> dict[str, Any]: ...


class EmbeddingRuntime(Protocol):
    dimension: int

    def embed(self, text: str) -> Sequence[float]: ...

    def info(self) -> dict[str, Any]: ...


class LlamaCppGenerationRuntime:
    """Resident desktop wrapper for the official merged Q4_0 checkpoint."""

    def __init__(
        self,
        model_path: Path,
        *,
        n_ctx: int = 8192,
        n_threads: int | None = None,
        n_gpu_layers: int = -1,
        verify_checksum: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.model_sha256 = sha256_file(self.model_path)
        if verify_checksum and self.model_sha256 != OFFICIAL_GENERATION_SHA256:
            raise ValueError("generation checkpoint is not the pinned official asset")
        self.supports_official_trigger_policy = (
            self.model_sha256 == OFFICIAL_GENERATION_SHA256
        )
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self._model = None
        self._load_ms: float | None = None
        self._calls = 0
        self._cache_hits = 0
        self._completion_cache: dict[tuple[Any, ...], RuntimeGeneration] = {}

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required; install requirements-phase4f.txt"
            ) from exc
        kwargs: dict[str, Any] = {
            "model_path": str(self.model_path),
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "last_n_tokens_size": 16,
            "logits_all": True,
            "verbose": False,
        }
        if self.n_threads is not None:
            kwargs["n_threads"] = self.n_threads
        start = time.perf_counter()
        self._model = Llama(**kwargs)
        self._load_ms = (time.perf_counter() - start) * 1000.0
        return self._model

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        seed: int,
        top_k: int,
        top_p: float,
        temperature: float,
        repeat_penalty: float,
        repeat_last_n: int,
    ) -> RuntimeGeneration:
        key = (
            prompt,
            max_tokens,
            seed,
            top_k,
            top_p,
            temperature,
            repeat_penalty,
            repeat_last_n,
        )
        cached = self._completion_cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return RuntimeGeneration(cached.text, cached.score, cached.seed, 0.0)
        model = self._ensure_loaded()
        start = time.perf_counter()
        result = model.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            seed=seed,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
            stop=["<|im_end|>", "<|endoftext|>"],
            echo=False,
            logprobs=1,
        )
        elapsed = (time.perf_counter() - start) * 1000.0
        self._calls += 1
        choice = result["choices"][0]
        logprobs = (choice.get("logprobs") or {}).get("token_logprobs") or []
        finite = [float(value) for value in logprobs if value is not None and math.isfinite(value)]
        score = sum(finite) / len(finite) if finite else None
        output = RuntimeGeneration(
            text=str(choice.get("text", "")),
            score=score,
            seed=seed,
            elapsed_ms=elapsed,
        )
        self._completion_cache[key] = output
        return output

    def info(self) -> dict[str, Any]:
        try:
            import llama_cpp

            version = getattr(llama_cpp, "__version__", "unknown")
        except ImportError:
            version = "not installed"
        return {
            "runtime": "llama-cpp-python",
            "runtime_version": version,
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "model_size_bytes": self.model_path.stat().st_size,
            "quantization": "Q4_0",
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "model_load_ms": self._load_ms,
            "generation_calls": self._calls,
            "completion_cache_hits": self._cache_hits,
            "official_checkpoint_verified": self.supports_official_trigger_policy,
        }


class LlamaCppEmbeddingRuntime:
    """Resident embedding wrapper for the official BGE Q8_0 checkpoint."""

    dimension = 512

    def __init__(
        self,
        model_path: Path,
        *,
        n_ctx: int = 512,
        n_threads: int | None = None,
        n_gpu_layers: int = -1,
        verify_checksum: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.model_sha256 = sha256_file(self.model_path)
        if verify_checksum and self.model_sha256 != OFFICIAL_EMBEDDING_SHA256:
            raise ValueError("embedding checkpoint is not the pinned official asset")
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self._model = None
        self._load_ms: float | None = None
        self._calls = 0
        self._cache_hits = 0
        self._embedding_cache: dict[str, tuple[float, ...]] = {}

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        try:
            from llama_cpp import Llama, LLAMA_POOLING_TYPE_MEAN
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required; install requirements-phase4f.txt"
            ) from exc
        kwargs: dict[str, Any] = {
            "model_path": str(self.model_path),
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "embedding": True,
            "pooling_type": LLAMA_POOLING_TYPE_MEAN,
            "verbose": False,
        }
        if self.n_threads is not None:
            kwargs["n_threads"] = self.n_threads
        start = time.perf_counter()
        self._model = Llama(**kwargs)
        self._load_ms = (time.perf_counter() - start) * 1000.0
        return self._model

    def embed(self, text: str) -> tuple[float, ...]:
        cached = self._embedding_cache.get(text)
        if cached is not None:
            self._cache_hits += 1
            return cached
        # The pinned llama-cpp-python API returns pooled vectors directly; the
        # audited HuoziIME path L2-normalizes at the HNSW boundary.
        result = self._ensure_loaded().create_embedding(text)
        self._calls += 1
        vector = result["data"][0]["embedding"]
        if len(vector) != self.dimension:
            raise ValueError(f"unexpected embedding dimension: {len(vector)}")
        output = tuple(float(value) for value in vector)
        self._embedding_cache[text] = output
        return output

    def info(self) -> dict[str, Any]:
        return {
            "runtime": "llama-cpp-python",
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "model_size_bytes": self.model_path.stat().st_size,
            "quantization": "Q8_0",
            "dimension": self.dimension,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "model_load_ms": self._load_ms,
            "embedding_calls": self._calls,
            "embedding_cache_hits": self._cache_hits,
        }
