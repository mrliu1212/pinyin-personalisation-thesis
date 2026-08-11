"""Prepare models, smoke-test, or run the frozen Phase 4E experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from src.phase_04c_evaluation import read_jsonl
from src.phase_04e_evaluation import (
    PHASE_04B6_CHECKSUM,
    PHASE_04C_CHECKSUM,
    PHASE_04D_CHECKSUM,
    Phase04EFeatureExtractor,
    evaluate_phase_04e,
    sha256_file,
)
from src.semantic_lm import (
    SEMANTIC_LM_REPO,
    CausalLMCandidateScorer,
    select_device,
)
from src.semantic_memory import (
    SEMANTIC_EMBEDDING_REPO,
    CachedEmbeddingModel,
    SentenceTransformerEmbeddingBackend,
    cosine,
)


MODEL_MANIFEST = Path("results/experiments/phase_04e/model_manifest.json")
SMOKE_RESULT = Path("results/experiments/phase_04e/smoke_test.json")
FINAL_RESULT = Path("results/experiments/phase_04e/evaluation.json")
ZHU_INTERACTIONS = Path(
    "data/processed/interactions/zhu_ziqing_simplified_rime/interactions.jsonl"
)
LU_INTERACTIONS = Path(
    "data/processed/interactions/lu_xun_simplified_rime/interactions.jsonl"
)
PHASE_04C_RESULT = Path("results/experiments/phase_04c/evaluation.json")
PHASE_04D_RESULT = Path("results/experiments/phase_04d/evaluation.json")
CACHE_DIR = Path(".cache/phase_04e")


def prepare_model_manifest(path: Path = MODEL_MANIFEST) -> dict[str, Any]:
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = {SEMANTIC_LM_REPO, SEMANTIC_EMBEDDING_REPO}
        actual = {item["repository"] for item in manifest["models"]}
        if actual != expected or any(not item["revision_sha"] for item in manifest["models"]):
            raise ValueError("existing Phase 4E model manifest is incompatible")
        return manifest
    api = HfApi()
    models = []
    for role, repository in (
        ("semantic_causal_lm", SEMANTIC_LM_REPO),
        ("semantic_embedding", SEMANTIC_EMBEDDING_REPO),
    ):
        info = api.model_info(repository)
        if not info.sha:
            raise RuntimeError(f"Hugging Face did not resolve a revision for {repository}")
        models.append(
            {"role": role, "repository": repository, "revision_sha": info.sha}
        )
    manifest = {
        "schema_version": 1,
        "resolved_at": datetime.now(timezone.utc).isoformat(),
        "source": "Hugging Face model repository API",
        "models": models,
        "reuse_policy": "official runs must use these exact revision SHAs",
        "weights_committed_to_git": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def revision(manifest: dict[str, Any], role: str) -> str:
    return next(item["revision_sha"] for item in manifest["models"] if item["role"] == role)


def load_services(manifest: dict[str, Any]):
    import torch

    device = select_device(torch)
    lm, lm_runtime = CausalLMCandidateScorer.from_pretrained(
        revision=revision(manifest, "semantic_causal_lm"),
        cache_dir=CACHE_DIR,
        device=device,
    )
    embedding_backend = SentenceTransformerEmbeddingBackend.from_pretrained(
        revision=revision(manifest, "semantic_embedding"), device=device
    )
    embeddings = CachedEmbeddingModel(
        embedding_backend,
        revision=revision(manifest, "semantic_embedding"),
        cache_dir=CACHE_DIR,
    )
    runtime = {
        "device": device,
        "lm_dtype": lm_runtime["dtype"],
        "embedding_dtype": embedding_backend.dtype,
    }
    return lm, embeddings, runtime


def smoke_test() -> dict[str, Any]:
    manifest = prepare_model_manifest()
    lm, embeddings, runtime = load_services(manifest)
    context = "我们可以在这里继续"
    candidates = ("使用", "实用")
    first = lm.score_candidates(context, candidates)
    second = lm.score_candidates(context, candidates)
    query_first = embeddings.encode_query(context)
    query_second = embeddings.encode_query(context)
    document = embeddings.encode_document("我们可以采用这个办法")
    similarity = cosine(query_first, document)
    if first != second or query_first != query_second:
        raise AssertionError("Phase 4E neural cache is not deterministic")
    values = [
        value
        for item in first
        for value in (
            item.lm_conditional_logprob,
            item.lm_prior_logprob,
            item.lm_context_gain,
        )
    ]
    if not all(math.isfinite(value) for value in values):
        raise AssertionError("non-finite LM smoke-test score")
    if not all(math.isfinite(value) for value in (*query_first, *document, similarity)):
        raise AssertionError("non-finite embedding smoke-test value")
    result = {
        "schema_version": 1,
        "purpose": "engineering smoke test only; not benchmark evaluation",
        "models": manifest["models"],
        "runtime": runtime,
        "lm_scores": [item.__dict__ for item in first],
        "embedding_dimensions": len(query_first),
        "cosine_similarity": similarity,
        "cache_reuse_verified": True,
    }
    SMOKE_RESULT.parent.mkdir(parents=True, exist_ok=True)
    SMOKE_RESULT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def verify_frozen_artifacts() -> None:
    expected = {
        PHASE_04C_RESULT: PHASE_04C_CHECKSUM,
        PHASE_04D_RESULT: PHASE_04D_CHECKSUM,
        ZHU_INTERACTIONS: PHASE_04B6_CHECKSUM,
    }
    for path, checksum in expected.items():
        actual = sha256_file(path)
        if actual != checksum:
            raise ValueError(f"frozen artifact checksum mismatch: {path}: {actual}")


def run_final(output: Path = FINAL_RESULT) -> dict[str, Any]:
    verify_frozen_artifacts()
    manifest = prepare_model_manifest()
    lm, embeddings, runtime = load_services(manifest)
    phase_04d = json.loads(PHASE_04D_RESULT.read_text(encoding="utf-8"))
    result = evaluate_phase_04e(
        read_jsonl(ZHU_INTERACTIONS),
        read_jsonl(LU_INTERACTIONS),
        phase_04d,
        Phase04EFeatureExtractor(lm, embeddings),
    )
    result["model_manifest"] = manifest
    result["runtime"] = runtime
    result["frozen_artifact_checksums"] = {
        str(PHASE_04C_RESULT): PHASE_04C_CHECKSUM,
        str(PHASE_04D_RESULT): PHASE_04D_CHECKSUM,
        str(ZHU_INTERACTIONS): PHASE_04B6_CHECKSUM,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Phase 4E result: {output}")
    for subset, conditions in result["subsets"].items():
        print(f"\n{subset}")
        for condition, summary in conditions.items():
            metrics = summary["metrics"]
            print(
                f"{condition}: Top-1={metrics['top1_accuracy']:.4f}, "
                f"MRR={metrics['mrr']:.4f}, missing={metrics['missing_target_count']}"
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-models", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output", type=Path, default=FINAL_RESULT)
    args = parser.parse_args()
    if args.prepare_models:
        print(json.dumps(prepare_model_manifest(), ensure_ascii=False, indent=2))
    elif args.smoke_test:
        print(json.dumps(smoke_test(), ensure_ascii=False, indent=2))
    else:
        run_final(args.output)


if __name__ == "__main__":
    main()
