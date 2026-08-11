"""Audit, prepare, smoke-test, or manually evaluate the Phase 4F backend."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
import urllib.request
import zipfile
from typing import Any

from src.phase_04c_evaluation import (
    LU_TRAIN_WORK_IDS,
    ZHU_TEST_WORK_IDS,
    ZHU_TRAIN_WORK_IDS,
    frozen_split,
    read_jsonl,
)
from src.phase_04f_evaluation import (
    CONDITIONS,
    PHASE_04B6_CHECKSUM,
    PHASE_04C_CHECKSUM,
    PHASE_04D_CHECKSUM,
    PHASE_04E_MANIFEST_CHECKSUM,
    evaluate_phase_04f,
)
from src.reference_backend.backend import ReferencePersonalisedIMEBackend
from src.reference_backend.benchmark_adapter import Phase04FBenchmarkAdapter
from src.reference_backend.candidate_generator import HuoziIMECandidateGenerator
from src.reference_backend.hierarchical_memory import BackgroundMemoryProcessor
from src.reference_backend.interaction_store import InteractionTraceStore
from src.reference_backend.memory_extractor import HuoziIMEMemoryExtractor
from src.reference_backend.memory_store import MemoryRecord, MemoryStore
from src.reference_backend.model_runtime import (
    LlamaCppEmbeddingRuntime,
    LlamaCppGenerationRuntime,
    OFFICIAL_EMBEDDING_SHA256,
    OFFICIAL_GENERATION_SHA256,
    sha256_file,
)
from src.reference_backend.pinyin_decoder import LibrimeLunaPinyinDecoder
from src.reference_backend.pinyin_integration import integrate_separate_channels
from src.reference_backend.vector_index import HNSWMemoryIndex


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache/phase_04f"
APK_PATH = CACHE_DIR / "official_release/HuoziIME-v1.0.1.apk"
MODEL_DIR = CACHE_DIR / "models"
GENERATION_MODEL = MODEL_DIR / "scirime_grpo_v2_744-q4_0.gguf"
EMBEDDING_MODEL = MODEL_DIR / "bge-small-zh-v1.5-q8_0.gguf"
PRE_MEMORY_ASSET = MODEL_DIR / "pre_mem.jsonl"
PREPARED_ASSET_MANIFEST = CACHE_DIR / "prepared_assets.json"
STATE_DIR = CACHE_DIR / "frozen_state"
STATE_MANIFEST = STATE_DIR / "state_manifest.json"

BACKEND_MANIFEST = ROOT / "results/experiments/phase_04f/backend_manifest.json"
REPRODUCTION_MATRIX = ROOT / "results/audits/phase_04f/reproduction_matrix.json"
SMOKE_RESULT = ROOT / "results/experiments/phase_04f/smoke_test.json"
FINAL_RESULT = ROOT / "results/experiments/phase_04f/evaluation.json"
PINYIN_AUDIT = ROOT / "results/audits/phase_04f/pinyin_integration_audit.json"
RIME_EXECUTABLE = ROOT / ".build/rime_candidate_cli"
RIME_ROOT = ROOT / "data/rime"
RIME_SETUP_MANIFEST = RIME_ROOT / "setup_manifest.json"

ZHU_INTERACTIONS = ROOT / "data/processed/interactions/zhu_ziqing_simplified_rime/interactions.jsonl"
LU_INTERACTIONS = ROOT / "data/processed/interactions/lu_xun_simplified_rime/interactions.jsonl"
PHASE_04C_RESULT = ROOT / "results/experiments/phase_04c/evaluation.json"
PHASE_04D_RESULT = ROOT / "results/experiments/phase_04d/evaluation.json"
PHASE_04E_MANIFEST = ROOT / "results/experiments/phase_04e/model_manifest.json"

APK_URL = "https://github.com/Shan-HIT/HuoziIME/releases/download/v1.0.1-beta/HuoziIME.apk"
APK_SHA256 = "6ce98a804e503aa2d6dc426ff6284d5064ffff09c9527dcaacfc050f6ab99207"
APK_SIZE = 995_361_342
LU_INTERACTIONS_SHA256 = "cc022956f36ba21e61f70677355cdd6c31a5b52854e744e3920a63c124a94861"
ASSETS = {
    "assets/scirime_grpo_v2_744-q4_0.gguf": (
        GENERATION_MODEL,
        OFFICIAL_GENERATION_SHA256,
        468_700_896,
    ),
    "assets/bge-small-zh-v1.5-q8_0.gguf": (
        EMBEDDING_MODEL,
        OFFICIAL_EMBEDDING_SHA256,
        26_472_640,
    ),
    "assets/pre_mem.jsonl": (
        PRE_MEMORY_ASSET,
        "08de8820ea8772216545dac2eb690715108fd76ff3539ef9416335aacbe5333e",
        263,
    ),
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_frozen_artifacts() -> dict[str, str]:
    expected = {
        ZHU_INTERACTIONS: PHASE_04B6_CHECKSUM,
        LU_INTERACTIONS: LU_INTERACTIONS_SHA256,
        PHASE_04C_RESULT: PHASE_04C_CHECKSUM,
        PHASE_04D_RESULT: PHASE_04D_CHECKSUM,
        PHASE_04E_MANIFEST: PHASE_04E_MANIFEST_CHECKSUM,
    }
    verified = {}
    for path, expected_hash in expected.items():
        actual = sha256_file(path)
        if actual != expected_hash:
            raise ValueError(f"frozen artifact checksum mismatch: {path}: {actual}")
        verified[str(path.relative_to(ROOT))] = actual
    return verified


def audit() -> dict[str, Any]:
    manifest = json.loads(BACKEND_MANIFEST.read_text(encoding="utf-8"))
    matrix = json.loads(REPRODUCTION_MATRIX.read_text(encoding="utf-8"))
    required = {
        "LLM base model", "post-trained IME model", "prompt/template",
        "candidate generation", "special action tokens", "memory trigger",
        "memory extraction", "L1", "L2", "L3", "plaintext memory",
        "embedding model", "HNSW", "memory-grounded generation",
        "asynchronous memory update", "quantization", "KV/prefix caching",
        "mobile scheduling", "Android UI", "MCP/chat context", "evaluation assets",
        "Pinyin decoder", "candidate-surface integration",
    }
    observed = {item["component"] for item in matrix["components"]}
    missing = sorted(required - observed)
    if missing:
        raise ValueError(f"reproduction matrix is incomplete: {missing}")
    invalid = [
        item["component"] for item in matrix["components"]
        if item["classification"] not in {"A", "B", "C", "D", "E"}
    ]
    if invalid:
        raise ValueError(f"invalid classifications: {invalid}")
    if manifest["audit"]["repository_commit"] != matrix["upstream_commit"]:
        raise ValueError("upstream SHA mismatch between audit artifacts")
    pinyin_audit = json.loads(PINYIN_AUDIT.read_text(encoding="utf-8"))
    if pinyin_audit["upstream_commit"] != matrix["upstream_commit"]:
        raise ValueError("Pinyin audit used a different upstream SHA")
    assets = {}
    for path, expected_hash in (
        (APK_PATH, APK_SHA256),
        (GENERATION_MODEL, OFFICIAL_GENERATION_SHA256),
        (EMBEDDING_MODEL, OFFICIAL_EMBEDDING_SHA256),
    ):
        assets[str(path.relative_to(ROOT))] = (
            {"present": True, "sha256": sha256_file(path), "verified": sha256_file(path) == expected_hash}
            if path.exists()
            else {"present": False, "verified": False}
        )
    return {
        "phase": manifest["phase"],
        "final_reproduction_label": manifest["final_reproduction_label"],
        "upstream_commit": manifest["audit"]["repository_commit"],
        "upstream_release": manifest["audit"]["upstream_release"],
        "component_count": len(matrix["components"]),
        "classification_counts": {
            label: sum(item["classification"] == label for item in matrix["components"])
            for label in ("A", "B", "C", "D", "E")
        },
        "unavailable_upstream_artifacts": manifest["unavailable_upstream_artifacts"],
        "frozen_artifacts": verify_frozen_artifacts(),
        "local_assets": assets,
        "pinyin_integration": pinyin_audit["conclusion"],
        "final_evaluation_present": FINAL_RESULT.exists(),
    }


def _download_apk() -> None:
    if APK_PATH.exists():
        if APK_PATH.stat().st_size == APK_SIZE and sha256_file(APK_PATH) == APK_SHA256:
            return
        raise ValueError(f"existing APK failed verification: {APK_PATH}")
    APK_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = APK_PATH.with_suffix(".apk.partial")
    print(f"Downloading official v1.0.1-beta APK ({APK_SIZE} bytes)...")
    try:
        urllib.request.urlretrieve(APK_URL, partial)
        if partial.stat().st_size != APK_SIZE or sha256_file(partial) != APK_SHA256:
            raise ValueError("downloaded APK checksum/size mismatch")
        partial.replace(APK_PATH)
    finally:
        if partial.exists() and not APK_PATH.exists():
            partial.unlink()


def prepare_assets() -> dict[str, Any]:
    _download_apk()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(APK_PATH) as archive:
        for member, (destination, expected_hash, expected_size) in ASSETS.items():
            if destination.exists():
                if destination.stat().st_size != expected_size or sha256_file(destination) != expected_hash:
                    raise ValueError(f"existing extracted asset failed verification: {destination}")
                continue
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            if destination.stat().st_size != expected_size or sha256_file(destination) != expected_hash:
                raise ValueError(f"extracted asset failed verification: {member}")
    if not RIME_EXECUTABLE.exists() or not RIME_SETUP_MANIFEST.exists():
        raise FileNotFoundError(
            "Phase 4F.1 decoder is not prepared; run `make rime-adapter` and "
            "`.venv/bin/python -m interactions.setup_rime` first"
        )
    rime_manifest = json.loads(RIME_SETUP_MANIFEST.read_text(encoding="utf-8"))
    if rime_manifest.get("schema_id") != "luna_pinyin":
        raise ValueError("Phase 4F.1 requires the pinned Luna Pinyin schema")
    result = {
        "schema_version": 1,
        "source_release": "v1.0.1-beta",
        "source_apk": {
            "url": APK_URL,
            "path": str(APK_PATH.relative_to(ROOT)),
            "sha256": APK_SHA256,
            "size_bytes": APK_SIZE,
        },
        "assets": [
            {
                "archive_member": member,
                "path": str(destination.relative_to(ROOT)),
                "sha256": expected_hash,
                "size_bytes": expected_size,
            }
            for member, (destination, expected_hash, expected_size) in ASSETS.items()
        ],
        "pinyin_decoder": {
            "implementation": "desktop librime C API adapter",
            "version": rime_manifest["librime"],
            "schema": "luna_pinyin",
            "dictionary": "pinned rime-luna-pinyin source set",
            "dictionary_revision": "rime-luna-pinyin@56b934b099dfbeab842320f13aa8b461a6ab3e42 with rime-essay@e9b1a374a6ea015fca5bdd04318924b4483ac35a",
            "simplified_mode": "zh_hans",
            "candidate_count": 10,
            "configuration": "data/rime/setup_manifest.json",
            "configuration_sha256": sha256_file(RIME_SETUP_MANIFEST),
            "integration_status": "FAITHFUL DESKTOP ADAPTATION",
        },
        "weights_committed_to_git": False,
    }
    _write_json(PREPARED_ASSET_MANIFEST, result)
    return result


def _services():
    generation = LlamaCppGenerationRuntime(GENERATION_MODEL, n_gpu_layers=-1)
    embedding = LlamaCppEmbeddingRuntime(EMBEDDING_MODEL, n_gpu_layers=-1)
    generator = HuoziIMECandidateGenerator(generation)
    extractor = HuoziIMEMemoryExtractor(generation)
    return generation, embedding, generator, extractor


def _pinyin_decoder() -> LibrimeLunaPinyinDecoder:
    manifest = json.loads(RIME_SETUP_MANIFEST.read_text(encoding="utf-8"))
    return LibrimeLunaPinyinDecoder(
        executable=RIME_EXECUTABLE,
        shared_data=RIME_ROOT / "shared",
        prebuilt_data=RIME_ROOT / "build",
        version=manifest["librime"],
        max_candidates=10,
    )


def _expected_state_inputs() -> dict[str, str]:
    return {
        "generation_model_sha256": OFFICIAL_GENERATION_SHA256,
        "embedding_model_sha256": OFFICIAL_EMBEDDING_SHA256,
        "zhu_interactions_sha256": PHASE_04B6_CHECKSUM,
        "lu_interactions_sha256": LU_INTERACTIONS_SHA256,
        "upstream_commit": "63f249e711f6501169e6baafec7e12318b3c765b",
    }


def prepare_frozen_state() -> dict[str, Any]:
    if STATE_MANIFEST.exists():
        existing = json.loads(STATE_MANIFEST.read_text(encoding="utf-8"))
        if existing.get("inputs") != _expected_state_inputs():
            raise ValueError("existing Phase 4F frozen state has incompatible inputs")
        return existing
    verify_frozen_artifacts()
    generation, embedding, _, extractor = _services()
    zhu = read_jsonl(ZHU_INTERACTIONS)
    lu = read_jsonl(LU_INTERACTIONS)
    zhu_split = frozen_split(zhu, ZHU_TRAIN_WORK_IDS, ZHU_TEST_WORK_IDS)
    # Lu's frozen file contains exactly the Phase 4C train+test works; only train is selected.
    lu_train = tuple(record for record in lu if record["work_id"] in set(LU_TRAIN_WORK_IDS))
    if not lu_train:
        raise ValueError("Lu frozen training history is empty")
    adapter = Phase04FBenchmarkAdapter()
    staging = CACHE_DIR / "frozen_state.preparing"
    if staging.exists():
        shutil.rmtree(staging)
    summaries = {}
    try:
        for user_id, train in (("zhu_ziqing", zhu_split.train), ("lu_xun", lu_train)):
            store = MemoryStore(staging / "l2", user_id=user_id)
            index = HNSWMemoryIndex(staging / "hnsw", user_id=user_id, dimension=embedding.dimension)
            trajectories = adapter.training_trajectories(train, user_id=user_id)
            processor = BackgroundMemoryProcessor(
                user_id=user_id,
                store=store,
                index=index,
                extractor=extractor,
                embedding_runtime=embedding,
                trace_path=staging / "l3" / user_id / "background_memory.jsonl",
            )
            processed = processor.process(trajectories)
            test_ids = {
                record["interaction_id"] for record in (zhu_split.test if user_id == "zhu_ziqing" else ())
            }
            if store.source_interaction_ids() & test_ids:
                raise ValueError("test interaction entered prepared personal memory")
            summaries[user_id] = {
                "training_interaction_count": len(train),
                "training_interaction_ids_sha256": hashlib.sha256(
                    "\n".join(sorted(record["interaction_id"] for record in train)).encode()
                ).hexdigest(),
                "trajectory_count": len(trajectories),
                "memory_count": len(store),
                "memory_ids": [record.memory_id for record in store.list()],
                "indexed_count": len(index),
                "background_status_counts": {
                    status: sum(item.status == status for item in processed)
                    for status in sorted({item.status for item in processed})
                },
                "test_memory_overlap_count": len(store.source_interaction_ids() & test_ids),
            }
        state_manifest = {
            "schema_version": 1,
            "purpose": "frozen training-derived Phase 4F user state; no test-time updates",
            "inputs": _expected_state_inputs(),
            "users": summaries,
            "memory_extraction": "official checkpoint and IdleMemoryWorker prompt/schema",
            "benchmark_adapter": "per-work training trajectory, upstream 4000-character cap",
        }
        _write_json(staging / "state_manifest.json", state_manifest)
        staging.replace(STATE_DIR)
        return state_manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def prepare() -> dict[str, Any]:
    assets = prepare_assets()
    state = prepare_frozen_state()
    return {"assets": assets, "frozen_state": state}


def _backend(
    user_id: str,
    *,
    generator: HuoziIMECandidateGenerator,
    embedding: LlamaCppEmbeddingRuntime,
    state_root: Path = STATE_DIR,
) -> ReferencePersonalisedIMEBackend:
    store = MemoryStore(state_root / "l2", user_id=user_id, read_only=True)
    index = HNSWMemoryIndex(state_root / "hnsw", user_id=user_id, dimension=embedding.dimension)
    index.validate_against(store)
    return ReferencePersonalisedIMEBackend(
        user_id=user_id,
        candidate_generator=generator,
        embedding_runtime=embedding,
        memory_store=store,
        memory_index=index,
        interaction_store=None,
        official_trigger_policy=True,
    )


def smoke_test() -> dict[str, Any]:
    prepare_assets()
    generation, embedding, generator, _ = _services()
    smoke_root = CACHE_DIR / "smoke_state"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    user_id = "phase_04f_smoke_user"
    store = MemoryStore(smoke_root / "l2", user_id=user_id)
    index = HNSWMemoryIndex(smoke_root / "hnsw", user_id=user_id, dimension=embedding.dimension)
    memory = MemoryRecord.create(
        user_id=user_id,
        plaintext="客户张总本周五来访 | 事项: 接待 | 细节: 准备他喜欢的红茶。",
        creation_position="2026-02-01|000000000001|official_release_smoke_fixture",
        source_interaction_ids=("official_release_pre_mem_smoke_fixture",),
        provenance={"source": "official v1.0.1-beta pre_mem demonstration asset"},
    )
    memory = index.add(memory, embedding.embed(memory.plaintext))
    store.add(memory)
    index.validate_against(store)
    traces = InteractionTraceStore(smoke_root / "l3", user_id=user_id)
    backend = ReferencePersonalisedIMEBackend(
        user_id=user_id,
        candidate_generator=generator,
        embedding_runtime=embedding,
        memory_store=store,
        memory_index=index,
        interaction_store=traces,
        official_trigger_policy=True,
    )
    first = backend.predict(
        user_id,
        "请告诉我上次约定的客户张总来访时要准备什么",
        "beijing",
        4,
        external_context=None,
        seed_base=404_600,
        chronological_position="2026-08-11|000000000001|smoke",
        record_trace=True,
    )
    with _pinyin_decoder() as pinyin_decoder:
        pinyin_result = pinyin_decoder.decode("beijing", top_k=10)
    integrated = integrate_separate_channels(pinyin_result, first)
    if "北京" not in [candidate.text for candidate in pinyin_result.candidates]:
        raise RuntimeError("Pinyin smoke failed: beijing did not produce 北京")
    query = embedding.embed("客户张总来访准备什么")
    direct_index_results = index.search(user_id=user_id, query_vector=query, k=20)
    forced_grounded = generator.generate(
        "客户张总本周五来访，我应该",
        top_k=1,
        external_context=None,
        memory_plaintext=(memory.plaintext,),
        seed_base=404_700,
    )
    if not forced_grounded.candidates:
        raise RuntimeError("HuoziIME smoke failed: grounded generation returned no suggestion")
    result = {
        "schema_version": 2,
        "phase": "Phase 4F.1 — Pinyin Integration Correction",
        "purpose": "engineering smoke test only; no Phase 4F benchmark result",
        "prediction": first.to_dict(),
        "pinyin_decoder": pinyin_result.to_dict(),
        "integrated_result": integrated.to_dict(),
        "subsystem_status": {
            "pinyin_decoder": "PASS: beijing produced 北京",
            "huoziime": "PASS: contextual generation and explicit memory-grounded suggestion completed",
            "channels_unified": False,
        },
        "official_trigger_observed": first.memory_trigger.should_retrieve,
        "direct_hnsw_diagnostic": [asdict(item) for item in direct_index_results],
        "explicit_grounded_path_diagnostic": {
            "reason": "verifies real-model grounded generation even if stochastic official policy did not trigger in this single smoke query",
            "memory_id": memory.memory_id,
            "candidates": [asdict(item) for item in forced_grounded.candidates],
            "elapsed_ms": forced_grounded.elapsed_ms,
        },
        "runtime": generation.info(),
        "embedding_runtime": embedding.info(),
        "peak_process_ram_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "platform": platform.platform(),
        "l3_trace_count": len(traces.list()),
        "final_evaluation_run": False,
    }
    _write_json(SMOKE_RESULT, result)
    return result


def run_final(output: Path = FINAL_RESULT) -> dict[str, Any]:
    """Manual-only final evaluation entrypoint. Never called by prepare/smoke/audit."""
    frozen = verify_frozen_artifacts()
    if not STATE_MANIFEST.exists():
        raise FileNotFoundError("run --prepare before the final evaluation")
    state = json.loads(STATE_MANIFEST.read_text(encoding="utf-8"))
    if state.get("inputs") != _expected_state_inputs():
        raise ValueError("frozen Phase 4F state inputs changed")
    generation, embedding, generator, _ = _services()
    backends = {
        "generic_no_memory": _backend("generic_no_memory", generator=generator, embedding=embedding),
        "correct_user_memory": _backend("zhu_ziqing", generator=generator, embedding=embedding),
        "wrong_user_memory": _backend("lu_xun", generator=generator, embedding=embedding),
    }
    zhu = read_jsonl(ZHU_INTERACTIONS)
    test = frozen_split(zhu, ZHU_TRAIN_WORK_IDS, ZHU_TEST_WORK_IDS).test
    with _pinyin_decoder() as pinyin_decoder:
        result = evaluate_phase_04f(test, backends=backends, pinyin_decoder=pinyin_decoder)
    result["backend_manifest"] = json.loads(BACKEND_MANIFEST.read_text(encoding="utf-8"))
    result["frozen_state_manifest"] = state
    result["frozen_artifact_checksums"] = frozen
    result["runtime"] = generation.info()
    result["embedding_runtime"] = embedding.info()
    _write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output", type=Path, default=FINAL_RESULT)
    args = parser.parse_args()
    if args.audit:
        value = audit()
    elif args.prepare:
        value = prepare()
    elif args.smoke_test:
        value = smoke_test()
    else:
        value = run_final(args.output)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
