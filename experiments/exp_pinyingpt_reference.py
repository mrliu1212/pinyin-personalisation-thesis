"""Run a small real-model PinyinGPT2-Concat reference smoke test."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

from src.reference_backend_pinyingpt import PinyinGPTConcatBackend


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / ".build/pinyingpt2-concat"
DEFAULT_OUTPUT = ROOT / "results/experiments/pinyingpt/smoke_test_windows_cuda.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def smoke_test(checkpoint: Path = DEFAULT_CHECKPOINT) -> dict[str, Any]:
    backend = PinyinGPTConcatBackend(checkpoint)
    examples = [
        ("这个工具真的很", "shi yong"),
        ("奥斯卡组委会", "qing xiang yu kan hao"),
    ]
    generated = [
        backend.generate(context, pinyin, top_k=10, beam_size=16).to_dict()
        for context, pinyin in examples
    ]
    fixed = backend.score_candidates(
        "这个工具真的很",
        "shi yong",
        ["使用", "实用", "适用", "试用"],
    )
    return {
        "schema_version": 1,
        "purpose": "PinyinGPT real-model engineering smoke test; not a benchmark",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "runtime": backend.runtime_info(),
        "checkpoint_files": {
            "pytorch_model.bin": sha256_file(checkpoint / "pytorch_model.bin"),
            "config.json": sha256_file(checkpoint / "config.json"),
            "vocab.txt": sha256_file(checkpoint / "vocab.txt"),
            "pinyin2char.json": sha256_file(checkpoint / "pinyin2char.json"),
        },
        "decoding": {
            "mode": "Pinyin-constrained character-level beam search",
            "beam_size": 16,
            "returned_candidates": 10,
            "score": "sum of full-vocabulary autoregressive token log-probabilities",
            "oracle_pinyin_segmentation": True,
        },
        "generated_examples": generated,
        "fixed_candidate_example": {
            "context": "这个工具真的很",
            "typed_pinyin": "shi yong",
            "candidates": [candidate.to_dict() for candidate in fixed],
        },
        "final_benchmark_run": False,
        "personalisation_implemented": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = smoke_test(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
