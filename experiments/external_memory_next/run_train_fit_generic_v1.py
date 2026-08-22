"""Run/resume frozen PinyinGPT Generic inference on standardized Train-Fit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.evaluation.deep_author_v2 import (
    BACKEND_INTEGRATION_REVISION,
    BACKEND_SOURCE_REVISION,
    CHECKPOINT_REVISION,
    OFFICIAL_CODE_REVISION,
)
from src.personalisation.pilot_a import GENERIC_CONTEXT_SEMANTICS
from src.personalisation.standardized_generic import generate_resumable, read_jsonl
from src.reference_backend_pinyingpt import PinyinGPTConcatBackend


EXPECTED_FIT_SHA256 = "547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6"
EXPECTED_FIT_ROWS = 144526


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if sha256_file(args.manifest) != EXPECTED_FIT_SHA256:
        raise RuntimeError("standardized Train-Fit manifest SHA changed")
    rows = read_jsonl(args.manifest)
    if len(rows) != EXPECTED_FIT_ROWS or any(row.get("standardized_partition") != "train_fit" for row in rows):
        raise RuntimeError("standardized Train-Fit population changed")
    if any(str(row.get("source_split", "")).lower() == "test" for row in rows):
        raise RuntimeError("STOP: Test row detected in Train-Fit")
    backend = PinyinGPTConcatBackend(args.checkpoint, device="cuda")
    result = generate_resumable(
        rows, backend, args.output, batch_size=args.batch_size,
        checkpoint_revision=CHECKPOINT_REVISION,
        official_code_revision=OFFICIAL_CODE_REVISION,
        backend_source_revision=BACKEND_SOURCE_REVISION,
        backend_integration_revision=BACKEND_INTEGRATION_REVISION,
        context_semantics=GENERIC_CONTEXT_SEMANTICS,
    )
    result["runtime"] = backend.runtime_info()
    result["manifest_sha256"] = EXPECTED_FIT_SHA256
    result["used_dev3000"] = False
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
