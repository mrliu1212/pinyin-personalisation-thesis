"""Validate and freeze provenance for the completed Train-Fit Generic cache."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable


EXPECTED_FIT_SHA256 = "547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6"
EXPECTED_ROWS = 144526
EXPECTED_REVISIONS = {
    "checkpoint_revision": "76dd20dc92d8236a350fb732e99dde6fa15e2263",
    "official_code_revision": "8f1573ed0bd4d1f3d8d3f10a05f7e870725646f1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def parse_final_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    start = text.rfind("\n{")
    if start < 0:
        start = text.find("{")
    else:
        start += 1
    if start < 0:
        raise RuntimeError("Generic stdout has no final JSON object")
    return json.loads(text[start:])


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.fit) != EXPECTED_FIT_SHA256:
        raise RuntimeError("Train-Fit SHA changed")
    count = 0
    for count, pair in enumerate(itertools.zip_longest(iter_jsonl(args.fit), iter_jsonl(args.predictions)), start=1):
        source, prediction = pair
        if source is None or prediction is None:
            raise RuntimeError("Train-Fit and prediction counts differ")
        if str(source["row_id"]) != str(prediction["row_id"]):
            raise RuntimeError(f"Prediction order differs at {count}")
        if prediction.get("used_test") is not False or str(prediction.get("source_split", "")).lower() == "test":
            raise RuntimeError(f"Test dependency at {prediction['row_id']}")
        if int(prediction.get("beam_size", 0)) != 16 or int(prediction.get("top_k", 0)) != 10:
            raise RuntimeError(f"Frozen decoding changed at {prediction['row_id']}")
        for key, expected in EXPECTED_REVISIONS.items():
            if str(prediction.get(key)) != expected:
                raise RuntimeError(f"{key} changed at {prediction['row_id']}")
    if count != EXPECTED_ROWS:
        raise RuntimeError(f"Prediction count changed: {count}")
    runtime = parse_final_json(args.stdout)
    if runtime.get("status") != "complete" or int(runtime.get("rows", 0)) != EXPECTED_ROWS:
        raise RuntimeError("Generic completion record is invalid")
    if runtime.get("used_test") is not False or runtime.get("used_dev3000") is not False:
        raise RuntimeError("Generic completion boundary changed")
    if runtime.get("runtime", {}).get("device") != "cuda":
        raise RuntimeError("Generic runtime was not CUDA")
    result = {"schema_version": 1, "status": "complete_and_verified",
              "rows": count, "fit_sha256": EXPECTED_FIT_SHA256,
              "predictions": str(args.predictions.resolve()),
              "predictions_sha256": sha256_file(args.predictions),
              "runtime": runtime, "used_dev3000": False, "used_test": False}
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.output_root / "run_manifest.json"
    write_json(manifest, result)
    write_json(args.output_root / "artifact_checksums.json", {
        "finalizer": sha256_file(Path(__file__)), "generator": sha256_file(args.generator),
        "helper": sha256_file(args.helper), "predictions.jsonl": result["predictions_sha256"],
        "run_manifest.json": sha256_file(manifest), "used_dev3000": False, "used_test": False})
    print(json.dumps({"status": result["status"], "rows": count,
                      "predictions_sha256": result["predictions_sha256"],
                      "runtime": runtime["runtime"]}, indent=2))


if __name__ == "__main__":
    main()
