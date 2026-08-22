"""Materialize compact author-free LambdaMART matrices from audited features."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.external_memory_next.audit_learned_fusion_inputs_v1 import (
    EXPECTED_FIT_ROWS,
    EXPECTED_VAL_ROWS,
    FEATURE_NAMES,
    extract_group,
    iter_jsonl,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--fit-supports", type=Path, required=True)
    parser.add_argument("--val-stage1", type=Path, required=True)
    parser.add_argument("--val-stage2", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit.get("status") != "complete" or audit.get("used_dev3000") is not False or audit.get("used_test") is not False:
        raise RuntimeError("Learned-fusion input audit is not complete and closed")
    if tuple(audit.get("feature_names", ())) != FEATURE_NAMES:
        raise RuntimeError("Audited runtime feature schema changed")
    fit_candidates = int(audit["fit"]["positive_group_candidates"])
    fit_groups = int(audit["fit"]["positive_groups"])
    val_candidates = int(audit["val"]["candidates"])
    val_groups = int(audit["val"]["groups"] - audit["val"]["empty_groups"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    fit_x = np.lib.format.open_memmap(args.output_root / "fit_X.npy", mode="w+", dtype="<f4",
                                      shape=(fit_candidates, len(FEATURE_NAMES)))
    fit_y = np.lib.format.open_memmap(args.output_root / "fit_y.npy", mode="w+", dtype="u1",
                                      shape=(fit_candidates,))
    fit_group = np.lib.format.open_memmap(args.output_root / "fit_group.npy", mode="w+", dtype="<i4",
                                          shape=(fit_groups,))
    fit_offset = fit_group_index = fit_rows = 0
    for row in iter_jsonl(args.fit_supports):
        vectors, labels, _ = extract_group(row, row)
        fit_rows += 1
        if not any(labels):
            continue
        size = len(vectors)
        fit_x[fit_offset:fit_offset + size] = np.asarray(vectors, dtype=np.float32)
        fit_y[fit_offset:fit_offset + size] = np.asarray(labels, dtype=np.uint8)
        fit_group[fit_group_index] = size
        fit_offset += size
        fit_group_index += 1
    if fit_rows != EXPECTED_FIT_ROWS or fit_offset != fit_candidates or fit_group_index != fit_groups:
        raise RuntimeError("Fit matrix allocation audit differs from materialized data")
    fit_x.flush(); fit_y.flush(); fit_group.flush()

    val_x = np.lib.format.open_memmap(args.output_root / "val_X.npy", mode="w+", dtype="<f4",
                                      shape=(val_candidates, len(FEATURE_NAMES)))
    val_y = np.lib.format.open_memmap(args.output_root / "val_y.npy", mode="w+", dtype="u1",
                                      shape=(val_candidates,))
    val_group = np.lib.format.open_memmap(args.output_root / "val_group.npy", mode="w+", dtype="<i4",
                                          shape=(val_groups,))
    meta_path = args.output_root / "val_query_meta.jsonl"
    val_offset = val_group_index = val_rows = 0
    inputs = (iter_jsonl(args.val_stage1), iter_jsonl(args.val_stage2), iter_jsonl(args.val_predictions))
    with meta_path.open("w", encoding="utf-8", newline="\n") as sink:
        for group in itertools.zip_longest(*inputs):
            feature, support, prediction = group
            if feature is None or support is None or prediction is None:
                raise RuntimeError("Val input counts differ")
            if len({str(row["row_id"]) for row in group}) != 1:
                raise RuntimeError(f"Val row order differs at {val_rows + 1}")
            vectors, labels, candidates = extract_group(feature, support)
            size = len(vectors)
            if size:
                val_x[val_offset:val_offset + size] = np.asarray(vectors, dtype=np.float32)
                val_y[val_offset:val_offset + size] = np.asarray(labels, dtype=np.uint8)
                val_group[val_group_index] = size
                matrix_group = val_group_index
                val_group_index += 1
            else:
                matrix_group = None
            sink.write(json.dumps({"row_id": feature["row_id"], "author": feature["author"],
                                   "gold": feature["gold"], "ambiguous": bool(feature["ambiguous"]),
                                   "conflict": bool(feature["conflict"]), "offset": val_offset,
                                   "candidate_count": size, "matrix_group": matrix_group,
                                   "candidates": candidates,
                                   "baseline_rank": prediction["RetunedFinal_rank"],
                                   "baseline_top10": prediction["RetunedFinal_top10"],
                                   "used_dev3000": False, "used_test": False},
                                  ensure_ascii=False, sort_keys=True) + "\n")
            val_offset += size
            val_rows += 1
    if val_rows != EXPECTED_VAL_ROWS or val_offset != val_candidates or val_group_index != val_groups:
        raise RuntimeError("Val matrix allocation audit differs from materialized data")
    val_x.flush(); val_y.flush(); val_group.flush()
    del fit_x, fit_y, fit_group, val_x, val_y, val_group
    paths = {name: args.output_root / name for name in
             ("fit_X.npy", "fit_y.npy", "fit_group.npy", "val_X.npy", "val_y.npy", "val_group.npy", "val_query_meta.jsonl")}
    manifest = {"schema_version": 1, "status": "complete",
                "feature_names": list(FEATURE_NAMES), "author_identity_feature": False,
                "gold_labels_are_separate_from_runtime_features": True,
                "fit": {"groups": fit_groups, "candidates": fit_candidates,
                        "zero_positive_groups_excluded": int(audit["fit"]["zero_positive_groups"])},
                "val": {"groups": EXPECTED_VAL_ROWS, "nonempty_matrix_groups": val_groups,
                        "candidates": val_candidates, "empty_groups": int(audit["val"]["empty_groups"])},
                "provenance": {"audit": {"path": str(args.audit.resolve()), "sha256": sha256_file(args.audit)},
                               "fit_supports": sha256_file(args.fit_supports),
                               "val_stage1": sha256_file(args.val_stage1),
                               "val_stage2": sha256_file(args.val_stage2),
                               "val_predictions": sha256_file(args.val_predictions)},
                "artifacts": {name: {"path": str(path.resolve()), "sha256": sha256_file(path),
                                     "bytes": path.stat().st_size} for name, path in paths.items()},
                "used_dev3000": False, "used_test": False}
    write_json(args.output_root / "matrix_manifest.json", manifest)
    write_json(args.output_root / "artifact_checksums.json", {
        "runner": sha256_file(Path(__file__)), "matrix_manifest.json": sha256_file(args.output_root / "matrix_manifest.json"),
        "used_dev3000": False, "used_test": False})
    print(json.dumps({"status": "complete", "fit": manifest["fit"], "val": manifest["val"],
                      "output": str(args.output_root / "matrix_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
