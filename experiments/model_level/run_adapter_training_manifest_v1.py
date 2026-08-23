"""Exact row-ID manifest wrapper around the validated Adapter V1 trainer.

This wrapper changes only training-population selection and, optionally,
forces Full-only Pinyin exposure.  The canonical model, loss, optimizer,
batching, deterministic epoch ordering, Adapter implementation, checkpoint
validation and persistence remain in run_adapter_training_v1.py.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any

from transformers import AutoTokenizer

from experiments.model_level import run_adapter_training_policy_v1 as policy
from experiments.model_level import run_adapter_training_v1 as base


ORIGINAL_CHOOSE_POPULATION = base.choose_training_population
SELECTION_AUDIT: dict[str, Any] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_row_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = None

        for key in ("row_ids", "selected_row_ids", "ids"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                values = candidate
                break

        if values is None:
            candidates = [
                value
                for value in payload.values()
                if isinstance(value, list)
                and all(isinstance(item, str) for item in value)
            ]
            if len(candidates) == 1:
                values = candidates[0]

        if values is None:
            raise RuntimeError(
                f"Could not locate row-ID list in manifest {path}"
            )
    else:
        raise RuntimeError(
            f"Unsupported manifest payload type: {type(payload)!r}"
        )

    row_ids = [str(value) for value in values]

    if not row_ids:
        raise RuntimeError("Manifest contains no row IDs")

    if len(set(row_ids)) != len(row_ids):
        raise RuntimeError("Manifest contains duplicate row IDs")

    return row_ids


def make_manifest_selector(
    row_ids: list[str],
    tokenizer: Any,
    *,
    manifest_path: Path,
):
    signature = inspect.signature(ORIGINAL_CHOOSE_POPULATION)

    def patched(*args: Any, **kwargs: Any):
        bound = signature.bind_partial(*args, **kwargs)

        row_name = None
        for candidate in (
            "values",
            "rows",
            "author_rows",
            "population",
        ):
            if candidate in bound.arguments:
                row_name = candidate
                break

        if row_name is None:
            raise RuntimeError(
                "STOP: could not identify row population argument in "
                f"choose_training_population{signature}"
            )

        source_rows = list(bound.arguments[row_name])

        source_by_id: dict[str, Any] = {}
        for row in source_rows:
            row_id = str(row["row_id"])
            if row_id in source_by_id:
                raise RuntimeError(
                    f"Duplicate source row_id: {row_id}"
                )
            source_by_id[row_id] = row

        missing = [
            row_id for row_id in row_ids
            if row_id not in source_by_id
        ]
        if missing:
            raise RuntimeError(
                "Manifest contains IDs outside the selected author "
                f"population: {missing[:10]}"
            )

        max_rows = bound.arguments.get("max_rows")
        if max_rows is not None and int(max_rows) != len(row_ids):
            raise RuntimeError(
                f"--max-rows={max_rows} does not match manifest "
                f"size={len(row_ids)}"
            )

        selected = [source_by_id[row_id] for row_id in row_ids]

        incompatible = []
        for row in selected:
            compatible, reason = policy.tokenizer_target_compatibility(
                row,
                tokenizer,
            )
            if not compatible:
                incompatible.append(reason)

        # These manifests were constructed after the tokenizer gate.
        # Any incompatibility now means provenance drift and must stop.
        if incompatible:
            raise RuntimeError(
                "Frozen manifest is no longer tokenizer-compatible: "
                f"{incompatible[:5]}"
            )

        SELECTION_AUDIT.clear()
        SELECTION_AUDIT.update(
            {
                "selection": "exact_row_id_manifest",
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "manifest_rows": len(row_ids),
                "source_population_rows": len(source_rows),
                "effective_training_rows": len(selected),
                "first_row_id": row_ids[0],
                "last_row_id": row_ids[-1],
                "tokenizer_incompatible_rows": 0,
                "used_test": False,
            }
        )

        return selected

    return patched


def main() -> None:
    wrapper = argparse.ArgumentParser(add_help=False)

    wrapper.add_argument(
        "--selected-row-ids",
        type=Path,
        required=True,
    )
    wrapper.add_argument(
        "--training-pinyin-policy",
        choices=("full_only", "mixed_3_1_2"),
        default="full_only",
    )

    wrapper_args, remaining = wrapper.parse_known_args()

    checkpoint_raw = policy.cli_value(
        remaining,
        "--checkpoint",
    )
    output_root_raw = policy.cli_value(
        remaining,
        "--output-root",
    )

    if checkpoint_raw is None:
        raise RuntimeError("--checkpoint is required")

    if output_root_raw is None:
        raise RuntimeError("--output-root is required")

    checkpoint = Path(checkpoint_raw)
    output_root = Path(output_root_raw)

    row_ids = load_row_ids(wrapper_args.selected_row_ids)

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
    )

    base.choose_training_population = make_manifest_selector(
        row_ids,
        tokenizer,
        manifest_path=wrapper_args.selected_row_ids,
    )

    if wrapper_args.training_pinyin_policy == "full_only":
        base.choose_pinyin_representation = (
            policy.make_full_only_representation_selector()
        )

    sys.argv = [sys.argv[0], *remaining]

    base.main()

    output_root.mkdir(parents=True, exist_ok=True)

    provenance = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "adapter_exact_manifest_training_v1",
        "training_pinyin_policy": (
            wrapper_args.training_pinyin_policy
        ),
        "selection_audit": SELECTION_AUDIT,
        "wrapper_sha256": sha256_file(
            Path(__file__).resolve()
        ),
        "base_runner_sha256": sha256_file(
            Path(base.__file__).resolve()
        ),
        "policy_runner_sha256": sha256_file(
            Path(policy.__file__).resolve()
        ),
        "used_test": False,
    }

    (
        output_root
        / "selection_manifest_provenance.json"
    ).write_text(
        json.dumps(
            provenance,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
