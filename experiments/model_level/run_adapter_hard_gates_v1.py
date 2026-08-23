"""Run pre-training correctness gates for Model-Level Adapter V1.

This runner uses only frozen Train-Fit rows.  It never reads Dev3000 or Test.
It performs zero-init identity, Adapter-only gradient, exact target-loss,
short overfit, and Adapter-only save/reload gates before formal training.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import torch

from src.model_level.concat_training import (
    collate_concat_examples,
    compute_target_loss,
    per_example_log_probabilities,
    prepare_concat_example,
)
from src.model_level.pinyin_modes import choose_pinyin_representation
from src.model_level.pinyingpt_adapter import (
    SerialAdapterConfig,
    gradient_audit,
    install_adapters,
    load_adapter_bundle,
    save_adapter_bundle,
    set_adapter_training_mode,
)
from src.reference_backend_pinyingpt.backend import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    OFFICIAL_CODE_REVISION,
    PinyinGPTConcatBackend,
)


EXPERIMENT = "model_level_adapter_v1_hard_gates"
EXPECTED_FIT_SHA256 = "547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6"
EXPECTED_FIT_ROWS = 144_526
EXPECTED_AUTHOR = "Agent Phage"
EXPECTED_AUTHOR_ROWS = 55_926
DEFAULT_SEED = 20_260_822


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_gate_rows(path: Path, backend: Any, *, author: str) -> tuple[list[dict[str, Any]], int, int]:
    """Select early, short-context rows for implementation gates only."""

    rows = 0
    author_rows = 0
    single: list[dict[str, Any]] = []
    multi: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            if str(row.get("source_split", "")).lower() == "test":
                raise RuntimeError("STOP: Test row detected in Train-Fit input")
            if str(row.get("author")) != author:
                continue
            author_rows += 1
            if (
                str(row.get("condition")) != "full_short"
                or str(row.get("target_type")) != "short"
                or str(row.get("standardized_partition")) != "train_fit"
            ):
                raise RuntimeError("Train-Fit row violates the frozen Full+Short schema")
            segments = tuple(map(str, row["pinyin_segments"]))
            if len(str(row["gold"])) != len(segments):
                raise RuntimeError("Gold/Pinyin alignment failure")
            need_single = len(segments) == 1 and len(single) < 2
            need_multi = len(segments) >= 2 and len(multi) < 1
            if not need_single and not need_multi:
                continue
            # Gate rows are intentionally small to make CPU debugging practical.
            context_tokens = len(backend.tokenizer.encode(str(row["context"]), add_special_tokens=False))
            if context_tokens > 512:
                continue
            if need_single:
                single.append(row)
            elif need_multi:
                multi.append(row)
    if rows != EXPECTED_FIT_ROWS:
        raise RuntimeError(f"Train-Fit row count changed: {rows}")
    if author == EXPECTED_AUTHOR and author_rows != EXPECTED_AUTHOR_ROWS:
        raise RuntimeError(f"Agent Phage Train-Fit count changed: {author_rows}")
    if len(single) < 2 or not multi:
        raise RuntimeError("could not select Full/Initial/Mixed hard-gate rows")
    return [single[0], single[1], multi[0]], rows, author_rows


def build_gate_examples(backend: Any, rows: Sequence[Mapping[str, Any]], *, seed: int) -> list[Any]:
    modes = ("full", "initial", "mixed")
    examples = []
    for row, mode in zip(rows, modes):
        representation = choose_pinyin_representation(
            row["pinyin_segments"],
            row_id=str(row["row_id"]),
            epoch=0,
            seed=seed,
            forced_mode=mode,  # implementation gate, not a training exposure
        )
        example = prepare_concat_example(backend, row, representation)
        # score_candidates includes the final Gold input token.  Keep one-token
        # slack so the exact scorer regression remains legal at n_positions.
        if len(example.input_ids) + 1 > int(backend.model.config.n_positions):
            raise RuntimeError("hard-gate scorer row has no one-token regression slack")
        examples.append(example)
    if examples[-1].pinyin.effective_mode != "mixed":
        raise AssertionError("multi-syllable forced Mixed gate became degenerate")
    return examples


def generate_fixed(backend: Any, examples: Sequence[Any]) -> list[dict[str, Any]]:
    outputs = []
    backend.model.eval()
    for example in examples:
        result = backend.generate(
            example.used_context,
            example.pinyin.segmented_input,
            top_k=10,
            beam_size=16,
        )
        outputs.append(
            {
                "row_id": example.row_id,
                "mode": example.pinyin.effective_mode,
                "segments": list(example.pinyin.segmented_input),
                "candidates": [candidate.to_dict() for candidate in result.candidates],
            }
        )
    return outputs


def compare_predictions(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    score_tolerance: float,
) -> dict[str, Any]:
    if len(left) != len(right):
        raise AssertionError("prediction population length changed")
    maximum_difference = 0.0
    for old, new in zip(left, right):
        if old["row_id"] != new["row_id"] or old["segments"] != new["segments"]:
            raise AssertionError("prediction row identity changed")
        old_candidates = old["candidates"]
        new_candidates = new["candidates"]
        if [item["text"] for item in old_candidates] != [item["text"] for item in new_candidates]:
            raise AssertionError(f"candidate text/order changed for {old['row_id']}")
        for old_item, new_item in zip(old_candidates, new_candidates):
            difference = abs(float(old_item["log_probability"]) - float(new_item["log_probability"]))
            maximum_difference = max(maximum_difference, difference)
    if maximum_difference > score_tolerance:
        raise AssertionError(
            f"candidate scores changed by {maximum_difference}, tolerance={score_tolerance}"
        )
    return {
        "candidate_texts_exact": True,
        "candidate_order_exact": True,
        "maximum_absolute_log_probability_difference": maximum_difference,
        "score_tolerance": score_tolerance,
        "passed": True,
    }


def scorer_regression(backend: Any, examples: Sequence[Any]) -> dict[str, Any]:
    differences = []
    details = []
    for example in examples:
        batch = collate_concat_examples(
            [example],
            pad_token_id=backend.tokenizer.pad_token_id,
            device=backend.device,
        )
        manual = per_example_log_probabilities(backend.model, batch)[0]
        reference = backend.score_candidates(
            example.used_context,
            example.pinyin.segmented_input,
            [example.target],
        )[0].log_probability
        difference = abs(manual - reference)
        differences.append(difference)
        details.append(
            {
                "row_id": example.row_id,
                "mode": example.pinyin.effective_mode,
                "manual_log_probability": manual,
                "score_candidates_log_probability": reference,
                "absolute_difference": difference,
            }
        )
    tolerance = 1e-5
    maximum = max(differences)
    if maximum > tolerance:
        raise AssertionError(f"Concat target loss does not reproduce score_candidates: {maximum}")
    return {"passed": True, "tolerance": tolerance, "maximum_absolute_difference": maximum, "rows": details}


def sha256_optional(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--author", default=EXPECTED_AUTHOR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke-steps", type=int, default=8)
    parser.add_argument("--smoke-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--platform-reference",
        type=Path,
        help="Windows generic_platform_reference.json to verify on the cluster",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_steps < 2 or args.smoke_learning_rate <= 0:
        raise ValueError("smoke gate requires at least two positive-learning-rate steps")
    fit_sha = sha256_file(args.fit)
    if fit_sha != EXPECTED_FIT_SHA256:
        raise RuntimeError(f"Train-Fit SHA mismatch: {fit_sha}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    backend = PinyinGPTConcatBackend(args.checkpoint, device=args.device)
    rows, fit_rows, author_rows = load_gate_rows(args.fit, backend, author=args.author)
    examples = build_gate_examples(backend, rows, seed=args.seed)
    baseline_predictions = generate_fixed(backend, examples)
    platform_reference_payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "purpose": "frozen_generic_windows_cluster_equivalence_reference",
        "checkpoint": CHECKPOINT_ID,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "official_code_revision": OFFICIAL_CODE_REVISION,
        "train_fit_sha256": fit_sha,
        "author": args.author,
        "seed": args.seed,
        "predictions": baseline_predictions,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(backend.device),
            "device_name": (
                torch.cuda.get_device_name(backend.device)
                if backend.device.type == "cuda"
                else "CPU"
            ),
        },
        "used_dev3000": False,
        "used_test": False,
    }
    platform_reference_path = args.output_root / "generic_platform_reference.json"
    platform_reference_path.write_text(
        json.dumps(platform_reference_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.platform_reference:
        expected_platform = json.loads(args.platform_reference.read_text(encoding="utf-8"))
        if (
            expected_platform.get("checkpoint_revision") != CHECKPOINT_REVISION
            or expected_platform.get("train_fit_sha256") != fit_sha
            or expected_platform.get("author") != args.author
            or int(expected_platform.get("seed")) != args.seed
        ):
            raise RuntimeError("platform reference provenance mismatch")
        platform_equivalence = {
            "mode": "comparison",
            "reference_path": str(args.platform_reference),
            **compare_predictions(
                expected_platform["predictions"],
                baseline_predictions,
                score_tolerance=1e-4,
            ),
        }
    else:
        platform_equivalence = {
            "mode": "reference_created",
            "reference_path": str(platform_reference_path),
            "passed": None,
        }

    adapter_config = SerialAdapterConfig()
    parameter_audit = install_adapters(backend.model, adapter_config)
    if parameter_audit["adapter_parameters"] != 894_528:
        raise AssertionError(f"unexpected Adapter parameter count: {parameter_audit}")
    zero_predictions = generate_fixed(backend, examples)
    identity = compare_predictions(
        baseline_predictions,
        zero_predictions,
        score_tolerance=1e-6,
    )
    target_loss_regression = scorer_regression(backend, examples)

    set_adapter_training_mode(backend.model, enabled=True)
    batch = collate_concat_examples(
        examples,
        pad_token_id=backend.tokenizer.pad_token_id,
        device=backend.device,
    )
    trainable = [parameter for parameter in backend.model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.smoke_learning_rate, weight_decay=0.0)
    smoke_losses: list[float] = []
    first_gradient_audit: dict[str, Any] | None = None
    for step in range(args.smoke_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_target_loss(backend.model, batch)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite smoke loss")
        loss.backward()
        if step == 0:
            first_gradient_audit = gradient_audit(backend.model)
            if not first_gradient_audit["adapter_gradient_present"]:
                raise AssertionError("Adapter gradients are absent")
            if not first_gradient_audit["adapter_nonzero_gradient_present"]:
                raise AssertionError("every Adapter gradient is numerically zero")
            if not first_gradient_audit["base_gradient_absent"]:
                raise AssertionError("frozen base received gradients")
        optimizer.step()
        smoke_losses.append(float(loss.detach().cpu()))
    if not smoke_losses[-1] < smoke_losses[0]:
        raise AssertionError(f"smoke loss did not decrease: {smoke_losses}")

    set_adapter_training_mode(backend.model, enabled=False)
    trained_predictions = generate_fixed(backend, examples)
    weights_path = args.output_root / "agent_phage_smoke_adapter_v1.safetensors"
    save_adapter_bundle(
        backend.model,
        weights_path,
        config=adapter_config,
        metadata={
            "experiment": EXPERIMENT,
            "author": args.author,
            "checkpoint": CHECKPOINT_ID,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "official_code_revision": OFFICIAL_CODE_REVISION,
            "train_fit_sha256": fit_sha,
            "seed": args.seed,
            "formal_training_checkpoint": False,
            "smoke_gate_only": True,
            "used_dev3000": False,
            "used_test": False,
        },
    )
    manifest_path = args.output_root / "hard_gate_rows.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as destination:
        for example in examples:
            destination.write(canonical_json(example.audit_dict()) + "\n")

    # Release the trained instance before loading a second full frozen base.
    del optimizer, batch, trainable
    del backend
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reloaded = PinyinGPTConcatBackend(args.checkpoint, device=args.device)
    reloaded_parameter_audit = install_adapters(reloaded.model, adapter_config)
    loaded_metadata = load_adapter_bundle(reloaded.model, weights_path)
    if (
        loaded_metadata.get("train_fit_sha256") != fit_sha
        or loaded_metadata.get("checkpoint_revision") != CHECKPOINT_REVISION
        or loaded_metadata.get("author") != args.author
    ):
        raise RuntimeError("reloaded Adapter metadata does not match the frozen run")
    set_adapter_training_mode(reloaded.model, enabled=False)
    reloaded_examples = build_gate_examples(reloaded, rows, seed=args.seed)
    reloaded_predictions = generate_fixed(reloaded, reloaded_examples)
    save_reload = compare_predictions(
        trained_predictions,
        reloaded_predictions,
        score_tolerance=1e-6,
    )

    runner_path = Path(__file__).resolve()
    checkpoint_files = {
        name: sha256_optional(args.checkpoint / name)
        for name in (
            "pytorch_model.bin",
            "config.json",
            "vocab.txt",
            "pinyin2char.json",
            "additional_special_tokens.json",
        )
    }
    result = {
        "schema_version": 1,
        "status": "complete",
        "experiment": EXPERIMENT,
        "author": args.author,
        "fit_rows": fit_rows,
        "author_fit_rows": author_rows,
        "fit_sha256": fit_sha,
        "gate_row_ids": [example.row_id for example in examples],
        "gate_modes": [example.pinyin.effective_mode for example in examples],
        "parameter_audit": parameter_audit,
        "reloaded_parameter_audit": reloaded_parameter_audit,
        "zero_init_identity_gate": identity,
        "target_loss_scorer_regression_gate": target_loss_regression,
        "gradient_gate": first_gradient_audit,
        "smoke_training_gate": {
            "steps": args.smoke_steps,
            "learning_rate": args.smoke_learning_rate,
            "losses": smoke_losses,
            "finite": all(math.isfinite(value) for value in smoke_losses),
            "loss_decreased": smoke_losses[-1] < smoke_losses[0],
            "passed": True,
        },
        "save_reload_gate": save_reload,
        "adapter_metadata_identity_gate": {"passed": True},
        "generic_platform_equivalence_gate": platform_equivalence,
        "generic_platform_reference_sha256": sha256_file(platform_reference_path),
        "adapter_weights_sha256": sha256_file(weights_path),
        "adapter_metadata_sha256": sha256_file(weights_path.with_suffix(".json")),
        "gate_manifest_sha256": sha256_file(manifest_path),
        "runner_sha256": sha256_file(runner_path),
        "checkpoint_file_sha256": checkpoint_files,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(reloaded.device),
            "device_name": (
                torch.cuda.get_device_name(reloaded.device)
                if reloaded.device.type == "cuda"
                else "CPU"
            ),
        },
        "base_dropout_training_mode": False,
        "adapter_dropout": 0.0,
        "formal_training_run": False,
        "used_dev3000": False,
        "used_test": False,
    }
    result_path = args.output_root / "hard_gate_result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
