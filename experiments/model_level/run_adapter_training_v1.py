"""Train one per-user Model-Level Adapter V1 on frozen Train-Fit rows.

The same frozen PinyinGPT2-Concat base and exact Concat target-only loss used by
the hard gates are reused here. Dev3000 and Test are never read by this runner.
Pilot runs are distinguished from full formal Train-Fit runs in metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

import torch

from src.model_level.concat_training import (
    collate_concat_examples,
    compute_target_loss,
    prepare_concat_example,
)
from src.model_level.pinyin_modes import choose_pinyin_representation
from src.model_level.pinyingpt_adapter import (
    SerialAdapterConfig,
    gradient_audit,
    install_adapters,
    save_adapter_bundle,
    set_adapter_training_mode,
)
from src.reference_backend_pinyingpt.backend import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    OFFICIAL_CODE_REVISION,
    PinyinGPTConcatBackend,
)


EXPERIMENT = "model_level_adapter_v1_training"
EXPECTED_FIT_SHA256 = (
    "547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6"
)
EXPECTED_FIT_ROWS = 144_526
EXPECTED_AUTHOR_ROWS = {
    "Agent Phage": 55_926,
    "Etinjat": 32_906,
    "breaddddd": 55_694,
}
DEFAULT_SEED = 20_260_822


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_optional(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def stable_digest(
    tag: str,
    row_id: str,
    *,
    seed: int,
    epoch: int | None = None,
) -> bytes:
    payload = json.dumps(
        {
            "tag": tag,
            "row_id": row_id,
            "seed": seed,
            "epoch": epoch,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def author_slug(author: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", author.lower()).strip("_")
    return value or "author"


def load_author_rows(
    path: Path,
    *,
    author: str,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    total_rows = 0

    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue

            row = json.loads(line)
            total_rows += 1

            if str(row.get("source_split", "")).lower() == "test":
                raise RuntimeError("STOP: Test row detected in Train-Fit input")

            if str(row.get("author")) != author:
                continue

            if (
                str(row.get("condition")) != "full_short"
                or str(row.get("target_type")) != "short"
                or str(row.get("standardized_partition")) != "train_fit"
            ):
                raise RuntimeError(
                    f"Train-Fit schema violation for row {row.get('row_id')}"
                )

            segments = tuple(map(str, row["pinyin_segments"]))
            if len(str(row["gold"])) != len(segments):
                raise RuntimeError(
                    f"Gold/Pinyin alignment failure for row {row.get('row_id')}"
                )

            rows.append(row)

    if total_rows != EXPECTED_FIT_ROWS:
        raise RuntimeError(
            f"Train-Fit row count changed: {total_rows} != {EXPECTED_FIT_ROWS}"
        )

    expected = EXPECTED_AUTHOR_ROWS.get(author)
    if expected is not None and len(rows) != expected:
        raise RuntimeError(
            f"{author} Train-Fit row count changed: {len(rows)} != {expected}"
        )

    if not rows:
        raise RuntimeError(f"No Train-Fit rows found for author {author!r}")

    return rows, total_rows


def choose_training_population(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    max_rows: int | None,
) -> list[dict[str, Any]]:
    values = list(rows)

    if max_rows is None:
        return values

    if max_rows <= 0:
        raise ValueError("--max-rows must be positive")

    if max_rows > len(values):
        raise ValueError(
            f"--max-rows={max_rows} exceeds author population {len(values)}"
        )

    values.sort(
        key=lambda row: stable_digest(
            "training-subset",
            str(row["row_id"]),
            seed=seed,
        )
    )
    return values[:max_rows]


def epoch_order(
    rows: Sequence[dict[str, Any]],
    *,
    epoch: int,
    seed: int,
) -> list[dict[str, Any]]:
    values = list(rows)
    values.sort(
        key=lambda row: stable_digest(
            "epoch-order",
            str(row["row_id"]),
            seed=seed,
            epoch=epoch,
        )
    )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--author", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)

    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)

    parser.add_argument(
        "--max-rows",
        type=int,
        help="Deterministic Train-Fit subset for pilot/debug runs.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Optional global optimizer-step cap for pilot/debug runs.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative")
    if args.max_grad_norm < 0:
        raise ValueError("--max-grad-norm cannot be negative")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    fit_sha = sha256_file(args.fit)
    if fit_sha != EXPECTED_FIT_SHA256:
        raise RuntimeError(f"Train-Fit SHA mismatch: {fit_sha}")

    args.output_root.mkdir(parents=True, exist_ok=True)

    result_path = args.output_root / "training_result.json"
    if result_path.exists():
        raise RuntimeError(
            f"Refusing to overwrite completed output: {result_path}"
        )

    # Freeze stochastic and CUDA numerical surfaces for reproducible training.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    backend = PinyinGPTConcatBackend(
        args.checkpoint,
        device=args.device,
    )

    author_rows, fit_rows = load_author_rows(
        args.fit,
        author=args.author,
    )

    training_rows = choose_training_population(
        author_rows,
        seed=args.seed,
        max_rows=args.max_rows,
    )

    selected_row_ids = sorted(str(row["row_id"]) for row in training_rows)
    selected_population_sha256 = sha256_text(
        "\n".join(selected_row_ids) + "\n"
    )

    adapter_config = SerialAdapterConfig()
    parameter_audit = install_adapters(
        backend.model,
        adapter_config,
    )

    if parameter_audit["adapter_parameters"] != 894_528:
        raise AssertionError(
            f"Unexpected Adapter parameter count: {parameter_audit}"
        )

    set_adapter_training_mode(
        backend.model,
        enabled=True,
    )

    trainable = [
        parameter
        for parameter in backend.model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    config = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "author": args.author,
        "checkpoint": CHECKPOINT_ID,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "official_code_revision": OFFICIAL_CODE_REVISION,
        "train_fit_sha256": fit_sha,
        "fit_rows": fit_rows,
        "author_fit_rows": len(author_rows),
        "selected_training_rows": len(training_rows),
        "selected_population_sha256": selected_population_sha256,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "max_rows": args.max_rows,
        "max_steps": args.max_steps,
        "optimizer": "AdamW",
        "scheduler": None,
        "mixed_precision": False,
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "tf32_allowed": False,
        "cublas_workspace_config": __import__("os").environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "pinyin_mode_weights": [3, 1, 2],
        "one_underlying_row_per_epoch": True,
        "base_model_frozen": True,
        "base_dropout_training_mode": False,
        "adapter_dropout": 0.0,
        "used_dev3000": False,
        "used_test": False,
    }

    config_path = args.output_root / "training_config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    history_path = args.output_root / "training_history.jsonl"

    global_step = 0
    processed_rows = 0
    supervised_tokens = 0
    first_step_loss: float | None = None
    final_step_loss: float | None = None
    first_gradient_audit: dict[str, Any] | None = None

    requested_mode_counts = {
        "full": 0,
        "initial": 0,
        "mixed": 0,
    }
    effective_mode_counts = {
        "full": 0,
        "initial": 0,
        "mixed": 0,
    }

    epoch_summaries: list[dict[str, Any]] = []
    stop_training = False
    started = time.time()

    with history_path.open("w", encoding="utf-8", newline="\n") as history:
        for epoch in range(args.epochs):
            ordered = epoch_order(
                training_rows,
                epoch=epoch,
                seed=args.seed,
            )

            epoch_rows = 0
            epoch_tokens = 0
            epoch_weighted_loss = 0.0
            batch_examples = []

            for row_index, row in enumerate(ordered):
                representation = choose_pinyin_representation(
                    row["pinyin_segments"],
                    row_id=str(row["row_id"]),
                    epoch=epoch,
                    seed=args.seed,
                )

                example = prepare_concat_example(
                    backend,
                    row,
                    representation,
                )

                requested_mode_counts[representation.requested_mode] += 1
                effective_mode_counts[representation.effective_mode] += 1
                batch_examples.append(example)

                is_last_row = row_index + 1 == len(ordered)
                if (
                    len(batch_examples) < args.batch_size
                    and not is_last_row
                ):
                    continue

                batch = collate_concat_examples(
                    batch_examples,
                    pad_token_id=backend.tokenizer.pad_token_id,
                    device=backend.device,
                )

                optimizer.zero_grad(set_to_none=True)

                loss = compute_target_loss(
                    backend.model,
                    batch,
                )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss at epoch={epoch} step={global_step + 1}"
                    )

                loss.backward()

                if first_gradient_audit is None:
                    first_gradient_audit = gradient_audit(
                        backend.model
                    )
                    if not first_gradient_audit["adapter_gradient_present"]:
                        raise AssertionError(
                            "Adapter gradients are absent"
                        )
                    if not first_gradient_audit[
                        "adapter_nonzero_gradient_present"
                    ]:
                        raise AssertionError(
                            "Adapter gradients are numerically zero"
                        )
                    if not first_gradient_audit["base_gradient_absent"]:
                        raise AssertionError(
                            "Frozen base received gradients"
                        )

                if args.max_grad_norm > 0:
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            trainable,
                            args.max_grad_norm,
                        ).detach().cpu()
                    )
                else:
                    grad_norm = None

                optimizer.step()

                global_step += 1

                loss_value = float(loss.detach().cpu())
                batch_rows = len(batch_examples)
                batch_tokens = int(batch["supervised_tokens"])

                if first_step_loss is None:
                    first_step_loss = loss_value
                final_step_loss = loss_value

                processed_rows += batch_rows
                supervised_tokens += batch_tokens

                epoch_rows += batch_rows
                epoch_tokens += batch_tokens
                epoch_weighted_loss += loss_value * batch_tokens

                record = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "batch_rows": batch_rows,
                    "supervised_tokens": batch_tokens,
                    "loss": loss_value,
                    "gradient_norm_before_clip": grad_norm,
                    "elapsed_seconds": time.time() - started,
                }

                history.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

                if global_step == 1 or global_step % args.log_every == 0:
                    print(
                        f"epoch={epoch + 1}/{args.epochs} "
                        f"step={global_step} "
                        f"rows={processed_rows} "
                        f"loss={loss_value:.6f}",
                        flush=True,
                    )

                batch_examples = []

                if (
                    args.max_steps is not None
                    and global_step >= args.max_steps
                ):
                    stop_training = True
                    break

            epoch_average_loss = (
                epoch_weighted_loss / epoch_tokens
                if epoch_tokens
                else None
            )

            epoch_summaries.append(
                {
                    "epoch": epoch,
                    "processed_rows": epoch_rows,
                    "supervised_tokens": epoch_tokens,
                    "token_weighted_mean_loss": epoch_average_loss,
                    "completed_full_selected_population": (
                        epoch_rows == len(training_rows)
                    ),
                }
            )

            print(
                f"epoch={epoch + 1} complete "
                f"rows={epoch_rows} "
                f"mean_loss={epoch_average_loss}",
                flush=True,
            )

            if stop_training:
                break

    set_adapter_training_mode(
        backend.model,
        enabled=False,
    )

    pilot_run = (
        args.max_rows is not None
        or args.max_steps is not None
    )

    slug = author_slug(args.author)
    weights_path = (
        args.output_root
        / f"{slug}_adapter_v1.safetensors"
    )

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
            "selected_population_sha256": selected_population_sha256,
            "seed": args.seed,
            "formal_training_checkpoint": not pilot_run,
            "pilot_run": pilot_run,
            "epochs_requested": args.epochs,
            "epochs_completed": len(epoch_summaries),
            "global_steps": global_step,
            "processed_rows": processed_rows,
            "used_dev3000": False,
            "used_test": False,
        },
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
        "pilot_run": pilot_run,
        "formal_training_run": not pilot_run,
        "author": args.author,
        "fit_rows": fit_rows,
        "author_fit_rows": len(author_rows),
        "selected_training_rows": len(training_rows),
        "selected_population_sha256": selected_population_sha256,
        "fit_sha256": fit_sha,
        "seed": args.seed,
        "parameter_audit": parameter_audit,
        "first_gradient_audit": first_gradient_audit,
        "epochs_requested": args.epochs,
        "epochs_completed": len(epoch_summaries),
        "epoch_summaries": epoch_summaries,
        "global_steps": global_step,
        "processed_rows": processed_rows,
        "supervised_tokens": supervised_tokens,
        "first_step_loss": first_step_loss,
        "final_step_loss": final_step_loss,
        "requested_mode_counts": requested_mode_counts,
        "effective_mode_counts": effective_mode_counts,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "scheduler": None,
        "mixed_precision": False,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "tf32_allowed": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None,
        "cublas_workspace_config": __import__("os").environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "max_grad_norm": args.max_grad_norm,
        "adapter_weights_sha256": sha256_file(weights_path),
        "adapter_metadata_sha256": sha256_file(
            weights_path.with_suffix(".json")
        ),
        "training_config_sha256": sha256_file(config_path),
        "training_history_sha256": sha256_file(history_path),
        "runner_sha256": sha256_file(runner_path),
        "checkpoint_file_sha256": checkpoint_files,
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
            "elapsed_seconds": time.time() - started,
        },
        "base_dropout_training_mode": False,
        "adapter_dropout": 0.0,
        "used_dev3000": False,
        "used_test": False,
    }

    result_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("===== TRAINING RESULT =====")
    print("status:", result["status"])
    print("pilot_run:", result["pilot_run"])
    print("author:", result["author"])
    print("selected rows:", result["selected_training_rows"])
    print("processed rows:", result["processed_rows"])
    print("steps:", result["global_steps"])
    print("first loss:", result["first_step_loss"])
    print("final loss:", result["final_step_loss"])
    print("requested modes:", result["requested_mode_counts"])
    print("effective modes:", result["effective_mode_counts"])
    print("adapter:", weights_path)


if __name__ == "__main__":
    main()
