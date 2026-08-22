"""Smoke, select, and refit the frozen task-specific context bi-encoder."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from src.personalisation.task_specific_biencoder import (
    SEED,
    SharedContextEncoder,
    evaluate_groups,
    group_batch_loss,
    load_groups,
    refuse_closed_path,
    select_epoch,
    sha256_file,
    sha256_tree,
    write_json,
)


BASE_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
GRADIENT_CLIP = 1.0
EPOCHS = 2


def seed_everything() -> None:
    import torch

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random_seed = SEED
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def ordered_batches(groups: Sequence[Mapping[str, Any]], epoch: int) -> list[list[int]]:
    import random

    indices = list(range(len(groups)))
    random.Random(SEED + epoch).shuffle(indices)
    return [indices[start : start + BATCH_SIZE] for start in range(0, len(indices), BATCH_SIZE)]


def train(
    *,
    base_model: Path,
    groups: Sequence[Mapping[str, Any]],
    epochs: int,
    output_root: Path,
    evaluate: Sequence[Mapping[str, Any]] | None,
) -> tuple[SharedContextEncoder, list[dict[str, Any]]]:
    import torch
    from transformers import get_linear_schedule_with_warmup

    seed_everything()
    encoder = SharedContextEncoder(base_model, device="cuda")
    optimizer = torch.optim.AdamW(encoder.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    batches_per_epoch = math.ceil(len(groups) / BATCH_SIZE)
    updates_per_epoch = math.ceil(batches_per_epoch / GRADIENT_ACCUMULATION)
    total_updates = updates_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(total_updates * WARMUP_FRACTION),
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    records = []
    update = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        encoder.model.train()
        losses = []
        batches = ordered_batches(groups, epoch)
        for batch_number, indices in enumerate(batches, start=1):
            batch = [groups[index] for index in indices]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = group_batch_loss(encoder, batch)
            losses.append(float(loss.detach().cpu()))
            scaler.scale(loss / GRADIENT_ACCUMULATION).backward()
            should_step = batch_number % GRADIENT_ACCUMULATION == 0 or batch_number == len(batches)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(encoder.model.parameters(), GRADIENT_CLIP)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                    update += 1
                optimizer.zero_grad(set_to_none=True)
            if batch_number % 250 == 0 or batch_number == len(batches):
                elapsed = time.perf_counter() - started
                processed = (epoch - 1) * len(groups) + min(batch_number * BATCH_SIZE, len(groups))
                print(
                    f"train epoch={epoch}/{epochs} batch={batch_number}/{len(batches)} "
                    f"groups={processed:,} rate={processed/max(elapsed, 1e-9):.2f}/s "
                    f"loss={sum(losses[-100:])/len(losses[-100:]):.6f}",
                    flush=True,
                )
        checkpoint = output_root / f"epoch_{epoch}"
        encoder.save(checkpoint)
        checkpoint_sha, checkpoint_files = sha256_tree(checkpoint)
        record: dict[str, Any] = {
            "epoch": epoch,
            "mean_training_loss": sum(losses) / len(losses),
            "updates": update,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_files": checkpoint_files,
        }
        if evaluate is not None:
            metrics = evaluate_groups(encoder, evaluate)["overall"]
            record["metrics"] = metrics
            print(f"inner gate epoch={epoch}: {json.dumps(metrics, sort_keys=True)}", flush=True)
        records.append(record)
        write_json(output_root / "progress.json", {"epochs": records, "used_dev3000": False, "used_test": False})
    return encoder, records


def reload_check(encoder: SharedContextEncoder, checkpoint: Path, texts: Sequence[str]) -> float:
    import numpy as np

    before = encoder.embed(texts, batch_size=len(texts))
    reloaded = SharedContextEncoder(checkpoint, device="cuda")
    after = reloaded.embed(texts, batch_size=len(texts))
    difference = float(np.max(np.abs(before - after)))
    del reloaded
    return difference


def release(encoder: SharedContextEncoder) -> None:
    import torch

    del encoder.model
    del encoder
    gc.collect()
    torch.cuda.empty_cache()


def smoke(args: argparse.Namespace, groups: Sequence[Mapping[str, Any]], base_provenance: Mapping[str, Any]) -> None:
    smoke_groups = [row for row in groups if row["split"] == "inner_fit"][:8]
    output = args.output_root / "smoke"
    encoder, records = train(base_model=args.base_model, groups=smoke_groups, epochs=1, output_root=output, evaluate=smoke_groups)
    checkpoint = Path(records[-1]["checkpoint"])
    difference = reload_check(encoder, checkpoint, [row["query_context"] for row in smoke_groups[:4]])
    result = {
        "schema_version": 1,
        "status": "passed" if difference <= 1e-6 else "failed",
        "engineering_smoke_only": True,
        "groups": len(smoke_groups),
        "save_reload_max_abs_difference": difference,
        "base_model": base_provenance,
        "runtime": runtime_info(),
        "used_dev3000": False,
        "used_test": False,
    }
    write_json(output / "smoke_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result["status"] != "passed":
        raise RuntimeError("save/reload smoke check failed")


def runtime_info() -> dict[str, Any]:
    import torch
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def full(args: argparse.Namespace, groups: Sequence[Mapping[str, Any]], base_provenance: Mapping[str, Any]) -> None:
    inner_fit = [row for row in groups if row["split"] == "inner_fit"]
    inner_gate = [row for row in groups if row["split"] == "inner_gate"]
    started = time.perf_counter()
    gate_root = args.output_root / "inner_gate_training"
    encoder, gate_records = train(
        base_model=args.base_model,
        groups=inner_fit,
        epochs=EPOCHS,
        output_root=gate_root,
        evaluate=inner_gate,
    )
    selected = dict(select_epoch(gate_records))
    selected_epoch = int(selected["epoch"])
    release(encoder)
    print(f"selected epoch={selected_epoch} by frozen inner-gate rule", flush=True)

    final_root = args.output_root / "final_refit"
    final_encoder, final_records = train(
        base_model=args.base_model,
        groups=groups,
        epochs=selected_epoch,
        output_root=final_root,
        evaluate=None,
    )
    final_checkpoint = Path(final_records[-1]["checkpoint"])
    check_texts = [row["query_context"] for row in groups[:8]]
    reload_difference = reload_check(final_encoder, final_checkpoint, check_texts)
    final_sha, final_files = sha256_tree(final_checkpoint)
    result = {
        "schema_version": 1,
        "status": "complete" if reload_difference <= 1e-6 else "failed",
        "protocol": {
            "epochs_considered": [1, 2],
            "selection_rule": "inner-gate Macro-author Recall@1, Micro Recall@1, MRR, earlier epoch",
            "batch_size_groups": BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_fraction": WARMUP_FRACTION,
            "gradient_clip": GRADIENT_CLIP,
            "temperature": 0.05,
            "seed": SEED,
            "pooling": "attention-mask mean then L2",
            "context_chars": 64,
            "max_length": 128,
            "unrestricted_in_batch_negatives": False,
        },
        "population": {"all_train_fit_groups": len(groups), "inner_fit_groups": len(inner_fit), "inner_gate_groups": len(inner_gate)},
        "inner_gate_records": gate_records,
        "selected_epoch": selected_epoch,
        "final_refit_records": final_records,
        "final_checkpoint": str(final_checkpoint.resolve()),
        "final_checkpoint_sha256": final_sha,
        "final_checkpoint_files": final_files,
        "save_reload_max_abs_difference": reload_difference,
        "base_model": base_provenance,
        "runtime": runtime_info(),
        "runtime_seconds": time.perf_counter() - started,
        "used_dev3000": False,
        "used_test": False,
    }
    write_json(args.output_root / "training_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result["status"] != "complete":
        raise RuntimeError("final save/reload check failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.groups, args.audit, args.base_model):
        refuse_closed_path(path)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("used_dev3000") or audit.get("used_test"):
        raise ValueError("training input audit is not closed-data clean")
    audited_groups = load_groups(args.groups)
    if len(audited_groups) != 99_671 or any(row.get("used_dev3000") or row.get("used_test") for row in audited_groups):
        raise ValueError("prepared group population differs from the frozen audit")
    groups = [row for row in audited_groups if row.get("trainable")]
    if len(groups) != 66_672 or any(len(row["history_contexts"]) < 2 for row in groups):
        raise ValueError("trainable query-local group population differs from the frozen audit")
    base_sha, base_files = sha256_tree(args.base_model)
    base_provenance = {
        "model_id": "BAAI/bge-small-zh-v1.5",
        "revision": BASE_REVISION,
        "path": str(args.base_model.resolve()),
        "tree_sha256": base_sha,
        "files": base_files,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.phase == "smoke":
        smoke(args, groups, base_provenance)
    else:
        full(args, groups, base_provenance)


if __name__ == "__main__":
    main()
