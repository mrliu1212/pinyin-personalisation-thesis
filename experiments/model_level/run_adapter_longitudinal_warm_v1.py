from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from experiments.model_level import run_adapter_dev_evaluation_v1 as eval_base
from experiments.model_level import run_adapter_training_v1 as train_base
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


EXPERIMENT = "model_level_adapter_longitudinal_warm_v1"

AUTHOR = "Agent Phage"
SEED = 20260822
BATCH_SIZE = 8
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0

EXPECTED_PROTOCOL_SHA256 = (
    "dfaf14e0e47af27bb173edc67632e902"
    "b5030e31b275d9e4821d76c7f553bc3d"
)

EXPECTED_FIT_SHA256 = (
    "547a4f8179f5d664a862188823659993"
    "8a2f967f055ef0c262be658b3500c8a6"
)

EXPECTED_VAL_SHA256 = (
    "d7ae1cc21ee029dde8458189b9dc7a0"
    "989b2b3a372627e079c3e2699307f2220"
)

EXPECTED_TRAIN_ROWS = {
    1: 5805,
    2: 5806,
    3: 5805,
    4: 5805,
    5: 5806,
    6: 5805,
}

EXPECTED_PROBE_ROWS = 500
EXPECTED_STEPS = 726


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def load_manifest_ids(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, list):
        raise RuntimeError(f"Manifest is not a JSON list: {path}")

    ids = [str(x) for x in value]

    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate row_id in {path}")

    return ids


def load_population(
    fit_path: Path,
    val_path: Path,
) -> dict[str, dict[str, Any]]:
    if sha256_file(fit_path) != EXPECTED_FIT_SHA256:
        raise RuntimeError("Train-Fit SHA mismatch")

    if sha256_file(val_path) != EXPECTED_VAL_SHA256:
        raise RuntimeError("Train-Val SHA mismatch")

    rows: dict[str, dict[str, Any]] = {}

    for path, partition in (
        (fit_path, "train_fit"),
        (val_path, "train_val"),
    ):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                row = json.loads(line)

                if str(row.get("source_split", "")).lower() == "test":
                    raise RuntimeError("STOP: Test row detected")

                if str(row.get("author")) != AUTHOR:
                    continue

                if str(row.get("condition")) != "full_short":
                    raise RuntimeError(
                        f"Condition violation: {row.get('row_id')}"
                    )

                if str(row.get("target_type")) != "short":
                    raise RuntimeError(
                        f"Target violation: {row.get('row_id')}"
                    )

                expected_partition = str(
                    row.get("standardized_partition")
                )

                if expected_partition != partition:
                    raise RuntimeError(
                        f"Partition violation: {row.get('row_id')}"
                    )

                row_id = str(row["row_id"])

                if row_id in rows:
                    raise RuntimeError(
                        f"Duplicate row_id across population: {row_id}"
                    )

                rows[row_id] = row

    if len(rows) != 69667:
        raise RuntimeError(
            f"Unexpected nominal Agent population: {len(rows)}"
        )

    return rows


def select_rows(
    population: Mapping[str, dict[str, Any]],
    manifest: Path,
) -> list[dict[str, Any]]:
    ids = load_manifest_ids(manifest)

    selected = []

    for row_id in ids:
        row = population.get(row_id)

        if row is None:
            raise RuntimeError(
                f"Manifest row not in Agent population: {row_id}"
            )

        selected.append(row)

    return selected


def configure_determinism(seed: int) -> None:
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(
        True,
        warn_only=False,
    )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")


def train_one_cycle(
    backend: PinyinGPTConcatBackend,
    rows: list[dict[str, Any]],
    *,
    cycle: int,
    output_root: Path,
) -> dict[str, Any]:
    if len(rows) != EXPECTED_TRAIN_ROWS[cycle]:
        raise RuntimeError(
            f"Cycle {cycle} row mismatch: {len(rows)}"
        )

    expected_steps = math.ceil(
        len(rows) / BATCH_SIZE
    )

    if expected_steps != EXPECTED_STEPS:
        raise RuntimeError(
            f"Cycle {cycle} step mismatch: {expected_steps}"
        )

    # Important experimental definition:
    # Adapter parameters persist across cycles,
    # optimizer state does NOT.
    set_adapter_training_mode(
        backend.model,
        enabled=True,
    )

    trainable = [
        p
        for p in backend.model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    ordered = train_base.epoch_order(
        rows,
        epoch=0,
        seed=SEED,
    )

    history_path = (
        output_root
        / f"cycle_{cycle:02d}"
        / "training_history.jsonl"
    )

    history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    global_step = 0
    processed_rows = 0
    supervised_tokens = 0
    weighted_loss = 0.0
    first_gradient_audit = None
    first_step_loss = None
    final_step_loss = None
    started = time.time()

    batch_examples = []

    with history_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as history:
        for row_index, row in enumerate(ordered):

            # Frozen longitudinal baseline = Full-only.
            representation = choose_pinyin_representation(
                row["pinyin_segments"],
                row_id=str(row["row_id"]),
                epoch=0,
                seed=SEED,
                forced_mode="full",
                weights=(1, 0, 0),
            )

            example = prepare_concat_example(
                backend,
                row,
                representation,
            )

            batch_examples.append(example)

            is_last = row_index + 1 == len(ordered)

            if (
                len(batch_examples) < BATCH_SIZE
                and not is_last
            ):
                continue

            batch = collate_concat_examples(
                batch_examples,
                pad_token_id=(
                    backend.tokenizer.pad_token_id
                ),
                device=backend.device,
            )

            optimizer.zero_grad(set_to_none=True)

            loss = compute_target_loss(
                backend.model,
                batch,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss cycle={cycle} "
                    f"step={global_step + 1}"
                )

            loss.backward()

            if first_gradient_audit is None:
                first_gradient_audit = gradient_audit(
                    backend.model
                )

                if not first_gradient_audit[
                    "adapter_gradient_present"
                ]:
                    raise AssertionError(
                        "Adapter gradients absent"
                    )

                if not first_gradient_audit[
                    "adapter_nonzero_gradient_present"
                ]:
                    raise AssertionError(
                        "Adapter gradients numerically zero"
                    )

                if not first_gradient_audit[
                    "base_gradient_absent"
                ]:
                    raise AssertionError(
                        "Frozen base received gradients"
                    )

            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    MAX_GRAD_NORM,
                ).detach().cpu()
            )

            optimizer.step()

            global_step += 1

            loss_value = float(
                loss.detach().cpu()
            )

            batch_rows = len(batch_examples)
            batch_tokens = int(
                batch["supervised_tokens"]
            )

            if first_step_loss is None:
                first_step_loss = loss_value

            final_step_loss = loss_value
            processed_rows += batch_rows
            supervised_tokens += batch_tokens
            weighted_loss += (
                loss_value * batch_tokens
            )

            history.write(
                json.dumps(
                    {
                        "cycle": cycle,
                        "step": global_step,
                        "batch_rows": batch_rows,
                        "processed_rows": processed_rows,
                        "supervised_tokens": batch_tokens,
                        "loss": loss_value,
                        "gradient_norm_before_clip": (
                            grad_norm
                        ),
                        "elapsed_seconds": (
                            time.time() - started
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

            if (
                global_step == 1
                or global_step % 100 == 0
                or global_step == EXPECTED_STEPS
            ):
                print(
                    f"cycle={cycle} "
                    f"step={global_step}/{EXPECTED_STEPS} "
                    f"rows={processed_rows}/{len(rows)} "
                    f"loss={loss_value:.6f}",
                    flush=True,
                )

            batch_examples = []

    if global_step != EXPECTED_STEPS:
        raise AssertionError(
            f"Cycle {cycle}: {global_step} "
            f"!= {EXPECTED_STEPS}"
        )

    if processed_rows != len(rows):
        raise AssertionError(
            f"Cycle {cycle}: processed rows mismatch"
        )

    set_adapter_training_mode(
        backend.model,
        enabled=False,
    )

    return {
        "cycle": cycle,
        "rows": processed_rows,
        "optimizer_steps": global_step,
        "supervised_tokens": supervised_tokens,
        "token_weighted_mean_loss": (
            weighted_loss / supervised_tokens
        ),
        "first_step_loss": first_step_loss,
        "final_step_loss": final_step_loss,
        "first_gradient_audit": first_gradient_audit,
        "optimizer_state_persisted": False,
        "adapter_state_persisted": True,
    }


def evaluate_probe(
    backend: PinyinGPTConcatBackend,
    rows: list[dict[str, Any]],
    *,
    cycle: int,
    output_root: Path,
) -> dict[str, Any]:
    if len(rows) != EXPECTED_PROBE_ROWS:
        raise RuntimeError(
            f"P{cycle}: expected 500 rows, got {len(rows)}"
        )

    set_adapter_training_mode(
        backend.model,
        enabled=False,
    )

    eval_dir = (
        output_root
        / f"cycle_{cycle:02d}"
        / "probe"
    )

    predictions = eval_base.evaluate_backend(
        backend,
        rows,
        condition=f"warm_s{cycle}_p{cycle}",
        output_path=(
            eval_dir / "predictions.jsonl"
        ),
        top_k=10,
        beam_size=16,
        log_every=100,
    )

    metrics = eval_base.compute_metrics(
        predictions
    )

    write_json(
        eval_dir / "metrics.json",
        {
            "cycle": cycle,
            "rows": len(rows),
            "metrics": metrics,
            "used_test": False,
        },
    )

    micro = metrics["micro"]

    print(
        f"P{cycle}: "
        f"Top1={micro['top1']:.6f} "
        f"Top3={micro['top3']:.6f} "
        f"MRR={micro['mrr_at_10']:.6f} "
        f"Missing={micro['missing_at_10_rate']:.6f}",
        flush=True,
    )

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--train-fit",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--train-val",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--protocol-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    protocol_path = (
        args.protocol_root / "frozen_protocol.json"
    )

    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)

    protocol_sha = sha256_file(protocol_path)

    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            "Frozen longitudinal protocol SHA mismatch: "
            f"{protocol_sha}"
        )

    protocol = json.loads(
        protocol_path.read_text(
            encoding="utf-8"
        )
    )

    if protocol.get("status") != "FROZEN":
        raise RuntimeError(
            "Protocol is not FROZEN"
        )

    if protocol["population"]["test_used"]:
        raise RuntimeError(
            "STOP: frozen protocol says Test used"
        )

    population = load_population(
        args.train_fit,
        args.train_val,
    )

    train_manifests = {}
    probe_manifests = {}

    all_train = set()
    all_probe = set()

    for cycle in range(1, 7):
        train_path = (
            args.protocol_root
            / f"cycle_{cycle:02d}_train_row_ids.json"
        )

        probe_path = (
            args.protocol_root
            / f"cycle_{cycle:02d}_probe_row_ids.json"
        )

        train_rows = select_rows(
            population,
            train_path,
        )

        probe_rows = select_rows(
            population,
            probe_path,
        )

        if len(train_rows) != EXPECTED_TRAIN_ROWS[cycle]:
            raise RuntimeError(
                f"Cycle {cycle} train manifest mismatch"
            )

        if len(probe_rows) != EXPECTED_PROBE_ROWS:
            raise RuntimeError(
                f"Cycle {cycle} probe manifest mismatch"
            )

        train_ids = {
            str(x["row_id"])
            for x in train_rows
        }

        probe_ids = {
            str(x["row_id"])
            for x in probe_rows
        }

        if train_ids & probe_ids:
            raise RuntimeError(
                f"Cycle {cycle} train/probe overlap"
            )

        if all_train & train_ids:
            raise RuntimeError(
                f"Train overlap at cycle {cycle}"
            )

        if all_probe & probe_ids:
            raise RuntimeError(
                f"Probe overlap at cycle {cycle}"
            )

        all_train.update(train_ids)
        all_probe.update(probe_ids)

        train_manifests[cycle] = train_rows
        probe_manifests[cycle] = probe_rows

    if all_train & all_probe:
        raise RuntimeError(
            "Global train/probe overlap"
        )

    if len(all_train) != 34832:
        raise RuntimeError(
            f"Train union mismatch: {len(all_train)}"
        )

    if len(all_probe) != 3000:
        raise RuntimeError(
            f"Probe union mismatch: {len(all_probe)}"
        )

    print(
        "===== LONGITUDINAL WARM PROTOCOL GATE ====="
    )
    print("protocol_sha256 =", protocol_sha)
    print("train_rows      =", len(all_train))
    print("probe_rows      =", len(all_probe))
    print("cycles          = 6")
    print("steps/cycle     = 726")
    print("total_steps     = 4356")
    print("optimizer       = reset each cycle")
    print("adapter_state   = persistent")
    print("training_pinyin = full_only")
    print("test_used       = false")

    if args.validate_only:
        print("STATUS = VALIDATED_ONLY")
        return

    configure_determinism(SEED)

    if (
        args.output_root
        / "longitudinal_result.json"
    ).exists():
        raise RuntimeError(
            "Refusing to overwrite completed run"
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    backend = PinyinGPTConcatBackend(
        args.checkpoint,
        device=args.device,
    )

    adapter_config = SerialAdapterConfig()

    parameter_audit = install_adapters(
        backend.model,
        adapter_config,
    )

    if (
        parameter_audit["adapter_parameters"]
        != 894_528
    ):
        raise AssertionError(
            f"Unexpected Adapter params: "
            f"{parameter_audit}"
        )

    cycle_results = []

    for cycle in range(1, 7):
        print()
        print(
            "=" * 60,
            flush=True,
        )
        print(
            f" CYCLE {cycle}/6",
            flush=True,
        )
        print(
            "=" * 60,
            flush=True,
        )

        training_result = train_one_cycle(
            backend,
            train_manifests[cycle],
            cycle=cycle,
            output_root=args.output_root,
        )

        checkpoint_path = (
            args.output_root
            / f"cycle_{cycle:02d}"
            / f"agent_phage_warm_s{cycle}.safetensors"
        )

        save_adapter_bundle(
            backend.model,
            checkpoint_path,
            config=adapter_config,
            metadata={
                "experiment": EXPERIMENT,
                "author": AUTHOR,
                "cycle": cycle,
                "checkpoint": CHECKPOINT_ID,
                "checkpoint_revision": CHECKPOINT_REVISION,
                "official_code_revision": OFFICIAL_CODE_REVISION,
                "protocol_sha256": protocol_sha,
                "seed": SEED,
                "training_pinyin_policy": "full_only",
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "max_grad_norm": MAX_GRAD_NORM,
                "optimizer": "AdamW",
                "optimizer_state_persisted": False,
                "adapter_state_persisted": True,
                "cycle_train_rows": (
                    EXPECTED_TRAIN_ROWS[cycle]
                ),
                "cycle_optimizer_steps": (
                    EXPECTED_STEPS
                ),
                "used_test": False,
            },
        )

        probe_metrics = evaluate_probe(
            backend,
            probe_manifests[cycle],
            cycle=cycle,
            output_root=args.output_root,
        )

        cycle_result = {
            "cycle": cycle,
            "training": training_result,
            "adapter_checkpoint": str(
                checkpoint_path
            ),
            "adapter_sha256": sha256_file(
                checkpoint_path
            ),
            "probe_metrics": probe_metrics,
        }

        cycle_results.append(
            cycle_result
        )

        write_json(
            args.output_root
            / f"cycle_{cycle:02d}"
            / "cycle_result.json",
            cycle_result,
        )

    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "complete",
        "author": AUTHOR,
        "protocol_sha256": protocol_sha,
        "cycles": cycle_results,
        "training": {
            "total_rows": 34832,
            "steps_per_cycle": 726,
            "total_optimizer_steps": 4356,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "pinyin_policy": "full_only",
            "adapter_state_persisted": True,
            "optimizer_state_persisted": False,
        },
        "evaluation": {
            "primary_probe_rows": 3000,
            "probe_rows_per_cycle": 500,
            "top_k": 10,
            "beam_size": 16,
        },
        "used_test": False,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
        },
    }

    write_json(
        args.output_root
        / "longitudinal_result.json",
        result,
    )

    print()
    print(
        "===== LONGITUDINAL WARM COMPLETE ====="
    )

    for x in cycle_results:
        m = x["probe_metrics"]["micro"]
        print(
            f"P{x['cycle']} "
            f"Top1={m['top1']:.6f} "
            f"Top3={m['top3']:.6f} "
            f"MRR={m['mrr_at_10']:.6f}"
        )

    print("TEST USED = false")


if __name__ == "__main__":
    main()
