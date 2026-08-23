from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from experiments.model_level import run_adapter_longitudinal_warm_v1 as warm
from src.model_level.pinyingpt_adapter import (
    SerialAdapterConfig,
    install_adapters,
    load_adapter_bundle,
    save_adapter_bundle,
)


EXPERIMENT = "model_level_adapter_longitudinal_warm_resume_v1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cycle_dir(output_root: Path, cycle: int) -> Path:
    return output_root / f"cycle_{cycle:02d}"


def checkpoint_path(output_root: Path, cycle: int) -> Path:
    return (
        cycle_dir(output_root, cycle)
        / f"agent_phage_warm_s{cycle}.safetensors"
    )


def cycle_result_path(output_root: Path, cycle: int) -> Path:
    return cycle_dir(output_root, cycle) / "cycle_result.json"


def cycle_is_complete(
    output_root: Path,
    cycle: int,
) -> bool:
    result_path = cycle_result_path(
        output_root,
        cycle,
    )

    weights_path = checkpoint_path(
        output_root,
        cycle,
    )

    metadata_path = weights_path.with_suffix(".json")

    probe_predictions = (
        cycle_dir(output_root, cycle)
        / "probe"
        / "predictions.jsonl"
    )

    probe_metrics = (
        cycle_dir(output_root, cycle)
        / "probe"
        / "metrics.json"
    )

    required = (
        result_path,
        weights_path,
        metadata_path,
        probe_predictions,
        probe_metrics,
    )

    if not all(path.is_file() for path in required):
        return False

    try:
        result = read_json(result_path)

        if int(result.get("cycle", -1)) != cycle:
            return False

        training = result.get("training", {})

        if int(training.get("optimizer_steps", -1)) != warm.EXPECTED_STEPS:
            return False

        if int(training.get("rows", -1)) != warm.EXPECTED_TRAIN_ROWS[cycle]:
            return False

        if training.get("adapter_state_persisted") is not True:
            return False

        if training.get("optimizer_state_persisted") is not False:
            return False

        metrics = result.get("probe_metrics", {})

        micro = metrics.get("micro", {})

        if int(micro.get("interaction_count", -1)) != warm.EXPECTED_PROBE_ROWS:
            return False

        metadata = read_json(metadata_path)

        if metadata.get("protocol_sha256") != warm.EXPECTED_PROTOCOL_SHA256:
            return False

        if int(metadata.get("cycle", -1)) != cycle:
            return False

        if metadata.get("used_test") is not False:
            return False

    except Exception:
        return False

    return True


def archive_partial_cycle(
    output_root: Path,
    cycle: int,
) -> None:
    path = cycle_dir(
        output_root,
        cycle,
    )

    if not path.exists():
        return

    archived = output_root / (
        f"cycle_{cycle:02d}.incomplete.{int(time.time())}"
    )

    print(
        f"ARCHIVE PARTIAL CYCLE {cycle}: "
        f"{path} -> {archived}",
        flush=True,
    )

    shutil.move(
        str(path),
        str(archived),
    )


def load_manifests(
    population: dict[str, dict[str, Any]],
    protocol_root: Path,
):
    train_manifests = {}
    probe_manifests = {}

    all_train: set[str] = set()
    all_probe: set[str] = set()

    for cycle in range(1, 7):
        train_path = (
            protocol_root
            / f"cycle_{cycle:02d}_train_row_ids.json"
        )

        probe_path = (
            protocol_root
            / f"cycle_{cycle:02d}_probe_row_ids.json"
        )

        train_rows = warm.select_rows(
            population,
            train_path,
        )

        probe_rows = warm.select_rows(
            population,
            probe_path,
        )

        if len(train_rows) != warm.EXPECTED_TRAIN_ROWS[cycle]:
            raise RuntimeError(
                f"Cycle {cycle} training-row mismatch"
            )

        if len(probe_rows) != warm.EXPECTED_PROBE_ROWS:
            raise RuntimeError(
                f"Cycle {cycle} probe-row mismatch"
            )

        train_ids = {
            str(row["row_id"])
            for row in train_rows
        }

        probe_ids = {
            str(row["row_id"])
            for row in probe_rows
        }

        if train_ids & probe_ids:
            raise RuntimeError(
                f"Cycle {cycle} train/probe overlap"
            )

        if all_train & train_ids:
            raise RuntimeError(
                f"Training overlap at cycle {cycle}"
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

    return train_manifests, probe_manifests


def main() -> None:
    args = warm.parse_args()

    protocol_path = (
        args.protocol_root
        / "frozen_protocol.json"
    )

    protocol_sha = warm.sha256_file(
        protocol_path
    )

    if protocol_sha != warm.EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            "Frozen longitudinal protocol SHA mismatch"
        )

    protocol = read_json(
        protocol_path
    )

    if protocol.get("status") != "FROZEN":
        raise RuntimeError(
            "Protocol is not FROZEN"
        )

    if protocol["population"]["test_used"]:
        raise RuntimeError(
            "STOP: frozen protocol says Test used"
        )

    population = warm.load_population(
        args.train_fit,
        args.train_val,
    )

    train_manifests, probe_manifests = load_manifests(
        population,
        args.protocol_root,
    )

    print(
        "===== LONGITUDINAL WARM RESUME GATE ====="
    )
    print("protocol_sha256 =", protocol_sha)
    print("train_rows      = 34832")
    print("probe_rows      = 3000")
    print("cycles          = 6")
    print("steps/cycle     = 726")
    print("total_steps     = 4356")
    print("optimizer       = reset each cycle")
    print("adapter_state   = persistent")
    print("training_pinyin = full_only")
    print("resume          = cycle_level")
    print("test_used       = false")

    if args.validate_only:
        print("STATUS = VALIDATED_ONLY")
        return

    warm.configure_determinism(
        warm.SEED
    )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_result_path = (
        args.output_root
        / "longitudinal_result.json"
    )

    if final_result_path.is_file():
        final = read_json(
            final_result_path
        )

        if (
            final.get("status") == "complete"
            and len(final.get("cycles", [])) == 6
            and final.get("used_test") is False
        ):
            print(
                "STATUS = ALREADY COMPLETE"
            )
            return

        raise RuntimeError(
            "Existing longitudinal_result.json is not "
            "a valid completed run"
        )

    completed_cycles = []

    for cycle in range(1, 7):
        if cycle_is_complete(
            args.output_root,
            cycle,
        ):
            completed_cycles.append(cycle)
        else:
            break

    # A later complete cycle without all previous cycles
    # would violate the sequential state trajectory.
    for cycle in range(
        len(completed_cycles) + 2,
        7,
    ):
        if cycle_is_complete(
            args.output_root,
            cycle,
        ):
            raise RuntimeError(
                "Non-contiguous completed-cycle state detected"
            )

    next_cycle = len(completed_cycles) + 1

    print()
    print(
        "===== RESUME STATE ====="
    )
    print(
        "completed_cycles =",
        completed_cycles,
    )

    if next_cycle <= 6:
        print(
            "next_cycle       =",
            next_cycle,
        )
    else:
        print(
            "next_cycle       = none"
        )

    if next_cycle <= 6:
        archive_partial_cycle(
            args.output_root,
            next_cycle,
        )

    backend = warm.PinyinGPTConcatBackend(
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

    if completed_cycles:
        last_completed = completed_cycles[-1]

        resume_checkpoint = checkpoint_path(
            args.output_root,
            last_completed,
        )

        metadata = load_adapter_bundle(
            backend.model,
            resume_checkpoint,
        )

        if int(metadata["cycle"]) != last_completed:
            raise RuntimeError(
                "Resume checkpoint cycle mismatch"
            )

        if metadata["protocol_sha256"] != protocol_sha:
            raise RuntimeError(
                "Resume checkpoint protocol mismatch"
            )

        print(
            f"LOADED S{last_completed}: "
            f"{resume_checkpoint}",
            flush=True,
        )
    else:
        print(
            "STARTING FROM S0 ZERO-INIT",
            flush=True,
        )

    cycle_results = []

    for cycle in completed_cycles:
        cycle_results.append(
            read_json(
                cycle_result_path(
                    args.output_root,
                    cycle,
                )
            )
        )

    for cycle in range(
        next_cycle,
        7,
    ):
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

        training_result = warm.train_one_cycle(
            backend,
            train_manifests[cycle],
            cycle=cycle,
            output_root=args.output_root,
        )

        weights_path = checkpoint_path(
            args.output_root,
            cycle,
        )

        save_adapter_bundle(
            backend.model,
            weights_path,
            config=adapter_config,
            metadata={
                "experiment": EXPERIMENT,
                "author": warm.AUTHOR,
                "cycle": cycle,
                "checkpoint": warm.CHECKPOINT_ID,
                "checkpoint_revision": (
                    warm.CHECKPOINT_REVISION
                ),
                "official_code_revision": (
                    warm.OFFICIAL_CODE_REVISION
                ),
                "protocol_sha256": protocol_sha,
                "seed": warm.SEED,
                "training_pinyin_policy": "full_only",
                "batch_size": warm.BATCH_SIZE,
                "learning_rate": warm.LEARNING_RATE,
                "weight_decay": warm.WEIGHT_DECAY,
                "max_grad_norm": warm.MAX_GRAD_NORM,
                "optimizer": "AdamW",
                "optimizer_state_persisted": False,
                "adapter_state_persisted": True,
                "resume_semantics": "cycle_level",
                "cycle_train_rows": (
                    warm.EXPECTED_TRAIN_ROWS[cycle]
                ),
                "cycle_optimizer_steps": (
                    warm.EXPECTED_STEPS
                ),
                "used_test": False,
            },
        )

        probe_metrics = warm.evaluate_probe(
            backend,
            probe_manifests[cycle],
            cycle=cycle,
            output_root=args.output_root,
        )

        cycle_result = {
            "cycle": cycle,
            "training": training_result,
            "adapter_checkpoint": str(
                weights_path
            ),
            "adapter_sha256": warm.sha256_file(
                weights_path
            ),
            "probe_metrics": probe_metrics,
        }

        warm.write_json(
            cycle_result_path(
                args.output_root,
                cycle,
            ),
            cycle_result,
        )

        cycle_results.append(
            cycle_result
        )

        print(
            f"CYCLE {cycle} COMPLETE + CHECKPOINTED",
            flush=True,
        )

    if len(cycle_results) != 6:
        raise RuntimeError(
            f"Expected 6 completed cycles, "
            f"got {len(cycle_results)}"
        )

    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "complete",
        "author": warm.AUTHOR,
        "protocol_sha256": protocol_sha,
        "resume_semantics": "cycle_level",
        "cycles": cycle_results,
        "training": {
            "total_rows": 34832,
            "steps_per_cycle": 726,
            "total_optimizer_steps": 4356,
            "batch_size": warm.BATCH_SIZE,
            "learning_rate": warm.LEARNING_RATE,
            "weight_decay": warm.WEIGHT_DECAY,
            "max_grad_norm": warm.MAX_GRAD_NORM,
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
    }

    warm.write_json(
        final_result_path,
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
            f"N={m['interaction_count']} "
            f"Top1={m['top1']:.6f} "
            f"Top3={m['top3']:.6f} "
            f"MRR={m['mrr_at_10']:.6f} "
            f"Missing={m['missing_at_10_rate']:.6f}"
        )

    print(
        "TEST USED = false"
    )


if __name__ == "__main__":
    main()
