from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from experiments.model_level import run_adapter_longitudinal_warm_v1 as warm
from experiments.model_level import run_adapter_training_v1 as train_base
from src.model_level.concat_training import (
    collate_concat_examples,
    compute_target_loss,
    prepare_concat_example,
)
from src.model_level.pinyin_modes import choose_pinyin_representation
from src.model_level.pinyingpt_adapter import (
    SerialAdapterConfig,
    adapter_state_dict,
    install_adapters,
    set_adapter_training_mode,
)
from src.reference_backend_pinyingpt.backend import PinyinGPTConcatBackend


ROOT = Path("/home/3160454/work/thesis-model-level")
CHECKPOINT = Path(
    "/home/3160454/work/model-level-assets-20260822/pinyingpt2-concat"
)
FIT = Path(
    "/home/3160454/work/model-level-assets-20260822/clean3_train_fit_v1.jsonl"
)
VAL = Path(
    "/home/3160454/work/model-level-assets-20260822/clean3_train_val_v1.jsonl"
)
PROTOCOL = (
    ROOT
    / "results/model_level/long_term_qualification_v1/"
      "alternating_six_train_six_probe_v1"
)
OUT = (
    ROOT
    / "results/model_level/longitudinal_warm_smoke_v1"
)

STEPS_PER_CYCLE = 2
PROBE_ROWS = 20


def state_sha(model) -> str:
    h = hashlib.sha256()
    state = adapter_state_dict(model)

    for name in sorted(state):
        h.update(name.encode("utf-8"))
        h.update(state[name].numpy().tobytes())

    return h.hexdigest()


def train_two_steps(backend, rows, cycle: int):
    set_adapter_training_mode(
        backend.model,
        enabled=True,
    )

    trainable = [
        p for p in backend.model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable,
        lr=warm.LEARNING_RATE,
        weight_decay=warm.WEIGHT_DECAY,
    )

    ordered = train_base.epoch_order(
        rows,
        epoch=0,
        seed=warm.SEED,
    )

    step = 0
    used = 0

    for start in range(0, len(ordered), warm.BATCH_SIZE):
        chunk = ordered[start:start + warm.BATCH_SIZE]

        examples = []

        for row in chunk:
            rep = choose_pinyin_representation(
                row["pinyin_segments"],
                row_id=str(row["row_id"]),
                epoch=0,
                seed=warm.SEED,
                forced_mode="full",
                weights=(1, 0, 0),
            )

            examples.append(
                prepare_concat_example(
                    backend,
                    row,
                    rep,
                )
            )

        batch = collate_concat_examples(
            examples,
            pad_token_id=backend.tokenizer.pad_token_id,
            device=backend.device,
        )

        optimizer.zero_grad(set_to_none=True)

        loss = compute_target_loss(
            backend.model,
            batch,
        )

        if not torch.isfinite(loss):
            raise RuntimeError("non-finite smoke loss")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            trainable,
            warm.MAX_GRAD_NORM,
        )

        optimizer.step()

        step += 1
        used += len(chunk)

        print(
            f"cycle={cycle} smoke_step={step}/2 "
            f"rows={used} loss={float(loss.detach().cpu()):.6f}",
            flush=True,
        )

        if step == STEPS_PER_CYCLE:
            break

    if step != STEPS_PER_CYCLE:
        raise RuntimeError("smoke did not reach two steps")

    set_adapter_training_mode(
        backend.model,
        enabled=False,
    )


def main():
    if OUT.exists():
        raise RuntimeError(
            f"Refusing existing smoke output: {OUT}"
        )

    OUT.mkdir(parents=True)

    protocol_sha = warm.sha256_file(
        PROTOCOL / "frozen_protocol.json"
    )

    if protocol_sha != warm.EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("protocol SHA mismatch")

    population = warm.load_population(
        FIT,
        VAL,
    )

    warm.configure_determinism(
        warm.SEED,
    )

    backend = PinyinGPTConcatBackend(
        CHECKPOINT,
        device="cuda",
    )

    audit = install_adapters(
        backend.model,
        SerialAdapterConfig(),
    )

    if audit["adapter_parameters"] != 894_528:
        raise RuntimeError(audit)

    initial_sha = state_sha(
        backend.model
    )

    results = []

    previous_end_sha = initial_sha

    for cycle in (1, 2):
        train_manifest = (
            PROTOCOL
            / f"cycle_{cycle:02d}_train_row_ids.json"
        )
        probe_manifest = (
            PROTOCOL
            / f"cycle_{cycle:02d}_probe_row_ids.json"
        )

        train_rows = warm.select_rows(
            population,
            train_manifest,
        )

        probe_rows = warm.select_rows(
            population,
            probe_manifest,
        )[:PROBE_ROWS]

        start_sha = state_sha(
            backend.model
        )

        if start_sha != previous_end_sha:
            raise RuntimeError(
                f"Cycle {cycle} did not inherit prior Adapter state"
            )

        train_two_steps(
            backend,
            train_rows,
            cycle,
        )

        end_sha = state_sha(
            backend.model
        )

        if end_sha == start_sha:
            raise RuntimeError(
                f"Cycle {cycle} Adapter did not change"
            )

        predictions = warm.eval_base.evaluate_backend(
            backend,
            probe_rows,
            condition=f"smoke_s{cycle}_p{cycle}",
            output_path=(
                OUT
                / f"cycle_{cycle:02d}_probe_predictions.jsonl"
            ),
            top_k=10,
            beam_size=16,
            log_every=10,
        )

        metrics = warm.eval_base.compute_metrics(
            predictions
        )

        results.append(
            {
                "cycle": cycle,
                "train_steps": STEPS_PER_CYCLE,
                "train_rows_seen": (
                    STEPS_PER_CYCLE * warm.BATCH_SIZE
                ),
                "probe_rows": PROBE_ROWS,
                "start_adapter_sha256": start_sha,
                "end_adapter_sha256": end_sha,
                "inherited_previous_state": (
                    start_sha == previous_end_sha
                ),
                "adapter_changed": (
                    end_sha != start_sha
                ),
                "metrics": metrics,
            }
        )

        previous_end_sha = end_sha

    payload = {
        "status": "complete",
        "experiment": "longitudinal_warm_smoke_v1",
        "protocol_sha256": protocol_sha,
        "cycles_tested": 2,
        "steps_per_cycle": STEPS_PER_CYCLE,
        "probe_rows_per_cycle": PROBE_ROWS,
        "optimizer_reset_each_cycle": True,
        "adapter_state_persistent": True,
        "used_test": False,
        "results": results,
    }

    warm.write_json(
        OUT / "smoke_result.json",
        payload,
    )

    print()
    print("===== WARM CONTINUAL SMOKE PASS =====")

    for x in results:
        m = x["metrics"]["micro"]
        print(
            f"Cycle {x['cycle']}: "
            f"inherit={x['inherited_previous_state']} "
            f"changed={x['adapter_changed']} "
            f"Top1={m['top1']:.4f} "
            f"MRR={m['mrr_at_10']:.4f}"
        )

    print("TEST USED = false")


if __name__ == "__main__":
    main()
