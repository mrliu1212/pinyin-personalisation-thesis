from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.reference_backend_pinyingpt.backend import (
    PinyinGPTConcatBackend,
)

from src.model_level.pinyingpt_adapter import (
    SerialAdapterConfig,
    install_adapters,
    save_adapter_bundle,
    set_adapter_training_mode,
)

from src.model_level.concat_training import (
    collate_concat_examples,
    compute_target_loss,
    prepare_concat_example,
)

from src.model_level.pinyin_modes import (
    choose_pinyin_representation,
)


EXPERIMENT = "controlled_preference_dynamics_r4_aba_clean_training_v2"

AUTHOR = "Agent Phage"

SEED = 20260824

BATCH_SIZE = 8
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0

CHECKPOINT = Path(
    "/home/3160454/work/model-level-assets-20260822/"
    "pinyingpt2-concat"
)

STAGE_ROOT = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "r4_aba_clean_v2/stage_data"
)

OUT = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "r4_aba_clean_v2/training"
)

STAGES = [
    "stage_01_A",
    "stage_02_B",
    "stage_03_A",
]


def write_json(path: Path, value: Any):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_stage(path: Path):
    rows = []

    with path.open(
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def train_stage(
    backend,
    rows,
    stage_name,
):

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

    batch_examples = []

    steps = 0
    processed = 0

    backend.model.train()

    for idx, row in enumerate(rows):

        representation = choose_pinyin_representation(
            row["pinyin_segments"],
            row_id=str(
                row["row_id"]
            ),
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

        batch_examples.append(
            example
        )

        if (
            len(batch_examples)
            == BATCH_SIZE
        ) or (
            idx + 1 == len(rows)
        ):

            batch = collate_concat_examples(
                batch_examples,
                pad_token_id=(
                    backend.tokenizer.pad_token_id
                ),
                device=backend.device,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss = compute_target_loss(
                backend.model,
                batch,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable,
                MAX_GRAD_NORM,
            )

            optimizer.step()

            steps += 1
            processed += len(
                batch_examples
            )

            batch_examples = []

            if (
                steps == 1
                or steps % 100 == 0
            ):
                print(
                    stage_name,
                    f"step={steps}",
                    f"rows={processed}/{len(rows)}",
                    f"loss={float(loss):.6f}",
                    flush=True,
                )

    set_adapter_training_mode(
        backend.model,
        enabled=False,
    )

    return {
        "stage": stage_name,
        "rows": processed,
        "optimizer_steps": steps,
    }


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    backend = PinyinGPTConcatBackend(
        CHECKPOINT,
        device="cuda",
    )

    install_adapters(
        backend.model
    )

    adapter_config = SerialAdapterConfig()

    results = []

    for stage in STAGES:

        print(
            "=" * 60
        )
        print(
            stage
        )
        print(
            "=" * 60,
            flush=True,
        )

        rows = load_stage(
            STAGE_ROOT /
            f"{stage}.jsonl"
        )

        result = train_stage(
            backend,
            rows,
            stage,
        )

        checkpoint = (
            OUT /
            f"{stage}.safetensors"
        )

        save_adapter_bundle(
            backend.model,
            checkpoint,
            config=adapter_config,
            metadata={
                "experiment": EXPERIMENT,
                "stage": stage,
                "rows": len(rows),
                "test_used": False,
                "adapter_state_persisted": True,
                "optimizer_state_persisted": False,
            },
        )

        results.append(
            {
                **result,
                "checkpoint": str(
                    checkpoint
                ),
            }
        )

    write_json(
        OUT /
        "aba_training_result.json",
        {
            "experiment": EXPERIMENT,
            "stages": results,
            "test_used": False,
        },
    )

    print()
    print(
        "===== R4 ABA TRAINING COMPLETE ====="
    )


if __name__ == "__main__":
    main()
