from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.reference_backend_pinyingpt.backend import PinyinGPTConcatBackend
from src.model_level.pinyingpt_adapter import (
    install_adapters,
    load_adapter_bundle,
    save_adapter_bundle,
    SerialAdapterConfig,
)

from src.model_level.concat_training import (
    collate_concat_examples,
    compute_target_loss,
    prepare_concat_example,
)

from src.model_level.pinyin_modes import choose_pinyin_representation


AUTHOR = "Agent Phage"

BATCH_SIZE = 8
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0

EXPOSURES = [1, 2, 4, 8, 16, 32]

SMOKE_PAIRS = {
    "R4_PAIR_001",
    "R4_PAIR_002",
    "R4_PAIR_004",
}

PAIRS = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "frozen_pairs_v1/"
    "frozen_pairs.json"
)

INIT_DIR = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "preference_init_smoke_v1"
)

OUT = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "persistent_shift_v1"
)

FIT = Path(
    "/home/3160454/work/model-level-assets-20260822/"
    "clean3_train_fit_v1.jsonl"
)

VAL = Path(
    "/home/3160454/work/model-level-assets-20260822/"
    "clean3_train_val_v1.jsonl"
)

CHECKPOINT = Path(
    "/home/3160454/work/model-level-assets-20260822/"
    "pinyingpt2-concat"
)


def read_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_rows(path: Path):
    rows = []

    with path.open(
        encoding="utf-8"
    ) as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if row.get("author") == AUTHOR:
                rows.append(row)

    return rows


def pinyin(row):
    value = (
        row.get("segmented_pinyin")
        or row.get("pinyin_input")
        or row.get("pinyin")
        or row.get("typed_pinyin")
    )

    if isinstance(value, list):
        return " ".join(
            str(x).lower()
            for x in value
        )

    return " ".join(
        str(value).lower().split()
    )


def target(row):
    for key in (
        "target",
        "gold",
        "text",
    ):
        if isinstance(row.get(key), str):
            return row[key]

    raise RuntimeError("missing target")


def score_margin(
    backend,
    pinyin_value,
    a,
    b,
):
    scores = backend.score_candidates(
        "",
        pinyin_value,
        [a, b],
    )

    return float(
        scores[0].log_probability
        -
        scores[1].log_probability
    )


def train_examples(
    backend,
    examples,
):
    optimizer = torch.optim.AdamW(
        [
            p
            for p in backend.model.parameters()
            if p.requires_grad
        ],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    backend.model.train()

    batch = []
    steps = 0

    for row in examples:

        representation = choose_pinyin_representation(
            row["pinyin_segments"],
            row_id=str(row["row_id"]),
            epoch=0,
            seed=20260822,
            forced_mode="full",
            weights=(1, 0, 0),
        )

        batch.append(
            prepare_concat_example(
                backend,
                row,
                representation,
            )
        )

        if len(batch) == BATCH_SIZE:

            data = collate_concat_examples(
                batch,
                pad_token_id=backend.tokenizer.pad_token_id,
                device=backend.device,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss = compute_target_loss(
                backend.model,
                data,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                backend.model.parameters(),
                MAX_GRAD_NORM,
            )

            optimizer.step()

            steps += 1
            batch = []

    return {
        "steps": steps,
        "rows": len(examples),
    }


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    frozen = read_json(PAIRS)

    frozen["pairs"] = [
        x
        for x in frozen["pairs"]
        if x["pair_id"] in SMOKE_PAIRS
    ]

    rows = (
        load_rows(FIT)
        +
        load_rows(VAL)
    )

    index = {}

    for row in rows:
        index.setdefault(
            (
                pinyin(row),
                target(row),
            ),
            [],
        ).append(row)


    backend = PinyinGPTConcatBackend(
        CHECKPOINT,
        device="cuda",
    )

    install_adapters(
        backend.model
    )


    results = []

    for pair in frozen["pairs"]:

        pair_id = pair["pair_id"]

        init_path = (
            INIT_DIR
            /
            f"{pair_id}.safetensors"
        )

        if not init_path.is_file():
            continue

        load_adapter_bundle(
            backend.model,
            init_path,
        )

        a = pair["candidate_a"]
        b = pair["candidate_b"]

        b_examples = index.get(
            (
                pair["pinyin"],
                b,
            ),
            [],
        )

        b_examples = b_examples[:32]

        print()
        print(
            pair_id,
            a,
            "->",
            b,
            "B_examples=",
            len(b_examples),
        )

        pair_results = []

        for n in EXPOSURES:

            train_examples(
                backend,
                b_examples[:n],
            )

            margin = score_margin(
                backend,
                pair["pinyin"],
                a,
                b,
            )

            pair_results.append(
                {
                    "exposure": n,
                    "margin_a_minus_b": margin,
                    "flipped": margin < 0,
                }
            )

            print(
                " exposure=",
                n,
                "margin=",
                margin,
                "flipped=",
                margin < 0,
            )

        results.append(
            {
                "pair_id": pair_id,
                "candidate_a": a,
                "candidate_b": b,
                "results": pair_results,
            }
        )


    (OUT / "persistent_shift_result.json").write_text(
        json.dumps(
            {
                "experiment":
                    "controlled_preference_dynamics_persistent_shift_v1",
                "pairs": results,
                "exposures": EXPOSURES,
                "test_used": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("===== COMPLETE =====")


if __name__ == "__main__":
    main()
