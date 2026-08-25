from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path
from typing import Any

import torch

from experiments.model_level import run_adapter_longitudinal_warm_v1 as warm
from experiments.model_level import run_adapter_training_v1 as train_base
from experiments.model_level import run_adapter_dev_evaluation_v1 as eval_base
from src.reference_backend_pinyingpt.backend import PinyinGPTConcatBackend

from src.model_level.pinyingpt_adapter import (
    SerialAdapterConfig,
    install_adapters,
    save_adapter_bundle,
    set_adapter_training_mode,
)

from src.model_level.pinyin_modes import choose_pinyin_representation

from src.model_level.concat_training import (
    collate_concat_examples,
    compute_target_loss,
    prepare_concat_example,
)


EXPERIMENT = "controlled_preference_dynamics_preference_init_smoke_v1"

AUTHOR = "Agent Phage"

BATCH_SIZE = 8
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0

SMOKE_PAIRS = None

PAIRS_PATH = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "frozen_pairs_v1/"
    "frozen_pairs.json"
)

AUDIT_PATH = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "evidence_audit_v1/"
    "evidence_pairs.csv"
)

OUT = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "preference_init_smoke_v1"
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

            if row.get("author") != AUTHOR:
                continue

            rows.append(row)

    return rows


def get_pinyin(row):
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


def get_target(row):
    for k in (
        "target",
        "gold",
        "text",
    ):
        if isinstance(row.get(k), str):
            return row[k]

    raise RuntimeError("missing target")


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_examples():
    frozen = read_json(PAIRS_PATH)

    selected = frozen["pairs"]

    if SMOKE_PAIRS is not None:
        selected = [
            x
            for x in selected
            if x["pair_id"] in SMOKE_PAIRS
        ]

    rows = (
        load_rows(FIT)
        +
        load_rows(VAL)
    )

    index = {}

    for row in rows:
        key = (
            get_pinyin(row),
            get_target(row),
        )

        index.setdefault(
            key,
            []
        ).append(row)

    return selected, index


def score_margin(
    backend,
    pinyin: str,
    a: str,
    b: str,
):
    scores = backend.score_candidates(
        "",
        pinyin,
        [a, b],
    )

    return {
        "score_a": float(
            scores[0].log_probability
        ),
        "score_b": float(
            scores[1].log_probability
        ),
        "margin_a_minus_b": float(
            scores[0].log_probability
            -
            scores[1].log_probability
        ),
    }


def train_evidence(
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

    steps = 0

    batch = []

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

            collated = collate_concat_examples(
                batch,
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
                collated,
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
        "examples": len(examples),
    }


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs, index = load_examples()

    print(
        "===== R4 PREFERENCE INIT SMOKE ====="
    )

    print(
        "pairs =",
        len(pairs),
    )

    backend = PinyinGPTConcatBackend(
        CHECKPOINT,
        device="cuda",
    )

    install_adapters(
        backend.model
    )

    results = []

    for pair in pairs:

        a_rows = index.get(
            (
                pair["pinyin"],
                pair["candidate_a"],
            ),
            [],
        )

        evidence = a_rows[:16]

        print()
        print(
            pair["pair_id"],
            pair["candidate_a"],
            "vs",
            pair["candidate_b"],
            "evidence=",
            len(evidence),
        )

        before = score_margin(
            backend,
            pair["pinyin"],
            pair["candidate_a"],
            pair["candidate_b"],
        )


        train_result = train_evidence(
            backend,
            evidence,
        )

        after_path = (
            OUT
            / f"{pair['pair_id']}.safetensors"
        )

        after = score_margin(
            backend,
            pair["pinyin"],
            pair["candidate_a"],
            pair["candidate_b"],
        )

        adapter_config = SerialAdapterConfig()

        save_adapter_bundle(
            backend.model,
            after_path,
            config=adapter_config,
            metadata={
                "experiment": EXPERIMENT,
                "pair_id": pair["pair_id"],
                "candidate_a": pair["candidate_a"],
                "candidate_b": pair["candidate_b"],
            },
        )

        results.append(
            {
                "pair_id": pair["pair_id"],
                "candidate_a": pair["candidate_a"],
                "candidate_b": pair["candidate_b"],
                "evidence_rows": len(evidence),
                "training": train_result,
                "before": before,
                "after": after,
                "adapter_checkpoint": str(after_path),
            }
        )

    (OUT / "smoke_result.json").write_text(
        json.dumps(
            {
                "experiment": EXPERIMENT,
                "pairs": results,
                "test_used": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "===== COMPLETE ====="
    )


if __name__ == "__main__":
    main()
