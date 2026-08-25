from __future__ import annotations

import json
from pathlib import Path

import torch

from src.reference_backend_pinyingpt.backend import PinyinGPTConcatBackend
from src.model_level.pinyingpt_adapter import (
    install_adapters,
    load_adapter_bundle,
)

from src.model_level.concat_training import (
    collate_concat_examples,
    compute_target_loss,
    prepare_concat_example,
)

from src.model_level.pinyin_modes import choose_pinyin_representation


AUTHOR = "Agent Phage"

BATCH_SIZE = 8
LR = 5e-4

SMOKE_PAIRS = None

STREAMS = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "temporal_shift_audit_v1/"
    "temporal_shift_streams.json"
)

INIT_DIR = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "preference_init_smoke_v1"
)

OUT = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "temporal_drift_v1"
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


def read_json(path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_rows(path):
    rows = []

    with path.open(
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)

                if row.get("author") == AUTHOR:
                    rows.append(row)

    return rows


def get_pinyin(row):
    value = (
        row.get("segmented_pinyin")
        or row.get("pinyin_input")
        or row.get("pinyin")
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


def score_margin(
    backend,
    context_rows,
    pinyin,
    a,
    b,
):
    values = []

    for row in context_rows:

        scores = backend.score_candidates(
            row.get("context", ""),
            pinyin,
            [a, b],
        )

        values.append(
            float(
                scores[0].log_probability
                -
                scores[1].log_probability
            )
        )

    return sum(values) / len(values)


def train_rows(
    backend,
    rows,
):

    optimizer = torch.optim.AdamW(
        [
            p
            for p in backend.model.parameters()
            if p.requires_grad
        ],
        lr=LR,
    )

    batch = []

    for row in rows:

        representation = choose_pinyin_representation(
            row["pinyin_segments"],
            row_id=str(row["row_id"]),
            epoch=0,
            seed=20260822,
            forced_mode="full",
            weights=(1,0,0),
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
                data,
            )

            loss.backward()

            optimizer.step()

            batch = []


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    streams = read_json(STREAMS)

    if SMOKE_PAIRS is not None:
        streams["results"] = [
            x
            for x in streams["results"]
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
                get_pinyin(row),
                get_target(row),
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


    for pair in streams["results"]:

        pid = pair["pair_id"]

        init = (
            INIT_DIR /
            f"{pid}.safetensors"
        )

        if not init.is_file():
            continue


        load_adapter_bundle(
            backend.model,
            init,
        )


        a = pair["candidate_a"]
        b = pair["candidate_b"]
        py = pair["pinyin"]


        a_rows = index.get(
            (py, a),
            [],
        )

        b_rows = index.get(
            (py, b),
            [],
        )


        pair_result = {
            "pair_id": pid,
            "candidate_a": a,
            "candidate_b": b,
            "gradual": [],
            "abrupt": [],
            "temporary": [],
        }


        # gradual
        for step in [
            0.2,
            0.4,
            0.6,
            0.8,
        ]:

            n_b = max(
                1,
                int(
                    len(b_rows) * step
                )
            )

            train_rows(
                backend,
                b_rows[:n_b],
            )

            margin = score_margin(
                backend,
                a_rows[:8],
                py,
                a,
                b,
            )

            pair_result["gradual"].append(
                {
                    "b_ratio": step,
                    "margin": margin,
                }
            )


        # abrupt
        train_rows(
            backend,
            b_rows,
        )

        pair_result["abrupt"].append(
            {
                "margin": score_margin(
                    backend,
                    a_rows[:8],
                    py,
                    a,
                    b,
                )
            }
        )


        # temporary drift:
        # S_A -> B drift -> A recovery

        load_adapter_bundle(
            backend.model,
            init,
        )

        recovery_results = []

        # temporary B drift
        train_rows(
            backend,
            b_rows[:8],
        )

        drift_margin = score_margin(
            backend,
            a_rows[:8],
            py,
            a,
            b,
        )

        recovery_results.append(
            {
                "stage": "after_B_drift",
                "margin": drift_margin,
            }
        )

        # recovery with A evidence
        for recovery_n in [1, 2, 4, 8, 16]:

            load_adapter_bundle(
                backend.model,
                init,
            )

            train_rows(
                backend,
                b_rows[:8],
            )

            train_rows(
                backend,
                a_rows[:recovery_n],
            )

            recovery_margin = score_margin(
                backend,
                a_rows[:8],
                py,
                a,
                b,
            )

            recovery_results.append(
                {
                    "stage": "A_recovery",
                    "a_examples": recovery_n,
                    "margin": recovery_margin,
                    "recovered": recovery_margin > 0,
                }
            )

        pair_result["temporary"] = recovery_results


        results.append(
            pair_result
        )

        print(
            pid,
            "done",
            flush=True,
        )


    (OUT / "temporal_drift_result.json").write_text(
        json.dumps(
            {
                "experiment":
                "controlled_preference_temporal_drift_v1",
                "pairs": results,
                "test_used": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "===== COMPLETE ====="
    )


if __name__ == "__main__":
    main()
