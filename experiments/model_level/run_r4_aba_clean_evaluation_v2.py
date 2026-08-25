from __future__ import annotations

import csv
import gc
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from src.model_level.concat_training import truncate_recent_context
from src.model_level.pinyingpt_adapter import (
    SerialAdapterConfig,
    install_adapters,
    load_adapter_bundle,
    set_adapter_training_mode,
)
from src.reference_backend_pinyingpt.backend import PinyinGPTConcatBackend


EXPERIMENT = "controlled_preference_dynamics_r4_aba_clean_evaluation_v2"
AUTHOR = "Agent Phage"
SEED = 20260824
CONTEXTS_PER_PAIR = 8

CHECKPOINT = Path(
    "/home/3160454/work/model-level-assets-20260822/pinyingpt2-concat"
)

FIT = Path(
    "/home/3160454/work/model-level-assets-20260822/clean3_train_fit_v1.jsonl"
)

VAL = Path(
    "/home/3160454/work/model-level-assets-20260822/clean3_train_val_v1.jsonl"
)

ROOT = Path(
    "results/model_level/controlled_preference_dynamics_v1"
)

EVAL_PROBES = (
    ROOT /
    "r4_aba_clean_v2/"
    "evaluation_probes.jsonl"
)

PAIR_FILE = ROOT / "r4_aba_clean_v2/selected_pairs.json"

STAGE_DATA_ROOT = ROOT / "r4_aba_clean_v2/stage_data"
TRAIN_ROOT = ROOT / "r4_aba_clean_v2/training"

OUT = ROOT / "r4_aba_clean_v2/evaluation"

STATES = {
    "generic": None,
    "stage_01_A": TRAIN_ROOT / "stage_01_A.safetensors",
    "stage_02_B": TRAIN_ROOT / "stage_02_B.safetensors",
    "stage_03_A": TRAIN_ROOT / "stage_03_A.safetensors",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("row_id", ""))


def row_pinyin(row: dict[str, Any]) -> str:
    if row.get("pinyin_segments"):
        return " ".join(map(str, row["pinyin_segments"]))

    value = (
        row.get("pinyin")
        or row.get("typed_pinyin")
        or row.get("segmented_pinyin")
    )

    if isinstance(value, list):
        return " ".join(map(str, value))

    if value is None:
        raise KeyError(
            f"Cannot find pinyin in row keys={sorted(row.keys())}"
        )

    return " ".join(str(value).split())


def first_value(
    item: dict[str, Any],
    keys: tuple[str, ...],
) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]

    raise KeyError(
        f"None of {keys} found. Available keys={sorted(item.keys())}"
    )


def normalize_pair(
    item: dict[str, Any],
    index: int,
) -> dict[str, str]:

    pair_id = str(
        item.get("pair_id")
        or item.get("id")
        or f"R4_PAIR_{index:03d}"
    )

    pinyin_raw = first_value(
        item,
        (
            "pinyin",
            "typed_pinyin",
            "pinyin_key",
            "pinyin_segments",
        ),
    )

    if isinstance(pinyin_raw, list):
        pinyin = " ".join(map(str, pinyin_raw))
    else:
        pinyin = " ".join(str(pinyin_raw).split())

    candidate_a = str(
        first_value(
            item,
            (
                "candidate_a",
                "a",
                "A",
                "target_a",
                "candidate_A",
            ),
        )
    )

    candidate_b = str(
        first_value(
            item,
            (
                "candidate_b",
                "b",
                "B",
                "target_b",
                "candidate_B",
            ),
        )
    )

    return {
        "pair_id": pair_id,
        "pinyin": pinyin,
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
    }


def load_pairs() -> list[dict[str, str]]:
    payload = json.loads(
        PAIR_FILE.read_text(encoding="utf-8")
    )

    if isinstance(payload, list):
        raw_pairs = payload
    elif isinstance(payload, dict):
        raw_pairs = (
            payload.get("pairs")
            or payload.get("frozen_pairs")
            or payload.get("shortlist")
        )
        if raw_pairs is None:
            raise RuntimeError(
                f"Cannot locate pair list in {PAIR_FILE}; "
                f"keys={sorted(payload.keys())}"
            )
    else:
        raise TypeError("Unexpected frozen_pairs.json structure")

    pairs = [
        normalize_pair(item, i + 1)
        for i, item in enumerate(raw_pairs)
    ]

    if len(pairs) != 15:
        raise RuntimeError(
            f"Expected 15 clean pairs, got {len(pairs)}"
        )

    return pairs


def collect_training_row_ids() -> set[str]:
    used: set[str] = set()

    for stage in (
        "stage_01_A",
        "stage_02_B",
        "stage_03_A",
    ):
        path = STAGE_DATA_ROOT / f"{stage}.jsonl"

        if not path.is_file():
            raise FileNotFoundError(path)

        for row in load_jsonl(path):
            rid = row_id(row)
            if rid:
                used.add(rid)

    return used


def build_context_probes(
    pairs: list[dict[str, str]],
    used_training_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:

    source_rows = load_jsonl(FIT) + load_jsonl(VAL)

    by_pinyin: dict[str, list[dict[str, Any]]] = {}

    for row in source_rows:
        if str(row.get("author")) != AUTHOR:
            continue

        if str(row.get("source_split", "")).lower() == "test":
            raise RuntimeError("STOP: Test row detected")

        rid = row_id(row)

        if rid and rid in used_training_ids:
            continue

        pinyin = row_pinyin(row)
        by_pinyin.setdefault(pinyin, []).append(row)

    probes: dict[str, list[dict[str, Any]]] = {}

    for pair_index, pair in enumerate(pairs):
        candidates = by_pinyin.get(pair["pinyin"], [])

        unique = []
        seen_contexts = set()

        for row in candidates:
            context = str(row.get("context", ""))

            if context in seen_contexts:
                continue

            seen_contexts.add(context)
            unique.append(row)

        rng = random.Random(SEED + pair_index)
        rng.shuffle(unique)

        probes[pair["pair_id"]] = unique[:CONTEXTS_PER_PAIR]

    return probes


def make_backend(
    adapter_path: Path | None,
) -> PinyinGPTConcatBackend:

    backend = PinyinGPTConcatBackend(
        CHECKPOINT,
        device="cuda",
    )

    if adapter_path is not None:
        install_adapters(
            backend.model,
            SerialAdapterConfig(),
        )

        load_adapter_bundle(
            backend.model,
            adapter_path,
        )

        set_adapter_training_mode(
            backend.model,
            enabled=False,
        )

    backend.model.eval()
    return backend


def score_pair(
    backend: PinyinGPTConcatBackend,
    pair: dict[str, str],
    context: str,
) -> dict[str, float]:

    typed_pinyin = pair["pinyin"].split()

    segments = backend.segment_pinyin(typed_pinyin)

    used_context, _, _, _ = truncate_recent_context(
        backend,
        context,
        segments,
    )

    scores = backend.score_candidates(
        used_context,
        typed_pinyin,
        [
            pair["candidate_a"],
            pair["candidate_b"],
        ],
    )

    score_map = {
        item.text: float(item.log_probability)
        for item in scores
    }

    a = score_map[pair["candidate_a"]]
    b = score_map[pair["candidate_b"]]

    return {
        "a_logp": a,
        "b_logp": b,
        "margin_a_minus_b": a - b,
    }


def evaluate_state(
    state: str,
    adapter_path: Path | None,
    pairs: list[dict[str, str]],
    probes: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:

    print()
    print("=" * 70)
    print(f"EVALUATING {state}")
    print("=" * 70, flush=True)

    backend = make_backend(adapter_path)

    output: dict[str, dict[str, Any]] = {}

    try:
        for i, pair in enumerate(pairs, 1):

            # Universal clean probe: no preceding context.
            pinyin_only = score_pair(
                backend,
                pair,
                "",
            )

            contextual_scores = []

            for row in probes[pair["pair_id"]]:
                contextual_scores.append(
                    score_pair(
                        backend,
                        pair,
                        str(row.get("context", "")),
                    )
                )

            if contextual_scores:
                contextual_margin = mean(
                    x["margin_a_minus_b"]
                    for x in contextual_scores
                )
                contextual_a = mean(
                    x["a_logp"]
                    for x in contextual_scores
                )
                contextual_b = mean(
                    x["b_logp"]
                    for x in contextual_scores
                )
            else:
                contextual_margin = None
                contextual_a = None
                contextual_b = None

            output[pair["pair_id"]] = {
                "pair_id": pair["pair_id"],
                "pinyin": pair["pinyin"],
                "candidate_a": pair["candidate_a"],
                "candidate_b": pair["candidate_b"],
                "pinyin_only_a_logp": pinyin_only["a_logp"],
                "pinyin_only_b_logp": pinyin_only["b_logp"],
                "pinyin_only_margin": pinyin_only[
                    "margin_a_minus_b"
                ],
                "context_probe_n": len(contextual_scores),
                "context_a_logp_mean": contextual_a,
                "context_b_logp_mean": contextual_b,
                "context_margin_mean": contextual_margin,
            }

            print(
                f"{i:02d}/30 "
                f"{pair['pair_id']} "
                f"{pair['pinyin']} "
                f"{pair['candidate_a']}/{pair['candidate_b']} "
                f"pinyin_margin="
                f"{pinyin_only['margin_a_minus_b']:+.4f} "
                f"context_n={len(contextual_scores)} "
                f"context_margin="
                f"{contextual_margin if contextual_margin is not None else 'NA'}",
                flush=True,
            )

    finally:
        del backend
        gc.collect()
        torch.cuda.empty_cache()

    return output


def summarize_scope(
    rows: list[dict[str, Any]],
    prefix: str,
) -> dict[str, Any]:

    g_key = f"generic_{prefix}"
    s1_key = f"stage_01_A_{prefix}"
    s2_key = f"stage_02_B_{prefix}"
    s3_key = f"stage_03_A_{prefix}"

    valid = [
        row
        for row in rows
        if all(
            row.get(k) is not None
            for k in (g_key, s1_key, s2_key, s3_key)
        )
    ]

    if not valid:
        return {
            "n_pairs": 0,
        }

    return {
        "n_pairs": len(valid),

        "generic_mean_margin": mean(
            row[g_key] for row in valid
        ),

        "stage_01_A_mean_margin": mean(
            row[s1_key] for row in valid
        ),

        "stage_02_B_mean_margin": mean(
            row[s2_key] for row in valid
        ),

        "stage_03_A_mean_margin": mean(
            row[s3_key] for row in valid
        ),

        "stage_01_A_preferred_pairs": sum(
            row[s1_key] > 0 for row in valid
        ),

        "stage_02_B_preferred_pairs": sum(
            row[s2_key] < 0 for row in valid
        ),

        "stage_03_A_preferred_pairs": sum(
            row[s3_key] > 0 for row in valid
        ),

        "A_to_B_flip_pairs": sum(
            row[s1_key] > 0 and row[s2_key] < 0
            for row in valid
        ),

        "B_to_A_recovery_pairs": sum(
            row[s2_key] < 0 and row[s3_key] > 0
            for row in valid
        ),

        "full_ABA_success_pairs": sum(
            row[s1_key] > 0
            and row[s2_key] < 0
            and row[s3_key] > 0
            for row in valid
        ),

        "mean_stage1_to_stage2_delta": mean(
            row[s2_key] - row[s1_key]
            for row in valid
        ),

        "mean_stage2_to_stage3_delta": mean(
            row[s3_key] - row[s2_key]
            for row in valid
        ),
    }


def main() -> None:

    OUT.mkdir(parents=True, exist_ok=True)

    for name, path in STATES.items():
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)

    pairs = load_pairs()

    used_training_ids = collect_training_row_ids()

    probe_rows = load_jsonl(
        EVAL_PROBES
    )

    probes = {}

    for row in probe_rows:
        probes.setdefault(
            row["pair_id"],
            []
        ).append(row)

    probe_manifest = []

    for pair in pairs:
        rows = probes[pair["pair_id"]]

        probe_manifest.append(
            {
                **pair,
                "heldout_context_rows": len(rows),
                "row_ids": [row_id(x) for x in rows],
            }
        )

    (
        OUT / "context_probe_manifest.json"
    ).write_text(
        json.dumps(
            {
                "experiment": EXPERIMENT,
                "author": AUTHOR,
                "contexts_per_pair_max": CONTEXTS_PER_PAIR,
                "training_unique_row_ids_excluded": len(
                    used_training_ids
                ),
                "pairs": probe_manifest,
                "test_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    state_results = {}

    for state, adapter_path in STATES.items():
        state_results[state] = evaluate_state(
            state,
            adapter_path,
            pairs,
            probes,
        )

    final_rows = []

    for pair in pairs:
        pid = pair["pair_id"]

        row: dict[str, Any] = {
            **pair,
            "context_probe_n": state_results[
                "generic"
            ][pid]["context_probe_n"],
        }

        for state in STATES:
            result = state_results[state][pid]

            row[f"{state}_pinyin_margin"] = result[
                "pinyin_only_margin"
            ]

            row[f"{state}_context_margin"] = result[
                "context_margin_mean"
            ]

        row["pinyin_shift_delta"] = (
            row["stage_02_B_pinyin_margin"]
            - row["stage_01_A_pinyin_margin"]
        )

        row["pinyin_recovery_delta"] = (
            row["stage_03_A_pinyin_margin"]
            - row["stage_02_B_pinyin_margin"]
        )

        row["pinyin_ABA_success"] = (
            row["stage_01_A_pinyin_margin"] > 0
            and row["stage_02_B_pinyin_margin"] < 0
            and row["stage_03_A_pinyin_margin"] > 0
        )

        if row["stage_01_A_context_margin"] is not None:
            row["context_shift_delta"] = (
                row["stage_02_B_context_margin"]
                - row["stage_01_A_context_margin"]
            )

            row["context_recovery_delta"] = (
                row["stage_03_A_context_margin"]
                - row["stage_02_B_context_margin"]
            )

            row["context_ABA_success"] = (
                row["stage_01_A_context_margin"] > 0
                and row["stage_02_B_context_margin"] < 0
                and row["stage_03_A_context_margin"] > 0
            )
        else:
            row["context_shift_delta"] = None
            row["context_recovery_delta"] = None
            row["context_ABA_success"] = None

        final_rows.append(row)

    csv_path = OUT / "pair_dynamics.csv"

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(final_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(final_rows)

    summary = {
        "experiment": EXPERIMENT,
        "author": AUTHOR,
        "pairs": len(pairs),
        "training_unique_row_ids_excluded": len(
            used_training_ids
        ),
        "pinyin_only": summarize_scope(
            final_rows,
            "pinyin_margin",
        ),
        "heldout_context": summarize_scope(
            final_rows,
            "context_margin",
        ),
        "test_used": False,
    }

    (
        OUT / "evaluation_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("R4 ABA EVALUATION COMPLETE")
    print("=" * 70)

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print(f"CSV: {csv_path}")
    print(
        f"SUMMARY: {OUT / 'evaluation_summary.json'}"
    )


if __name__ == "__main__":
    main()
