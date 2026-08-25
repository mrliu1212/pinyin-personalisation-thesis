from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPERIMENT = "controlled_preference_dynamics_r4_aba_clean_v2"
AUTHOR = "Agent Phage"
SEED = 20260824

N_PAIRS = 15
EVAL_PER_SIDE = 4

TOTAL_ROWS_PER_STAGE = 5000
CONTROL_ROWS_PER_STAGE = 600
BACKGROUND_ROWS_PER_STAGE = 4400

CONTROL_PER_PAIR = CONTROL_ROWS_PER_STAGE // N_PAIRS

assert N_PAIRS * CONTROL_PER_PAIR == CONTROL_ROWS_PER_STAGE
assert CONTROL_PER_PAIR == 40

ASSET_ROOT = Path(
    "/home/3160454/work/model-level-assets-20260822"
)

FIT = ASSET_ROOT / "clean3_train_fit_v1.jsonl"
VAL = ASSET_ROOT / "clean3_train_val_v1.jsonl"

ROOT = Path(
    "results/model_level/controlled_preference_dynamics_v1"
)

FROZEN_PAIRS = ROOT / "frozen_pairs_v1/frozen_pairs.json"

OUT = ROOT / "r4_aba_clean_v2"
STAGE_ROOT = OUT / "stage_data"

STAGES = (
    ("stage_01_A", "A"),
    ("stage_02_B", "B"),
    ("stage_03_A", "A"),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def row_id(row: dict[str, Any]) -> str:
    value = row.get("row_id")

    if value is None:
        raise RuntimeError(
            f"Row without row_id: keys={sorted(row.keys())}"
        )

    return str(value)


def row_target(row: dict[str, Any]) -> str:
    return str(
        row.get("gold")
        or row.get("target")
        or ""
    )


def row_pinyin(row: dict[str, Any]) -> str:
    if row.get("pinyin_segments"):
        return " ".join(
            map(str, row["pinyin_segments"])
        )

    value = (
        row.get("pinyin")
        or row.get("typed_pinyin")
        or row.get("segmented_pinyin")
    )

    if isinstance(value, list):
        return " ".join(map(str, value))

    if value is None:
        raise RuntimeError(
            f"Cannot find pinyin: {row_id(row)}"
        )

    return " ".join(str(value).split())


def first_value(
    item: dict[str, Any],
    keys: tuple[str, ...],
) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]

    raise RuntimeError(
        f"Missing keys {keys}; "
        f"available={sorted(item.keys())}"
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
        pinyin = " ".join(
            map(str, pinyin_raw)
        )
    else:
        pinyin = " ".join(
            str(pinyin_raw).split()
        )

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


def load_frozen_pairs() -> list[dict[str, str]]:
    payload = json.loads(
        FROZEN_PAIRS.read_text(
            encoding="utf-8"
        )
    )

    if isinstance(payload, list):
        raw = payload
    elif isinstance(payload, dict):
        raw = (
            payload.get("pairs")
            or payload.get("frozen_pairs")
            or payload.get("shortlist")
        )

        if raw is None:
            raise RuntimeError(
                "Cannot locate frozen pair list"
            )
    else:
        raise RuntimeError(
            "Unexpected frozen pair payload"
        )

    pairs = [
        normalize_pair(x, i + 1)
        for i, x in enumerate(raw)
    ]

    if len(pairs) != 30:
        raise RuntimeError(
            f"Expected 30 frozen pairs, got {len(pairs)}"
        )

    return pairs


def unique_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    result = []
    seen = set()

    for row in rows:
        rid = row_id(row)

        if rid in seen:
            continue

        seen.add(rid)
        result.append(row)

    return result


def main() -> None:

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    STAGE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = load_jsonl(FIT) + load_jsonl(VAL)

    rows = []

    for row in all_rows:

        if str(row.get("author")) != AUTHOR:
            continue

        if (
            str(
                row.get(
                    "source_split",
                    "",
                )
            ).lower()
            == "test"
        ):
            raise RuntimeError(
                "STOP: Test row detected"
            )

        rows.append(row)

    print(
        f"Agent rows={len(rows)}",
        flush=True,
    )

    index: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        index[
            (
                row_pinyin(row),
                row_target(row),
            )
        ].append(row)

    for key in list(index):
        index[key] = unique_rows(
            index[key]
        )

    frozen = load_frozen_pairs()

    candidates = []

    for pair in frozen:

        a_rows = index[
            (
                pair["pinyin"],
                pair["candidate_a"],
            )
        ]

        b_rows = index[
            (
                pair["pinyin"],
                pair["candidate_b"],
            )
        ]

        support_a = len(a_rows)
        support_b = len(b_rows)

        min_support = min(
            support_a,
            support_b,
        )

        candidates.append(
            {
                **pair,
                "support_a": support_a,
                "support_b": support_b,
                "min_support": min_support,
                "total_support": (
                    support_a + support_b
                ),
            }
        )

    # One pair only for each unique Pinyin.
    # Within the same Pinyin, choose the pair with
    # strongest balanced real evidence.
    best_by_pinyin: dict[
        str,
        dict[str, Any],
    ] = {}

    for pair in candidates:

        pinyin = pair["pinyin"]

        current = best_by_pinyin.get(
            pinyin
        )

        score = (
            pair["min_support"],
            pair["total_support"],
            -int(
                pair["pair_id"].split("_")[-1]
            ),
        )

        if current is None:
            best_by_pinyin[pinyin] = pair
            continue

        current_score = (
            current["min_support"],
            current["total_support"],
            -int(
                current[
                    "pair_id"
                ].split("_")[-1]
            ),
        )

        if score > current_score:
            best_by_pinyin[pinyin] = pair

    unique_pair_candidates = list(
        best_by_pinyin.values()
    )

    # Need 5 held-out examples per side,
    # and at least one remaining real row
    # available for controlled training.
    eligible = [
        pair
        for pair in unique_pair_candidates
        if pair["min_support"]
        >= EVAL_PER_SIDE + 1
    ]

    eligible.sort(
        key=lambda x: (
            -x["min_support"],
            -x["total_support"],
            x["pair_id"],
        )
    )

    if len(eligible) < N_PAIRS:
        raise RuntimeError(
            f"Only {len(eligible)} eligible unique-pinyin "
            f"pairs; need {N_PAIRS}"
        )

    selected = eligible[:N_PAIRS]

    selected.sort(
        key=lambda x: x["pair_id"]
    )

    selected_pinyins = {
        pair["pinyin"]
        for pair in selected
    }

    if len(selected_pinyins) != N_PAIRS:
        raise AssertionError(
            "Selected Pinyin values are not unique"
        )

    print()
    print(
        "===== SELECTED CLEAN PAIRS ====="
    )

    for pair in selected:
        print(
            f"{pair['pair_id']} "
            f"{pair['pinyin']} | "
            f"{pair['candidate_a']} "
            f"({pair['support_a']}) vs "
            f"{pair['candidate_b']} "
            f"({pair['support_b']})"
        )

    # Freeze held-out evaluation rows FIRST.
    eval_ids: set[str] = set()
    evaluation_probes = []
    train_pools = {}

    for pair_index, pair in enumerate(
        selected
    ):

        pinyin = pair["pinyin"]
        a = pair["candidate_a"]
        b = pair["candidate_b"]

        a_rows = list(
            index[(pinyin, a)]
        )

        b_rows = list(
            index[(pinyin, b)]
        )

        rng_a = random.Random(
            SEED
            + 10000
            + pair_index * 2
        )

        rng_b = random.Random(
            SEED
            + 10001
            + pair_index * 2
        )

        rng_a.shuffle(a_rows)
        rng_b.shuffle(b_rows)

        eval_a = a_rows[
            :EVAL_PER_SIDE
        ]

        eval_b = b_rows[
            :EVAL_PER_SIDE
        ]

        train_a = a_rows[
            EVAL_PER_SIDE:
        ]

        train_b = b_rows[
            EVAL_PER_SIDE:
        ]

        if not train_a or not train_b:
            raise RuntimeError(
                f"No training pool after eval holdout: "
                f"{pair['pair_id']}"
            )

        for side, candidate, selected_rows in (
            ("A", a, eval_a),
            ("B", b, eval_b),
        ):
            for row in selected_rows:

                rid = row_id(row)

                if rid in eval_ids:
                    raise RuntimeError(
                        f"Duplicate eval row: {rid}"
                    )

                eval_ids.add(rid)

                evaluation_probes.append(
                    {
                        "pair_id": pair["pair_id"],
                        "pinyin": pinyin,
                        "candidate_a": a,
                        "candidate_b": b,
                        "context_side": side,
                        "context_gold": candidate,
                        "source_row_id": rid,
                        "context": str(
                            row.get(
                                "context",
                                "",
                            )
                        ),
                    }
                )

        train_pools[
            pair["pair_id"]
        ] = {
            "A": train_a,
            "B": train_b,
        }

    expected_eval = (
        N_PAIRS
        * 2
        * EVAL_PER_SIDE
    )

    if (
        len(evaluation_probes)
        != expected_eval
    ):
        raise AssertionError(
            "Evaluation probe count mismatch"
        )

    if len(eval_ids) != expected_eval:
        raise AssertionError(
            "Evaluation row IDs are not unique"
        )

    # Controlled-but-natural background:
    # exclude only the selected A/B targets for each
    # controlled Pinyin. Other naturally occurring
    # candidates with the same Pinyin remain.
    controlled_targets = {
        (
            pair["pinyin"],
            pair["candidate_a"],
        )
        for pair in selected
    } | {
        (
            pair["pinyin"],
            pair["candidate_b"],
        )
        for pair in selected
    }

    background_pool = []

    for row in rows:

        rid = row_id(row)

        if rid in eval_ids:
            continue

        key = (
            row_pinyin(row),
            row_target(row),
        )

        if key in controlled_targets:
            continue

        background_pool.append(row)

    background_pool = unique_rows(
        background_pool
    )

    needed_background = (
        BACKGROUND_ROWS_PER_STAGE
        * len(STAGES)
    )

    if (
        len(background_pool)
        < needed_background
    ):
        raise RuntimeError(
            f"Need {needed_background} unique "
            f"background rows, have "
            f"{len(background_pool)}"
        )

    rng_bg = random.Random(
        SEED + 50000
    )

    rng_bg.shuffle(
        background_pool
    )

    background_blocks = {}

    for stage_index, (
        stage_name,
        _,
    ) in enumerate(STAGES):

        start = (
            stage_index
            * BACKGROUND_ROWS_PER_STAGE
        )

        end = (
            start
            + BACKGROUND_ROWS_PER_STAGE
        )

        background_blocks[
            stage_name
        ] = background_pool[
            start:end
        ]

    bg_id_sets = [
        {
            row_id(row)
            for row in background_blocks[
                stage_name
            ]
        }
        for stage_name, _ in STAGES
    ]

    for i in range(
        len(bg_id_sets)
    ):
        for j in range(
            i + 1,
            len(bg_id_sets),
        ):
            overlap = (
                bg_id_sets[i]
                & bg_id_sets[j]
            )

            if overlap:
                raise RuntimeError(
                    "Background stage overlap "
                    f"detected: {len(overlap)}"
                )

    stage_audits = []

    for stage_index, (
        stage_name,
        preferred_side,
    ) in enumerate(STAGES):

        stage_rows = list(
            background_blocks[
                stage_name
            ]
        )

        controlled_audit = []

        for pair_index, pair in enumerate(
            selected
        ):

            pool = list(
                train_pools[
                    pair["pair_id"]
                ][preferred_side]
            )

            rng_control = random.Random(
                SEED
                + 70000
                + stage_index * 1000
                + pair_index
            )

            rng_control.shuffle(pool)

            controlled_rows = [
                pool[
                    i % len(pool)
                ]
                for i in range(
                    CONTROL_PER_PAIR
                )
            ]

            stage_rows.extend(
                controlled_rows
            )

            controlled_audit.append(
                {
                    "pair_id": pair[
                        "pair_id"
                    ],
                    "pinyin": pair[
                        "pinyin"
                    ],
                    "side": preferred_side,
                    "candidate": (
                        pair[
                            "candidate_a"
                        ]
                        if preferred_side
                        == "A"
                        else pair[
                            "candidate_b"
                        ]
                    ),
                    "real_training_pool_rows": len(
                        pool
                    ),
                    "controlled_exposures": len(
                        controlled_rows
                    ),
                    "unique_control_row_ids_used": len(
                        {
                            row_id(x)
                            for x
                            in controlled_rows
                        }
                    ),
                }
            )

        if (
            len(stage_rows)
            != TOTAL_ROWS_PER_STAGE
        ):
            raise AssertionError(
                f"{stage_name}: "
                f"{len(stage_rows)} rows"
            )

        if any(
            row_id(row) in eval_ids
            for row in stage_rows
        ):
            raise RuntimeError(
                f"{stage_name}: evaluation leakage"
            )

        rng_stage = random.Random(
            SEED
            + 90000
            + stage_index
        )

        rng_stage.shuffle(
            stage_rows
        )

        path = (
            STAGE_ROOT
            / f"{stage_name}.jsonl"
        )

        write_jsonl(
            path,
            stage_rows,
        )

        stage_audits.append(
            {
                "stage": stage_name,
                "preferred_side": preferred_side,
                "rows": len(stage_rows),
                "background_rows": (
                    BACKGROUND_ROWS_PER_STAGE
                ),
                "controlled_rows": (
                    CONTROL_ROWS_PER_STAGE
                ),
                "control_per_pair": (
                    CONTROL_PER_PAIR
                ),
                "unique_row_ids": len(
                    {
                        row_id(x)
                        for x in stage_rows
                    }
                ),
                "controlled": controlled_audit,
                "path": str(path),
            }
        )

    selected_payload = []

    for pair in selected:

        selected_payload.append(
            {
                "pair_id": pair["pair_id"],
                "pinyin": pair["pinyin"],
                "candidate_a": pair[
                    "candidate_a"
                ],
                "candidate_b": pair[
                    "candidate_b"
                ],
                "support_a_total": pair[
                    "support_a"
                ],
                "support_b_total": pair[
                    "support_b"
                ],
                "eval_a_rows": (
                    EVAL_PER_SIDE
                ),
                "eval_b_rows": (
                    EVAL_PER_SIDE
                ),
                "train_a_real_rows": len(
                    train_pools[
                        pair["pair_id"]
                    ]["A"]
                ),
                "train_b_real_rows": len(
                    train_pools[
                        pair["pair_id"]
                    ]["B"]
                ),
            }
        )

    write_json(
        OUT / "selected_pairs.json",
        {
            "experiment": EXPERIMENT,
            "selection_policy": (
                "one pair per unique pinyin; "
                "maximize min(A_count,B_count), "
                "then total evidence; choose "
                "top 15 eligible unique pinyins"
            ),
            "pairs": selected_payload,
            "test_used": False,
        },
    )

    with (
        OUT / "selected_pairs.csv"
    ).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                selected_payload[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            selected_payload
        )

    write_jsonl(
        OUT / "evaluation_probes.jsonl",
        evaluation_probes,
    )

    write_json(
        OUT / "evaluation_manifest.json",
        {
            "experiment": EXPERIMENT,
            "pairs": N_PAIRS,
            "contexts_per_pair": (
                EVAL_PER_SIDE * 2
            ),
            "contexts_per_side": (
                EVAL_PER_SIDE
            ),
            "evaluation_rows": len(
                evaluation_probes
            ),
            "evaluation_unique_row_ids": len(
                eval_ids
            ),
            "balanced_A_B_contexts": True,
            "frozen_before_training": True,
            "training_leakage_allowed": False,
            "test_used": False,
        },
    )

    write_json(
        OUT / "stage_plan.json",
        {
            "experiment": EXPERIMENT,
            "author": AUTHOR,
            "seed": SEED,
            "pairs": N_PAIRS,
            "unique_pinyin_pairs": True,
            "eval_per_side": EVAL_PER_SIDE,
            "control_per_pair": CONTROL_PER_PAIR,
            "rows_per_stage": (
                TOTAL_ROWS_PER_STAGE
            ),
            "controlled_rows_per_stage": (
                CONTROL_ROWS_PER_STAGE
            ),
            "background_rows_per_stage": (
                BACKGROUND_ROWS_PER_STAGE
            ),
            "background_policy": (
                "exclude only selected A/B targets; retain other same-pinyin candidates; "
                "three disjoint deterministic "
                "background blocks"
            ),
            "controlled_policy": (
                "real Agent Phage rows only; "
                "held-out eval rows excluded; "
                "oversample remaining real rows "
                "when fewer than 40"
            ),
            "stage_sequence": [
                {
                    "stage": name,
                    "preferred_side": side,
                }
                for name, side in STAGES
            ],
            "stage_audits": stage_audits,
            "test_used": False,
        },
    )

    print()
    print(
        "===== R4 CLEAN V2 COMPLETE ====="
    )
    print(
        f"selected_pairs={N_PAIRS}"
    )
    print(
        f"unique_pinyins="
        f"{len(selected_pinyins)}"
    )
    print(
        f"evaluation_rows="
        f"{len(evaluation_probes)}"
    )
    print(
        f"background_pool="
        f"{len(background_pool)}"
    )

    for audit in stage_audits:
        print(
            f"{audit['stage']} "
            f"rows={audit['rows']} "
            f"background="
            f"{audit['background_rows']} "
            f"controlled="
            f"{audit['controlled_rows']} "
            f"unique_ids="
            f"{audit['unique_row_ids']}"
        )

    print("test_used=false")


if __name__ == "__main__":
    main()
