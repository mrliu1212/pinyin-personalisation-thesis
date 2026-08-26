from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np


ROOT = Path(r"C:\Users\chiar\Desktop\LBH")

LF_REPO = ROOT / "thesis-learned-fusion-lab"

FULL_MATRIX_ROOT = (
    ROOT
    / "thesis-external-memory-next"
    / "results"
    / "personalisation"
    / "external_memory_next"
    / "lambdamart_matrices_v1"
)

FULL_STAGE1 = (
    ROOT
    / "thesis-context-compare"
    / "results"
    / "personalisation"
    / "context_comparison_followup_v1"
    / "full_retune_final_trainval_dev_v1"
    / "tune"
    / "train_val_stage1_features.jsonl"
)

FULL_SHAPE = (
    LF_REPO
    / "results"
    / "personalisation"
    / "learned_fusion_lab"
    / "lf005_task_shape_fusion_v1"
    / "features"
    / "val_shape_X.npy"
)

ILT_MODEL = (
    LF_REPO
    / "results"
    / "personalisation"
    / "initial_learned_fusion_transfer"
    / "ilt004_task_fusion_v1"
    / "generic_task_shape_model.txt"
)

FULL_MODEL = (
    ROOT
    / "thesis-external-memory-next"
    / "results"
    / "personalisation"
    / "external_memory_next"
    / "lambdamart_fusion_v1"
    / "models"
    / "d5_m500_r100.txt"
)

OUT_ROOT = (
    Path("results")
    / "personalisation"
    / "ilt004d_to_full_zero_shot_v1"
)

AUTHOR = "Agent Phage"


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            row = json.loads(line)

            if (
                str(row.get("source_split", "")).lower() == "test"
                or bool(row.get("used_test", False))
            ):
                raise RuntimeError(
                    f"Closed Test marker: {path}:{number}"
                )

            yield row


def rank_of(gold: str, ranked: list[str]):
    try:
        return ranked.index(gold) + 1
    except ValueError:
        return None


def metrics(ranks):
    n = len(ranks)

    def top(k):
        return sum(
            rank is not None and rank <= k
            for rank in ranks
        ) / n

    return {
        "n": n,
        "top1": top(1),
        "top3": top(3),
        "top5": top(5),
        "mrr_at_10": sum(
            0.0 if rank is None else 1.0 / rank
            for rank in ranks
        ) / n,
        "missing10": sum(
            rank is None
            for rank in ranks
        ) / n,
    }


def transition(before, after):
    out = {
        "n": len(before),
        "rescue": 0,
        "harm": 0,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
    }

    for old, new in zip(before, after):
        old_correct = old == 1
        new_correct = new == 1

        if new_correct and not old_correct:
            out["rescue"] += 1
        elif old_correct and not new_correct:
            out["harm"] += 1
        elif old_correct:
            out["unchanged_correct"] += 1
        else:
            out["unchanged_wrong"] += 1

    out["net"] = out["rescue"] - out["harm"]
    return out


def rank_queries(scores, meta, groups):
    ranks = []
    top10s = []

    offset = 0

    for group_size, row in zip(groups, meta):
        n = int(group_size)

        candidates = list(map(str, row["candidates"]))

        if len(candidates) != n:
            raise RuntimeError(
                f"Candidate/group mismatch: {row['row_id']}"
            )

        query_scores = scores[offset:offset + n]

        order = sorted(
            range(n),
            key=lambda i: (
                -float(query_scores[i]),
                i,
            ),
        )

        ranked = [
            candidates[i]
            for i in order
        ]

        top10s.append(ranked)
        ranks.append(
            rank_of(str(row["gold"]), ranked)
        )

        offset += n

    if offset != len(scores):
        raise RuntimeError("Score/group coverage mismatch")

    return ranks, top10s


# Import the exact Initial feature definitions.
sys.path.insert(0, str(LF_REPO))

from experiments.initial_learned_fusion_transfer.initial_features_v1 import (
    FEATURE_NAMES,
    INTERACTION_FEATURE_NAMES,
    compact_interactions,
)

from src.personalisation.task_retrieval_shape import (
    SHAPE_FEATURE_NAMES,
)


manifest = json.loads(
    (FULL_MATRIX_ROOT / "matrix_manifest.json").read_text(
        encoding="utf-8"
    )
)

full_feature_names = tuple(manifest["feature_names"])

full_x = np.load(
    FULL_MATRIX_ROOT / "val_X.npy",
    mmap_mode="r",
)

groups = np.load(
    FULL_MATRIX_ROOT / "val_group.npy",
    mmap_mode="r",
)

meta = list(
    iter_jsonl(
        FULL_MATRIX_ROOT / "val_query_meta.jsonl"
    )
)

shape_x = np.load(
    FULL_SHAPE,
    mmap_mode="r",
)


# META-DERIVED GROUP RECONSTRUCTION
#
# Full val_query_meta retains every Train-Val query, including any
# zero-candidate query. LightGBM val_group.npy cannot encode a
# zero-length group, so reconstruct the complete query grouping from
# candidate_count and require the stored group array to match exactly
# after zero groups are removed.
groups_file = np.asarray(groups, dtype=np.int64)

meta_groups = np.asarray(
    [int(row["candidate_count"]) for row in meta],
    dtype=np.int64,
)

zero_group_indices = np.flatnonzero(meta_groups == 0)

if len(groups_file) != len(meta_groups):
    nonzero_meta_groups = meta_groups[meta_groups > 0]

    if not np.array_equal(groups_file, nonzero_meta_groups):
        raise RuntimeError(
            "Stored val_group.npy does not exactly equal the "
            "non-zero candidate_count sequence from val_query_meta."
        )

    if int(meta_groups.sum()) != len(full_x):
        raise RuntimeError(
            f"Meta candidate counts do not cover Full matrix: "
            f"{int(meta_groups.sum())} vs {len(full_x)}"
        )

    print(
        "GROUP RECONSTRUCTION:",
        {
            "meta_queries": len(meta_groups),
            "stored_nonzero_groups": len(groups_file),
            "zero_candidate_queries": len(zero_group_indices),
            "zero_group_indices": zero_group_indices.tolist(),
            "candidate_rows": int(meta_groups.sum()),
            "stored_group_rows": int(groups_file.sum()),
        },
    )

    groups = meta_groups
else:
    groups = meta_groups

if len(meta) != len(groups):
    raise RuntimeError(
        f"Meta/group mismatch: {len(meta)} vs {len(groups)}"
    )

if int(np.asarray(groups).sum()) != len(full_x):
    raise RuntimeError("Full matrix group coverage changed")

if len(shape_x) != len(full_x):
    raise RuntimeError(
        f"Shape/matrix mismatch: "
        f"{len(shape_x)} vs {len(full_x)}"
    )


full_col = {
    name: np.asarray(
        full_x[:, index],
        dtype=np.float32,
    )
    for index, name in enumerate(full_feature_names)
}


stage1 = {
    str(row["row_id"]): row
    for row in iter_jsonl(FULL_STAGE1)
}

if len(stage1) != len(meta):
    raise RuntimeError(
        f"Stage1/meta mismatch: "
        f"{len(stage1)} vs {len(meta)}"
    )


# Allocate exact Initial raw feature matrix:
# 27 Initial features, but on the frozen Full candidate surface.
initial_raw = np.zeros(
    (len(full_x), len(FEATURE_NAMES)),
    dtype=np.float32,
)


offset = 0

personal_candidates = 0
count_reconstruction_max_error = 0.0


for group_index, (group_size, m) in enumerate(
    zip(groups, meta)
):
    n = int(group_size)

    rid = str(m["row_id"])

    if rid not in stage1:
        raise RuntimeError(
            f"Missing Full Stage1 row: {rid}"
        )

    s1 = stage1[rid]

    candidates = list(map(str, m["candidates"]))

    if len(candidates) != n:
        raise RuntimeError(
            f"Candidate count changed: {rid}"
        )

    # Full Train-Val contains two legitimate zero-candidate queries.
    # They remain in the query population and will evaluate as missing,
    # but contribute no candidate-level feature rows.
    if n == 0:
        continue

    sl = slice(offset, offset + n)

    source_personal = (
        np.asarray(
            full_col["source_personal"][sl],
            dtype=np.float64,
        )
        > 0.5
    )

    # -----------------------------
    # Initial frequency semantics
    # -----------------------------

    same_n = int(
        s1.get("same_pinyin_history_count", 0)
    )

    personal_k5 = list(
        map(str, s1.get("personal_k5", []))
    )

    choice_map = {
        str(k): float(v)
        for k, v in s1.get(
            "choice_share", {}
        ).items()
    }

    p_ng_map = {
        str(k): float(v)
        for k, v in s1.get(
            "p_ng", {}
        ).items()
    }

    personal_counts = {}

    for name in personal_k5:
        if name not in choice_map:
            raise RuntimeError(
                f"Missing choice_share: {rid} / {name}"
            )

        raw_count = (
            choice_map[name] * same_n
        )

        count = int(round(raw_count))

        error = abs(raw_count - count)

        count_reconstruction_max_error = max(
            count_reconstruction_max_error,
            error,
        )

        if error > 1e-8:
            raise RuntimeError(
                f"Non-integral recovered count: "
                f"{rid} / {name} / {raw_count}"
            )

        personal_counts[name] = count

    personal_denominator = max(
        (
            math.log1p(personal_counts[name])
            for name in personal_k5
        ),
        default=0.0,
    )

    # Initial personal rank definition:
    # sorted by descending p_ng, then original
    # personal_k5 order, then candidate string.
    p_order = sorted(
        range(len(personal_k5)),
        key=lambda i: (
            -float(
                p_ng_map.get(
                    personal_k5[i],
                    0.0,
                )
            ),
            i,
            personal_k5[i],
        ),
    )

    personal_rank = {
        personal_k5[index]: rank
        for rank, index in enumerate(
            p_order,
            start=1,
        )
    }

    generic_rows = s1.get(
        "generic_frequency_candidates",
        [],
    )

    if not generic_rows:
        raise RuntimeError(
            f"No generic frequency rows: {rid}"
        )

    boundary = min(
        float(row["normalized_generic_score"])
        for row in generic_rows
    )

    concentration = float(
        s1["entropy_concentration"]
    )

    # -----------------------------
    # Candidate-level raw values
    # -----------------------------

    generic_score = np.zeros(
        n, dtype=np.float64
    )
    normalized_generic = np.zeros(
        n, dtype=np.float64
    )
    generic_rank = np.zeros(
        n, dtype=np.float64
    )
    frequency_count = np.zeros(
        n, dtype=np.float64
    )
    frequency_support = np.zeros(
        n, dtype=np.float64
    )
    personal_candidate_rank = np.zeros(
        n, dtype=np.float64
    )
    p_ng = np.zeros(
        n, dtype=np.float64
    )
    choice_share = np.zeros(
        n, dtype=np.float64
    )
    base_score = np.zeros(
        n, dtype=np.float64
    )

    for i, candidate in enumerate(candidates):

        if not source_personal[i]:
            generic_score[i] = float(
                full_col["generic_score"][
                    offset + i
                ]
            )

            normalized_generic[i] = float(
                full_col[
                    "normalized_generic_score"
                ][offset + i]
            )

            generic_rank[i] = float(
                full_col["generic_rank"][
                    offset + i
                ]
            )

            frequency_count[i] = float(
                full_col["frequency_count"][
                    offset + i
                ]
            )

            # Exact bridge:
            # Full generic personal_score ==
            # Initial generic frequency_support.
            frequency_support[i] = float(
                full_col["personal_score"][
                    offset + i
                ]
            )

            base_score[i] = (
                normalized_generic[i]
                + 4.0 * frequency_support[i]
            )

        else:
            personal_candidates += 1

            if candidate not in personal_counts:
                raise RuntimeError(
                    f"Personal candidate not in "
                    f"personal_k5: {rid} / "
                    f"{candidate}"
                )

            count = personal_counts[candidate]

            frequency_count[i] = float(count)

            if personal_denominator <= 0.0:
                raise RuntimeError(
                    f"Invalid personal denominator: "
                    f"{rid}"
                )

            frequency_support[i] = (
                math.log1p(count)
                / personal_denominator
            )

            personal_candidate_rank[i] = float(
                personal_rank[candidate]
            )

            p_ng[i] = float(
                p_ng_map.get(candidate, 0.0)
            )

            choice_share[i] = float(
                choice_map[candidate]
            )

            # Exact Initial balanced Stage1
            # personal recovery semantics.
            base_score[i] = (
                boundary
                + 4.0 * p_ng[i]
                + 4.0 * choice_share[i]
                + 2.0 * concentration
            )

    # -----------------------------
    # Initial-style base ranking
    # on the frozen Full surface
    # -----------------------------

    base_order = sorted(
        range(n),
        key=lambda i: (
            -float(base_score[i]),
            1 if source_personal[i] else 0,
            int(
                personal_candidate_rank[i]
                if source_personal[i]
                else generic_rank[i]
            ),
            candidates[i],
        ),
    )

    base_rank = np.zeros(
        n,
        dtype=np.float64,
    )

    for rank, index in enumerate(
        base_order,
        start=1,
    ):
        base_rank[index] = rank

    # Existing memory retrieval evidence is
    # regime-appropriate Full evidence.
    ngram = np.asarray(
        full_col["ngram_support"][sl],
        dtype=np.float64,
    )

    bge = np.asarray(
        full_col["bge_support"][sl],
        dtype=np.float64,
    )

    # Exact Initial frozen linear formula.
    frozen_score = (
        base_score
        + 4.0 * ngram
        + 6.0 * bge
    )

    frozen_order = sorted(
        range(n),
        key=lambda i: (
            -float(frozen_score[i]),
            int(base_rank[i]),
            candidates[i],
        ),
    )

    frozen_rank = np.zeros(
        n,
        dtype=np.float64,
    )

    for rank, index in enumerate(
        frozen_order,
        start=1,
    ):
        frozen_rank[index] = rank

    max_base = float(
        np.max(base_score)
    )
    max_ngram = float(
        np.max(ngram)
    )
    max_bge = float(
        np.max(bge)
    )

    values = {
        "base_score":
            base_score,
        "base_rank":
            base_rank,
        "source_personal":
            source_personal.astype(np.float64),
        "has_generic":
            (~source_personal).astype(np.float64),
        "generic_score":
            generic_score,
        "normalized_generic_score":
            normalized_generic,
        "generic_rank":
            generic_rank,
        "frequency_count":
            frequency_count,
        "frequency_support":
            frequency_support,
        "has_personal":
            source_personal.astype(np.float64),
        "personal_candidate_rank":
            personal_candidate_rank,
        "p_ng":
            p_ng,
        "choice_share":
            choice_share,
        "entropy_concentration":
            np.full(
                n,
                concentration,
                dtype=np.float64,
            ),
        "query_ambiguous":
            np.full(
                n,
                float(bool(m["ambiguous"])),
                dtype=np.float64,
            ),
        "log1p_same_pinyin_history":
            np.asarray(
                full_col[
                    "log1p_same_pinyin_history"
                ][sl],
                dtype=np.float64,
            ),
        "log1p_raw_history":
            np.asarray(
                full_col[
                    "log1p_raw_history"
                ][sl],
                dtype=np.float64,
            ),
        "ngram_support":
            ngram,
        "bge_support":
            bge,
        "log1p_bge_history_count":
            np.asarray(
                full_col[
                    "log1p_bge_history_count"
                ][sl],
                dtype=np.float64,
            ),
        "ngram_effective_n":
            np.asarray(
                full_col[
                    "ngram_effective_n"
                ][sl],
                dtype=np.float64,
            ),
        "log1p_ngram_matched_history":
            np.asarray(
                full_col[
                    "log1p_ngram_matched_history"
                ][sl],
                dtype=np.float64,
            ),

        # IMPORTANT:
        # Initial signs are max - candidate.
        "base_gap_to_top":
            max_base - base_score,
        "ngram_gap_to_top":
            max_ngram - ngram,
        "bge_gap_to_top":
            max_bge - bge,

        "frozen_linear_score":
            frozen_score,
        "frozen_rank":
            frozen_rank,
    }

    for feature_index, name in enumerate(
        FEATURE_NAMES
    ):
        initial_raw[
            sl,
            feature_index,
        ] = np.asarray(
            values[name],
            dtype=np.float32,
        )

    offset += n


if offset != len(full_x):
    raise RuntimeError(
        "Full candidate coverage mismatch"
    )

if not np.isfinite(initial_raw).all():
    raise RuntimeError(
        "Non-finite Initial-style feature"
    )


print("FULL RAW MATRIX:", full_x.shape)
print(
    "INITIAL-SEMANTIC RAW MATRIX:",
    initial_raw.shape,
)
print("FULL TASK SHAPE:", shape_x.shape)
print(
    "PERSONAL CANDIDATES:",
    personal_candidates,
)
print(
    "MAX COUNT RECONSTRUCTION ERROR:",
    count_reconstruction_max_error,
)


# Exact Initial compact interaction function.
interactions = compact_interactions(
    initial_raw,
    FEATURE_NAMES,
    groups_file,
)

ilt_x = np.concatenate(
    (
        initial_raw,
        np.asarray(
            interactions,
            dtype=np.float32,
        ),
        np.asarray(
            shape_x,
            dtype=np.float32,
        ),
    ),
    axis=1,
)


expected_names = [
    *FEATURE_NAMES,
    *INTERACTION_FEATURE_NAMES,
    *SHAPE_FEATURE_NAMES,
]


ilt_model = lgb.Booster(
    model_file=str(ILT_MODEL)
)

if ilt_model.num_feature() != len(
    expected_names
):
    raise RuntimeError(
        f"ILT model feature count changed: "
        f"{ilt_model.num_feature()} vs "
        f"{len(expected_names)}"
    )

model_names = list(
    ilt_model.feature_name()
)

if model_names != expected_names:
    raise RuntimeError(
        "ILT-004 D feature names/order changed.\n"
        f"Expected: {expected_names}\n"
        f"Model: {model_names}"
    )


print(
    "ILT-004 D MATRIX:",
    ilt_x.shape,
)
print(
    "ILT MODEL FEATURES:",
    ilt_model.num_feature(),
)


# ----------------------------------
# Frozen ILT-004 D -> Full
# ----------------------------------

ilt_scores = ilt_model.predict(
    ilt_x,
    num_iteration=
        ilt_model.current_iteration(),
)

ilt_ranks, ilt_top10 = rank_queries(
    ilt_scores,
    meta,
    groups,
)


# ----------------------------------
# Frozen Full-specialized LambdaMART
# reference on its native matrix
# ----------------------------------

full_model = lgb.Booster(
    model_file=str(FULL_MODEL)
)

full_scores = full_model.predict(
    full_x,
    num_iteration=
        full_model.current_iteration(),
)

full_ranks, full_top10 = rank_queries(
    full_scores,
    meta,
    groups,
)


agent_indices = [
    i
    for i, row in enumerate(meta)
    if str(row["author"]) == AUTHOR
]

full_agent = [
    full_ranks[i]
    for i in agent_indices
]

ilt_agent = [
    ilt_ranks[i]
    for i in agent_indices
]


both_correct = sum(
    full_agent[i] == 1
    and ilt_agent[i] == 1
    for i in range(len(agent_indices))
)

full_only = sum(
    full_agent[i] == 1
    and ilt_agent[i] != 1
    for i in range(len(agent_indices))
)

ilt_only = sum(
    full_agent[i] != 1
    and ilt_agent[i] == 1
    for i in range(len(agent_indices))
)

neither = (
    len(agent_indices)
    - both_correct
    - full_only
    - ilt_only
)


result = {
    "status": "complete",
    "experiment":
        "ilt004d_to_full_zero_shot_v1",
    "description": (
        "Frozen Initial-specialized ILT-004 D "
        "applied zero-shot to the frozen Full "
        "candidate surface. Full evidence was "
        "reconstructed under exact Initial "
        "27-feature semantics; Full task-shape "
        "features were reused in exact candidate "
        "order."
    ),
    "candidate_surface":
        "frozen Full LambdaMART surface",
    "feature_semantics":
        "Initial / ILT-004 D",
    "feature_count":
        int(ilt_x.shape[1]),
    "personal_frequency_reconstruction": {
        "personal_candidates":
            personal_candidates,
        "max_integer_error":
            count_reconstruction_max_error,
    },
    "metrics": {
        "FullFrozenLambdaMART_Overall":
            metrics(full_ranks),
        "ILT004D_to_Full_Overall":
            metrics(ilt_ranks),
        "FullFrozenLambdaMART_Agent":
            metrics(full_agent),
        "ILT004D_to_Full_Agent":
            metrics(ilt_agent),
    },
    "agent_complementarity": {
        "both_top1_correct":
            both_correct,
        "full_only_top1_correct":
            full_only,
        "ilt_only_top1_correct":
            ilt_only,
        "neither_top1_correct":
            neither,
        "oracle_either_top1":
            (
                both_correct
                + full_only
                + ilt_only
            )
            / len(agent_indices),
    },
    "agent_transition": transition(
        full_agent,
        ilt_agent,
    ),
    "training": False,
    "tuning": False,
    "used_dev3000": False,
    "used_test": False,
}


OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

(
    OUT_ROOT / "result.json"
).write_text(
    json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


with (
    OUT_ROOT / "predictions.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:

    for i, row in enumerate(meta):

        payload = {
            "row_id":
                str(row["row_id"]),
            "author":
                str(row["author"]),
            "gold":
                str(row["gold"]),
            "FullFrozenLambdaMART_rank":
                full_ranks[i],
            "ILT004D_to_Full_rank":
                ilt_ranks[i],
            "FullFrozenLambdaMART_top10":
                full_top10[i],
            "ILT004D_to_Full_top10":
                ilt_top10[i],
            "used_dev3000": False,
            "used_test": False,
        }

        f.write(
            json.dumps(
                payload,
                ensure_ascii=False,
            )
            + "\n"
        )


print()
print("===== ZERO-SHOT RESULTS =====")
print(
    json.dumps(
        result["metrics"],
        indent=2,
    )
)

print()
print(
    "===== AGENT COMPLEMENTARITY ====="
)
print(
    json.dumps(
        result["agent_complementarity"],
        indent=2,
    )
)

print()
print("===== AGENT TRANSITION =====")
print(
    json.dumps(
        result["agent_transition"],
        indent=2,
    )
)

print()
print(
    "RESULT:",
    OUT_ROOT / "result.json",
)
print(
    "PREDICTIONS:",
    OUT_ROOT / "predictions.jsonl",
)
