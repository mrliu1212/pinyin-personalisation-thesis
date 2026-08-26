from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np


ROOT = Path(r"C:\Users\chiar\Desktop\LBH")

INITIAL_ROOT = (
    ROOT
    / "thesis-learned-fusion-lab"
    / "results"
    / "personalisation"
    / "initial_learned_fusion_transfer"
)

MATRIX_ROOT = INITIAL_ROOT / "ilt001_matrix_v1"

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

ILT004_PREDICTIONS = (
    INITIAL_ROOT
    / "ilt004_task_fusion_v1"
    / "predictions.jsonl"
)

OUT_ROOT = (
    Path("results")
    / "personalisation"
    / "full_em_to_initial_zero_shot_v1"
)

FULL_FEATURES = (
    "base_score",
    "base_rank",
    "source_personal",
    "has_generic",
    "generic_score",
    "normalized_generic_score",
    "generic_rank",
    "frequency_count",
    "personal_score",
    "has_personal",
    "personal_candidate_rank",
    "p_ng",
    "choice_share",
    "entropy_concentration",
    "log1p_same_pinyin_history",
    "log1p_raw_history",
    "ngram_support",
    "bge_support",
    "log1p_bge_history_count",
    "ngram_effective_n",
    "log1p_ngram_matched_history",
    "base_gap_to_top",
    "ngram_gap_to_top",
    "bge_gap_to_top",
    "frozen_linear_score",
)


def load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if bool(row.get("used_test", False)):
                    raise RuntimeError(f"used_test=true in {path}")
                rows.append(row)
    return rows


def rank_of(gold, ranked):
    try:
        return ranked.index(gold) + 1
    except ValueError:
        return None


def metrics(ranks):
    n = len(ranks)

    def top(k):
        return sum(r is not None and r <= k for r in ranks) / n

    return {
        "n": n,
        "top1": top(1),
        "top3": top(3),
        "top5": top(5),
        "mrr_at_10": sum(
            0.0 if r is None else 1.0 / r
            for r in ranks
        ) / n,
        "missing10": sum(r is None for r in ranks) / n,
    }


manifest = json.loads(
    (MATRIX_ROOT / "matrix_manifest.json").read_text(encoding="utf-8")
)

initial_features = tuple(manifest["feature_names"])

X_initial = np.load(MATRIX_ROOT / "val_X.npy")
groups = np.load(MATRIX_ROOT / "val_group.npy")
meta = load_jsonl(MATRIX_ROOT / "val_query_meta.jsonl")
ilt004 = load_jsonl(ILT004_PREDICTIONS)

if len(meta) != len(groups):
    raise RuntimeError(
        f"meta/groups mismatch: {len(meta)} vs {len(groups)}"
    )

if int(groups.sum()) != len(X_initial):
    raise RuntimeError("group sizes do not cover Initial matrix")

if any(int(x) != 10 for x in groups):
    raise RuntimeError("Expected fixed Initial Top10 surface")

col = {
    name: X_initial[:, index]
    for index, name in enumerate(initial_features)
}

missing = [
    name
    for name in FULL_FEATURES
    if name != "personal_score" and name not in col
]

if missing:
    raise RuntimeError(
        f"Cannot reconstruct Full feature schema: {missing}"
    )

source_personal = np.asarray(
    col["source_personal"],
    dtype=np.float32,
)

frequency_support = np.asarray(
    col["frequency_support"],
    dtype=np.float32,
)

# Exact Full semantics:
# generic_frequency row -> original personal_score
# personal_recovery row -> missing personal_score -> 0.0
personal_score = np.where(
    source_personal > 0.5,
    0.0,
    frequency_support,
).astype(np.float32)

columns = []

for name in FULL_FEATURES:
    if name == "personal_score":
        columns.append(personal_score)
    else:
        columns.append(
            np.asarray(col[name], dtype=np.float32)
        )

X_full_schema = np.column_stack(columns).astype(np.float32)

if X_full_schema.shape != (len(X_initial), 25):
    raise RuntimeError(
        f"Unexpected reconstructed shape: {X_full_schema.shape}"
    )

if not np.isfinite(X_full_schema).all():
    raise RuntimeError("Non-finite Full-schema feature detected")

model = lgb.Booster(model_file=str(FULL_MODEL))

if model.num_feature() != 25:
    raise RuntimeError(
        f"Full LambdaMART expects {model.num_feature()} features"
    )

scores = model.predict(X_full_schema)

ranks_all = []
ranks_agent = []
output = []

offset = 0

for query_index, (group_size, row) in enumerate(zip(groups, meta)):
    n = int(group_size)

    candidates = list(map(str, row["candidates"]))
    if len(candidates) != n:
        raise RuntimeError(
            f"candidate/group mismatch at query {query_index}"
        )

    query_scores = scores[offset : offset + n]

    ranked_indices = sorted(
        range(n),
        key=lambda i: (
            -float(query_scores[i]),
            i,
        ),
    )

    ranked = [
        candidates[i]
        for i in ranked_indices
    ]

    gold = str(row["gold"])
    rank = rank_of(gold, ranked)

    ranks_all.append(rank)

    if str(row["author"]) == "Agent Phage":
        ranks_agent.append(rank)

    output.append({
        "row_id": str(row["row_id"]),
        "author": str(row["author"]),
        "gold": gold,
        "rank": rank,
        "top10": ranked,
        "used_dev3000": False,
        "used_test": False,
    })

    offset += n


ilt004_by_id = {
    str(row["row_id"]): row
    for row in ilt004
}

ilt_agent_ranks = []

for row in meta:
    if str(row["author"]) != "Agent Phage":
        continue

    frozen = ilt004_by_id[str(row["row_id"])]

    ilt_agent_ranks.append(
        rank_of(
            str(row["gold"]),
            list(map(str, frozen["generic_task_shape_top10"])),
        )
    )


result = {
    "experiment": "full_em_to_initial_zero_shot_v1",
    "status": "complete",
    "description": (
        "Frozen Full LambdaMART applied zero-shot to Initial-Pinyin "
        "features reconstructed under the exact Full 25-feature schema."
    ),
    "full_model": str(FULL_MODEL),
    "feature_count": 25,
    "personal_score_reconstruction": {
        "generic": "Initial frequency_support",
        "personal_recovery": 0.0,
        "basis": (
            "Full builder uses row.get('personal_score', 0.0); "
            "Initial generic frequency_support preserves personal_score."
        ),
    },
    "metrics": {
        "FullEM_to_Initial_Overall": metrics(ranks_all),
        "FullEM_to_Initial_Agent": metrics(ranks_agent),
        "ILT004D_Agent_reference": metrics(ilt_agent_ranks),
    },
    "training": False,
    "tuning": False,
    "used_dev3000": False,
    "used_test": False,
}

OUT_ROOT.mkdir(parents=True, exist_ok=True)

(OUT_ROOT / "result.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

with (OUT_ROOT / "predictions.jsonl").open(
    "w",
    encoding="utf-8",
) as f:
    for row in output:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


print("FULL MODEL:", FULL_MODEL)
print("INITIAL MATRIX:", MATRIX_ROOT / "val_X.npy")
print("RECONSTRUCTED MATRIX:", X_full_schema.shape)
print("LIGHTGBM FEATURES:", model.num_feature())

print()
print("===== ZERO-SHOT RESULTS =====")
print(json.dumps(result["metrics"], indent=2))

print()
print("RESULT:", OUT_ROOT / "result.json")
print("PREDICTIONS:", OUT_ROOT / "predictions.jsonl")


if __name__ == "__main__":
    pass
