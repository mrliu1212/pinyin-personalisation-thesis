#!/bin/bash
#SBATCH --job-name=final-val-rich30
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=results/finalmodel_fiveauthor_v1/val_end_to_end_v1/slurm-%j.out
#SBATCH --error=results/finalmodel_fiveauthor_v1/val_end_to_end_v1/slurm-%j.err

set -euo pipefail

cd /home/3160454/work/thesis-finalmodel-sixauthor-dataset-v1

PY=/home/3160454/work/envs/model-level-py312/bin/python
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

ROOT=results/finalmodel_fiveauthor_v1
OUT=$ROOT/val_end_to_end_v1

VAL=$ROOT/adapter_input_preparation_v1/adapter_val_manifest_frozen_v1.jsonl
FIT=$ROOT/adapter_input_preparation_v1/adapter_fit_manifest_frozen_v1.jsonl

BOUNDARIES=$ROOT/h5000_preparation_v1/h5000_query_boundaries_frozen_v1.jsonl

WORK_SPLIT=$ROOT/dataset_preparation_v1/work_split_manifest_frozen_v1.csv
FULL_MANIFEST=$ROOT/dataset_preparation_v1/final_fiveauthor_manifest_frozen_v1.jsonl

RAW=/home/3160454/work/final-sixauthor-source-v1/data/processed/deep_author/interactions_t1_ready.jsonl

CHECKPOINT=/home/3160454/work/model-level-assets-20260822/pinyingpt2-concat

ADAPTER_ROOT=$ROOT/adapter_training_v1

ADAPTER_OUT=$OUT/adapter_val_top10
H5000_OUT=$OUT/val_h5000_compact_v1.jsonl
H5000_SUMMARY=$OUT/val_h5000_compact_summary_v1.json
RICH_OUT=$OUT/rich30_training_v1

mkdir -p \
"$OUT" \
"$ADAPTER_OUT" \
"$RICH_OUT"

echo "========================================"
echo "FINAL FIVE-AUTHOR VAL END-TO-END V1"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA}"
echo "========================================"

echo
echo "===== ENVIRONMENT ====="

"$PY" - <<'PY'
import sys
import torch
import numpy
import lightgbm
import safetensors

print("PYTHON =", sys.version)
print("TORCH =", torch.__version__)
print("CUDA =", torch.version.cuda)
print("NUMPY =", numpy.__version__)
print("LIGHTGBM =", lightgbm.__version__)
print("SAFETENSORS =", safetensors.__version__)

assert lightgbm.__version__ == "4.6.0"

print("ENV_GATE=PASS")
PY


echo
echo "========================================"
echo "STAGE 1 / 3"
echo "AUTHOR-SPECIFIC ADAPTER VAL TOP10"
echo "========================================"

"$PY" \
experiments/model_level/run_final_fiveauthor_adapter_val_candidates_v1.py \
--checkpoint "$CHECKPOINT" \
--val "$VAL" \
--adapter-root "$ADAPTER_ROOT" \
--output-root "$ADAPTER_OUT" \
--device cuda \
--log-every 250

echo
echo "STAGE1_ADAPTER_VAL=PASS"


echo
echo "========================================"
echo "STAGE 2 / 3"
echo "H5000 COMPATIBLE EXACT + COMPOSITION"
echo "========================================"

# Run H5000 only for Val rows.
VAL_IDS=$OUT/val_row_ids_v1.json

"$PY" - <<PY
import json

ids = []

with open(
    "$VAL",
    encoding="utf-8",
) as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        assert str(r["split"]).lower() == "val"
        ids.append(str(r["row_id"]))

assert len(ids) == 20000
assert len(set(ids)) == 20000

with open(
    "$VAL_IDS",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        ids,
        f,
        ensure_ascii=False,
        indent=2,
    )
    f.write("\n")

print("VAL_SELECTED_ROW_IDS=20000")
print("TEST_SELECTED_ROW_IDS=0")
PY

"$PY" \
experiments/model_level/build_final_fiveauthor_h5000_raw_recovery_v1.py \
--fit "$FIT" \
--val "$VAL" \
--boundaries "$BOUNDARIES" \
--raw-history "$RAW" \
--work-split "$WORK_SPLIT" \
--full-manifest "$FULL_MANIFEST" \
--selected-row-ids "$VAL_IDS" \
--artifact-mode compact \
--output "$H5000_OUT" \
--summary "$H5000_SUMMARY"

echo
echo "STAGE2_H5000_VAL=PASS"


echo
echo "========================================"
echo "STAGE 3 / 3"
echo "PERSONAL5 + RICH30 + LAMBDAMART"
echo "========================================"

"$PY" \
experiments/model_level/run_final_fiveauthor_val_rich30_train_v1.py \
--adapter-root "$ADAPTER_OUT" \
--h5000 "$H5000_OUT" \
--full-manifest "$FULL_MANIFEST" \
--raw-history "$RAW" \
--work-split "$WORK_SPLIT" \
--output-root "$RICH_OUT"

echo
echo "STAGE3_RICH30_TRAIN=PASS"


echo
echo "========================================"
echo "FINAL AUDIT"
echo "========================================"

"$PY" - <<PY
import hashlib
import json
from pathlib import Path

root = Path("$OUT")
rich = root / "rich30_training_v1"

model = (
    rich
    / "final_fiveauthor_rich30_lambdamart_v1.txt"
)

summary = (
    rich
    / "final_fiveauthor_val_rich30_training_summary_v1.json"
)

surface = (
    rich
    / "val_rich30_training_surface_v1.jsonl"
)

h5000 = Path("$H5000_OUT")

for p in (
    model,
    summary,
    surface,
    h5000,
):
    assert p.is_file(), p
    assert p.stat().st_size > 0, p

def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(b)
    return h.hexdigest()

with summary.open(
    encoding="utf-8"
) as f:
    s = json.load(f)

assert s["training_population"] == "Val"
assert s["train_queries"] == 20000
assert s["feature_count"] == 30
assert s["used_test"] is False
assert s["test_rows_used_as_queries"] == 0
assert s["test_model_evaluation"] is False

print("MODEL_SHA256 =", sha(model))
print("SURFACE_SHA256 =", sha(surface))
print("H5000_SHA256 =", sha(h5000))

print("VAL_TRAIN_QUERIES = 20000")
print("TEST_ROWS_USED_AS_QUERIES = 0")
print("TEST_MODEL_EVALUATION = FALSE")
print("FINAL_VAL_END_TO_END_GATE=PASS")
PY

echo
echo "========================================"
echo "FINAL VAL MODEL TRAINING COMPLETE"
echo "TEST HAS NOT BEEN USED BY THIS PIPELINE"
echo "========================================"
