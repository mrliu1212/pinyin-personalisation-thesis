#!/bin/bash
#SBATCH --job-name=final-test
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/3160454/work/thesis-finalmodel-sixauthor-dataset-v1/results/finalmodel_fiveauthor_v1/test_end_to_end_v1/slurm-%j.out
#SBATCH --error=/home/3160454/work/thesis-finalmodel-sixauthor-dataset-v1/results/finalmodel_fiveauthor_v1/test_end_to_end_v1/slurm-%j.err

set -euo pipefail

ROOT=/home/3160454/work/thesis-finalmodel-sixauthor-dataset-v1
PY=/home/3160454/work/envs/model-level-py312/bin/python

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

RESULTS="$ROOT/results/finalmodel_fiveauthor_v1"
OUT="$RESULTS/test_end_to_end_v1"

CHECKPOINT=/home/3160454/work/model-level-assets-20260822/pinyingpt2-concat

FULL_MANIFEST="$RESULTS/dataset_preparation_v1/final_fiveauthor_manifest_frozen_v1.jsonl"
WORK_SPLIT="$RESULTS/dataset_preparation_v1/work_split_manifest_frozen_v1.csv"

RAW_HISTORY=/home/3160454/work/final-sixauthor-source-v1/data/processed/deep_author/interactions_t1_ready.jsonl

ADAPTER_ROOT="$RESULTS/adapter_training_v1"
ADAPTER_TEST="$OUT/adapter_test_top10_v1"

H5000_OUT="$OUT/h5000_test_compact_v1.jsonl"
H5000_SUMMARY="$OUT/h5000_test_summary_v1.json"

RICH30_OUT="$OUT/rich30_test_v1"

VAL_MODULE="$ROOT/experiments/model_level/run_final_fiveauthor_val_rich30_train_v1.py"
FROZEN_H5000="$ROOT/experiments/model_level/build_final_fiveauthor_h5000_raw_recovery_v1.py"

mkdir -p "$OUT"
mkdir -p "$ADAPTER_TEST"
mkdir -p "$RICH30_OUT"

echo "=============================================="
echo "FINAL FIVE-AUTHOR PERSONALISED TEST V1"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "START=$(date --iso-8601=seconds)"
echo "=============================================="

echo
echo "===== ENVIRONMENT ====="

"$PY" - <<'PY'
import lightgbm
import numpy
import safetensors
import torch

print("PYTHON =", __import__("sys").version)
print("TORCH =", torch.__version__)
print("CUDA =", torch.version.cuda)
print("NUMPY =", numpy.__version__)
print("LIGHTGBM =", lightgbm.__version__)
print("SAFETENSORS =", safetensors.__version__)
print("ENV_GATE=PASS")
PY

echo
echo "===== LOCATE FROZEN VAL LAMBDAMART ====="

mapfile -t MODELS < <(
    find \
    "$RESULTS/val_end_to_end_v1" \
    -type f \
    -name 'final_fiveauthor_rich30_lambdamart_v1.txt' \
    -print
)

if [ "${#MODELS[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly one frozen Val LambdaMART."
    echo "FOUND=${#MODELS[@]}"
    printf '%s\n' "${MODELS[@]:-}"
    exit 1
fi

MODEL="${MODELS[0]}"

echo "MODEL=$MODEL"
sha256sum "$MODEL"

echo
echo "=============================================="
echo "STAGE 1 / 3"
echo "AUTHOR-SPECIFIC ADAPTER TEST TOP10"
echo "=============================================="

"$PY" \
"$ROOT/experiments/model_level/run_final_fiveauthor_adapter_test_candidates_v1.py" \
    --checkpoint "$CHECKPOINT" \
    --full-manifest "$FULL_MANIFEST" \
    --adapter-root "$ADAPTER_ROOT" \
    --output-root "$ADAPTER_TEST" \
    --device cuda \
    --log-every 250

echo
echo "=============================================="
echo "STAGE 2 / 3"
echo "H5000 TEST CAUSAL RECOVERY"
echo "=============================================="

"$PY" \
"$ROOT/experiments/model_level/build_final_fiveauthor_h5000_test_recovery_v1.py" \
    --frozen-builder "$FROZEN_H5000" \
    --full-manifest "$FULL_MANIFEST" \
    --raw-history "$RAW_HISTORY" \
    --work-split "$WORK_SPLIT" \
    --output "$H5000_OUT" \
    --summary "$H5000_SUMMARY" \
    --artifact-mode compact

echo
echo "=============================================="
echo "STAGE 3 / 3"
echo "FROZEN RICH30 LAMBDAMART TEST"
echo "=============================================="

"$PY" \
"$ROOT/experiments/model_level/run_final_fiveauthor_test_rich30_evaluation_v1.py" \
    --val-module "$VAL_MODULE" \
    --adapter-root "$ADAPTER_TEST" \
    --h5000 "$H5000_OUT" \
    --full-manifest "$FULL_MANIFEST" \
    --raw-history "$RAW_HISTORY" \
    --work-split "$WORK_SPLIT" \
    --model "$MODEL" \
    --output-root "$RICH30_OUT"

echo
echo "=============================================="
echo "FINAL AUDIT"
echo "=============================================="

"$PY" - "$RICH30_OUT/final_fiveauthor_test_rich30_summary_v1.json" <<'PY'
import json
import pathlib
import sys

p = pathlib.Path(sys.argv[1])

if not p.is_file():
    raise SystemExit(f"Missing summary: {p}")

s = json.loads(p.read_text(encoding="utf-8"))

assert s["test_queries"] == 40000
assert s["feature_count"] == 30
assert s["lambdaMART_training_on_test"] is False
assert s["neural_parameter_updates_on_test"] is False
assert s["current_test_visible"] == 0
assert s["future_test_visible"] == 0

print("TEST_QUERIES =", s["test_queries"])
print("FEATURE_COUNT =", s["feature_count"])
print("MODEL_SHA256 =", s["model_sha256"])
print("SURFACE_SHA256 =", s["surface_sha256"])
print()
print("OVERALL =", json.dumps(
    s["overall"],
    ensure_ascii=False,
    sort_keys=True,
))
print()
print("FINAL_TEST_GATE=PASS")
PY

echo
echo "=============================================="
echo "FINAL PERSONALISED TEST FINISHED"
echo "END=$(date --iso-8601=seconds)"
echo "=============================================="
