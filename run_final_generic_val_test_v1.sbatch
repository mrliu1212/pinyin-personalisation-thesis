#!/bin/bash

#SBATCH --job-name=final5_generic
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=results/finalmodel_fiveauthor_v1/generic_baseline_v1/slurm-%j.out
#SBATCH --error=results/finalmodel_fiveauthor_v1/generic_baseline_v1/slurm-%j.err

set -euo pipefail

ROOT=/home/3160454/work/thesis-finalmodel-sixauthor-dataset-v1
PY=/home/3160454/work/envs/model-level-py312/bin/python

CHECKPOINT=/home/3160454/work/model-level-assets-20260822/pinyingpt2-concat

MANIFEST=$ROOT/results/finalmodel_fiveauthor_v1/dataset_preparation_v1/final_fiveauthor_manifest_frozen_v1.jsonl

OUT=$ROOT/results/finalmodel_fiveauthor_v1/generic_baseline_v1/run_v2

cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p \
results/finalmodel_fiveauthor_v1/generic_baseline_v1

echo "=============================================="
echo "FINAL GENERIC VAL + TEST"
echo "JOB_ID=${SLURM_JOB_ID:-unknown}"
echo "HOST=$(hostname)"
echo "START=$(date --iso-8601=seconds)"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
echo "=============================================="

echo
echo "===== GPU ====="
nvidia-smi

echo
echo "===== INPUT SHA ====="
sha256sum "$MANIFEST"

GLOBAL_START=$(date +%s)

"$PY" \
experiments/model_level/run_final_generic_val_test_v1.py \
  --checkpoint "$CHECKPOINT" \
  --manifest "$MANIFEST" \
  --output-root "$OUT" \
  --device cuda \
  --log-every 500 \
  --open-test

STATUS=$?

GLOBAL_END=$(date +%s)
SECONDS_TOTAL=$((GLOBAL_END-GLOBAL_START))

echo
echo "=============================================="
echo "GENERIC JOB FINISHED"
echo "END=$(date --iso-8601=seconds)"
echo "RUNTIME_SECONDS=$SECONDS_TOTAL"
echo "RUNTIME_MINUTES=$(awk "BEGIN {printf \"%.2f\", $SECONDS_TOTAL/60}")"
echo "EXIT_STATUS=$STATUS"
echo "=============================================="

exit "$STATUS"
