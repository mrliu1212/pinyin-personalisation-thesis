#!/bin/bash
#SBATCH --job-name=gen-freq-test
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/3160454/work/thesis-finalmodel-sixauthor-dataset-v1/results/finalmodel_fiveauthor_v1/generic_frequency_test_v1/slurm-%j.out
#SBATCH --error=/home/3160454/work/thesis-finalmodel-sixauthor-dataset-v1/results/finalmodel_fiveauthor_v1/generic_frequency_test_v1/slurm-%j.err

set -euo pipefail

ROOT=/home/3160454/work/thesis-finalmodel-sixauthor-dataset-v1
PY=/home/3160454/work/envs/model-level-py312/bin/python

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "=============================================="
echo "GENERIC + FREQUENCY FINAL TEST"
echo "START=$(date --iso-8601=seconds)"
echo "=============================================="

"$PY" \
experiments/model_level/run_final_fiveauthor_generic_frequency_test_v1.py \
  --generic-predictions \
results/finalmodel_fiveauthor_v1/generic_baseline_v1/run_v2/generic_test_predictions.jsonl \
  --h5000 \
results/finalmodel_fiveauthor_v1/test_end_to_end_v1/h5000_test_compact_v1.jsonl \
  --val-module \
experiments/model_level/run_final_fiveauthor_val_rich30_train_v1.py \
  --raw-history \
/home/3160454/work/final-sixauthor-source-v1/data/processed/deep_author/interactions_t1_ready.jsonl \
  --work-split \
results/finalmodel_fiveauthor_v1/dataset_preparation_v1/work_split_manifest_frozen_v1.csv \
  --output-root \
results/finalmodel_fiveauthor_v1/generic_frequency_test_v1

echo
echo "=============================================="
echo "GENERIC + FREQUENCY TEST FINISHED"
echo "END=$(date --iso-8601=seconds)"
echo "=============================================="
