#!/bin/bash
#SBATCH --job-name=stats-recovery
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/home/3160454/work/thesis-finalmodel-sixauthor-dataset-v1/results/finalmodel_fiveauthor_v1/statistical_robustness_and_recovery_v1/slurm-%j.out
#SBATCH --error=/home/3160454/work/thesis-finalmodel-sixauthor-dataset-v1/results/finalmodel_fiveauthor_v1/statistical_robustness_and_recovery_v1/slurm-%j.err

set -euo pipefail

ROOT=/home/3160454/work/thesis-finalmodel-sixauthor-dataset-v1
PY=/home/3160454/work/envs/model-level-py312/bin/python

cd "$ROOT"

echo "=============================================="
echo "FINAL STATISTICAL + RECOVERY ANALYSIS"
echo "START=$(date --iso-8601=seconds)"
echo "=============================================="

"$PY" \
experiments/model_level/analyse_final_fiveauthor_statistical_robustness_and_recovery_v1.py

echo
echo "=============================================="
echo "FINISHED"
echo "END=$(date --iso-8601=seconds)"
echo "=============================================="
