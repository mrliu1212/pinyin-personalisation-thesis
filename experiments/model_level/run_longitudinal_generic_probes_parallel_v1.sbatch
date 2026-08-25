#!/bin/bash
#SBATCH --chdir=/home/3160454/work/thesis-model-level
#SBATCH --job-name=ml-gen-p
#SBATCH --partition=short_gpunew
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=1-6
#SBATCH --output=logs/model_level/ml_gen_parallel_%A_%a.out
#SBATCH --error=logs/model_level/ml_gen_parallel_%A_%a.err

set -euo pipefail

PROBE="${SLURM_ARRAY_TASK_ID}"
ROOT="results/model_level/longitudinal_generic_probes_parallel_v1/p${PROBE}"

mkdir -p "$ROOT"

echo "============================================================"
echo " GENERIC PARALLEL P${PROBE}"
echo "============================================================"
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

exec /home/3160454/work/envs/model-level-py312/bin/python \
  -m experiments.model_level.run_longitudinal_generic_probes_v1 \
  --checkpoint /home/3160454/work/model-level-assets-20260822/pinyingpt2-concat \
  --train-fit /home/3160454/work/model-level-assets-20260822/clean3_train_fit_v1.jsonl \
  --train-val /home/3160454/work/model-level-assets-20260822/clean3_train_val_v1.jsonl \
  --output-root "$ROOT" \
  --probe "$PROBE" \
  --device cuda
