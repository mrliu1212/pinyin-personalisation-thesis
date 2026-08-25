#!/bin/bash
#SBATCH --chdir=/home/3160454/work/thesis-model-level
#SBATCH --job-name=r4-aba-eval-v2
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/model_level/r4_aba_clean_eval_v2_%j.out
#SBATCH --error=logs/model_level/r4_aba_clean_eval_v2_%j.err

set -euo pipefail

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export TOKENIZERS_PARALLELISM=false

PY=/home/3160454/work/envs/model-level-py312/bin/python

echo "===== R4 CLEAN ABA V2 EVALUATION ====="
date
hostname

"$PY" \
-m experiments.model_level.run_r4_aba_clean_evaluation_v2

echo "===== COMPLETE ====="
date
