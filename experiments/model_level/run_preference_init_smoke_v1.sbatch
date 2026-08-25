#!/bin/bash
#SBATCH --chdir=/home/3160454/work/thesis-model-level
#SBATCH --job-name=r4-init-smoke
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/model_level/r4_init_smoke_%j.out
#SBATCH --error=logs/model_level/r4_init_smoke_%j.err

set -euo pipefail

PY=/home/3160454/work/envs/model-level-py312/bin/python

echo "============================================================"
echo " R4 PREFERENCE INIT SMOKE"
echo "============================================================"

date
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

"$PY" \
  -m experiments.model_level.run_preference_init_smoke_v1
