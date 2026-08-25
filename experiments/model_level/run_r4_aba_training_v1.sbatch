#!/bin/bash
#SBATCH --chdir=/home/3160454/work/thesis-model-level
#SBATCH --job-name=r4-aba-train
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/model_level/r4_aba_train_%j.out
#SBATCH --error=logs/model_level/r4_aba_train_%j.err

set -euo pipefail

PY=/home/3160454/work/envs/model-level-py312/bin/python

echo "============================================================"
echo "R4 ABA TRAINING"
echo "============================================================"

date
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

"$PY" \
-m experiments.model_level.run_r4_aba_training_v1
