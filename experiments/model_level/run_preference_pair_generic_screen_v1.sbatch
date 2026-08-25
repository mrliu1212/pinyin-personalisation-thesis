#!/bin/bash
#SBATCH --chdir=/home/3160454/work/thesis-model-level
#SBATCH --job-name=ml-pref-screen
#SBATCH --partition=gpuh200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/model_level/ml_pref_screen_%j.out
#SBATCH --error=logs/model_level/ml_pref_screen_%j.err

set -euo pipefail

cd /home/3160454/work/thesis-model-level

echo "============================================================"
echo " CONTROLLED PREFERENCE GENERIC MARGIN SCREEN"
echo "============================================================"
date
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

srun bash -c '
  set -euo pipefail

  /home/3160454/work/envs/model-level-py312/bin/python \
    -m experiments.model_level.screen_preference_pairs_generic_v1 \
    --device cuda
'
