#!/bin/bash
#SBATCH --chdir=/home/3160454/work/thesis-model-level
#SBATCH --job-name=ml-generic-p1p6
#SBATCH --partition=gpuh200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/model_level/ml_generic_p1p6_%j.out
#SBATCH --error=logs/model_level/ml_generic_p1p6_%j.err

set -euo pipefail

cd /home/3160454/work/thesis-model-level

mkdir -p logs/model_level

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=20260822

PYTHON=/home/3160454/work/envs/model-level-py312/bin/python

echo "============================================================"
echo " LONGITUDINAL GENERIC MATCHED P1-P6"
echo "============================================================"
date
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

srun \
  --exclusive \
  --exact \
  -n1 \
  -c6 \
  --mem=36G \
  --gres=gpu:1 \
  bash -c "
    set -euo pipefail
    cd /home/3160454/work/thesis-model-level
    echo INNER_CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}

    $PYTHON -m experiments.model_level.run_longitudinal_generic_probes_v1 \
      --checkpoint /home/3160454/work/model-level-assets-20260822/pinyingpt2-concat \
      --train-fit /home/3160454/work/model-level-assets-20260822/clean3_train_fit_v1.jsonl \
      --train-val /home/3160454/work/model-level-assets-20260822/clean3_train_val_v1.jsonl \
      --output-root results/model_level/longitudinal_generic_probes_v1
  "

echo "============================================================"
echo " COMPLETE"
echo "============================================================"
date
