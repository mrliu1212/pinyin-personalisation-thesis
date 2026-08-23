#!/usr/bin/env bash
#SBATCH --job-name=ml-warm-smoke
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/3160454/work/thesis-model-level/logs/model_level/ml_warm_smoke_%j.out
#SBATCH --error=/home/3160454/work/thesis-model-level/logs/model_level/ml_warm_smoke_%j.err

set -euo pipefail

ROOT=/home/3160454/work/thesis-model-level
PY=/home/3160454/work/envs/model-level-py312/bin/python

mkdir -p "$ROOT/logs/model_level"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "$ROOT"

srun \
  --exclusive \
  --exact \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=6 \
  --mem=36G \
  --gres=gpu:1 \
  bash -c '
    set -euo pipefail
    ROOT="$1"
    PY="$2"
    cd "$ROOT"
    exec "$PY" -m experiments.model_level.run_longitudinal_warm_smoke_v1
  ' _ "$ROOT" "$PY"
