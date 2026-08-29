#!/bin/bash
#SBATCH --job-name=five-test-report
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/3160454/work/thesis-finalmodel-sixauthor-dataset-v1/results/finalmodel_fiveauthor_v1/five_system_test_comparison_v1/slurm-%j.out
#SBATCH --error=/home/3160454/work/thesis-finalmodel-sixauthor-dataset-v1/results/finalmodel_fiveauthor_v1/five_system_test_comparison_v1/slurm-%j.err

set -euo pipefail

ROOT=/home/3160454/work/thesis-finalmodel-sixauthor-dataset-v1
PY=/home/3160454/work/envs/model-level-py312/bin/python

cd "$ROOT"

"$PY" \
experiments/model_level/build_final_fiveauthor_five_system_test_report_v1.py
