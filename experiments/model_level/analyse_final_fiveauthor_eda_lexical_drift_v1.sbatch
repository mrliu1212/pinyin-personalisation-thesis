#!/bin/bash
#SBATCH --job-name=final5-eda1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/finalmodel_fiveauthor_v1/eda_v1/eda1_%j.out
#SBATCH --error=results/finalmodel_fiveauthor_v1/eda_v1/eda1_%j.err

set -euo pipefail

cd /home/3160454/work/thesis-finalmodel-sixauthor-dataset-v1

mkdir -p results/finalmodel_fiveauthor_v1/eda_v1

/home/3160454/work/envs/model-level-py312/bin/python \
  experiments/model_level/analyse_final_fiveauthor_eda_lexical_drift_v1.py
