#!/bin/bash
#SBATCH --job-name=final5_adapter_v1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=results/finalmodel_fiveauthor_v1/adapter_training_v1/logs/slurm-%j.out
#SBATCH --error=results/finalmodel_fiveauthor_v1/adapter_training_v1/logs/slurm-%j.err

set -euo pipefail

ROOT=/home/3160454/work/thesis-finalmodel-sixauthor-dataset-v1
PY=/home/3160454/work/envs/model-level-py312/bin/python

CHECKPOINT=/home/3160454/work/model-level-assets-20260822/pinyingpt2-concat

FIT=$ROOT/results/finalmodel_fiveauthor_v1/adapter_input_preparation_v1/adapter_fit_manifest_frozen_v1.jsonl

OUT=$ROOT/results/finalmodel_fiveauthor_v1/adapter_training_v1

TIMING=$OUT/training_runtime_v1.tsv

cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUT"

printf \
"author\tstart_time\tend_time\truntime_seconds\truntime_minutes\texit_status\n" \
> "$TIMING"


# ============================================================
# Global timing
# ============================================================

GLOBAL_START=$(date +%s)
GLOBAL_START_ISO=$(date --iso-8601=seconds)

echo "============================================================"
echo "FINAL FIVE-AUTHOR ADAPTER TRAINING"
echo "JOB_ID=${SLURM_JOB_ID:-unknown}"
echo "HOST=$(hostname)"
echo "START=$GLOBAL_START_ISO"
echo "============================================================"

echo
echo "===== GPU ====="
nvidia-smi

echo
echo "===== PROVENANCE ====="
echo "git_commit=$(git rev-parse HEAD)"

sha256sum "$FIT"

sha256sum \
results/finalmodel_fiveauthor_v1/adapter_input_preparation_v1/adapter_fit_compatible_row_ids_frozen_v1.json


run_author () {

    AUTHOR="$1"
    SLUG="$2"
    EXPECTED="$3"

    AUTHOR_OUT="$OUT/$SLUG"

    if [ -e "$AUTHOR_OUT" ]; then
        echo "STOP: output already exists:"
        echo "$AUTHOR_OUT"
        exit 1
    fi

    mkdir -p "$AUTHOR_OUT"

    START=$(date +%s)
    START_ISO=$(date --iso-8601=seconds)

    echo
    echo "============================================================"
    echo "AUTHOR=$AUTHOR"
    echo "COMPATIBLE_FIT_ROWS=$EXPECTED"
    echo "START=$START_ISO"
    echo "============================================================"

    set +e

    "$PY" \
      -m experiments.model_level.run_final_fiveauthor_adapter_compatible_training_v1 \
      --checkpoint "$CHECKPOINT" \
      --fit "$FIT" \
      --output-root "$AUTHOR_OUT" \
      --author "$AUTHOR" \
      --device cuda \
      --seed 20260822 \
      --epochs 1 \
      --batch-size 8 \
      --learning-rate 0.0005 \
      --weight-decay 0 \
      --max-grad-norm 1 \
      --log-every 100

    STATUS=$?

    set -e

    END=$(date +%s)
    END_ISO=$(date --iso-8601=seconds)

    RUNTIME=$((END - START))

    RUNTIME_MIN=$(
        awk -v s="$RUNTIME" \
        'BEGIN {printf "%.2f", s / 60}'
    )

    printf \
    "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$AUTHOR" \
    "$START_ISO" \
    "$END_ISO" \
    "$RUNTIME" \
    "$RUNTIME_MIN" \
    "$STATUS" \
    >> "$TIMING"

    echo
    echo "AUTHOR=$AUTHOR"
    echo "END=$END_ISO"
    echo "RUNTIME_SECONDS=$RUNTIME"
    echo "RUNTIME_MINUTES=$RUNTIME_MIN"
    echo "EXIT_STATUS=$STATUS"

    if [ "$STATUS" -ne 0 ]; then
        echo "TRAINING_FAILED=$AUTHOR"
        exit "$STATUS"
    fi
}


# ============================================================
# Five author-specific Adapters
# ============================================================

run_author \
"Re_spectators" \
"re_spectators" \
"29951"

run_author \
"Etinjat" \
"etinjat" \
"29131"

run_author \
"Agent Phage" \
"agent_phage" \
"29991"

run_author \
"QBLevi" \
"qblevi" \
"29967"

run_author \
"breaddddd" \
"breaddddd" \
"29981"


# ============================================================
# Global timing
# ============================================================

GLOBAL_END=$(date +%s)
GLOBAL_END_ISO=$(date --iso-8601=seconds)

TOTAL_RUNTIME=$((GLOBAL_END - GLOBAL_START))

TOTAL_MIN=$(
    awk -v s="$TOTAL_RUNTIME" \
    'BEGIN {printf "%.2f", s / 60}'
)

TOTAL_HOURS=$(
    awk -v s="$TOTAL_RUNTIME" \
    'BEGIN {printf "%.3f", s / 3600}'
)

echo
echo "============================================================"
echo "ALL_FIVE_ADAPTERS_COMPLETE"
echo "START=$GLOBAL_START_ISO"
echo "END=$GLOBAL_END_ISO"
echo "TOTAL_RUNTIME_SECONDS=$TOTAL_RUNTIME"
echo "TOTAL_RUNTIME_MINUTES=$TOTAL_MIN"
echo "TOTAL_RUNTIME_HOURS=$TOTAL_HOURS"
echo "TIMING_FILE=$TIMING"
echo "TEST_USED=FALSE"
echo "============================================================"
