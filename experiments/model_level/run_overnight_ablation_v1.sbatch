#!/usr/bin/env bash

#SBATCH --job-name=ml-overnight-v1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/home/3160454/work/thesis-model-level/logs/model_level/ml_overnight_v1_%j.out
#SBATCH --error=/home/3160454/work/thesis-model-level/logs/model_level/ml_overnight_v1_%j.err

set -euo pipefail

ROOT=/home/3160454/work/thesis-model-level
PY=/home/3160454/work/envs/model-level-py312/bin/python

CHECKPOINT=/home/3160454/work/model-level-assets-20260822/pinyingpt2-concat
FIT=/home/3160454/work/model-level-assets-20260822/clean3_train_fit_v1.jsonl
VAL=/home/3160454/work/model-level-assets-20260822/clean3_train_val_v1.jsonl

OUT=$ROOT/results/model_level/overnight_ablation_v1
LOG=$ROOT/logs/model_level/overnight_ablation_v1

DEVROOT=$ROOT/results/model_level/fixed_dev1000_v1

mkdir -p "$OUT" "$LOG"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "$ROOT"

echo "===== JOB ====="
echo "JOB_ID=${SLURM_JOB_ID:-}"
echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "START=$(date -Is)"
echo


run_gpu() {
    local logfile="$1"
    shift

    srun \
      --exclusive \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task=4 \
      --gres=gpu:1 \
      bash -c '
        set -euo pipefail

        ROOT="$1"
        shift

        cd "$ROOT"

        echo "HOST=$(hostname)"
        echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

        exec "$@"
      ' _ "$ROOT" "$@" \
      > "$logfile" 2>&1
}


wait_pair() {
    local p1="$1"
    local p2="$2"
    local name="$3"

    set +e
    wait "$p1"
    local s1=$?
    wait "$p2"
    local s2=$?
    set -e

    if [[ "$s1" -ne 0 || "$s2" -ne 0 ]]; then
        echo "PAIR FAILED: $name status1=$s1 status2=$s2"
        exit 1
    fi

    echo "PAIR COMPLETE: $name"
}


training_complete() {
    local out="$1"

    [[ -s "$out/training_result.json" ]] || return 1
    compgen -G "$out/*_adapter_v1.safetensors" >/dev/null || return 1

    "$PY" - "$out/training_result.json" <<'PY'
import json
import sys

x = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if x.get("status") == "complete" else 1)
PY
}


train_model() {
    local name="$1"
    local author="$2"
    local max_rows="$3"
    local pinyin_policy="$4"
    local history_selection="$5"

    local out="$OUT/train/$name"
    local logfile="$LOG/train_${name}.log"

    if training_complete "$out"; then
        echo "SKIP COMPLETE TRAINING: $name"
        return 0
    fi

    if [[ -d "$out" ]]; then
        mv "$out" "${out}.incomplete.$(date +%s)"
    fi

    mkdir -p "$(dirname "$out")"

    local cmd=(
      "$PY"
      -m experiments.model_level.run_adapter_training_policy_v1
      --training-pinyin-policy "$pinyin_policy"
      --history-selection "$history_selection"
      --checkpoint "$CHECKPOINT"
      --fit "$FIT"
      --output-root "$out"
      --author "$author"
      --device cuda
      --seed 20260822
      --epochs 1
      --batch-size 8
      --learning-rate 5e-4
      --log-every 250
    )

    if [[ "$max_rows" != "0" ]]; then
        cmd+=(--max-rows "$max_rows")
    fi

    echo "START TRAIN: $name"

    run_gpu "$logfile" "${cmd[@]}"

    training_complete "$out"

    echo "DONE TRAIN: $name"
}


eval_complete() {
    local out="$1"

    [[ -s "$out/evaluation_result.json" ]] || return 1

    "$PY" - "$out/evaluation_result.json" <<'PY'
import json
import sys

x = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if x.get("status") == "complete" else 1)
PY
}


eval_model() {
    local name="$1"
    local author="$2"
    local adapter="$3"
    local mode="$4"
    local scope="$5"
    local devfile="$6"

    local out="$OUT/eval/$scope/$name/$mode"
    local logfile="$LOG/eval_${scope}_${name}_${mode}.log"
    local label="${name}_${mode}_${scope}"

    if eval_complete "$out"; then
        echo "SKIP COMPLETE EVAL: $scope $name $mode"
        return 0
    fi

    mkdir -p "$out"

    local cmd=(
      "$PY"
      -m experiments.model_level.run_adapter_condition_evaluation_v1
      --checkpoint "$CHECKPOINT"
      --train-val "$VAL"
      --adapter "$adapter"
      --output-root "$out"
      --author "$author"
      --label "$label"
      --pinyin-mode "$mode"
      --device cuda
      --seed 20260822
      --top-k 10
      --beam-size 16
      --log-every 250
    )

    if [[ "$scope" == "dev1000" ]]; then
        cmd+=(--selected-rows "$devfile")
    fi

    echo "START EVAL: $scope $name $mode"

    run_gpu "$logfile" "${cmd[@]}"

    eval_complete "$out"

    echo "DONE EVAL: $scope $name $mode"
}


###############################################################################
# MODEL DEFINITIONS
###############################################################################

AG500=agent_recent500_full_only
AG5K=agent_recent5000_full_only
AG25K=agent_recent25000_full_only
AGA=agent_recent27963_full_only
AGB=agent_full55926_full_only
AGC=agent_full55926_mixed312
ET=etinjat_full32906_full_only
BR=breaddddd_full55694_full_only

AG500_OUT=$OUT/train/$AG500
AG5K_OUT=$OUT/train/$AG5K
AG25K_OUT=$OUT/train/$AG25K
AGA_OUT=$OUT/train/$AGA
AGB_OUT=$OUT/train/$AGB
AGC_OUT=$OUT/train/$AGC
ET_OUT=$OUT/train/$ET
BR_OUT=$OUT/train/$BR

AG500_ADAPTER=$AG500_OUT/agent_phage_adapter_v1.safetensors
AG5K_ADAPTER=$AG5K_OUT/agent_phage_adapter_v1.safetensors
AG25K_ADAPTER=$AG25K_OUT/agent_phage_adapter_v1.safetensors
AGA_ADAPTER=$AGA_OUT/agent_phage_adapter_v1.safetensors
AGB_ADAPTER=$AGB_OUT/agent_phage_adapter_v1.safetensors
AGC_ADAPTER=$AGC_OUT/agent_phage_adapter_v1.safetensors
ET_ADAPTER=$ET_OUT/etinjat_adapter_v1.safetensors
BR_ADAPTER=$BR_OUT/breaddddd_adapter_v1.safetensors

AG_DEV=$DEVROOT/agent_phage_dev1000.jsonl
ET_DEV=$DEVROOT/etinjat_dev1000.jsonl
BR_DEV=$DEVROOT/breaddddd_dev1000.jsonl


###############################################################################
# STAGE 1: TRAIN ALL 8 DISTINCT ADAPTERS
###############################################################################

echo
echo "===== STAGE 1: TRAINING ====="

train_model "$AG500" "Agent Phage" 500 full_only recent &
P1=$!
train_model "$AG5K" "Agent Phage" 5000 full_only recent &
P2=$!
wait_pair "$P1" "$P2" "train round 1"

train_model "$AG25K" "Agent Phage" 25000 full_only recent &
P1=$!
train_model "$AGA" "Agent Phage" 27963 full_only recent &
P2=$!
wait_pair "$P1" "$P2" "train round 2"

train_model "$AGB" "Agent Phage" 0 full_only deterministic &
P1=$!
train_model "$AGC" "Agent Phage" 0 mixed_3_1_2 deterministic &
P2=$!
wait_pair "$P1" "$P2" "train round 3"

train_model "$ET" "Etinjat" 0 full_only deterministic &
P1=$!
train_model "$BR" "breaddddd" 0 full_only deterministic &
P2=$!
wait_pair "$P1" "$P2" "train round 4"


###############################################################################
# STAGE 2: ALL DEV1000 INITIAL FIRST
###############################################################################

echo
echo "===== STAGE 2: DEV1000 INITIAL ====="

eval_model "$AG500" "Agent Phage" "$AG500_ADAPTER" initial dev1000 "$AG_DEV" &
P1=$!
eval_model "$AG5K" "Agent Phage" "$AG5K_ADAPTER" initial dev1000 "$AG_DEV" &
P2=$!
wait_pair "$P1" "$P2" "initial dev round 1"

eval_model "$AG25K" "Agent Phage" "$AG25K_ADAPTER" initial dev1000 "$AG_DEV" &
P1=$!
eval_model "$AGA" "Agent Phage" "$AGA_ADAPTER" initial dev1000 "$AG_DEV" &
P2=$!
wait_pair "$P1" "$P2" "initial dev round 2"

eval_model "$AGB" "Agent Phage" "$AGB_ADAPTER" initial dev1000 "$AG_DEV" &
P1=$!
eval_model "$AGC" "Agent Phage" "$AGC_ADAPTER" initial dev1000 "$AG_DEV" &
P2=$!
wait_pair "$P1" "$P2" "initial dev round 3"

eval_model "$ET" "Etinjat" "$ET_ADAPTER" initial dev1000 "$ET_DEV" &
P1=$!
eval_model "$BR" "breaddddd" "$BR_ADAPTER" initial dev1000 "$BR_DEV" &
P2=$!
wait_pair "$P1" "$P2" "initial dev round 4"


###############################################################################
# STAGE 3: ALL DEV1000 FULL SECOND
###############################################################################

echo
echo "===== STAGE 3: DEV1000 FULL ====="

eval_model "$AG500" "Agent Phage" "$AG500_ADAPTER" full dev1000 "$AG_DEV" &
P1=$!
eval_model "$AG5K" "Agent Phage" "$AG5K_ADAPTER" full dev1000 "$AG_DEV" &
P2=$!
wait_pair "$P1" "$P2" "full dev round 1"

eval_model "$AG25K" "Agent Phage" "$AG25K_ADAPTER" full dev1000 "$AG_DEV" &
P1=$!
eval_model "$AGA" "Agent Phage" "$AGA_ADAPTER" full dev1000 "$AG_DEV" &
P2=$!
wait_pair "$P1" "$P2" "full dev round 2"

eval_model "$AGB" "Agent Phage" "$AGB_ADAPTER" full dev1000 "$AG_DEV" &
P1=$!
eval_model "$AGC" "Agent Phage" "$AGC_ADAPTER" full dev1000 "$AG_DEV" &
P2=$!
wait_pair "$P1" "$P2" "full dev round 3"

eval_model "$ET" "Etinjat" "$ET_ADAPTER" full dev1000 "$ET_DEV" &
P1=$!
eval_model "$BR" "breaddddd" "$BR_ADAPTER" full dev1000 "$BR_DEV" &
P2=$!
wait_pair "$P1" "$P2" "full dev round 4"


###############################################################################
# STAGE 4: FORMAL FULL-HISTORY TRAIN-VAL INITIAL
#
# Agent reuses B.
###############################################################################

echo
echo "===== STAGE 4: FULL TRAIN-VAL INITIAL ====="

eval_model "$AGB" "Agent Phage" "$AGB_ADAPTER" initial train_val "" &
P1=$!
eval_model "$BR" "breaddddd" "$BR_ADAPTER" initial train_val "" &
P2=$!
wait_pair "$P1" "$P2" "initial train-val round 1"

eval_model "$ET" "Etinjat" "$ET_ADAPTER" initial train_val ""


###############################################################################
# STAGE 5: FORMAL FULL-HISTORY TRAIN-VAL FULL
###############################################################################

echo
echo "===== STAGE 5: FULL TRAIN-VAL FULL ====="

eval_model "$AGB" "Agent Phage" "$AGB_ADAPTER" full train_val "" &
P1=$!
eval_model "$BR" "breaddddd" "$BR_ADAPTER" full train_val "" &
P2=$!
wait_pair "$P1" "$P2" "full train-val round 1"

eval_model "$ET" "Etinjat" "$ET_ADAPTER" full train_val ""


###############################################################################
# FINAL SUMMARY
###############################################################################

echo
echo "===== BUILD SUMMARY ====="

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path(
    "/home/3160454/work/thesis-model-level/"
    "results/model_level/overnight_ablation_v1/eval"
)

rows = []

for result_path in sorted(root.glob("*/*/*/evaluation_result.json")):
    x = json.loads(result_path.read_text(encoding="utf-8"))
    m = x["metrics"]["micro"]

    rows.append(
        {
            "scope": x["analysis_scope"],
            "author": x["author"],
            "model": result_path.parent.parent.name,
            "mode": x["pinyin_mode"],
            "n": x["selected_rows"],
            "top1": m["top1"],
            "top3": m["top3"],
            "top5": m["top5"],
            "top10": m["top10"],
            "mrr": m["mrr_at_10"],
            "missing": m["missing_at_10_rate"],
        }
    )

summary_path = root.parent / "overnight_summary.json"

summary_path.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "status": "complete",
            "results": rows,
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print()
print(
    f"{'SCOPE':14s} "
    f"{'AUTHOR':14s} "
    f"{'MODEL':34s} "
    f"{'MODE':8s} "
    f"{'N':>6s} "
    f"{'TOP1':>9s} "
    f"{'MRR':>9s} "
    f"{'MISS':>9s}"
)

for row in rows:
    print(
        f"{row['scope'][:14]:14s} "
        f"{row['author'][:14]:14s} "
        f"{row['model'][:34]:34s} "
        f"{row['mode']:8s} "
        f"{row['n']:6d} "
        f"{row['top1']:9.6f} "
        f"{row['mrr']:9.6f} "
        f"{row['missing']:9.6f}"
    )

print()
print("summary:", summary_path)
PY

echo
echo "===== OVERNIGHT JOB COMPLETE ====="
echo "END=$(date -Is)"
