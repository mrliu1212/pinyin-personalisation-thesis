#!/usr/bin/env bash
#SBATCH --job-name=ml-warm-resume-v1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/3160454/work/thesis-model-level/logs/model_level/ml_warm_resume_v1_%j.out
#SBATCH --error=/home/3160454/work/thesis-model-level/logs/model_level/ml_warm_resume_v1_%j.err

set -euo pipefail

ROOT=/home/3160454/work/thesis-model-level
PY=/home/3160454/work/envs/model-level-py312/bin/python

CHECKPOINT=/home/3160454/work/model-level-assets-20260822/pinyingpt2-concat
FIT=/home/3160454/work/model-level-assets-20260822/clean3_train_fit_v1.jsonl
VAL=/home/3160454/work/model-level-assets-20260822/clean3_train_val_v1.jsonl

PROTOCOL=$ROOT/results/model_level/long_term_qualification_v1/alternating_six_train_six_probe_v1
OUT=$ROOT/results/model_level/longitudinal_warm_v1

mkdir -p "$ROOT/logs/model_level"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "$ROOT"

echo "============================================================"
echo " LONGITUDINAL WARM V1"
echo " SINGLE GPU + CYCLE RESUME"
echo "============================================================"
echo "START=$(date -Is)"
echo "HOST=$(hostname)"
echo "COMMIT=$(git rev-parse HEAD)"
echo

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

echo
echo "============================================================"
echo " PREFLIGHT"
echo "============================================================"

"$PY" \
  -m experiments.model_level.run_adapter_longitudinal_warm_resume_v1 \
  --checkpoint "$CHECKPOINT" \
  --train-fit "$FIT" \
  --train-val "$VAL" \
  --protocol-root "$PROTOCOL" \
  --output-root "$OUT" \
  --validate-only

echo
echo "============================================================"
echo " GPU RUN"
echo "============================================================"

# Exactly ONE GPU is requested here.
# Slurm controls CUDA visibility for the process.
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
    CHECKPOINT="$3"
    FIT="$4"
    VAL="$5"
    PROTOCOL="$6"
    OUT="$7"

    cd "$ROOT"

    echo "INNER CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

    exec "$PY" \
      -m experiments.model_level.run_adapter_longitudinal_warm_resume_v1 \
      --checkpoint "$CHECKPOINT" \
      --train-fit "$FIT" \
      --train-val "$VAL" \
      --protocol-root "$PROTOCOL" \
      --output-root "$OUT" \
      --device cuda
  ' _ \
  "$ROOT" \
  "$PY" \
  "$CHECKPOINT" \
  "$FIT" \
  "$VAL" \
  "$PROTOCOL" \
  "$OUT"

echo
echo "============================================================"
echo " FINAL RESULT GATE"
echo "============================================================"

"$PY" - "$OUT/longitudinal_result.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])

if not path.is_file():
    raise RuntimeError(
        f"Missing result: {path}"
    )

x = json.loads(
    path.read_text(encoding="utf-8")
)

if x.get("status") != "complete":
    raise RuntimeError(
        "Run is not complete"
    )

if x.get("used_test") is not False:
    raise RuntimeError(
        "Test-use gate failed"
    )

if x.get("resume_semantics") != "cycle_level":
    raise RuntimeError(
        "Resume semantics mismatch"
    )

cycles = x.get("cycles", [])

if len(cycles) != 6:
    raise RuntimeError(
        f"Expected 6 cycles, got {len(cycles)}"
    )

for expected_cycle, cycle in enumerate(
    cycles,
    start=1,
):
    if int(cycle["cycle"]) != expected_cycle:
        raise RuntimeError(
            "Cycle ordering mismatch"
        )

    training = cycle["training"]

    if int(training["optimizer_steps"]) != 726:
        raise RuntimeError(
            f"Cycle {expected_cycle}: step mismatch"
        )

    if training["adapter_state_persisted"] is not True:
        raise RuntimeError(
            "Adapter persistence mismatch"
        )

    if training["optimizer_state_persisted"] is not False:
        raise RuntimeError(
            "Optimizer-state policy mismatch"
        )

print("STATUS = COMPLETE")
print("CYCLES = 6")
print("STEPS_PER_CYCLE = 726")
print("TOTAL_STEPS = 4356")
print("GPU_REQUEST = 1")
print("RESUME = cycle_level")
print("TEST_USED = false")

print()
print("===== PRIMARY PROBE RESULTS =====")

for cycle in cycles:
    m = cycle["probe_metrics"]["micro"]

    print(
        f"P{cycle['cycle']} "
        f"N={m['interaction_count']} "
        f"Top1={m['top1']:.6f} "
        f"Top3={m['top3']:.6f} "
        f"MRR={m['mrr_at_10']:.6f} "
        f"Missing={m['missing_at_10_rate']:.6f}"
    )
PY

echo
echo "============================================================"
echo " COMPLETE"
echo "============================================================"
echo "END=$(date -Is)"
