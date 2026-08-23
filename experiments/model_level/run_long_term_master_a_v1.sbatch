#!/usr/bin/env bash

#SBATCH --job-name=ml-longterm-a-v1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/3160454/work/thesis-model-level/logs/model_level/ml_longterm_a_v1_%j.out
#SBATCH --error=/home/3160454/work/thesis-model-level/logs/model_level/ml_longterm_a_v1_%j.err

set -euo pipefail

ROOT=/home/3160454/work/thesis-model-level
PY=/home/3160454/work/envs/model-level-py312/bin/python

CHECKPOINT=/home/3160454/work/model-level-assets-20260822/pinyingpt2-concat
FIT=/home/3160454/work/model-level-assets-20260822/clean3_train_fit_v1.jsonl
VAL=/home/3160454/work/model-level-assets-20260822/clean3_train_val_v1.jsonl

OV=$ROOT/results/model_level/overnight_ablation_v1

AUDIT=$ROOT/results/model_level/long_term_qualification_v1/audit_pre_freeze_v1
OUT=$ROOT/results/model_level/long_term_qualification_v1/master_a_v1
LOG=$ROOT/logs/model_level/long_term_qualification_v1/master_a_v1

AG_DEV=$ROOT/results/model_level/fixed_dev1000_v1/agent_phage_dev1000.jsonl

OLDEST_IDS=$AUDIT/oldest5000_row_ids.json
RANDOM_IDS=$AUDIT/random5000_row_ids.json
RECENT_IDS=$AUDIT/recent5000_row_ids.json

OLD_NAME=agent_oldest5000_full_only
RAND_NAME=agent_random5000_full_only
LONG_NAME=agent_recent5000_full_only_steps6991

OLD_OUT=$OUT/train/$OLD_NAME
RAND_OUT=$OUT/train/$RAND_NAME
LONG_OUT=$OUT/train/$LONG_NAME

OLD_ADAPTER=$OLD_OUT/agent_phage_adapter_v1.safetensors
RAND_ADAPTER=$RAND_OUT/agent_phage_adapter_v1.safetensors
LONG_ADAPTER=$LONG_OUT/agent_phage_adapter_v1.safetensors

RECENT_NAME=agent_recent5000_full_only
FULL_NAME=agent_full55926_full_only

RECENT_OUT=$OV/train/$RECENT_NAME
FULL_OUT=$OV/train/$FULL_NAME

RECENT_ADAPTER=$RECENT_OUT/agent_phage_adapter_v1.safetensors
FULL_ADAPTER=$FULL_OUT/agent_phage_adapter_v1.safetensors

ET_NAME=etinjat_full32906_full_only
ET_ADAPTER=$OV/train/$ET_NAME/etinjat_adapter_v1.safetensors
ET_FULL_OUT=$OV/eval/train_val/$ET_NAME/full

mkdir -p "$OUT/train" "$OUT/eval/dev1000" "$LOG"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "$ROOT"


run_gpu() {
    local logfile="$1"
    shift

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
        shift
        cd "$ROOT"
        exec "$@"
      ' _ "$ROOT" "$@" \
      > "$logfile" 2>&1
}


eval_complete() {
    local out="$1"
    local expected_n="$2"

    [[ -s "$out/evaluation_result.json" ]] || return 1

    "$PY" - "$out/evaluation_result.json" "$expected_n" <<'PY'
import json
import sys

path = sys.argv[1]
expected = int(sys.argv[2])

x = json.load(open(path, encoding="utf-8"))

ok = (
    x.get("status") == "complete"
    and int(x.get("selected_rows", -1)) == expected
)

raise SystemExit(0 if ok else 1)
PY
}


training_complete_steps() {
    local out="$1"
    local expected_steps="$2"

    [[ -s "$out/training_result.json" ]] || return 1
    [[ -s "$out/training_history.jsonl" ]] || return 1
    [[ -s "$out/agent_phage_adapter_v1.safetensors" ]] || return 1

    "$PY" - \
      "$out/training_result.json" \
      "$out/training_history.jsonl" \
      "$expected_steps" <<'PY'
import json
import sys

result_path, history_path, expected = sys.argv[1:]
expected = int(expected)

result = json.load(open(result_path, encoding="utf-8"))

last = None
with open(history_path, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            last = json.loads(line)

ok = (
    result.get("status") == "complete"
    and last is not None
    and int(last["global_step"]) == expected
)

raise SystemExit(0 if ok else 1)
PY
}


archive_incomplete_training() {
    local out="$1"

    if [[ -d "$out" ]]; then
        mv "$out" "${out}.incomplete.$(date +%s)"
    fi
}


train_manifest_model() {
    local name="$1"
    local manifest="$2"
    local epochs="$3"
    local max_steps="$4"
    local expected_steps="$5"

    local out="$OUT/train/$name"
    local logfile="$LOG/train_${name}.log"

    if training_complete_steps "$out" "$expected_steps"; then
        echo "SKIP COMPLETE TRAINING: $name"
        return 0
    fi

    archive_incomplete_training "$out"
    mkdir -p "$(dirname "$out")"

    local cmd=(
      "$PY"
      -m experiments.model_level.run_adapter_training_manifest_v1
      --selected-row-ids "$manifest"
      --training-pinyin-policy full_only
      --checkpoint "$CHECKPOINT"
      --fit "$FIT"
      --output-root "$out"
      --author "Agent Phage"
      --device cuda
      --seed 20260822
      --epochs "$epochs"
      --batch-size 8
      --learning-rate 5e-4
      --max-rows 5000
      --log-every 250
    )

    if [[ "$max_steps" != "0" ]]; then
        cmd+=(--max-steps "$max_steps")
    fi

    echo "START TRAIN: $name"

    run_gpu "$logfile" "${cmd[@]}"

    training_complete_steps "$out" "$expected_steps"

    echo "DONE TRAIN: $name"
}


eval_agent() {
    local name="$1"
    local adapter="$2"
    local mode="$3"

    local out="$OUT/eval/dev1000/$name/$mode"
    local logfile="$LOG/eval_${name}_${mode}.log"
    local label="${name}_${mode}_dev1000"

    if eval_complete "$out" 1000; then
        echo "SKIP COMPLETE EVAL: $name $mode"
        return 0
    fi

    mkdir -p "$out"

    echo "START EVAL: $name $mode"

    run_gpu "$logfile" \
      "$PY" \
      -m experiments.model_level.run_adapter_condition_evaluation_v1 \
      --checkpoint "$CHECKPOINT" \
      --train-val "$VAL" \
      --adapter "$adapter" \
      --output-root "$out" \
      --author "Agent Phage" \
      --label "$label" \
      --pinyin-mode "$mode" \
      --device cuda \
      --seed 20260822 \
      --top-k 10 \
      --beam-size 16 \
      --log-every 250 \
      --selected-rows "$AG_DEV"

    eval_complete "$out" 1000

    echo "DONE EVAL: $name $mode"
}


###############################################################################
# STAGE 0: PREFLIGHT
###############################################################################

echo
echo "============================================================"
echo " STAGE 0: PREFLIGHT"
echo "============================================================"

for path in \
    "$CHECKPOINT" \
    "$FIT" \
    "$VAL" \
    "$AG_DEV" \
    "$OLDEST_IDS" \
    "$RANDOM_IDS" \
    "$RECENT_IDS" \
    "$RECENT_ADAPTER" \
    "$FULL_ADAPTER" \
    "$ET_ADAPTER"
do
    [[ -e "$path" ]] || {
        echo "MISSING REQUIRED PATH: $path"
        exit 2
    }
done

"$PY" - "$OLDEST_IDS" "$RANDOM_IDS" "$RECENT_IDS" <<'PY'
import json
import sys
from pathlib import Path


def ids(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(payload, list):
        values = payload
    else:
        values = None

        for key in ("row_ids", "selected_row_ids", "ids"):
            if isinstance(payload.get(key), list):
                values = payload[key]
                break

        if values is None:
            lists = [
                value
                for value in payload.values()
                if isinstance(value, list)
                and all(isinstance(x, str) for x in value)
            ]
            if len(lists) == 1:
                values = lists[0]

    if values is None:
        raise RuntimeError(f"Cannot read row IDs from {path}")

    values = list(map(str, values))

    if len(values) != 5000:
        raise RuntimeError(
            f"{path}: expected 5000 IDs, got {len(values)}"
        )

    if len(set(values)) != 5000:
        raise RuntimeError(
            f"{path}: row IDs are not unique"
        )

    return values


oldest = ids(sys.argv[1])
random = ids(sys.argv[2])
recent = ids(sys.argv[3])

print("oldest5000 =", len(oldest))
print("random5000 =", len(random))
print("recent5000 =", len(recent))

print(
    "oldest<->random overlap =",
    len(set(oldest) & set(random)),
)
print(
    "oldest<->recent overlap =",
    len(set(oldest) & set(recent)),
)
print(
    "random<->recent overlap =",
    len(set(random) & set(recent)),
)

if set(oldest) & set(recent):
    raise RuntimeError(
        "Oldest5K and Recent5K must have zero overlap"
    )
PY

echo
echo "===== VALIDATE REUSED TRAINING STEP COUNTS ====="

training_complete_steps "$RECENT_OUT" 625 || {
    echo "Existing Recent5K is not the expected 625-step run"
    exit 3
}

training_complete_steps "$FULL_OUT" 6991 || {
    echo "Existing Full-B is not the expected 6991-step run"
    exit 3
}

echo "Existing Recent5K = 625 steps: OK"
echo "Existing Full-B    = 6991 steps: OK"


###############################################################################
# STAGE 1: ETINJAT FULL TRAIN-VAL FINALIZATION
###############################################################################

echo
echo "============================================================"
echo " STAGE 1: ETINJAT FULL FINALIZATION"
echo "============================================================"

if eval_complete "$ET_FULL_OUT" 8030; then
    echo "SKIP: Etinjat Full Train-Val already complete"
else
    if ! grep -q \
      'no tokenizer candidates for Pinyin ' \
      experiments/model_level/run_adapter_dev_evaluation_v1.py
    then
        echo "STOP: unsupported-Pinyin evaluation patch is not present"
        exit 4
    fi

    mkdir -p "$ET_FULL_OUT"

    run_gpu "$LOG/eval_etinjat_full_train_val.log" \
      "$PY" \
      -m experiments.model_level.run_adapter_condition_evaluation_v1 \
      --checkpoint "$CHECKPOINT" \
      --train-val "$VAL" \
      --adapter "$ET_ADAPTER" \
      --output-root "$ET_FULL_OUT" \
      --author "Etinjat" \
      --label "etinjat_full32906_full_only_full_train_val" \
      --pinyin-mode full \
      --device cuda \
      --seed 20260822 \
      --top-k 10 \
      --beam-size 16 \
      --log-every 250

    eval_complete "$ET_FULL_OUT" 8030
fi


###############################################################################
# STAGE 2: TRAIN FIXED-BUDGET TEMPORAL CONTROLS
###############################################################################

echo
echo "============================================================"
echo " STAGE 2: TRAIN TEMPORAL CONTROLS"
echo "============================================================"

# Exactly 5000 unique compatible rows, one epoch.
# batch=8 => ceil(5000/8) = 625 optimizer updates.
train_manifest_model \
    "$OLD_NAME" \
    "$OLDEST_IDS" \
    1 \
    0 \
    625

train_manifest_model \
    "$RAND_NAME" \
    "$RANDOM_IDS" \
    1 \
    0 \
    625

# Optimization confound:
# same exact Recent5K population, repeated until EXACTLY 6991 optimizer steps.
#
# 11 complete epochs = 6875 steps.
# Epoch 12 supplies the remaining 116 updates.
train_manifest_model \
    "$LONG_NAME" \
    "$RECENT_IDS" \
    12 \
    6991 \
    6991


###############################################################################
# STAGE 3: DEV1000 FULL + INITIAL
###############################################################################

echo
echo "============================================================"
echo " STAGE 3: AGENT DEV1000 EVALUATION"
echo "============================================================"

for mode in full initial
do
    eval_agent "$OLD_NAME" "$OLD_ADAPTER" "$mode"
    eval_agent "$RAND_NAME" "$RAND_ADAPTER" "$mode"
    eval_agent "$LONG_NAME" "$LONG_ADAPTER" "$mode"
done


###############################################################################
# STAGE 4: VALIDATE REUSED DEV1000 RESULTS
###############################################################################

echo
echo "============================================================"
echo " STAGE 4: VALIDATE REUSED RESULTS"
echo "============================================================"

for mode in full initial
do
    eval_complete \
      "$OV/eval/dev1000/$RECENT_NAME/$mode" \
      1000 || {
        echo "Missing/incomplete reused Recent5K $mode"
        exit 5
      }

    eval_complete \
      "$OV/eval/dev1000/$FULL_NAME/$mode" \
      1000 || {
        echo "Missing/incomplete reused Full-B $mode"
        exit 5
      }
done


###############################################################################
# STAGE 5: SUMMARY
###############################################################################

echo
echo "============================================================"
echo " STAGE 5: MASTER A SUMMARY"
echo "============================================================"

"$PY" - <<'PY'
from __future__ import annotations

import csv
import json
from pathlib import Path


root = Path(
    "results/model_level/long_term_qualification_v1/master_a_v1"
)

overnight = Path(
    "results/model_level/overnight_ablation_v1"
)

models = {
    "oldest5k_625": {
        "name": "agent_oldest5000_full_only",
        "unique_rows": 5000,
        "optimizer_steps": 625,
        "eval_root": root / "eval" / "dev1000",
        "source": "new",
    },
    "random5k_625": {
        "name": "agent_random5000_full_only",
        "unique_rows": 5000,
        "optimizer_steps": 625,
        "eval_root": root / "eval" / "dev1000",
        "source": "new",
    },
    "recent5k_625": {
        "name": "agent_recent5000_full_only",
        "unique_rows": 5000,
        "optimizer_steps": 625,
        "eval_root": overnight / "eval" / "dev1000",
        "source": "reused",
    },
    "recent5k_6991": {
        "name": "agent_recent5000_full_only_steps6991",
        "unique_rows": 5000,
        "optimizer_steps": 6991,
        "eval_root": root / "eval" / "dev1000",
        "source": "new",
    },
    "full55925_6991": {
        "name": "agent_full55926_full_only",
        "unique_rows": 55925,
        "optimizer_steps": 6991,
        "eval_root": overnight / "eval" / "dev1000",
        "source": "reused",
    },
}


rows = []

for key, spec in models.items():
    for mode in ("full", "initial"):
        path = (
            spec["eval_root"]
            / spec["name"]
            / mode
            / "evaluation_result.json"
        )

        if not path.is_file():
            raise RuntimeError(
                f"Missing evaluation result: {path}"
            )

        x = json.loads(
            path.read_text(encoding="utf-8")
        )

        if x.get("status") != "complete":
            raise RuntimeError(
                f"Incomplete evaluation: {path}"
            )

        if int(x["selected_rows"]) != 1000:
            raise RuntimeError(
                f"Expected 1000 rows: {path}"
            )

        m = x["metrics"]["micro"]

        rows.append(
            {
                "condition": key,
                "model_directory": spec["name"],
                "source": spec["source"],
                "mode": mode,
                "n": x["selected_rows"],
                "unique_training_rows": spec["unique_rows"],
                "optimizer_steps": spec["optimizer_steps"],
                "top1": m["top1"],
                "top3": m["top3"],
                "top5": m["top5"],
                "top10": m["top10"],
                "mrr_at_10": m["mrr_at_10"],
                "missing_at_10_rate": m[
                    "missing_at_10_rate"
                ],
            }
        )


by = {
    (row["condition"], row["mode"]): row
    for row in rows
}


def comparison(
    name: str,
    left: str,
    right: str,
) -> dict:
    result = {
        "name": name,
        "left": left,
        "right": right,
        "definition": "left_minus_right",
        "modes": {},
    }

    for mode in ("full", "initial"):
        a = by[(left, mode)]
        b = by[(right, mode)]

        result["modes"][mode] = {
            "top1_delta": a["top1"] - b["top1"],
            "top3_delta": a["top3"] - b["top3"],
            "mrr_at_10_delta": (
                a["mrr_at_10"] - b["mrr_at_10"]
            ),
            "missing_at_10_rate_delta": (
                a["missing_at_10_rate"]
                - b["missing_at_10_rate"]
            ),
        }

    return result


comparisons = [
    comparison(
        "temporal_recent_minus_oldest_matched_625_steps",
        "recent5k_625",
        "oldest5k_625",
    ),
    comparison(
        "temporal_recent_minus_random_matched_625_steps",
        "recent5k_625",
        "random5k_625",
    ),
    comparison(
        "optimization_recent6991_minus_recent625_same_rows",
        "recent5k_6991",
        "recent5k_625",
    ),
    comparison(
        "history_full55925_minus_recent5k_matched_6991_steps",
        "full55925_6991",
        "recent5k_6991",
    ),
]


summary = {
    "schema_version": 1,
    "status": "complete",
    "experiment": "model_level_long_term_master_a_v1",
    "author": "Agent Phage",
    "development_evaluation": "fixed Agent Dev1000 Train-Val subset",
    "test_used": False,
    "interpretation": {
        "temporal_position_control": (
            "Oldest5K, Random5K and Recent5K all use "
            "5000 unique compatible Train-Fit rows and "
            "625 optimizer steps."
        ),
        "optimization_control": (
            "Recent5K-625 versus the identical Recent5K "
            "population repeated to 6991 optimizer steps."
        ),
        "matched_step_history_control": (
            "Recent5K repeated to 6991 steps versus "
            "Full55925 trained for one epoch / 6991 steps."
        ),
        "scope_note": (
            "This static Train-Fit history ablation is "
            "separate from the later Train-Fit+Train-Val "
            "prequential continual-learning benchmark."
        ),
    },
    "results": rows,
    "comparisons": comparisons,
}


json_path = root / "master_a_summary.json"
csv_path = root / "master_a_summary.csv"

json_path.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

with csv_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0]),
    )
    writer.writeheader()
    writer.writerows(rows)


print()
print(
    f"{'CONDITION':22s} "
    f"{'MODE':8s} "
    f"{'STEPS':>6s} "
    f"{'TOP1':>9s} "
    f"{'TOP3':>9s} "
    f"{'MRR':>9s} "
    f"{'MISS':>9s}"
)

for row in rows:
    print(
        f"{row['condition']:22s} "
        f"{row['mode']:8s} "
        f"{row['optimizer_steps']:6d} "
        f"{row['top1']:9.6f} "
        f"{row['top3']:9.6f} "
        f"{row['mrr_at_10']:9.6f} "
        f"{row['missing_at_10_rate']:9.6f}"
    )

print()
print("===== COMPARISONS =====")

for item in comparisons:
    print()
    print(item["name"])

    for mode, delta in item["modes"].items():
        print(
            f"  {mode:8s} "
            f"Top1={delta['top1_delta']:+.6f} "
            f"Top3={delta['top3_delta']:+.6f} "
            f"MRR={delta['mrr_at_10_delta']:+.6f} "
            f"Miss={delta['missing_at_10_rate_delta']:+.6f}"
        )

print()
print("SUMMARY JSON =", json_path)
print("SUMMARY CSV  =", csv_path)
PY


echo
echo "============================================================"
echo " MASTER A COMPLETE"
echo "============================================================"
echo "END=$(date -Is)"
