#!/usr/bin/env bash

#SBATCH --job-name=ml-et-full
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/3160454/work/thesis-model-level/logs/model_level/ml_etinjat_full_%j.out
#SBATCH --error=/home/3160454/work/thesis-model-level/logs/model_level/ml_etinjat_full_%j.err

set -euo pipefail

ROOT=/home/3160454/work/thesis-model-level
PY=/home/3160454/work/envs/model-level-py312/bin/python

CHECKPOINT=/home/3160454/work/model-level-assets-20260822/pinyingpt2-concat
VAL=/home/3160454/work/model-level-assets-20260822/clean3_train_val_v1.jsonl

OUT=$ROOT/results/model_level/overnight_ablation_v1
ADAPTER=$OUT/train/etinjat_full32906_full_only/etinjat_adapter_v1.safetensors
EVAL_OUT=$OUT/eval/train_val/etinjat_full32906_full_only/full

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "$ROOT"

echo "===== ETINJAT FULL TRAIN-VAL FINALIZATION ====="
echo "JOB_ID=${SLURM_JOB_ID:-}"
echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "START=$(date -Is)"
echo

test -s "$ADAPTER"

mkdir -p "$EVAL_OUT"

echo "===== RUN ETINJAT FULL TRAIN-VAL ====="

"$PY" \
  -m experiments.model_level.run_adapter_condition_evaluation_v1 \
  --checkpoint "$CHECKPOINT" \
  --train-val "$VAL" \
  --adapter "$ADAPTER" \
  --output-root "$EVAL_OUT" \
  --author "Etinjat" \
  --label "etinjat_full32906_full_only_full_train_val" \
  --pinyin-mode full \
  --device cuda \
  --seed 20260822 \
  --top-k 10 \
  --beam-size 16 \
  --log-every 250

echo
echo "===== VERIFY ALL 22 EVALUATIONS ====="

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path(
    "results/model_level/overnight_ablation_v1"
)

eval_root = root / "eval"

results = sorted(
    eval_root.glob("*/*/*/evaluation_result.json")
)

print("evaluation_result count =", len(results))

if len(results) != 22:
    raise RuntimeError(
        f"Expected exactly 22 completed evaluations, got {len(results)}"
    )

expected_train_val = {
    ("Agent Phage", "initial"): 13741,
    ("Agent Phage", "full"): 13741,
    ("Etinjat", "initial"): 8030,
    ("Etinjat", "full"): 8030,
    ("breaddddd", "initial"): 12645,
    ("breaddddd", "full"): 12645,
}

seen = {}

for path in results:
    x = json.loads(
        path.read_text(encoding="utf-8")
    )

    if x.get("status") != "complete":
        raise RuntimeError(
            f"Incomplete result: {path}"
        )

    if x["analysis_scope"] == "full_train_val":
        key = (
            x["author"],
            x["pinyin_mode"],
        )

        seen[key] = x["selected_rows"]

for key, n in expected_train_val.items():
    actual = seen.get(key)

    print(
        key,
        "rows=",
        actual,
        "expected=",
        n,
    )

    if actual != n:
        raise RuntimeError(
            f"Train-Val population mismatch for {key}: "
            f"{actual} != {n}"
        )

print()
print("ALL 22 EVALUATIONS: PASS")
PY

echo
echo "===== BUILD FINAL SUMMARY ====="

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path(
    "results/model_level/overnight_ablation_v1"
)

eval_root = root / "eval"

rows = []

for result_path in sorted(
    eval_root.glob("*/*/*/evaluation_result.json")
):
    x = json.loads(
        result_path.read_text(encoding="utf-8")
    )

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

if len(rows) != 22:
    raise RuntimeError(
        f"Expected 22 summary rows, got {len(rows)}"
    )

summary = {
    "schema_version": 1,
    "status": "complete",
    "evaluation_count": len(rows),
    "results": rows,
}

summary_path = root / "overnight_summary.json"

summary_path.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

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

for r in rows:
    print(
        f"{r['scope'][:14]:14s} "
        f"{r['author'][:14]:14s} "
        f"{r['model'][:34]:34s} "
        f"{r['mode']:8s} "
        f"{r['n']:6d} "
        f"{r['top1']:9.6f} "
        f"{r['mrr']:9.6f} "
        f"{r['missing']:9.6f}"
    )

print()
print("SUMMARY =", summary_path)
PY

echo
echo "===== FINALIZATION COMPLETE ====="
echo "END=$(date -Is)"
