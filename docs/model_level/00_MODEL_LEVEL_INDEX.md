# Model-Level Personalisation — File Index

**Last updated:** 2026-08-23

This index tracks the implementation, protocols, runners, datasets, checkpoints,
and evaluation artifacts for the Model-Level Adapter personalisation line.

---

## 1. Core documentation

### `README_MODEL_LEVEL_ADAPTER_V1.md`
High-level entry point for Model-Level Adapter V1.

### `docs/model_level/01_ADAPTER_V1_HARD_GATE_PROTOCOL_2026-08-22.md`
Frozen architecture, correctness gates, and hard-gate protocol.

### `docs/model_level/02_ADAPTER_V1_TRAINING_CALIBRATION_CHECKPOINT_2026-08-23.md`
Research checkpoint covering:
- frozen architecture
- training objective
- deterministic training
- Train-Fit / Train-Val separation
- pilot training
- LR calibration
- held-out 2,048-row LR Dev evaluation
- current research status

### `docs/model_level/03_ADAPTER_V1_REPRODUCIBILITY_2026-08-23.md`
Reproduction record covering:
- environment
- checkpoint identity
- data identity
- deterministic settings
- training calibration
- evaluation settings
- resume behavior
- current LR-selection result

---

## 2. Core implementation

### `src/model_level/pinyingpt_adapter.py`
Serial residual bottleneck Adapter implementation.

Frozen Adapter V1:
- 12 adapters
- hidden size 768
- bottleneck 48
- reduction factor 16
- ReLU
- residual form
- zero-initialized up projection
- 894,528 trainable parameters/user

### `src/model_level/concat_training.py`
Exact PinyinGPT2-Concat target-only teacher-forcing loss.

### `src/model_level/pinyin_modes.py`
Deterministic Full / Initial / Mixed Pinyin exposure policy.

---

## 3. Tests

### `tests/test_model_level_adapter.py`
Adapter architecture and behavior tests.

---

## 4. Experiment runners

### `experiments/model_level/run_adapter_hard_gates_v1.py`
Architecture, gradient, save/reload, scorer-regression, and platform gates.

### `experiments/model_level/run_adapter_training_v1.py`
Adapter training runner.

Supports:
- deterministic population selection
- deterministic row order
- deterministic Pinyin-mode assignment
- mini-batch training
- Adapter-only optimization
- provenance output
- pilot/formal distinction

### `experiments/model_level/run_adapter_dev_evaluation_v1.py`
Held-out Standard Train-Val evaluation.

Current frozen primary calibration setup:
- Agent Phage
- Full Pinyin
- Short target
- Beam 16
- Top-10
- deterministic row selection
- Generic + multiple Adapter checkpoints
- safe prediction resume

Backup before resume modification:

    experiments/model_level/run_adapter_dev_evaluation_v1.py.pre_resume_20260823

---

## 5. Frozen external assets

Cluster asset root:

    /home/3160454/work/model-level-assets-20260822

### Base checkpoint

    pinyingpt2-concat/

Checkpoint:

    aihijo/transformers4ime-pinyingpt-concat

Revision:

    76dd20dc92d8236a350fb732e99dde6fa15e2263

### Train-Fit

    clean3_train_fit_v1.jsonl

Rows:

    144,526

SHA256:

    547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6

### Train-Val

    clean3_train_val_v1.jsonl

Rows:

    34,416

SHA256:

    d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220

Agent Phage Train-Val rows:

    13,741

Test has not been used for Model-Level training-policy selection.

---

## 6. Important result directories

### Hard gates

    results/model_level/adapter_v1_hard_gates_cluster_a100_v2/

Status:

    PASS

### Initial pilots

    results/model_level/agent_phage_pilot_512_b8_lr1e4_v1/
    results/model_level/agent_phage_pilot_512_b8_lr1e4_repro_v1/

### Strict deterministic repetitions

    results/model_level/agent_phage_pilot_512_deterministic_a/
    results/model_level/agent_phage_pilot_512_deterministic_b/

Result:

    Adapter SHA identical
    loss history identical
    max weight difference = 0

### LR Train-Fit calibration

    results/model_level/lr_calibration_2048_v1/

Learning rates:

    5e-5
    1e-4
    2e-4
    5e-4

### LR held-out Dev calibration

    results/model_level/lr_dev_eval_2048_v1/

Files:

    evaluation_config.json
    evaluation_result.json
    selected_rows.jsonl

    generic_metrics.json
    generic_predictions.jsonl

    lr_5e-5_metrics.json
    lr_5e-5_predictions.jsonl

    lr_1e-4_metrics.json
    lr_1e-4_predictions.jsonl

    lr_2e-4_metrics.json
    lr_2e-4_predictions.jsonl

    lr_5e-4_metrics.json
    lr_5e-4_predictions.jsonl

Current calibration leader:

    LR = 5e-4

This is not yet the final frozen LR.

---

## 7. Current research stage

Completed:

    architecture
    exact training loss
    Pinyin exposure policy
    Adapter-only gradient gate
    save / reload gate
    cross-platform equivalence
    deterministic GPU training
    training runner
    Train-Fit LR calibration
    Train-Val provenance
    Train/Val leakage audit
    Dev inference smoke
    resumable Dev evaluation
    2,048-row held-out LR comparison

Current next stage:

    larger/full Agent Phage Train-Val confirmation

Then:

    freeze LR
    calibrate training duration
    full-user training
    multi-user evaluation
    Model-Level vs reranking
    optional hybrid
    final untouched Test

## Deferred Future Work

- `docs/model_level/04_ADAPTER_FUTURE_WORK_2026-08-23.md`
  - Unified Full / Initial / Mixed Pinyin training.
  - Controlled Short / Multi1-Multi5 training mixtures.
  - Training-mixture ablations are intentionally deferred from Adapter V1.
  - Future hybrid integration with external-memory reranking.

## Remaining Research Plan

- `docs/model_level/05_ADAPTER_V1_REMAINING_RESEARCH_PLAN_2026-08-23.md`
  - Full Agent Phage LR=5e-4 confirmation.
  - Agent Phage training-data learning curve.
  - Minimal training-duration decision.
  - Formal three-user Adapter training and validation.
  - External-memory versus Model-Level comparison.
  - Planned Hybrid experiment.
  - Final untouched Test protocol.
  - Explicitly deferred experiments and Future Work boundaries.

## Full Train-Val Confirmation

- `docs/model_level/06_ADAPTER_V1_FULL_VAL_CONFIRMATION_2026-08-23.md`
  - Full 13,741-row Agent Phage Train-Val confirmation.
  - LR=5e-4 Top1 = 0.935813.
  - Generic Top1 = 0.819955.
  - Top1 gain = +11.586 percentage points.
  - LR=5e-4 frozen for Adapter V1.
