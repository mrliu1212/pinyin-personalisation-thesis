# Model-Level Personalisation — File Index

**Last updated:** 2026-08-23
**Current stage:** Static Adapter qualification complete; longitudinal continual personalization next.

This index tracks the implementation, protocols, runners, datasets, checkpoints,
results, and reproducibility records for the Model-Level Adapter line.

---

## 1. Documentation

### `docs/model_level/01_ADAPTER_V1_HARD_GATE_PROTOCOL_2026-08-22.md`

Frozen Adapter architecture and correctness gates.

Status: **COMPLETE**

### `docs/model_level/02_ADAPTER_V1_TRAINING_CALIBRATION_CHECKPOINT_2026-08-23.md`

Training implementation, deterministic calibration, LR sweep, and initial
held-out evaluation.

Status: **COMPLETE**

### `docs/model_level/03_ADAPTER_V1_REPRODUCIBILITY_2026-08-23.md`

Canonical reproducibility record including:

- runtime;
- checkpoint identity;
- dataset SHA256;
- deterministic settings;
- frozen LR;
- tokenizer compatibility;
- Dev1000 identities;
- Overnight reproduction;
- unsupported-Pinyin semantics;
- Master A manifests and exact step controls;
- current longitudinal provenance boundary.

Status: **ACTIVE CANONICAL REPRODUCIBILITY RECORD**

### `docs/model_level/04_ADAPTER_FUTURE_WORK_2026-08-23.md`

Deferred architecture/extension ideas.

Status: **REFERENCE ONLY**

### `docs/model_level/05_ADAPTER_V1_REMAINING_RESEARCH_PLAN_2026-08-23.md`

Current remaining research plan.

Current scope:

- combined Train-Fit + Train-Val chronological stream;
- episode/probe freeze;
- prequential continual evaluation;
- warm Adapter;
- adaptation / rescue / harm;
- retention / forgetting;
- locality;
- Warm versus Rebuild;
- replay only if warranted;
- Temporary versus Persistent controlled stress test.

### `docs/model_level/06_ADAPTER_V1_FULL_VAL_CONFIRMATION_2026-08-23.md`

Full Agent Phage Train-Val confirmation.

Key result:

- Generic Full Top1: 0.819955
- Adapter Full Top1: 0.935813

Result:

**LR `5e-4` frozen.**

### `docs/model_level/07_OVERNIGHT_ABLATION_PROTOCOL_2026-08-23.md`

Pre-result specification for:

- Agent A/B/C;
- Recent500 / Recent5K / Recent25K / full;
- Full-only versus mixed 3:1:2;
- cross-author evaluation;
- fixed Dev1000;
- tokenizer representability gate.

Status: **COMPLETED**

### `docs/model_level/08_ADAPTER_STATIC_RESULTS_CHECKPOINT_2026-08-23.md`

Canonical static-result checkpoint.

Contains:

- Overnight results;
- Full Train-Val results;
- Agent history-size curve;
- Full-only versus mixed results;
- three-author results;
- Etinjat completion;
- unsupported-Pinyin semantics;
- Master A temporal controls;
- optimization control;
- matched-step diversity control;
- revised scientific interpretation.

Status: **STATIC PHASE COMPLETE**

---

## 2. Core implementation

### `src/model_level/pinyingpt_adapter.py`

Frozen serial bottleneck Adapter architecture.

- 12 Adapters
- hidden 768
- bottleneck 48
- ReLU
- residual serial insertion
- 894,528 trainable parameters/user

### `src/model_level/concat_training.py`

Exact target-only teacher-forcing implementation aligned with Concat candidate
scoring.

### `src/model_level/pinyin_modes.py`

Full / Initial / Mixed training representations.

---

## 3. Tests

### `tests/test_model_level_adapter.py`

Architecture, gradient, persistence, and behavior tests.

### `tests/test_adapter_evaluation_unsupported_pinyin.py`

Regression tests for denominator-preserving unsupported-Pinyin evaluation.

---

## 4. Experiment runners

### Hard gates

`experiments/model_level/run_adapter_hard_gates_v1.py`

### Canonical Adapter training

`experiments/model_level/run_adapter_training_v1.py`

### Training-policy wrapper

`experiments/model_level/run_adapter_training_policy_v1.py`

Supports static Full-only / mixed policies and recent/deterministic history
selection while retaining the canonical training implementation.

### Exact-manifest training wrapper

`experiments/model_level/run_adapter_training_manifest_v1.py`

Used by Master A for exact Oldest5K / Random5K / Recent5K row populations.

### Dev evaluation

`experiments/model_level/run_adapter_dev_evaluation_v1.py`

Includes safe resume and unsupported-Pinyin explicit-miss semantics.

### Condition evaluation

`experiments/model_level/run_adapter_condition_evaluation_v1.py`

Evaluates Full or Initial input using the shared evaluation implementation.

### Full Train-Val confirmation

`experiments/model_level/run_adapter_full_val_confirmation_v1.py`

### Overnight orchestration

`experiments/model_level/run_overnight_ablation_v1.sbatch`

### Etinjat finalizer

`experiments/model_level/finalize_overnight_etinjat_full_v1.sbatch`

### Master A

`experiments/model_level/run_long_term_master_a_v1.sbatch`

---

## 5. Frozen assets

Asset root:

`/home/3160454/work/model-level-assets-20260822`

### Base checkpoint

`pinyingpt2-concat/`

Revision:

`76dd20dc92d8236a350fb732e99dde6fa15e2263`

Model SHA256:

`1c5ebb9e7b15d75ea8899b914fc8363f4745703115253071f7834780263c74bb`

### Train-Fit

Rows:

`144,526`

SHA256:

`547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6`

### Train-Val

Rows:

`34,416`

SHA256:

`d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220`

Test remains sealed.

---

## 6. Effective static training populations

| Author | Nominal Train-Fit | Effective |
|---|---:|---:|
| Agent Phage | 55,926 | 55,925 |
| Etinjat | 32,906 | 32,555 |
| breaddddd | 55,694 | 55,682 |

Training compatibility audit:

`results/model_level/train_fit_tokenizer_unknown_audit_v1.json`

---

## 7. Important result directories

### Hard gates

`results/model_level/adapter_v1_hard_gates_cluster_a100_v2/`

Status: PASS

### LR calibration

`results/model_level/lr_calibration_2048_v1/`

### LR held-out evaluation

`results/model_level/lr_dev_eval_2048_v1/`

Frozen LR:

`5e-4`

### Fixed Dev1000

`results/model_level/fixed_dev1000_v1/`

### Overnight static experiments

`results/model_level/overnight_ablation_v1/`

### Long-term qualification / Master A

`results/model_level/long_term_qualification_v1/master_a_v1/`

Master A:

- Slurm job `634528`
- COMPLETED
- ExitCode `0:0`
- elapsed `01:30:45`
- Test used: false

Summary:

`results/model_level/long_term_qualification_v1/master_a_v1/master_a_summary.json`

---

## 8. Static headline results

### History-size Dev1000 Full Top1

- Recent500: 0.911
- Recent5K: **0.949**
- Recent25K: 0.943
- Recent27963: 0.943
- Full55925: 0.941

### Full-only versus mixed

Full55925 Full-only:

- Full Top1: 0.941
- Initial Top1: 0.630

Full55925 mixed 3:1:2:

- Full Top1: 0.936
- Initial Top1: 0.681

### Cross-author Full Dev1000 Top1

- Agent Phage: 0.941
- breaddddd: 0.944
- Etinjat: 0.707

### Master A Full Top1

- Oldest5K @625: 0.945
- Random5K @625: 0.941
- Recent5K @625: **0.949**
- Recent5K @6991: 0.868
- Full55925 @6991: 0.941

Primary interpretation:

- recency benefit: modest;
- repeated small-population optimization effect: large;
- history diversity: strongly protective.

---

## 9. Current research stage

### Complete

- architecture;
- hard gates;
- deterministic training;
- LR calibration;
- LR freeze;
- tokenizer compatibility;
- full Agent confirmation;
- Overnight static matrix;
- three-user static evaluation;
- full Train-Val evaluation;
- unsupported-Pinyin evaluation semantics;
- matched temporal controls;
- matched optimization control;
- matched-step history/diversity control;
- static result checkpoint.

### Next

1. combined longitudinal corpus freeze;
2. episode/probe manifests;
3. prequential test-before-train protocol;
4. warm continual Adapter;
5. adaptation / rescue / harm;
6. retention / forgetting;
7. generic preservation / locality;
8. Warm versus Rebuild control;
9. replay only if warranted;
10. Temporary versus Persistent controlled stress test.

---

## 10. Current scope boundary

The following are not part of the current main research line unless explicitly
reopened:

- bottleneck-size search;
- alternative Adapter architectures;
- LoRA;
- Controlled Write;
- Consolidation;
- Controlled Read;
- external-memory Hybrid;
- long-term × short-term control;
- broad engineering phase;
- final Test evaluation.

The current Model-Level line ends after the continual-learning and controlled
temporary/persistent adaptation experiments.
