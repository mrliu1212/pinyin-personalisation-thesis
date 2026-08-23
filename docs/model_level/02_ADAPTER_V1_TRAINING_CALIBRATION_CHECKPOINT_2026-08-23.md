# Model-Level Adapter V1 — Training Calibration Checkpoint

**Date:** 2026-08-23  
**Status:** LR screening complete; larger validation confirmation pending  
**Test status:** untouched

---

## 1. Research question

Model-Level Adapter V1 tests whether a small per-user trainable Adapter can
personalize frozen PinyinGPT2-Concat candidate ranking.

The current open question is no longer architecture correctness.

The current question is:

> Which training policy gives the best held-out candidate-ranking
> personalisation for the frozen Adapter V1 architecture?

The current sub-question is learning-rate selection.

---

## 2. Frozen architecture

Base:

- PinyinGPT2-Concat
- 12 GPT-2 blocks
- hidden size 768
- frozen base model
- frozen final LayerNorm
- frozen tied LM head

Adapter:

    h' = h + W_up(ReLU(W_down(h)))

Frozen configuration:

- one Adapter after each GPT2Block
- 12 independent Adapters
- bottleneck 48
- reduction factor 16
- no extra LayerNorm
- no dropout
- no gate
- no learned residual scale
- zero-initialized up projection
- 894,528 trainable parameters/user

Only Adapter parameters train.

---

## 3. Frozen data

Train-Fit:

    /home/3160454/work/model-level-assets-20260822/clean3_train_fit_v1.jsonl

Rows:

    144,526

SHA256:

    547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6

Per author:

    Agent Phage    55,926
    Etinjat        32,906
    breaddddd      55,694

Train-Val:

    /home/3160454/work/model-level-assets-20260822/clean3_train_val_v1.jsonl

Rows:

    34,416

SHA256:

    d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220

Per author:

    Agent Phage    13,741
    Etinjat         8,030
    breaddddd      12,645

Train-Fit / Train-Val audit:

    exact row-ID overlap = 0
    exact source-span overlap = 0

Test has not been used.

---

## 4. Frozen training semantics

Each Train-Fit row appears once per epoch.

Requested Pinyin exposure:

    Full : Initial : Mixed = 3 : 1 : 2

Mode assignment is deterministic from:

    SHA256(seed, epoch, row_id)

Training loss is exact target-only PinyinGPT2-Concat teacher forcing.

Context budget:

    n_positions - (2 + 2*m)

Hard-gate scorer regression maximum difference:

    4.76837158203125e-07

Tolerance:

    1e-5

PASS.

---

## 5. Strict GPU reproducibility

Strict deterministic settings:

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

Environment:

    CUBLAS_WORKSPACE_CONFIG=:4096:8
    NVIDIA_TF32_OVERRIDE=0

Repeated 512-row runs produced:

    identical population
    identical Pinyin mode counts
    identical loss history
    identical Adapter SHA256
    maximum weight difference = 0

PASS.

---

## 6. Train-Fit LR calibration

Agent Phage deterministic subset:

    2,048 Train-Fit rows
    1 epoch
    batch size 8
    256 steps
    seed 20260822

Results:

| LR | Final loss | Mean loss |
|---:|---:|---:|
| 5e-5 | 4.175084 | 3.902741 |
| 1e-4 | 3.045703 | 3.682965 |
| 2e-4 | 2.731161 | 3.508314 |
| 5e-4 | 2.629570 | 3.361018 |

Training loss alone was explicitly not used to select the LR.

---

## 7. Held-out LR Dev calibration

Runner:

    experiments/model_level/run_adapter_dev_evaluation_v1.py

Population:

    Agent Phage Standard Train-Val

Selection:

    deterministic 2,048-row subset
    seed = 20260822

Frozen inference condition:

    Pinyin = Full
    target = Short
    Beam = 16
    Top-K = 10

Every condition used exactly the same rows.

### Results

| Condition | Top1 | Top3 | Top5 | Top10 | MRR@10 | Missing@10 |
|---|---:|---:|---:|---:|---:|---:|
| Generic | 0.833984 | 0.947266 | 0.963867 | 0.972656 | 0.892432 | 0.027344 |
| 5e-5 | 0.911621 | 0.969727 | 0.979980 | 0.988281 | 0.942354 | 0.011719 |
| 1e-4 | 0.921875 | 0.978027 | 0.984863 | 0.991211 | 0.950061 | 0.008789 |
| 2e-4 | 0.928223 | 0.979980 | 0.984375 | 0.992676 | 0.954739 | 0.007324 |
| 5e-4 | 0.938965 | 0.981934 | 0.986328 | 0.991699 | 0.960669 | 0.008301 |

Paired Top-1 outcomes versus Generic:

| LR | Rescue | Harm | Net |
|---:|---:|---:|---:|
| 5e-5 | 213 | 54 | +159 |
| 1e-4 | 228 | 48 | +180 |
| 2e-4 | 238 | 45 | +193 |
| 5e-4 | 255 | 40 | +215 |

For LR=5e-4:

    Generic Top1 = 0.833984
    Adapter Top1 = 0.938965
    absolute gain = +0.104981

    rescue = 255
    harm = 40
    net = +215

    rescue / harm = 6.375

LR=5e-4 is the current primary candidate.

LR=2e-4 remains the runner-up and has slightly better Top10 / Missing@10.

The LR is not yet considered finally frozen.

---

## 8. Resume incident and recovery

The first LR Dev evaluation was interrupted because its shared parent Slurm
allocation terminated.

At interruption:

    Generic predictions safely written = 1,887 / 2,048

The original evaluation runner opened output with write mode and therefore did
not support safe continuation.

The runner was modified to:

- load existing prediction JSONL
- validate row IDs
- reject unknown rows
- reject duplicate rows
- validate condition
- validate gold
- validate Pinyin
- validate Top-K
- validate Beam size
- skip already-completed rows
- append only missing rows
- return predictions in frozen selected-row order

Resume validation:

    generic: resume_validated=1887/2048 remaining=161

Recovery:

    generic: 2048/2048

The four Adapter conditions then completed successfully.

This resume behavior is now part of the Model-Level Dev evaluation runner.

---

## 9. Current interpretation

The 2,048-row held-out comparison provides strong evidence that Adapter
personalisation is producing a real positive ranking signal for Agent Phage.

However, the current result must not yet be reported as the final system result
because:

- only Agent Phage has been evaluated in this calibration
- only 2,048 of 13,741 Agent Phage Train-Val rows were used
- these Adapter checkpoints were trained on only 2,048 Train-Fit rows
- training duration has not yet been frozen
- Test remains untouched

Current defensible conclusion:

> LR=5e-4 is the leading learning-rate candidate on the deterministic
> 2,048-row held-out Agent Phage Standard Train-Val calibration subset.

---

## 10. Next stage

Recommended next confirmation:

    full/larger Agent Phage Standard Train-Val

Primary conditions:

    Generic
    LR=5e-4

Useful runner-up confirmation:

    LR=2e-4

After larger validation:

    freeze LR
    calibrate training duration
    perform formal full-user training
