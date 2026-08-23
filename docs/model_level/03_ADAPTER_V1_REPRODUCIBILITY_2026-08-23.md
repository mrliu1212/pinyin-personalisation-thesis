# Model-Level Adapter V1 — Reproducibility Record

**Date:** 2026-08-23

---

## 1. Repository

Cluster repository:

    /home/3160454/work/thesis-model-level

Branch:

    work/model-level-adapter

Base commit used when this line was created:

    c594e2a641a757700a48e8a796deaf50becb0b65

---

## 2. Runtime

Python:

    /home/3160454/work/envs/model-level-py312/bin/python

Versions:

    Python       3.12.14
    torch        2.11.0+cu128
    CUDA         12.8
    transformers 4.57.6
    safetensors  0.8.0

GPU used during current calibration:

    NVIDIA A100 80GB PCIe MIG 4g.40gb

---

## 3. Base checkpoint identity

Path:

    /home/3160454/work/model-level-assets-20260822/pinyingpt2-concat

Checkpoint ID:

    aihijo/transformers4ime-pinyingpt-concat

Revision:

    76dd20dc92d8236a350fb732e99dde6fa15e2263

Checksums:

    pytorch_model.bin
    1c5ebb9e7b15d75ea8899b914fc8363f4745703115253071f7834780263c74bb

    config.json
    86bb492283d576cc845cd2e9f2b67f8e07423a546659c5f1d427f146dd492cb4

    pinyin2char.json
    f344e2dd29253b50b8cc7a512b793996427d1400d7c2d0c79e0e12ff3628e142

    vocab.txt
    f7863b040bae29ac474065729355252248c92d41141c1e09fbf21dd3e593a238

    additional_special_tokens.json
    f1a5aa58b6549ae57048bd3a61e9060d4a8599e1372c2b92d0d8229b937cc11e

---

## 4. Dataset identity

Train-Fit:

    /home/3160454/work/model-level-assets-20260822/clean3_train_fit_v1.jsonl

Rows:

    144,526

SHA256:

    547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6

Train-Val:

    /home/3160454/work/model-level-assets-20260822/clean3_train_val_v1.jsonl

Rows:

    34,416

SHA256:

    d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220

Train-Val Agent Phage rows:

    13,741

No Test data was used for LR calibration.

---

## 5. Adapter architecture

Configuration:

    number of Adapters = 12
    hidden dimension = 768
    bottleneck dimension = 48
    reduction factor = 16
    activation = ReLU

Transformation:

    h' = h + W_up(ReLU(W_down(h)))

Trainable parameters:

    894,528

Base parameters remain frozen.

---

## 6. Determinism

Training runner enables:

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

Shell environment:

    CUBLAS_WORKSPACE_CONFIG=:4096:8
    NVIDIA_TF32_OVERRIDE=0

Strict repeat test:

    identical loss history = true
    identical Adapter SHA = true
    maximum Adapter tensor difference = 0

---

## 7. LR Train-Fit calibration reproduction

Runner:

    experiments/model_level/run_adapter_training_v1.py

Author:

    Agent Phage

Selection:

    2,048 deterministic Train-Fit rows

Seed:

    20260822

Configuration:

    epochs = 1
    batch size = 8
    weight decay = 0
    max grad norm = 1

Learning rates:

    5e-5
    1e-4
    2e-4
    5e-4

Output root:

    results/model_level/lr_calibration_2048_v1/

These checkpoints are calibration pilots, not formal full-user models.

---

## 8. LR held-out evaluation reproduction

Runner:

    experiments/model_level/run_adapter_dev_evaluation_v1.py

Output:

    results/model_level/lr_dev_eval_2048_v1/

Author:

    Agent Phage

Selection seed:

    20260822

Rows:

    2,048 deterministic Standard Train-Val rows

Condition:

    Pinyin = Full
    target = Short

Inference:

    Beam = 16
    Top-K = 10

Conditions:

    Generic
    5e-5
    1e-4
    2e-4
    5e-4

Selected population is persisted in:

    results/model_level/lr_dev_eval_2048_v1/selected_rows.jsonl

Evaluation configuration:

    results/model_level/lr_dev_eval_2048_v1/evaluation_config.json

Aggregate result:

    results/model_level/lr_dev_eval_2048_v1/evaluation_result.json

---

## 9. LR Dev results

Generic:

    Top1    0.833984
    Top3    0.947266
    Top5    0.963867
    Top10   0.972656
    MRR@10  0.892432
    Missing 0.027344

LR=5e-5:

    Top1    0.911621
    MRR@10  0.942354
    rescue  213
    harm     54
    net     +159

LR=1e-4:

    Top1    0.921875
    MRR@10  0.950061
    rescue  228
    harm     48
    net     +180

LR=2e-4:

    Top1    0.928223
    MRR@10  0.954739
    rescue  238
    harm     45
    net     +193

LR=5e-4:

    Top1    0.938965
    Top3    0.981934
    Top5    0.986328
    Top10   0.991699
    MRR@10  0.960669
    Missing 0.008301
    rescue  255
    harm     40
    net     +215

Current LR leader:

    5e-4

Not yet frozen pending larger/full validation.

---

## 10. Evaluation resume semantics

The Dev runner supports safe condition-level and row-level resume.

If a prediction JSONL already exists, it verifies:

    row_id belongs to frozen selected population
    no duplicate row_id
    condition matches
    gold matches
    segmented Pinyin matches
    top_k matches
    beam_size matches

Existing valid rows are reused.

Missing rows are appended.

Returned predictions are reordered into frozen selected-row order before
metric computation.

Observed real recovery:

    persisted before interruption = 1,887 / 2,048 Generic rows
    resumed remaining = 161
    final = 2,048 / 2,048

---

## 11. Slurm / GPU note

During final LR Dev completion, the shared allocation was:

    Job 634214
    gnode02
    8 CPUs
    2 GPUs

M0 occupied one exclusive step.

Model-Level used another exclusive step:

    --cpus-per-task=4
    --gres=gpu:1

For the Model-Level step:

    SLURM_STEP_GPUS=1

Inside the isolated process:

    CUDA_VISIBLE_DEVICES=0
    torch.cuda.device_count()=1

The internal CUDA index is remapped by Slurm and does not mean the step used
the same allocation GPU as M0.

---

## 12. Reproducibility status

Architecture reproducibility:

    PASS

Exact training-loss regression:

    PASS

Cross-platform candidate equivalence:

    PASS

Strict CUDA training repeatability:

    PASS

Frozen held-out subset persistence:

    PASS

Interrupted evaluation resume:

    PASS

Test isolation:

    PASS

---

## 11. Post-calibration protocol freeze

The earlier LR-calibration status in this document is historical.

Subsequent full-development confirmation froze:

`LR = 5e-4`

for Adapter V1.

Full Agent Phage Train-Val confirmation:

- N = 13,741;
- Generic Top1 = 0.8199548796;
- Adapter Top1 = 0.935812532;
- Adapter Top3 = 0.982897897;
- Adapter MRR = 0.959568716;
- Adapter Missing = 0.008660214.

The static experiments after this point use LR `5e-4`.

---

## 12. Tokenizer-compatible training-target gate

A complete frozen-tokenizer audit found the following Train-Fit populations.

Nominal:

- Agent Phage = 55,926;
- Etinjat = 32,906;
- breaddddd = 55,694.

Effective tokenizer-compatible training populations:

- Agent Phage = 55,925;
- Etinjat = 32,555;
- breaddddd = 55,682.

Total excluded rows:

`364`

The training gate is applied after nominal experiment-population selection.

Recent-N therefore means:

1. select the nominal most-recent N rows;
2. apply the frozen-checkpoint target-compatibility gate.

Evaluation populations are not filtered by this training gate.

Audit output:

`results/model_level/train_fit_tokenizer_unknown_audit_v1.json`

Implementation commit:

`2c23bb5`

---

## 13. Fixed author Dev1000 identity

Selection seed:

`20260822`

SHA256:

Agent Phage:

`aebdc7e60831ac5d67b5459db0c172afa7e7bf29d7d6df9bfc380a459a2e89ed`

Etinjat:

`a38c9d3feceb99f47046770c8bcfb68fc3a0a149897e548ab15545cfb283b454`

breaddddd:

`20bf977540d000c7021674df674d0c07464842c5c161ef12ddac1c7d49a84814`

Root:

`results/model_level/fixed_dev1000_v1/`

These manifests are subsets of Train-Val.

---

## 14. Overnight static experiment reproduction

Protocol:

`docs/model_level/07_OVERNIGHT_ABLATION_PROTOCOL_2026-08-23.md`

Runner:

`experiments/model_level/run_overnight_ablation_v1.sbatch`

Output root:

`results/model_level/overnight_ablation_v1/`

Training:

- LR = `5e-4`;
- batch size = 8;
- epoch = 1;
- seed = `20260822`;
- Short targets;
- Full-only unless explicitly mixed;
- Full:Initial:Mixed mixed condition = `3:1:2`.

Primary Agent conditions:

- recent500 Full-only;
- recent5000 Full-only;
- recent25000 Full-only;
- recent27963 Full-only;
- full55925 Full-only;
- full55925 mixed 3:1:2.

Cross-author full-history Full-only:

- Agent Phage;
- Etinjat;
- breaddddd.

Evaluation:

- Beam = 16;
- Top-K = 10;
- fixed Dev1000 Full and Initial;
- complete Train-Val Full and Initial.

GPU-isolation commit:

`3ba4fdc`

Etinjat-finalizer commit:

`dddaa04`

---

## 15. Unsupported-Pinyin evaluation reproduction

Evaluation denominator is preserved when the frozen checkpoint has no tokenizer
candidate set for an input Pinyin segment.

Caught condition:

a `ValueError` whose message begins exactly with:

`no tokenizer candidates for Pinyin `

Recorded semantics:

- candidates = `[]`;
- candidate scores = `[]`;
- `gold_top10_rank = null`;
- Top1 = false;
- Top3 = false;
- Top5 = false;
- Top10 present = false;
- reciprocal rank = 0;
- `unsupported_input = true`;
- explicit unsupported reason.

All unrelated `ValueError`s still propagate.

Regression test:

`tests/test_adapter_evaluation_unsupported_pinyin.py`

Implementation commit:

`18bf358`

This policy completed the previously interrupted Etinjat Full Train-Val run.

Etinjat Full Train-Val:

- N = 8,030;
- Top1 = 0.688044832;
- Top3 = 0.823163138;
- Top5 = 0.864508095;
- Top10 = 0.900000000;
- MRR = 0.763310354;
- Missing = 0.100000000.

---

## 16. Master A history-manifest identity

Pre-freeze audit root:

`results/model_level/long_term_qualification_v1/audit_pre_freeze_v1/`

Manifests:

- `oldest5000_row_ids.json`
- `random5000_row_ids.json`
- `recent5000_row_ids.json`

Each contains exactly 5,000 Agent Phage compatible Train-Fit row IDs.

Selection seed:

`20260822`

Selection definitions:

- Oldest5K = first 5,000 chronological compatible rows;
- Recent5K = last 5,000 chronological compatible rows;
- Random5K = seeded sample of 5,000 compatible rows, then chronological sort.

Observed overlap:

- Oldest5K vs Random5K = 453;
- Oldest5K vs Recent5K = 0;
- Random5K vs Recent5K = 432.

Manifest training wrapper:

`experiments/model_level/run_adapter_training_manifest_v1.py`

The wrapper changes only exact row-population selection and delegates the
canonical architecture/loss/optimizer/batching/save behavior to the existing
training runner.

---

## 17. Master A optimizer-step controls

Batch size:

`8`

5K one-pass condition:

`5000 / 8 = 625 optimizer steps`

Full Agent effective history:

`55,925 / 8 -> ceil = 6,991 optimizer steps`

Recent5K-long:

- same 5,000 unique Recent5K rows;
- maximum epochs = 12;
- exact `max_steps = 6991`;
- 11 complete epochs = 6,875 steps;
- epoch 12 contributes 116 additional steps;
- total = exactly 6,991 steps;
- row exposures = 55,928.

Full55925:

- unique rows = 55,925;
- one epoch;
- optimizer steps = 6,991;
- row exposures = 55,925;
- last batch size = 5.

Therefore the matched-step comparison has a three-row exposure difference:

`55,928 versus 55,925`

This is recorded explicitly. The experiment is an exact optimizer-step control,
not an exact row-exposure control.

---

## 18. Master A reproduction

Runner:

`experiments/model_level/run_long_term_master_a_v1.sbatch`

Commit introducing Master A:

`5b0f19e`

Slurm job:

`634528`

State:

`COMPLETED`

Exit code:

`0:0`

Elapsed:

`01:30:45`

Summary JSON:

`results/model_level/long_term_qualification_v1/master_a_v1/master_a_summary.json`

Summary CSV:

`results/model_level/long_term_qualification_v1/master_a_v1/master_a_summary.csv`

Test used:

`false`

Conditions:

| Condition | Unique rows | Steps |
|---|---:|---:|
| Oldest5K | 5,000 | 625 |
| Random5K | 5,000 | 625 |
| Recent5K | 5,000 | 625 |
| Recent5K-long | 5,000 | 6,991 |
| Full55925 | 55,925 | 6,991 |

Dev1000 Top1:

| Condition | Full | Initial |
|---|---:|---:|
| Oldest5K | 0.945 | 0.652 |
| Random5K | 0.941 | 0.652 |
| Recent5K | 0.949 | 0.659 |
| Recent5K-long | 0.868 | 0.567 |
| Full55925 | 0.941 | 0.630 |

Primary controls:

- Recent5K minus Oldest5K @625:
  Full Top1 `+0.4 pp`;
- Recent5K minus Random5K @625:
  Full Top1 `+0.8 pp`;
- Recent5K @6991 minus Recent5K @625:
  Full Top1 `-8.1 pp`;
- Full55925 minus Recent5K-long @6991:
  Full Top1 `+7.3 pp`.

---

## 19. Static-phase interpretation

Static results support:

- modest temporal-relevance benefit;
- large repeated-exposure optimization effect;
- strong protection from training-set diversity;
- substantial cross-author heterogeneity;
- clear Full-versus-Initial training-policy tradeoff.

They do not support the simple claim that older history is intrinsically
harmful.

Canonical result checkpoint:

`docs/model_level/08_ADAPTER_STATIC_RESULTS_CHECKPOINT_2026-08-23.md`

---

## 20. Longitudinal development corpus

The next Model-Level phase uses the chronological Agent Phage stream formed from:

`Train-Fit + Train-Val`

Nominal combined rows:

`69,667`

Tokenizer-compatible effective rows:

`69,664`

Works:

`45`

Work indices:

`0–44`

Train-Fit:

- work indices 0–30;
- 55,926 nominal rows.

Train-Val:

- work indices 31–44;
- 13,741 rows.

No row-ID or work-index overlap exists between Train-Fit and Train-Val.

Test remains sealed.

Once Train-Val is used in this longitudinal branch, it is no longer described
as untouched validation for that branch.

---

## 21. Current reproducibility boundary

Static Adapter qualification is complete.

The next provenance objects to freeze are:

1. combined chronological corpus audit;
2. episode manifests;
3. future-probe manifests;
4. prequential ordering;
5. warm Adapter checkpoint lineage;
6. retention-evaluation lineage.

No Test data is included in the current Model-Level research line.
