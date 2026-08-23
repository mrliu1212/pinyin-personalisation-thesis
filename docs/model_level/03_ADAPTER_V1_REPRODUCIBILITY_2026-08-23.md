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
