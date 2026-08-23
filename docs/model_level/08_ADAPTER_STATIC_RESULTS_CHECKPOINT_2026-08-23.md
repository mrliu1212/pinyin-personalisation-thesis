# Model-Level Adapter V1 — Static Results Checkpoint

**Date:** 2026-08-23
**Status:** STATIC ADAPTER QUALIFICATION COMPLETE
**Test used:** No

---

## 1. Scope

This checkpoint records the completed static Model-Level Adapter experiments
before the longitudinal prequential continual-learning phase.

The following are now complete:

- Adapter V1 architecture and hard gates;
- deterministic training verification;
- learning-rate calibration;
- full Agent Phage Train-Val confirmation;
- tokenizer-compatibility training gate;
- Agent history-size ablation;
- Full-only versus mixed-Pinyin training comparison;
- cross-author static Adapter evaluation;
- full Train-Val confirmation for Agent Phage, breaddddd, and Etinjat;
- matched-budget temporal controls;
- matched-population optimization control;
- matched-step history/diversity control.

The next phase is chronological continual personalization, not additional
static hyperparameter search.

---

## 2. Frozen Adapter configuration

Base model:

- PinyinGPT2-Concat;
- base parameters frozen;
- 12 GPT2 blocks;
- one independent serial residual bottleneck Adapter after each completed block.

Adapter:

- hidden dimension: 768;
- bottleneck dimension: 48;
- reduction factor: 16;
- activation: ReLU;
- no Adapter LayerNorm;
- no Adapter dropout;
- no gate;
- no learned scale;
- zero-initialized up projection;
- residual transformation:

  `h' = h + W_up(ReLU(W_down(h)))`

Parameters:

- base parameters: 102,408,960;
- Adapter trainable parameters per user: 894,528.

Training:

- target-only teacher forcing aligned with `score_candidates()`;
- Short targets;
- AdamW;
- weight decay: 0;
- max gradient norm: 1;
- frozen learning rate: `5e-4`;
- batch size: 8;
- deterministic seed: `20260822`.

---

## 3. Frozen datasets

### Train-Fit

Path:

`/home/3160454/work/model-level-assets-20260822/clean3_train_fit_v1.jsonl`

Rows:

`144,526`

SHA256:

`547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6`

Nominal author rows:

- Agent Phage: 55,926
- Etinjat: 32,906
- breaddddd: 55,694

Tokenizer-compatible effective training rows:

- Agent Phage: 55,925
- Etinjat: 32,555
- breaddddd: 55,682

### Train-Val

Path:

`/home/3160454/work/model-level-assets-20260822/clean3_train_val_v1.jsonl`

Rows:

`34,416`

SHA256:

`d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220`

Author rows:

- Agent Phage: 13,741
- Etinjat: 8,030
- breaddddd: 12,645

Train-Fit / Train-Val leakage audit:

- row-ID overlap: 0;
- source-span overlap: 0.

Test remained sealed throughout all experiments in this checkpoint.

---

## 4. Frozen checkpoint

Path:

`/home/3160454/work/model-level-assets-20260822/pinyingpt2-concat`

Checkpoint ID:

`aihijo/transformers4ime-pinyingpt-concat`

Revision:

`76dd20dc92d8236a350fb732e99dde6fa15e2263`

Important SHA256 values:

- `pytorch_model.bin`:
  `1c5ebb9e7b15d75ea8899b914fc8363f4745703115253071f7834780263c74bb`
- `config.json`:
  `86bb492283d576cc845cd2e9f2b67f8e07423a546659c5f1d427f146dd492cb4`
- `pinyin2char.json`:
  `f344e2dd29253b50b8cc7a512b793996427d1400d7c2d0c79e0e12ff3628e142`
- `vocab.txt`:
  `f7863b040bae29ac474065729355252248c92d41141c1e09fbf21dd3e593a238`
- `additional_special_tokens.json`:
  `f1a5aa58b6549ae57048bd3a61e9060d4a8599e1372c2b92d0d8229b937cc11e`

---

## 5. Frozen Dev1000 manifests

All Dev1000 subsets are deterministic Train-Val subsets using seed
`20260822`.

SHA256:

- Agent Phage:
  `aebdc7e60831ac5d67b5459db0c172afa7e7bf29d7d6df9bfc380a459a2e89ed`
- Etinjat:
  `a38c9d3feceb99f47046770c8bcfb68fc3a0a149897e548ab15545cfb283b454`
- breaddddd:
  `20bf977540d000c7021674df674d0c07464842c5c161ef12ddac1c7d49a84814`

These are fixed development subsets, not additional data partitions.

---

## 6. LR calibration

Agent Phage, deterministic 2,048-row Train-Fit calibration:

| LR | Final calibration loss |
|---|---:|
| 5e-5 | 3.902741 |
| 1e-4 | 3.682965 |
| 2e-4 | 3.508314 |
| 5e-4 | **3.361018** |

Held-out Agent 2,048-row Train-Val evaluation:

### Generic

- Top1: 0.833984
- Top3: 0.947266
- Top5: 0.963867
- Top10: 0.972656
- MRR@10: 0.892432
- Missing@10: 0.027344

### Adapter LR=5e-4

- Top1: 0.938965
- Top3: 0.981934
- Top5: 0.986328
- Top10: 0.991699
- MRR@10: 0.960669
- Missing@10: 0.008301
- rescue: 255
- harm: 40
- net rescue: +215

Full Agent Train-Val confirmation also remained strongly positive.

**Frozen Adapter V1 learning rate: `5e-4`.**

---

## 7. Overnight static experiments

Frozen static training settings unless stated otherwise:

- LR: `5e-4`
- batch size: 8
- seed: `20260822`
- one epoch
- Short targets
- Full-only training except mixed condition C.

### 7.1 Agent history-size ablation

Fixed Agent Dev1000:

| Training history | Unique rows | Full Top1 | Initial Top1 |
|---|---:|---:|---:|
| Recent500 | 500 | 0.911 | 0.579 |
| Recent5000 | 5,000 | **0.949** | 0.659 |
| Recent25000 | 25,000 | 0.943 | 0.671 |
| Recent27963 | 27,963 | 0.943 | 0.661 |
| Full55925 | 55,925 | 0.941 | 0.630 |

Observed Full Top1 curve:

`91.1% -> 94.9% -> 94.3% -> 94.3% -> 94.1%`

This curve is descriptive only because one epoch implies different optimizer
step counts for different history sizes.

---

## 8. Full-only versus mixed training

Same effective Agent full Train-Fit population.

### B — Full-only

Dev1000:

| Mode | Top1 | Top3 | MRR@10 | Missing@10 |
|---|---:|---:|---:|---:|
| Full | 0.941 | 0.987 | 0.963769 | 0.007 |
| Initial | 0.630 | 0.771 | 0.719298 | 0.083 |

### C — Full:Initial:Mixed = 3:1:2

Dev1000:

| Mode | Top1 | Top3 | MRR@10 | Missing@10 |
|---|---:|---:|---:|---:|
| Full | 0.936 | 0.988 | 0.961575 | 0.005 |
| Initial | 0.681 | 0.840 | 0.767967 | 0.073 |

B -> C:

- Full Top1: `-0.5 pp`
- Initial Top1: `+5.1 pp`

Interpretation:

Mixed-input exposure substantially improves Initial-Pinyin robustness while
incurring a small Full-Pinyin Top1 cost.

---

## 9. Cross-author Dev1000 results

Full-history, Full-only, one-epoch Adapters:

| Author | Effective training rows | Full Top1 | Initial Top1 |
|---|---:|---:|---:|
| Agent Phage | 55,925 | 0.941 | 0.630 |
| breaddddd | 55,682 | 0.944 | 0.600 |
| Etinjat | 32,555 | 0.707 | 0.412 |

The large Etinjat gap demonstrates substantial user heterogeneity.

---

## 10. Full Train-Val confirmation

### Agent Phage — Full55925 Full-only

Full:

- N: 13,741
- Top1: 0.93952
- Top3: 0.98617
- MRR@10: 0.962606
- Missing@10: 0.00735

Initial:

- N: 13,741
- Top1: 0.61582
- Top3: 0.79266
- MRR@10: 0.715847
- Missing@10: 0.09330

### breaddddd — Full55682 Full-only

Full:

- N: 12,645
- Top1: 0.94124
- Top3: 0.98687
- MRR@10: 0.964111
- Missing@10: 0.00474

Initial:

- N: 12,645
- Top1: 0.59359
- Top3: 0.77541
- MRR@10: 0.696939
- Missing@10: 0.10320

### Etinjat — Full32555 Full-only

Full:

- N: 8,030
- Top1: 0.688044832
- Top3: 0.823163138
- Top5: 0.864508095
- Top10: 0.900000000
- MRR@10: 0.763310354
- Missing@10: 0.100000000

Initial:

- N: 8,030
- Top1: 0.37397
- Top3: 0.50971
- MRR@10: 0.456990
- Missing@10: 0.35679

---

## 11. Unsupported-Pinyin evaluation policy

Training and evaluation compatibility are intentionally different.

Training:

- tokenizer-incompatible Gold targets are excluded;
- exclusions are recorded in provenance.

Evaluation:

- denominator is preserved;
- unsupported Pinyin input is not removed.

If generation raises exactly a `ValueError` whose message starts with:

`no tokenizer candidates for Pinyin `

then the row is retained with:

- empty candidates;
- empty candidate scores;
- `gold_top10_rank = null`;
- Top1/Top3/Top5/Top10 = false;
- reciprocal rank = 0;
- `unsupported_input = true`;
- explicit unsupported reason.

All unrelated `ValueError`s remain hard failures.

Regression tests cover:

1. supported generation unchanged;
2. unsupported Pinyin becomes an explicit miss;
3. unrelated `ValueError` still propagates.

This policy allowed Etinjat Full Train-Val to complete all 8,030 rows.

---

## 12. Master A temporal / optimization controls

Master A compares five Agent conditions on the same fixed Dev1000.

| Condition | Unique rows | Steps | Full Top1 | Initial Top1 |
|---|---:|---:|---:|---:|
| Oldest5K | 5,000 | 625 | 0.945 | 0.652 |
| Random5K | 5,000 | 625 | 0.941 | 0.652 |
| Recent5K | 5,000 | 625 | **0.949** | **0.659** |
| Recent5K-long | 5,000 | 6,991 | 0.868 | 0.567 |
| Full55925 | 55,925 | 6,991 | 0.941 | 0.630 |

Full metric table:

| Condition | Mode | Steps | Top1 | Top3 | Top5 | Top10 | MRR@10 | Missing@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Oldest5K | Full | 625 | 0.945 | 0.987 | 0.990 | 0.993 | 0.965869 | 0.007 |
| Oldest5K | Initial | 625 | 0.652 | 0.837 | 0.877 | 0.927 | 0.749895 | 0.073 |
| Random5K | Full | 625 | 0.941 | 0.989 | 0.992 | 0.995 | 0.964586 | 0.005 |
| Random5K | Initial | 625 | 0.652 | 0.828 | 0.888 | 0.934 | 0.752552 | 0.066 |
| Recent5K | Full | 625 | 0.949 | 0.989 | 0.990 | 0.994 | 0.968512 | 0.006 |
| Recent5K | Initial | 625 | 0.659 | 0.841 | 0.882 | 0.933 | 0.756531 | 0.067 |
| Recent5K-long | Full | 6991 | 0.868 | 0.954 | 0.962 | 0.978 | 0.911956 | 0.022 |
| Recent5K-long | Initial | 6991 | 0.567 | 0.750 | 0.806 | 0.855 | 0.668300 | 0.145 |
| Full55925 | Full | 6991 | 0.941 | 0.987 | 0.990 | 0.993 | 0.963769 | 0.007 |
| Full55925 | Initial | 6991 | 0.630 | 0.771 | 0.850 | 0.917 | 0.719298 | 0.083 |

---

## 13. Master A comparisons

### 13.1 Temporal position at matched budget

All use exactly:

- 5,000 unique compatible rows;
- 625 optimizer steps.

Recent minus Oldest:

Full:

- Top1: +0.4 pp
- Top3: +0.2 pp
- MRR: +0.002642
- Missing: -0.1 pp

Initial:

- Top1: +0.7 pp
- Top3: +0.4 pp
- MRR: +0.006636
- Missing: -0.6 pp

Recent minus Random:

Full:

- Top1: +0.8 pp
- Top3: 0.0 pp
- MRR: +0.003926
- Missing: +0.1 pp

Initial:

- Top1: +0.7 pp
- Top3: +1.3 pp
- MRR: +0.003979
- Missing: +0.1 pp

Conclusion:

Recent history has a consistent but modest matched-budget advantage.

---

### 13.2 Optimization control

Identical Recent5K population:

`625 steps -> 6991 steps`

Full:

- Top1: `0.949 -> 0.868` = **-8.1 pp**
- Top3: `0.989 -> 0.954` = -3.5 pp
- MRR: -0.056556
- Missing: `0.006 -> 0.022` = +1.6 pp

Initial:

- Top1: `0.659 -> 0.567` = **-9.2 pp**
- Top3: `0.841 -> 0.750` = -9.1 pp
- MRR: -0.088231
- Missing: `0.067 -> 0.145` = +7.8 pp

Recent5K-long:

- exactly 6,991 optimizer steps;
- 55,928 row exposures;
- final reported batch loss: 0.0145754.

This is strong evidence of repeated-population overfitting.

---

### 13.3 Matched-step history/diversity control

Both conditions use 6,991 optimizer steps.

Recent5K-long:

- 5,000 unique rows repeatedly revisited;
- 55,928 row exposures.

Full55925:

- 55,925 unique compatible rows;
- one epoch;
- 6,991 optimizer steps;
- final batch contains 5 rows.

Thus the optimizer-step comparison differs by only three total row exposures
(`55,928` versus `55,925`) while changing training-set diversity dramatically.

Full55925 minus Recent5K-long:

Full:

- Top1: +7.3 pp
- Top3: +3.3 pp
- MRR: +0.051813
- Missing: -1.5 pp

Initial:

- Top1: +6.3 pp
- Top3: +2.1 pp
- MRR: +0.050998
- Missing: -6.2 pp

Conclusion:

6,991 steps are not intrinsically harmful. Severe degradation occurs when the
same small population is repeatedly revisited. Training-set diversity strongly
buffers the optimization budget.

---

## 14. Revised interpretation

The original history-size curve must not be interpreted as evidence that old
history is intrinsically harmful.

Current evidence supports:

1. recency has a modest matched-budget benefit;
2. repeated optimization exposure has a much larger effect;
3. small-population repetition can strongly overfit;
4. diverse lifetime history can sustain many more optimizer steps;
5. Oldest5K itself remains strong at 94.5% Full Top1.

Therefore:

> Under a matched 5,000-example / 625-step budget, recent history gives a modest
> advantage over oldest and random history. However, repeated optimization on
> the same Recent5K population to 6,991 steps causes severe generalization
> degradation, while training for the same 6,991 steps over the much more
> diverse Full55925 history remains strong. The previously observed Recent5K
> advantage is therefore not attributable to recency alone; optimization
> exposure and training-set diversity are major confounding factors.

---

## 15. Master A provenance

Runner:

`experiments/model_level/run_long_term_master_a_v1.sbatch`

Manifest training wrapper:

`experiments/model_level/run_adapter_training_manifest_v1.py`

Slurm job:

`634528`

Status:

`COMPLETED`

Exit code:

`0:0`

Elapsed:

`01:30:45`

Summary:

`results/model_level/long_term_qualification_v1/master_a_v1/master_a_summary.json`

CSV:

`results/model_level/long_term_qualification_v1/master_a_v1/master_a_summary.csv`

Important commits:

- `2c23bb5` — tokenizer-compatible training-target filter
- `3ba4fdc` — isolated concurrent overnight GPU steps
- `dddaa04` — overnight evaluation finalizer
- `5b0f19e` — long-term Master A controls
- `18bf358` — unsupported-Pinyin evaluation preservation

---

## 16. Next research phase

Static Adapter qualification is complete.

The next phase uses the chronological combined Agent development stream:

- Train-Fit + Train-Val;
- Test remains sealed;
- Train-Val ceases to be untouched validation once included in longitudinal
  development training.

Current combined Agent stream:

- nominal rows: 69,667;
- tokenizer-compatible effective rows: 69,664;
- works: 45;
- work indices: 0 through 44.

The next experiment family is:

1. freeze chronological episode/probe manifests;
2. prequential test-before-train evaluation;
3. warm continual Adapter updates;
4. adaptation / rescue / harm;
5. retention / forgetting;
6. generic-preservation / locality;
7. Warm versus Rebuild controls;
8. replay only if meaningful forgetting is observed;
9. controlled Temporary versus Persistent stress test.

Test remains outside this research line.
