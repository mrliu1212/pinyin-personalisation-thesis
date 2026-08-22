# Standardized Context-Model Comparison — Completion Report

Date: 2026-08-22
Worktree: `C:\Users\chiar\Desktop\LBH\thesis-context-compare`
Branch: `work/context-model-comparison`
Status: **standardized Train-Val selection and sealed Dev3000 evaluation complete**
Test used: **false**

## 1. Purpose

This work established a controlled horizontal comparison of seven Full+Short
Pinyin-input ranking systems under one shared candidate surface, history policy,
selection population, and evaluation implementation:

1. Generic PinyinGPT;
2. personal Frequency reranking;
3. M1 with BGE retrieval;
4. M2 with BGE retrieval and generic cross-encoder reranking;
5. Hidden-M1 with PinyinGPT hidden-state retrieval;
6. Hidden-M2 with hidden-state retrieval and generic cross-encoder reranking;
7. EM3-Clean3 with hidden-state retrieval and a Clean3-trained personal
   cross-encoder.

The comparison was designed to isolate the effect of contextual personalisation.
Every method reranked the same frozen Generic Top-10 candidates. No method was
allowed to inject, remove, or fuse candidates.

## 2. Frozen protocol

### 2.1 Data sequence

The experimental order was:

`Clean3 Train -> Train-Fit/Train-Val -> training and retuning -> PRE_DEV_FREEZE -> sealed Dev3000`

The frozen populations were:

| Population | Rows | SHA256 |
|---|---:|---|
| Train-Fit | 144,526 | `547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6` |
| Train-Val | 34,416 | `d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220` |
| Dev3000 | 3,000 | `9181f895eb19d0c36852e511263bfaefb34459dcd44efa6f45a44252e6b03f93` |

Dev3000 contains 1,000 interactions each for Agent Phage, Etinjat, and
breaddddd. It was not used for neural training, checkpoint selection,
hyperparameter selection, method selection, or threshold selection.

### 2.2 Causal history

All personal systems used the same history semantics:

`same author -> strictly prior -> latest up-to-5000 raw interactions -> exact segmented-Pinyin match`

Train-Val histories could contain Train-Fit and earlier Train-Val interactions.
Dev3000 used rolling causal online history, so an earlier Dev interaction could
become history for a later Dev interaction. Future interactions were never
admitted.

H5000 denotes 5,000 prior raw interactions, not 5,000 matching memories. Exact
Pinyin filtering occurs only after the bounded causal history has been formed.

### 2.3 Frozen Generic surface

The Generic backend remained:

- checkpoint: `aihijo/transformers4ime-pinyingpt-concat`;
- revision: `76dd20dc92d8236a350fb732e99dde6fa15e2263`;
- beam size: 16;
- candidate surface: Top-10;
- production-compatible maximum-position policy: `n_positions=1024`.

All compared systems evaluated exactly this surface.

## 3. Engineering completed

The comparison added a versioned, resumable implementation for:

- chronological whole-work Train-Fit/Train-Val construction;
- causal H5000 history resolution;
- exact frozen Generic generation with shape-compatible batching;
- PinyinGPT hidden-state extraction;
- BGE cache reuse and miss-only generation;
- deterministic Stage-1 retrieval;
- exact pair-request registries;
- resumable Generic and EM3 pair-score SQLite caches;
- Train-Val grid evaluation and deterministic tie-breaking;
- machine-readable model/configuration freezes;
- freeze-gated Dev3000 preparation and evaluation;
- checksummed machine and human result finalization.

A narrow guard was also added to `src/personalisation/context_memory.py` for an
empty candidate surface. It does not alter non-empty ranking behavior.

The implementation preserved completed artifacts throughout. No standardized
reset, completed inference stage, or EM3 training run was restarted.

## 4. Reuse and new computation

### 4.1 Reused or already complete

- Generic Train-Val predictions: 34,416/34,416;
- hidden representations: 42,454/42,454;
- BGE representations: 39,993 exact cache hits;
- newly needed BGE representations already completed: 2,365;
- EM3 Train-Fit pair dataset: 269,071 pairs with causal audit passed;
- EM3 training: 8,409/8,409 optimizer steps;
- Frequency and M1 Train-Val searches;
- Stage-1 registries and the exact 381,295-pair Train-Val registry;
- Generic cross-encoder scores: 381,295/381,295.

### 4.2 Completed in the final continuation

- EM3 pair scoring: 302,649/302,649;
- remaining M2, Hidden-M2, and EM3 Train-Val grids;
- final selection for all seven systems;
- machine-readable and human-readable PRE_DEV_FREEZE artifacts;
- one sealed standardized Dev3000 evaluation.

The Dev Generic cache contained 1,568 exact reusable rows. Only the 1,432
missing rows were generated after the freeze. Dev representation work reused
11,124 of 16,768 unique BGE contexts and 6,166 of 16,779 hidden states. It
generated only the 5,644 BGE and 10,613 hidden-state misses.

For Dev pair scoring, 17,419 exact Generic scores were reusable. The run scored
only 25,217 new Generic pairs and the 33,567 required EM3 pairs.

## 5. Train-Val selections

Selection used Macro-author Top1 on Train-Val. Ties followed the predeclared
lower-lambda, then lower-K/Top-N, then canonical-order rule.

| Method | Frozen configuration |
|---|---|
| Generic | revision `76dd20dc...e2263`, beam 16, Top-10, `n_positions=1024` |
| Frequency | lambda `4.0` |
| M1 | BGE Full cosine, Top-N `5`, lambda `4.0` |
| M2 | BGE Stage-1 K `10`, lambda `4.0` |
| Hidden-M1 | PinyinGPT hidden cosine, Top-N `5`, lambda `4.0` |
| Hidden-M2 | hidden Stage-1 K `10`, lambda `4.0` |
| EM3-Clean3 | hidden Stage-1 K `10`, lambda `4.0`, final 8,409-step checkpoint |

The selected Train-Val Macro-author Top1 values for the three pair-scored
systems were:

- M2: `0.7769184347`;
- Hidden-M2: `0.7762991774`;
- EM3: `0.7771533900`.

The principal model identities were:

- BGE model SHA256:
  `5a88d266870fbd27c6f329df60de80e2d4cf3bbd5e6f080bd5c1b2e5abb12039`;
- generic cross-encoder revision:
  `2cfc18c9415c912f9d8155881c133215df768a70`;
- generic cross-encoder model SHA256:
  `ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd`;
- EM3 `model.safetensors` SHA256:
  `0e846deeeaf06c3e5c61dc39bfae1c1f986d37a82668146dd030a1ccca793dfa`;
- EM3 tokenizer SHA256:
  `4a8d0b7573869188be52cca17a27a84f3cfbc0a5536c28ee1eca82903e8c68c6`.

## 6. Sealed Dev3000 results

Because Dev3000 is exactly balanced by author, Macro-author and Micro Top1 are
equal in this evaluation.

| Method | Macro Top1 | Micro Top1 | Micro Top3 | Micro MRR@10 | Micro Missing@10 |
|---|---:|---:|---:|---:|---:|
| Generic | 72.267% | 72.267% | 87.667% | 0.806844 | 5.133% |
| Frequency | 82.500% | 82.500% | 91.533% | 0.872403 | 5.133% |
| M1 | **82.833%** | **82.833%** | 91.333% | **0.874323** | 5.133% |
| M2 | 82.433% | 82.433% | 91.333% | 0.872074 | 5.133% |
| Hidden-M1 | 82.633% | 82.633% | 91.200% | 0.872864 | 5.133% |
| Hidden-M2 | 82.367% | 82.367% | 91.300% | 0.871268 | 5.133% |
| EM3 | 82.733% | 82.733% | 91.200% | 0.873047 | 5.133% |

### 6.1 Per-author Top1

| Method | Agent Phage | Etinjat | breaddddd |
|---|---:|---:|---:|
| Generic | 76.200% | 56.800% | 83.800% |
| Frequency | 89.600% | 65.200% | 92.700% |
| M1 | **90.300%** | 65.300% | **92.900%** |
| M2 | 89.600% | 65.000% | 92.700% |
| Hidden-M1 | 90.100% | 64.900% | **92.900%** |
| Hidden-M2 | 89.800% | 64.500% | 92.800% |
| EM3 | 89.800% | **65.500%** | **92.900%** |

### 6.2 Ambiguous and Conflict diagnostics

The Ambiguous subset contains 1,330 rows. The formal Conflict subset contains
310 rows and follows the repository rule: an ambiguous Pinyin has a unique
frequency winner, the gold differs from that winner, and frequency ties are
excluded.

| Method | Ambiguous Macro Top1 | Conflict Macro Top1 |
|---|---:|---:|
| Generic | 63.736% | **28.062%** |
| Frequency | 77.537% | 14.425% |
| M1 | **78.412%** | 24.756% |
| M2 | 77.433% | 17.887% |
| Hidden-M1 | 78.033% | 22.234% |
| Hidden-M2 | 77.457% | 19.395% |
| EM3 | 78.087% | 18.035% |

### 6.3 Paired rescue and harm

| Method | vs Frequency rescue/harm/net | vs Generic rescue/harm/net |
|---|---|---|
| M1 | 40 / 30 / **+10** | 396 / 79 / **+317** |
| M2 | 39 / 41 / **-2** | 388 / 83 / **+305** |
| Hidden-M1 | 38 / 34 / **+4** | 391 / 80 / **+311** |
| Hidden-M2 | 34 / 38 / **-4** | 382 / 79 / **+303** |
| EM3 | 34 / 27 / **+7** | 389 / 75 / **+314** |

## 7. Interpretation

All personal systems substantially improved Top1 over Generic on this Dev set.
Frequency supplied most of the gain: its Top1 was 10.233 percentage points
above Generic. The more complex contextual systems clustered within 0.433
percentage points of Frequency.

M1 produced the highest overall Dev Top1 and MRR@10. It also had the largest
positive paired net relative to Frequency. EM3 ranked second in overall Top1
and had a positive paired net of seven relative to Frequency. M2 and Hidden-M2
had small negative paired nets relative to Frequency despite remaining far
above Generic.

The Conflict subset exposes the central difficulty more clearly. Frequency
performed poorly when the gold target differed from the unique personal
frequency winner. M1 recovered part of this loss and was the strongest
personal method on Conflict, but Generic remained highest on that subset.
Thus, the results support contextual reranking as a useful correction to the
Generic baseline, while also showing that the present contextual models do not
consistently override misleading frequency evidence.

These are descriptive Dev results. No statistical significance claim is made,
and no method may be changed in response to these results before the frozen
Test run.

## 8. Runtime and integrity

- Hardware: NVIDIA GeForce RTX 4060 Laptop GPU;
- PyTorch: `2.11.0+cu128`;
- CUDA runtime: 12.8;
- EM3 training: 8,409 optimizer steps, 7,103.38 seconds;
- EM3 mean training loss: `0.5595846`;
- Dev Generic miss-only generation: 1,432 rows in 42.68 seconds;
- Train-Val Generic pair scoring final rate: 123.72 pairs/second;
- Train-Val EM3 pair scoring final rate: 123.23 pairs/second;
- Dev BGE miss generation final rate: 200.35 contexts/second;
- Dev hidden-state miss generation final rate: 43.94 rows/second;
- Dev Generic pair scoring final rate: 124.81 pairs/second;
- Dev EM3 pair scoring final rate: 128.04 pairs/second.

SQLite `PRAGMA integrity_check` returned `ok` for every Train-Val and Dev pair
registry and score cache. Exact final counts were:

| Database | Rows |
|---|---:|
| Train-Val pair registry | 381,295 |
| Train-Val Generic scores | 381,295 |
| Train-Val EM3 scores | 302,649 |
| Dev pair registry | 42,636 |
| Dev Generic scores | 42,636 |
| Dev EM3 scores | 33,567 |

## 9. Reproducibility artifacts

| Artifact | SHA256 |
|---|---|
| `results/personalisation/context_comparison_v2/pre_dev_freeze_v1.json` | `7c0fcf69823f0b4b7d8b914a81ea54a097e12c03cb61c515c2400be46df46824` |
| `docs/context_comparison/13_PRE_DEV_FREEZE_2026-08-21.md` | `80146cd29ef92b2aa6243cf71e9a7ebeccf17a2446abec4331763985a1297d8c` |
| `results/personalisation/context_comparison_v2/dev3000/standardized_dev3000_result.json` | `99e1b6960f96c39f107916873f80d5461d143be529f3970d275fded7ef9ab35f` |
| `results/personalisation/context_comparison_v2/dev3000/predictions.jsonl` | `dd219bfcb28fcad6a65f31eb14ddb16fc03c80f54a8b62a1cfe2504113c84233` |
| `results/personalisation/context_comparison_v2/dev3000/checksums.json` | `61962570748e493eaf21de975cea853c058ba3ce552c22c0bc23334ea78104c8` |
| `docs/context_comparison/14_STANDARDIZED_DEV3000_RESULT_2026-08-21.md` | `3fb4592b0f2f8d2865c2ddec61b3dd3eabc64e999be0f204cbb2e55885fb2d39` |

Both the pre-Dev freeze and final result explicitly record `used_test=false`.
The freeze also records `dev3000_used_for_selection=false`.

## 10. Validation

- focused context-comparison tests: 22 passed;
- complete `tests/` suite: 128 passed, 6 skipped;
- Python compilation: passed;
- `git diff --check`: passed.

A root-level `pytest` invocation also collected
`experiments/context_lab/ctx64_m1_test.py`, which raises during import when its
unrelated generated `ctx64` embedding cache is absent. The actual `tests/`
suite completed successfully without creating or copying that unrelated cache.

## 11. Scientific boundary and remaining work

The standardized comparison is method-frozen after observation of Dev3000.
Train-Val selection must not be reopened and the current method must not be
changed based on these Dev results.

The only remaining experimental step is the separately authorized, one-time
frozen Test evaluation. Until that authorization, Test remains closed. Repository
review, archival, and an explicit commit may be performed independently, but
no commit, push, tag, reset, clean, or staging action was performed as part of
this comparison completion.
