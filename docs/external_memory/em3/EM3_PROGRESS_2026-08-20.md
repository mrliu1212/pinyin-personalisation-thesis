# EM3 Progress Note — 2026-08-20

## Current status

EM3-BCE v1 has been trained and evaluated on the three-author Dev tune surface.

Current three-author set:
- Etinjat
- Re_spectators
- breaddddd

The current EM3-BCE v1 pointwise Cross-Encoder was trained from `BAAI/bge-reranker-base` using causal same-Pinyin positive / hard-negative history pairs.

Training result:
- train pairs: 233,154
- positive pairs: 86,959
- negative pairs: 146,195
- epochs: 1
- final mean training loss: 0.543399
- final model:
  `results/personalisation/external_memory/em3_cross_encoder_v1/train/final`

## Existing Dev results

### Frozen candidate-level baselines

Macro-author Top1 on the fixed 5,608-query three-author Dev tune surface:

| Method | Overall | History | Ambiguous | Conflict |
|---|---:|---:|---:|---:|
| Generic | 0.722948 | 0.759399 | 0.683605 | 0.467924 |
| F | 0.765240 | 0.823281 | 0.754831 | 0.197192 |
| Original-M1 | 0.768888 | 0.828351 | 0.769454 | 0.297163 |
| Hidden-M1 | 0.768748 | 0.828688 | 0.766692 | 0.299645 |
| Original-M2 | 0.766869 | 0.825123 | 0.763249 | 0.258543 |

### EM3-BCE history discrimination

On the 1,609 pair-trainable Dev-tune queries:

- Macro-author Top1: 0.736181
- Macro-author MRR: 0.815373
- Mean margin: 2.622738
- Median margin: 1.285817
- PositiveBeatsNegative: 0.735853

Micro:
- Top1: 0.722809
- MRR: 0.809840
- Mean margin: 1.498866
- Median margin: 0.825073
- PositiveBeatsNegative: 0.722188

Important diagnostic:
- `G wrong, F right`: EM3 history Top1 = 0.761364
- `G right, F wrong`: EM3 history Top1 = 0.305556
- formal Conflict: EM3 history Top1 = 0.314763

Interpretation:
EM3 learns meaningful personal-memory evidence, but performs poorly when historical preference conflicts with the current contextual signal. The current pointwise BCE baseline appears to over-trust historical evidence in difficult conflict cases.

### EM3 on Hidden retrieval surface

Frozen PinyinGPT hidden-state retrieval + EM3-BCE Cross-Encoder + M2-style target aggregation/fusion:

Selected on Dev:
- K = 10
- lambda = 4.0

Macro-author Top1:

| Method | Overall | History | Ambiguous | Conflict |
|---|---:|---:|---:|---:|
| Generic | 0.722948 | 0.759399 | 0.683605 | 0.467924 |
| F | 0.765240 | 0.823281 | 0.754831 | 0.197192 |
| EM3-Hidden | 0.767633 | 0.827217 | 0.761592 | 0.248205 |

Conclusion:
EM3-Hidden is only slightly better than F overall (+0.2393 percentage points) and does not materially exceed the existing M1 / Hidden-M1 results. Conflict performance remains weak.

## Current interpretation

The current EM3-BCE v1 result is not strong enough to claim a clear end-to-end improvement.

Two likely issues should be treated separately:

1. **Data / author quality**
   - The current three-author experimental set is uneven.
   - `Re_spectators` has a very small EM3 pair-trainable Dev population (41 queries in the earlier aligned evaluation), so its author-level metric is volatile.
   - The dataset also contains script / encoding / reconstruction noise discovered in earlier stages.
   - Therefore, the next experiment should replace the weak / low-support author and rebuild a cleaner three-author comparison set before drawing strong conclusions.

2. **Learning objective**
   - Pointwise BCE learns whether a history interaction looks individually useful.
   - It does not directly optimize relative ordering between positive and hard-negative histories.
   - The strongest failure cases are conflict cases, suggesting that the model still needs better context-sensitive negative supervision / ranking objectives.

## Decision

EM3-BCE v1 is retained as the **pointwise BCE baseline**.

Do not spend more time trying to rescue this exact run by tuning many checkpoints or repeatedly adjusting lambda on the same Dev surface.

Proceed to EM3-v2.

## Files / outputs to preserve

### EM3 training
- `experiments/external_memory/em3_train_cross_encoder.py`
- `results/personalisation/external_memory/em3_train_pairs_v1/train_pairs.jsonl`
- `results/personalisation/external_memory/em3_cross_encoder_v1/train/final`

### EM3 evaluation
- `experiments/external_memory/em3_eval_final_ranking_cached.py`
- `results/personalisation/external_memory/em3_bce_v1_final_dev_tune/scores.jsonl`
- `results/personalisation/external_memory/em3_bce_v1_final_dev_tune/summary.json`

### Candidate-level comparison
- `results/personalisation/external_memory/em2_four_way_dev_compare/rows.jsonl`
- `results/personalisation/external_memory/em2_original_m2_dev/rows.jsonl`
- `results/personalisation/external_memory/em3_bce_v1_final_compare/summary.json`
- `results/personalisation/external_memory/em3_bce_v1_final_compare/rows_1609.jsonl`

### Hidden-surface EM3
- `experiments/external_memory/em3_hidden_surface_dev.py`
- `results/personalisation/external_memory/em3_hidden_dev/em3_pair_scores.jsonl`
- `results/personalisation/external_memory/em3_hidden_dev/summary.json`
- `results/personalisation/external_memory/em3_hidden_dev/grid.json`
- `results/personalisation/external_memory/em3_hidden_dev/selected_rows.jsonl`

## Test policy

Test has not been used for EM3 selection.
Keep Test untouched until the next method and comparison protocol are frozen.

## Final 2026-08-20 Dev checkpoint

EM3 Dev analysis progressed from the preliminary 124/59 failure slices to a
complete 5,608-row G/F/Hidden-M1 outcome audit on the old exploratory
Full+Short/H5000 three-author surface. The eight correctness groups are:

`3361, 24, 42, 100, 403, 35, 45, 1598`

in `G✓F✓H✓, G✓F✓H✗, G✓F✗H✓, G✓F✗H✗, G✗F✓H✓, G✗F✓H✗,
G✗F✗H✓, G✗F✗H✗` order. Relative to F, Context rescues 87 rows, harms
59, and nets `+28` diagnostic Micro-correct rows.

The 1,598 all-wrong rows contain 1,275 without a Gold same-Pinyin precedent and
323 with Gold in history. Gold is also in the candidate set for 53.9% of the
323, implying approximately 174 ranking/fusion opportunities and 149
candidate-recovery failures. The 45 Context-only rescues demonstrate that
Context can correctly choose a minority historical target; the 24 pure Context
regressions and 35 lost F rescues show the need for calibrated suppression.

Current working direction, not a frozen method: replace raw user frequency with
user-vs-global frequency lift and learn candidate-specific historical utility.
Do not reduce Context to a single global frequency weight. The main clean3
development authors remain Agent Phage, Etinjat, and breaddddd.

The formal pair generator now exists at
`experiments/external_memory/em3_generate_train_pairs.py` and exactly
regression-reproduces the frozen old-three-author counts
`30,968 / 86,959 / 146,195 / 233,154` in audit-only mode, with no Test,
non-prior pairs, or reused query-history rows.

Main work is paused before new heavy training. On resume, first generate and
audit clean3 pairs, then freeze the prediction-visible PersonalLift and
candidate-utility definitions and run a small Dev-only experiment. See:

- `EM3_ALL_OUTCOME_DISTRIBUTION_RECORD_2026-08-20.md` for complete tables;
- `EM3_V2_FAILURE_AUDIT_2026-08-20.md` for the final taxonomy;
- `EM3_DEV_CLOSEOUT_2026-08-20.md` for provenance and exact resume order.
