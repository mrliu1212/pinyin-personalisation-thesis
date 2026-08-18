# Diagnostic A1 — Short Retrieval Results

**Date:** 2026-08-18  
**Status:** Completed exploratory diagnostic  
**History budget:** H5000  
**Exploratory authors:** Etinjat, Re_spectators, breaddddd

## Purpose

Diagnostic A1 asks where the contextual-personalisation pipeline first fails:

1. useful Gold-target history does not exist;
2. Gold history exists but BGE does not retrieve it;
3. Gold history is retrieved, so the remaining failure must occur downstream.

Recall is measured only among rows where legal strictly-prior H5000 history contains the Gold target.

## Full + Short

Test rows: 3,000

- History Available: 1,891 / 3,000 = 63.03%
- Gold-target history exists: 1,698 / 3,000 = 56.60%
- Recall@1: 85.22%
- Recall@3: 93.70%
- Recall@5: 96.17%
- Recall@10: 98.53%
- Recall@20: 99.41%

### Ambiguous

- rows: 836
- Gold history exists: 763
- Recall@1: 67.10%
- Recall@10: 96.72%
- Recall@20: 98.69%

### Conflict

- rows: 233
- Gold history exists: 173
- Recall@1: 24.28%
- Recall@3: 52.60%
- Recall@5: 69.36%
- Recall@10: 86.13%
- Recall@20: 94.80%

Interpretation:

For Full + Short, useful Gold history is usually present in the upper BGE retrieval set once it exists. Therefore the remaining investigation should focus on how retrieved contextual evidence is converted into candidate scores and final decisions.

## Initial + Short

Test rows: 3,000

- History Available: 2,796 / 3,000 = 93.20%
- Gold-target history exists: 1,698 / 3,000 = 56.60%
- Recall@1: 39.22%
- Recall@3: 59.72%
- Recall@5: 67.96%
- Recall@10: 79.15%
- Recall@20: 86.63%

### Ambiguous

- rows: 2,618
- Gold history exists: 1,625
- Recall@1: 36.49%
- Recall@10: 78.22%
- Recall@20: 86.03%

### Conflict

- rows: 1,438
- Gold history exists: 742
- Recall@1: 14.69%
- Recall@3: 31.27%
- Recall@5: 42.45%
- Recall@10: 60.51%
- Recall@20: 75.88%

Interpretation:

Initial input exposes much more competing history while Gold-history coverage remains unchanged. This substantially reduces retrieval precision and motivates a separate Initial-to-Full-Pinyin diagnostic rather than immediately applying the Full + Short A2 strategy.

## Current Decision

Full + Short:
- proceed to A2 scoring / decision diagnosis.

Initial + Short:
- retain A1 results;
- prioritise Initial → Full-Pinyin expansion and Personal Vocabulary diagnostics.

Multi3:
- preserve A1 coverage observations;
- defer active method development;
- investigate a dedicated long-composition mechanism later.

## A2 Question

A2 will not recompute ordinary model accuracy.

Existing F/M1/M2 prediction transitions already establish that contextual methods can both rescue Frequency errors and introduce new errors.

A2 instead asks why each transition occurs by joining:

- Gold;
- Generic prediction;
- Frequency prediction;
- M1 prediction;
- M2 prediction;
- BGE Gold-history retrieval rank;
- history ambiguity/conflict information.

The main categories are:

1. F wrong → Context correct: rescue;
2. F correct → Context wrong: harm;
3. Gold retrieved but Context still wrong: evidence available but unused;
4. Gold not retrieved: retrieval failure;
5. Generic correct → F wrong → Context correct: protection from Frequency harm;
6. Generic wrong + F wrong → Context correct: unique contextual rescue.

All causal explanations beyond the observed transition/retrieval statistics remain hypotheses until A2 is completed.
