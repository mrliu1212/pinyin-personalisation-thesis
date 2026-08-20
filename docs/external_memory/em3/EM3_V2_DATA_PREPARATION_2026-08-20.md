# EM3-v2 Data Preparation — 2026-08-20

## Status

This document freezes the data-preparation decision for the next EM3 experiment.

No Test data is used.

## 1. Clean author set

The EM3-v2 main development population is frozen as:

- Agent Phage
- Etinjat
- breaddddd

This replaces the exploratory EM3-v1 population:

- Etinjat
- Re_spectators
- breaddddd

## 2. Why the author set changed

The change is based on the existing six-author Train/Dev population audits, not on selecting authors by model accuracy.

### Main reason

`Re_spectators` has substantially less useful EM3 supervision and Dev support than the stronger available authors.

Dev pair-trainable queries:

- Agent Phage: 2,851
- breaddddd: 2,003
- Etinjat: 1,326
- QBLevi: 544
- Re_spectators: 247
- MScarlet: 2,108

The previous EM3 tune-aligned evaluation was even more unstable for `Re_spectators`, where only 41 pair-trainable tune queries were available.

Because the primary evaluation metric is Macro-author Top1, each author receives equal weight. A very small author-level evaluation population can therefore create large variance.

### Why Agent Phage is added

Agent Phage provides strong Train and Dev support:

Train:
- all rows: 69,667
- pair-trainable: 15,811
- ambiguous: 16,136
- conflict: 2,764

Dev:
- all rows: 10,110
- pair-trainable: 2,851
- ambiguous: 2,931
- conflict: 439

This makes Agent Phage substantially more useful for a stable author-balanced experiment.

### Why MScarlet is not used in the clean3 main set

MScarlet has strong numerical support, but earlier project work identified a script-normalisation / script-alignment confound.

Therefore MScarlet is not used in the clean3 main experiment until that issue is formally repaired.

MScarlet can remain useful later as:
- an ablation;
- a repaired-script experiment;
- or an additional robustness author.

### Why QBLevi is not selected

QBLevi is not known to have the same script confound, but its usable EM3 support is much smaller than Agent Phage.

Dev pair-trainable:
- QBLevi: 544
- Agent Phage: 2,851

Therefore Agent Phage is the stronger replacement for the main clean3 experiment.

## 3. Population comparison

### Old exploratory three-author set

Authors:
- Etinjat
- Re_spectators
- breaddddd

Train:
- all rows: 120,272
- pair-trainable: 30,968
- ambiguous: 32,861
- conflict: 6,178

Dev:
- all rows: 13,895
- pair-trainable: 3,576
- ambiguous: 3,892
- conflict: 840

### New clean3 set

Authors:
- Agent Phage
- Etinjat
- breaddddd

Train:
- all rows: 178,942
- pair-trainable: 44,788
- ambiguous: 46,961
- conflict: 8,596

Dev:
- all rows: 22,723
- pair-trainable: 6,180
- ambiguous: 6,570
- conflict: 1,235

### Increase from old set to clean3

- Train rows: +48.8%
- Train pair-trainable: +44.6%
- Train ambiguous: +42.9%
- Train conflict: +39.1%
- Dev rows: +63.5%
- Dev pair-trainable: +72.8%
- Dev ambiguous: +68.8%
- Dev conflict: +47.0%

## 4. Frozen causal semantics

The following semantics remain unchanged:

- same author only;
- strictly-prior interactions only;
- H5000 history budget;
- history budget applied before exact segmented-Pinyin filtering;
- exact segmented-Pinyin matching for EM3 supervision;
- current interaction becomes history only after it is processed;
- earlier Dev interactions may become history for later Dev queries;
- no future history;
- no Test.

## 5. EM3-v1 baseline policy

The existing EM3-BCE v1 model remains the exploratory pointwise BCE baseline.

However, because the author set changes, a fair clean3 comparison requires a new clean3 BCE baseline trained on:

- Agent Phage
- Etinjat
- breaddddd

The old BCE model must not be compared directly against a new clean3-trained EM3-v2 method as the sole supervised baseline.

## 6. EM3-v2 training-data direction

The next dataset should support ranking-oriented supervision.

The key requirement is to learn relative preference:

`score(query, positive_history) > score(query, hard_negative_history)`

rather than only independent pointwise labels.

### Positive

Strictly-prior same-Pinyin history interaction whose target equals the current Gold target.

### Hard negative

Strictly-prior same-Pinyin history interaction whose target differs from the current Gold target.

### Hard-negative priority

The next generator should preserve enough metadata to distinguish at least:

1. ordinary same-Pinyin wrong-target negatives;
2. high-frequency wrong-target negatives;
3. retrieval-similar wrong-target negatives;
4. conflict-like negatives where personal history is strong but current contextual evidence should resist it.

The exact v2 loss is not frozen yet.

Possible methods already documented in `EM3_V2_METHOD_OPTIONS_2026-08-20.md` include ranking-oriented objectives such as listwise / pairwise alternatives.

## 7. Evaluation requirements

Future clean3 evaluation should preserve row-level outputs.

Primary:
- Macro-author Top1

Secondary:
- Micro Top1
- Ambiguous Top1
- Conflict Top1
- candidate-level MRR when consistently available

Paired diagnostics:
- rescue
- harm
- net rescue
- Generic wrong / Personal right
- Generic right / Personal wrong
- method disagreement subsets

Statistical comparison:
- paired bootstrap 95% CI
- McNemar test for paired Top1 correctness

## 8. Interaction with Codex research

Codex is independently investigating:
- Full vs Initial;
- Short vs Multi3;
- condition-specific personalisation headroom;
- cheap gating / fusion alternatives;
- other performance-improving interventions.

This EM3-v2 data-preparation track must not duplicate those experiments.

The clean3 data and evaluation framework should be prepared first.

The final EM3-v2 condition should be chosen after the Codex headroom results are available.

## 9. Next actions

Checkpoint update: the canonical generator now exists at
`experiments/external_memory/em3_generate_train_pairs.py`. Its audit-only run
regression-reproduces the old three-author counts exactly: 30,968 eligible
queries, 86,959 positive pairs, 146,195 negative pairs, and 233,154 total, with
zero non-prior pairs or query-history reuse and no Test.

1. Generate the clean3 manifest with the canonical generator and freeze:
   - Agent Phage
   - Etinjat
   - breaddddd
2. Preserve source, runner, output, and audit hashes.
3. Regenerate the clean3 pointwise BCE baseline manifest.
4. Generate a clean3 ranking-oriented EM3-v2 manifest with explicit positive / hard-negative grouping.
5. Audit:
   - chronology;
   - same-author constraint;
   - H5000-before-Pinyin semantics;
   - no duplicate query-history pair;
   - no Test;
   - per-author counts;
   - positive / negative / conflict support.
6. Persist all summaries and row-level manifests.
7. Do not start expensive training until the data audit passes and the Codex condition-headroom result is available.
