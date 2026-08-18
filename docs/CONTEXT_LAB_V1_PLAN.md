# Context Personalisation Lab V1 Plan

## Purpose

This branch is an exploratory diagnostic and method-development branch for understanding why contextual personalisation has not consistently outperformed the frequency baseline, and for testing structured alternatives before further expensive model development.

## Current Priority Override ? Short Focus

As of 2026-08-18, active Context Lab method development is restricted to:

- Full + Short
- Initial + Short

Multi3 remains preserved but is deferred from active method development after Diagnostic A1 showed extremely low exact-history coverage under the current H5000 memory definition.

See:

`docs/context_lab/SHORT_FOCUS_DECISION_2026-08-18.md`

This is a prioritisation decision, not a claim that Multi3 is impossible to personalise and not a deletion of the original four-condition research scope.

---

## Baseline

Base commit:

`617a20f`

Base tag:

`personalisation-v1-long-context-matrix`

The paused Multi3 128-context implementation is preserved separately at:

`multi3-128-context-wip-20260818`

and is not part of this lab baseline.

---

## Fixed Exploratory Scope

### History Budget

H5000 only.

History semantics remain:

- same user;
- strictly prior interactions only;
- history budget applied before Pinyin filtering;
- no future information;
- no Test Gold used for personal vocabulary construction or parameter tuning.

### Exploratory Authors

The Phase 1 exploratory subset is fixed to:

- Etinjat
- Re_spectators
- breaddddd

These authors are used only for exploratory diagnostics and method development.

MScarlet is temporarily excluded from Phase 1 because a dataset-level script audit identified a strong author-specific Chinese-script confound:

- MScarlet Gold t2s-change rate: 57.6702%;
- MScarlet context t2s-change rate: 98.6360%;
- the other five authors have much lower Gold t2s-change rates (0.0107%?1.7771%).

This exclusion is based on preprocessing properties of the dataset, not on Test accuracy or method performance.

Final formal evaluation must return to all six authors after the script-normalisation issue is resolved.

### Conditions

The four existing evaluation conditions are:

- Full + Short
- Initial + Short
- Full + Multi3
- Initial + Multi3

Not every diagnostic needs all four conditions.

---

# Phase 1 Diagnostics

## Diagnostic A — Context Retrieval Diagnostic

### Research Question

Does the current BGE-based memory retrieval fail because useful Gold-target history is absent, or because the similarity method fails to retrieve useful history?

### Conditions

Run all four:

- Full + Short
- Initial + Short
- Full + Multi3
- Initial + Multi3

### Metrics

For each relevant case:

- whether legal Gold-target history exists;
- Gold-history Recall@1;
- Recall@3;
- Recall@5;
- Recall@10;
- Recall@20.

Where possible, separately report:

- all history-available cases;
- ambiguous cases;
- conflict cases;
- per-author results.

### Failure Decomposition

Classify failures into:

1. Gold-target history does not exist;
2. Gold-target history exists but is not retrieved into BGE Top-K;
3. Gold-target history is retrieved but contextual reranking does not support Gold;
4. contextual support favors Gold but final score/ranking is still wrong.

### Goal

Determine whether the main bottleneck is:

- memory coverage;
- Stage-1 semantic retrieval;
- generic cross-encoder discrimination;
- final score aggregation.

No new model training is required.

---

## Diagnostic B — Initial-to-Full-Pinyin Expansion Diagnostic

### Research Question

Can same-user chronological history reduce Initial ambiguity by predicting a small set of likely Full-Pinyin expansions before Chinese candidate selection?

### Conditions

Run only:

- Initial + Short
- Initial + Multi3

### Metrics

For each case, rank the Gold Full-Pinyin expansion using legal H5000 same-Initial history.

Report:

- Gold Expansion Top1;
- Top3;
- Top5;
- Top10;
- Unseen.

Also report:

- expansion count / diversity;
- per-author results.

### Conflict Decomposition

Split existing Initial Conflict cases into:

#### Cross-expansion Conflict

The frequency-winning Chinese target and Gold have different Full Pinyin.

#### Within-expansion Conflict

The frequency-winning Chinese target and Gold share the same Full Pinyin.

### Goal

Estimate the headroom of a future hierarchical:

Initial → Full Pinyin → Chinese

personalisation method.

This diagnostic does not itself implement the hierarchy.

---

## Diagnostic C — PV0 Initial Recoverability Extension

### Research Question

How much of the high Initial Missing@10 rate can be recovered from the user's legal H5000 personal history?

### Conditions

Run:

- Initial + Short
- Initial + Multi3

Full + Short PV0 has already been completed and should not be rerun unless needed for validation.

### Metrics

Among Generic Missing@10 cases:

- number of Missing cases;
- recoverable Missing;
- unrecoverable Missing;
- recoverability rate;
- personal lexicon size distribution;
- per-author recoverability.

### Goal

Determine whether Personal Vocabulary candidate recovery should be extended to Initial conditions.

This is an extension of the previously completed PV0 recoverability analysis, not a new candidate-recovery concept.

---

# Known Dataset Issue ? Mixed Chinese Script

A read-only audit of frozen Dataset V1 found substantial Simplified/Traditional Chinese inconsistency:

- 1,074,032 interactions audited;
- 102,982 Gold targets (9.5884%) change under OpenCC t2s;
- 226,817 contexts (21.1183%) change under OpenCC t2s;
- the issue is strongly author-dependent, with MScarlet accounting for most affected Gold targets.

A separate audit of the existing Full + Short Generic candidate pool found that 186 of 538 exact Missing@10 cases (34.57%) contain a candidate that becomes equivalent to Gold after t2s canonicalisation.

Therefore, the current mixed-script Dataset V1 and its existing results are retained as historical baselines, but they should not be treated as the final formal dataset configuration.

The planned formal repair is:

1. derive a new version from frozen Dataset V1;
2. apply OpenCC t2s canonicalisation to context, Gold, and history targets;
3. do not reintroduce the earlier Dataset V1.1 metadata-deletion changes;
4. canonicalise PinyinGPT candidate surfaces with the same t2s policy;
5. deduplicate canonical candidates while preserving ranking information;
6. retain the Top-K unique Simplified-Chinese candidates;
7. rerun the final formal six-author evaluation under the new dataset version.

Until that repair is implemented, Phase 1 remains an exploratory diagnostic using existing frozen artifacts and avoids MScarlet to reduce the known script confound.

---

# Phase 1 Methodology Rules

- H5000 only.
- Three exploratory authors only.
- Reuse frozen Generic predictions and existing caches where possible.
- Avoid new Generic inference unless strictly necessary.
- No new neural training in Phase 1.
- Do not tune any new hyperparameter on Test.
- Existing Test failure analysis may motivate hypotheses, but new parameter selection must use Dev.
- Do not overwrite previous experiment outputs.
- Invalid or superseded results must remain archived and explicitly marked.

---

# Expected Decision After Phase 1

Diagnostic A determines whether to prioritize:

- local-context retrieval;
- improved retrieval;
- task-specific Memory Matcher;
- scoring changes.

Diagnostic B determines whether to prioritize:

- Initial → Full-Pinyin → Chinese hierarchical personalisation.

Diagnostic C determines whether to prioritize:

- Personal Vocabulary recovery under Initial conditions.

Only methods supported by Phase 1 diagnostics should proceed to implementation.