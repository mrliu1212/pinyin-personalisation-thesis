# Context Lab — Short-Focus Decision

**Date:** 2026-08-18  
**Status:** Active Phase 1 research decision

---

## 1. Decision

The active Context Personalisation Lab will now focus on:

- Full + Short
- Initial + Short

Multi3 is temporarily deferred from active method development.

The existing Multi3 manifests, caches, diagnostics, and partial/previous experiment artifacts must be preserved. Nothing is deleted or reclassified as failed.

This decision applies to the current exploratory Context Lab only. It does not remove Multi3 from the overall thesis problem definition.

---

## 2. Why Short Is Prioritised

Diagnostic A1 showed that the current exact-Pinyin H5000 memory formulation behaves very differently between Short and Multi3.

### Full + Short

Three exploratory authors:

- Etinjat
- Re_spectators
- breaddddd

Test rows:

- 3,000

Observed:

- History Available: 1,891 / 3,000 = 63.03%
- Gold-target history exists: 1,698 / 3,000 = 56.60%
- Gold exists given available history: 89.79%

BGE retrieval among rows where legal Gold-target history exists:

- Recall@1: 85.22%
- Recall@3: 93.70%
- Recall@5: 96.17%
- Recall@10: 98.53%
- Recall@20: 99.41%

Interpretation:

For Full + Short, Stage-1 BGE retrieval is already strong when useful Gold-target history exists.

The next question is therefore not primarily whether BGE can retrieve the Gold history, but why contextual scoring/reranking does not consistently outperform the Frequency baseline after useful history has already been retrieved.

This motivates an A2 scoring/decision diagnostic.

---

## 3. Initial + Short Finding

Test rows:

- 3,000

Observed:

- History Available: 2,796 / 3,000 = 93.20%
- Gold-target history exists: 1,698 / 3,000 = 56.60%
- Gold exists given available history: 60.73%

BGE retrieval among rows where legal Gold-target history exists:

- Recall@1: 39.22%
- Recall@3: 59.72%
- Recall@5: 67.96%
- Recall@10: 79.15%
- Recall@20: 86.63%

The number of rows containing Gold history is identical to Full + Short, but Initial input exposes substantially more non-Gold history.

This is consistent with the hypothesis that Initial input merges multiple Full-Pinyin intents into the same history pool and therefore makes semantic retrieval substantially more difficult.

This motivates:

- Initial-to-Full-Pinyin expansion diagnostics;
- analysis of cross-expansion vs within-expansion ambiguity;
- Personal Vocabulary recoverability for Initial input.

The structural explanation remains a research hypothesis until further diagnostics are completed.

---

## 4. Multi3 A1 Coverage Observation

A1 was also able to inspect the already-constructed Multi3 condition/history manifests.

These are legitimate dataset/condition artifacts even though the formal Multi3 personalisation experiment was not completed as an accepted final result.

### Full + Multi3

- Test rows: 3,000
- History Available: 27 / 3,000 = 0.90%
- Gold-target history exists: 26 / 3,000 = 0.87%

### Initial + Multi3

- Test rows: 3,000
- History Available: 89 / 3,000 = 2.97%
- Gold-target history exists: 26 / 3,000 = 0.87%

The very high retrieval Recall values in these conditions are based on only 26 Gold-history rows and therefore should not be interpreted as evidence that BGE is especially strong on Multi3.

The important observation is instead:

> Under the current H5000 + exact-current-condition Pinyin-history definition, Multi3 has extremely low useful historical coverage.

---

## 5. Consequence for Multi3

The current Short personalisation mechanism should not automatically be assumed to be appropriate for Multi3.

Multi3 may require a dedicated personalisation mechanism.

Possible future directions include:

- subphrase-level history;
- prefix or suffix memory;
- lexical-unit decomposition;
- back-off from exact phrase history;
- composition of shorter personalised evidence;
- other long-composition-specific mechanisms.

These are future hypotheses, not implemented or validated methods.

For now:

> Multi3 is preserved and deferred, not abandoned.

---

## 6. Active Short Research Path

Current priority:

```text
Full + Short
    ↓
A1 retrieval diagnostic complete
    ↓
inspect Ambiguous / Conflict / per-author results
    ↓
A2 scoring / decision diagnostic
    ↓
determine why retrieved contextual evidence
does not consistently beat Frequency
```

and:

```text
Initial + Short
    ↓
A1 retrieval diagnostic complete
    ↓
inspect Ambiguous / Conflict / per-author results
    ↓
Diagnostic B:
Initial → Full-Pinyin expansion
    ↓
Diagnostic C:
Personal Vocabulary recoverability
```

---

## 7. Methodological Boundaries

Current exploratory authors:

- Etinjat
- Re_spectators
- breaddddd

MScarlet remains temporarily excluded because of the separately documented mixed-script confound.

H5000 semantics remain unchanged:

- same user;
- strictly prior interactions;
- history budget applied before Pinyin filtering;
- exact current-condition segmented-Pinyin matching.

No Test result may be used to tune new hyperparameters.

Test diagnostics may motivate hypotheses and future method designs, but parameter selection must remain Dev-only.

Existing Multi3 and mixed-script artifacts must remain archived and must not be overwritten.

---

## 8. Immediate Next Steps

1. Complete detailed Short A1 subset analysis:
   - Overall
   - History Available
   - Ambiguous
   - Conflict
   - per-author

2. Full + Short:
   - run A2 scoring/decision diagnosis.

3. Initial + Short:
   - run Initial-to-Full-Pinyin expansion diagnostic;
   - run Initial Personal Vocabulary recoverability diagnostic.

4. Defer new Multi3 method development until the Short pipeline is better understood.

5. Before final formal evaluation:
   - resolve the mixed-script dataset issue;
   - return to all six authors;
   - separately decide the formal Multi3 strategy.

---

## 9. Reporting Guidance

Safe current conclusion:

> Diagnostic A1 indicates that the bottleneck differs substantially by input condition. Full + Short has high Gold-history retrieval recall when useful history exists, while Initial + Short exposes much more competing history and has substantially lower retrieval recall. Multi3 has very low exact-history coverage under the current memory definition.

Do not yet write:

> Context is useless.

Do not yet write:

> BGE fails on all conditions.

Do not yet write:

> Multi3 cannot be personalised.

The currently supported conclusion is narrower:

> The same exact-Pinyin contextual-memory formulation does not expose the same bottleneck across Short and Multi3, motivating Short-focused analysis and a future dedicated Multi3 mechanism.
