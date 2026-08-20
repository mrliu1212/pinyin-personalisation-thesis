# EM3-v2 Failure Audit and Data-Diversity Findings — 2026-08-20

## Status

ACTIVE RESEARCH NOTE

This note records the findings from the manual positive-history inspection and the first failure-subset audit conducted before EM3-v2 training.

No Test data was used.

---

## 1. Why this audit was run

The EM3 population audit showed that queries with positive history can have many matching prior interactions.

For the current clean3 authors:

- Agent Phage
- Etinjat
- breaddddd

the raw average positive-history count among queries with at least one positive was approximately 42 per query.

However, manual inspection was needed to determine whether raw interaction count corresponds to independent personalization evidence.

---

## 2. Manual positive-history inspection

Representative examples were printed from Full+Short / H5000 history.

Examples included:

- `shou rong -> 收容`
- `de -> 的`

A typical inspected query had approximately 8–15 positive history interactions.

### Key observation

Many positives were:

- from the same local document;
- close in chronological position;
- highly overlapping in surrounding context;
- repetitions of the same common target.

For example, several `de -> 的` positives were consecutive occurrences from nearly the same surrounding paragraph.

### Interpretation

Raw positive-history count should not be interpreted as the amount of independent personalization evidence.

Conceptually:

`raw positive count != effective independent evidence count`

A query with 10 positive interactions may contain only a few genuinely distinct contextual patterns.

This motivates measuring positive-history diversity in addition to quantity.

---

## 3. Literature connection

The observed redundancy problem is consistent with personalization literature that distinguishes history quantity/relevance from history utility or representativeness.

Relevant directions identified during literature review include:

- representative-history selection rather than treating all history as equally useful;
- history utility rather than semantic relevance alone;
- history reweighting / selection under noisy or redundant user history;
- memory triggering / selective personalization rather than always applying history.

This literature supports treating history redundancy, diversity and override decisions as explicit EM3-v2 design questions.

---

## 4. Failure subsets audited

The first targeted failure audit used frozen Full+Short Dev results and Hidden-M1 as the representative contextual personalization model.

Two subsets were inspected:

### A. Generic right / Hidden-M1 wrong

Definition:

`G Top1 correct AND Hidden-M1 Top1 wrong`

Observed:

- total cases: 124
- positive-history count mean: 6.36
- median: 1
- min: 0
- max: 65

### Interpretation

Most of this failure class occurs with weak or absent Gold-history support.

This supports a conservative decision rule:

When personal evidence is sparse, contextual personalization should not easily override a correct Generic prediction.

This is evidence for an evidence/confidence gate or a learned "should personalize?" decision.

---

### B. Frequency right / Hidden-M1 wrong

Definition:

`F Top1 correct AND Hidden-M1 Top1 wrong`

Observed:

- total cases: 59
- positive-history count mean: 20.81
- median: 11
- min: 1
- max: 65

### Interpretation

This subset cannot be explained primarily by lack of positive history.

In many cases, the Gold target had strong historical frequency support, but a contextually similar minority target was promoted by Hidden-M1.

This is a more important failure mode for EM3-v2.

---

## 5. Representative real examples

### Example 1 — `you -> 有`

Observed case:

- Gold: `有`
- Generic: `有` — correct
- Frequency: `有` — correct
- Hidden-M1: `由` — wrong
- same-Pinyin history:
  - `有`: 65
  - `又`: 14
  - `由`: 5
- Gold history share: 77.4%

Interpretation:

A contextually preferred minority target (`由`) overrode a much stronger long-run historical preference (`有`).

This demonstrates that contextual relevance can dominate frequency even when the historical preference is strong.

---

### Example 2 — `er -> 而`

Observed case:

- Gold: `而`
- Generic: `而` — correct
- Frequency: `而` — correct
- Hidden-M1: `尔` — wrong
- same-Pinyin history:
  - `而`: 57
  - `尔`: 3
  - `耳`: 1
- Gold history share: 93.4%

Interpretation:

This is a strong example of:

`semantic/contextual similarity != personalization utility`

The minority `尔` histories are plausible in the same classical-language style, but they should not automatically override a 57-to-3 historical preference for `而`.

---

### Example 3 — `qi -> 其`

Observed case:

- Gold: `其`
- Generic: `气` — wrong
- Frequency: `其` — correct
- Hidden-M1: `气` — wrong
- same-Pinyin history:
  - `其`: 57
  - `起`: 3
  - `七`: 1
  - `魌`: 1
  - `气`: 1
- Gold history share: 90.5%

Interpretation:

A target with only one prior occurrence (`气`) can still dominate the contextual ranking.

This shows that contextual evidence can over-amplify a very low-frequency personal target.

---

### Example 4 — `shi -> 是`

Observed case:

- Gold: `是`
- Generic: `食` — wrong
- Frequency: `是` — correct
- Hidden-M1: `食` — wrong
- same-Pinyin history:
  - `是`: 55
  - `时`: 11
  - `使`: 8
  - `氏`: 4
  - `似`: 4
  - others
- Gold history share: 61.8%

Current local context strongly contains the semantic environment of `食`.

Interpretation:

This is a genuine conflict rather than a simple contextual mistake.

A correct personalized system must sometimes override long-run frequency when the current context strongly supports another target.

Therefore EM3-v2 should not simply lock onto the highest-frequency personal target.

---

## 6. Emerging failure taxonomy

### Type 1 — Evidence sparse

Pattern:

- Generic correct
- Context model wrong
- zero or very few positive histories

Likely remedy:

- conservative personalization;
- Generic-confidence / personal-evidence gate;
- abstain from personalization when evidence is weak.

---

### Type 2 — Strong frequency, misleading contextual minority

Pattern:

- Frequency correct
- Context model wrong
- Gold has strong historical support
- minority target receives disproportionate contextual support

Likely remedy:

- learn when contextual evidence is strong enough to override frequency;
- explicit frequency-vs-context margin features;
- hard negatives drawn from contextually similar minority targets;
- ranking supervision that compares useful vs misleading history.

This is currently the most important EM3-v2 failure class.

---

### Type 3 — Genuine current-context conflict

Pattern:

- long-run personal preference supports one target;
- current context strongly supports another target.

Likely remedy:

- do not hard-code frequency dominance;
- learn an override decision using current-context evidence;
- separate "should personalize?" from "which personal history should be trusted?"

---

## 7. Important caution: row-level failures may be clustered

Several `breaddddd / you -> 有` failures occur in the same or closely related MTF-list material.

Therefore:

59 `F right / Hidden-M1 wrong` rows must not automatically be interpreted as 59 independent failure phenomena.

They may be concentrated in:

- a small number of Pinyin/Gold pairs;
- the same document;
- the same local section;
- adjacent chronological queries.

This must be audited before using the 59 rows as evidence of a general systematic failure.

---

## 8. Immediate next audit

Before EM3-v2 training, measure clustering and effective diversity in the 59-row failure subset.

Required statistics:

1. count of unique `(author, Pinyin, Gold)` patterns;
2. frequency distribution of those patterns;
3. top repeated patterns;
4. positive-history count distribution;
5. negative-history count distribution;
6. Gold-share distribution;
7. number of failures per author;
8. if document/work identity is available, number of unique documents and failures per document;
9. position-distance / local clustering if document identity is not available;
10. compare clustered vs non-clustered failures.

Decision question:

Are contextual regressions a broad phenomenon, or are they dominated by a few repeated Pinyin/Gold/document clusters?

---

## 9. EM3-v2 design implications so far

The working EM3-v2 direction is now:

**ranking-oriented supervision + diversity-aware history sampling + personalization-informative query selection + explicit conflict/override handling**

This is stronger than the earlier plan of only replacing pointwise BCE with a ranking loss.

Candidate data-design changes:

- diversify positive histories across position/document/context;
- avoid selecting three near-duplicate positives;
- prioritize queries with multiple plausible same-Pinyin targets;
- include contextually similar minority targets as hard negatives;
- retain frequency evidence as an explicit competing signal;
- train or learn an override decision rather than always adding contextual evidence.

No final loss/objective is frozen yet.

---

## 10. Next update rule

This note should be updated after:

- Pinyin/Gold clustering audit;
- positive-history diversity audit;
- document/position clustering audit;
- clean3 reproduction manifest generation;
- EM3-v2 method freeze.

Do not overwrite negative findings. Preserve them as research evidence.

---

## 11. Final full-surface outcome audit

The preliminary 124 `G right / Hidden-M1 wrong` and 59
`F right / Hidden-M1 wrong` analyses above remain valid exploratory slices.
The later canonical audit covers every row on the fixed Full+Short/H5000 old
exploratory three-author Dev surface (Etinjat, Re_spectators, breaddddd):

| G | F | Hidden-M1 | Rows |
|---|---|---|---:|
| ✓ | ✓ | ✓ | 3,361 |
| ✓ | ✓ | ✗ | 24 |
| ✓ | ✗ | ✓ | 42 |
| ✓ | ✗ | ✗ | 100 |
| ✗ | ✓ | ✓ | 403 |
| ✗ | ✓ | ✗ | 35 |
| ✗ | ✗ | ✓ | 45 |
| ✗ | ✗ | ✗ | 1,598 |

The total is 5,608. Micro correctness is G `3527/5608 = 62.89%`, F
`3823/5608 = 68.17%`, and Hidden-M1 `3851/5608 = 68.67%`. Relative to F,
Context has 87 rescues, 59 harms, and net `+28` rows (approximately `+0.50`
percentage points Micro Top1). These counts are diagnostic and must not replace
the primary Macro-author evaluation table.

Document concentration remains a serious interpretation caution. Top documents
account for large shares of several small groups, so individual rows must not be
treated as independent phenomena without clustered analysis.

## 12. Oracle diagnostics versus runtime features

Gold, Gold count/share, Gold-in-history, Gold-in-candidates, group membership,
and correctness are analysis-only oracle quantities. They may define diagnostic
or training-label subsets but must not become inference-time features.

Prediction-visible features may include:

- visible and exact-Pinyin history counts;
- target-frequency distribution, distinct target count, winner/share/margin,
  and entropy;
- Generic score/confidence/margin when available;
- retrieval/Context support and margins;
- G/F/H agreement and disagreement.

Formal Conflict is still defined by the repository's existing
`subset_membership` semantics. It is not synonymous with `G != F`.

## 13. All-wrong decomposition

The largest raw block is `G✗F✗H✗ = 1,598`:

- 1,275 have no Gold target in same-Pinyin history;
- 323 have Gold in same-Pinyin history;
- 53.9% of those 323 also have Gold in the current candidate set.

Approximately 174 rows are therefore the principal EM3 ranking/fusion
opportunity, while approximately 149 have a historical Gold precedent but lack
Gold from the candidate set and are candidate-recovery/EM1-type failures. The
174/149 figures are approximate implications of the exact 53.9% statistic, not
separately counted exact values. The 1,275 no-precedent rows must not all be
called EM3 ranking failures.

## 14. Final success and harm mechanisms

### Context-only rescue: `G✗F✗H✓ = 45`

All 45 have Gold in same-Pinyin history and the current candidate set. Gold is
the raw history-frequency winner in only 20%. Thus, in about 80% of these
successful rescues, Context selects a minority historical target. Removing
Context or using only one global scalar on Frequency would destroy this useful
mechanism.

### Pure Context regression: `G✓F✓H✗ = 24`

G and F are both correct and Context alone creates the error. This is the
cleanest harmful-override and hard-negative diagnostic set.

### Lost Frequency rescue: `G✗F✓H✗ = 35`

Compare with the 403 `G✗F✓H✓` successes. The failure group's median raw history
winner share is about 57.1%, entropy 1.684 bits, and distinct targets 4, versus
88.9%, 0.531 bits, and 2 for the successes. Same-Pinyin ambiguity is strongly
associated with Context difficulty.

### Generic right, Frequency/Context wrong: `G✓F✗H✗ = 100`

Gold occurs in same-Pinyin history for only 39% and is the raw history winner
for 0%. User history frequency can override a correct current-context prediction
even when the current target is absent or rare in personal history.

## 15. Final failure taxonomy

1. **No direct historical precedent:** history reranking alone cannot recover
   most of the 1,275 rows.
2. **Candidate recovery failure:** Gold exists in history but not the current
   candidate surface (approximately 149 rows).
3. **Candidate-level ranking/fusion failure:** Gold exists in both history and
   candidates but all methods fail (approximately 174 rows).
4. **Sparse/misleading personal override:** personal evidence harms a correct
   Generic result, including the 24 pure Context regressions.
5. **Ambiguous historical distribution:** lower winner dominance and higher
   entropy make Context selection unreliable, including the 35 lost F rescues.
6. **Minority-target opportunity:** candidate-specific Context correctly
   selects a non-frequency-winner target in most of the 45 Context-only rescues.
7. **Clustered evidence:** repeated Pinyin/target/document patterns reduce the
   effective independent sample size.

This supports aggregation, fusion, calibration, and utility estimation as the
current bottleneck. It does not justify claiming that hidden kNN retrieval is
worse than BGE; hidden retrieval discrimination was stronger but did not
materially improve end-to-end M1.

## 16. Current working direction

Raw user frequency can reflect globally common language rather than a personal
preference. Investigate user-vs-global frequency lift:

```text
PersonalLift(c,p) = log(P_user(c|p) / P_global(c|p))
```

The current conceptual direction is:

```text
Final(c) = Generic(c) + PersonalScore(c)
PersonalScore(c) ≈ PersonalLift(c) × ContextUtility(query, candidate, history)
```

This is not a frozen architecture. The central concept is candidate-specific
historical utility, not semantic similarity alone, raw frequency alone, or one
global Context weight. The design must preserve minority-target rescues while
reducing misleading overrides.

## 17. Reproducibility

Canonical runner:
`experiments/external_memory/em3_all_outcome_audit.py`.

Output root:
`results/personalisation/external_memory/em3_all_outcome_audit/`.

```powershell
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
& $python -m experiments.external_memory.em3_all_outcome_audit
```

Input hashes:

- history manifest:
  `7c85c38728d03985856d742f452992b3b3072af5f1c07845e099d9d07854da68`;
- Dev manifest:
  `cf072d9323328b77e3d47d8a0c1beed8c40edc8767e075fb58593d6b72120606`;
- four-way rows:
  `7bc20cddc5a772e7c1f9fb3fdd60ec17e8c2813667b7c32ec835b4cbc15d87d7`;
- fixed G/F/Context surface:
  `6e4007b2ba7cd0bffea4c869a7860cc08c3671bf078c22e957ad09d6ce18ea25`.

Detailed tables, history distributions, document concentration, and exact
output paths are preserved in
`EM3_ALL_OUTCOME_DISTRIBUTION_RECORD_2026-08-20.md`. No Test data was used.
