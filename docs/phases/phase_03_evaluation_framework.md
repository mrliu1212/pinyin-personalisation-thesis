# Phase 3 — Evaluation Framework

Status: complete.

## Objective

Build a deterministic, leakage-safe evaluation framework that quantifies when
personalisation improves or harms candidate ranking and verifies that gains are
specific to the correct user.

## Scope / Work Performed

- Defined reusable per-instance evaluation records and chronological history
  filtering.
- Added deterministic synthetic multi-user data before considering real
  chronological author data.
- Added three-condition evaluation on the same test instances:
  - base ranking without personalisation;
  - correct-user personalisation;
  - wrong-user personalisation as a control.
- Added machine-readable aggregate metrics:
  - Top-1 accuracy;
  - Top-3 accuracy;
  - mean reciprocal rank (MRR);
  - mean target rank;
  - helpful, harmful, and unchanged reranking counts.
- Retained per-instance ranks, timestamps, selected control user, and history
  sizes to validate metric calculations and
  classify ranking changes without introducing a new persistence layer.

## Required / Verified Behaviours

- Top-1 is correct exactly when the target is ranked first.
- Top-3 is correct exactly when the target appears in the first three results.
- Reciprocal rank is `1 / target_rank`; MRR is its mean across evaluated
  instances.
- Mean target rank is computed consistently across all conditions.
- Reranking is helpful when target rank improves, harmful when it worsens, and
  unchanged otherwise, always relative to the base ranking.
- Correct-user and wrong-user conditions use identical test cases and candidate
  lists.
- Every personal model is fitted only on that user's interactions strictly
  earlier than the evaluated interaction.
- Repeated evaluation with the same inputs produces identical results.
- A missing target is incorrect for Top-K and contributes zero reciprocal rank.
  It is excluded from mean target rank and reported through a separate missing
  count, so no arbitrary numeric rank is invented.

## Completion Criteria

- Unit tests verify each metric on hand-calculated rankings.
- Tests verify helpful, harmful, and unchanged classification.
- A deterministic synthetic multi-user evaluation produces base, correct-user,
  and wrong-user results from the same chronological test sequence.
- Tests demonstrate that no current or future interaction enters a test
  instance's personal history.
- Aggregate outputs contain all required metrics and reranking counts and can be
  inspected against per-instance outcomes.
- No real author dataset was introduced; the synthetic framework is
  correct and reproducible.

These criteria were met.

## Important Design Decisions

- Chronology is part of evaluation semantics, not merely a preprocessing step.
- The wrong-user control uses another user's past history without mixing it into
  the target user's model.
- Base, correct-user, and wrong-user comparisons share deterministic candidate
  generation so only personal evidence changes between conditions.
- Metrics should remain independent of the current personal-scoring formula so
  later ablations can reuse the same evaluator.
- Phase 3 evaluates the existing algorithm; it does not optimise evidence
  weights or add profiles, interventions, UI, or external IME integration.

## Known Limitations / Deferred Questions

- Mean target rank and missing-target count must be read together because the
  mean includes only targets present in the candidate list.
- Synthetic results establish correctness but cannot establish real-world
  effectiveness or ecological validity.
- History-size experiments, component ablations, context variants, and real
  author data should follow only after the core evaluator is verified.
- Weight selection must use training or validation history without consulting
  future test outcomes.

### Deferred Future Research: Abbreviated Pinyin / Jianpin

Example: `shiyong -> sy`.

Potential question: does personalisation provide greater benefit when Pinyin
input is more ambiguous?

Evaluate this only if the eventual base candidate generator already supports
abbreviated Pinyin. Building a custom abbreviated-Pinyin decoder remains out of
scope.
