# Phase 2 — Context Personalisation

Status: complete.

## Objective

Make personalisation context-sensitive while keeping the model lightweight,
interpretable, deterministic, and isolated by user and time.

## Scope / Work Performed

- Added three user-specific frequency components:
  - global candidate evidence;
  - exact-Pinyin candidate evidence;
  - exact-context-and-Pinyin candidate evidence.
- Added configurable `EvidenceWeights`, currently defaulting to `0.1/0.3/0.6`.
- Preserved `score(...)` as the combined-score interface and added
  `score_details(...)` for component inspection.
- Exposed global, Pinyin, and context evidence alongside combined personal and
  final scores in ranked output.

## Required / Verified Behaviours

- For one user, `我们可以 + shiyong` can prefer `使用`, while
  `这个软件很 + shiyong` prefers `实用`.
- Different users can learn different rankings for the same context and Pinyin.
- Other users' interactions never affect the selected user's evidence.
- Interactions at or after a cutoff never affect any evidence component.
- An unseen context retains global and Pinyin evidence while its exact-context
  evidence is zero.
- All scoring components are directly inspectable and the Phase 1 behaviours
  remain valid.

## Completion Criteria

- Context can change personalised ranking for otherwise identical Pinyin.
- User and temporal isolation hold across all three evidence levels.
- Broader evidence provides fallback for unseen contexts.
- Returned candidates expose every component used to calculate their scores.
- The full Phase 1 and Phase 2 test suite passes.

These criteria were met.

## Important Design Decisions

- Context evidence uses exact matching of the complete context string.
- Evidence consists of raw selection counts and the personal score is their
  weighted sum.
- Weights must be non-negative and at least one must be positive.
- The reranker min-max normalizes combined personal scores within the candidate
  list before base/personal interpolation.
- Debug output reports the ranking model's actual statistics rather than a
  separate explanation layer.

## Known Limitations / Deferred Questions

- Exact context matching may become sparse on real text.
- Evidence weights `0.1/0.3/0.6` are not optimised.
- Current min-max normalization does not explicitly encode evidence confidence.
- Raw counts do not discount old interactions and the evidence levels are
  correlated.
- Context suffixes, smoothing, confidence-aware fallback, and recency weighting
  remain possible later experiments, subject to interpretability and
  leakage-safe validation.

