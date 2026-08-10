# Phase 1 — Minimal Pipeline

Status: complete.

## Objective

Establish the smallest deterministic pipeline from Chinese context and Pinyin
input to a personalised Top-K ranking, using manually verifiable synthetic data.

## Scope / Work Performed

- Defined immutable interaction, base-candidate, and ranked-candidate records.
- Added a deterministic in-memory base candidate ranker.
- Added a per-user candidate-frequency model conditioned on exact Pinyin.
- Added min-max score normalization and linear base/personal interpolation.
- Added a standard-library-only synthetic `shiyong` demonstration.

## Required / Verified Behaviours

- User history can promote a lower-ranked base candidate to Top-1.
- Only the selected user's interactions affect their model.
- When a cutoff is supplied, only interactions strictly before it are used.
- Base and personalised ranking are deterministic.
- The synthetic example changes `实用, 使用, 试用` to `使用, 实用, 试用`.

## Completion Criteria

- A complete `(context, pinyin) -> base candidates -> personal score -> Top-K`
  path runs without external dependencies.
- Candidate order can be checked manually on the synthetic fixture.
- Reranking, user isolation, and temporal isolation tests pass.

These criteria were met.

## Important Design Decisions

- Input Pinyin is assumed to be normalized and tone-free.
- Candidate generation remains separate from personalisation and is represented
  by an in-memory lexicon at this stage.
- Scores are min-max normalized within each candidate list; a constant vector
  maps to zero and contributes no ordering preference.
- Ties preserve deterministic base-ranking behaviour.

## Known Limitations / Deferred Questions

- Context is accepted by the interface but does not affect Phase 1 scoring.
- The synthetic lexicon is not representative of a real IME.
- Raw frequency does not model recency, uncertainty, or confidence.
- Research metrics and history-size evaluation are deferred.
- The eventual base candidate generator and principled score calibration remain
  open questions.

