# Transparent, User-Controllable Pinyin Personalisation

## Current Phase

Phase 3 — Evaluation Framework

## Current Objective

The current phase provides a reproducible framework for measuring personalised
Pinyin candidate ranking. It compares:

- base ranking;
- correct-user personalisation;
- wrong-user personalisation.

These comparisons are established on deterministic synthetic users before any
real author data is introduced.

## Current System / Pipeline

```text
(context, pinyin)
        ↓
base candidates
        ↓
chronological user history
        ↓
personal model
        ↓
reranking
        ↓
evaluation metrics
```

The current system includes context-sensitive personal evidence,
strict chronological history filtering, directly inspectable scoring
components, and quantitative evaluation across multiple ranking conditions.

## Why This Phase

Before evaluating real-world personalisation, the project needs a controlled
framework that can determine:

- whether personalisation improves target ranking;
- whether any improvement is specific to the correct user;
- whether wrong-user history behaves differently from correct-user history;
- whether evaluation excludes all future information.

This framework separates evaluator correctness from later questions about real
data quality and generalisation.

## Current Model / Important Assumptions

- Pinyin is already normalized and tone-free.
- Candidate generation is external to the personal model and currently uses an
  in-memory source.
- Personal evidence combines:
  - global candidate frequency;
  - exact Pinyin candidate frequency;
  - exact context + Pinyin candidate frequency.
- Current evidence weights are:
  - global: `0.1`;
  - Pinyin: `0.3`;
  - context: `0.6`.
- The weights are configuration values, not optimised parameters.
- Personal and base scores are min-max normalized within the current candidate
  list before interpolation.
- Every evaluated interaction uses only history strictly earlier than its
  timestamp, preventing future leakage.

## Evaluation Framework

The current metrics are:

- Top-1 accuracy;
- Top-3 accuracy;
- Mean Reciprocal Rank (MRR);
- mean target rank;
- helpful, harmful, and unchanged reranking counts relative to Base.

The three evaluation settings are:

1. Base ranking
2. Correct-user personalised ranking
3. Wrong-user personalised ranking

Wrong-user personalisation applies another user's earlier history to the target
user's examples. This acts as a control for generic benefits that are not truly
user-specific.

If a target is absent from the candidate list, it is incorrect for Top-K and
contributes zero reciprocal rank. It is excluded from mean target rank and
reported through a separate missing-target count rather than receiving an
invented rank.

## How to Run

Run the full test suite from the repository root:

```bash
python3 -m unittest discover -s tests -v
```

The command prints each test result and a final pass/failure summary covering
the Phase 1–3 behaviours.

Run the current Phase 3 synthetic evaluation:

```bash
python3 -m experiments.exp_phase_03_evaluation
```

The experiment prints Top-1, Top-3, MRR, and mean target rank for Base,
correct-user, and wrong-user conditions. It also prints reranking change counts
for the personalised conditions.

## Current Limitations

- Evaluation currently uses deterministic synthetic multi-user data.
- A real author/text benchmark has not started.
- Exact context matching may become sparse on real text and could require a
  revised context representation.
- Evidence weights have not been optimised.
- Candidate generation remains simplified and in-memory.

## Next Phase

Phase 4 — Real Author Benchmark

The next phase will investigate whether the approach generalises from
controlled synthetic users to real chronological writing data while preserving
strict temporal separation.

## Git Phase Snapshots

Accepted phase versions are preserved using Git commits and, when explicitly
approved, phase tags such as:

- `phase-01`
- `phase-02`
- `phase-03`

Tags are not created automatically. Phase specifications live in
[`docs/phases/`](docs/phases/), while completed outcome summaries live in
[`results/phases/`](results/phases/).

