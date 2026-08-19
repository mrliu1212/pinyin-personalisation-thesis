# EM-2E1 Hidden-M1 End-to-End Design

Status: DESIGN FROZEN BEFORE DEV GRID
Date: 2026-08-19

## Research question

If the original M1 reranker is kept unchanged except for replacing its
general-purpose BGE retrieval representation with the Frozen PinyinGPT
task-native hidden representation, does end-to-end candidate ranking improve?

## Scope

- Full+Short
- H5000
- Dev tune only
- Authors:
  - Etinjat
  - Re_spectators
  - breaddddd
- Frozen Generic Top10 candidate surface
- No candidate recovery
- No frequency fusion

## Controlled comparison

Original M1:

    BGE context representation
        -> cosine retrieval
        -> Top-N histories
        -> positive similarity weights
        -> aggregate weights by historical target
        -> G + lambda_memory * context_support

Hidden-M1:

    Frozen PinyinGPT hidden representation
        -> cosine retrieval
        -> Top-N histories
        -> positive similarity weights
        -> aggregate weights by historical target
        -> G + lambda_hidden * context_support

The intended methodological change is only:

    BGE representation
        ->
    Frozen PinyinGPT hidden representation

## Frozen hidden representation

- Frozen PinyinGPT2-Concat
- final Transformer layer
- final prompt [SEP] token
- 768 dimensions
- cosine similarity

This definition was frozen by EM-2A before retrieval results.

## Legal history

Reuse the existing HistoryIndex semantics:

- same author
- strictly prior
- H5000 applied before Pinyin filtering
- exact same segmented Pinyin

## Candidate context support

Retrieve Top-N legal histories.

For each selected history i:

    weight_i = max(cosine_i, 0)

For candidate c:

    C(c) =
        sum(weight_i where historical_target_i == c)
        /
        sum(all positive selected weights)

If the denominator is zero:

    C(c) = 0

This is the same transparent target-support aggregation used by M1.

## Final score

    score(c) =
        z(GenericScore(c))
        + lambda_hidden * C(c)

## Dev grid

Top-N:

    {1, 3, 5, 10, 20}

lambda_hidden:

    {0, 0.25, 0.5, 1, 2, 4}

This matches the original M1 search surface.

If and only if lambda_hidden = 4 is selected at the upper boundary,
perform one pre-registered boundary check at lambda_hidden = 8 before
freezing the final Dev selection.

## Primary selection metric

Macro-author Overall Top1.

Exact ties:

1. lower lambda_hidden;
2. lower Top-N.

## Secondary reporting

Report Micro and Macro-author metrics on:

- Overall
- History Available
- Ambiguous
- Conflict

Metrics:

- Top1
- Top3
- MRR@10
- Missing@10

Also report Generic -> Hidden-M1:

- rescue
- harm
- net

for each subset.

## No-tuning rule

Do not:

- change the hidden layer;
- change extraction position;
- add pooling;
- change similarity;
- add Frequency;
- add Recovery;

to improve this experiment after inspecting results.

## Future hypotheses registered before Hidden-M1 result

### Fixed fusion

Later investigate:

    G + lambda_F * F + lambda_C * C_hidden

This tests whether long-term frequency preference and situation-specific
context evidence are complementary.

### Adaptive fusion

Later investigate query-dependent weights:

    G + lambda_F(q) * F + lambda_C(q) * C_hidden

Possible pre-registered gating information includes:

- visible history count;
- distinct historical target count;
- frequency winner share;
- frequency margin;
- Top-1 retrieval similarity;
- similarity margin;
- retrieved-target agreement.

These are future hypotheses and are NOT part of Hidden-M1 selection.

## Test rule

No Test evaluation until Hidden-M1 Dev selection is complete and frozen.
