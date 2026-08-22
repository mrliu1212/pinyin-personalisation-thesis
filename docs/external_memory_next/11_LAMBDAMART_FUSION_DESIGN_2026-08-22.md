# 11 - LambdaMART Fusion Design

Date: 2026-08-22

Status: **PREDECLARED / EXECUTION GATED ON INPUT AUDIT**

## 1. Research question

Can shallow tree interactions among the existing runtime evidence improve
candidate ordering over the frozen Full RetunedFinal staged linear formula?

This is the nonlinear-only experiment. It uses raw Choice Share, not the
selected smoother, so smoothing and nonlinear fusion remain isolated.

## 2. Candidate surface and populations

- Fit only on causal Train-Fit-derived frozen RetunedFinal Top10 groups.
- Select only on Train-Val by Macro-author Top1.
- Preserve the frozen Train-Val Top10 surface exactly.
- Exclude zero-positive Train-Fit groups from the ranking objective because
  they provide no within-query supervised ordering signal.
- Retain all Train-Val groups, including empty/zero-positive groups, in metrics.
- Dev3000 and Test are forbidden.

Execution must stop if the preceding feature audit does not pass exact frozen
Val reconstruction or reports a schema different from this record.

## 3. Runtime feature schema

The 25 author-free features are:

```text
base_score, base_rank, source_personal, has_generic,
generic_score, normalized_generic_score, generic_rank,
frequency_count, personal_score, has_personal,
personal_candidate_rank, p_ng, choice_share,
entropy_concentration, log1p_same_pinyin_history,
log1p_raw_history, ngram_support, bge_support,
log1p_bge_history_count, ngram_effective_n,
log1p_ngram_matched_history, base_gap_to_top,
ngram_gap_to_top, bge_gap_to_top, frozen_linear_score
```

Gold correctness is a separate binary training label. Author identity,
rescue/harm status, post-hoc error class, future interaction information,
Dev/Test information, and any oracle category are absent from `X`.

## 4. Library and objective

Use isolated LightGBM 4.7.0 core API with:

```text
objective = lambdarank
binary label gain = [0, 1]
lambdarank truncation = 10
learning rate = 0.05
deterministic = true
force_col_wise = true
seed family = 1729
CPU threads = 8
```

The current frozen staged linear score is the primary control. A learned
depth-1 additive-stump model is a same-table no-interaction control.

## 5. Predeclared grid

Additive control:

```text
max_depth=1, num_leaves=2, min_data_in_leaf=500, rounds=100
```

Nonlinear grid:

```text
max_depth in {2, 3, 5}
num_leaves = 2^max_depth - 1
min_data_in_leaf in {100, 500}
rounds in {50, 100}
```

This is 12 nonlinear points. No early stopping or post-result boundary
extension is pre-authorized.

Select among nonlinear points by Macro-author Top1, then Micro Top1, MRR@10,
then lower depth, fewer rounds, larger minimum leaf size, and stable config ID.

Within a query, descending learned score determines the candidate order. Exact
score ties retain the frozen RetunedFinal order, with candidate text as the
final deterministic tie-break. This prevents a learned-score tie from changing
the candidate surface or introducing filesystem/order dependence.

## 6. Required outputs

- overall, ambiguous, conflict, per-author, Top1/3/5, MRR@10, Missing@10;
- rescue/harm/net against frozen RetunedFinal and smoothing alpha 128;
- comparison with the additive-stump control;
- feature importance by split and gain;
- mean absolute TreeSHAP contribution on Train-Val, labelled descriptive;
- model/config/library hashes, exact commands, runtime, and closed-data flags.

No significance claim is authorized by this development experiment.
