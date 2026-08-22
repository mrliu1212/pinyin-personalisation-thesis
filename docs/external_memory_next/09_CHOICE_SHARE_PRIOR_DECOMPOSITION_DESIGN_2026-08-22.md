# 09 - Choice Share Prior Decomposition Design

Date: 2026-08-22

Status: **PREDECLARED BEFORE RESULTS**

## 1. Why this diagnostic is necessary

The selected all-author-prior smoother at `(alpha=128, w_CS=6)` reached Macro
`.796515499`, but raw Choice Share with a lower fixed-surface coefficient
`w_CS=2` reached `.796500349`. The difference is only `.000015150`.

The initial positive result could therefore reflect suppression of an
overconfident feature rather than useful candidate-specific population prior
information. Those mechanisms must not be conflated.

## 2. Fixed comparison

On the same frozen RetunedFinal Top10/support surface, compare:

1. raw Choice Share, `w_CS=6` (frozen reference);
2. raw Choice Share, `w_CS=2` (scale control);
3. zero-prior shrinkage `n_c/(N+128)`, `w_CS=6`;
4. all-author Train-Fit prior shrinkage, alpha 128, `w_CS=6`;
5. other-author-only Train-Fit prior shrinkage, alpha 128, `w_CS=6`.

All other Stage-1 and Stage-2 weights/supports remain frozen. This is a
mechanism diagnostic, not another hyperparameter search. Report the full metric
set and Top1 transitions against raw `w_CS=6`.

## 3. Causal boundary

Both priors use only Train-Fit, which precedes Train-Val. The other-author prior
excludes the query author's Train-Fit counts. Unseen pairs receive zero mass.
No Dev3000 or Test input is accepted.
