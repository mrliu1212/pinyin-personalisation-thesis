# 04 - Choice Share Smoothing Design

Date: 2026-08-22

Status: **PREDECLARED BEFORE RESULTS**

## 1. Research question

Does shrinking raw personal Choice Share improve the frozen Full RetunedFinal
ranking when same-Pinyin history is sparse?

## 2. Controlled ablation

Only the candidate-specific Choice Share term changes. The following remain
fixed:

- the frozen Generic-frequency candidate rows;
- the frozen RetunedFinal Top10 candidate surface;
- Personal-K5 construction and `P_NG`;
- entropy concentration;
- Stage-1 weights `w_P=2`, `w_CS=6`, `w_E=4`;
- existing NGramRecency and BGERecency supports;
- Stage-2 weights `lambda_N=6`, `lambda_B=6`;
- all ranking and tie-breaking semantics.

Keeping the Top10 surface fixed is necessary for a clean arithmetic-only
ablation because the durable Stage-2 support artifact covers that exact
surface. It also avoids confounding smoothing with new candidate recovery or
new BGE inference.

## 3. Estimator

For a Train-Val query with exact segmented Pinyin `p`, personal candidate `c`,
visible same-author count `N`, candidate count `n_c`, and all-author Train-Fit
prior:

```text
P_smooth(c | u,p) = (n_c + alpha * P_Fit(c|p)) / (N + alpha)
```

All Train-Fit interactions precede Train-Val, so `P_Fit` is causal at this
evaluation point. It does not use the current or future Train-Val gold.

If the Pinyin exists in Train-Fit but the candidate does not, prior mass is
zero. If the Pinyin itself is unseen, prior mass is also zero. This deliberately
shrinks unsupported personal evidence toward the Generic decision boundary;
it does not invent a uniform candidate preference.

`alpha=0` is defined as the exact raw Choice Share baseline. The runner
reconstructs `n_c` from the durable raw share and `N` and rejects non-integral
inconsistencies.

## 4. Grid and selection

The predeclared grid is:

```text
alpha in {0, 1, 2, 4, 8, 16, 32, 64, 128}
```

It spans the observed Train-Val history regime: median `N=5`, P75 `N=27`,
P90 `N=83`. Selection uses Macro-author Top1, then Micro Top1, then MRR@10,
then the smaller alpha. No fusion coefficient is retuned in this experiment.

## 5. Required reporting

Report overall, ambiguous, conflict, and history-count-bin metrics; per-author
Top1; Top3; Top5; MRR@10; Missing@10; and Top1 rescue/harm/net against
`alpha=0`. The alpha-zero output must exactly reproduce the frozen baseline or
the experiment stops.

## 6. Boundaries and limitations

- Train-Fit constructs the prior; Train-Val selects alpha.
- Dev3000 is not accepted and Test is closed.
- Because the candidate surface is fixed, this experiment estimates ranking
  effects on the existing surface. It does not measure whether smoothing would
  admit a different Personal-K5 candidate into Top10.
- A promising result may justify a separate versioned dynamic-surface study;
  it does not authorize silently reusing incomplete Stage-2 supports.
