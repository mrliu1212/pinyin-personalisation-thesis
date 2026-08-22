# 06 - Smoothing Fusion Retune Design

Date: 2026-08-22

Status: **PREDECLARED BEFORE RESULTS**

## 1. Purpose

The smoothing-only study selected an interior alpha and produced a small,
consistent Train-Val improvement. Because shrinkage changes the scale of the
Choice Share feature, the next controlled question is whether its coefficient
should change. This is not a new candidate generator or nonlinear model.

## 2. Sequential search

Stage A keeps `w_P=2`, `w_E=4`, `lambda_N=6`, and `lambda_B=6` fixed and
searches:

```text
alpha in {0, 2, 8, 32, 128, 512}
w_CS in {2, 4, 6, 8, 12}
```

The pair is selected by Macro-author Top1, then Micro Top1, MRR@10, then
distance to the smoothing-only reference `(alpha=128, w_CS=6)`, then numeric
order.

Stage B freezes the selected Stage-A point and searches the existing lattice:

```text
lambda_N in {0, 2, 4, 6, 8}
lambda_B in {0, 2, 4, 6, 8}
```

Selection uses Macro-author Top1, then Micro Top1, MRR@10, then distance to
the current `(6,6)` point, then numeric order.

## 3. Fixed terms

The Generic rows, RetunedFinal Top10 surface, Personal-K5, `P_NG`, entropy,
NGram/BGE support values, causal history semantics, and tie-breaking remain
unchanged. Alpha zero must again reproduce the raw-Choice-Share baseline.

## 4. Evaluation boundary

Train-Fit constructs the causal prior and Train-Val selects parameters.
Dev3000 and Test are forbidden. This follow-up is exploratory development on
the same Train-Val used by the earlier grid; any gain must be reported as such.
