# Reranking Personalisation Matrix

> **IMPLEMENTED; FINAL MATRIX RUN PENDING**

## Scope

The frozen experiment evaluates G0/F/M1/M2 across four T1 conditions and
H0/H500/H5000/HFull. It contains 36 non-H0 method cells. Full+Short/H5000 F,
M1, and M2 are reused from completed artifacts; 33 cells are newly required.

The method, cache identities, chronology, Dev selection, resume rules,
wrong-user control, and commands are documented in
[Reranking Personalisation Matrix](../research/reranking_personalisation_matrix.md).

## Known Reused Full+Short H5000 Results

| System | Macro-author Top-1 | Top-3 | MRR@10 | Missing@10 |
| --- | ---: | ---: | ---: | ---: |
| G0 | 0.7231667 | 0.8535000 | 0.7934288 | 0.0896667 |
| F-H5000 | 0.7718333 | 0.8723333 | 0.8249612 | 0.0896667 |
| M1-H5000 | 0.7675000 | 0.8713333 | 0.8225745 | 0.0896667 |
| M2-H5000 | 0.7650000 | 0.8716667 | 0.8209835 | 0.0896667 |

Frozen selections are F lambda `4.0`, M1 Top-N `5` with lambda `4.0`, and M2
retrieval K `20` with lambda `4.0`.

## Pre-run Audit

The real audit verifies 24,000/24,000 frozen Test Generic rows, zero expected
Test inference, 3 reused cells, 33 new cells, all prior artifact hashes, and
the shared BGE/M2 caches. Condition-aware HFull legitimately requires new
context embeddings; existing cached vectors are reused and never recomputed.

## Pending Results

The following remain pending until the detached worker writes `COMPLETE.json`:

- the 4 x 3 x 3 condition matrix;
- per-author and Overall/History Available/Ambiguous/Conflict metrics;
- per-cell Dev selections;
- H0/H500/H5000/HFull learning curves;
- consolidated contextual diagnostics;
- Full+Short/HFull wrong-user control;
- final runtime, cache reuse, and artifact checksums.

No values will be inferred from smoke output. After completion, this report
must be finalized from `condition_matrix.csv`, `learning_curves.csv`,
`context_diagnostics.csv`, `wrong_user_summary.json`, and
`metrics_summary.json`.

## Expected Output Root

`results/personalisation/reranking_matrix/` contains the durable manifest,
audit/smoke metadata, structured cell results, aggregate CSV/JSON files,
operational logs, and final `COMPLETE.json`.
