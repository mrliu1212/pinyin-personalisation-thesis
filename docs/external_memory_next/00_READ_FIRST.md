# External Memory Next — Read First

Status: **EXPERIMENTS A/B COMPLETE / REPRODUCIBLE TRAIN-VAL IMPROVEMENT**

Worktree: `C:\Users\chiar\Desktop\LBH\thesis-external-memory-next`

Branch: `work/external-memory-next`

Base commit: `fb09ca2fa50589a0fc72130552212c5b47ed4365`

## Scientific boundary

- Current research surface: Full+Short with H5000 causal history.
- Main reference: frozen Full RetunedFinal.
- Development resources: Train-Fit for fitting and Train-Val for selection.
- Dev3000 has already been observed and is forbidden for feature design,
  architecture choice, hyperparameter selection, or method selection here.
- Test is closed.
- Candidate generation and the frozen Generic surface are not being changed.
- Model-level personalisation, LoRA, adapters, and per-user fine-tuning are out
  of scope.

## Current order

1. Phase 0 evidence and reusable-artifact audit: complete.
2. Exact Full RetunedFinal baseline reproduction: complete.
3. Choice Share smoothing ablation: complete; alpha 128 formally selected,
   but prior-specific value is weak relative to simpler suppression controls.
4. Nonlinear evidence-fusion study: complete; LambdaMART improved the primary
   Train-Val metric.
5. Task-specific bi-encoder design/cost gate: complete; training deferred on
   current evidence.

## Records

- `01_BASE_AND_PROVENANCE_2026-08-22.md`: worktree/base decision and source
  provenance.
- `02_PHASE0_EVIDENCE_AUDIT_2026-08-22.md`: real data distributions, reusable
  artifacts, hashes, and implications for method design.
- `03_FULL_RETUNED_BASELINE_REPRODUCTION_2026-08-22.md`: exact baseline gate.
- `04_...` through `07_...`: smoothing designs and completed results.
- `08_NONLINEAR_FUSION_READINESS_AND_DATA_PLAN_2026-08-22.md`: current
  learned-fusion dependency and method gate.
- `09_...` and `10_...`: Choice Share prior-mechanism design and result.
- `11_LAMBDAMART_FUSION_DESIGN_2026-08-22.md`: predeclared nonlinear grid.
- `12_LEARNED_FUSION_INPUT_GATE_2026-08-22.md`: Generic/support generation,
  feature audit, empty-surface correction, and matrix provenance.
- `13_LAMBDAMART_FUSION_RESULTS_2026-08-22.md`: selected nonlinear result,
  controls, breakdowns, interpretation, and hashes.
- `14_TASK_SPECIFIC_BIENCODER_DESIGN_COST_GATE_2026-08-22.md`: evidence-based
  design/cost gate and decision to defer training.

## Current result checkpoint

```text
Frozen Full RetunedFinal Macro-author Top1 = 0.7960049265502147
Smoothed fixed-surface alpha=128 Macro = 0.7965154987791901
Top1 transition = 25 rescue / 9 harm / net +16
Fusion retune = original w_CS=6, lambda_N=6, lambda_B=6 reselected
Mechanism conclusion = raw Choice Share over-weighted; population-prior
                       contribution not clearly separated from suppression
LambdaMART Macro-author Top1 = 0.7988390633366215
LambdaMART vs frozen = 267 rescue / 171 harm / net +96
LambdaMART Conflict Macro Top1 = 0.2965464515079956
Selected tree = depth 5 / 31 leaves / min leaf 500 / 100 rounds
Dev3000 used = false
Test used = false
```

Completed local-only artifacts and logs are under:

```text
results/personalisation/external_memory_next/train_fit_generic_v1/
results/personalisation/external_memory_next/train_fit_ranking_features_v1/
results/personalisation/external_memory_next/learned_fusion_input_audit_v1/
results/personalisation/external_memory_next/lambdamart_matrices_v1/
results/personalisation/external_memory_next/lambdamart_fusion_v1/
```

The Generic cache and all downstream feature/matrix/result artifacts are
complete and checksum-recorded. Generated artifacts remain local-only.

## Non-negotiable causal rule

```text
same author
-> strictly prior interactions
-> latest up-to-5000 raw interactions
-> exact segmented-Pinyin filtering
```

Gold may be used for supervised fitting and evaluation, but it must never enter
runtime features or candidate construction.

## Resume rule

Read this file, then the numbered records in order, followed by the living
`docs/FILE_INDEX.md` and `docs/REPRODUCIBILITY_INDEX.md`. Preserve every
versioned result namespace and do not overwrite historical artifacts.
