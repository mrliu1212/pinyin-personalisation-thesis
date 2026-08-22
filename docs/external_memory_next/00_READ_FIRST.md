# External Memory Next — Read First

Status: **EXPERIMENTS A/B/C + POST-HOC CALIBRATION COMPLETE / TRAIN-VAL DEVELOPMENT ONLY**

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

1. Phase 0 - evidence/readiness audit, including exact Full RetunedFinal
   reproduction: complete.
2. Experiment A - Choice Share smoothing: complete; a small diagnostic
   improvement, with the evidence mainly indicating that raw Choice Share was
   over-weighted.
3. Experiment B - LambdaMART nonlinear fusion: complete; positive end-to-end
   result, Macro Top1 `.7960049265502147 -> .7988390633366215`.
4. Experiment C - task-specific bi-encoder: complete; strong intrinsic
   retrieval improvement but a slight fixed-fusion end-to-end regression. The
   predeclared gate failed, so no task-specific LambdaMART refit was run.
5. Post-hoc recovery/calibration diagnostic: complete on separate Initial and
   Full Train-Val tracks. Equal calibration, frozen Recency, and Personal-K5
   recovery use did not convert the intrinsic Task retrieval gain into an
   overall end-to-end gain. Q8/Q8+F remain the strongest candidate scorers.

The current best Train-Val External Memory configuration remains the existing
generic-BGE LambdaMART result from Experiment B.

```text
Experiment C: task-specific BiEncoder   COMPLETE
Post-hoc calibration/recovery follow-up COMPLETE
Task-BiEncoder further retuning         CLOSED FOR THIS PHASE
Dev3000                                 CLOSED / UNUSED HERE
Test                                    CLOSED / UNUSED
```

Current scientific direction: representation-level Task-BiEncoder improvement
did not yield an end-to-end gain; nonlinear candidate-level evidence
arbitration remains the stronger observed direction. This is not a causal or
statistical-significance claim.

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
  design/cost gate and original decision to defer training.
- `15_TASK_SPECIFIC_BIENCODER_PREDECLARED_PROTOCOL_2026-08-22.md`: protocol
  frozen before the later explicitly authorized training run.
- `16_TASK_SPECIFIC_BIENCODER_RESULTS_2026-08-22.md`: causal audit, training,
  intrinsic retrieval, fixed-fusion result, hashes, and reproduction commands.
- `17_...` and `18_...`: frozen post-hoc protocol and comparability/cost gate.
- `19_POSTHOC_TASK_BIENCODER_RECOVERY_CALIBRATION_RESULTS_2026-08-22.md`:
  separate Initial/Full calibration, recovery, Q8, latency/Pareto results,
  answers to the primary questions, hashes, and reproduction commands.

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
Task bi-encoder intrinsic Macro Recall@1 = 0.8109711910595357
Generic BGE intrinsic Macro Recall@1 = 0.7789437773409569
Task fixed-fusion Macro Top1 = 0.7957117243433173
Task vs frozen = 34 rescue / 46 harm / net -12
Task-specific LambdaMART refit = not authorized by frozen gate
Post-hoc Initial Generic joint Macro = 0.4370578839609785
Post-hoc Initial Task joint Macro = 0.4356462413968509
Post-hoc Full Generic joint Macro = 0.7960049265502147
Post-hoc Full Task joint Macro = 0.7957480665207601
Post-hoc conclusion = calibration/Recency/recovery do not produce a Task win
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
results/personalisation/external_memory_next/task_specific_biencoder_v1/
results/personalisation/external_memory_next/posthoc_task_biencoder_calibration_v1/
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
