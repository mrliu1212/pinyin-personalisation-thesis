# Repository File Index

Purpose: answer **what each meaningful file or directory is for, and what its current status is**. This is a static inventory of `work/external-memory-completion` at commit `a9a9351c85fe7f40f17c5232e5f77b6c84e7b35c`, inspected on 2026-08-19. It follows [FILE_MANAGEMENT_RULES.md](FILE_MANAGEMENT_RULES.md).

**Maintenance:** this is a living repository index. Update `docs/FILE_INDEX.md` in place as the repository evolves; do not create dated/version-suffixed replacements for routine updates. Git history is the revision history.

Statuses are `ACTIVE`, `FROZEN`, `HELPER`, `LEGACY`, `DEFERRED`, `GENERATED`, and `LOCAL-ONLY`. Compound statuses are used where a generated artifact is also intentionally local-only. No status authorizes deletion.

## External Memory Next — active 2026-08-22

| Path | Type | Purpose and dependencies/outputs | Related stage | Status |
|---|---|---|---|---|
| `docs/external_memory_next/00_READ_FIRST.md` | Current-status entry point | Scientific boundary, execution order, records, and resume rule for the isolated next-phase worktree. | External Memory Next | ACTIVE |
| `docs/external_memory_next/01_BASE_AND_PROVENANCE_2026-08-22.md` | Provenance record | Base-selection evidence, branch divergence, supplied-index identities, and source checkpoint map. | Phase 0 | FROZEN RECORD |
| `docs/external_memory_next/02_PHASE0_EVIDENCE_AUDIT_2026-08-22.md` | Evidence audit | Real Train-Fit/Train-Val distributions, feature availability, reusable artifacts, and method-design implications. | Phase 0 | COMPLETE |
| `docs/external_memory_next/03_FULL_RETUNED_BASELINE_REPRODUCTION_2026-08-22.md` | Reproduction record | Exact arithmetic reconstruction of every frozen Full RetunedFinal Train-Val candidate order and rank. | Baseline gate | COMPLETE |
| `docs/external_memory_next/04_CHOICE_SHARE_SMOOTHING_DESIGN_2026-08-22.md` | Predeclared design | Fixed-surface empirical-Bayes estimator, causal prior, alpha grid, selection rule, and limits. | Experiment A | FROZEN RECORD |
| `docs/external_memory_next/05_CHOICE_SHARE_SMOOTHING_FIXED_SURFACE_RESULTS_2026-08-22.md` | Result record | Positive alpha=128 Train-Val ablation, boundary extension, breakdowns, hashes, and cautious interpretation. | Experiment A | COMPLETE |
| `docs/external_memory_next/06_SMOOTHING_FUSION_RETUNE_DESIGN_2026-08-22.md` | Predeclared design | Sequential Choice Share coefficient and Stage-2 lambda follow-up grid. | Experiment A | FROZEN RECORD |
| `docs/external_memory_next/07_SMOOTHING_FUSION_RETUNE_RESULTS_2026-08-22.md` | Result record | Negative retune result: original fusion weights reselected, isolating the smoothing effect. | Experiment A | COMPLETE |
| `docs/external_memory_next/08_NONLINEAR_FUSION_READINESS_AND_DATA_PLAN_2026-08-22.md` | Readiness gate | Missing Train-Fit Generic surface, causal data-generation plan, cost, and learned-ranking boundary. | Experiment B | COMPLETE RECORD |
| `docs/external_memory_next/09_CHOICE_SHARE_PRIOR_DECOMPOSITION_DESIGN_2026-08-22.md` | Predeclared diagnostic | Separates population-prior information from generic Choice Share suppression. | Experiment A | FROZEN RECORD |
| `docs/external_memory_next/10_CHOICE_SHARE_PRIOR_DECOMPOSITION_RESULTS_2026-08-22.md` | Result record | Shows that raw Choice Share is over-weighted but prior-specific added value is weak. | Experiment A | COMPLETE |
| `docs/external_memory_next/11_LAMBDAMART_FUSION_DESIGN_2026-08-22.md` | Predeclared design | Runtime feature schema, group policy, LightGBM objective, controls, grid, and selection rule. | Experiment B | FROZEN RECORD |
| `docs/external_memory_next/12_LEARNED_FUSION_INPUT_GATE_2026-08-22.md` | Input/result gate | Validated 144,526-row Generic/support generation, runtime-only feature audit, matrices, hashes, and empty-surface no-op correction. | Experiment B | COMPLETE |
| `docs/external_memory_next/13_LAMBDAMART_FUSION_RESULTS_2026-08-22.md` | Result record | Positive nonlinear-fusion Train-Val result, controls, subset/per-author metrics, feature contributions, limitations, and hashes. | Experiment B | COMPLETE |
| `docs/external_memory_next/14_TASK_SPECIFIC_BIENCODER_DESIGN_COST_GATE_2026-08-22.md` | Historical design/cost gate | Causal training/evaluation design, compute estimate, literature links, and the original evidence-based deferral before later explicit authorization. | Experiment C | COMPLETE RECORD |
| `docs/external_memory_next/15_TASK_SPECIFIC_BIENCODER_PREDECLARED_PROTOCOL_2026-08-22.md` | Frozen protocol | Exact inner split, model revision, loss, two-epoch gate, fixed-fusion evaluation, and conditional nonlinear-refit rule frozen before training. | Experiment C | FROZEN RECORD |
| `docs/external_memory_next/16_TASK_SPECIFIC_BIENCODER_RESULTS_2026-08-22.md` | Result record | Causal pair audit, training/checkpoint provenance, retrieval-positive/final-ranking-negative result, breakdowns, hashes, and commands. | Experiment C | COMPLETE |
| `experiments/external_memory_next/audit_phase0_evidence_v1.py` | Audit runner | Hash-gated Phase-0 distributions and initial frozen-metric check; accepts no Dev/Test path. | Phase 0 | ACTIVE |
| `experiments/external_memory_next/reproduce_full_retuned_baseline_v1.py` | Reproduction runner | Rebuilds frozen Stage-1/final ranking arithmetic and verifies all 34,416 rows. | Baseline gate | ACTIVE |
| `experiments/external_memory_next/run_choice_share_smoothing_v1.py` | Experiment runner | Original alpha grid for fixed-surface causal Choice Share smoothing. | Experiment A | ACTIVE |
| `experiments/external_memory_next/run_choice_share_smoothing_boundary_v2.py` | Boundary runner | Versioned alpha-boundary extension; selected alpha=128 as an interior point. | Experiment A | ACTIVE |
| `experiments/external_memory_next/run_smoothing_fusion_retune_v1.py` | Experiment runner | Sequential `w_CS` and Stage-2 lambda retune; reselected original coefficients. | Experiment A | ACTIVE |
| `experiments/external_memory_next/run_choice_share_prior_decomposition_v1.py` | Diagnostic runner | Fixed-surface raw-scale, zero-prior, all-author, and other-author comparison. | Experiment A | ACTIVE |
| `experiments/external_memory_next/run_train_fit_generic_v1.py` | Resumable GPU runner | Hash-pinned frozen PinyinGPT Generic generation for Clean3 Train-Fit only. | Experiment B input | ACTIVE |
| `experiments/external_memory_next/finalize_train_fit_generic_v1.py` | Validation runner | Verifies 144,526-row order, frozen decoding/revisions, CUDA runtime, and final cache checksums. | Experiment B input | ACTIVE |
| `experiments/external_memory_next/prepare_train_fit_ranking_features_v1.py` | Feature runner | Builds causal frozen Stage-1 Top10, then existing NGram/BGE supports, for Train-Fit learned-ranking groups. | Experiment B input | ACTIVE |
| `experiments/external_memory_next/audit_learned_fusion_inputs_v1.py` | Feature-audit runner | Runtime-only feature distributions, group/positive coverage, source composition, and zero-positive policy. | Experiment B input | ACTIVE |
| `experiments/external_memory_next/prepare_lambdamart_matrices_v1.py` | Matrix runner | Materializes compact author-free `X`, separate gold labels, query groups, and Val metadata after audit. | Experiment B input | ACTIVE |
| `experiments/external_memory_next/run_lambdamart_fusion_v1.py` | Experiment runner | Resumable additive-stump control and 12-point deterministic LambdaMART grid with exact external Macro selection. | Experiment B | ACTIVE |
| `experiments/external_memory_next/prepare_task_specific_biencoder_v1.py` | Audit/preparation runner | Verifies every frozen Train-Fit pair against chronology/author/Pinyin/label/context and materializes compact query-local groups. | Experiment C | ACTIVE |
| `experiments/external_memory_next/run_task_specific_biencoder_v1.py` | CUDA training runner | Bounded smoke, two-epoch inner Train-Fit gate, fresh all-Train-Fit refit, checkpoint hashing, and save/reload check. | Experiment C | ACTIVE |
| `experiments/external_memory_next/evaluate_task_specific_biencoder_v1.py` | Evaluation runner | Exact generic-BGE reconstruction plus one-shot Train-Val intrinsic and frozen fixed-fusion task-encoder evaluation. | Experiment C | ACTIVE |
| `src/personalisation/standardized_generic.py` | Runtime helper | Shape-safe deterministic bucketing, durable resume, and original-row-order restoration for frozen Generic generation. | Experiment B input | ACTIVE |
| `src/personalisation/standardized_reranking.py` | Runtime helper | Standardized query/candidate adapters imported byte-for-byte from the audited comparison worktree. | Experiment B input | ACTIVE |
| `src/personalisation/task_specific_biencoder.py` | Training/runtime helper | Frozen last64 serialization, masked mean pooling, group loss, chronological split, metrics, hashing, and checkpoint helpers. | Experiment C | ACTIVE |
| `tests/external_memory_next/` | Focused tests | Audit math, exact ranking arithmetic, smoothing, Generic bucketing/resume/Test rejection, empty-Generic no-op, matrix policy, LambdaMART selection/ties, bi-encoder split/pooling/selection/metrics, and closed-path rejection. | External Memory Next | ACTIVE |
| `tests/external_memory_next/test_task_specific_biencoder.py` | Focused regression tests | Group construction, chronological split, pooling, checkpoint selection, intrinsic metrics, and closed-resource path rejection. | Experiment C | ACTIVE |
| `results/personalisation/external_memory_next/task_specific_biencoder_v1/` | Generated experiment artifacts | Audited groups, checkpoints, vectors, predictions, metrics, hashes, and logs; intentionally excluded from the research commit. | Experiment C | GENERATED / LOCAL-ONLY |
| `.build/external_memory_next_biencoder/` | Ignored local model assets | Pinned full-precision base snapshot and other local model material required by the bi-encoder run; not portable or tracked. | Experiment C | LOCAL-ONLY / IGNORED |

Generated audit/result namespaces for this phase will remain under
`results/personalisation/external_memory_next/` and are local-only unless an
explicit later policy says otherwise. Dev3000 is excluded from design and
selection; Test remains closed.

## Repository governance and entry points

| Path | Type | Purpose and dependencies/outputs | Related stage | Status |
| --- | --- | --- | --- | --- |
| `README.md` | Repository entry point | High-level Deep Author setup and historical personalisation commands. This audit adds a narrow current-worktree scope notice and links to the policy/index documents. | Repository-wide | ACTIVE |
| `RESEARCH_TARGETS.md` | Research direction | Authoritative thesis targets and the boundary that HuoziIME is historical. | Thesis-wide | ACTIVE |
| `docs/FILE_MANAGEMENT_RULES.md` | Policy | Authoritative V1 file-management and lifecycle policy; currently untracked. | Repository governance | ACTIVE |
| `docs/FILE_INDEX.md` | Index | Canonical file/directory role and lifecycle inventory. | Repository governance | ACTIVE |
| `docs/REPRODUCIBILITY_INDEX.md` | Index | Canonical checkpoint reproduction inventory. | Repository governance | ACTIVE |
| `docs/VERSION_HISTORY.md` | Chronology | Navigation for historical Deep Author/personalisation checkpoints, including Context Diagnostic A and Context Strengthening. | Repository-wide | ACTIVE |
| `docs/TECHNICAL_HANDOFF.md` | Operator handoff | Detailed reranking-matrix worktree/environment snapshot from 2026-08-18; not the current Context Lab handoff. This audit adds a historical-scope notice without rewriting the snapshot. | Long-context matrix | LEGACY |
| `docs/technical_handoff_manifest.json` | Machine-readable handoff | Paths and versions for the reranking-matrix snapshot. | Long-context matrix | LEGACY |

## Reusable implementation under `src/`

| Path | Type | Purpose and dependencies/outputs | Related stage | Status |
| --- | --- | --- | --- | --- |
| `src/` and package `__init__.py` files | Package structure | Python package roots for datasets, evaluation, personalisation, and PinyinGPT backend. | Repository-wide | ACTIVE |
| `src/datasets/deep_author/pipeline.py` | Reusable dataset implementation | SCP-CN acquisition/provenance, cleaning, OpenCC normalization, tokenization, Pinyin alignment, interaction generation, hashes, and Dataset V1.1 build. Reads `config/deep_author/*`; writes local dataset/audit trees. | Dataset V1/V1.1 | FROZEN |
| `src/evaluation/deep_author_v2.py` | Evaluation implementation | Chronological work split, frozen anchors/conditions, T1 PinyinGPT inference, cache validation, metrics, and diagnostic exports. | Evaluation V2 / T1 | FROZEN |
| `src/evaluation/ranking.py` | Metrics helper | Model-independent Top-K/rank metrics used by evaluation code. | Evaluation | FROZEN |
| `src/reference_backend_pinyingpt/backend.py` | Model adapter | Pinned PinyinGPT2-Concat tokenizer/model loading, constrained beam decoding, batching, context fitting, and candidate scoring. Requires the local pinned checkpoint. | T1 and later Generic surfaces | FROZEN |
| `src/personalisation/context_memory.py` | Ranking implementation | Prediction-visible query schema; strict-prior same-user/same-Pinyin filtering; F and M1 formulas; retrieval, metrics, and Conflict definition with tied winners excluded. | Pilot A, Matrix, Context Lab | FROZEN |
| `src/personalisation/pilot_a.py` | Pilot orchestration library | Dev manifest construction, Generic cache, BGE embedding/cache, HistoryIndex, F/M1 tuning and evaluation. | Pilot A / Context Lab dependency | FROZEN |
| `src/personalisation/h5000.py` | Formal H5000 implementation | T1-aligned Full+Short F/M1 runner, frozen T1 hashes, cache reuse, Dev selection, Test evaluation. | F/M1 H5000 | FROZEN |
| `src/personalisation/candidate_memory_m2.py` | M2 implementation | Candidate-aware pair schema/template, balanced recent-context serialization, pair cache identity, pretrained reranker, and M2 ranking formula. | M2 | FROZEN |
| `src/personalisation/m2_h5000.py` | Formal M2 runner | Dev/Test pair scoring, cache/resume, Dev-only selection, and result generation over the frozen candidate surface. | M2 H5000 | FROZEN |
| `src/personalisation/personal_vocabulary.py` | PV implementation | Strict-prior personal lexicon, candidate recovery/merge, frequency and contextual support, transitions, and invariant helpers. | Personal Vocabulary | FROZEN |
| `src/personalisation/pv_h5000.py` | Formal PV runner | PV0/PV1/PV2 preparation, Dev state cache, tuning, Test evaluation, metrics and provenance validation. | Personal Vocabulary H5000 | FROZEN |
| `src/personalisation/reranking_matrix.py` | Matrix implementation | Resumable 36-cell F/M1/M2 matrix, condition-aware Dev Generic cache, BGE/M2 reuse, per-cell persistence, diagnostics and finalization. | v1 long-context matrix | DEFERRED / LEGACY |

## General experiment entry points

| Path | Type | Purpose and dependencies/outputs | Related stage | Status |
| --- | --- | --- | --- | --- |
| `experiments/prepare_deep_author_dataset.py` | Formal CLI | Runs `DeepAuthorBuilder`; produces local Dataset V1.1 data and audits. | Dataset V1.1 | FROZEN |
| `experiments/deep_author_evaluation_v2.py` | Formal CLI | `--phase design`, `t1`, or `metrics` for Evaluation V2. | Evaluation V2 / T1 | FROZEN |
| `experiments/exp_pinyingpt_reference.py` | Engineering smoke CLI | Small real-model PinyinGPT smoke; output is explicitly not a benchmark. | PinyinGPT integration | HELPER |
| `experiments/personalisation_pilot_a.py` | Formal exploratory CLI | Prepare/Generic/embedding/tune/evaluate/smoke/all phases for Dev Full+Short Pilot A. | Pilot A | LEGACY |
| `experiments/personalisation_pilot_a_h5000.py` | Formal CLI | Resumable F/M1 H5000 Dev selection and frozen T1 Test evaluation. | F/M1 H5000 | FROZEN |
| `experiments/personalisation_m2_h5000.py` | Formal CLI | Resumable M2 preparation, pair scoring, tuning and Test evaluation. | M2 H5000 | FROZEN |
| `experiments/personal_vocabulary_h5000.py` | Formal CLI | Resumable PV0/PV1/PV2 H5000 workflow. | Personal Vocabulary | FROZEN |
| `experiments/reranking_personalisation_matrix.py` | Formal CLI | Audit/smoke/run/finalize entry point for the old 36-cell long-context matrix. | v1 long-context matrix | DEFERRED / LEGACY |

## `experiments/context_lab/`

| Path | Type | Purpose and dependencies/outputs | Related stage | Status |
| --- | --- | --- | --- | --- |
| `experiments/context_lab/audit_dataset_v1_script.py` | One-off audit utility | Audits mixed/traditional-script characteristics of the inherited Dataset V1 artifacts; writes `script_audit_v1/`. | Context Lab data diagnosis | HELPER |
| `experiments/context_lab/diagnostic_a_retrieval.py` | Formal diagnostic runner | A1 retrieval audit/run for four T1 conditions, three exploratory authors, H5000. Reads frozen Test/work manifests, old matrix history manifests and the old BGE cache; writes A1 rows/metrics/summary. | Context Diagnostic A | FROZEN |
| `experiments/context_lab/diagnostic_a2_decision.py` | Formal read-only runner | Joins A1 with frozen F/M1/M2 predictions and computes transition/decision diagnostics. | Context Diagnostic A | FROZEN |
| `experiments/context_lab/diagnostic_a2b_evidence_competition.py` | Formal corrected read-only runner | Produces corrected A2b v2 evidence-competition cases. The earlier output namespace remains provenance only. | Context Diagnostic A | FROZEN |
| `experiments/context_lab/local_context_retrieval.py` | Exploratory Test diagnostic runner | Compares ctx16/ctx64 Test retrieval using separate BGE caches; Test observations were hypothesis-generating, not selection. | Context Strengthening | FROZEN |
| `experiments/context_lab/local_context_retrieval_dev.py` | Formal Dev representation runner | Full/64/16/8 local-context retrieval on tune/evaluation partitions; generates isolated embedding caches and metrics. | Context Strengthening | FROZEN |
| `experiments/context_lab/ctx64_m1_retune.py` | Formal Dev tuning runner | Tunes original M1 Top-N/lambda grid after ctx64 is fixed; consumes old Pilot manifests/Generic cache and ctx64 Dev caches. | Context Strengthening | FROZEN |
| `experiments/context_lab/ctx64_m1_retune_lambda8.py` | Formal boundary-check runner | Repeats Dev tuning with lambda 8 appended; confirms lambda 4 remains selected. | Context Strengthening | FROZEN |
| `experiments/context_lab/ctx64_m1_test.py` | Final Test runner | Applies frozen ctx64, H5000, Top-N 3, lambda 4 to 3,000 Full+Short Test rows. Depends on local Pilot manifests, ctx64 cache, and an untracked HFull matrix prediction artifact used as the Generic surface. | Context Strengthening | FROZEN |
| `experiments/context_lab/find_full_short_test_manifest.py` | Path-discovery utility | Recursively searches local result roots for a 3,000-anchor Test manifest with context/Pinyin; no formal semantics. Currently untracked. | Context Strengthening engineering | HELPER |
| `experiments/context_lab/generate_dev_evaluation_generic.py` | One-off generation utility | Hard-coded PilotRunner invocation for Dev-evaluation Generic rows in a separate local namespace. No current formal runner references that namespace. Currently untracked. | Context Strengthening engineering | HELPER |

## Configuration

| Path | Type | Purpose and dependencies/outputs | Related stage | Status |
| --- | --- | --- | --- | --- |
| `config/deep_author/authors_v1.json` | Frozen configuration | Six fixed proxy authors and source identities. | Dataset V1/V1.1 | FROZEN |
| `config/deep_author/run_config.yaml` | Frozen configuration | Dataset V1.1 acquisition, eligibility, cleaning, segmentation, context and audit parameters. | Dataset V1.1 | FROZEN |
| `config/deep_author/evaluation_v2.yaml` | Frozen configuration | Dataset V1 source/hash, chronological split, sampling, model revisions, beam and Top-10 evaluation settings. | Evaluation V2 / T1 | FROZEN |

## Research and report documentation

| Path | Type | Purpose and dependencies/outputs | Related stage | Status |
| --- | --- | --- | --- | --- |
| `docs/third_party/pinyingpt.md` | Third-party audit | Pinned PinyinGPT checkpoint/code provenance and compatibility notes. | Generic backend | FROZEN |
| `docs/reports/01_dataset_preparation.md` | Completed report | Dataset V1 result/provenance. | Dataset V1 | FROZEN |
| `docs/reports/01b_dataset_preparation_v1_1.md` | Completed report | Dataset V1.1 correction and audit. | Dataset V1.1 | FROZEN |
| `docs/reports/02_deep_author_evaluation_v2.md` | Design report | Frozen chronological Evaluation V2 protocol. | Evaluation V2 | FROZEN |
| `docs/reports/03_t1_generic_pinyingpt_baseline.md` | Completed report | 24,000-condition T1 Generic baseline results and runtime. | T1 | FROZEN |
| `docs/research/context_aware_personal_memory.md` | Method protocol | F/M1 definitions, information boundary, grids, BGE hash and H5000 semantics. | Pilot A / M1 | FROZEN |
| `docs/reports/04_personalisation_pilot_a_context_memory.md` | Completed report | Pilot A and formal F/M1 H5000 results. | F/M1 H5000 | FROZEN |
| `docs/research/candidate_aware_personal_memory_m2.md` | Method protocol | Candidate-aware M2 pair/scoring/cache design. | M2 | FROZEN |
| `docs/reports/05_personalisation_m2_h5000.md` | Completed report | M2 H5000 selection/result/provenance. | M2 | FROZEN |
| `docs/research/personal_vocabulary.md` | Method protocol | PV0/PV1/PV2 recovery and ranking design. | Personal Vocabulary | FROZEN |
| `docs/reports/06_personal_vocabulary_h5000.md` | Completed report | Personal Vocabulary H5000 results. | Personal Vocabulary | FROZEN |
| `docs/research/reranking_personalisation_matrix.md` | Method protocol | Frozen design and commands for the 36-cell matrix. | v1 long-context matrix | DEFERRED / LEGACY |
| `docs/reports/07_reranking_personalisation_matrix.md` | Incomplete report | Describes planned aggregate matrix outputs; the run was intentionally stopped before completion. | v1 long-context matrix | DEFERRED / LEGACY |
| `docs/CONTEXT_LAB_V1_PLAN.md` | Frozen plan | Original Context Personalisation Lab scope and decision boundaries. | Context Lab | FROZEN |
| `docs/context_lab/A1_SHORT_RESULTS_2026-08-18.md` | Intermediate report | Early A1 Short-only interpretation, superseded by the consolidated Diagnostic A report. | Context Diagnostic A | LEGACY |
| `docs/context_lab/SHORT_FOCUS_DECISION_2026-08-18.md` | Research decision | Historical decision to focus Short and defer Multi3; retained as provenance. | Context Lab | FROZEN |
| `docs/context_lab/CONTEXT_DIAGNOSTIC_A_REPORT_2026-08-18.md` | Completed report | Consolidated A1/A2/corrected-A2b evidence, metrics, limitations and commands. | Context Diagnostic A | FROZEN |
| `docs/context_lab/CONTEXT_STRENGTHENING_REPORT_2026-08-18.md` | Completed report | Dev window selection, M1 retune/boundary check, final Test result and conclusion. Tracked at the tagged checkpoint. | Context Strengthening | FROZEN |
| `docs/external_memory/EXTERNAL_MEMORY_COMPLETION_PLAN_2026-08-19.md` | Active plan | EM-1 recovery+F, EM-2 hidden-state kNN, EM-3 task-specific cross-encoder, EM-4 freeze. Currently untracked; no experiment was launched by this audit. | External Memory Completion | ACTIVE |
| `docs/script_normalisation/README.md` | Stage note | Script-normalisation audit context inherited from Dataset work. | Dataset quality | DEFERRED |

## Tests

| Path | Type | Purpose and dependencies/outputs | Related stage | Status |
| --- | --- | --- | --- | --- |
| `tests/test_deep_author_dataset.py` | Unit/invariant tests | Deterministic IDs/cleaning/segmentation, OpenCC, boundaries, raw immutability and no-inference guarantees. | Dataset V1.1 | FROZEN |
| `tests/test_deep_author_evaluation_v2.py` | Unit/invariant tests | Conditions, chronology, deterministic sampling, metric definitions and cache provenance. | Evaluation V2 | FROZEN |
| `tests/test_pinyingpt_reference.py` | Unit/optional real-model tests | Pinyin normalization, checkpoint errors, constrained generation/scoring, context fitting and batch equivalence. | Generic backend | FROZEN |
| `tests/test_personalisation_pilot_a.py` | Unit/integration tests | History visibility, F/M1 fallback/formulas, Conflict ties, Dev/Test separation and caches. | Pilot A | FROZEN |
| `tests/test_personalisation_h5000.py` | Invariant tests | Frozen 6,000-row T1 population/hash, H5000 ordering and read-only Generic reuse. | F/M1 H5000 | FROZEN |
| `tests/test_personalisation_m2.py` | Unit/invariant tests | M2 information boundary, pair identity/provenance, truncation, resume and candidate-pool invariants. | M2 | FROZEN |
| `tests/test_personal_vocabulary.py` | Unit/invariant tests | PV chronology, recovery/merge, F/PV formulas, grids, transitions and frozen prior hashes. | Personal Vocabulary | FROZEN |
| `tests/test_reranking_matrix.py` | Matrix regression tests | Batch-shape fix, resume/order, CUDA transition, cell states, shared formulas/cache identity and finalization. | v1 long-context matrix | DEFERRED / LEGACY |

## Important generated and local-only result roots

| Path | Type | Purpose and dependencies/outputs | Related stage | Status |
| --- | --- | --- | --- | --- |
| `results/evaluation/deep_author_v2/design/` | Design artifacts | Work split and 24,000-row T1 condition manifest. In this Context Lab workflow the authoritative copies are in the `thesis-deep-author` worktree. | Evaluation V2 | GENERATED / LOCAL-ONLY |
| `results/evaluation/deep_author_v2/t1/` | T1 result artifacts | Predictions, metrics, runtime and regression summaries; authoritative local cache resides in `thesis-deep-author`. | T1 | GENERATED / LOCAL-ONLY |
| `results/personalisation/pilot_a_context_memory/` | Pilot/M1 result root | History/Dev/Test manifests, Generic cache, BGE cache, F/M1 predictions and metrics; authoritative local copy resides in `thesis-personalisation`. | Pilot A / F/M1 | GENERATED / LOCAL-ONLY |
| `results/personalisation/m2_h5000/` | M2 result root | Pair-score cache, predictions, selections, metrics and logs in `thesis-personalisation`. | M2 | GENERATED / LOCAL-ONLY |
| `results/personalisation/personal_vocabulary_h5000/` | PV result root | PV state/predictions/selections/metrics in `thesis-personalisation`. | Personal Vocabulary | GENERATED / LOCAL-ONLY |
| `results/personalisation/reranking_matrix/` | Old matrix result root | Partial cells, manifests, caches and logs. Preserve; intentionally stopped and not a completed formal matrix result. | v1 long-context matrix | GENERATED / LOCAL-ONLY / DEFERRED |
| `results/personalisation/context_lab/script_audit_v1/` | Audit output | Dataset V1 script audit log. | Context Lab data diagnosis | GENERATED / LOCAL-ONLY |
| `results/personalisation/context_lab/diagnostic_a1_retrieval/` | Canonical A1 output | Cache audit, four condition row files/metrics, summary and logs. | Context Diagnostic A | GENERATED / LOCAL-ONLY |
| `results/personalisation/context_lab/diagnostic_a2_decision/` | Canonical A2 output | Joined row diagnostics and summary. | Context Diagnostic A | GENERATED / LOCAL-ONLY |
| `results/personalisation/context_lab/diagnostic_a2b_evidence_competition/` | Superseded output | First A2b output; retained because its rescue comparison was corrected later. | Context Diagnostic A | GENERATED / LOCAL-ONLY / LEGACY |
| `results/personalisation/context_lab/diagnostic_a2b_evidence_competition_v2/` | Canonical corrected A2b output | Strong regressions, unique rescues, summary and log. | Context Diagnostic A | GENERATED / LOCAL-ONLY |
| `results/personalisation/context_lab/local_context_retrieval/` | Exploratory Test outputs | ctx16/ctx64 Test retrieval rows, metrics, summaries and isolated SQLite caches. | Context Strengthening | GENERATED / LOCAL-ONLY |
| `results/personalisation/context_lab/local_context_retrieval_dev/` | Dev representation outputs | Tune Full/64/16/8 and evaluation ctx64 metrics/rows/caches. | Context Strengthening | GENERATED / LOCAL-ONLY |
| `results/personalisation/context_lab/ctx64_m1_retune/` | Dev selection output | Original-grid search and selected Top-N 3/lambda 4. | Context Strengthening | GENERATED / LOCAL-ONLY |
| `results/personalisation/context_lab/ctx64_m1_retune_lambda8/` | Boundary-check output | Extended-grid search confirming lambda 4. | Context Strengthening | GENERATED / LOCAL-ONLY |
| `results/personalisation/context_lab/ctx64_m1_test_h5000/` | Canonical final Test output | 3,000 predictions and `result.json` for frozen ctx64 M1. | Context Strengthening | GENERATED / LOCAL-ONLY |
| `results/personalisation/context_lab/generic_dev_evaluation/` | Helper-generated intermediate | Large alternate Pilot manifests and Dev-evaluation Generic cache; not referenced by the frozen Context Strengthening runners. Preserve pending cleanup review. | Context Strengthening engineering | GENERATED / LOCAL-ONLY / HELPER |
| `results/personalisation/context_lab/generic_dev_evaluation_3authors/` | Incomplete helper output | Contains only a history manifest/cache directory in the inspected tree; producer and intended consumer are not established. | Context Strengthening engineering | GENERATED / LOCAL-ONLY (purpose partly ambiguous) |

## Critical local dependencies outside this worktree

| Path | Type | Purpose and dependencies/outputs | Related stage | Status |
| --- | --- | --- | --- | --- |
| `C:/Users/chiar/Desktop/LBH/thesis-deep-author/.build/dataset-v1-reconstruction/` | Frozen dataset worktree | Exact Dataset V1 source used by Evaluation V2 and personalisation. | T1 onward | LOCAL-ONLY |
| `C:/Users/chiar/Desktop/LBH/thesis/.build/pinyingpt2-concat/` | Model checkpoint | Pinned PinyinGPT2-Concat files. | Generic inference | LOCAL-ONLY |
| `C:/Users/chiar/Desktop/LBH/thesis/.cache/phase_04f/models/bge-small-zh-v1.5-q8_0.gguf` | Model asset | Frozen BGE GGUF, SHA-256 `5a88d266870fbd27c6f329df60de80e2d4cf3bbd5e6f080bd5c1b2e5abb12039`. | M1 / Context Lab | LOCAL-ONLY |
| `C:/Users/chiar/Desktop/LBH/thesis/.build/bge-reranker-base/` | Model checkpoint | Pinned BAAI reranker used by M2. | M2 | LOCAL-ONLY |
| `.../pilot_a_context_memory/cache/embedding_cache.sqlite3` | Large cache | Original BGE vectors consumed by Diagnostic A; approximately 1.05 GB in the inspected local copy. | Diagnostic A | LOCAL-ONLY |
| `.../pilot_a_context_memory/cache/generic_predictions.jsonl` | Generic Dev cache | Consumed by ctx64 Dev tuning/evaluation; inspected SHA-256 `588aa84c6397e8cb1a13576c0d5dfecd9dd2c4305b45be351328dd83ef62007d`. | Context Strengthening | LOCAL-ONLY |
| `.../reranking_matrix/cells/full_short/HFull/M1/predictions.jsonl` | Frozen-surface dependency | `ctx64_m1_test.py` reads its Generic candidate/rank fields. Inspected SHA-256 `2fb513523674a790e1e5e7e1485a741fe4414709fee2ecd11dc9fea85793bcab`. | Context Strengthening | LOCAL-ONLY |

## Placement recommendations for later review

No files were moved during this audit. Under the policy, `find_full_short_test_manifest.py`, `generate_dev_evaluation_generic.py`, and probably `audit_dataset_v1_script.py` belong under `experiments/helpers/`. The helper-generated Generic directories should remain untouched until their provenance and consumers are reviewed. `TECHNICAL_HANDOFF.md` should eventually be replaced or clearly scoped to the matrix worktree; this audit does not rewrite it.


### EM-1 external memory completion

FROZEN / ACTIVE REFERENCE:

- `src/personalisation/external_memory.py`
  Reusable exact-scored recovery and recovery+frequency fusion logic.

- `experiments/external_memory/em1_score_compatibility.py`
  Fixed-score vs cached Generic score engineering compatibility gate.

- `experiments/external_memory/em1_recovery_coverage.py`
  Backend-compatible H5000 recovery coverage diagnostic.

- `experiments/external_memory/em1_gold_reachability.py`
  Dev Gold/backend reachability audit.

- `experiments/external_memory/em1_gold_reachability_test.py`
  Test-only aggregate Gold/backend reachability audit; not parameter tuning.

- `experiments/external_memory/em1_score_recovered_dev.py`
  Exact scoring of Dev recovered candidates.

- `experiments/external_memory/em1_dev_comparison.py`
  Dev G0/F/R/R+F comparison and frozen parameter selection.

- `experiments/external_memory/em1_score_recovered_test.py`
  Frozen K=1 three-author Test recovered-candidate exact scoring.

- `experiments/external_memory/em1_test_evaluation.py`
  Frozen three-author Test evaluation.

Documentation:

- `docs/external_memory/EM1_DEV_SELECTION_2026-08-19.md`
- `docs/external_memory/EM1_TEST_RESULT_2026-08-19.md`
- `docs/external_memory/EM1_REPRODUCIBILITY_2026-08-19.md`
- `docs/data_quality/KNOWN_ISSUES.md`

GENERATED / LOCAL:

- `results/personalisation/external_memory/`

Do not treat generated result files as the sole source of truth.

<!-- EM2-2026-08-19-INDEX -->
## External Memory EM-2 -2026-08-19

Canonical stage report:

- `docs/external_memory/em2/stages/EM2_PROGRESS_REPORT_2026-08-19.md` -ACTIVE stage report covering hidden-state engineering validation, hidden kNN retrieval, Hidden-M1, Hidden-M2, and fixed G+F+C.

Formal experiment runners:

- `experiments/external_memory/em2_hidden_state_gate.py`
- `experiments/external_memory/em2_cache_hidden_dev.py`
- `experiments/external_memory/em2_hidden_knn_dev.py`
- `experiments/external_memory/em2_hidden_m1_dev.py`
- `experiments/external_memory/em2_hidden_m1_dev_boundary8.py`
- `experiments/external_memory/em2_hidden_m2_dev.py`
- `experiments/external_memory/em2_original_m2_same_surface_dev.py`
- `experiments/external_memory/em2_fixed_gfc_dev.py`

Helper/comparison:

- `experiments/external_memory/em2_four_way_dev_compare.py` -HELPER / comparison-only.

Generated local result roots:

- `results/personalisation/external_memory/em2_hidden_dev/`
- `results/personalisation/external_memory/em2_hidden_m1_dev/`
- `results/personalisation/external_memory/em2_hidden_m1_dev_boundary8/`
- `results/personalisation/external_memory/em2_original_m2_dev/`
- `results/personalisation/external_memory/em2_hidden_m2_dev/`
- `results/personalisation/external_memory/em2_fixed_gfc_dev/`

Generated result trees and SQLite caches are GENERATED / LOCAL-ONLY and should not be Git-added as normal source artifacts.



<!-- EM2-FINAL-CLOSE-2026-08-19 -->
## EM-2 final closure

Status: **FROZEN / CLOSED**

Canonical EM-2 files:

- `docs/external_memory/em2/EM2_FINAL_REPORT_2026-08-19.md` - final Dev-stage scientific report and closure.
- `docs/external_memory/em2/EM2_REPRODUCIBILITY_2026-08-19.md` - canonical commands, dependencies, hashes, and expected checkpoints.
- `docs/external_memory/em2/EM2_TO_EM3_HANDOFF_2026-08-19.md` - handoff to EM-3.
- `docs/external_memory/em2/stages/` - retained process/design/diagnostic documentation.

Formal runners:

- `experiments/external_memory/em2_hidden_state_gate.py`
- `experiments/external_memory/em2_cache_hidden_dev.py`
- `experiments/external_memory/em2_hidden_knn_dev.py`
- `experiments/external_memory/em2_hidden_m1_dev.py`
- `experiments/external_memory/em2_hidden_m1_dev_boundary8.py`
- `experiments/external_memory/em2_original_m2_same_surface_dev.py`
- `experiments/external_memory/em2_hidden_m2_dev.py`
- `experiments/external_memory/em2_fixed_gfc_dev.py`
- `experiments/external_memory/em2_adaptive_gfc_dev.py`

Helper/comparison:

- `experiments/external_memory/em2_four_way_dev_compare.py` - HELPER / same-surface comparison.

Generated result/caches remain under `results/personalisation/external_memory/` and are GENERATED / LOCAL-ONLY.

No new EM-2 Test result was opened.

<!-- EM1_PV_AUDIT_20260819_START -->
## Post-hoc PV1/PV2 vs EM-1 same-surface audit

- `docs/external_memory/EM1_PV_COMPARISON_ADDENDUM_2026-08-19.md`
  - Status: FROZEN / POST-HOC EXPLANATORY AUDIT.
  - Records the aligned 3,000-row PV1/PV2 vs EM-1 comparison, candidate-identity audit,
    interpretation boundary, and provenance.
- `experiments/external_memory/em1_pv_same_surface_audit.py`
  - Status: ACTIVE REPRODUCIBILITY HELPER.
  - Read-only audit runner. No parameter tuning, model training, or new Test inference.
- Generated evidence under
  `results/personalisation/external_memory/em1_pv_same_surface_audit/`
  is GENERATED / LOCAL-ONLY and must not be committed as normal source.
<!-- EM1_PV_AUDIT_20260819_END -->

<!-- EM3-DEV-CHECKPOINT-20260820 -->
## EM3 Dev-analysis checkpoint — 2026-08-20

Status: **ACTIVE DEV CHECKPOINT / PAUSED BEFORE HEAVY TRAINING**. This is not a
final EM3 method freeze. Benchmark Test remains closed.

### Canonical navigation and documentation

| Path | Role | Status |
|---|---|---|
| `docs/CURRENT_RESEARCH_INDEX_2026-08-20.md` | Canonical first-read worktree, state, provenance, and resume index | ACTIVE |
| `docs/external_memory/em3/EM3_PROGRESS_2026-08-20.md` | Concise EM3-BCE and final Dev-audit progress checkpoint | ACTIVE |
| `docs/external_memory/em3/EM3_V2_EXECUTION_PLAN_2026-08-20.md` | Preserved v2 execution plan; closeout controls current resume order | ACTIVE PLANNING |
| `docs/external_memory/em3/EM3_V2_METHOD_OPTIONS_2026-08-20.md` | Literature-backed method options; not a frozen method | ACTIVE RESEARCH NOTE |
| `docs/external_memory/em3/EM3_V2_DATA_PREPARATION_2026-08-20.md` | Frozen clean3 author/data policy | ACTIVE |
| `docs/external_memory/em3/EM3_V2_FAILURE_AUDIT_2026-08-20.md` | Preliminary and final full-surface failure taxonomy | ACTIVE |
| `docs/external_memory/em3/EM3_ALL_OUTCOME_DISTRIBUTION_RECORD_2026-08-20.md` | Complete 5,608-row G/F/Hidden-M1 distributions and provenance | CANONICAL DIAGNOSTIC |
| `docs/external_memory/em3/EM3_DEV_CLOSEOUT_2026-08-20.md` | Canonical pause/handoff note and exact resume order | ACTIVE |

### Canonical and provenance runners

| Path | Role | Status |
|---|---|---|
| `experiments/external_memory/em3_generate_train_pairs.py` | Deterministic causal pair generator; old-v1 count regression and future clean3 generation | ACTIVE |
| `experiments/external_memory/em3_all_outcome_audit.py` | Consolidated 5,608-row G/F/Hidden-M1 audit | CANONICAL DIAGNOSTIC |
| `experiments/external_memory/em3_context_failure_examples.py` | Preliminary 124/59 example extraction | PROVENANCE HELPER |
| `experiments/external_memory/em3_context_failure_cluster_audit.py` | Failure clustering/diversity audit | PROVENANCE HELPER |
| `experiments/external_memory/em3_context_outcome_examples.py` | Focused outcome-group examples | PROVENANCE HELPER |

### Generated local-only evidence

- `results/personalisation/external_memory/em3_all_outcome_audit/`
- `results/personalisation/external_memory/em3_train_pairs_v1/`
- `results/personalisation/external_memory/em3_train_pairs_v1_regression_audit/`
- `results/personalisation/external_memory/em3_train_population_audit/`
- `results/personalisation/external_memory/em3_dev_population_audit/`
- `results/personalisation/external_memory/em3_bce_v1_final_dev_tune/`
- `results/personalisation/external_memory/em3_hidden_dev/`

All result trees, JSONL, SQLite, logs, caches, embeddings, checkpoints, and
generated models remain GENERATED / LOCAL-ONLY and must not be staged as normal
source artifacts.
<!-- EM3-DEV-CHECKPOINT-20260820-END -->
---

---

<!-- CONTEXT-COMPARISON-PREP-20260820 -->
## Context-model horizontal comparison preparation - 2026-08-20

| Path | Type | Purpose and dependencies/outputs | Related stage | Status |
|---|---|---|---|---|
| `docs/context_comparison/01_CONTEXT_COMPARISON_PROTOCOL_2026-08-20.md` | Protocol | Five routes, Clean3 Full+Short/H5000 surface, cache-blind sampling, metrics, cache/Test/freeze policy. | Phases 0-2 | ACTIVE / FROZEN PROTOCOL |
| `docs/context_comparison/02_CONTEXT_COMPARISON_ARTIFACT_AUDIT_2026-08-20.md` | Research record | Exact model/cache/result inventory, compatibility decisions, registry, and frozen-3000 coverage. | Phases 3 and 6 | ACTIVE |
| `docs/context_comparison/05_CONTEXT_COMPARISON_DATASET_RECORD_2026-08-20.md` | Dataset record | Clean3 Train/Dev accounting, identity bridge, deterministic balanced-3000 composition and SHA. | Phases 4-5 | FROZEN |
| `docs/context_comparison/07_CONTEXT_COMPARISON_DB_RECORD_2026-08-20.md` | Database record | Local DB schema, joins/counts, oracle/runtime separation, prediction/missing coverage, regeneration. | Phase 7 | ACTIVE |
| `experiments/context_comparison/prepare_context_comparison.py` | Reproducibility runner | Reads canonical/Pilot/EM3 Dev manifests and immutable caches; freezes rows before coverage; writes registry/audits/local DB. No inference. | Phases 4-7 | ACTIVE |
| `experiments/context_comparison/__init__.py` | Package structure | Package marker for comparison preparation workflows. | Context comparison | ACTIVE |
| `tests/context_comparison/test_prepare_context_comparison.py` | Test | Namespace-safe bridge, deterministic cache-blind balance, accounting with/without partition fields. | Validation | ACTIVE |
| `results/personalisation/context_comparison_v1/clean3_history_balanced_3000.jsonl` | Generated manifest | Frozen 3,000-row Clean3 Dev identity/analysis surface. | Phase 5 | GENERATED / LOCAL-ONLY / DO NOT STAGE |
| `results/personalisation/context_comparison_v1/clean3_history_balanced_3000_audit.json` | Generated audit | Source hashes, bridge checks, parent/eligible/sample accounting, seed/rule, manifest SHA. | Phase 5 | GENERATED / LOCAL-ONLY / DO NOT STAGE |
| `results/personalisation/context_comparison_v1/model_registry.json` | Generated registry | Exact five-route semantics, checkpoints, K/lambda, history/candidate/cache provenance. | Phase 3 | GENERATED / LOCAL-ONLY / DO NOT STAGE |
| `results/personalisation/context_comparison_v1/cache_coverage.json` | Generated audit | Row/model direct, reconstructable, partial, inference-required, unresolved and per-author counts. | Phase 6 | GENERATED / LOCAL-ONLY / DO NOT STAGE |
| `results/personalisation/context_comparison_v1/dev_context_eval_v1.sqlite` | Generated database | 3,000 interactions/subsets/runtime/oracle rows, five models, 3,984 predictions, 15,000 coverage states. | Phase 7 | GENERATED / LOCAL-ONLY / DO NOT STAGE |

No Test data, model inference, GPU job, training, or score-driven sampling was
used. The inherited 5,608 surface is a legacy regression surface, not Clean3.
<!-- CONTEXT-COMPARISON-PREP-20260820-END -->

<!-- STANDARDIZED-CONTEXT-RESET-20260820 -->
## Standardized context-model reset — active 2026-08-20

| Path | Role | Status |
|---|---|---|
| `docs/context_comparison/10_STANDARDIZED_RESET_PLAN_2026-08-20.md` | Authoritative standardized Train-Fit/Train-Val/Dev3000 plan | ACTIVE / FROZEN PROTOCOL |
| `docs/context_comparison/09_STANDARDIZED_RESET_DECISION_LOG_2026-08-20.md` | Scientific decisions and rejected alternatives | ACTIVE |
| `docs/context_comparison/11_STANDARDIZED_RESET_EXECUTION_LOG_2026-08-20.md` | Exact commands, environment, failures/reruns, outputs | ACTIVE |
| `docs/context_comparison/06_TRAIN_VAL_SPLIT_RECORD_2026-08-20.md` | Frozen whole-work split, counts, hashes | FROZEN |
| `docs/context_comparison/04_HISTORY_SEMANTICS_RECORD_2026-08-20.md` | Rolling causal H5000 resolution and Dev audit | FROZEN |
| `docs/context_comparison/12_MODEL_RETUNE_REGISTRY_2026-08-20.md` | Model identities, grids, tie breaks, EM3 recipe | ACTIVE / PRE-RESULT FROZEN |
| `docs/context_comparison/03_WORKLOAD_CACHE_AUDIT_2026-08-20.md` | Exact representation reuse/miss audit | ACTIVE |
| `src/personalisation/standardized_context_comparison.py` | Split/history/evaluator invariants | ACTIVE |
| `src/personalisation/standardized_generic.py` | Shape-safe resumable frozen Generic orchestration | ACTIVE |
| `experiments/context_comparison/prepare_standardized_reset.py` | Versioned split/registry/history/regression preparation | ACTIVE |
| `experiments/context_comparison/audit_standardized_workload.py` | Exact Train-Val representation workload audit | ACTIVE |
| `experiments/context_comparison/run_standardized_generic.py` | Resumable frozen Generic Train-Val runner | ACTIVE |
| `experiments/context_comparison/run_standardized_hidden.py` | Resumable frozen PinyinGPT hidden-state runner | ACTIVE |
| `experiments/context_comparison/fill_standardized_bge.py` | Read-only-base BGE cache copy/overlay miss filler | ACTIVE |
| `experiments/context_comparison/train_standardized_em3.py` | SHA-checked historical-recipe EM3 trainer wrapper | ACTIVE |
| `tests/context_comparison/` | Evaluator, split, history, and resumability invariants | ACTIVE |
| `results/personalisation/context_comparison_v2/` | Manifests, caches, logs, checkpoints, registries, results | GENERATED / LOCAL-ONLY / DO NOT STAGE |

Dev3000 remains frozen at SHA256 `9181f895...b03f93`; Test is closed. No
generated v2 JSONL, SQLite, logs, embeddings, or checkpoints should be staged.
<!-- STANDARDIZED-CONTEXT-RESET-20260820-END -->

### Historical history-depth provenance

- `docs/context_comparison/08_HISTORICAL_HISTORY_DEPTH_PROVENANCE_2026-08-20.md`
  - Recovered historical Full+Short H500/H5000/HFull results for Frequency, M1, and M2.
  - Records original result paths, selected hyperparameters, Test-selection safety, history semantics, interpretation limits, and SHA256 provenance.
- Historical Test evidence only; does not modify the current standardized H5000 protocol.

<!-- STANDARDIZED-COMPARISON-FINAL-20260821 -->
## Standardized context-model comparison — completed 2026-08-21

| Path | Role | Status |
|---|---|---|
| `docs/context_comparison/15_CONTEXT_COMPARISON_COMPLETION_REPORT_2026-08-22.md` | Consolidated protocol, implementation, results, provenance, and limitations report | COMPLETED |
| `docs/context_comparison/13_PRE_DEV_FREEZE_2026-08-21.md` | Human-readable frozen Train-Val selections and provenance | FROZEN |
| `docs/context_comparison/14_STANDARDIZED_DEV3000_RESULT_2026-08-21.md` | Canonical sealed Dev3000 result and diagnostics | COMPLETED |
| `src/personalisation/standardized_reranking.py` | Exact pair registries, score caches, and standardized reranking | ACTIVE |
| `experiments/context_comparison/run_standardized_rerankers.py` | Resumable Stage-1/pair-scoring/Train-Val orchestration | ACTIVE |
| `experiments/context_comparison/run_standardized_dev3000.py` | Freeze-gated sealed Dev3000 orchestration | ACTIVE |
| `experiments/context_comparison/finalize_standardized_comparison.py` | Machine/human result finalization | ACTIVE |
| `results/personalisation/context_comparison_v2/pre_dev_freeze_v1.json` | Machine-readable selection/model/config freeze | GENERATED / LOCAL-ONLY / DO NOT STAGE |
| `results/personalisation/context_comparison_v2/dev3000/` | Sealed Dev predictions, caches, checksums, and result | GENERATED / LOCAL-ONLY / DO NOT STAGE |

The standardized comparison is complete on Dev3000. All selection was confined
to Train-Val; `used_test=false`; Test remains closed.
<!-- STANDARDIZED-COMPARISON-FINAL-20260821-END -->

---

## Initial-Pinyin Personalisation

### Placement and maintenance rule

```text
All numbered Initial-Pinyin research documents belong under:
  docs/initial_personalisation/

This includes 01_..., 02_..., ..., 19_..., and decimal-numbered records such as 13.1_....

Repository-wide living indexes remain at docs/ root:
  docs/FILE_INDEX.md
  docs/REPRODUCIBILITY_INDEX.md

These two index files are updated in place. Their filenames are stable; Git history carries revisions.
```

### Canonical current documentation

```text
docs/initial_personalisation/19_INITIAL_RECOVERY_CONTEXT_TRAINVAL_REPRODUCIBILITY_2026-08-21.md
  Canonical reproduction record for the final Train-Val Recovery -> NGramRecency -> BGERecency activity,
  including V1/V2/V3, diagnosis, Top-k transitions, hashes, commands, regressions, and safety invariants.

docs/initial_personalisation/18_INITIAL_RECOVERY_CONTEXT_TRAINVAL_FINAL_CONCLUSIONS_2026-08-21.md
  Canonical data/conclusion record for the completed Train-Val recovery + context activity.
  This is the main standalone record of the new result numbers and diagnosis.

docs/initial_personalisation/17_INITIAL_PV1_CONTEXT_RERANKING_RESULTS_AND_REPRODUCIBILITY_2026-08-21.md
  Historical PV1 fixed-candidate context-reranking checkpoint used as the control line for later recovery+context work.

docs/initial_personalisation/16_INITIAL_REPRODUCIBILITY_2026-08-21_v4.md
  Earlier broad Initial+Short reproducibility record through candidate scoring and Stage-1 recovery development.

docs/initial_personalisation/15_INITIAL_PERSONALISATION_RECOVERY_REPRODUCIBILITY_2026-08-21.md
  Earlier compact recovery/controllability reproducibility record.

docs/initial_personalisation/14_INITIAL_PERSONALISATION_FILE_INDEX_2026-08-21.md
  Historical compact Initial-Pinyin phase-local file map; repository-wide current navigation is this FILE_INDEX.md.

docs/initial_personalisation/13.1_INITIAL_PERSONALISATION_RECOVERY_METHOD_COMPARISON_AND_REVISED_CONCLUSIONS_2026-08-21_v2.md
  Stage-1 recovery-method comparison and revised conclusions before Stage-2 context fusion.

docs/initial_personalisation/13_INITIAL_PERSONALISATION_CURRENT_CONCLUSIONS_AND_CONTROLLABILITY.md
  Earlier current-conclusion checkpoint; superseded by the later 18/19 activity and the living current-conclusions file.

docs/initial_personalisation/12_INITIAL_PERSONALISATION_METRIC_PURPOSE_AND_PRIORITIES.md
  Metric purpose and priority: primary, secondary, recovery, and diagnostic metrics.

docs/initial_personalisation/11_INITIAL_PERSONALISATION_EVALUATION_METRICS_AND_DESIGN.md
  Metric definitions, calculation formulas, evaluation populations, and design rationale.
```

The earlier numbered `01_` through `10_` Initial-Pinyin research/design/diagnostic records remain in the same `docs/initial_personalisation/` directory as provenance. Do not move numbered Initial-Pinyin documents back to `docs/` root.

### Living current-conclusions record

```text
docs/initial_personalisation/INITIAL_PERSONALISATION_CURRENT_CONCLUSIONS_AND_CONTROLLABILITY_v2.md
  Current cross-phase Initial-Pinyin conclusions and controllability interpretation after the final Train-Val context activity.
```

This file is unnumbered because it is a living synthesis rather than a chronological experiment record.

### Current final Train-Val recovery + context runners

```text
experiments/initial_personalisation/run_initial_recovery_ngram_context_fusion_v1.py
  Stage-1 recovery bases + Stage-2 NGramRecency grid.
  SHA256: e6dcd1f68028ad5065064b6b714eaa88d92f74363a328570bfcc777b13271dc2

experiments/initial_personalisation/run_initial_recovery_bge_ngram_context_fusion_v2.py
  Two-dimensional NGramRecency + BGERecency grid.
  SHA256: b7d95374aa421cbc364699e44e0850ba2e72e50a2a5f816ad37f85b138d1435a

experiments/initial_personalisation/run_initial_recovery_bge_ngram_context_fusion_v3.py
  Expanded lambda_B boundary verification; arithmetic-only reuse of completed support.
  SHA256: 2b29a86957b4f2adf17a13de37648766e1423d0ec99a57ea257c5aa155d89335

experiments/initial_personalisation/run_initial_recovery_context_diagnostics_v1.py
  Read-only final Train-Val diagnosis.
  SHA256: 7c4a12a5f447405f024d8e8008253da23aab4775d2ae4500f5c44545583d3256

experiments/initial_personalisation/run_initial_recovery_context_topk_transitions_v1.py
  Read-only Top1/Top3/Top5 rescue-harm transition diagnosis.
  SHA256: 3966111844719f29a07b580a10d18021b0cdf4a6846c71157de611e1a92eaef1
```

### Current final Train-Val result roots

```text
results/personalisation/initial_recovery_comparison_v1/recovery_ngram_context_fusion_v1/
  NGram-only context fusion support and selected predictions.

results/personalisation/initial_recovery_comparison_v1/recovery_bge_ngram_context_fusion_v2/
  Original NGram+BGE two-dimensional grid and BGE support/cache products.

results/personalisation/initial_recovery_comparison_v1/recovery_bge_ngram_context_fusion_v3/
  Canonical final selected full-context operating points after expanded-boundary verification.
  Durable files:
    grid_results.csv
    selected_metrics.csv
    selected_predictions.jsonl
    full_comparison.csv
    comparison.json
    run_manifest.json
    artifact_checksums.json

results/personalisation/initial_recovery_comparison_v1/recovery_context_diagnostics_v1/
  Post-hoc read-only diagnostic outputs:
    diagnostic_summary.json
    headline_comparison.csv
    per_author.csv
    subset_metrics.csv
    top1_transitions.csv
    rank_movement.csv
    recovery_diagnostics.csv
    context_increment.csv
    base_disagreement.csv
    margin_diagnostics.csv
    error_examples.jsonl
    diagnostic_report.md
    run_manifest.json
    artifact_checksums.json

results/personalisation/initial_recovery_comparison_v1/recovery_context_topk_transitions_v1/
  Top-k transition outputs:
    topk_transitions.csv
    topk_transitions.json
```

### Current frozen Train-Val operating points

```text
Primary overall development selection:
  4P+4CS+2E + NGramRecency(lambda_N=4) + BGERecency(lambda_B=6)
  Macro=.437058  Micro=.460571  Top3=.631392  Top5=.696478
  MRR=.559755  Missing=.243172
  Recovery: Rec1=.4246 Rec3=.6969 Rec5=.8153 Rec10=.9485 RecMRR=.5892

Coverage-oriented comparison:
  K5+Entropy + NGramRecency(lambda_N=6) + BGERecency(lambda_B=8)
  Macro=.436767  Micro=.459990  Top3=.626453  Top5=.688139
  MRR=.557836  Missing=.243288
  Recovery: Rec1=.4430 Rec3=.7481 Rec5=.8778 Rec10=.9876 RecMRR=.6218

Front-rank comparison:
  6P+2CS+.25E + NGramRecency(lambda_N=4) + BGERecency(lambda_B=6)
  Macro=.436477  Micro=.459786  Top3=.630085  Top5=.696013
  MRR=.558806  Missing=.243869
  Recovery: Rec1=.4415 Rec3=.6961 Rec5=.8069 Rec10=.9283 RecMRR=.5951
```

Development interpretation: the balanced model is the Train-Val selection under the pre-specified Macro-author Top1 criterion, but the gap to K5+Entropy is very small. Do not claim significance or generalization before holdout evaluation.

### Latest diagnostic checkpoints

```text
Primary 4P+4CS+2E context increments:
  Recovery -> NG-R: Delta Macro = +.027644; Top1 rescue=2375 harm=1468 net=+907
  NG-R -> Full:     Delta Macro = +.004607; Top1 rescue=681  harm=514  net=+167

Top3 rescue/harm on recoverable R=4910, Recovery -> Full:
  K5+Entropy:    rescue=745 harm=80 net=+665
  4P+4CS+2E:     rescue=745 harm=20 net=+725
  6P+2CS+.25E:   rescue=617 harm=16 net=+601

Per-author primary full-context Top1 / Missing@10:
  Agent Phage: Top1=.493923 Missing=.171312
  Etinjat:     Top1=.275218 Missing=.485056
  breaddddd:   Top1=.542032 Missing=.167655
```

Etinjat is substantially harder on this Initial-Pinyin surface, primarily because its candidate-set missing rate is much higher. Context still improves Etinjat from Stage-1 Top1 `.237235` to full-context `.275218`; do not infer a causal reason for the author difference without a dedicated upstream data/history audit.

### Protocol state

```text
Train-Val development: complete
V3 upper-boundary check: passed
Post-hoc diagnosis: complete
Dev3000 used: false
Test used: false
Next formal step: PRE-DEV FREEZE -> Dev3000
```

All generated result trees remain GENERATED / LOCAL-ONLY unless separately designated for tracking. Preserve versioned historical experiment outputs; do not overwrite them.

---

<!-- FULL-TRANSFER-INITIAL-FINAL-20260822 -->
## Full+Short zero-shot Initial-final transfer follow-up — 2026-08-22

This block records a **post-Dev follow-up** performed after completion of the standardized seven-system context comparison. It does not reopen the frozen Train-Val method selection and is not an eighth pre-Dev comparison method. `used_dev3000=false`, `used_test=false`, and no Full-specific hyperparameter search was performed.

| Path | Type | Purpose and dependencies/outputs | Related stage | Status |
|---|---|---|---|---|
| `experiments/context_comparison/run_full_transfer_initial_final_v1.py` | Follow-up experiment runner | Zero-shot transfer of the frozen Initial-Pinyin `4P+4CS+2E + NG-R4 + BGE-R6` architecture to standardized Full+Short Train-Val. Rebuilds causal Personal K5 from Full history, preserves H5000-before-Pinyin semantics, performs Stage1 recovery then fixed-surface NG-R/BGE-R reranking. Runner SHA256 `f75d40f381e966f85cd4b20647ba7dc6a95df9116ad8657ca9a07505949a37b0`. | Post-Dev Full transfer follow-up | FROZEN |
| `docs/context_comparison/16_FULL_TRANSFER_INITIAL_FINAL_TRAINVAL_2026-08-22.md` | Research/result record | Full experimental purpose, transferred formulas/parameters, overall/Ambiguous/Conflict/per-author/recovery/transition data, interpretation limits, and exact reproduction command. SHA256 `228e4c404ae8a369831ac1f0fe1bfd79cf2cdf91a1972b788057bfddb69884bc`. | Post-Dev Full transfer follow-up | FROZEN |
| `results/personalisation/context_comparison_followup_v1/full_transfer_initial_final_v1/result.json` | Generated result | Canonical machine-readable Full Train-Val result. SHA256 `604a74d212ff16954b09f375a8db88f527cc07d12333fab0a7c18a7f712743a3`. | Follow-up evaluation | GENERATED / LOCAL-ONLY / DO NOT STAGE |
| `results/personalisation/context_comparison_followup_v1/full_transfer_initial_final_v1/run_setup.json` | Generated setup/provenance | Frozen transferred parameters, paths, hashes, history policy, and `used_dev3000=false`, `used_test=false`. SHA256 `28ee66721e4ffcdad82f141d763c537e3fdcc60ca7faa4c0e2e2ed82c27e69e1`. | Follow-up reproducibility | GENERATED / LOCAL-ONLY / DO NOT STAGE |
| `results/personalisation/context_comparison_followup_v1/full_transfer_initial_final_v1/stage1_predictions.jsonl` | Generated predictions | Stage1 `4P+4CS+2E` recovered candidate surface and ranking. SHA256 `eacc6c37c53e581bc667483eb6b29816cc81c3239aad9c16acf16788611ec53f`. | Follow-up Stage1 | GENERATED / LOCAL-ONLY / DO NOT STAGE |
| `results/personalisation/context_comparison_followup_v1/full_transfer_initial_final_v1/final_predictions.jsonl` | Generated predictions | Final `Stage1 + NG-R4 + BGE-R6` ranking. SHA256 `fcc9b44c06fe0dc7bf629ad81d79476a4d27e52be934a37f4ac9c9d8d293973d`. | Follow-up Final | GENERATED / LOCAL-ONLY / DO NOT STAGE |
| `results/personalisation/context_comparison_followup_v1/full_transfer_initial_final_v1/artifact_checksums.json` | Generated checksum record | Hashes for runner and canonical follow-up artifacts plus `used_dev3000=false` and `used_test=false`. | Follow-up provenance | GENERATED / LOCAL-ONLY / DO NOT STAGE |
| `results/personalisation/context_comparison_followup_v1/full_transfer_initial_final_v1/bge_context_cache.sqlite3` | Generated cache | Candidate-conditioned BGERecency context cache; 42,278 required unique contexts and 42,278 stored rows in the completed run. | Follow-up BGERecency | GENERATED / LOCAL-ONLY / DO NOT STAGE |

Headline descriptive Train-Val result:

- Generic Macro Top1 `0.700795`;
- Frequency `0.776875`;
- M1 `0.778239`;
- Stage1 `0.788523`;
- Final `0.795367`;
- Final Missing@10 `0.051953` versus `0.069212` for Generic/Frequency/M1;
- Final Ambiguous Macro Top1 `0.802172`;
- Final Conflict Macro Top1 `0.228973`;
- Stage1 -> Final Conflict Macro Top1 improvement `+7.618 pp`, but Final remains below M1 and Generic at Conflict Top1.

Scientific interpretation: this result is descriptive evidence that the frozen Initial recovery-plus-recency architecture transfers to Full+Short without retuning. It must not be presented as a statistically established superiority claim or as part of the already frozen seven-system Dev comparison.
<!-- FULL-TRANSFER-INITIAL-FINAL-20260822-END -->

<!-- FULL-RETUNED-FINAL-DEV-CLOSEOUT-20260822 -->
## Full-retuned Final Train-Val selection + Dev3000 closeout — 2026-08-22

| Path | Role | Status |
|---|---|---|
| `docs/context_comparison/17_FULL_RETUNED_FINAL_DEV3000_CLOSEOUT_2026-08-22.md` | Canonical Full-specific retuning protocol, selected configuration, Dev3000 horizontal comparison, transition accounting, provenance, and freeze decision | COMPLETED / DEVELOPMENT CLOSED |
| `experiments/context_comparison/run_full_retune_final_trainval_dev_v1.py` | Two-phase runner: Full Train-Val-only sequential weight selection followed by frozen Dev3000 evaluation; SHA256 `89d526cb61d3bb93a1caa3d401679db9f1f8b8efdc31d4daa4590adcce3dee8d` | FROZEN DEVELOPMENT RUNNER |
| `results/personalisation/context_comparison_followup_v1/full_retune_final_trainval_dev_v1/selected_config.json` | Machine-readable Full Train-Val selected Stage1/Stage2 configuration | GENERATED / LOCAL-ONLY / DO NOT STAGE |
| `results/personalisation/context_comparison_followup_v1/full_retune_final_trainval_dev_v1/tune/train_val_result.json` | Full Train-Val grid-selection result and selected predictions provenance | GENERATED / LOCAL-ONLY / DO NOT STAGE |
| `results/personalisation/context_comparison_followup_v1/full_retune_final_trainval_dev_v1/dev/dev_result.json` | Frozen selected configuration on Dev3000 with horizontal metrics and rescue/harm transitions | GENERATED / LOCAL-ONLY / DO NOT STAGE |

Selected weights: `w_P=2.0`, `w_CS=6.0`, `w_E=4.0`, `lambda_N=6.0`, `lambda_B=6.0`.

Dev headline: RetunedFinal Macro/Micro Top1 `0.843667`, Top3 `0.934333`, MRR@10 `0.892041`, Missing@10 `0.029000`. It is the highest observed Dev system on Top1, Top3, and MRR among the horizontal comparators; no significance claim is made.

Train-Val performs parameter selection; Dev3000 is a development comparison surface; Test remains CLOSED. Generated result trees remain local-only and must not be staged.
<!-- FULL-RETUNED-FINAL-DEV-CLOSEOUT-20260822-END -->
