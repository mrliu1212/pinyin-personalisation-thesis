# Repository File Index

Purpose: answer **what each meaningful file or directory is for, and what its current status is**. This is a static inventory of `work/external-memory-completion` at commit `a9a9351c85fe7f40f17c5232e5f77b6c84e7b35c`, inspected on 2026-08-19. It follows [FILE_MANAGEMENT_RULES.md](FILE_MANAGEMENT_RULES.md).

Statuses are `ACTIVE`, `FROZEN`, `HELPER`, `LEGACY`, `DEFERRED`, `GENERATED`, and `LOCAL-ONLY`. Compound statuses are used where a generated artifact is also intentionally local-only. No status authorizes deletion.

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
