# Reproducibility Index

Purpose: answer **how a frozen checkpoint could be rerun later**. This is a static evidence audit, not a record of reruns performed on 2026-08-19. Commands appear only where preserved CLI/source/report evidence establishes them.

Use a separate worktree for historical checkpoints; do not move the active worktree backwards:

```powershell
git worktree add --detach C:\Users\chiar\Desktop\LBH\thesis-reproduce-<name> <tag>
```

Status meanings follow [FILE_MANAGEMENT_RULES.md](FILE_MANAGEMENT_RULES.md): `COMPLETE`, `PARTIAL`, `RESULT-ONLY`, `LOCAL-ARTIFACT-DEPENDENT`, and `LEGACY`.

## Current environment and shared local dependencies

The preserved Windows commands use:

```powershell
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$dataset = 'C:\Users\chiar\Desktop\LBH\thesis-deep-author\.build\dataset-v1-reconstruction'
$pinyingpt = 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat'
$bge = 'C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf'
$reranker = 'C:\Users\chiar\Desktop\LBH\thesis\.build\bge-reranker-base'
$t1 = 'C:\Users\chiar\Desktop\LBH\thesis-deep-author\results\evaluation\deep_author_v2\t1\predictions.jsonl'
```

Pinned provenance shared by later checkpoints:

- PinyinGPT2-Concat: `aihijo/transformers4ime-pinyingpt-concat`, revision `76dd20dc92d8236a350fb732e99dde6fa15e2263`; official code reference `8f1573ed0bd4d1f3d8d3f10a05f7e870725646f1`.
- Frozen T1 condition-manifest normalized SHA-256: `45b9cafedd7a8269d1f0b66d3f7f135ee990140e4b5b3668c67645863ab00d39`.
- Frozen T1 predictions SHA-256: `764db39887f3db04b913d1739d9dbd46295f0e46e5a2bffa649f1563b56ee4e2`.
- BGE GGUF SHA-256: `5a88d266870fbd27c6f329df60de80e2d4cf3bbd5e6f080bd5c1b2e5abb12039`.
- M2 model: `BAAI/bge-reranker-base`, revision `2cfc18c9415c912f9d8155881c133215df768a70`.

## Checkpoint summary

| Checkpoint | Tag / exact commit | Reproducibility status | Current HEAD suitable? |
| --- | --- | --- | --- |
| Deep Author Dataset V1 | `deep-author-dataset-preparation-v1` / `d886b6558fab2898fe2deba5113b8490fd87ac04` | COMPLETE | No; use tag |
| Deep Author Dataset V1.1 | `deep-author-dataset-preparation-v1.1` / `d871f1f8d1b524389decbe535e5ec1e26f501ff4` | COMPLETE | Current code descends from it, but use tag for exact semantics |
| Evaluation V2 Design | `deep-author-evaluation-v2-design` / `b145f2d0037f55abda071ee025f6adca2381c765` | LOCAL-ARTIFACT-DEPENDENT | No; use tag plus frozen V1 artifact |
| Evaluation V2 T1 Generic Baseline | `deep-author-evaluation-v2-t1` / `14d584a17c4ae0a284b25bcdc892d3b12e439745` | LOCAL-ARTIFACT-DEPENDENT | Use tag for exact runner; local prediction cache is preserved |
| Personalisation Pilot A implementation | `personalisation-pilot-a-implementation-v1` / `4f217fe2028abfd071ebe333bb5d06eae45ed201` | LEGACY | No; superseded by formal H5000 runner |
| Frequency / M1 H5000 | `personalisation-pilot-a-h5000-implementation-v1` / `7d9f15ae3857c27c9d87e3e0acc6d0850eec1598` | LOCAL-ARTIFACT-DEPENDENT | Yes for code, but use tag and local inputs for exact provenance |
| M2 implementation | `personalisation-m2-h5000-implementation-v1` / `dd0753ba6d04a6110a765ff8bc19c4a1b19e01e1` | LEGACY | No; use completed-result tag |
| M2 completed result | `personalisation-m2-h5000-result-v1` / `fb7abcaf71446ca4f9c7e4e8992222e7bc2d6072` | LOCAL-ARTIFACT-DEPENDENT | Code is present; local M1/BGE/pair artifacts remain dependencies |
| Personal Vocabulary implementation | `personal-vocabulary-h5000-implementation-v1` / `483aa73efc56a7316e65692d05cb817da2dae13f` | LEGACY | No; use completed-result tag |
| Personal Vocabulary completed result | `personal-vocabulary-h5000-result-v1` / `a83757e607b4bb5b89cb00143e159ea07aaa3941` | LOCAL-ARTIFACT-DEPENDENT | Code is present; local prior artifacts remain dependencies |
| Reranking Personalisation Matrix | `reranking-personalisation-matrix-implementation-v1` / `4e2c6ea529fa7fb38b419e178640720d1fc8af3b` | PARTIAL | Do not resume as current research; use historical branch/tag only |
| Context Diagnostic A | `context-diagnostic-a-complete-20260818` / `54f05b76fa7b553baec62e260b1c74ed72a83e0f` | LOCAL-ARTIFACT-DEPENDENT | Yes at tag/current HEAD, with preserved external artifacts |
| Context Strengthening | `context-strengthening-complete-20260819` / `a9a9351c85fe7f40f17c5232e5f77b6c84e7b35c` | LOCAL-ARTIFACT-DEPENDENT | Yes; current HEAD is the tagged implementation |
| External Memory Completion | no completed checkpoint; active untracked plan | PARTIAL | Planning only; no formal runner/result exists |

## 1. Deep Author Dataset V1

- **Purpose:** initial six-author SCP-CN corpus and contextual interaction build.
- **Inputs:** public source pages/files selected by `config/deep_author/authors_v1.json` and the V1 configuration frozen at the tag.
- **Implementation:** `src/datasets/deep_author/pipeline.py`; `experiments/prepare_deep_author_dataset.py`.
- **Exact command at the tag:**

```powershell
Set-Location C:\Users\chiar\Desktop\LBH\thesis-reproduce-dataset-v1
& $python -m experiments.prepare_deep_author_dataset
```

- **Outputs:** `data/raw/deep_author/`, `data/processed/deep_author/`, and dataset audit/result metadata described in `docs/reports/01_dataset_preparation.md` at the tag.
- **Cache:** not required, but network/source availability affects reacquisition. Immutable raw hashes are recorded.
- **Status rationale:** the historical implementation, config, report and command are preserved. Use the tag because current `pipeline.py` implements V1.1 semantics.

## 2. Deep Author Dataset V1.1

- **Purpose:** targeted cleaning correction with source-confirmed credit removal, Simplified-Chinese normalization, and hard non-Han interaction boundaries.
- **Inputs/model:** same public corpus; OpenCC `t2s.json`; Pinyin conversion dependencies from `requirements-deep-author.txt` at the tag.
- **Implementation/config:** `src/datasets/deep_author/pipeline.py`, `config/deep_author/run_config.yaml`, `config/deep_author/authors_v1.json`.
- **Exact command:**

```powershell
Set-Location C:\Users\chiar\Desktop\LBH\thesis-reproduce-dataset-v1-1
& $python -m experiments.prepare_deep_author_dataset
```

- **Outputs/report:** local raw/processed/audit trees; `docs/reports/01b_dataset_preparation_v1_1.md`.
- **Cache:** optional; immutable raw files are reusable if their recorded identities match.
- **Status rationale:** exact code/config/entry point are preserved. The report records that no evaluation was run as part of this correction.

## 3. Evaluation V2 Design

- **Purpose:** freeze six-author chronological History/Dev/Test works and 6,000 anchors expanded to 24,000 Full/Initial × Short/Multi3 conditions.
- **Required input:** exact Dataset V1 artifact, 2,048,557,493 bytes, SHA-256 `8d1a98e18a5f7ed997930b65bbd1149c3d52daaa22ac2c59771256a966648da2`.
- **Implementation/config:** `src/evaluation/deep_author_v2.py`, `config/deep_author/evaluation_v2.yaml`, tag-specific CLI that enables only design.
- **Exact command:**

```powershell
Set-Location C:\Users\chiar\Desktop\LBH\thesis-reproduce-evaluation-v2-design
& $python -m experiments.deep_author_evaluation_v2 --phase design
```

- **Expected outputs:** `results/evaluation/deep_author_v2/design/`, including `work_split_manifest.csv` and `t1_condition_manifest.jsonl`.
- **Cache:** the exact local Dataset V1 source is required. Preserve `thesis-deep-author/.build/dataset-v1-reconstruction`.
- **Status rationale:** the path is established, but exact reproduction depends on the large local frozen Dataset V1 artifact.

## 4. Evaluation V2 T1 Generic PinyinGPT Baseline

- **Purpose:** generic PinyinGPT2-Concat baseline on all 24,000 frozen conditions.
- **Inputs:** frozen design and pinned PinyinGPT checkpoint above.
- **Implementation:** `src/reference_backend_pinyingpt/backend.py`, `src/evaluation/deep_author_v2.py`, `experiments/deep_author_evaluation_v2.py` at the tag.
- **Exact full/resume command and metrics-only command:**

```powershell
Set-Location C:\Users\chiar\Desktop\LBH\thesis-reproduce-t1
& $python -m experiments.deep_author_evaluation_v2 --phase t1
& $python -m experiments.deep_author_evaluation_v2 --phase metrics
```

- **Outputs:** `results/evaluation/deep_author_v2/t1/predictions.jsonl`, `metrics_summary.json`, runtime/regression/cache validation and checksums.
- **Headline:** Macro-author Top-1 `0.3752083`; predictions contain 24,000 rows and hash to `764db398...ee4e2`.
- **Cache:** reusable/resumable predictions are an important local artifact; preserve the `thesis-deep-author` result tree.
- **Status rationale:** exact runner is preserved, but the frozen design, checkpoint and durable prediction cache are large local artifacts.

## 5. Personalisation Pilot A and formal F/M1 H5000

### Pilot A implementation — LEGACY

The exploratory Dev-only runner is `experiments/personalisation_pilot_a.py`; it established strict chronology, F, M1, BGE caching, and Dev separation. It is superseded for formal T1 comparison by the H5000 runner. The CLI phases are preserved, but this checkpoint should not be presented as the completed H5000 Test result.

### F/M1 H5000 completed local result — LOCAL-ARTIFACT-DEPENDENT

- **Purpose:** reuse 6,000 frozen Full+Short Test Generic rows, tune F/M1 on earlier Dev, and evaluate H5000.
- **Inputs/models:** `$dataset`, `$pinyingpt`, `$bge`, `$t1`.
- **Implementation:** `src/personalisation/context_memory.py`, `pilot_a.py`, `h5000.py`; CLI `experiments/personalisation_pilot_a_h5000.py`.
- **Exact full/resume command:**

```powershell
Set-Location C:\Users\chiar\Desktop\LBH\thesis-personalisation
& $python -m experiments.personalisation_pilot_a_h5000 --phase all `
  --dataset-root $dataset --pinyingpt-model $pinyingpt `
  --embedding-model $bge --t1-predictions $t1
```

- **Outputs:** `results/personalisation/pilot_a_context_memory/h5000/` plus shared manifests/caches under its parent.
- **Headline Macro-author Top-1:** G0 `0.7231667`, F `0.7718333`, M1 `0.7675000`.
- **Provenance:** completed metrics SHA-256 `e35fb9efbe3bdd31d7f8354c227efbed2aa178855061955b3ac16a70137e424d`.
- **Cache:** BGE and Dev Generic caches are reusable and expensive; local result/caches are required for later Context Lab diagnostics.

## 6. M2 H5000

### Implementation checkpoint — LEGACY

`personalisation-m2-h5000-implementation-v1` preserves the pre-result implementation. Use the completed-result tag for reproduction of the reported checkpoint.

### Completed result — LOCAL-ARTIFACT-DEPENDENT

- **Purpose:** candidate-aware pretrained cross-encoder reranking after unchanged BGE Stage-1 retrieval.
- **Inputs/models:** completed M1 artifacts, `$dataset`, `$pinyingpt`, `$bge`, `$reranker`, `$t1`.
- **Implementation:** `candidate_memory_m2.py`, `m2_h5000.py`; CLI `experiments/personalisation_m2_h5000.py`.
- **Exact full/resume command:**

```powershell
Set-Location C:\Users\chiar\Desktop\LBH\thesis-personalisation
& $python -m experiments.personalisation_m2_h5000 --phase all `
  --dataset-root $dataset --pinyingpt-model $pinyingpt `
  --embedding-model $bge --reranker-model $reranker --t1-predictions $t1
```

- **Outputs:** `results/personalisation/m2_h5000/`; canonical files include `m2_predictions.jsonl`, `metrics_summary.json`, `selected_hyperparameters.json`, `hyperparameter_search.csv`, checksums, and `cache/pair_scores.sqlite3`.
- **Headline:** Overall Macro-author Top-1 `0.765`; candidate surface and Missing@10 unchanged.
- **Cache:** pair-score SQLite and M1/BGE inputs are reusable and costly. Preserve them.

## 7. Personal Vocabulary H5000

### Implementation checkpoint — LEGACY

`personal-vocabulary-h5000-implementation-v1` is superseded by the completed-result tag.

### Completed result — LOCAL-ARTIFACT-DEPENDENT

- **Purpose:** PV0 recoverability plus PV1 frequency injection and PV2 context support over frozen prior results.
- **Inputs:** completed T1/M1/M2 artifacts and their hashes; `$dataset`, `$pinyingpt`, `$bge`, `$t1`.
- **Implementation:** `personal_vocabulary.py`, `pv_h5000.py`; CLI `experiments/personal_vocabulary_h5000.py`.
- **Exact full/resume command:**

```powershell
Set-Location C:\Users\chiar\Desktop\LBH\thesis-personalisation
& $python -m experiments.personal_vocabulary_h5000 --phase all `
  --dataset-root $dataset --pinyingpt-model $pinyingpt `
  --embedding-model $bge --t1-predictions $t1
```

- **Outputs:** `results/personalisation/personal_vocabulary_h5000/`.
- **Headline:** 160/538 Generic-missing targets PV0-recoverable; PV1 Macro-author Top-1 `0.779`, Missing@10 `0.0661667`; PV2 context net Top-1 help `-5` versus PV1.
- **Cache:** prior results and BGE cache are reusable; local-only states/predictions must be preserved.

## 8. Reranking Personalisation Matrix — PARTIAL

- **Purpose:** planned 4 conditions × H500/H5000/HFull × F/M1/M2 matrix with Dev selection and cache reuse.
- **Identity:** implementation tag above; later fixes at `2318dbb`, `4c95c54`, and operator docs at `617a20f`. Archival tag `personalisation-v1-long-context-matrix` points to `617a20f`.
- **Implementation/CLI:** `src/personalisation/reranking_matrix.py`; `experiments/reranking_personalisation_matrix.py`.
- **Preserved command:**

```powershell
Set-Location C:\Users\chiar\Desktop\LBH\thesis-personalisation
$common = @('--dataset-root',$dataset,'--pinyingpt-model',$pinyingpt,
  '--embedding-model',$bge,'--reranker-model',$reranker,'--t1-predictions',$t1)
& $python -m experiments.reranking_personalisation_matrix --phase audit @common
& $python -m experiments.reranking_personalisation_matrix --phase run @common
```

- **Outputs:** `results/personalisation/reranking_matrix/`, including manifest, partial cells, caches and logs.
- **Status rationale:** the 36-cell result did not complete and was intentionally stopped. Do not infer aggregate results from partial cells and do not resume it as the current research direction.
- **Cache:** all Generic/BGE/M2 caches and partial cells are important local-only evidence.

## 9. Context Diagnostic A — LOCAL-ARTIFACT-DEPENDENT

- **Purpose:** determine whether contextual failure arises from missing history, retrieval, or evidence/decision competition.
- **Population:** Full+Short/H5000, 3 exploratory authors, 3,000 Test anchors for the consolidated A2/A2b analysis; A1 also recorded the other three conditions.
- **Inputs:** frozen Test/work manifests; old matrix history manifests; original Pilot BGE cache; local F/M1/M2 predictions.
- **Implementation:** the three `diagnostic_a*.py` files listed in [FILE_INDEX.md](FILE_INDEX.md).
- **Exact commands from the frozen report:**

```powershell
Set-Location C:\Users\chiar\Desktop\LBH\thesis-context-lab
& $python experiments\context_lab\diagnostic_a_retrieval.py --phase audit
& $python experiments\context_lab\diagnostic_a_retrieval.py --phase run
& $python experiments\context_lab\diagnostic_a2_decision.py
& $python experiments\context_lab\diagnostic_a2b_evidence_competition.py
```

- **Outputs:** `diagnostic_a1_retrieval/`, `diagnostic_a2_decision/`, and corrected `diagnostic_a2b_evidence_competition_v2/` under the Context Lab result root.
- **Headline:** Full+Short Gold-history retrieval R@1 `85.22%`; F micro Top-1 `81.07%`, M1 `79.83%`, M2 `80.10%`; unique contextual rescues M1 `4`, M2 `2`; strong regressions M1 `45`, M2 `33`.
- **Local provenance:** inspected A1 summary SHA-256 `bd22268f9f32bb4e6b7a181a0d9cc6e79565b33f6a9c69ee5abbc9fceac50704`.
- **Cache:** the A1 audit does not generate missing vectors; the original local BGE cache is required.

## 10. Context Strengthening — LOCAL-ARTIFACT-DEPENDENT

- **Purpose:** test local personal-memory context windows, select representation on Dev, retune M1, and run the frozen 3-author Test configuration.
- **Population:** Full+Short/H5000; Etinjat, Re_spectators, breaddddd.
- **Models:** frozen Generic surface unchanged; BGE GGUF only for personal-memory embeddings.
- **Implementation:** `local_context_retrieval.py`, `local_context_retrieval_dev.py`, `ctx64_m1_retune.py`, `ctx64_m1_retune_lambda8.py`, `ctx64_m1_test.py`.
- **Reproduction sequence established by the frozen scripts:**

```powershell
Set-Location C:\Users\chiar\Desktop\LBH\thesis-context-lab

# Exploratory Test retrieval diagnostics; not parameter selection.
& $python experiments\context_lab\local_context_retrieval.py --window 16 --phase all --model $bge
& $python experiments\context_lab\local_context_retrieval.py --window 64 --phase all --model $bge

# Dev-tune representation comparison.
foreach ($window in 'full','64','16','8') {
  & $python experiments\context_lab\local_context_retrieval_dev.py `
    --window $window --partition tune --phase all --model $bge
}

# Frozen ctx64 check on later Dev evaluation.
& $python experiments\context_lab\local_context_retrieval_dev.py `
  --window 64 --partition evaluation --phase all --model $bge

# M1 Dev tuning and boundary check.
& $python experiments\context_lab\ctx64_m1_retune.py --phase all
& $python experiments\context_lab\ctx64_m1_retune_lambda8.py --phase all

# Final frozen Test runner.
& $python experiments\context_lab\ctx64_m1_test.py
```

- **Outputs:** the five Context Strengthening result roots indexed in `FILE_INDEX.md`.
- **Selected configuration:** ctx64 characters, H5000, Top-N 3, lambda 4; the lambda-8 boundary check selected the same point.
- **Headline Test result:** 3,000 rows; ctx64 M1 Overall Macro-author Top-1 `0.7976667` versus Generic `0.776`; Frequency remains stronger at micro `0.8107` versus ctx64 M1 `0.7977`. Conflict Macro-author Top-1 is `0.2515` for ctx64 M1 versus Generic `0.5357`.
- **Local hashes:** `ctx64_m1_retune/selected_hyperparameters.json` `9d8927d6...702d`; lambda-8 selection `82eaddb8...3f9a`; final `result.json` `c832f7d1...e219`.
- **Critical cache/input status:** local ctx64 SQLite caches are reusable. The final Test script also requires `pilot_a_context_memory/h5000/test_manifest.jsonl`, `history_manifest.jsonl`, and `reranking_matrix/cells/full_short/HFull/M1/predictions.jsonl`. The last file is an untracked partial-matrix artifact, hash `2fb51352...bcab`, and must be preserved.
- **Discrepancy:** the completed report contains no reproduction section; the sequence above is derived from the tagged frozen CLIs. The helper `generate_dev_evaluation_generic.py` writes an alternate namespace not referenced by the frozen runners, so it is not included in the canonical sequence.

## 11. External Memory Completion — PARTIAL

- **Purpose:** active planned sequence EM-1 recovery+F, EM-2 PinyinGPT hidden-state kNN, EM-3 IME-specific cross-encoder, EM-4 final fusion/freeze.
- **Evidence:** untracked `docs/external_memory/EXTERNAL_MEMORY_COMPLETION_PLAN_2026-08-19.md`.
- **Missing:** no source implementation, formal CLI, config, result namespace manifest, command, tag, or result exists yet.
- **Status rationale:** this is a plan, not a reproducible checkpoint. No command is documented because inventing one would violate provenance rules.

## Additional historical formal checkpoints

These tags are genuine preserved checkpoints but are not the current research line.

| Checkpoint(s) | Tag(s) / evidence | Status | Reproduction note |
| --- | --- | --- | --- |
| IME Simulator v0.1/v0.2 | `ime-simulator-v0.1` (`6750bee`), `ime-simulator-v0.2` (`ad69543`) | PARTIAL | Use separate tagged worktrees; tag-specific tool/research docs and code are preserved, but this audit did not establish one exact end-to-end launch/reproduction command. |
| LiveChat Generic Baseline v1 | `livechat-generic-baseline-v1` (`c594e2a`) | LOCAL-ARTIFACT-DEPENDENT | Frozen baseline code/tag exists; large LiveChat data/predictions are local. Inspect the tag in a separate worktree before running. |
| Historical HuoziIME Phases 3–4E | `phase-03`, `phase-04a`, `phase-04b`, `phase-04b6`, `phase-04b7`, `phase-04c-setup`, `phase-04c`, `phase-04c-complete`, `phase-04d-setup`, `phase-04d`, `phase-04e-implementation` | LEGACY | Superseded thesis direction. Tag-specific `docs/phases/`, configs, runners and reports must be used; current PinyinGPT/personalisation code must not reinterpret them. |
| HuoziIME Phase 4F | `phase-04f` (`3e0cde2`) | PARTIAL | The tag contains implementation, audit artifacts, backend manifest and smoke test, but the code-defined final `results/experiments/phase_04f/evaluation.json` is absent from the tag/current history. Do not infer final benchmark numbers from smoke output. |
| Phase 4F Windows compatibility | `phase-04f-windows-compat` (`d23e9a7`) | LEGACY | Platform/build compatibility checkpoint. It preserves Windows smoke output separately and does not create the missing Phase 4F final evaluation artifact. |

## Known documentation and implementation discrepancies

1. `docs/VERSION_HISTORY.md` does not yet list Context Diagnostic A or Context Strengthening despite both having annotated tags.
2. The prior `README.md` opening described only the earlier non-personalised baseline; this audit adds a current-worktree scope notice rather than rewriting historical instructions.
3. `docs/TECHNICAL_HANDOFF.md` describes `work/reranking-matrix`, old PIDs and matrix operations, not the active External Memory branch; this audit adds a historical-scope notice.
4. Context Strengthening has a completed report but no report-level reproduction section; exact commands had to be reconstructed from tagged CLIs.
5. `ctx64_m1_test.py` labels an HFull M1 matrix prediction file as its `generic_source`. It reads Generic fields from that file, but the dependency is scientifically non-obvious and local-only.
6. `generate_dev_evaluation_generic.py` writes `generic_dev_evaluation/`, while the frozen retune scripts read the old Pilot Generic cache. No consumer of the helper namespace was found.
7. HuoziIME Phase 4F's commit message claims completion, but its code-defined final evaluation artifact does not survive in Git.

## Preservation priorities

Do not delete the following before a separate archival review:

- the Dataset V1 reconstruction worktree and frozen Evaluation V2 design/T1 predictions;
- PinyinGPT, BGE GGUF and BGE reranker model directories;
- Pilot history/Dev/Test manifests and Generic/BGE caches;
- M2 pair-score cache and completed predictions;
- Personal Vocabulary states/results;
- all partial long-context matrix cells/manifests/logs/caches;
- all Context Lab result directories, especially corrected A2b v2, Dev ctx64 caches, frozen selections, final predictions/result;
- `reranking_matrix/cells/full_short/HFull/M1/predictions.jsonl`, because Context Strengthening reproduction currently depends on it.
