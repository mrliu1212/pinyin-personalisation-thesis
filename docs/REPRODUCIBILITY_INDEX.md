# Reproducibility Index

Purpose: answer **how a frozen checkpoint could be rerun later**. This is a static evidence audit, not a record of reruns performed on 2026-08-19. Commands appear only where preserved CLI/source/report evidence establishes them.

**Maintenance:** this is a living reproducibility index. Update `docs/REPRODUCIBILITY_INDEX.md` in place when a new checkpoint or reproduction record is added; do not fork routine updates into dated/versioned index filenames. Git history records revisions.

Use a separate worktree for historical checkpoints; do not move the active worktree backwards:

```powershell
git worktree add --detach C:\Users\chiar\Desktop\LBH\thesis-reproduce-<name> <tag>
```

Status meanings follow [FILE_MANAGEMENT_RULES.md](FILE_MANAGEMENT_RULES.md): `COMPLETE`, `PARTIAL`, `RESULT-ONLY`, `LOCAL-ARTIFACT-DEPENDENT`, and `LEGACY`.

## External Memory Next — Experiments A/B/C complete

- Worktree: `C:\Users\chiar\Desktop\LBH\thesis-external-memory-next`.
- Branch: `work/external-memory-next`.
- Base: `fb09ca2fa50589a0fc72130552212c5b47ed4365`.
- Read first: `docs/external_memory_next/00_READ_FIRST.md`.
- Base/provenance record:
  `docs/external_memory_next/01_BASE_AND_PROVENANCE_2026-08-22.md`.
- Phase 0 evidence audit:
  `docs/external_memory_next/02_PHASE0_EVIDENCE_AUDIT_2026-08-22.md`.
- Exact baseline gate:
  `docs/external_memory_next/03_FULL_RETUNED_BASELINE_REPRODUCTION_2026-08-22.md`.
- Choice Share smoothing design/result:
  `docs/external_memory_next/04_CHOICE_SHARE_SMOOTHING_DESIGN_2026-08-22.md` and
  `docs/external_memory_next/05_CHOICE_SHARE_SMOOTHING_FIXED_SURFACE_RESULTS_2026-08-22.md`.
- Smoothing coefficient follow-up:
  `docs/external_memory_next/06_SMOOTHING_FUSION_RETUNE_DESIGN_2026-08-22.md` and
  `docs/external_memory_next/07_SMOOTHING_FUSION_RETUNE_RESULTS_2026-08-22.md`.
- Learned-fusion data gate:
  `docs/external_memory_next/08_NONLINEAR_FUSION_READINESS_AND_DATA_PLAN_2026-08-22.md`.
- Learned-fusion input/result records:
  `docs/external_memory_next/12_LEARNED_FUSION_INPUT_GATE_2026-08-22.md` and
  `docs/external_memory_next/13_LAMBDAMART_FUSION_RESULTS_2026-08-22.md`.
- Task-specific bi-encoder design/cost gate:
  `docs/external_memory_next/14_TASK_SPECIFIC_BIENCODER_DESIGN_COST_GATE_2026-08-22.md`.
- Task-specific bi-encoder frozen protocol and completed result:
  `docs/external_memory_next/15_TASK_SPECIFIC_BIENCODER_PREDECLARED_PROTOCOL_2026-08-22.md`
  and `docs/external_memory_next/16_TASK_SPECIFIC_BIENCODER_RESULTS_2026-08-22.md`.
- Development boundary: Train-Fit fitting and Train-Val selection only;
  Dev3000 is already observed and excluded from design/selection; Test closed.
- Frozen Full RetunedFinal reproduction: **EXACT**, 34,416 candidate orders and
  ranks, Macro-author Top1 `.7960049265502147`.
- Smoothing-only selection: alpha `128`, Macro-author Top1
  `.7965154987791901`; `25` rescues, `9` harms, net `+16`; all historical
  fusion coefficients were reselected unchanged.
- Mechanism decomposition: raw `w_CS=2`, zero-prior shrinkage, all-author
  shrinkage, and other-author shrinkage all produce net `+15` or `+16` Top1;
  prior-specific added value is weak. See records `09_...` and `10_...`.
- LambdaMART selection: depth `5`, leaves `31`, min leaf `500`, rounds `100`;
  Macro-author Top1 `.7988390633366215`, `267/171/+96` rescue/harm/net versus
  frozen RetunedFinal. Dev3000/Test remained unused.
- Task bi-encoder: BAAI revision
  `7999e1d3359715c523056ef9478215996d62a620`, final checkpoint SHA256
  `f9b87af11fcff692ad7c25fb6330f44f9f23ffedb480af9aec36af0e7cd08a8e`;
  intrinsic Macro Recall@1 `.8109711910595357` versus generic BGE
  `.7789437773409569`, but fixed-fusion Macro Top1 `.7957117243433173`
  versus frozen `.7960049265502147`. The nonlinear-refit gate failed.
- Exact Experiment C Windows commands are in record 16, section 8. Generated
  groups, checkpoints, vectors, predictions, results, and logs are local-only.
- Current reproducibility status: **EXPERIMENTS A/B/C COMPLETE / LOCAL-ARTIFACT-DEPENDENT**.

Current Windows input-generation commands (run from the isolated worktree):

```powershell
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$compare = 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_v2'
$followup = 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune'
$next = '.\results\personalisation\external_memory_next'

New-Item -ItemType Directory -Force "$next\train_fit_generic_v1" | Out-Null

& $python -m experiments.external_memory_next.run_train_fit_generic_v1 `
  --manifest "$compare\clean3_train_fit_v1.jsonl" `
  --checkpoint 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat' `
  --output "$next\train_fit_generic_v1\predictions.jsonl" `
  --batch-size 2 `
  2> "$next\train_fit_generic_v1\stderr.log" `
  | Tee-Object -FilePath "$next\train_fit_generic_v1\stdout.log"

& $python -m experiments.external_memory_next.finalize_train_fit_generic_v1 `
  --fit "$compare\clean3_train_fit_v1.jsonl" `
  --predictions "$next\train_fit_generic_v1\predictions.jsonl" `
  --stdout "$next\train_fit_generic_v1\stdout.log" `
  --generator '.\experiments\external_memory_next\run_train_fit_generic_v1.py' `
  --helper '.\src\personalisation\standardized_generic.py' `
  --output-root "$next\train_fit_generic_v1"

& $python -m experiments.external_memory_next.prepare_train_fit_ranking_features_v1 `
  --phase stage1 `
  --fit "$compare\clean3_train_fit_v1.jsonl" `
  --generic "$next\train_fit_generic_v1\predictions.jsonl" `
  --checkpoint 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat' `
  --output-root "$next\train_fit_ranking_features_v1"

& $python -m experiments.external_memory_next.prepare_train_fit_ranking_features_v1 `
  --phase supports `
  --fit "$compare\clean3_train_fit_v1.jsonl" `
  --bge-model 'C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf' `
  --seed-bge-cache 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\bge_context_cache.sqlite3' `
  --cuda-path 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8' `
  --output-root "$next\train_fit_ranking_features_v1"

& $python -m experiments.external_memory_next.audit_learned_fusion_inputs_v1 `
  --fit-supports "$next\train_fit_ranking_features_v1\train_fit_candidate_supports.jsonl" `
  --val-stage1 "$followup\train_val_stage1_features.jsonl" `
  --val-stage2 "$followup\train_val_stage2_supports.jsonl" `
  --val-predictions "$followup\train_val_selected_predictions.jsonl" `
  --output-root "$next\learned_fusion_input_audit_v1"

& $python -m experiments.external_memory_next.prepare_lambdamart_matrices_v1 `
  --audit "$next\learned_fusion_input_audit_v1\audit.json" `
  --fit-supports "$next\train_fit_ranking_features_v1\train_fit_candidate_supports.jsonl" `
  --val-stage1 "$followup\train_val_stage1_features.jsonl" `
  --val-stage2 "$followup\train_val_stage2_supports.jsonl" `
  --val-predictions "$followup\train_val_selected_predictions.jsonl" `
  --output-root "$next\lambdamart_matrices_v1"

& $python -m pip install --target '.\.build\external_memory_next_deps' 'lightgbm==4.7.0'

& $python -m experiments.external_memory_next.run_lambdamart_fusion_v1 `
  --matrix-root "$next\lambdamart_matrices_v1" `
  --audit "$next\learned_fusion_input_audit_v1\audit.json" `
  --deps-root '.\.build\external_memory_next_deps' `
  --smoothing-predictions "$next\choice_share_smoothing_fixed_surface_boundary_v2\selected_predictions.jsonl" `
  --output-root "$next\lambdamart_fusion_v1"
```

The Generic output is resumable through its `.partial.jsonl`; rerunning the
same command skips completed row IDs and restores canonical original order.
Generated Generic/features/matrices/models/results are local-only and should
not be staged.

### Task-Specific Bi-Encoder checkpoint

- **Purpose:** test whether strictly causal, query-local same-Pinyin
  supervision improves historical retrieval and frozen final IME ranking.
- **Preceding closeout commit:**
  `330204ae83ed24134befc0fb5ddb99d9b15239c5`.
- **Protocol/result records:**
  `docs/external_memory_next/15_TASK_SPECIFIC_BIENCODER_PREDECLARED_PROTOCOL_2026-08-22.md`
  and `docs/external_memory_next/16_TASK_SPECIFIC_BIENCODER_RESULTS_2026-08-22.md`.
- **Base model:** full-precision `BAAI/bge-small-zh-v1.5`, revision
  `7999e1d3359715c523056ef9478215996d62a620`; shared four-layer encoder,
  512-dimensional masked-mean/L2-normalized embeddings.
- **Frozen dependencies:** Clean3 Train-Fit for fitting, Clean3 Train-Val for
  the one-shot checkpoint evaluation, frozen Full RetunedFinal Stage-1 and
  support/prediction artifacts, completed generic-BGE LambdaMART result, and
  the exact EM3 Train-Fit pair registry. Dev3000 and Test are rejected.
- **Causal groups:** 99,671 audited positive rounds; 32,999 positive-only
  groups excluded from optimization; 66,672 trainable groups split into
  59,686 inner-fit and 6,986 inner-gate groups.
- **Selection:** epoch 2 by inner-gate Macro Recall@1 `.594849963`, then a
  fresh two-epoch refit on all 66,672 trainable Train-Fit groups.
- **Headline result:** intrinsic Macro Recall@1 `.778943777 -> .810971191`,
  Micro Recall@1 `.788203753 -> .821233244`, and MRR@10
  `.863742033 -> .883514537`; fixed-fusion Macro Top1
  `.796004927 -> .795711724`, with `34/46/-12` rescue/harm/net. The frozen
  nonlinear-refit gate failed, so no task-specific LambdaMART was fitted.
- **Hashes:** base tree
  `4d71fdf52d2c78025befad48d042d2bafa9199e19cdcf2b635c678a1e436b252`;
  final checkpoint
  `f9b87af11fcff692ad7c25fb6330f44f9f23ffedb480af9aec36af0e7cd08a8e`;
  final weights
  `81f1bc54ec80567cc15c3f986b4acc88033b0a9d268b78fa6c1a893360e63364`;
  group registry
  `9b9eda5629842ec2b57428a53c0e2b6e273c533d24dd918ed1914afbfb4c4441`;
  evaluation result
  `493d7901e7295ae58e2dcfc7d267bfe44ea797bd638357f47ca8fcf1791da0ad`;
  predictions
  `cbc5ce85605d2ddd035254e3a02a623512319976e0c931c5f4767996cf368ddf`.
- **Artifact policy:** prepared groups, checkpoints, embeddings, predictions,
  results, and logs remain local-only under
  `results/personalisation/external_memory_next/task_specific_biencoder_v1/`
  and `.build/external_memory_next_biencoder/`.
- **Resource flags:** `used_dev3000=false`; `used_test=false`.
- **Status:** **LOCAL-ARTIFACT-DEPENDENT**.

Exact Windows commands (run from the isolated worktree):

```powershell
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$compare = 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_v2'
$follow = 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune'
$root = '.\results\personalisation\external_memory_next\task_specific_biencoder_v1'
$revision = '7999e1d3359715c523056ef9478215996d62a620'
$model = ".\.build\external_memory_next_biencoder\base\$revision"

& $python -m experiments.external_memory_next.prepare_task_specific_biencoder_v1 `
  --fit "$compare\clean3_train_fit_v1.jsonl" `
  --pairs "$compare\em3_train_pairs_v1\train_pairs.jsonl" `
  --output-root "$root\preparation"

& $python -m experiments.external_memory_next.run_task_specific_biencoder_v1 `
  --phase full `
  --groups "$root\preparation\groups.jsonl" `
  --audit "$root\preparation\audit.json" `
  --base-model $model `
  --output-root "$root\training"

& $python -m experiments.external_memory_next.evaluate_task_specific_biencoder_v1 `
  --fit "$compare\clean3_train_fit_v1.jsonl" `
  --val "$compare\clean3_train_val_v1.jsonl" `
  --stage2 "$follow\train_val_stage2_supports.jsonl" `
  --frozen-predictions "$follow\train_val_selected_predictions.jsonl" `
  --generic-bge-cache "$follow\bge_context_cache.sqlite3" `
  --training-result "$root\training\training_result.json" `
  --lambdamart-result '.\results\personalisation\external_memory_next\lambdamart_fusion_v1\result.json' `
  --checkpoint "$root\training\final_refit\epoch_2" `
  --output-root "$root\evaluation"
```

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

- **Purpose:** freeze six-author chronological History/Dev/Test works and 6,000 anchors expanded to 24,000 Full/Initial 脳 Short/Multi3 conditions.
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

### Pilot A implementation -LEGACY

The exploratory Dev-only runner is `experiments/personalisation_pilot_a.py`; it established strict chronology, F, M1, BGE caching, and Dev separation. It is superseded for formal T1 comparison by the H5000 runner. The CLI phases are preserved, but this checkpoint should not be presented as the completed H5000 Test result.

### F/M1 H5000 completed local result -LOCAL-ARTIFACT-DEPENDENT

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

### Implementation checkpoint -LEGACY

`personalisation-m2-h5000-implementation-v1` preserves the pre-result implementation. Use the completed-result tag for reproduction of the reported checkpoint.

### Completed result -LOCAL-ARTIFACT-DEPENDENT

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

### Implementation checkpoint -LEGACY

`personal-vocabulary-h5000-implementation-v1` is superseded by the completed-result tag.

### Completed result -LOCAL-ARTIFACT-DEPENDENT

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

## 8. Reranking Personalisation Matrix -PARTIAL

- **Purpose:** planned 4 conditions 脳 H500/H5000/HFull 脳 F/M1/M2 matrix with Dev selection and cache reuse.
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

## 9. Context Diagnostic A -LOCAL-ARTIFACT-DEPENDENT

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

## 10. Context Strengthening -LOCAL-ARTIFACT-DEPENDENT

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

## 11. External Memory Completion -PARTIAL

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
| Historical HuoziIME Phases 3-E | `phase-03`, `phase-04a`, `phase-04b`, `phase-04b6`, `phase-04b7`, `phase-04c-setup`, `phase-04c`, `phase-04c-complete`, `phase-04d-setup`, `phase-04d`, `phase-04e-implementation` | LEGACY | Superseded thesis direction. Tag-specific `docs/phases/`, configs, runners and reports must be used; current PinyinGPT/personalisation code must not reinterpret them. |
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


## EM-1 -Exact-Scored Personal Candidate Recovery

Status: FROZEN

Condition:
Full+Short / H5000 / Etinjat + Re_spectators + breaddddd.

Dev-frozen configuration:
- K = 1
- frequency lambda = 4

Primary detailed reproduction record:

`docs/external_memory/EM1_REPRODUCIBILITY_2026-08-19.md`

Dev selection record:

`docs/external_memory/EM1_DEV_SELECTION_2026-08-19.md`

Frozen Test result:

`docs/external_memory/EM1_TEST_RESULT_2026-08-19.md`

Reusable implementation:

`src/personalisation/external_memory.py`

Formal runners:

- `experiments/external_memory/em1_score_compatibility.py`
- `experiments/external_memory/em1_recovery_coverage.py`
- `experiments/external_memory/em1_gold_reachability.py`
- `experiments/external_memory/em1_gold_reachability_test.py`
- `experiments/external_memory/em1_score_recovered_dev.py`
- `experiments/external_memory/em1_dev_comparison.py`
- `experiments/external_memory/em1_score_recovered_test.py`
- `experiments/external_memory/em1_test_evaluation.py`

Generated outputs live under:

`results/personalisation/external_memory/`

Generated result files are not the sole source of truth. Use the frozen
method/configuration, provenance hashes, and reproduction record above.

<!-- EM2-2026-08-19-INDEX -->
## External Memory EM-2 -2026-08-19

Status: **SUPERSEDED by the final EM-2 closure record below.**

Primary report:

- `docs/external_memory/em2/stages/EM2_PROGRESS_REPORT_2026-08-19.md`

Critical dependencies:

- frozen PinyinGPT2-Concat checkpoint under `C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat`
- frozen Pilot/Dev manifests and Generic cache under `C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory`
- frozen Original M2 pair cache under `C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\m2_h5000\cache\pair_scores.sqlite3`
- frozen `bge-reranker-base` checkpoint under `C:\Users\chiar\Desktop\LBH\thesis\.build\bge-reranker-base`
- EM-2 hidden cache `results/personalisation/external_memory/em2_hidden_dev/hidden_states.sqlite3`

Important hashes:

- Generic Dev cache: `588aa84c6397e8cb1a13576c0d5dfecd9dd2c4305b45be351328dd83ef62007d`
- EM-2 hidden cache: `9a80a3314c184ccf3f0540916203c651474fad162dc3dab1fc97f7451f441df1`

Exact verified commands for Hidden cache, Hidden-M1, four-way comparison, Original M2 control, Hidden-M2, and Fixed G+F+C are preserved in the stage report.




<!-- EM2-FINAL-CLOSE-2026-08-19 -->
## EM-2 final closure

Status: **COMPLETE / LOCAL-ARTIFACT-DEPENDENT**

Canonical reproduction record:

- `docs/external_memory/em2/EM2_REPRODUCIBILITY_2026-08-19.md`

Canonical scientific report:

- `docs/external_memory/em2/EM2_FINAL_REPORT_2026-08-19.md`

Stage checkpoint tag:

- `external-memory-em2-closed-20260819`

The canonical record contains runner commands, required model/data/cache paths, frozen hashes, expected Dev checkpoints, and the recovered EM-2A / EM-2C invocations.

Large generated caches/results remain local evidence and are not committed as normal source files.

EM-2 is frozen at Dev stage. No new EM-2 Test was opened.

<!-- EM1_PV_AUDIT_20260819_START -->
## PV1/PV2 vs EM-1 same-surface audit - 2026-08-19

Status: CLOSED POST-HOC EXPLANATORY AUDIT.

Runner:

```text
experiments/external_memory/em1_pv_same_surface_audit.py
```

Primary inputs:

```text
C:\Users\chiar\Desktop\LBH\thesis-personalisation\
results\personalisation\personal_vocabulary_h5000\predictions.jsonl

results\personalisation\external_memory\
em1_test_evaluation\rows.jsonl
```

Primary generated outputs:

```text
results\personalisation\external_memory\em1_pv_same_surface_audit\
    summary.json
    rows.jsonl
    candidate_identity_split.json
```

Key provenance:

```text
Generic predictions SHA256:
764db39887f3db04b913d1739d9dbd46295f0e46e5a2bffa649f1563b56ee4e2

Recovered candidate scores SHA256:
5151d462bd3594fe63d81b244083ec557886d2018f286a92a105c334b307185d

PV / EM-1 Test-state SHA256:
2912d32b8cd88843e825cb5592dfbc0a06e88e4a58831c632a126d2b8452b061
```

Observed audit invariants:

```text
rows = 3000
authors = Etinjat, Re_spectators, breaddddd
PV.condition_id == EM1.row_id
G_rank_exact_equal = True
F_rank_exact_equal = True
new Test tuning = False
new PinyinGPT Test inference = False
```

Scientific result is recorded in
`docs/external_memory/EM1_PV_COMPARISON_ADDENDUM_2026-08-19.md`.

Generated result artifacts remain LOCAL-ONLY.
<!-- EM1_PV_AUDIT_20260819_END -->

<!-- EM3-DEV-CHECKPOINT-20260820 -->
## EM3 Dev-analysis checkpoint — 2026-08-20

Status: **DEV-STAGE CHECKPOINT / LOCAL-ARTIFACT-DEPENDENT**. EM3 is paused
before new heavy training; this is not a final method freeze. Test used: **No**.

Canonical records:

- `docs/CURRENT_RESEARCH_INDEX_2026-08-20.md`;
- `docs/external_memory/em3/EM3_DEV_CLOSEOUT_2026-08-20.md`;
- `docs/external_memory/em3/EM3_ALL_OUTCOME_DISTRIBUTION_RECORD_2026-08-20.md`;
- `docs/external_memory/em3/EM3_V2_FAILURE_AUDIT_2026-08-20.md`.

### Consolidated outcome audit

Runner:

```text
experiments\external_memory\em3_all_outcome_audit.py
```

Output root:

```text
results\personalisation\external_memory\em3_all_outcome_audit\
```

Expected outputs: `summary.json`, `provenance.json`, `report.txt`,
`all_rows.jsonl`, `groups\`, and `focused_subsets\`.

```powershell
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
& $python -m experiments.external_memory.em3_all_outcome_audit
```

| Input | Path | SHA256 |
|---|---|---|
| history manifest | `C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\history_manifest.jsonl` | `7c85c38728d03985856d742f452992b3b3072af5f1c07845e099d9d07854da68` |
| Dev manifest | `C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\dev_manifest.jsonl` | `cf072d9323328b77e3d47d8a0c1beed8c40edc8767e075fb58593d6b72120606` |
| four-way rows | `C:\Users\chiar\Desktop\LBH\thesis-context-lab\results\personalisation\external_memory\em2_four_way_dev_compare\rows.jsonl` | `7bc20cddc5a772e7c1f9fb3fdd60ec17e8c2813667b7c32ec835b4cbc15d87d7` |
| fixed G/F/Context surface | `C:\Users\chiar\Desktop\LBH\thesis-context-lab\results\personalisation\external_memory\em2_fixed_gfc_dev\selected_rows.jsonl` | `6e4007b2ba7cd0bffea4c869a7860cc08c3671bf078c22e957ad09d6ce18ea25` |

Expected eight-way group counts, in
`G✓F✓H✓, G✓F✓H✗, G✓F✗H✓, G✓F✗H✗, G✗F✓H✓, G✗F✓H✗,
G✗F✗H✓, G✗F✗H✗` order:

```text
3361, 24, 42, 100, 403, 35, 45, 1598
```

### Formal pair-generator regression

Canonical runner:
`experiments/external_memory/em3_generate_train_pairs.py`.

Source manifest:
`C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\reranking_matrix\manifests\history_full_short.jsonl`.

Source SHA256:
`6d32d44189c0824d7973a5a9a50359dce3fb8111f6f7a9078580eb69fac58597`.

```powershell
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
& $python -m experiments.external_memory.em3_generate_train_pairs `
  --authors Etinjat Re_spectators breaddddd `
  --audit-only `
  --output-root results\personalisation\external_memory\em3_train_pairs_v1_regression_audit
```

Required checkpoint:

| Item | Expected |
|---|---:|
| Eligible queries | 30,968 |
| Positive pairs | 86,959 |
| Negative pairs | 146,195 |
| Total pairs | 233,154 |

The audit must also report zero non-prior pairs, zero query-history reuse, and
`test_used: false`. The existing frozen v1 pair manifest hash is
`8729f0db9ea2d4cd5c82ef812d743cdb37f551b6ddfa591b3d788b42d5a8dee2`;
its summary hash is
`c9161b187e4cace65d8c33e55b96c2a109e5aecabba525107c3e2fa89f6fc0bd`.
Audit-only mode deliberately does not rewrite or duplicate that 708 MB file.

Planned checkpoint tag after explicit approval:
`external-memory-em3-dev-audit-20260820`.

Generated results remain local-only. Do not stage `results/`, JSONL, SQLite,
logs, caches, embeddings, checkpoints, or model files.
<!-- EM3-DEV-CHECKPOINT-20260820-END -->
---

---

## Context-model comparison Dev preparation - 2026-08-20

Purpose: freeze the Clean3 balanced-3000 Full+Short/H5000 Dev surface, verify
canonical/Pilot identity, audit five-model cache coverage, and build the
local-only unified DB. CPU/SQLite only; external artifacts are read-only;
model inference: **No**; GPU: **No**; Test used: **No**.

```powershell
$worktree = 'C:\Users\chiar\Desktop\LBH\thesis-context-compare'
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
Set-Location $worktree
& $python -m experiments.context_comparison.prepare_context_comparison `
  --personalisation-root 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation' `
  --external-root 'C:\Users\chiar\Desktop\LBH\thesis-context-lab\results\personalisation\external_memory' `
  --output-root 'results\personalisation\context_comparison_v1'
```

Primary inputs are the 32,212-row canonical Full+Short Dev manifest (SHA256
`a62cb7bcc25c3c6938e5ab1d9b789a83bf0a2c506ee1765dfe82ab043d800235`),
the 32,212-row Pilot Dev manifest (SHA256
`cf072d9323328b77e3d47d8a0c1beed8c40edc8767e075fb58593d6b72120606`),
the EM3 Dev population rows (SHA256
`0c79db7a7f6fad2bee30b2cae82b1327f022ed4beeb53aa56af8055eea604059`),
and the Generic/BGE/hidden/M2/EM3 artifacts in the Artifact Audit.

Required assertions: exact 32,212 mapping; zero duplicate/unmapped/author/
Pinyin/Gold mismatches; Clean3 Dev 22,723; eligible 16,794; exactly 1,000
selected per author; all selected history-available; Test absent. Expected
manifest SHA256:
`9181f895eb19d0c36852e511263bfaefb34459dcd44efa6f45a44252e6b03f93`.

Outputs under `results/personalisation/context_comparison_v1/` are generated,
local-only, and must not be staged. Reruns byte-validate frozen JSON/JSONL and
validate an existing database rather than silently replacing it.

```powershell
& $python -m pytest tests\context_comparison\test_prepare_context_comparison.py -q
& $python -m py_compile experiments\context_comparison\prepare_context_comparison.py
```
<!-- CONTEXT-COMPARISON-PREP-20260820-END -->

<!-- STANDARDIZED-CONTEXT-RESET-20260820 -->
## Standardized context-model comparison reset

Canonical protocol and live execution records:

- `docs/context_comparison/10_STANDARDIZED_RESET_PLAN_2026-08-20.md`
- `docs/context_comparison/09_STANDARDIZED_RESET_DECISION_LOG_2026-08-20.md`
- `docs/context_comparison/11_STANDARDIZED_RESET_EXECUTION_LOG_2026-08-20.md`
- `docs/context_comparison/06_TRAIN_VAL_SPLIT_RECORD_2026-08-20.md`
- `docs/context_comparison/04_HISTORY_SEMANTICS_RECORD_2026-08-20.md`
- `docs/context_comparison/12_MODEL_RETUNE_REGISTRY_2026-08-20.md`
- `docs/context_comparison/03_WORKLOAD_CACHE_AUDIT_2026-08-20.md`

Frozen source/split identities:

| Artifact | SHA256 |
|---|---|
| authoritative Clean3 Train | `6d32d44189c0824d7973a5a9a50359dce3fb8111f6f7a9078580eb69fac58597` |
| Train-Fit v1 | `547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6` |
| Train-Val v1 | `d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220` |
| frozen Dev3000 | `9181f895eb19d0c36852e511263bfaefb34459dcd44efa6f45a44252e6b03f93` |
| legacy 5,608 evaluator source | `7bc20cddc5a772e7c1f9fb3fdd60ec17e8c2813667b7c32ec835b4cbc15d87d7` |

The legacy regression must reproduce
`3361,24,42,100,403,35,45,1598` and currently passes. Focused validation:

```powershell
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$env:PYTHONPYCACHEPREFIX = 'C:\Users\chiar\Desktop\LBH\thesis\.tmp_context_compare_pyc'
& $python -m pytest -p no:cacheprovider --basetemp 'C:\Users\chiar\Desktop\LBH\thesis\.tmp_context_compare_pytest' tests\context_comparison -q
& $python -m py_compile src\personalisation\standardized_context_comparison.py src\personalisation\standardized_generic.py experiments\context_comparison\prepare_standardized_reset.py
```

The exact preparation, Generic, hidden, BGE, pair-generation, and training
commands are maintained in the execution log. All outputs under
`results/personalisation/context_comparison_v2/` are local-only. `used_test`
must be false throughout; Test is not part of this protocol execution.
<!-- STANDARDIZED-CONTEXT-RESET-20260820-END -->

### Historical Full+Short history-depth recovery

Primary provenance note:

- `docs/context_comparison/08_HISTORICAL_HISTORY_DEPTH_PROVENANCE_2026-08-20.md`

Recovered comparison:

| History | Frequency | M1 | M2 |
|---|---:|---:|---:|
| H500 | 74.0000% | 74.0333% | 73.6500% |
| H5000 | 77.1833% | 76.7500% | 76.5000% |
| HFull | 80.3500% | 80.6500% | 80.4000% |

Recovered selected configurations:

- H500 F: lambda_frequency=4
- H500 M1: lambda_memory=4, top_n=5
- H500 M2: lambda_m2=4, retrieval_k=10
- H5000 F: lambda_frequency=4
- H5000 M1: lambda_memory=4, top_n=5
- H5000 M2: lambda_m2=4, retrieval_k=20
- HFull F: lambda_frequency=4
- HFull M1: lambda_memory=4, top_n=20
- HFull M2: lambda_m2=4, retrieval_k=10

Provenance-note SHA256:

`C93572201955F0E3BE008BB4A546F26181F63470563EE3FBDD810240DE2C4DEC`

Scientific status:

- Historical Test evidence only.
- Do not use for current model or hyperparameter selection.
- H5000 remains the controlled bounded-memory setting for the current standardized protocol.
- The historical history-depth curve is not a pure fixed-hyperparameter ablation because retrieval parameters differ across history budgets.

## Standardized Full+Short comparison completion — 2026-08-21

- Protocol and execution:
  `docs/context_comparison/10_STANDARDIZED_RESET_PLAN_2026-08-20.md` and
  `docs/context_comparison/11_STANDARDIZED_RESET_EXECUTION_LOG_2026-08-20.md`.
- Frozen Train-Val selections:
  `docs/context_comparison/13_PRE_DEV_FREEZE_2026-08-21.md` and
  `results/personalisation/context_comparison_v2/pre_dev_freeze_v1.json`.
- Sealed Dev3000 result:
  `docs/context_comparison/14_STANDARDIZED_DEV3000_RESULT_2026-08-21.md` and
  `results/personalisation/context_comparison_v2/dev3000/standardized_dev3000_result.json`.
- Dev manifest SHA256:
  `9181f895eb19d0c36852e511263bfaefb34459dcd44efa6f45a44252e6b03f93`.
- PRE_DEV_FREEZE SHA256:
  `7c0fcf69823f0b4b7d8b914a81ea54a097e12c03cb61c515c2400be46df46824`.
- Predictions SHA256:
  `dd219bfcb28fcad6a65f31eb14ddb16fc03c80f54a8b62a1cfe2504113c84233`.
- All selection used Train-Val only; `used_dev3000=false` during selection and
  `used_test=false` throughout. Test remains closed.

<!-- FULL-TRANSFER-INITIAL-FINAL-REPRO-20260822 -->

---

## Initial-Pinyin Personalisation — current Train-Val recovery + context checkpoint

Status: **DEVELOPMENT COMPLETE / LOCAL-ARTIFACT-DEPENDENT / PRE-DEV FREEZE PENDING**.

The numbered Initial-Pinyin research records live under `docs/initial_personalisation/`. `docs/REPRODUCIBILITY_INDEX.md` itself is a living repository-wide index and is updated in place.

### Canonical current records

```text
docs/initial_personalisation/19_INITIAL_RECOVERY_CONTEXT_TRAINVAL_REPRODUCIBILITY_2026-08-21.md
  Primary reproduction record for the latest activity.

docs/initial_personalisation/18_INITIAL_RECOVERY_CONTEXT_TRAINVAL_FINAL_CONCLUSIONS_2026-08-21.md
  Standalone scientific/data record for the same activity.

docs/initial_personalisation/17_INITIAL_PV1_CONTEXT_RERANKING_RESULTS_AND_REPRODUCIBILITY_2026-08-21.md
  Historical PV1 context-reranking control line.

docs/initial_personalisation/16_INITIAL_REPRODUCIBILITY_2026-08-21_v4.md
  Earlier broad Initial+Short reproducibility checkpoint.

docs/initial_personalisation/15_INITIAL_PERSONALISATION_RECOVERY_REPRODUCIBILITY_2026-08-21.md
  Earlier recovery/controllability reproducibility checkpoint.
```

### Frozen inputs / protocol identity

```text
Train-Fit rows = 144,526
Train-Val rows = 34,416
Initial Train-Fit SHA256 = 162f5c98daa86cc69947571e6d8f20fc401f0a82cdd3fd6e517eb7be2addbdb4
Initial Train-Val SHA256 = d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4
Candidate surface SHA256 = 205c0ba01cd0678d7a4341c503fa2e74cf126a70182ff582687025e4946764b2
Generic predictions SHA256 = bd0fb4dc304e0b266b90fae6fe3ac65424d2f52b23fedfa881212706ba2c2873
Frequency/PV1 predictions SHA256 = 7fd8aa158d8cd50bced36b55610f8d932bc65e3aae1dbbd5bd65907ff1707ea7

History semantics:
  same author
  -> strictly prior interactions
  -> latest up-to-5000 RAW interactions
  -> exact Initial-Pinyin filtering afterward

Gold used for candidate construction/scoring = false
Gold used for Train-Val evaluation/selection/diagnosis only = true
Dev3000 used = false
Test used = false
```

### Exact current runner identities

```text
run_initial_recovery_ngram_context_fusion_v1.py
  SHA256 e6dcd1f68028ad5065064b6b714eaa88d92f74363a328570bfcc777b13271dc2

run_initial_recovery_bge_ngram_context_fusion_v2.py
  SHA256 b7d95374aa421cbc364699e44e0850ba2e72e50a2a5f816ad37f85b138d1435a

run_initial_recovery_bge_ngram_context_fusion_v3.py
  SHA256 2b29a86957b4f2adf17a13de37648766e1423d0ec99a57ea257c5aa155d89335

run_initial_recovery_context_diagnostics_v1.py
  SHA256 7c4a12a5f447405f024d8e8008253da23aab4775d2ae4500f5c44545583d3256

run_initial_recovery_context_topk_transitions_v1.py
  SHA256 3966111844719f29a07b580a10d18021b0cdf4a6846c71157de611e1a92eaef1
```

For exact commands, parameter grids, BGE cache behavior, expected console checkpoints, and failure rules, use document `19_...REPRODUCIBILITY...md`; do not reconstruct them from this index.

### Reproduction sequence

```text
V1: Stage-1 recovery bases + NGramRecency grid
  -> V2: NGramRecency + BGERecency 2-D grid
  -> V3: expanded lambda_B boundary verification, no BGE recomputation
  -> read-only context diagnosis
  -> read-only Top1/Top3/Top5 rescue-harm diagnosis
```

Required V3 regressions:

```text
V1 NGram-only selected points reproduced: PASS
V2 selected full-context points reproduced: PASS
V3 selected points identical to V2: PASS
No selected lambda_B at expanded upper boundary: PASS
BGE recomputed in V3: false
Dev3000 used: false
Test used: false
```

### Canonical current result roots

```text
results/personalisation/initial_recovery_comparison_v1/recovery_ngram_context_fusion_v1/
results/personalisation/initial_recovery_comparison_v1/recovery_bge_ngram_context_fusion_v2/
results/personalisation/initial_recovery_comparison_v1/recovery_bge_ngram_context_fusion_v3/
results/personalisation/initial_recovery_comparison_v1/recovery_context_diagnostics_v1/
results/personalisation/initial_recovery_comparison_v1/recovery_context_topk_transitions_v1/
```

The V3 root is the canonical final selected-prediction/result root for this Train-Val stage. The diagnosis roots are post-hoc/read-only and must not be used to reopen tuning.

### Frozen Train-Val operating points

```text
Primary overall:
  4P+4CS+2E + NG-R lambda_N=4 + BGE-R lambda_B=6
  Macro=.437058 Micro=.460571 Top3=.631392 Top5=.696478 MRR=.559755 Missing=.243172
  Rec1=.4246 Rec3=.6969 Rec5=.8153 Rec10=.9485 RecMRR=.5892

Coverage-oriented:
  K5+Entropy + NG-R lambda_N=6 + BGE-R lambda_B=8
  Macro=.436767 Micro=.459990 Top3=.626453 Top5=.688139 MRR=.557836 Missing=.243288
  Rec1=.4430 Rec3=.7481 Rec5=.8778 Rec10=.9876 RecMRR=.6218

Front-rank comparison:
  6P+2CS+.25E + NG-R lambda_N=4 + BGE-R lambda_B=6
  Macro=.436477 Micro=.459786 Top3=.630085 Top5=.696013 MRR=.558806 Missing=.243869
  Rec1=.4415 Rec3=.6961 Rec5=.8069 Rec10=.9283 RecMRR=.5951
```

Primary selection is by pre-specified Macro-author Top1. The Balanced-vs-K5 Macro gap is only `.000291`; this is a Train-Val selection result, not evidence of statistical significance or holdout superiority.

### Diagnostic reproduction checkpoints

```text
Primary 4P+4CS+2E:
  Recovery -> NG-R:
    Delta Macro = +.027644
    Top1 rescue=2375 harm=1468 net=+907

  NG-R -> Full:
    Delta Macro = +.004607
    Top1 rescue=681 harm=514 net=+167

Recoverable R = 4,910
Generic Missing = 12,565

Top3 Recovery -> Full on R:
  K5+Entropy:  rescue=745 harm=80 net=+665
  4P+4CS+2E:   rescue=745 harm=20 net=+725
  6P+2CS+.25E: rescue=617 harm=16 net=+601
```

Per-author primary final checkpoints:

```text
Agent Phage: Stage1=.470344 -> NG=.487374 -> Full=.493923; Missing=.171312
Etinjat:     Stage1=.237235 -> NG=.271980 -> Full=.275218; Missing=.485056
breaddddd:   Stage1=.506841 -> NG=.537999 -> Full=.542032; Missing=.167655
```

### Reproduction boundary

This checkpoint is **Train-Val development only**. The current selected lambdas and recovery coefficients must be treated as frozen before opening Dev3000. Post-hoc diagnosis is explanatory only and must not trigger new gates, features, coefficients, or lambda tuning on the same Train-Val data.

Next formal sequence:

```text
PRE-DEV FREEZE
-> Dev3000 evaluation of frozen operating points / control
-> pre-declared selection
-> final freeze
-> Test
```

---

## Full+Short zero-shot Initial-final transfer — LOCAL-ARTIFACT-DEPENDENT

- **Date:** 2026-08-22.
- **Scientific status:** post-Dev follow-up; Full Train-Val descriptive evaluation only.
- **Purpose:** apply the frozen Initial-Pinyin primary architecture `4P+4CS+2E + NGramRecency(lambda_N=4) + BGERecency(lambda_B=6)` directly to standardized Full+Short Train-Val without Full-specific tuning.
- **Dev3000 used:** No.
- **Test used:** No.
- **Hyperparameter search on Full:** No.
- **Canonical report:** `docs/context_comparison/16_FULL_TRANSFER_INITIAL_FINAL_TRAINVAL_2026-08-22.md`.
- **Report SHA256:** `228e4c404ae8a369831ac1f0fe1bfd79cf2cdf91a1972b788057bfddb69884bc`.
- **Runner:** `experiments/context_comparison/run_full_transfer_initial_final_v1.py`.
- **Runner SHA256:** `f75d40f381e966f85cd4b20647ba7dc6a95df9116ad8657ca9a07505949a37b0`.

### Frozen inputs

| Artifact | Rows | SHA256 |
|---|---:|---|
| `results/personalisation/context_comparison_v2/clean3_train_fit_v1.jsonl` | 144,526 | `547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6` |
| `results/personalisation/context_comparison_v2/clean3_train_val_v1.jsonl` | 34,416 | `d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220` |
| `results/personalisation/context_comparison_v2/train_val_generic/predictions.jsonl` | 34,416 | `cf4ae382fa23e5ec1154bf28320d13ac1d6ca9600e9dcf8a6aa599600bc28eab` |
| `results/personalisation/context_comparison_v2/stage1/train_val.jsonl` | standardized Train-Val comparator artifact | `69e44c6b4d91c679b1ebcd7043f0fe98e093d9ae83849b542a37a875488c2a45` |
| BGE GGUF | model | `5a88d266870fbd27c6f329df60de80e2d4cf3bbd5e6f080bd5c1b2e5abb12039` |

History semantics must remain:

```text
same author -> strictly prior -> latest H5000 raw -> exact segmented-Pinyin
```

The H5000 budget is applied before exact-Pinyin filtering.

### Frozen transferred configuration

Stage1:

```text
Personal K = 5
formula = boundary + 4*P_NG + 4*ChoiceShare + 2*EntropyConcentration
P_NG type = InterpolatedNGramRecency
P_NG maxN = 2
P_NG kappa = 1
P_NG tau = 2048
```

Stage2 NGramRecency:

```text
lambda_N = 4
maxN = 2
tau_N = 2048
```

Stage2 BGERecency:

```text
context = last 64 Chinese characters
retrieval = cosine only, candidate-conditioned
TopN per candidate = 5
tau_B = 2048
lambda_B = 6
aggregation = max(0, cosine) * exp(-age/tau)
```

For the two frozen Generic rows with zero candidates, the runner applies a conservative no-op because the transferred Generic boundary is undefined; it does not invent a Full-specific recovery anchor.

### Exact reproduction command

```powershell
Set-Location 'C:\Users\chiar\Desktop\LBH\thesis-context-compare'

& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  '.\experiments\context_comparison\run_full_transfer_initial_final_v1.py' `
  --fit '.\results\personalisation\context_comparison_v2\clean3_train_fit_v1.jsonl' `
  --val '.\results\personalisation\context_comparison_v2\clean3_train_val_v1.jsonl' `
  --generic '.\results\personalisation\context_comparison_v2\train_val_generic\predictions.jsonl' `
  --checkpoint 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat' `
  --bge-model 'C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf' `
  --standardized-stage1 '.\results\personalisation\context_comparison_v2\stage1\train_val.jsonl' `
  --output-root '.\results\personalisation\context_comparison_followup_v1\full_transfer_initial_final_v1' `
  --compatibility-device cpu `
  --cuda-path 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8' `
  --progress-every 500
```

### Expected completed result

Overall Macro-author Top1:

```text
Generic   0.7007949151
Frequency 0.7768750752
M1        0.7782391648
Stage1    0.7885230323
Final     0.7953665798
```

Expected Final diagnostics:

```text
Micro Top1       0.8245002325
Top3             0.9119014412
Top5             0.9306717806
MRR@10           0.8710377879
Missing@10       0.0519525802
Ambiguous Macro  0.8021720889
Conflict Macro   0.2289725953
```

Recovery expectation among the 620 Generic-missing rows with Gold in Personal K5:

```text
Final Rec@1  = 505 / 620 = 0.8145161290
Final Rec@3  = 585 / 620 = 0.9435483871
Final Rec@5  = 602 / 620 = 0.9709677419
Final Rec@10 = 617 / 620 = 0.9951612903
Final recovery MRR@10 = 0.8835701485
```

Transition expectations:

```text
Frequency -> Stage1: rescue=472 harm=119 net=+353
Stage1 -> Final:     rescue=464 harm=252 net=+212
Frequency -> Final:  rescue=893 harm=328 net=+565
M1 -> Final:         rescue=899 harm=400 net=+499
```

### Generated output identity

Output root:

```text
results/personalisation/context_comparison_followup_v1/full_transfer_initial_final_v1/
```

Known canonical hashes:

| Artifact | SHA256 |
|---|---|
| `result.json` | `604a74d212ff16954b09f375a8db88f527cc07d12333fab0a7c18a7f712743a3` |
| `run_setup.json` | `28ee66721e4ffcdad82f141d763c537e3fdcc60ca7faa4c0e2e2ed82c27e69e1` |
| `stage1_predictions.jsonl` | `eacc6c37c53e581bc667483eb6b29816cc81c3239aad9c16acf16788611ec53f` |
| `final_predictions.jsonl` | `fcc9b44c06fe0dc7bf629ad81d79476a4d27e52be934a37f4ac9c9d8d293973d` |

The completed BGE cache must contain **42,278 / 42,278** required unique contexts.

Generated results, JSONL, SQLite caches, and logs remain local-only and should not be staged. The human-authored runner/report/index files may be staged only under the repository's normal explicit-path Git policy and only with user authorization.

### Interpretation boundary

This checkpoint is a **descriptive zero-shot transfer result**. It must not be used to reopen the already completed standardized Train-Val selection or retroactively alter the seven-system sealed Dev3000 comparison. No statistical-significance claim is established by this checkpoint alone.
<!-- FULL-TRANSFER-INITIAL-FINAL-REPRO-20260822-END -->

<!-- FULL-RETUNED-FINAL-DEV-REPRO-20260822 -->
## Full-retuned Final Train-Val selection + Dev3000 — LOCAL-ARTIFACT-DEPENDENT

- **Date:** 2026-08-22.
- **Scientific status:** post-Dev development extension; Full Train-Val parameter selection followed by a frozen Dev3000 development comparison.
- **Test used:** No. Test remains CLOSED.
- **Dev used for hyperparameter selection:** No.
- **Dev role:** development comparison / feedback surface; not untouched final evaluation.
- **Canonical report:** `docs/context_comparison/17_FULL_RETUNED_FINAL_DEV3000_CLOSEOUT_2026-08-22.md`.
- **Runner:** `experiments/context_comparison/run_full_retune_final_trainval_dev_v1.py`.
- **Runner SHA256:** `89d526cb61d3bb93a1caa3d401679db9f1f8b8efdc31d4daa4590adcce3dee8d`.
- **Frozen base transfer runner SHA256:** `f75d40f381e966f85cd4b20647ba7dc6a95df9116ad8657ca9a07505949a37b0`.

### Selection grid and rule

```text
Stage1 w_P  = [0.0, 2.0, 4.0, 6.0]
Stage1 w_CS = [0.0, 2.0, 4.0, 6.0]
Stage1 w_E  = [0.0, 2.0, 4.0]
Stage2 lambda_N = [0.0, 2.0, 4.0, 6.0, 8.0]
Stage2 lambda_B = [0.0, 2.0, 4.0, 6.0, 8.0]
selection = Macro-author Top1 -> Micro Top1 -> MRR@10 -> distance to transferred reference -> lexicographic tie-break
```

Selected configuration:

```text
w_P=2.0  w_CS=6.0  w_E=4.0
lambda_N=6.0  lambda_B=6.0
```

Fixed architecture remains Personal K5, H5000-before-Pinyin causal history, P_NG maxN=2/kappa=1/tau=2048, Stage2 NG maxN=2/tau=2048, and BGE context64/Top5/tau=2048.

### Expected selected results

```text
Train-Val RetunedFinal Macro Top1 = 0.7960049266
Train-Val RetunedFinal Micro Top1 = 0.8249941887
Dev RetunedFinal Macro Top1       = 0.8436666667
Dev RetunedFinal Micro Top1       = 0.8436666667
Dev RetunedFinal Top3             = 0.9343333333
Dev RetunedFinal MRR@10           = 0.8920410053
Dev RetunedFinal Missing@10       = 0.0290000000
Dev Stage1->Final rescue/harm/net = 67/38/+29
```

### Generated output identity

- `selected_config.json` SHA256 `3dc3fb908aeeaa853526ad71cf85de7400f47d261ed7c09acdd8197446f5fa3d`
- `tune/train_val_result.json` SHA256 `2899270eca2c474957afbb7cb1943140bd576ad3aa76514223ee2a0b0f4c7b48`
- `dev/dev_result.json` SHA256 `07a9fb80a138681db6de05cba7361e948a880b61b02868ec2c06037ff69e48da`
- Generated JSON/JSONL/SQLite/cache artifacts are LOCAL-ONLY / DO NOT STAGE.

### Reproduction

Use the exact two commands recorded in the canonical closeout report. Always run `--phase tune` first; the subsequent `--phase dev` requires and reuses the frozen `selected_config.json` and performs no Dev hyperparameter search.

### Freeze boundary

This development segment is closed. Do not change the selected Full-retuned configuration based on this Dev result unless the development phase is explicitly reopened. Test must remain untouched until a separately authorized final frozen evaluation.
<!-- FULL-RETUNED-FINAL-DEV-REPRO-20260822-END -->

<!-- POSTHOC-TASK-BIENCODER-CALIBRATION-REPRO-20260822 -->
## Post-hoc Task-BiEncoder recovery/calibration - LOCAL-ARTIFACT-DEPENDENT

- Date: 2026-08-22.
- Selection/evaluation surface: separate Initial and Full Train-Val tracks.
- Task checkpoint retrained: No.
- Dev3000 used: No.
- Test used: No.
- Protocol: `docs/external_memory_next/17_POSTHOC_TASK_BIENCODER_RECOVERY_CALIBRATION_PROTOCOL_2026-08-22.md`.
- Results: `docs/external_memory_next/19_POSTHOC_TASK_BIENCODER_RECOVERY_CALIBRATION_RESULTS_2026-08-22.md`.
- Generated root: `results/personalisation/external_memory_next/posthoc_task_biencoder_calibration_v1/`.

Set shared PowerShell paths:

```powershell
Set-Location 'C:\Users\chiar\Desktop\LBH\thesis-external-memory-next'
$py = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$initial = 'C:\Users\chiar\Desktop\LBH\thesis-initial-research\results\personalisation\initial_recovery_comparison_v1'
$full = 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation'
$root = '.\results\personalisation\external_memory_next\posthoc_task_biencoder_calibration_v1'
$task = '.\results\personalisation\external_memory_next\task_specific_biencoder_v1'
$checkpoint = "$task\training\final_refit\epoch_2"
```

Exact Initial/Full preflight and support commands:

```powershell
$initialSupport = @(
  '--track','initial',
  '--fit',"$initial\initial_train_fit_v1.jsonl",
  '--val',"$initial\initial_train_val_v1.jsonl",
  '--stage1',"$initial\recovery_ngram_context_fusion_v1\stage1_frozen.jsonl",
  '--existing-support',"$initial\recovery_bge_ngram_context_fusion_v2\bge_recency_support.jsonl",
  '--frequency-predictions',"$initial\frequency_pv1\predictions.jsonl",
  '--generic-seed-cache',"$initial\recovery_bge_ngram_context_fusion_v2\bge_history_embedding_cache.sqlite3",
  '--task-seed-cache',"$task\evaluation\task_vectors.sqlite3",
  '--task-checkpoint',$checkpoint,
  '--generic-bge-model','C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf'
)
& $py -m experiments.external_memory_next.prepare_posthoc_context_support_v1 `
  --phase preflight @initialSupport --output-root "$root\preflight"
& $py -m experiments.external_memory_next.prepare_posthoc_context_support_v1 `
  --phase score @initialSupport --output-root "$root\support"

$fullSupport = @(
  '--track','full',
  '--fit',"$full\context_comparison_v2\clean3_train_fit_v1.jsonl",
  '--val',"$full\context_comparison_v2\clean3_train_val_v1.jsonl",
  '--stage1',"$full\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_stage1_features.jsonl",
  '--existing-support',"$full\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_stage2_supports.jsonl",
  '--generic-seed-cache',"$full\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\bge_context_cache.sqlite3",
  '--task-seed-cache',"$task\evaluation\task_vectors.sqlite3",
  '--task-checkpoint',$checkpoint,
  '--generic-bge-model','C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf'
)
& $py -m experiments.external_memory_next.prepare_posthoc_context_support_v1 `
  --phase preflight @fullSupport --output-root "$root\preflight"
& $py -m experiments.external_memory_next.prepare_posthoc_context_support_v1 `
  --phase score @fullSupport --output-root "$root\support"
```

Exact resumable Full-Q8 command:

```powershell
& $py -m experiments.external_memory_next.score_full_personal_k5_q8_v1 `
  --val "$full\context_comparison_v2\clean3_train_val_v1.jsonl" `
  --features "$full\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_stage1_features.jsonl" `
  --generic "$full\context_comparison_v2\train_val_generic\predictions.jsonl" `
  --checkpoint 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat' `
  --output-root "$root\q8_full" --device cuda
```

Exact accepted evaluation command:

```powershell
& $py -m experiments.external_memory_next.evaluate_posthoc_task_biencoder_calibration_v1 `
  --initial-fit "$initial\initial_train_fit_v1.jsonl" `
  --initial-val "$initial\initial_train_val_v1.jsonl" `
  --initial-stage1 "$initial\recovery_ngram_context_fusion_v1\stage1_frozen.jsonl" `
  --initial-ngram "$initial\recovery_ngram_context_fusion_v1\ngram_recency_support.jsonl" `
  --initial-support "$root\support\initial_support.jsonl" `
  --initial-frequency "$initial\frequency_pv1\predictions.jsonl" `
  --initial-q8 "$initial\candidate_scoring_q8_bge64_v1\q8\q8_k5_exact_scores.jsonl" `
  --full-fit "$full\context_comparison_v2\clean3_train_fit_v1.jsonl" `
  --full-val "$full\context_comparison_v2\clean3_train_val_v1.jsonl" `
  --full-stage1 "$full\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_stage1_features.jsonl" `
  --full-stage2 "$full\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_stage2_supports.jsonl" `
  --full-support "$root\support\full_support.jsonl" `
  --full-frozen "$full\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_selected_predictions.jsonl" `
  --full-q8 "$root\q8_full\full_q8_scores.jsonl" `
  --intrinsic-result "$task\evaluation\result.json" `
  --lambdamart-result '.\results\personalisation\external_memory_next\lambdamart_fusion_v1\result.json' `
  --output-root "$root\evaluation"
```

Exact bounded task latency command:

```powershell
& $py -m experiments.external_memory_next.benchmark_posthoc_task_latency_v1 `
  --fit "$initial\initial_train_fit_v1.jsonl" `
  --val "$initial\initial_train_val_v1.jsonl" `
  --support "$root\support\initial_support.jsonl" `
  --task-checkpoint $checkpoint `
  --task-vectors "$root\support\initial_task_vectors.sqlite3" `
  --output "$root\latency\task_biencoder_latency.json" `
  --queries 500 --warmup 20
```

Exact latency/Pareto finalization command:

```powershell
& $py -m experiments.external_memory_next.finalize_posthoc_latency_pareto_v1 `
  --evaluation "$root\evaluation\result.json" `
  --task-latency "$root\latency\task_biencoder_latency.json" `
  --historical-summary "$initial\all_results_summary_v1\all_results_summary.json" `
  --support-initial "$root\support\support_initial_result.json" `
  --support-full "$root\support\support_full_result.json" `
  --q8-full "$root\q8_full\full_q8_summary.json" `
  --output "$root\latency\latency_pareto.json" `
  --plot "$root\latency\accuracy_latency_pareto.png"
```

Exact read-only invariant audit and final validation commands:

```powershell
& $py -m experiments.external_memory_next.audit_posthoc_closeout_v1 `
  --result "$root\evaluation\result.json" `
  --predictions "$root\evaluation\selected_predictions.jsonl" `
  --initial-fit "$initial\initial_train_fit_v1.jsonl" `
  --initial-val "$initial\initial_train_val_v1.jsonl" `
  --initial-frequency "$initial\frequency_pv1\predictions.jsonl" `
  --initial-support "$root\support\initial_support.jsonl" `
  --full-fit "$full\context_comparison_v2\clean3_train_fit_v1.jsonl" `
  --full-val "$full\context_comparison_v2\clean3_train_val_v1.jsonl" `
  --full-stage1 "$full\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_stage1_features.jsonl" `
  --full-support "$root\support\full_support.jsonl"

& $py -m pytest tests\external_memory_next\test_posthoc_context_calibration.py -q
& $py -m pytest tests\external_memory_next -q

& $py -m py_compile `
  src\personalisation\posthoc_context_calibration.py `
  experiments\external_memory_next\prepare_posthoc_context_support_v1.py `
  experiments\external_memory_next\score_full_personal_k5_q8_v1.py `
  experiments\external_memory_next\evaluate_posthoc_task_biencoder_calibration_v1.py `
  experiments\external_memory_next\benchmark_posthoc_task_latency_v1.py `
  experiments\external_memory_next\finalize_posthoc_latency_pareto_v1.py `
  experiments\external_memory_next\audit_posthoc_closeout_v1.py `
  tests\external_memory_next\test_posthoc_context_calibration.py

git diff --check
rg -n '"(used_dev3000|used_test|dev3000_used|test_used)"\s*:\s*true' `
  "$root" -g '*.json' -g '*.jsonl'
Get-Process python -ErrorAction SilentlyContinue
nvidia-smi
```

Expected primary values:

```text
Initial NGram+Generic-R Macro = 0.4370578839609785
Initial NGram+Task-R Macro    = 0.4356462413968509
Full NGram+Generic-R Macro    = 0.7960049265502147
Full NGram+Task-R Macro       = 0.7957480665207601
Initial Q8 K2+ Macro          = 0.6365306668058097
Initial Q8+F K2+ Macro        = 0.6691641408627556
Full historical LambdaMART    = 0.7988390633366215
```

Canonical generated artifact inventory:

| Path under `$root` | Bytes | SHA256 |
|---|---:|---|
| `evaluation/result.json` | 227,542 | `5b622c0288c482adee584857801cf13358db3a86ea7171735edc0a83b98d4eac` |
| `evaluation/grid.json` | 635,740 | `e8799c1ff765e05089db025de7ecfbb91fd6805877bb677bb60fdbd1663267f3` |
| `evaluation/selected_predictions.jsonl` | 37,492,212 | `85eac5f6533de3f439f289811e2524f42b8e8146cf5a4b5fdc6c04e13184386c` |
| `latency/latency_pareto.json` | 4,424 | `f9df5dafcc6af85fcc15e6745dbacf9371de60474b5f9d3aed311a72e8bc49ec` |
| `latency/accuracy_latency_pareto.png` | 59,783 | `7d5c1efff99ab75f47f89b4d8c9165981021188970cd2f7c68b6a748557f5e42` |
| `support/initial_support.jsonl` | 48,853,320 | `2c2bc7faddab4c032baf58d23a0767e6c31881bdd4c08f348cb865f43e4fced3` |
| `support/full_support.jsonl` | 31,321,549 | `564ec16bd623d722a42b17eec5ee05daffff1918049cce6eaf8bb4ef9902ef4a` |
| `q8_full/full_q8_scores.jsonl` | 1,763,192 | `99b9e7095cd6a6a457366aaddd185cf781f9e04dfccb60fdf939bdd0ff957ab6` |

Frozen input SHA256 provenance is recorded in section 8 of the canonical result
report. Generated artifacts are not Git inputs and must remain
`GENERATED / LOCAL-ONLY / DO NOT STAGE`.
Generated JSON/JSONL/SQLite/PNG artifacts remain local-only and must not be
staged.
<!-- POSTHOC-TASK-BIENCODER-CALIBRATION-REPRO-20260822-END -->
