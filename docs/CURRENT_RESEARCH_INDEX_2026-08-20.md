# CURRENT_RESEARCH_INDEX — 2026-08-20

Purpose: this is the **first file to read before searching the repository**.
It is a live navigation and status index for the current thesis research line.

Project:
**Transparent and User-Controllable Personalisation for Chinese Pinyin Input**

---

## 0. Current checkpoint — read this first

**Status:** EM3 is paused at a **Dev-analysis checkpoint**. This is not a final
EM3 method freeze. Benchmark Test remains closed.

The canonical consolidated diagnostic is Full+Short / H5000 on the **old
exploratory** three-author Dev surface (Etinjat, Re_spectators, breaddddd),
5,608 rows. It must not be relabelled as clean3, and its Micro counts must not
replace the frozen primary Macro-author evaluation table.

| G | F | Hidden-M1 | Rows |
|---|---|---|---:|
| ✓ | ✓ | ✓ | 3,361 |
| ✓ | ✓ | ✗ | 24 |
| ✓ | ✗ | ✓ | 42 |
| ✓ | ✗ | ✗ | 100 |
| ✗ | ✓ | ✓ | 403 |
| ✗ | ✓ | ✗ | 35 |
| ✗ | ✗ | ✓ | 45 |
| ✗ | ✗ | ✗ | 1,598 |

Diagnostic-surface Micro correctness is G `3527/5608 = 62.89%`, F
`3823/5608 = 68.17%`, and Hidden-M1 `3851/5608 = 68.67%`. Relative to F,
Context rescues 87 rows, harms 59, and nets `+28` rows (about `+0.50`
percentage points Micro Top1).

The 1,598 all-wrong rows split into 1,275 with no Gold precedent in
same-Pinyin history and 323 with Gold in history. Of those 323, exactly 53.9%
also have Gold in the current candidate set: approximately 174 ranking/fusion
opportunities and 149 candidate-recovery failures. The approximate counts are
interpretations of the exact percentage, not independently counted integers.

Key groups are:

- 45 Context-only rescues (`G✗F✗H✓`); all have Gold in history and candidates,
  but only 20% have Gold as the raw history winner;
- 24 pure harmful Context overrides (`G✓F✓H✗`);
- 35 cases where F rescues G but Context loses the rescue (`G✗F✓H✗`), versus
  403 successful `G✗F✓H✓` rows. The failure group has much lower raw-winner
  dominance and greater same-Pinyin entropy/diversity.

Current working direction, not a frozen architecture:

```text
PersonalLift(c,p) = log(P_user(c|p) / P_global(c|p))
Final(c) = Generic(c) + PersonalScore(c)
PersonalScore(c) ≈ PersonalLift(c) × ContextUtility(query, candidate, history)
```

The aim is candidate-specific historical utility: preserve minority-target
Context rescues while suppressing misleading overrides and separating common
language frequency from user-specific lift. Gold-based quantities are oracle
diagnostics only. Runtime features must be prediction-visible.

The EM3-v2 development population is frozen as **clean3**: Agent Phage,
Etinjat, and breaddddd. MScarlet remains excluded because of the script
normalization confound; Re_spectators remains part of the old diagnostic only.

Canonical audit:

- runner: `experiments/external_memory/em3_all_outcome_audit.py`;
- result root: `results/personalisation/external_memory/em3_all_outcome_audit/`;
- detailed record:
  `docs/external_memory/em3/EM3_ALL_OUTCOME_DISTRIBUTION_RECORD_2026-08-20.md`.

Canonical pair generator:

- `experiments/external_memory/em3_generate_train_pairs.py`;
- frozen old-three-author regression: 30,968 eligible queries, 86,959 positive
  pairs, 146,195 negatives, 233,154 total;
- audit root:
  `results/personalisation/external_memory/em3_train_pairs_v1_regression_audit/`.

Independent Codex Dev result (record only; do not merge its branch or copy its
generated result trees): the isolated Initial+Short clean3 G/F/M1 utility gate
reached Macro-author Top1 `0.349221` versus F `0.346863` on 3,296 untouched
confirmation rows (`+0.002357`), with rescue/harm/net `80/71/+9`. Its 95%
bootstrap interval `[-0.005365, 0.010024]` and McNemar `p=0.5152` are not
statistically conclusive. Full+Short support was `+0.001633`. Canonical report:
`C:\Users\chiar\Desktop\LBH\thesis-codex-em3-research\docs\external_memory\em3\CODEX_PERFORMANCE_RESEARCH_2026-08-20.md`.

Canonical pause/handoff and exact resume order:
`docs/external_memory/em3/EM3_DEV_CLOSEOUT_2026-08-20.md`.

---

## 1. Working directories / isolation

### Main active research worktree
`C:\Users\chiar\Desktop\LBH\thesis-context-lab`

Current research branch:
`work/external-memory-completion`

Use this worktree for the user's main EM1/EM2/EM3 work.

### Codex isolated research worktree
`C:\Users\chiar\Desktop\LBH\thesis-codex-em3-research`

Codex branch:
`codex/em3-performance-research-20260820`

Codex may read `thesis-context-lab` as read-only reference, but should write only to its isolated worktree.

### Other important local worktrees / assets

`C:\Users\chiar\Desktop\LBH\thesis-personalisation`
- authoritative Pilot / reranking-matrix manifests and several large caches/results.

`C:\Users\chiar\Desktop\LBH\thesis-deep-author`
- frozen Dataset V1 reconstruction and frozen T1 evaluation artifacts.

`C:\Users\chiar\Desktop\LBH\thesis`
- shared Python venv and pinned model checkpoints.

Python:
`C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe`

---

## 2. Canonical repository navigation files

Read these before doing broad repository searches:

`docs/FILE_INDEX.md`
- static role/status inventory for important repository files and result roots.

`docs/REPRODUCIBILITY_INDEX.md`
- exact historical checkpoints, tags, local dependencies and reproduction commands.

`docs/external_memory/em3/EM3_PROGRESS_2026-08-19.md`
- earlier EM3 design/progress.

`docs/external_memory/em3/EM3_PROGRESS_2026-08-20.md`
- current EM3-BCE v1 results, diagnostics and file locations.

`docs/external_memory/em3/EM3_V2_METHOD_OPTIONS_2026-08-20.md`
- candidate v2 losses / sampling / ranking directions.

`docs/external_memory/em3/EM3_V2_EXECUTION_PLAN_2026-08-20.md`
- current v2 execution plan.

`docs/external_memory/em3/EM3_V2_DATA_PREPARATION_2026-08-20.md`
- clean-author decision and v2 data-preparation policy.

---

## 3. Frozen core task semantics

Task:

preceding Chinese context
+ current Pinyin
+ strictly-prior same-author history
→ ranked Pinyin-compatible Chinese candidates.

Causal history semantics:
- same author only;
- strictly prior only;
- H5000 budget;
- budget applied BEFORE exact-Pinyin filtering;
- exact segmented-Pinyin matching where the method requires it;
- current row enters history only after evaluation;
- earlier Dev may become history for later Dev;
- no future history.

Test policy:
**Do not use Test for current EM3 development or selection.**

---

## 4. Frozen Generic baseline

Model:
PinyinGPT / PinyinGPT2-Concat

Pinned checkpoint:
`C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat`

Frozen T1 predictions:
`C:\Users\chiar\Desktop\LBH\thesis-deep-author\results\evaluation\deep_author_v2\t1\predictions.jsonl`

24k prediction SHA256:
`764db39887f3db04b913d1739d9dbd46295f0e46e5a2bffa649f1563b56ee4e2`

Condition-manifest SHA256:
`45b9cafedd7a8269d1f0b66d3f7f135ee990140e4b5b3668c67645863ab00d39`

Frozen Generic Test Macro Top1:
- Full+Short: 72.3167%
- Initial+Short: 32.90%
- Full+Multi3: 37.6333%
- Initial+Multi3: 7.2333%

These Test numbers are historical background only.
Do not use Test for current method selection.

---

## 5. Author populations

Six authors:
- Agent Phage
- Etinjat
- MScarlet
- QBLevi
- Re_spectators
- breaddddd

Old exploratory EM3 three-author set:
- Etinjat
- Re_spectators
- breaddddd

Current EM3-v2 clean3 set:
- Agent Phage
- Etinjat
- breaddddd

Why:
- Re_spectators has low usable EM3 Dev support and unstable Macro-author weight.
- MScarlet has a known script-normalisation / script-alignment confound.
- Agent Phage has substantially stronger Train/Dev pair-trainable support.

### Train population audit

Source:
`results\personalisation\external_memory\em3_train_population_audit\summary.json`

Pair-trainable:
- Agent Phage: 15,811
- Etinjat: 11,799
- MScarlet: 12,696
- QBLevi: 3,543
- Re_spectators: 1,991
- breaddddd: 17,178

### Dev population audit

Source:
`results\personalisation\external_memory\em3_dev_population_audit\summary.json`

Pair-trainable:
- Agent Phage: 2,851
- Etinjat: 1,326
- MScarlet: 2,108
- QBLevi: 544
- Re_spectators: 247
- breaddddd: 2,003

Clean3 totals:
- Train rows: 178,942
- Train pair-trainable: 44,788
- Train ambiguous: 46,961
- Train conflict: 8,596
- Dev rows: 22,723
- Dev pair-trainable: 6,180
- Dev ambiguous: 6,570
- Dev conflict: 1,235

---

## 6. Important manifests

Authoritative reranking-matrix manifests:
`C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\reranking_matrix\manifests\`

Important condition files include:
- `history_full_short.jsonl`
- `dev_full_short.jsonl`
- `history_initial_short.jsonl`
- `dev_initial_short.jsonl`
- corresponding Full/Initial × Short/Multi3 manifests.

Full+Short historical source SHA256:
`6d32d44189c0824d7973a5a9a50359dce3fb8111f6f7a9078580eb69fac58597`

Full+Short Dev source SHA256:
`a62cb7bcc25c3c6938e5ab1d9b789a83bf0a2c506ee1765dfe82ab043d800235`

Pilot manifests:
`C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\`

Important:
- `history_manifest.jsonl`
- `dev_manifest.jsonl`

The matrix and Pilot rows may use different `row_id` values.
Stable cross-pipeline mapping can use `anchor_id`.

---

## 7. Important reusable caches

### Generic Dev cache
Pilot:
`C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\cache\generic_predictions.jsonl`

Reranking-matrix per-condition Generic:
`C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\reranking_matrix\cache\dev_generic\`

### BGE embedding cache
`C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\cache\embedding_cache.sqlite3`

### Original M2 pair cache
`C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\m2_h5000\cache\pair_scores.sqlite3`

### EM2 hidden-state cache
`results\personalisation\external_memory\em2_hidden_dev\hidden_states.sqlite3`

Hidden cache SHA256:
`9a80a3314c184ccf3f0540916203c651474fad162dc3dab1fc97f7451f441df1`

---

## 8. Method map

### G / Generic
No personal history.

### F
Exact segmented-Pinyin historical target frequency.

Core:
`src/personalisation/context_memory.py`

### Original-M1
BGE semantic retrieval + memory support.

### Hidden-M1
PinyinGPT final hidden-state cosine retrieval + M1-style support.

Runner:
`experiments/external_memory/em2_hidden_m1_dev.py`

Selected frozen variant:
TopN=3, lambda=4.

### Original-M2
BGE Stage-1 + generic pretrained Cross-Encoder + target aggregation + Generic fusion.

Runner:
`experiments/external_memory/em2_original_m2_same_surface_dev.py`

### Hidden-M2
PinyinGPT hidden retrieval + generic Cross-Encoder + same M2 aggregation/fusion.

Runner:
`experiments/external_memory/em2_hidden_m2_dev.py`

### EM3-BCE v1
Task-specific Cross-Encoder fine-tuned from `BAAI/bge-reranker-base`.

Training runner:
`experiments/external_memory/em3_train_cross_encoder.py`

Final model:
`results\personalisation\external_memory\em3_cross_encoder_v1\train\final`

Training:
- pairs: 233,154
- positive: 86,959
- negative: 146,195
- epochs: 1
- mean loss: 0.543399

### EM3-Hidden
Hidden retrieval + EM3-BCE CE + M2-style aggregation/fusion.

Runner:
`experiments/external_memory/em3_hidden_surface_dev.py`

Selected Dev:
- K=10
- lambda=4

---

## 9. Current three-author Full+Short Dev baseline table

Primary metric: Macro-author Top1.

Fixed 5,608-query old three-author Dev-tune surface:

| Method | Overall | History | Ambiguous | Conflict |
|---|---:|---:|---:|---:|
| G | 0.722948 | 0.759399 | 0.683605 | 0.467924 |
| F | 0.765240 | 0.823281 | 0.754831 | 0.197192 |
| Original-M1 | 0.768888 | 0.828351 | 0.769454 | 0.297163 |
| Hidden-M1 | 0.768748 | 0.828688 | 0.766692 | 0.299645 |
| Original-M2 | 0.766869 | 0.825123 | 0.763249 | 0.258543 |
| EM3-Hidden | 0.767633 | 0.827217 | 0.761592 | 0.248205 |

Interpretation:
- current EM3-BCE v1 is a valid baseline but not a convincing end-to-end improvement;
- Conflict remains weak;
- better retrieval did not consistently convert to better candidate Top1.

---

## 10. EM3 history-discrimination result

Result root:
`results\personalisation\external_memory\em3_bce_v1_final_dev_tune\`

Score cache:
`scores.jsonl`

Summary:
`summary.json`

Pair-trainable old three-author Dev-tune queries:
1,609

Macro-author:
- Top1: 0.736181
- MRR: 0.815373
- MeanMargin: 2.622738
- MedianMargin: 1.285817
- PositiveBeatsNegative: 0.735853

Micro:
- Top1: 0.722809
- MRR: 0.809840
- MeanMargin: 1.498866
- MedianMargin: 0.825073
- PositiveBeatsNegative: 0.722188

Important failure pattern:
- Generic wrong / F right: EM3 history Top1 ≈ 0.761
- Generic right / F wrong: EM3 history Top1 ≈ 0.306
- formal Conflict: EM3 history Top1 ≈ 0.315

Working interpretation:
EM3 learns personal-memory evidence but often fails to suppress misleading history when Generic/current context is already correct.

---

## 11. Row-level comparison artifacts

Four-way:
`results\personalisation\external_memory\em2_four_way_dev_compare\rows.jsonl`

Original-M2:
`results\personalisation\external_memory\em2_original_m2_dev\rows.jsonl`

EM3 aligned comparison:
`results\personalisation\external_memory\em3_bce_v1_final_compare\`
- `summary.json`
- `rows_1609.jsonl`

EM3-Hidden:
`results\personalisation\external_memory\em3_hidden_dev\`
- `summary.json`
- `grid.json`
- `selected_rows.jsonl`
- `em3_pair_scores.jsonl`

Future evaluations MUST persist row-level output.

---

## 12. Current EM3-v2 direction

Do not blindly continue pointwise BCE.

Current clean-data direction:
- clean3 = Agent Phage + Etinjat + breaddddd;
- rebuild a fair clean3 BCE baseline if supervised comparison is needed;
- prepare ranking-oriented supervision;
- emphasize hard negatives / conflict negatives;
- keep no-Test discipline.

Possible v2 methods are documented in:
`docs/external_memory/em3/EM3_V2_METHOD_OPTIONS_2026-08-20.md`

Important:
the final condition (Full/Initial × Short/Multi3) is not frozen yet.

---

## 13. Parallel Codex research — completed Dev report

Codex isolated worktree:
`C:\Users\chiar\Desktop\LBH\thesis-codex-em3-research`

Final isolated Dev result:

- best method: transparent regularized utility gate over frozen G/F/M1;
- condition/population: Initial+Short clean3;
- untouched confirmation rows: 3,296;
- Macro-author Top1: Gate `0.349221`, F `0.346863`, delta `+0.002357`;
- Micro delta `+0.002731`; Ambiguous delta `+0.002702`; Conflict delta
  `+0.004177`;
- rescue/harm/net versus F: `80/71/+9`;
- 95% bootstrap CI `[-0.005365, 0.010024]`;
- exact McNemar `p=0.5152`;
- Full+Short independent support: `+0.001633` Macro-author Top1;
- no Test, no new GPU inference, existing caches reused.

Interpret this as a promising frozen Dev candidate, not a statistically proven
general performance advance. Do not merge or cherry-pick the isolated branch.
The canonical report and supporting records remain in that isolated worktree.

---

## 14. Exact resume order

1. Read this index, `EM3_DEV_CLOSEOUT_2026-08-20.md`, the all-outcome
   distribution record, and the final failure audit.
2. Re-run the old-three-author pair-generator audit and require the exact
   `30,968 / 86,959 / 146,195 / 233,154` checkpoint before changing sampling.
3. Generate and audit the clean3 pair manifest with the canonical generator;
   preserve source, runner, and output hashes. Do not start training before the
   chronology/no-Test audit passes.
4. Specify the prediction-visible global-frequency source and freeze the
   user-vs-global PersonalLift definition without using Dev labels as runtime
   features.
5. Implement a small Dev-only candidate-specific historical-utility experiment
   that preserves the 45 minority-target rescue mechanism and tests suppression
   on the 24/35 harmful subsets.
6. If supervised comparison remains justified, rebuild the BCE baseline on the
   same clean3 population before comparing any v2 objective.
7. Validate on a separate Dev confirmation surface with row-level outputs,
   paired bootstrap, and McNemar. Keep benchmark Test closed until the complete
   method and protocol are frozen later.

---

## 15. Search rules for future work

Before asking the user to search broadly:

1. Read this file.
2. Read `docs/FILE_INDEX.md`.
3. Read `docs/REPRODUCIBILITY_INDEX.md`.
4. Read the latest relevant EM3 progress/data-preparation note.
5. Check exact known result roots above.
6. Only then ask for a narrow command if a field/schema is still missing.

Do NOT repeatedly ask the user to run broad `Get-ChildItem -Recurse` searches.

When a new important script/result is created:
- update this file;
- update the relevant EM3 progress note;
- record exact path and purpose.

---

## 16. Known gotchas

- Matrix and Pilot use different row IDs; use `anchor_id` for stable mapping.
- Macro-author and Micro metrics must not be mixed.
- Candidate Top1 and history-discrimination Top1 are different tasks.
- Re_spectators is low-support in EM3 evaluation.
- MScarlet has script-normalisation confound.
- Tokenizer warning seen for EM3 final model; do not silently change tokenizer semantics mid-baseline.
- Old Full+Short Test results are historical evidence only; no current Test tuning.
- Large generated caches are local-only and should not be Git-added normally.
- Use targeted Git staging; never `git add .`.
