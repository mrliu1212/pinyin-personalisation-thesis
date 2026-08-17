# Reranking Personalisation Matrix

## Purpose

This experiment completes the first personalisation research line by evaluating
the already-frozen ranking methods across input ambiguity, composition length,
and legal personal-history size. It does not introduce a new ranking method.
It preserves Personal Vocabulary as a separate completed candidate-availability
line and does not implement M3.

The matrix contains four T1 input conditions, three non-zero history budgets,
and three personal ranking methods: 36 cells. G0 supplies the H0 point in each
learning curve. The completed Full+Short/H5000 F, M1, and M2 cells are reused
byte-for-provenance and are never recomputed.

## Frozen Conditions and Population

The conditions are `full_short`, `initial_short`, `full_multi3`, and
`initial_multi3`. Each contains the exact frozen 6,000 T1 Test anchors: 1,000
per proxy user. The durable 24,000-row T1 prediction file is validated against
the frozen condition manifest and its SHA-256 before use. Test PinyinGPT
inference must remain zero.

H500 and H5000 retain the latest 500 or 5,000 strictly prior same-user records
before exact segmented-Pinyin filtering. HFull retains all legal strictly prior
same-user History-split records before that filter. H0 is G0; F/M1/M2 are not
rerun at H0.

## Frozen Ranking Methods

G0 is the frozen Generic Top-10 candidate surface and scores.

F calls the existing `rank_frequency` implementation. It uses normalized
`log(1 + count)` support and the existing lambda grid `{0, 0.25, 0.5, 1, 2,
4}`. No second frequency implementation exists.

M1 calls the existing BGE retrieval and memory-ranking functions. It uses
normalized context embeddings, positive cosine support, candidate-target
aggregation, Top-N `{1, 3, 5, 10, 20}`, and lambda `{0, 0.25, 0.5, 1, 2, 4}`.

M2 uses the frozen BGE Stage-1 retrieval and pretrained
`BAAI/bge-reranker-base` Cross-Encoder. It does not train or fine-tune the
model. Its grid is retrieval K `{10, 20}` and lambda `{0.5, 1, 2, 4}`. The
candidate surface remains Generic Top-10.

## Dev-only Selection and Diagnostics

Every new condition/budget group selects F, M1, and M2 parameters on the
corresponding chronologically earlier whole-work Dev-tune partition. Selection
maximizes Macro-author Top-1, with the existing lower-weight/lower-width tie
breaks. Test rows seen during selection are zero.

Diagnostic membership is recomputed from the exact history visible under each
budget:

- History Available: at least one visible same-Pinyin record;
- Ambiguous: at least two distinct historical targets;
- Conflict: Ambiguous, one unique frequency winner, and Gold differs from it;
  tied winners are excluded.

Gold is used only after ranking for metrics and diagnostic labels. The
prediction-visible `PredictionQuery` contains no Gold field.

## Shared History and Cache Architecture

One condition-aware History/Dev manifest is generated from the frozen Dataset
V1 work split. `HistoryIndex` groups by user and pre-indexes segmented Pinyin.
For each query it first derives the author-level prior ordinal window and then
bisects matching-Pinyin records inside that window. This is exactly equivalent
to budget-before-Pinyin filtering while avoiding an HFull corpus rescan.

For a condition/budget group, Generic predictions, visible histories,
frequency statistics, BGE retrieval, and diagnostic flags are prepared once
and consumed by F/M1/M2.

The 24,000 Test Generic predictions are read-only. New Dev conditions may need
Dev-only Generic inference for legal parameter selection; those rows are
resumable and separate from Test.

The BGE cache key depends only on the frozen model/preprocessing and context
text. It contains no condition, budget, method, or experiment name. Exact
contexts are shared across budgets and conditions.

The M2 pair key depends only on current ID/context/Pinyin, historical
ID/context/target, candidate, and frozen model/template/truncation provenance.
It contains no budget or matrix-cell name. H500, H5000, HFull, and the focused
wrong-user control therefore share identical pair scores whenever their
semantic inputs coincide.

## Manifest, Resume, and Failure Semantics

`matrix_manifest.json` has one entry per method/condition/budget cell. States
are `reused_complete`, `pending`, `running`, `complete`, or `failed`. Every
completed method writes its predictions and `result.json` immediately. A
restart uses the same command, skips `complete` and `reused_complete` cells,
and retries incomplete or failed groups without editing the manifest.

Important JSON metadata is written to a temporary sibling and atomically
renamed. One group failure records its traceback without deleting completed
cells. The finalizer refuses `complete` while any required cell is unfinished.

The worker self-finalizes `condition_matrix.csv`, `learning_curves.csv`,
`context_diagnostics.csv`, `wrong_user_summary.json`, selections, metrics,
cache/runtime summaries, checksums, and `COMPLETE.json`.

## Wrong-user Control

One Full+Short/HFull control uses the frozen cyclic `AUTHORS` mapping. Every
user maps to a different user, and F/M1/M2 use the same mapping. Test
performance does not choose it. The output records correct-user and wrong-user
metrics, per-author data, and correct-minus-wrong Top-1 deltas.

## Execution Order and Scientific Questions

After audit, Full+Short H500 and HFull run first because H5000 is already
complete. The three remaining conditions then run H500, H5000, and HFull. The
worker aggregates learning curves and conflict diagnostics, runs the one
wrong-user control, revalidates all prior hashes, and writes completion only
after every integrity check succeeds.

The outputs support later analysis of monotonic history gains, frequency
saturation, contextual methods under Initial/Multi3/HFull, Overall versus
Conflict trade-offs, user specificity, and evidence relevant to a later M3.
They do not answer those questions before results exist.

## Commands

From the `thesis-personalisation` worktree:

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$common = @(
  '--dataset-root', 'C:\Users\chiar\Desktop\LBH\thesis-deep-author\.build\dataset-v1-reconstruction',
  '--pinyingpt-model', 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat',
  '--embedding-model', 'C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf',
  '--reranker-model', 'C:\Users\chiar\Desktop\LBH\thesis\.build\bge-reranker-base',
  '--t1-predictions', 'C:\Users\chiar\Desktop\LBH\thesis-deep-author\results\evaluation\deep_author_v2\t1\predictions.jsonl'
)
& $python -m experiments.reranking_personalisation_matrix --phase audit @common
& $python -m experiments.reranking_personalisation_matrix --phase smoke @common
& $python -m experiments.reranking_personalisation_matrix --phase run @common
```

The `run` command is also the exact resume command. Monitor with:

```powershell
Get-Content results\personalisation\reranking_matrix\matrix_stdout.log -Tail 30 -Wait
Get-Content results\personalisation\reranking_matrix\matrix_stderr.log -Tail 50
Get-Content -Raw results\personalisation\reranking_matrix\matrix_manifest.json
Get-Content -Raw results\personalisation\reranking_matrix\COMPLETE.json
```

## Limitations

The matrix uses six proxy authors, reconstructed Pinyin, frozen Dataset V1,
fixed candidate surfaces, and pretrained generic contextual models. Initial
Pinyin can expose much larger exact-history buckets, so HFull has substantial
legitimate embedding and pair-score work. The experiment has no recency decay,
online adaptation, task-specific training, candidate injection, Personal
Vocabulary extension, M3, or user interface.
