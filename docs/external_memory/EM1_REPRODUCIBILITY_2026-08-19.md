# EM-1 Reproducibility Record

Status: FROZEN

Date: 2026-08-19

## Scope

Experiment:
EM-1 Recovery + Frequency Fusion

Condition:
Full+Short

History:
H5000 strictly-prior same-user history

Development and Test authors:
- Etinjat
- Re_spectators
- breaddddd

Frozen Dev-selected parameters:
- Recovery K = 1
- Frequency lambda = 4

Primary Dev selection metric:
Macro-author Overall Top1.

Test data was not used for parameter selection.

## Why this experiment exists

Earlier frequency and context-memory methods could only rerank the Frozen
Generic candidate surface.

EM-1 tests a separate question: can strictly-prior personal history recover
a Pinyin-compatible target that the Frozen Generic Top10 omitted?

The experiment therefore separates:
1. candidate recovery; and
2. reranking after recovery.

## Method definitions

G0:
Frozen Generic PinyinGPT Top10.

F:
Frozen frequency-only reranking over the original Generic candidate set.

R:
Add the first backend-compatible personal-only candidate from H5000 history
and score it exactly with the same Frozen PinyinGPT backend. The unified pool
is ranked by exact model log probability.

R+F:
Use the same exact-scored unified pool as R, then combine the
Generic-reference normalized model score with normalized historical frequency
using frozen lambda = 4.

This R is not the older PV1 approximate-boundary recovery.

## Critical recovery semantics

Personal candidates come from strictly-prior same-user H5000 history.

Candidates incompatible with the Frozen PinyinGPT tokenizer/constrained
Pinyin vocabulary are skipped.

K is counted after backend-compatibility filtering.

The selected personal-only candidate is exact-scored using the same Frozen
PinyinGPT model with fixed-candidate / teacher-forced scoring.

Gold is never used to select or score recovery candidates.

## Engineering compatibility gate

Before using exact fixed-candidate scoring for personal-only candidates,
cached Generic beam scores were compared with fresh fixed-candidate scores
for candidates already present on the Generic candidate surface.

Audit:
- 30 Dev rows
- 3 Generic candidates per row
- 90 comparisons
- tolerance: 1e-4
- maximum absolute difference: 2.47955322266e-05
- mean absolute difference: 3.53716313839e-06
- median absolute difference: 2.02655792236e-06
- gate: PASS

The 1e-4 tolerance matches the existing backend regression criterion using
four decimal places.

## Frozen input provenance

Frozen T1 prediction file:

C:\Users\chiar\Desktop\LBH\thesis-deep-author\results\evaluation\deep_author_v2\t1\predictions.jsonl

SHA256:

764db39887f3db04b913d1739d9dbd46295f0e46e5a2bffa649f1563b56ee4e2

H5000 Test state cache:

C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\personal_vocabulary_h5000\cache\test_states.jsonl

SHA256:

2912d32b8cd88843e825cb5592dfbc0a06e88e4a58831c632a126d2b8452b061

Frozen PinyinGPT checkpoint:

C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat

## Dev artifacts

Recovered fixed scores:

results\personalisation\external_memory\em1_recovered_scores_dev\recovered_candidate_scores.jsonl

SHA256:

ab80cb31d72383d2c9fbe887da4dc3082067a3e573893dcee565384099ac15f2

Dev comparison summary:

results\personalisation\external_memory\em1_dev_comparison\summary.json

SHA256:

01848cdc947b46ca2b5d3c03e78318bba7c4d19473842eb3200a55d0cae02446

Dev per-row evaluation:

results\personalisation\external_memory\em1_dev_comparison\rows.jsonl

SHA256:

02e63d9a3bf3d2a126a3624956a8f56c98fd94b36d3633dc7f6c46e44ba7106f

Dev selected:
- K = 1
- lambda_frequency = 4
- Macro-author Overall Top1 = 77.390%

The Dev selection was frozen before formal EM-1 Test evaluation.

Pre-Test Git checkpoint:

8179678 freeze EM-1 recovery and reranking dev selection

## Test artifacts

The exact Test artifact hashes should be copied from the local repository
after the final Test run:

```powershell
Get-FileHash `
  'results\personalisation\external_memory\em1_recovered_scores_test\recovered_candidate_scores.jsonl' `
  -Algorithm SHA256

Get-FileHash `
  'results\personalisation\external_memory\em1_test_evaluation\summary.json' `
  -Algorithm SHA256

Get-FileHash `
  'results\personalisation\external_memory\em1_test_evaluation\rows.jsonl' `
  -Algorithm SHA256
```

Record those three hashes here before marking the Test archive fully frozen.

## Frozen Test result

G0:
- Top1: 77.600%
- Top3: 90.967%
- MRR@10: 0.8465
- Missing@10: 4.400%

F:
- Top1: 81.067%
- Top3: 92.200%
- MRR@10: 0.8685
- Missing@10: 4.400%

R:
- Top1: 77.700%
- Top3: 91.067%
- MRR@10: 0.8476
- Missing@10: 4.200%

R+F:
- Top1: 81.033%
- Top3: 92.700%
- MRR@10: 0.8708
- Missing@10: 3.733%

F -> R+F:
- rescue: 10
- harm: 11
- net Top1: -1

Recovery:
- Raw Generic Missing: 132
- Backend-reachable Generic Missing: 122
- Backend-unreachable Generic Missing: 10
- Recovered to pool: 23
- Recovered to Top10: 21
- Recovered to Top3: 15
- Recovered to Top1: 10

## Reproduction commands

### Dev recovered-candidate scoring

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  -m experiments.external_memory.em1_score_recovered_dev `
  --generic-cache 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\cache\generic_predictions.jsonl' `
  --dev-states 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\personal_vocabulary_h5000\cache\dev_states.jsonl' `
  --checkpoint 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat' `
  --output-root 'results\personalisation\external_memory\em1_recovered_scores_dev' `
  --device cuda
```

### Dev comparison and parameter selection

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  -m experiments.external_memory.em1_dev_comparison `
  --generic-cache 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\cache\generic_predictions.jsonl' `
  --dev-states 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\personal_vocabulary_h5000\cache\dev_states.jsonl' `
  --recovered-scores 'results\personalisation\external_memory\em1_recovered_scores_dev\recovered_candidate_scores.jsonl' `
  --reachability 'results\personalisation\external_memory\em1_gold_reachability\rows.jsonl' `
  --output-root 'results\personalisation\external_memory\em1_dev_comparison'
```

### Frozen Test recovered-candidate scoring

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  -m experiments.external_memory.em1_score_recovered_test `
  --predictions 'C:\Users\chiar\Desktop\LBH\thesis-deep-author\results\evaluation\deep_author_v2\t1\predictions.jsonl' `
  --test-states 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\personal_vocabulary_h5000\cache\test_states.jsonl' `
  --checkpoint 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat' `
  --output-root 'results\personalisation\external_memory\em1_recovered_scores_test' `
  --device cuda
```

### Frozen Test evaluation

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  -m experiments.external_memory.em1_test_evaluation `
  --predictions 'C:\Users\chiar\Desktop\LBH\thesis-deep-author\results\evaluation\deep_author_v2\t1\predictions.jsonl' `
  --test-states 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\personal_vocabulary_h5000\cache\test_states.jsonl' `
  --recovered-scores 'results\personalisation\external_memory\em1_recovered_scores_test\recovered_candidate_scores.jsonl' `
  --checkpoint 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat' `
  --output-root 'results\personalisation\external_memory\em1_test_evaluation'
```

## Known limitations

A small number of benchmark Gold targets cannot be represented by the
Frozen PinyinGPT backend vocabulary.

These rows remain in primary evaluation for comparability.

Backend-reachable subsets are reported separately for recovery analysis.

Script-normalisation repair is also deferred and must use a separately
versioned dataset rather than silently changing Frozen Dataset V1.

## Related documentation

Dev parameter freeze:

docs/external_memory/EM1_DEV_SELECTION_2026-08-19.md

Frozen Test result:

docs/external_memory/EM1_TEST_RESULT_2026-08-19.md

Known issues:

docs/data_quality/KNOWN_ISSUES.md
