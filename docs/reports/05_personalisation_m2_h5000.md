# Personalisation M2-H5000 — Completed Result

## Purpose and Motivation

M2 tested whether a stronger candidate-aware second-stage scorer could improve
long-term personal candidate ranking beyond frozen Generic PinyinGPT (`G0`),
same-Pinyin Frequency (`F-H5000`), and BGE bi-encoder memory (`M1-H5000`). M1
asks which previous contexts are generally similar. M2 instead asks whether a
specific historical interaction supports a candidate in the current context.

The experiment used the exact 6,000 frozen T1 Full+Short Test anchors, with
1,000 rows for each of six proxy users. Test Gold did not select the method or
its parameters.

## Architecture

Stage 1 reuses M1 unchanged. It selects the 5,000 most recent strictly prior
legal History records for the same user, then filters exact segmented Pinyin.
The pinned `bge-small-zh-v1.5-q8_0.gguf` embeddings and cosine order retrieve
Top-K history from the existing profile-neutral embedding cache.

Stage 2 uses the pretrained Cross-Encoder `BAAI/bge-reranker-base`, revision
`2cfc18c9415c912f9d8155881c133215df768a70`. The model jointly receives current
preceding context, segmented Pinyin, candidate, historical preceding context,
and historical selected target. Current Gold, future text, other-user history,
and author name are absent. Raw logits are mapped by sigmoid, summed by
historical target, normalized within the query, and combined with the unchanged
Generic z-score:

```text
M2_support(c) = sum(sigmoid(logit(h)) where target(h) = c)
                / sum(sigmoid(logit(h)) over retrieved h)
Score_M2(c) = Z_generic(c) + lambda_m2 * M2_support(c)
```

M2 only reorders the frozen Generic Top-10 surface; it cannot introduce a
missing candidate.

## Dev Selection

The chronological 16,171-row Dev-tune partition selected parameters by
Macro-author Top-1. The frozen grids were K in `{10, 20}` and `lambda_m2` in
`{0.5, 1, 2, 4}`. Selection chose:

- `retrieval_k = 20`
- `lambda_m2 = 4.0`

The selection artifact records `test_rows_seen_during_selection = 0` and
`test_gold_used_for_selection = false`.

## Completed Results

All values are exact Macro-author metrics. `MeanRank|Top10` excludes missing
targets.

| Subset | System | Top-1 | Top-3 | MRR@10 | Missing@10 | MeanRank\|Top10 |
|---|---|---:|---:|---:|---:|---:|
| Overall | G0 | 0.7231666666666667 | 0.8535 | 0.793428835978836 | 0.08966666666666667 | 1.5391761581065844 |
| Overall | F-H5000 | 0.7718333333333334 | 0.8723333333333333 | 0.8249612433862433 | 0.08966666666666667 | 1.366685554535331 |
| Overall | M1-H5000 | 0.7675000000000001 | 0.8713333333333333 | 0.8225744708994709 | 0.08966666666666667 | 1.3769072114411491 |
| Overall | M2-H5000 | 0.765 | 0.8716666666666666 | 0.8209834656084656 | 0.08966666666666667 | 1.3823712353345725 |
| History Available | G0 | 0.7567131615581154 | 0.8895599009681594 | 0.8273955882250704 | 0.05919995737819245 | 1.4809319863299928 |
| History Available | F-H5000 | 0.8290200604214916 | 0.9182803936124361 | 0.8743708114770946 | 0.05919995737819245 | 1.2391212591868312 |
| History Available | M1-H5000 | 0.8190291288899766 | 0.9167333460939123 | 0.8691770267721989 | 0.05919995737819245 | 1.257078016239148 |
| History Available | M2-H5000 | 0.8157311355048013 | 0.9172886702588047 | 0.8670266735992009 | 0.05919995737819245 | 1.2631369717861334 |
| Ambiguous | G0 | 0.7068355123795343 | 0.8784510968844751 | 0.7969496139616061 | 0.06143930268165531 | 1.5911274115759955 |
| Ambiguous | F-H5000 | 0.7712972990214865 | 0.9131769325891655 | 0.8419084656448366 | 0.06143930268165531 | 1.3133934278593462 |
| Ambiguous | M1-H5000 | 0.759040112296474 | 0.9093407138177344 | 0.8348421050458926 | 0.06143930268165531 | 1.3458619106140073 |
| Ambiguous | M2-H5000 | 0.7503615549870789 | 0.9107746203813439 | 0.8293867632375697 | 0.06143930268165531 | 1.361394647291802 |
| Conflict | G0 | 0.43811986933227537 | 0.741105329282021 | 0.5969967415555996 | 0.1486167336778239 | 2.1202114378537793 |
| Conflict | F-H5000 | 0.18207160436483746 | 0.7671344464765517 | 0.47190672140319695 | 0.1486167336778239 | 2.2994045538351875 |
| Conflict | M1-H5000 | 0.1944904309518971 | 0.7551393263564316 | 0.4747994928669271 | 0.1486167336778239 | 2.358416145024004 |
| Conflict | M2-H5000 | 0.1643947485347861 | 0.7588066792625064 | 0.4531953458392603 | 0.1486167336778239 | 2.452531398922738 |

### Per-author Overall Top-1

| Author | G0 | F-H5000 | M1-H5000 | M2-H5000 |
|---|---:|---:|---:|---:|
| Agent Phage | 0.793 | 0.844 | 0.847 | 0.842 |
| Etinjat | 0.717 | 0.722 | 0.684 | 0.69 |
| MScarlet | 0.402 | 0.493 | 0.502 | 0.486 |
| QBLevi | 0.816 | 0.862 | 0.861 | 0.859 |
| Re_spectators | 0.814 | 0.846 | 0.848 | 0.848 |
| breaddddd | 0.797 | 0.864 | 0.863 | 0.865 |

## Candidate and Missing Invariance

`candidate_pool_invariant = true`. Missing counts were exactly 538 for G0,
F-H5000, M1-H5000, and M2-H5000. M2 performed zero Test PinyinGPT inference
and reused all 6,000 frozen T1 Generic rows. Completed M1 artifacts were
hash-checked before and after M2 and remained unchanged.

## Interpretation

Personal history clearly improves G0, but Frequency remains the strongest
completed personal-ranking method. M1 contains contextual signal without
outperforming Frequency overall. M2 replaces general bi-encoder cosine as the
final support score with a stronger candidate-aware pretrained Cross-Encoder,
yet its Overall Top-1 (`0.765`) remains below Frequency
(`0.7718333333333334`) and M1 (`0.7675000000000001`). It also does not improve
the Ambiguous or Conflict comparisons.

This is a useful negative and diagnostic result: stronger generic semantic
relevance is not automatically stronger personal preference modelling. The
result motivates studying candidate-set personalisation, where prior personal
vocabulary may recover targets absent from Generic Top-10. No statistical
significance claim is made because no significance test was run.

## Runtime and Limitations

DEV pair scoring added 133,845 scores in 992.4895486999885 seconds. Test added
45,117 scores in 336.8242548999842 seconds. Final arithmetic and artifact
generation took 55.25801590003539 seconds. The final pair cache contained
178,991 scores.

Limitations include proxy users, reconstructed Pinyin interactions, a fixed
H5000 window, a generic pretrained rather than task-trained reranker, generic
semantic training data, and the unchanged candidate surface. M2 cannot recover
personal vocabulary omitted by PinyinGPT and does not model temporal drift.

## Durable Artifacts

- `results/personalisation/m2_h5000/metrics_summary.json`
- `results/personalisation/m2_h5000/m2_predictions.jsonl`
- `results/personalisation/m2_h5000/selected_hyperparameters.json`
- `results/personalisation/m2_h5000/hyperparameter_search.csv`
- `results/personalisation/m2_h5000/metrics_by_author.csv`
- `results/personalisation/m2_h5000/metrics_by_subset.csv`
- `results/personalisation/m2_h5000/artifact_checksums.json`
- local resumable cache: `results/personalisation/m2_h5000/cache/pair_scores.sqlite3`

Key SHA-256 values:

- metrics: `9ad6acecf41b9f36aa1a1bf1bd702cfc729322c4226a4a6a9e3fde4082c6f6d8`
- predictions: `0a199c31e9fc7b9a35c39aef1cdf48f8a8514b1663fb37416844657eacac79fb`
- selected configuration: `e47e765b950804ceaed2d2fff5a4d2d1dba0ddeb652ac9bf20ccd89a42a182f4`

## Reproduction

From the personalisation worktree, with the pinned models already prepared:

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
& C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe `
  -m experiments.personalisation_m2_h5000 `
  --phase all `
  --dataset-root C:\Users\chiar\Desktop\LBH\thesis-deep-author\.build\dataset-v1-reconstruction `
  --pinyingpt-model C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat `
  --embedding-model C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf `
  --reranker-model C:\Users\chiar\Desktop\LBH\thesis\.build\bge-reranker-base `
  --t1-predictions C:\Users\chiar\Desktop\LBH\thesis-deep-author\results\evaluation\deep_author_v2\t1\predictions.jsonl `
  --batch-size 32
```
