# Personalisation Pilot A — Context-Aware Memory M1-H5000

## Purpose

This experiment tested whether strictly prior same-user history improves the
frozen Generic PinyinGPT candidate ranking. It compared Generic ordering
(`G0`), same-Pinyin frequency (`F-H5000`), and Context-Aware Personal Memory
(`M1-H5000`).

## H5000 Definition

The population is the exact 6,000 frozen T1 Full+Short Test anchors: 1,000 for
each of six proxy authors. H5000 first selects the 5,000 most recent strictly
prior legal History-split interactions for the same author and only then
filters exact segmented Pinyin. Dev Gold selected parameters; Test Gold did
not enter prediction or tuning.

## Methods

`F-H5000` aggregates normalized `log(1 + count)` support by historical target.
`M1-H5000` uses the pinned `bge-small-zh-v1.5-q8_0.gguf` model to retrieve
same-Pinyin histories by context cosine similarity, clips negative similarity
to zero, and normalizes target support. Both combine personal support with the
unchanged within-query Generic z-score and reorder only the frozen Generic
Top-10 surface.

Dev-only selection chose:

- `lambda_frequency = 4.0`
- `top_n = 5`
- `lambda_memory = 4.0`

## Completed Results

The values below are Macro-author metrics from
`results/personalisation/pilot_a_context_memory/h5000/metrics_summary.json`.

| Subset | Rows | Method | Top-1 | Top-3 | MRR@10 | Missing@10 | MeanRank\|Top10 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 6000 | G0 | 0.7231666666666667 | 0.8535 | 0.793428835978836 | 0.08966666666666667 | 1.5391761581065844 |
| Overall | 6000 | F-H5000 | 0.7718333333333334 | 0.8723333333333333 | 0.8249612433862433 | 0.08966666666666667 | 1.366685554535331 |
| Overall | 6000 | M1-H5000 | 0.7675000000000001 | 0.8713333333333333 | 0.8225744708994709 | 0.08966666666666667 | 1.3769072114411491 |
| History available | 3904 | G0 | 0.7567131615581154 | 0.8895599009681594 | 0.8273955882250704 | 0.05919995737819245 | 1.4809319863299928 |
| History available | 3904 | F-H5000 | 0.8290200604214916 | 0.9182803936124361 | 0.8743708114770946 | 0.05919995737819245 | 1.2391212591868312 |
| History available | 3904 | M1-H5000 | 0.8190291288899766 | 0.9167333460939123 | 0.8691770267721989 | 0.05919995737819245 | 1.257078016239148 |
| Ambiguous | 1661 | G0 | 0.7068355123795343 | 0.8784510968844751 | 0.7969496139616061 | 0.06143930268165531 | 1.5911274115759955 |
| Ambiguous | 1661 | F-H5000 | 0.7712972990214865 | 0.9131769325891655 | 0.8419084656448366 | 0.06143930268165531 | 1.3133934278593462 |
| Ambiguous | 1661 | M1-H5000 | 0.759040112296474 | 0.9093407138177344 | 0.8348421050458926 | 0.06143930268165531 | 1.3458619106140073 |
| Conflict | 377 | G0 | 0.43811986933227537 | 0.741105329282021 | 0.5969967415555996 | 0.1486167336778239 | 2.1202114378537793 |
| Conflict | 377 | F-H5000 | 0.18207160436483746 | 0.7671344464765517 | 0.47190672140319695 | 0.1486167336778239 | 2.2994045538351875 |
| Conflict | 377 | M1-H5000 | 0.1944904309518971 | 0.7551393263564316 | 0.4747994928669271 | 0.1486167336778239 | 2.358416145024004 |

The candidate pool was invariant across all three methods. Missing counts were
identical: 538 for `G0`, 538 for `F-H5000`, and 538 for `M1-H5000`.

## Interpretation

Personal history clearly improved `G0` overall and on rows with visible
history. Frequency slightly outperformed M1 overall. M1 nevertheless showed
contextual signal: it did not always reproduce the frequency result, and it
slightly improved the Conflict Top-1 value relative to frequency. General BGE
context cosine similarity was not sufficiently precise to outperform the
strong frequency baseline. Conflict was the main failure case: both personal
methods over-trusted the historical majority when the current intended target
differed from it.

## Limitations and M2 Direction

The six authors are proxies, Pinyin interactions are reconstructed, the
candidate pool cannot recover personal vocabulary, and H5000 is only one
history budget. M1 asks approximately, “Which previous contexts are
semantically similar?” M2 retains the same legal H5000 and BGE retrieval but
asks, “Does this historical interaction support this candidate in the current
context?” with a pretrained candidate-aware Cross-Encoder. Personal vocabulary,
other history budgets, wrong-user controls, and trained M3 scoring remain later
work.

## Provenance

- M1 implementation tag: `personalisation-pilot-a-h5000-implementation-v1`
- T1 prediction SHA-256: `764db39887f3db04b913d1739d9dbd46295f0e46e5a2bffa649f1563b56ee4e2`
- BGE GGUF SHA-256: `5a88d266870fbd27c6f329df60de80e2d4cf3bbd5e6f080bd5c1b2e5abb12039`
- Completed metrics SHA-256: `e35fb9efbe3bdd31d7f8354c227efbed2aa178855061955b3ac16a70137e424d`
- `status = complete`, `rows = 6000`, `generic_test_inference_rows = 0`,
  `test_gold_used_for_tuning = false`
