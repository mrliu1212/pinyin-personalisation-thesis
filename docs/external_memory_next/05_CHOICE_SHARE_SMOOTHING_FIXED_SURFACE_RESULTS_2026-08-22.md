# 05 - Choice Share Smoothing Fixed-Surface Results

Date: 2026-08-22

Status: **POSITIVE TRAIN-VAL ABLATION / SMALL EFFECT / NO HOLDOUT CLAIM**

## 1. What was done

The predeclared fixed-surface empirical-Bayes Choice Share ablation was run on
34,416 Full+Short Train-Val rows. Version 1 tested alpha through 128. Because
128 was the upper boundary, a versioned boundary extension tested 256, 512,
1,024, 2,048, and 4,096 without changing any other term.

The boundary extension selected the interior point `alpha=128`; performance
declined above it. Alpha zero reproduced every frozen baseline Top10 and rank
exactly.

## 2. Frozen method

```text
P_smooth(c|u,p) = (n_c + alpha * P_all-Train-Fit(c|p)) / (N + alpha)
selected alpha = 128
candidate surface = frozen RetunedFinal Top10
w_P=2, w_CS=6, w_E=4, lambda_N=6, lambda_B=6
```

All Train-Fit observations precede Train-Val. Unseen candidate/Pinyin pairs
receive zero prior mass. Dev3000 and Test were unused.

## 3. Main result

| Metric | Raw Choice Share | Smoothed alpha=128 | Delta |
|---|---:|---:|---:|
| Macro-author Top1 | .796004927 | .796515499 | +.000510572 |
| Micro Top1 | .824994189 | .825459089 | +.000464900 |
| Top3 | .912075779 | .912337285 | +.000261506 |
| Top5 | .930758949 | .930729893 | -.000029056 |
| MRR@10 | .871377873 | .871652086 | +.000274213 |
| Missing@10 | .051981636 | .051981636 | 0 |

Top1 transitions against raw Choice Share are 25 rescues and 9 harms, net
`+16` rows.

Per-author Top1 changed as follows:

| Author | Raw | Smoothed | Delta |
|---|---:|---:|---:|
| Agent Phage | .887271669 | .887344444 | +.000072775 |
| Etinjat | .601494396 | .602241594 | +.000747198 |
| breaddddd | .899248715 | .899960459 | +.000711744 |

Conflict Macro-author Top1 rose from .252588201 to .255722738
(`+.003134537`). The improvement therefore is not produced by sacrificing one
author, although the absolute number of changed decisions is small.

## 4. Alpha curve

| Alpha | Macro Top1 | Micro Top1 | MRR@10 | Rescue | Harm | Net |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | .796004927 | .824994189 | .871377873 | 0 | 0 | 0 |
| 2 | .796232800 | .825197583 | .871497488 | 7 | 0 | +7 |
| 8 | .796255222 | .825255695 | .871539861 | 12 | 3 | +9 |
| 32 | .796432477 | .825400976 | .871603981 | 20 | 6 | +14 |
| 64 | .796432477 | .825400976 | .871618187 | 22 | 8 | +14 |
| **128** | **.796515499** | **.825459089** | **.871652086** | **25** | **9** | **+16** |
| 256 | .796473988 | .825430033 | .871637638 | 25 | 10 | +15 |
| 512 | .796432477 | .825400976 | .871623110 | 25 | 11 | +14 |
| 4,096 | .796390966 | .825371920 | .871614393 | 25 | 12 | +13 |

The interior optimum closes the initial upper-boundary concern.

## 5. History-depth diagnosis

No rank changed for `N=0`, `N=1`, or `N>100`. The largest Macro gains were
in `N=2` (`+.001943`) and `N=3-5` (`+.002037`), matching the sparse-history
hypothesis. `N=21-50` had a small Macro decline (`-.000576`) with unchanged
Micro Top1. Other nonzero bins improved modestly.

## 6. Interpretation and limitation

The data supports the direction of the overconfidence hypothesis: the causal
smoother improves Top1, Top3, MRR, every author's Top1, and conflict performance
while leaving coverage fixed. The later prior-decomposition record 10 shows
that most of this gain is reproduced by simpler Choice Share suppression, so
this record must not be read as strong evidence for population-prior value.
The effect is only 16 net Top1 rows, was selected on Train-Val, and slightly
reduces Top5 by one row. It is not evidence of statistical significance or
holdout generalisation.

The fixed surface deliberately excludes admission/removal effects. This is a
clean independent estimator result, not a complete dynamic-surface system.

## 7. Reproduction command

Use the command in record 04, replacing the module and output root with:

```powershell
-m experiments.external_memory_next.run_choice_share_smoothing_boundary_v2
--output-root '.\results\personalisation\external_memory_next\choice_share_smoothing_fixed_surface_boundary_v2'
```

All other arguments are identical to the version-1 command encoded in the
runner invocation history and its provenance manifest.

## 8. Hashes

- Boundary runner: `15db0f56be816877f8d55131d3ecfa17a4111aeb4fa55ee9bb6d5a28681535ce`
- Shared smoothing core: `a09fde45440d036fa50f8341f90c9f7f596c08b59de01b2eb5f08f0487451632`
- `result.json`: `7b5f8167d5be994bda19f94ce8c9aee99650bf67330c2b28ca3b906a34e519d0`
- `grid_results.json`: `120096168b4958be56848baf7f24b343730ff32af14911659de20bf302f6d84f`
- `selected_predictions.jsonl`:
  `41863119c16590bc67a8d39892cd8e45dceb783044814288c313f30313d8c2c2`

## 9. Next decision

The independent ablation is sufficiently consistent to justify one versioned
Train-Val retuning experiment. It will retune the coefficient attached to the
changed estimator and then the existing Stage-2 lambdas, while keeping all
other features and the candidate surface fixed.
