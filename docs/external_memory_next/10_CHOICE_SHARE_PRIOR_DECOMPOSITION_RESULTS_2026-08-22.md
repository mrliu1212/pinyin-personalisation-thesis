# 10 - Choice Share Prior Decomposition Results

Date: 2026-08-22

Status: **COMPLETE / OVERWEIGHTING SUPPORTED / PRIOR-SPECIFIC VALUE WEAK**

## 1. What was done

The five predeclared fixed-surface mechanisms in record 09 were evaluated on
all 34,416 Train-Val rows. This was a diagnostic comparison, not a new
parameter search. Raw `w_CS=6` reproduced the frozen result exactly.

## 2. Result

| Mechanism | Macro Top1 | Micro Top1 | Top3 | Top5 | MRR@10 | Rescue/Harm/Net |
|---|---:|---:|---:|---:|---:|---:|
| Raw CS, `w_CS=6` | .796004927 | .824994189 | .912075779 | .930758949 | .871377873 | 0/0/0 |
| Raw CS, `w_CS=2` | .796500349 | .825459089 | .912221060 | .930700837 | .871652558 | 24/8/+16 |
| Zero-prior shrink, alpha 128 | .796460940 | .825430033 | **.912366341** | .930700837 | **.871662809** | 29/14/+15 |
| All-author prior, alpha 128 | **.796515499** | **.825459089** | .912337285 | **.930729893** | .871652086 | 25/9/+16 |
| Other-author prior, alpha 128 | .796485199 | **.825459089** | .912337285 | .930642724 | .871634652 | 28/12/+16 |

All-author smoothing wins the pre-specified Macro-author Top1 criterion, but
its advantage over raw `w_CS=2` is only `.000015150`, and over zero-prior
shrinkage only `.000054559`. Micro Top1 and net Top1 are tied with the simpler
scale control. Zero-prior shrinkage has the best Top3 and MRR.

Conflict Macro Top1 is `.252588` for the frozen reference, `.257242` for raw
`w_CS=2`, `.257670` for zero-prior shrinkage, `.255723` for all-author prior,
and `.257306` for other-author prior. The all-author prior is not the best
conflict correction.

## 3. Interpretation

The reproducible finding is:

```text
Raw personal Choice Share is over-weighted on this frozen surface.
```

The data does not provide strong evidence that candidate-specific population
prior information is responsible for the gain. Conservative rescaling or
zero-prior shrinkage recovers nearly all of it. The all-author smoother remains
the formal Train-Val Macro selection, but should be described as a small
regularized-Choice-Share operating point, not a demonstrated Bayesian-prior
breakthrough.

This result strengthens the rationale for nonlinear fusion: the useful
operation appears conditional and signal-dependent, and a tree ranker can be
given raw Choice Share, history depth, concentration, source, and contextual
supports separately rather than hiding the interaction in one global weight.

## 4. Limitations

- Candidate surface and Missing@10 are fixed.
- Every comparison uses the same repeatedly observed Train-Val set.
- Differences are small and no significance test was predeclared.
- No Dev3000 or Test data was read.

## 5. Reproduction

Use the same input arguments as record 04 with:

```powershell
-m experiments.external_memory_next.run_choice_share_prior_decomposition_v1
--output-root '.\results\personalisation\external_memory_next\choice_share_prior_decomposition_fixed_surface_v1'
```

## 6. Hashes

- Runner: `5d1af44aa8cedc32b343800581de3583af13a7e003af5739f0e2dd6c7c4d701e`
- Smoothing core: `a09fde45440d036fa50f8341f90c9f7f596c08b59de01b2eb5f08f0487451632`
- `result.json`: `af832bb0549589181f2905a19df728c103d82fd2143d9cdafa995fb0d747a796`
- `used_dev3000=false`; `used_test=false`.

## 7. Current decision

Experiment A is closed. Carry the all-author alpha-128 point as the formal
Macro-selected smoothing result, while treating conservative Choice Share
suppression as the more strongly supported mechanism. Proceed to nonlinear
fusion only after its causal Train-Fit feature table passes audit.
