# EM-1 Frozen Test Result

Status: FROZEN TEST RESULT

Condition: Full+Short  
History semantics: H5000

Authors:
- Etinjat
- Re_spectators
- breaddddd

Dev-frozen configuration:
- Recovery K = 1
- Frequency lambda = 4

No Test parameter tuning was performed.

## Why EM-1 was run

Frequency reranking can only reorder candidates already present in the
Frozen Generic Top10.

EM-1 tests whether strictly-prior personal history can also recover
Pinyin-compatible personal targets that are absent from the Generic
candidate surface.

The method deliberately separates candidate recovery from reranking.

## Method definitions

G0:
Frozen Generic PinyinGPT Top10.

F:
Frozen frequency-only reranking over the original Generic candidate set.

R:
Add the first backend-compatible personal-only candidate from H5000 history,
score it exactly with the same Frozen PinyinGPT backend, and rank the unified
candidate pool by exact PinyinGPT log probability.

R+F:
Use the same exact-scored unified candidate pool as R and add the frozen
frequency signal.

This R is the EM-1 exact-scored recovery method and is not the older PV1
approximate-boundary recovery.

## Frozen Test results

| Method | Top1 | Top3 | MRR@10 | Missing@10 |
|---|---:|---:|---:|---:|
| G0 | 77.600% | 90.967% | 0.8465 | 4.400% |
| F | 81.067% | 92.200% | 0.8685 | 4.400% |
| R | 77.700% | 91.067% | 0.8476 | 4.200% |
| R+F | 81.033% | 92.700% | 0.8708 | 3.733% |

## Per-author Top1

G0:
- Etinjat: 71.700%
- Re_spectators: 81.400%
- breaddddd: 79.700%

F:
- Etinjat: 72.200%
- Re_spectators: 84.600%
- breaddddd: 86.400%

R:
- Etinjat: 71.900%
- Re_spectators: 81.400%
- breaddddd: 79.800%

R+F:
- Etinjat: 71.700%
- Re_spectators: 84.600%
- breaddddd: 86.800%

## Incremental F -> R+F

Overall:
- Rescue: 10
- Harm: 11
- Net Top1 change: -1

History Available:
- Rescue: 10
- Harm: 11
- Net: -1

Ambiguous:
- Rescue: 3
- Harm: 0
- Net: +3

Conflict:
- Rescue: 0
- Harm: 0
- Net: 0

## Recovery analysis

Raw Generic Missing:
- 132

Backend-reachable Generic Missing:
- 122

Backend-unreachable Generic Missing:
- 10

Recovered into the unified candidate pool:
- 23

Of these recovered Gold targets:
- Final Top10: 21
- Final Top3: 15
- Final Top1: 10

## Interpretation

EM-1 validates personal-history recovery as a candidate-coverage mechanism.

It improves Top3, MRR@10, and Missing@10 by restoring targets omitted from
the Frozen Generic Top10.

It does not improve overall Test Top1 beyond the frequency-only F baseline:
R+F has 10 rescues and 11 harms relative to F, giving a net Top1 change of
-1 example.

Therefore EM-1 must not be described as a Top1 improvement over F.

Its main demonstrated value is candidate availability and ranking depth.

The lack of improvement on Conflict examples also shows that candidate
recovery alone does not solve context-sensitive personal preference
selection. This motivates EM-2 and EM-3.

No Test retuning of K or lambda is permitted.

## Recovery opportunity funnel

The limited overall effect of EM-1 is partly explained by the size of the
recovery opportunity.

Across the 3,000 frozen Test examples:

- Total Test rows: 3,000
- Rows with at least one backend-compatible personal-only candidate: 306
- Share of Test rows with an injectable personal-only candidate: 10.2%
- Generic Top10 already contained Gold: 2,868
- Generic Missing: 132
- Backend-reachable Generic Missing: 122
- Backend-unreachable Generic Missing: 10
- Gold successfully recovered into the unified pool: 23
- Recovered Gold reaching final Top10: 21
- Recovered Gold reaching final Top3: 15
- Recovered Gold reaching final Top1: 10

The 306 eligible rows should not be interpreted as 306 opportunities to
recover the correct answer. They only indicate that personal history contains
at least one additional backend-compatible candidate outside the Generic
Top10.

The actual useful recovery opportunity is much smaller: among the 122
backend-reachable cases where Generic Top10 missed Gold, EM-1 recovered Gold
in 23 cases.

Therefore the main limitation of EM-1 is not that exact candidate scoring
fails. Rather, the number of examples where the correct missing candidate is
both available in personal history and recoverable is relatively small.

This explains why Recovery improves candidate coverage and ranking depth but
produces little overall Top1 gain.

## EM-1 final conclusion

EM-1 demonstrates that External Memory can recover valid personal candidates
that are absent from the Frozen Generic candidate surface.

However, its contribution is primarily candidate coverage rather than final
Top1 accuracy.

The frozen Test evidence is:

- G0 Top1: 77.600%
- F Top1: 81.067%
- R Top1: 77.700%
- R+F Top1: 81.033%

Relative to F, R+F produces:

- 10 Top1 rescues
- 11 Top1 harms
- net Top1 change: -1 example

At the same time, R+F improves:

- Top3: 92.200% -> 92.700%
- MRR@10: 0.8685 -> 0.8708
- Missing@10: 4.400% -> 3.733%

The correct interpretation is therefore:

EM-1 is useful for recovering missing personal candidates, but the recovery
opportunity is limited and Recovery alone is not sufficient to solve
context-sensitive personalisation. This motivates EM-2 and EM-3, which focus
on selecting the relevant historical evidence rather than only expanding the
candidate surface.
