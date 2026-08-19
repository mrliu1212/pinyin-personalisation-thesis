# EM-1 Frozen Test Result

Status: FROZEN TEST RESULT

Condition: Full+Short
History: H5000

Authors:
- Etinjat
- Re_spectators
- breaddddd

Dev-frozen configuration:
- Recovery K = 1
- Frequency lambda = 4

No Test parameter tuning was performed.

## Overall Test results

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

## Incremental R+F effect over F

Overall:
- Rescue: 10
- Harm: 11
- Net Top1: -1

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
132

Backend-reachable Generic Missing:
122

Backend-unreachable Generic Missing:
10

Recovered into unified candidate pool:
23

Of the 23 recovered Gold targets:
- Final Top10: 21
- Final Top3: 15
- Final Top1: 10

## Interpretation

EM-1 validates personal-history candidate recovery as a candidate-coverage
mechanism.

Recovery successfully restores Gold targets that are absent from the Frozen
Generic Top10 and improves Top3, MRR@10, and Missing@10.

However, on the frozen Test set, R+F does not improve overall Top1 beyond
the frequency-only F baseline. The incremental F -> R+F Top1 change is
10 rescues versus 11 harms, for a net change of -1 example.

Therefore EM-1 should not be claimed as a Top1 improvement over F.

Its main contribution is improved candidate availability and ranking depth.

The lack of improvement on Conflict examples also shows that candidate
recovery alone does not solve context-sensitive personal preference
selection. This motivates EM-2 and EM-3.

No Test retuning of K or lambda is permitted.
