# EM-1 Dev Selection

Status: FROZEN DEV SELECTION

Condition: Full+Short
History semantics: H5000
Development authors:
- Etinjat
- Re_spectators
- breaddddd

## Method

EM-1 separates:

1. Recovery:
   inject backend-compatible personal-only candidates from strictly-prior
   same-user H5000 history.

2. Reranking:
   exact Frozen PinyinGPT fixed-candidate score plus frequency evidence.

Recovered candidates are exact-scored by the same Frozen PinyinGPT backend.

## Development grid

Recovery K:
- 1
- 3
- 5

Frequency lambda:
- 0.5
- 1
- 2
- 4
- 8

Primary selection metric:
Macro-author Overall Top1.

Tie-break:
lower frequency lambda, then lower K.

No Test result was used for parameter selection.

## Selected configuration

K = 1
lambda_frequency = 4

Macro-author Overall Top1:
77.390%

Micro Overall Top1:
68.937%

## Main Dev comparison

Generic G0:
- Micro Top1: 62.892%
- Macro-author Top1: 72.295%

Frozen F:
- Micro Top1: 68.170%
- Macro-author Top1: 76.524%

Recovery only, K=1:
- Micro Top1: 63.195%
- Macro-author Top1: 72.832%

Selected R+F:
- Micro Top1: 68.937%
- Macro-author Top1: 77.390%

Increment over Frozen F:
- Micro Top1: +0.767 percentage points
- Macro-author Top1: +0.866 percentage points

## Incremental F -> R+F effect

Overall:
- Rescue: 47
- Harm: 4
- Net: +43

Backend-Reachable:
- Rescue: 47
- Harm: 4
- Net: +43

History Available:
- Rescue: 47
- Harm: 4
- Net: +43

Ambiguous:
- Rescue: 7
- Harm: 0
- Net: +7

Conflict:
- Rescue: 1
- Harm: 0
- Net: +1

Per author:
- Etinjat: +15 net
- Re_spectators: +2 net
- breaddddd: +26 net

## Recovery conversion

Backend-reachable Generic Missing:
669

Recovered into candidate pool at K=1:
125

Of these:
- final Top10: 105
- final Top3: 78
- final Top1: 46

Pool-to-Top1 conversion:
46 / 125 = 36.8%

## Interpretation

Frequency remains the main personalisation signal.

However, exact-scored candidate recovery provides additional value beyond
frequency-only reranking.

The gain is mainly a candidate-coverage gain: personal history can restore
targets that the Frozen Generic Top10 omitted.

Only a small part of the incremental gain occurs on Ambiguous or Conflict
examples. Therefore EM-1 should not be interpreted as solving
context-sensitive disambiguation.

Context-sensitive conflicts remain the motivation for later EM-2 and EM-3.

## Frozen decision

Use:

- Recovery K = 1
- Frequency lambda = 4

for subsequent EM-1 evaluation.

Do not retune these values using Test results.

## Frozen artifact hashes

Recovered candidate fixed-score cache:

`AB80CB31D72383D2C9FBE887DA4DC3082067A3E573893DCEE565384099AC15F2`

Dev comparison summary:

`01848CDC947B46CA2B5D3C03E78318BBA7C4D19473842EB3200A55D0CAE02446`

Dev comparison per-row output:

`02E63D9A3BF3D2A126A3624956A8F56C98FD94B36D3633DC7F6C46E44BA7106F`

These hashes were recorded before formal EM-1 Test evaluation.
