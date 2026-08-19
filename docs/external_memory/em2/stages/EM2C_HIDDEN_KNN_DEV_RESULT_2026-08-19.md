# EM-2C Hidden-State kNN Dev Retrieval Result

Status: COMPLETE / FROZEN

Date: 2026-08-19

## Scope

- Full+Short
- H5000
- Dev tune only
- Authors:
  - Etinjat
  - Re_spectators
  - breaddddd

Representation:

- Frozen PinyinGPT2-Concat
- final layer
- final prompt [SEP] hidden state
- 768 dimensions
- cosine similarity

No tuning was performed.

Test was not used.

## Population

All queries:
- 5,608

History Available:
- 3,625

Gold History Available:
- 3,213

Ambiguous + Gold History:
- 1,609

Conflict + Gold History:
- 359

## Overall Gold-History retrieval

Micro:
- R@1: 0.869281
- R@5: 0.971366
- R@10: 0.989107

Macro-author:
- R@1: 0.890920
- R@5: 0.977833
- R@10: 0.992256

## Ambiguous Gold-History retrieval

Micro:
- R@1: 0.738968
- R@5: 0.942822
- R@10: 0.978247

Macro-author:
- R@1: 0.768247
- R@5: 0.957042
- R@10: 0.986311

Primary metric:

- Macro-author Ambiguous R@1 = 0.768247

## Conflict Gold-History retrieval

Micro:
- R@1: 0.431755
- R@5: 0.768802
- R@10: 0.902507

Macro-author:
- R@1: 0.424646
- R@5: 0.808620
- R@10: 0.931717

## Interpretation

The Frozen PinyinGPT hidden representation contains useful retrieval signal,
including on Ambiguous and Conflict cases.

This result does NOT yet establish superiority over BGE.

A strict same-surface BGE comparison is required before making that claim.

## Leakage / tuning checks

Gold used for representation construction:
- No

Gold used for retrieval ordering:
- No

Gold used for post-retrieval evaluation:
- Yes

Test used:
- No

Tuning performed:
- No

## Next

EM-2D:

Re-evaluate BGE full-context and frozen BGE ctx64 retrieval on the exact same
EM-2C query/history/subset surface.
