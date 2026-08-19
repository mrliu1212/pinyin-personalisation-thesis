# EM-2D Same-Surface Representation Comparison

Status: COMPLETE / GO TO END-TO-END

Date: 2026-08-19

## Question

Does the Frozen PinyinGPT task-native hidden representation retrieve
useful personal history better than the existing generic BGE semantic
representation?

## Frozen comparison surface

All compared methods use:

- Full+Short
- H5000
- Dev tune only
- Etinjat
- Re_spectators
- breaddddd
- 5,608 queries
- same-user history
- strictly-prior History + earlier Dev interactions
- H5000 applied before exact segmented-Pinyin filtering
- exact same segmented Pinyin
- cosine similarity
- retrieval recall conditional on Gold existing in legal history
- identical Ambiguous and Conflict definitions

Population:

- History Available: 3,625
- Gold History Available: 3,213
- Ambiguous + Gold History: 1,609
- Conflict + Gold History: 359

No Test result was used.

## Representations

### BGE Full

bge-small-zh-v1.5 using the full stored preceding context.

### BGE ctx64

bge-small-zh-v1.5 using the most recent 64 characters of the preceding
context for both query and historical representations.

ctx64 was the previously Dev-selected BGE representation.

### PinyinGPT hidden

Frozen PinyinGPT2-Concat final-layer hidden state at the final prompt [SEP]
token.

The extraction definition was frozen by EM-2A before retrieval performance
was inspected.

## Macro-author retrieval results

| Representation | Overall R@1 | Ambiguous R@1 | Conflict R@1 |
|---|---:|---:|---:|
| BGE Full | 88.41% | 73.03% | 34.53% |
| BGE ctx64 | 88.36% | 75.09% | 41.05% |
| PinyinGPT hidden | 89.09% | 76.82% | 42.46% |

Primary metric:

Macro-author Ambiguous R@1.

PinyinGPT hidden vs BGE ctx64:

- +1.74 percentage points

PinyinGPT hidden vs BGE Full:

- +3.80 percentage points

## Conflict retrieval depth

| Representation | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| BGE Full | 34.53% | 73.86% | 84.39% |
| BGE ctx64 | 41.05% | 70.28% | 80.69% |
| PinyinGPT hidden | 42.46% | 80.86% | 93.17% |

Relative to BGE ctx64, PinyinGPT hidden improves:

- Conflict R@1: +1.41 pp
- Conflict R@5: +10.59 pp
- Conflict R@10: +12.48 pp

## Interpretation

The task-native Frozen PinyinGPT representation provides stronger retrieval
evidence than both the generic full-context BGE representation and the
previously selected local-context BGE ctx64 representation.

The result is particularly encouraging on Ambiguous cases, which form the
pre-registered primary diagnostic, and on deeper Conflict retrieval.

The Full-vs-ctx64 controls also indicate that the result cannot be explained
simply by PinyinGPT seeing a longer context: PinyinGPT also outperforms the
full-context BGE representation.

## Important limitation

This is a retrieval-stage result only.

Prior Context Strengthening experiments demonstrated that improved retrieval
does not necessarily translate into improved final candidate ranking.

Therefore EM-2 is not yet considered an end-to-end personalisation success.

## Decision

GO.

Proceed to an independently designed Dev end-to-end experiment using the
frozen PinyinGPT hidden-state retrieval representation.

Do not change:
- hidden layer;
- extraction position;
- pooling;
- representation definition;

based on downstream results.

## Next

EM-2E:

Test whether PinyinGPT hidden-state historical evidence can improve final
candidate ranking beyond the existing Generic/Frequency baselines.
