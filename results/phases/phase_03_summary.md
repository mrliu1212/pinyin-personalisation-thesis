# Phase 3 Result Summary

## Status

Phase 3 — Evaluation Framework: COMPLETE

## Objective

Validate that the evaluation framework can measure:
- base ranking;
- correct-user personalisation;
- wrong-user personalisation.

## Test Result

16 tests passed.

## Synthetic Evaluation Result

| Method | Top-1 | Top-3 | MRR | Mean Rank |
|---|---:|---:|---:|---:|
| Base | 0.485 | 1.000 | 0.732 | 1.576 |
| Correct-user | 0.636 | 1.000 | 0.813 | 1.394 |
| Wrong-user | 0.364 | 1.000 | 0.657 | 1.788 |

## Reranking Behaviour

Correct-user:
- Helpful: 9
- Harmful: 4
- Unchanged: 20

Wrong-user:
- Helpful: 4
- Harmful: 11
- Unchanged: 18

## Interpretation

The synthetic experiment verifies that the evaluation framework can distinguish
correct-user and wrong-user personalisation behaviour.

These results are validation of the experimental setup, not a claim of
real-world performance.

## Limitations

- Synthetic dataset only.
- Real author benchmark not evaluated.
- Candidate generation remains simplified.