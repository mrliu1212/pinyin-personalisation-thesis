# Personalisation M2-H5000

> **PENDING MANUAL/BACKGROUND RUN**

## Research Question

Can a pretrained candidate-aware second-stage Cross-Encoder improve the frozen
M1 H5000 result beyond `G0`, `F-H5000`, and `M1-H5000`, especially on Ambiguous
and Conflict rows?

## Frozen Method

M2 reuses the exact T1 Full+Short population, legal H5000 history, Generic
candidate surface, BGE Stage-1 embeddings, M1 diagnostic definitions, and
within-query Generic z-score. `BAAI/bge-reranker-base` jointly scores the
current context, segmented Pinyin, candidate, historical context, and
historical selected target. Full details are in
[Candidate-Aware Personal Memory M2](../research/candidate_aware_personal_memory_m2.md).

## Dev Selection

- Stage-1 K: `{10, 20}`
- `lambda_m2`: `{0.5, 1, 2, 4}`
- metric: Macro-author Top-1
- tie-break: lower lambda, then lower K
- Test Gold used for selection: false

Selected configuration: **PENDING MANUAL/BACKGROUND RUN**

## Final Results

**PENDING MANUAL/BACKGROUND RUN**

Report Overall, History Available, Ambiguous, Conflict, and per-author Top-1,
Top-3, MRR@10, Missing@10, and MeanRank|Top10 for `G0`, `F-H5000`,
`M1-H5000`, and `M2-H5000`. Missing@10 must be identical and
`candidate_pool_invariant` must be true before any result is claimed.

## Runtime and Provenance

**PENDING MANUAL/BACKGROUND RUN**

Durable outputs are under `results/personalisation/m2_h5000/`. Pair logits are
resumable in `cache/pair_scores.sqlite3`. The completed M1 artifacts are
validated by SHA-256 before and after M2 and are never overwritten.

## Limitations

M2 is an untrained pretrained reranker over a fixed candidate surface. It does
not implement personal vocabulary, H500, HFull, wrong-user controls, M3,
transparency, or user control.
