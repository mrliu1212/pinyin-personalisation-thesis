# Phase 4D — Transparent Contextual Memory Retrieval

## Objective

Test whether same-Pinyin retrieval from a user's frozen history can provide
useful, inspectable contextual evidence while confidence-aware backoff limits
the damage caused by weak matches.

Phase 4D is motivated by the completed Phase 4C result: Luna Base Top-1 was
77.97%, frequency personalisation reduced it to 75.27%, correct-user and
wrong-user results were close, and exact-context evidence was usually zero.
The Phase 4D experiment is not assumed to improve performance and must be run
before any performance claim is made.

## Scope

- Reuse the unchanged Phase 4C Zhu and Lu training histories and Zhu test set.
- Reuse each interaction's existing derived 12-Chinese-character context.
- Add transparent character TF-IDF retrieval from one active user's training
  history.
- Preserve the Phase 4C Base ordinal representation and `alpha=0.5` blending.
- Implement only the frozen no-gate and full Phase 4D conditions.
- Retain full per-query retrieval and candidate-score traces.

Phase 2 scoring, Phase 4C code/results, benchmark interactions, context length,
weights, and candidate generation remain unchanged.

## Frozen contextual memory

Each memory item is `(context, pinyin, selected_candidate)` with provenance.
Only history whose normalized Pinyin exactly matches the current Pinyin is
eligible. Correct-user memory contains only Zhu training interactions;
wrong-user memory contains only Lu training interactions. Both are frozen
before the first Zhu test interaction, with no test-time updates.

The representation is fitted independently for the active user's entire
training history:

- analyzer: character;
- n-gram range: `(1,2)`;
- term frequency: raw count;
- IDF: `log((1+n_documents)/(1+document_frequency))+1`;
- vector normalization: L2;
- similarity: cosine.

The implementation is a deterministic sparse representation and introduces no
external embeddings, neural models, pretrained models, or LLMs.

## Retrieval and contextual evidence

From eligible same-Pinyin history, retrieve the five highest positive cosine
similarities. `K=5` is frozen. Similarity ties use chronological interaction ID
order. If no positive match exists, retrieval is empty and `q=0`.

For retrieved items with similarities `s_i` and selections `y_i`:

```text
C(y) = sum(s_i where y_i = y) / sum(all retrieved s_i)
```

If the denominator is zero, `C(y)=0`. Each `C(y)` is bounded to `[0,1]` and all
retrieved contributors remain in the trace. Context confidence is:

```text
q = max(retrieved similarity), or 0 when retrieval is empty
```

## Frequency fallback

For candidates in the current Luna list, global and same-Pinyin training
counts are separately divided by the maximum count in that candidate list.
An all-zero channel remains all zero.

```text
F(y) = 0.25 * normalized_global(y)
     + 0.75 * normalized_pinyin(y)
```

The weights preserve the frozen Phase 4C global-to-Pinyin ratio `0.1:0.3`.

## Ablation and full model

Only two Phase 4D variants exist:

1. `phase_04d_no_gate`: remove confidence interpolation by fixing contextual
   use at full strength, so `U(y)=C(y)`.
2. `phase_04d_full`: use confidence-aware fallback:

```text
U(y) = (1-q) * F(y) + q * C(y)
```

The primary system is `phase_04d_full`.

## Base blending

Luna's rank-derived ordinal utility is
`candidate_count - base_rank + 1` and is min-max normalized within the current
candidate list. With frozen `alpha=0.5`:

```text
S_final(y) = 0.5 * normalized_Base(y) + 0.5 * U(y)
```

Candidates sort by descending final score. Exact score ties preserve the
original deterministic Base order.

## Evaluation design

The five reported conditions are:

1. Base Luna;
2. Phase 4C frequency personalisation;
3. Phase 4D no-gate, correct-user history;
4. Phase 4D full, correct-user history;
5. Phase 4D full, wrong-user history.

The unchanged full benchmark and Base-rerankable subset report Top-1/3/5/10,
MRR, mean target rank, and improved/unchanged/harmed counts. Phase 4D conditions
also report:

- queries with eligible same-Pinyin history;
- queries and percentage with non-zero contextual similarity;
- mean similarity across retrieved records;
- mean maximum similarity across evaluated queries, using zero for no match;
- ranking changes with non-zero contextual evidence;
- improved and harmed cases with non-zero contextual evidence.

## Transparency schema

Every Phase 4D evaluation row records:

- query interaction ID, work, date, context, Pinyin, and target;
- Base and personalised target ranks;
- eligible same-Pinyin history count;
- every retrieved interaction's context, Pinyin, selected candidate, and
  similarity;
- `q`;
- for every current candidate: Base rank, ordinal and normalized Base utility,
  raw/normalized global and Pinyin counts, `F(y)`, `C(y)`, contributing
  retrieved records, `q`, `U(y)`, final score, and final rank.

These fields are sufficient to reconstruct the ranking. Full traces are stored
under `evaluation_rows`; deterministic changed examples are duplicated under
`transparency_examples` for convenient inspection.

## Required behaviours and completion criteria

- TF-IDF fitting and transformation are deterministic.
- Identical non-empty contexts have cosine similarity 1 where represented.
- Retrieval is positive-similarity, same-Pinyin only, and at most five items.
- Correct-user and wrong-user memories cannot mix.
- Any test-time or future history is rejected.
- Context and frequency evidence remain within `[0,1]`.
- Full and no-gate formulas are directly testable from traces.
- Base order resolves final-score ties.
- Phase 4C and Phase 4B.6 artifact checksums remain unchanged.
- The full existing test suite passes before the experiment is run manually.

## How to run

After accepting the implementation, run:

```bash
.venv/bin/python -m experiments.exp_phase_04d_context_personalisation
```

The command writes `results/experiments/phase_04d/evaluation.json`. It is not
run automatically during implementation.

## Known limitations and deferred questions

- Character n-gram overlap is lexical rather than semantic similarity.
- Context vectors remain limited to the existing 12-character prefix.
- Only same-Pinyin history is eligible, even when another pronunciation might
  be linguistically related.
- Parameters are frozen for this experiment and are not tuned after inspecting
  test performance.
- Base Top-10 misses remain unrecoverable by reranking.

Abbreviated Pinyin/Jianpin remains outside this phase.
