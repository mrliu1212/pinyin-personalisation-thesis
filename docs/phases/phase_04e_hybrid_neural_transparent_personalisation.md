# Phase 4E — Hybrid Neural-Transparent Personalisation

## Objective

Combine frozen neural semantic signals with explicit user history and an
interpretable learned linear reranker. The neural feature extractors may be
opaque, but every final score must decompose into named, auditable factors and
support deterministic counterfactual recomputation.

Phase 4E is a frozen post-hoc extension after the Zhu test set was observed in
earlier phases. No model, feature, parameter, pool rule, or normalization may
be changed after inspecting its final result.

## Preserved baselines and data

- Phase 4B.6 interactions, segmentation, Pinyin, targets, and Luna candidates
  remain unchanged.
- Phase 4C and Phase 4D implementations and result JSON files remain unchanged.
- Phase 4D no-gate is imported as the existing lexical-retrieval baseline.
- Correct-user history is the existing Zhu training partition.
- Wrong-user history is the existing Lu training partition.
- Zhu test and Lu held-out selections never enter training or memory.
- Test histories are frozen; there are no online updates.

## Frozen neural models

| Role | Repository | Resolved revision |
| --- | --- | --- |
| Causal semantic scorer | `Qwen/Qwen3-0.6B-Base` | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| Semantic embedding | `Qwen/Qwen3-Embedding-0.6B` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |

The revisions are stored in
`results/experiments/phase_04e/model_manifest.json`. Official runs reuse that
manifest. Models run in evaluation mode under `torch.no_grad()` on MPS, CUDA,
or CPU fallback. Frozen inference uses float16 on MPS/CUDA and float32 on CPU.
No fine-tuning, generation, chat template, prompting for
answers, or sampling is used. Model weights are never committed.

Semantic features are cached under ignored `.cache/phase_04e/` paths keyed by
model revision, context, candidate where applicable, and scoring configuration.

## Context and causal-LM scoring

`context_12` preserves the existing derived context. `semantic_context_64` is
the final 64 Chinese characters from the existing raw preceding context; it
contains no target or following text.

Context and candidate are tokenized separately with
`add_special_tokens=False`, then manually concatenated. Candidate-token
autoregressive log probabilities are averaged:

```text
L_cond(y) = sum candidate-token log probabilities conditioned on context
            / candidate token count

L_prior(y) = mean candidate-token log probability from the minimal
             BOS/EOS-compatible starting prefix

L_gain(y) = L_cond(y) - L_prior(y)
```

Conditional and gain values are independently min-max normalized within the
active candidate pool; constant vectors become zero. The LM only scores supplied
candidates and never generates strings.

## Semantic personal memory

Historical and query contexts use `semantic_context_64`. Only exact
same-normalized-Pinyin history is eligible. Queries use the frozen instruction:

> Given a current Chinese text context, retrieve previous contexts in which the
> same Pinyin input was used in a semantically similar way.

Historical documents have no retrieval instruction. Embeddings are L2
normalized and ranked by cosine similarity. Retrieve at most `K=5`, retaining
negative similarities in the trace but using `w_i=max(similarity,0)` for
evidence.

For candidate `y`:

```text
memory_weighted_share(y) = sum(w_i selecting y) / sum(all w_i)
memory_max_similarity(y) = max non-negative supporting similarity, else 0
memory_support_count(y)  = number of retrieved items selecting y
memory_any_support(y)    = 1 when any retrieved item selects y, else 0
```

If every non-negative weight is zero, all four candidate ranking features above
are zero; the retrieved records remain visible as audit metadata. No
hand-written confidence gate is used.

## Behaviour and personal vocabulary

Behavioural metadata retains raw global and same-Pinyin counts for audit and
uses the specified ranking features: their `log1p` transforms, same-Pinyin
selection share, seen flag, and `recency=1/(1+d)`, where `d` is the number of
active-user interactions since the latest same-Pinyin selection.

Personal vocabulary maps Pinyin to historical selections. At most three
candidates absent from Luna Top-10 are injected, ordered by descending
same-Pinyin count, most recent selection, then Unicode lexical order. Every
injection records all responsible history IDs. It has no Base rank, zero Base
utility, and `personal_vocab_injected=1`; the pool may contain 13 candidates.

## Exact feature vectors

`GENERIC_CONTEXT_MODEL` uses exactly:

1. `normalized_base_utility`
2. `candidate_char_length`
3. `normalized_lm_conditional`
4. `normalized_lm_context_gain`

`HYBRID_PERSONAL_MODEL` uses exactly:

1. `normalized_base_utility`
2. `candidate_char_length`
3. `personal_vocab_injected`
4. `normalized_lm_conditional`
5. `normalized_lm_context_gain`
6. `memory_weighted_share`
7. `memory_max_similarity`
8. `memory_support_count`
9. `memory_any_support`
10. `log1p_global_count`
11. `log1p_same_pinyin_count`
12. `same_pinyin_selection_share`
13. `candidate_seen_same_pinyin`
14. `recency`

No additional ranking feature is included.

## Pairwise linear training

Generic fusion weights use both authors' designated training interactions but
no personal features. Hybrid weights also use both authors, with each training
interaction seeing only the same author's strict chronological prefix.

For interactions whose target is in the active pool, fit `StandardScaler` on
candidate rows, then construct target-minus-competitor positive pairs and their
reversed negative pairs. Fit exactly:

```text
LogisticRegression(
    fit_intercept=False,
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    random_state=40408,
)
```

Inference score is the coefficient-weighted sum of standardized candidate
features. The same Hybrid model handles fixed and augmented pools.

## Evaluation conditions

Exactly seven conditions are prepared:

1. `base`
2. `phase_04d_no_gate_correct_user` imported unchanged
3. `phase_04e_generic_context`
4. `phase_04e_hybrid_fixed_correct_user`
5. `phase_04e_hybrid_fixed_wrong_user`
6. `phase_04e_hybrid_augmented_correct_user`
7. `phase_04e_hybrid_augmented_wrong_user`

Full and original-Luna-rerankable subsets report Top-1/3/5/10, MRR, mean rank,
missing count, coverage, and rank-change counts. Augmented conditions report
recovery at Top-1/3/5/10. Per-work results cover `to_my_late_wife` and `spring`.

Frozen statistical comparisons use paired exact McNemar for Top-1 and 10,000
paired bootstrap resamples for MRR and mean-rank effects, with seed `40408`.
Statistics describe effects and uncertainty; they are not tuning criteria.

## Auditability and counterfactuals

Every candidate row stores identity/source, Base fields, all LM scores, complete
retrieval traces and memory evidence, behaviour, standardized feature values,
coefficients, exact per-feature contributions, grouped factor contributions,
final score, and final rank. Feature and factor sums are programmatically
verified against the score.

Factor families are Base IME, semantic context, personal semantic memory,
historical behaviour, and personal vocabulary. Selected examples rerank after
zeroing one family at a time without retraining. Supported examples also delete
one retrieved memory at a time, recompute memory features, and rerank with the
same trained model.

## Contamination limitation

Pretraining membership for these famous historical works is unknown. Absolute
neural results are not clean unseen-text generalisation. The explicit risk is
recorded under `results/audits/phase_04e/`.

## Completion criteria and command

Unit tests use deterministic stubs and never download neural weights. A tiny
real-model smoke test validates loading, finite scores/embeddings, device, and
caching without evaluating Zhu test performance.

After implementation acceptance, the frozen final experiment command is:

```bash
.venv/bin/python -m experiments.exp_phase_04e_hybrid_personalisation
```

The command is not run automatically.
