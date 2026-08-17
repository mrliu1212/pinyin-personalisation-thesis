# Context-Aware Personal Memory

## Motivation

Personal frequency can identify a user's recurrent output for a segmented Pinyin sequence, but it cannot distinguish different meanings that share that Pinyin. Pilot A asks whether similarity between the current preceding context and strictly prior same-user contexts provides useful ranking evidence beyond both frozen Generic PinyinGPT (`G0`) and a transparent same-Pinyin frequency baseline (`F`). Context-Aware Personal Memory (`M`) is evaluated as a ranking method, not as a source of new candidates.

## Research Scope

M1-H5000 tunes hyperparameters on the frozen Dev tune partition, then evaluates once on the exact 6,000 Test anchors used by T1 Full + Short. It ranks the frozen Generic Top-10 surface. It does not augment vocabulary, adapt PinyinGPT weights, add internal model conditioning, or implement the complete T2 learning curve. Frequency is a simple baseline rather than the claimed technical contribution.

M1 means the first Context-Aware Personal Memory implementation: BGE bi-encoder embeddings plus cosine-similarity retrieval. It is not assumed to be the strongest final method, and cosine similarity is not claimed as novel. A possible future M2 would retain BGE first-stage retrieval and add pairwise/Cross-Encoder reranking; M2 is not implemented here.

## System Architecture

1. Frozen PinyinGPT2-Concat produces Generic candidates and cumulative log probabilities.
2. A chronology index exposes the most recent 5,000 strictly prior legal History-split records for the same author.
3. Exact segmented-Pinyin filtering is applied inside that bounded history.
4. `F-H5000` aggregates same-Pinyin target counts.
5. `M1-H5000` embeds the current and historical contexts, retrieves the Top-N most similar visible records, and aggregates target-specific support.
6. Both methods reorder only the Generic candidate surface. Gold is used only after prediction for metrics and diagnostic subset membership.

## History Record

Each history record contains a stable row/interaction ID, author, work, work date and chronological index, source position, chronological position, preceding context, segmented Pinyin, and selected target. The legal Test-time pool is the frozen Evaluation V2 `history` split recorded for T2 feasibility; Dev and Test outcomes are not added to that pool. Records are ordered by work chronology and source position.

The prepared source contains 248,082 History rows and 32,212 Dev Full+Short rows. H5000 means the 5,000 most recent strictly prior same-author legal records, selected before Pinyin filtering. It does not mean 5,000 same-Pinyin examples.

## Information Boundary

Prediction receives only the current context, segmented Pinyin, frozen Generic candidates/scores, author identifier for isolation, chronological position, and the bounded strictly prior same-author history. The query type has no Gold field. Another author, the current interaction, future interactions, Dev/Test outcomes, and current Test Gold are unavailable to prediction. Test Gold is used only after ranking for metrics and diagnostics.

## Generic Candidate Pool

`G0`, `F`, and `M` have exactly the same up-to-Top-10 candidate texts. The runner asserts candidate-set equality and identical Missing@10 status on every evaluation row. No missing candidate can be introduced or recovered in this pilot.

Test-time G0 is read directly from the completed T1 cache at `results/evaluation/deep_author_v2/t1/predictions.jsonl`, SHA-256 `764db39887f3db04b913d1739d9dbd46295f0e46e5a2bffa649f1563b56ee4e2`. Its 6,000 Full+Short rows use PinyinGPT2-Concat revision `76dd20dc92d8236a350fb732e99dde6fa15e2263`, official-code revision `8f1573ed0bd4d1f3d8d3f10a05f7e870725646f1`, beam 16, Top-K 10, oracle Full Pinyin, and frozen T1 context semantics. H5000 performs zero Test Generic inference.

## Frequency Baseline

For candidate `c`, visible frequency is:

```text
count(c) = number of strictly prior same-author, same-Pinyin records with target c
raw_frequency(c) = log(1 + count(c))
frequency_component(c) = raw_frequency(c) / max_candidate raw_frequency
score_F(c) = z_generic(c) + lambda_frequency * frequency_component(c)
```

If all candidate counts are zero, the personal component is zero. The frozen grid is `{0, 0.25, 0.5, 1, 2, 4}`. Ties in final score preserve Generic rank.

## Context Representation

The fixed embedding model is `bge-small-zh-v1.5`, the Q8_0 GGUF bundled with the audited HuoziIME v1.0.1-beta asset. Its independent upstream revision is unavailable, so reproducibility is fixed by SHA-256 `5a88d266870fbd27c6f329df60de80e2d4cf3bbd5e6f080bd5c1b2e5abb12039` and size 26,472,640 bytes.

The runtime is `llama-cpp-python 0.3.16`, mean pooling, 512 dimensions, and L2 normalization. Contexts exceeding the 512-token model budget are tokenizer-aware left-truncated to the most recent 510 content tokens before model-added tokens. This is the embedding model's capacity boundary, not a Simulator Interactive-32/64/128 profile.

Vectors are stored as normalized float32 BLOBs in one canonical SQLite cache. Keys hash model SHA-256, preprocessing version, pooling, normalization, and the complete original context. They contain no experiment name, history budget, split, or profile, so compatible entries remain reusable by Dev tuning, H500, H5000, HFull, T3, and a future M2 first stage. Metadata additionally fixes model identity, dimension, runtime, and runtime version; a mismatch stops reuse.

## Retrieval

Memory first selects the 5,000 most recent strictly prior same-author records, then filters to exact same Pinyin. Cosine similarity is computed on normalized context embeddings. Results are ordered by decreasing similarity, then earlier chronology, then stable interaction ID. The frozen Top-N grid is `{1, 3, 5, 10, 20}`.

Negative similarity has zero weight:

```text
weight(h) = max(cosine(current_context, historical_context_h), 0)
```

## Score Normalisation

Generic cumulative log probabilities are population-z-scored within each query's candidate surface:

```text
z_generic(c) = (generic_log_probability(c) - candidate_mean) / candidate_population_stddev
```

If all Generic scores are equal, every Generic component is zero. This normalization is frozen before the full run.

## Memory Score

For the selected Top-N retrieved records:

```text
memory_component(c) =
    sum(weight(h) for retrieved h with target c)
    / sum(weight(h) for all retrieved h)

score_M(c) = z_generic(c) + lambda_memory * memory_component(c)
```

If total positive weight is zero, every memory component is zero. The frozen `lambda_memory` grid is `{0, 0.25, 0.5, 1, 2, 4}`. Ties preserve Generic rank.

## Dev-Internal Tuning Boundary

Within each author, Dev works retain their frozen chronological order. The earlier `floor(number of Dev works / 2)`, with a minimum of one work, form the tune partition. This produces 16,171 tune rows. Later Dev works remain excluded from parameter selection. Test Gold is never used for tuning.

The selected configuration maximizes tune Macro-author Top-1. Frequency ties choose lower lambda. Memory ties choose lower lambda and then lower Top-N.

## Zero-History Fallback

When there is no visible prior same-Pinyin history, both `F` and `M` return the exact `G0` text/order. This is tested directly. The prepared manifest has prior same-author history for all 32,212 Dev rows and prior same-Pinyin history for 28,661 rows; the fallback remains required for the other rows and future use.

## Ambiguous Subset

A Test row is Ambiguous when its H5000-visible same-Pinyin history contains at least two distinct targets. Counts remain pending the manual run.

## Conflict Subset

A row is Conflict when it is Ambiguous, the historical same-Pinyin frequency winner is unique, and current Gold differs from that winner. If maximum historical frequency is tied, the row is excluded from Conflict rather than selecting an arbitrary winner. Conflict membership is computed after prediction because it uses Gold only as a diagnostic label.

## Transparency / Provenance

Every `M` output preserves candidate text, Generic rank and score, normalized Generic score, memory score, final score, and final rank. Retrieved evidence records historical interaction ID, historical target, similarity, non-negative weight, and chronological position. Generic, embedding, manifest, cache, hyperparameter, and runtime provenance are written to structured artifacts.

## Runtime Design

The H5000 CLI has `prepare`, `dev-generic`, `dev-embeddings`, `tune`, `test-embeddings`, `evaluate`, `smoke`, and `all` phases. Dev Generic JSONL is durable and resumable. Test G0 is validated and reused read-only from T1. One provenance-checked SQLite embedding cache is shared by Dev and Test and can be extended by later history budgets. Progress reports cache hits, missing work, and completed rows.

The corrected six-row engineering smoke validated six T1 Full+Short cache hits, H5000 visibility, and zero Generic inference. Smoke values are not research results. No H5000 runtime estimate is claimed before the manual run.

## Limitations

- Authors are proxy users, not observed IME users.
- Pinyin and composition boundaries are reconstructed or simulated.
- Dataset V1 and proxy authors remain development research assets rather than final cleaned-dataset thesis evidence.
- The candidate pool is frozen, so personal vocabulary recovery is impossible.
- PinyinGPT is not internally adapted and no new learned personal conditioning is added.
- Exact same-Pinyin filtering does not share evidence across related spellings.
- Context embedding quality and the pinned quantized model constrain retrieval.
- This is only the H5000 point; H500, HFull, and wrong-user HFull are required for the complete controlled history-size curve.

## Implementation Version History

| Date | Stage | Commit or tag | Status |
| --- | --- | --- | --- |
| 2026-08-17 | Chronological frequency/context-memory ranking | `f335715` | Implemented and unit-tested |
| 2026-08-17 | Dev Full+Short manifests, caches, runner, metrics, and smoke | `22eb1e0` | Implemented; 12-row isolated smoke passed |
| 2026-08-17 | Implementation checkpoint | `personalisation-pilot-a-implementation-v1` | Full manual run pending |
| 2026-08-17 | T1-aligned M1-H5000 implementation | `personalisation-pilot-a-h5000-implementation-v1` | 6,000-anchor manual run pending |
