# Context-Aware Personal Memory

## Motivation

Personal frequency can identify a user's recurrent output for a segmented Pinyin sequence, but it cannot distinguish different meanings that share that Pinyin. Pilot A asks whether similarity between the current preceding context and strictly prior same-user contexts provides useful ranking evidence beyond both frozen Generic PinyinGPT (`G0`) and a transparent same-Pinyin frequency baseline (`F`). Context-Aware Personal Memory (`M`) is evaluated as a ranking method, not as a source of new candidates.

## Research Scope

The pilot is Dev-only and Full + Short only. It ranks the frozen Generic Top-10 surface. It does not augment vocabulary, adapt PinyinGPT weights, add internal model conditioning, use Test data, or implement T2/T3/T4. Frequency is a simple evaluation baseline rather than the claimed technical contribution.

## System Architecture

1. Frozen PinyinGPT2-Concat produces Generic candidates and cumulative log probabilities.
2. A chronology index exposes strictly prior records with the same author and exact segmented Pinyin.
3. `F` aggregates same-Pinyin target counts.
4. `M` embeds the current and historical contexts, retrieves the Top-N most similar visible records, and aggregates target-specific support.
5. Both methods reorder only the Generic candidate surface. Gold is used only after prediction for metrics and diagnostic subset membership.

## History Record

Each history record contains a stable row/interaction ID, author, work, work date and chronological index, source position, global chronological position, preceding context, segmented Pinyin, and selected target. History works precede Dev works by the frozen Evaluation V2 chronological split; records within a work are ordered by source position and stable anchor ID.

The prepared source contains 248,082 History rows and 32,212 Dev Full+Short rows. The Dev rows are not all visible at once: every query filters by a strict `historical_position < current_position` boundary.

## Information Boundary

Prediction receives only the current context, segmented Pinyin, frozen Generic candidates/scores, author identifier for isolation, chronological position, and strictly prior same-author history. The query type has no Gold field. The history filter also requires exact segmented-Pinyin equality. Another author, the current interaction, later Dev interactions, and all Test interactions are unavailable.

## Generic Candidate Pool

`G0`, `F`, and `M` have exactly the same up-to-Top-10 candidate texts. The runner asserts candidate-set equality and identical Missing@10 status on every evaluation row. No missing candidate can be introduced or recovered in this pilot.

Generic inference remains PinyinGPT2-Concat revision `76dd20dc92d8236a350fb732e99dde6fa15e2263`, official-code revision `8f1573ed0bd4d1f3d8d3f10a05f7e870725646f1`, beam 16, Top-K 10, oracle Full-Pinyin segments, frozen Evaluation V2 tokenizer-aware context handling, CUDA, and float32. Padding-free scheduling is capped at two independent requests because the previously probed batch-4 score delta exceeded the established semantic-equivalence tolerance.

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

Vectors are stored as normalized float32 BLOBs in SQLite. Cache keys hash the model SHA-256 and complete original context. Metadata fixes model hash, dimension, pooling, and normalization; a provenance mismatch stops reuse. The full prepared workload has 203,091 required unique contexts.

## Retrieval

Memory first filters to strictly prior, same-author, exact same-Pinyin records. Cosine similarity is computed on normalized context embeddings. Results are ordered by decreasing similarity, then earlier chronology, then stable interaction ID. The frozen Top-N grid is `{1, 3, 5, 10, 20}`.

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

Within each author, Dev works retain their frozen chronological order. The earlier `floor(number of Dev works / 2)`, with a minimum of one work, form the tune partition; all later Dev works form the reported pilot-evaluation partition. This produces 16,171 tune rows and 16,041 evaluation rows. Tune and evaluation work IDs are disjoint, every tune work precedes every evaluation work for its author, and evaluation Gold is not used for hyperparameter selection.

The selected configuration maximizes tune Macro-author Top-1. Frequency ties choose lower lambda. Memory ties choose lower lambda and then lower Top-N.

## Zero-History Fallback

When there is no visible prior same-Pinyin history, both `F` and `M` return the exact `G0` text/order. This is tested directly. The prepared manifest has prior same-author history for all 32,212 Dev rows and prior same-Pinyin history for 28,661 rows; the fallback remains required for the other rows and future use.

## Ambiguous Subset

A row is Ambiguous when visible strictly prior same-author, same-Pinyin history contains at least two distinct targets. The prepared Dev manifest identifies 17,017 such rows before any model result is inspected.

## Conflict Subset

A row is Conflict when it is Ambiguous, the historical same-Pinyin frequency winner is unique, and current Gold differs from that winner. If maximum historical frequency is tied, the row is excluded from Conflict rather than selecting an arbitrary winner. Conflict membership is computed after prediction because it uses Gold only as a diagnostic label.

## Transparency / Provenance

Every `M` output preserves candidate text, Generic rank and score, normalized Generic score, memory score, final score, and final rank. Retrieved evidence records historical interaction ID, historical target, similarity, non-negative weight, and chronological position. Generic, embedding, manifest, cache, hyperparameter, and runtime provenance are written to structured artifacts.

## Runtime Design

The CLI has `prepare`, `generic`, `embeddings`, `tune`, `evaluate`, `smoke`, and `all` phases. Generic JSONL appends and flushes durably and validates every existing row before resume. SQLite avoids repeated context embeddings across configurations. Tune and evaluation reuse both caches. Progress reports completed rows, throughput, and ETA where meaningful. Runtime artifacts separate Generic inference, embedding, retrieval, and reranking timings with mean, median, and nearest-rank P90.

The isolated 12-row engineering smoke completed real CUDA Generic inference, 36 real BGE embeddings, both ranking paths, provenance, cache writes, and metrics in 3.06 seconds. Smoke metrics are not research results. A conservative extrapolation for 32,212 Generic rows and 203,091 embeddings is several hours, approximately up to 5–6 hours plus content- and hardware-dependent tuning/evaluation overhead; this is not a measured full-run duration.

## Limitations

- Authors are proxy users, not observed IME users.
- Pinyin and composition boundaries are reconstructed or simulated.
- The pilot is Dev-only on Dataset V1 and is not a final cleaned-dataset thesis result.
- The candidate pool is frozen, so personal vocabulary recovery is impossible.
- PinyinGPT is not internally adapted and no new learned personal conditioning is added.
- Exact same-Pinyin filtering does not share evidence across related spellings.
- Context embedding quality and the pinned quantized model constrain retrieval.
- Hyperparameters are selected on an earlier Dev partition; no Test result is produced.

## Implementation Version History

| Date | Stage | Commit or tag | Status |
| --- | --- | --- | --- |
| 2026-08-17 | Chronological frequency/context-memory ranking | `f335715` | Implemented and unit-tested |
| 2026-08-17 | Dev Full+Short manifests, caches, runner, metrics, and smoke | `22eb1e0` | Implemented; 12-row isolated smoke passed |
| 2026-08-17 | Implementation checkpoint | `personalisation-pilot-a-implementation-v1` | Full manual run pending |
