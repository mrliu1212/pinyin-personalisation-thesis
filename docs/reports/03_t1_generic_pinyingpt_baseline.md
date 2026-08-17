# T1 Generic PinyinGPT Baseline

## 1. Research Question

T1 measures how well the frozen generic contextual Pinyin backend ranks the intended Chinese output on held-out future works before any personal history or author identity is introduced.

## 2. Scope and Dataset

This is a development evaluation on **Deep Author Dataset V1**. The six authors are proxy users rather than observed IME users. Pinyin, composition boundaries, and targets are reconstructed or simulated from text. Works are split chronologically within author, and T1 uses Test works only. Dataset V1.1 remains a separate cleaning experiment and was not substituted into this run.

## 3. Frozen Protocol

- Authors: Re_spectators, MScarlet, Etinjat, Agent Phage, QBLevi, and breaddddd
- Sampling: 1,000 Test anchors per author; 6,000 anchors total
- Conditions per anchor: Full + Short, Initial + Short, Full + Multi3, Initial + Multi3
- Total conditions: 24,000, with 6,000 per condition
- Multi3: exactly three consecutive frozen Jieba tokens
- Seed: 40408
- Sampling: deterministic, work-balanced round-robin within Test works
- Pinyin segmentation: oracle segments from the frozen manifest

## 4. Generic Backend

The model is `aihijo/transformers4ime-pinyingpt-concat` at checkpoint revision `76dd20dc92d8236a350fb732e99dde6fa15e2263`, with official-code reference `8f1573ed0bd4d1f3d8d3f10a05f7e870725646f1`.

It is a decoder-only GPT-2-style model with 12 layers, hidden size 768, 12 attention heads, vocabulary size 21,571, and 102,408,960 parameters. It is generic and population-level: author identity, history, and personalisation were all disabled, and Dev data was not scored.

## 5. Decoding

Decoding uses the frozen Pinyin compatibility masks, beam size 16, Top-K 10, and cumulative autoregressive log probability. Full conditions constrain each position by its oracle complete syllable; Initial conditions use the frozen single-letter compatibility keys. The model weights, tokenizer, prompt construction, compatibility rules, beam ownership, and ranking semantics were not changed.

## 6. Context Handling

T1 uses preceding context from the frozen Evaluation V2 manifest, not an IME Simulator profile. Immediately before inference, tokenizer-aware left truncation retains the most recent suffix that fits the model's 1,024-position limit after accounting for prompt, Pinyin, separators, and generated positions. The completed run reports `context_truncation_count = 0`: all 24,000 stored contexts fit without truncation.

## 7. KV-Cache / Inference Optimisation

The reference decoder recomputed the full prompt as output characters were extended. The integrated implementation reuses transformer attention key/value states and may schedule padding-free requests of identical prompt and target length together. Every condition retains its own independent beam-16 search.

This is a semantic-equivalent inference optimisation only. It does not intentionally change model weights, prompt semantics, constraints, cumulative score semantics, or candidate ordering. A batch size of four was not adopted because its score delta exceeded the established tolerance even though order was unchanged; the completed runner retained the validated batch-2 behavior.

## 8. Semantic Regression Validation

The formal gate compared fresh legacy-reference decoding with preserved batched/KV cache results for 12 frozen-manifest examples: three each from Full + Short, Initial + Short, Full + Multi3, and Initial + Multi3. It covered contexts up to 509 model tokens and targets up to nine characters.

- Candidate text/order equality: **12/12 exact**
- Rank changes: **0**
- Maximum absolute cumulative-log-probability delta: `0.0000324249267578125`
- Accepted established tolerance: `0.00005`
- Separate batch-2 probe maximum delta: `0.00003814697265625`, with identical order
- Tokenizer-aware left-truncation test: passed, retaining the most recent 1,018 of 1,100 tokens

The small floating-point differences are accepted only because candidate text/order did not change and the deltas remained within the established semantic-equivalence tolerance.

## 9. Runtime

The final invocation resumed 3,294 validated rows and added 20,706 rows without recomputing the prefix.

| Field | Value |
| --- | ---: |
| Inference elapsed | 1,174.6908 s (19 min 34.7 s) |
| Effective resumed throughput | 17.6268 conditions/s |
| Model load, excluded from inference time | 7.5458 s |
| Device | NVIDIA GeForce RTX 4060 Laptop GPU / CUDA 12.8 |
| Dtype | `torch.float32` |
| PyTorch | 2.11.0+cu128 |
| Transformers | 4.57.6 |
| Python | 3.12.13 |
| Physical GPU memory reported | 8,585,216,000 bytes |
| Peak allocated counter | 3,192,194,048 bytes |
| Peak reserved allocator counter | 12,213,813,248 bytes |

The recorded peak-reserved counter exceeds reported physical VRAM, so it is preserved as PyTorch allocator telemetry and is not interpreted as simultaneous resident memory. The actual peak-allocated counter is the more directly useful figure here.

## 10. Primary Result

> **Macro-author Top-1: 0.3752083 (37.52%)**

The metric is calculated separately for each of the six authors and then averaged with equal author weight.

## 11. Secondary Metrics

| Averaging | Top-1 | Top-3 | MRR@10 | Missing@10 | MeanRank\|Top10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Macro-author (primary family) | 0.375208 | 0.493417 | 0.442748 | 0.425708 | 2.044790 |
| Micro overall | 0.375208 | 0.493417 | 0.442748 | 0.425708 | 1.987521 |

The first four macro and micro values coincide because every author contributes exactly 4,000 rows. Conditional MeanRank differs because macro averaging gives each author's within-Top-10 mean equal weight.

## 12. Results by Condition

These are macro-author results; every condition has 1,000 rows per author.

| Condition | Top-1 | Top-3 | MRR@10 | Missing@10 | MeanRank\|Top10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full + Short | 0.723167 | 0.853500 | 0.793429 | 0.089667 | 1.539176 |
| Initial + Short | 0.329000 | 0.485833 | 0.422376 | 0.375000 | 2.513326 |
| Full + Multi3 | 0.376333 | 0.524167 | 0.458986 | 0.387167 | 2.214689 |
| Initial + Multi3 | 0.072333 | 0.110167 | 0.096201 | 0.851000 | 2.793537 |

## 13. Results by Author

Each author contributes 4,000 rows across the four conditions.

| Author | Top-1 | Top-3 | MRR@10 | Missing@10 | MeanRank\|Top10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Re_spectators | 0.449750 | 0.584750 | 0.525805 | 0.334750 | 1.860579 |
| MScarlet | 0.160500 | 0.231250 | 0.203711 | 0.697500 | 2.624793 |
| Etinjat | 0.366750 | 0.487250 | 0.436236 | 0.426750 | 2.029655 |
| Agent Phage | 0.417500 | 0.551000 | 0.492054 | 0.370250 | 1.893609 |
| QBLevi | 0.438500 | 0.555250 | 0.505328 | 0.360750 | 1.911224 |
| breaddddd | 0.418250 | 0.551000 | 0.493354 | 0.364250 | 1.948879 |

## 14. Full vs Initial

Across target lengths, Full conditions average 0.549750 Top-1 and Initial conditions average 0.200667, a difference of 0.349083 (34.91 percentage points). On the paired anchors:

- Short: Initial minus Full is −0.394167. There were 2,411 Full-correct/Initial-wrong pairs and 46 Full-wrong/Initial-correct pairs.
- Multi3: Initial minus Full is −0.304000. There were 1,887 Full-correct/Initial-wrong pairs and 63 Full-wrong/Initial-correct pairs.

Initial input was therefore harder than Full input in this frozen development run. No significance test was added.

## 15. Short vs Multi3

Across input forms, Short conditions average 0.526083 Top-1 and Multi3 conditions average 0.224333, a difference of 0.301750 (30.18 percentage points). Full drops from 0.723167 on Short to 0.376333 on Multi3; Initial drops from 0.329000 to 0.072333. Multi3 was harder under both Pinyin forms.

## 16. Error / Missing Diagnostics

Gold was missing from the candidate surface in 10,217 of 24,000 rows (42.57%). Missing@10 was 8.97% for Full + Short, 37.50% for Initial + Short, 38.72% for Full + Multi3, and 85.10% for Initial + Multi3. MScarlet had the highest author-level missing rate (69.75%); Re_spectators had the lowest (33.48%).

No inference failures occurred. In 564 rows, deterministic deduplication or the available compatible surface yielded fewer than ten distinct candidates; these remain valid up-to-Top-10 outputs. The context and gold-length diagnostics, per-work table, first 1,000 missing examples, and manual-review sample are retained as descriptive artifacts. Their observed associations are not interpreted causally here.

## 17. Generic Candidate Cache

`results/evaluation/deep_author_v2/t1/predictions.jsonl` contains exactly 24,000 durable JSONL rows. Each row preserves the frozen condition and provenance, actual model-used context, Pinyin constraint sequence, Gold, candidate text/rank/generic log probability, derived Gold rank metrics, beam/Top-K, checkpoint and official-code revisions, and runtime device.

The cache is a reusable generic-output artifact for later analysis or deliberately frozen reranking work. This report does not pre-commit future experiments to a particular personalisation method.

- Bytes: 122,698,571
- SHA-256: `764db39887f3db04b913d1739d9dbd46295f0e46e5a2bffa649f1563b56ee4e2`
- Cache validation: 24,000 unique known condition IDs; status `valid`

## 18. Limitations

- Authors are proxy users, not observed IME users.
- Pinyin and composition boundaries are reconstructed or simulated.
- Dataset V1 is a development source with known residual cleanliness limitations.
- T1 evaluates a generic backend only; no author identity, user history, or personalisation is available.
- This is not a claim that PinyinGPT is current commercial state of the art.
- These are not final cleaned-dataset thesis numbers; a later deliberately frozen run may differ.

## 19. Reproducibility

- Branch: `work/deep-author-evaluation-v2`
- Frozen design: `b145f2d0037f55abda071ee025f6adca2381c765` / `deep-author-evaluation-v2-design`
- Cached backend integration: `8c608f106ee7bb49ca5573e72de3da5eeb2290af`
- Resumable T1 runner and metrics: `5d270cd`
- Final tag: `deep-author-evaluation-v2-t1`
- Manifest: `results/evaluation/deep_author_v2/design/t1_condition_manifest.jsonl`
- Manifest SHA-256: `45b9cafedd7a8269d1f0b66d3f7f135ee990140e4b5b3668c67645863ab00d39`
- Predictions: `results/evaluation/deep_author_v2/t1/predictions.jsonl`
- Metrics: `results/evaluation/deep_author_v2/t1/metrics_summary.json`
- Runtime: `results/evaluation/deep_author_v2/t1/runtime_summary.json`
- Regression: `results/evaluation/deep_author_v2/t1/regression_summary.json`
- Cache validation: `results/evaluation/deep_author_v2/t1/cache_validation.json`
- Checksums: `results/evaluation/deep_author_v2/t1/artifact_checksums.json`
