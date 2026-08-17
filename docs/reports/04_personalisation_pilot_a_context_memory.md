# Personalisation Pilot A — Context-Aware Memory

> **Implementation-stage report skeleton. Final numerical results: PENDING MANUAL FULL RUN.**

## 1. Research Question

Can strictly prior same-user contextual history improve Generic PinyinGPT candidate ranking beyond both frozen Generic ordering (`G0`) and a transparent same-Pinyin frequency baseline (`F`)?

## 2. Scope

- Dataset V1 development source
- Dev works only; Test is unavailable
- Full + Short only
- Six proxy authors
- Ranking over the frozen Generic up-to-Top-10 surface
- No vocabulary augmentation or internal PinyinGPT adaptation

## 3. Frozen Method

`G0` is frozen PinyinGPT2-Concat. `F` adds normalized `log(1 + same-Pinyin target count)` support. `M` retrieves strictly prior same-author, exact same-Pinyin contexts using pinned normalized BGE embeddings and adds normalized non-negative similarity support. Generic scores use within-query population z-score normalization. Full details are in [Context-Aware Personal Memory](../research/context_aware_personal_memory.md).

## 4. Dev Split

The deterministic manifest has 32,212 Dev rows. Within author, earlier whole Dev works form the tune population and later whole Dev works form the evaluation population: 16,171 tune rows and 16,041 evaluation rows. Work sets are disjoint and chronological. Test rows: zero.

## 5. Frozen Hyperparameter Grid

- `lambda_frequency`: `{0, 0.25, 0.5, 1, 2, 4}`
- memory Top-N: `{1, 3, 5, 10, 20}`
- `lambda_memory`: `{0, 0.25, 0.5, 1, 2, 4}`
- Selection: tune Macro-author Top-1
- Tie-breaking: lower lambda; for memory, then lower Top-N

Selected parameters: **PENDING MANUAL FULL RUN**

## 6. Metrics

Primary: Macro-author Top-1. Secondary: Top-3, MRR@10, Missing@10, and MeanRank|Top10. Report `G0`, `F`, and `M` overall and for history-available, Ambiguous, and Conflict subsets, plus per-author results.

## 7. Final Results

**PENDING MANUAL FULL RUN**

Do not infer values from the isolated smoke test.

## 8. Runtime

**PENDING MANUAL FULL RUN**

The runtime report will separate Generic inference, embedding/cache work, memory retrieval, and reranking.

## 9. Expected Result Artifacts

All paths are below `results/personalisation/pilot_a_context_memory/`:

- `dev_manifest.jsonl`, `history_manifest.jsonl`, `dev_split_summary.json`
- `generic_predictions.jsonl`, `generic_runtime.json`
- `embedding_cache.sqlite3`, `embedding_runtime.json`
- `frequency_hyperparameter_search.csv`, `memory_hyperparameter_search.csv`
- `selected_hyperparameters.json`
- `frequency_predictions.jsonl`, `memory_predictions.jsonl`
- `ambiguous_subset.jsonl`, `conflict_subset.jsonl`
- `metrics_summary.json`, `metrics_by_author.csv`, `metrics_by_subset.csv`
- `runtime_summary.json`, `artifact_checksums.json`

## 10. Limitations

Proxy authors, reconstructed input, Dataset V1 Dev-only scope, fixed Generic candidate pool, no vocabulary recovery, no internal model adaptation, and no Test result. This pilot cannot establish final thesis performance or commercial-state-of-the-art claims.

## 11. Reproducibility

- Branch: `work/personalisation-pilot-a`
- Starting checkpoint: `deep-author-evaluation-v2-t1` / `14d584a17c4ae0a284b25bcdc892d3b12e439745`
- Framework commit: `f335715`
- Runner commit: `22eb1e0`
- Implementation tag: `personalisation-pilot-a-implementation-v1`
- PinyinGPT checkpoint revision: `76dd20dc92d8236a350fb732e99dde6fa15e2263`
- Official code revision: `8f1573ed0bd4d1f3d8d3f10a05f7e870725646f1`
- Embedding SHA-256: `5a88d266870fbd27c6f329df60de80e2d4cf3bbd5e6f080bd5c1b2e5abb12039`
- Full numerical result completion: **PENDING MANUAL FULL RUN**
