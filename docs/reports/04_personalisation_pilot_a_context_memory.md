# Personalisation Pilot A — Context-Aware Memory M1-H5000

> **Implementation-stage report skeleton. M1-H5000 numerical results: PENDING MANUAL RUN.**

## 1. Research Question

Can strictly prior same-user contextual history improve Generic PinyinGPT candidate ranking beyond both frozen Generic ordering (`G0`) and a transparent same-Pinyin frequency baseline (`F`)?

## 2. Scope

- Dataset V1 frozen Evaluation V2 split
- Hyperparameter selection on Dev only
- One evaluation on the exact 6,000 frozen T1 Test anchors
- Full + Short only
- H5000 personal-history budget
- Six proxy authors
- Ranking over the frozen Generic up-to-Top-10 surface
- No vocabulary augmentation or internal PinyinGPT adaptation

## 3. Frozen Method

`G0` is reused read-only from the completed T1 Full+Short cache. `F-H5000` adds normalized `log(1 + same-Pinyin target count)` support. `M1-H5000` first takes the 5,000 most recent strictly prior same-author History-split interactions, then filters exact same Pinyin, retrieves by pinned BGE cosine similarity, and adds normalized non-negative support. Generic scores use within-query population z-score normalization. Full details are in [Context-Aware Personal Memory](../research/context_aware_personal_memory.md).

## 4. Dev Split

The deterministic Dev manifest has 32,212 rows. Earlier whole Dev works form the 16,171-row tune population. Later Dev works are not used for parameter selection. After parameters are frozen, evaluation uses all 6,000 T1 Full+Short Test anchors, exactly 1,000 per author. Test Gold is never available to tuning or prediction.

## 5. Frozen Hyperparameter Grid

- `lambda_frequency`: `{0, 0.25, 0.5, 1, 2, 4}`
- memory Top-N: `{1, 3, 5, 10, 20}`
- `lambda_memory`: `{0, 0.25, 0.5, 1, 2, 4}`
- Selection: tune Macro-author Top-1
- Tie-breaking: lower lambda; for memory, then lower Top-N

Selected parameters: **PENDING MANUAL RUN**

## 6. Metrics

Primary: Macro-author Top-1. Secondary: Top-3, MRR@10, Missing@10, and MeanRank|Top10. Report `G0`, `F-H5000`, and `M1-H5000` overall and for history-available, Ambiguous, and Conflict subsets, plus per-author results.

## 7. M1-H5000 Results

**PENDING MANUAL RUN**

Do not infer values from the isolated smoke test.

## 8. Runtime

**PENDING MANUAL FULL RUN**

The runtime report will separate Generic inference, embedding/cache work, memory retrieval, and reranking.

## 9. Expected Result Artifacts

Shared Dev/cache paths are below `results/personalisation/pilot_a_context_memory/`; H5000 outputs are below its `h5000/` child:

- `dev_manifest.jsonl`, `history_manifest.jsonl`, `dev_split_summary.json`
- `cache/generic_predictions.jsonl`, `generic_runtime.json` (Dev tune only)
- `cache/embedding_cache.sqlite3` (shared across history budgets)
- `frequency_hyperparameter_search.csv`, `memory_hyperparameter_search.csv`
- `selected_hyperparameters.json`
- `h5000/test_manifest.jsonl`, `h5000/manifest_summary.json`
- `h5000/frozen_hyperparameters.json`
- `h5000/frequency_predictions.jsonl`, `h5000/memory_predictions.jsonl`
- `h5000/ambiguous_subset.jsonl`, `h5000/conflict_subset.jsonl`
- `h5000/metrics_summary.json`, `h5000/metrics_by_author.csv`, `h5000/metrics_by_subset.csv`
- `h5000/runtime_summary.json`, `h5000/artifact_checksums.json`

## 10. Limitations

Proxy authors, reconstructed input, Dataset V1, fixed Generic candidate pool, no vocabulary recovery, and no internal model adaptation remain limitations. H5000 alone is not the complete T2 history-size curve; H500, HFull, and wrong-user controls remain future work.

## 11. Reproducibility

- Branch: `work/personalisation-pilot-a`
- Starting checkpoint: `deep-author-evaluation-v2-t1` / `14d584a17c4ae0a284b25bcdc892d3b12e439745`
- Framework commit: `f335715`
- Runner commit: `22eb1e0`
- Implementation tag: `personalisation-pilot-a-implementation-v1`
- PinyinGPT checkpoint revision: `76dd20dc92d8236a350fb732e99dde6fa15e2263`
- Official code revision: `8f1573ed0bd4d1f3d8d3f10a05f7e870725646f1`
- Embedding SHA-256: `5a88d266870fbd27c6f329df60de80e2d4cf3bbd5e6f080bd5c1b2e5abb12039`
- T1 Full+Short manifest: 6,000 anchors / SHA-256 `45b9cafedd7a8269d1f0b66d3f7f135ee990140e4b5b3668c67645863ab00d39`
- T1 completed predictions: SHA-256 `764db39887f3db04b913d1739d9dbd46295f0e46e5a2bffa649f1563b56ee4e2`
- H5000 implementation tag: `personalisation-pilot-a-h5000-implementation-v1`
- M1-H5000 numerical result completion: **PENDING MANUAL RUN**
