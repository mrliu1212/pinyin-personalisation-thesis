# Final Research Closure — 2026-08-29

## Project

**Transparent and User-Controllable Personalisation for Chinese Pinyin Input**

This document closes the final five-author experimental cycle of the thesis project.

The following are frozen by this closure:

- final dataset and split;
- final architecture;
- training and evaluation protocol;
- five-system held-out Test comparison;
- Test-condition analyses;
- candidate-recovery analysis;
- interpretation boundaries and future-version rules.

No further architecture selection or hyperparameter tuning should use the Test results recorded here. Any future experimental change must be treated as a new version, with a new branch/output directory.

---

## 1. Final study population

The final study uses five authors:

1. Re_spectators
2. Etinjat
3. Agent Phage
4. QBLevi
5. breaddddd

MScarlet was excluded because of source-data quality problems before final splitting, training, model selection, and Test evaluation. This was a data-quality exclusion, not a result-based exclusion.

Final eligible works: **248**

Historical source population: **904,162 raw interactions**

---

## 2. Frozen final dataset

| Split | Rows |
|---|---:|
| Fit | 150,000 |
| Validation | 20,000 |
| Test | 40,000 |
| **Total** | **210,000** |

Each author contributes:

| Split | Rows per author |
|---|---:|
| Fit | 30,000 |
| Validation | 4,000 |
| Test | 8,000 |
| Total | 42,000 |

The split is chronological at whole-work level.

Roles:

- **Fit**: author-specific Adapter training.
- **Validation**: Rich30/LambdaMART development and freezing.
- **Test**: final held-out evaluation only.

Frozen final manifest:

`results/finalmodel_fiveauthor_v1/dataset_preparation_v1/final_fiveauthor_manifest_frozen_v1.jsonl`

SHA256:

`e05b07d5020ebf0ccf090e98b34e4c66db7dab9b565541d39feb2d6f1d1bb20b`

Frozen work split:

`results/finalmodel_fiveauthor_v1/dataset_preparation_v1/work_split_manifest_frozen_v1.csv`

SHA256:

`631f0ca1b769589c2f0294b10a8783aa95da93fe100c71d066df924922150813`

---

## 3. Test population and conditions

### By author

Each author contributes exactly **8,000 Test queries**, for **40,000 total**.

### By target length

`M` is the number of consecutive processed tokens, not necessarily the number of Chinese characters.

| M | N |
|---|---:|
| M1 | 9,600 |
| M2 | 8,800 |
| M3 | 8,800 |
| M4 | 6,960 |
| M5 | 5,840 |
| **Total** | **40,000** |

### Assigned typing mode

| Mode | N |
|---|---:|
| Full | 20,000 |
| Initial | 8,560 |
| Mixed | 11,440 |
| **Total** | **40,000** |

### Effective typing mode

| Mode | N |
|---|---:|
| Full | 20,487 |
| Initial | 9,067 |
| Mixed | 10,446 |
| **Total** | **40,000** |

Effective typing mode is the main typing-condition analysis because it reflects the actual input after deterministic fallback for single-syllable Mixed cases.

Pinyin generation was frozen with:

- `pypinyin 0.55.0`
- `Style.NORMAL`
- `strict=True`
- tone-free lowercase Pinyin

A 50,000-row historical regression check produced zero mismatches.

---

## 4. Frozen neural model and Adapter

Base model:

`aihijo/transformers4ime-pinyingpt-concat`

Revision:

`76dd20dc92d8236a350fb732e99dde6fa15e2263`

Decoding:

- beam size = 16
- Top-K = 10

Final Adapter architecture:

- serial bottleneck residual Adapter after every GPT-2 block;
- 12 Adapter layers;
- hidden size 768;
- bottleneck size 48;
- ReLU;
- reduction factor 16;
- zero-init up projection;
- no Adapter dropout;
- no gate;
- no Adapter layer norm;
- no learned Adapter scale.

Trainable Adapter parameters: **894,528**

Frozen base parameters: **102,408,960**

Training:

- AdamW
- learning rate `5e-4`
- weight decay `0`
- max grad norm `1`
- 1 epoch
- batch size `8`
- seed `20260822`

Final Adapter SHA256 values:

- Re_spectators: `8ca40dacaf0ad1c9e24119f5af9c85325e5ddd372c4e2a3535f4483a53b52242`
- Etinjat: `b26b4f6934b0084a2e2b38d38db7c19ecc7401d6bbcb15035f321afdbdfff8cd`
- Agent Phage: `a27ed86b48226161b9fcba1c6fe19943395129949da906752fa1195d36e87345`
- QBLevi: `e977825641797610153bd9aa3b795157d9a1f38d4d3e447387feb5bfb826cc7e`
- breaddddd: `740cbf84d32a361f8f654c3a9bb8d2486c23307626ad0c3035a054a06247bec1`

---

## 5. Explicit Memory

The final explicit-memory system uses **H5000**.

For each query, memory contains at most the latest **5,000 strictly prior raw interactions** of the same author.

This is a **bounded causal sliding window**, not an unbounded history store.

At Test time, memory may contain:

- Fit history;
- Validation history;
- strictly earlier Test history.

It may never contain:

- the current Test interaction;
- future Test interactions.

Final Test leakage gates:

- `CURRENT_TEST_VISIBLE = 0`
- `FUTURE_TEST_VISIBLE = 0`

39,999 of 40,000 Test queries had some visible strictly prior Test history; the first chronological Test query is the expected exception.

### Exact recovery

Pinyin compatibility allows either:

1. complete canonical Pinyin syllable; or
2. a one-letter initial equal to the first letter of that syllable.

The query and historical interaction must have the same number of Pinyin segments.

Strongest-only context suffix backoff:

1. suffix length 2;
2. suffix length 1;
3. context-free fallback.

Only the strongest non-empty level is used.

Recency weighting:

`exp(-age / 2048)`

### Composition recovery

Composition runs in parallel with Exact recovery.

It uses proper consecutive Pinyin spans and excludes the complete query span.

Each span retrieves up to 5 candidates.

Dynamic-programming beam:

`300`

A valid composition contains at least 2 pieces.

Composition score:

`S_comp = (1/n) * sum_i [ l_i * log(max(p_i, EPS)) ]`

with `EPS = 1e-12`.

`p_composition = exp(clip(S_comp, -50, 0))`

### Candidate fusion

Exact and Composition evidence are merged by candidate text.

Recovery confidence:

`max(p_exact, p_composition)`

The neural model contributes Top-10.

Explicit Memory contributes up to 5 memory-only candidates.

Memory overlap with a neural candidate retains memory evidence and does not consume the Personal5 budget.

Maximum fused pool size: **15**.

---

## 6. Rich30 LambdaMART

The final reranker uses 30 frozen features combining:

- neural candidate evidence;
- Exact and Composition evidence;
- candidate-pool/query characteristics;
- H5000 frequency/history statistics;
- context-sensitive n-gram history evidence.

Final LambdaMART configuration:

- objective = `lambdarank`
- metric = `ndcg`
- max_depth = 5
- num_leaves = 31
- min_data_in_leaf = 500
- learning_rate = 0.05
- lambda_l2 = 0
- rounds = 100
- seed = 1729
- deterministic = true
- force_col_wise = true
- num_threads = 4

Final Adapter + Explicit model SHA256:

`059b5e6c092a00cf1c9510d65ae1a7192cfe441bc7fd56f327bead45f5aa06bd`

Generic + Explicit model SHA256:

`9284e4227eba95441e5c7f0185a6e9b7728f43fac3d677dede453cdbddf750e2`

C1 is not part of the final architecture.

---

## 7. Final five-system held-out Test comparison

Systems:

1. Generic
2. Generic + Frequency
3. Generic + Explicit Memory
4. Adapter
5. Adapter + Explicit Memory

All systems were aligned over the same 40,000 Test row IDs and evaluated with the same metric implementation.

| System | N | Top-1 | Top-3 | Top-5 | Top-10 | MRR | MRR@5 | Pool Hit | Missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Generic | 40,000 | 34.918% | 46.170% | 49.858% | 53.535% | 0.41324 | 0.40821 | 53.535% | 46.465% |
| Generic + Frequency | 40,000 | 36.820% | 48.605% | 52.100% | 55.618% | 0.43461 | 0.42963 | 55.785% | 44.215% |
| Generic + Explicit Memory | 40,000 | 45.115% | 55.310% | 58.518% | 61.683% | 0.50898 | 0.50416 | 62.260% | 37.740% |
| Adapter | 40,000 | 64.398% | 75.095% | 77.820% | 80.405% | 0.70235 | 0.69882 | 80.405% | 19.595% |
| **Adapter + Explicit Memory** | **40,000** | **65.683%** | **76.218%** | **79.013%** | **81.440%** | **0.71492** | **0.71109** | **82.078%** | **17.923%** |

Final held-out result:

- Top-1 = **65.6825%**
- Top-3 = **76.2175%**
- Top-5 = **79.0125%**
- Top-10 = **81.4400%**
- MRR = **0.714915**
- MRR@5 = **0.711089**
- Pool Hit = **82.0775%**
- Missing = **17.9225%**

---

## 8. Pairwise gains

### Generic -> Generic + Frequency

- Top-1: +1.903 pp
- Top-3: +2.435 pp
- Top-5: +2.243 pp
- Top-10: +2.082 pp
- MRR: +0.02137
- Missing: -2.250 pp

### Generic -> Generic + Explicit Memory

- Top-1: +10.197 pp
- Top-3: +9.140 pp
- Top-5: +8.660 pp
- Top-10: +8.147 pp
- MRR: +0.09575
- Missing: -8.725 pp

### Generic -> Adapter

- Top-1: +29.480 pp
- Top-3: +28.925 pp
- Top-5: +27.963 pp
- Top-10: +26.870 pp
- MRR: +0.28912
- Missing: -26.870 pp

### Adapter -> Adapter + Explicit Memory

- Top-1: +1.285 pp
- Top-3: +1.123 pp
- Top-5: +1.192 pp
- Top-10: +1.035 pp
- MRR: +0.01256
- Missing: -1.673 pp

### Generic + Frequency -> Generic + Explicit Memory

- Top-1: +8.295 pp
- Top-3: +6.705 pp
- Top-5: +6.417 pp
- Top-10: +6.065 pp
- MRR: +0.07437
- Missing: -6.475 pp

---

## 9. Candidate recovery

### Generic + Frequency

- neural Top-10 hit: 21,414 / 40,000 = 53.535%
- newly recovered: 900 / 40,000 = 2.250%
- fused pool hit: 22,314 / 40,000 = 55.785%
- gold in pool but outside final Top-10: 67 / 40,000 = 0.168%

### Generic + Explicit Memory

- neural Top-10 hit: 21,414 / 40,000 = 53.535%
- newly recovered: 3,490 / 40,000 = 8.725%
- fused pool hit: 24,904 / 40,000 = 62.260%
- gold in pool but outside final Top-10: 231 / 40,000 = 0.578%

Rich Explicit Memory recovers **2,590 more Test targets** than simple Frequency.

### Adapter + Explicit Memory

- neural Top-10 hit: 32,162 / 40,000 = 80.405%
- newly recovered: 669 / 40,000 = 1.6725%
- final pool hit: 32,831 / 40,000 = 82.0775%
- gold in pool but outside final Top-10: 255 / 40,000 = 0.6375%

Final Top-10 failures:

- total: 7,424
- gold absent from candidate pool: 7,169
- gold in pool but outside Top-10: 255

Approximately **96.6%** of remaining Top-10 failures are therefore candidate-coverage failures rather than post-pool ranking failures.

---

## 10. Results by author

Each author has N = 8,000.

| Author | Generic | Generic+Frequency | Generic+Explicit | Adapter | Adapter+Explicit |
|---|---:|---:|---:|---:|---:|
| Agent Phage | 31.662% | 33.125% | 41.875% | 65.838% | **66.850%** |
| Etinjat | 29.200% | 31.300% | 40.062% | 55.837% | **57.700%** |
| QBLevi | 20.300% | 20.950% | 25.812% | 48.400% | **49.300%** |
| Re_spectators | 47.000% | 49.175% | 58.200% | 72.763% | **74.875%** |
| breaddddd | 46.425% | 49.550% | 59.625% | 79.150% | **79.688%** |

The ordering is stable for all authors:

`Generic < Generic+Frequency < Generic+Explicit < Adapter < Adapter+Explicit`

Adapter + Explicit improves Adapter Top-1 for all five authors.

---

## 11. Results by target length

| M | N | Generic | Generic+Frequency | Generic+Explicit | Adapter | Adapter+Explicit |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 9,600 | 62.479% | 69.552% | 74.750% | 81.490% | **83.198%** |
| M2 | 8,800 | 41.920% | 42.477% | 53.227% | 70.875% | **72.057%** |
| M3 | 8,800 | 27.034% | 27.307% | 37.398% | 60.511% | **61.818%** |
| M4 | 6,960 | 17.902% | 18.032% | 26.365% | 52.543% | **53.621%** |
| M5 | 5,840 | 11.216% | 11.216% | 18.151% | 46.524% | **47.483%** |

Generic -> Adapter Top-1 gains:

- M1: +19.01 pp
- M2: +28.96 pp
- M3: +33.48 pp
- M4: +34.64 pp
- M5: +35.31 pp

Simple Frequency becomes progressively less useful as target length increases and produces no Top-1 improvement at M5.

Explicit Memory remains beneficial at every M.

---

## 12. Results by effective typing condition

| Effective condition | N | Generic | Generic+Frequency | Generic+Explicit | Adapter | Adapter+Explicit |
|---|---:|---:|---:|---:|---:|---:|
| Full | 20,487 | 51.960% | 54.166% | 63.923% | 84.205% | **85.381%** |
| Initial | 9,067 | 14.261% | 16.334% | 20.558% | 35.690% | **36.859%** |
| Mixed | 10,446 | 19.424% | 20.582% | 29.542% | 50.469% | **52.068%** |

Difficulty order:

`Full > Mixed > Initial`

Adapter + Explicit improves Adapter in all three effective input conditions.

Adapter -> Adapter + Explicit Top-1 gains:

- Full: +1.176 pp
- Initial: +1.169 pp
- Mixed: +1.599 pp

---

## 13. Main findings

### Adapter personalisation is the dominant performance source

Generic Top-1: 34.918%

Adapter Top-1: 64.398%

Gain: **+29.480 pp**

This improvement occurs for every author, every target length, and every typing condition.

### Simple frequency is useful but insufficient

Generic + Frequency improves Top-1 by only **+1.903 pp** and recovers only **900** additional Test targets.

Its benefit collapses as target length increases and disappears at M5.

### Rich Explicit Memory provides substantially more useful information than raw frequency

Generic + Explicit improves Top-1 by **+10.197 pp** and recovers **3,490** additional Test targets.

It recovers **2,590 more queries** than Generic + Frequency.

### Adapter and Explicit Memory are complementary

Adding Explicit Memory to Adapter improves:

- Top-1: +1.285 pp
- Top-10: +1.035 pp
- MRR: +0.01256
- Missing: -1.673 pp

The gain is positive for every author, every M, and every effective typing condition.

A useful interpretation is:

- **Adapter** provides broad author-specific generalisation.
- **Explicit Memory** provides targeted historically grounded candidate recovery and evidence.

### Personalisation matters most under difficult prediction conditions

At M5:

- Generic: 11.216%
- Generic + Frequency: 11.216%
- Generic + Explicit: 18.151%
- Adapter: 46.524%
- Adapter + Explicit: 47.483%

Under effective Initial:

- Generic: 14.261%
- Generic + Explicit: 20.558%
- Adapter: 35.690%
- Adapter + Explicit: 36.859%

Under effective Mixed:

- Generic: 19.424%
- Generic + Explicit: 29.542%
- Adapter: 50.469%
- Adapter + Explicit: 52.068%

### Remaining errors are dominated by candidate coverage

Final pool hit: 82.0775%

Final Top-10: 81.4400%

Only 255 queries have the gold inside the pool but outside Top-10.

Approximately 96.6% of final Top-10 failures occur because the gold target never enters the candidate pool.

Further improvement should therefore prioritise candidate generation and recovery rather than increasing final-reranker complexity.

---

## 14. Final architecture

```text
Pinyin input + context
        |
        v
Author-specific Adapter PinyinGPT
        |
        | neural Top-10
        v
Candidate fusion
        ^
        |
H5000 bounded causal Explicit Memory
    |               |
 Exact          Composition
    \               /
     \             /
      memory evidence
      + up to 5 memory-only candidates
        |
        v
maximum 15-candidate fused pool
        |
        v
Rich30 features
        |
        v
Validation-frozen LambdaMART
        |
        v
Final ranked candidates
```

No C1 component is used.

---

## 15. Experimental integrity

Final Test:

- 40,000 held-out queries
- 5 authors
- 8,000 queries per author
- 30 Rich30 features

Final-system Test gates:

- LambdaMART training on Test = 0
- neural parameter updates on Test = 0
- current Test visible = 0
- future Test visible = 0

Final Test job:

- Job: `640032`
- State: `COMPLETED`
- ExitCode: `0:0`
- Runtime: `09:10:41`

Final Test surface SHA256:

`09fd02b942cee646de03bec5b13202efe432f5087c40fb2467dbe982bbf8de6f`

Final Test H5000 SHA256:

`aeee83143d9c7ef7cf5b1cc9cfff1622d6ff522427872d5b18df1f50ccdca78f`

Unified comparison gate:

`FIVE_SYSTEM_TEST_COMPARISON_GATE=PASS`

---

## 16. Canonical final comparison artifacts

Unified comparison JSON:

`results/finalmodel_fiveauthor_v1/five_system_test_comparison_v1/five_system_test_comparison_v1.json`

Overall CSV:

`results/finalmodel_fiveauthor_v1/five_system_test_comparison_v1/overall_v1.csv`

Condition CSV:

`results/finalmodel_fiveauthor_v1/five_system_test_comparison_v1/conditions_v1.csv`

Aligned row-level rank surface:

`results/finalmodel_fiveauthor_v1/five_system_test_comparison_v1/row_level_ranks_v1.csv`

These artifacts are the canonical source for the final five-system Test comparison.

---

## 17. Final conclusion

Final Top-1 progression:

```text
Generic
34.918%

    -> Generic + Frequency
36.820%

    -> Generic + Explicit Memory
45.115%

    -> Adapter
64.398%

    -> Adapter + Explicit Memory
65.683%
```

The experiments show that:

1. simple recent-user frequency provides a small measurable benefit;
2. richer explicit memory substantially outperforms simple frequency;
3. Adapter-based neural personalisation provides the largest overall gain;
4. Adapter and Explicit Memory provide complementary benefits;
5. the combined system is strongest overall and across all major tested conditions;
6. remaining failures are dominated by candidate coverage rather than final reranking.

Final held-out Test result:

- **Top-1: 65.6825%**
- **Top-3: 76.2175%**
- **Top-5: 79.0125%**
- **Top-10: 81.4400%**
- **MRR: 0.714915**
- **MRR@5: 0.711089**
- **Pool Hit: 82.0775%**
- **Missing: 17.9225%**

---

## 18. Research freeze declaration

This experimental cycle is now **CLOSED**.

The following are frozen:

- final five-author population;
- dataset generator and split;
- Fit / Validation / Test roles;
- author-specific Adapter architecture;
- H5000 causal memory semantics;
- Exact recovery;
- Composition recovery;
- Personal5 candidate expansion;
- Rich30 features;
- LambdaMART configuration;
- final five-system baseline definitions;
- final Test metric definitions;
- final held-out Test results.

The Test results in this document must not be used for additional architecture selection or hyperparameter tuning within this experimental version.

Any future change to model architecture, Adapter configuration, H5000 semantics, memory scoring, candidate budget, Rich30 features, reranker, training population, dataset generator, or evaluation protocol must be treated as a **new experimental version**.

**Research status: FINAL / FROZEN / TEST COMPLETE.**
