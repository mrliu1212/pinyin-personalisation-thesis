# Hybrid Personalisation and Cross-Regime Transfer Report

**Date:** 2026-08-26
**Research status:** Train-Val / development-safe only
**Test status:** CLOSED
**Primary personalised author for Adapter experiments:** Agent Phage

---

## 1. Purpose

This phase investigates whether model-level personalisation and external-memory personalisation provide complementary information, and whether they can be combined into a stronger Pinyin-to-character ranking system.

The systems considered are:

1. a frozen PinyinGPT2-Concat base model with a user-specific serial Adapter;
2. External Memory / Learned Fusion rerankers;
3. zero-shot Hybrid combinations of Adapter and External Memory;
4. cross-regime transfer between Full-Pinyin and Initial-Pinyin fusion policies.

The main methodological question is not only whether one expert has higher standalone Top-1 accuracy, but whether weaker experts can safely rescue errors made by the strongest expert without introducing excessive harms.

No Test data were used in the experiments documented here.

---

## 2. Model-level Adapter

The formal Agent Phage Adapter is a serial bottleneck Adapter installed into the frozen PinyinGPT2-Concat model.

Configuration:

- layers: 12
- hidden size: 768
- bottleneck size: 48
- frozen base parameters: 102,408,960
- Adapter parameters: 894,528
- trainable parameters: 894,528

The formal checkpoint is:

`local_artifacts/adapters/static/agent_full55926.safetensors`

The same formal Adapter checkpoint can be evaluated under Full, Initial, and Mixed Pinyin manifestations. Initial here means first-letter abbreviated Pinyin, not an early stage of user adaptation.

---

## 3. Full-Pinyin Hybrid: naive Borda control

A first diagnostic combined the Adapter and frozen Full LambdaMART rankings using equal-weight Borda fusion.

On a 256-query Agent smoke subset:

| System | Top-1 | Top-3 | Top-5 | MRR |
|---|---:|---:|---:|---:|
| Adapter | 0.894531 | 0.941406 | 0.968750 | 0.923670 |
| Frozen Full LambdaMART | 0.867188 | 0.960938 | 0.976563 | 0.916769 |
| Equal Borda | 0.878906 | 0.968750 | 0.984375 | 0.926693 |

Transitions:

- LambdaMART -> Borda: rescue 9, harm 6, net +3
- Adapter -> Borda: rescue 8, harm 12, net -4

This established that the two systems contain complementary information, but that unconditional equal-weight ranking fusion can damage the stronger Adapter.

This experiment is retained only as a naive control.

---

## 4. Full-Pinyin Proper Hybrid H0

A stricter zero-shot Hybrid protocol was then constructed.

Protocol:

1. keep the frozen External Memory Stage-1 candidate surface fixed;
2. rescore that same surface with the formal Agent Adapter;
3. replace neural-base-dependent features using Adapter scores;
4. keep the frozen memory evidence and frozen LambdaMART policy unchanged;
5. perform no training or tuning;
6. keep Test closed.

The frozen LambdaMART reproduction gate was required to match the original ranking before Hybrid evaluation.

### Agent Full Train-Val

Population: `n = 13,741`

| System | Top-1 | Top-3 | Top-5 | MRR | Missing@10 |
|---|---:|---:|---:|---:|---:|
| Adapter on fixed EM surface | **0.933047** | 0.977367 | 0.980642 | **0.954850** | 0.017757 |
| Frozen Full LambdaMART | 0.890401 | 0.963976 | 0.974820 | 0.927977 | 0.017757 |
| Proper Hybrid H0 | 0.924751 | **0.978168** | **0.981006** | 0.950963 | 0.017757 |

Transition from Frozen LambdaMART to Hybrid:

- rescue: 610
- harm: 138
- net: +472

Transition from Adapter-on-fixed-surface to Hybrid:

- rescue: 222
- harm: 336
- net: -114

### Interpretation

External Memory clearly contains useful information that the Adapter does not: it rescues 222 Adapter Top-1 errors. However, unconditional fusion overrides too many correct Adapter decisions, producing 336 harms.

Therefore the remaining Hybrid problem is primarily an **arbitration problem**: determine when External Memory evidence is strong enough to override the Adapter, rather than always blending the two rankings.

The Hybrid improves Top-3 and Top-5 over the Adapter-on-fixed-surface baseline, but does not improve Adapter Top-1.

---

## 5. Initial-Pinyin Hybrid diagnostic

Initial Pinyin uses first-letter abbreviation, for example:

`zhong guo ren -> z g r`

The formal Adapter checkpoint was evaluated on the fixed Initial External Memory candidate surface.

### 256-query Agent smoke

| System | Top-1 | Top-3 | Top-5 | MRR | Missing@10 |
|---|---:|---:|---:|---:|---:|
| Adapter on fixed Initial surface | **0.574219** | 0.726563 | 0.765625 | **0.652429** | 0.210938 |
| Frozen ILT-004 D | 0.496094 | 0.671875 | 0.718750 | 0.591558 | 0.210938 |
| Equal Borda | 0.535156 | 0.710938 | 0.757813 | 0.628974 | 0.210938 |

Complementarity:

- both correct: 101
- Adapter only: 46
- ILT-004 D only: 26
- neither: 83
- oracle either Top-1: **0.675781**

Again, the weaker memory expert contains many useful corrections, but naive fusion reduces Adapter Top-1.

This remains a smoke diagnostic. A full 13,741-query Agent Initial run should be completed before making final quantitative claims.

---

## 6. Full and Initial External Memory relationship

Full and Initial use the same underlying causal External Memory framework, but their final learned ranking policies are not identical.

The original Full LambdaMART uses a 25-feature representation.

The Initial-specialised ILT-004 D (`generic_task_shape`) uses:

- 27 Initial base features;
- 14 compact interaction features;
- 16 Task-specific retrieval-shape features;

for a total of **57 features**.

The additional representation is designed to handle the much higher ambiguity of Initial-Pinyin input.

---

## 7. Full-policy -> Initial zero-shot transfer

A policy-level cross-regime transfer control was performed.

This is **not** an end-to-end transfer of the entire Full pipeline. Instead, the Initial candidate/evidence surface is kept fixed, the frozen Full LambdaMART policy is reconstructed over the compatible Initial evidence representation, and no retraining or tuning is performed.

### Overall Train-Val

| System | Top-1 | Top-3 | Top-5 | MRR | Missing@10 |
|---|---:|---:|---:|---:|---:|
| Full policy -> Initial | 0.446072 | 0.632322 | 0.696420 | 0.552090 | 0.243172 |

### Agent

| System | Top-1 | Top-3 | Top-5 | MRR | Missing@10 |
|---|---:|---:|---:|---:|---:|
| Full policy -> Initial | 0.489338 | 0.695801 | 0.766029 | 0.605995 | 0.171312 |
| Initial-native ILT-004 D | **0.523979** | **0.709264** | **0.772360** | **0.628806** | 0.171312 |

Agent Top-1 difference: **ILT-004 D +3.464 percentage points**.

The identical Missing@10 demonstrates that the difference is ranking quality, not candidate coverage.

---

## 8. ILT-004 D -> Full zero-shot transfer

The reverse transfer was then tested. This required a stricter feature-semantic reconstruction because Full and Initial contain several similarly named features with different definitions.

The Full candidate surface was fixed, but all ILT-004 D inputs were rebuilt using the exact Initial semantics.

Important differences handled explicitly include:

- Initial gap features use `max - candidate`;
- the original Full features use the opposite sign;
- Initial frozen linear score uses `base + 4 * NGram + 6 * BGE`;
- Full originally used `base + 6 * NGram + 6 * BGE`;
- Initial personal-recovery Stage-1 weights differ from Full;
- Initial uses a dedicated `frequency_support`;
- Initial includes `query_ambiguous` and `frozen_rank`.

### Personal-frequency reconstruction audit

Full Train-Val contains:

- 3,556 queries with personal-recovery candidates;
- 6,736 personal-recovery candidates.

Audit:

- missing Stage-1 rows: 0
- missing personal_k5: 0
- missing choice entries: 0
- candidate not in personal_k5: 0
- non-integral reconstructed counts: 0
- bad denominators: 0
- maximum count reconstruction floating-point error: `3.552713678800501e-15`

### Zero-candidate queries

The Full Train-Val query population contains 34,416 queries but only 34,414 non-empty LightGBM groups.

Two legitimate zero-candidate queries occur at query indices 19,143 and 21,126. They remain in the evaluation population and count as Missing@10. They are not silently removed.

Total Full candidate rows: `333,099`.

### Final transferred matrix

- Initial-semantic base matrix: `333099 x 27`
- compact interactions: 14
- Full Task-retrieval shape: 16
- final ILT-004 D input: `333099 x 57`
- frozen ILT-004 D model feature count: 57

### Overall Full Train-Val

| System | Top-1 | Top-3 | Top-5 | MRR | Missing@10 |
|---|---:|---:|---:|---:|---:|
| Native Full LambdaMART | 0.827784 | 0.911989 | 0.930614 | 0.873043 | 0.051982 |
| **ILT-004 D -> Full** | **0.830137** | **0.912018** | **0.930962** | **0.874240** | 0.051982 |

### Agent Full Train-Val

| System | Top-1 | Top-3 | Top-5 | MRR | Missing@10 |
|---|---:|---:|---:|---:|---:|
| Native Full LambdaMART | 0.890401 | 0.963976 | 0.974820 | 0.927977 | 0.017757 |
| **ILT-004 D -> Full** | **0.893821** | 0.963976 | 0.974820 | **0.929814** | 0.017757 |

Agent transition from native Full LambdaMART:

- rescue: 110
- harm: 63
- net: **+47**
- both correct: 12,172
- Full-only correct: 63
- ILT-only correct: 110
- neither correct: 1,396
- pairwise oracle Top-1: **0.898406**

### Interpretation

Cross-regime transfer is asymmetric. The Full-trained ranking policy loses substantially when transferred to Initial, whereas the richer Initial-specialised ILT-004 D policy transfers successfully back to Full and slightly improves the native 25-feature Full LambdaMART baseline.

This result must **not** be described as ILT-004 D outperforming the final Full personalisation system. The comparison above is against the native 25-feature Full LambdaMART. The stronger Full Learned Fusion LF-007/LF-008 line remains a separate comparison.

---

## 9. Current Full Learned Fusion context

The later Full Learned Fusion line reaches approximately:

- LF-007 Train-Val micro Top-1: 0.828975
- LF-007 MRR: 0.873645

LF-008 is a cheaper gated policy:

- Task invocation rate: approximately 52.43%
- projected mean latency: approximately 15.51 ms/query

These systems must be included when selecting the final External Memory expert.

---

## 10. Main scientific conclusion so far

Across Full and Initial experiments, the same pattern repeatedly appears:

1. the Adapter is usually the stronger standalone personalised expert;
2. External Memory contains complementary corrections;
3. naive Borda / unconditional fusion can harm Adapter Top-1;
4. pairwise oracle performance shows substantial remaining headroom;
5. the main unsolved problem is safe expert arbitration.

Therefore the leading final architecture is not a simple average of ranking scores. The Adapter should be treated as a strong default expert, while memory or Multi-level experts should override it only when runtime evidence indicates a sufficiently reliable correction.

---

## 11. Multi integration decision

Before selecting the final combined model, the Multi system should be evaluated under the same development-safe framework.

Required Multi diagnostics include:

- Agent Full Train-Val Top-1 / Top-3 / Top-5 / MRR / Missing@10;
- Agent Initial Train-Val metrics where applicable;
- candidate-surface provenance;
- training split and any Train-Val tuning;
- transition against Adapter: rescue / harm / net;
- transition against the strongest memory expert;
- pairwise oracle performance;
- latency / runtime cost.

A Multi model does not need to beat Adapter standalone to be useful. If it provides orthogonal rescues, it can still be valuable as an expert in a gated final system.

---

## 12. Data-use constraints

All model-selection work documented here obeys the following rule: **Test remains closed.**

Current experiments use Train-Fit / Train-Val or Train-Fit internal splits.

Any final arbitration gate must be selected using Train-Fit or an internal Train-Fit gate/validation split and then evaluated once on Train-Val.

Thresholds or policies must not be selected by sweeping Train-Val and reporting the best Train-Val value as if it were held-out evidence.

---

## 13. Reporting cautions

1. `Adapter on fixed EM surface` is not the same quantity as standalone Adapter generation accuracy.
2. Equal Missing@10 is important when attributing improvements to ranking rather than candidate coverage.
3. `Full -> Initial` and `ILT -> Full` experiments are policy-level cross-regime transfers on the target-regime candidate/evidence surface.
4. They are not full end-to-end transfers of candidate generation pipelines.
5. The 256-query Initial Hybrid experiment is a smoke diagnostic, not a final Train-Val result.
6. The serialized Python Hybrid evaluator runtime must not be presented as an optimized production IME latency measurement.
7. ILT-004 D -> Full currently beats the native 25-feature Full LambdaMART, not necessarily LF-007/LF-008.
8. Test must remain closed until the final protocol and model-selection policy have been frozen.

---

## 14. Next decision point

The immediate next research step is to inspect the existing Multi personalisation system under the same protocol and determine whether it adds orthogonal rescue signal.

After that comparison, the final comprehensive model can be selected from candidate architectures such as Adapter + one strongest memory expert, Adapter + Multi, Adapter + memory + Multi, a shared Adapter-aware arbitration gate, or regime-aware Full / Initial gating if the two input regimes require different decision boundaries.

The choice should be based on held-out Train-Val performance, rescue/harm behaviour, candidate coverage, and runtime cost rather than standalone Top-1 alone.
