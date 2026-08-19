# EM-2 External Memory -Final Dev-Stage Report and Closure

**Project:** Transparent and User-Controllable Personalisation for Chinese Pinyin Input
**Stage:** EM-2 -Frozen PinyinGPT Hidden-State External Memory
**Date:** 2026-08-19
**Status:** **CLOSED / FROZEN AT DEV STAGE**
**Next stage:** EM-3 -IME-specific task-trained historical relevance model
**New EM-2 Test opened:** **No**

---

## 1. Closure decision

EM-2 is closed after the Dev-stage investigation.

No further EM-2 representation sweep, retrieval tuning, M1/M2 tuning, fixed fusion tuning, adaptive-gate feature engineering, or EM-2 Test evaluation will be performed.

This is a deliberate stage boundary. EM-2 is treated as an exploratory but methodologically controlled investigation that established where a task-native hidden-state memory helps and where the remaining bottleneck lies.

The central conclusion is:

> Frozen PinyinGPT hidden states are better personal-history retrieval keys than the generic BGE representations tested here, but the stronger retrieval signal does not materially improve the existing M1/M2 end-to-end candidate-decision rules. Fixed and transparent adaptive Frequency/Context fusion also failed to produce a meaningful Overall gain. The remaining problem is therefore not only retrieval; it is how historical evidence is converted into candidate-level decisions.

EM-3 will address that next problem with a task-specific learned historical relevance model rather than further modifying EM-2.

---

# 2. Scientific scope

All new EM-2 development reported here used:

- condition: **Full+Short**
- history budget: **H5000**
- authors:
  - `Etinjat`
  - `Re_spectators`
  - `breaddddd`
- Dev tune population: **5,608 queries**
- same-user / same-author history only
- strictly-prior history
- H5000 applied **before** exact segmented-Pinyin filtering
- exact same segmented Pinyin for legal memory
- frozen Generic PinyinGPT candidate surface
- no current Gold in retrieval or scoring features
- no future history
- no Test tuning

Important limitation:

> The new EM-2 methods were closed without opening their Test results. Therefore the numerical EM-2 conclusions below are Dev-stage conclusions and must not be presented as final held-out generalisation results.

Previously frozen Test results for older F/M1/M2 methods remain separate and are not replaced by these Dev comparisons.

---

# 3. Why EM-2 was needed

Earlier context-personalisation experiments showed that a generic BGE semantic retriever could often find useful same-Pinyin personal history, but better retrieval did not consistently become better final candidate ranking.

A local 64-character BGE context window improved retrieval discrimination, yet the final M1 result remained almost unchanged.

This created a new question:

> Is the generic semantic representation itself the wrong representation for an IME memory lookup?

PinyinGPT already computes an internal state immediately before predicting the first Chinese target character. EM-2 therefore tested whether that task-native state could serve as a better external-memory key.

The research logic was intentionally staged:

```text
EM-2A
Is the hidden state extracted from the correct prediction position?

         ->
EM-2B
Can all required Dev representations be cached under the frozen backend semantics?

         ->
EM-2C
Does hidden-state kNN retrieve the correct personal history?

         ->
EM-2D
Is hidden retrieval stronger than BGE on the exact same retrieval surface?

         ->
EM-2E1
If only BGE  -> Hidden is changed inside M1, does final ranking improve?

         ->
EM-2E2
If only M2 Stage-1 BGE  -> Hidden is changed, does the Cross-Encoder benefit?

         ->
Fixed G+F+C
Are Frequency and Hidden Context complementary under fixed weights?

         ->
Adaptive G+F+C
Can transparent prediction-visible confidence choose F/C weights per query?
```

---

# 4. EM-2A -hidden-state engineering gate

## Purpose

Before inspecting retrieval quality, the project verified that the representation was taken from the exact PinyinGPT prediction state relevant to the first target character.

Without this gate, a later result could be confounded by selecting the wrong layer, pooling rule, or token position.

## Frozen extraction

The frozen PinyinGPT2-Concat prompt is:

```text
[CLS] + preceding context + [SEP] + Pinyin tokens + [SEP]
```

The first target character is predicted from the final prompt position.

EM-2 therefore froze:

```text
final Transformer layer
+
final prompt [SEP] hidden state
```

Hidden size:

```text
768
```

## Engineering result

Nine Dev-only samples:

```text
9 / 9 PASS
hidden size = 768

max hidden -> LM-head logits abs diff
= 2.67028808594e-05

max allowed distribution diff
= 1.14440917969e-05

max first-step direct vs teacher abs diff
= 3.81469726562e-06

max cached beam vs fixed score diff
= 9.17911529541e-06

best allowed next-token agreement
= 9 / 9
```

Gold was not used. Retrieval performance was not inspected.

## Decision

The final-layer final-[SEP] state was frozen before retrieval results.

No later EM-2 result was allowed to justify changing the hidden layer, extraction position, or pooling rule.

---

# 5. EM-2B -legal memory workload and hidden-state cache

## Why measure workload first

The legal personal-memory surface was profiled before running expensive representation extraction.

Dev tune workload:

```text
History manifest rows        248,082
Dev manifest rows             32,212
Three-author tune queries      5,608
Legal query-history edges    122,067
```

Per author:

```text
Etinjat          3,047
Re_spectators      199
breaddddd        2,362
```

Visible exact-Pinyin H5000 histories per query:

```text
mean      21.767
median     2
max      435
```

Availability:

```text
history available   3,625 / 5,608 = 64.64%
no history          1,983 / 5,608 = 35.36%
```

Required unique representation rows:

```text
11,475
```

The workload is strongly skewed: the mean is much larger than the median.

## Hidden cache result

```text
Cached rows: 11,475
Hidden size: 768
Context-truncated rows: 0

Prompt tokens:
min  = 6
mean = 436.26
max  = 517
```

Frozen hidden cache SHA256:

```text
9a80a3314c184ccf3f0540916203c651474fad162dc3dab1fc97f7451f441df1
```

This also confirmed that no extra model-limit truncation was introduced for the required EM-2 Dev representations.

---

# 6. EM-2C -hidden-state kNN retrieval

## Question

Can the frozen PinyinGPT prediction-state representation retrieve the correct historical target when that target exists in the legal personal history?

## Evaluation populations

```text
all queries                5,608
history available          3,625
Gold history available     3,213
Ambiguous + Gold history   1,609
Conflict + Gold history      359
```

Retrieval recall is conditioned on Gold existing in legal history. Gold is used only after retrieval for evaluation.

## Result

Overall Gold-History:

```text
Micro R@1  = 0.869281
Micro R@5  = 0.971366
Micro R@10 = 0.989107

Macro R@1  = 0.890920
Macro R@5  = 0.977833
Macro R@10 = 0.992256
```

Ambiguous Gold-History:

```text
Micro R@1  = 0.738968
Micro R@5  = 0.942822
Micro R@10 = 0.978247

Macro R@1  = 0.768247
Macro R@5  = 0.957042
Macro R@10 = 0.986311
```

Conflict Gold-History:

```text
Micro R@1  = 0.431755
Micro R@5  = 0.768802
Micro R@10 = 0.902507

Macro R@1  = 0.424646
Macro R@5  = 0.808620
Macro R@10 = 0.931717
```

Primary retrieval diagnostic:

```text
Macro-author Ambiguous R@1 = 0.768247
```

## Interpretation

The PinyinGPT prediction state contains useful personal-history retrieval signal.

This stage did not yet establish superiority over BGE or final ranking improvement.

---

# 7. EM-2D -same-surface representation comparison

The retrieval comparison held the query/history surface, history semantics, cosine ranking, Gold-history conditioning, Ambiguous definition, and Conflict definition fixed.

Only the representation changed.

| Representation | Overall Macro R@1 | Ambiguous Macro R@1 | Conflict Macro R@1 |
|---|---:|---:|---:|
| BGE Full | 88.41% | 73.03% | 34.53% |
| BGE ctx64 | 88.36% | 75.09% | 41.05% |
| **PinyinGPT hidden** | **89.09%** | **76.82%** | **42.46%** |

Primary Ambiguous comparison:

```text
Hidden vs BGE ctx64
+1.74 percentage points

Hidden vs BGE Full
+3.80 percentage points
```

Conflict depth:

| Representation | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| BGE Full | 34.53% | 73.86% | 84.39% |
| BGE ctx64 | 41.05% | 70.28% | 80.69% |
| PinyinGPT hidden | **42.46%** | **80.86%** | **93.17%** |

## Decision

Hidden-state retrieval was considered a retrieval-stage success.

The next question became whether the stronger retrieved evidence could improve end-to-end candidate ranking.

---

# 8. EM-2E1 -Hidden-M1

## Controlled change

Original M1:

```text
BGE representation
-> cosine retrieval
-> Top-N histories
-> positive similarity target support
-> z(Generic) + lambda * support
```

Hidden-M1:

```text
PinyinGPT hidden representation
-> same cosine retrieval
-> same Top-N logic
-> same positive similarity target support
-> same final ranking rule
```

The intended methodological change was only:

```text
BGE representation
->
Frozen PinyinGPT hidden representation
```

## Dev selection

Grid:

```text
Top-N = {1, 3, 5, 10, 20}
lambda = {0, 0.25, 0.5, 1, 2, 4}
```

Selected:

```text
Top-N = 3
lambda_hidden = 4
```

Because lambda 4 was the original upper boundary, a pre-registered one-time lambda 8 check was performed:

```text
Top-N=3, lambda=4 -> 0.768748
Top-N=3, lambda=8 -> 0.764567
```

No further boundary expansion was allowed.

## Four-way same-surface Dev result

| Method | Overall | History | Ambiguous | Conflict |
|---|---:|---:|---:|---:|
| G | 72.2948% | 75.9399% | 68.3605% | **46.7924%** |
| F | 76.5240% | 82.3281% | 75.4831% | 19.7192% |
| Original M1 | **76.8888%** | 82.8351% | **76.9454%** | 29.7163% |
| Hidden-M1 | 76.8748% | **82.8688%** | 76.6692% | **29.9645%** |

Hidden-M1 vs Original M1:

```text
Overall          -0.0140 pp
History          +0.0337 pp
Ambiguous        -0.2762 pp
Conflict         +0.2482 pp
```

Original M1 -> Hidden-M1 micro transitions:

```text
Overall:
rescue = 65
harm   = 54
net    = +11

Conflict:
rescue = 37
harm   = 16
net    = +21
```

## Conclusion

The stronger hidden-state retrieval signal did **not** materially improve M1 end-to-end Macro-author Top1.

This was the critical diagnostic:

```text
retrieval representation improved
but
candidate-decision result stayed almost unchanged
```

Therefore retrieval quality alone was no longer the main bottleneck.

---

# 9. EM-2E2 -Hidden-M2

## Why test M2

M1 uses a simple target-support aggregation. M2 adds a generic pretrained Cross-Encoder after Stage-1 retrieval.

The question was:

> Can the existing Cross-Encoder make better use of hidden-state Stage-1 retrieval?

Only Stage-1 retrieval representation was changed.

## Original M2

Frozen Original M2:

```text
BGE Full retrieval
K = 20
BAAI/bge-reranker-base
lambda_m2 = 4
```

Same-surface Dev:

```text
Overall            0.766869
History Available  0.825123
Ambiguous          0.763249
Conflict           0.258543
```

## Hidden-M2

Grid:

```text
K = {10, 20}
lambda_m2 = {0.5, 1, 2, 4}
```

Selected:

```text
K = 10
lambda_m2 = 4
```

Result:

```text
Overall            0.768372
History Available  0.827441
Ambiguous          0.768332
Conflict           0.295521
```

Strict same-parameter representation control:

```text
Original M2
BGE, K20, lambda4
= 0.766869

Hidden-M2
Hidden, K20, lambda4
= 0.767776

delta
= +0.0907 pp
```

## Conclusion

Hidden retrieval helped M2 slightly, particularly on Conflict, but Hidden-M2 still did not outperform the M1 family.

The generic Cross-Encoder route was therefore stopped rather than repeatedly retuned.

This result directly motivates EM-3, where the historical relevance model will be trained for the IME task rather than remaining generic.

---

# 10. Fixed G+F+C fusion

## Motivation

Frequency and Hidden Context extract different evidence from the same legal personal history.

```text
F:
long-term target preference for this Pinyin

C:
situation-specific support from similar historical contexts
```

A fixed fusion tested whether the two signals were complementary.

## Definition

```text
score(c)
=
G(c)
+ lambda_F * F(c)
+ lambda_C * C_hidden(c)
```

Hidden Top-N was frozen at 3.

Grid:

```text
lambda_F, lambda_C
in {0, 0.25, 0.5, 1, 2, 4, 8}
```

True-fusion selection required both weights to be positive.

## Result

Selected:

```text
lambda_F = 0.5
lambda_C = 4
```

| Method | Overall | History | Ambiguous | Conflict |
|---|---:|---:|---:|---:|
| G | 72.2948% | 75.9399% | 68.3605% | 46.7924% |
| F | 76.5240% | 82.3281% | 75.4831% | 19.7192% |
| Hidden-M1 | 76.8748% | 82.8688% | 76.6692% | 29.9645% |
| GFC 4/4 | 76.7227% | 82.6492% | 76.0751% | 22.3070% |
| **GFC selected** | **76.8825%** | **82.8857%** | **76.6689%** | **28.9951%** |

Selected GFC vs Hidden-M1:

```text
Overall     +0.0077 pp
History     +0.0169 pp
Ambiguous   -0.0003 pp
Conflict    -0.9694 pp
```

Hidden-M1 -> selected GFC:

```text
Overall:
rescue = 10
harm   = 9
net    = +1
```

## Conclusion

A single global pair of Frequency/Context weights did not provide meaningful extra value.

The optimum heavily down-weighted Frequency and almost collapsed back to Hidden-M1.

---

# 11. Transparent Adaptive G+F+C

## Registered motivation

Before the Fixed Fusion result was observed, a transparent adaptive-fusion hypothesis was registered:

> Different queries may need different Frequency/Context weights because the quantity and confidence of available evidence differ.

No Gold-derived input was allowed.

## Primary count-aware rule

Visible history quantity:

```text
H(q) = n / (n + 5)
```

Frequency confidence:

```text
CF(q)
=
H(q) * frequency_margin
```

Context confidence:

```text
CC(q)
=
H(q)
* clamp(top1_hidden_cosine, 0, 1)
* retrieved_target_agreement
```

Frequency and Context shared one total personalisation budget.

Global scale grid:

```text
L = {1, 2, 4, 8, 16}
```

No expansion beyond 16 was allowed.

A no-count control was also pre-registered.

## Result

Selected count-aware:

```text
L = 16
```

Selected no-count control:

```text
L = 4
```

| Method | Overall | History | Ambiguous | Conflict |
|---|---:|---:|---:|---:|
| G | 72.2948% | 75.9399% | 68.3605% | **46.7924%** |
| F | 76.5240% | 82.3281% | 75.4831% | 19.7192% |
| Hidden-M1 | 76.8748% | 82.8688% | 76.6692% | 29.9645% |
| Fixed-GFC | **76.8825%** | **82.8857%** | 76.6689% | 28.9951% |
| Adaptive count-aware | 76.5894% | 82.5190% | 75.3736% | 20.2374% |
| Adaptive NoCount | 76.8591% | 82.7811% | **76.8895%** | **31.1041%** |

Count-aware vs Hidden-M1:

```text
Overall:
rescue = 41
harm   = 49
net    = -8

Conflict:
rescue = 10
harm   = 35
net    = -25
```

Gate statistics for the count-aware selected method:

```text
mean lambda_F      = 2.510153
mean lambda_C      = 2.729159
median lambda_F    = 1.575817
median lambda_C    = 1.843168

lambda_F > lambda_C : 2,355
lambda_C > lambda_F : 1,270
lambda_F = lambda_C : 1,983

zero personalisation: 1,983
```

The 1,983 zero-personalisation rows correspond to the queries without legal same-Pinyin history.

## Interpretation

The specific count-aware rule was worse than Hidden-M1 and Fixed GFC.

The no-count confidence control was close to Hidden-M1 Overall and slightly stronger on Ambiguous and Conflict.

Important scientific boundary:

> This result does **not** prove that history quantity can never help dynamic F/C weighting.

The tested count-aware formula used history quantity as a shared shrinkage term for both F and C. It mostly reduced total personalisation strength rather than changing the relative F:C preference.

However, because Adaptive Fusion was pre-registered as the final transparent EM-2 experiment, no new count function, gating feature, threshold, or adaptive formula was introduced after seeing this result.

That prevents result-driven tuning.

---

# 12. Consolidated Dev result

All new EM-2 development methods below use the same three-author, Full+Short, H5000, 5,608-query Dev tune surface.

| Method | Overall | History | Ambiguous | Conflict |
|---|---:|---:|---:|---:|
| G | 72.2948% | 75.9399% | 68.3605% | **46.7924%** |
| F | 76.5240% | 82.3281% | 75.4831% | 19.7192% |
| Original M1 | **76.8888%** | 82.8351% | **76.9454%** | 29.7163% |
| Hidden-M1 | 76.8748% | 82.8688% | 76.6692% | 29.9645% |
| Original M2 | 76.6869% | 82.5123% | 76.3249% | 25.8543% |
| Hidden-M2 | 76.8372% | 82.7441% | 76.8332% | 29.5521% |
| Fixed GFC | 76.8825% | **82.8857%** | 76.6689% | 28.9951% |
| Adaptive count-aware | 76.5894% | 82.5190% | 75.3736% | 20.2374% |
| Adaptive NoCount | 76.8591% | 82.7811% | 76.8895% | **31.1041%** |

Do not interpret small Dev differences as final held-out method ranking.

---

# 13. Final EM-2 conclusions

## Conclusion A -representation

The PinyinGPT prediction state is a useful task-native personal-memory representation and retrieves better than the BGE alternatives tested here.

## Conclusion B -end-to-end conversion

The stronger retrieval signal does not automatically become a better candidate ranking under the existing M1 target-support rule.

Hidden-M1 and Original M1 are essentially tied on the primary Dev metric.

## Conclusion C -generic Cross-Encoder

Replacing BGE Stage-1 retrieval with Hidden retrieval improves M2 slightly, but the generic pretrained Cross-Encoder still does not outperform the simpler M1 family.

## Conclusion D -Frequency/Context fusion

A global fixed F/C fusion does not provide meaningful extra gain.

The tested transparent count-aware adaptive rule is worse.

The no-count confidence control is interesting on Ambiguous/Conflict but does not produce a meaningful Overall improvement.

## Conclusion E -bottleneck

The main unresolved problem is:

```text
historical evidence
->
task-specific candidate-level relevance / decision
```

This is the handoff to EM-3.

---

# 14. What is frozen and must not be silently changed

EM-2 is now closed.

The following must not be retrospectively changed under the EM-2 name:

- final-layer final-[SEP] hidden representation;
- hidden dimensionality 768;
- cosine retrieval;
- causal H5000 history semantics;
- exact segmented-Pinyin memory bucket;
- Hidden-M1 Top-N 3 / lambda 4;
- Hidden-M2 Dev selection K10 / lambda 4;
- Fixed GFC selection lambda_F 0.5 / lambda_C 4;
- Adaptive count-aware rule;
- Adaptive no-count control definition;
- Dev metrics reported here.

Any future method that changes these semantics belongs to a new stage/name.

---

# 15. Reproducibility inventory

Canonical runners:

```text
experiments/external_memory/em2_hidden_state_gate.py
experiments/external_memory/em2_cache_hidden_dev.py
experiments/external_memory/em2_hidden_knn_dev.py
experiments/external_memory/em2_hidden_m1_dev.py
experiments/external_memory/em2_hidden_m1_dev_boundary8.py
experiments/external_memory/em2_four_way_dev_compare.py
experiments/external_memory/em2_original_m2_same_surface_dev.py
experiments/external_memory/em2_hidden_m2_dev.py
experiments/external_memory/em2_fixed_gfc_dev.py
experiments/external_memory/em2_adaptive_gfc_dev.py
```

Important local artifacts:

```text
Frozen Generic / manifests:
C:\Users\chiar\Desktop\LBH\thesis-personalisation\
results\personalisation\pilot_a_context_memory

Original BGE embedding cache:
...\pilot_a_context_memory\cache\embedding_cache.sqlite3

Original M2 pair cache:
C:\Users\chiar\Desktop\LBH\thesis-personalisation\
results\personalisation\m2_h5000\cache\pair_scores.sqlite3

Frozen PinyinGPT:
C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat

Frozen generic reranker:
C:\Users\chiar\Desktop\LBH\thesis\.build\bge-reranker-base

EM-2 hidden cache:
results\personalisation\external_memory\
em2_hidden_dev\hidden_states.sqlite3
```

Important hashes:

```text
Frozen Generic Dev cache SHA256
588aa84c6397e8cb1a13576c0d5dfecd9dd2c4305b45be351328dd83ef62007d

EM-2 hidden cache SHA256
9a80a3314c184ccf3f0540916203c651474fad162dc3dab1fc97f7451f441df1
```

Exact canonical commands are maintained in:

```text
docs/external_memory/em2/EM2_REPRODUCIBILITY_2026-08-19.md
docs/REPRODUCIBILITY_INDEX.md
```

Generated `results/` trees and SQLite caches are evidence/local artifacts and are not normal Git source files.

---

# 16. Research provenance

EM-2 does not claim to invent nearest-neighbour language-model memory, task-model hidden-state retrieval, or Cross-Encoder reranking.

Relevant precedents include:

- Khandelwal et al. (2020), *Generalization through Memorization: Nearest Neighbor Language Models*.
- Khandelwal et al. (2021), *Nearest Neighbor Machine Translation*.
- Tan et al. (2022), *Exploring and Adapting Chinese GPT to Pinyin Input Method*.
- Nogueira and Cho (2019), *Passage Re-ranking with BERT*.

Thesis-specific adaptation includes:

- causal same-user external memory;
- H5000 before exact-Pinyin filtering;
- exact segmented-Pinyin retrieval;
- Frozen PinyinGPT prediction-state keys;
- Ambiguous / Conflict diagnostics;
- explicit separation of retrieval quality from candidate decision;
- comparison of Frequency and context memory;
- transparency/control framing.

Do not claim that the prior work proposed this exact personal Pinyin memory system.

---

# 17. Known limitations at closure

- New EM-2 methods were closed without opening Test.
- EM-2 development used three exploratory authors, not the final repaired six-author formal benchmark.
- MScarlet remained excluded from this exploratory branch because of the known script-normalisation confound.
- The tested adaptive history-count rule was only one transparent rule; it should not be generalised into a claim that history quantity is never useful.
- Generated local caches are required for fast reproduction and are not committed as normal Git source files.

These limitations are accepted at stage closure and are not reasons to reopen EM-2.

---

# 18. Handoff to EM-3

EM-3 should not ask again whether a generic semantic retriever can find useful history. EM-2 already established that the Frozen PinyinGPT hidden representation gives a stronger task-native retrieval signal.

The new EM-3 research question is:

> Can a task-specific learned relevance model predict which strictly-prior personal historical interactions are actually useful for the current Pinyin candidate decision?

The design should preserve the same causal information boundary and should define training labels, train/dev/test separation, leakage controls, candidate/history pair construction, and evaluation before opening Test.

EM-2 is closed.
