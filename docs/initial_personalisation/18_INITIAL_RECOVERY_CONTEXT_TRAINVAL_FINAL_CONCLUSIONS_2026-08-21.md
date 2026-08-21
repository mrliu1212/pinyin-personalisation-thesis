# 18. Initial-Pinyin Recovery + Context Reranking — Train-Val Final Conclusions and Data Record

**Date:** 2026-08-21  
**Scope:** Initial-Pinyin personalisation, Train-Fit / Train-Val development only  
**Status:** Train-Val development and post-hoc diagnosis complete; Dev3000 and Test untouched

---

## 1. Purpose of this record

This file is intended to be the **standalone record of the latest Initial-Pinyin recovery + context-reranking activity**, so that the key numerical results, selected operating points, diagnostic findings, and protocol state do not need to be reconstructed from `results/` artifacts.

It consolidates:

- Stage-1 recovery baselines and selected recovery philosophies;
- Stage-2 NGramRecency-only reranking (V1);
- Stage-2 NGramRecency + BGERecency joint reranking (V2);
- expanded BGE-weight boundary check (V3);
- post-hoc diagnostic analysis;
- Top-3 rescue / harm diagnosis;
- per-author behavior;
- current Train-Val selection and PRE-DEV freeze recommendation.

This document is a **development record**, not a Dev3000 or Test result report.

---

## 2. Frozen protocol and safety boundary

Current protocol:

```text
Clean3 Train
  -> Train-Fit / Train-Val
  -> development / method selection
  -> diagnosis
  -> PRE-DEV FREEZE
  -> Dev3000
  -> final freeze
  -> Test
```

Current state:

```text
Train-Val rows = 34,416
Dev3000 used   = false
Test used      = false
```

Gold usage rule:

```text
Gold may be used for:
- Train-Val evaluation
- Train-Val hyperparameter selection
- post-hoc diagnostic subsets
- rescue / harm classification

Gold must NOT be used for:
- candidate construction
- candidate scoring features
- online reranking features
- personal-history construction
```

Causal personal-history semantics:

```text
same author
-> strictly prior interactions only
-> latest up-to-5000 RAW same-author interactions
-> exact current Initial-Pinyin filtering afterward
```

Important consequence:

```text
H5000 is applied BEFORE exact-Pinyin filtering.
Earlier Train-Val rows may become history for later Train-Val rows.
Current/future target is never visible to the scorer.
```

---

## 3. Core system decomposition

The final system is interpreted as two stages:

```text
Initial Pinyin
 -> Generic PinyinGPT
 -> Generic Top10
 -> Generic Frequency F
 -> Personal recovery
 -> frozen final candidate set / Top10
 -> Context reranking
 -> final ordered Top10
```

Core thesis statement:

$$
\boxed{\text{Recovery determines availability; Context determines ordering.}}
$$

Stage-2 score:

$$
S_{\mathrm{final}}(c)
=
S_{\mathrm{REC}}(c)
+
\lambda_N P_{\mathrm{NG-R}}(c)
+
\lambda_B P_{\mathrm{BGE-R}}(c)
$$

Pure Stage-2 reranking uses a fixed candidate set for each Stage-1 recovery base. Therefore:

```text
Missing@10 is invariant under Stage-2 reranking.
Recovery Rec@10 is invariant under Stage-2 reranking on fixed R.
Only rank-position metrics such as Rec@1, Rec@3, Rec@5, and MRR@10 can change.
```

---

## 4. Initial-Pinyin baseline difficulty

### 4.1 Generic baseline

Population:

```text
N = 34,416
```

Generic Initial-Pinyin baseline:

| Metric | Value |
|---|---:|
| Macro-author Top1 | 0.307099 |
| Micro Top1 | 0.330573 |
| Top3 | 0.491341 |
| Top5 | 0.557996 |
| MRR@10 | 0.426472 |
| Missing@10 | 0.365092 |

Thus, approximately **36.51%** of Gold targets are absent from Generic Top10.

### 4.2 Full-vs-Initial same-anchor comparison

Same frozen 34,416 anchors:

| Condition | Generic Top1 | Missing@10 |
|---|---:|---:|
| Full Pinyin | 0.736024 | 0.069212 |
| Initial Pinyin | 0.330573 | 0.365092 |

Derived paired values:

```text
Observed Top1 gap              = 40.5451 percentage points
Candidate-coverage gap         = 29.5880 percentage points
Additional within-Top10 gap    = 10.9571 percentage points
Coverage share of observed gap ≈ 72.98%
```

Interpretation boundary:

> The 72.98% value is an **error-accounting decomposition**, not a causal attribution.

Additional paired result:

```text
Full-covered -> Initial-missing = 10,223 / 34,416 = 29.704%
```

This establishes that Initial-Pinyin difficulty is strongly associated with candidate-set deterioration, not only with within-list ordering.

---

## 5. Recovery population and candidate-surface ceiling

Generic Missing:

```text
12,565 / 34,416
```

Fixed recovery population:

$$
R=\{Gold\notin GenericTop10 \land Gold\in PersonalK5\}
$$

```text
|R| = 4,910
```

Theoretical Personal-K5 recoverability among Generic Missing:

$$
\frac{4910}{12565}=0.390768=39.0768\%
$$

This is a **candidate-surface upper bound**, not an achieved model metric.

The fixed Recovery evaluation denominator throughout the recovery analysis is:

```text
R = 4,910
```

---

## 6. Historical controls

### 6.1 Frequency F

| Metric | Value |
|---|---:|
| Macro Top1 | 0.382495 |
| Micro Top1 | 0.408473 |
| Top3 | 0.555120 |
| Top5 | 0.601000 |
| MRR@10 | 0.489432 |
| Missing@10 | 0.365092 |

### 6.2 Historical PV1

| Metric | Value |
|---|---:|
| Macro Top1 | 0.401872 |
| Micro Top1 | 0.426749 |
| Top3 | 0.598907 |
| Top5 | 0.663093 |
| MRR@10 | 0.524450 |
| Missing@10 | 0.291144 |

F -> PV1 Top1 transition:

```text
rescue = 933
harm   = 304
net    = +629
```

PV1 recovery on R=4,910:

| Metric | Value |
|---|---:|
| Rec1 | 0.1900 |
| Rec3 | 0.4923 |
| Rec5 | 0.5363 |
| Rec10 | 0.5401 |
| Recovery MRR | 0.3363 |

### 6.3 Historical PV1 + full context control

Historical frozen full-context control:

```text
PV1 + NGramRecency + BGERecency
```

Metrics:

| Metric | Value |
|---|---:|
| Macro Top1 | 0.429506 |
| Micro Top1 | 0.453423 |
| Top3 | 0.612099 |
| Top5 | 0.667335 |
| MRR@10 | 0.542766 |
| Missing@10 | 0.291144 |
| Rec1 | 0.3582 |
| Rec3 | 0.5165 |
| Rec5 | 0.5381 |
| Rec10 | 0.5401 |
| Recovery MRR | 0.4356 |

Historical context ablation summary:

| Method | Macro Top1 | Micro Top1 | Top3 | Top5 | MRR@10 | Missing@10 |
|---|---:|---:|---:|---:|---:|---:|
| PV1 | .401872 | .426749 | .598907 | .663093 | .524450 | .291144 |
| + BGERecency | .413942 | .438052 | .605532 | .665766 | .532635 | .291144 |
| + NGramRecency | .429091 | .452755 | .611285 | .665940 | .541944 | .291144 |
| + NGramRecency + BGERecency | .429506 | .453423 | .612099 | .667335 | .542766 | .291144 |

Historical full-context gain over NG-only:

```text
Delta Macro = +0.000415
```

No statistical-significance claim is made for this difference.

---

## 7. Stage-1 recovery bases selected for downstream context reranking

Three recovery philosophies were retained.

### 7.1 Coverage-first: K5+Entropy

Architecture:

```text
Personal K5
-> frozen Interpolated-NGram ordering/admission over all K5
-> exact PV1 candidate-frequency support
-> Generic-F boundary merge
```

Stage-1 score:

$$
S = B + 4F_{PV}(c) + 0.25C_E(q)
$$

where

$$
F_{PV}(c)=\frac{\log(1+n_c)}{\max_j\log(1+n_j)}
$$

Stage-1 overall metrics:

| Metric | Value |
|---|---:|
| Macro Top1 | 0.403790 |
| Micro Top1 | 0.428405 |
| Top3 | 0.602336 |
| Top5 | 0.677069 |
| MRR@10 | 0.533534 |
| Missing@10 | 0.243288 |

Recovery on R=4,910:

| Metric | Value | Count |
|---|---:|---:|
| Rec1 | 0.212627 | 1,044 |
| Rec3 | 0.612627 | 3,008 |
| Rec5 | 0.803462 | 3,945 |
| Rec10 | 0.987576 | 4,849 |
| Recovery MRR | 0.455236 | — |

Interpretation:

```text
Primary role = maximum candidate availability / recovery coverage
```

### 7.2 Balanced: 4P+4CS+2E

Stage-1 score:

$$
S=B+4P_{NG}+4CS+2C_E
$$

Stage-1 overall metrics:

| Metric | Value |
|---|---:|
| Macro Top1 | 0.404807 |
| Micro Top1 | 0.429364 |
| Top3 | 0.614801 |
| Top5 | 0.685815 |
| MRR@10 | 0.537433 |
| Missing@10 | 0.243172 |

Recovery on R=4,910:

| Metric | Value | Count |
|---|---:|---:|
| Rec1 | 0.258859 | 1,271 |
| Rec3 | 0.549287 | 2,697 |
| Rec5 | 0.716904 | 3,520 |
| Rec10 | 0.948473 | 4,657 |
| Recovery MRR | 0.454217 | — |

Interpretation:

```text
Primary role = strongest overall Stage-1 ranking balance
```

### 7.3 Front-rank: 6P+2CS+.25E

Stage-1 score:

$$
S=B+6P_{NG}+2CS+0.25C_E
$$

Stage-1 overall metrics:

| Metric | Value |
|---|---:|
| Macro Top1 | 0.400638 |
| Micro Top1 | 0.423989 |
| Top3 | 0.615876 |
| Top5 | 0.686047 |
| MRR@10 | 0.534545 |
| Missing@10 | 0.243869 |

Recovery on R=4,910:

| Metric | Value | Count |
|---|---:|---:|
| Rec1 | 0.276986 | 1,360 |
| Rec3 | 0.573727 | 2,817 |
| Rec5 | 0.710998 | 3,491 |
| Rec10 | 0.928310 | 4,558 |
| Recovery MRR | 0.467900 | — |

Interpretation:

```text
Primary role = aggressive front-rank / early-rank recovery
```

---

## 8. Stage-2 context definitions

### 8.1 NGramRecency

Primary lexical-context scorer:

```text
Hard suffix backoff
maxN = 2
tau  = 2048
```

Semantics:

1. On the current fixed final candidate set, choose the largest suffix order `<=2` with matching legal history whose target is in the current candidate set.
2. Use strict causal same-author H5000 history.
3. Apply recency:

$$
\exp(-age/2048)
$$

4. Normalize candidate support over the current candidate set.
5. If candidate mass is zero, use uniform support so ordering is unchanged.

Important distinction:

> This Stage-2 NGramRecency uses **hard suffix backoff**. It is different from the Stage-1 interpolated NGram used inside recovery.

### 8.2 BGERecency

Semantic complement:

```text
Current query context = last 64 characters
Candidate-conditioned same-Pinyin causal history
Top-5 historical rows selected by cosine similarity ONLY
Negative cosine clamped to zero
Recency applied only during aggregation
tau_B = 2048
```

For candidate `c`:

$$
R_{BGE-R}(c)=
\sum_{h\in Top5_{cos}(H_c)}
\max(0,\cos(E(q),E(h)))e^{-age(h)/\tau_B}
$$

Then normalize over the current final candidate set.

---

## 9. V1 — NGramRecency-only reranking

Runner:

```text
run_initial_recovery_ngram_context_fusion_v1.py
```

All three recovery bases selected:

```text
lambda_N = 6
```

All candidate-set / missing / Rec10 invariants passed.

### 9.1 K5+Entropy + NG-R

```text
lambda_N = 6
```

| Metric | Value |
|---|---:|
| Macro Top1 | 0.434673 |
| Micro Top1 | 0.457985 |
| Top3 | 0.624390 |
| Top5 | 0.686832 |
| MRR@10 | 0.555949 |
| Missing@10 | 0.243288 |
| Rec1 | 0.412831 |
| Rec3 | 0.718126 |
| Rec5 | 0.855804 |
| Rec10 | 0.987576 |
| Recovery MRR | 0.595977 |

Top1 transitions:

```text
rescue = 2,446
harm   = 1,428
net    = +1,018
```

Rank movement:

```text
improved = 3,904
worsened = 2,822
same     = 19,317
net      = +1,082
```

### 9.2 4P+4CS+2E + NG-R

```text
lambda_N = 6
```

| Metric | Value |
|---|---:|
| Macro Top1 | 0.432451 |
| Micro Top1 | 0.455718 |
| Top3 | 0.630434 |
| Top5 | 0.694183 |
| MRR@10 | 0.556494 |
| Missing@10 | 0.243172 |
| Rec1 | 0.422403 |
| Rec3 | 0.663544 |
| Rec5 | 0.785336 |
| Rec10 | 0.948473 |
| Recovery MRR | 0.577497 |

Top1 transitions:

```text
rescue = 2,375
harm   = 1,468
net    = +907
```

Rank movement:

```text
improved = 4,042
worsened = 3,006
same     = 18,999
net      = +1,036
```

### 9.3 6P+2CS+.25E + NG-R

```text
lambda_N = 6
```

| Metric | Value |
|---|---:|
| Macro Top1 | 0.431802 |
| Micro Top1 | 0.454818 |
| Top3 | 0.629184 |
| Top5 | 0.693776 |
| MRR@10 | 0.555338 |
| Missing@10 | 0.243869 |
| Rec1 | 0.434623 |
| Rec3 | 0.664562 |
| Rec5 | 0.774338 |
| Rec10 | 0.928310 |
| Recovery MRR | 0.580702 |

Top1 transitions:

```text
rescue = 2,530
harm   = 1,469
net    = +1,061
```

Rank movement:

```text
improved = 4,148
worsened = 2,993
same     = 18,882
net      = +1,155
```

---

## 10. V2 — joint NGramRecency + BGERecency reranking

Runner:

```text
run_initial_recovery_bge_ngram_context_fusion_v2.py
```

Grid:

```text
lambda_N = [0, .25, .5, 1, 2, 4, 6, 8, 12]
lambda_B = [0, .25, .5, 1, 2, 4, 6, 8]
```

BGE historical-context audit:

```text
required unique historical contexts = 42,717
seed-cache rows reused               = 38,847 required contexts
new historical embeddings           = 3,870
```

Online BGE scoring:

```text
rows                 = 34,416
final rate           ≈ 364.75 rows/s
mean embedding time  ≈ 1.881 ms
mean online time     ≈ 2.443 ms
```

All V1 NG-only regression checks passed.

### 10.1 V2 selected full-context result — K5+Entropy

```text
lambda_N = 6
lambda_B = 8
```

| Metric | Value |
|---|---:|
| Macro Top1 | 0.436767 |
| Micro Top1 | 0.459990 |
| Top3 | 0.626453 |
| Top5 | 0.688139 |
| MRR@10 | 0.557836 |
| Missing@10 | 0.243288 |
| Rec1 | 0.4430 |
| Rec3 | 0.7481 |
| Rec5 | 0.8778 |
| Rec10 | 0.9876 |
| Recovery MRR | 0.6218 |
| Top1Net vs Stage1 | +1,087 |

### 10.2 V2 selected full-context result — 4P+4CS+2E

```text
lambda_N = 4
lambda_B = 6
```

| Metric | Value |
|---|---:|
| Macro Top1 | 0.437058 |
| Micro Top1 | 0.460571 |
| Top3 | 0.631392 |
| Top5 | 0.696478 |
| MRR@10 | 0.559755 |
| Missing@10 | 0.243172 |
| Rec1 | 0.4246 |
| Rec3 | 0.6969 |
| Rec5 | 0.8153 |
| Rec10 | 0.9485 |
| Recovery MRR | 0.5892 |
| Top1Net vs Stage1 | +1,074 |

### 10.3 V2 selected full-context result — 6P+2CS+.25E

```text
lambda_N = 4
lambda_B = 6
```

| Metric | Value |
|---|---:|
| Macro Top1 | 0.436477 |
| Micro Top1 | 0.459786 |
| Top3 | 0.630085 |
| Top5 | 0.696013 |
| MRR@10 | 0.558806 |
| Missing@10 | 0.243869 |
| Rec1 | 0.4415 |
| Rec3 | 0.6961 |
| Rec5 | 0.8069 |
| Rec10 | 0.9283 |
| Recovery MRR | 0.5951 |
| Top1Net vs Stage1 | +1,232 |

### 10.4 Full-context improvement over historical PV1 full-context control

Historical control Macro:

```text
PV1 + NG-R + BGE-R = 0.429506
```

New full-context Macro gains:

```text
K5+Entropy       0.436767 - 0.429506 = +0.007261
4P+4CS+2E        0.437058 - 0.429506 = +0.007552
6P+2CS+.25E      0.436477 - 0.429506 = +0.006971
```

No significance claim is made.

### 10.5 BGE incremental gain over NG-only

#### K5+Entropy

```text
Delta Macro  = +0.002094
Delta MRR    = +0.001887
Delta Rec1   = +0.0302
Delta Rec3   = +0.0300
Delta Rec5   = +0.0220
Delta RecMRR = +0.0258
```

#### 4P+4CS+2E

```text
Delta Macro  = +0.004607
Delta MRR    = +0.003261
Delta Rec1   = +0.0022
Delta Rec3   = +0.0334
Delta Rec5   = +0.0300
Delta RecMRR = +0.0117
```

#### 6P+2CS+.25E

```text
Delta Macro  = +0.004675
Delta MRR    = +0.003468
Delta Rec1   = +0.0069
Delta Rec3   = +0.0315
Delta Rec5   = +0.0326
Delta RecMRR = +0.0144
```

V2 caveat at the time:

```text
K5+Entropy selected lambda_B=8,
which was the V2 upper grid boundary.
```

Therefore a boundary follow-up was required before freezing.

---

## 11. V3 — expanded BGE boundary check

Runner:

```text
run_initial_recovery_bge_ngram_context_fusion_v3.py
```

Expanded grid:

```text
lambda_N = [0, .25, .5, 1, 2, 4, 6, 8, 12]
lambda_B = [0, .25, .5, 1, 2, 4, 6, 8, 12, 16]
```

V3 reused V1/V2 support files and performed arithmetic reranking only.

Runtime:

```text
193.1 s
```

No BGE embeddings were recomputed.

### 11.1 Regression checks

V1 NGram-only selected-point regressions:

```text
PASS K5+Entropy      lambda_N=6  Macro=0.434673  MRR=0.555949
PASS 4P+4CS+2E       lambda_N=6  Macro=0.432451  MRR=0.556494
PASS 6P+2CS+.25E     lambda_N=6  Macro=0.431802  MRR=0.555338
```

V2 full-context selected-point regressions:

```text
PASS K5+Entropy      lambda_N=6 lambda_B=8  Macro=0.436767
PASS 4P+4CS+2E       lambda_N=4 lambda_B=6  Macro=0.437058
PASS 6P+2CS+.25E     lambda_N=4 lambda_B=6  Macro=0.436477
```

### 11.2 V3 selected results

| Recovery base | lambda_N | lambda_B | Macro | Micro | Top3 | Top5 | MRR | Missing | Rec1 | Rec3 | Rec5 | Rec10 | RecMRR | Top1Net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K5+Entropy | 6 | 8 | .436767 | .459990 | .626453 | .688139 | .557836 | .243288 | .4430 | .7481 | .8778 | .9876 | .6218 | +1087 |
| **4P+4CS+2E** | **4** | **6** | **.437058** | **.460571** | **.631392** | **.696478** | **.559755** | **.243172** | .4246 | .6969 | .8153 | .9485 | .5892 | +1074 |
| 6P+2CS+.25E | 4 | 6 | .436477 | .459786 | .630085 | .696013 | .558806 | .243869 | .4415 | .6961 | .8069 | .9283 | .5951 | **+1232** |

### 11.3 V2 -> V3 stability

```text
K5+Entropy       (6,8) -> (6,8)  DeltaMacro=+0.000000  DeltaMRR=+0.000000
4P+4CS+2E        (4,6) -> (4,6)  DeltaMacro=+0.000000  DeltaMRR=+0.000000
6P+2CS+.25E      (4,6) -> (4,6)  DeltaMacro=+0.000000  DeltaMRR=+0.000000
```

### 11.4 Boundary status

```text
K5+Entropy       lambda_B=8  upper=16  hit_upper_boundary=false
4P+4CS+2E        lambda_B=6  upper=16  hit_upper_boundary=false
6P+2CS+.25E      lambda_B=6  upper=16  hit_upper_boundary=false
```

Final boundary conclusion:

```text
PASS: no selected lambda_B hits the expanded upper boundary.
```

Therefore **no further lambda-grid expansion is required**.

---

## 12. Final Train-Val overall selection

Under the pre-specified primary criterion:

```text
Primary selection criterion = Macro-author Top1 on all 34,416 Train-Val rows
```

the current Train-Val development selection is:

$$
\boxed{
4P+4CS+2E
+4P_{\text{NG-R}}
+6P_{\text{BGE-R}}
}
$$

Plain form:

```text
Primary overall model:
4P+4CS+2E Recovery
+ NGramRecency lambda_N=4
+ BGERecency   lambda_B=6
```

Final metrics:

```text
Macro Top1 = 0.437058
Micro Top1 = 0.460571
Top3       = 0.631392
Top5       = 0.696478
MRR@10     = 0.559755
Missing@10 = 0.243172
Rec1       = 0.4246
Rec3       = 0.6969
Rec5       = 0.8153
Rec10      = 0.9485
RecMRR     = 0.5892
```

Among the three frozen recovery bases, this model has the highest:

```text
Macro Top1
Micro Top1
Top3
Top5
MRR@10
```

However, its Macro advantage over K5+Entropy is very small:

$$
0.437058-0.436767=0.000291
$$

or approximately:

```text
+0.0291 percentage points
```

Therefore the correct conclusion is:

> **4P+4CS+2E is the Train-Val development selection under the pre-specified Macro-author Top1 criterion, while K5+Entropy remains extremely competitive overall and substantially stronger on recovery-oriented metrics.**

Do **not** describe Balanced as clearly or significantly superior without formal statistical testing.

---

## 13. Final recovery-oriented winner

The final full-context K5+Entropy system is:

```text
K5+Entropy Recovery
+ NGramRecency lambda_N=6
+ BGERecency   lambda_B=8
```

Recovery metrics:

```text
Rec1   = 0.4430
Rec3   = 0.7481
Rec5   = 0.8778
Rec10  = 0.9876
RecMRR = 0.6218
```

Compared with Balanced:

| Recovery metric | K5+Entropy | 4P+4CS+2E |
|---|---:|---:|
| Rec1 | **.4430** | .4246 |
| Rec3 | **.7481** | .6969 |
| Rec5 | **.8778** | .8153 |
| Rec10 | **.9876** | .9485 |
| Recovery MRR | **.6218** | .5892 |

Thus:

$$
\boxed{\text{best recovery system} \neq \text{best overall system}}
$$

Operational interpretation:

```text
K5+Entropy  = strongest recovery / coverage-oriented operating point
4P+4CS+2E   = strongest overall Train-Val operating point
6P+2CS+.25E = strongest aggressive front-rank comparison point
```

---

## 14. Primary-context increment diagnosis

For the selected Balanced model:

```text
Stage1 Recovery        Macro = 0.404807
+ NGramRecency         Macro = 0.432451
+ BGERecency           Macro = 0.437058
```

Incremental gains:

```text
NGram increment = +0.027644 Macro
BGE increment   = +0.004607 Macro
```

Approximate share of the total Macro gain:

```text
NGram ≈ 86%
BGE   ≈ 14%
```

This is a descriptive arithmetic decomposition only.

### 14.1 Top1 rescue / harm

Recovery -> NGramRecency:

```text
rescue = 2,375
harm   = 1,468
net    = +907
```

NGramRecency -> Full context:

```text
rescue = 681
harm   = 514
net    = +167
```

Recovery -> Full context:

```text
net = +1,074
```

and:

```text
907 + 167 = 1,074
```

Main interpretation:

> **NGramRecency is the dominant contextual reranker; BGERecency provides a smaller complementary semantic refinement.**

---

## 15. Population diagnosis: why K5 wins Recovery but Balanced wins Overall

Full-context diagnostic result:

```text
K5+Entropy:
Generic-covered Macro ≈ 0.6081
Recovery-R Macro      ≈ 0.4420

4P+4CS+2E:
Generic-covered Macro ≈ 0.6155
Recovery-R Macro      ≈ 0.4219
```

Population sizes:

```text
Generic-covered ≈ 21,851 rows
Recovery R      = 4,910 rows
```

Interpretation:

```text
K5 wins the recovery population.
Balanced better preserves / ranks the much larger Generic-covered population.
```

This explains how Balanced can achieve the best overall Macro while K5 dominates Recovery metrics.

---

## 16. BGE behavior differs by recovery surface

### 16.1 K5+Entropy: BGE increment

On Generic-covered:

```text
Delta Macro ≈ -0.004509
Top1 net    = -79
```

On Recovery R:

```text
Delta Macro ≈ +0.030332
Top1 net    = +148
```

Interpretation:

> On the wide K5 surface, BGE strongly helps recovered personal candidates but sacrifices some originally covered cases.

### 16.2 Balanced: BGE increment

On Generic-covered:

```text
Delta Macro ≈ +0.007178
Top1 net    = +156
```

On Recovery R:

```text
Delta Macro ≈ +0.002409
Top1 net    = +11
```

Interpretation:

> On the Balanced surface, BGE contributes more to overall / covered-case Top1 ordering than to additional Top1 recovery.

This is a key reason that **the same BGE scorer behaves differently depending on Stage-1 candidate geometry**.

---

## 17. Conflict-subset diagnosis

Formal Conflict definition:

```text
Ambiguous
AND unique frequency winner
AND Gold != winner
```

Conflict is a **Gold-derived diagnostic subset only** and must never become a runtime feature without a new development protocol.

Observed BGE Top1 effect on Conflict:

| Base | BGE Delta Macro on Conflict | Top1 net |
|---|---:|---:|
| K5+Entropy | -0.002293 | -38 |
| 4P+4CS+2E | -0.005313 | -84 |
| 6P+2CS+.25E | -0.004107 | -67 |

For the primary Balanced model:

```text
BGE Conflict rescue = 273
BGE Conflict harm   = 357
BGE Conflict net    = -84
```

At the same time, Conflict Top3 / Top5 deltas were positive:

```text
Top3 Delta = +0.002801
Top5 Delta = +0.004429
```

Interpretation:

> BGE can improve list structure while still making rank-1 decisions worse on preference-conflict cases.

Do not redesign the system around this finding at this stage, because that would reopen Train-Val adaptive tuning.

---

## 18. Top-3 rescue / harm diagnosis

Definition:

```text
rescue@3 = outside Top3 -> inside Top3
harm@3   = inside Top3  -> outside Top3
net@3    = rescue@3 - harm@3
```

### 18.1 Overall Top3 transitions

| Base | Transition | Rescue | Harm | Net |
|---|---|---:|---:|---:|
| K5+Entropy | Recovery -> NG-R | 1,145 | 386 | +759 |
| K5+Entropy | NG-R -> Full | 384 | 313 | +71 |
| K5+Entropy | Recovery -> Full | 1,434 | 604 | +830 |
| 4P+4CS+2E | Recovery -> NG-R | 989 | 451 | +538 |
| 4P+4CS+2E | NG-R -> Full | 309 | 276 | +33 |
| 4P+4CS+2E | Recovery -> Full | 1,198 | 627 | +571 |
| 6P+2CS+.25E | Recovery -> NG-R | 906 | 448 | +458 |
| 6P+2CS+.25E | NG-R -> Full | 310 | 279 | +31 |
| 6P+2CS+.25E | Recovery -> Full | 1,104 | 615 | +489 |

### 18.2 Generic-covered Top3 transitions

| Base | Transition | Rescue | Harm | Net |
|---|---|---:|---:|---:|
| K5+Entropy | Recovery -> NG-R | 567 | 326 | +241 |
| K5+Entropy | NG-R -> Full | 183 | 259 | -76 |
| K5+Entropy | Recovery -> Full | 689 | 524 | +165 |
| 4P+4CS+2E | Recovery -> NG-R | 409 | 432 | -23 |
| 4P+4CS+2E | NG-R -> Full | 119 | 250 | -131 |
| 4P+4CS+2E | Recovery -> Full | 453 | 607 | -154 |
| 6P+2CS+.25E | Recovery -> NG-R | 438 | 426 | +12 |
| 6P+2CS+.25E | NG-R -> Full | 129 | 253 | -124 |
| 6P+2CS+.25E | Recovery -> Full | 487 | 599 | -112 |

### 18.3 Recoverable-R Top3 transitions

| Base | Transition | Rescue | Harm | Net |
|---|---|---:|---:|---:|
| K5+Entropy | Recovery -> NG-R | 578 | 60 | +518 |
| K5+Entropy | NG-R -> Full | 201 | 54 | +147 |
| K5+Entropy | Recovery -> Full | 745 | 80 | +665 |
| 4P+4CS+2E | Recovery -> NG-R | 580 | 19 | +561 |
| 4P+4CS+2E | NG-R -> Full | 190 | 26 | +164 |
| 4P+4CS+2E | Recovery -> Full | 745 | 20 | **+725** |
| 6P+2CS+.25E | Recovery -> NG-R | 468 | 22 | +446 |
| 6P+2CS+.25E | NG-R -> Full | 181 | 26 | +155 |
| 6P+2CS+.25E | Recovery -> Full | 617 | 16 | +601 |

Net Top3 gain on R=4,910:

```text
K5+Entropy       +665 / 4910 ≈ +13.54 pp
4P+4CS+2E        +725 / 4910 ≈ +14.77 pp
6P+2CS+.25E      +601 / 4910 ≈ +12.24 pp
```

Important distinction:

```text
Highest final Recovery Rec3           = K5+Entropy (.7481)
Largest Context-driven Recovery Top3 gain = 4P+4CS+2E (+725 rows, ≈ +14.77 pp)
```

This happens because K5 starts from a higher Stage-1 Rec3 baseline.

---

## 19. Top3 cutoff dependence of BGE

On recoverable R, BGE is consistently positive at Top3:

```text
K5       +201 rescue -54 harm = +147
Balanced +190 rescue -26 harm = +164
Front    +181 rescue -26 harm = +155
```

But on Generic-covered, BGE Top3 balance is negative for all three:

```text
K5       +183 -259 = -76
Balanced +119 -250 = -131
Front    +129 -253 = -124
```

For Balanced this coexists with a positive Generic-covered Top1 net (`+156`).

Therefore BGE has **cutoff-dependent effects**:

> It can improve some rank-1 decisions while simultaneously moving another set of Gold candidates from rank 2/3 to rank 4+.

The correct statement is therefore not that BGE uniformly preserves covered cases, but that:

> **BGE provides positive overall semantic refinement with subset- and cutoff-dependent trade-offs.**

---

## 20. Per-author metrics

Train-Val row counts:

```text
Agent Phage = 13,741
Etinjat     = 8,030
breaddddd   = 12,645
```

Etinjat has the smallest Train-Val population.

### 20.1 K5+Entropy per-author progression

| Author | Stage | Top1 | Top3 | Top5 | MRR@10 | Missing@10 | Mean rank given Top10 |
|---|---|---:|---:|---:|---:|---:|---:|
| Agent Phage | Recovery | .472819 | .666618 | .747326 | .587836 | .171676 | 2.2955 |
| Agent Phage | +NG-R | .491085 | .681464 | .752929 | .602567 | .171676 | 2.2031 |
| Agent Phage | +NG-R+BGE-R | .494142 | .679863 | .752420 | .603886 | .171676 | 2.1967 |
| Etinjat | Recovery | .236613 | .359278 | .429016 | .317610 | .483935 | 2.9100 |
| Etinjat | +NG-R | .274222 | .388418 | .441594 | .346207 | .483935 | 2.6108 |
| Etinjat | +NG-R+BGE-R | .277210 | .390162 | .442590 | .348812 | .483935 | 2.5811 |
| breaddddd | Recovery | .501938 | .686833 | .758244 | .611646 | .168288 | 2.1601 |
| breaddddd | +NG-R | .538711 | .712218 | .770739 | .638484 | .168288 | 1.9948 |
| breaddddd | +NG-R+BGE-R | .538948 | .718466 | .774219 | .640533 | .168288 | 1.9606 |

### 20.2 Balanced per-author progression

| Author | Stage | Top1 | Top3 | Top5 | MRR@10 | Missing@10 | Mean rank given Top10 |
|---|---|---:|---:|---:|---:|---:|---:|
| Agent Phage | Recovery | .470344 | .682192 | .759042 | .590938 | .171312 | 2.2177 |
| Agent Phage | +NG-R | .487374 | .691216 | .764209 | .603335 | .171312 | 2.1446 |
| Agent Phage | +NG-R+BGE-R | .493923 | .689542 | .764937 | .606609 | .171312 | 2.1343 |
| Etinjat | Recovery | .237235 | .363138 | .428892 | .318799 | .485056 | 2.8907 |
| Etinjat | +NG-R | .271980 | .388169 | .442839 | .345472 | .485056 | 2.5918 |
| Etinjat | +NG-R+BGE-R | .275218 | .388917 | .447198 | .347801 | .485056 | 2.5497 |
| breaddddd | Recovery | .506841 | .701384 | .769395 | .618130 | .167655 | 2.0790 |
| breaddddd | +NG-R | .537999 | .718229 | .777699 | .639600 | .167655 | 1.9618 |
| breaddddd | +NG-R+BGE-R | .542032 | .722183 | .780388 | .643439 | .167655 | 1.9281 |

### 20.3 Front-rank per-author progression

| Author | Stage | Top1 | Top3 | Top5 | MRR@10 | Missing@10 | Mean rank given Top10 |
|---|---|---:|---:|---:|---:|---:|---:|
| Agent Phage | Recovery | .460010 | .681319 | .758897 | .584885 | .171021 | 2.2444 |
| Agent Phage | +NG-R | .486791 | .688887 | .763409 | .602055 | .171021 | 2.1590 |
| Agent Phage | +NG-R+BGE-R | .492031 | .687723 | .765228 | .605032 | .171021 | 2.1445 |
| Etinjat | Recovery | .240598 | .368867 | .431009 | .321964 | .487547 | 2.8226 |
| Etinjat | +NG-R | .273225 | .388418 | .444707 | .345726 | .487547 | 2.5655 |
| Etinjat | +NG-R+BGE-R | .275841 | .388543 | .445953 | .347864 | .487547 | 2.5235 |
| breaddddd | Recovery | .501305 | .701621 | .768841 | .614839 | .168288 | 2.0870 |
| breaddddd | +NG-R | .535389 | .717200 | .776275 | .637683 | .168288 | 1.9696 |
| breaddddd | +NG-R+BGE-R | .541558 | .720838 | .779597 | .642527 | .168288 | 1.9293 |

---

## 21. Etinjat-specific diagnosis

Etinjat is substantially harder than the other two authors under Initial-Pinyin.

For the final Balanced model:

```text
Top1       = 0.275218
Top3       = 0.388917
Top5       = 0.447198
MRR@10     = 0.347801
Missing@10 = 0.485056
```

By comparison:

```text
Agent Missing@10      = 0.171312
breaddddd Missing@10  = 0.167655
```

Thus Etinjat has approximately **48.5%** Gold missing from the frozen Top10, versus roughly **17%** for the other two authors.

Candidate-presence ceiling for Etinjat:

$$
1-0.485056\approx0.514944
$$

So only about **51.5%** of Etinjat rows even contain Gold in the final candidate set.

The difficulty predates the new recovery/context method.

Generic Initial-Pinyin Top1 by author:

```text
Agent Phage = 0.389273
Etinjat     = 0.151557
breaddddd   = 0.380467
```

Frequency F Top1 by author:

```text
Agent Phage = 0.460738
Etinjat     = 0.207347
breaddddd   = 0.479399
```

PV1 Top1 by author:

```text
Agent Phage = 0.471436
Etinjat     = 0.232877
breaddddd   = 0.501305
```

Therefore Etinjat was already substantially harder at the Generic / Frequency / PV1 stages.

Context still improves Etinjat meaningfully under the Balanced system:

```text
Stage1       0.237235
+ NGram      0.271980
+ BGE        0.275218
```

Total Top1 gain:

```text
+0.037983
```

Mean rank when Gold is present also improves:

```text
2.8907 -> 2.5918 -> 2.5497
```

Therefore the correct interpretation is:

> **Etinjat is intrinsically much harder under the Initial-Pinyin candidate surface; contextual reranking helps substantially but cannot compensate for the much higher candidate-missing rate.**

Etinjat also has fewer Train-Val rows:

```text
Etinjat     = 8,030
Agent Phage = 13,741
breaddddd   = 12,645
```

This is compatible with the hypothesis that less available author-specific material may reduce useful personal-history coverage, but **row count alone does not prove that fewer works / less history causes the lower accuracy**. A causal statement would require explicit per-author work-count / Train-Fit-history / same-Initial-evidence analysis.

---

## 22. Author-level consistency of context gains

Balanced Top1 progression:

| Author | Recovery | +NG | +NG+BGE | Full gain vs Recovery |
|---|---:|---:|---:|---:|
| Agent Phage | .470344 | .487374 | .493923 | +.023579 |
| Etinjat | .237235 | .271980 | .275218 | +.037983 |
| breaddddd | .506841 | .537999 | .542032 | +.035192 |

BGE incremental Top1 gains are positive for all three authors:

```text
Agent Phage  +0.006550
Etinjat      +0.003238
breaddddd    +0.004033
```

Safe interpretation:

> Context gains are observed for all three Train-Val authors.

Do not claim statistical consistency or significance without formal testing.

---

## 23. Three-base disagreement

Final full-context systems choose the same Top1 candidate for approximately:

```text
31,793 / 34,416 ≈ 92.38%
```

Disagreement:

```text
2,623 / 34,416 ≈ 7.62%
```

Thus the three recovery philosophies differ mainly on a relatively small set of difficult rows.

Balanced vs K5 disagreement diagnostic:

```text
Balanced correct, K5 wrong = 473
K5 correct, Balanced wrong = 453
net                         = +20 rows for Balanced
```

This is consistent with the very small Macro difference:

```text
Balanced = 0.437058
K5       = 0.436767
Delta    = +0.000291
```

Per-author final Top1 comparison also shows that Balanced is not uniformly superior:

```text
Agent Phage:
K5       = 0.494142
Balanced = 0.493923

Etinjat:
K5       = 0.277210
Balanced = 0.275218

breaddddd:
K5       = 0.538948
Balanced = 0.542032
```

Therefore the selection is driven by the pre-specified Macro-author Top1 criterion, not by uniform dominance.

---

## 24. Margin diagnosis

Post-hoc margin analysis indicates that larger final score margins correspond to substantially higher observed Top1 accuracy.

For the Balanced full model, high-margin rows (`final margin >= 1`) reach approximately:

```text
Top1 ≈ 0.516
```

while lower-margin bins are much weaker, around:

```text
Top1 ≈ 0.11–0.23
```

Interpretation:

> Final score margin is useful as a post-hoc confidence / controllability diagnostic.

It is **not** used as a new gate or runtime feature in the current frozen model.

---

## 25. Final operating points to freeze before Dev3000

### 25.1 Primary overall model

```text
Recovery:
4P+4CS+2E

Stage2:
NGramRecency lambda_N = 4
BGERecency   lambda_B = 6
```

### 25.2 Coverage-oriented comparison

```text
Recovery:
K5+Entropy

Stage2:
NGramRecency lambda_N = 6
BGERecency   lambda_B = 8
```

### 25.3 Front-rank comparison

```text
Recovery:
6P+2CS+.25E

Stage2:
NGramRecency lambda_N = 4
BGERecency   lambda_B = 6
```

### 25.4 Historical control

```text
PV1
+ NGramRecency
+ BGERecency
```

---

## 26. Final development conclusions

The Train-Val evidence supports the following development-level conclusions:

1. **Initial-Pinyin candidate availability is a major bottleneck.** Generic Missing@10 rises to 0.365092, and Personal K5 exposes a bounded recovery opportunity of 4,910 / 12,565 Generic-missing rows.
2. **Recovery and contextual ordering are distinct roles.** Recovery determines whether the Gold candidate can enter the final list; context reranking determines where it is placed.
3. **K5+Entropy is the strongest recovery-oriented surface.** Its full-context Recovery metrics are Rec1=.4430, Rec3=.7481, Rec5=.8778, Rec10=.9876, RecMRR=.6218.
4. **4P+4CS+2E is the current Train-Val overall selection.** With lambda_N=4 and lambda_B=6 it achieves Macro=.437058, Micro=.460571, Top3=.631392, Top5=.696478, MRR=.559755.
5. **The Balanced-vs-K5 overall difference is very small.** The Macro gap is only 0.000291, and the two systems differ by only +20 net correct rows in their direct disagreement set. Therefore no claim of clear or significant superiority is justified.
6. **NGramRecency is the main contextual mechanism.** For Balanced, it contributes +0.027644 Macro and +907 Top1 net from Stage1, compared with BGE's additional +0.004607 Macro and +167 Top1 net.
7. **NGram is especially effective on recoverable missing cases.** On R, Balanced Recovery->NG produces Top3 rescue=580, harm=19, net=+561.
8. **BGE is complementary, not uniformly beneficial.** It improves overall metrics, but its effect depends on recovery surface, subset, and cutoff. In particular, it is positive on recoverable-R Top3 but negative on Generic-covered Top3 transition balance.
9. **Etinjat is a substantially harder author condition.** Its final Balanced Missing@10 is 0.485056 and Top1 is 0.275218, but context still provides a +0.037983 absolute Top1 gain from Stage1.
10. **The three recovery philosophies are mostly aligned.** Their final Top1 predictions are identical on about 92.38% of Train-Val rows; the differences concentrate in difficult cases.

Concise thesis-level summary:

> Initial-Pinyin input substantially increases candidate ambiguity and candidate-set failure relative to Full Pinyin, making recovery a prerequisite for effective personalisation. Personal K5 history exposes a substantial but bounded recovery opportunity, with 4,910 of 12,565 Generic-missing cases theoretically recoverable. Coverage-first recovery maximizes this opportunity, reaching Rec@10=.9876 after contextual reranking, whereas the balanced 4P+4CS+2E recovery surface achieves the best overall Train-Val ranking under the pre-specified Macro-author Top1 criterion. NGramRecency provides the dominant contextual improvement by moving recovered candidates into high ranks, while BGERecency contributes a smaller complementary semantic gain with subset- and cutoff-dependent trade-offs. The results therefore support a two-stage interpretation: **recovery determines candidate availability, while context determines candidate ordering**.

---

## 27. Current status before Dev3000

```text
TRAIN-VAL DEVELOPMENT: COMPLETE
HYPERPARAMETER SEARCH: COMPLETE
V1 NGRAM REGRESSION CHECKS: PASSED
V2 FULL-CONTEXT REGRESSION CHECKS: PASSED
V3 BOUNDARY CHECK: PASSED
POST-HOC DIAGNOSIS: COMPLETE
TOP3 RESCUE/HARM DIAGNOSIS: COMPLETE
DEV3000: UNTOUCHED
TEST: UNTOUCHED
```

Next step:

```text
PRE-DEV FREEZE
-> Dev3000
```

No further Train-Val lambda search, feature addition, conflict gate, margin gate, or recovery-score redesign should be performed before Dev3000 unless the development protocol is explicitly reopened and documented as such.

---

## 28. Main result artifact locations

Main result root:

```text
results\personalisation\initial_recovery_comparison_v1
```

Key output directories:

```text
recovery_ngram_context_fusion_v1\
recovery_bge_ngram_context_fusion_v2\
recovery_bge_ngram_context_fusion_v3\
recovery_context_diagnostics_v1\
recovery_context_topk_transitions_v1\
```

Key V3 files:

```text
grid_results.csv
selected_metrics.csv
selected_predictions.jsonl
full_comparison.csv
comparison.json
run_manifest.json
artifact_checksums.json
```

Key diagnosis files:

```text
diagnostic_summary.json
headline_comparison.csv
per_author.csv
subset_metrics.csv
top1_transitions.csv
rank_movement.csv
recovery_diagnostics.csv
context_increment.csv
base_disagreement.csv
margin_diagnostics.csv
error_examples.jsonl
diagnostic_report.md
run_manifest.json
artifact_checksums.json
```

Top-K transition diagnosis:

```text
topk_transitions.csv
topk_transitions.json
```

---

## 29. Provenance notes

Known frozen Initial Train-Val identity:

```text
initial_train_val_v1.jsonl
SHA256 = d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4
```

Known candidate surface identity:

```text
candidate_surface\train_val_candidate_surface.jsonl
SHA256 = 205c0ba01cd0678d7a4341c503fa2e74cf126a70182ff582687025e4946764b2
rows = 34,416
eligible rows = 30,509
candidate pairs = 123,738
```

Known frozen Generic prediction identity:

```text
train_val_generic\predictions.jsonl
SHA256 = bd0fb4dc304e0b266b90fae6fe3ac65424d2f52b23fedfa881212706ba2c2873
```

Known Stage-1 / Stage-2 runner checksums recorded during development:

```text
run_initial_recovery_ngram_context_fusion_v1.py
SHA256 = e6dcd1f68028ad5065064b6b714eaa88d92f74363a328570bfcc777b13271dc2

run_initial_recovery_bge_ngram_context_fusion_v2.py
SHA256 = b7d95374aa421cbc364699e44e0850ba2e72e50a2a5f816ad37f85b138d1435a

run_initial_recovery_bge_ngram_context_fusion_v3.py
SHA256 = 2b29a86957b4f2adf17a13de37648766e1423d0ec99a57ea257c5aa155d89335
```

---

## 30. Interpretation boundaries to preserve in the thesis

Do not overstate the following:

```text
- K5 recoverability is a candidate-surface ceiling, not model accuracy.
- The 72.98% Full-vs-Initial decomposition is not causal attribution.
- Balanced's Train-Val advantage over K5 is very small and is not a significance claim.
- Conflict diagnosis is Gold-derived and analysis-only.
- Margin diagnosis is post-hoc and not part of the frozen runtime model.
- Etinjat's lower row count does not by itself prove that fewer works / less history causes lower accuracy.
- Dev3000 and Test have not yet validated generalisation.
```

End of Train-Val final conclusions and data record.
