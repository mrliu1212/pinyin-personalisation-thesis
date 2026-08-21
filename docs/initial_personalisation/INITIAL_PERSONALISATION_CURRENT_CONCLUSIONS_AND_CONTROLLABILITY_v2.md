# Initial-Pinyin Personalisation: Current Conclusions, Model Comparison, Context Reranking, and Controllability — v2

**Updated:** 2026-08-21  
**Status:** Train-Val development complete; post-hoc diagnosis complete; PRE-DEV FREEZE is the next protocol step  
**Evaluation surface:** 34,416 standardized Clean3 Train-Val rows  
**Dev3000 used:** No  
**Test used:** No

This file supersedes the earlier `INITIAL_PERSONALISATION_CURRENT_CONCLUSIONS_AND_CONTROLLABILITY.md` as the **current master summary**. The earlier file remains a historical record of the pre-Stage2 development state.

Detailed latest data record:

```text
18_INITIAL_RECOVERY_CONTEXT_TRAINVAL_FINAL_CONCLUSIONS_2026-08-21.md
```

Latest reproducibility record:

```text
19_INITIAL_RECOVERY_CONTEXT_TRAINVAL_REPRODUCIBILITY_2026-08-21.md
```

---

## 1. Current high-level conclusion

The final Train-Val evidence supports a two-stage architecture:

```text
Initial Pinyin
 -> Generic PinyinGPT Top10
 -> Generic Frequency F
 -> Stage-1 personal recovery
 -> frozen final candidate set
 -> Stage-2 context reranking
 -> final ordered Top10
```

The central interpretation is:

\[
\boxed{\text{Recovery determines availability; Context determines ordering.}}
\]

The current **primary overall Train-Val development selection** is:

```text
Stage1 Recovery = 4P+4CS+2E
Stage2 NGramRecency lambda_N = 4
Stage2 BGERecency   lambda_B = 6
```

with:

| Macro Top1 | Micro Top1 | Top3 | Top5 | MRR@10 | Missing@10 |
|---:|---:|---:|---:|---:|---:|
| **.437058** | **.460571** | **.631392** | **.696478** | **.559755** | **.243172** |

The strongest recovery-oriented operating point is instead:

```text
K5+Entropy + NGramRecency(6) + BGERecency(8)
```

with:

```text
Rec1  = .4430
Rec3  = .7481
Rec5  = .8778
Rec10 = .9876
RecMRR= .6218
```

Therefore:

\[
\boxed{\text{best recovery system} \neq \text{best overall ranking system}}
\]

The Balanced-vs-K5 Macro difference is only:

\[
.437058-.436767=.000291
\]

or about **0.0291 percentage points**. It is a development-selection difference under the pre-specified Macro-author Top1 criterion, not evidence of clear or statistically significant superiority.

---

## 2. Protocol and interpretation boundary

Protocol:

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
Dev3000 used = false
Test used = false
```

Gold may be used for Train-Val evaluation, hyperparameter selection, rescue/harm classification, and post-hoc diagnostic subsets. It must not be used for candidate construction, online features, or scoring.

Causal personal history:

```text
same author
-> strictly prior rows
-> latest up-to-5000 RAW interactions
-> exact Initial-Pinyin filtering afterward
```

No claim in this document should be phrased as confirmed generalization or statistical significance unless a later holdout/statistical analysis supports it.

---

## 3. Evaluation layers remain distinct

### 3.1 End-to-end population

```text
N = 34,416
primary selection metric = Macro-author Top1
```

### 3.2 Generic Missing

```text
Generic Missing = 12,565
```

### 3.3 Recovery population

\[
R=\{Gold\notin GenericTop10 \land Gold\in PersonalK5\}
\]

```text
|R| = 4,910
```

K5 theoretical recoverability among Generic Missing:

\[
4910/12565=39.0768\%
\]

This is candidate-surface availability, not model accuracy.

### 3.4 Candidate-only scoring population

Historical candidate-scoring experiments use the non-trivial candidate-ranking population:

```text
Gold in Personal K5 AND K >= 2
n = 4,471
```

Candidate-only percentages must not be compared directly with 34,416-row end-to-end percentages.

---

## 4. Why Initial Pinyin is difficult

Generic Initial-Pinyin baseline:

| Macro | Micro | Top3 | Top5 | MRR | Missing |
|---:|---:|---:|---:|---:|---:|
| .307099 | .330573 | .491341 | .557996 | .426472 | .365092 |

Same-anchor Full-vs-Initial comparison:

| Condition | Generic Top1 | Missing@10 |
|---|---:|---:|
| Full Pinyin | .736024 | .069212 |
| Initial Pinyin | .330573 | .365092 |

Derived error-accounting decomposition:

```text
Observed Top1 gap = 40.5451 pp
Candidate-coverage gap = 29.5880 pp
Additional within-Top10 gap = 10.9571 pp
Coverage share of observed gap ≈ 72.98%
Full-covered -> Initial-missing = 10,223 / 34,416 = 29.704%
```

The 72.98% value is an accounting decomposition, not causal attribution.

The empirical system problem is therefore not only ranking ambiguity. A large fraction of Initial-Pinyin errors begin with the Gold candidate being absent from Generic Top10.

---

## 5. Historical baseline progression

| Model | Macro Top1 | Micro Top1 | Top3 | Top5 | MRR | Missing@10 |
|---|---:|---:|---:|---:|---:|---:|
| Generic G | .307099 | .330573 | .491341 | .557996 | .426472 | .365092 |
| Frequency F | .382495 | .408473 | .555120 | .601000 | .489432 | .365092 |
| PV1 | .401872 | .426749 | .598907 | .663093 | .524450 | .291144 |

F -> PV1 Top1 transition:

```text
rescue = 933
harm = 304
net = +629
```

PV1 recovery on R=4,910:

```text
Rec1=.1900
Rec3=.4923
Rec5=.5363
Rec10=.5401
RecMRR=.3363
```

Historical PV1 full-context control:

```text
PV1 + NGramRecency + BGERecency
Macro = .429506
Micro = .453423
Top3 = .612099
Top5 = .667335
MRR = .542766
Missing = .291144
Rec1 = .3582
Rec3 = .5165
Rec5 = .5381
Rec10 = .5401
RecMRR = .4356
```

---

## 6. Candidate-scoring conclusions retained from upstream development

Candidate-only population:

```text
Gold in Personal K5, K>=2, n=4,471
```

| Candidate scorer | Macro Top1 | Micro Top1 | Top3 | MRR | Approx. online latency |
|---|---:|---:|---:|---:|---:|
| Frequency | .491082 | .494968 | .847685 | .683822 | not fully recorded |
| BGE64 | .536167 | .538582 | .885484 | .718840 | 2.136 ms mean |
| Hard NGramRecency | .591558 | .594945 | .895996 | .750563 | ~0.084 ms |
| Interpolated NGram | .590783 | .594051 | .898009 | .750760 | ~0.084 ms |
| Q8 | .636531 | .642362 | .912995 | .783568 | 32.453 ms mean |
| **Q8+F** | **.669164** | **.675688** | **.925744** | **.804015** | ~Q8 |

Candidate-only accuracy winner remains Q8+F. Interpolated NGram remains the practical accuracy-latency choice used in Stage-1 recovery because of near-Hard-NGram accuracy, slightly better Top3/MRR, smooth backoff, transparency, and very low latency.

Important distinction:

```text
Stage-1 Interpolated NGram
!=
Stage-2 NGramRecency
```

Stage-1 uses smooth interpolated backoff inside personal recovery. Stage-2 uses hard suffix backoff plus recency to rerank the already-frozen final candidate set. They belong to the same lexical-context signal family and should not be described as independent sources of novelty.

---

## 7. Stage-1 recovery development

Three recovery philosophies are retained for the final Stage-2 comparison.

### 7.1 Coverage-first — K5+Entropy

```text
Score = boundary + 4*F_PV + .25*C_E
```

Overall:

```text
Macro=.403790
Micro=.428405
Top3=.602336
Top5=.677069
MRR=.533534
Missing=.243288
```

Recovery:

```text
Rec1=.2126
Rec3=.6126
Rec5=.8035
Rec10=.9876
RecMRR=.4552
```

This is the availability-first Stage-1 base.

### 7.2 Balanced — 4P+4CS+2E

```text
Score = boundary + 4*P_NG + 4*CS + 2*C_E
```

Overall:

```text
Macro=.404807
Micro=.429364
Top3=.614801
Top5=.685815
MRR=.537433
Missing=.243172
```

Recovery:

```text
Rec1=.2589
Rec3=.5493
Rec5=.7169
Rec10=.9485
RecMRR=.4542
```

This is the strongest Stage-1 overall balanced point.

### 7.3 Front-rank — 6P+2CS+.25E

```text
Score = boundary + 6*P_NG + 2*CS + .25*C_E
```

Overall:

```text
Macro=.400638
Micro=.423989
Top3=.615876
Top5=.686047
MRR=.534545
Missing=.243869
```

Recovery:

```text
Rec1=.2770
Rec3=.5737
Rec5=.7110
Rec10=.9283
RecMRR=.4679
```

This provides a more aggressive early-rank recovery alternative.

---

## 8. Stage-2 Context Reranking

Final Stage-2 score:

\[
S_{final}(c)=S_{REC}(c)+\lambda_NP_{NG-R}(c)+\lambda_BP_{BGE-R}(c)
\]

Candidate sets are fixed separately for each recovery base, so Stage-2 is pure reranking.

Consequences:

```text
Missing@10 must remain invariant
Rec10 on fixed R must remain invariant
Rec1/Rec3/Rec5/MRR may change
```

### NGramRecency

```text
HardBackoff maxN=2
tau_N=2048
candidate-conditioned
largest supported suffix order <=2
recency exp(-age/2048)
```

### BGERecency

```text
current context = last 64 chars
candidate-conditioned same-Pinyin causal history
historical Top-5 selected by cosine only
negative cosine clamped to zero
recency exp(-age/2048) applied in aggregation
tau_B=2048
```

---

## 9. V1 NGram-only result

All three Stage-1 bases selected:

```text
lambda_N = 6
```

| Base | Macro | Micro | Top3 | Top5 | MRR | Missing | Rec1 | Rec3 | Rec5 | Rec10 | RecMRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K5+Entropy + NG | .434673 | .457985 | .624390 | .686832 | .555949 | .243288 | .4128 | .7181 | .8558 | .9876 | .5960 |
| 4P+4CS+2E + NG | .432451 | .455718 | **.630434** | **.694183** | **.556494** | .243172 | .4224 | .6635 | .7853 | .9485 | .5775 |
| 6P+2CS+.25E + NG | .431802 | .454818 | .629184 | .693776 | .555338 | .243869 | **.4346** | .6646 | .7743 | .9283 | .5807 |

For the Balanced base, NGram alone adds:

```text
Delta Macro = +.027644
Top1 rescue = 2,375
Top1 harm = 1,468
Top1 net = +907
```

NGramRecency is therefore the dominant Stage-2 contextual mechanism.

---

## 10. V2/V3 full-context result and boundary closure

V2 jointly searched NGramRecency and BGERecency. K5 selected lambda_B=8 at the original upper boundary, so V3 expanded the BGE grid to 16 using the already-computed support.

V3 result:

```text
K5+Entropy       (lambda_N=6, lambda_B=8)
4P+4CS+2E        (lambda_N=4, lambda_B=6)
6P+2CS+.25E      (lambda_N=4, lambda_B=6)
```

The selected points are identical to V2, with zero Macro/MRR delta, and none hits the expanded `lambda_B=16` upper boundary.

Final full-context comparison:

| Base | lambda_N | lambda_B | Macro | Micro | Top3 | Top5 | MRR | Missing | Rec1 | Rec3 | Rec5 | Rec10 | RecMRR | Top1 net vs Stage1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K5+Entropy | 6 | 8 | .436767 | .459990 | .626453 | .688139 | .557836 | .243288 | **.4430** | **.7481** | **.8778** | **.9876** | **.6218** | +1087 |
| **4P+4CS+2E** | **4** | **6** | **.437058** | **.460571** | **.631392** | **.696478** | **.559755** | **.243172** | .4246 | .6969 | .8153 | .9485 | .5892 | +1074 |
| 6P+2CS+.25E | 4 | 6 | .436477 | .459786 | .630085 | .696013 | .558806 | .243869 | .4415 | .6961 | .8069 | .9283 | .5951 | **+1232** |

The current primary development selection is therefore Balanced full context because it maximizes the pre-specified Macro-author Top1 criterion and also has the highest Micro, Top3, Top5, and MRR among these three operating points.

However, the Macro gap to K5 is only .000291. This ordering should be treated as a Train-Val selection, not a claim of robust superiority.

---

## 11. Improvement over the historical PV1 full-context control

Historical control Macro:

```text
PV1 + NG-R + BGE-R = .429506
```

New full-context Macro deltas:

```text
K5+Entropy full   = .436767 -> +.007261
Balanced full     = .437058 -> +.007552
Front-rank full   = .436477 -> +.006971
```

These are development-set deltas only. No significance claim is made.

---

## 12. Diagnosis: why K5 wins Recovery but Balanced wins Overall

Full-context subset behavior:

```text
K5+Entropy:
Generic-covered Macro = .608057
Recovery-R Macro      = .442017

Balanced:
Generic-covered Macro = .615512
Recovery-R Macro      = .421864
```

The system therefore exposes a real trade-off:

\[
\boxed{\text{K5 wins recovery; Balanced wins overall preservation/ranking.}}
\]

The Generic-covered population is 21,851 rows, while R contains 4,910 rows. A small advantage on the much larger covered population can offset a substantially stronger recovery score on R.

Final direct Top1 agreement among the three systems:

```text
all same = 31,793 / 34,416 = 92.38%
any disagreement = 2,623 / 34,416 = 7.62%
```

Balanced vs K5 direct correctness disagreement:

```text
Balanced correct / K5 wrong = 473
K5 correct / Balanced wrong = 453
net = +20 rows for Balanced
```

This reinforces that the overall ranking difference is extremely small.

---

## 13. Diagnosis: NGram vs BGE contribution

Primary Balanced:

```text
Stage1                 Macro = .404807
+ NGramRecency         Macro = .432451
+ BGERecency           Macro = .437058
```

Incremental contributions:

```text
NGram DeltaMacro = +.027644
BGE DeltaMacro   = +.004607
```

Relative to the sum of these two incremental Macro gains, approximately 86% comes from NGram and 14% from BGE. This is a descriptive decomposition, not a causal estimate.

Top1 transition counts:

```text
Recovery -> NG:
rescue=2,375
harm=1,468
net=+907

NG -> Full:
rescue=681
harm=514
net=+167
```

On recoverable R:

```text
Balanced Recovery -> NG Top1 net = +803
Balanced NG -> Full Top1 net = +11
```

The main mechanism for converting recovered availability into correct Top1 placement is therefore NGramRecency. BGERecency is a smaller semantic refinement.

---

## 14. Top3 rescue/harm diagnosis

Top3 transition definition:

```text
rescue@3 = rank >3 / missing -> rank <=3
harm@3   = rank <=3 -> rank >3 / missing
```

### Overall Recovery -> Full

```text
K5       1,434 rescue - 604 harm = +830
Balanced 1,198 rescue - 627 harm = +571
Front    1,104 rescue - 615 harm = +489
```

### Recoverable R, Recovery -> Full

```text
K5       745 - 80 = +665
Balanced 745 - 20 = +725
Front    617 - 16 = +601
```

Thus K5 has the highest final Rec3, but Balanced obtains the largest **context-driven Top3 net gain on R**.

Balanced NGram-only transition on R:

```text
580 rescue - 19 harm = +561
```

Balanced BGE increment on R:

```text
190 rescue - 26 harm = +164
```

This gives strong diagnostic support to the two-stage interpretation: recovery exposes the Gold candidate and context moves it into a user-visible early rank.

---

## 15. BGE is complementary but cutoff-dependent

BGE increment on Generic-covered Top3:

```text
K5       183 rescue - 259 harm = -76
Balanced 119 rescue - 250 harm = -131
Front    129 rescue - 253 harm = -124
```

BGE increment on recoverable-R Top3:

```text
K5       201 - 54 = +147
Balanced 190 - 26 = +164
Front    181 - 26 = +155
```

For Balanced, BGE on Generic-covered Top1 is positive (`+156 net`) while Generic-covered Top3 transition is negative (`-131 net`). Therefore BGE cannot be described as uniformly improving preservation. Its effect depends on subset and cutoff.

Formal Conflict diagnosis also shows negative BGE Top1 net:

```text
K5       -38
Balanced -84
Front    -67
```

Conflict is Gold-derived and diagnosis-only. It must not be converted into a runtime gate under the current freeze.

---

## 16. Per-author heterogeneity

Primary Balanced progression:

| Author | Stage1 Top1 | +NG Top1 | Full Top1 | Full Top3 | Full Top5 | Full MRR | Full Missing |
|---|---:|---:|---:|---:|---:|---:|---:|
| Agent Phage | .470344 | .487374 | .493923 | .689542 | .764937 | .606609 | .171312 |
| Etinjat | .237235 | .271980 | .275218 | .388917 | .447198 | .347801 | .485056 |
| breaddddd | .506841 | .537999 | .542032 | .722183 | .780388 | .643439 | .167655 |

Stage1 -> Full Top1 gain:

```text
Agent Phage +.023579
Etinjat     +.037983
breaddddd   +.035192
```

Etinjat is substantially harder from the baseline onward. Its final Missing@10 is .485056, versus about .17 for the other two authors. Context nevertheless improves Etinjat substantially, so the data do not support the interpretation that context itself fails for Etinjat.

Train-Val row counts:

```text
Agent Phage = 13,741
Etinjat     = 8,030
breaddddd   = 12,645
```

Etinjat therefore also has the smallest Train-Val sample, but row count alone does not establish that fewer works or less history causes its lower accuracy. A causal explanation would require separate work-count/history-density/recoverability analysis.

---

## 17. Controllability: updated interpretation

The strongest controllability claim is no longer simply a choice among Stage-1 coefficients. The end-to-end experiments show distinct operating philosophies that survive contextual reranking.

### 17.1 Balanced overall mode

```text
4P+4CS+2E
+ NG-R lambda_N=4
+ BGE-R lambda_B=6
```

Purpose:

```text
best current overall Train-Val ranking under Macro-author Top1 selection
```

### 17.2 Coverage-oriented mode

```text
K5+Entropy
+ NG-R lambda_N=6
+ BGE-R lambda_B=8
```

Purpose:

```text
maximize recoverable-candidate availability and Recovery ranking
Rec10=.9876
RecMRR=.6218
```

### 17.3 Front-rank mode

```text
6P+2CS+.25E
+ NG-R lambda_N=4
+ BGE-R lambda_B=6
```

Purpose:

```text
more aggressive early-rank behavior
highest Stage1 Rec1 and highest full-context Top1 net movement (+1232)
```

These are interpretable operating points, not yet validated user-facing controls. The thesis can argue that the architecture exposes controllable trade-offs between recovery breadth, preservation, and early-rank aggressiveness, while avoiding the stronger claim that the current coefficient settings are final UX controls.

Historical NGramSelector K1/K3/K5 remains useful evidence that recovery breadth can also be controlled structurally through K. It is retained as prior architectural evidence, but it is not part of the current three-base Stage2 freeze set.

---

## 18. Current model set to carry to PRE-DEV / Dev3000

The compact frozen comparison set should be:

1. **Historical control:** PV1 + NGramRecency + BGERecency.
2. **Coverage:** K5+Entropy + NG(6) + BGE(8).
3. **Primary Balanced:** 4P+4CS+2E + NG(4) + BGE(6).
4. **Front-rank:** 6P+2CS+.25E + NG(4) + BGE(6).

Dev3000 should evaluate these frozen configurations without lambda search or feature redesign.

The principal Dev questions are:

```text
Does Balanced remain the overall winner?
Does K5 remain the recovery-oriented winner?
Do the Train-Val trade-offs persist outside the tuning surface?
```

Dev is not a new unrestricted development surface.

---

## 19. What is frozen before Dev3000

Freeze at minimum:

```text
- frozen Train-Fit / Train-Val identities and hashes
- frozen Personal-K5 candidate construction
- Stage-1 formulas and tie semantics
- Stage-2 NGramRecency definition
- Stage-2 BGERecency definition
- tau_N = 2048
- tau_B = 2048
- BGE Top-5 cosine-only historical retrieval
- current-query last-64-char BGE context
- selected lambda_N/lambda_B for each recovery base
- primary Macro-author Top1 criterion
- diagnostic subsets as analysis-only
- no Conflict gate
- no margin gate
- no further Train-Val lambda search
```

---

## 20. Current thesis-level conclusions

### Conclusion 1 — Initial Pinyin creates a major candidate-availability problem

Generic Missing@10 rises to .365092 and 10,223 same-anchor queries move from Full-covered to Initial-missing. Candidate recovery is therefore not optional if the system is expected to personalize Initial-Pinyin input effectively.

### Conclusion 2 — Recovery and ordering are different optimization problems

Personal K5 determines whether a missing Gold can become available. Stage-2 context determines whether that available Gold is moved into rank 1/3/5.

### Conclusion 3 — Maximum recovery does not maximize overall ranking

K5+Entropy dominates final Recovery metrics, but Balanced slightly wins overall Macro/Micro/Top3/Top5/MRR. The difference is explained by a trade-off between stronger recovery on R and better behavior on the much larger Generic-covered population.

### Conclusion 4 — NGramRecency is the dominant contextual mechanism

Balanced receives +.027644 Macro and +907 Top1 net from NGram, versus an additional +.004607 Macro and +167 Top1 net from BGE.

### Conclusion 5 — BGE provides complementary semantic value, not uniform improvement

BGE is positive overall and strongly useful for recovered Top3 placement, but it can be negative on Generic-covered Top3 and Conflict Top1. Its behavior is subset- and cutoff-dependent.

### Conclusion 6 — Context gains are not driven by one author only

All three Train-Val authors improve from Stage1 to full context. Etinjat remains much harder because of substantially worse candidate availability and ranking, not because contextual reranking provides no gain.

### Conclusion 7 — Controllability is supported as a trade-off structure

The experiments expose distinct balanced, coverage-first, and front-rank operating philosophies. This supports transparent personalisation controls over recovery breadth and ranking aggressiveness, while final UX semantics remain future work.

---

## 21. Interpretation boundaries

Do not overstate:

```text
- 39.0768% K5 recoverability is a candidate-surface ceiling, not model accuracy.
- 72.98% Full-vs-Initial decomposition is not causal attribution.
- Balanced's .000291 Macro advantage over K5 is not a significance claim.
- Conflict is Gold-derived and analysis-only.
- Margin is post-hoc and not a runtime gate.
- Etinjat's smaller row count does not prove that fewer works cause its lower performance.
- Stage-1 Interpolated NGram and Stage-2 NGramRecency are related signal families, not independent novelty.
- Dev3000 and Test have not yet validated generalization.
```

---

## 22. Current status

```text
TRAIN-VAL DEVELOPMENT: COMPLETE
STAGE-1 RECOVERY SELECTION: COMPLETE
V1 NGRAM-ONLY RERANKING: COMPLETE
V2 FULL CONTEXT: COMPLETE
V3 BOUNDARY VERIFICATION: COMPLETE
POST-HOC DIAGNOSIS: COMPLETE
TOP3 RESCUE/HARM DIAGNOSIS: COMPLETE
DEV3000: UNTOUCHED
TEST: UNTOUCHED
```

Next protocol step:

```text
PRE-DEV FREEZE
-> Dev3000
```

No further Train-Val feature addition, lambda search, Conflict gating, margin gating, or recovery redesign should occur unless the development protocol is explicitly reopened and documented.

---

## 23. Canonical current records

Latest standalone numerical record:

```text
18_INITIAL_RECOVERY_CONTEXT_TRAINVAL_FINAL_CONCLUSIONS_2026-08-21.md
SHA256 = 98a83076e9de8473e6ccfb99997d27431c3433c9a1d2ad6b3ede2b3336d5afdc
```

Latest reproducibility record:

```text
19_INITIAL_RECOVERY_CONTEXT_TRAINVAL_REPRODUCIBILITY_2026-08-21.md
```

Main result root:

```text
results\personalisation\initial_recovery_comparison_v1
```

Key latest result directories:

```text
recovery_ngram_context_fusion_v1\
recovery_bge_ngram_context_fusion_v2\
recovery_bge_ngram_context_fusion_v3\
recovery_context_diagnostics_v1\
recovery_context_topk_transitions_v1\
```

---

## 24. Concise thesis-ready summary

> The Initial-Pinyin experiments show that personalisation is fundamentally a two-stage problem. Initial-only input substantially degrades Generic candidate coverage, creating a bounded recovery opportunity in personal history. Stage-1 recovery determines which personal targets become available, while Stage-2 context determines how those available targets are ordered. Coverage-first K5+Entropy produces the strongest final recovery metrics, reaching Rec@10=.9876 and Recovery MRR=.6218, whereas the balanced 4P+4CS+2E recovery surface combined with NGramRecency and BGERecency achieves the best current overall Train-Val ranking under the pre-specified Macro-author Top1 criterion (Macro=.437058, Micro=.460571, Top3=.631392, Top5=.696478, MRR=.559755). NGramRecency supplies most of the contextual gain, while BGERecency adds a smaller semantic refinement whose effect varies by subset and cutoff. The small overall difference between the balanced and coverage-first systems, together with their contrasting recovery behavior, supports a transparent controllability interpretation based on recovery breadth, preservation, and early-rank aggressiveness rather than a single universally optimal personalisation policy.
