# Initial-Pinyin Personalisation: Current Conclusions, Model Comparison, and Controllability

**Status:** Train-Val development / exploratory model selection
**Evaluation surface:** 34,416 standardized Clean3 Train-Val rows
**Personal candidate pool:** frozen Personal K5
**Dev3000 used:** No
**Test used:** No
**Gold used for candidate construction/scoring:** No
**Gold used for Train-Val evaluation/model-selection diagnostics:** Yes

---

## 1. Current high-level conclusion

The current results do **not** support the idea that one model is best for every purpose.

Different models dominate different layers of the system:

1. **Candidate availability / recoverability ceiling** is determined by the Personal K5 surface.
2. **Candidate scoring** asks which candidate inside Personal K5 is correct.
3. **Recovery** asks whether a Generic-missing but K5-available Gold can be inserted high enough into the final list.
4. **End-to-end ranking** asks whether the whole IME candidate list improves across all queries.
5. **Controllability** concerns how the system should expose the trade-off between first-choice accuracy, candidate-list quality, and personal recovery aggressiveness.

The current strongest overall development operating point is:

\[
\boxed{
S(c)=B+4P_{NG}(c)+4CS(c)+2C_E(q)
}
\]

where:

- \(B\) = Generic/Frequency boundary score,
- \(P_{NG}(c)\) = Interpolated NGram contextual support,
- \(CS(c)\) = candidate Choice Share,
- \(C_E(q)\) = query-level entropy concentration.

This model is referred to below as:

\[
\boxed{\textbf{4P + 4CS + 2E}}
\]

It currently gives the best **Macro Top1**, **Micro Top1**, and **MRR@10**, while remaining very close to the best Top3 and Top5 and achieving low Missing@10.

However, **NGramSelector@K3 remains an important alternative architecture**, especially for transparent and conservative recovery control.

---

# 2. Evaluation layers must remain separate

## 2.1 Candidate availability / recoverability ceiling

Generic Missing count:

\[
12,565
\]

Generic-missing rows whose Gold exists in frozen Personal K5:

\[
4,910
\]

Therefore:

\[
\boxed{
\text{K5 theoretical recoverability}
=
\frac{4910}{12565}
=
39.08\%
}
\]

This is a hard candidate-surface limit.

No scorer can recover a Generic-missing Gold that is not present in Personal K5.

---

## 2.2 Candidate-scoring population

Candidate scoring is evaluated only when ranking is non-trivial:

\[
\boxed{
Gold\in PersonalK5,\quad K\ge2
}
\]

Population:

\[
\boxed{n=4,471}
\]

This population must **not** be numerically compared directly with the full 34,416-row end-to-end evaluation.

---

## 2.3 Recovery population

Recovery evaluation uses:

\[
R=
\{
q:
Gold\notin GenericTop10
\land
Gold\in PersonalK5
\}
\]

with:

\[
\boxed{|R|=4,910}
\]

This population answers:

> When recovery is theoretically possible, how often does the method actually place Gold into the final Top1 / Top3 / Top5 / Top10?

---

## 2.4 End-to-end population

Final system ranking is evaluated on:

\[
\boxed{n=34,416}
\]

all standardized Train-Val queries.

This is the population used for primary model selection.

---

# 3. Candidate scoring conclusions

## 3.1 Candidate-only scorer comparison

Population:

\[
Gold\in PersonalK5,\quad K\ge2,\quad n=4471
\]

| Candidate scorer | Macro Top1 | Micro Top1 | Top3 | MRR | Online latency |
|---|---:|---:|---:|---:|---:|
| Frequency | .491082 | .494968 | .847685 | .683822 | not fully recorded |
| BGE64 | .536167 | .538582 | .885484 | .718840 | 2.136 ms mean |
| Hard NGramRecency | .591558 | .594945 | .895996 | .750563 | ~0.084 ms |
| Interpolated NGram | .590783 | .594051 | .898009 | .750760 | ~0.084 ms |
| Q8 | .636531 | .642362 | .912995 | .783568 | 32.453 ms mean |
| **Q8 + F** | **.669164** | **.675688** | **.925744** | **.804015** | ~Q8 |

### Candidate-scoring conclusion

Accuracy winner:

\[
\boxed{\textbf{Q8 + F}}
\]

Candidate Top3:

\[
\boxed{92.57\%}
\]

Practical accuracy-latency winner:

\[
\boxed{\textbf{Interpolated NGram}}
\]

because it reaches:

\[
Top3=89.80\%
\]

at roughly:

\[
0.084\text{ ms}
\]

instead of approximately 32.45 ms for Q8.

---

# 4. Why Interpolated NGram is used downstream

Hard and Interpolated NGram are extremely close in Top1.

### Hard NGramRecency

\[
MacroTop1=.591558
\]

\[
Top3=.895996
\]

\[
MRR=.750563
\]

### Interpolated NGram

\[
MacroTop1=.590783
\]

\[
Top3=.898009
\]

\[
MRR=.750760
\]

Hard therefore has a very small Top1 advantage, but Interpolated has slightly better Top3 and MRR.

More importantly, Interpolated NGram uses smooth evidence-dependent backoff.

For suffix order \(n\):

\[
\lambda_n=\frac{m_n}{m_n+\kappa}
\]

and:

\[
P_n
=
\lambda_n\hat P_n
+
(1-\lambda_n)P_{n-1}
\]

Therefore sparse high-order context evidence does not completely replace lower-order evidence.

This is especially suitable for personal history, where exact contextual matches are often sparse.

The downstream choice is therefore:

\[
\boxed{\textbf{Interpolated NGram}}
\]

for its combination of:

- near-identical Top1,
- slightly stronger Top3,
- slightly stronger MRR,
- smoother evidence-dependent backoff,
- transparent behaviour,
- extremely low latency.

---

# 5. End-to-end baseline progression

Population:

\[
n=34,416
\]

| Model | Macro Top1 | Micro Top1 | Top3 | Top5 | MRR | Missing@10 |
|---|---:|---:|---:|---:|---:|---:|
| Generic G | .307099 | .330573 | .491341 | .557996 | .426472 | .365092 |
| Frequency F | .382495 | .408473 | .555120 | .601000 | .489432 | .365092 |
| EM1-R | .310531 | .334292 | .500349 | — | .434164 | .345857 |
| EM1-R+F | .394722 | .421199 | .592719 | — | .517784 | .300616 |
| PV1 | .401872 | .426749 | .598907 | .663093 | .524450 | .291144 |

This progression established:

- Generic PinyinGPT alone has substantial Missing@10.
- Generic-side frequency reranking helps Top1 strongly.
- PV1 provides substantial additional recovery.
- Recovery must be calibrated because aggressive insertion can improve coverage while harming already-correct Generic rankings.

---

# 6. NGram recovery and NGram-CS interpolation

## 6.1 Pure NGram recovery

\[
S(c)=B+\lambda P_{NG}(c)
\]

Selected \(\lambda=4\).

| Macro | Micro | Top3 | Top5 | MRR | Missing |
|---:|---:|---:|---:|---:|---:|
| .395803 | .421577 | .602075 | .674105 | .528567 | .249274 |

Recovery:

| Rec@1 | Rec@3 | Rec@5 | Rec@10 | Recovery MRR |
|---:|---:|---:|---:|---:|
| .1157 | .3876 | .5737 | .8725 | .3143 |

Interpretation:

> NGram is strong for candidate identity/order, but its contextual probability is not by itself an ideal absolute injection-strength calibration.

---

## 6.2 NGram × Choice Share

The earlier multiplicative NGram × CS family did not become the main formulation.

Selected result:

| Macro | Micro | Top3 | Top5 | MRR | Missing |
|---:|---:|---:|---:|---:|---:|
| .399119 | .424134 | .600360 | .662744 | .527534 | .260083 |

Recovery:

| Rec@1 | Rec@3 | Rec@5 | Rec@10 | Recovery MRR |
|---:|---:|---:|---:|---:|
| .2405 | .3727 | .4701 | .7764 | .3580 |

---

## 6.3 NGram-CS interpolation

Define:

\[
D_\alpha(c)
=
(1-\alpha)P_{NG}(c)
+
\alpha CS(c)
\]

then:

\[
S(c)=B+\lambda_DD_\alpha(c)
\]

The two most important operating points are:

### \(\alpha=.25,\lambda_D=8\)

Equivalent to:

\[
\boxed{B+6P_{NG}+2CS}
\]

| Macro | Micro | Top3 | Top5 | MRR | Missing |
|---:|---:|---:|---:|---:|---:|
| .400212 | .423553 | **.615731** | **.685669** | .534314 | .244392 |

Recovery:

| Rec@1 | Rec@3 | Rec@5 | Rec@10 | Recovery MRR |
|---:|---:|---:|---:|---:|
| .2664 | .5692 | .7057 | .9230 | .4603 |

This is strongly candidate-list oriented.

---

### \(\alpha=.5,\lambda_D=8\)

Equivalent to:

\[
\boxed{B+4P_{NG}+4CS}
\]

| Macro | Micro | Top3 | Top5 | MRR | Missing |
|---:|---:|---:|---:|---:|---:|
| **.402628** | **.427272** | .612593 | .683461 | **.535609** | .245613 |

Recovery:

| Rec@1 | Rec@3 | Rec@5 | Rec@10 | Recovery MRR |
|---:|---:|---:|---:|---:|
| .2169 | .5037 | .6660 | .9106 | .4118 |

This became the balanced NGram-CS anchor.

---

# 7. NGramSelector remains an important architecture

The Selector family uses a different decomposition:

\[
\boxed{\text{NGram decides WHO}}
\]

while:

\[
\boxed{F_{PV}\text{ decides HOW STRONGLY}}
\]

Pipeline:

```text
Personal K5
    ↓
Interpolated NGram contextual ranking
    ↓
keep top K selected personal candidates
    ↓
original PV1 frequency support
    ↓
boundary injection
    ↓
merge with Generic F
```

This is architecturally different from using NGram probability directly inside the injection score.

---

## 7.1 K1 / K3 / K5 selector comparison

| Selector | Macro | Micro | Top3 | Top5 | MRR | Missing | Rec@1 | Rec@3 | Rec@5 | Rec@10 | PV1 net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K1 | .403678 | .428318 | .608409 | .674744 | .530086 | .278272 | .1959 | .5393 | .6130 | .6303 | +54 |
| **K3** | **.403964** | **.428522** | .604544 | **.681079** | **.534120** | .247007 | **.2016** | **.6051** | .7923 | .9051 | **+61** |
| K5 | .403772 | .428376 | .603266 | .677708 | .533908 | **.243172** | .2014 | .6026 | **.7986** | **.9857** | +56 |

### Interpretation

#### K1

Conservative:

\[
Rec@10=.6303
\]

It limits personal injection heavily.

#### K3

Balanced:

\[
Macro=.403964
\]

\[
Rec@3=.6051
\]

\[
Rec@10=.9051
\]

It achieved the best Top1 and net transition among the selector K-ablation.

#### K5

Coverage-oriented:

\[
Rec@10=.9857
\]

but slightly weaker ranking than K3.

### Current conclusion about Selector

\[
\boxed{\textbf{NGramSelector@K3 should remain in the thesis comparison}}
\]

It is especially valuable because:

- it is simple and transparent;
- contextual scoring and injection calibration are separated;
- Rec@3 is strong;
- harmful override is limited;
- \(K\) itself is a natural controllability parameter.

---

# 8. Entropy as query-level confidence

Define:

\[
C_E(q)=1-H_{norm}(q)
\]

where \(H_{norm}\) is normalized entropy over the full legal same-Pinyin historical target distribution.

Interpretation:

- \(C_E\approx1\): history is concentrated / stable;
- \(C_E\approx0\): history is diffuse / uncertain.

For a fixed query, \(C_E(q)\) is shared by all personal candidates.

Therefore it does **not** directly determine candidate identity.

Instead, it controls the strength of the whole personal block relative to Generic.

This gives a clean decomposition:

\[
\boxed{P_{NG}\rightarrow Context}
\]

\[
\boxed{CS\rightarrow Preference}
\]

\[
\boxed{C_E\rightarrow Confidence}
\]

---

# 9. K5 + Entropy: useful negative result

Earlier K5 additive concentration:

\[
S=B+4F_{PV}(c)+\lambda_E C_E(q)
\]

gave only a negligible end-to-end improvement.

Best Entropy point:

\[
\lambda_E=.25
\]

| Model | Macro | Micro | Top3 | Top5 | MRR | Missing | Rec@3 | Rec@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K5 | .403772 | .428376 | .603266 | .677708 | .533908 | .243172 | .6026 | .9857 |
| K5 + Entropy | .403790 | .428405 | .602336 | .677069 | .533534 | .243288 | **.6126** | **.9876** |

Interpretation:

> Entropy can increase recovery aggressiveness, but frequency-only candidate-specific scoring is not strong enough for that extra recovery to improve overall ranking.

This result became important later because Entropy **does** become useful once candidate-specific Context + Preference scoring is stronger.

---

# 10. New Context–Preference–Confidence experiments

General formulation:

\[
\boxed{
S(c)
=
B+
\beta P_{NG}(c)
+
\gamma CS(c)
+
\lambda_E C_E(q)
}
\]

Two frozen NGram-CS anchors were tested.

---

# 11. Top3-oriented anchor

Fix:

\[
\beta=6,\quad\gamma=2
\]

so:

\[
\boxed{
S=B+6P_{NG}+2CS+\lambda_EC_E
}
\]

Full grid:

| \(\lambda_E\) | Macro | Micro | Top3 | Top5 | MRR | Missing | Rec@3 | Rec@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .400212 | .423553 | .615731 | .685669 | .534314 | .244392 | .5692 | .9230 |
| **.25** | .400638 | .423989 | **.615876** | .686047 | .534545 | .243869 | .5737 | .9283 |
| .5 | **.400701** | .424047 | .615789 | .686338 | **.534577** | .243608 | .5790 | .9328 |
| 1 | .399749 | .422972 | .615760 | **.686715** | .534055 | .243026 | .5876 | .9409 |
| 2 | .391700 | .414400 | .614627 | .686425 | .529343 | **.242620** | .6049 | .9540 |
| 4 | .371792 | .392201 | .610762 | .682996 | .516460 | .243201 | **.6407** | .9692 |

---

## 11.1 Top3-oriented operating point

\[
\boxed{
B+6P_{NG}+2CS+0.25C_E
}
\]

gives:

\[
\boxed{Top3=.615876}
\]

This is the current highest final end-to-end Top3.

Its recovery:

\[
Rec@3=.5737
\]

\[
Rec@10=.9283
\]

Therefore it should be described as:

\[
\boxed{\textbf{Top3-oriented moderate-recovery operating point}}
\]

It is **not** the maximum-recovery point.

---

## 11.2 Top5-oriented operating point

\[
\boxed{
B+6P_{NG}+2CS+1C_E
}
\]

gives:

\[
\boxed{Top5=.686715}
\]

the current highest observed final Top5.

---

## 11.3 Coverage-oriented aggressive point

\[
\boxed{
B+6P_{NG}+2CS+2C_E
}
\]

gives:

\[
Missing=.242620
\]

the lowest observed Missing@10 in the current comparison.

Recovery:

\[
Rec@3=.6049
\]

\[
Rec@10=.9540
\]

However:

\[
MacroTop1=.391700
\]

which is substantially worse than the balanced models.

Thus this is a useful:

\[
\boxed{\textbf{coverage-oriented aggressive operating point}}
\]

not a recommended main model.

---

## 11.4 Maximum Rec@3 aggressive point

\[
\boxed{
B+6P_{NG}+2CS+4C_E
}
\]

gives:

\[
\boxed{Rec@3=.6407}
\]

the highest observed Rec@3 in this family.

But:

\[
MacroTop1=.371792
\]

\[
MRR=.516460
\]

Therefore:

\[
\boxed{\text{maximum recovery is not maximum end-to-end quality}}
\]

This is a key negative/control result demonstrating harmful override from excessive personalisation.

---

# 12. BalancedAnchor + Entropy

Fix:

\[
\beta=4,\quad\gamma=4
\]

so:

\[
\boxed{
S=B+4P_{NG}+4CS+\lambda_EC_E
}
\]

Full grid:

| \(\lambda_E\) | Macro | Micro | Top3 | Top5 | MRR | Missing | Rec@3 | Rec@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .402628 | .427272 | .612593 | .683461 | .535609 | .245613 | .5037 | .9106 |
| .25 | .402742 | .427388 | .613029 | .683752 | .535797 | .245060 | .5094 | .9165 |
| .5 | .402756 | .427359 | .613610 | .684420 | .535995 | .244305 | .5163 | .9236 |
| 1 | .403584 | .428231 | .614278 | .684798 | .536724 | .243666 | .5285 | .9328 |
| **2** | **.404807** | **.429364** | **.614801** | **.685815** | **.537433** | **.243172** | .5493 | .9485 |
| 4 | .401933 | .425848 | .612506 | .684304 | .534646 | .243520 | **.5910** | **.9652** |

---

# 13. Current overall development-best

The strongest balanced point is:

\[
\boxed{
S(c)=
B+
4P_{NG}(c)
+
4CS(c)
+
2C_E(q)
}
\]

Results:

| Metric | Result |
|---|---:|
| **Macro Top1** | **.404807** |
| **Micro Top1** | **.429364** |
| Top3 | .614801 |
| Top5 | .685815 |
| **MRR@10** | **.537433** |
| Missing@10 | .243172 |
| Rec@3 | .5493 |
| Rec@10 | .9485 |
| PV1 net | +90 |

This currently gives:

- highest Macro Top1;
- highest Micro Top1;
- highest MRR;
- Top3 within ~0.108 percentage points of the Top3-specific winner;
- Top5 within ~0.090 percentage points of the Top5-specific winner;
- very low Missing.

Therefore:

\[
\boxed{\textbf{4P + 4CS + 2E is the current overall development operating point}}
\]

It should still be described as **exploratory development-best**, not confirmed generalization.

Dev3000 and Test remain untouched.

---

# 14. Main end-to-end horizontal comparison

Population:

\[
n=34,416
\]

| Model | Macro Top1 ↑ | Micro Top1 ↑ | Top3 ↑ | Top5 ↑ | MRR ↑ | Missing ↓ | Rec@3 ↑ | Rec@10 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Generic G | .307099 | .330573 | .491341 | .557996 | .426472 | .365092 | — | — |
| Frequency F | .382495 | .408473 | .555120 | .601000 | .489432 | .365092 | — | — |
| PV1 | .401872 | .426749 | .598907 | .663093 | .524450 | .291144 | .4923 | .5401 |
| Pure NGram | .395803 | .421577 | .602075 | .674105 | .528567 | .249274 | .3876 | .8725 |
| NGram × CS | .399119 | .424134 | .600360 | .662744 | .527534 | .260083 | .3727 | .7764 |
| 6P + 2CS | .400212 | .423553 | .615731 | .685669 | .534314 | .244392 | .5692 | .9230 |
| 4P + 4CS | .402628 | .427272 | .612593 | .683461 | .535609 | .245613 | .5037 | .9106 |
| Selector K1 | .403678 | .428318 | .608409 | .674744 | .530086 | .278272 | .5393 | .6303 |
| **Selector K3** | .403964 | .428522 | .604544 | .681079 | .534120 | .247007 | **.6051** | .9051 |
| Selector K5 | .403772 | .428376 | .603266 | .677708 | .533908 | .243172 | .6026 | .9857 |
| K5 + Entropy | .403790 | .428405 | .602336 | .677069 | .533534 | .243288 | .6126 | **.9876** |
| **6P + 2CS + .25E** | .400638 | .423989 | **.615876** | .686047 | .534545 | .243869 | .5737 | .9283 |
| 6P + 2CS + .5E | .400701 | .424047 | .615789 | .686338 | .534577 | .243608 | .5790 | .9328 |
| **6P + 2CS + 1E** | .399749 | .422972 | .615760 | **.686715** | .534055 | .243026 | .5876 | .9409 |
| **6P + 2CS + 2E** | .391700 | .414400 | .614627 | .686425 | .529343 | **.242620** | .6049 | .9540 |
| **6P + 2CS + 4E** | .371792 | .392201 | .610762 | .682996 | .516460 | .243201 | **.6407** | .9692 |
| 4P + 4CS + .5E | .402756 | .427359 | .613610 | .684420 | .535995 | .244305 | .5163 | .9236 |
| 4P + 4CS + 1E | .403584 | .428231 | .614278 | .684798 | .536724 | .243666 | .5285 | .9328 |
| **4P + 4CS + 2E** | **.404807** | **.429364** | .614801 | .685815 | **.537433** | .243172 | .5493 | .9485 |
| 4P + 4CS + 4E | .401933 | .425848 | .612506 | .684304 | .534646 | .243520 | .5910 | .9652 |

---

# 15. Winners by metric purpose

| Purpose | Current winner | Result |
|---|---|---:|
| Candidate-scoring Macro Top1 | Q8 + F | .669164 |
| Candidate-scoring Top3 | Q8 + F | **.925744** |
| Candidate-scoring efficiency | Interpolated NGram | Top3 .898009 at ~.084 ms |
| **Overall Macro Top1** | **4P + 4CS + 2E** | **.404807** |
| **Overall Micro Top1** | **4P + 4CS + 2E** | **.429364** |
| **Overall Top3** | **6P + 2CS + .25E** | **.615876** |
| **Overall Top5** | **6P + 2CS + 1E** | **.686715** |
| **Overall MRR** | **4P + 4CS + 2E** | **.537433** |
| Lowest Missing@10 | 6P + 2CS + 2E | **.242620** |
| Highest Rec@3 | 6P + 2CS + 4E | **.6407** |
| Highest Rec@10 among listed main methods | K5 + Entropy | **.9876** |
| Best selector balance | NGramSelector@K3 | Macro .403964, Rec@3 .6051 |

---

# 16. Why NGramSelector@K3 should still be retained

Although:

\[
4P+4CS+2E
\]

currently has stronger overall ranking metrics, the difference in Macro Top1 versus Selector K3 is small:

\[
.404807-.403964=.000843
\]

or:

\[
\boxed{0.0843\text{ percentage points}}
\]

Meanwhile Selector K3 has:

\[
Rec@3=.6051
\]

compared with:

\[
Rec@3=.5493
\]

for 4P+4CS+2E.

This makes Selector K3 scientifically valuable.

It represents:

\[
\boxed{\textbf{Select first, then calibrate}}
\]

rather than:

\[
\boxed{\textbf{Joint transparent scoring}}
\]

The two architectures are:

### Architecture A — NGramSelector

```text
NGram
  ↓
WHO should enter?
  ↓
PV1 frequency support
  ↓
HOW STRONGLY should it enter?
```

### Architecture B — Context–Preference–Confidence

```text
NGram → Context relevance
CS    → Personal preference
E     → Query-level confidence
           ↓
      joint score
```

Both should remain in the final thesis comparison until holdout evaluation.

---

# 17. Controllability: the strongest emerging thesis implication

The results support a richer notion of controllability than simply:

```text
Personalisation ON / OFF
```

Two distinct control dimensions emerge.

---

## 17.1 Ranking-preference control

Different users or products may value different candidate-list objectives.

### First-choice / balanced mode

Use:

\[
\boxed{4P+4CS+2E}
\]

because it maximizes:

- Macro Top1,
- Micro Top1,
- MRR,

while keeping Top3 and Top5 close to their maxima.

Possible product interpretation:

> **First-choice accuracy / Balanced mode**

---

### Top3-oriented mode

Use:

\[
\boxed{6P+2CS+.25E}
\]

because:

\[
\boxed{Top3=.615876}
\]

Possible product interpretation:

> **Candidate-list / Top-3 mode**

This sacrifices some first-choice accuracy to maximize the chance that Gold is visible among the first few candidates.

---

### Top5-oriented mode

Use:

\[
\boxed{6P+2CS+1E}
\]

because:

\[
\boxed{Top5=.686715}
\]

Possible product interpretation:

> **Broader shortlist mode**

---

## 17.2 Personalisation-strength / recovery control

For a fixed candidate-specific model, increasing Entropy weight increases personal recovery aggressiveness.

Example:

\[
6P+2CS+\lambda_EE
\]

| \(\lambda_E\) | Macro | Top3 | Missing | Rec@3 | Rec@10 |
|---:|---:|---:|---:|---:|---:|
| 0 | .400212 | .615731 | .244392 | .5692 | .9230 |
| .25 | .400638 | **.615876** | .243869 | .5737 | .9283 |
| .5 | **.400701** | .615789 | .243608 | .5790 | .9328 |
| 1 | .399749 | .615760 | .243026 | .5876 | .9409 |
| 2 | .391700 | .614627 | **.242620** | .6049 | .9540 |
| 4 | .371792 | .610762 | .243201 | **.6407** | .9692 |

The pattern is:

\[
\boxed{
\lambda_E\uparrow
\Rightarrow
Recovery\uparrow
}
\]

but after a point:

\[
\boxed{
\lambda_E\uparrow
\Rightarrow
Harmful\ override\uparrow
\Rightarrow
Top1/MRR\downarrow
}
\]

Therefore \(\lambda_E\) is a natural candidate for a user-visible or system-level:

\[
\boxed{\textbf{Personalisation Strength}}
\]

control.

---

# 18. Controllability through Selector K

NGramSelector provides an even more discrete and interpretable control.

| K | Interpretation | Macro | Rec@3 | Rec@10 |
|---:|---|---:|---:|---:|
| 1 | Conservative | .403678 | .5393 | .6303 |
| **3** | **Balanced** | **.403964** | **.6051** | .9051 |
| 5 | High coverage | .403772 | .6026 | **.9857** |

This gives a very intuitive control:

```text
K1  → Conservative
K3  → Balanced
K5  → High-Coverage
```

Therefore \(K\) itself can be interpreted as:

\[
\boxed{\textbf{recovery breadth}}
\]

or:

\[
\boxed{\textbf{how many personal candidates are allowed to compete}}
\]

This is highly relevant to the thesis goal of **user-controllable personalisation**.

---

# 19. Proposed controllability framework

The current results suggest:

\[
\boxed{
\text{Controllability}
=
\text{Ranking Preference}
+
\text{Personalisation Strength}
}
\]

## Ranking Preference

Controls whether the user/system prefers:

- strongest Top1,
- strongest Top3,
- strongest Top5 / shortlist quality.

Potentially related to:

\[
\beta:\gamma
\]

in:

\[
\beta P_{NG}+\gamma CS
\]

---

## Personalisation Strength

Controls how aggressively personal candidates are allowed to override Generic candidates.

Potentially represented by:

\[
\lambda_E
\]

in the joint-scoring family, or:

\[
K
\]

in the Selector family.

---

# 20. Important caution about coefficient interpretation

Increasing the coefficient on:

\[
P_{NG}
\]

does **not** automatically mean greater recoverability.

A larger \(P_{NG}\) coefficient means:

> place more relative trust in current contextual relevance.

This primarily changes candidate-specific ranking preference.

By contrast:

\[
\lambda_E
\]

is much closer to a recovery-strength control because \(C_E(q)\) is a shared query-level boost for the whole personal block.

Therefore:

- \(\beta/\gamma\) should be interpreted mainly as **Context vs Preference weighting**;
- \(\lambda_E\) should be interpreted mainly as **Personalisation / recovery strength**;
- \(K\) should be interpreted mainly as **Recovery breadth**.

---

# 21. Current thesis-level conclusions

## Conclusion 1 — Candidate identity and injection strength are different problems

NGram is especially useful for determining:

\[
\boxed{\text{WHO is contextually appropriate}}
\]

while historical frequency / Choice Share / confidence are useful for determining:

\[
\boxed{\text{HOW strongly personal evidence should affect Generic ranking}}
\]

---

## Conclusion 2 — Candidate scoring accuracy alone is not enough

Q8+F is the strongest candidate scorer, but it is much slower than Interpolated NGram.

Therefore:

\[
\boxed{
\text{best scorer}
\neq
\text{best practical scorer}
}
\]

---

## Conclusion 3 — Recoverability is not end-to-end quality

Aggressive models can produce excellent:

\[
Rec@3,\ Rec@10,\ Missing
\]

while harming:

\[
Top1,\ MRR
\]

For example:

\[
6P+2CS+4E
\]

achieves:

\[
Rec@3=.6407
\]

but only:

\[
MacroTop1=.371792
\]

Thus:

\[
\boxed{
\text{more recovery}
\neq
\text{better IME}
}
\]

---

## Conclusion 4 — Context, Preference, and Confidence are complementary

The strongest current balanced model combines:

\[
\boxed{P_{NG}}
\]

for Context,

\[
\boxed{CS}
\]

for Personal Preference,

and:

\[
\boxed{C_E}
\]

for Confidence.

This gives the interpretable decomposition:

\[
\boxed{
\text{Context}
+
\text{Preference}
+
\text{Confidence}
}
\]

---

## Conclusion 5 — Entropy is useful only when the candidate-specific score is good enough

Entropy added to frequency-only K5 produced essentially no meaningful end-to-end gain.

Entropy added to balanced NGram + CS produced a clear gain.

Therefore the evidence supports:

> Query-level confidence is useful when it modulates a sufficiently strong candidate-specific personal scoring function.

---

## Conclusion 6 — NGramSelector@K3 remains scientifically important

Selector K3 remains competitive because it:

- separates candidate selection from injection calibration;
- achieves high Rec@3;
- limits harmful override;
- gives a highly interpretable K-based control;
- is only 0.0843 pp below the current overall Macro Top1 development-best.

It should remain a main comparison model until untouched holdout evaluation.

---

## Conclusion 7 — Controllability is supported empirically

The experiments expose meaningful operating points rather than a single universally optimal configuration.

Examples:

### Balanced / first-choice oriented

\[
\boxed{4P+4CS+2E}
\]

### Top3 oriented

\[
\boxed{6P+2CS+.25E}
\]

### Top5 oriented

\[
\boxed{6P+2CS+1E}
\]

### Coverage-oriented aggressive

\[
\boxed{6P+2CS+2E}
\]

### Maximum-recovery / diagnostic aggressive

\[
\boxed{6P+2CS+4E}
\]

### Selector conservative / balanced / high-coverage

\[
K=1,\quad3,\quad5
\]

This supports a thesis argument that personalisation can be made **transparent and controllable**, not merely switched on or off.

---

# 22. Recommended main model set for future holdout evaluation

Do not carry every Train-Val ablation forward as a primary model.

A compact, scientifically useful set is:

1. **PV1** — historical recovery baseline.
2. **NGramSelector@K3** — selective / calibrated architecture.
3. **4P + 4CS + 2E** — current overall balanced development-best.
4. **6P + 2CS + .25E** — Top3-oriented operating point.
5. Optionally **K5 / aggressive point** as a coverage/recovery diagnostic rather than a main winner.

This preserves both:

- architecture comparison;
- controllability comparison.

---

# 23. Current status and interpretation boundary

All conclusions in this document are based on Train-Val development experiments.

Therefore the appropriate wording is:

\[
\boxed{\textbf{current exploratory development result}}
\]

or:

\[
\boxed{\textbf{development-best operating point}}
\]

Do **not** claim:

- confirmed generalization,
- final test superiority,
- statistical significance,

until appropriate holdout evaluation and significance testing are completed.

Dev3000 remains untouched.

Test remains untouched.

---

# 24. Concise thesis-ready summary

> The Initial-Pinyin experiments reveal that personalisation quality is governed by several distinct decisions: whether the correct personal candidate is available, which personal candidate is contextually appropriate, how strongly that candidate should compete with the Generic list, and how aggressively the system should prioritise personal recovery. Q8+F achieved the strongest candidate-only ranking accuracy, while Interpolated NGram provided a substantially better accuracy–latency trade-off. At the end-to-end level, the strongest current balanced formulation combines contextual NGram support, candidate-specific Choice Share, and query-level entropy concentration, yielding a transparent Context–Preference–Confidence decomposition. The current development-best balanced point is \(B+4P_{NG}+4CS+2C_E\), whereas \(B+6P_{NG}+2CS+0.25C_E\) maximises final Top-3 accuracy. More aggressive entropy weighting further increases recovery but eventually harms Top1 and MRR, demonstrating a clear recovery–precision trade-off. The NGramSelector@K3 architecture remains an important alternative because it separates contextual candidate selection from frequency-based injection calibration and provides a natural K-based control over recovery breadth. Together, these results motivate controllable personalisation through both ranking preference and personalisation-strength settings rather than a single fixed personalisation policy.
