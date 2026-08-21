# Initial-Pinyin Personalisation: Evaluation Metrics and Design Rationale

## 1. Purpose

This document defines **how the Initial-Pinyin personalisation experiments are evaluated**, **why each metric is needed**, and **how the evaluation design maps to the model architecture**.

The central point is that Chinese IME personalisation is **not a single-label classification problem**. A practical IME exposes a ranked candidate list, and a personalisation method can improve one part of the system while damaging another. For that reason, the evaluation is intentionally decomposed into three layers:

1. **Candidate availability / recoverability** — is the Gold target available in the personal candidate pool at all?
2. **Personal candidate scoring** — if Gold is already in Personal K5, can the contextual scorer rank it near the top?
3. **End-to-end recovery and merge** — after personal candidates are injected into the Generic ranking, does the final Top10 improve without harmful overrides?

All current development results use the frozen Clean3 Train-Val protocol. Dev3000 and Test remain untouched.

---

## 2. Evaluation populations

The most important rule is that metrics from different populations must **not** be compared as if they were the same task.

### 2.1 End-to-end population

All final-system metrics are evaluated on the complete Train-Val set:

\[
N_{all}=34,416
\]

This population includes:

- queries already solved by Generic;
- queries where Generic contains Gold but ranks it poorly;
- Generic-Missing queries;
- queries with and without useful personal history.

This is the correct population for final system selection because an aggressive recovery method can rescue missing Gold while simultaneously damaging many already-correct Generic predictions.

---

### 2.2 Candidate-scoring population

Candidate scoring asks a narrower question:

> If Gold is already inside Personal K5, can the scorer identify it?

The common comparison population is:

\[
Gold \in PersonalK5, \qquad K \ge 2
\]

with:

\[
N_{score}=4,471
\]

The condition \(K\ge2\) is important because when \(K=1\), candidate selection is trivial: there is only one personal candidate, so no scorer is required to discriminate between alternatives.

Candidate-only metrics therefore measure **candidate discrimination**, not end-to-end IME quality.

---

### 2.3 Generic-Missing recoverable population

For recovery diagnostics we define:

\[
R=\{q: Gold\notin GenericTop10 \land Gold\in PersonalK5\}
\]

Current counts are:

\[
|GenericMissing|=12,565
\]

\[
|R|=4,910
\]

Therefore the current K5 recovery ceiling is:

\[
\frac{4910}{12565}=39.0768\%
\]

This is an **availability limit**. A better scorer cannot recover Generic-Missing rows whose Gold target is not present in Personal K5.

---

## 3. Rank representation

For each query \(q\), let:

\[
r_q \in \{1,2,\dots,10\}\cup\{\varnothing\}
\]

where:

- \(r_q=1\): Gold is the first candidate;
- \(r_q=3\): Gold is ranked third;
- \(r_q=\varnothing\): Gold is absent from the final Top10.

All end-to-end ranking metrics are derived from these ranks.

---

# 4. End-to-end metrics

## 4.1 Micro Top1

### Definition

\[
MicroTop1=\frac{1}{N}\sum_{q=1}^{N}\mathbf{1}[r_q=1]
\]

### Meaning

The percentage of all queries for which the first displayed candidate is correct.

### Why we use it

This is the most direct measure of first-choice user experience. If users usually accept the first candidate immediately, Micro Top1 closely approximates direct acceptance accuracy.

### Limitation

Authors with more rows contribute more weight. Therefore Micro Top1 alone can hide poor performance on smaller authors.

---

## 4.2 Macro-author Top1

For author \(a\):

\[
Top1_a=\frac{\#\{q\in a:r_q=1\}}{N_a}
\]

Then:

\[
MacroAuthorTop1=\frac{1}{A}\sum_{a=1}^{A}Top1_a
\]

where \(A\) is the number of authors represented in the evaluation population.

### Why we use it

Macro-author Top1 gives every author equal weight. This prevents one author with many rows from dominating model selection.

For the current Initial-Pinyin development protocol, **Macro-author Top1 remains the formal primary selection metric**.

### Why we do not use it alone

An IME is a ranked-list interface. A model may have slightly lower Top1 but substantially better Top3, Top5, or MRR. Therefore Macro Top1 determines formal selection, while the other rank metrics are required to understand the actual interaction trade-off.

---

## 4.3 Top3

### Definition

\[
Top3=\frac{1}{N}\sum_{q=1}^{N}\mathbf{1}[r_q\le3]
\]

Missing rows contribute zero.

### Meaning

The probability that Gold is visible among the first three candidates.

### Why Top3 is especially important for IME

IME users do not necessarily require the first candidate to be correct. If the desired candidate appears in positions 2 or 3, the interaction is still highly usable.

Therefore Top3 measures **candidate-list usefulness**, not only first-choice accuracy.

This is one of the most important descriptive metrics in the current project.

---

## 4.4 Top5

### Definition

\[
Top5=\frac{1}{N}\sum_{q=1}^{N}\mathbf{1}[r_q\le5]
\]

### Meaning

The probability that Gold is visible in a slightly wider shortlist.

### Why we use it

Top5 distinguishes systems that may have similar Top3 but different deeper-list coverage. It is useful for measuring whether personalisation improves the candidate list beyond only the first few positions.

### Important candidate-only note

On the **candidate-only K5 population**, Top5 is generally trivial because Gold is defined to already be inside Personal K5 and all K5 candidates are ranked. Therefore candidate-only evaluation should focus mainly on:

- Top1;
- Top3;
- MRR.

Top5 is much more informative for the final merged Top10.

---

## 4.5 MRR@10

### Definition

For each query:

\[
RR_q=
\begin{cases}
1/r_q, & r_q\in\{1,\dots,10\}\\
0, & r_q=\varnothing
\end{cases}
\]

Then:

\[
MRR@10=\frac{1}{N}\sum_q RR_q
\]

### Meaning

MRR rewards systems that place Gold near the top even when it is not rank 1.

For example:

- rank 1 contributes \(1\);
- rank 2 contributes \(0.5\);
- rank 3 contributes \(0.333\);
- missing contributes \(0\).

### Why we use it

Top3 only says whether Gold crossed a threshold. MRR distinguishes rank 1, rank 2, and rank 3 directly.

Therefore MRR is a strong measure of **overall ranked-list quality**.

### Candidate-only version

For Personal K5 scoring, the corresponding MRR is effectively MRR over the K5 ranking, because the candidate pool contains at most five candidates.

---

## 4.6 Missing@10

### Definition

\[
Missing@10=\frac{1}{N}\sum_q\mathbf{1}[r_q=\varnothing]
\]

Lower is better.

### Meaning

The fraction of queries where the correct target is completely absent from the final Top10.

### Why we use it

Top1/Top3 can improve while Gold coverage remains poor. Missing@10 directly measures whether the system gives the user any opportunity to select the correct target.

It is therefore the main **coverage metric** for the final IME list.

### Important limitation

A method can reduce Missing@10 simply by aggressively injecting many personal candidates. This can damage already-good Generic rankings. Missing must therefore always be interpreted together with Top1, MRR, and rescue/harm transitions.

---

## 4.7 Mean rank when present

Let:

\[
Q_{present}=\{q:r_q\neq\varnothing\}
\]

Then:

\[
MeanRank_{present}=\frac{1}{|Q_{present}|}\sum_{q\in Q_{present}}r_q
\]

Lower is better.

### Why we use it

This separates two effects:

1. whether Gold is present at all;
2. when present, how deep in the list it tends to appear.

Unlike MRR, missing rows are excluded from this diagnostic, so it should never replace MRR or Missing@10.

---

# 5. Transition metrics: rescue, harm, and net

End-to-end averages can hide what changed row by row. Therefore every important recovery method is compared against a baseline using Top1 transitions.

For baseline rank \(r_q^{base}\) and new rank \(r_q^{new}\):

## 5.1 Rescue

\[
Rescue=\#\{q:r_q^{base}\neq1 \land r_q^{new}=1\}
\]

A previously wrong first candidate becomes correct.

## 5.2 Harm

\[
Harm=\#\{q:r_q^{base}=1 \land r_q^{new}\neq1\}
\]

A previously correct first candidate becomes wrong.

## 5.3 Net

\[
Net=Rescue-Harm
\]

### Why we use these metrics

Personal recovery is inherently risky. Increasing personal injection strength often rescues missing targets but can override good Generic predictions.

A method with high recovery but high harm may be unsuitable even if Missing@10 improves.

The transition view therefore makes the **benefit–override trade-off** explicit.

---

# 6. Recovery-specific metrics

Recovery metrics are computed only on:

\[
R=\{Gold\notin GenericTop10 \land Gold\in PersonalK5\}
\]

with \(|R|=4,910\).

This isolates the rows that are both:

- genuinely missing from Generic;
- legally recoverable by the current Personal K5 candidate surface.

---

## 6.1 Recovered@k

For \(k\in\{1,3,5,10\}\):

\[
Recovered@k=\frac{1}{|R|}\sum_{q\in R}\mathbf{1}[r_q\le k]
\]

### Interpretation

- **Rec@1**: recovery becomes the first candidate;
- **Rec@3**: recovered Gold is visible in the first three candidates;
- **Rec@5**: recovered Gold is visible in the first five;
- **Rec@10**: recovered Gold enters the final list at all.

### Why Rec@3 matters

For an IME, merely entering Top10 may be insufficient. A recovered candidate at rank 9 is much less useful than one at rank 2 or 3.

Rec@3 therefore measures **useful recovery**, not only recovery coverage.

---

## 6.2 Recovery MRR

For each \(q\in R\):

\[
RR_q^{rec}=
\begin{cases}
1/r_q, & r_q\le10\\
0, & \text{not recovered into Top10}
\end{cases}
\]

Then:

\[
RecoveryMRR=\frac{1}{|R|}\sum_{q\in R}RR_q^{rec}
\]

### Why we use it

Rec@10 treats rank 1 and rank 10 equally as "recovered". Recovery MRR instead rewards high-quality recovery positions.

---

## 6.3 Mean recovered rank

Let:

\[
R_{recovered}=\{q\in R:r_q\le10\}
\]

Then:

\[
MeanRecoveredRank=\frac{1}{|R_{recovered}|}\sum_{q\in R_{recovered}}r_q
\]

Lower is better.

### Why we use it

This answers:

> When recovery succeeds, where does the recovered Gold usually appear?

It complements Rec@10 and Recovery MRR.

---

# 7. Candidate-scoring metrics

Candidate scoring is evaluated before Generic/Personal merging.

The scorer receives a Personal K5 candidate set and produces an ordering.

For the common population:

\[
Gold\in PersonalK5,\quad K\ge2,\quad N=4471
\]

we compute:

- Macro-author Top1;
- Micro Top1;
- Top3;
- MRR over the personal candidate ranking;
- online latency.

### Why this stage is separated from end-to-end recovery

A candidate scorer can be excellent at identifying the correct personal candidate but still perform poorly as an injection-strength model.

This is exactly what the experiments showed with direct NGram recovery: contextual NGram probabilities are strong for **candidate identity**, but using them directly as absolute recovery strength does not necessarily produce the best final ranking.

Therefore candidate discrimination and recovery calibration are evaluated separately.

---

# 8. Latency metrics

For online candidate scoring we report at least:

- mean latency;
- p95 latency.

### Mean latency

\[
MeanLatency=\frac{1}{N}\sum_q t_q
\]

### p95 latency

The 95th percentile of per-query online scoring time.

### Why both are needed

Mean latency measures average computational cost. p95 shows tail latency, which matters for interactive typing: occasional slow responses can still make an IME feel unresponsive.

Current candidate-scoring results illustrate why latency is necessary:

- Q8 is more accurate but much slower;
- Interpolated NGram is slightly less accurate but extremely fast.

Thus model quality must be considered as an **accuracy–latency trade-off**, not accuracy alone.

---

# 9. Why we do not select models on Generic-Missing rows only

It would be easy to maximize recovery by injecting personal candidates very aggressively.

For example, increasing a query-level personalisation boost can monotonically increase Rec@3 or Rec@10 while simultaneously:

- lowering Top1;
- lowering MRR;
- creating many harmful overrides.

Therefore formal model selection uses the complete 34,416-row Train-Val population rather than only Generic-Missing cases.

Recovery metrics are diagnostics explaining *how* a method changes the system, not the sole optimization objective.

---

# 10. Why Top1 is not enough for an IME

A conventional classification system often evaluates whether the single predicted label is correct. An IME exposes a ranked list.

Consider two systems:

- System A: Gold rank = 1, 1, 8, missing
- System B: Gold rank = 2, 2, 2, 2

Depending on the product objective:

- System A may have better Top1;
- System B may provide a much more reliable candidate list.

Therefore we deliberately keep several complementary metrics:

- **Top1** — immediate acceptance;
- **Top3** — short candidate-list usefulness;
- **Top5** — wider candidate availability;
- **MRR** — fine-grained rank quality;
- **Missing** — total list coverage.

No single metric fully describes IME quality.

---

# 11. How the metrics map to the model architecture

The current recovery architecture separates three questions.

## 11.1 Context relevance

Interpolated NGram provides:

\[
P_{NG}(c\mid context)
\]

Question answered:

> Which personal candidate best matches the current lexical context?

This component is mainly evaluated through **candidate Top1 / Top3 / MRR**.

---

## 11.2 Personal preference

Choice Share is:

\[
CS(c)=\frac{n_c}{N_{same\text{-}Pinyin}}
\]

where \(n_c\) is the causal same-Pinyin history count for candidate \(c\).

Question answered:

> For this Pinyin, how strongly does the user historically prefer this candidate?

This is candidate-specific and therefore can change the internal Personal ranking.

---

## 11.3 Query-level personal confidence

Let historical same-Pinyin target shares be \(p_1,\dots,p_M\).

Entropy is:

\[
H=-\sum_i p_i\log p_i
\]

Normalized entropy is:

\[
H_{norm}=\frac{H}{\log M}
\]

and entropy concentration is:

\[
C_E(q)=1-H_{norm}
\]

For a single distinct historical target, \(C_E=1\). With no history, concentration is treated as zero.

Question answered:

> Is this user's historical preference for the current Pinyin concentrated and stable enough to trust strongly?

Because \(C_E(q)\) is shared by all personal candidates for the same query, it does **not** alter Personal candidate ordering. It moves the entire Personal block relative to Generic.

---

# 12. Current Context–Preference–Confidence model

The current strongest balanced development model is:

\[
\boxed{
Score(c)=B+4P_{NG}(c)+4CS(c)+2C_E(q)
}
\]

where:

- \(B\): Generic boundary score;
- \(P_{NG}(c)\): contextual candidate relevance;
- \(CS(c)\): candidate-specific personal preference;
- \(C_E(q)\): query-level personal confidence.

This decomposition is intentionally interpretable:

```text
Context relevance     -> P_NG(c)
Personal preference   -> CS(c)
Personal confidence   -> C_E(q)
                         |
                         v
                 Personal recovery score
                         |
                         v
              merge with Generic-F ranking
```

The three components correspond to three different evaluation concerns:

| Component | Main question | Most relevant metrics |
|---|---|---|
| Personal K5 availability | Can Gold be recovered at all? | recoverable rate, Missing@10 |
| NGram contextual scoring | Can we identify the right personal candidate? | candidate Top1, Top3, MRR, latency |
| CS + Entropy recovery strength | How strongly should Personal compete with Generic? | Overall Top1/Top3/MRR, Rec@3, rescue/harm |

---

# 13. Why Entropy must be evaluated together with harm

Increasing \(C_E\)'s weight can increase recovery strength.

Empirically, this creates a characteristic pattern:

```text
small/moderate entropy weight
    -> useful recovery increases
    -> Top1 / Top3 / MRR may improve

very large entropy weight
    -> Rec@3 / Rec@10 continue increasing
    -> Personal candidates become too aggressive
    -> harmful Generic overrides increase
    -> overall Top1 / MRR decline
```

This is why Rec@3 alone cannot select the final model.

The desired operating point is a **recovery-benefit / override-cost balance**.

---

# 14. Formal selection versus product-oriented interpretation

The current protocol keeps two ideas separate.

## Formal development selection

Primary criterion:

\[
\boxed{Macro\text{-}author\ Top1}
\]

Reason: equal author weighting and protection against over-optimizing a subset.

## Product-oriented interpretation

For an actual IME candidate list, we also place strong emphasis on:

- Top3;
- Top5;
- MRR;
- Missing@10;
- Rec@3;
- online latency.

Therefore it is valid for two operating points to serve different goals:

- a **balanced model** maximizing overall accuracy/ranking quality;
- a **Top3-oriented model** maximizing short candidate-list visibility.

These should be described as different operating points rather than claiming that one metric universally defines the best IME.

---

# 15. Current headline interpretation

Based on the completed Train-Val experiments:

- **Q8+F** is the strongest candidate scorer by candidate-only Top1/Top3/MRR, but is much slower.
- **Interpolated NGram** gives the strongest practical accuracy–latency trade-off among tested contextual scorers.
- **K5+Entropy** is extremely strong in pure recovery coverage, but stronger recovery does not automatically improve the overall ranking.
- **6P_NG + 2CS + 0.25Entropy** is the current Top3-oriented end-to-end operating point.
- **4P_NG + 4CS + 2Entropy** is the current strongest balanced end-to-end development model across Macro Top1, Micro Top1, MRR, Top5 and coverage while remaining very close to the best Top3.

These are **development findings**, not final generalization claims. Dev3000 and Test have not been used for these selections.

---

# 16. Reporting checklist for future models

Every future end-to-end recovery method should report the same evaluation package.

## Overall — all 34,416 Train-Val rows

- Macro-author Top1
- Micro Top1
- Top3
- Top5
- MRR@10
- Missing@10
- Mean rank when present

## Transition analysis

- rescue
- harm
- net
- unchanged correct
- unchanged wrong

## Recoverable Generic-Missing subset — R = 4,910

- Rec@1
- Rec@3
- Rec@5
- Rec@10
- Recovery MRR
- Mean recovered rank

## Candidate scorer evaluation

- Macro Top1
- Micro Top1
- Top3
- MRR
- Mean latency
- p95 latency

## Fairness / robustness diagnostics

- per-author metrics
- concentration-bin diagnostics where applicable

## Provenance

- Gold not used for candidate construction or online scoring
- same-author strictly-prior causal history
- H5000 applied before exact-Pinyin filtering
- Dev3000 untouched during Train-Val development
- Test untouched until final evaluation

---

# 17. Short thesis-ready summary

> We evaluate Initial-Pinyin personalisation as a ranked candidate-generation and recovery problem rather than a single-label classification task. Final systems are evaluated on all Train-Val queries using Macro-author Top1, Micro Top1, Top3, Top5, MRR@10, and Missing@10. Macro-author Top1 is retained as the formal development-selection metric to weight authors equally, while Top3 and Top5 capture practical IME candidate-list usability and MRR measures fine-grained ranking quality. Recovery is evaluated separately on Generic-Missing queries whose Gold target is legally available in Personal K5, using Recovered@1/3/5/10, Recovery MRR, and mean recovered rank. Rescue/harm transitions are reported because aggressive personal candidate injection can improve recovery coverage while overriding already-correct Generic predictions. Candidate-scoring quality is evaluated on Gold-in-Personal-K5 queries with at least two candidates, separating contextual candidate discrimination from final recovery calibration. This layered evaluation directly mirrors the final interpretable architecture: contextual NGram evidence estimates candidate relevance, Choice Share represents candidate-specific personal preference, and entropy concentration represents query-level confidence in the user's historical preference distribution.

---

## 18. Main local result artifacts

The current evaluation is grounded in the following local experiment families:

```text
results/personalisation/initial_recovery_comparison_v1/

candidate_scoring_q8_bge64_v1/
candidate_scoring_ngram_recency_v1/
candidate_scoring_adaptive_ngram_top10_v1/
candidate_scoring_ngram_frequency_fusion_v1/

recovery_ngram_cs_concentration_k5_v1/
recovery_ngram_cs_interpolation_k5_v1/
pv1_ngram_selector_k135_v1/
pv1_ngram_k5_additive_concentration_v1/
pv1_ngram_k5_joint_frequency_concentration_v1/
ngram_cs_entropy_two_anchors_v1/

all_results_summary_v1/
```
