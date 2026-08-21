# Initial-Pinyin Personalisation: Metric Purpose and Priority Guide

## 1. Why this document exists

The Initial-Pinyin personalisation system is not a single classification model. It contains several distinct stages:

```text
Generic candidate generation
        ↓
Generic-side frequency reranking
        ↓
Personal candidate availability
        ↓
Personal candidate scoring / selection
        ↓
Recovery strength / merge
        ↓
Final Top10 candidate list
```

Because these stages solve different problems, no single metric is sufficient.

For example:

- Top1 measures whether the first candidate is correct;
- Top3 measures whether the correct candidate appears near the top of the visible IME list;
- Missing@10 measures whether the correct candidate is absent entirely;
- Recovery@3 measures whether a Generic-missing but recoverable target has been restored near the top;
- latency measures whether a scorer is practical for interactive typing.

The key principle is therefore:

> **Use different metrics for different questions, and do not treat all metrics as interchangeable.**

---

# 2. Metric hierarchy

For the current Initial-Pinyin experiments, metrics can be grouped into four levels.

## Level A — Primary model-selection metric

### Macro-author Top1

This remains the primary model-selection metric during development.

For each author \(a\):

\[
Top1_a = \frac{\#\{q : rank_q = 1\}}{N_a}
\]

Then:

\[
MacroTop1 = \frac{1}{A}\sum_{a=1}^{A} Top1_a
\]

where \(A\) is the number of authors.

### Purpose

Macro-author Top1 answers:

> **If every user is weighted equally, how often is the first candidate correct?**

This is important because authors may contribute different numbers of rows. A simple pooled Top1 could otherwise be dominated by the largest author.

### Priority

\[
\boxed{\textbf{Primary metric}}
\]

Use it to decide the main development operating point unless the evaluation objective has been explicitly changed in advance.

---

# 3. Key IME ranking metrics

## 3.1 Micro Top1

\[
MicroTop1 = \frac{\#\{q : rank_q = 1\}}{N}
\]

### Purpose

Answers:

> **Across all queries, how often is the first candidate correct?**

Unlike Macro Top1, this weights every query equally rather than every author equally.

### Interpretation

- useful for aggregate product-level accuracy;
- should be reported beside Macro Top1;
- should not replace Macro Top1 when author balance matters.

### Priority

\[
\boxed{\text{Key secondary metric}}
\]

---

## 3.2 Top3

\[
Top3 = \frac{\#\{q : rank_q \le 3\}}{N}
\]

### Purpose

Top3 answers:

> **How often is the correct answer visible within the first three IME candidates?**

This is especially important for an input method because the task is not identical to ordinary single-label classification. A user can still choose the correct candidate when it is ranked second or third.

### Why Top3 is critical for this thesis

Top3 captures candidate-list quality more directly than Top1.

Two systems can have similar Top1 but very different user experience:

```text
System A: Gold rank distribution = mostly 1 or missing
System B: Gold rank distribution = often 1, 2, or 3
```

Top1 may make them look similar, while Top3 reveals that System B gives the user a much more useful shortlist.

### Priority

\[
\boxed{\textbf{Critical metric}}
\]

For IME ranking, Top3 should be treated as one of the main headline metrics alongside Macro Top1 and MRR.

---

## 3.3 Top5

\[
Top5 = \frac{\#\{q : rank_q \le 5\}}{N}
\]

### Purpose

Answers:

> **How often is the correct candidate still present in a slightly wider visible shortlist?**

Top5 is useful for understanding whether a method improves broad candidate-list coverage even when the candidate is not in the top three.

### Interpretation

Top5 is especially useful when comparing:

- aggressive recovery methods;
- K1 vs K3 vs K5 injection;
- methods that reduce Missing but may not improve Top1.

### Priority

\[
\boxed{\text{Important secondary metric}}
\]

It is less central than Top3 because the first few IME positions matter more, but it helps diagnose ranking-depth effects.

---

# 4. MRR@10 — overall ranking quality

For each query:

\[
RR_q =
\begin{cases}
1/rank_q, & rank_q \le 10\\
0, & \text{otherwise}
\end{cases}
\]

Then:

\[
MRR@10 = \frac{1}{N}\sum_q RR_q
\]

### Purpose

MRR answers:

> **On average, how close to the top is the correct candidate?**

Unlike Top3, MRR distinguishes between rank 1, 2, and 3.

For example:

```text
Gold at rank 1 → contribution 1.000
Gold at rank 2 → contribution 0.500
Gold at rank 3 → contribution 0.333
Gold at rank 5 → contribution 0.200
Missing      → contribution 0
```

### Why MRR is important

A system may have the same Top3 as another system but systematically place Gold at rank 2 instead of rank 3. Top3 cannot see this difference; MRR can.

### Priority

\[
\boxed{\textbf{Critical metric}}
\]

For this thesis, MRR should be treated as the main rank-quality metric together with Top3.

---

# 5. Missing@10 — coverage failure

\[
Missing@10 = \frac{\#\{q : Gold \notin FinalTop10_q\}}{N}
\]

Lower is better.

### Purpose

Answers:

> **How often is the correct answer completely unavailable in the final candidate list?**

This is fundamentally different from Top1 or Top3.

If Gold is rank 7:

- Top1 = failure;
- Top3 = failure;
- Top5 = failure;
- Missing@10 = success, because the target is still present.

If Gold is absent:

- the user cannot select it at all from the candidate list.

### Why it matters for personal vocabulary recovery

The purpose of Personal Vocabulary Recovery is partly to solve cases where Generic PinyinGPT does not include the target at all.

Therefore Missing@10 measures whether the recovery system improves candidate availability.

### Priority

\[
\boxed{\textbf{Critical coverage metric}}
\]

However, Missing must never be optimized alone because aggressive personal injection can reduce Missing while damaging already-correct Generic rankings.

---

# 6. Mean rank when present

For queries where Gold appears in the final Top10:

\[
MeanRank = \frac{1}{|S|}\sum_{q\in S} rank_q
\]

where \(S\) contains only queries where Gold is present.

### Purpose

Answers:

> **Conditional on the target being available, how deep in the list does it usually appear?**

### Priority

\[
\boxed{\text{Diagnostic metric}}
\]

Useful for interpretation, but not appropriate as the primary selection criterion because it ignores missing cases.

---

# 7. Rescue / Harm / Net — personalization safety

For two methods, baseline \(B\) and new method \(M\):

## Rescue

\[
Rescue = \#\{q : B\text{ Top1 wrong},\ M\text{ Top1 correct}\}
\]

## Harm

\[
Harm = \#\{q : B\text{ Top1 correct},\ M\text{ Top1 wrong}\}
\]

## Net

\[
Net = Rescue - Harm
\]

### Purpose

Answers:

> **How many mistakes did personalisation fix, and how many previously-correct predictions did it break?**

This is essential because recovery methods can become too aggressive.

A model could recover many missing personal candidates while simultaneously pushing correct Generic candidates down.

### Example interpretation

```text
rescue = 120
harm   = 40
net    = +80
```

This indicates a healthy personalization trade-off.

But:

```text
rescue = 500
harm   = 650
net    = -150
```

means the method is recovering aggressively but doing more overall damage than benefit at Top1.

### Priority

\[
\boxed{\textbf{Critical diagnostic metric}}
\]

It should always be reported for recovery methods even though it is not itself the main optimization objective.

---

# 8. Recoverability metrics

Recovery metrics use a special population:

\[
R = \{q : Gold \notin GenericTop10 \land Gold \in PersonalK5\}
\]

In the current frozen K5 setup:

\[
|R| = 4910
\]

This population answers a different question from the full 34,416-row end-to-end evaluation.

---

## 8.1 Recovery@1

\[
Rec@1 = \frac{\#\{q\in R : rank_q=1\}}{|R|}
\]

### Purpose

Measures how often a previously missing but theoretically recoverable personal target is restored directly to rank 1.

### Priority

\[
\boxed{\text{Useful but not sufficient}}
\]

Very strict; it does not capture useful rank-2 or rank-3 recovery.

---

## 8.2 Recovery@3

\[
Rec@3 = \frac{\#\{q\in R : rank_q\le3\}}{|R|}
\]

### Purpose

Answers:

> **Among Generic-missing cases that K5 could theoretically recover, how often does the method place Gold in the first three final candidates?**

This directly measures useful personal vocabulary recovery near the top of the IME list.

### Priority

\[
\boxed{\textbf{Critical recovery metric}}
\]

For recovery quality, Rec@3 is more informative than Rec@1 alone.

---

## 8.3 Recovery@5

\[
Rec@5 = \frac{\#\{q\in R : rank_q\le5\}}{|R|}
\]

### Purpose

Measures medium-depth recovery.

### Priority

\[
\boxed{\text{Important secondary recovery metric}}
\]

---

## 8.4 Recovery@10

\[
Rec@10 = \frac{\#\{q\in R : rank_q\le10\}}{|R|}
\]

### Purpose

Answers:

> **How much of the theoretically recoverable K5 opportunity did the final system actually recover somewhere into Top10?**

### Why it matters

A high Rec@10 but modest Rec@3 indicates:

> the method is good at candidate availability but not necessarily good at placing the candidate near the top.

This is exactly the type of behaviour seen when increasing recovery breadth or using aggressive injection.

### Priority

\[
\boxed{\textbf{Critical recovery-coverage metric}}
\]

---

# 9. Recovery MRR

On the recoverable population \(R\):

\[
RecoveryMRR = \frac{1}{|R|}\sum_{q\in R}
\begin{cases}
1/rank_q, & rank_q\le10\\
0, & \text{otherwise}
\end{cases}
\]

### Purpose

Measures both:

- whether recovery succeeded;
- how high the recovered candidate appears.

### Priority

\[
\boxed{\textbf{Critical recovery ranking metric}}
\]

It is especially useful when comparing two methods with similar Rec@10 but different recovered ranks.

---

# 10. Mean recovered rank

For successfully recovered rows only:

\[
MeanRecoveredRank = \frac{1}{|R'|}\sum_{q\in R'}rank_q
\]

where \(R'\subseteq R\) contains recoverable rows that actually enter the final Top10.

### Purpose

Answers:

> **When recovery succeeds, how deep does the recovered Gold usually appear?**

### Priority

\[
\boxed{\text{Diagnostic recovery metric}}
\]

Useful for understanding placement quality, but should not be optimized alone because unsuccessful recoveries are excluded.

---

# 11. Candidate-scoring metrics

Candidate-scoring experiments use a different evaluation population:

\[
Gold\in PersonalK5,\quad K\ge2
\]

Current common population:

\[
n=4471
\]

The question is:

> **Assuming the correct candidate is already available in Personal K5, can the scorer rank it correctly?**

This isolates candidate discrimination from candidate availability and final recovery strength.

Key candidate-scoring metrics are:

- Macro Top1;
- Micro Top1;
- Top3;
- MRR;
- latency.

---

# 12. Candidate-scoring Top3

### Purpose

Measures whether a candidate scorer can place the correct personal candidate among its top three choices.

This is especially relevant when the scorer is later used to order or select multiple personal candidates rather than only a single winner.

### Priority

\[
\boxed{\textbf{Critical candidate-scoring metric}}
\]

For example, a scorer with slightly lower Top1 but higher Top3 may be preferable for a K3 recovery architecture.

---

# 13. Candidate-scoring MRR

### Purpose

Measures overall ordering quality inside the Personal candidate pool.

This becomes important when the downstream recovery system can use more than one candidate.

### Priority

\[
\boxed{\textbf{Critical candidate-ranking metric}}
\]

---

# 14. Latency

For interactive IME use, latency should be reported with at least:

\[
MeanLatency
\]

and

\[
P95Latency
\]

### Mean latency

Represents typical runtime cost.

### P95 latency

Represents tail responsiveness:

> **How slow is the system for the slower 5% of queries?**

This matters because typing systems are highly latency-sensitive.

### Priority

\[
\boxed{\textbf{Critical deployment metric for candidate scorers}}
\]

A scorer that gains only a few percentage points but is hundreds of times slower may not be the preferred production choice.

---

# 15. Which metrics are actually key?

## 15.1 End-to-end model selection

### Tier 1 — Key headline metrics

\[
\boxed{
MacroTop1,
Top3,
MRR@10,
Missing@10
}
\]

These four metrics together answer the four main questions:

| Metric | Main question |
|---|---|
| Macro Top1 | Is the first candidate correct consistently across users? |
| Top3 | Is Gold visible in the first few IME candidates? |
| MRR | How high is Gold ranked overall? |
| Missing@10 | Is Gold available at all? |

### Tier 2 — Important supporting metrics

\[
\boxed{MicroTop1,\ Top5}
\]

These improve interpretation but usually should not independently determine the final model.

### Tier 3 — Diagnostics

\[
\boxed{MeanRank,\ rescue,\ harm,\ net}
\]

These explain *why* performance changed.

---

# 16. Recovery-specific priorities

For Personal Vocabulary Recovery, the most important recovery metrics are:

\[
\boxed{
Rec@3,
Rec@10,
RecoveryMRR
}
\]

with:

- Rec@3 = useful near-top recovery;
- Rec@10 = recovery coverage;
- Recovery MRR = balance between success and rank quality.

Also always report:

\[
\boxed{rescue,
 harm,
 net}
\]

because a recovery system that improves Rec@10 while damaging many correct Generic predictions is not necessarily better overall.

---

# 17. Candidate-scoring priorities

For Personal-K5 candidate scoring, the key metrics are:

\[
\boxed{
MacroTop1,
Top3,
MRR,
Latency
}
\]

These answer:

```text
Macro Top1 → who is the best first choice?
Top3       → is Gold among the best few candidates?
MRR        → how good is the entire personal-candidate ordering?
Latency    → can we afford this scorer interactively?
```

---

# 18. Why we do not optimize only one metric

## Only Top1 is insufficient

A model may lose a very small amount of Top1 but dramatically improve Top3, MRR, or Missing.

For an IME, this can still represent better user experience.

## Only Top3 is insufficient

A very aggressive recovery method can push many personal candidates into the top three while harming rank-1 correctness.

## Only Missing is insufficient

A method can reduce Missing simply by injecting more personal candidates, but this may crowd out useful Generic candidates.

## Only Recovery@10 is insufficient

A system may recover almost every theoretically recoverable Gold somewhere in Top10 but place them too low to be useful.

## Only latency is insufficient

A very fast scorer with poor ranking quality is not useful either.

Therefore the evaluation is intentionally multi-dimensional.

---

# 19. Recommended interpretation framework

When reading a new result, use this order.

## Step 1 — Did primary accuracy improve?

Check:

\[
MacroTop1
\]

## Step 2 — Did visible candidate-list quality improve?

Check:

\[
Top3,
Top5,
MRR
\]

## Step 3 — Did coverage improve?

Check:

\[
Missing@10
\]

## Step 4 — Was the improvement caused by useful recovery or aggressive override?

Check:

\[
rescue,
 harm,
 net
\]

## Step 5 — On genuinely recoverable Generic-missing cases, what happened?

Check:

\[
Rec@3,
Rec@10,
RecoveryMRR
\]

## Step 6 — If this is a candidate scorer, is it practical?

Check:

\[
mean\ latency,
P95\ latency
\]

---

# 20. Metric-to-system-component mapping

The current personalisation architecture separates three functions:

\[
P_{NG}(c)
\rightarrow
\text{Context relevance}
\]

\[
CS(c)
\rightarrow
\text{Personal preference}
\]

\[
C_E(q)
\rightarrow
\text{Personal confidence}
\]

Different metrics diagnose different parts of this design.

| System component | Most relevant metrics |
|---|---|
| Candidate availability | Missing@10, recoverable rate |
| Candidate identity / ordering | Candidate Top1, Candidate Top3, Candidate MRR |
| Recovery breadth | Rec@5, Rec@10, Missing@10 |
| Near-top recovery | Rec@1, Rec@3, Recovery MRR |
| Final user-facing ranking | Macro Top1, Top3, Top5, MRR |
| Harmful override control | rescue, harm, net |
| Runtime practicality | mean latency, P95 latency |

---

# 21. Recommended thesis reporting order

For every final end-to-end model, report:

```text
Macro-author Top1
Micro Top1
Top3
Top5
MRR@10
Missing@10
```

Then report recovery diagnostics:

```text
rescue
harm
net
Rec@1
Rec@3
Rec@5
Rec@10
Recovery MRR
Mean recovered rank
```

For candidate scorers, report separately:

```text
Macro Top1
Micro Top1
Top3
MRR
Mean latency
P95 latency
```

Do not combine candidate-only and end-to-end metrics in one numerical ranking because they use different evaluation populations.

---

# 22. Short thesis-ready summary

> **The evaluation is deliberately multi-dimensional because personalised Pinyin input involves candidate availability, candidate ranking, recovery strength, and final user-facing list quality. Macro-author Top1 is retained as the primary development selection metric to give equal weight to each user, while Top3 and MRR are treated as critical IME ranking metrics because the correct candidate remains useful when it appears near the top rather than strictly at rank one. Missing@10 measures candidate availability, whereas Recovery@3, Recovery@10, and Recovery MRR isolate the effectiveness of Personal Vocabulary Recovery on Generic-missing but theoretically recoverable cases. Rescue/harm transitions are reported to distinguish useful recovery from harmful overrides. Candidate-scoring models are evaluated separately on the Gold-in-Personal-K5 population using Top1, Top3, MRR, and online latency, since candidate discrimination and final recovery are distinct problems.**

---

# 23. Final priority summary

## Main end-to-end metrics

1. **Macro-author Top1** — primary selection objective.
2. **Top3** — critical IME shortlist quality.
3. **MRR@10** — critical overall rank quality.
4. **Missing@10** — critical coverage metric.

## Main recovery metrics

1. **Recovery@3** — useful near-top recovery.
2. **Recovery@10** — recovery coverage.
3. **Recovery MRR** — quality + coverage together.
4. **Rescue / Harm / Net** — safety and trade-off diagnosis.

## Main candidate-scoring metrics

1. **Macro Top1** — best candidate identity.
2. **Top3** — quality of the top personal shortlist.
3. **MRR** — full personal-candidate ordering quality.
4. **Mean / P95 latency** — interactive deployability.

## Supporting metrics

- Micro Top1;
- Top5;
- Mean rank;
- Mean recovered rank.

These are important for interpretation, but they should normally support rather than replace the primary metrics above.
