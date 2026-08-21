# Initial Personalisation — Evidence & Background Lift Deferred Study

**Date:** 2026-08-21
**Status:** Deferred diagnostic / future stability study
**Scope:** Initial-Pinyin personalisation; current 3-author Train-Val diagnostics; future 6-author + richer-history / Full-Pinyin comparison
**Protocol:** No Dev3000 / No Test / No parameter selection from sealed evaluation data

---

## 1. Purpose

This note records findings about **Evidence / Exposure / Concentration** and **Background Lift / Personal Distinctiveness** that should **not yet be treated as final model conclusions**.

The current diagnostics use only a small cross-user background (3 authors total, so each leave-one-user-out background contains only 2 other authors) and Initial Pinyin, where one abbreviated key such as `sy` can merge many distinct Full-Pinyin lexical competition sets.

Therefore, two questions are intentionally deferred until larger and richer data are available:

1. How should **Evidence Count / Exposure Rate** interact with concentration when estimating preference reliability?
2. Does **Background Lift / Personal Distinctiveness** become more stable and useful with more authors, richer history, and Full Pinyin?

---

## 2. Definitions to keep separate

### 2.1 Exposure Rate

\[
ExposureRate_u(p)=\frac{N_u(p)}{|H_t|}
\]

Interpretation: how common the current Pinyin/Initial key is in the user's visible recent causal history.

This measures **prevalence**, not directly preference reliability.

### 2.2 Evidence Count

\[
EvidenceCount_u(p)=N_u(p)
\]

Interpretation: how many same-key historical observations support the estimated personal distribution.

This is useful for uncertainty / smoothing, but current diagnostics show that count alone is not a sufficient reliability measure.

### 2.3 Choice Share

\[
ChoiceShare_u(c\mid p)=P_u(c\mid p)=\frac{N_u(c,p)}{N_u(p)}
\]

Interpretation: when this user typed this key, how often did they choose candidate `c`?

### 2.4 Distribution Concentration

Two current diagnostics:

\[
Margin_u(p)=P_u(c_{(1)}\mid p)-P_u(c_{(2)}\mid p)
\]

and normalized entropy:

\[
H_u^{norm}(p)=\frac{-\sum_c P_u(c\mid p)\log P_u(c\mid p)}{\log |C_p|}
\]

High Margin / low Entropy indicate a more concentrated historical choice distribution.

### 2.5 Background Lift

Current raw diagnostic:

\[
Lift_u(c,p)=\log\frac{P_u(c\mid p)+\epsilon}{P_{bg,-u}(c\mid p)+\epsilon}
\]

Interpretation: how much more strongly the current user prefers candidate `c` than the available leave-one-user-out background users.

**Important:** Lift measures **personal distinctiveness**, not current contextual correctness.

---

## 3. Current canonical diagnostic data

Dataset: Initial Train-Val, 34,416 rows.

### Conflict semantics

- Formal Conflict: **15,353**
- Legacy B2 winner-mismatch subset: **16,192**
- Legacy-only extra rows: **839**

F → PV1 total transition accounting:

- Rescue: **933**
- Harm: **304**
- Net: **+629**

Formal Conflict:

- Rescue: **0**
- Harm: **111**
- Net: **−111**

Ambiguous non-Conflict:

- Rescue: **575**
- Harm: **105**
- Net: **+470**

Non-ambiguous winner-mismatch:

- Rescue: **0**
- Harm: **88**
- Net: **−88**

---

## 4. Evidence × Concentration: current finding

The strongest current observation is that **low Evidence Count is not automatically weak when the historical choice distribution is highly concentrated**.

### Evidence Count = 2

High concentration:

\[
Margin\ge0.75:\quad 54\ rescue,\ 14\ harm,\ net=+40
\]

Low concentration:

\[
Margin<0.10:\quad 34\ rescue,\ 42\ harm,\ net=-8
\]

### Evidence Count = 3–4

High concentration:

\[
Margin\ge0.75:\quad 65\ rescue,\ 6\ harm,\ net=+59
\]

Low concentration:

\[
Margin<0.10:\quad 26\ rescue,\ 33\ harm,\ net=-7
\]

### Evidence Count = 5–9

High concentration:

\[
Margin\ge0.75:\quad 41\ rescue,\ 8\ harm,\ net=+33
\]

Low concentration:

\[
Margin<0.10:\quad 23\ rescue,\ 25\ harm,\ net=-2
\]

Entropy shows the same qualitative pattern: low-evidence, low-entropy histories are safer than low-evidence, high-entropy histories.

### Current interpretation

Do **not** use a rule such as:

\[
EvidenceCount<k\Rightarrow \text{ignore personalisation}
\]

A better working hypothesis is:

\[
Reliability=f(EvidenceCount,\ Concentration)
\]

This is especially relevant to rare but consistent personal vocabulary.

### Status

**Supported diagnostic observation, but defer final functional form until larger data.**

Do not yet freeze:

- an Evidence threshold;
- a specific reliability formula;
- Margin vs Entropy as the final confidence measure;
- a smoothing strength based only on the current 3-author Initial dataset.

---

## 5. Exposure Rate: current interpretation

Current transition medians:

| Transition | Exposure median | Evidence median |
|---|---:|---:|
| Harm | 0.0008 | 4 |
| Rescue | 0.0014 | 7 |
| Unchanged correct | 0.0086 | 43 |
| Unchanged wrong | 0.0046 | 23 |

For H5000-saturated histories, Exposure Rate is approximately Evidence Count divided by 5000 and therefore contains nearly the same ordering information.

However, the concepts remain different:

- **Exposure Rate** = prevalence of the key in recent user history.
- **Evidence Count** = amount of evidence available for estimating the conditional choice distribution.

Low Exposure can still be useful when concentration is high. For Exposure <0.1%:

\[
Margin\ge0.75:\quad 257\ rescue,\ 80\ harm,\ net=+177
\]

while:

\[
Margin<0.10:\quad 60\ rescue,\ 75\ harm,\ net=-15
\]

### Status

Keep Exposure Rate for interpretation / reporting and early-history cases, but **do not assume it should replace Evidence Count in reliability estimation**.

Revisit with richer histories where \(|H_t|\) varies more substantially.

---

## 6. Background Lift: current finding

Current transition medians:

| Transition | Raw Log Lift median | Excess Share median |
|---|---:|---:|
| Harm | 3.0263 | 0.4835 |
| Rescue | 1.4714 | 0.3202 |
| Unchanged correct | 0.3275 | 0.1189 |
| Unchanged wrong | 0.6634 | 0.0941 |

This is **opposite to the naive expectation** that larger Lift should directly imply safer personalisation.

Changing from log-ratio to simple absolute difference does not fully solve the issue:

\[
ExcessShare=P_u(c\mid p)-P_{bg,-u}(c\mid p)
\]

Harm still has a higher median than rescue.

### Candidate-zero problem

A major measurement issue is candidate-level zero probability in the small background.

Examples from the current diagnostic:

- Harm, `candidate_zero`, both other users have seen the Initial: **112 rows**, median raw Lift ≈ **13.12**.
- Rescue, same background status: **174 rows**, median raw Lift ≈ **13.12**.

Therefore the current raw Lift often cannot distinguish:

1. genuinely strong personal distinctiveness; and
2. a candidate that simply did not appear in the two available background users.

### Important conceptual correction

Even a perfectly measured distinctiveness signal would not necessarily solve current-intention errors:

\[
Personal\ Distinctiveness \neq Current\ Appropriateness
\]

A candidate can be highly user-specific and still be wrong for the current context.

Therefore Background should answer:

> **Is this candidate unusually characteristic of this user?**

not:

> **Should this candidate be Top-1 right now?**

The latter requires local contextual evidence such as position / recency / current-context compatibility.

---

## 7. Why the current 3-author Initial setting may underestimate Background Lift

### 7.1 Too few background users

With 3 authors total, leave-one-user-out background for user `u` contains only 2 users:

\[
P_{bg,-u}(c\mid p)=\frac{P_{v_1}(c\mid p)+P_{v_2}(c\mid p)}{2}
\]

A zero can therefore mean only:

> neither of these two users happened to produce this candidate.

It should not be interpreted as population-level rarity.

With 6 authors, each user would have 5 background authors. Candidate absence would be more informative and estimates should be less volatile.

### 7.2 Initial Pinyin mixes multiple lexical competition sets

An Initial key such as `sy` can merge Full-Pinyin forms such as:

- `shi yong`
- `shou yi`
- `sheng yi`
- `shi yi`
- etc.

Therefore:

\[
P(c\mid sy)
\]

mixes variation in both:

1. which Full-Pinyin reading the user tends to produce; and
2. which Chinese candidate the user chooses within that reading.

This weakens the construct validity of a candidate-level background comparison.

---

## 8. Why Background Lift may be more meaningful in Full Pinyin

For a Full-Pinyin key such as:

\[
shi\ yong
\]

the candidate competition set may more directly contain items such as:

- 使用
- 实用
- 试用
- 适用

Then:

\[
\frac{P_u(c\mid shi\ yong)}{P_{bg,-u}(c\mid shi\ yong)}
\]

more cleanly asks:

> Given the same lexical pronunciation competition, does this user prefer candidate `c` more than other users?

This is conceptually cleaner than Initial-level Lift.

However, Full Pinyin also produces more specific keys and may therefore have lower per-key sample counts.

Expected trade-off:

\[
Initial:\ more\ observations,\ more\ lexical\ mixing
\]

\[
Full:\ cleaner\ competition\ set,\ potentially\ more\ sparsity
\]

Larger author count and richer history may help Full Pinyin substantially.

---

## 9. Deferred Background Stability Study

When the larger dataset is available, run a dedicated study rather than silently reusing the current conclusions.

### Minimum comparison matrix

| Setting | Initial | Full Pinyin |
|---|---|---|
| Current 3-author background | existing diagnostics | compute if available |
| 6-author + richer history / HFull | rerun | **primary stability condition** |

### Recompute at minimum

For every setting:

- Evidence Count distribution
- Exposure Rate distribution
- Choice Share
- Margin
- normalized Entropy
- background Initial/Pinyin evidence count
- background candidate evidence count
- number of background users who saw the key
- number of background users who saw the candidate
- candidate-zero rate
- raw log Lift
- Excess Share
- smoothed background probability variants
- rescue / harm / net by signal bin
- Formal Conflict and Ambiguous-nonConflict diagnostics

### Specific questions

#### Q1 — Evidence stability

Does the low-evidence + high-concentration advantage remain with more authors / richer histories?

If yes, it supports a general reliability formulation rather than a small-sample artifact.

#### Q2 — Candidate-zero stability

Does candidate-zero rate fall substantially with 6 authors and richer history?

#### Q3 — Lift ordering

Does Background Lift begin to separate rescue from harm more cleanly after increasing background coverage?

#### Q4 — Initial vs Full

Is Lift more stable / interpretable / predictive under Full Pinyin than Initial Pinyin?

#### Q5 — Ranking contribution

After measurement is stabilized, does a background-distinctiveness contribution add positive paired net rescue over a strong ChoiceShare + Concentration model?

---

## 10. Smoothing candidates to test later

Do not tune these yet on the current small diagnostic alone.

### User-side smoothing

\[
\widetilde P_u(c\mid p)
=
\frac{N_u(c,p)+\alpha P_0(c\mid p)}{N_u(p)+\alpha}
\]

### Background-side smoothing

Candidate-level smoothing is required so that `0/5` and `0/1000` do not both become exactly zero.

One possible form:

\[
\widetilde P_{bg}(c\mid p)
=
\frac{N_{bg}(c,p)+\beta P_0(c\mid p)}{N_{bg}(p)+\beta}
\]

### Smoothed distinctiveness diagnostics

\[
SmoothedLogLift
=
\log\frac{\widetilde P_u(c\mid p)}{\widetilde P_{bg}(c\mid p)}
\]

and:

\[
SmoothedExcessShare
=
\widetilde P_u(c\mid p)-\widetilde P_{bg}(c\mid p)
\]

Possible reliability terms should be evaluated separately rather than hidden inside an opaque gate.

---

## 11. Current decisions

### Keep

- Choice Share as the main long-term candidate preference concept.
- Margin / Entropy as concentration diagnostics and likely confidence components.
- Evidence Count as an uncertainty / reliability input.
- Exposure Rate as a prevalence / interpretability measure.
- Background distinctiveness as a **formal future ablation / stability question**.

### Do not freeze yet

- minimum Evidence threshold;
- exact Evidence-to-confidence function;
- Margin vs Entropy final choice;
- raw Lift as a ranking bonus;
- smoothing constants;
- background reliability formula;
- claim that Lift is ineffective;
- claim that Lift is effective;
- claim that Full Pinyin is superior for Lift.

### Current wording for thesis notes

> Current Train-Val diagnostics indicate that sparse personal evidence can still be useful when the historical choice distribution is highly concentrated. Raw leave-one-user-out background lift is unstable in the present 3-author Initial-Pinyin setting, especially under candidate-level zero counts, and high personal distinctiveness does not by itself imply current contextual correctness. Evidence reliability and background distinctiveness should therefore be re-evaluated with a larger author pool, richer history, and a Full-Pinyin comparison before their final functional forms are frozen.

---

## 12. Revisit trigger

Reopen this note when **6-author + richer-history / HFull** data are ready.

At that point:

1. reproduce the current 3-author diagnostics;
2. run the same diagnostics on the larger author pool;
3. compare Initial vs Full Pinyin;
4. only then decide the final Evidence reliability and Background Distinctiveness formulations.

Until then, these components remain **recorded but intentionally unresolved**.
