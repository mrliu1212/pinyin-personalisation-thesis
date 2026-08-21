# Initial Personalisation: Recoverability-First Research Plan

## 1. Purpose

This document defines the next research step for the **Initial+Short** condition under the new standardized reset.

The main question is not yet "which reranker is best?". The first question is:

> When Generic misses the correct answer in its Top-10 candidates, how often can that answer be legally recovered from the user's own causal personal history?

This is the **recoverability** question.

The result of this audit will decide whether the next priority should be **candidate recovery** (EM1 / PV1 style) or whether the main bottleneck lies elsewhere.

---

## 2. Current standardized data base

The standardized reset has already produced a frozen Train-Fit / Train-Val split from Clean3 Train.

| Dataset | Rows | SHA256 | Role |
|---|---:|---|---|
| Clean3 Train | 178,942 | — | Source population |
| Train-Fit | 144,526 | `547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6` | Model fitting / training |
| Train-Val | 34,416 | `d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220` | Method development and tuning |
| Frozen Dev3000 | 3,000 | `9181f895eb19d0c36852e511263bfaefb34459dcd44efa6f45a44252e6b03f93` | Sealed confirmation after method freeze |
| Test | closed | — | Not used during current method development |

Current Initial research should use **Train-Fit / Train-Val only**. Dev3000 must not be used until the Initial method is frozen. Test remains closed.

### How to obtain the standardized files

The current standardized split is available in the comparison worktree:

```text
C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_v2\clean3_train_fit_v1.jsonl
C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_v2\clean3_train_val_v1.jsonl
```

Recommended PowerShell check:

```powershell
$fit = 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_v2\clean3_train_fit_v1.jsonl'
$val = 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_v2\clean3_train_val_v1.jsonl'

(Get-Content $fit -Encoding UTF8 | Measure-Object -Line).Lines
(Get-FileHash $fit -Algorithm SHA256).Hash

(Get-Content $val -Encoding UTF8 | Measure-Object -Line).Lines
(Get-FileHash $val -Algorithm SHA256).Hash
```

Expected values:

```text
Train-Fit rows: 144526
Train-Fit SHA : 547A4F8179F5D664A8621888236599938A2F967F055EF0C262BE658B3500C8A6

Train-Val rows: 34416
Train-Val SHA : D7AE1CC21EE029DDE8458189B9DC7A0989B2B3A372627E079C3E2699307F2220
```

When reading JSONL in Windows PowerShell, use explicit UTF-8:

```powershell
Get-Content $val -TotalCount 1 -Encoding UTF8 |
    ConvertFrom-Json |
    Format-List *
```

---

## 3. Important warning about existing fields

The standardized Train-Val rows were generated for the **Full+Short** condition.

Fields such as:

```text
ambiguous
conflict
same_pinyin_history_count
distinct_history_targets
frequency_winner
positive_history_count
negative_history_count
```

must **not** be reused directly as Initial-condition labels.

They were computed under Full-Pinyin history matching.

For Initial research, the history-derived quantities must be recomputed after applying the frozen historical Initial transformation.

---

## 4. Historical evidence motivating this study

Historical Test results show that Initial has a much larger candidate-coverage problem than Full.

### Generic historical performance

| Condition | Generic Macro Top-1 |
|---|---:|
| Full+Short | 72.32% |
| Initial+Short | 32.90% |
| Full+Multi3 | 37.63% |
| Initial+Multi3 | 7.23% |

### Historical Missing@10

| Condition | Missing@10 |
|---|---:|
| Full+Short, H5000 | 8.97% |
| Initial+Short, H500 | 37.50% |
| Initial+Short, H5000 | 37.50% |

Therefore:

\[
37.50\%-8.97\%=28.53\text{ percentage points}
\]

and Initial Missing@10 was approximately:

\[
\frac{37.5}{8.97}\approx4.18\times
\]

that of Full.

This is the main reason to investigate candidate recovery before building a stronger reranker.

---

## 5. Historical Initial structure

### H500

| Metric | Value |
|---|---:|
| Total | 6,000 |
| History Available | 70.12% |
| Ambiguous | 49.85% |
| Conflict | 20.38% |
| Missing@10 | 37.50% |
| Frequency Top1 | 33.42% |
| M1 Top1 | 34.85% |
| M2 Top1 | 34.63% |

### H5000

| Metric | Value |
|---|---:|
| Total | 6,000 |
| History Available | 93.48% |
| Ambiguous | 87.87% |
| Conflict | 47.50% |
| Missing@10 | 37.50% |
| Frequency Top1 | 37.22% |
| M1 Top1 | 36.40% |
| M2 Top1 | 36.03% |

The important structural pattern is:

\[
History\ Depth\uparrow
\Rightarrow
History\ Availability\uparrow
\]

but also:

\[
Ambiguity\uparrow,\quad Conflict\uparrow
\]

At H5000, M1 still helps Conflict relative to Frequency, but loses Overall Top1. This suggests that context may still be useful for difficult cases while becoming harmful when applied broadly.

A separate hypothesis is that older history may be less predictive than recent history. This should be tested later, not assumed.

---

## 6. Primary research question: recoverability

For query \(q_t\), let:

- \(C_G^{10}(q_t)\): Generic Initial Top-10 candidates;
- \(y_t\): Gold target;
- \(H_t^I\): legal Initial-matched causal personal history.

The key event is:

\[
y_t\notin C_G^{10}(q_t)
\quad\land\quad
y_t\in H_t^I
\]

This means Generic omitted the Gold, but the user's own legal personal history contains it.

### Main metrics

#### Generic Missing@10

\[
R_{missing}
=
P(y_t\notin C_G^{10})
\]

#### Personal-history availability of Gold

\[
R_{history-gold}
=
P(y_t\in H_t^I)
\]

#### Recoverable missing rate over all queries

\[
R_{recoverable}
=
P(y_t\notin C_G^{10}\land y_t\in H_t^I)
\]

#### Recoverability conditional on Generic missing

\[
R_{recoverable\mid missing}
=
P(y_t\in H_t^I\mid y_t\notin C_G^{10})
\]

This is the most interpretable quantity:

> Among queries missed by Generic Top-10, what percentage contain the Gold in legal personal history?

---

## 7. Causal history semantics

For every query \(q_t\) from author \(a\):

\[
Prior(a,q_t)=\{\text{same-author interactions strictly earlier than }q_t\}
\]

Then:

\[
H_t=\text{most recent }\min(5000,|Prior(a,q_t)|)\text{ raw rows}
\]

Only after this raw-history cap is applied should the Initial condition transform and match be performed.

Conceptually:

\[
H_t^I
=
\{x\in H_t:I(p_x)=I(p_t)\}
\]

where \(I(\cdot)\) is the frozen Initial transformation.

The required order is:

```text
same author
-> strictly prior
-> latest up-to-5000 raw interactions
-> Initial transform / Initial match
-> candidate counts / recovery / ranking
```

Do not use:

```text
all same-Initial history over unlimited past
-> then take 5000
```

History is query-specific, not batch-specific.

---

## 8. First audit outputs

Before training or tuning a new model, produce one Train-Val audit containing at least:

| Metric | Value |
|---|---:|
| Train-Val queries | 34,416 |
| Initial Generic Top1 | ? |
| Initial Generic Recall@10 | ? |
| Initial Missing@10 | ? |
| History available | ? |
| Gold in legal personal history | ? |
| Recoverable missing | ? |
| Recoverable given missing | ? |
| Initial ambiguous rate | ? |
| Initial conflict rate | ? |
| Mean distinct personal targets | ? |
| Median distinct personal targets | ? |

These values must be recomputed on the standardized Train-Val population.

---

## 9. Personal Top-K candidate recovery

Presence anywhere in history is only an oracle-style opportunity measure. A practical method must select a small number of personal candidates.

Therefore measure:

\[
Gold\in PersonalTopK
\]

for preregistered values such as:

\[
K\in\{1,3,5,10\}
\]

Recommended coverage table:

| Candidate surface | Gold coverage |
|---|---:|
| Generic Top10 | ? |
| Generic Top10 + Personal Top1 | ? |
| Generic Top10 + Personal Top3 | ? |
| Generic Top10 + Personal Top5 | ? |
| Generic Top10 + Personal Top10 | ? |

This tells us how much candidate recovery is available before introducing a complicated reranker.

---

## 10. Decision rule after the recoverability audit

If:

\[
R_{recoverable\mid missing}
\]

is substantial, candidate recovery becomes the primary next step.

The next comparison should then be a strict reproduction on the new standardized data:

\[
Generic
\rightarrow
Frequency
\rightarrow
EM1
\rightarrow
PV1
\]

EM1 and PV1 must be reproduced from their historical definitions rather than re-invented.

They must use the same:

- Train-Fit / Train-Val query population;
- Initial transformation;
- causal history semantics;
- Generic candidate outputs;
- history budget;
- evaluation metrics.

---

## 11. Planned EM1 vs PV1 comparison

After recoverability is established, compare EM1 and PV1 on the new Train-Val data using at least:

| Metric | Generic | Frequency | EM1 | PV1 |
|---|---:|---:|---:|---:|
| Macro-author Top1 | ? | ? | ? | ? |
| Recall@10 | ? | ? | ? | ? |
| Missing@10 | ? | ? | ? | ? |
| History subset Top1 | ? | ? | ? | ? |
| Ambiguous Top1 | ? | ? | ? | ? |
| Conflict Top1 | ? | ? | ? | ? |
| Rescue vs baseline | — | ? | ? | ? |
| Harm vs baseline | — | ? | ? | ? |
| Net rescue-harm | — | ? | ? | ? |

Definitions:

\[
Rescue=\#(baseline\ wrong,\ method\ correct)
\]

\[
Harm=\#(baseline\ correct,\ method\ wrong)
\]

\[
Net=Rescue-Harm
\]

The first goal is to determine which method actually recovers missing candidates more effectively and safely under Initial.

---

## 12. Historical EM1 / PV1 evidence to reproduce, not reuse as new evidence

Historical Full+Short PV results showed that candidate recovery could reduce Missing@10:

| Method | Macro Top1 |
|---|---:|
| Generic | 72.32% |
| Frequency | 77.18% |
| PV1 | 77.90% |
| PV2 | 77.82% |

Historical Missing@10 decreased from about 8.97% to 6.62% under PV1.

This is only historical evidence that candidate recovery can work. It does **not** establish that PV1 will be best for Initial.

The standardized Initial Train-Val comparison must be run again from scratch.

---

## 13. Recency hypothesis for later analysis

The historical Initial result has an interesting pattern:

- M1 is better than Frequency at H500;
- Frequency is better than M1 Overall at H5000;
- M1 still helps on Conflict at H5000.

One plausible explanation is temporal drift:

\[
Recent\ personal\ history
>
old\ personal\ history
\]

in predictive value.

After the recoverability and EM1/PV1 study, test this hypothesis by grouping legal history by age, for example:

```text
1-500
501-1000
1001-2000
2001-5000
```

and measure quantities such as:

\[
P(target_h=gold\mid age\ bucket)
\]

and retrieval purity for rescue vs harm cases.

Do not introduce recency weighting before showing evidence that age matters.

---

## 14. What is out of scope for the first audit

The first recoverability study should **not**:

- use Dev3000;
- use Test;
- tune Context models;
- tune recency decay;
- modify M1 / M2 / EM3;
- change the frozen Train-Fit / Train-Val population;
- reuse Full-condition ambiguity/conflict labels as if they were Initial labels.

The goal is diagnostic first.

---

## 15. Expected research sequence

The planned sequence is:

\[
Initial\ Transform\ Verification
\]

\[
\downarrow
\]

\[
Train\text{-}Val\ Recoverability\ Audit
\]

\[
\downarrow
\]

\[
EM1\ vs\ PV1\ Standardized\ Reproduction
\]

\[
\downarrow
\]

\[
Candidate\ Recovery\ Method\ Selection
\]

\[
\downarrow
\]

\[
Context\ /\ Conflict\ Disambiguation
\]

\[
\downarrow
\]

\[
Recency\ Analysis\ if\ justified
\]

Only after the Initial method is fixed should a PRE-DEV freeze be produced and the sealed Dev3000 evaluated.

---

## 16. Immediate next action

Before writing the recoverability script, verify the frozen historical Initial implementation:

- exact Full-Pinyin -> Initial transform;
- syllable segmentation behavior;
- handling of heteronyms / condition-manifest generation;
- how historical interactions are converted for Initial matching;
- historical EM1 candidate-recovery semantics;
- historical PV1 scoring and candidate-injection semantics.

The historical implementation should be inspected from the existing personalisation repository rather than reconstructed from memory.

Useful historical repository:

```text
C:\Users\chiar\Desktop\LBH\thesis-personalisation
```

Current Initial research workspace:

```text
C:\Users\chiar\Desktop\LBH\thesis-initial-research
```

Current standardized comparison workspace, treated as read-only reference for this study:

```text
C:\Users\chiar\Desktop\LBH\thesis-context-compare
```
