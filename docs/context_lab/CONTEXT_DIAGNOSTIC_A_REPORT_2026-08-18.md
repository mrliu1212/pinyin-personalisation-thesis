# Context Diagnostic A — Retrieval, Decision, and Evidence Competition

**Date:** 2026-08-18  
**Status:** Completed exploratory diagnostic checkpoint  
**Condition:** Full + Short  
**History budget:** H5000  
**Exploratory authors:** Etinjat, Re_spectators, breaddddd  
**Rows:** 3,000 Test anchors

> This document records an exploratory diagnostic stage. It is not the final six-author formal evaluation. Mixed-script confounding remains a known issue in Dataset V1 and must be addressed before final formal evaluation.

---

# 1. Purpose

Diagnostic A was designed to locate the failure point of the current contextual-personalisation pipeline.

The central question was not simply whether Context outperforms Frequency. Instead, the diagnostic decomposed the problem into three stages:

1. Does useful strictly-prior Gold-target history exist?
2. If it exists, can the contextual retriever find it?
3. If it is retrieved, does contextual scoring / decision use the evidence correctly?

This produced three diagnostic stages:

- **A1 — Retrieval diagnosis**
- **A2 — Decision-transition diagnosis**
- **A2b — Evidence-competition diagnosis**

---

# 2. Diagnostic A1 — Retrieval

## 2.1 Full + Short overall

Test rows: **3,000**

- History Available: **1,891 / 3,000 = 63.03%**
- Gold-target History Exists: **1,698 / 3,000 = 56.60%**
- Gold exists given History Available: **89.79%**

Among the 1,698 rows where legal strictly-prior H5000 history contains the Gold target:

| Metric | Recall |
|---|---:|
| R@1 | 85.22% |
| R@3 | 93.70% |
| R@5 | 96.17% |
| R@10 | 98.53% |
| R@20 | 99.41% |

### Interpretation

Once useful Gold-target history exists, the current BGE retrieval mechanism usually succeeds in placing at least one Gold-target historical interaction near the top of the retrieved set.

Therefore, under Full + Short, poor final contextual performance cannot in general be explained simply by saying that the correct history is absent from retrieval.

## 2.2 Full + Short Ambiguous subset

- Rows: **836**
- Gold history exists: **763**

| Metric | Recall |
|---|---:|
| R@1 | 67.10% |
| R@3 | 85.98% |
| R@5 | 91.48% |
| R@10 | 96.72% |
| R@20 | 98.69% |

## 2.3 Full + Short Conflict subset

Conflict is defined as:

- at least two distinct historical targets;
- Frequency has a unique winner;
- the Frequency winner is not the Gold target.

Rows:

- Conflict rows: **233**
- Conflict rows with Gold history: **173**

Among those 173 recoverable Conflict cases:

| Metric | Recall |
|---|---:|
| R@1 | 24.28% |
| R@3 | 52.60% |
| R@5 | 69.36% |
| R@10 | 86.13% |
| R@20 | 94.80% |

### Interpretation

Conflict cases are substantially harder at rank 1, but Gold-target history is still usually present in the upper retrieved set. This creates many contextual rescue opportunities: the system frequently has access to useful Gold evidence even when Frequency prefers a different target.

A1 alone does not show whether this evidence is used successfully.

---

# 3. Diagnostic A2 — Final Decision Behaviour

A2 joined the A1 rows with the previously frozen Full + Short / H5000 Generic, Frequency, M1, and M2 prediction artifacts.

Join integrity:

- A1 rows: **3,000**
- A1 duplicate IDs: **0**
- F missing from A1: **0**
- M1 missing from A1: **0**
- M2 missing from A1: **0**
- Gold/author mismatches: **0**
- Shared across A1/F/M1/M2: **3,000**

The A1 `row_id` matches the prediction artifacts' `condition_id`.

## 3.1 Overall accuracy

| Method | Top-1 |
|---|---:|
| Generic | 77.60% |
| Frequency | 81.07% |
| M1 | 79.83% |
| M2 | 80.10% |

Frequency remains stronger than both contextual methods in this exploratory Full + Short comparison.

## 3.2 Frequency-to-Context transitions

### M1

- F wrong -> M1 correct: **18**
- F correct -> M1 wrong: **55**
- Net vs F: **-37**

### M2

- F wrong -> M2 correct: **13**
- F correct -> M2 wrong: **42**
- Net vs F: **-29**

### Interpretation

Context sometimes repairs Frequency errors, but the number of newly introduced errors is substantially larger than the number of rescues. The current contextual methods therefore have negative net value relative to Frequency in this exploratory sample.

---

# 4. Independent Context Rescues

A Frequency-to-Context rescue is not automatically evidence that Context independently discovered the correct answer. Some apparent rescues occur when Generic was already correct, Frequency changed it to an incorrect candidate, and Context merely restored or preserved the Generic answer.

## 4.1 M1

Of the 18 F-wrong -> M1-correct transitions:

- **14** had Generic already correct;
- only **4** had Generic wrong and Frequency wrong.

Therefore:

**M1 unique contextual rescues = 4 / 3,000.**

## 4.2 M2

Of the 13 F-wrong -> M2-correct transitions:

- **11** had Generic already correct;
- only **2** had Generic wrong and Frequency wrong.

Therefore:

**M2 unique contextual rescues = 2 / 3,000.**

### Interpretation

The current contextual mechanisms have genuine independent predictive value, but such clean rescues are rare. Most apparent rescues are better described as protection against Frequency damage rather than independent contextual discovery.

---

# 5. Strong Context Regressions

A particularly important failure type is:

> Generic correct + Frequency correct + Context wrong.

Observed counts:

- M1 strong regression: **45**
- M2 strong regression: **33**

This failure type motivated A2b.

---

# 6. Diagnostic A2b — Evidence Competition

A2b compares the contextual support assigned to the Gold candidate and the incorrect final Context winner. It also separately examines the very small number of genuine unique Context rescues.

# 7. Why Does Context Break Correct Predictions?

## 7.1 M1 strong regressions

Cases: **45**

Evidence:

- Gold evidence present in actual M1 retrieved evidence: **21 / 45**
- Wrong-winner evidence present: **45 / 45**
- Wrong winner Context support > Gold Context support: **45 / 45**

The 45 M1 regressions contain two broad situations.

### A. Gold did not enter the actual M1 Top-5 evidence

- **24 / 45 cases**

In these cases, deeper retrieval may contain the Gold history, but it is not available to the actual M1 Top-5 evidence stage. This indicates a retrieval-depth / selection component.

### B. Gold and wrong winner both entered M1 evidence

- **21 / 45 cases**

Nevertheless:

> wrong-winner Context support > Gold Context support.

This indicates contextual discrimination / evidence-competition failure.

## 7.2 M2 strong regressions

Cases: **33**

Evidence:

- Gold evidence present: **33 / 33**
- Wrong-winner evidence present: **33 / 33**
- Wrong-winner Context support > Gold Context support: **33 / 33**

For these strong M2 regressions, the failure cannot be explained by simple absence of Gold evidence. The system sees Gold-supporting history but assigns stronger contextual support to an incorrect alternative.

---

# 8. What Does a Successful Context Rescue Look Like?

## 8.1 M1 unique rescues

Cases: **4**

All four have:

- Generic wrong;
- Frequency wrong;
- Generic and Frequency choose the same wrong candidate;
- M1 correct;
- Gold Context support > the shared wrong candidate;
- Gold evidence present.

Gold retrieval rank:

- rank 1: **3**
- rank 2: **1**

## 8.2 M2 unique rescues

Cases: **2**

Both have:

- Generic wrong;
- Frequency wrong;
- Generic and Frequency choose the same wrong candidate;
- M2 correct;
- Gold Context support > the shared wrong candidate;
- Gold evidence present.

Gold retrieval rank:

- rank 1: **1**
- rank 2: **1**

---

# 9. Main Diagnostic A Conclusion

The simplest interpretation of Diagnostic A is:

> Context sometimes works, but reliable independent contextual rescue is currently rare.

More importantly:

> The main Full + Short bottleneck is not simply retrieval coverage.

When legal Gold history exists, retrieval usually places Gold evidence in the upper candidate-history set. The more important weakness is **contextual discrimination**.

The current contextual mechanism is not reliably able to distinguish history that is genuinely predictive of the user's current intended candidate from history that is semantically similar but misleading for the current IME completion decision.

In strong regressions, incorrect candidates consistently receive stronger contextual support than the Gold candidate. Conversely, the rare genuine Context rescues occur when Gold history is retrieved extremely highly (rank 1-2) and the Context scorer gives Gold stronger support than the common wrong Generic/Frequency candidate.

A concise description is:

> **The current Context mechanism can usually find useful evidence, but it is not yet reliable at deciding which evidence to trust.**

Or more formally:

> **The principal bottleneck is contextual discrimination rather than retrieval coverage.**

---

# 10. Methodological Boundary

Diagnostic A does **not** establish that:

- Context is useless;
- BGE can never work for this task;
- M2 is inherently unsuitable;
- gating will necessarily improve performance;
- any proposed replacement method is already superior.

The observed evidence supports the diagnosis of the current implementation, not a universal conclusion about contextual personalisation.

---

# 11. Research Decision After Diagnostic A

The immediate next step is **not** to prioritise selective gating.

The current Context signal itself is still too weak and unreliable. A gating mechanism built directly on an unreliable contextual confidence signal may simply become confident about the wrong candidate.

Therefore the next research priority is:

## Strengthen Context first.

Only after contextual discrimination improves should the project evaluate whether and when Context should be allowed to override Generic/Frequency.

The intended research order is:

1. improve Context representation / matching;
2. demonstrate stronger contextual discrimination;
3. re-evaluate independent Context rescues and regressions;
4. only then investigate confidence-aware gating / selective intervention.

---

# 12. Next Context-Strengthening Hypothesis

The current M1/M2 mechanisms rely heavily on generic semantic similarity. However, IME contextual relevance is not necessarily the same as general semantic similarity.

The next experiment should therefore test **local context matching**.

Candidate memory-query windows:

- last 8;
- last 16;
- last 32;
- last 64;
- last 128;
- Full context.

Important experimental boundary:

> The frozen Generic PinyinGPT input/context remains unchanged.

Only the personal-memory Context representation is varied.

Primary diagnostic targets should include:

- Gold retrieval R@1 / R@5 / R@10;
- Conflict retrieval quality;
- contextual discrimination between Gold and competing historical targets;
- strong-regression rate;
- unique Context rescue rate.

This is a method-strengthening experiment, not yet a gating experiment.

---

# 13. Reproduction

## 13.1 Required source code

Diagnostic A depends on these Context Lab scripts:

```text
experiments/context_lab/diagnostic_a_retrieval.py
experiments/context_lab/diagnostic_a2_decision.py
experiments/context_lab/diagnostic_a2b_evidence_competition.py
```

The existing frozen Full + Short / H5000 prediction artifacts must also be preserved.

### Frequency

```text
C:\Users\chiar\Desktop\LBH\thesis-personalisation\
results\personalisation\pilot_a_context_memory\h5000\
frequency_predictions.jsonl
```

### M1

```text
C:\Users\chiar\Desktop\LBH\thesis-personalisation\
results\personalisation\pilot_a_context_memory\h5000\
memory_predictions.jsonl
```

### M2

```text
C:\Users\chiar\Desktop\LBH\thesis-personalisation\
results\personalisation\m2_h5000\
m2_predictions.jsonl
```

## 13.2 A1 outputs

```text
results/personalisation/context_lab/diagnostic_a1_retrieval/
```

Important files:

```text
rows_full_short.jsonl
metrics_full_short.json
summary.json
short_detailed_summary.txt
```

## 13.3 A2 outputs

```text
results/personalisation/context_lab/diagnostic_a2_decision/
```

Important files:

```text
rows.jsonl
summary.json
stdout.log
```

## 13.4 A2b v2 outputs

```text
results/personalisation/context_lab/diagnostic_a2b_evidence_competition_v2/
```

Important files:

```text
m1_strong_regression.jsonl
m1_unique_rescue.jsonl
m2_strong_regression.jsonl
m2_unique_rescue.jsonl
summary.json
stdout.log
```

The earlier A2b output directory is retained for provenance but its unique-rescue comparison should not be used because v1 compared the Gold candidate against the method winner, which is itself Gold in rescue cases.

The corrected v2 explicitly compares Gold against the incorrect Generic/Frequency candidate.

## 13.5 Python environment

Python:

```text
C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe
```

Run from:

```text
C:\Users\chiar\Desktop\LBH\thesis-context-lab
```

## 13.6 Compile diagnostic scripts

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
-m py_compile `
experiments\context_lab\diagnostic_a_retrieval.py `
experiments\context_lab\diagnostic_a2_decision.py `
experiments\context_lab\diagnostic_a2b_evidence_competition.py
```

Expected exit code:

```text
0
```

## 13.7 Reproduce A1

Run the cache/audit phase first:

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
experiments\context_lab\diagnostic_a_retrieval.py `
--phase audit
```

Only proceed when the required embedding-cache audit reports no unresolved cache misses.

Then run:

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
experiments\context_lab\diagnostic_a_retrieval.py `
--phase run
```

## 13.8 Reproduce A2

A2 performs read-only joins over A1 plus the existing F/M1/M2 prediction artifacts. It does not run neural inference.

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
experiments\context_lab\diagnostic_a2_decision.py
```

Expected integrity condition:

```text
shared A1/F/M1/M2 rows = 3000
no duplicate IDs
no missing A1 rows
no Gold/author mismatch
```

## 13.9 Reproduce corrected A2b

A2b is also read-only and performs no model inference.

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
experiments\context_lab\diagnostic_a2b_evidence_competition.py
```

Corrected output namespace:

```text
diagnostic_a2b_evidence_competition_v2
```

---

# 14. Checkpoint Status

Diagnostic A is considered complete at this checkpoint.

Next active stage:

> **Context Strengthening — local-context retrieval / matching diagnostics**

Do not reinterpret hypotheses from the next stage as verified findings until their experiments are completed.
