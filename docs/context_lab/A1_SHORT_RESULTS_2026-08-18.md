# Diagnostic A1 — Short Retrieval Results

**Date:** 2026-08-18  
**Status:** Completed exploratory diagnostic  
**History budget:** H5000  
**Exploratory authors:** Etinjat, Re_spectators, breaddddd

---

## 1. Scope

Diagnostic A1 asks whether contextual personalisation is limited because:

1. useful Gold-target history does not exist; or
2. useful history exists but BGE similarity retrieval fails to retrieve it.

The active analysis now focuses on:

- Full + Short
- Initial + Short

Multi3 observations are retained separately but are not part of active Short method development.

Recall is calculated only among rows where legal strictly-prior H5000 history contains the Gold target.

---

## 2. Full + Short — Overall

Test rows:

- 3,000

History:

- History Available: 1,891 / 3,000 = 63.03%
- Gold-target history exists: 1,698 / 3,000 = 56.60%
- Gold exists given available history: 89.79%

BGE retrieval among Gold-history rows:

- Recall@1: 85.22%
- Recall@3: 93.70%
- Recall@5: 96.17%
- Recall@10: 98.53%
- Recall@20: 99.41%

Macro-author:

- Recall@1: 82.01%
- Recall@3: 91.56%
- Recall@5: 95.01%
- Recall@10: 98.07%
- Recall@20: 99.29%

### Per-author Recall@1

- Etinjat: 64.36%
- Re_spectators: 88.85%
- breaddddd: 92.80%

Etinjat is substantially harder than the other two exploratory authors.

---

## 3. Full + Short — Ambiguous

Rows:

- 836

Gold-target history exists:

- 763

Retrieval:

- Recall@1: 67.10%
- Recall@3: 85.98%
- Recall@5: 91.48%
- Recall@10: 96.72%
- Recall@20: 98.69%

Even under ambiguity, Gold history is usually present in the upper retrieval set.

---

## 4. Full + Short — Conflict

Rows:

- 233

Gold-target history exists:

- 173

Retrieval:

- Recall@1: 24.28%
- Recall@3: 52.60%
- Recall@5: 69.36%
- Recall@10: 86.13%
- Recall@20: 94.80%

Per-author Recall@1 / Recall@10:

- Etinjat: 26.85% / 86.11%
- Re_spectators: 10.26% / 82.05%
- breaddddd: 34.62% / 92.31%

### Interpretation

Conflict cases expose an important distinction:

Gold history is often not the top-ranked retrieved memory, but it is still usually present within the Top10 or Top20.

Therefore, for Full + Short, the main remaining question is not simply whether BGE can access useful Gold history.

The next diagnostic should investigate:

- which competing historical targets outrank Gold;
- whether retrieved contextual evidence supports Gold strongly enough;
- whether frequency evidence overwhelms contextual evidence;
- whether M1/M2 scoring preserves or destroys useful contextual information;
- whether a decision rule should choose among Generic, Frequency, and contextual override rather than always blending them.

This motivates Diagnostic A2.

---

## 5. Initial + Short — Overall

Test rows:

- 3,000

History:

- History Available: 2,796 / 3,000 = 93.20%
- Gold-target history exists: 1,698 / 3,000 = 56.60%
- Gold exists given available history: 60.73%

BGE retrieval among Gold-history rows:

- Recall@1: 39.22%
- Recall@3: 59.72%
- Recall@5: 67.96%
- Recall@10: 79.15%
- Recall@20: 86.63%

Macro-author:

- Recall@1: 36.10%
- Recall@3: 55.41%
- Recall@5: 63.53%
- Recall@10: 75.32%
- Recall@20: 83.18%

### Per-author Recall@1

- Etinjat: 18.78%
- Re_spectators: 44.27%
- breaddddd: 45.23%

Etinjat is particularly difficult under Initial input.

---

## 6. Initial + Short — Ambiguous

Rows:

- 2,618

Gold-target history exists:

- 1,625

Retrieval:

- Recall@1: 36.49%
- Recall@3: 57.91%
- Recall@5: 66.52%
- Recall@10: 78.22%
- Recall@20: 86.03%

The large ambiguous subset confirms that Initial input exposes much more competing history than Full input.

---

## 7. Initial + Short — Conflict

Rows:

- 1,438

Gold-target history exists:

- 742

Retrieval:

- Recall@1: 14.69%
- Recall@3: 31.27%
- Recall@5: 42.45%
- Recall@10: 60.51%
- Recall@20: 75.88%

Per-author Recall@1 / Recall@10:

- Etinjat: 6.60% / 42.64%
- Re_spectators: 20.33% / 68.67%
- breaddddd: 14.29% / 64.90%

### Interpretation

Initial + Short has a different failure mode from Full + Short.

The number of rows containing Gold history is unchanged relative to Full + Short, but History Available rises from 63.03% to 93.20%.

Thus Initial input introduces substantially more non-Gold competing history without increasing Gold-history coverage.

This is consistent with the hypothesis that multiple Full-Pinyin intents are merged by Initial representation.

Diagnostic A1 alone does not prove that Full-Pinyin expansion is the correct solution, but it provides strong motivation for Diagnostic B:

Initial → Full Pinyin → Chinese

---

## 8. Cross-Condition Comparison

| Metric | Full + Short | Initial + Short |
|---|---:|---:|
| History Available | 63.03% | 93.20% |
| Gold History Exists | 56.60% | 56.60% |
| Overall R@1 | 85.22% | 39.22% |
| Overall R@10 | 98.53% | 79.15% |
| Ambiguous R@1 | 67.10% | 36.49% |
| Conflict R@1 | 24.28% | 14.69% |
| Conflict R@10 | 86.13% | 60.51% |

The strongest structural observation is:

> Initial input substantially increases competing visible history while Gold-history coverage remains unchanged.

---

## 9. Current Research Decision

### Full + Short

Proceed to Diagnostic A2.

Primary question:

> When Gold history has already been retrieved, why does contextual personalisation still fail to consistently outperform Frequency?

A2 should inspect candidate-level and history-level evidence rather than redesign Stage-1 retrieval immediately.

### Initial + Short

Do not prioritise the same A2 path first.

Proceed toward:

- Diagnostic B: Initial-to-Full-Pinyin expansion;
- cross-expansion vs within-expansion conflict decomposition;
- Diagnostic C: Personal Vocabulary recoverability.

### Multi3

Deferred from active Short method development.

Existing Multi3 A1 observations remain archived.

---

## 10. Safe Reporting Conclusion

Supported:

> Diagnostic A1 indicates different bottlenecks for Full and Initial Short input. Full + Short has high Gold-history retrieval coverage within the upper BGE ranks, including on many conflict cases, suggesting that subsequent evidence aggregation and decision logic deserve investigation. Initial + Short introduces substantially more competing history and exhibits much lower retrieval recall, motivating a structured Initial-to-Full-Pinyin analysis.

Not yet supported:

- Context is ineffective.
- BGE is generally unsuitable.
- Initial-to-Full-Pinyin expansion is proven superior.
- Multi3 cannot be personalised.
- A particular new scoring rule is already validated.

All proposed explanations beyond the observed retrieval statistics remain hypotheses until tested.
