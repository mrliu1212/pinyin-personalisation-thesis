# Phase 4C Error Analysis Results

## Purpose

This analysis investigates why the Phase 4C frequency-based personalisation
model did not outperform the baseline Luna Pinyin system.

The analysis focuses on selected transparency examples extracted from the
evaluation artifact.

These examples are used to understand model behaviour. They are not a
statistical estimate over the full test benchmark.

---

# 1. Improved Cases

Improved cases are examples where:

personalised rank < base rank.

The analysed examples show that personalisation can successfully correct
baseline ranking when historical frequency agrees with the target candidate.

However, improvements are mainly caused by Pinyin-level and global frequency
evidence rather than context-specific evidence.

## Example: 老是

Observation:

- The baseline ranked "老师" above "老是".
- Personalisation promoted "老是" to the first position.
- The target candidate received historical evidence from previous user
  selections.

Evidence:

- Global evidence: non-zero
- Pinyin evidence: non-zero
- Context evidence: zero

Interpretation:

The model successfully used user history to resolve a lexical ambiguity, but the
improvement was frequency-based rather than context-aware.

Category:

frequency_success

---

## Example: 实在

Observation:

- The baseline ranked another candidate above "实在".
- Personalisation promoted "实在" to the first position.

Evidence:

- Global evidence: non-zero
- Pinyin evidence: non-zero
- Context evidence: zero

Interpretation:

The model learned that the user frequently selected this candidate for the same
Pinyin input.

However, it did not use the surrounding sentence context.

Category:

frequency_success

---

# 2. Harmed Cases

Harmed cases are examples where:

personalised rank > base rank.

These cases reveal the limitations of frequency-only adaptation.

---

## Example: 他们 / 她们

Observation:

- The baseline correctly ranked "他们" first.
- Personalisation promoted "她们" above "他们".

Evidence:

- "她们" had stronger historical frequency evidence.
- Context evidence was zero.

Interpretation:

The model increased a historically frequent homophone without considering the
current sentence meaning.

This represents a same-Pinyin lexical confusion problem.

Category:

same_pinyin_lexical_confusion

Implication:

Frequency evidence alone cannot reliably distinguish candidates with identical
Pinyin but different semantics.

---

## Example: 偷偷 / 偷偷地

Observation:

- The baseline ranked "偷偷地" first.
- Personalisation promoted "偷偷" above the target.

Evidence:

- The promoted candidate received limited historical evidence.
- Context evidence was absent.

Interpretation:

A weak frequency signal changed an already reasonable baseline ranking.

This represents frequency bias.

Category:

frequency_bias

Implication:

Personalisation should consider evidence confidence and contextual relevance
before overriding a strong baseline ranking.

---

# 3. Wrong-user Comparison

The correct-user and wrong-user results were close:

- Correct-user:
  - Top-1: 75.27%
  - MRR: 0.8277

- Wrong-user:
  - Top-1: 74.95%
  - MRR: 0.8252

The analysed examples suggest that both histories can provide similar
frequency evidence.

Observation:

- Improvements often come from general candidate frequency.
- User identity contributes less than expected.
- Context-specific evidence is rarely activated.

Interpretation:

The current model captures general lexical preference more strongly than
individual writing style.

This explains why using another author's history does not produce a large
performance difference.

---

# 4. Main Findings

The error analysis identifies three main limitations:

## 4.1 Frequency evidence can help but is insufficient

Historical frequency can correct some baseline ranking errors.

However, it cannot reliably resolve semantic ambiguity.

---

## 4.2 Context evidence is too sparse

Although the model includes context evidence, the current implementation uses
exact context matching.

In analysed examples, context evidence is usually zero.

Therefore, the model behaves mainly as a frequency-based reranker.

---

## 4.3 User-specific adaptation is limited

The small difference between correct-user and wrong-user performance suggests
that the current model does not strongly capture individual user preference.

---

# 5. Motivation for Phase 4D

The results motivate a context-aware extension.

The goal is not to replace the transparent scoring framework, but to provide
richer contextual evidence while preserving interpretability.

Future work should investigate:

- similar-context matching;
- confidence-aware contextual evidence;
- gradual fallback from context-specific evidence to broader frequency evidence.

Phase 4C demonstrates the limitation of frequency-only personalisation and
provides the motivation for Phase 4D.