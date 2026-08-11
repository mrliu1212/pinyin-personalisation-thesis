# Phase 4D — Transparent Contextual Memory Retrieval: Results Summary

## Purpose

Phase 4D was introduced after Phase 4C showed that simple frequency-based
personalisation did not outperform the strong Luna Pinyin baseline.

The Phase 4C error analysis indicated that exact-context evidence was too sparse
and that the model mainly relied on global and same-Pinyin frequency. This could
improve some cases, but it also produced harmful frequency-driven reranking.

Phase 4D therefore replaced exact-context matching with transparent retrieval
over previous same-Pinyin interactions.

The main research question was:

> Can similarity-based retrieval from a user's historical contexts provide more
> useful and transparent personalisation evidence than frequency-only
> adaptation?

The frozen Phase 4D design used:

- exact same-Pinyin historical retrieval;
- character unigram + bigram TF-IDF;
- cosine similarity;
- Top-5 positive-similarity historical contexts;
- Luna rank-derived Base utility;
- no test-time history updates.

Two Phase 4D variants were evaluated:

1. `phase_04d_no_gate_correct_user`

   Contextual evidence only:

   `U(y) = C(y)`

2. `phase_04d_full_correct_user`

   Frequency fallback combined with contextual evidence using the maximum
   retrieved similarity as confidence:

   `U(y) = (1 - q)F(y) + qC(y)`

A wrong-user version of the full model used Lu Xun training history instead of
Zhu Ziqing history.

---

## 1. Full Benchmark Results

The full benchmark contains 926 Zhu Ziqing test interactions.

| Condition | Top-1 | Top-3 | Top-5 | Top-10 | MRR | Mean target rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base Luna | 77.97% | 90.06% | 92.01% | 92.55% | 0.8401 | 1.2707 |
| Phase 4C frequency | 75.27% | 90.17% | 92.12% | 92.55% | 0.8277 | 1.2905 |
| Phase 4D no-gate correct-user | **79.05%** | **90.28%** | 92.12% | 92.55% | **0.8461** | **1.2520** |
| Phase 4D full correct-user | 76.67% | 90.17% | 92.12% | 92.55% | 0.8348 | 1.2742 |
| Phase 4D full wrong-user | 76.13% | 90.17% | 92.01% | 92.55% | 0.8315 | 1.2859 |

The strongest Phase 4D condition was the no-gate correct-user model.

Relative to Base, it increased Top-1 accuracy from 77.97% to 79.05%, an
absolute improvement of 1.08 percentage points.

The number of Top-1-correct interactions increased from 722 to 732.

MRR also increased from 0.8401 to 0.8461, while mean target rank improved from
1.2707 to 1.2520.

This is the first evaluated personalisation condition in the project to
outperform the Luna Base on Top-1 accuracy, MRR, and mean target rank.

---

## 2. Rerankable Subset

The rerankable subset contains the 857 interactions whose target already
appeared in the original Luna Top-10 candidate list.

| Condition | Top-1 | Top-3 | Top-5 | Top-10 | MRR | Mean target rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base Luna | 84.25% | 97.32% | 99.42% | 100.00% | 0.9077 | 1.2707 |
| Phase 4C frequency | 81.33% | 97.43% | 99.53% | 100.00% | 0.8943 | 1.2905 |
| Phase 4D no-gate correct-user | **85.41%** | **97.55%** | 99.53% | 100.00% | **0.9142** | **1.2520** |
| Phase 4D full correct-user | 82.85% | 97.43% | 99.53% | 100.00% | 0.9021 | 1.2742 |
| Phase 4D full wrong-user | 82.26% | 97.43% | 99.42% | 100.00% | 0.8985 | 1.2859 |

The same ordering is visible on the rerankable subset.

The no-gate contextual model improves Base Top-1 from 84.25% to 85.41% and MRR
from 0.9077 to 0.9142.

Because Phase 4D only reranks the existing Luna candidate list, Top-10 coverage
cannot improve. The 69 full-benchmark interactions whose targets were absent
from Luna remain unrecoverable at this stage.

---

## 3. Rank-Change Behaviour

On the full benchmark:

### Phase 4C frequency personalisation

- improved: 25
- unchanged: 846
- harmed: 55

### Phase 4D no-gate correct-user

- improved: 20
- unchanged: 895
- harmed: 11

### Phase 4D full correct-user

- improved: 25
- unchanged: 860
- harmed: 41

The no-gate contextual model changes substantially fewer rankings than the
frequency model.

Although it produces slightly fewer improved interactions than Phase 4C, the
number of harmed interactions decreases from 55 to 11.

This suggests that contextual retrieval acts as a more conservative source of
personalisation evidence than broad historical frequency.

The no-gate model therefore achieves its performance gain primarily by avoiding
harmful over-personalisation rather than by aggressively changing large numbers
of candidate rankings.

---

## 4. Context Retrieval Coverage

For the no-gate correct-user condition on the full benchmark:

- evaluated queries: 926
- queries with eligible same-Pinyin history: 436
- queries with non-zero contextual similarity: 374
- percentage with non-zero contextual similarity: 40.39%
- mean retrieved similarity: 0.0364
- mean maximum similarity: 0.0207
- ranking changes with non-zero contextual evidence: 31
- improved cases with contextual evidence: 20
- harmed cases with contextual evidence: 11

Unlike the exact-context mechanism used previously, Phase 4D's similarity-based
retrieval is therefore genuinely activated on a substantial portion of the
benchmark.

This confirms that replacing exact-context equality with similarity-based
retrieval addresses the sparsity problem observed in Phase 4C.

However, the absolute TF-IDF cosine similarities remain small.

---

## 5. No-Gate vs Confidence-Gated Fusion

The main unexpected result is that the no-gate model clearly outperforms the
full confidence-gated model.

The no-gate model uses:

`U(y) = C(y)`

whereas the full model uses:

`U(y) = (1 - q)F(y) + qC(y)`

with `q` equal to the maximum retrieved cosine similarity.

The mean maximum similarity for the correct-user evaluation is only 0.0207.

Therefore, in the full model, contextual evidence generally receives only a
small weight, while the frequency component remains dominant.

Since Phase 4C already showed that frequency-based personalisation is prone to
harmful reranking, the full Phase 4D model partially inherits this weakness.

This provides evidence that raw TF-IDF cosine similarity should not
automatically be interpreted as a calibrated confidence probability.

The failure of the gated variant does not invalidate contextual retrieval.
Instead, the ablation shows that the retrieved contextual evidence itself is
useful, while the specific hand-designed fusion rule is not.

---

## 6. Wrong-User Control

The full correct-user model achieves:

- Top-1: 76.67%
- MRR: 0.8348
- mean target rank: 1.2742

The full wrong-user model achieves:

- Top-1: 76.13%
- MRR: 0.8315
- mean target rank: 1.2859

Correct-user performance is slightly better than wrong-user performance under
the same full fusion architecture.

However, this result should not yet be interpreted as strong evidence of
user-specific contextual personalisation.

The no-gate condition, which is the strongest Phase 4D model, was not evaluated
with wrong-user history in this phase.

In addition, wrong-user retrieval actually produced non-zero contextual
similarity for more queries (44.06% versus 40.39%) and had higher mean
similarities.

Therefore, Phase 4D does not yet establish that TF-IDF contextual similarity
itself reliably distinguishes an author's individual semantic preferences.

A same-model correct-user versus wrong-user comparison would be required for a
clean user-specificity conclusion.

---

## 7. Main Findings

Phase 4D produces four main findings.

### 7.1 Similarity-based context retrieval solves the exact-context sparsity problem

Phase 4C's exact-context feature was effectively inactive in real chronological
evaluation.

Phase 4D retrieves non-zero contextual evidence for approximately 40% of test
queries.

The context mechanism therefore becomes operational rather than merely existing
in the model specification.

### 7.2 Contextual retrieval is more reliable than frequency-only personalisation

Frequency-only Phase 4C reduced Base Top-1 accuracy from 77.97% to 75.27%.

The Phase 4D no-gate contextual model instead improves Top-1 to 79.05%.

It also reduces harmful rank changes from 55 to 11.

This suggests that contextual historical evidence is a more selective and
useful source of adaptation than global or same-Pinyin frequency alone.

### 7.3 Raw cosine similarity is not a suitable confidence gate

The full Phase 4D fusion model remains below the Base.

The observed similarity magnitudes are too small for direct use as a confidence
coefficient, causing the full model to fall back primarily to the weaker
frequency component.

This demonstrates an important distinction between:

- retrieval similarity;
- calibrated confidence in a ranking decision.

They should not be assumed to be equivalent.

### 7.4 Candidate-generation coverage remains an upper bound

All Phase 4D systems retain the same Top-10 accuracy of 92.55%.

The remaining 69 missing-target interactions cannot be recovered by reranking
because the correct candidate is absent from Luna's candidate list.

Future models should distinguish improvements in candidate generation or
candidate augmentation from improvements in reranking.

---

## 8. Interpretation

Phase 4D provides the first positive evidence that user history can improve the
existing Luna candidate ranking when historical evidence is conditioned on
context rather than used only as broad frequency.

The strongest model is intentionally simple and conservative:

Luna Base
+
same-Pinyin contextual memory retrieval
+
transparent context evidence.

Its improvement is modest, and statistical significance has not yet been
established.

Therefore, the result should not be described as demonstrating general
superiority over modern IMEs or commercial input methods.

Instead, Phase 4D establishes a useful experimental result within the current
benchmark:

> transparent contextual retrieval can improve ranking over a strong
> non-personalised Luna baseline while producing substantially fewer harmful
> interventions than frequency-only personalisation.

At the same time, the relatively shallow character-level TF-IDF representation
limits the model's semantic understanding.

Phase 4D therefore serves as both:

1. a successful transparent contextual baseline; and
2. motivation for a richer semantic context and personal-memory model.

---

## 9. Decision

Phase 4D is accepted as a completed experimental stage.

The Phase 4D no-gate system is retained as the strongest transparent lexical
context baseline.

The confidence-gated model is retained as a negative ablation rather than being
silently replaced or tuned on the observed test results.

No Phase 4D parameters will be tuned after seeing the final Zhu Ziqing test
performance.

The next method extension will be implemented as a separately named phase,
preserving Phase 4D results unchanged.

The next phase will investigate whether stronger semantic context modelling and
semantic personal memory can improve accuracy while retaining an auditable
factor-level personalisation layer.