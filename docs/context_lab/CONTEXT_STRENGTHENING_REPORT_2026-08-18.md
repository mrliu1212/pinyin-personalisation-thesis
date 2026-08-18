# Context Strengthening Report

**Project:** Transparent and User-Controllable Personalisation for Chinese Pinyin Input  
**Date:** 2026-08-18  
**Scope:** Full+Short, H5000, exploratory three-author Context Lab  
**Authors:** Etinjat, Re_spectators, breaddddd

## 1. Purpose

The original M1 method uses a generic BGE sentence embedding and cosine similarity to retrieve same-Pinyin personal history. Earlier diagnostics showed that useful Gold history was often present, but M1 still made many wrong decisions when several historical targets competed.

The purpose of Context Strengthening was therefore to test whether a more local context representation could improve contextual discrimination.

The Generic PinyinGPT system was kept frozen. Only the context used by the personal-memory matcher was changed.

## 2. Diagnostic A: what problem were we trying to solve?

For the three exploratory authors on Full+Short Test:

- History Available: 1,891 / 3,000
- Gold target exists in legal history: 1,698 / 3,000

Original full-context retrieval:

| Subset | R@1 | R@3 | R@5 | R@10 | R@20 |
|---|---:|---:|---:|---:|---:|
| Overall | 85.22% | 93.70% | 96.17% | 98.53% | 99.41% |
| Ambiguous | 67.10% | 85.98% | 91.48% | 96.72% | 98.69% |
| Conflict | 24.28% | 52.60% | 69.36% | 86.13% | 94.80% |

This showed that retrieval coverage was already high at larger K. The more important problem was contextual discrimination: when several possible historical targets existed, the correct history was not reliably ranked first.

The earlier case analysis supported the same conclusion. Strong M1/M2 regressions often occurred even when Gold evidence was present. The main failure was therefore not simply "Gold history is unavailable", but "the system does not reliably decide which historical evidence should be trusted".

## 3. Local-context hypothesis

The hypothesis was that whole-context semantic embeddings may over-weight broad topic similarity. Pinyin input prediction may instead depend more on local lexical, syntactic, and continuation context.

We therefore tested symmetric suffix windows for the personal-memory matcher:

- Full context
- last 64 characters
- last 16 characters
- last 8 characters

The Generic PinyinGPT context was not changed.

### Important methodological note

The first Full/64/16 comparison had already been run on Test and was treated only as exploratory diagnostic evidence. It was not used as a formal end-to-end hyperparameter search.

The formal representation-selection comparison was then performed on the Dev tune partition.


## 3A. Exploratory Test local-context diagnostic

Before the formal Dev selection, an exploratory diagnostic was run on the same three-author Full+Short / H5000 Test population used by Diagnostic A.

These Test results were treated as **hypothesis-generating only** and were not used to select the final context window.

| Setting | Subset | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|
| Full | Overall | 85.22% | 96.17% | 98.53% |
| ctx64 | Overall | 85.22% | 96.17% | 98.70% |
| ctx16 | Overall | 87.10% | 97.41% | 98.70% |
| Full | Ambiguous | 67.10% | 91.48% | 96.72% |
| ctx64 | Ambiguous | 67.10% | 91.48% | 97.12% |
| ctx16 | Ambiguous | 71.30% | 94.23% | 97.12% |
| Full | Conflict | 24.28% | 69.36% | 86.13% |
| ctx64 | Conflict | 28.32% | 67.05% | 89.02% |
| ctx16 | Conflict | 36.99% | 76.88% | 87.28% |

The most notable exploratory result was the Conflict R@1 increase from 24.28% with Full context to 36.99% with ctx16. This supported the hypothesis that whole-context semantic similarity might dilute more local cues useful for IME personal-history matching.

However, the effect was not monotonic across retrieval depths, and the Test Gold outcomes had already been inspected. Therefore these results were not used for formal parameter selection.

This distinction is important because the exploratory Test diagnostic visually favored ctx16, whereas the independent Dev tune selection later favored ctx64 under the pre-defined Macro-author Ambiguous R@1 criterion. The final representation choice was therefore based on Dev rather than on the previously observed Test diagnostic.


## 4. Why Macro-author Ambiguous R@1 was used

The primary representation-selection metric was **Macro-author Ambiguous R@1**.

### Why R@1?

R@1 asks whether the highest-ranked retrieved historical interaction supports the Gold target.

Diagnostic A showed that Gold history was often already present in the upper retrieval set. Therefore the main issue was not broad retrieval coverage, but whether the matcher could put the most useful evidence first.

### Why Ambiguous?

Ambiguous cases contain at least two distinct historical targets for the same Pinyin. These are the cases where Context must genuinely distinguish between competing personal usages.

If only one historical target exists, Frequency already has little target-level ambiguity to resolve.

### Why Macro-author?

The metric is calculated per author and then averaged equally across authors. This prevents an author with more ambiguous samples from dominating the result and better reflects personalisation across users.

Overall and Conflict retrieval metrics were retained as secondary diagnostics.

## 5. Dev tune context-window results

### Retrieval-stage results

| Window | Overall Macro R@1 | Ambiguous Macro R@1 | Conflict Macro R@1 |
|---|---:|---:|---:|
| Full | 88.41% | 73.03% | 34.53% |
| ctx64 | 88.36% | **75.09%** | **41.05%** |
| ctx16 | 87.16% | 72.68% | 23.39% |
| ctx8 | 86.95% | 71.65% | 25.54% |

Under the pre-defined representation-selection criterion, **ctx64 was selected**.

This selection means that ctx64 was the strongest retrieval representation under the chosen diagnostic metric. It does not claim that ctx64 is mathematically guaranteed to be the globally best end-to-end configuration over all possible combinations of window, top_n, and lambda.

## 6. Dev evaluation retrieval check

After selecting ctx64, the chronologically later Dev evaluation partition was also checked at the retrieval level.

| Subset | Micro R@1 | Macro R@1 | Micro R@5 | Micro R@10 |
|---|---:|---:|---:|---:|
| Overall | 95.11% | 90.80% | 99.15% | 99.55% |
| Ambiguous | 84.49% | 79.41% | 97.31% | 98.58% |
| Conflict | 43.51% | 41.66% | 80.75% | 88.70% |

The selected ctx64 representation therefore did not collapse on the later Dev works. In particular, Ambiguous Macro R@1 increased from 75.09% on Dev tune to 79.41% on Dev evaluation.

No downstream M1 hyperparameters were changed using these evaluation results.

## 7. M1 hyperparameter tuning under ctx64

After fixing ctx64, M1 was tuned on the Dev tune partition.

The original search grid was:

- top_n = 1, 3, 5, 10, 20
- lambda_memory = 0, 0.25, 0.5, 1, 2, 4

Selection criterion:

- primary: Macro-author final candidate Top-1
- tie-break: lower lambda_memory, then lower top_n

The selected configuration was:

- **top_n = 3**
- **lambda_memory = 4.0**
- Macro-author Top-1 = **77.08%**
- Macro-author Top-3 = 86.03%
- MRR@10 = 82.03%

Because lambda_memory = 4.0 was the upper boundary of the original grid, an adaptive boundary check was performed with lambda_memory = 8.0.

The selected configuration remained:

- **top_n = 3**
- **lambda_memory = 4.0**

Therefore the original upper boundary did not truncate the optimum. No lambda values above 8 were tested.

The frozen ctx64 M1 configuration was therefore:

- context window = 64 characters
- history budget = 5000
- top_n = 3
- lambda_memory = 4.0

## 8. Final Test

The final M1 Test used the frozen Generic T1 candidate pool and the previously generated ctx64 Test embedding cache. PinyinGPT was not rerun and no Test hyperparameter search was performed.

Population:

- Full+Short
- H5000
- 3 exploratory authors
- 3,000 Test anchors

### New ctx64 M1 Test result

| Subset | n | Generic Top-1 | ctx64 M1 Top-1 | ctx64 M1 Top-3 | ctx64 M1 MRR@10 |
|---|---:|---:|---:|---:|---:|
| Overall | 3000 | 77.60% | 79.77% | 91.93% | 0.8616 |
| History Available | 1891 | 79.85% | 83.29% | 94.92% | 0.8925 |
| Ambiguous | 836 | 74.88% | 74.40% | 93.42% | 0.8407 |
| Conflict | 233 | 62.66% | 33.05% | 82.83% | 0.5828 |

Macro-author Top-1:

| Subset | Generic | ctx64 M1 | Delta |
|---|---:|---:|---:|
| Overall | 77.60% | 79.77% | +2.17pp |
| History Available | 79.03% | 81.41% | +2.38pp |
| Ambiguous | 75.41% | 75.69% | +0.29pp |
| Conflict | 53.57% | 25.15% | -28.43pp |

## 9. Comparison with Frequency and the old M1

The most important comparison is not only against Generic, but also against the simple Frequency baseline and the original full-context M1.

The following values are micro Top-1 on the same three-author Full+Short Test population:

| Subset | Generic | F | Old M1 Full Context | New M1 ctx64 |
|---|---:|---:|---:|---:|
| Overall | 77.60% | **81.07%** | 79.83% | 79.77% |
| Ambiguous | 74.88% | **79.07%** | 74.64% | 74.40% |
| Conflict | 62.66% | **41.63%** | 32.19% | 33.05% |

Change from old M1 to ctx64 M1:

- Overall: 79.83% -> 79.77% (-0.06pp)
- Ambiguous: 74.64% -> 74.40% (-0.24pp)
- Conflict: 32.19% -> 33.05% (+0.86pp)

The local-context modification therefore produced **almost no end-to-end improvement over the original M1**.

Frequency remained clearly stronger:

- Overall: F 81.07% vs ctx64 M1 79.77% (+1.30pp for F)
- Ambiguous: F 79.07% vs ctx64 M1 74.40% (+4.67pp for F)
- Conflict: F 41.63% vs ctx64 M1 33.05% (+8.58pp for F)

## 10. Conclusion

The Context Strengthening experiment gives a negative but useful result.

Reducing the personal-memory context to the most recent 64 characters improved retrieval-level discrimination. In particular, the Dev tune Macro-author Ambiguous R@1 increased from 73.03% with the full context to 75.09% with ctx64, and Conflict Macro R@1 increased from 34.53% to 41.05%.

However, this retrieval improvement did **not** translate into better end-to-end M1 personalisation.

On Test, the new ctx64 M1:

- did not materially improve over the old full-context M1;
- remained below the Frequency baseline;
- still performed poorly on Conflict cases.

Therefore, the evidence does not support the hypothesis that context length alone is the main problem.

A more precise conclusion is:

> **Generic semantic embeddings combined with cosine similarity are not sufficiently aligned with the fine-grained contextual distinctions required for Pinyin candidate selection.**

The matcher can often retrieve relevant personal history, but the resulting semantic similarity signal does not reliably identify which historical usage should influence the current candidate ranking.

This means the remaining bottleneck is not simply retrieval coverage. It is the quality and task alignment of the contextual matching signal and the way that signal is converted into candidate-level support.

## 11. What was completed in this stage

This stage completed the following work:

1. Diagnosed M1 retrieval failures using Overall, Ambiguous, and Conflict subsets.
2. Tested local personal-memory context windows Full / 64 / 16 / 8.
3. Defined and justified Macro-author Ambiguous R@1 as the representation-selection metric.
4. Selected ctx64 on Dev tune.
5. Checked ctx64 retrieval on chronologically later Dev evaluation works.
6. Retuned M1 top_n and lambda_memory under ctx64.
7. Performed a lambda upper-bound check with lambda_memory = 8.
8. Froze ctx64, top_n = 3, lambda_memory = 4.
9. Reused the frozen Generic Test candidate pool.
10. Ran the final three-author Full+Short H5000 M1 Test.
11. Compared the new M1 against Generic, Frequency, and the original full-context M1.
12. Concluded that local context improves retrieval diagnostics but does not solve the end-to-end Context personalisation problem.

## 12. Experimental status

This Context Strengthening route is now considered complete for the current M1 cosine-similarity formulation.

The Test result should be treated as final for:

- window = 64
- H5000
- top_n = 3
- lambda_memory = 4

No further tuning of these parameters should be performed using this Test result.
