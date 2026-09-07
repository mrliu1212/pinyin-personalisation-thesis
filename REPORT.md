# Research Report

## 1. Research goal

Chinese Pinyin input is fundamentally a ranking problem. A single Pinyin sequence may correspond to many possible Chinese words or phrases, and the correct output depends strongly on context and on the individual user's writing habits.

A generic input model can learn broad language regularities, but it cannot fully represent the vocabulary, topics, names, expressions, and repeated preferences of one user.

This project studies whether candidate ranking can be improved through personalisation while also making that personalisation transparent and controllable.

The main questions are:

- How much can user-specific adaptation improve Pinyin candidate ranking?
- How much can explicit user history recover candidates that a generic neural model misses?
- Are neural adaptation and explicit memory complementary?
- Can personalisation be designed so that users can inspect or change its effects?

## 2. Experimental data

Long-form modern Chinese writing was used to approximate user histories.

Five authors were selected as proxy users:

- Re_spectators
- Etinjat
- Agent Phage
- QBLevi
- breaddddd

Using authors as proxy users provides long, identifiable histories while keeping each user's textual history separated.

The final dataset contains 210,000 interactions:

| Split | Interactions |
|---|---:|
| Fit | 150,000 |
| Validation | 20,000 |
| Test | 40,000 |
| Total | 210,000 |

The split is chronological and performed at whole-work level.

## 3. Generic neural baseline

The neural backbone is PinyinGPT2-Concat:

`aihijo/transformers4ime-pinyingpt-concat`

On the frozen 40,000-row Test set:

- Top-1 = 34.92%
- Top-5 = 49.86%
- Top-10 = 53.54%
- MRR = 0.4132

## 4. Frequency personalisation

A simple frequency baseline uses at most the latest 5,000 strictly prior interactions from the same proxy user.

Historical target frequency is combined with the Generic ranking using reciprocal-rank fusion.

Top-1 improves from 34.92% to 36.82%.

## 5. Explicit Memory

Explicit Memory keeps recent user history outside the neural model.

For every query, it uses the latest 5,000 strictly prior raw interactions from the same user.

Two recovery mechanisms operate in parallel:

- Exact recovery
- Composition recovery

In the Generic + Explicit system, Explicit Memory newly recovered 3,490 Test targets that were absent from the Generic neural candidate pool.

Generic + Explicit reaches:

- Top-1 = 45.12%
- Top-10 = 61.68%
- MRR = 0.5090

## 6. Adapter personalisation

A user-specific serial residual Adapter is inserted after every transformer block while the base PinyinGPT parameters remain frozen.

The final configuration uses:

- 12 Adapter layers
- hidden size 768
- bottleneck size 48
- ReLU
- one training epoch
- AdamW
- learning rate 5e-4

The Adapter system reaches:

- Top-1 = 64.40%
- Top-10 = 80.41%
- MRR = 0.7024

## 7. Hybrid system

The final system combines Adapter neural candidates, H5000 Explicit Memory, Exact recovery, Composition recovery, memory-only candidate insertion, and a Rich30 LambdaMART reranker.

| Metric | Generic | Hybrid | Gain |
|---|---:|---:|---:|
| Top-1 | 34.92% | 65.68% | +30.77 pp |
| Top-3 | 46.17% | 76.22% | +30.05 pp |
| Top-5 | 49.86% | 79.01% | +29.16 pp |
| Top-10 | 53.54% | 81.44% | +27.91 pp |
| MRR | 0.4132 | 0.7149 | +0.3017 |

The Adapter supplies most of the gain, while Explicit Memory remains complementary.

## 8. Candidate recovery

For Generic + Explicit, 3,490 new targets are recovered.

For Adapter + Explicit, 669 additional targets are recovered beyond the Adapter candidate pool.

The final system's remaining Top-10 failures are dominated by candidate-pool coverage rather than reranking errors.

## 9. Transparency and controllability

The Adapter is adaptive and powerful, but its learned preferences are distributed across parameters and are therefore less directly traceable.

Explicit Memory is directly connected to observable historical evidence and supports direct interventions such as deleting, restoring, or suppressing remembered preferences.

The repository currently includes the compact R4 reversible preference adaptation report. Other compact controllability artifacts from the thesis will be added in a later repository version.

## 10. Robustness

The Generic → Adapter Top-1 improvement is approximately +29.48 percentage points.

The Adapter → Hybrid improvement is approximately +1.29 percentage points and is statistically robust.

## 11. Supplementary EDA

Supplementary lexical analysis was performed only after the final evaluation was frozen.

It was not used for model selection or Test tuning.

## 12. Main conclusion

The final evaluation supports three main conclusions:

1. Personalisation can substantially improve Chinese Pinyin candidate ranking.
2. Neural adaptation and explicit memory solve different parts of the problem.
3. A hybrid design can combine adaptive performance with more direct transparency and control.

The final Hybrid system improves Top-1 accuracy from 34.92% to 65.68% on the frozen five-user Test set.

## 13. Limitations

The evaluation uses authors as proxy users rather than real deployed IME users.

Typing interactions are simulated from written text.

The study does not directly compare the system against commercial input methods.

Deployment latency was not evaluated in a production environment.

The final user-profile visualisation discussed in the thesis is conceptual rather than an implemented and evaluated interface.

## 14. Repository scope

This cleaned repository intentionally excludes large model checkpoints, raw source corpora, complete row-level prediction surfaces, deprecated pilots, calibration experiments, and earlier Phase 1–4 pipelines.

It preserves the final predictive experiment, compact evidence, frozen evaluation protocol, reproducibility code, statistical analysis, and supplementary EDA.
