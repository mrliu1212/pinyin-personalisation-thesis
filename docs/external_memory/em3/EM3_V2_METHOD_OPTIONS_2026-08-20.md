# EM3-v2 Method Options and Evidence Base

**Project:** Transparent and User-Controllable Personalisation for Chinese Pinyin Input
**Status:** Design note; no EM3-v2 method is frozen yet
**Date:** 2026-08-20

## 1. Purpose

This note records the feasible next-stage methods for EM3 and the literature that motivates them. The goal is to move beyond the current pointwise BCE baseline while preserving causal history semantics and keeping runtime suitable for an interactive Pinyin IME.

The current EM3-BCE experiment should be retained as a baseline. However, if the author set is changed for EM3-v2, a BCE baseline should be rerun on the new author set before making direct method comparisons.

## 2. Fixed task semantics

For a current query \(q\):

- history is from the same author/user;
- only strictly-prior interactions are legal;
- history budget is H5000 before Pinyin filtering;
- positive history: same segmented Pinyin and \(target(h)=gold(q)\);
- negative history: same segmented Pinyin and \(target(h)\neq gold(q)\);
- Test is never used for training or model selection.

These are project constraints, not claims taken from the papers below.

---

# 3. Option A — Localized Contrastive Estimation (LCE)

## Literature basis

Gao, Dai, and Callan (2021) study BERT rerankers in a two-stage retrieval pipeline. Their vanilla training uses independent binary cross-entropy:

\[
\mathcal{L}_{v}:=
\begin{cases}
\mathrm{BCE}(\mathrm{score}(q,d),+) & d\text{ is positive}\\
\mathrm{BCE}(\mathrm{score}(q,d),-) & d\text{ is negative}
\end{cases}
\]

Their query-level LCE objective is:

\[
\mathcal{L}_{q}
=
-\log
\frac{\exp(\mathrm{dist}(q,d_q^+))}
{\sum_{d\in G_q}\exp(\mathrm{dist}(q,d))}
\]

and the batch loss is:

\[
\mathcal{L}_{\mathrm{LCE}}
=
\frac{1}{|Q|}
\sum_{q\in Q,\,G_q\sim R_q^m}
-\log
\frac{\exp(\mathrm{dist}(q,d_q^+))}
{\sum_{d\in G_q}\exp(\mathrm{dist}(q,d))}
\]

**Short original excerpt:** “We localize negative sample distribution by sampling from the target retriever top results.” — Gao et al. (2021)

The paper reports a large improvement when moving from one positive + one negative to one positive + three negatives, with some additional improvement from larger groups.

## Adaptation to this project

\[
q=(\text{current context},\text{current Pinyin})
\]

and:

\[
G_q=\{h^+,h^-_1,\ldots,h^-_k\}
\]

The Cross-Encoder should give the correct-target history the highest score in the group.

## Why it may help

The current BCE baseline asks whether each \((q,h)\) pair is positive or negative independently. The downstream problem is ranking: useful histories must outrank confusing wrong-target histories. LCE trains that competition directly.

## Advantages

- Better alignment with reranking than independent binary classification.
- Direct evidence from a BERT-reranker study.
- No additional inference-time cost if the model and reranking depth are unchanged.
- Clean ablation against the BCE baseline.

## Main risk

Standard LCE is naturally single-positive per group and does not fully solve how to exploit many valid same-target histories.

**Reference:** Luyu Gao, Zhuyun Dai, Jamie Callan. *Rethink Training of BERT Rerankers in Multi-Stage Retrieval Pipeline*. 2021. arXiv:2101.08751.

---

# 4. Option B — Retriever-localized hard negatives

## Literature basis

LCE samples negatives from top results produced by the target retriever. Official FlagEmbedding tooling follows the same practical pattern.

**Short original excerpt:** “retrieve top-k documents for each query, and random sample negatives from the top-k documents” — FlagEmbedding documentation.

## Adaptation to this project

Legal negative pool:

\[
H_q^-=
\{h:\text{same user, prior, H5000, same Pinyin, }target(h)\neq gold(q)\}
\]

Frozen hidden-state kNN then scores:

\[
r(q,h)=\cos(z_q,z_h)
\]

For cosine similarity, **higher is harder**.

Proposed procedure:

1. build the legal same-Pinyin wrong-target pool;
2. score with frozen hidden-state kNN;
3. sort cosine descending;
4. keep a high-ranked pool such as Top-20 or Top-30;
5. reproducibly sample negatives from that pool;
6. do not repeat the exact same history row inside a group;
7. allow the same wrong target multiple times if the histories are different contexts.

Example:

```text
使用 H1  cosine .95
使用 H2  cosine .93
试用 H3  cosine .91
食用 H4  cosine .72
```

Both `使用 H1` and `使用 H2` can be valid hard negatives.

## Why it may help

The reranker should learn from mistakes it is likely to see at deployment. Easy random negatives can be solved using superficial cues and may teach little about the real decision boundary.

## Advantages

- Better train/inference distribution alignment.
- Makes multiple histories of the same wrong target useful when contexts differ.
- Mining is offline and adds no user-time latency by itself.

## Open hyperparameter

Top-20 vs Top-30 should be audited/tuned on Dev. The important principle is high-ranked localized sampling, not one fixed number.

---

# 5. Option C — Multi-positive Log-Sum-Exp Pairwise loss (LSEPair)

## Literature basis

Wang, Tang, Zhang, Guo, and Bi (KDD 2026) systematically study training dense retrievers when a query has multiple valid positives.

**Short original excerpt:** “LSEPair consistently achieves superior robustness and performance across settings.” — Wang et al. (2026)

The paper defines:

\[
\boxed{
\mathcal{L}_{\mathrm{LSEPair}}
=
\log\left(
1+
\sum_{p\in P_q}
\sum_{n\in N_q}
\exp(s(q,n)-s(q,p))
\right)
}
\]

It therefore aggregates all positive-negative ordering constraints:

\[
s(q,p)>s(q,n),
\qquad
\forall p\in P_q,\;n\in N_q
\]

The paper shows that lower-scoring “hard” positives receive stronger gradients.

## What model did the 2026 paper train?

The paper studies **dense retrievers / dual encoders**, not Cross-Encoder rerankers. Applying LSEPair to `BAAI/bge-reranker-base` is therefore our adaptation of the objective.

## Adaptation to this project

\[
P_q=\{\text{legal prior histories with target = current gold}\}
\]

\[
N_q=\{\text{retriever-localized legal wrong-target histories}\}
\]

The Cross-Encoder computes one score per selected history, then LSEPair is computed over all positive-negative score differences.

## Computational cost

If there are \(|P|\) positives and \(|N|\) negatives, the loss contains \(|P||N|\) scalar comparisons, but the expensive Transformer work is only \(|P|+|N|\) Cross-Encoder forward scores.

So the loss arithmetic is cheap; training cost mainly grows with how many histories are actually scored.

## Why it may help

The task naturally has multiple histories supporting the same current gold. LSEPair uses those contexts together instead of pretending there is only one positive.

## Advantages

- Direct use of multi-positive supervision.
- Query-level objective.
- Strong pairwise ranking interpretation.
- No extra inference-time cost once the model is trained.

## Risks

- Published evidence is from dense retrievers, not Cross-Encoders.
- Low-scoring positives receive strong pressure. If “same target” includes contextually weak positives, this may over-promote noisy histories.
- Scoring every positive of a frequent target can still make training expensive.

## Proposed mitigation

Use representative / balanced positive selection before LSEPair, based on a distribution audit and diversity-aware selection.

**Reference:** Benben Wang, Minghao Tang, Hengran Zhang, Jiafeng Guo, Keping Bi. *Training Dense Retrievers with Multiple Positive Passages*. KDD 2026. arXiv:2602.12727.

---

# 6. Option D — Coreset / diversity-aware deletion

## Motivation

Equal weighting fixes optimization imbalance but does not reduce GPU work. Downsampling can both balance the data and reduce Cross-Encoder training cost.

## Literature basis

Sener and Savarese formulate active-learning selection as a core-set problem.

**Short original excerpt:** “choosing set of points such that a model learned over the selected subset is competitive” — Sener & Savarese (2018)

A suitable adaptation is greedy farthest-first / k-center-style selection in frozen hidden-state space.

Let \(z_h\) be the history representation and \(S\) the retained set:

\[
h^*
=
\arg\max_{h\notin S}
\min_{s\in S}
\left(1-\cos(z_h,z_s)\right)
\]

This repeatedly retains the history least covered by the current subset, encouraging context diversity.

## Proposed balancing unit

Operate inside:

\[
(\text{author},\text{Pinyin},\text{target})
\]

groups.

Do **not** automatically force every target down to the count of the rarest target if this would discard most of the data. First audit the distribution, then define:

- a minimum-support rule;
- a per-Pinyin target quota or capped quota;
- diversity-aware selection when a group must be reduced.

The exact quota is project-specific and must be Dev-validated.

## Advantages

- Reduces training time.
- Reduces redundant examples.
- Preserves context diversity better than random deletion.
- Can be audited before model training.

## Risk

The frozen hidden-state geometry may consider two histories redundant even if the Cross-Encoder would distinguish them. Compare against random-balanced deletion as a control.

**Reference:** Ozan Sener, Silvio Savarese. *Active Learning for Convolutional Neural Networks: A Core-Set Approach*. ICLR 2018. arXiv:1708.00489.

---

# 7. Option E — RankNet

## Literature basis

Burges et al. introduced RankNet as a neural pairwise learning-to-rank method.

**Short original excerpt:** “we propose a simple probabilistic cost function, and we introduce RankNet” — Burges et al. (2005)

For a positive \(p\) and negative \(n\):

\[
P(p>n)=\sigma(s_p-s_n)
\]

\[
\mathcal L_{\mathrm{RankNet}}
=
-\log\sigma(s_p-s_n)
=
\log(1+\exp(s_n-s_p))
\]

## Why it may help

It directly optimizes the ordering relation needed by reranking.

## Advantages

- Classic and well-established.
- Easy to explain.
- Same inference cost as BCE/LCE/LSEPair for the same model architecture.

## Limitation

It treats positive-negative relations pairwise rather than using the entire query group jointly. LCE and LSEPair exploit grouped structure more naturally.

**Reference:** Chris J.C. Burges et al. *Learning to Rank using Gradient Descent*. ICML 2005 / Microsoft Research.

---

# 8. Option F — LambdaLoss / LambdaRank-style metric weighting

## Literature basis

Wang et al. (2018) present LambdaLoss as a probabilistic framework connecting learning-to-rank losses to ranking metrics.

**Short original excerpt:** “a probabilistic framework for ranking metric optimization.” — Wang et al. (2018)

Conceptually:

\[
\mathcal L
=
\sum_{i,j}
w_{ij}
\log(1+\exp(-(s_i-s_j)))
\]

where \(w_{ij}\) can depend on the ranking-metric impact of swapping items \(i\) and \(j\), such as \(|\Delta\mathrm{NDCG}_{ij}|\).

## Why it may help

Top-ranked mistakes matter more to an IME than low-ranked changes that the user never sees.

## Why not first

It becomes most meaningful when training examples resemble a real Top-K candidate list. For small artificially sampled groups, LCE/LSEPair are cleaner first experiments.

## Runtime

Training objective only. No intrinsic inference-time penalty.

**Reference:** Xuanhui Wang, Cheng Li, Nadav Golbandi, Mike Bendersky, Marc Najork. *The LambdaLoss Framework for Ranking Metric Optimization*. CIKM 2018.

---

# 9. EM4 candidate — Target-level evidence aggregation

## Important status

The proposed target-level EM4 is **not copied from one Pinyin paper**. It is a project-specific adaptation motivated by prior work on aggregating multiple local relevance signals into a higher-level score.

## PARADE inspiration

PARADE studies how passage-level relevance signals can be aggregated into a document-level ranking score.

**Short original excerpt:** “aggregating relevance signals from a document's passages into a final ranking score.” — Li et al. (2020)

Project analogy:

```text
PARADE                         This project
Document                       Chinese target
 ├─ passage 1                   ├─ history 1
 ├─ passage 2                   ├─ history 2
 └─ passage 3                   └─ history 3
      ↓                              ↓
document score                  target evidence score
```

## Multiple-Instance Learning inspiration

MIL provides a second conceptual basis: many local instances form a bag, and instance evidence supports a bag-level prediction.

For this project:

\[
\text{target}=\text{bag},
\qquad
\text{history interactions}=\text{instances}
\]

## Possible aggregation

After retrieving and Cross-Encoding a **small Top-K**:

\[
S_t(q)
=
\operatorname{LogSumExp}_{h\in H_t^{(K)}} s(q,h)
\]

or compare max, mean, top-r mean, and LogSumExp on Dev.

## Runtime rule

**Exclude any design that Cross-Encodes all H5000 histories at user time.**

Deployable form:

```text
H5000
  ↓ cheap retriever
Top-K
  ↓ Cross-Encoder only on K histories
history scores
  ↓ cheap target grouping / aggregation
target scores
```

Grouping and LogSumExp are cheap; Cross-Encoder forwards dominate latency.

## Advantages

- Directly ranks the final object of interest: Chinese targets.
- Can combine repeated evidence from the same target.
- Adds little aggregation latency if K is fixed.

## Risk

Raw aggregation may unintentionally reward targets simply because they have more histories. This must be separated from the explicit frequency-personalization component or normalized.

**References:**
Canjia Li et al. *PARADE: Passage Representation Aggregation for Document Reranking*. arXiv:2008.09093.
Yunjie Ji et al. *Diversified Multiple Instance Learning for Document-Level Multi-Aspect Sentiment Classification*. EMNLP 2020.

---

# 10. Author and within-Pinyin target balancing

## Author set

The current exploratory three-author set is strongly imbalanced. A plausible cleaner next set is:

- Etinjat
- Agent Phage
- breaddddd

because their available training populations are closer in scale. MScarlet should remain excluded from this clean comparison until the script-normalization confound is repaired.

This is a project-data decision, not literature-derived.

## Within-Pinyin target balance

Next audit should measure, per author and Pinyin:

- number of distinct current-query targets;
- query count per target;
- positive-history count per query;
- legal wrong-target history count;
- concentration of the largest target;
- how much strict equal-count downsampling would discard.

Current design preference:

- different Pinyins do not need equal global frequency;
- within the same Pinyin, competing targets should have comparable training influence;
- where deletion is used, retain a diversity-aware coreset rather than random examples;
- extremely rare targets should not force all common targets down to a tiny count without a minimum-support rule.

---

# 11. Training-time versus inference-time cost

| Method | Training cost | User-time inference cost if architecture/K unchanged | Keep? |
|---|---:|---:|---|
| BCE | baseline | none extra | baseline |
| LCE | small grouped-loss overhead | none extra | yes |
| Hard-negative mining | offline retrieval/mining | none extra beyond deployed retriever | yes |
| LSEPair | more forward scores if more positives/negatives are selected | none extra | yes |
| Coreset balancing | offline selection; then reduces training work | none | yes |
| RankNet | pairwise training overhead | none extra | optional |
| LambdaLoss | list/metric-weighted training overhead | none extra | optional later |
| EM4 Top-K target aggregation | small grouping cost | cheap aggregation only | possible |
| Cross-Encode all H5000 | huge | huge | **exclude** |

The loss function itself is not the deployment bottleneck. Runtime is mainly controlled by the number \(K\) of histories passed through the Cross-Encoder.

---

# 12. Recommended experiment sequence

## Stage 0 — Finish current BCE baseline

Keep the current pointwise BCE run as an exploratory baseline.

## Stage 1 — Freeze a cleaner author population

Audit and likely use:

- Etinjat
- Agent Phage
- breaddddd

If the author set changes, rerun BCE on the new set for a fair comparison.

## Stage 2 — Balance and reduce redundancy

1. Audit \((author,Pinyin,target)\) counts.
2. Define minimum-support and quota rules.
3. Compare random-balanced deletion against hidden-state diversity-aware coreset deletion.
4. Freeze the retained population before comparing losses.

## Stage 3 — Build localized hard-negative pools

For each legal Train query:

1. strictly-prior H5000;
2. exact same segmented Pinyin;
3. wrong-target histories only;
4. frozen hidden-state kNN cosine;
5. sort descending;
6. create Top-K hard-negative pool;
7. reproducibly sample from that pool;
8. allow repeated target surfaces, but not repeated exact history rows.

## Stage 4 — Loss experiments on the same frozen data

### EM3-v2-A — BCE-HN
Same data + hard negatives, pointwise BCE. Isolates the effect of negative mining.

### EM3-v2-B — LCE-HN
One positive competes directly with localized hard negatives.

### EM3-v2-C — LSEPair-HN
Multiple representative positives compete against multiple localized hard negatives in one query-level objective.

### Optional EM3-v2-D — RankNet-HN
Classical pairwise ranking baseline.

### Optional EM3-v2-E — LambdaLoss-HN
Only after a realistic Top-K candidate-list training format exists.

## Stage 5 — EM4 only if fusion remains the bottleneck

Retrieve Top-K histories, Cross-Encode only those K, group by Chinese target, and compare target-level aggregation functions.

---

# 13. Main hypotheses for the thesis report

**H1 — LCE:** Query-local contrastive training should better match reranking than independent BCE.

**H2 — Hard negatives:** Negatives localized by the same retriever used at inference should be more informative than arbitrary legal negatives.

**H3 — Coreset:** Diversity-aware deletion should reduce redundant training work while preserving broader contextual coverage than random deletion.

**H4 — LSEPair:** Multi-positive supervision may be valuable because several distinct historical contexts can support the same current target.

**H5 — LSEPair caution:** Its emphasis on low-scoring positives may over-promote contextually weak same-target histories; representative positive selection is therefore important.

**H6 — EM4:** If history-level ranking is strong but repeated evidence is not fused well, target-level aggregation over retrieved Top-K histories may improve candidate ranking without large latency cost.

---

# 14. Source-backed statements vs project-specific adaptations

## Directly supported by cited work

- Independent BCE is a vanilla BERT-reranker training objective in Gao et al.
- LCE combines target-retriever-localized negatives with a contrastive reranker loss.
- Official FlagEmbedding tools mine negatives from top retrieval results.
- LSEPair aggregates all positive-negative score-difference constraints and was robust in the 2026 dense-retriever study.
- Core-set selection uses representation geometry to choose a representative subset.
- RankNet is a pairwise probabilistic learning-to-rank method.
- LambdaLoss provides metric-driven ranking losses.
- PARADE provides precedent for aggregating local relevance evidence into a higher-level ranking score.
- MIL provides precedent for learning a bag-level prediction from multiple local instances.

## Project-specific adaptations requiring our own validation

- Replacing Re_spectators with Agent Phage.
- Balancing targets within the same Pinyin while leaving different Pinyin frequencies unbalanced.
- Using frozen PinyinGPT hidden-state kNN as the target retriever for hard-negative mining.
- Allowing multiple histories from the same wrong target when contexts differ.
- Exact Top-20/Top-30 hard-negative pool size.
- Hidden-state k-center-style coreset selection inside `(author, Pinyin, target)` groups.
- Applying LSEPair, studied for dual-encoder retrieval, to a Cross-Encoder history scorer.
- Grouping retrieved histories by Chinese target for EM4 evidence fusion.

These distinctions should remain explicit in the final thesis.

# References

1. Gao, L., Dai, Z., & Callan, J. (2021). *Rethink Training of BERT Rerankers in Multi-Stage Retrieval Pipeline*. arXiv:2101.08751.
2. FlagOpen. *FlagEmbedding — Finetune Reranker / Hard Negatives*. Official repository documentation.
3. Wang, B., Tang, M., Zhang, H., Guo, J., & Bi, K. (2026). *Training Dense Retrievers with Multiple Positive Passages*. KDD 2026, arXiv:2602.12727.
4. Sener, O., & Savarese, S. (2018). *Active Learning for Convolutional Neural Networks: A Core-Set Approach*. ICLR 2018, arXiv:1708.00489.
5. Burges, C. J. C., et al. (2005). *Learning to Rank using Gradient Descent*. ICML 2005 / Microsoft Research.
6. Wang, X., Li, C., Golbandi, N., Bendersky, M., & Najork, M. (2018). *The LambdaLoss Framework for Ranking Metric Optimization*. CIKM 2018.
7. Li, C., Yates, A., MacAvaney, S., He, B., & Sun, Y. (2020/2021). *PARADE: Passage Representation Aggregation for Document Reranking*. arXiv:2008.09093.
8. Ji, Y., Liu, H., He, B., Xiao, X., Wu, H., & Yu, Y. (2020). *Diversified Multiple Instance Learning for Document-Level Multi-Aspect Sentiment Classification*. EMNLP 2020.
