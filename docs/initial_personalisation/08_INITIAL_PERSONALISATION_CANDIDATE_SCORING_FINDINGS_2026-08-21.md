# Initial-Pinyin Personal Candidate Scoring Findings

**Date:** 2026-08-21
**Project:** Transparent and User-Controllable Personalisation for Chinese Pinyin Input
**Status:** Train-Val development findings; Dev3000 and Test remain closed

## 1. Executive summary

The Initial-Pinyin work now separates two different problems:

1. **Candidate availability / recovery:** whether the correct personal target is present in the recovered Personal candidate pool.
2. **Candidate scoring:** once the correct target is present, which scorer ranks it highest among the Personal candidates.

The strongest empirical finding from the candidate-scoring stage is that a very simple **lexical N-gram memory with recency** is much stronger than raw Frequency and substantially stronger than the BGE64 semantic-similarity memory, while being dramatically cheaper online.

On the fixed Personal K5 candidate-scoring population (`Gold in Personal K5`, `K>=2`, n=4,471):

| Method | Macro Top1 | Micro Top1 | Top3 | MRR | Mean online latency |
|---|---:|---:|---:|---:|---:|
| Frequency F | 49.11% | 49.50% | 84.77% | .6838 | near-zero CPU baseline |
| BGE64 | 53.62% | 53.86% | 88.55% | .7188 | 2.136 ms |
| NGram@2 | 56.01% | 56.32% | 87.61% | .7283 | 0.0835 ms selected scorer |
| **NGramRecency@2, tau=2048** | **59.16%** | **59.49%** | **89.60%** | **.7506** | **0.0844 ms selected scorer** |
| Interpolated NGramRecency | 59.08% | 59.41% | **89.80%** | **.7508** | ~0.025 ms within adaptive-grid timer |
| Interpolated NGramRecency + F | **59.31%** | **59.63%** | 89.67% | .7525 | arithmetic fusion; no model inference |
| Q8 | 63.65% | 64.24% | 91.30% | .7836 | 32.453 ms |
| **Q8 + F** | **66.92%** | **67.57%** | **92.57%** | **.8040** | approximately Q8 cost |

Important latency note: the original NGramRecency experiment reported selected-method online latency (~0.084 ms), whereas the adaptive-grid experiment records smaller per-method arithmetic timing inside a larger sweep. These timer scopes should not be treated as perfectly interchangeable. The robust cross-family comparison is that the original selected NGramRecency scorer was about **25x faster than BGE64 online** and about **385x faster than Q8** on the measured runs.

The second major finding is that expanding the recovery pool from K5 to K10 gives meaningful extra candidate availability:

- Gold in Personal K5: **4,910** rows;
- Gold in Personal K10: **5,537** rows;
- incremental Gold availability: **+627** rows;
- Generic-Missing recoverability: **39.08% -> 44.07%** (+4.99 pp).

On the same original 4,471 K5 ranking rows, expanding the pool to K10 causes only a modest ranking penalty for the best adaptive scorer:

- Interpolated Macro Top1: **59.08% -> 58.13%** (-0.95 pp);
- Hard NGramRecency Macro Top1: **59.16% -> 57.75%** (-1.40 pp).

Therefore K10 is promising as a **recovery pool**, but final K should not be frozen from candidate-only results. The decisive comparison should happen after GenericTop10 + PersonalK integration and Distribution-strength control.

## 2. Experimental boundary

These experiments are standardized Clean3 Train-Val development experiments.

They do **not** use:

- Dev3000 for model or hyperparameter selection;
- Test for model or hyperparameter selection.

Current Gold is not used for:

- candidate construction;
- legal-history retrieval;
- Frequency scoring;
- N-gram scoring;
- BGE64 scoring;
- Q8 candidate scoring.

Gold is used after scoring for:

- Train-Val metric calculation;
- diagnostic subsets;
- disagreement correctness;
- alpha / N / tau / beta / kappa selection on Train-Val.

Consequently, all selected improvements in this document are **development-selected findings**, not sealed holdout confirmation.

## 3. Frozen data and candidate surface

Root:

```text
results\personalisation\initial_recovery_comparison_v1
```

Frozen inputs:

| Artifact | Rows | SHA256 |
|---|---:|---|
| Initial Train-Fit | 144,526 | `162f5c98daa86cc69947571e6d8f20fc401f0a82cdd3fd6e517eb7be2addbdb4` |
| Initial Train-Val | 34,416 | `d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4` |
| Frozen Personal K5 candidate surface | 34,416 | `205c0ba01cd0678d7a4341c503fa2e74cf126a70182ff582687025e4946764b2` |
| Frozen Initial Generic predictions | 34,416 | `bd0fb4dc304e0b266b90fae6fe3ac65424d2f52b23fedfa881212706ba2c2873` |

Frozen K5 workload:

```text
rows = 34,416
eligible rows = 30,509
candidate pairs = 123,738
candidate count distribution:
  K0 = 3,907
  K1 = 3,194
  K2 = 2,787
  K3 = 2,635
  K4 = 2,400
  K5 = 19,493
```

History semantics are unchanged throughout:

```text
same author
-> strictly prior chronological position
-> latest up-to-5000 RAW interactions
-> exact same segmented Initial Pinyin afterward
```

This is **H5000-before-Pinyin**, with chronologically earlier Train-Val rows becoming legal history for later Train-Val rows.

## 4. Candidate-scoring task definition

The main scorer-quality population is:

\[
Gold \in PersonalK \quad \land \quad K\ge2
\]

For K5 this gives n=4,471 rows.

This is deliberately not the final IME Top1. It asks:

> If the correct answer has already been recovered into the Personal candidate pool, can the scorer select it?

The final system still needs a later unified ranking stage over GenericTop10 + Personal candidates.

## 5. Baselines: Frequency, BGE64, Q8

### 5.1 Frequency

For candidate \(c\) with visible same-Pinyin historical count \(n_c\):

\[
F_{raw}(c)=\log(1+n_c)
\]

Frequency is the transparent long-term preference baseline.

K5 result:

```text
Macro Top1 = 0.491082
Micro Top1 = 0.494968
Top3 = 0.847685
MRR = 0.683822
```

### 5.2 BGE64

Current and historical contexts are truncated to the last 64 Python Unicode code points. Candidate support is the normalized target-conditioned Top-5 positive cosine sum.

K5 result:

```text
Macro Top1 = 0.536167
Micro Top1 = 0.538582
Top3 = 0.885484
MRR = 0.718840
online mean = 2.136 ms
online p95 = 3.052 ms
```

### 5.3 Q8

Q8 uses frozen PinyinGPT fixed-candidate scoring with the last 8 Python Unicode code points of current context and ranks candidates by normalized `fixed_mean_log_probability`.

K5 result:

```text
Macro Top1 = 0.636531
Micro Top1 = 0.642362
Top3 = 0.912995
MRR = 0.783568
mean = 32.453 ms
p95 = 54.540 ms
```

Q8 is the strongest standalone context scorer, but it is much more expensive online.

### 5.4 Q8 + Frequency

Fusion:

\[
S(c)=(1-\alpha)Q8(c)+\alpha F(c)
\]

with \(\alpha\in\{0,.25,.5,.75,1\}\).

Selected:

```text
alpha = 0.75
Macro Top1 = 0.669164
Micro Top1 = 0.675688
Top3 = 0.925744
MRR = 0.804015
```

Q8 + F improves Q8 by about **+3.26 pp Macro Top1**, showing that Q8 and long-term personal Frequency contain strongly complementary information.

## 6. N-gram discovery

### 6.1 Hard-backoff N-gram

The first lexical scorer uses an exact suffix match over current and historical contexts.

For max order \(N\), it searches:

\[
N\rightarrow N-1\rightarrow\cdots\rightarrow1\rightarrow0
\]

and uses the longest level with candidate-target evidence.

The original grid was:

\[
N\in\{1,2,3,4,6,8\}
\]

The best pure N-gram was:

```text
NGram@2
Macro Top1 = 0.560105
Micro Top1 = 0.563185
Top3 = 0.876090
MRR = 0.728331
```

The best order being N=2 strongly suggests that very local lexical context carries substantial predictive information under Initial Pinyin, while longer exact suffixes become sparse.

### 6.2 Recency

For a legal historical record \(h\):

\[
w_{recency}(h)=e^{-age(h)/\tau}
\]

where age is measured in the user's H5000 interaction sequence, not only in the same-Pinyin subset.

The original grid selected:

```text
N = 2
tau = 2048
```

with:

```text
Macro Top1 = 0.591558
Micro Top1 = 0.594945
Top3 = 0.895996
MRR = 0.750563
```

Relative to pure NGram@2:

```text
Macro Top1 +3.15 pp
Micro Top1 +3.18 pp
Top3 +1.99 pp
MRR +0.0222
```

All three authors improved.

The adaptive follow-up later tested tau values up to 8192 and still selected tau=2048 for the best Hard and Interpolated configurations. Therefore the earlier concern that tau=2048 was only a boundary optimum is substantially reduced within the current tested range.

## 7. Adaptive lexical-context variants

Three variants were compared on K5 and K10.

### 7.1 HardBackoffNGramRecency

Use only the longest exact suffix level with evidence:

\[
S(c)=\sum_{h:y_h=c,\ suffix_k(q)=suffix_k(h)}e^{-age(h)/\tau}
\]

### 7.2 SoftSuffixRecency

Keep all candidate-target history and weight each row by its capped longest suffix match length:

\[
w_h=e^{\beta m(q,h)}e^{-age(h)/\tau}
\]

### 7.3 InterpolatedNGramRecency

Build recency-weighted candidate distributions at levels \(k=0,1,\ldots,N\), then recursively smooth a specific-context distribution toward the shorter-context distribution:

\[
\lambda_k=\frac{mass_k}{mass_k+\kappa}
\]

\[
P_k=\lambda_k\hat P_k+(1-\lambda_k)P_{k-1}
\]

This keeps long context when there is evidence but backs off smoothly when the context is sparse.

### 7.4 Selected K5 variants

| Method | Selected configuration | Macro Top1 | Micro Top1 | Top3 | MRR@10 |
|---|---|---:|---:|---:|---:|
| Hard | `maxN=2, tau=2048` | **59.156%** | **59.495%** | 89.600% | .750563 |
| Soft | `maxN=4, beta=1, tau=2048` | 57.404% | 57.795% | 89.488% | .741657 |
| Interpolated | `maxN=2, kappa=1, tau=2048` | 59.078% | 59.405% | **89.801%** | **.750760** |

Hard and Interpolated are effectively tied on K5. Hard has a tiny Top1 advantage; Interpolated has a tiny Top3/MRR advantage.

SoftSuffix underperforms both, suggesting that preserving all weakly related history and merely downweighting it can dilute the useful local signal.

## 8. Why the N-gram result matters

Compared with BGE64 on the same K5 candidate-scoring task:

\[
59.16\%-53.62\%=\mathbf{+5.54\ pp}
\]

for Macro Top1.

Using the original selected-scorer latency measurements:

```text
NGramRecency ~0.0844 ms/query
BGE64 online ~2.136 ms/query
```

so BGE64 is about **25.3x slower** on the measured runs while also being less accurate.

This supports the following bounded empirical conclusion:

> In the current Initial-Pinyin personal-candidate discrimination setting, short lexical suffix context plus recency is a better accuracy-latency trade-off than the tested BGE64 semantic-similarity scorer.

This should **not** yet be generalized to all context-aware personalisation settings. A later Full+Short / six-author M1-style comparison is appropriate future work.

## 9. K5 -> K10 candidate expansion

The K10 experiment reconstructs Personal candidates from legal H5000 history and accepts K10 only after verifying that its first five compatible candidates exactly reproduce the frozen K5 for every Train-Val row.

Audit:

```text
frozen K5 exact prefix matches = 34,416 / 34,416
match rate = 1.0
K10 candidate surface SHA256 = 46072df2a24e892de9906efb349fc5d1bc980a00758ac36f422c437083608ddd
```

Availability:

| Metric | K5 | K10 | Change |
|---|---:|---:|---:|
| Gold in Personal pool | 4,910 | **5,537** | **+627** |
| Generic-Missing Gold recoverable | 4,910 | **5,537** | **+627** |
| Generic-Missing recoverable rate | 39.08% | **44.07%** | **+4.99 pp** |

K10 candidate-specific scorer population (`Gold in K10`, `K>=2`) is n=5,098.

### 9.1 K10 selected-method results

| Method | Macro Top1 | Micro Top1 | Top3 | Top5 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| F | 43.174% | 43.409% | 74.343% | 87.701% | .616555 |
| Hard NGramRecency | 52.732% | 52.962% | 80.541% | 90.585% | .686540 |
| SoftSuffixRecency | 51.280% | 51.530% | 80.836% | 90.898% | .679712 |
| **Interpolated NGramRecency** | **52.951%** | **53.178%** | **81.149%** | **90.957%** | **.689352** |

Interpolated becomes the best overall K10 lexical scorer.

### 9.2 Apples-to-apples K5 vs K10 ranking cost

K5-specific and K10-specific metrics have different evaluation populations, so they should not be directly subtracted.

The fair comparison fixes the original 4,471 `Gold in K5, K>=2` rows and changes only the size of the candidate pool.

| Method | K5-pool Macro Top1 | K10-pool Macro Top1 on same rows | Change |
|---|---:|---:|---:|
| F | 49.108% | 49.108% | 0.00 pp |
| Hard NGramRecency | **59.156%** | 57.754% | -1.40 pp |
| Interpolated NGramRecency | **59.078%** | **58.129%** | **-0.95 pp** |
| SoftSuffixRecency | 57.404% | 56.891% | -0.51 pp |

For Interpolated, the original K5 rows have:

```text
K5 pool Top1 correct = 2,656 / 4,471
K10 pool Top1 correct on same rows = 2,616 / 4,471
loss from extra competitors = 40 Top1 cases
```

Across the 627 newly available K10-only Gold rows, Interpolated produces approximately:

```text
Top1 correct = 95 / 627 = 15.15%
Top3 ~= 35.1%
Top5 ~= 56.6%
```

Therefore the candidate-only Top1 accounting is approximately:

```text
+95 newly solvable Top1 cases
-40 previously correct K5 cases displaced by added competitors
= +55 net Top1 cases
```

This is evidence that K10 is useful as a recovery pool, but it is not yet an end-to-end system result because Generic candidates and Distribution strength are not included.

## 10. NGramRecency + Frequency fusion

Fusion uses the same convention as Q8 + F:

\[
S(c)=(1-\alpha)N(c)+\alpha F(c)
\]

with:

\[
\alpha\in\{0,.25,.5,.75,1\}
\]

### 10.1 Selected results

| Pool | Base | Best alpha | Base Macro Top1 | Fused Macro Top1 | Change | Rescue | Harm | Net |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| K5 | Hard | .25 | 59.156% | 59.203% | +0.047 pp | 34 | 31 | +3 |
| K5 | Interpolated | .25 | 59.078% | **59.307%** | **+0.228 pp** | 43 | 33 | +10 |
| K10 | Hard | .50 | 52.732% | 53.033% | +0.301 pp | 84 | 69 | +15 |
| K10 | Interpolated | .25 | 52.951% | **53.375%** | **+0.423 pp** | 49 | 28 | **+21** |

The fusion gain is real on Train-Val but small compared with Q8 + F.

Interpretation:

- Q8 is a largely model-based local-context signal, so long-term Frequency adds strong complementary information.
- NGramRecency already contains historical occurrence, lexical context, and recency. Therefore explicit Frequency overlaps with information already present in the scorer.
- This explains why Frequency adds only a small development-selected improvement to NGramRecency.

The cleanest N-gram fusion result is Interpolated + 0.25F, especially at K10, but the gain is small enough that pure Interpolated remains attractive for simplicity and transparency.

Do not freeze the fusion solely from the +0.23/+0.42 pp Train-Val gains without later holdout confirmation.

## 11. Current candidate-scoring conclusions

### 11.1 Strongest conclusions

1. **Frequency alone is not enough.** Context-aware candidate scoring improves substantially over F.
2. **Lexical context is surprisingly strong.** NGramRecency improves K5 Macro Top1 over Frequency by about +10.05 pp.
3. **Recency matters.** NGram@2 -> NGramRecency@2 gives about +3.15 pp Macro Top1.
4. **Very short context is sufficient in the current task.** Hard and Interpolated both select maxN=2; longer context is not required for the best result.
5. **BGE64 is not competitive on the current accuracy-latency frontier.** NGramRecency is both more accurate and much faster.
6. **Q8 remains the strongest standalone context scorer but is expensive.** Q8+F is the current candidate-scoring accuracy ceiling.
7. **K10 meaningfully improves candidate availability.** It adds 627 Gold rows at modest shared-population ranking cost.
8. **Interpolated smoothing becomes more useful as the pool grows.** It is almost tied with Hard at K5 and is the best lexical scorer at K10.
9. **Explicit F adds little to NGramRecency.** NGramRecency already captures much of the historical-preference signal.

### 11.2 What should remain provisional

Do not yet freeze:

- final Personal candidate pool K;
- final unified Generic+Personal ranking rule;
- Distribution / concentration strength;
- NGram+F as mandatory instead of optional;
- any claim that lexical N-gram universally beats semantic embedding methods.

The candidate-only results motivate carrying **K10 + Interpolated NGramRecency** into the unified-ranking stage, while preserving K5 as the clean reference condition.

## 12. Recommended next stage: Distribution / trust strength

Candidate scoring answers:

> Which Personal candidate should rank above the other Personal candidates?

The next problem is different:

> How strongly should Personal evidence affect the final Generic + Personal ranking?

For the Distribution stage, keep the semantic distinction:

- **Choice Share** is candidate-specific and indicates direction / which candidate the user prefers.
- **Concentration** is query-level and indicates how strongly the historical distribution should be trusted.

A clean family to test later is:

\[
D_\gamma(c)=Conc^\gamma\left(CS(c)-\frac{1}{K}\right)
\]

with a small grid such as:

\[
\gamma\in\{0,0.5,1,2\}
\]

This should be staged after candidate-scorer selection rather than mixed into the current N-gram experiment.

## 13. Future work recorded but deferred

### 13.1 Lexical context inside M1 / Full+Short context models

The N-gram result is important enough to justify a later controlled experiment on the richer Full+Short / six-author context setting.

Potential comparison:

```text
Frequency
vs BGE-M1
vs Lexical-NGram-M1
vs Hybrid lexical + semantic M1
```

Possible hybrid history weight:

\[
w_h=w_{lexical}(q,h)\cdot w_{semantic}(q,h)\cdot w_{recency}(h)
\]

This is future work only; do not retroactively modify frozen M1 results.

### 13.2 PinyinGPT fixed-candidate scoring acceleration

PinyinGPT speed work is intentionally deferred until after the current candidate-scoring write-up.

Priority audit / engineering ideas to record:

1. instrument `score_candidates()` and count actual model forward calls per query;
2. check whether K candidates are currently scored sequentially internally;
3. batch candidate continuations when possible;
4. reuse the shared query prefix through Transformer KV cache;
5. use a candidate trie so shared candidate prefixes are evaluated once;
6. for single-character candidates, obtain all candidate token probabilities from one next-token logits vector where semantics permit exact equivalence;
7. only after computational reuse, evaluate lower-level optimizations such as inference-mode / precision / compilation;
8. optionally use NGramRecency as a transparent fast path and invoke Q8 only on uncertain / low-margin cases.

These are engineering hypotheses, not measured speedups yet.

## 14. Reproducibility map

### 14.1 Q8 + BGE64

Runner:

```text
experiments\initial_personalisation\run_initial_candidate_scoring_q8_bge64_v1.py
```

Canonical fixed-master SHA256:

```text
cee95ee85fd69f2de7deab07cb50ab8f56bb50fb9dbc29315261ca89c578b494
```

Output:

```text
results\personalisation\initial_recovery_comparison_v1\candidate_scoring_q8_bge64_v1\candidate_scoring_comparison.json
```

### 14.2 Original NGram + Recency

Runner:

```text
experiments\initial_personalisation\run_initial_candidate_scoring_ngram_recency_v1.py
```

SHA256:

```text
7a24b100aa20e26d80bb8bf5900260864c7c71ea03ad2f1295d9cc3b933090d2
```

Output:

```text
results\personalisation\initial_recovery_comparison_v1\candidate_scoring_ngram_recency_v1\ngram_recency\candidate_scoring_comparison.json
```

### 14.3 Adaptive NGram K5/K10

Runner:

```text
experiments\initial_personalisation\run_initial_candidate_scoring_adaptive_ngram_top10_v1.py
```

SHA256:

```text
261020f767e0501840df87bcdd15863652c7c3db660937ed8450287c3592ba7f
```

Selected grid:

```text
candidate K = {5,10}
maxN = {2,4,8}
tau = {512,2048,8192}
beta = {0.25,0.5,1}
kappa = {1,4,16,64}
```

Output:

```text
results\personalisation\initial_recovery_comparison_v1\candidate_scoring_adaptive_ngram_top10_v1\adaptive_ngram_top10_v1\comparison.json
```

K5/K10 surface SHA256:

```text
46072df2a24e892de9906efb349fc5d1bc980a00758ac36f422c437083608ddd
```

### 14.4 NGram + Frequency fusion

Runner:

```text
experiments\initial_personalisation\run_initial_ngram_frequency_fusion_v1.py
```

SHA256:

```text
341d194a60fc571a3ef1e6376dffdab87ac7015c988aace611b888ab19f4e147
```

Output:

```text
results\personalisation\initial_recovery_comparison_v1\candidate_scoring_ngram_frequency_fusion_v1\ngram_frequency_fusion_comparison.json
```

Alpha grid:

```text
{0, 0.25, 0.5, 0.75, 1}
```

### 14.5 Output hashes still to freeze locally

Before the final thesis record, capture SHA256 for the completed result artifacts:

```powershell
$root = '.\results\personalisation\initial_recovery_comparison_v1'

Get-FileHash "$root\candidate_scoring_q8_bge64_v1\candidate_scoring_comparison.json" -Algorithm SHA256
Get-FileHash "$root\candidate_scoring_ngram_recency_v1\ngram_recency\candidate_scoring_comparison.json" -Algorithm SHA256
Get-FileHash "$root\candidate_scoring_adaptive_ngram_top10_v1\adaptive_ngram_top10_v1\comparison.json" -Algorithm SHA256
Get-FileHash "$root\candidate_scoring_adaptive_ngram_top10_v1\adaptive_ngram_top10_v1\candidate_surface_k5_k10.jsonl" -Algorithm SHA256
Get-FileHash "$root\candidate_scoring_ngram_frequency_fusion_v1\ngram_frequency_fusion_comparison.json" -Algorithm SHA256
```

Record these output hashes in the durable repository reproducibility index after verification.

## 15. Suggested thesis wording

A cautious summary suitable for later thesis drafting is:

> On the standardized Initial-Pinyin Train-Val candidate-scoring task, short lexical context proved highly informative. A causal same-user N-gram memory with recency improved Macro-author Top1 from 49.11% for Frequency to 59.16%, and outperformed the tested BGE64 semantic-similarity scorer (53.62%) while requiring far less online computation. Adaptive interpolation provided little additional benefit at K5 but became slightly stronger when the Personal candidate pool was expanded to K10. Expanding K5 to K10 increased Generic-Missing personal recoverability from 39.08% to 44.07% at a modest shared-population ranking cost. Frozen PinyinGPT Q8 remained more accurate (63.65%, 66.92% with Frequency) but was substantially slower. These results motivate lexical N-gram memory as the primary transparent low-latency candidate scorer to carry into unified Generic+Personal ranking experiments.

## 16. Immediate next action

1. Freeze and hash the current completed JSON outputs.
2. Add this findings note and the updated reproducibility note to the repository as explicit files only.
3. Do not commit/push/tag unless explicitly authorized.
4. Move next to Distribution / Choice Share / Concentration over the selected Personal candidate scorer(s).
5. Keep PinyinGPT acceleration and M1 lexical-context transfer recorded as deferred follow-up work.
