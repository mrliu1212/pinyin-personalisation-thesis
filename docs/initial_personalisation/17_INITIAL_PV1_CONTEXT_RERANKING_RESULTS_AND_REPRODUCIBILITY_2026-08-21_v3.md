# Initial-Pinyin PV1 Final-List Context Reranking: Results and Reproducibility

**Date:** 2026-08-21  
**Status:** Train-Val development record; Dev3000 untouched; Test untouched  

Formatting note: viewer-compatible formula formatting. Display equations use `$$ ... $$`; important equations include a plain-text fallback.
**Repository:** `C:\Users\chiar\Desktop\LBH\thesis-initial-research`  
**Branch:** `work/initial-personalisation`

This document freezes the current evidence for final-list context reranking on top of frozen PV1. It also records the interpretation of these results relative to the historical PV2 context-aware personal-vocabulary method. It records the complete numeric results currently observed in the terminal/logs, the method definitions, the recovery results, the additive-vs-interaction ablation, runtime evidence, script hashes, input hashes, commands, output paths, and scientific boundaries needed for reproducibility.

Important scope distinction: this document is about ordering the frozen PV1 Top10, not changing candidate availability. Recovery/candidate-construction experiments are upstream and remain separate.

## 1. Executive conclusion

The current evidence supports the following hierarchy:

```text
PV1
  -> PV1 + NGramRecency                 [main / practical context method]
  -> PV1 + NGramRecency + BGERecency    [full / accuracy-oriented augmentation]
```

**Main conclusions:**

NGramRecency is the core context method. It gives the large improvement while remaining transparent and lightweight.

BGERecency is useful, but secondary. Adding semantic-temporal evidence to NGramRecency gives the best overall Train-Val metrics, but only a small additional gain.

Recency should condition relevant history rather than merely be added as an independent score. Both NGram and BGE perform better when interaction recency is multiplied into context-relevant historical evidence than when plain context and a separate recency-only factor are added independently.

Recovery and overall ranking are not identical objectives. The full interaction model is best overall, while NGramRecency alone is slightly better on Recovery Rec@1 and Recovery MRR.

Final-list context reranking does not change candidate coverage. Missing@10 and Recovery Rec@10 remain invariant because the candidate set is frozen.

The current evidence favors a two-stage architecture: recovery first, context second. Historical PV2 used BGE mainly as a personal-candidate injection-strength term and did not improve over PV1 on the frozen Full+Short benchmark (.779000 -> .778167 Macro Top1). In the current Initial Train-Val setting, freezing PV1 first and then reranking the whole final list with context produces positive gains, especially with NGramRecency (.401872 -> .429091). This cross-setting comparison is supportive rather than a controlled causal proof; a same-surface stage-placement ablation would be required for that stronger claim.

**Current recommended development methods:**

Primary practical method: PV1 + NGramRecency

Full accuracy-oriented method: PV1 + BGERecency + NGramRecency

Semantic ablation: PV1 + BGERecency

Short-term preference ablation: PV1 + RecencyOnly

Do not claim that the small gain of the full model over NGramRecency is statistically significant without a paired test / later holdout confirmation.

## 2. Frozen experimental protocol

### 2.1 Development split

```text
Clean3 Train
  -> Train-Fit / Train-Val
  -> development and method selection
  -> PRE-DEV FREEZE
  -> Dev3000
  -> final freeze
  -> Test
```

Current context experiments use only:

Train-Fit rows = 144,526
Train-Val rows = 34,416
Authors = Agent Phage, Etinjat, breaddddd

**Protocol assertions:**

Gold used for candidate construction = false
Gold used for context scoring/features = false
Gold used for Train-Val evaluation/model selection = true
Dev3000 used = false
Test used = false

### 2.2 Causal personal history

History semantics are fixed:

```text
same author
-> strictly prior interactions only
-> latest up-to-5000 RAW same-author interactions
-> exact current Initial-Pinyin filtering afterward
```

Therefore:

H5000 is applied before exact-Pinyin filtering.

Earlier Train-Val rows may become history for later Train-Val rows.

Current/future target is never visible to the scorer.

age=0 means the immediately previous same-author interaction inside the H5000 author stream.

### 2.3 Frozen candidate set

All final-list context methods rerank only the already-built frozen PV1 Top10.

candidate_set_changed = false

Therefore the following invariants must hold:

Missing@10_new = Missing@10_PV1
Recovery Rec@10_new = Recovery Rec@10_PV1

Observed in all current context runs:

Missing@10 = 0.291144
Recovery Rec@10 = 0.5401

## 3. Frozen inputs and hashes

### 3.1 Standardized Full source split

These are the source Clean3 Full-Pinyin files from which the deterministic Initial split was constructed:

Full Train-Fit rows = 144,526
Full Train-Val rows = 34,416

Full Train-Fit SHA256 = 547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6
Full Train-Val SHA256 = d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220

### 3.2 Deterministically transformed Initial inputs used by the context runners

Initial Train-Fit rows = 144,526
Initial Train-Val rows = 34,416

initial_train_fit_v1.jsonl
SHA256 = 162f5c98daa86cc69947571e6d8f20fc401f0a82cdd3fd6e517eb7be2addbdb4

initial_train_val_v1.jsonl
SHA256 = d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4

### 3.3 Frozen candidate and prediction artifacts

candidate_surface\train_val_candidate_surface.jsonl
SHA256 = 205c0ba01cd0678d7a4341c503fa2e74cf126a70182ff582687025e4946764b2
rows = 34,416
eligible = 30,509
candidate pairs = 123,738

Frozen Generic predictions:

train_val_generic\predictions.jsonl
SHA256 = bd0fb4dc304e0b266b90fae6fe3ac65424d2f52b23fedfa881212706ba2c2873

Frozen Frequency/PV1 predictions used directly by the final-list context runners:

frequency_pv1\predictions.jsonl
SHA256 = 7fd8aa158d8cd50bced36b55610f8d932bc65e3aae1dbbd5bd65907ff1707ea7

## 4. Frozen PV1 baseline

PV1 is fixed at:

Kpv = 1
lambda_F = 4
lambda_PV = 4

For a personal-only injected candidate c:

final_score(c) = generic_boundary_score + 4 * frequency_support(c)

Generic candidates retain their frozen Frequency/PV1 final scores. Context reranking must preserve these PV1 scores as the base, not replace them with rank-only scores.

The generic-side score was normalized per query before Frequency/PV1 using a z-score.

Personal-frequency support:

$$
F_{PV}(c)=\frac{\log(1+n_c)}{\max_j\log(1+n_j)}
$$

Plain text:

`Personal-frequency support = log-scaled candidate frequency normalized by the maximum log-scaled frequency for the query.`

### 4.1 Overall baseline metrics — all Train-Val, N=34,416

| Method | Macro Top1 | Micro Top1 | Top3 | Top5 | MRR@10 | Missing@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Generic G | 0.307099 | 0.330573 | 0.491341 | 0.557996 | 0.426472 | 0.365092 |
| Frequency F | 0.382495 | 0.408473 | 0.555120 | 0.601000 | 0.489432 | 0.365092 |
| PV1 | 0.401872 | 0.426749 | 0.598907 | 0.663093 | 0.524450 | 0.291144 |

F -> PV1 Top1 transition diagnostic:

rescue = 933
harm = 304
net = +629

### 4.2 Candidate availability vs actual PV1 recovery

Generic Missing count:

12,565 / 34,416

Gold theoretically available in Personal K5 among Generic Missing:

4,910 / 12,565 = 39.0768%

This is theoretical K5 recoverability, not actual PV1 final-list recovery. PV1 injects K=1 only.

Recovery population used below:

$$
R=\{Gold\notin GenericTop10 \land Gold\in PersonalK5\}
$$

Plain text:

`R = queries where Generic Top10 misses Gold but Personal K5 contains Gold.`

|R| = 4,910

PV1 on this fixed recovery population:

| Method | Rec@1 | Rec@3 | Rec@5 | Rec@10 | Recovery MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| PV1 | 0.1900 | 0.4923 | 0.5363 | 0.5401 | 0.3363 |

## 5. Context feature definitions

The final reranking score always starts from frozen PV1:

$$
S_{new}(c)=S_{PV1}(c)+\text{context terms}
$$

Plain text:

`New score = frozen PV1 score + context terms.`

### 5.1 Plain BGE64 semantic context

Current query context:

last 64 Chinese/context characters

For each candidate, retrieve candidate-conditioned same-Pinyin causal historical interactions. Select the candidate's historical Top-5 by cosine similarity only, clamp negative cosine to zero, sum, and normalize over the frozen PV1 Top10.

Conceptually:

$$
R_{BGE}(c)=\sum_{h\in Top5_{cos}(H_c)}\max(0,\cos(E(q),E(h)))
$$

Plain text:

`BGE support = sum of non-negative cosine similarities over the candidate-conditioned cosine Top-5 histories.`

$$
P_{BGE}(c)=\frac{R_{BGE}(c)}{\sum_jR_{BGE}(j)}
$$

Plain text:

`Normalized BGE support = candidate BGE support divided by total BGE support over the final candidate list.`

### 5.2 Plain HardBackoff NGram — no recency

Use lexical suffix matching, not position-decay weighting.

Choose the largest exact suffix order k <= maxN with candidate-target historical evidence. With current maxN=2:

try exact suffix-2
-> if no evidence, suffix-1
-> if no evidence, k=0 same-Pinyin history

Then count matched history rows per candidate and normalize.

$$
P_{NG}(c)\propto \#\{h:y_h=c,\ suffix_{k^*}(h)=suffix_{k^*}(q)\}
$$

Plain text:

`NGram support is proportional to the number of candidate histories matching the selected backoff suffix.`

### 5.3 Recency-only — no context matching

This factor measures recent personal choice frequency independent of semantic/lexical context:

$$
R_R(c)=\sum_{h:y_h=c}e^{-age(h)/\tau_R}
$$

Plain text:

`Recency-only support = sum of exponentially decayed same-candidate historical interactions.`

$$
P_R(c)=\frac{R_R(c)}{\sum_jR_R(j)}
$$

Plain text:

`Normalized recency support = candidate recency support divided by total recency support over the candidate list.`

Current fixed:

tau_R = 2048

### 5.4 NGramRecency interaction

First perform HardBackoff suffix gating; then weight each matched historical interaction by age:

$$
R_{NG-R}(c)=\sum_{\substack{h:y_h=c\\suffix_{k^*}(h)=suffix_{k^*}(q)}}e^{-age(h)/2048}
$$

Plain text:

`NGramRecency = suffix-relevant candidate history weighted by exponential interaction recency.`

This is:

lexical context relevance × interaction recency

It is not the Position method where characters closer to the current Pinyin receive a continuous positional decay weight.

### 5.5 BGERecency interaction

Top-5 retrieval remains cosine-only. Recency does not change retrieval; it only weights the aggregation:

$$
R_{BGE-R}(c)=\sum_{h\in Top5_{cos}(H_c)}\max(0,\cos(E(q),E(h)))e^{-age(h)/\tau_B}
$$

Plain text:

`BGERecency = cosine-relevant Top-5 history weighted by exponential interaction recency after retrieval.`

$$
P_{BGE-R}(c)=\frac{R_{BGE-R}(c)}{\sum_jR_{BGE-R}(j)}
$$

Plain text:

`Normalized BGERecency support = candidate BGERecency support divided by total support over the candidate list.`

Current fixed:

tau_B = 2048

### 5.6 Position / PositionRecency exploratory ablation

Position scoring asks whether the characters nearer the current Pinyin match more strongly:

$$
S_{pos}(q,h)=\sum_{d=1}^{L}\alpha^{d-1}\mathbf{1}[x^q_{-d}=x^h_{-d}]
$$

Plain text:

`Position score = exponentially position-weighted character matches between current and historical contexts.`

PositionRecency additionally multiplies the history interaction by:

$$
e^{-age(h)/\tau}
$$

Plain text:

`Recency weight = exponential decay by history age.`

This was tested but was weaker than Hard NGramRecency and is not the current mainline.

## 6. Experiment A — Position vs Hard NGramRecency final-list reranking

Runner:

experiments\initial_personalisation\run_initial_pv1_context_reranking_v2.py

Provided-copy SHA256:

3f140f7c5ce32b5e37297472c2b5e63b603085f07453bfa236dffbe421b96b44

**Selected configurations:**

HardNGramRecency: maxN=2, tau=2048, lambda_C=4
Position: alpha=.25, L=all, lambda_C=2
PositionRecency: alpha=.75, tau=2048, L=all, lambda_C=2

### 6.1 Overall Train-Val

| Method | Macro | Micro | Top3 | Top5 | MRR | Missing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PV1 | .401872 | .426749 | .598907 | .663093 | .524450 | .291144 |
| HardNGramRecency | .428085 | .451970 | .610646 | .665592 | .541368 | .291144 |
| Position | .412552 | .437209 | .606462 | .665127 | .532486 | .291144 |
| PositionRecency | .413536 | .438226 | .605707 | .665098 | .532670 | .291144 |

### 6.2 Recovery population

| Method | Rec@1 | Rec@3 | Rec@5 | Rec@10 | RecMRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| PV1 | .1900 | .4923 | .5363 | .5401 | .3363 |
| HardNGramRecency | .3310 | .5136 | .5379 | .5401 | .4203 |
| Position | .2434 | .5061 | .5377 | .5401 | .3713 |
| PositionRecency | .2401 | .5071 | .5371 | .5401 | .3696 |

**Interpretation:**

Hard NGramRecency is the dominant lexical-temporal signal.

Position contains signal but is clearly weaker.

Position + history recency gives only marginal/mixed improvement over Position.

Position remains an exploratory / appendix ablation, not the main context method.

## 7. Experiment B — Plain BGE + NGramRecency final-list reranking V1

Runner:

experiments\initial_personalisation\run_initial_pv1_bge_ngram_context_reranking_v1.py

Provided-copy SHA256:

9082d0f72143c86e11d0f4addceb219e4dd3f5cc69bf58a55c2b7b4d4b8a87cb

BGE model:

C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf

Initial grid:

lambda_B = {0,.25,.5,1,2,4}
lambda_N = {0,.25,.5,1,2,4}

**Selected:**

BGE                     lambda_B=4,    lambda_N=0
NGramRecency            lambda_B=0,    lambda_N=4
BGE+NGramRecency        lambda_B=.25,  lambda_N=4
GlobalBest              lambda_B=.25,  lambda_N=4

### 7.1 Overall Train-Val, N=34,416

| Method | Macro | Micro | Top3 | Top5 | MRR | Missing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PV1 | .401872 | .426749 | .598907 | .663093 | .524450 | .291144 |
| BGE | .405477 | .429945 | .601523 | .664720 | .526895 | .291144 |
| NGramRecency | .428085 | .451970 | .610646 | .665592 | .541368 | .291144 |
| BGE+NGramRecency | .428283 | .452173 | .610966 | .665650 | .541496 | .291144 |

### 7.2 Recovery population

| Method | Rec@1 | Rec@3 | Rec@5 | Rec@10 | RecMRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| PV1 | .1900 | .4923 | .5363 | .5401 | .3363 |
| BGE | .2379 | .5086 | .5371 | .5401 | .3681 |
| NGramRecency | .3310 | .5136 | .5379 | .5401 | .4203 |
| BGE+NGramRecency | .3334 | .5145 | .5381 | .5401 | .4217 |

**Interpretation at this stage:**

BGE alone was useful but much weaker than NGramRecency.

The joint model gave a tiny improvement over NGramRecency on the original grid.

Both BGE and NGram hit the old upper boundary lambda=4, so the old grid did not establish the optimum.

### 7.3 BGE V1 cache / runtime evidence

Historical BGE contexts:

required unique historical contexts = 38,878
seed cache vectors reused = 29,680
new historical embeddings = 9,198
final observed new-history embedding mean ~= 2.128 ms/context

Online query-scoring progress ended at approximately:

rate ~= 487.80 rows/s
query embedding ~= 1.563 ms
online ~= 1.776 ms

Observed total wall time:

148.8 s

Post-completion llama/ctypes callback messages appeared after outputs were written. They are treated as shutdown logging noise, not failed scoring.

## 8. Experiment C — BGE history-recency V2 + expanded lambda grid

Runner:

experiments\initial_personalisation\run_initial_pv1_bge_recency_ngram_context_reranking_v2.py

SHA256:

0d02ede6819ca824b703816d30fa7276b429f1e834baa866c0add73b65d6a3db

Important BGERecency design:

Top-5 retrieval = cosine only
recency affects aggregation only
tau_B = 2048

**Expanded grids:**

lambda_B = {0,.25,.5,1,2,4,6,8}
lambda_N = {0,.25,.5,1,2,4,6,8,12}

**Selected:**

BGE                            mode=plain    lambda_B=6   lambda_N=0
BGERecency                     mode=recency  lambda_B=6   lambda_N=0
NGramRecency                   mode=plain    lambda_B=0   lambda_N=6
BGE+NGramRecency               mode=plain    lambda_B=.25 lambda_N=6
BGERecency+NGramRecency        mode=recency  lambda_B=4   lambda_N=4
GlobalBest                     mode=recency  lambda_B=4   lambda_N=4

### 8.1 Overall Train-Val

| Method | Macro | Micro | Top3 | Top5 | MRR | Missing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PV1 | .401872 | .426749 | .598907 | .663093 | .524450 | .291144 |
| BGE | .405679 | .429887 | .601958 | .665214 | .526979 | .291144 |
| BGERecency | .413942 | .438052 | .605532 | .665766 | .532635 | .291144 |
| NGramRecency | .429091 | .452755 | .611285 | .665940 | .541944 | .291144 |
| BGE+NGramRecency | .428791 | .452406 | .611344 | .666027 | .541755 | .291144 |
| BGERecency+NGramRecency | .429506 | .453423 | .612099 | .667335 | .542766 | .291144 |

### 8.2 Recovery population

| Method | Rec@1 | Rec@3 | Rec@5 | Rec@10 | RecMRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| PV1 | .1900 | .4923 | .5363 | .5401 | .3363 |
| BGE | .2656 | .5100 | .5369 | .5401 | .3835 |
| BGERecency | .2912 | .5106 | .5369 | .5401 | .3982 |
| NGramRecency | .3601 | .5149 | .5379 | .5401 | .4361 |
| BGE+NGramRecency | .3605 | .5153 | .5381 | .5401 | .4364 |
| BGERecency+NGramRecency | .3582 | .5165 | .5381 | .5401 | .4356 |

### 8.3 Key V2 comparisons

BGE history recency effect:

BGE Macro        .405679 -> .413942   (+0.8263 pp)
BGE Recovery@1   .2656   -> .2912     (+2.56 pp)

Expanded NGram weight effect relative to the earlier lambda_N=4 result:

NGramRecency Macro .428085 -> .429091 (+0.1006 pp)
selected lambda_N  4 -> 6

Plain BGE augmentation after expanding the grid:

NGramRecency              .429091
BGE+NGramRecency          .428791
Delta                     -0.0300 pp Macro

Recency-weighted semantic augmentation:

NGramRecency                         .429091
BGERecency+NGramRecency              .429506
Delta                                +0.0415 pp Macro

The joint interaction model also improved Micro, Top3, Top5, and MRR, but the gain is small; do not call it materially/significantly superior without statistical validation.

### 8.4 V2 BGERecency runtime

The V2 run loaded the completed V1 historical cache:

history vectors loaded = 38,878
new history embeddings = 0
current query rows embedded/scored = 34,416

Final progress approximately:

rate ~= 491.35 rows/s
query embedding ~= 1.631 ms
online ~= 1.911 ms

Observed total wall time:

128.8 s

The same post-completion llama/ctypes callback logging noise was observed after successful completion.

## 9. Experiment D — factorial BGE + NGram + Recency ablation V3

Runner:

experiments\initial_personalisation\run_initial_pv1_factorial_context_ablation_v3.py

SHA256:

b8f71397c9ab7271b2cac2f2340bce76ed9a5ae9d42b5d15b4c47214a4942a8c

Purpose: separate the three factors into independent additive terms and compare them directly with the existing context-by-recency interaction scorers.

Additive model:

$$
S(c)=S_{PV1}(c)+\lambda_B P_{BGE}(c)+\lambda_N P_{NG}(c)+\lambda_R P_R(c)
$$

Plain text:

`Additive score = PV1 + weighted BGE + weighted NGram + weighted independent Recency.`

Fixed parameters:

Plain NGram maxN = 2
Recency-only tau_R = 2048
NGramRecency control tau = 2048
BGE V1 / BGERecency V2 supports = read-only
BGE embeddings recomputed = false

Grids:

lambda_B = {0,.25,.5,1,2,4,6,8}
lambda_N = {0,.25,.5,1,2,4,6,8,12}
lambda_R = {0,.25,.5,1,2,4,6,8,12}

Full additive grid size:

8 * 9 * 9 = 648 configurations

### 9.1 Selected additive configurations

| Family | lambda_B | lambda_N | lambda_R |
| --- | ---: | ---: | ---: |
| PV1 | 0 | 0 | 0 |
| Recency | 0 | 0 | 8 |
| NGram | 0 | 6 | 0 |
| NGram+Recency | 0 | 6 | .25 |
| BGE | 6 | 0 | 0 |
| BGE+Recency | 1 | 0 | 4 |
| BGE+NGram | .25 | 6 | 0 |
| BGE+NGram+Recency | .25 | 6 | .25 |

**Global additive best:**

family = NGram+Recency
lambda_B=0, lambda_N=6, lambda_R=.25

### 9.2 Complete V3 overall results — all Train-Val, N=34,416

| Method | Macro | Micro | Top3 | Top5 | MRR | Missing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PV1 | .401872 | .426749 | .598907 | .663093 | .524450 | .291144 |
| Recency | .410765 | .433839 | .603208 | .664458 | .529197 | .291144 |
| NGram | .426304 | .449820 | .609600 | .665679 | .539729 | .291144 |
| NGram+Recency additive | .426361 | .449878 | .609833 | .665766 | .539774 | .291144 |
| BGE | .405679 | .429887 | .601958 | .665214 | .526979 | .291144 |
| BGE+Recency additive | .411150 | .434682 | .602481 | .664081 | .529607 | .291144 |
| BGE+NGram additive | .426318 | .449762 | .609920 | .665766 | .539753 | .291144 |
| BGE+NGram+Recency additive | .426284 | .449791 | .609949 | .665882 | .539766 | .291144 |
| BGERecency interaction | .413942 | .438052 | .605532 | .665766 | .532635 | .291144 |
| NGramRecency interaction | .429091 | .452755 | .611285 | .665940 | .541944 | .291144 |
| BGERecency+NGramRecency interaction | .429506 | .453423 | .612099 | .667335 | .542766 | .291144 |

### 9.3 Complete V3 recovery results — fixed R, N=4,910

| Method | Rec@1 | Rec@3 | Rec@5 | Rec@10 | RecMRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| PV1 | .1900 | .4923 | .5363 | .5401 | .3363 |
| Recency | .3505 | .5191 | .5383 | .5401 | .4316 |
| NGram | .3477 | .5167 | .5387 | .5401 | .4301 |
| NGram+Recency additive | .3513 | .5173 | .5391 | .5401 | .4322 |
| BGE | .2656 | .5100 | .5369 | .5401 | .3835 |
| BGE+Recency additive | .2927 | .5145 | .5381 | .5401 | .4007 |
| BGE+NGram additive | .3495 | .5171 | .5389 | .5401 | .4313 |
| BGE+NGram+Recency additive | .3530 | .5177 | .5391 | .5401 | .4332 |
| BGERecency interaction | .2912 | .5106 | .5369 | .5401 | .3982 |
| NGramRecency interaction | .3601 | .5149 | .5379 | .5401 | .4361 |
| BGERecency+NGramRecency interaction | .3582 | .5165 | .5381 | .5401 | .4356 |

### 9.4 V3 runtime

factor-support construction = 34,416 / 34,416
factorial configs = 648 / 648
final grid throughput ~= 204.64 configs/s
observed total wall time = 35.9 s

V3 does not re-embed BGE histories or queries; it reads BGE/BGERecency supports from completed read-only V1/V2 artifacts.

## 10. Cross-method comparison and deltas

### 10.1 Overall gain over PV1

| Method | Macro | Macro gain vs PV1 | MRR | Recovery@1 |
| --- | ---: | ---: | ---: | ---: |
| PV1 | .401872 | — | .524450 | .1900 |
| BGE | .405679 | +0.3807 pp | .526979 | .2656 |
| Recency-only | .410765 | +0.8893 pp | .529197 | .3505 |
| BGERecency | .413942 | +1.2070 pp | .532635 | .2912 |
| Plain NGram | .426304 | +2.4432 pp | .539729 | .3477 |
| NGram + independent Recency | .426361 | +2.4489 pp | .539774 | .3513 |
| NGramRecency interaction | .429091 | +2.7219 pp | .541944 | .3601 |
| Full interaction | .429506 | +2.7634 pp | .542766 | .3582 |

### 10.2 Additive recency vs interaction recency

Lexical

NGram                        Macro=.426304
NGram + independent Recency  Macro=.426361
NGramRecency interaction     Macro=.429091

**Key differences:**

Independent R added to NGram: +0.0057 pp Macro
Interaction vs additive:      +0.2730 pp Macro

**Recovery@1:**
NGram+R additive = .3513
NGramRecency     = .3601
Delta            = +0.88 pp

**Interpretation:**

Recent history is most useful when it is conditioned on lexical relevance to the current query, rather than merely added as a global recent-preference score.

Semantic

BGE                        Macro=.405679
BGE + independent Recency  Macro=.411150
BGERecency interaction     Macro=.413942

Interaction vs additive: +0.2792 pp Macro

However on the recovery population:

BGE+R additive Rec@1 = .2927
BGERecency Rec@1     = .2912

So BGERecency's overall gain is not primarily a recovery-Top1 gain; it improves broader final-list ordering.

### 10.3 Practical NGramRecency vs full semantic augmentation

| Metric | NGramRecency | Full interaction | Full - NGramRecency |
| --- | ---: | ---: | ---: |
| Macro Top1 | .429091 | .429506 | +0.0415 pp |
| Micro Top1 | .452755 | .453423 | +0.0668 pp |
| Top3 | .611285 | .612099 | +0.0814 pp |
| Top5 | .665940 | .667335 | +0.1395 pp |
| MRR | .541944 | .542766 | +0.000822 |
| Recovery Rec@1 | .3601 | .3582 | -0.19 pp |
| Recovery Rec@3 | .5149 | .5165 | +0.16 pp |
| Recovery Rec@5 | .5379 | .5381 | +0.02 pp |
| Recovery MRR | .4361 | .4356 | -0.0005 |

**Interpretation:**

Full interaction is the overall Train-Val winner.

NGramRecency is the Recovery@1 / Recovery MRR winner and avoids BGE inference.

The full model's overall advantage is very small; this is why NGramRecency remains the recommended practical main method.


### 10.4 Short-term recency signal

Recency-only is surprisingly strong for recovery:

PV1 Recovery@1       = .1900
Recency-only Rec@1   = .3505
Delta                = +16.05 pp

But its overall Macro only reaches .410765, far below NGramRecency .429091.

This suggests:

Recent personal usage is highly informative for promoting recoverable personal candidates, but lexical context is needed to determine when that recent preference is appropriate for the current expression.

## 10A. Consolidated scientific findings and relation to historical PV2

This section records the main scientific interpretation that should be preserved for the thesis. The key result is not merely that one final configuration has the highest Macro Top1. The ablations show where context helps, which type of context helps most, and how recency should be combined with context.

### 10A.1 Main finding: context is useful when used for final-list ordering

The current Initial-Pinyin experiments show that context contains useful information once candidate availability has already been established by PV1.

On all standardized Initial Train-Val rows (N=34,416):

| Method | Macro Top1 | Delta vs PV1 | Micro Top1 | Top3 | Top5 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PV1 | .401872 | - | .426749 | .598907 | .663093 | .524450 |
| BGE | .405679 | +0.3807 pp | .429887 | .601958 | .665214 | .526979 |
| NGram | .426304 | +2.4432 pp | .449820 | .609600 | .665679 | .539729 |
| Recency-only | .410765 | +0.8893 pp | .433839 | .603208 | .664458 | .529197 |
| BGERecency | .413942 | +1.2070 pp | .438052 | .605532 | .665766 | .532635 |
| NGramRecency | .429091 | +2.7219 pp | .452755 | .611285 | .665940 | .541944 |
| BGERecency + NGramRecency | .429506 | +2.7634 pp | .453423 | .612099 | .667335 | .542766 |

Therefore the current evidence rejects the broad interpretation that "context does not help". A more accurate statement is:

Context is useful for ordering candidates that are already available, even though earlier ways of using context inside the personal-vocabulary recovery/injection stage did not improve the final result.

The strongest practical signal is short lexical context through NGramRecency. Semantic BGE context is also useful, but its contribution is smaller.

### 10A.2 Historical PV2 versus the current design

The historical PV2 mechanism and the current PV1 + BGE mechanism use closely related BGE evidence, but they apply it differently.

Historical PV2 used BGE context as a personal-side injection-strength term. Generic candidates retained their Frequency scores and explicitly had context_support = 0. A personal-only candidate was scored conceptually as:

$$
S_{PV2}(c)=b(q)+\lambda_{PV}F(c)+\lambda_{ctx}C_{BGE}(c)
$$

Plain text:

`Historical PV2 score = Generic boundary + personal-frequency support + BGE context injection support.`

where b(q) is the Generic boundary score. The BGE context support was target-conditioned positive cosine over the candidate's historical interactions, using Top-5 historical contexts and normalization across the personal candidate pool.

In other words, PV2 mainly asked:

How strongly should this personal-only candidate be injected into the Generic ranking?

The current final-list design separates recovery from contextual ordering:

Generic -> Frequency -> PV1 recovery -> frozen PV1 Top10 -> context reranking

After PV1 has produced a final Top10, context is scored across the final candidate list:

$$
S_{new}(c)=S_{PV1}(c)+\lambda_C P_{context}(c)
$$

Plain text:

`Final-list context score = frozen PV1 score + weighted context support.`

and in the full model:

$$
S_{new}(c)=S_{PV1}(c)+\lambda_N P_{NG-R}(c)+\lambda_B P_{BGE-R}(c)
$$

Plain text:

`Full final-list score = frozen PV1 + NGramRecency support + BGERecency support.`

This means context can now support a recovered personal candidate or support a Generic candidate against an incorrect personal override. The task changes from one-sided personal promotion to candidate-wide contextual discrimination.

**Historical PV2 evidence**

On the frozen historical six-author Full+Short/H5000 Test:

| Method | Macro Top1 |
| --- | ---: |
| Generic | .723167 |
| Frequency | .771833 |
| PV1 | .779000 |
| PV2 | .778167 |

Thus:

PV2 - PV1 = -0.000833 = -0.0833 percentage points

The historical PV family reduced Missing from .089667 to .066167, and the selected PV1 configuration was Kpv=1, lambda_pv=4. However, adding the PV2 BGE context term did not improve Macro Top1 over PV1 on that historical benchmark.

**Current final-list evidence**

On the current standardized Initial Train-Val final-list task:

PV1       = .401872
PV1+BGE   = .405679   (+0.3807 pp)
PV1+NGram = .426304   (+2.4432 pp)

So BGE by itself is not useless: when used as a final-list reranking signal it gives a positive, although modest, improvement. The much larger gain comes from the lexical NGram signal.

**Scientific boundary on the PV2 comparison**

The historical PV2 result and the current Initial Train-Val result are not the same dataset/split/condition, so they cannot be used as a controlled causal proof that changing only the stage of context application caused the improvement.

What the combined evidence supports is the following bounded interpretation:

Historical PV2 showed that using BGE primarily to strengthen personal-candidate injection did not improve over PV1. The current Initial-Pinyin study shows that, after candidate recovery is fixed, applying context as final-list reranking does improve ranking. This supports a two-stage design in which recovery determines availability and context determines ordering.

A strict causal test of the stage-placement hypothesis would require a matched same-surface ablation on the current Train-Val data: personal-only BGE injection versus all-candidate BGE reranking using the same history, BGE support, candidate set, and tuning protocol.

### 10A.3 Lexical context is the dominant context signal

Plain BGE is useful but small:

PV1 = .401872
BGE = .405679
Delta = +0.3807 pp Macro Top1

Plain NGram is much stronger:

NGram = .426304
Delta vs PV1 = +2.4432 pp Macro Top1

Directly comparing the two current plain context signals:

NGram - BGE = +2.0625 pp Macro Top1
MRR: .539729 - .526979 = +.012750
Top3: .609600 - .601958 = +0.7642 pp
Recovery Rec@1: .3477 - .2656 = +8.21 pp

The same ordering was already visible in the earlier personal-candidate scoring study (Gold in Personal K5, K>=2, n=4,471):

| Candidate-only scorer | Macro Top1 | Micro Top1 | Top3 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Frequency | .491082 | .494968 | .847685 | .683822 |
| BGE64 | .536167 | .538582 | .885484 | .718840 |
| Plain NGram@2 | .560105 | .563185 | .876090 | .728331 |
| Hard NGramRecency@2,tau=2048 | .591558 | .594945 | .895996 | .750563 |

The converging interpretation is:

BGE64 captures longer-range semantic similarity between the current context and historical contexts.

NGram captures very short lexical context, with the selected maxN=2, i.e. exact matching/backoff over the last one or two context characters.

For this Initial-Pinyin task, short lexical context is substantially more predictive than the tested BGE semantic signal.

This should remain a task-bounded claim, not a universal claim that lexical context is always better than semantic context.

### 10A.4 BGE and NGram are conceptually complementary, but the measured extra BGE gain is small

The conceptual roles are complementary:

BGE64  -> longer-range semantic relevance
NGram  -> short-range lexical relevance

However, the data show that simply adding plain BGE to plain NGram contributes almost nothing to Macro Top1:

NGram             = .426304
BGE + NGram       = .426318
Delta             = +0.0014 pp

Therefore it would be too strong to write that plain BGE+NGram gives a large combined improvement.

The complementary gain appears more clearly after recency is integrated into both context signals:

NGramRecency                    = .429091
BGERecency + NGramRecency       = .429506
Delta                           = +0.0415 pp Macro

Additional differences for the full interaction model over NGramRecency:

| Metric | NGramRecency | Full interaction | Full - NGramRecency |
| --- | ---: | ---: | ---: |
| Macro Top1 | .429091 | .429506 | +0.0415 pp |
| Micro Top1 | .452755 | .453423 | +0.0668 pp |
| Top3 | .611285 | .612099 | +0.0814 pp |
| Top5 | .665940 | .667335 | +0.1395 pp |
| MRR | .541944 | .542766 | +.000822 |

Thus the thesis-safe interpretation is:

Semantic context provides a small complementary signal beyond the stronger lexical-temporal signal. The full semantic-temporal + lexical-temporal model is the overall Train-Val winner, but the additional gain over NGramRecency is small and should not be described as statistically significant without a paired test or holdout confirmation.

### 10A.5 Recency is useful, but works better when integrated into relevant context evidence

Recency-only already contains strong personal preference information:

PV1 Macro                = .401872
Recency-only Macro       = .410765
Delta                    = +0.8893 pp

PV1 Recovery Rec@1       = .1900
Recency-only Rec@1       = .3505
Delta                    = +16.05 pp

However, the factorial ablation shows that recency is more useful when it weights relevant historical evidence than when it is simply added as another independent feature.

**Lexical case**

NGram                         = .426304
NGram + independent Recency   = .426361
NGramRecency interaction      = .429091

Therefore:

Independent Recency gain over NGram = +0.0057 pp
NGramRecency over additive NGram+R  = +0.2730 pp

**Recovery@1:**

NGram + independent R = .3513
NGramRecency           = .3601
Delta                  = +0.88 pp

**Semantic case**

BGE                         = .405679
BGE + independent Recency   = .411150
BGERecency interaction      = .413942

Therefore:

BGERecency - additive BGE+R = +0.2792 pp Macro

**The common interpretation is:**

ContextRelevance(q,h) \times Recency(h)

is more effective than:

ContextScore(c) + RecencyScore(c)

The intended interpretation is not merely "recent words are better". It is:

Among historical interactions that are relevant to the current context, more recent interactions should count more strongly.

This is supported independently by both the lexical and semantic ablations.

### 10A.6 Recovery availability is not the main discriminating metric in this stage

Because every current context experiment reranks the same frozen PV1 Top10, context cannot create new candidate availability.

Across all final-list context methods:

Missing@10 = .291144   [invariant]
Recovery Rec@10 = .5401 [invariant on fixed R]

Therefore Missing@10 and Rec@10 cannot tell the context methods apart. The availability problem has already been decided upstream by Generic + Personal K5 + PV1.

What still changes is the rank of an already recovered candidate:

| Method | Recovery Rec@1 | Delta vs PV1 | Rec@3 | Rec@5 | RecMRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| PV1 | .1900 | - | .4923 | .5363 | .3363 |
| BGE | .2656 | +7.56 pp | .5100 | .5369 | .3835 |
| NGram | .3477 | +15.77 pp | .5167 | .5387 | .4301 |
| Recency-only | .3505 | +16.05 pp | .5191 | .5383 | .4316 |
| NGramRecency | .3601 | +17.01 pp | .5149 | .5379 | .4361 |
| BGERecency + NGramRecency | .3582 | +16.82 pp | .5165 | .5381 | .4356 |

So the correct distinction is:

Recovery availability:
    Can the Gold candidate enter the final Top10?
    -> decided upstream; fixed in this experiment.

Recovered-candidate ranking:
    Once the candidate is present, can it be moved to rank 1 / rank 3?
    -> strongly affected by context reranking.

Therefore Recovery@1 / Recovery MRR remain useful diagnostics, but they should be described as ranking of already-recovered candidates, not as increased recoverability.

### 10A.7 Resulting two-stage architecture

The combined evidence supports the following architecture:

```text
Stage 1: Candidate availability / personal recovery
Generic PinyinGPT
    -> Frequency
    -> Personal Vocabulary / PV1 recovery
    -> final candidate set

Stage 2: Contextual ordering
frozen PV1 Top10
    -> lexical-temporal reranking (NGramRecency)
    -> optional semantic-temporal augmentation (BGERecency)
    -> final ordered Top10
```

**The conceptual separation is:**

Recovery / preference answers:
    "Which personal candidate should be available?"

Context reranking answers:
    "Given the available candidates, which candidate is appropriate now?"

This is the central methodological conclusion of the current study.

### 10A.8 Thesis-safe consolidated conclusion

**A thesis-ready bounded statement is:**

The experiments indicate that personal recovery and contextual selection are better treated as separate stages. Historical PV2 used BGE context mainly to strengthen personal-candidate injection and did not improve over PV1 on the frozen six-author Full+Short benchmark (Macro Top1 .779000 for PV1 versus .778167 for PV2). In the current standardized Initial-Pinyin Train-Val study, freezing PV1 candidate availability and then applying context to final-list reranking produces clear gains: plain BGE improves Macro Top1 from .401872 to .405679, while short lexical NGram reranking reaches .426304. Adding history recency directly as an independent factor provides limited additional benefit once NGram is present, whereas integrating recency into context-relevant historical evidence is consistently stronger, reaching .429091 with NGramRecency. The full BGERecency + NGramRecency model reaches the best overall Macro Top1 of .429506, although its additional gain over NGramRecency is small. Because the candidate set is frozen, these improvements should be interpreted as better contextual ordering, not increased recoverability: Missing@10 remains .291144 and Recovery Rec@10 remains .5401, while the rank of already-recovered candidates improves substantially (PV1 Recovery@1 .1900 to NGramRecency .3601).

### 10A.9 Reproducibility boundary for this interpretation

Historical PV2 evidence is a frozen historical comparison, not a source for current method tuning:

Historical PV/EM1 audit HEAD = e500417
Historical tag = em1-pv-same-surface-audit-20260819
Historical six-author Full+Short Test must not be reopened for current selection.

Historical PV2 implementation semantics to preserve:

PV2 context version = target-conditioned-positive-cosine-top5-normalized-v1
PV2 context lambda grid = {0.5, 1, 2, 4}
Generic candidates: context_support = 0
Personal candidate final score:
    generic_boundary + lambda_pv * frequency + lambda_ctx * context

Current final-list conclusions are based only on the standardized Initial Train-Fit / Train-Val development protocol recorded in this document:

Train-Fit = 144,526
Train-Val = 34,416
Dev3000 used = false
Test used = false
Gold used for context feature/scoring construction = false

All frozen hashes, runner hashes, commands, output paths, and artifact-freeze commands remain recorded in the reproducibility sections below. The historical PV2 numbers must be labeled as historical Full+Short Test evidence and must never be numerically pooled with the current Initial Train-Val results.

## 11. Recommended thesis framing

The evidence now favors this decomposition:

Long-term Personal Preference
+ Lexical-Temporal Relevance
+ optional Semantic-Temporal Relevance

### 11.1 Main practical model

$$
S(c)=S_{PV1}(c)+\lambda_N P_{NG-R}(c)
$$

Plain text:

`Practical model = frozen PV1 score + weighted NGramRecency support.`

Development-selected:

maxN = 2
tau_N = 2048
lambda_N = 6

### 11.2 Full model

$$
S(c)=S_{PV1}(c)+\lambda_N P_{NG-R}(c)+\lambda_B P_{BGE-R}(c)
$$

Plain text:

`Full model = frozen PV1 score + weighted NGramRecency support + weighted BGERecency support.`

Development-selected:

lambda_N = 4
lambda_B = 4
tau_N = 2048
tau_B = 2048
BGE context = last 64 chars
BGE historical retrieval = candidate-conditioned Top-5 by cosine only

### 11.3 Thesis-safe wording

Recommended bounded statement:

On the standardized Initial-Pinyin Train-Val final-list reranking task, short lexical suffix matching combined with same-user interaction recency provides the strongest practical context signal. A recency-weighted semantic BGE signal provides a small additional improvement in overall ranking metrics, while NGramRecency alone remains slightly stronger on Recovery@1 and Recovery MRR. Factorial ablation shows that recency is more effective when it modulates context-relevant historical interactions than when it is introduced as an independent additive preference score.

Do not convert this into a universal claim that lexical context is always better than semantic context.

## 12. Candidate-scoring background that motivated the final-list study

Earlier candidate-only population:

- Gold in Personal K5
- candidate count >= 2
- n = 4,471

| Method | Macro Top1 | Micro Top1 | Top3 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Frequency F | .491082 | .494968 | .847685 | .683822 |
| BGE64 | .536167 | .538582 | .885484 | .718840 |
| Plain NGram@2 | .560105 | .563185 | .876090 | .728331 |
| Interpolated NGramRecency | .590783 | .594051 | .898009 | .750760 |
| Hard NGramRecency@2,tau=2048 | .591558 | .594945 | .895996 | .750563 |
| Q8 | .636531 | .642362 | .912995 | .783568 |
| Q8+F @ alpha=.75 | .669164 | .675688 | .925744 | .804015 |

Selected candidate-scoring latency from the historical NGram runner:

Plain NGram@2 mean = 0.0834867 ms
NGramRecency@2,tau=2048 mean = 0.0843688 ms

Historical BGE64 candidate scorer:

online mean = 2.136 ms
p95 = 3.052 ms

These historical candidate-only latencies are useful background but are not perfectly apples-to-apples with the current final-list context runner latency scope.

## 13. Reproducibility

### 13.1 Environment

Worktree:

C:\Users\chiar\Desktop\LBH\thesis-initial-research

Branch:

work/initial-personalisation

Python:

C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe

Frozen PinyinGPT checkpoint used upstream:

C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat

Frozen Generic identity:

aihijo/transformers4ime-pinyingpt-concat@76dd20dc92d8236a350fb732e99dde6fa15e2263
beam_size = 16
top_k = 10
production-compatible n_positions = 1024

BGE model:

C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf

Observed CUDA path in V1/V2 runs:

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8

### 13.2 Script hashes

| Runner | SHA256 |
| --- | ---: |
| run_initial_pv1_context_reranking_v2.py | 3f140f7c5ce32b5e37297472c2b5e63b603085f07453bfa236dffbe421b96b44 |
| run_initial_pv1_bge_ngram_context_reranking_v1.py | 9082d0f72143c86e11d0f4addceb219e4dd3f5cc69bf58a55c2b7b4d4b8a87cb |
| run_initial_pv1_bge_recency_ngram_context_reranking_v2.py | 0d02ede6819ca824b703816d30fa7276b429f1e834baa866c0add73b65d6a3db |
| run_initial_pv1_factorial_context_ablation_v3.py | b8f71397c9ab7271b2cac2f2340bce76ed9a5ae9d42b5d15b4c47214a4942a8c |

Before reproducing, verify your local copies match these hashes.

### 13.3 Reproduce BGE + NGram V1

From the repository root:

```powershell
$py   = "C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe"
$root = ".\results\personalisation\initial_recovery_comparison_v1"
$bge  = "C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf"

& $py -m experiments.initial_personalisation.run_initial_pv1_bge_ngram_context_reranking_v1 `
    --fit "$root\initial_train_fit_v1.jsonl" `
    --val "$root\initial_train_val_v1.jsonl" `
    --candidate-surface "$root\candidate_surface\train_val_candidate_surface.jsonl" `
    --frequency-pv1-predictions "$root\frequency_pv1\predictions.jsonl" `
    --bge-model "$bge" `
    --seed-bge-cache "$root\candidate_scoring_q8_bge64_v1\bge64\history_embedding_cache.sqlite3" `
    --output-root "$root\pv1_bge_ngram_context_reranking_v1"
```

Expected output root:

results\personalisation\initial_recovery_comparison_v1\pv1_bge_ngram_context_reranking_v1

Key output files include:

run_manifest.json
bge_history_embedding_cache.sqlite3
bge_scores.jsonl
bge_scoring_summary.json
grid_results.csv
family_selection.json
selected_method_metrics.csv
selected_subset_metrics.csv
per_author_metrics.csv
recovery_metrics.csv
rank_transition_metrics.csv
selected_predictions.jsonl
context_diagnostics.json
latency.json
comparison.json
artifact_checksums.json

### 13.4 Reproduce BGERecency V2

```powershell
$py   = "C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe"
$root = ".\results\personalisation\initial_recovery_comparison_v1"
$bge  = "C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf"

& $py -m experiments.initial_personalisation.run_initial_pv1_bge_recency_ngram_context_reranking_v2 `
    --fit "$root\initial_train_fit_v1.jsonl" `
    --val "$root\initial_train_val_v1.jsonl" `
    --candidate-surface "$root\candidate_surface\train_val_candidate_surface.jsonl" `
    --frequency-pv1-predictions "$root\frequency_pv1\predictions.jsonl" `
    --bge-model "$bge" `
    --base-bge-root "$root\pv1_bge_ngram_context_reranking_v1" `
    --output-root "$root\pv1_bge_recency_ngram_context_reranking_v2"
```

Important V2 provenance rule:

V1 BGE artifacts are read-only.
V2 reuses:
  bge_scores.jsonl
  bge_history_embedding_cache.sqlite3
  V1 run_manifest.json / comparison.json

Expected output root:

results\personalisation\initial_recovery_comparison_v1\pv1_bge_recency_ngram_context_reranking_v2

Expected outputs:

run_manifest.json
bge_recency_scores.jsonl
bge_recency_scoring_summary.json
grid_results.csv
family_selection.json
selected_method_metrics.csv
selected_subset_metrics.csv
per_author_metrics.csv
recovery_metrics.csv
rank_transition_metrics.csv
selected_predictions.jsonl
context_diagnostics.json
latency.json
comparison.json
artifact_checksums.json

### 13.5 Reproduce factorial additive-vs-interaction V3

V3 performs no BGE model inference; it reads completed V1/V2 supports.

```powershell
$py   = "C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe"
$root = ".\results\personalisation\initial_recovery_comparison_v1"

& $py -m experiments.initial_personalisation.run_initial_pv1_factorial_context_ablation_v3 `
    --fit "$root\initial_train_fit_v1.jsonl" `
    --val "$root\initial_train_val_v1.jsonl" `
    --candidate-surface "$root\candidate_surface\train_val_candidate_surface.jsonl" `
    --frequency-pv1-predictions "$root\frequency_pv1\predictions.jsonl" `
    --base-bge-root "$root\pv1_bge_ngram_context_reranking_v1" `
    --base-bge-recency-root "$root\pv1_bge_recency_ngram_context_reranking_v2" `
    --output-root "$root\pv1_factorial_context_ablation_v3"
```

Expected V3 output root:

results\personalisation\initial_recovery_comparison_v1\pv1_factorial_context_ablation_v3

Expected outputs:

run_manifest.json
grid_results.csv
family_selection.json
selected_method_metrics.csv
selected_subset_metrics.csv
per_author_metrics.csv
recovery_metrics.csv
rank_transition_metrics.csv
factor_supports.jsonl
selected_predictions.jsonl
context_diagnostics.json
latency.json
comparison.json
artifact_checksums.json

V3 safety behavior:

- refuses a non-empty output directory
- validates V1/V2 are complete
- validates V1/V2 input hashes equal current frozen input hashes
- validates BGE context length and Top-N
- validates V1/V2 candidate lists equal the frozen PV1 rankings
- records prior artifact hashes in run_manifest.json
- writes artifact_checksums.json
- checks candidate-set / Missing@10 / Recovery@10 invariants

### 13.6 Verify local script hashes

Get-FileHash '.\experiments\initial_personalisation\run_initial_pv1_context_reranking_v2.py' -Algorithm SHA256
Get-FileHash '.\experiments\initial_personalisation\run_initial_pv1_bge_ngram_context_reranking_v1.py' -Algorithm SHA256
Get-FileHash '.\experiments\initial_personalisation\run_initial_pv1_bge_recency_ngram_context_reranking_v2.py' -Algorithm SHA256
Get-FileHash '.\experiments\initial_personalisation\run_initial_pv1_factorial_context_ablation_v3.py' -Algorithm SHA256

### 13.7 Verify frozen input hashes

```powershell
$root = '.\results\personalisation\initial_recovery_comparison_v1'

Get-FileHash "$root\initial_train_fit_v1.jsonl" -Algorithm SHA256
Get-FileHash "$root\initial_train_val_v1.jsonl" -Algorithm SHA256
Get-FileHash "$root\candidate_surface\train_val_candidate_surface.jsonl" -Algorithm SHA256
Get-FileHash "$root\train_val_generic\predictions.jsonl" -Algorithm SHA256
Get-FileHash "$root\frequency_pv1\predictions.jsonl" -Algorithm SHA256
```

Expected:

Initial Train-Fit              162f5c98daa86cc69947571e6d8f20fc401f0a82cdd3fd6e517eb7be2addbdb4
Initial Train-Val              d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4
Candidate surface              205c0ba01cd0678d7a4341c503fa2e74cf126a70182ff582687025e4946764b2
Generic predictions            bd0fb4dc304e0b266b90fae6fe3ac65424d2f52b23fedfa881212706ba2c2873
Frequency/PV1 predictions      7fd8aa158d8cd50bced36b55610f8d932bc65e3aae1dbbd5bd65907ff1707ea7

### 13.8 Freeze output artifact hashes

Each current runner writes artifact_checksums.json. Before PRE-DEV freeze, additionally record the checksum of the main comparison and checksum manifest themselves:

```powershell
$root = '.\results\personalisation\initial_recovery_comparison_v1'

Get-FileHash "$root\pv1_bge_ngram_context_reranking_v1\comparison.json" -Algorithm SHA256
Get-FileHash "$root\pv1_bge_ngram_context_reranking_v1\artifact_checksums.json" -Algorithm SHA256

Get-FileHash "$root\pv1_bge_recency_ngram_context_reranking_v2\comparison.json" -Algorithm SHA256
Get-FileHash "$root\pv1_bge_recency_ngram_context_reranking_v2\artifact_checksums.json" -Algorithm SHA256

Get-FileHash "$root\pv1_factorial_context_ablation_v3\comparison.json" -Algorithm SHA256
Get-FileHash "$root\pv1_factorial_context_ablation_v3\artifact_checksums.json" -Algorithm SHA256
```

Do not invent these local output hashes from filenames. Copy them from the actual completed local run / artifact_checksums.json.

### 13.9 Exact protocol checks before citing

Before citing the final-list context numbers, verify all of the following:

[ ] branch = work/initial-personalisation
[ ] Initial Train-Fit hash matches frozen value
[ ] Initial Train-Val hash matches frozen value
[ ] candidate-surface hash matches frozen value
[ ] Frequency/PV1 predictions hash matches frozen value
[ ] 34,416 Train-Val rows
[ ] same three authors
[ ] H5000 before exact-Pinyin filtering
[ ] strictly prior same-author history
[ ] current/future Gold absent from scoring/features
[ ] PV1 K=1, lambda_F=4, lambda_PV=4
[ ] PV1 final scores preserved as score base
[ ] candidate set unchanged
[ ] Missing@10 exactly invariant
[ ] Recovery Rec@10 exactly invariant
[ ] BGE last-64 context
[ ] BGE Top-5 candidate-conditioned history
[ ] BGERecency Top-5 selected by cosine only
[ ] recency used only during BGERecency aggregation
[ ] NGram HardBackoff maxN=2
[ ] interaction age uses same-author interaction distance
[ ] Dev3000 used = false
[ ] Test used = false

## 14. Diagnostics generated by the runners but not printed in the supplied terminal summaries

The V1/V2/V3 runners also produce the following detailed dimensions:

selected_method_metrics.csv
selected_subset_metrics.csv
per_author_metrics.csv
rank_transition_metrics.csv
context_diagnostics.json
latency.json

These files contain / are intended to contain:

PV1 rescue retention and PV1 harm repair

Conflict vs non-Conflict subsets

Context Opportunity rank improved / worsened / unchanged

per-author results

selected-method rank transitions

detailed scorer/rerank latency

The exact values from these files were not included in the terminal text available when this note was written, so this document deliberately does not fabricate them.

Before the context method is PRE-DEV frozen, append or archive these exact local outputs. Recommended inspection:

```powershell
$root = '.\results\personalisation\initial_recovery_comparison_v1\pv1_factorial_context_ablation_v3'
```

Import-Csv "$root\selected_method_metrics.csv" | Format-Table -AutoSize
Import-Csv "$root\selected_subset_metrics.csv" | Format-Table -AutoSize
Import-Csv "$root\per_author_metrics.csv" | Format-Table -AutoSize
Import-Csv "$root\rank_transition_metrics.csv" | Format-Table -AutoSize
Get-Content "$root\context_diagnostics.json"
Get-Content "$root\latency.json"
Get-Content "$root\artifact_checksums.json"

This is the remaining data-completeness step if the goal is a single archival note containing every diagnostic row, rather than every headline metric observed so far.

## 15. Scientific boundary and freeze decision

All results in this note are:

standardized Train-Val development results

They are not:

Dev3000 confirmation results
Test results
independent held-out significance evidence

Current bounded conclusion:

NGramRecency is the strongest practical final-list context reranker found so far. It substantially improves PV1 overall Top1 and Recovery@1 by combining short lexical suffix relevance with causal interaction recency. BGERecency adds a weaker semantic-temporal signal; adding it to NGramRecency gives the best overall Train-Val ranking metrics, but only by a very small margin and with additional BGE inference cost. Factorial ablation indicates that recency is more effective as a modifier of context-relevant history than as an independent additive factor.

Recommended pre-Dev candidates:

Primary practical:
  PV1 + NGramRecency
  maxN=2, tau=2048, lambda_N=6

Full accuracy-oriented:
  PV1 + BGERecency + NGramRecency
  lambda_B=4, lambda_N=4
  tau_B=2048, tau_N=2048

Before final freeze, judge the choice using not only Macro Top1, but also:

PV1 rescue retention
PV1 harm repair
Conflict / non-Conflict
Context Opportunity rank movement
Recovery Rec@1 / Rec@3 / Rec@5
per-author stability
latency

Do not change the method using Dev3000 until the development method is explicitly frozen.
