# EM-2 External Memory Progress Report -2026-08-19

**Purpose:** preserve the complete reasoning, implementation path, results, and reproduction information for the PinyinGPT hidden-state External Memory experiments before the next Adaptive Fusion stage.

**Status:** ACTIVE STAGE REPORT -completed through Fixed G+F+C Dev fusion. Test for the new EM-2 methods has not been opened.

**Scope:** Full+Short, H5000, three exploratory authors (`Etinjat`, `Re_spectators`, `breaddddd`), Dev tune unless explicitly stated otherwise.

**Scientific boundary:** this document records what has actually been run. Hypotheses and future work are labelled separately. New EM-2 Test results have not been inspected.

---

## 1. Why EM-2 was started

Earlier Context experiments showed two facts at the same time:

1. Generic BGE context retrieval often found useful same-Pinyin personal history.
2. Better retrieval did not reliably become better final candidate ranking.

The local-context experiment strengthened this diagnosis. A 64-character BGE context window improved retrieval discrimination, but the final M1 Test result remained almost unchanged and below the simple Frequency baseline.

This motivated a more task-aligned representation rather than another generic semantic-context tweak.

The EM-2 question therefore became:

> Can a representation taken directly from the frozen PinyinGPT prediction process retrieve personal history that is more relevant to the actual Pinyin candidate decision?

The design intentionally reuses the existing causal memory semantics:

- same author / proxy user;
- strictly prior history;
- H5000 applied to all prior interactions before Pinyin filtering;
- exact same segmented Pinyin;
- no current Gold in retrieval;
- no Test tuning.

---

# 2. EM-2A -Hidden-State Engineering Gate

## 2.1 Why this step was necessary

Before evaluating retrieval quality, the project had to establish that the extracted hidden state actually corresponded to the PinyinGPT state used to predict the first Chinese target character.

Without this gate, a later retrieval result could be confounded by extracting the wrong layer, wrong token position, or a representation unrelated to the backend's actual prediction semantics.

## 2.2 Frozen representation

The frozen PinyinGPT2-Concat prompt is:

```text
[CLS] + preceding context + [SEP] + Pinyin tokens + [SEP]
```

The first target character is predicted from the final prompt position.

EM-2 therefore froze:

```text
representation
= final Transformer layer
+ final prompt [SEP] hidden state
```

Hidden size:

```text
768
```

No layer or pooling sweep was allowed after retrieval results.

## 2.3 Engineering result

Nine Dev-only samples were checked.

```text
9 / 9 PASS
hidden size = 768

max hidden -> LM-head logits abs diff
= 2.67028808594e-05

max allowed distribution diff
= 1.14440917969e-05

max first-step direct vs teacher abs diff
= 3.81469726562e-06

max cached beam vs fixed score diff
= 9.17911529541e-06

best allowed next-token agreement
= 9 / 9
```

Gold was not used and retrieval metrics were not inspected.

## 2.4 Interpretation

This stage was not a performance experiment. It established an engineering invariant:

> The selected 768-dimensional final-[SEP] hidden state is consistent with the frozen backend semantics used for first-target prediction.

Poor downstream retrieval would therefore not justify silently switching to another layer or pooling rule.

## 2.5 Runner

```text
experiments/external_memory/em2_hidden_state_gate.py
```

**Reproduction status:** PARTIAL in this report. The runner is preserved, but the original exact CLI invocation should be copied from the stage documentation or shell history into `docs/REPRODUCIBILITY_INDEX.md` before final stage freeze.

---

# 3. EM-2B -Workload Profile and Hidden-State Cache

## 3.1 Why profile the workload first

Before computing thousands of transformer states, the project measured the legal retrieval surface.

This answered:

- how many Dev queries exist;
- how many have legal same-Pinyin history;
- how large the retrieval buckets are;
- how many unique PinyinGPT representations must actually be cached.

This prevents expensive computation from being started without understanding the workload and also creates useful descriptive statistics for the thesis.

## 3.2 Dev workload profile

Population:

```text
History manifest rows      248,082
Dev manifest rows           32,212
Three-author tune queries    5,608
Legal query-history edges  122,067
```

Queries per author:

```text
Etinjat          3,047
Re_spectators      199
breaddddd        2,362
```

Visible same-Pinyin H5000 history per query:

```text
mean      21.767
median     2
max      435

history available   3,625 / 5,608 = 64.64%
no history          1,983 / 5,608 = 35.36%
```

Unique representation rows required:

```text
11,475
```

The distribution is strongly skewed: the mean history depth is much larger than the median.

This is descriptive only; it does not prove effectiveness.

## 3.3 Hidden-state caching

Runner:

```text
experiments/external_memory/em2_cache_hidden_dev.py
```

Verified command:

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  -m experiments.external_memory.em2_cache_hidden_dev `
  --pilot-root 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory' `
  --checkpoint 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat' `
  --output-root 'results\personalisation\external_memory\em2_hidden_dev' `
  --device cuda `
  --batch-size 64
```

Final result:

```text
Cached rows: 11,475
Hidden size: 768
Context-truncated rows: 0

Prompt length:
min = 6
mean = 436.26
max = 517

SQLite SHA256:
9a80a3314c184ccf3f0540916203c651474fad162dc3dab1fc97f7451f441df1

Gold used: False
Target used in representation: False
Retrieval metrics inspected: False
PASS
```

Canonical local cache:

```text
results/personalisation/external_memory/em2_hidden_dev/hidden_states.sqlite3
```

The cache is GENERATED / LOCAL-ONLY and should not be committed as normal source code.

---

# 4. EM-2C -Hidden-State kNN Retrieval Diagnostic

## 4.1 Why retrieval was tested before reranking

The project deliberately separated:

```text
Can the representation find useful history?
```

from:

```text
Can the final ranker use that history correctly?
```

This prevents an end-to-end failure from being incorrectly blamed on the representation when the real problem could be the evidence aggregation rule.

## 4.2 Method

For each Dev tune query:

1. Build the legal H5000 history with the existing `HistoryIndex`.
2. Keep exact same segmented Pinyin.
3. Compare normalized 768-dimensional PinyinGPT hidden states with cosine similarity.
4. Sort deterministically.
5. Evaluate retrieval only after retrieval is complete.

Gold is used only for evaluation, never to construct keys or order history.

## 4.3 Evaluation populations

```text
all queries                5,608
history available          3,625
Gold history available     3,213
Ambiguous + Gold history   1,609
Conflict + Gold history      359
```

`Gold History Available` means the correct target actually exists somewhere in the legal visible history. Retrieval recall is conditional on this because a retriever cannot retrieve a target that is absent from its legal memory.

## 4.4 Result

### Overall Gold-History

```text
Micro:
R@1  = 0.869281
R@5  = 0.971366
R@10 = 0.989107

Macro:
R@1  = 0.890920
R@5  = 0.977833
R@10 = 0.992256
```

### Ambiguous Gold-History

```text
Micro:
R@1  = 0.738968
R@5  = 0.942822
R@10 = 0.978247

Macro:
R@1  = 0.768247
R@5  = 0.957042
R@10 = 0.986311
```

### Conflict Gold-History

```text
Micro:
R@1  = 0.431755
R@5  = 0.768802
R@10 = 0.902507

Macro:
R@1  = 0.424646
R@5  = 0.808620
R@10 = 0.931717
```

Pre-registered primary diagnostic:

```text
Macro-author Ambiguous R@1 = 0.768247
```

## 4.5 Interpretation

The frozen PinyinGPT prediction state contains meaningful information for retrieving personal history.

This stage alone did **not** establish that hidden-state retrieval is superior to BGE or that it improves the final candidate ranking. It only justified continuing to a controlled representation comparison.

Runner:

```text
experiments/external_memory/em2_hidden_knn_dev.py
```

**Reproduction status:** PARTIAL in this report. The runner and output are preserved; the exact original CLI should be copied into the canonical reproducibility index from the stage documentation or shell history.

---

# 5. EM-2D -Same-Surface Representation Comparison

## 5.1 Why this comparison was necessary

The hidden-state retrieval result was not meaningful without a BGE control on the **same exact query/history population**.

The comparison therefore held constant:

- 5,608 Dev tune queries;
- H5000 history semantics;
- exact same-Pinyin filtering;
- Gold-history conditioning;
- Ambiguous / Conflict definitions;
- cosine ranking;
- deterministic evaluation.

Only the representation changed.

Three representations were compared:

```text
BGE Full
BGE ctx64
PinyinGPT hidden
```

BGE Full was included to control for the fact that PinyinGPT hidden uses the full frozen prompt context, while ctx64 uses only the last 64 characters.

## 5.2 Macro-author retrieval result

| Representation | Overall R@1 | Ambiguous R@1 | Conflict R@1 |
|---|---:|---:|---:|
| BGE Full | 88.41% | 73.03% | 34.53% |
| BGE ctx64 | 88.36% | 75.09% | 41.05% |
| **PinyinGPT hidden** | **89.09%** | **76.82%** | **42.46%** |

Primary comparison:

```text
PinyinGPT hidden vs BGE ctx64
Ambiguous Macro R@1
+1.74 percentage points
```

Against BGE Full:

```text
+3.80 pp
```

Conflict retrieval depth:

| Representation | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| BGE Full | 34.53% | 73.86% | 84.39% |
| BGE ctx64 | 41.05% | 70.28% | 80.69% |
| PinyinGPT hidden | **42.46%** | **80.86%** | **93.17%** |

## 5.3 Interpretation

The task-native PinyinGPT representation produced stronger retrieval evidence than both the generic full-context BGE representation and the previously selected ctx64 BGE representation.

The Full-vs-hidden comparison is important: the result cannot be explained only by PinyinGPT seeing a longer context than ctx64.

This established a **retrieval-stage success**, not yet an end-to-end personalisation success.

---

# 6. EM-2E1 -Hidden-M1 End-to-End Reranking

## 6.1 Why Hidden-M1 was the cleanest next experiment

Original M1 is:

```text
BGE representation
-> cosine retrieval
-> Top-N histories
-> positive similarity weights
-> aggregate support by historical target
-> z(Generic) + lambda * context support
```

Hidden-M1 kept this logic and changed only:

```text
BGE representation
->
PinyinGPT hidden representation
```

Frequency and candidate recovery were deliberately excluded.

This is the cleanest test of whether the stronger retrieval representation actually improves the same end-to-end M1 decision rule.

## 6.2 Dev grid

```text
Top-N:
{1, 3, 5, 10, 20}

lambda_hidden:
{0, 0.25, 0.5, 1, 2, 4}
```

Primary:

```text
Macro-author Overall Top1
```

Tie break:

```text
lower lambda, then lower Top-N
```

Because lambda 4 was selected at the grid boundary, a pre-registered single check at lambda 8 was performed. No lambda 16 search was allowed.

## 6.3 Selected configuration

```text
Top-N = 3
lambda_hidden = 4
```

Boundary check at Top-N 3:

```text
lambda 4 -> 0.768748 Macro Overall Top1
lambda 8 -> 0.764567
```

Therefore lambda 4 was frozen.

## 6.4 Hidden-M1 vs Generic

| Subset | G | Hidden-M1 | Delta |
|---|---:|---:|---:|
| Overall | 72.2948% | 76.8748% | +4.5800 pp |
| History Available | 75.9399% | 82.8688% | +6.9289 pp |
| Ambiguous | 68.3605% | 76.6692% | +8.3087 pp |
| Conflict | 46.7924% | 29.9645% | -16.8279 pp |

Generic -> Hidden-M1 transitions:

```text
Overall:
rescue = 448
harm   = 124
net    = +324

Ambiguous:
rescue = 274
harm   = 83
net    = +191

Conflict:
rescue = 45
harm   = 59
net    = -14
```

## 6.5 Hidden-M1 vs Frequency

Frozen Frequency uses `lambda_frequency = 4`.

| Subset | F | Hidden-M1 | Hidden - F |
|---|---:|---:|---:|
| Overall | 76.5240% | 76.8748% | +0.3507 pp |
| History Available | 82.3281% | 82.8688% | +0.5407 pp |
| Ambiguous | 75.4831% | 76.6692% | +1.1861 pp |
| Conflict | 19.7192% | 29.9645% | +10.2454 pp |

F -> Hidden-M1 transitions:

```text
Overall:
rescue = 87
harm   = 59
net    = +28

Conflict:
rescue = 69
harm   = 11
net    = +58
```

This showed that Hidden Context was competitive with Frequency on Dev and was substantially less vulnerable than Frequency on the Conflict subset.

However, Conflict remained far below Generic.

## 6.6 Exact Original M1 control

Original M1 frozen parameters on this surface:

```text
BGE Full
Top-N = 5
lambda_memory = 4
```

Four-way Dev comparison:

| Method | Overall | History | Ambiguous | Conflict |
|---|---:|---:|---:|---:|
| G | 72.2948% | 75.9399% | 68.3605% | 46.7924% |
| F | 76.5240% | 82.3281% | 75.4831% | 19.7192% |
| Original M1 | **76.8888%** | 82.8351% | **76.9454%** | 29.7163% |
| Hidden-M1 | 76.8748% | **82.8688%** | 76.6692% | **29.9645%** |

Hidden-M1 - Original M1:

```text
Overall          -0.0140 pp
History          +0.0337 pp
Ambiguous        -0.2762 pp
Conflict         +0.2482 pp
```

Original M1 -> Hidden-M1 micro transitions:

```text
Overall:
rescue = 65
harm   = 54
net    = +11

Conflict:
rescue = 37
harm   = 16
net    = +21
```

## 6.7 Main conclusion

This was the most important negative/neutral result of EM-2 so far:

> PinyinGPT hidden states were clearly better retrieval representations, but replacing BGE with hidden states inside the unchanged M1 target-support rule did not materially improve end-to-end Macro-author Top1.

Therefore the bottleneck moved from representation quality toward:

```text
retrieved evidence
->
candidate-level decision / evidence aggregation
```

This result must not be rewritten as "hidden states failed." Retrieval improved; end-to-end aggregation did not exploit that improvement.

## 6.8 Reproduction

Runner:

```text
experiments/external_memory/em2_hidden_m1_dev.py
```

Verified command:

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  -m experiments.external_memory.em2_hidden_m1_dev `
  --pilot-root 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory' `
  --generic-cache 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\cache\generic_predictions.jsonl' `
  --hidden-cache 'results\personalisation\external_memory\em2_hidden_dev\hidden_states.sqlite3' `
  --output-root 'results\personalisation\external_memory\em2_hidden_m1_dev'
```

Boundary wrapper:

```text
experiments/external_memory/em2_hidden_m1_dev_boundary8.py
```

Four-way comparison:

```text
experiments/external_memory/em2_four_way_dev_compare.py
```

Command:

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  -m experiments.external_memory.em2_four_way_dev_compare
```

---

# 7. EM-2E2 -Hidden-M2

## 7.1 Why M2 was tested

A stronger retriever may not help M1 if M1's aggregation is too simple.

Original M2 adds a pretrained candidate-aware Cross-Encoder after Stage-1 retrieval.

The experiment therefore asked:

> If only M2 Stage-1 BGE retrieval is replaced by PinyinGPT hidden retrieval, can the existing Cross-Encoder make better use of the improved history set?

The Cross-Encoder itself was **not** changed or fine-tuned.

## 7.2 Original M2 control

Frozen Original M2:

```text
Stage 1: BGE Full retrieval
K = 20

Stage 2:
BAAI/bge-reranker-base

lambda_m2 = 4
```

Old pair-score cache:

```text
C:\Users\chiar\Desktop\LBH\thesis-personalisation\
results\personalisation\m2_h5000\cache\pair_scores.sqlite3
```

The same-surface control found:

```text
Required pair uses: 39,415
Missing pair scores: 0
```

Original M2 Dev:

| Subset | Macro Top1 |
|---|---:|
| Overall | 76.6869% |
| History Available | 82.5123% |
| Ambiguous | 76.3249% |
| Conflict | 25.8543% |

Runner:

```text
experiments/external_memory/em2_original_m2_same_surface_dev.py
```

Command:

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  -m experiments.external_memory.em2_original_m2_same_surface_dev
```

## 7.3 Hidden-M2 workload

Hidden Top-20 preflight:

```text
Queries: 5,608
Requested unique Top20 pairs: 39,415
Cache hits: 0
Missing pair scores: 39,415
Test used: False
```

## 7.4 Hidden-M2 grid and result

Grid:

```text
K in {10, 20}
lambda_m2 in {0.5, 1, 2, 4}
```

Selected:

```text
K = 10
lambda_m2 = 4
```

Selected Hidden-M2:

| Subset | Macro Top1 |
|---|---:|
| Overall | 76.8372% |
| History Available | 82.7441% |
| Ambiguous | 76.8332% |
| Conflict | 29.5521% |

Hidden-M2 - Original M2:

```text
Overall     +0.1503 pp
History     +0.2318 pp
Ambiguous   +0.5083 pp
Conflict    +3.6978 pp
```

Strict same-parameter representation control:

```text
Original M2
BGE K20 lambda4
= 76.6869%

Hidden-M2
Hidden K20 lambda4
= 76.7776%

difference
= +0.0907 pp
```

## 7.5 Interpretation

Hidden retrieval helped M2 slightly, especially on Conflict, which supports the retrieval-stage finding.

However:

```text
Original M1   76.8888%
Hidden-M1     76.8748%
Hidden-M2     76.8372%
Original M2   76.6869%
```

The generic Cross-Encoder still did not outperform the simpler M1 family.

Therefore the generic M2 route was stopped rather than repeatedly retuned.

This also motivates EM-3 later: an IME-specific task-trained matcher is a different research question from continuing to tune the generic pretrained Cross-Encoder.

## 7.6 Reproduction

Runner:

```text
experiments/external_memory/em2_hidden_m2_dev.py
```

Verified command:

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  -m experiments.external_memory.em2_hidden_m2_dev `
  --pilot-root 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory' `
  --generic-cache 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\cache\generic_predictions.jsonl' `
  --hidden-cache 'results\personalisation\external_memory\em2_hidden_dev\hidden_states.sqlite3' `
  --reranker-model 'C:\Users\chiar\Desktop\LBH\thesis\.build\bge-reranker-base' `
  --output-root 'results\personalisation\external_memory\em2_hidden_m2_dev' `
  --batch-size 32
```

The pair cache is resume-safe and stored under the output directory.

---

# 8. Fixed G+F+C Fusion

## 8.1 Why fusion was tested

Frequency and Hidden Context use the same legal personal history but extract different information:

```text
F:
long-term same-Pinyin target preference

C:
which historical situations are most similar to the current situation
```

Row-level transitions showed that Hidden Context could rescue many Frequency failures, especially on Conflict.

Therefore a natural next question was:

> Are Frequency and Hidden Context complementary if both are included in the final score?

## 8.2 Fixed fusion definition

```text
score(c)
=
G(c)
+ lambda_F * F(c)
+ lambda_C * C_hidden(c)
```

Hidden retrieval depth was frozen at:

```text
Top-N = 3
```

The fixed grid was registered before results:

```text
lambda_F, lambda_C
in {0, 0.25, 0.5, 1, 2, 4, 8}
```

Axis controls:

```text
0,0 = G
4,0 = frozen F
0,4 = frozen Hidden-M1
4,4 = equal strong fusion
```

Primary fixed-fusion selection required:

```text
lambda_F > 0
and
lambda_C > 0
```

so that the selected method could not silently collapse to a single-signal baseline.

## 8.3 Result

Selected:

```text
lambda_F = 0.5
lambda_C = 4.0
```

The best-any grid point was the same.

| Method | Overall | History | Ambiguous | Conflict |
|---|---:|---:|---:|---:|
| G | 72.2948% | 75.9399% | 68.3605% | 46.7924% |
| F | 76.5240% | 82.3281% | 75.4831% | 19.7192% |
| Hidden-M1 | 76.8748% | 82.8688% | 76.6692% | 29.9645% |
| GFC 4/4 | 76.7227% | 82.6492% | 76.0751% | 22.3070% |
| **GFC selected 0.5/4** | **76.8825%** | **82.8857%** | **76.6689%** | **28.9951%** |

Selected GFC - Hidden-M1:

```text
Overall     +0.0077 pp
History     +0.0169 pp
Ambiguous   -0.0003 pp
Conflict    -0.9694 pp
```

Hidden-M1 -> selected GFC:

```text
Overall:
rescue = 10
harm   = 9
net    = +1

Conflict:
rescue = 2
harm   = 8
net    = -6
```

F -> selected GFC:

```text
Overall:
rescue = 79
harm   = 50
net    = +29

Conflict:
rescue = 62
harm   = 10
net    = +52
```

## 8.4 Interpretation

A single global pair of Frequency/Context weights did **not** produce meaningful improvement beyond Hidden-M1.

The optimum heavily down-weighted Frequency:

```text
lambda_F = 0.5
lambda_C = 4
```

and was effectively tied with Hidden-M1.

Equal strong weighting (`4,4`) was worse, especially on Conflict.

This indicates that Frequency and Context should not simply be added with the same static confidence for every query.

The result motivates the already registered Adaptive Fusion hypothesis.

## 8.5 Reproduction

Runner:

```text
experiments/external_memory/em2_fixed_gfc_dev.py
```

Verified command:

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  -m experiments.external_memory.em2_fixed_gfc_dev `
  --pilot-root 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory' `
  --generic-cache 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\cache\generic_predictions.jsonl' `
  --hidden-cache 'results\personalisation\external_memory\em2_hidden_dev\hidden_states.sqlite3' `
  --output-root 'results\personalisation\external_memory\em2_fixed_gfc_dev'
```

---

# 9. Current Consolidated Dev Table

All values below use the same three-author, Full+Short, H5000, 5,608-query Dev tune surface.

| Method | Overall | History | Ambiguous | Conflict |
|---|---:|---:|---:|---:|
| G | 72.2948% | 75.9399% | 68.3605% | **46.7924%** |
| F | 76.5240% | 82.3281% | 75.4831% | 19.7192% |
| Original M1 | **76.8888%** | 82.8351% | **76.9454%** | 29.7163% |
| Hidden-M1 | 76.8748% | 82.8688% | 76.6692% | 29.9645% |
| Original M2 | 76.6869% | 82.5123% | 76.3249% | 25.8543% |
| Hidden-M2 | 76.8372% | 82.7441% | 76.8332% | 29.5521% |
| Fixed GFC | 76.8825% | **82.8857%** | 76.6689% | 28.9951% |

Important reading rule:

- These are **Dev** results for method development.
- They do not replace the previously frozen Test conclusions for F/M1/M2.
- New EM-2 Test must remain closed until the remaining design choices are frozen.

---

# 10. What the EM-2 sequence has established

## Verified result 1 -hidden states are technically valid task-native keys

EM-2A established the extraction point before retrieval performance was inspected.

## Verified result 2 -hidden states retrieve better than BGE

On the same Dev retrieval surface, PinyinGPT hidden states improved the primary Ambiguous Macro R@1 and also improved deeper Conflict recall.

## Verified result 3 -retrieval gain does not automatically become end-to-end gain

Hidden-M1 was almost identical to Original M1 despite stronger retrieval.

This strongly suggests that the remaining bottleneck includes the mapping:

```text
retrieved history
->
candidate-level support / final decision
```

## Verified result 4 -generic Cross-Encoder complexity is not the answer

Hidden-M2 improved over Original M2 but remained below the M1 family.

## Verified result 5 -fixed Frequency + Context fusion is insufficient

The best fixed fusion almost collapsed back to Hidden-M1 and produced essentially zero additional Overall gain.

---

# 11. Next Registered Hypothesis -Adaptive Fusion

This hypothesis was discussed and registered **before** the Fixed GFC result was observed.

The idea is to replace global fixed weights:

```text
lambda_F
lambda_C
```

with prediction-visible query-dependent weights:

```text
lambda_F(q)
lambda_C(q)
```

Possible already registered gating information:

```text
visible_history_count
distinct_target_count
frequency_winner_share
frequency_margin
retrieval_top1_similarity
retrieval_similarity_margin
retrieved_target_agreement
```

Important information boundary:

- these features are available at prediction time;
- Gold cannot be used;
- the `Conflict` label cannot be used as an input because its definition depends on Gold.

Conceptual goal:

```text
F confident, C weak
-> trust F more

F weak, C confident
-> trust C more

both weak
-> reduce personalisation and trust G

both strong and agree
-> stronger personalisation

both strong and disagree
-> resolve using prediction-visible confidence
```

This stage should remain transparent before considering a learned black-box gating network.

---

# 12. Research Provenance

EM-2 does not claim to invent nearest-neighbour language-model memory, task-model hidden-state retrieval, or Cross-Encoders.

Relevant precedents:

- Khandelwal et al. (2020), *Generalization through Memorization: Nearest Neighbor Language Models*. External datastore with language-model context representations and nearest-neighbour retrieval.
- Khandelwal et al. (2021), *Nearest Neighbor Machine Translation*. Task-model internal representations used as external-memory keys in conditional generation.
- Tan et al. (2022), *Exploring and Adapting Chinese GPT to Pinyin Input Method*. PinyinGPT task/backend context.
- Nogueira and Cho (2019), *Passage Re-ranking with BERT*. Cross-Encoder reranking precedent.

Thesis-specific adaptation:

- same-user causal external memory;
- H5000 before exact-Pinyin filtering;
- exact segmented-Pinyin retrieval buckets;
- Frozen PinyinGPT prediction-state keys;
- Ambiguous/Conflict diagnostics;
- explicit separation of retrieval, candidate decision, Frequency, and Recovery;
- transparency/control orientation.

Do not claim that the cited work proposed this exact personal Pinyin memory system.

---

# 13. Reproducibility Dependencies

Important local dependencies:

```text
Frozen PinyinGPT checkpoint:
C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat

Frozen Pilot / Dev manifests and Generic cache:
C:\Users\chiar\Desktop\LBH\thesis-personalisation\
results\personalisation\pilot_a_context_memory

Original BGE cache:
...\pilot_a_context_memory\cache\embedding_cache.sqlite3

Original M2 pair cache:
C:\Users\chiar\Desktop\LBH\thesis-personalisation\
results\personalisation\m2_h5000\cache\pair_scores.sqlite3

Frozen generic reranker checkpoint:
C:\Users\chiar\Desktop\LBH\thesis\.build\bge-reranker-base

EM-2 hidden cache:
results\personalisation\external_memory\em2_hidden_dev\hidden_states.sqlite3
```

Key frozen hashes:

```text
Generic Dev cache SHA256:
588aa84c6397e8cb1a13576c0d5dfecd9dd2c4305b45be351328dd83ef62007d

EM-2 hidden cache SHA256:
9a80a3314c184ccf3f0540916203c651474fad162dc3dab1fc97f7451f441df1
```

---

# 14. Repository Files Created / Used in This EM-2 Sequence

Formal experiment runners:

```text
experiments/external_memory/em2_hidden_state_gate.py
experiments/external_memory/em2_cache_hidden_dev.py
experiments/external_memory/em2_hidden_knn_dev.py
experiments/external_memory/em2_hidden_m1_dev.py
experiments/external_memory/em2_hidden_m1_dev_boundary8.py
experiments/external_memory/em2_hidden_m2_dev.py
experiments/external_memory/em2_original_m2_same_surface_dev.py
experiments/external_memory/em2_fixed_gfc_dev.py
```

Comparison / inspection utilities:

```text
experiments/external_memory/em2_four_way_dev_compare.py
```

Generated result roots:

```text
results/personalisation/external_memory/em2_hidden_dev/
results/personalisation/external_memory/em2_hidden_m1_dev/
results/personalisation/external_memory/em2_hidden_m1_dev_boundary8/
results/personalisation/external_memory/em2_original_m2_dev/
results/personalisation/external_memory/em2_hidden_m2_dev/
results/personalisation/external_memory/em2_fixed_gfc_dev/
```

Stage documentation already created or intended:

```text
docs/external_memory/em2/stages/EM2_HIDDEN_KNN_DESIGN_2026-08-19.md
docs/external_memory/em2/stages/EM2A_HIDDEN_STATE_GATE_2026-08-19.md
docs/external_memory/em2/stages/EM2B_DEV_WORKLOAD_PROFILE_2026-08-19.md
docs/external_memory/em2/stages/EM2B_HIDDEN_CACHE_RESULT_2026-08-19.md
docs/external_memory/em2/stages/EM2C_HIDDEN_KNN_DEV_RESULT_2026-08-19.md
docs/external_memory/em2/stages/EM2D_REPRESENTATION_COMPARISON_2026-08-19.md
docs/external_memory/em2/stages/EM2E1_HIDDEN_M1_DESIGN_2026-08-19.md
docs/external_memory/EM2E1_HIDDEN_M1_DEV_RESULT_2026-08-19.md
```

Some of these documents were prepared during the interactive research session; their actual presence in the repository should be audited rather than assumed.

---

# 15. Stage-Close Checklist for EM-2

Before calling EM-2 fully frozen, complete all of the following.

## Method / design

- canonical method/protocol document;
- research question;
- information boundary;
- population;
- grids;
- selection rules;
- tie rules;
- no-Test rule;
- provenance / prior-work note.

## Implementation

- formal runner(s) in `experiments/external_memory/`;
- reusable logic in `src/` only if it is genuinely shared;
- helpers explicitly classified as HELPER;
- tests for reusable scientific invariants where appropriate.

## Results

- generated result namespace preserved;
- summary/grid/prediction artifacts not manually edited;
- output paths documented;
- cache hashes/provenance documented;
- negative results retained.

## Report

- why the step was done;
- what changed relative to the previous step;
- what was held fixed;
- what population was used;
- what was selected;
- numerical results with Dev/Test scope;
- interpretation;
- limitations;
- next-step motivation.

## Reproducibility

- exact runner;
- exact command;
- required local model/data/cache dependencies;
- hashes or manifest identity;
- expected output directory;
- reproduction status.

## Repository navigation

Update:

```text
docs/FILE_INDEX.md
docs/REPRODUCIBILITY_INDEX.md
docs/VERSION_HISTORY.md
```

Update the current technical handoff when the active stage changes substantially.

## Freeze / Git

When the stage is genuinely complete:

- inspect changes;
- explicitly stage reviewed human-authored files;
- do not `git add .`;
- do not add large generated results/caches;
- commit;
- tag/checkpoint if this is a formal stage close;
- record commit/tag in the report and reproduction index.

---

# 16. Current Status and Next Action

Current status:

```text
EM-2A engineering gate              COMPLETE
EM-2B hidden cache                  COMPLETE
EM-2C hidden retrieval diagnostic   COMPLETE
EM-2D BGE vs hidden comparison      COMPLETE
EM-2E1 Hidden-M1                    DEV COMPLETE
EM-2E2 Hidden-M2                    DEV COMPLETE
Fixed G+F+C                         DEV COMPLETE
Adaptive G+F+C                      NEXT
New EM-2 Test                       NOT OPENED
```

Immediate next research action:

> Design and freeze a transparent prediction-visible Adaptive Fusion rule on Dev before inspecting any new EM-2 Test result.
