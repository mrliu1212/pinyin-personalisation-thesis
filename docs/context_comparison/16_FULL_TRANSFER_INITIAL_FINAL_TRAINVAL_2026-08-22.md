# Full+Short Zero-Shot Transfer of the Frozen Initial-Pinyin Recovery鈥揅ontext Model

**Date:** 2026-08-22
**Worktree:** `C:\Users\chiar\Desktop\LBH\thesis-context-compare`
**Scientific status:** post-Dev follow-up; zero-shot transferred frozen Initial-final configuration; **Train-Val descriptive evaluation only**
**Hyperparameter search on Full:** No
**Dev3000 used:** No
**Test used:** No

---

## 1. Purpose

This follow-up experiment asks whether the final recovery-plus-context architecture selected under the **Initial-Pinyin** condition transfers to the standardized **Full+Short** setting without any Full-specific retuning.

The transferred architecture is:

$$
\text{Full Generic}
\rightarrow
\text{Personal K5 Recovery}
\rightarrow
\text{Balanced Stage1}
\rightarrow
\text{NGramRecency}
\rightarrow
\text{BGERecency}
$$

The experiment is deliberately kept separate from the already frozen seven-system standardized context comparison. It is **not** treated as an eighth pre-Dev method because the idea was introduced after the standardized Dev comparison had already been observed.

---

## 2. Population and causal history

Evaluation population:

- standardized Full+Short Clean3 Train-Val;
- Train-Fit rows: **144,526**;
- Train-Val rows: **34,416**;
- same-author history only;
- strictly prior interactions only;
- latest **H5000 raw interactions** first;
- exact segmented-Pinyin filtering only after the H5000 budget is applied;
- earlier causally prior Train-Val rows may enter history for later Train-Val queries;
- Gold is never used for candidate construction or online scoring.

Frozen Train-Fit SHA256:

```text
547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6
```

Frozen Train-Val SHA256:

```text
d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220
```

Frozen Full Train-Val Generic predictions SHA256:

```text
cf4ae382fa23e5ec1154bf28320d13ac1d6ca9600e9dcf8a6aa599600bc28eab
```

---

## 3. Transferred frozen model

### 3.1 Stage 1 鈥?balanced personal recovery

The transferred Stage-1 score is:

$$
S_{\mathrm{REC}}(c)
=
B
+
4P_{\mathrm{NG}}(c)
+
4CS(c)
+
2C_E
$$

where:

- $B$ is the Generic boundary score;
- $P_{\mathrm{NG}}(c)$ is the candidate-specific interpolated personal N-gram signal;
- $CS(c)$ is the candidate Choice Share in visible same-Pinyin history;
- $C_E$ is entropy-based concentration of the same-Pinyin target distribution.

The transferred personal N-gram component is:

```text
Type    = InterpolatedNGramRecency
maxN    = 2
kappa   = 1
tau     = 2048
```

Personal candidate budget:

$$
K=5
$$

### 3.2 Stage 2 鈥?NGramRecency

After Stage 1, the candidate set is fixed. NGramRecency uses hard suffix backoff with:

$$
\lambda_N=4
$$

$$
N_{\max}=2
$$

$$
\tau_N=2048
$$

### 3.3 Stage 2 鈥?BGERecency

The final score is:

$$
S_{\mathrm{final}}(c)
=
S_{\mathrm{REC}}(c)
+
4P_{\mathrm{NG-R}}(c)
+
6P_{\mathrm{BGE-R}}(c)
$$

For candidate $c$, BGERecency support is:

$$
R_B(c)
=
\sum_{h\in\mathrm{Top5}_{\cos}(H_c)}
\max(0,\cos(E(q),E(h)))
\exp\left(-\frac{\mathrm{age}(h)}{2048}\right)
$$

Frozen BGERecency settings:

```text
context               = last 64 Chinese characters
retrieval              = candidate-conditioned, cosine only
TopN per candidate     = 5
recency tau            = 2048
lambda_B               = 6
aggregation            = max(0, cosine) * exp(-age/tau)
BGE model SHA256       = 5a88d266870fbd27c6f329df60de80e2d4cf3bbd5e6f080bd5c1b2e5abb12039
```

No Full-specific $K$, $\lambda$, $\tau$, $N_{\max}$, or gating parameter was searched.

---

## 4. Candidate availability

Observed candidate-surface statistics:

| Quantity | Value |
|---|---:|
| Train-Val rows | 34,416 |
| Empty frozen Generic surfaces | 2 |
| Generic Missing@10 rows | 2,382 |
| Generic-missing rows with Gold in Personal K5 | 620 |
| Recoverability among Generic-missing | 26.03% |
| Rows with at least one Personal K5 candidate | 3,556 |
| Total Personal K5 candidates | 6,942 |

The theoretical Full recoverability among Generic-missing rows is therefore:

$$
\frac{620}{2382}
=
26.03\%
$$

For the two rows where the frozen Generic surface is empty, the Initial generic-boundary anchor is undefined. The runner therefore applies the conservative zero-shot policy: **no Personal-K5 injection and no Stage-2 reranking on those rows**. This avoids inventing a Full-specific boundary rule.

---

## 5. Overall results

| Method | Macro Top1 | Micro Top1 | Top3 | Top5 | MRR@10 | Missing@10 | Mean rank given Top10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Generic | 70.079% | 73.602% | 87.166% | 90.292% | .80940 | 6.921% | 1.491 |
| Frequency | 77.688% | 80.808% | 89.563% | 91.391% | .85465 | 6.921% | **1.304** |
| M1 | 77.824% | 81.000% | 89.293% | 91.251% | .85498 | 6.921% | 1.312 |
| Stage1 | 78.852% | 81.834% | 91.004% | 93.015% | .86722 | **5.195%** | 1.319 |
| **Final** | **79.537%** | **82.450%** | **91.190%** | **93.067%** | **.87104** | **5.195%** | 1.305 |

Relative to M1, the Final transferred model changes Macro Top1 by:

$$
79.537-77.824
=
+1.713\text{ pp}
$$

Micro Top1:

$$
82.450-81.000
=
+1.450\text{ pp}
$$

Top3:

$$
91.190-89.293
=
+1.897\text{ pp}
$$

Top5:

$$
93.067-91.251
=
+1.816\text{ pp}
$$

The gain is therefore not restricted to Top1; it is also visible in Top3, Top5, MRR, and candidate coverage.

---

## 6. Decomposition: recovery vs contextual ordering

### 6.1 Frequency to Stage1

Macro Top1:

$$
77.688\%
\rightarrow
78.852\%
=
+1.165\text{ pp}
$$

Other changes:

- Micro Top1: **+1.026 pp**;
- Top3: **+1.441 pp**;
- Top5: **+1.624 pp**;
- MRR@10: **.85465 -> .86722**;
- Missing@10: **6.921% -> 5.195%**.

Stage1 therefore provides both candidate availability gain and ranking gain.

### 6.2 Stage1 to Final

Macro Top1:

$$
78.852\%
\rightarrow
79.537\%
=
+0.684\text{ pp}
$$

Other changes:

- Micro Top1: **+0.616 pp**;
- Top3: **+0.186 pp**;
- Top5: **+0.052 pp**;
- MRR@10: **.86722 -> .87104**;
- Missing@10: **5.195% -> 5.195%**.

The invariant Missing@10 after Stage1 is expected because Stage2 reranks a fixed recovered candidate set.

The observed decomposition directly supports:

$$
\boxed{\text{Recovery determines availability; Context determines ordering.}}
$$

---

## 7. Recovery diagnostics

Among the **620** Generic-missing rows whose Gold is available in Personal K5:

| Metric | Stage1 | Final | Change |
|---|---:|---:|---:|
| Rec@1 | 76.13% (472/620) | **81.45% (505/620)** | +5.32 pp |
| Rec@3 | 93.23% (578/620) | **94.35% (585/620)** | +1.13 pp |
| Rec@5 | 96.29% (597/620) | **97.10% (602/620)** | +0.81 pp |
| Rec@10 | **99.52% (617/620)** | **99.52% (617/620)** | 0 |
| Recovery MRR@10 | .85115 | **.88357** | +.03242 |

Thus Stage1 almost completely exposes the theoretically recoverable surface, while Stage2 mainly pushes the recovered Gold candidate toward the top.

Only:

$$
620-617=3
$$

recoverable rows fail to retain Gold in the final Top10.

---

## 8. Ambiguous subset

Ambiguous population:

$$
n=10{,}053
$$

| Method | Macro Top1 | Micro Top1 | Top3 | Top5 | MRR@10 | Missing@10 |
|---|---:|---:|---:|---:|---:|---:|
| Generic | 67.508% | 68.189% | 87.098% | 91.296% | .78278 | 5.242% |
| Frequency | 77.642% | 78.156% | 91.853% | 93.604% | .85046 | 5.242% |
| M1 | 78.267% | 78.812% | 90.928% | 93.126% | .85160 | 5.242% |
| Stage1 | 78.081% | 78.574% | 93.315% | 95.633% | .86054 | 2.925% |
| **Final** | **80.217%** | **80.682%** | **93.952%** | **95.812%** | **.87363** | **2.925%** |

Final vs M1 Macro Top1:

$$
80.217-78.267
=
+1.950\text{ pp}
$$

Stage2 alone on Ambiguous rows:

$$
80.217-78.081
=
+2.136\text{ pp}
$$

This is substantially larger than the overall Stage2 gain of +0.684 pp, indicating that lexical and semantic recency are especially useful where candidate ambiguity is genuinely high.

---

## 9. Conflict subset

Conflict population:

$$
n=1{,}865
$$

| Method | Macro Top1 | Micro Top1 | Top3 | Top5 | MRR@10 | Missing@10 |
|---|---:|---:|---:|---:|---:|---:|
| **Generic** | **32.293%** | **29.276%** | 64.182% | 74.370% | .48804 | 15.550% |
| Frequency | 16.273% | 15.442% | 71.903% | 79.249% | .43852 | 15.550% |
| M1 | 25.854% | 24.075% | 67.989% | 76.997% | .47465 | 15.550% |
| Stage1 | 15.280% | 14.531% | 72.976% | 82.735% | .44329 | **10.831%** |
| Final | 22.897% | 21.555% | **76.193%** | **83.646%** | **.48919** | **10.831%** |

Stage1 is slightly worse than Frequency at Conflict Top1, consistent with stronger personal-distribution evidence reinforcing a misleading personal prior.

However, Stage2 produces a large correction:

$$
22.897-15.280
=
+7.618\text{ pp}
$$

Despite this correction, Final remains below M1 on Conflict Top1:

$$
22.897\%
<
25.854\%
$$

and below Generic:

$$
22.897\%
<
32.293\%
$$

At deeper ranks, however, Final is substantially stronger than M1:

$$
\Delta\text{Top3}
=
76.193-67.989
=
+8.204\text{ pp}
$$

$$
\Delta\text{Top5}
=
83.646-76.997
=
+6.649\text{ pp}
$$

Final MRR is also slightly higher than M1:

$$
0.48919
>
0.47465
$$

This suggests that the correct alternative is often already retrieved and promoted near the top, but a misleading personal-preference candidate still occupies rank 1.

The remaining problem is therefore increasingly one of final arbitration:

$$
\boxed{\text{When should personal history be overridden by current-context evidence?}}
$$

---

## 10. Per-author overall Top1

| Author | Frequency | M1 | Stage1 | Final | Final - M1 |
|---|---:|---:|---:|---:|---:|
| Agent Phage | 87.42% | 87.77% | 88.28% | **88.76%** | +0.98 pp |
| Etinjat | 56.72% | 56.50% | 58.85% | **60.00%** | **+3.50 pp** |
| breaddddd | 88.92% | 89.20% | 89.43% | **89.85%** | +0.66 pp |

All three authors improve relative to M1. The aggregate gain is therefore not caused by a single author, although Etinjat receives the largest benefit.

---

## 11. Rescue / harm transitions

| Transition | Rescue | Harm | Net |
|---|---:|---:|---:|
| Frequency -> Stage1 | 472 | 119 | +353 |
| Stage1 -> Final | 464 | 252 | +212 |
| Frequency -> Final | 893 | 328 | +565 |
| M1 -> Final | 899 | 400 | +499 |

Stage1 rescue-to-harm ratio:

$$
\frac{472}{119}
\approx
3.97
$$

Stage2 rescue-to-harm ratio:

$$
\frac{464}{252}
\approx
1.84
$$

Stage1 is therefore comparatively conservative, whereas Stage2 performs more aggressive context-driven corrections and accepts more reversals in exchange for additional rescue.

---

## 12. Main conclusions

### Conclusion 1 鈥?The frozen Initial architecture transfers to Full

Without Full-specific tuning, the Final model reaches:

$$
\boxed{79.537\%\ \text{Macro-author Top1}}
$$

compared with:

$$
77.824\%
$$

for M1 and:

$$
77.688\%
$$

for Frequency.

This is a strong descriptive zero-shot transfer result, but not yet a statistical superiority claim.

### Conclusion 2 鈥?Recovery and context make distinct contributions

The gain separates into:

$$
\text{Frequency}\rightarrow\text{Stage1}
=
+1.165\text{ pp}
$$

followed by:

$$
\text{Stage1}\rightarrow\text{Final}
=
+0.684\text{ pp}
$$

Only Stage1 changes Missing@10. Stage2 improves ranking without changing candidate availability.

### Conclusion 3 鈥?Context gain concentrates under ambiguity

On Ambiguous rows, Stage2 contributes:

$$
+2.136\text{ pp Macro Top1}
$$

which is substantially larger than its overall gain. This supports the intended role of context as a disambiguation mechanism.

### Conclusion 4 鈥?Context strongly repairs misleading personal priors, but arbitration remains unsolved

On Conflict rows:

$$
\text{Stage1}\rightarrow\text{Final}
=
+7.618\text{ pp Macro Top1}
$$

but Final still trails M1 and Generic at rank 1. The strong Final Top3/Top5 suggests the correct alternative is often present near the top, while the remaining failure is deciding when the dominant personal preference should be overridden.

### Conclusion 5 鈥?The combined thesis picture is structurally coherent

The evidence supports the staged interpretation:

$$
\boxed{\text{Recovery solves availability.}}
$$

$$
\boxed{\text{Recency-aware context improves ordering.}}
$$

$$
\boxed{\text{The remaining challenge is evidence arbitration.}}
$$

This strengthens the broader thesis argument that interpretable IME personalisation should separate candidate recovery, persistent user preference, lexical recency, semantic context, and final arbitration rather than collapse all evidence into one opaque score.

---

## 13. Reproducibility

Runner:

```text
experiments\context_comparison\run_full_transfer_initial_final_v1.py
```

Runner SHA256:

```text
f75d40f381e966f85cd4b20647ba7dc6a95df9116ad8657ca9a07505949a37b0
```

Output root:

```text
results\personalisation\context_comparison_followup_v1\full_transfer_initial_final_v1
```

Primary generated artifacts:

```text
result.json
run_setup.json
stage1_predictions.jsonl
final_predictions.jsonl
artifact_checksums.json
bge_context_cache.sqlite3
```

Known output SHA256 values:

```text
result.json              604a74d212ff16954b09f375a8db88f527cc07d12333fab0a7c18a7f712743a3
run_setup.json           28ee66721e4ffcdad82f141d763c537e3fdcc60ca7faa4c0e2e2ed82c27e69e1
stage1_predictions.jsonl eacc6c37c53e581bc667483eb6b29816cc81c3239aad9c16acf16788611ec53f
final_predictions.jsonl  fcc9b44c06fe0dc7bf629ad81d79476a4d27e52be934a37f4ac9c9d8d293973d
```

BGE cache:

```text
required unique contexts = 42,278
stored rows              = 42,278
```

Exact PowerShell reproduction command:

```powershell
Set-Location 'C:\Users\chiar\Desktop\LBH\thesis-context-compare'

& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  '.\experiments\context_comparison\run_full_transfer_initial_final_v1.py' `
  --fit '.\results\personalisation\context_comparison_v2\clean3_train_fit_v1.jsonl' `
  --val '.\results\personalisation\context_comparison_v2\clean3_train_val_v1.jsonl' `
  --generic '.\results\personalisation\context_comparison_v2\train_val_generic\predictions.jsonl' `
  --checkpoint 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat' `
  --bge-model 'C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf' `
  --standardized-stage1 '.\results\personalisation\context_comparison_v2\stage1\train_val.jsonl' `
  --output-root '.\results\personalisation\context_comparison_followup_v1\full_transfer_initial_final_v1' `
  --compatibility-device cpu `
  --cuda-path 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8' `
  --progress-every 500
```

The result must continue to record:

```text
hyperparameter_search = false
used_dev3000          = false
used_test             = false
```

Generated result trees, JSONL, SQLite caches, and logs remain **LOCAL-ONLY / DO NOT STAGE** unless a later repository policy explicitly changes that status.
