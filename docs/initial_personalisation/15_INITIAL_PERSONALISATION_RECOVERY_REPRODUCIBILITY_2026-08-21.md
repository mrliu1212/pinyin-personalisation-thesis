# Initial-Pinyin Recovery / Context–Preference–Confidence Reproducibility

**Date:** 2026-08-21
**Scope:** post-candidate-scoring recovery, NGramSelector, NGram-CS, concentration, and the final two-anchor Context–Preference–Confidence experiment.
**Status:** Train-Val development only; Dev3000 and Test remain untouched.

---

## 1. Repository and frozen protocol

Worktree:

```text
C:\Users\chiar\Desktop\LBH\thesis-initial-research
```

Branch:

```text
work/initial-personalisation
```

Python:

```text
C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe
```

Main result root:

```text
results\personalisation\initial_recovery_comparison_v1
```

Protocol:

```text
Clean3 Train
  -> Train-Fit / Train-Val
  -> development / method selection
  -> PRE-DEV FREEZE
  -> Dev3000
  -> final freeze
  -> Test
```

Current assertions:

```text
Gold used for candidate construction = false
Gold used for runtime feature construction = false
Gold used for online scoring = false
Gold used for Train-Val evaluation / hyperparameter selection = true
Dev3000 used = false
Test used = false
K10 used in the current main recovery line = false
```

---

## 2. Frozen inputs

| Artifact | Rows | SHA256 |
|---|---:|---|
| `initial_train_fit_v1.jsonl` | 144,526 | `162f5c98daa86cc69947571e6d8f20fc401f0a82cdd3fd6e517eb7be2addbdb4` |
| `initial_train_val_v1.jsonl` | 34,416 | `d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4` |
| `candidate_surface\train_val_candidate_surface.jsonl` | 34,416 | `205c0ba01cd0678d7a4341c503fa2e74cf126a70182ff582687025e4946764b2` |
| Generic predictions | 34,416 | `bd0fb4dc304e0b266b90fae6fe3ac65424d2f52b23fedfa881212706ba2c2873` |
| `frequency_pv1\predictions.jsonl` | 34,416 | `7fd8aa158d8cd50bced36b55610f8d932bc65e3aae1dbbd5bd65907ff1707ea7` |

Frozen Personal K5 statistics:

```text
eligible rows = 30,509
candidate pairs = 123,738
candidate-count distribution = {0:3907, 1:3194, 2:2787, 3:2635, 4:2400, 5:19493}
```

Causal history semantics MUST remain:

1. same author only;
2. strictly prior rows only;
3. latest H5000 selected before exact-Pinyin filtering;
4. Train-Fit plus causally earlier Train-Val may be history;
5. current/future target never visible.

---

## 3. Evaluation populations

### End-to-end

```text
n = 34,416 Train-Val rows
primary selection metric = Macro-author Top1
```

### Candidate-only scoring

```text
Gold in Personal K5 AND K >= 2
n = 4,471
```

### Recovery subset

```text
R = {Gold not in Generic Top10 AND Gold in Personal K5}
|R| = 4,910
Generic Missing = 12,565
K5 theoretical recoverability = 4,910 / 12,565 = 39.08%
```

Do not compare candidate-only percentages directly with full end-to-end percentages.

---

## 4. Frozen Interpolated NGram used downstream

The downstream lexical scorer is:

```text
Personal K5
Interpolated NGramRecency
maxN = 2
kappa = 1
tau = 2048
```

Candidate-only result:

```text
Macro Top1 = 0.590783
Micro Top1 = 0.594051
Top3       = 0.898009
MRR        = 0.750760
online latency ~0.084 ms/query
```

Hard NGram has a marginally higher Top1 but slightly lower Top3/MRR. Interpolated is retained because its evidence-dependent interpolation gives smoother backoff under sparse personal history.

---

## 5. NGramSelector reproducibility checkpoint

Architecture:

```text
Interpolated NGram -> decides candidate identity/order (WHO)
PV1 historical frequency support -> decides injection strength (HOW STRONGLY)
```

Canonical K1/K3/K5 output root:

```text
results\personalisation\initial_recovery_comparison_v1\pv1_ngram_selector_k135_v1
```

Key results:

| K | Macro Top1 | Micro Top1 | Top3 | Top5 | MRR | Missing | Rec@3 | Rec@10 | PV1 net |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | .403678 | .428318 | .608409 | .674744 | .530086 | .278272 | .5393 | .6303 | +54 |
| **3** | **.403964** | **.428522** | .604544 | **.681079** | **.534120** | .247007 | **.6051** | .9051 | **+61** |
| 5 | .403772 | .428376 | .603266 | .677708 | .533908 | **.243172** | .6026 | **.9857** | +56 |

Interpretation:

```text
K1 = conservative
K3 = balanced selective recovery
K5 = high coverage
```

`NGramSelector@K3` remains a main comparison model.

---

## 6. NGram-CS interpolation checkpoints

General form:

\[
D_\alpha=(1-\alpha)P_{NG}+\alpha CS
\]

\[
Score=B+\lambda_D D_\alpha
\]

Important exact equivalents at \(\lambda_D=8\):

```text
alpha=.25 -> B + 6 P_NG + 2 CS
alpha=.50 -> B + 4 P_NG + 4 CS
```

Key results:

| Model | Macro | Micro | Top3 | Top5 | MRR | Missing | Rec@3 | Rec@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6P+2CS | .400212 | .423553 | .615731 | .685669 | .534314 | .244392 | .5692 | .9230 |
| 4P+4CS | .402628 | .427272 | .612593 | .683461 | .535609 | .245613 | .5037 | .9106 |

---

## 7. Frequency-only K5 + concentration checkpoint

Form:

\[
Score=B+4F_{PV}(c)+\lambda C(q)
\]

Joint/fixed tuning conclusion:

```text
best F-only remains gamma_F = 4
Margin selects lambda = 0
Dual selects lambda = 0
Entropy selects lambda = .25
```

Entropy result:

```text
Macro = .403790
Micro = .428405
Top3 = .602336
Top5 = .677069
MRR = .533534
Missing = .243288
Rec@3 = .6126
Rec@10 = .9876
```

This increases recovery but does not meaningfully improve overall ranking relative to K5.

---

## 8. Context–Preference–Confidence two-anchor experiment

Canonical runner:

```text
experiments\initial_personalisation\run_initial_ngram_cs_entropy_two_anchors_v1.py
```

Runner SHA256:

```text
e5460a81435a76375619bdac809d464f567a9cdc8ae4f9c37115388d3b25d9cc
```

Canonical output root:

```text
results\personalisation\initial_recovery_comparison_v1\ngram_cs_entropy_two_anchors_v1
```

The runner reuses the already-computed frozen Interpolated NGram cache. It does not rerun Q8/BGE/PinyinGPT inference.

Two frozen anchors:

\[
\text{Top3Anchor}=B+6P_{NG}+2CS+\lambda_EC_E
\]

\[
\text{BalancedAnchor}=B+4P_{NG}+4CS+\lambda_EC_E
\]

Grid:

```text
lambda_E = {0, .25, .5, 1, 2, 4}
```

Entropy concentration:

\[
C_E=1-H_{norm}
\]

where the entropy distribution uses the full legal visible same-Pinyin target distribution, not only Personal K5.

---

## 9. Exact execution command

```powershell
$py = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$root = '.\results\personalisation\initial_recovery_comparison_v1'

& $py .\experiments\initial_personalisation\run_initial_ngram_cs_entropy_two_anchors_v1.py `
    --fit "$root\initial_train_fit_v1.jsonl" `
    --val "$root\initial_train_val_v1.jsonl" `
    --candidate-surface "$root\candidate_surface\train_val_candidate_surface.jsonl" `
    --frequency-pv1-predictions "$root\frequency_pv1\predictions.jsonl" `
    --adaptive-dir "$root\candidate_scoring_adaptive_ngram_top10_v1\adaptive_ngram_top10_v1" `
    --output-root "$root\ngram_cs_entropy_two_anchors_v1" `
    --entropy-lambdas "0,.25,.5,1,2,4" `
    --progress-every 1000
```

---

## 10. Required control reproduction

At \(\lambda_E=0\), the new runner MUST exactly reproduce the earlier NGram-CS anchors.

### Top3Anchor control

```text
Macro   = .400212
Top3    = .615731
Top5    = .685669
MRR     = .534314
Missing = .244392
```

### BalancedAnchor control

```text
Macro   = .402628
Top3    = .612593
Top5    = .683461
MRR     = .535609
Missing = .245613
```

Observed run:

```text
Top3Anchor lambda_E=0 PASSED
BalancedAnchor lambda_E=0 PASSED
```

A future rerun should be rejected if either control fails.

---

## 11. Full two-anchor result grid

### Top3Anchor: B + 6P + 2CS + lambda_E E

| lambda_E | Macro | Micro | Top3 | Top5 | MRR | Missing | Rec@3 | Rec@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .400212 | .423553 | .615731 | .685669 | .534314 | .244392 | .5692 | .9230 |
| .25 | .400638 | .423989 | **.615876** | .686047 | .534545 | .243869 | .5737 | .9283 |
| .5 | **.400701** | .424047 | .615789 | .686338 | **.534577** | .243608 | .5790 | .9328 |
| 1 | .399749 | .422972 | .615760 | **.686715** | .534055 | .243026 | .5876 | .9409 |
| 2 | .391700 | .414400 | .614627 | .686425 | .529343 | **.242620** | .6049 | .9540 |
| 4 | .371792 | .392201 | .610762 | .682996 | .516460 | .243201 | **.6407** | .9692 |

### BalancedAnchor: B + 4P + 4CS + lambda_E E

| lambda_E | Macro | Micro | Top3 | Top5 | MRR | Missing | Rec@3 | Rec@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .402628 | .427272 | .612593 | .683461 | .535609 | .245613 | .5037 | .9106 |
| .25 | .402742 | .427388 | .613029 | .683752 | .535797 | .245060 | .5094 | .9165 |
| .5 | .402756 | .427359 | .613610 | .684420 | .535995 | .244305 | .5163 | .9236 |
| 1 | .403584 | .428231 | .614278 | .684798 | .536724 | .243666 | .5285 | .9328 |
| **2** | **.404807** | **.429364** | **.614801** | **.685815** | **.537433** | **.243172** | .5493 | .9485 |
| 4 | .401933 | .425848 | .612506 | .684304 | .534646 | .243520 | **.5910** | **.9652** |

---

## 12. Current operating points to freeze for later holdout comparison

### Overall balanced development point

\[
\boxed{B+4P_{NG}+4CS+2C_E}
\]

```text
Macro Top1 = .404807
Micro Top1 = .429364
Top3       = .614801
Top5       = .685815
MRR        = .537433
Missing    = .243172
Rec@3      = .5493
Rec@10     = .9485
PV1 net    = +90
```

### Top3-oriented point

\[
\boxed{B+6P_{NG}+2CS+.25C_E}
\]

```text
Macro Top1 = .400638
Top3       = .615876
Top5       = .686047
MRR        = .534545
Missing    = .243869
Rec@3      = .5737
Rec@10     = .9283
```

### Top5-oriented point

\[
\boxed{B+6P_{NG}+2CS+1C_E}
\]

```text
Top5 = .686715
```

### Coverage-oriented aggressive diagnostic

\[
\boxed{B+6P_{NG}+2CS+2C_E}
\]

```text
Missing = .242620
Rec@3   = .6049
Rec@10  = .9540
Macro   = .391700
```

### Maximum Rec@3 diagnostic

\[
\boxed{B+6P_{NG}+2CS+4C_E}
\]

```text
Rec@3 = .6407
Rec@10 = .9692
Macro = .371792
MRR = .516460
```

This is intentionally retained as evidence that maximum recovery is not maximum end-to-end quality.

---

## 13. Durable outputs produced by the two-anchor runner

```text
ngram_cs_entropy_two_anchors_v1\
  features.jsonl
  predictions.jsonl
  grid_results.csv
  feature_summary.json
  comparison.json
  artifact_checksums.json
```

`artifact_checksums.json` is the canonical local source for exact output-file hashes.

Because the local Windows result directory is not available inside this conversation runtime, output hashes are deliberately **not invented here**. Freeze them from the local run with:

```powershell
$dir = '.\results\personalisation\initial_recovery_comparison_v1\ngram_cs_entropy_two_anchors_v1'
Get-FileHash "$dir\features.jsonl" -Algorithm SHA256
Get-FileHash "$dir\predictions.jsonl" -Algorithm SHA256
Get-FileHash "$dir\grid_results.csv" -Algorithm SHA256
Get-FileHash "$dir\feature_summary.json" -Algorithm SHA256
Get-FileHash "$dir\comparison.json" -Algorithm SHA256
Get-FileHash "$dir\artifact_checksums.json" -Algorithm SHA256
```

---

## 14. Reproducibility interpretation boundary

These are Train-Val development results.

Safe wording:

```text
current exploratory development result
current development-best operating point
```

Do not claim:

```text
confirmed generalization
final Test superiority
statistical significance
```

until untouched holdout evaluation and paired significance testing are completed.
