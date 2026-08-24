# 19. Initial-Pinyin Recovery + Context Reranking — Train-Val Reproducibility

**Date:** 2026-08-21  
**Scope:** Stage-1 recovery -> Stage-2 NGramRecency / BGERecency reranking -> post-hoc diagnosis  
**Status:** Train-Val development complete; Dev3000 and Test untouched  
**Companion result record:** `18_INITIAL_RECOVERY_CONTEXT_TRAINVAL_FINAL_CONCLUSIONS_2026-08-21.md`

---

## 1. Purpose

This document is the reproducibility record for the final Train-Val recovery + context-reranking activity. It records the exact scripts, inputs, commands, parameter grids, expected checkpoints, selected results, diagnostic commands, and safety invariants needed to reproduce the activity without reconstructing the procedure from terminal history.

This file covers five new runners:

```text
33_run_initial_recovery_ngram_context_fusion_v1.py
34_run_initial_recovery_bge_ngram_context_fusion_v2.py
35_run_initial_recovery_bge_ngram_context_fusion_v3.py
36_run_initial_recovery_context_diagnostics_v1.py
37_run_initial_recovery_context_topk_transitions_v1.py
```

The numerical interpretation and thesis-level conclusions are recorded in document 18; this document focuses on **how to reproduce and verify** those results.

---

## 2. Repository and environment

Worktree:

```text
C:\Users\chiar\Desktop\LBH\thesis-initial-research
```

Branch used during development:

```text
work/initial-personalisation
```

Python interpreter:

```text
C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe
```

Main result root:

```text
results\personalisation\initial_recovery_comparison_v1
```

BGE model path used by V2:

```text
C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf
```

All commands below assume execution from:

```text
C:\Users\chiar\Desktop\LBH\thesis-initial-research
```

---

## 3. Frozen protocol boundary

Protocol:

```text
Clean3 Train
  -> Train-Fit / Train-Val
  -> development / method selection
  -> diagnosis
  -> PRE-DEV FREEZE
  -> Dev3000
  -> final freeze
  -> Test
```

For every runner documented here:

```text
Dev3000 used = false
Test used = false
Gold used for candidate construction = false
Gold used for runtime scoring/features = false
Gold used for Train-Val selection/evaluation = true
```

Post-hoc diagnosis may use Gold only to classify evaluation subsets and rescue/harm transitions. Diagnosis does not alter scores, candidates, or selected lambdas.

Causal personal-history semantics remain:

```text
same author
-> strictly prior interactions
-> latest up-to-5000 RAW same-author interactions
-> exact current Initial-Pinyin filtering afterward
```

Therefore H5000 is applied **before** exact-Pinyin filtering. Earlier Train-Val rows may become legal history for later Train-Val rows. Current/future Gold is never available to the scorer.

---

## 4. Frozen input identities

| Artifact | Rows | SHA256 |
|---|---:|---|
| `initial_train_fit_v1.jsonl` | 144,526 | `162f5c98daa86cc69947571e6d8f20fc401f0a82cdd3fd6e517eb7be2addbdb4` |
| `initial_train_val_v1.jsonl` | 34,416 | `d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4` |
| `candidate_surface\train_val_candidate_surface.jsonl` | 34,416 | `205c0ba01cd0678d7a4341c503fa2e74cf126a70182ff582687025e4946764b2` |
| `train_val_generic\predictions.jsonl` | 34,416 | `bd0fb4dc304e0b266b90fae6fe3ac65424d2f52b23fedfa881212706ba2c2873` |
| `frequency_pv1\predictions.jsonl` | 34,416 | `7fd8aa158d8cd50bced36b55610f8d932bc65e3aae1dbbd5bd65907ff1707ea7` |

Frozen Personal-K5 surface statistics:

```text
eligible rows = 30,509
candidate pairs = 123,738
candidate-count distribution = {0:3907, 1:3194, 2:2787, 3:2635, 4:2400, 5:19493}
```

Evaluation populations:

```text
all Train-Val rows = 34,416
Generic Missing = 12,565
R = Gold not in Generic Top10 AND Gold in Personal K5
|R| = 4,910
K5 theoretical recoverability = 4,910 / 12,565 = 39.0768%
```

---

## 5. New runner identities

| Runner | SHA256 |
|---|---|
| `33_run_initial_recovery_ngram_context_fusion_v1.py` | `e6dcd1f68028ad5065064b6b714eaa88d92f74363a328570bfcc777b13271dc2` |
| `34_run_initial_recovery_bge_ngram_context_fusion_v2.py` | `b7d95374aa421cbc364699e44e0850ba2e72e50a2a5f816ad37f85b138d1435a` |
| `35_run_initial_recovery_bge_ngram_context_fusion_v3.py` | `2b29a86957b4f2adf17a13de37648766e1423d0ec99a57ea257c5aa155d89335` |
| `36_run_initial_recovery_context_diagnostics_v1.py` | `7c4a12a5f447405f024d8e8008253da23aab4775d2ae4500f5c44545583d3256` |
| `37_run_initial_recovery_context_topk_transitions_v1.py` | `3966111844719f29a07b580a10d18021b0cdf4a6846c71157de611e1a92eaef1` |

Companion final-conclusions record:

```text
18_INITIAL_RECOVERY_CONTEXT_TRAINVAL_FINAL_CONCLUSIONS_2026-08-21.md
SHA256 = 98a83076e9de8473e6ccfb99997d27431c3433c9a1d2ad6b3ede2b3336d5afdc
```

---

## 6. Frozen Stage-1 recovery bases

The Stage-2 experiments do not retune or redesign Stage-1 recovery. Three Stage-1 operating points are frozen as inputs.

### 6.1 K5+Entropy — coverage-first

Form:

```text
Personal K5
-> frozen Interpolated-NGram ordering/admission of all K5
-> exact PV1 frequency support
-> Generic-F boundary merge

personal score = boundary + 4*F_PV + .25*C_E
```

Stage-1 metrics:

| Macro | Micro | Top3 | Top5 | MRR | Missing |
|---:|---:|---:|---:|---:|---:|
| .403790 | .428405 | .602336 | .677069 | .533534 | .243288 |

Recovery on R=4,910:

```text
Rec1  = .212627  (1,044)
Rec3  = .612627  (3,008)
Rec5  = .803462  (3,945)
Rec10 = .987576  (4,849)
RecMRR= .455236
```

### 6.2 4P+4CS+2E — balanced

Form:

```text
personal score = boundary + 4*P_NG + 4*CS + 2*C_E
```

Stage-1 metrics:

| Macro | Micro | Top3 | Top5 | MRR | Missing |
|---:|---:|---:|---:|---:|---:|
| .404807 | .429364 | .614801 | .685815 | .537433 | .243172 |

Recovery:

```text
Rec1  = .258859  (1,271)
Rec3  = .549287  (2,697)
Rec5  = .716904  (3,520)
Rec10 = .948473  (4,657)
RecMRR= .454217
```

### 6.3 6P+2CS+.25E — front-rank

Form:

```text
personal score = boundary + 6*P_NG + 2*CS + .25*C_E
```

Stage-1 metrics:

| Macro | Micro | Top3 | Top5 | MRR | Missing |
|---:|---:|---:|---:|---:|---:|
| .400638 | .423989 | .615876 | .686047 | .534545 | .243869 |

Recovery:

```text
Rec1  = .276986  (1,360)
Rec3  = .573727  (2,817)
Rec5  = .710998  (3,491)
Rec10 = .928310  (4,558)
RecMRR= .467900
```

---

## 7. Stage-2 scorer definitions

Stage-2 always reranks a **fixed Stage-1 final candidate set**. It does not add or remove candidates.

Final score:

\[
S_{final}(c)=S_{REC}(c)+\lambda_NP_{NG-R}(c)+\lambda_BP_{BGE-R}(c)
\]

### 7.1 NGramRecency

Stage-2 NGramRecency is distinct from the Stage-1 Interpolated NGram scorer.

Stage-2 definition:

```text
HardBackoff
maxN = 2
tau_N = 2048
candidate-conditioned on current frozen final candidate set
```

For each candidate, use the largest suffix order `<=2` for which candidate-target history evidence exists. Apply recency weight:

\[
\exp(-age/2048)
\]

and normalize candidate support over the current candidate set.

If total candidate support is zero, use uniform support, leaving the ordering unchanged.

### 7.2 BGERecency

Current query representation:

```text
last 64 context characters
```

For each candidate:

```text
same-Pinyin causal candidate-conditioned history
-> cosine similarity to current context
-> choose historical Top-5 by cosine ONLY
-> clamp negative cosine to zero
-> apply exp(-age/2048) in aggregation
-> normalize over current candidate set
```

Conceptually:

\[
R_{BGE-R}(c)=\sum_{h\in Top5_{cos}(H_c)}\max(0,\cos(E(q),E(h)))e^{-age(h)/2048}
\]

Recency does not alter which Top-5 historical items are selected; it is applied after cosine-only retrieval.

---

## 8. V1 — NGramRecency-only fusion

### 8.1 Command

```powershell
Set-Location 'C:\Users\chiar\Desktop\LBH\thesis-initial-research'

$py   = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$root = '.\results\personalisation\initial_recovery_comparison_v1'

& $py -m experiments.initial_personalisation.run_initial_recovery_ngram_context_fusion_v1 `
    --fit "$root\initial_train_fit_v1.jsonl" `
    --val "$root\initial_train_val_v1.jsonl" `
    --candidate-surface "$root\candidate_surface\train_val_candidate_surface.jsonl" `
    --frequency-pv1-predictions "$root\frequency_pv1\predictions.jsonl" `
    --adaptive-scores "$root\candidate_scoring_adaptive_ngram_top10_v1\adaptive_ngram_top10_v1\scores.jsonl" `
    --output-root "$root\recovery_ngram_context_fusion_v1"
```

Default lambda_N grid:

```text
[0, .25, .5, 1, 2, 4, 6, 8, 12]
```

### 8.2 Required invariants

V1 must verify:

```text
candidate set unchanged under Stage-2 reranking
lambda_N=0 reproduces exact Stage-1 order
Missing@10 invariant
Rec@10 invariant on fixed R
Gold not used for scoring/features
Dev3000 used = false
Test used = false
```

### 8.3 Expected selected points

All three bases select:

```text
lambda_N = 6
```

Expected selected results:

| Base | Macro | Micro | Top3 | Top5 | MRR | Missing | Rec1 | Rec3 | Rec5 | Rec10 | RecMRR | Top1 net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K5+Entropy | .434673 | .457985 | .624390 | .686832 | .555949 | .243288 | .4128 | .7181 | .8558 | .9876 | .5960 | +1018 |
| 4P+4CS+2E | .432451 | .455718 | .630434 | .694183 | .556494 | .243172 | .4224 | .6635 | .7853 | .9485 | .5775 | +907 |
| 6P+2CS+.25E | .431802 | .454818 | .629184 | .693776 | .555338 | .243869 | .4346 | .6646 | .7743 | .9283 | .5807 | +1061 |

Expected output root:

```text
results\personalisation\initial_recovery_comparison_v1\recovery_ngram_context_fusion_v1
```

Expected files:

```text
stage1_frozen.jsonl
ngram_recency_support.jsonl
grid_results.csv
selected_metrics.csv
selected_predictions.jsonl
comparison.json
run_manifest.json
artifact_checksums.json
```

---

## 9. V2 — joint NGramRecency + BGERecency fusion

### 9.1 Command

```powershell
Set-Location 'C:\Users\chiar\Desktop\LBH\thesis-initial-research'

$py   = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$root = '.\results\personalisation\initial_recovery_comparison_v1'
$bge  = 'C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf'

& $py -m experiments.initial_personalisation.run_initial_recovery_bge_ngram_context_fusion_v2 `
    --fit "$root\initial_train_fit_v1.jsonl" `
    --val "$root\initial_train_val_v1.jsonl" `
    --base-ngram-root "$root\recovery_ngram_context_fusion_v1" `
    --bge-model "$bge" `
    --seed-bge-cache "$root\pv1_bge_ngram_context_reranking_v1\bge_history_embedding_cache.sqlite3" `
    --output-root "$root\recovery_bge_ngram_context_fusion_v2"
```

Default V2 grids:

```text
lambda_N = [0, .25, .5, 1, 2, 4, 6, 8, 12]
lambda_B = [0, .25, .5, 1, 2, 4, 6, 8]
```

Selection rule per recovery base:

```text
1. higher Macro-author Top1
2. higher MRR@10
3. smaller total context weight
4. smaller lambda_B
5. smaller lambda_N
```

### 9.2 Expected BGE cache behavior

Observed formal run:

```text
required unique historical contexts = 42,717
required contexts reused from seed  = 38,847
new historical embeddings           = 3,870
BGE embeddings are stored in a new versioned local cache
seed cache is read-only
```

Observed performance record:

```text
BGE online rows = 34,416
final online scoring rate ≈ 364.75 rows/s
mean embedding time ≈ 1.881 ms/context
mean online scoring time ≈ 2.443 ms/row
```

### 9.3 Required regression checks

V2 must exactly preserve the completed V1 inputs and support:

```text
V1 Stage-1 hashes match
V1 NGram support hashes match
V1 NGram-only selected points reproduce at lambda_B=0
candidate order alignment passes
pure-rerank Missing/Rec10 invariants pass
Dev3000/Test provenance remains false
```

### 9.4 Expected V2 selected results

| Base | lambda_N | lambda_B | Macro | Micro | Top3 | Top5 | MRR | Missing | Rec1 | Rec3 | Rec5 | Rec10 | RecMRR | Top1 net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K5+Entropy | 6 | 8 | .436767 | .459990 | .626453 | .688139 | .557836 | .243288 | .4430 | .7481 | .8778 | .9876 | .6218 | +1087 |
| 4P+4CS+2E | 4 | 6 | **.437058** | **.460571** | **.631392** | **.696478** | **.559755** | **.243172** | .4246 | .6969 | .8153 | .9485 | .5892 | +1074 |
| 6P+2CS+.25E | 4 | 6 | .436477 | .459786 | .630085 | .696013 | .558806 | .243869 | .4415 | .6961 | .8069 | .9283 | .5951 | **+1232** |

The V2 K5 point selected `lambda_B=8`, which was the upper boundary of the original BGE grid. This is why V3 is required before freeze.

Expected V2 output root:

```text
results\personalisation\initial_recovery_comparison_v1\recovery_bge_ngram_context_fusion_v2
```

Expected files:

```text
bge_recency_support.jsonl
bge_recency_scoring_summary.json
grid_results.csv
selected_metrics.csv
selected_predictions.jsonl
full_comparison.csv
comparison.json
run_manifest.json
artifact_checksums.json
```

---

## 10. V3 — expanded BGE boundary verification

V3 performs **arithmetic-only grid evaluation** from completed V1/V2 support. It must not recompute BGE embeddings.

### 10.1 Command

```powershell
Set-Location 'C:\Users\chiar\Desktop\LBH\thesis-initial-research'

$py   = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$root = '.\results\personalisation\initial_recovery_comparison_v1'

& $py -m experiments.initial_personalisation.run_initial_recovery_bge_ngram_context_fusion_v3 `
    --fit "$root\initial_train_fit_v1.jsonl" `
    --val "$root\initial_train_val_v1.jsonl" `
    --base-ngram-root "$root\recovery_ngram_context_fusion_v1" `
    --base-bge-root "$root\recovery_bge_ngram_context_fusion_v2" `
    --output-root "$root\recovery_bge_ngram_context_fusion_v3"
```

Default expanded grids:

```text
lambda_N = [0, .25, .5, 1, 2, 4, 6, 8, 12]
lambda_B = [0, .25, .5, 1, 2, 4, 6, 8, 12, 16]
```

Total configurations:

```text
3 recovery bases x 9 lambda_N x 10 lambda_B = 270
```

Observed runtime:

```text
193.1 s
BGE embeddings recomputed = false
```

### 10.2 Required V1 regression checkpoints

Expected terminal lines:

```text
PASS K5+Entropy       lambda_N=6  Macro=0.434673 MRR=0.555949
PASS 4P+4CS+2E        lambda_N=6  Macro=0.432451 MRR=0.556494
PASS 6P+2CS+.25E      lambda_N=6  Macro=0.431802 MRR=0.555338
```

### 10.3 Required V2 selected-point regression checkpoints

```text
PASS K5+Entropy       lambda_N=6 lambda_B=8  Macro=0.436767
PASS 4P+4CS+2E        lambda_N=4 lambda_B=6  Macro=0.437058
PASS 6P+2CS+.25E      lambda_N=4 lambda_B=6  Macro=0.436477
```

### 10.4 Expected V3 selection

V3 must leave all three selected points unchanged:

```text
K5+Entropy       (6,8) -> (6,8)  DeltaMacro=+0.000000  DeltaMRR=+0.000000
4P+4CS+2E        (4,6) -> (4,6)  DeltaMacro=+0.000000  DeltaMRR=+0.000000
6P+2CS+.25E      (4,6) -> (4,6)  DeltaMacro=+0.000000  DeltaMRR=+0.000000
```

Boundary verification:

```text
K5+Entropy       lambda_B=8  upper=16  hit_upper_boundary=false
4P+4CS+2E        lambda_B=6  upper=16  hit_upper_boundary=false
6P+2CS+.25E      lambda_B=6  upper=16  hit_upper_boundary=false
PASS: no selected lambda_B hits the expanded upper boundary.
```

This closes the V2 boundary concern. No further lambda-grid expansion is required for the frozen Train-Val development line.

Expected output root:

```text
results\personalisation\initial_recovery_comparison_v1\recovery_bge_ngram_context_fusion_v3
```

Expected files:

```text
grid_results.csv
selected_metrics.csv
selected_predictions.jsonl
full_comparison.csv
comparison.json
run_manifest.json
artifact_checksums.json
run_setup.json
```

---

## 11. Final frozen Train-Val operating points

After V3:

### Primary overall

```text
Recovery = 4P+4CS+2E
lambda_N = 4
lambda_B = 6

Macro = .437058
Micro = .460571
Top3 = .631392
Top5 = .696478
MRR = .559755
Missing = .243172
Rec1 = .4246
Rec3 = .6969
Rec5 = .8153
Rec10 = .9485
RecMRR = .5892
```

### Coverage-oriented

```text
Recovery = K5+Entropy
lambda_N = 6
lambda_B = 8

Macro = .436767
Micro = .459990
Top3 = .626453
Top5 = .688139
MRR = .557836
Missing = .243288
Rec1 = .4430
Rec3 = .7481
Rec5 = .8778
Rec10 = .9876
RecMRR = .6218
```

### Front-rank

```text
Recovery = 6P+2CS+.25E
lambda_N = 4
lambda_B = 6

Macro = .436477
Micro = .459786
Top3 = .630085
Top5 = .696013
MRR = .558806
Missing = .243869
Rec1 = .4415
Rec3 = .6961
Rec5 = .8069
Rec10 = .9283
RecMRR = .5951
```

---

## 12. Post-hoc diagnosis runner

The diagnosis runner is analysis-only. It does not retune lambdas and does not recompute context support.

### 12.1 Command

```powershell
Set-Location 'C:\Users\chiar\Desktop\LBH\thesis-initial-research'

$py   = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$root = '.\results\personalisation\initial_recovery_comparison_v1'

& $py -m experiments.initial_personalisation.run_initial_recovery_context_diagnostics_v1 `
    --val "$root\initial_train_val_v1.jsonl" `
    --base-ngram-root "$root\recovery_ngram_context_fusion_v1" `
    --base-v3-root "$root\recovery_bge_ngram_context_fusion_v3" `
    --output-root "$root\recovery_context_diagnostics_v1"
```

### 12.2 Expected terminal checkpoint

```text
=== INITIAL RECOVERY -> CONTEXT DIAGNOSTICS V1 COMPLETE ===
Rows: 34416
Generic Missing: 12565
Recoverable R: 4910
Frozen V3 lambdas:
  K5+Entropy       lambda_N=6 lambda_B=8
  4P+4CS+2E        lambda_N=4 lambda_B=6
  6P+2CS+.25E      lambda_N=4 lambda_B=6

Primary 4P+4CS+2E context increments:
  NGram: DeltaMacro=+0.027644 rescue=2375 harm=1468 net=+907
  BGE  : DeltaMacro=+0.004607 rescue=681 harm=514 net=+167

Dev3000 used: false
Test used: false
Diagnosis only: no tuning performed
```

Expected diagnosis files:

```text
diagnostic_summary.json
headline_comparison.csv
per_author.csv
subset_metrics.csv
top1_transitions.csv
rank_movement.csv
recovery_diagnostics.csv
context_increment.csv
base_disagreement.csv
margin_diagnostics.csv
error_examples.jsonl
diagnostic_report.md
run_manifest.json
artifact_checksums.json
```

### 12.3 Key diagnosis verification values

Primary Balanced overall increments:

```text
Recovery -> NG-R:
DeltaMacro = +.027644
rescue = 2,375
harm = 1,468
Top1 net = +907
rank improved = 4,042
rank worsened = 3,006
rank net = +1,036

NG-R -> Full:
DeltaMacro = +.004607
rescue = 681
harm = 514
Top1 net = +167
rank improved = 2,004
rank worsened = 1,579
rank net = +425
```

Balanced subset Top1 transitions:

```text
Generic-covered, Recovery -> NG: +104 net
Generic-covered, NG -> Full: +156 net
Recoverable R, Recovery -> NG: +803 net
Recoverable R, NG -> Full: +11 net
Conflict, Recovery -> NG: +409 net
Conflict, NG -> Full: -84 net
```

K5 BGE increment:

```text
Generic-covered DeltaMacro = -.004509, Top1 net = -79
Recoverable-R DeltaMacro = +.030332, Top1 net = +148
```

Balanced BGE increment:

```text
Generic-covered DeltaMacro = +.007178, Top1 net = +156
Recoverable-R DeltaMacro = +.002409, Top1 net = +11
```

Final Top1 candidate agreement across the three full-context recovery bases:

```text
all same Top1 = 31,793 / 34,416 = 92.3785%
any disagreement = 2,623 / 34,416 = 7.6215%
```

---

## 13. Top-K rescue/harm diagnosis

This second diagnostic runner adds explicit Top1/Top3/Top5 transition counts. It is also analysis-only.

### 13.1 Command

```powershell
Set-Location 'C:\Users\chiar\Desktop\LBH\thesis-initial-research'

$py   = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$root = '.\results\personalisation\initial_recovery_comparison_v1'

& $py -m experiments.initial_personalisation.run_initial_recovery_context_topk_transitions_v1 `
    --val "$root\initial_train_val_v1.jsonl" `
    --base-ngram-root "$root\recovery_ngram_context_fusion_v1" `
    --base-v3-root "$root\recovery_bge_ngram_context_fusion_v3" `
    --output-root "$root\recovery_context_topk_transitions_v1"
```

Definition:

```text
rescue@k = outside Top-k -> inside Top-k
harm@k   = inside Top-k -> outside Top-k
net@k    = rescue@k - harm@k
```

### 13.2 Expected Top3 transitions — overall

| Base | Transition | Rescue | Harm | Net |
|---|---|---:|---:|---:|
| K5+Entropy | Recovery->NG-R | 1,145 | 386 | +759 |
| K5+Entropy | NG-R->Full | 384 | 313 | +71 |
| K5+Entropy | Recovery->Full | 1,434 | 604 | +830 |
| 4P+4CS+2E | Recovery->NG-R | 989 | 451 | +538 |
| 4P+4CS+2E | NG-R->Full | 309 | 276 | +33 |
| 4P+4CS+2E | Recovery->Full | 1,198 | 627 | +571 |
| 6P+2CS+.25E | Recovery->NG-R | 906 | 448 | +458 |
| 6P+2CS+.25E | NG-R->Full | 310 | 279 | +31 |
| 6P+2CS+.25E | Recovery->Full | 1,104 | 615 | +489 |

### 13.3 Expected Top3 transitions — Generic-covered

| Base | Transition | Rescue | Harm | Net |
|---|---|---:|---:|---:|
| K5+Entropy | Recovery->NG-R | 567 | 326 | +241 |
| K5+Entropy | NG-R->Full | 183 | 259 | -76 |
| K5+Entropy | Recovery->Full | 689 | 524 | +165 |
| 4P+4CS+2E | Recovery->NG-R | 409 | 432 | -23 |
| 4P+4CS+2E | NG-R->Full | 119 | 250 | -131 |
| 4P+4CS+2E | Recovery->Full | 453 | 607 | -154 |
| 6P+2CS+.25E | Recovery->NG-R | 438 | 426 | +12 |
| 6P+2CS+.25E | NG-R->Full | 129 | 253 | -124 |
| 6P+2CS+.25E | Recovery->Full | 487 | 599 | -112 |

### 13.4 Expected Top3 transitions — recoverable R

| Base | Transition | Rescue | Harm | Net |
|---|---|---:|---:|---:|
| K5+Entropy | Recovery->NG-R | 578 | 60 | +518 |
| K5+Entropy | NG-R->Full | 201 | 54 | +147 |
| K5+Entropy | Recovery->Full | 745 | 80 | +665 |
| 4P+4CS+2E | Recovery->NG-R | 580 | 19 | +561 |
| 4P+4CS+2E | NG-R->Full | 190 | 26 | +164 |
| 4P+4CS+2E | Recovery->Full | 745 | 20 | **+725** |
| 6P+2CS+.25E | Recovery->NG-R | 468 | 22 | +446 |
| 6P+2CS+.25E | NG-R->Full | 181 | 26 | +155 |
| 6P+2CS+.25E | Recovery->Full | 617 | 16 | +601 |

Expected output files:

```text
topk_transitions.csv
topk_transitions.json
```

---

## 14. Per-author verification checkpoint

For the primary Balanced system:

| Author | Stage1 Top1 | +NG Top1 | Full Top1 | Full Top3 | Full Top5 | Full MRR | Full Missing |
|---|---:|---:|---:|---:|---:|---:|---:|
| Agent Phage | .470344 | .487374 | .493923 | .689542 | .764937 | .606609 | .171312 |
| Etinjat | .237235 | .271980 | .275218 | .388917 | .447198 | .347801 | .485056 |
| breaddddd | .506841 | .537999 | .542032 | .722183 | .780388 | .643439 | .167655 |

Context Top1 gain from Stage1 to Full:

```text
Agent Phage = +.023579
Etinjat     = +.037983
breaddddd   = +.035192
```

This verifies that Etinjat remains a much harder candidate-availability/ranking condition, but context gain is positive for all three authors.

---

## 15. Pure-reranking invariants

For each recovery base, Stage-2 does not change the candidate set. Therefore these values must remain exactly invariant between Stage1, NG-only, and full context:

### K5+Entropy

```text
Missing@10 = .243288
Rec10 = .987576
```

### 4P+4CS+2E

```text
Missing@10 = .243172
Rec10 = .948473
```

### 6P+2CS+.25E

```text
Missing@10 = .243869
Rec10 = .928310
```

A change in either value indicates that a supposed Stage-2 reranker has accidentally changed the candidate surface and the run should be treated as invalid.

---

## 16. Verification commands after reproduction

Check runner hashes from the project copy:

```powershell
Get-FileHash .\experiments\initial_personalisation\33_run_initial_recovery_ngram_context_fusion_v1.py -Algorithm SHA256
Get-FileHash .\experiments\initial_personalisation\34_run_initial_recovery_bge_ngram_context_fusion_v2.py -Algorithm SHA256
Get-FileHash .\experiments\initial_personalisation\35_run_initial_recovery_bge_ngram_context_fusion_v3.py -Algorithm SHA256
Get-FileHash .\experiments\initial_personalisation\36_run_initial_recovery_context_diagnostics_v1.py -Algorithm SHA256
Get-FileHash .\experiments\initial_personalisation\37_run_initial_recovery_context_topk_transitions_v1.py -Algorithm SHA256
```

Inspect final V3 selection directly:

```powershell
$root = '.\results\personalisation\initial_recovery_comparison_v1'
Import-Csv "$root\recovery_bge_ngram_context_fusion_v3\selected_metrics.csv" | Format-Table -AutoSize
```

Inspect primary context increments:

```powershell
Import-Csv "$root\recovery_context_diagnostics_v1\context_increment.csv" |
    Where-Object { $_.base -eq '4P+4CS+2E' -and $_.subset -eq 'overall' } |
    Format-Table -AutoSize
```

Inspect Top3 rescue/harm:

```powershell
Import-Csv "$root\recovery_context_topk_transitions_v1\topk_transitions.csv" |
    Where-Object { $_.k -eq '3' } |
    Format-Table -AutoSize
```

Every formal output directory contains `artifact_checksums.json`; use that file as the authoritative checksum inventory for the generated artifacts of that run.

---

## 17. Failure / retry policy

Do not silently overwrite historical outputs after a non-trivial failed run.

If a completed output root already exists and a changed runner or changed input must be tested, use a new versioned root, for example:

```text
recovery_bge_ngram_context_fusion_v3b
recovery_context_diagnostics_v1b
```

If V3 reports an exact-regression mismatch, inspect provenance and hashes first. Do not loosen numerical tolerances simply to make the check pass.

If a selected lambda again lands on the maximum of a newly declared grid, treat it as an unresolved boundary condition rather than an interior optimum.

---

## 18. Reproducibility closure

The completed formal chain is:

```text
Frozen Stage1 recovery bases
  -> V1 NGramRecency-only reranking
  -> V2 NGramRecency + BGERecency joint reranking
  -> V3 expanded lambda_B boundary verification
  -> frozen post-hoc diagnosis
  -> frozen Top-K rescue/harm diagnosis
```

Final verification state:

```text
V1 NGram regression checks: PASS
V2 full-context selected-point regressions: PASS
V3 expanded-boundary check: PASS
selected V2 and V3 operating points: identical
diagnosis is non-tuning: true
BGE embeddings recomputed in V3: false
Dev3000 used: false
Test used: false
```

The Train-Val recovery + context development line is therefore reproducibly closed and ready for PRE-DEV FREEZE.
