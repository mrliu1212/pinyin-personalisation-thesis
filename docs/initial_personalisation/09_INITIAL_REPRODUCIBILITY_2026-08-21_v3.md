# Initial+Short Standardized Candidate-Coverage Reproducibility

Date: 2026-08-21

This note records how to reproduce the standardized Initial+Short candidate-coverage and Full-vs-Initial paired findings without using Dev3000 or Test.

## 1. Repository and environment

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

Frozen PinyinGPT local checkpoint:

```text
C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat
```

Frozen Generic identity:

```text
aihijo/transformers4ime-pinyingpt-concat@76dd20dc92d8236a350fb732e99dde6fa15e2263
beam_size = 16
top_k = 10
production-compatible n_positions = 1024
```

Generic must use current context + current Pinyin only. It must not read personal history.

## 2. Standardized Full source split

Reference worktree:

```text
C:\Users\chiar\Desktop\LBH\thesis-context-compare
```

Result root:

```text
C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_v2
```

Frozen standardized split:

```text
Train-Fit rows = 144,526
Train-Val rows = 34,416
```

Expected SHA256:

```text
Train-Fit = 547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6
Train-Val = d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220
```

Files:

```text
...\context_comparison_v2\clean3_train_fit_v1.jsonl
...\context_comparison_v2\clean3_train_val_v1.jsonl
```

## 3. A0 — deterministic Full -> Initial transformation and rolling H5000 history

Script:

```text
experiments\initial_personalisation\prepare_initial_standardized.py
```

Run from the Initial worktree root:

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  '.\experiments\initial_personalisation\prepare_initial_standardized.py'
```

The Initial transform is deterministic:

```text
Full segmented Pinyin syllable -> first letter of each syllable
```

Example:

```text
shi yong -> s y
```

History semantics are mandatory:

```text
same author
-> strictly prior interactions
-> latest up-to-5000 RAW interactions
-> Initial-Pinyin exact match afterward
```

Do not take unlimited same-Initial history and cap afterward.

Expected output root:

```text
results\personalisation\initial_recovery_comparison_v1
```

Expected files include:

```text
initial_train_fit_v1.jsonl
initial_train_val_v1.jsonl
initial_transform_audit.json
history_semantics_audit.json
manifest.json
```

Expected Train-Val audit:

```text
rows = 34,416
raw_history_available = 34,416
initial_history_available = 32,752
initial_history_available_rate = 0.9516503951650395
ambiguous = 30,527
ambiguous_rate = 0.8870002324500232
conflict = 15,353
conflict_rate = 0.44610065086006506
frequency_winner_tied = 3,874
frequency_winner_tied_rate = 0.11256392375639238
```

The A0 manifest must report:

```text
dev3000_used = false
test_used = false
generic_inference_performed = false
```

## 4. A1 — standardized Initial Generic Top10

Script:

```text
experiments\initial_personalisation\run_initial_train_val_generic.py
```

Run:

```powershell
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$ckpt  = 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat'

& $python -m experiments.initial_personalisation.run_initial_train_val_generic `
  --input '.\results\personalisation\initial_recovery_comparison_v1\initial_train_val_v1.jsonl' `
  --checkpoint $ckpt `
  --output-root '.\results\personalisation\initial_recovery_comparison_v1\train_val_generic' `
  --device cuda
```

Expected final file:

```text
results\personalisation\initial_recovery_comparison_v1\train_val_generic\predictions.jsonl
```

Expected prediction SHA256:

```text
bd0fb4dc304e0b266b90fae6fe3ac65424d2f52b23fedfa881212706ba2c2873
```

Expected runtime summary fields:

```text
status = complete
rows = 34,416
beam_size = 16
top_k = 10
device = cuda
dev3000_used = false
test_used = false
```

Observed run:

```text
run_seconds = 1175.826129800058
mean_inference_ms_per_row = 29.689704216040887
median_inference_ms_per_row = 32.10877499077469
```

## 5. A2 — Initial candidate recoverability

Script:

```text
experiments\initial_personalisation\evaluate_initial_recoverability.py
```

The evaluator must import:

```python
from src.personalisation.pilot_a import HistoryIndex
from src.personalisation.context_memory import PredictionQuery
```

Run:

```powershell
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$ckpt  = 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat'

& $python -m experiments.initial_personalisation.evaluate_initial_recoverability `
  --initial-train-fit '.\results\personalisation\initial_recovery_comparison_v1\initial_train_fit_v1.jsonl' `
  --initial-train-val '.\results\personalisation\initial_recovery_comparison_v1\initial_train_val_v1.jsonl' `
  --generic-predictions '.\results\personalisation\initial_recovery_comparison_v1\train_val_generic\predictions.jsonl' `
  --checkpoint $ckpt `
  --output-root '.\results\personalisation\initial_recovery_comparison_v1\recoverability' `
  --device cpu
```

Expected headline results:

```text
Rows = 34,416
Generic Top1 = 0.33057298930729895
Generic Missing@10 = 0.36509181775918176
Raw recoverable | missing = 0.4771985674492638
Compatible recoverable | missing = 0.470274572224433
Compatible recoverable@1 | missing = 0.2110624751293275
Compatible recoverable@3 | missing = 0.33656983684838837
Compatible recoverable@5 | missing = 0.3907680063668922
```

Expected summary:

```text
results\personalisation\initial_recovery_comparison_v1\recoverability\recoverability_summary.json
```

Interpretation boundary: recoverability and oracle candidate coverage are not actual reranking accuracy.

## 6. Same-anchor standardized Full vs Initial candidate-coverage audit

Script:

```text
experiments\initial_personalisation\evaluate_full_vs_initial_candidate_coverage.py
```

This audit must compare the exact same 34,416 standardized Train-Val anchors and verify author/work/chronology/context/Gold identity, with only Full Pinyin -> deterministic Initial changed.

Run:

```powershell
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$ckpt  = 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat'
$cmp   = 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_v2'

& $python -m experiments.initial_personalisation.evaluate_full_vs_initial_candidate_coverage `
  --full-train-fit "$cmp\clean3_train_fit_v1.jsonl" `
  --full-train-val "$cmp\clean3_train_val_v1.jsonl" `
  --full-generic-predictions "$cmp\train_val_generic\predictions.jsonl" `
  --initial-train-val '.\results\personalisation\initial_recovery_comparison_v1\initial_train_val_v1.jsonl' `
  --initial-recoverability-rows '.\results\personalisation\initial_recovery_comparison_v1\recoverability\recoverability_rows.jsonl' `
  --checkpoint $ckpt `
  --output-root '.\results\personalisation\initial_recovery_comparison_v1\full_vs_initial_candidate_coverage' `
  --device cpu
```

Expected headline results:

```text
Rows = 34,416
Full Generic Top1 = 0.736024
Initial Generic Top1 = 0.330573
Full Missing@10 = 0.069212
Initial Missing@10 = 0.365092
Missing delta (Initial-Full) = 0.295880
Missing ratio (Initial/Full) = 5.275
Full-covered -> Initial-missing = 10,223 (29.704%)
Initial-loss recoverable from compatible personal history = 0.5188300890149663
Dev3000 used = false
Test used = false
```

Useful derived decomposition:

```text
Observed Top1 gap = 73.6024% - 33.0573% = 40.5451 percentage points
Candidate-coverage deterioration = 36.5092% - 6.9212% = 29.5880 percentage points
Additional within-Top10 ranking difficulty = 40.5451 - 29.5880 = 10.9571 percentage points
Coverage share of observed Top1 gap = 29.5880 / 40.5451 ~= 72.98%
```

Use careful wording: the 72.98% value is an error-accounting decomposition, not a causal attribution.

Expected output files:

```text
full_vs_initial_candidate_coverage\full_recoverability_rows.jsonl
full_vs_initial_candidate_coverage\full_vs_initial_paired_rows.jsonl
full_vs_initial_candidate_coverage\full_vs_initial_summary.json
full_vs_initial_candidate_coverage\full_vs_initial_headline.csv
```

## 7. Reproduction checks before citing results

Before using these numbers in the thesis, verify:

```text
1. git branch is work/initial-personalisation
2. input Full Train-Fit/Train-Val SHA256 match the frozen values above
3. A0 manifest reports dev3000_used=false and test_used=false
4. A1 prediction SHA256 matches bd0fb4dc...
5. A1 runtime_summary reports 34,416 rows, beam=16, top_k=10
6. A2 completes on exactly 34,416 rows
7. Full-vs-Initial paired audit completes on exactly 34,416 rows
8. paired audit reports dev3000_used=false and test_used=false
9. no result is regenerated after changing history semantics, Pinyin transform, checkpoint, beam, Top-K, or context semantics
```

## 8. Scientific boundary

These are standardized Train-Val research/development findings. They are not sealed Dev3000 confirmation results and they are not Test results.

The current finding is:

> On the same 34,416 standardized queries, converting Full Pinyin to Initial input increases Generic Missing@10 from 6.92% to 36.51%. Of the 10,223 queries whose Gold is covered under Full but lost under Initial, 51.88% remain recoverable from backend-compatible causal personal history.

Do not use Dev3000 or Test to change the current method based on these findings.

## 9. B1 — Frozen unified Initial personal candidate surface

Purpose: construct one prediction-visible, backend-compatible personal candidate list that must be reused by PV1 and EM1. The surface is built on standardized Initial Train-Val only; Dev3000 and Test remain closed.

History semantics:

```text
same author
-> strictly prior interactions
-> latest up-to-5000 RAW interactions
-> exact Initial-segment match
-> frequency lexicon
-> remove candidates already in Generic Top10
-> frozen PinyinGPT backend compatibility filter
-> frequency-descending personal list with lexical tie-break
```

Run:

```powershell
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$ckpt = 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat'
$root = '.\results\personalisation\initial_recovery_comparison_v1'

& $python -m experiments.initial_personalisation.build_initial_candidate_surface `
  --initial-train-fit "$root\initial_train_fit_v1.jsonl" `
  --initial-train-val "$root\initial_train_val_v1.jsonl" `
  --generic-predictions "$root\train_val_generic\predictions.jsonl" `
  --a2-recoverability-rows "$root\recoverability\recoverability_rows.jsonl" `
  --checkpoint $ckpt `
  --output-root "$root\candidate_surface" `
  --device cpu
```

Frozen B1 result:

```text
rows = 34,416
Generic Missing@10 = 0.36509181775918176
Compatible recoverable | missing = 0.470274572224433
Recovered@1 | missing = 0.2110624751293275
Recovered@3 | missing = 0.33656983684838837
Recovered@5 | missing = 0.3907680063668922
A2 cross-check = PASSED 34,416 rows
candidate surface SHA256 = 205c0ba01cd0678d7a4341c503fa2e74cf126a70182ff582687025e4946764b2
Dev3000 used = false
Test used = false
```

Frozen output files:

```text
results/personalisation/initial_recovery_comparison_v1/candidate_surface/train_val_candidate_surface.jsonl
results/personalisation/initial_recovery_comparison_v1/candidate_surface/candidate_surface_summary.json
results/personalisation/initial_recovery_comparison_v1/candidate_surface/candidate_surface_manifest.json
```

Before PV1 or EM1, verify the candidate-surface SHA256 exactly matches the value above. PV1 and EM1 must not independently rebuild, add, delete, or reorder personal candidates outside the frozen K-prefix rule.

## 10. B3 — Q8 vs BGE64 candidate scoring

Candidate-scoring work keeps the frozen Personal K5 surface unchanged and evaluates within-personal ranking on Train-Val only.

Canonical runner:

```text
experiments\initial_personalisation\run_initial_candidate_scoring_q8_bge64_v1.py
```

Canonical fixed-master SHA256:

```text
cee95ee85fd69f2de7deab07cb50ab8f56bb50fb9dbc29315261ca89c578b494
```

Canonical output:

```text
results\personalisation\initial_recovery_comparison_v1\candidate_scoring_q8_bge64_v1\candidate_scoring_comparison.json
```

Headline `Gold-in-personal K>=2` result (n=4,471):

```text
F              MacroTop1=0.491082 MicroTop1=0.494968 Top3=0.847685 MRR=0.683822
Q8             MacroTop1=0.636531 MicroTop1=0.642362 Top3=0.912995 MRR=0.783568
BGE64          MacroTop1=0.536167 MicroTop1=0.538582 Top3=0.885484 MRR=0.718840
Q8+F@0.75      MacroTop1=0.669164 MicroTop1=0.675688 Top3=0.925744 MRR=0.804015
BGE64+F@0      MacroTop1=0.536167 MicroTop1=0.538582 Top3=0.885484 MRR=0.718840
```

Latency:

```text
Q8 mean = 32.453 ms; p95 = 54.540 ms
BGE64 online mean = 2.136 ms; p95 = 3.052 ms
```

Protocol checks:

```text
Gold used for scoring/candidate selection = false
Gold used for Train-Val evaluation / alpha selection = true
Dev3000 used = false
Test used = false
```

## 11. B4 — NGram and NGramRecency candidate scoring

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

Original grids:

```text
N = {1,2,3,4,6,8}
tau = {32,128,512,2048}
```

Selected results on `Gold in Personal K5, K>=2`:

```text
NGram@2:
  MacroTop1 = 0.5601050146271284
  MicroTop1 = 0.5631849698054127
  Top3 = 0.876090360098412
  MRR = 0.7283307239245508

NGramRecency@2,tau=2048:
  MacroTop1 = 0.5915582674700545
  MicroTop1 = 0.594945202415567
  Top3 = 0.8959964213822411
  MRR = 0.7505628867516588
```

Selected-method online latency from that runner:

```text
NGram@2 mean = 0.0834867 ms
NGramRecency@2,tau=2048 mean = 0.0843688 ms
```

## 12. B5 — Adaptive NGram K5/K10

Runner:

```text
experiments\initial_personalisation\run_initial_candidate_scoring_adaptive_ngram_top10_v1.py
```

SHA256:

```text
261020f767e0501840df87bcdd15863652c7c3db660937ed8450287c3592ba7f
```

Run:

```powershell
Set-Location 'C:\Users\chiar\Desktop\LBH\thesis-initial-research'

$py = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$ckpt = 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat'
$root = '.\results\personalisation\initial_recovery_comparison_v1'
$out = "$root\candidate_scoring_adaptive_ngram_top10_v1"

& $py -m experiments.initial_personalisation.run_initial_candidate_scoring_adaptive_ngram_top10_v1 `
    --phase all `
    --fit "$root\initial_train_fit_v1.jsonl" `
    --val "$root\initial_train_val_v1.jsonl" `
    --candidate-surface "$root\candidate_surface\train_val_candidate_surface.jsonl" `
    --generic-predictions "$root\train_val_generic\predictions.jsonl" `
    --pinyin-checkpoint $ckpt `
    --output-root $out `
    --candidate-ks 5 10 `
    --progress-every 500
```

Grid:

```text
candidate K = {5,10}
maxN = {2,4,8}
tau = {512,2048,8192}
beta = {0.25,0.5,1}
kappa = {1,4,16,64}
```

Required surface audit:

```text
frozen_k5_exact_prefix_matches = 34416
frozen_k5_exact_prefix_match_rate = 1.0
K10 surface SHA256 = 46072df2a24e892de9906efb349fc5d1bc980a00758ac36f422c437083608ddd
candidate_selection_used_gold = false
Dev3000 used = false
Test used = false
```

Availability:

```text
Gold in K5 = 4910
Gold in K10 = 5537
incremental = +627
Generic Missing = 12565
recoverable K5 = 0.3907680063668922
recoverable K10 = 0.4406685236768802
```

Selected K5:

```text
HardBackoff maxN=2 tau=2048:
  MacroTop1=0.5915582674700545
  MicroTop1=0.594945202415567
  Top3=0.8959964213822411
  Top5=1.0
  MRR10=0.7505628867516588

SoftSuffix maxN=4 beta=1 tau=2048:
  MacroTop1=0.5740437623816178
  MicroTop1=0.5779467680608364
  Top3=0.8948781033325878
  Top5=1.0
  MRR10=0.7416573473495862

Interpolated maxN=2 kappa=1 tau=2048:
  MacroTop1=0.5907826703699341
  MicroTop1=0.5940505479758443
  Top3=0.898009393871617
  Top5=1.0
  MRR10=0.7507604562737643
```

Selected K10 (n=5,098):

```text
HardBackoff:
  MacroTop1=0.5273164887946317
  MicroTop1=0.5296194586112201
  Top3=0.8054138877991369
  Top5=0.9058454295802275
  MRR10=0.6865399285122705

SoftSuffix:
  MacroTop1=0.5128002758438617
  MicroTop1=0.515300117693213
  Top3=0.8083562181247548
  Top5=0.9089839152608866
  MRR10=0.679711776171172

Interpolated:
  MacroTop1=0.5295146538333798
  MicroTop1=0.5317771675166733
  Top3=0.8114947038054139
  Top5=0.9095723813260101
  MRR10=0.6893521113166072
```

Shared-K5 apples-to-apples comparison for Interpolated:

```text
K5 pool MacroTop1 = 0.5907826703699341
K10 pool on same 4471 rows = 0.5812870103838225
delta = -0.9496 percentage points
```

## 13. B6 — NGramRecency + Frequency fusion

Runner:

```text
experiments\initial_personalisation\run_initial_ngram_frequency_fusion_v1.py
```

SHA256:

```text
341d194a60fc571a3ef1e6376dffdab87ac7015c988aace611b888ab19f4e147
```

Run:

```powershell
$root = '.\results\personalisation\initial_recovery_comparison_v1'

& $py -m experiments.initial_personalisation.run_initial_ngram_frequency_fusion_v1 `
    --val "$root\initial_train_val_v1.jsonl" `
    --adaptive-dir "$root\candidate_scoring_adaptive_ngram_top10_v1\adaptive_ngram_top10_v1" `
    --output-root "$root\candidate_scoring_ngram_frequency_fusion_v1"
```

Fusion:

```text
score = (1-alpha) * NGram + alpha * F
alpha = {0, 0.25, 0.5, 0.75, 1}
```

Selected Train-Val results:

```text
K5 Hard: alpha=.25
  MacroTop1 0.591558 -> 0.592025
  rescue=34 harm=31 net=+3

K5 Interpolated: alpha=.25
  MacroTop1 0.590783 -> 0.593066
  rescue=43 harm=33 net=+10

K10 Hard: alpha=.50
  MacroTop1 0.527316 -> 0.530326
  rescue=84 harm=69 net=+15

K10 Interpolated: alpha=.25
  MacroTop1 0.529515 -> 0.533746
  rescue=49 harm=28 net=+21
```

Interpretation boundary: these are same-Train-Val selected alpha gains and require later holdout confirmation. The gain is much smaller than Q8+F, consistent with NGramRecency already encoding historical preference through recency-weighted occurrence counts.

## 14. Candidate-scoring scientific record

Current bounded conclusion:

> In the standardized Initial-Pinyin Train-Val personal-candidate scoring task, short lexical suffix matching plus recency provides a substantially better accuracy-latency trade-off than the tested BGE64 context-similarity scorer. Q8 remains more accurate but is much slower. Expanding the recovery pool from K5 to K10 adds 627 Gold targets and only modestly reduces ranking quality on the shared K5 population, with Interpolated NGramRecency becoming the strongest lexical K10 scorer.

Do not convert this into a universal claim about lexical versus semantic context models. Later Full+Short / six-author M1 transfer is a separate future experiment.

## 15. Output-hash freeze checklist

The following completed local artifacts should be hashed and copied into the durable reproducibility record before the phase is closed:

```powershell
$root = '.\results\personalisation\initial_recovery_comparison_v1'

Get-FileHash "$root\candidate_scoring_q8_bge64_v1\candidate_scoring_comparison.json" -Algorithm SHA256
Get-FileHash "$root\candidate_scoring_ngram_recency_v1\ngram_recency\candidate_scoring_comparison.json" -Algorithm SHA256
Get-FileHash "$root\candidate_scoring_adaptive_ngram_top10_v1\adaptive_ngram_top10_v1\candidate_surface_k5_k10.jsonl" -Algorithm SHA256
Get-FileHash "$root\candidate_scoring_adaptive_ngram_top10_v1\adaptive_ngram_top10_v1\comparison.json" -Algorithm SHA256
Get-FileHash "$root\candidate_scoring_ngram_frequency_fusion_v1\ngram_frequency_fusion_comparison.json" -Algorithm SHA256
```

Required protocol assertions remain:

```text
Dev3000 used = false
Test used = false
Gold used for candidate construction / scoring = false
Gold used for Train-Val model-selection diagnostics = true
```

## 16. Deferred engineering notes

Record but do not pursue until after current documentation:

```text
PinyinGPT fixed-candidate speed audit:
- count model forward calls inside score_candidates()
- candidate batching
- shared-prefix KV-cache reuse
- candidate trie scoring
- exact one-forward next-token scoring for compatible one-character candidates where semantics match
- optional NGram fast-path / Q8 slow-path gating
```

Also record future lexical-context transfer to the existing Full+Short / M1 family as a separate experiment; do not alter frozen historical M1 artifacts.
