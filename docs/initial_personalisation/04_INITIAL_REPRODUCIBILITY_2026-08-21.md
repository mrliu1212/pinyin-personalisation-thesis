Initial+Short Standardized Candidate-Coverage Reproducibility

Date: 2026-08-21

This note records how to reproduce the standardized Initial+Short candidate-coverage and Full-vs-Initial paired findings without using Dev3000 or Test.

1. Repository and environment

Worktree:

C:\Users\chiar\Desktop\LBH\thesis-initial-research

Branch:

work/initial-personalisation

Python:

C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe

Frozen PinyinGPT local checkpoint:

C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat

Frozen Generic identity:

aihijo/transformers4ime-pinyingpt-concat@76dd20dc92d8236a350fb732e99dde6fa15e2263
beam_size = 16
top_k = 10
production-compatible n_positions = 1024

Generic must use current context + current Pinyin only. It must not read personal history.

2. Standardized Full source split

Reference worktree:

C:\Users\chiar\Desktop\LBH\thesis-context-compare

Result root:

C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_v2

Frozen standardized split:

Train-Fit rows = 144,526
Train-Val rows = 34,416

Expected SHA256:

Train-Fit = 547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6
Train-Val = d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220

Files:

...\context_comparison_v2\clean3_train_fit_v1.jsonl
...\context_comparison_v2\clean3_train_val_v1.jsonl

3. A0 — deterministic Full -> Initial transformation and rolling H5000 history

Script:

experiments\initial_personalisation\01_prepare_initial_standardized.py

Run from the Initial worktree root:

& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  '.\experiments\initial_personalisation\01_prepare_initial_standardized.py'

The Initial transform is deterministic:

Full segmented Pinyin syllable -> first letter of each syllable

Example:

shi yong -> s y

History semantics are mandatory:

same author
-> strictly prior interactions
-> latest up-to-5000 RAW interactions
-> Initial-Pinyin exact match afterward

Do not take unlimited same-Initial history and cap afterward.

Expected output root:

results\personalisation\initial_recovery_comparison_v1

Expected files include:

initial_train_fit_v1.jsonl
initial_train_val_v1.jsonl
initial_transform_audit.json
history_semantics_audit.json
manifest.json

Expected Train-Val audit:

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

The A0 manifest must report:

dev3000_used = false
test_used = false
generic_inference_performed = false

4. A1 — standardized Initial Generic Top10

Script:

experiments\initial_personalisation\02_run_initial_train_val_generic.py

Run:

$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$ckpt  = 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat'

& $python -m experiments.initial_personalisation.run_initial_train_val_generic `
  --input '.\results\personalisation\initial_recovery_comparison_v1\initial_train_val_v1.jsonl' `
  --checkpoint $ckpt `
  --output-root '.\results\personalisation\initial_recovery_comparison_v1\train_val_generic' `
  --device cuda

Expected final file:

results\personalisation\initial_recovery_comparison_v1\train_val_generic\predictions.jsonl

Expected prediction SHA256:

bd0fb4dc304e0b266b90fae6fe3ac65424d2f52b23fedfa881212706ba2c2873

Expected runtime summary fields:

status = complete
rows = 34,416
beam_size = 16
top_k = 10
device = cuda
dev3000_used = false
test_used = false

Observed run:

run_seconds = 1175.826129800058
mean_inference_ms_per_row = 29.689704216040887
median_inference_ms_per_row = 32.10877499077469

5. A2 — Initial candidate recoverability

Script:

experiments\initial_personalisation\03_evaluate_initial_recoverability.py

The evaluator must import:

from src.personalisation.pilot_a import HistoryIndex
from src.personalisation.context_memory import PredictionQuery

Run:

$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$ckpt  = 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat'

& $python -m experiments.initial_personalisation.evaluate_initial_recoverability `
  --initial-train-fit '.\results\personalisation\initial_recovery_comparison_v1\initial_train_fit_v1.jsonl' `
  --initial-train-val '.\results\personalisation\initial_recovery_comparison_v1\initial_train_val_v1.jsonl' `
  --generic-predictions '.\results\personalisation\initial_recovery_comparison_v1\train_val_generic\predictions.jsonl' `
  --checkpoint $ckpt `
  --output-root '.\results\personalisation\initial_recovery_comparison_v1\recoverability' `
  --device cpu

Expected headline results:

Rows = 34,416
Generic Top1 = 0.33057298930729895
Generic Missing@10 = 0.36509181775918176
Raw recoverable | missing = 0.4771985674492638
Compatible recoverable | missing = 0.470274572224433
Compatible recoverable@1 | missing = 0.2110624751293275
Compatible recoverable@3 | missing = 0.33656983684838837
Compatible recoverable@5 | missing = 0.3907680063668922

Expected summary:

results\personalisation\initial_recovery_comparison_v1\recoverability\recoverability_summary.json

Interpretation boundary: recoverability and oracle candidate coverage are not actual reranking accuracy.

6. Same-anchor standardized Full vs Initial candidate-coverage audit

Script:

experiments\initial_personalisation\04_evaluate_full_vs_initial_candidate_coverage.py

This audit must compare the exact same 34,416 standardized Train-Val anchors and verify author/work/chronology/context/Gold identity, with only Full Pinyin -> deterministic Initial changed.

Run:

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

Expected headline results:

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

Useful derived decomposition:

Observed Top1 gap = 73.6024% - 33.0573% = 40.5451 percentage points
Candidate-coverage deterioration = 36.5092% - 6.9212% = 29.5880 percentage points
Additional within-Top10 ranking difficulty = 40.5451 - 29.5880 = 10.9571 percentage points
Coverage share of observed Top1 gap = 29.5880 / 40.5451 ~= 72.98%

Use careful wording: the 72.98% value is an error-accounting decomposition, not a causal attribution.

Expected output files:

full_vs_initial_candidate_coverage\full_recoverability_rows.jsonl
full_vs_initial_candidate_coverage\full_vs_initial_paired_rows.jsonl
full_vs_initial_candidate_coverage\full_vs_initial_summary.json
full_vs_initial_candidate_coverage\full_vs_initial_headline.csv

7. Reproduction checks before citing results

Before using these numbers in the thesis, verify:

1. git branch is work/initial-personalisation
2. input Full Train-Fit/Train-Val SHA256 match the frozen values above
3. A0 manifest reports dev3000_used=false and test_used=false
4. A1 prediction SHA256 matches bd0fb4dc...
5. A1 runtime_summary reports 34,416 rows, beam=16, top_k=10
6. A2 completes on exactly 34,416 rows
7. Full-vs-Initial paired audit completes on exactly 34,416 rows
8. paired audit reports dev3000_used=false and test_used=false
9. no result is regenerated after changing history semantics, Pinyin transform, checkpoint, beam, Top-K, or context semantics

8. Scientific boundary

These are standardized Train-Val research/development findings. They are not sealed Dev3000 confirmation results and they are not Test results.

The current finding is:

On the same 34,416 standardized queries, converting Full Pinyin to Initial input increases Generic Missing@10 from 6.92% to 36.51%. Of the 10,223 queries whose Gold is covered under Full but lost under Initial, 51.88% remain recoverable from backend-compatible causal personal history.

Do not use Dev3000 or Test to change the current method based on these findings.

9. B1 — Frozen unified Initial personal candidate surface

Purpose: construct one prediction-visible, backend-compatible personal candidate list that must be reused by PV1 and EM1. The surface is built on standardized Initial Train-Val only; Dev3000 and Test remain closed.

History semantics:

same author
-> strictly prior interactions
-> latest up-to-5000 RAW interactions
-> exact Initial-segment match
-> frequency lexicon
-> remove candidates already in Generic Top10
-> frozen PinyinGPT backend compatibility filter
-> frequency-descending personal list with lexical tie-break

Run:

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

Frozen B1 result:

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

Frozen output files:

results/personalisation/initial_recovery_comparison_v1/candidate_surface/train_val_candidate_surface.jsonl
results/personalisation/initial_recovery_comparison_v1/candidate_surface/candidate_surface_summary.json
results/personalisation/initial_recovery_comparison_v1/candidate_surface/candidate_surface_manifest.json

Before PV1 or EM1, verify the candidate-surface SHA256 exactly matches the value above. PV1 and EM1 must not independently rebuild, add, delete, or reorder personal candidates outside the frozen K-prefix rule.