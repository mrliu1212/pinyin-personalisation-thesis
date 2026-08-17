# Deep Author Contextual Pinyin Research

The authoritative thesis direction is [RESEARCH_TARGETS.md](RESEARCH_TARGETS.md).
This branch contains the frozen Deep Author Evaluation V2 design and completed
T1 Generic PinyinGPT baseline on Dataset V1. It does not implement
personalisation.

## Retained generic backend

The frozen generic backend is PinyinGPT2-Concat. Its technical audit and pinned
model/code revisions are recorded in
[docs/third_party/pinyingpt.md](docs/third_party/pinyingpt.md). The reusable
adapter is under `src/reference_backend_pinyingpt/`; model-independent Top-K
metrics are under `src/evaluation/`.

## Dataset preparation

The six fixed proxy-user authors are Re_spectators, MScarlet, Etinjat, Agent
Phage, QBLevi and breaddddd. Public/open SCP-CN works created from 2014 through
2021 are discovered through SCPPER-CN, provenance-checked, cleaned, segmented,
and converted into Short and Multi-token contextual Pinyin interactions.

Windows setup and build:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-deep-author.txt
.\.venv\Scripts\python.exe -m experiments.prepare_deep_author_dataset
```

Raw and full processed text are intentionally local and ignored by Git:

- `data/raw/deep_author/`
- `data/processed/deep_author/interactions_t1_ready.jsonl`
- `data/processed/deep_author/works/`

V1 is the historical initial build frozen at `deep-author-dataset-preparation-v1`;
its report is [docs/reports/01_dataset_preparation.md](docs/reports/01_dataset_preparation.md).
V1.1 is the current cleaned dataset: source-confirmed SCP metadata is removed,
retained text is normalized to Simplified Chinese, and simulated IME context and
Gold use Han-only spans separated by hard non-Han boundaries. Its audit is under
`results/audits/deep_author_dataset_v1_1/`, and its correction report is
[docs/reports/01b_dataset_preparation_v1_1.md](docs/reports/01b_dataset_preparation_v1_1.md).
Frozen parameters are in `config/deep_author/run_config.yaml`.

Do not proceed to T1 evaluation until the manual review sample has been checked.

## Evaluation V2

The development evaluation protocol is documented in
[docs/reports/02_deep_author_evaluation_v2.md](docs/reports/02_deep_author_evaluation_v2.md).
It deliberately uses the verified Dataset V1 artifact for the frozen
chronological design and generic T1 baseline; it does not modify either dataset
checkpoint or introduce personalisation.

The completed T1 report is
[docs/reports/03_t1_generic_pinyingpt_baseline.md](docs/reports/03_t1_generic_pinyingpt_baseline.md),
and the project checkpoint index is [docs/VERSION_HISTORY.md](docs/VERSION_HISTORY.md).

## Viewing T1 Results Manually

From the repository root in PowerShell, inspect the frozen results without
loading the model:

```powershell
# Overall summary
Get-Content -Raw `
  results\evaluation\deep_author_v2\t1\metrics_summary.json

# By condition
Import-Csv `
  results\evaluation\deep_author_v2\t1\metrics_by_condition.csv |
  Format-Table

# By author
Import-Csv `
  results\evaluation\deep_author_v2\t1\metrics_by_author.csv |
  Format-Table

# Full versus Initial for paired anchors
Import-Csv `
  results\evaluation\deep_author_v2\t1\paired_full_initial.csv |
  Format-Table

# Runtime and semantic-equivalence validation
Get-Content -Raw `
  results\evaluation\deep_author_v2\t1\runtime_summary.json
Get-Content -Raw `
  results\evaluation\deep_author_v2\t1\regression_summary.json

# Durable prediction count
(Get-Content `
  results\evaluation\deep_author_v2\t1\predictions.jsonl |
  Measure-Object -Line).Lines
```

Metric definitions:

- **Top-1:** fraction of conditions where Gold is ranked first.
- **Top-3:** fraction where Gold appears among the first three candidates.
- **MRR@10:** reciprocal Gold rank within Top-10; missing Gold contributes zero.
- **Missing@10:** fraction where Gold does not appear in Top-10.
- **MeanRank\|Top10:** average Gold rank only when Gold appears in Top-10.
- **Macro-author:** calculate the metric separately for each of the six authors,
  then average authors equally.

**Macro-author Top-1 is the frozen primary T1 metric.** These are Dataset V1
development results for proxy users and reconstructed input, not final
cleaned-dataset thesis numbers.

## Running Personalisation Pilot A M1-H5000 Manually

Pilot A runs only in:

```text
C:\Users\chiar\Desktop\LBH\thesis-personalisation
```

This command tunes M1 on the frozen earlier Dev works, freezes the selected
parameters, and evaluates the exact 6,000 T1 Full+Short Test anchors under the
H5000 history budget. Test G0 candidates are reused from the completed T1
cache; no Test Generic inference is performed. Run it from PowerShell; Codex
does not need to remain active:

```powershell
Set-Location C:\Users\chiar\Desktop\LBH\thesis-personalisation
New-Item -ItemType Directory -Force `
  results\personalisation\pilot_a_context_memory | Out-Null
& C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe `
  -m experiments.personalisation_pilot_a_h5000 `
  --phase all `
  --dataset-root C:\Users\chiar\Desktop\LBH\thesis-deep-author\.build\dataset-v1-reconstruction `
  --pinyingpt-model C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat `
  --embedding-model C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf `
  --t1-predictions C:\Users\chiar\Desktop\LBH\thesis-deep-author\results\evaluation\deep_author_v2\t1\predictions.jsonl `
  1>> results\personalisation\pilot_a_context_memory\h5000_stdout.log `
  2>> results\personalisation\pilot_a_context_memory\h5000_stderr.log
```

The same command safely resumes. Valid Dev Generic rows and compatible BGE
embeddings are reused; provenance mismatches stop with an error. The phases are
`prepare`, `dev-generic`, `dev-embeddings`, `tune`, `test-embeddings`,
`evaluate`, `smoke`, and `all`.

Monitor progress from a second PowerShell window:

```powershell
Get-Content `
  C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\h5000_stdout.log `
  -Tail 30 -Wait
```

Inspect errors:

```powershell
Get-Content -Raw `
  C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\h5000_stderr.log
```

Count completed Dev-tune Generic predictions:

```powershell
(Get-Content `
  C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\cache\generic_predictions.jsonl |
  Measure-Object -Line).Lines
```

Verify full completion after the command exits:

```powershell
$result = Get-Content -Raw `
  C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\h5000\metrics_summary.json |
  ConvertFrom-Json
$result.status
$result.rows
$result.generic_test_inference_rows
$result.test_gold_used_for_tuning
```

Expected values are `complete`, `6000`, `0`, and `False`. Final metrics appear
under `results/personalisation/pilot_a_context_memory/h5000/`. The canonical
BGE cache is `cache/embedding_cache.sqlite3`; its identity excludes the H5000
label, so later H500 and HFull runs can reuse compatible vectors. Large caches
remain local and should not be committed.

## Running Personalisation M2-H5000

M2 retains the completed M1 BGE retrieval and adds the pinned
`BAAI/bge-reranker-base` candidate-aware second stage. See
[the M2 method](docs/research/candidate_aware_personal_memory_m2.md) and the
[pending result report](docs/reports/05_personalisation_m2_h5000.md).

Download the exact official snapshot once (the runner verifies every required
artifact hash):

```powershell
& C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe -c `
  "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-reranker-base', revision='2cfc18c9415c912f9d8155881c133215df768a70', local_dir=r'C:\Users\chiar\Desktop\LBH\thesis\.build\bge-reranker-base', allow_patterns=['*.json','*.txt','*.model','*.safetensors','README.md','LICENSE'])"
```

Inspect the method without running inference:

```powershell
Get-Content -Raw docs\research\candidate_aware_personal_memory_m2.md
Get-Content -Raw results\personalisation\m2_h5000\manifest_summary.json
```

Launch or resume the final pipeline independently from PowerShell:

```powershell
Set-Location C:\Users\chiar\Desktop\LBH\thesis-personalisation
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$stdout = 'results\personalisation\m2_h5000\m2_h5000_stdout.log'
$stderr = 'results\personalisation\m2_h5000\m2_h5000_stderr.log'
$arguments = @(
  '-m', 'experiments.personalisation_m2_h5000',
  '--phase', 'all',
  '--dataset-root', 'C:\Users\chiar\Desktop\LBH\thesis-deep-author\.build\dataset-v1-reconstruction',
  '--pinyingpt-model', 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat',
  '--embedding-model', 'C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf',
  '--reranker-model', 'C:\Users\chiar\Desktop\LBH\thesis\.build\bge-reranker-base',
  '--t1-predictions', 'C:\Users\chiar\Desktop\LBH\thesis-deep-author\results\evaluation\deep_author_v2\t1\predictions.jsonl',
  '--batch-size', '32'
)
$process = Start-Process -FilePath $python -ArgumentList $arguments `
  -WorkingDirectory (Get-Location) -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$process.Id
```

Monitor stdout and inspect stderr:

```powershell
Get-Content results\personalisation\m2_h5000\m2_h5000_stdout.log -Tail 30 -Wait
Get-Content results\personalisation\m2_h5000\m2_h5000_stderr.log -Tail 50
```

Check completion:

```powershell
$result = Get-Content -Raw `
  results\personalisation\m2_h5000\metrics_summary.json | ConvertFrom-Json
$result.status
$result.rows
$result.candidate_pool_invariant
$result.test_gold_used_for_tuning
$result.m1_artifacts_unchanged
```

Expected values after completion are `complete`, `6000`, `True`, `False`, and
`True`. To stop safely, read the launcher and worker IDs from
`results/personalisation/m2_h5000/background_status.json`, then run:

```powershell
$status = Get-Content -Raw `
  results\personalisation\m2_h5000\background_status.json | ConvertFrom-Json
Get-Process -Id $status.worker_pid,$status.launcher_pid `
  -ErrorAction SilentlyContinue | Stop-Process
```

The command is resumable. Valid BGE embeddings, T1 Generic predictions, and M2
pair scores are reused; M1 result artifacts are read-only and hash-checked.

## Running Personal Vocabulary H5000

The detailed method is [Bounded Personal Vocabulary H5000](docs/research/personal_vocabulary.md),
with results in the [completed H5000 report](docs/reports/06_personal_vocabulary_h5000.md).
Outputs and resumable state caches are isolated under
`results\personalisation\personal_vocabulary_h5000\`.

Set common arguments once in PowerShell:

```powershell
Set-Location C:\Users\chiar\Desktop\LBH\thesis-personalisation
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$common = @(
  '--dataset-root', 'C:\Users\chiar\Desktop\LBH\thesis-deep-author\.build\dataset-v1-reconstruction',
  '--pinyingpt-model', 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat',
  '--embedding-model', 'C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf',
  '--t1-predictions', 'C:\Users\chiar\Desktop\LBH\thesis-deep-author\results\evaluation\deep_author_v2\t1\predictions.jsonl'
)
```

Run individual frozen phases:

```powershell
# Audit artifacts and BGE reuse
& $python -m experiments.personal_vocabulary_h5000 --phase prepare @common

# PV0 Test recoverability only; it does not select parameters
& $python -m experiments.personal_vocabulary_h5000 --phase pv0 @common

# Build/resume shared DEV states
& $python -m experiments.personal_vocabulary_h5000 --phase dev-states @common

# Select PV1 Kpv/lambda, then PV2 context lambda on DEV only
& $python -m experiments.personal_vocabulary_h5000 --phase tune @common

# One shared frozen Test pass for PV1/PV2
& $python -m experiments.personal_vocabulary_h5000 --phase evaluate @common
```

The complete resumable workflow is:

```powershell
& $python -m experiments.personal_vocabulary_h5000 --phase all @common
```

Monitor an independently redirected run and inspect errors:

```powershell
Get-Content results\personalisation\personal_vocabulary_h5000\pv_stdout.log -Tail 30 -Wait
Get-Content results\personalisation\personal_vocabulary_h5000\pv_stderr.log -Tail 50
```

Check completion:

```powershell
$result = Get-Content -Raw `
  results\personalisation\personal_vocabulary_h5000\metrics_summary.json | ConvertFrom-Json
$result.status
$result.rows
$result.generic_test_inference_rows
$result.test_gold_used_for_tuning
$result.gold_used_for_vocabulary_construction
$result.previous_artifacts_unchanged
```

Expected values are `complete`, `6000`, `0`, `False`, `False`, and `True`.
The shared BGE cache remains
`results\personalisation\pilot_a_context_memory\cache\embedding_cache.sqlite3`.

## Running the Reranking Personalisation Matrix

The [matrix method](docs/research/reranking_personalisation_matrix.md) completes
F/M1/M2 across four T1 conditions and H500/H5000/HFull. Its result root is
`results\personalisation\reranking_matrix\`.

```powershell
Set-Location C:\Users\chiar\Desktop\LBH\thesis-personalisation
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$common = @(
  '--dataset-root', 'C:\Users\chiar\Desktop\LBH\thesis-deep-author\.build\dataset-v1-reconstruction',
  '--pinyingpt-model', 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat',
  '--embedding-model', 'C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf',
  '--reranker-model', 'C:\Users\chiar\Desktop\LBH\thesis\.build\bge-reranker-base',
  '--t1-predictions', 'C:\Users\chiar\Desktop\LBH\thesis-deep-author\results\evaluation\deep_author_v2\t1\predictions.jsonl'
)

# Audit only: no neural inference
& $python -m experiments.reranking_personalisation_matrix --phase audit @common

# Launch or resume the matrix
& $python -m experiments.reranking_personalisation_matrix --phase run @common

# Monitor and check completion
Get-Content results\personalisation\reranking_matrix\matrix_stdout.log -Tail 30 -Wait
Get-Content results\personalisation\reranking_matrix\matrix_stderr.log -Tail 50
Get-Content -Raw results\personalisation\reranking_matrix\matrix_manifest.json
Get-Content -Raw results\personalisation\reranking_matrix\COMPLETE.json
```

The same `--phase run` command resumes incomplete or failed cells. A detached
worker can be stopped safely with the PID recorded in
`results\personalisation\reranking_matrix\background_status.json`:

```powershell
$status = Get-Content -Raw results\personalisation\reranking_matrix\background_status.json | ConvertFrom-Json
Get-Process -Id $status.worker_pid -ErrorAction SilentlyContinue
Stop-Process -Id $status.worker_pid
```
