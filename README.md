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
