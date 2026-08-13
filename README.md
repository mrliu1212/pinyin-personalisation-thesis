# Deep Author Contextual Pinyin Research

The authoritative thesis direction is [RESEARCH_TARGETS.md](RESEARCH_TARGETS.md).
This branch prepares Deep Author Dataset V1.1 for a later generic PinyinGPT T1
evaluation. It does not run model inference or implement personalisation.

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
