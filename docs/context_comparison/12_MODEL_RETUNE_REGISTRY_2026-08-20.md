# Standardized model and retuning registry — 2026-08-20

Status: **identities and search spaces frozen; Train-Val selection in progress**  
Primary selection metric: **Macro-author Top1**  
Selection data: **Clean3 Train-Val only**  
Dev3000 involved in selection: **false**  
Test used: **false**

## Frozen identities

| System | Frozen neural identity | Allowed standardized change |
|---|---|---|
| Generic | PinyinGPT2-Concat `76dd20dc...e2263`, beam 16, Top-10 | none |
| Frequency | no neural model | lambda selection |
| M1 | BGE `bge-small-zh-v1.5`, SHA `5a88d266...12039` | Top-N and lambda |
| M2 | BGE Stage-1 + generic `BAAI/bge-reranker-base@2cfc18...` | Stage-1 K and lambda |
| Hidden-M1 | frozen PinyinGPT final-layer final-`[SEP]` state | Top-N and lambda |
| Hidden-M2 | Hidden Stage-1 + same frozen generic CE | Stage-1 K and lambda |
| EM3-Clean3 | initialize from pinned generic CE base; Train-Fit BCE | Stage-1 K and lambda after training |

Every personal system uses same-author strictly-prior rolling raw H5000 before
exact segmented-Pinyin filtering and the same frozen Generic Top-10 candidate
surface. Candidate injection/fusion is not introduced.

## Search-space freeze

- Frequency lambda: `0, 0.25, 0.5, 1, 2, 4`.
- M1/Hidden-M1 Top-N: `1, 3, 5, 10, 20`; lambda:
  `0, 0.25, 0.5, 1, 2, 4`.
- M2/Hidden-M2/EM3 Stage-1 K: `10, 20`; lambda: `0.5, 1, 2, 4`.
- No boundary expansion is allowed after results are observed.
- Exact ties choose lower lambda, then lower K/Top-N, then canonical config
  order.

## EM3 frozen recipe

- Base: pinned `BAAI/bge-reranker-base` revision `2cfc18...`.
- Loss: `BCEWithLogitsLoss`.
- Epochs: 1; learning rate: `2e-5`.
- Physical batch: 4; gradient accumulation: 8; effective batch: 32.
- Maximum pair length: 512.
- Optimizer: `torch.optim.AdamW`.
- Warmup: 10% of optimizer steps, at least one step; linear schedule.
- Seed: 42 for deterministic pair selection, shuffle, and training RNG.
- Mixed precision: fp16 on CUDA.
- Checkpoint interval: 1,000 optimizer steps; final checkpoint always saved.
- Pair construction: H5000 before exact Pinyin, up to three query-local rounds,
  one unused positive and up to three unused negatives per round, SHA256-derived
  query-local RNG, no fabricated repetitions.

Machine registries:

- `results/personalisation/context_comparison_v2/standardized_model_registry_v1.json`
- `results/personalisation/context_comparison_v2/search_space_registry_v1.json`

Selected configurations and all tried Train-Val metrics will be appended only
after the sealed searches complete.

## Completed Train-Val selection — 2026-08-21

All selections used Clean3 Train-Val only. `used_dev3000=false` and
`used_test=false` for every search.

| System | Frozen selection | Macro-author Top1 |
|---|---|---:|
| Frequency | lambda `4.0` | recorded in `pre_dev_freeze_v1.json` |
| M1 | BGE Full cosine, Top-N `5`, lambda `4.0` | recorded in `pre_dev_freeze_v1.json` |
| Hidden-M1 | PinyinGPT hidden cosine, Top-N `5`, lambda `4.0` | recorded in `pre_dev_freeze_v1.json` |
| M2 | BGE Stage-1 K `10`, lambda `4.0` | `0.7769184347` |
| Hidden-M2 | Hidden Stage-1 K `10`, lambda `4.0` | `0.7762991774` |
| EM3-Clean3 | Hidden Stage-1 K `10`, lambda `4.0` | `0.7771533900` |

The complete tried grids and exact configuration/model hashes are preserved in
`results/personalisation/context_comparison_v2/pre_dev_freeze_v1.json`. The
human-readable freeze is
`docs/context_comparison/13_PRE_DEV_FREEZE_2026-08-21.md`.
