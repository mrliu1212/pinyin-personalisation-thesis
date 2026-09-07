# Transparent and User-Controllable Personalisation for Chinese Pinyin Input

This repository contains the final research code, compact evaluation artifacts, and reproducibility material for a thesis on personalised Chinese Pinyin input.

The project studies whether a Pinyin input system can improve candidate ranking from a user's historical writing while keeping personalisation understandable and controllable.

## Final system

The final system combines two complementary forms of personalisation:

1. **Neural Adapter**
   - A small user-specific residual adapter is inserted into a frozen PinyinGPT model.
   - It learns broader user-specific lexical and contextual preferences.

2. **Explicit Memory**
   - A bounded causal history of the user's previous interactions is searched directly.
   - Exact and compositional recovery provide candidates and interpretable history-derived evidence.
   - A Rich30 LambdaMART reranker combines neural and memory signals.

The final Hybrid system is therefore:

> **PinyinGPT + user Adapter + H5000 Explicit Memory + Rich30 LambdaMART**

The base model is:

`aihijo/transformers4ime-pinyingpt-concat`

with the frozen revision used in the experiments documented in the repository.

## Final evaluation

The frozen Test set contains **40,000 interactions** from five proxy users.

| System | Top-1 | Top-3 | Top-5 | Top-10 | MRR |
|---|---:|---:|---:|---:|---:|
| Generic | 34.92% | 46.17% | 49.86% | 53.54% | 0.4132 |
| Generic + Frequency | 36.82% | 48.61% | 52.10% | 55.62% | 0.4346 |
| Generic + Explicit Memory | 45.12% | 55.31% | 58.52% | 61.68% | 0.5090 |
| Adapter | 64.40% | 75.10% | 77.82% | 80.41% | 0.7024 |
| Adapter + Explicit Memory | **65.68%** | **76.22%** | **79.01%** | **81.44%** | **0.7149** |

The Hybrid system improves Top-1 accuracy by **30.77 percentage points** over the Generic baseline.

The Adapter provides the largest improvement. Explicit Memory provides a smaller but complementary gain and also offers a more direct and traceable form of user control.

## Dataset design

Five authors are used as proxy users:

- Re_spectators
- Etinjat
- Agent Phage
- QBLevi
- breaddddd

The final frozen dataset contains:

- **150,000 Fit interactions**
- **20,000 Validation interactions**
- **40,000 Test interactions**
- **210,000 interactions total**

Splits are chronological and are made at whole-work level.

The original long-form source corpus and large generated interaction surfaces are **not bundled in this compact repository snapshot**. Compact manifests, audits, frozen split summaries, and evaluation outputs are included under `results/`.

## Repository structure

```text
.
├── README.md
├── REPORT.md
├── FINAL_RESEARCH_CLOSURE_20260829.md
├── requirements-pinyingpt.txt
│
├── experiments/
│   └── model_level/
│       ├── final five-author dataset / H5000 scripts
│       ├── Adapter training and candidate generation
│       ├── Explicit Memory and Rich30 evaluation
│       ├── five-system comparison
│       ├── robustness analysis
│       └── supplementary EDA
│
├── src/
│   ├── model_level/
│   │   ├── concat_training.py
│   │   ├── pinyin_modes.py
│   │   └── pinyingpt_adapter.py
│   └── reference_backend_pinyingpt/
│       └── backend.py
│
├── results/
│   └── finalmodel_fiveauthor_v1/
│       ├── dataset_preparation_v1/
│       ├── adapter_input_preparation_v1/
│       ├── five_system_test_comparison_v1/
│       ├── statistical_robustness_and_recovery_v1/
│       ├── eda_v1/
│       └── final_evaluation_protocol_v1.json
│
└── docs/
    ├── R4_CONTROLLED_REVERSIBLE_PREFERENCE_ADAPTATION_REPORT.md
    ├── model_level/
    │   └── 03_ADAPTER_V1_REPRODUCIBILITY_2026-08-23.md
    └── third_party/
        └── pinyingpt.md
```

## How to read this repository

For a quick overview:

1. Read `REPORT.md`.
2. Inspect `results/finalmodel_fiveauthor_v1/five_system_test_comparison_v1/overall_v1.csv`.
3. Read `FINAL_RESEARCH_CLOSURE_20260829.md` for the frozen final research state.

For reproducibility and implementation:

1. Start with `requirements-pinyingpt.txt`.
2. Read `docs/model_level/03_ADAPTER_V1_REPRODUCIBILITY_2026-08-23.md`.
3. Inspect `src/model_level/`.
4. Follow the final scripts in `experiments/model_level/`.

## Reproduction notes

The original experiments were executed on an HPC environment.

Large files are intentionally excluded from this repository, including model weights, trained Adapter checkpoints, large row-level prediction surfaces, the complete source corpus, and large generated interaction datasets.

Some `.sbatch` files retain the original HPC paths used in the final experiment. These paths document the executed environment rather than providing portable local defaults.

## Transparency and controllability

The thesis distinguishes between two forms of personalisation:

- **Adapter personalisation** is adaptive and implicit.
- **Explicit Memory personalisation** is external and inspectable.

This repository currently contains the compact R4 behavioural adaptation report. Additional compact controllability artifacts from the final thesis experiments are planned for a later repository version.

## Supplementary EDA

The `eda_v1` analysis was conducted after the final evaluation was frozen and was **not used for Test-set tuning or final model selection**.

## Historical work

Earlier research stages, calibration experiments, pilot datasets, long-term adaptation studies, and deprecated Phase 1–4 pipelines have been removed from the cleaned main snapshot.

They remain available through the repository's historical Git branches and commit history.

## Thesis status

The predictive evaluation is frozen. The current repository is intended as a compact final research snapshot rather than a full archive of every intermediate experiment.
