# 03 - Full RetunedFinal Baseline Reproduction

Date: 2026-08-22

Status: **EXACT REPRODUCTION PASSED**

## 1. What and why

Before evaluating a new estimator, the frozen Full RetunedFinal Train-Val
ranking was reconstructed from its hash-pinned Stage-1 features and Stage-2
supports. This is stronger than merely rereading the recorded metrics: every
Stage-1 Top10, Stage-1 score, final Top10, and gold rank was regenerated using
the frozen arithmetic and compared to the durable selected predictions.

No model inference was run. Dev3000 and Test were neither accepted nor read.

## 2. Frozen parameters verified

```text
w_P=2
w_CS=6
w_E=4
lambda_N=6
lambda_B=6
```

The machine-readable selected configuration hash is
`3dc3fb908aeeaa853526ad71cf85de7400f47d261ed7c09acdd8197446f5fa3d`.

## 3. Result

All 34,416 rows matched exactly in order, Stage-1 candidate surface, final
candidate order, and gold rank.

| Population | N | Macro Top1 | Micro Top1 | Top3 | Top5 | MRR@10 | Missing@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 34,416 | .796004927 | .824994189 | .912075779 | .930758949 | .871377873 | .051981636 |
| Ambiguous | 10,053 | .803894152 | .808514871 | .940117378 | .958420372 | .874789726 | .029344474 |
| Conflict | 1,865 | .252588201 | .235924933 | .765147453 | .838069705 | .500956849 | .108847185 |

Per-author overall Top1 is .887271669 for Agent Phage, .601494396 for
Etinjat, and .899248715 for breaddddd.

## 4. Reproduction command

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  -m experiments.external_memory_next.reproduce_full_retuned_baseline_v1 `
  --stage1 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_stage1_features.jsonl' `
  --stage2 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_stage2_supports.jsonl' `
  --predictions 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_selected_predictions.jsonl' `
  --config 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\selected_config.json' `
  --output-root '.\results\personalisation\external_memory_next\full_retuned_baseline_reproduction_v1'
```

## 5. Hashes

- Runner: `acd69de6c6a8a3d0c7c414627dfc8a6fc45b1414295204a91728228f03d0cfc5`
- `baseline_reproduction.json`:
  `f1e0ac3e0889bd107bd48f520d9b0667ac39e4c137ef04b951843c7d91013cd7`
- Frozen Stage-1 feature input:
  `e311261cb0c5ea252ce593fdaa43ff87826d19eef440f39e2efc67ddc4310b13`
- Frozen Stage-2 support input:
  `d413d02650a759c0a759e1845212e68b2d948c1a1d20fc88abfd89ea7973bc64`
- Frozen selected prediction input:
  `f3e902e5a9e7d25e62799b9abb719026c336381eacc42999d1e7edccf2731b22`

## 6. Interpretation and limitation

The mandatory comparison gate is satisfied: new arithmetic can be compared
against a reproducible reference rather than a prose number. This does not
revalidate Generic model inference or BGE embeddings; it verifies the exact
frozen evidence-fusion result using the already validated durable inputs.

Current decision: proceed to the predeclared smoothing-only ablation.
