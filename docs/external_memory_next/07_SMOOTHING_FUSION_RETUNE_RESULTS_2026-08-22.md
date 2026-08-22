# 07 - Smoothing Fusion Retune Results

Date: 2026-08-22

Status: **COMPLETE / NO ADDITIONAL RETUNE GAIN**

## 1. What was done

The predeclared sequential grid in record 06 was evaluated arithmetically on
the frozen Full RetunedFinal candidate/support surface. Stage A jointly tested
30 `(alpha, w_CS)` points. Stage B froze the Stage-A selection and tested the
existing 25 `(lambda_N, lambda_B)` points.

## 2. Result

Both stages reselected the smoothing-only configuration:

```text
alpha=128
w_P=2, w_CS=6, w_E=4
lambda_N=6, lambda_B=6
```

The final metrics and 25-rescue/9-harm transition are therefore identical to
record 05. No coefficient change improved Macro-author Top1.

The nearest Stage-A competitor, `(alpha=32, w_CS=2)`, tied the selected point
on Micro Top1 but had lower Macro Top1 (`.796500349` versus `.796515499`).
Raw Choice Share with `w_CS=2` also reached Macro `.796500349`, only
`.000015150` below the selected smoothed point. Because this fixed-surface
scale control is nearly tied, the current evidence does not yet show that
population-prior information itself matters beyond conservative suppression of
raw Choice Share.
The original Stage-2 `(6,6)` point remained the clear grid winner; the next
point `(6,4)` reached Macro `.796285963`.

## 3. Interpretation

The positive result is specifically attributable to the causal Choice Share
term rather than Stage-2 retuning, but the near-tied lower-weight control means
the mechanism may be generic shrinkage rather than use of the population prior.
A narrow prior-source decomposition is required before closing Experiment A.
The negative lambda result argues against expanding the Stage-2 linear grid
further on the same Train-Val surface.

## 4. Reproduction

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  -m experiments.external_memory_next.run_smoothing_fusion_retune_v1 `
  --fit 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_v2\clean3_train_fit_v1.jsonl' `
  --val 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_v2\clean3_train_val_v1.jsonl' `
  --stage1 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_stage1_features.jsonl' `
  --stage2 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_stage2_supports.jsonl' `
  --predictions 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_selected_predictions.jsonl' `
  --config 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\selected_config.json' `
  --output-root '.\results\personalisation\external_memory_next\choice_share_smoothing_fusion_retune_fixed_surface_v1'
```

## 5. Hashes and boundary

- Runner: `4631e04271c1d040e014e899fb5ae0e457dfa541908703eee49bd02f87648c64`
- Shared smoothing core: `a09fde45440d036fa50f8341f90c9f7f596c08b59de01b2eb5f08f0487451632`
- `result.json`: `4a5f02f4234cb3291f427803c656de9d1e881218c79907735e148d6738765945`
- Stage-A grid: `12fba9babcfb7e47d2084624a22c4f850e821b46933a4dc6c5eb348a5aa61a16`
- Stage-B grid: `f852037a06c7feb5408f2dee2d567dc7d27f121ad0dd11f4f99252bc337b2113`
- Selected predictions:
  `57f1daef8c687b03c2ec4754b4f659d011e70dd196a35da7720b592b3c13dbfe`
- `used_dev3000=false`; `used_test=false`.

This remains a Train-Val development result and does not establish holdout
generalisation.
