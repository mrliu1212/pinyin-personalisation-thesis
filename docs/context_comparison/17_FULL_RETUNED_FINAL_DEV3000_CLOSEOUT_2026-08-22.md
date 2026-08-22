# Full+Short Retuned Final — Train-Val Selection and Dev3000 Development Closeout

**Date:** 2026-08-22
**Experiment:** `full_retune_final_trainval_dev_v1`
**Status:** DEVELOPMENT CLOSED / TEST CLOSED

## 1. Purpose and scientific status

This follow-up gives the transferred Initial-final architecture the same kind of Full-specific Train-Val hyperparameter-selection opportunity used for model development elsewhere in the Full+Short comparison. The selected configuration is frozen before the Dev phase. Dev3000 is used as a development comparison surface, not as an untouched final evaluation. Test remains unused and closed.

The earlier seven-system standardized comparison remains a separate historical frozen comparison. This follow-up does not retroactively alter that pre-Dev selection record; it is a post-Dev development extension with explicit provenance.

## 2. Architecture held fixed

The method keeps the transferred two-stage architecture and causal history semantics:

```text
same author -> strictly prior -> latest H5000 raw -> exact segmented-Pinyin
```

- Personal recovery K: `5`.
- Frequency lambda: `4.0`.
- P_NG: maxN `2`, kappa `1.0`, tau `2048.0`.
- Stage-2 NGramRecency: maxN `2`, tau `2048.0`.
- Stage-2 BGERecency: context `64` Chinese characters, TopN/candidate `5`, tau `2048.0`.
- Empty Generic surface policy: `conservative no-op`.
- Gold is never used for candidate construction or online scoring.

## 3. Full Train-Val selection protocol

Selection is sequential. Stage1 weights are selected first on Full Train-Val; Stage1 is then frozen while Stage2 context weights are selected. Dev3000 is not read during parameter selection.

Stage1 grid (48 points):

```text
w_P  = [0.0, 2.0, 4.0, 6.0]
w_CS = [0.0, 2.0, 4.0, 6.0]
w_E  = [0.0, 2.0, 4.0]
```

Stage2 grid (25 points):

```text
lambda_N = [0.0, 2.0, 4.0, 6.0, 8.0]
lambda_B = [0.0, 2.0, 4.0, 6.0, 8.0]
```

Primary selection criterion: Macro-author Top1; ties are broken by Micro Top1, then MRR@10, then distance to the transferred Initial reference, then deterministic lexicographic order.

### Selected Full-specific configuration

```text
Stage1: w_P=2.0, w_CS=6.0, w_E=4.0
Stage2: lambda_N=6.0, lambda_B=6.0
```

Selected Train-Val headline:

| Method | Macro Top1 | Micro Top1 | Top3 | MRR@10 | Missing@10 |
|---|---:|---:|---:|---:|---:|
| RetunedStage1 | 78.942% | 81.904% | 91.036% | 0.867756 | 5.198% |
| **RetunedFinal** | **79.600%** | **82.499%** | **91.208%** | **0.871378** | **5.198%** |

## 4. Dev3000 horizontal comparison

Dev3000 is author-balanced, so Macro-author Top1 equals Micro Top1 on this surface.

| Method | Macro Top1 | Micro Top1 | Top3 | MRR@10 | Missing@10 |
|---|---:|---:|---:|---:|---:|
| Generic | 72.267% | 72.267% | 87.667% | 0.806844 | 5.133% |
| Frequency | 82.500% | 82.500% | 91.533% | 0.872403 | 5.133% |
| M1 | 82.833% | 82.833% | 91.333% | 0.874323 | 5.133% |
| M2 | 82.433% | 82.433% | 91.333% | 0.872074 | 5.133% |
| Hidden-M1 | 82.633% | 82.633% | 91.200% | 0.872864 | 5.133% |
| Hidden-M2 | 82.367% | 82.367% | 91.300% | 0.871268 | 5.133% |
| EM3 | 82.733% | 82.733% | 91.200% | 0.873047 | 5.133% |
| RetunedStage1 | 83.400% | 83.400% | 93.133% | 0.885842 | 2.900% |
| **RetunedFinal** | **84.367%** | **84.367%** | **93.433%** | **0.892041** | **2.900%** |

RetunedFinal is the highest observed Dev system on Top1, Top3, and MRR@10 among the compared systems. No statistical-significance claim is made here.

### Deltas versus key baselines

- vs M1: Top1 `+1.533 pp`, Top3 `+2.100 pp`, MRR `+0.017718`, Missing `-2.233 pp`.
- vs Frequency: Top1 `+1.867 pp`.
- vs RetunedStage1: Top1 `+0.967 pp`.

## 5. Top1 transition accounting

| Comparison | Rescue | Harm | Net |
|---|---:|---:|---:|
| Frequency -> RetunedFinal | 111 | 55 | +56 |
| M1 -> RetunedFinal | 105 | 59 | +46 |
| RetunedStage1 -> RetunedFinal | 67 | 38 | +29 |

The Stage1-to-Final transition is net positive on Dev, supporting the two-stage interpretation: candidate recovery/personal preference determines availability and a strong first ordering, while Stage2 contextual recency further improves ordering overall despite some harms.

## 6. Development conclusion and freeze decision

RetunedStage1 reaches `83.400%` Dev Top1 and RetunedFinal reaches `84.367%`. The Stage2 gain is `+0.967 pp`, corresponding to `67 rescue / 38 harm / net +29 rows.

Relative to M1 (`82.833%`), RetunedFinal improves Dev Top1 by `+1.533 pp` and reduces Missing@10 from `5.133%` to `2.900%`.

This closes the Full-retuned Final development segment. Do not change the selected weights, candidate policy, history semantics, or Stage2 configuration on the basis of this Dev result unless this research phase is explicitly reopened. Test remains CLOSED and is reserved for a separately authorized final frozen evaluation.

## 7. Reproducibility and provenance

- Retune runner: `experiments/context_comparison/run_full_retune_final_trainval_dev_v1.py`
- Retune runner SHA256: `89d526cb61d3bb93a1caa3d401679db9f1f8b8efdc31d4daa4590adcce3dee8d`
- Frozen base transfer runner: `experiments/context_comparison/run_full_transfer_initial_final_v1.py`
- Base runner SHA256: `f75d40f381e966f85cd4b20647ba7dc6a95df9116ad8657ca9a07505949a37b0`
- Machine-readable selected configuration: `results/personalisation/context_comparison_followup_v1/full_retune_final_trainval_dev_v1/selected_config.json`
- selected_config.json SHA256: `3dc3fb908aeeaa853526ad71cf85de7400f47d261ed7c09acdd8197446f5fa3d`
- Train-Val result: `results/personalisation/context_comparison_followup_v1/full_retune_final_trainval_dev_v1/tune/train_val_result.json`
- train_val_result.json SHA256: `2899270eca2c474957afbb7cb1943140bd576ad3aa76514223ee2a0b0f4c7b48`
- Dev result: `results/personalisation/context_comparison_followup_v1/full_retune_final_trainval_dev_v1/dev/dev_result.json`
- dev_result.json SHA256: `07a9fb80a138681db6de05cba7361e948a880b61b02868ec2c06037ff69e48da`
- Tune artifact-check record runner SHA: `89d526cb61d3bb93a1caa3d401679db9f1f8b8efdc31d4daa4590adcce3dee8d`
- Dev artifact-check record runner SHA: `89d526cb61d3bb93a1caa3d401679db9f1f8b8efdc31d4daa4590adcce3dee8d`
- Train-Val selection used Dev3000: `False`
- Train-Val selection used Test: `False`
- Dev hyperparameter search: `False`
- Dev used Test: `False`

Generated JSON/JSONL/SQLite/cache/result trees remain LOCAL-ONLY / DO NOT STAGE.

## 8. Exact commands

Train-Val selection:

```powershell
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
Set-Location 'C:\Users\chiar\Desktop\LBH\thesis-context-compare'

& $python `
  '.\experiments\context_comparison\run_full_retune_final_trainval_dev_v1.py' `
  --phase tune `
  --fit '.\results\personalisation\context_comparison_v2\clean3_train_fit_v1.jsonl' `
  --val '.\results\personalisation\context_comparison_v2\clean3_train_val_v1.jsonl' `
  --generic '.\results\personalisation\context_comparison_v2\train_val_generic\predictions.jsonl' `
  --standardized-stage1 '.\results\personalisation\context_comparison_v2\stage1\train_val.jsonl' `
  --checkpoint 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat' `
  --bge-model 'C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf' `
  --seed-bge-cache '.\results\personalisation\context_comparison_followup_v1\full_transfer_initial_final_v1\bge_context_cache.sqlite3' `
  --output-root '.\results\personalisation\context_comparison_followup_v1\full_retune_final_trainval_dev_v1' `
  --compatibility-device cpu `
  --cuda-path 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8' `
  --progress-every 500
```

Frozen Dev evaluation:

```powershell
& $python `
  '.\experiments\context_comparison\run_full_retune_final_trainval_dev_v1.py' `
  --phase dev `
  --pilot-history 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\history_manifest.jsonl' `
  --pilot-dev 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\dev_manifest.jsonl' `
  --frozen-dev '.\results\personalisation\context_comparison_v1\clean3_history_balanced_3000.jsonl' `
  --standardized-dev-root '.\results\personalisation\context_comparison_v2\dev3000' `
  --checkpoint 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat' `
  --bge-model 'C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf' `
  --output-root '.\results\personalisation\context_comparison_followup_v1\full_retune_final_trainval_dev_v1' `
  --compatibility-device cpu `
  --cuda-path 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8' `
  --progress-every 250
```

## 9. Interpretation boundary

The Dev numbers are development evidence. They support the claim that RetunedFinal has the highest observed Dev performance among the compared systems on the reported ranking metrics, but they do not establish statistical significance and they are not a substitute for the final untouched Test evaluation.
