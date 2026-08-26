# Hybrid Reproducibility Index

**Date:** 2026-08-26
**Branch:** `work/hybrid-personalisation`
**Evaluation policy:** Train-Val / development-safe only
**Test:** CLOSED

This index records the code, result summaries, frozen dependencies, feature semantics, and fixed dataset hashes required to reproduce the Hybrid and cross-regime transfer experiments.

## 1. In-repository runners and result summaries

| Kind | Path | SHA256 | Role |
|---|---|---|---|
| Runner | `experiments\hybrid\run_h0_agent_adapter_lambdamart_borda_v1.py` | `5d88587a33eff56082b6e4ffa542af0335e7e4204ea9eed4f8cc79a6d46fcfd5` | Naive Full Adapter + LambdaMART Borda smoke control |
| Runner | `experiments\hybrid\run_h0_proper_agent_adapter_frozen_lambdamart_v1.py` | `efa062226687c29a5f140e2387604fc034728bb763fdb15f40fe5812ca746e7e` | Proper Full Adapter + frozen LambdaMART H0 |
| Runner | `experiments\hybrid\run_initial_h0_adapter_ilt004d_borda_v1.py` | `347fd79bb49b763ee35df222c6a57d43779052294a54ad466f8480785cfb8843` | Initial Adapter + ILT-004 D Borda/complementarity diagnostic |
| Runner | `experiments\hybrid\run_full_em_to_initial_zero_shot_v1.py` | `d2ac3f3e6b575200c835c1ce00c614aa43ea6ab0fd5d068394db45d72426527b` | Frozen Full LambdaMART policy -> Initial zero-shot |
| Runner | `experiments\hybrid\run_ilt004d_to_full_zero_shot_v1.py` | `5d8ffd35ab7f3474caf40284b11a4adfdc98d2b96a7d0648ce577ce4d916b641` | Frozen ILT-004 D policy -> Full zero-shot |
| Result | `results\personalisation\hybrid_zero_shot_proper_v1\result_agent_full.json` | `8e08eca9b50e654667756527f7314d33b2c005569263244c6a5f09394cde60b1` | Proper Full Hybrid H0 Agent full Train-Val summary |
| Result | `results\personalisation\full_em_to_initial_zero_shot_v1\result.json` | `d14e7e643b1dc9ebbefe91718cf8b03959c10abd9b393e4764161a1b14e96c04` | Full-policy -> Initial zero-shot summary |
| Result | `results\personalisation\ilt004d_to_full_zero_shot_v1\result.json` | `d3ef0da464b29da678a98b6fd8faa6a5e7bbcb170d9d4e6d74371ff54e9b4f62` | ILT-004 D -> Full zero-shot summary |

## 2. Frozen external dependencies

| Kind | Local path | SHA256 | Role |
|---|---|---|---|
| Frozen dependency | `C:\Users\chiar\Desktop\LBH\thesis-model-level\local_artifacts\adapters\static\agent_full55926.safetensors` | `266e17099f64a6ea99306d89518e3fa30652135cece1e7819f008295395b1ac1` | Formal Agent Phage model-level Adapter |
| Frozen dependency | `C:\Users\chiar\Desktop\LBH\thesis-external-memory-next\results\personalisation\external_memory_next\lambdamart_fusion_v1\models\d5_m500_r100.txt` | `406b1693e5b8bb10b0af92c6bb31f494f8a78a13590d47ec5bf138fdba18df4e` | Native Full 25-feature LambdaMART |
| Frozen dependency | `C:\Users\chiar\Desktop\LBH\thesis-learned-fusion-lab\results\personalisation\initial_learned_fusion_transfer\ilt004_task_fusion_v1\generic_task_shape_model.txt` | `f354772ac0d0dd58ea4599a33d526e3df3a0fd36691e04790d07f83d62702ebc` | Frozen ILT-004 D generic_task_shape model |
| Frozen dependency | `C:\Users\chiar\Desktop\LBH\thesis-learned-fusion-lab\results\personalisation\learned_fusion_lab\lf005_task_shape_fusion_v1\features\val_shape_X.npy` | `d6ee4bd9006683cc3b0a3360fc68425c2e2969f5499e9199f6908c46553281f6` | Full 16-dimensional Task retrieval-shape matrix |
| Frozen dependency | `C:\Users\chiar\Desktop\LBH\thesis-learned-fusion-lab\results\personalisation\initial_learned_fusion_transfer\ilt001_matrix_v1\val_X.npy` | `579835b94e768aa2229d88ab55c43b8523e735ecb9e6d68bfe6c341955ec4669` | Initial base Train-Val matrix |
| Frozen dependency | `C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_stage1_features.jsonl` | `e311261cb0c5ea252ce593fdaa43ff87826d19eef440f39e2efc67ddc4310b13` | Frozen Full Stage-1 feature artifact |
| Frozen dependency | `C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_stage2_supports.jsonl` | `d413d02650a759c0a759e1845212e68b2d948c1a1d20fc88abfd89ea7973bc64` | Frozen Full Stage-2 memory-support artifact |

## 3. Frozen dataset / checkpoint provenance

| Artifact | SHA256 |
|---|---|
| Clean3 Train-Fit | `547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6` |
| Clean3 Train-Val | `d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220` |
| PinyinGPT2-Concat base checkpoint | `1c5ebb9e7b15d75ea8899b914fc8363f4745703115253071f7834780263c74bb` |
| Task-specific BiEncoder final checkpoint | `f9b87af11fcff692ad7c25fb6330f44f9f23ffedb480af9aec36af0e7cd08a8e` |
| Frozen Full Stage-1 Train-Val artifact | `e311261cb0c5ea252ce593fdaa43ff87826d19eef440f39e2efc67ddc4310b13` |
| Frozen Full Stage-2 Train-Val artifact | `d413d02650a759c0a759e1845212e68b2d948c1a1d20fc88abfd89ea7973bc64` |
| Frozen Full predictions artifact | `f3e902e5a9e7d25e62799b9abb719026c336381eacc42999d1e7edccf2731b22` |
| Initial ILT-001 Train-Val matrix | `579835b94e768aa2229d88ab55c43b8523e735ecb9e6d68bfe6c341955ec4669` |
| Initial ILT-004 D model | `f354772ac0d0dd58ea4599a33d526e3df3a0fd36691e04790d07f83d62702ebc` |
| Full Task-shape Train-Val matrix | `d6ee4bd9006683cc3b0a3360fc68425c2e2969f5499e9199f6908c46553281f6` |

## 4. Full proper Hybrid H0 protocol

- Population: Agent Phage Train-Val, `n=13,741`.
- Candidate surface: frozen Full External Memory Stage-1 surface.
- Adapter rescoring: formal `agent_full55926` Adapter.
- Memory evidence: frozen.
- LambdaMART: frozen.
- Training during H0: none.
- Train-Val tuning during H0: none.
- Test used: false.
- Frozen LambdaMART reproduction gate must pass before Hybrid scoring.

## 5. Full -> Initial policy-transfer protocol

- Target candidate/evidence surface: Initial.
- Transferred object: frozen Full LambdaMART ranking policy.
- This is policy-level transfer, not end-to-end candidate-generation transfer.
- Training: none.
- Tuning: none.
- Test used: false.

## 6. ILT-004 D -> Full policy-transfer protocol

- Target candidate/evidence surface: Full.
- Transferred object: frozen Initial-specialised ILT-004 D ranking policy.
- Initial semantic base features: 27.
- Compact interactions: 14.
- Task retrieval-shape features: 16.
- Final input dimensionality: 57.
- Full candidate rows: 333,099.
- Full Train-Val queries: 34,416.
- Non-empty LightGBM groups: 34,414.
- Legitimate zero-candidate queries retained in evaluation: 2.
- Personal-recovery candidates reconstructed: 6,736.
- Maximum history-count reconstruction floating-point error: `3.552713678800501e-15`.
- Training: none.
- Tuning: none.
- Test used: false.

### Initial feature semantics that must not be silently replaced by Full semantics

- Gap features: `max - candidate`.
- Frozen linear score: `base + 4 * ngram + 6 * bge`.
- Personal frequency support uses normalized `log1p(count)` within personal K5.
- `query_ambiguous` is explicit.
- `frozen_rank` is explicit.
- Initial Stage-1 personal-recovery weighting differs from Full.

## 7. Key result checks

| Experiment | Reference Top-1 | Transferred / Hybrid Top-1 | Key transition |
|---|---:|---:|---|
| Full proper Hybrid H0, Agent | Adapter surface 0.933047 | Hybrid 0.924751 | Adapter -> Hybrid rescue 222 / harm 336 / net -114 |
| Full policy -> Initial, Agent | Initial ILT-004 D 0.523979 | Full policy 0.489338 | Same Missing@10 = 0.171312 |
| ILT-004 D -> Full, Agent | Full LambdaMART 0.890401 | ILT transfer 0.893821 | rescue 110 / harm 63 / net +47 |

## 8. Generated prediction artifacts

- `results/personalisation/full_em_to_initial_zero_shot_v1/predictions.jsonl`
- `results/personalisation/ilt004d_to_full_zero_shot_v1/predictions.jsonl`

These are reproducible derivatives and are not staged by this compact provenance commit.

## 9. Closed-data guard

- `used_dev3000 = false` unless explicitly documented otherwise;
- `used_test = false`;
- no Test-path access during model selection;
- no Train-Val threshold sweep used as final model selection.

The final Test evaluation must occur only after the comprehensive model architecture and arbitration policy have been frozen.
