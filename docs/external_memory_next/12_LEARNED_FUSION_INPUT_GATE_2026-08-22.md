# 12 - Learned-Fusion Input Gate

Date: 2026-08-22

Status: **COMPLETE / CLEARED FOR PREDECLARED LAMBDAMART GRID**

## 1. What was done and why

The missing causal Clean3 Train-Fit Generic surface was generated with the
frozen PinyinGPT configuration. The existing Full RetunedFinal Stage-1 and
NGram/BGE support semantics were then applied to Train-Fit, audited against
the frozen Train-Val artifacts, and materialized as compact ranking matrices.
This gate prevents a learned result from hiding a changed candidate surface,
feature leak, group-policy change, or baseline mismatch.

## 2. Inputs and frozen behavior

- Train-Fit: 144,526 rows, SHA256
  `547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6`.
- Generic: PinyinGPT revision `76dd20dc92d8236a350fb732e99dde6fa15e2263`,
  official code revision `8f1573ed0bd4d1f3d8d3f10a05f7e870725646f1`,
  beam 16, Top10, long-context frozen semantics.
- Stage-1: `2P + 6CS + 4E`; Stage-2: `6NG + 6BGE`.
- History: same author, strictly prior, latest H5000 raw interactions, then
  exact segmented-Pinyin filtering.
- BGE: frozen `bge-small-zh-v1.5-q8_0.gguf`, last 64 characters, with the
  existing 42,278-row Train-Val cache used only as an exact-key seed.
- Dev3000 and Test were not read.

## 3. Generation and engineering result

Generic completed 144,526/144,526 rows in 4,756.83 seconds at 30.44 rows/s.
The final prediction SHA256 is
`0cbccbde87cfb03d415cb694fa42ea898716abec386beaf5a9a34eee1364f49c`.
The validated runtime was CUDA 12.8 / PyTorch 2.11.0+cu128 on the NVIDIA RTX
4060 Laptop GPU.

The first Stage-1 attempt exposed an empty-Generic edge case in the reused
Frequency helper. The frozen Full policy already specifies a conservative
no-op for this case. The new preparation runner was corrected to return an
empty Frequency surface before normalization when Generic is empty. It does
not invent, pad, or inject a candidate. Five Train-Fit groups use this policy.
The focused regression tests cover both the empty no-op and unchanged
delegation for non-empty surfaces.

Stage-1 then completed 144,526 rows. BGE required 143,891 exact context keys;
135,863 were generated and the local cache ended with 178,141 rows including
seed entries. The final support table SHA256 is
`51a24835354c1732406c779af48f53dcf33f9e195a101fcb22f962f27a3db4be`.

## 4. Feature and group audit

The 25 features are exactly those frozen in record 11. Gold correctness is a
separate label; author identity and post-hoc correctness categories are absent
from `X`.

| Population | Groups | Candidates | Positive groups | Zero-positive | Empty |
|---|---:|---:|---:|---:|---:|
| Train-Fit | 144,526 | 1,398,606 | 139,163 | 5,363 | 5 |
| Train-Val | 34,416 | 333,099 | 32,627 | 1,789 | 2 |

The 139,163 positive Train-Fit groups, containing 1,345,162 candidates, enter
LambdaRank. Zero-positive Fit groups are excluded because they carry no
within-query ordering label. All 34,416 Train-Val groups remain in metrics.
The frozen Train-Val candidate order/rank was reconstructed exactly for every
row before this gate passed.

NGram and BGE aggregate means are equal because both supports normalize to
unit mass on supported queries; they are not aliases. A 20,000-group raw check
found 11,081 candidate values where the two supports differ.

## 5. Reproduction

Use the Generic, Stage-1, and support commands in
`docs/REPRODUCIBILITY_INDEX.md`, then run:

```powershell
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$compare = 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation'
$next = '.\results\personalisation\external_memory_next'

& $python -m experiments.external_memory_next.audit_learned_fusion_inputs_v1 `
  --fit-supports "$next\train_fit_ranking_features_v1\train_fit_candidate_supports.jsonl" `
  --val-stage1 "$compare\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_stage1_features.jsonl" `
  --val-stage2 "$compare\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_stage2_supports.jsonl" `
  --val-predictions "$compare\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_selected_predictions.jsonl" `
  --output-root "$next\learned_fusion_input_audit_v1"

& $python -m experiments.external_memory_next.prepare_lambdamart_matrices_v1 `
  --audit "$next\learned_fusion_input_audit_v1\audit.json" `
  --fit-supports "$next\train_fit_ranking_features_v1\train_fit_candidate_supports.jsonl" `
  --val-stage1 "$compare\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_stage1_features.jsonl" `
  --val-stage2 "$compare\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_stage2_supports.jsonl" `
  --val-predictions "$compare\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_selected_predictions.jsonl" `
  --output-root "$next\lambdamart_matrices_v1"
```

## 6. Hashes and decision

- Generic runner: `c5c07bbcd4a890a4bcb40aaa261aeb87d8bcec07b3aa6c40dde7afc3bd28cc6b`.
- Feature runner: `c9fbca2b5e60cad46b3e4d6dd7a012be59fdbccc7fa4598ee914e5802eaba071`.
- Audit runner: `7da06c20b06b3cd8e56f52cfab8369a7c071cc65b7426ec62458dc061bf3b2b8`.
- Matrix runner: `1820c8e27a91a784e3d4ab0a82e9d2ef4a1c0a86193023185deb4a99c8eace74`.
- `audit.json`: `fe7a506042aa9bd00eaa9ba4487bfff39487b4dab2a9ea6cfa43b31af83260cd`.
- `matrix_manifest.json`: `61f722af39e873f3015fc661cc99f34d1aa452c6d0fd734d753da556da83d627`.
- `fit_X.npy`: `6eee8948c7dbe85877a59b95096beb1047866ede6ab42bb5f2e79125937f580a`.
- `val_X.npy`: `4ef627b0ac45478992aab2217a0bf38a7e06022152a98416f93c199b7dfc3fc3`.
- `used_dev3000=false`; `used_test=false`.

All declared gates passed. The predeclared additive-stump control and 12-point
nonlinear grid in record 11 may now run without changing their design.
