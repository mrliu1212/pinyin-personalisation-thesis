# 13 - LambdaMART Fusion Results

Date: 2026-08-22

Status: **POSITIVE TRAIN-VAL RESULT / NO HOLDOUT CLAIM**

## 1. What was done

LightGBM 4.7.0 fitted the predeclared additive-stump control and twelve
nonlinear LambdaRank configurations on the audited causal Train-Fit groups.
Selection used Train-Val Macro-author Top1 and the exact tie-break frozen in
record 11. The candidate surface and all upstream evidence were fixed.

## 2. Selected configuration

```text
max_depth=5
num_leaves=31
min_data_in_leaf=500
rounds=100
learning_rate=0.05
seed=1729
```

The selected point is at the maximum predeclared depth and round count. This is
a boundary limitation, not authorization for a post-result grid extension.

## 3. Main result

| Method | Macro Top1 | Micro Top1 | Top3 | Top5 | MRR@10 | Missing@10 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen Full RetunedFinal | .796004927 | .824994189 | .912075779 | .930758949 | .871377873 | .051981636 |
| Smoothing alpha 128 | .796515499 | .825459089 | .912337285 | .930729893 | .871652086 | .051981636 |
| Additive stumps | .792899650 | .822233845 | .911756160 | .930613668 | .869833741 | .051981636 |
| **LambdaMART** | **.798839063** | **.827783589** | .911988610 | .930613668 | **.873043096** | .051981636 |

Against frozen RetunedFinal, LambdaMART changes Macro Top1 by +.002834137
(+0.2834 percentage points), Micro Top1 by +.002789400, and MRR by
+.001665223. Top3 and Top5 change by -.000087169 and -.000145281,
respectively; they are slightly lower rather than improved. Missing is fixed
by the unchanged candidate surface.

Paired Top1 transitions:

| Reference | Rescue | Harm | Net |
|---|---:|---:|---:|
| Frozen RetunedFinal | 267 | 171 | +96 |
| Smoothing alpha 128 | 252 | 172 | +80 |
| Additive stumps | 358 | 167 | +191 |

## 4. Breakdowns

Per-author Top1 is .890400990 for Agent Phage, .604732254 for Etinjat, and
.901383946 for breaddddd. Relative to frozen RetunedFinal, all three improve
by +.003129321, +.003237858, and +.002135231, respectively.

| Subset | Frozen Macro Top1 | LambdaMART Macro Top1 | Delta |
|---|---:|---:|---:|
| Ambiguous, n=10,053 | .803894152 | .812689892 | +.008795740 |
| Conflict, n=1,865 | .252588201 | .296546452 | +.043958250 |

The formal Conflict rule remains unchanged: ambiguous, unique frequency
winner, gold differs from that winner, frequency ties excluded.

## 5. Interpretation and feature contributions

The nonlinear interaction hypothesis is supported on Train-Val. The additive
stump control is below the frozen formula, while the deeper model improves all
authors and has its largest relative gain on Conflict. This is consistent with
conditional arbitration among Generic, personal-frequency, history depth, and
contextual evidence rather than a better global linear coefficient.

The five largest descriptive mean-absolute TreeSHAP contributions are:

| Feature | Mean abs. contribution |
|---|---:|
| frozen linear score | .770206 |
| Stage-1 gap to top | .352305 |
| Stage-1 base score | .234409 |
| log same-Pinyin history | .194038 |
| normalized Generic score | .173442 |

NGram effective order/support and BGE gaps/support also contribute, but below
the main frozen-score and confidence features. TreeSHAP here explains this
fitted development model; it is not causal attribution.

## 6. Limitations and next decision

- The result is selected and reported on repeatedly observed Train-Val.
- No paired significance test was predeclared; no significance claim is made.
- The selected complexity lies on two grid boundaries.
- Top3/Top5 are marginally lower.
- Dev3000 and Test were not used, so no holdout-generalisation claim is made.

A smoothing-plus-tree experiment is not justified now. Record 10 found that
the prior-specific component of smoothing was weak, while this model already
receives raw Choice Share, history depth, concentration, and contextual
supports separately. Adding the selected smoother would entangle two
independently understood changes for little mechanistic motivation.

Proceed to the task-specific bi-encoder design/cost gate. The current result
also raises the bar for that proposal: a new representation should show
retrieval-level headroom that is likely to survive candidate-level fusion.

## 7. Reproduction and hashes

```powershell
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$next = '.\results\personalisation\external_memory_next'

& $python -m experiments.external_memory_next.run_lambdamart_fusion_v1 `
  --matrix-root "$next\lambdamart_matrices_v1" `
  --audit "$next\learned_fusion_input_audit_v1\audit.json" `
  --deps-root '.\.build\external_memory_next_deps' `
  --smoothing-predictions "$next\choice_share_smoothing_fixed_surface_boundary_v2\selected_predictions.jsonl" `
  --output-root "$next\lambdamart_fusion_v1"
```

- Runner: `e96a1013deca6e5fb34f693485ec2dec86057b10cd29d7fe5bb15c40424a776b`.
- LightGBM: `4.7.0`.
- Matrix manifest: `61f722af39e873f3015fc661cc99f34d1aa452c6d0fd734d753da556da83d627`.
- Selected model: `406b1693e5b8bb10b0af92c6bb31f494f8a78a13590d47ec5bf138fdba18df4e`.
- `result.json`: `0a6481eac497b4a63c1e48867227b175bb3fd14f64d0787b32ca94f9d95d073e`.
- `selected_predictions.jsonl`: `a7d98b5618cdd9f7a60705cd7d07da850a79f41f4985fe1d2a152cac07cff4ae`.
- `used_dev3000=false`; `used_test=false`.

## 8. Validation

A second bounded invocation reused the additive control and all twelve
nonlinear models; no model was retrained and the selected result was unchanged.
The final focused suite passed `22/22`, all new Python files compiled, and
tracked `git diff --check` returned exit 0 (with only Git's existing LF/CRLF
working-copy notices for the two living indexes).
