# Model-Level Adapter V1 — Hard-Gate Protocol

## Scope

This package implements the pre-training correctness gates for a per-user
serial bottleneck Adapter on the frozen PinyinGPT2-Concat backend. It does not
perform formal Adapter training, use Dev3000, or inspect Test.

## Frozen architecture

```text
12 frozen GPT2Blocks
-> one independent residual Adapter after each completed block output
-> frozen final LayerNorm
-> frozen tied LM head
```

Per-layer Adapter:

```text
h' = h + W_up(ReLU(W_down(h)))
hidden = 768
bottleneck = 48
reduction factor = 16
W_up and up bias = zero initialized
down bias = zero initialized
no extra LayerNorm / dropout / gate / learned scale
```

Expected trainable parameters per user: `894,528`.

The frozen base, including its configured dropout, remains in evaluation mode
during Adapter training. Only Adapter parameters require gradients.

## Pinyin manifestations

Each underlying Full+Short Train-Fit row appears once per epoch.

```text
Full : Initial : Mixed = 3 : 1 : 2
```

The selection is a deterministic SHA256 function of seed, epoch, and row ID.
Initial uses the official first-letter representation (`shi -> s`, `zhong -> z`,
`chi -> c`). For multi-syllable Mixed rows, the mask is uniform over every
non-degenerate Full/Initial binary mask. A requested one-syllable Mixed row has
a recorded deterministic 50/50 Full/Initial fallback.

## Exact Concat loss

For Gold target `c1...cm`, the causal input is:

```text
[CLS] context [SEP] p1...pm [SEP] c1...c(m-1)
```

Supervision is applied directly at:

```text
final [SEP] -> c1
c1          -> c2
...
c(m-1)      -> cm
```

This reproduces `score_candidates()` while retaining the exact standardized
Generic recent-context budget:

```text
n_positions - (2 + 2*m)
```

## Hard gates

The runner must pass, in this order:

1. Train-Fit SHA and population audit.
2. Zero-init exact Top-10 text/order identity and score tolerance.
3. Target-only loss regression against `score_candidates()`.
4. Adapter-only parameter/gradient audit.
5. Finite decreasing short overfit loss.
6. Adapter-only safetensors save/reload prediction identity.
7. On Windows, write a frozen Generic platform reference; on Linux, require
   exact Top-10 text/order equivalence against it with documented score tolerance.

The smoke Adapter is not a formal checkpoint.

## Frozen input

```text
Train-Fit rows = 144,526
SHA256 = 547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6

Agent Phage = 55,926
Etinjat = 32,906
breaddddd = 55,694
```

The first pilot author is Agent Phage because it has the largest frozen
Train-Fit population.

## Windows command

Run from `C:\Users\chiar\Desktop\LBH\thesis-model-level` after copying the
package files into their matching repository paths:

```powershell
$py = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$checkpoint = 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat'
$fit = 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_v2\clean3_train_fit_v1.jsonl'

& $py -m experiments.model_level.run_adapter_hard_gates_v1 `
    --checkpoint $checkpoint `
    --fit $fit `
    --output-root '.\results\model_level\adapter_v1_hard_gates' `
    --author 'Agent Phage' `
    --device cuda
```

Generated results and smoke weights remain local-only. Do not stage them.
