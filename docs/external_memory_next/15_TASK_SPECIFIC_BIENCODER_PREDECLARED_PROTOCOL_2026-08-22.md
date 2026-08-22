# 15 - Task-Specific Bi-Encoder Predeclared Protocol

Date: 2026-08-22

Status: **FROZEN BEFORE TRAINING / EXECUTED WITHOUT EXTENSION**

## 1. Scope

This executes the representation experiment designed in record 14. It is a
new named experiment and does not modify Generic PinyinGPT, the fixed
RetunedFinal candidate surface, NGramRecency, the frozen linear coefficients,
or the completed LambdaMART result. Dev3000 and Test are not accepted by the
new runners.

## 2. Frozen training data

Use the existing audited Clean3 Train-Fit EM3 pair registry only:

```text
same author
-> strictly prior
-> latest H5000 raw interactions before Pinyin filtering
-> exact segmented-Pinyin
-> matching-target positive / query-local wrong-target negatives
```

Each `(query_row_id, round)` is one sampled group with exactly one positive
and up to three wrong-target negatives. No unrestricted in-batch negatives are
used. The preparation audit must reconstruct every row against the frozen
Train-Fit manifest and stop on a non-prior, cross-author, cross-Pinyin,
mislabelled, duplicate, or malformed pair.

The first full audit, performed before any neural training, exposed a necessary
clarification to the cost gate: 32,999 of the 99,671 sampled positive rounds
have no remaining negative after without-replacement sampling. Their
single-class cross-entropy is identically zero, so they are retained in the
population audit but excluded from optimization. The frozen trainable surface
is therefore 66,672 groups: 11,845 with one negative, 6,926 with two, and
47,901 with three. No negatives are resampled, padded, or borrowed from other
queries.

## 3. Encoder and serialization

- Base checkpoint: full-precision `BAAI/bge-small-zh-v1.5` at Hugging Face
  revision `7999e1d3359715c523056ef9478215996d62a620`.
- Shared query/history encoder weights.
- Input is only the most recent 64 Unicode code points of context.
- Mean pooling over non-padding tokens followed by L2 normalization, matching
  the repository's existing generic-BGE pooling semantics.
- Tokenizer maximum length 128 with left truncation as a safety boundary after
  the last-64-character ablation.
- No author, Pinyin, target, Gold, correctness, future-row, rescue/harm, or
  candidate text is serialized into the encoder.

Historical target remains external metadata used only to build supervision
and aggregate legal runtime evidence.

## 4. Inner Train-Fit gate and fixed schedule

Within each author, eligible query IDs are ordered by
`(chronological_position, row_id)`. Whole chronological-position blocks in
the earliest approximately 90% form `inner_fit`; the remaining latest blocks
form `inner_gate`. Equal-position queries cannot cross the boundary. All
rounds of one query remain in the same split.

There is one optimizer configuration and only two predeclared checkpoint
choices:

```text
seed                 = 1729
optimizer            = AdamW
learning_rate        = 2e-5
weight_decay         = 0.01
warmup               = 10% of update steps
schedule             = linear decay
gradient_clip_norm   = 1.0
temperature          = 0.05
physical_batch       = 16 query-local groups
gradient_accumulation= 2
effective_batch      = 32 groups
precision            = FP16 on CUDA
epochs considered    = {1, 2}
```

Select the epoch checkpoint by inner-gate Macro-author Recall@1, then Micro
Recall@1, MRR, then the earlier epoch. Do not extend the epoch boundary or
change the optimizer after observing the gate. After selection, reinitialize
from the pinned base and refit on all 66,672 trainable Train-Fit groups for
exactly the selected epoch count. That final checkpoint is frozen before
Train-Val is read.

The loss is temperature-scaled listwise cross-entropy within each query-local
group. The positive history is the sole correct class. Different groups do not
act as negatives for one another.

## 5. Mandatory pre-full-run checks

Before the full fit:

1. verify registry counts, chronology, labels, and split isolation;
2. hash the model/tokenizer snapshot and record library/CUDA provenance;
3. train a bounded eight-group smoke artifact;
4. save/reload it and require embedding agreement within `1e-6`;
5. record `used_dev3000=false` and `used_test=false`.

Smoke outputs and checkpoints are engineering artifacts and cannot be used for
scientific selection.

## 6. Frozen Train-Val evaluation

The final checkpoint receives one Train-Val evaluation. Intrinsic comparison
uses identical deployed, candidate-conditioned history populations for the
generic BGE and task-specific encoders: strictly legal same-Pinyin histories
whose target is on the frozen RetunedFinal Stage-1 Top10 surface. Eligible
intrinsic queries contain at least one Gold-target and at least one wrong-target
history. Report Recall@1/5/10, MRR@10, and target-support Top1 diagnostics.

The primary end-to-end ablation substitutes only the representation in the
existing BGERecency aggregation:

```text
last64 contexts
candidate-conditioned histories
cosine Top5 per candidate
max(0, cosine) * exp(-age / 2048)
candidate-wise normalization
S_final = frozen RetunedStage1 + 6*NGramRecency + 6*task-specific support
```

Candidate order, tie-breaking, Missing@10, and all other evaluation semantics
remain unchanged. Compare the frozen generic-BGE RetunedFinal route, the new
fixed-fusion route, and the already completed LambdaMART result.

A task-specific LambdaMART refit is authorized only if both intrinsic
Macro-author Recall@1 and fixed-fusion Macro-author Top1 strictly exceed their
generic-BGE counterparts. Otherwise record the representation-only result and
stop; do not alter the protocol to obtain a win.

Selection and evaluation use Train-Val only. `used_dev3000=false` and
`used_test=false` are mandatory in every generated manifest.
