# External Memory Completion Plan — 2026-08-19

## Scope

Development scope:
- Full+Short
- H5000
- Etinjat, Re_spectators, breaddddd
- Frozen Generic PinyinGPT
- No Test-based hyperparameter selection

The purpose of this stage is to complete the non-parametric External Memory system before starting model-level Adapter/LoRA personalisation.

## EM-1 — Candidate Recovery + Reranking

Primary question:

Can personal candidate recovery and personal reranking work together better than either mechanism alone?

Compare:

- G0: Generic only
- F: Frequency reranking only
- R: Candidate recovery only
- R+F: Recovery + Frequency reranking

Method semantics:

- G0:
  frozen Generic Top-10 and original Generic ranking.

- F:
  existing Frequency method unchanged; rerank only the frozen Generic Top-10.
  No candidate recovery.

- R:
  recover personal-only candidates, obtain frozen-PinyinGPT fixed-candidate
  scores, merge them with the Generic pool, and rank using Generic-model
  evidence only. No Frequency reranking.

- R+F:
  use the same recovered unified pool as R, then add the personal Frequency
  signal for final reranking.

Architecture:

Frozen Generic Top-10
+ strictly-prior personal vocabulary
-> recover bounded Pinyin-compatible personal-only candidates
-> score recovered candidates with the SAME frozen PinyinGPT using
   fixed-candidate / teacher-forced autoregressive scoring under the
   current interaction's context + Pinyin
-> merge Generic and recovered candidates into a unified candidate pool
-> optionally apply Frequency personal support
-> final Top-K

Important scoring detail:

A recovered candidate being absent from Generic Top-10 does not mean that
PinyinGPT assigns it zero probability. The Generic Top-10 is produced by
finite beam search, so a valid candidate may have been pruned during search.

For a recovered candidate, fixed-candidate scoring forces that candidate path
under the same frozen PinyinGPT prompt and sums the autoregressive per-character
log probabilities. This provides a Generic-model score for the recovered path.

Before using these scores in the unified pool, run a compatibility audit:
re-score candidates already present in the frozen Generic Top-10 using the
fixed-candidate scorer and compare them against their cached Generic
log-probabilities.

If the two scoring paths are not numerically compatible within an explicitly
validated tolerance, STOP and do not mix their scores until the discrepancy is
understood.

This exact scoring is a NEW EM-1 design. It must not be described as the
scoring method used by the previous PV1 experiment, which used an approximate
Generic boundary score for personal-only candidates.

Report separately:

- Macro-author Top-1
- Micro Top-1
- Top-3
- MRR@10
- Generic Missing@10
- recovered-to-pool count/rate
- final Missing@10
- recovered-to-Top1 conversion
- Rescue
- Harm
- History Available
- Ambiguous
- Conflict

Recovery budget and ranking weights must be selected on Dev only.

The three-author scope in this plan is a development/method-selection stage,
not the final six-author formal thesis evaluation.

This experiment must distinguish:

1. candidate coverage improvement;
2. conversion of recovered candidates into ranking improvement.

## EM-2 — PinyinGPT Hidden-State kNN

Primary question:

Does a task-native PinyinGPT hidden representation retrieve more useful same-user, same-Pinyin history than generic BGE embeddings?

Comparison:

- BGE representation + cosine
- PinyinGPT hidden representation + cosine

Keep all other retrieval semantics fixed.

Primary diagnostic:
- Macro-author Ambiguous R@1

Secondary diagnostics:
- Overall R@1
- Conflict R@1
- R@5
- R@10

First evaluate retrieval quality.
Only run end-to-end reranking if the task-native representation provides useful retrieval evidence.

## EM-3 — IME-Specific Cross-Encoder

Primary question:

Can a supervised task-specific matcher learn whether a historical interaction is useful for the current Pinyin candidate decision better than generic semantic similarity?

Input:

Current:
- preceding context
- current Pinyin

History:
- historical context
- historical target

Training labels:

- positive: historical target == current Gold
- negative: same-Pinyin historical target != current Gold

The model must not receive author identity as an input feature.

All training/evaluation history must respect strict chronological visibility.

Compare:

- BGE + cosine
- PinyinGPT hidden-state kNN
- IME-specific Cross-Encoder

Focus especially on:
- Ambiguous
- Conflict

## EM-4 — Final External Memory Fusion

After EM-1 to EM-3, combine only signals that are supported by Dev evidence.

Candidate final system:

Frozen Generic
+ Personal Vocabulary Recovery
+ Frequency Memory
+ best validated Context Memory signal
-> unified final reranking

Do not assume that a Context signal must be included.
If Recovery + Frequency remains strongest, that is a valid final External Memory system.

Freeze all settings before final formal evaluation.

## Deferred

Not part of this stage:

- Initial+Short structural expansion
- Adapter / LoRA
- Hypernetwork
- six-author script-normalised formal evaluation

