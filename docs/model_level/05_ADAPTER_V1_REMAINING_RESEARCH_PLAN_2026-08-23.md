# Model-Level Adapter V1 — Remaining Research Plan

**Date:** 2026-08-23
**Status:** Static qualification complete; longitudinal continual phase next.

---

## 1. Completed work

The following Model-Level Adapter work is complete:

- Adapter architecture and hard gates;
- deterministic GPU reproducibility;
- exact target-only Concat training loss;
- learning-rate calibration;
- LR `5e-4` freeze;
- full Agent Train-Val confirmation;
- tokenizer-compatible training-target gate;
- Full-only and mixed-Pinyin static training;
- Agent history-size experiments;
- three-author static Adapter training/evaluation;
- full Train-Val evaluation;
- unsupported-Pinyin denominator-preserving evaluation semantics;
- Oldest5K / Random5K / Recent5K matched-budget controls;
- Recent5K 625-versus-6991 optimization control;
- Recent5K-long versus Full55925 matched-step diversity control.

Static results are frozen in:

`docs/model_level/08_ADAPTER_STATIC_RESULTS_CHECKPOINT_2026-08-23.md`

---

## 2. Frozen current configuration

Architecture:

- frozen PinyinGPT2-Concat base;
- 12 serial residual Adapters;
- hidden 768;
- bottleneck 48;
- ReLU;
- 894,528 trainable parameters per user.

Current continual update baseline:

- Full-Pinyin training;
- Short targets;
- LR `5e-4`;
- batch size 8;
- one-pass 5K update block;
- 625 optimizer steps per complete 5K update block;
- seed `20260822`;
- deterministic training.

Master A demonstrated that repeated training of a fixed 5K population to 6,991
steps strongly overfits. The continual protocol should therefore avoid repeated
multi-epoch optimization of the same small episode by default.

---

## 3. Longitudinal development corpus

Current longitudinal research uses Agent Phage only.

Allowed corpus:

`Train-Fit + Train-Val`

Combined nominal rows:

`69,667`

Tokenizer-compatible effective rows:

`69,664`

Works:

`45`

Chronology:

- Train-Fit: work indices 0–30;
- Train-Val: work indices 31–44;
- combined indices: 0–44.

Test remains sealed.

Important semantic change:

Once Train-Val is incorporated into the longitudinal stream, it is part of the
development/training corpus for this research branch and is no longer an
untouched validation partition.

---

## 4. Next Stage A — Freeze episode/probe manifests

Create deterministic chronological manifests before continual training.

Working design:

`[Update 5K] [Future Probe 500]`

repeated for 12 cycles.

This gives:

- update rows: 60,000;
- probe rows: 6,000;
- remaining terminal future rows: 3,664.

Exact manifests must be persisted and hashed before model results are inspected.

Probe rows must never be used for training.

---

## 5. Next Stage B — Prequential test-before-train protocol

For episode `k`:

1. load the warm Adapter state from episode `k-1`;
2. evaluate Future Probe `k` before Update `k`;
3. train once on Update `k`;
4. evaluate Future Probe `k` again;
5. retain all prior probe manifests for later retention evaluation.

This preserves causal ordering:

past history -> model -> future probe -> later update.

No future row may influence an earlier model state.

---

## 6. Next Stage C — Warm continual Adapter

The central experiment is a single per-user Adapter updated sequentially through
the chronological stream.

Baseline update unit:

- 5,000 compatible chronological rows;
- batch 8;
- one pass;
- 625 optimizer steps.

The purpose is to measure whether a fixed-size model-level personalization state
can continuously absorb new user history.

---

## 7. Adaptation metrics

For every future probe, report at least:

- Top1;
- Top3;
- Top5;
- Top10;
- MRR@10;
- Missing@10.

For paired pre/post-update predictions also compute:

- rescue;
- harm;
- unchanged-correct;
- unchanged-wrong;
- net rescue.

The primary adaptation quantity is the change on the future probe caused by the
immediately preceding chronological update.

---

## 8. Retention and forgetting

Earlier probes are re-evaluated after later updates.

For probe `i` at later time `t`, define:

`Forget(i,t) = Acc(i,i) - Acc(i,t)`

where `Acc(i,i)` is the post-learning performance associated with the time the
probe was first incorporated into the learned history, and `Acc(i,t)` is its
later performance.

Report:

- per-probe forgetting;
- average forgetting;
- worst-probe forgetting;
- temporal forgetting curves.

Replay is not part of the baseline.

Replay is introduced only if warm continual training exhibits meaningful
forgetting.

---

## 9. Generic preservation / locality

Continual personalization should not be evaluated only on same-user future
accuracy.

A preservation/control analysis should measure whether sequential updates cause
unwanted collateral drift outside the recently learned preference region.

The exact control manifest must be frozen before the corresponding results are
examined.

---

## 10. Warm versus Rebuild control

Warm continual training may benefit from accumulated optimization trajectory,
not only accumulated information.

Therefore compare the warm Adapter against rebuild controls at selected
checkpoints.

Planned checkpoints:

- after approximately 20K update rows;
- after approximately 40K update rows;
- after approximately 60K update rows.

A rebuild model is trained from the zero-initialized Adapter state using the
allowed historical data available up to that point.

This control tests whether warm sequential updating provides a benefit or cost
relative to rebuilding from accumulated history.

---

## 11. Replay decision

Do not add replay pre-emptively.

Decision rule:

- if warm continual training shows little meaningful forgetting, retain the
  simpler no-replay system;
- if substantial forgetting is observed, introduce a minimal replay condition
  as a targeted control.

This keeps the research question focused.

---

## 12. Controlled Temporary versus Persistent stress test

Natural ABA/ABB temporal shifts were audited but are not sufficiently clean to
serve as preference-ground-truth events because semantic/context confounds
remain.

They are retained as a methodological negative result.

A later controlled mechanistic stress test will therefore use ambiguous Pinyin
with two real tokenizer-compatible candidates A/B.

Measure:

- candidate scores;
- candidate ranks;
- margin `Score(A) - Score(B)`;
- takeover under persistent B exposure;
- recovery after temporary B exposure;
- collateral drift on unrelated Pinyin.

Counterbalance:

- A -> B;
- B -> A.

This is a stress test of the existing warm Adapter, not a new architecture.

---

## 13. Explicitly outside the current research line

The following are not part of the remaining main pipeline unless reopened
explicitly:

- Adapter bottleneck 48 versus 96;
- alternative Adapter architecture search;
- LoRA comparison;
- controlled-write architecture;
- consolidation architecture;
- controlled-read architecture;
- external-memory Hybrid;
- long-term × short-term hybrid control;
- broad engineering benchmark phase;
- final Test evaluation.

The current research line ends after the longitudinal continual experiments and
controlled temporary/persistent stress test.

---

## 14. Remaining sequence

1. Freeze combined longitudinal corpus provenance.
2. Freeze episode/probe manifests.
3. Run prequential test-before-train baseline.
4. Run warm continual Adapter.
5. Measure adaptation / rescue / harm.
6. Measure retention / forgetting.
7. Measure generic preservation / locality.
8. Run Warm versus Rebuild controls.
9. Add replay only if warranted by forgetting.
10. Run Temporary versus Persistent controlled stress test.
11. Freeze final Model-Level conclusions.

Test remains sealed.
