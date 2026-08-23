# Model-Level Adapter Overnight Ablation Protocol

Date: 2026-08-23

## Purpose

This protocol runs the remaining controlled Model-Level experiments required
before Hybrid integration.

No Test rows are used.

## Frozen Base Configuration

- PinyinGPT2-Concat base frozen
- Adapter V1 architecture
- bottleneck = 48
- LR = 5e-4
- epoch = 1
- batch size = 8
- seed = 20260822
- exact Concat target loss
- Beam16 / Top10 evaluation

## Agent Phage A/B/C

A:
- most recent 27,963 Train-Fit rows
- 100% Full-Pinyin training

B:
- all 55,926 Train-Fit rows
- 100% Full-Pinyin training

C:
- all 55,926 Train-Fit rows
- Full:Initial:Mixed = 3:1:2 training

All three are evaluated on the same frozen 1,000-row Agent Phage Dev under:

1. Initial Pinyin
2. Full Pinyin

Comparisons:

- A vs B: history amount under Full-only training
- B vs C: Full-only versus mixed-Pinyin training at equal history size
- A vs C: less but cleaner Full-only history versus more mixed-mode history

## Agent Phage Recent-History Experiment

Full-only Adapters are trained using:

- most recent 500 rows
- most recent 5,000 rows
- most recent 25,000 rows
- all 55,926 rows

The full-history model is B and is not retrained.

Recent history is defined using:

1. work_chronological_index
2. chronological_position
3. source_position_start
4. source_position_end

The subsets are nested:

500 subset 5,000 subset 25,000 subset Full.

Each model is evaluated on the same frozen Agent Phage Dev1000 under:

1. Initial Pinyin
2. Full Pinyin

## Cross-Author Full-History Models

Full-history, Full-only Adapters are trained for:

- Agent Phage: 55,926 rows (reuse B)
- Etinjat: 32,906 rows
- breaddddd: 55,694 rows

These three models are retained for later Hybrid experiments.

Each model is evaluated under:

1. Dev1000 Initial
2. Dev1000 Full
3. Full Train-Val Initial
4. Full Train-Val Full

The evaluation ordering intentionally runs Initial before Full.

## Fixed Dev

Because no pre-existing 1,000-row Dev manifest was present in the transferred
cluster worktree, this protocol freezes `fixed_dev1000_v1`.

For each author, exactly 1,000 rows are deterministically selected from the
held-out Train-Val population using seed 20260822.

The exact rows and hashes are stored under:

`results/model_level/fixed_dev1000_v1/`

## Scope

The overnight experiment does not:

- use Test;
- run Hybrid;
- retrain an Adapter for every evaluation condition;
- tune other authors independently.

Every distinct Adapter is trained once and reused for all of its evaluations.
