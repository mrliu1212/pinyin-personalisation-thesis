# Adapter V1 Full Agent Phage Train-Val Confirmation

Date: 2026-08-23

## Purpose

This experiment confirms the LR=5e-4 Model-Level Adapter on the complete
held-out Agent Phage Train-Val population after the earlier 2,048-row
learning-rate calibration.

No Test rows were used.

## Evaluation Protocol

- Author: Agent Phage
- Train-Val rows: 13,741
- Pinyin condition: Full
- Target type: Short
- Beam size: 16
- Top-K: 10
- Adapter LR: 5e-4
- Adapter architecture: frozen Adapter V1
- Base PinyinGPT2-Concat: frozen

Row manifest SHA256:

`bc101693e085576b26c92266c3b8c00e2cf608a46408b120b350157553821b65`

## Full Train-Val Results

### Existing Generic baseline

- N: 13,741
- Top1: 0.8199548796
- Top3: 0.9402518012
- Top5: 0.9581544284
- Top10: 0.9705989375
- MRR@10: 0.8820930699
- Missing@10: 0.0294010625

### Adapter LR=5e-4

- N: 13,741
- Top1: 0.935812532
- Top3: 0.982897897
- Top5: 0.987337166
- Top10: 0.991339786
- MRR@10: 0.959568716
- Missing@10: 0.008660214

## Delta Versus Generic

- Top1: +0.1158576524
- Top3: +0.0426460958
- Top5: +0.0291827376
- Top10: +0.0207408485
- MRR@10: +0.0774756461
- Missing@10: -0.0207408485

In percentage-point terms:

- Top1: +11.586 pp
- Top3: +4.265 pp
- Top5: +2.918 pp
- Top10: +2.074 pp
- Missing@10: -2.074 pp

## Comparison With 2,048-Row LR Calibration

Earlier 2,048-row held-out calibration:

- Generic Top1: 0.833984
- LR=5e-4 Top1: 0.938965
- Top1 gain: approximately +10.50 pp
- Generic MRR: 0.892432
- LR=5e-4 MRR: 0.960669

The complete 13,741-row confirmation therefore preserves the same qualitative
conclusion and shows an even slightly larger Top1 gain over Generic.

The earlier 2,048-row subset was somewhat easier in absolute Generic accuracy,
but the Adapter gain was not an artefact of that subset.

## Decision

LR=5e-4 is frozen as the Adapter V1 learning rate.

No full Train-Val rerun is required for LR=5e-5, 1e-4, or 2e-4 unless a later
critical implementation issue invalidates the current experiment.

The next planned experiment is the Agent Phage training-data learning curve:

- 2,048 Train-Fit rows
- 8,192 Train-Fit rows
- 32,768 Train-Fit rows
- 55,926 Train-Fit rows

All models should be compared on the same frozen 2,048-row Agent Phage
Train-Val subset.

## Scope

This is a development result, not final Test inference.

Train-Val remains held out from Adapter parameter updates.

The untouched Test partition remains reserved for the final frozen comparison.
