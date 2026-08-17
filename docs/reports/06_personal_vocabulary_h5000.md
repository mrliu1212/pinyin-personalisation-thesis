# Personal Vocabulary H5000

> **PENDING FINAL RUN**

## Purpose

This experiment separates candidate availability from candidate ranking. PV0
measures whether prior same-user vocabulary can supply Generic-missing targets.
PV1 injects bounded frequency-ranked personal candidates. PV2 tests whether
reused M1 BGE context support improves those personal-only candidates.

## Frozen Design

- exact 6,000 T1 Full+Short Test anchors;
- H5000 before exact segmented-Pinyin filtering;
- frozen T1 Generic predictions and zero Test inference;
- PV1 grid: Kpv `{1, 3, 5}`, lambda `{0.5, 1, 2, 4}`;
- PV2 grid: context lambda `{0.5, 1, 2, 4}` after freezing PV1;
- frozen M1 Top-N = 5;
- Dev Macro-author Top-1 selection only;
- no M2 Cross-Encoder and no neural training.

Full method details are in [Bounded Personal Vocabulary H5000](../research/personal_vocabulary.md).

## PV0 Recoverability

**PENDING FINAL RUN**

Report Generic Missing = 538, recoverable/unrecoverable counts, rate,
per-author values, occurrence distribution, and lexicon-size statistics.

## PV1 Selection and Test Result

**PENDING FINAL RUN**

Report selected Kpv/lambda, standard metrics, recovery at Top-10/Top-3/Top-1,
missing and recoverable recovery rates, F→PV1 helped/harmed accounting,
net help, covered-case harm, and per-author values.

## PV2 Selection and Test Result

**PENDING FINAL RUN**

Report selected context lambda, standard metrics, recovery metrics, PV1→PV2
context helped/harmed accounting, net context help, and per-author values.

## Full Comparison

**PENDING FINAL RUN**

Compare G0, F-H5000, M1-H5000, M2-H5000, PV1-H5000, and PV2-H5000. PV0 is a
separate availability analysis and must not be presented as Top-1 ranking.

## Reuse and Provenance

Preparation found 39,680 BGE cache hits, zero misses, and zero new embeddings.
Test PinyinGPT inference is zero. T1/M1/M2 artifacts are hash-checked. Final
artifacts will be under `results/personalisation/personal_vocabulary_h5000/`.

## Interpretation and Limitations

**PENDING FINAL RUN**

Interpret candidate recovery separately from ranking effects. Limitations
include the fixed H5000 window, frequency bias, Kpv pruning, cold start, unseen
vocabulary, approximate boundary score, erroneous history, generic BGE context,
and unmodelled temporal drift.
