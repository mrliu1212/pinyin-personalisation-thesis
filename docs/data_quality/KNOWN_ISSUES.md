# Known Data and Backend Quality Issues

Status: ACTIVE ISSUE REGISTRY

This document records known dataset, preprocessing, reconstruction, and
backend-compatibility issues discovered during development.

An issue being listed here does not mean that it is immediately repaired.
Some issues are deliberately deferred in order to preserve frozen datasets,
frozen model backends, and comparability with previously completed
experiments.

Detailed investigations may live in separate documents or directories.

---

## Issue 1 — Script normalisation and dataset representation

Status: DEFERRED / DOCUMENTED

The reconstructed author dataset and the Frozen PinyinGPT backend may use
different Chinese script representations. Simplified/Traditional
normalisation can therefore change Gold text, context text, tokenisation,
segmentation boundaries, and candidate matching.

This issue has been investigated separately under the script-normalisation
work.

Current decision:

- Preserve Frozen Dataset V1.
- Do not silently rewrite historical experimental inputs.
- Treat script-normalised data as a separate versioned dataset when formal
  repair is performed.
- Report the limitation explicitly in the thesis.

---

## Issue 2 — Frozen backend Gold reachability

Status: DEFERRED / DOCUMENTED

### Description

The benchmark interactions are reconstructed from author text rather than
collected from real IME keystroke logs.

Therefore, a target appearing in an author's historical text does not
guarantee that the Frozen PinyinGPT backend can generate that target for the
reconstructed Pinyin.

A small number of benchmark Gold targets are outside the Frozen PinyinGPT
tokenizer / constrained candidate vocabulary.

These targets are backend-unreachable: no reranker or personalisation method
using the current Frozen backend can produce them.

### Development audit — Full+Short

Rows: 5,608

Gold backend-compatible:
- 5,568 / 5,608
- 99.29%

Gold backend-incompatible:
- 40 / 5,608
- 0.71%

Generic Missing:
- 709

Generic Missing with backend-compatible Gold:
- 669

Generic Missing with backend-incompatible Gold:
- 40
- 5.64% of Generic Missing

Observed incompatibility reasons:
- tokenizer_unknown_at_0: 34
- tokenizer_unknown_at_1: 6

### Frozen Test audit — Full+Short

Rows: 6,000

Gold backend-compatible:
- 5,989 / 6,000
- 99.82%

Gold backend-incompatible:
- 11 / 6,000
- 0.18%

Generic Missing:
- 538

Generic Missing with backend-compatible Gold:
- 527

Generic Missing with backend-incompatible Gold:
- 11
- 2.04% of Generic Missing

Observed incompatibility reasons:
- tokenizer_unknown_at_0: 10
- tokenizer_unknown_at_1: 1

### Interpretation

This is primarily a Frozen backend vocabulary-coverage limitation rather
than evidence that personalisation itself failed.

The reconstructed proxy history records that an author used a target, but
does not prove that the Frozen PinyinGPT backend could have generated that
target from the reconstructed Pinyin.

### Current decision

Do NOT:

- modify Frozen Dataset V1;
- expand the Frozen PinyinGPT tokenizer or candidate vocabulary;
- remove these rows from the main evaluation;
- use this issue to change EM-1 parameters.

Maintain the original benchmark for comparability.

For reporting:

- Overall Top-1 / Top-3 / MRR remain calculated over the complete benchmark.
- Raw Generic Missing remains the ordinary Top-10 missing count.
- Recovery analysis should additionally report Backend-Reachable Generic
  Missing:
    Gold absent from Generic Top-10 AND Gold compatible with the Frozen
    backend.
- Backend-unreachable Gold should be reported as a known backend/dataset
  compatibility limitation rather than silently removed.

### Future repair option

A future dataset/backend-alignment version may explicitly validate every
reconstructed target against the chosen IME backend candidate space.

Such a repair must be versioned separately and must not overwrite Frozen
Dataset V1 or historical results.

---

## Issue status meanings

ACTIVE:
Currently under investigation.

DOCUMENTED:
Confirmed and recorded.

DEFERRED:
Known issue deliberately not repaired in the current experimental stage.

RESOLVED:
A versioned repair has been implemented and validated.

FROZEN:
Historical behaviour that must remain unchanged for reproducibility.
