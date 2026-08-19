# EM-2B Dev Hidden-State Cache Result

Status: COMPLETE / FROZEN

Date: 2026-08-19

## Scope

- Full+Short
- H5000
- Dev tune only
- Authors:
  - Etinjat
  - Re_spectators
  - breaddddd

## Frozen representation

Frozen PinyinGPT2-Concat final-layer hidden state at the final prompt [SEP]
token.

Hidden dimension:

- 768

The representation definition was frozen by the EM-2A engineering gate
before retrieval performance was inspected.

## Workload

Dev queries:

- 5,608

Queries with legal same-Pinyin H5000 history:

- 3,625

Legal query-history edges:

- 122,067

Unique representation inputs:

- 11,475

## Cache result

Cached representations:

- 11,475 / 11,475

Context rows requiring model-limit truncation:

- 0

Prompt-token length:

- minimum: 6
- mean: 436.26
- maximum: 517

All representation inputs therefore fit within the Frozen PinyinGPT
generation context semantics without additional model-limit truncation.

## Leakage checks

Gold used during representation construction:

- No

Historical target used as part of the representation input:

- No

Retrieval metrics inspected during cache construction:

- No

## Artifact

SQLite cache:

results/personalisation/external_memory/em2_hidden_dev/hidden_states.sqlite3

SHA256:

9a80a3314c184ccf3f0540916203c651474fad162dc3dab1fc97f7451f441df1

## Conclusion

EM-2B PASS.

The complete three-author Dev representation surface is cached and ready
for the frozen EM-2C kNN retrieval diagnostic.
