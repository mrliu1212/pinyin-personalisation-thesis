# Personal Vocabulary H5000

> **COMPLETED 2026-08-17** — exact frozen 6,000-row T1 Full+Short Test population.

## Purpose and Frozen Design

This experiment separates candidate availability from candidate ranking. PV0
measures whether strictly prior same-user vocabulary can supply targets missing
from Generic Top-10. PV1 injects bounded frequency-ranked personal candidates.
PV2 tests whether reused M1 BGE context support improves those personal-only
candidates.

- exact 6,000 T1 Full+Short Test anchors, 1,000 per proxy user;
- same-user, strictly prior H5000 applied before exact segmented-Pinyin filtering;
- frozen T1 Generic predictions and zero Test PinyinGPT inference;
- PV1 grid: `Kpv in {1, 3, 5}`, `lambda_pv in {0.5, 1, 2, 4}`;
- PV2 grid: `lambda_ctx in {0.5, 1, 2, 4}` after freezing PV1;
- frozen M1 Top-N = 5 and no M2 Cross-Encoder;
- selection on the chronological 16,171-row Dev-tune partition by Macro-author
  Top-1; Test Gold never selects parameters;
- one Gold-free per-query state shared by PV0, PV1, PV2, and all grid values.

Full definitions are in [Bounded Personal Vocabulary H5000](../research/personal_vocabulary.md).

## PV0 Recoverability

Generic missed 538 of 6,000 targets. Of those, 160 were present in the legal
prior same-Pinyin Personal Lexicon and 378 were absent. The recoverability rate
was therefore `160 / 538 = 0.2973977695` (29.74%). This is candidate
availability, not a Top-1 ranking result.

| Author | Generic missing | Recoverable | Unrecoverable | Recoverability |
| --- | ---: | ---: | ---: | ---: |
| Agent Phage | 33 | 8 | 25 | 0.2424 |
| Etinjat | 74 | 13 | 61 | 0.1757 |
| MScarlet | 335 | 120 | 215 | 0.3582 |
| QBLevi | 38 | 7 | 31 | 0.1842 |
| Re_spectators | 36 | 2 | 34 | 0.0556 |
| breaddddd | 22 | 10 | 12 | 0.4545 |

The active same-Pinyin lexicon contained a mean 1.215 targets per query
(median 1, p90 3, maximum 14). Among the 160 recoverable targets, 38 had only
one visible prior occurrence and 25 had two; the complete occurrence-count
distribution is preserved in `pv0_recoverability.json`.

## Dev Selection

PV1 selected `Kpv = 1` and `lambda_pv = 4.0`. PV2 froze those values and
selected `lambda_ctx = 1.0`, with frozen M1 Top-N = 5. Selection used 16,171
Dev rows and saw zero Test rows. The full 12-row PV1 and four-row PV2 searches
are preserved as CSV files.

## Test Results

| System | Top-1 | Top-3 | MRR@10 | Missing@10 | MeanRank given Top-10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| G0 | 0.723167 | 0.853500 | 0.793429 | 0.089667 | 1.539176 |
| F-H5000 | 0.771833 | 0.872333 | 0.824961 | 0.089667 | 1.366686 |
| M1-H5000 | 0.767500 | 0.871333 | 0.822574 | 0.089667 | 1.376907 |
| M2-H5000 | 0.765000 | 0.871667 | 0.820983 | 0.089667 | 1.382371 |
| **PV1-H5000** | **0.779000** | 0.891500 | **0.838785** | **0.066167** | 1.380187 |
| PV2-H5000 | 0.778167 | **0.891667** | 0.838374 | **0.066167** | 1.380911 |

PV1 recovered 143 original Generic-missing targets into Top-10, 140 into
Top-3, and 96 into Top-1. This is 26.58% of all 538 missing targets and 89.38%
of the 160 PV0-recoverable targets at Top-10. Relative to F-H5000, PV1 helped
96 Top-1 cases and harmed 53 (`net_help = +43`); 4,578 remained correct and
1,273 remained wrong. Of 5,462 originally Generic-covered cases, 53 were
harmed, a 0.9703% covered-case harm rate.

PV2 recovered the same 143 targets into Top-10, all 143 into Top-3, and 101
into Top-1. Its Top-10 recovery rates therefore remain 26.58% of all missing
and 89.38% of recoverable cases. However, relative to PV1, context helped five
Top-1 cases and harmed ten (`net_context_help = -5`), with 4,664 unchanged
correct and 1,321 unchanged wrong. PV2 thus did not improve the primary Test
metric over PV1.

## Per-author Top-1 and Missing@10

| Author | F Top-1 | PV1 Top-1 | PV2 Top-1 | PV1 Missing@10 | PV2 Missing@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Agent Phage | 0.844 | 0.841 | 0.842 | 0.025 | 0.025 |
| Etinjat | 0.722 | 0.701 | 0.699 | 0.063 | 0.063 |
| MScarlet | 0.493 | 0.558 | 0.554 | 0.232 | 0.232 |
| QBLevi | 0.862 | 0.863 | 0.863 | 0.031 | 0.031 |
| Re_spectators | 0.846 | 0.839 | 0.839 | 0.034 | 0.034 |
| breaddddd | 0.864 | 0.872 | 0.872 | 0.012 | 0.012 |

The largest availability and Top-1 gain occurs for MScarlet, the author with
most Generic-missing rows. Benefits are heterogeneous: bounded injection
improves the macro result despite small regressions for several authors.

## Reuse, Runtime, and Provenance

Preparation required 39,680 unique contexts and found 39,680 BGE cache hits,
zero misses, and zero new embeddings; the embedding model was not loaded.
Generic Test inference was zero. State building processed 6,000 Test and
16,171 Dev rows; the completed foreground `--phase all` command took 92.6
seconds wall time. Arithmetic Dev selection took 1.656 seconds and the final
shared 6,000-row evaluation took 0.769 seconds.

T1, M1, and M2 inputs were hash-checked before evaluation and remained
unchanged. The final prediction checksum is
`cb39d210c2c35453aa40ac250188f237742f0a3c7837c5945bc86721765ff3d7`;
the metrics checksum is
`ab5566991f85474ecc1dc6f3f6ab7216ee25a02515cad76fb7958218c0e6f29c`.
All durable artifacts are under
`results/personalisation/personal_vocabulary_h5000/`.

## Interpretation and Limitations

The main finding is that bounded prior-user vocabulary changes what the system
can output: it removes 141 Missing@10 cases net (`538 - 397`) and produces a
modest Top-1 gain over F-H5000 (`+0.007167`). This benefit is not obtainable by
F, M1, or M2 because they preserve Generic's candidate surface. Candidate
availability is therefore a distinct and useful personalisation axis.

The reused generic BGE context term did not improve PV1 overall. PV2's
`net_context_help = -5` and `-0.000833` Top-1 difference are a diagnostic
negative result, not evidence against personal vocabulary itself.

Limitations include the fixed H5000 window, frequency-biased Kpv pruning,
proxy users, reconstructed Pinyin, cold start, unseen vocabulary, an
approximate Generic boundary score for candidates without a PinyinGPT score,
possible erroneous historical selections, generic BGE similarity, and
unmodelled temporal drift. No task-specific training, M3, internal PinyinGPT
personalisation, temporal adaptation, or Transparency/Control interface was
introduced.
