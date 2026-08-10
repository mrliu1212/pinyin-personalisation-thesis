# Transparent, User-Controllable Pinyin Personalisation

## Current Phase

Phase 4B.6 — Rime Script Alignment Analysis

## Current Objective

This version measures Base candidate coverage when both the Zhu Ziqing corpus
and the Rime engine use Simplified Chinese. It compares three controlled
settings:

1. Phase 4B: original Traditional/mixed corpus with default Luna output;
2. Phase 4B.5: OpenCC T2S corpus with mismatched default Luna output;
3. Phase 4B.6: OpenCC T2S corpus with engine-side Luna `zh_hans` output.

This remains a data-construction and retrieval experiment. It does not change
or evaluate personalisation and does not start Phase 4C.

## Why This Phase

Phase 4B.5 was not a fair Simplified Chinese evaluation: only the corpus was
converted, while the candidate generator continued to emit predominantly
Traditional/mixed strings. Its coverage reduction measured script mismatch.

Phase 4B.6 aligns the two sides. Luna Pinyin's existing
`simplifier@zh_hans` engine filter applies OpenCC `t2s.json` when the `zh_hans`
schema option is enabled. The adapter sets that option inside an isolated
librime session; it never converts retrieved candidate strings afterwards.

## Current Pipeline

```text
Phase 4A canonical text (preserved)
        ↓
separate OpenCC t2s.json representation
        ↓
Jieba segmentation
        ↓
tone-free full Pinyin
        ↓
luna_pinyin + engine option zh_hans
        ↓
Simplified candidate order from librime
        ↓
coverage, script, recovery, and provenance diagnostics
```

Each interaction stores work identity/chronology, source offsets, original
target, Simplified target and contexts, Pinyin, ordered candidates, target rank,
and both normalization and Rime script-mode provenance.

## Observed Coverage

| Setting | Corpus | Rime output | Interactions | Top-1 | Top-3 | Top-5 | Top-10 | Missing |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 4B | Original Traditional/mixed | Default Luna | 4,531 | 64.67% | 76.91% | 79.50% | 80.53% | 19.47% |
| Phase 4B.5 | OpenCC T2S | Default Luna | 4,691 | 32.85% | 39.14% | 40.55% | 41.04% | 58.96% |
| Phase 4B.6 | OpenCC T2S | Luna `zh_hans` | 4,691 | 72.12% | 85.44% | 88.23% | 89.11% | 10.89% |

Engine alignment recovers 2,255 of Phase 4B.5's 2,766 missing targets, leaving
511. These are candidate-coverage results, not personalisation results.

Direct engine verification:

| Pinyin | Default | Simplified Rime |
| --- | --- | --- |
| `weishenme` | 爲什麼 | 为什么 |
| `women` | 我們 | 我们 |
| `shihou` | 時候 | 时候 |

## Candidate Script Diagnostics

Among 46,878 Phase 4B.6 candidate occurrences:

- Simplified-only: 20,135 (42.9519%)
- Traditional-only: 18 (0.0384%)
- Mixed: 2 (0.0043%)
- Script-invariant: 26,723 (57.0054%)

Script-invariant candidates are reported separately because many Chinese forms
are shared across both conventions. Classification compares each candidate with
OpenCC `t2s` and `s2t`; no candidate is removed or rewritten for diagnostics.

## Interaction Count Difference

OpenCC runs before Jieba, so normalized character forms change lexical
boundaries. Relative to Phase 4B, the 4,691-interaction set contains 865 added
spans and 705 removed spans, for a net increase of 160. This is a segmentation
effect, not candidate generation adding interactions. The detailed audit is in
`results/audits/phase_04b/script_normalization_interaction_delta.json`.

## Dependencies and Setup

From the repository root on macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-phase4b.txt
brew install opencc librime
.venv/bin/python -m interactions.setup_rime
make rime-adapter
```

Python dependencies remain `jieba==0.42.1` and `pypinyin==0.55.0`. Rime data
commits are locked in `config/rime/sources.json`; Simplified mode is declared in
`config/rime/simplified_candidate_mode.json`.

## How to Run

Regenerate the separate OpenCC-normalized corpus and Phase 4B.5 comparison:

```bash
.venv/bin/python -m normalization.phase_04b5
```

Generate the aligned Simplified Rime interactions and full diagnostics:

```bash
.venv/bin/python -m normalization.phase_04b6
```

Reproduce the segmentation-delta audit:

```bash
.venv/bin/python -m audits.phase_04b5_interaction_delta
```

Run all tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The Phase 4B.6 command prints three-way coverage and candidate-script counts.
Detailed results are written to
`data/processed/interactions/zhu_ziqing_simplified_rime/phase_04b6_comparison.json`.

## Data Layout

```text
data/processed/
├── authors/zhu_ziqing/                       # unchanged Phase 4A
├── normalized/authors/zhu_ziqing_t2s/       # Phase 4B.5 T2S text
└── interactions/
    ├── zhu_ziqing/                           # Phase 4B baseline
    ├── zhu_ziqing_t2s/                       # Phase 4B.5 mismatch
    └── zhu_ziqing_simplified_rime/           # Phase 4B.6 aligned
```

## Important Assumptions

- Phase 4B.5 and Phase 4B.6 use identical normalized text, segmentation,
  Pinyin, target filtering, context policy, schema data, and Top-10 limit.
- Their only candidate-side difference is the engine `zh_hans` option.
- Rime uses a fresh temporary user directory per run, preventing saved options
  or learned state from influencing results.
- Coverage uses exact candidate/target equality.
- Numeric Base scores remain unavailable; engine candidate rank is preserved.

## Current Limitations

- Results cover one author, one corpus, and one pinned Luna schema snapshot.
- T2S changes segmentation, so Phase 4B versus Phase 4B.6 is not fully paired;
  exact-span recovery is reported separately.
- A small residual of 18 Traditional-only and two mixed candidate occurrences
  remains under the documented OpenCC-based classifier.
- Candidate coverage does not show whether personalisation helps.
- Manual review and analysis of 511 remaining misses are incomplete.
- No second author or final chronology boundary has been selected.

## Next Planned Phase

Phase 4C remains deferred. Before starting it, the project must choose the
benchmark script configuration, review remaining data-quality issues, prepare a
second-author control, and define the chronological history/test boundary.

## Project History

- Phase specifications: [`docs/phases/`](docs/phases/)
- Completed outcomes: [`results/phases/`](results/phases/)
- Phase 4B audits: [`results/audits/phase_04b/`](results/audits/phase_04b/)
- Workflow: [`docs/WORKFLOW.md`](docs/WORKFLOW.md)

Existing Git tags are unchanged. Tags are not created automatically.
