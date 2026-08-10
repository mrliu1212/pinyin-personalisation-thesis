# Phase 4B — Interaction Construction and Base Coverage

## Status

Phase 4B — Real Pinyin-IME Interaction Construction and Base Candidate
Coverage: **COMPLETE**

This result measures construction and Base retrieval coverage only. It contains
no personalised-ranking experiment.

## Observed Dataset

- Included lexical interactions: 4,531
- Successfully converted to full Pinyin: 4,531
- Pinyin conversion failures: 0
- Candidate maximum K: 10
- Mean candidate-list size: 9.99
- Median candidate-list size: 10.00
- Potential polyphonic cases flagged for review: 2,695

Interactions by work:

| Work ID | Interactions |
| --- | ---: |
| `ahe` | 1,214 |
| `beiying` | 348 |
| `congcong` | 140 |
| `moonlight_over_lotus_pond` | 367 |
| `qinhuai_river` | 1,543 |
| `spring` | 194 |
| `to_my_late_wife` | 725 |

Interactions by target length:

| Chinese characters | Interactions |
| ---: | ---: |
| 2 | 4,019 |
| 3 | 461 |
| 4 | 51 |

Excluded segmented tokens were counted rather than silently discarded:

| Reason | Count |
| --- | ---: |
| Below two characters | 5,350 |
| Non-Chinese, punctuation, Latin text, or numbers | 2,430 |
| Above four characters | 1 |

## Base Target Coverage

| Retrieval depth | Present | Coverage |
| --- | ---: | ---: |
| Top-1 | 2,930 | 64.67% |
| Top-3 | 3,485 | 76.91% |
| Top-5 | 3,602 | 79.50% |
| Top-10 | 3,649 | 80.53% |

- Target absent from retrieved Top-10: 882
- Missing-target rate: 19.47%

Missing targets remain in the interaction file and in the denominator. These
figures are Base candidate-source coverage, not personalisation accuracy.

## One-Work Pilot

Before full generation, 50 interactions from `匆匆` were inspected. Pilot
coverage was 90.0% Top-1, 94.0% Top-3, 96.0% Top-5/Top-10, with two explicit
Top-10 misses. Representative records were:

| Derived context | Pinyin | Target | Base candidates (prefix) | Rank |
| --- | --- | --- | --- | ---: |
| *(empty)* | `yanzi` | 燕子 | 燕子, 燕姿, 驗資, 晏紫, … | 1 |
| 你吿訴我我們的日子爲什麼 | `yiqu` | 一去 | 一曲, 易趣, 一區, 一去, … | 4 |
| 已經從我手中溜去像針尖上 | `yidi` | 一滴 | 異地, 一滴, 一地, 易地, … | 2 |
| 藏在何處呢是他們自己逃走 | `leba` | 了罷 | 了吧, 樂吧, 了, 樂, … | absent |

The last case illustrates that targets are not rewritten to match Rime.

## Exact Configuration

- Segmentation: `jieba==0.42.1`, default tokenizer mode
- Target policy: all-Chinese tokens, 2–4 characters
- Pinyin: `pypinyin==0.55.0`, `Style.NORMAL`, strict, tone-free full syllables,
  concatenated for Rime input
- Context: complete preceding source text plus its last 12 Chinese characters
- Candidate engine: `librime 1.17.0_2` through the repository C++ command-line
  adapter
- Schema: `luna_pinyin`
- Candidate order: librime candidate-iterator order
- Numeric Base score: unavailable; stored as `null`
- Rime source revisions: exact commits in `config/rime/sources.json`, copied
  into each generated manifest

The full machine-readable configuration and aggregate coverage are in
`data/processed/interactions/zhu_ziqing/manifest.json`.

## Test Result

- 29 tests passed
- 0 failures

This includes all earlier tests plus offline tests for lexical conversion,
Pinyin normalization and review flags, filtering, context, chronology,
target-rank detection, missing targets, and deterministic construction.

## Unresolved Problems Before Phase 4C

- The 19.47% Top-10 miss population needs error analysis before deciding
  whether to change retrieval depth or eligibility policy.
- Automatic Jieba units may differ from realistic author input boundaries.
- Polyphonic review flags are broad; a reproducible review/correction policy is
  needed before treating every generated reading as ground truth.
- A second author is required for a meaningful wrong-user control.
- The chronological history/test boundary, including treatment of partial work
  dates, remains unselected.
- No evidence weights, interpolation parameters, recency behavior, or personal
  scoring logic were changed or evaluated.
