# Deep Author Evaluation V2 Design

## 1. Purpose

Evaluation V2 freezes a chronological generic-baseline protocol before any
personalisation experiment. It uses Deep Author Dataset V1, not V1.1, because
V1 is the previously specified development source. V1.1 remains preserved for
the later final cleaned-dataset rerun.

No model inference was performed while this design was prepared.

## 2. Verified source

Dataset V1 was reconstructed from tag `deep-author-dataset-preparation-v1`
using the immutable local raw snapshot in a separate ignored worktree. The
result exactly matched the frozen artifact:

- 2,048,557,493 bytes;
- SHA-256 `8d1a98e18a5f7ed997930b65bbd1149c3d52daaa22ac2c59771256a966648da2`;
- 282 included works, 1,074,032 V1 interactions, and 26 recorded alignment
  failures.

The six authors remain Re_spectators, MScarlet, Etinjat, Agent Phage, QBLevi,
and breaddddd.

## 3. Chronological split

Works were ordered independently per author by creation date and work ID. Two
chronological cut points were chosen to minimize squared deviation from the
70/10/20 History/Dev/Test target by eligible-anchor volume, subject to at least
5 History works, 2 Dev works, 3 Test works, and 1,000 valid Test anchors.

| Author | History works / anchors | Dev works / anchors | Test works / anchors |
|---|---:|---:|---:|
| Re_spectators | 11 / 10,997 | 2 / 1,282 | 3 / 11,442 |
| MScarlet | 24 / 42,315 | 2 / 6,172 | 8 / 11,524 |
| Etinjat | 63 / 40,936 | 12 / 4,045 | 9 / 17,663 |
| Agent Phage | 45 / 69,667 | 5 / 10,110 | 6 / 19,129 |
| QBLevi | 28 / 15,828 | 2 / 2,035 | 7 / 4,214 |
| breaddddd | 45 / 68,339 | 5 / 8,568 | 5 / 22,677 |

The target is an anchor-volume objective rather than an exact quota. In
particular, Re_spectators has only 16 eligible works and its last three works
are large, so the required work minima prevent a close 70/10/20 volume split.
No work appears in more than one split.

## 4. Frozen T1 anchors

Only Test works were sampled. A valid anchor begins at an eligible V1 Short
token and supports exactly three consecutive V1 Jieba tokens without an
intervening token or character gap. The Short and Multi-3 targets must both
align deterministically to Pinyin and every Gold character must be compatible
with the frozen checkpoint map for both its full syllable and official
single-letter initial.

Sampling uses seed 40408. Eligible anchors were deterministically shuffled
within each Test work and selected by round-robin across works. This produced
exactly 1,000 anchors per author and 6,000 total. The selected counts are as
work-balanced as availability permits: equal for MScarlet and breaddddd, within
one for Re_spectators and Agent Phage, and bounded by the smaller eligible pool
for one Etinjat and one QBLevi work.

## 5. Paired conditions

Every anchor has the same preceding V1 context and produces four rows:

1. Full + Short;
2. Initial + Short;
3. Full + Multi-3;
4. Initial + Multi-3.

There are 24,000 unique condition rows, 6,000 in each condition. Multi-3 always
contains exactly three consecutive frozen V1 Jieba tokens. Full and Initial use
oracle Pinyin segmentation and share the same Gold within each target type.

## 6. Frozen generic model protocol

T1 will use `aihijo/transformers4ime-pinyingpt-concat` at revision
`76dd20dc92d8236a350fb732e99dde6fa15e2263`, with official code reference
`8f1573ed0bd4d1f3d8d3f10a05f7e870725646f1`. The input is the established
Concat sequence `[CLS] context [SEP] Pinyin tokens [SEP]`. Decoding uses the
Pinyin compatibility map, beam size 16, and up to ten candidate sequences.

Immediately before inference, context will be truncated from the left using the
frozen tokenizer so the most recent preceding context fits the 1,024-position
model limit after prompt, Pinyin, separator, and generated-target requirements
are accounted for. No future text, author identity, history, personal
vocabulary, personal frequency, or Dev data enters T1.

## 7. Validation and limitations

The design validation confirmed unique anchor and condition IDs, chronological
work separation, required work minima, Test-only anchors, exactly 1,000 anchors
per author, all four paired rows per anchor, and the frozen Dataset V1 hash.

This is a development evaluation. Dataset V1 contains known residual SCP
platform/template material. Authors are proxy users, Pinyin is reconstructed,
and Multi-3 is simulated. The protocol will later be rerun on the final cleaned
dataset. T2 and T3 feasibility files are inventories only; no performance result
was generated for either task.
