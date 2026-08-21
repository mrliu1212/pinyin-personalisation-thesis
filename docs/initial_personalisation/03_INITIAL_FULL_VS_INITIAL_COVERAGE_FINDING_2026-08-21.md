# Standardized Full vs Initial Candidate-Coverage Finding

Date: 2026-08-21
Status: Train-Val diagnostic, pre-Dev3000, Test closed

## Question

Why does Initial-only Pinyin perform much worse than Full Pinyin?

One candidate mechanism is that abbreviation hurts the system before personalised reranking: the correct Gold target may fall out of the frozen Generic Top-10 candidate surface.

## Paired standardized design

The comparison uses the same 34,416 standardized Train-Val anchors under both input conditions. Author, work, chronological position, context, and Gold are held fixed; only the Pinyin representation changes from Full segmented Pinyin to deterministic first-letter Initials.

No Dev3000 or Test data were used.

## Main results

| Metric | Full | Initial | Difference |
|---|---:|---:|---:|
| Generic Top-1 | 73.6024% | 33.0573% | -40.5451 pp |
| Missing@10 | 6.9212% | 36.5092% | +29.5880 pp |
| Top-10 coverage | 93.0788% | 63.4908% | -29.5880 pp |

Initial Missing@10 is 5.275x the Full rate.

A total of 10,223 / 34,416 rows (29.704%) are Full-covered but Initial-missing. These are direct same-anchor cases where the Gold target is retained in the frozen Generic Top-10 under Full Pinyin but lost after converting the same query to Initial-only Pinyin.

Among all Initial-missing rows (12,565), 10,223 (81.36%) belong to this Full-covered -> Initial-missing transition.

## Error decomposition

The observed Full-to-Initial Top-1 gap is 40.5451 percentage points.

The increase in Missing@10 contributes 29.5880 percentage points of this gap, while the remaining 10.9571 percentage points come from additional ranking errors among queries whose Gold still remains inside Top-10.

Equivalently, 72.98% of the observed Top-1 gap can be accounted for by the increase in candidate-surface misses in this exact metric decomposition. This is an accounting decomposition of the observed errors, not a claim that candidate coverage is the only causal mechanism.

## Personal-history recovery opportunity

Within the 10,223 Full-covered -> Initial-missing rows, 51.883% are recoverable from backend-compatible personal history under the legal rolling H5000 protocol.

That is exactly 5,304 rows.

This supports the mechanism:

Full Pinyin -> stronger phonetic constraints -> high Generic candidate coverage

Initial-only Pinyin -> substantial phonetic information loss -> much larger candidate ambiguity -> correct Gold more often falls outside Generic Top-10

Long-term personal history can recover a meaningful fraction of those lost candidates.

## Safe thesis interpretation

A substantial part of the Initial-input degradation occurs before personalised reranking. On the same 34,416 anchors, reducing Full Pinyin to Initial-only input raises Generic Missing@10 from 6.92% to 36.51%, a 29.59-point increase. Nearly one third of all queries are specifically Full-covered but Initial-missing. Moreover, 51.9% of these lost candidates remain recoverable from backend-compatible personal history. This suggests that candidate recovery is especially important under Initial input, where abbreviation weakens the Generic candidate surface much more than under Full Pinyin.

## What this does not prove

- It does not show that all Full-to-Initial Top-1 degradation is caused by candidate coverage; ranking errors also increase.
- It does not yet show that PV1 or EM1 will convert all recoverable opportunities into Top-1 gains.
- It does not use Dev3000 or Test, so this remains a Train-Val diagnostic finding until later confirmation.
