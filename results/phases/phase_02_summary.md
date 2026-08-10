# Phase 2 Result Summary

## Status

Phase 2 — Context-Sensitive Interpretable Personalisation: **COMPLETE**

## Test Result

- 8 tests passed
- 0 failures

Verified behaviours:

- Future context evidence is excluded.
- Ranked candidates expose all scoring components.
- The same Pinyin can learn different preferences under different contexts.
- Unseen contexts retain broader Pinyin/global personal evidence.
- Different users can learn different preferences for the same context +
  Pinyin.
- Evidence weights are explicit and configurable.
- Other-user and future interactions remain excluded.
- Phase 1 reranking behaviour remains valid.

## Observed Minimal Pipeline Result

1. 使用 — base=0.80, global=3, pinyin=3, context=3, personal=3.00,
   final=0.80
2. 实用 — base=0.90, global=0, pinyin=0, context=0, personal=0.00,
   final=0.40
3. 试用 — base=0.70, global=0, pinyin=0, context=0, personal=0.00,
   final=0.00

The base ranker preferred `实用` over `使用`. After personalisation, historical
evidence promoted `使用` to Rank 1. The evidence responsible for the promotion
is directly inspectable through the global, Pinyin, and context evidence
fields.

## Current Model Configuration

- Global evidence weight: `0.1`
- Pinyin evidence weight: `0.3`
- Context evidence weight: `0.6`
- These values are configuration defaults and have not been optimised.
- Personal and base scores are min-max normalised within the current candidate
  list before interpolation.

## Known Limitations Carried Forward

- Exact full-context matching may become sparse on real text.
- Evidence weights are not optimised.
- Min-max normalisation does not explicitly represent evidence confidence.
- Raw frequency does not currently include recency.
- The current base candidate source remains synthetic/in-memory.

