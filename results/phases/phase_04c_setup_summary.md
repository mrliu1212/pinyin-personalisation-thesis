# Phase 4C — Personalisation Evaluation Setup

## Status

Experiment preparation is complete. The final Base/correct-user/wrong-user
comparison has not been run and no personalisation improvement is claimed.

## Selected users

- **User A — Zhu Ziqing:** correct-user history and future evaluation target.
- **User B — Lu Xun:** wrong-user history control.

## Frozen chronological partitions

| User | Partition | Work IDs | Interactions |
| --- | --- | --- | ---: |
| Zhu Ziqing | Train, before 1930 | `congcong`, `qinhuai_river`, `beiying`, `ahe`, `moonlight_over_lotus_pond` | 3,765 |
| Zhu Ziqing | Test, after 1930 | `to_my_late_wife`, `spring` | 926 |
| Lu Xun | Train, before 1930 | `madmans_diary`, `kong_yiji`, `medicine`, `hometown`, `new_years_sacrifice` | 7,014 |
| Lu Xun | Held out, 1934 | `takeism`, `have_chinese_lost_self_confidence` | 579 |

The Lu held-out partition is recorded for corpus integrity but is not used in
the wrong-user history condition.

## Lu Xun corpus outcome

Seven individual works were acquired from Chinese Wikisource and preserved as
revision-pinned raw API responses. `吶喊` and `彷徨` were recorded as excluded
collection/container pages. Processing reused the accepted path without an
author-specific variant:

```text
raw Wikisource response
    → conservative Phase 4A cleaning
    → OpenCC t2s
    → Jieba default segmentation
    → tone-free pypinyin
    → Luna Pinyin with engine-side zh_hans
```

The resulting Lu dataset contains 7,593 interactions. Its interaction JSONL
SHA-256 is
`cc022956f36ba21e61f70677355cdd6c31a5b52854e744e3920a63c124a94861`.

The accepted Zhu Phase 4B.6 interaction checksum remains
`2d0df837fed3cf6b1a141b9f43677733671cf1f08cb72ca3b9e2f0f2f13f5077`.

## Prepared evaluation settings

1. Original Luna `zh_hans` Base order with no personal history.
2. Zhu training history applied to later Zhu interactions.
3. Lu training history applied to the same later Zhu interactions.

Both the full 926-row test benchmark and the Base-rerankable subset are
reported. Prepared metrics are Top-1/3/5/10, MRR, mean target rank, and
improved/unchanged/harmed counts. Detailed JSON output includes deterministic,
candidate-level transparency examples.

## Remaining step

After accepting this setup, run the final experiment manually:

```bash
.venv/bin/python -m experiments.exp_phase_04c_personalisation
```

This writes `results/experiments/phase_04c/evaluation.json`. The result must be
interpreted without assuming that personalisation is necessarily better than
Base.
