# Phase 4C — Personalisation Evaluation

## Objective

Evaluate whether transparent, user-specific history improves later Zhu Ziqing
Pinyin candidate ranks relative to the original Luna `zh_hans` ranking and to
a wrong-user control built from Lu Xun history.

The experiment tests measurement capability; it does not assume that either
personalised condition must outperform Base.

## Scope

- Preserve the accepted 4,691-interaction Phase 4B.6 Zhu Ziqing benchmark.
- Build a separate Lu Xun corpus through the identical processing path:
  revision-pinned Wikisource raw text, conservative cleaning, OpenCC `t2s`,
  Jieba segmentation, tone-free pypinyin, and Luna `zh_hans` Top-10 candidates.
- Use fixed work-level chronological partitions, never a random split.
- Evaluate Base, correct-user Zhu history, and wrong-user Lu history on the
  same later Zhu interactions.
- Report full-benchmark and rerankable-subset results.
- Preserve candidate-level evidence in deterministic transparency examples.

No scoring weights, interpolation behaviour, candidate generation, or earlier
phase data are changed.

## Chronological design

### Zhu Ziqing — evaluation target

Training history consists only of works before 1930:

| Work ID | Work | Date | Interactions |
| --- | --- | --- | ---: |
| `congcong` | 匆匆 | 1922-03-28 | 149 |
| `qinhuai_river` | 槳聲燈影裏的秦淮河 | 1924-01-25 | 1,627 |
| `beiying` | 背影 | 1925-10 | 363 |
| `ahe` | 阿河 | 1926-01-11 | 1,236 |
| `moonlight_over_lotus_pond` | 荷塘月色 | 1927-07 | 390 |

The fixed test partition consists only of works after 1930:

| Work ID | Work | Date | Interactions |
| --- | --- | --- | ---: |
| `to_my_late_wife` | 給亡婦 | 1932-10-11 | 730 |
| `spring` | 春 | 1933-07 | 196 |

This yields 3,765 training-history interactions and 926 test interactions.
The personal model is frozen during test evaluation; test selections are not
fed back as online history.

### Lu Xun — wrong-user control

The Lu Xun training partition is likewise restricted to works before 1930:

| Work ID | Work | Date | Interactions |
| --- | --- | --- | ---: |
| `madmans_diary` | 狂人日記 | 1918-04 | 1,235 |
| `kong_yiji` | 孔乙己 | 1919-03 | 683 |
| `medicine` | 藥 | 1919-04 | 1,236 |
| `hometown` | 故鄉 | 1921-01 | 1,327 |
| `new_years_sacrifice` | 祝福 | 1924-02-07 | 2,533 |

The separately recorded Lu Xun test partition is:

| Work ID | Work | Date | Interactions |
| --- | --- | --- | ---: |
| `takeism` | 拿来主义 | 1934-06-04 | 384 |
| `have_chinese_lost_self_confidence` | 中國人失掉自信力了嗎 | 1934-09-25 | 195 |

The Lu corpus therefore contains 7,014 training-history interactions and 579
held-out interactions. The held-out Lu partition is not used to personalise
Zhu test examples. `吶喊` and `彷徨` are recorded but excluded because they are
collection/container pages rather than individual works.

## Evaluation settings

1. **Base:** original per-interaction Luna `zh_hans` candidate order, without
   personal history.
2. **Correct user:** the existing frequency model fitted only to the five Zhu
   training works.
3. **Wrong user:** the same model fitted only to the five Lu training works,
   then applied unchanged to Zhu test interactions.

The two personalised settings use the unchanged Phase 2 defaults: interpolation
`alpha=0.5`, with global/Pinyin/context evidence weights `0.1/0.3/0.6`.

Librime exposes candidate order but not numeric scores in this adapter. To pass
that order through the existing min-max interpolation without changing the
reranker, Phase 4C uses the deterministic ordinal utility
`candidate_count - base_rank + 1`. It preserves the Luna order and is explicitly
not a probability or a hidden librime score.

## Required behaviours

- Every Zhu and Lu history row is strictly earlier than the first Zhu test row.
- Correct-user history contains only Zhu training interactions.
- Wrong-user history contains only Lu training interactions.
- No Zhu test or Lu held-out interaction enters either personal model.
- Reversing input file order does not change evaluation output.
- Missing Base targets remain in the full benchmark and are not assigned an
  invented rank.
- Transparency examples expose context, Pinyin, target, Base rank, ordinal Base
  score, all three evidence components, combined personal score, final score,
  and personalised rank.

## Metrics and subsets

Each condition reports Top-1, Top-3, Top-5, Top-10, MRR, and mean target rank.
Each personalised condition additionally reports improved, unchanged, and
harmed counts relative to Base.

Missing targets count as incorrect for Top-K and contribute zero to MRR. They
are reported separately and excluded from mean target rank rather than being
assigned an arbitrary numeric rank.

The full benchmark contains all 926 Zhu test interactions. The rerankable
subset contains only rows whose target appears in Base Top-10. This distinction
is necessary because reranking cannot recover a target absent from candidate
generation.

## Completion criteria

- The separate, revision-pinned Lu corpus and aligned interaction data exist.
- Both fixed author splits and their counts are recorded.
- The deterministic runner supports all three settings, both subsets, required
  metrics, and transparency output.
- Tests cover split chronology, correct/wrong user isolation, future exclusion,
  deterministic evaluation, and extended metrics.
- The full test suite passes.
- The final comparison remains a deliberate manual run after setup acceptance.

## Known limitations and deferred questions

- Both authors are represented by a small, curated selection of works.
- Genre differs within and between the selected corpora and may affect the
  wrong-user control.
- Exact 12-character context matching may be sparse.
- Evidence weights and interpolation are not optimised.
- Ordinal Base utility preserves order but not unknown confidence gaps between
  Luna candidates.
- A reranker cannot address the accepted benchmark's Base Top-10 misses.
- Phase 4B.7 human labels may later inform interpretation, but are not used to
  filter or alter this benchmark.

Abbreviated Pinyin/Jianpin remains deferred and should only be considered if a
future base candidate generator already supports it.
