# Phase 4B Data-Quality Audit Summary

This is a factual audit of the existing interaction dataset. It does not infer
error causes, alter eligibility, or report personalisation performance.

## Top-10 Missing Diagnostics

- Source interactions: 4531
- Target absent from Top-10: 882
- Missing rate: 19.47%
- Stored Pinyin: 882
- Stored syllable lists: 882
- Polyphonic-review flagged: 625 (70.86%)
- Polyphonic-review unflagged: 257

Target-length distribution: `{"2": 671, "3": 190, "4": 21}`

Work distribution: `{"ahe": 327, "beiying": 43, "congcong": 11, "moonlight_over_lotus_pond": 56, "qinhuai_river": 298, "spring": 36, "to_my_late_wife": 111}`

Candidate-list-size distribution: `{"10": 880, "8": 2}`

## Most Frequent Repeated Missing Targets

| Target | Count | Stored Pinyin | Works |
| --- | ---: | --- | --- |
| 阿河 | 18 | ahe | ahe |
| 因為 | 14 | yinwei | ahe, qinhuai_river, to_my_late_wife |
| 什么 | 11 | shenme | ahe |
| 阿齊 | 11 | aqi | ahe |
| 房里 | 8 | fangli | ahe |
| 這里 | 7 | zheli | ahe, qinhuai_river |
| 韋君 | 6 | weijun | ahe |
| 大中橋 | 5 | dazhongqiao | qinhuai_river |
| 采蓮 | 5 | cailian | moonlight_over_lotus_pond |
| 不愿 | 4 | buyuan | ahe |
| 中橋 | 4 | zhongqiao | qinhuai_river |
| 几天 | 4 | jitian | ahe |
| 平伯 | 4 | pingbo | qinhuai_river |
| 我現 | 4 | woxian | ahe, beiying, to_my_late_wife |
| 李媽 | 4 | lima | ahe |
| 東關頭 | 4 | dongguantou | qinhuai_river |
| 艙前 | 4 | cangqian | qinhuai_river |
| 他終 | 3 | tazhong | beiying |
| 几乎 | 3 | jihu | ahe |
| 可怜 | 3 | kelian | ahe |

The complete repeated-target list and aggregate counts are in
`top10_missing_diagnostics.json`. Manual reviewers may consider Pinyin,
segmentation, character variants, vocabulary rarity, candidate truncation, and
other explanations, but this audit assigns none of those labels automatically.
