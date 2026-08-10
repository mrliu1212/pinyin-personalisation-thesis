# Phase 4B.5 — Script Normalisation Analysis

## Status

Phase 4B.5 — Script Normalisation Analysis: **COMPLETE**

This sub-phase measures corpus-side script normalization and Base retrieval
coverage only. No personalisation experiment or model change was performed.

Phase 4B.5 is **not a valid evaluation of Simplified Chinese IME
performance**. It converted only the corpus to Simplified Chinese while the
unchanged Luna Pinyin configuration continued to return predominantly
Traditional/mixed candidates. The result therefore measures script mismatch,
not the quality of a script-aligned Simplified IME pipeline.

## Method

The seven unchanged Phase 4A cleaned works were converted to a separate
Simplified Chinese representation with OpenCC `t2s.json`. Homebrew reports
OpenCC package `1.4.1`; the installed CLI identifies itself as OpenCC command
line tool version `1.4.0`.

The same Phase 4B settings were then applied to the normalized representation:

- Jieba `0.42.1`, default segmentation;
- all-Chinese lexical targets of 2–4 characters;
- pypinyin `0.55.0`, tone-free full Pinyin;
- 12-Chinese-character derived context;
- unchanged Luna Pinyin/librime candidate generator;
- unchanged maximum candidate K of 10;
- exact target/candidate string equality.

OpenCC changed 3,768 code-point positions. It did not change code-point length
in any work, allowing exact original/normalized offset correspondence. The
normalization pipeline verifies this property and stops if it is not true.

## Coverage Comparison

| Metric | Before normalization | After OpenCC T2S | Difference |
| --- | ---: | ---: | ---: |
| Interactions | 4,531 | 4,691 | +160 |
| Top-1 | 2,930 (64.67%) | 1,541 (32.85%) | −31.82 pp |
| Top-3 | 3,485 (76.91%) | 1,836 (39.14%) | −37.78 pp |
| Top-5 | 3,602 (79.50%) | 1,902 (40.55%) | −38.95 pp |
| Top-10 | 3,649 (80.53%) | 1,925 (41.04%) | −39.50 pp |
| Missing | 882 (19.47%) | 2,766 (58.96%) | +39.50 pp |

The denominators differ because applying the unchanged Jieba segmenter after
T2S conversion changes some lexical boundaries. The aggregate figures describe
each full derived interaction set rather than a paired-only subset.

## Why the Interaction Count Changed

The increase from 4,531 to 4,691 was not caused by simply adding 160 targets.
Exact work/start/end comparison found:

- interactions added after normalization: 865;
- interactions removed after normalization: 705;
- net change: +160.

OpenCC runs before Jieba. Changed character forms therefore alter Jieba's
lexical boundaries, and the unchanged 2–4-character eligibility filter is then
applied to a different token stream. OpenCC does not directly create
interactions.

Examples:

1. `鄉 / 下` are separate one-character tokens and are filtered; after T2S,
   Jieba produces eligible `乡下`, adding that interaction.
2. Traditional `那別墅` may be one token; after T2S, `那别墅` is split so
   `别墅` becomes a new eligible interaction.
3. Traditional `常如鏡子` is one eligible token; normalized `常如镜子` is
   split into `常如 / 镜子`, removing the original interaction and adding two
   different spans.
4. Boundaries around `不聲不響 → 不声不响` also change; surrounding converted
   characters can affect segmentation even when a particular target substring
   itself is unchanged.

The complete delta audit records 865 additions, 705 removals, their work/span
provenance, and overlapping token evidence in
`results/audits/phase_04b/script_normalization_interaction_delta.json`.

## Previously Missing Targets

Recovery is defined conservatively: the normalized interaction must have the
same work ID and exact source start/end offsets as the baseline interaction.

- Baseline missing targets: 882
- With an identical normalized source span: 578
- Recovered after normalization: 0
- Still missing at the identical span: 578
- Without an identical span after normalized re-segmentation: 304

There are no recovered examples to report because the observed recovered count
is zero. This is retained as a negative result rather than substituting or
loosening the matching definition.

Representative targets that remain missing:

| Source target | T2S target | Pinyin | Normalized candidate prefix |
| --- | --- | --- | --- |
| 而淚 | 而泪 | `erlei` | 二類, 二壘, 餌雷, 而, … |
| 遮挽時 | 遮挽时 | `zhewanshi` | 折彎是, 折彎, 遮挽, 這碗, … |
| 如輕煙 | 如轻烟 | `ruqingyan` | 如輕言, 乳清, 入情, 如, … |
| 蒸融 | 蒸融 | `zhengrong` | 整容, 崢嶸, 正, 整, … |

## Previously Present Targets Lost

For identical source spans, 1,366 baseline-present targets become missing after
T2S. Representative cases show the unchanged Luna candidate output retaining
Traditional forms:

| Source target | T2S target | Candidate prefix |
| --- | --- | --- |
| 時候 | 时候 | 時候, 事後, 侍候, … |
| 楊柳 | 杨柳 | 楊柳, 洋流, 樣, … |
| 聰明 | 聪明 | 聰明, 從命, 從, … |
| 我們 | 我们 | 我們, 我, 喔, … |
| 他們 | 他们 | 他們, 它們, 她們, … |

## Interpretation

Corpus-side T2S normalization does not improve coverage under the current
unchanged Luna Pinyin configuration. It produces a strong exact-script mismatch
for many otherwise retrievable targets. Because corpus and candidate scripts
are not aligned, Phase 4B.5 cannot answer how a real Simplified Chinese pipeline
performs. This result is specific to this schema, dictionary snapshot,
normalization direction, and exact-equality coverage definition; it is not a
general claim that script normalization is harmful.

No candidate strings were normalized, no candidates were reordered, and no
coverage cases were manually altered.

## Provenance and Reproducibility

- Phase 4A text files retained their original SHA-256 checksums.
- Normalized works occupy a separate directory and have their own checksums.
- Each normalized work manifest links source and normalized paths/hashes and
  records changed positions and OpenCC configuration.
- Each interaction records the original target at its source span and links
  both corpus manifests.
- Repeated full runs produce deterministic normalized interaction records for
  fixed inputs and configuration.

Run:

```bash
.venv/bin/python -m normalization.phase_04b5
```

The detailed machine-readable comparison is stored in
`data/processed/interactions/zhu_ziqing_t2s/coverage_comparison.json`.

## Deferred Questions

- Whether the benchmark should retain canonical Traditional source text and
  current Luna candidates.
- Whether a deliberately Simplified-output candidate schema should be tested as
  a separate controlled condition.
- Whether candidate-side normalization or variant-aware matching belongs in a
  retrieval analysis, and how it would affect candidate identity/rank.
- How normalized re-segmentation should be controlled in a paired experiment.

These questions are not implemented here. Phase 4C remains deferred.
