# Phase 4B.6 — Rime Script Alignment Analysis

## Status

Phase 4B.6 — Simplified Corpus + Simplified Rime: **COMPLETE**

This phase measures Base retrieval coverage only. It does not run Phase 4C or
modify the personalisation model/evaluation framework.

## Motivation

Phase 4B.5 converted the corpus to Simplified Chinese but left Rime in its
default Traditional/mixed output mode. Its large coverage reduction therefore
measured script mismatch, not Simplified IME quality.

Phase 4B.6 asks a fairer question:

> How does candidate coverage change when both the corpus and candidate
> generator use the same Simplified Chinese script convention?

## Rime Configuration

The existing `luna_pinyin` schema declares:

- engine filter: `simplifier@zh_hans`;
- schema option: `zh_hans`;
- filter OpenCC configuration: `t2s.json`.

The repository adapter now enables `zh_hans` through librime's session option
API after selecting `luna_pinyin`. Each experiment uses a fresh isolated
temporary Rime user directory, so no saved user preference or learned state is
required. Candidate conversion happens inside the Rime engine filter; candidate
strings are not converted after retrieval.

Direct verification with identical Pinyin input:

| Pinyin | Default Luna first candidate | `zh_hans` first candidate |
| --- | --- | --- |
| `weishenme` | 爲什麼 | 为什么 |
| `women` | 我們 | 我们 |
| `shihou` | 時候 | 时候 |

The machine-readable mode declaration is
`config/rime/simplified_candidate_mode.json`.

## Coverage Comparison

| Setting | Corpus | Rime output | Interactions | Top-1 | Top-3 | Top-5 | Top-10 | Missing |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 4B | Original Traditional/mixed | Default Luna Traditional/mixed | 4,531 | 64.67% | 76.91% | 79.50% | 80.53% | 19.47% |
| Phase 4B.5 | OpenCC T2S | Default Luna Traditional/mixed | 4,691 | 32.85% | 39.14% | 40.55% | 41.04% | 58.96% |
| Phase 4B.6 | OpenCC T2S | Luna `zh_hans` | 4,691 | 72.12% | 85.44% | 88.23% | 89.11% | 10.89% |

Phase 4B.6 contains 3,383 Top-1, 4,008 Top-3, 4,139 Top-5, and
4,180 Top-10 targets. There are 511 Top-10 misses.

This is candidate coverage, not personalised ranking performance. The higher
coverage demonstrates the importance of aligning scripts in this controlled
configuration; it does not establish general IME superiority.

## Candidate Script Distribution

Classification compares every complete candidate with OpenCC `t2s` and `s2t`.
Script-invariant strings are reported separately so totals remain exhaustive.

For 46,878 Phase 4B.6 candidate occurrences:

| Category | Occurrences | Rate | Unique candidates |
| --- | ---: | ---: | ---: |
| Simplified-only | 20,135 | 42.9519% | 7,392 |
| Traditional-only | 18 | 0.0384% | 3 |
| Mixed | 2 | 0.0043% | 2 |
| Script-invariant | 26,723 | 57.0054% | 7,621 |

The small non-Simplified residual consists of OpenCC/variant classifications
such as `昇`, `乾`, `於`, `乾图`, and `苧`; it is surfaced rather than silently
discarded.

For comparison, default Phase 4B candidate occurrences were 41.62%
Traditional-only, 0.82% Simplified-only, 0.10% mixed, and 57.46%
script-invariant.

## Interaction Delta

Phase 4B.6 uses exactly the Phase 4B.5 normalized corpus/segmentation, so it has
the same 4,691 interaction spans. Relative to Phase 4B:

- added interactions: 865;
- removed interactions: 705;
- net change: +160.

These changes are caused by OpenCC-before-Jieba segmentation changes, not by
the Rime `zh_hans` option. Examples include `鄉 / 下 → 乡下`,
`那別墅 → 那 / 别墅`, and `常如鏡子 → 常如 / 镜子`.

## Coverage Recovery

Relative to the script-mismatched Phase 4B.5 interaction set (identical spans):

- missing before: 2,766;
- recovered with engine-side `zh_hans`: 2,255;
- still missing: 511;
- no matching span: 0.

Recovered examples include `时候`, `杨柳`, and `聪明`, each returning to
Base rank 1 after the Rime engine emits Simplified candidates.

Relative to Phase 4B baseline missing targets:

- baseline missing: 882;
- recovered at an identical normalized span: 178;
- still missing at an identical span: 400;
- no matching normalized span after re-segmentation: 304.

Examples recovered from the baseline include `變為 → 变为`, `煙熏 → 烟熏`,
`遊踪 → 游踪`, and `因為 → 因为`, all at Base rank 1 in the aligned pipeline.

Representative remaining misses include `里算`, `流里`, `而泪`, `遮挽时`,
and `如轻烟`.

## Provenance and Safeguards

- Phase 4A raw/processed corpus checksums remain unchanged.
- Phase 4B and Phase 4B.5 interaction files remain unchanged.
- Phase 4B.6 occupies a new output directory.
- Each interaction retains original work/chronology/offsets, original target,
  Simplified target/context, Pinyin, ordered Simplified Rime candidates, and
  source/normalization/Rime configuration provenance.
- No final chronological split, wrong-user benchmark, or personalisation result
  is produced.

## Reproduction

```bash
brew install opencc librime
.venv/bin/python -m interactions.setup_rime
make rime-adapter
.venv/bin/python -m normalization.phase_04b5
.venv/bin/python -m normalization.phase_04b6
.venv/bin/python -m unittest discover -s tests -v
```

Detailed diagnostics are stored in
`data/processed/interactions/zhu_ziqing_simplified_rime/phase_04b6_comparison.json`.

## Remaining Decisions Before Phase 4C

- Select the coherent benchmark representation: canonical Traditional baseline
  or aligned Simplified pipeline.
- Decide whether the 18 Traditional-only and two mixed candidate occurrences
  require policy changes or simply documentation.
- Complete/review manual audit labels and remaining 511 misses.
- Prepare a second-author wrong-user control.
- Select a chronological history/test boundary, including partial-date policy.

Phase 4C remains deferred.
