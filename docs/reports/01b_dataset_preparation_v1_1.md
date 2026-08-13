# Deep Author Dataset Preparation V1.1

## 1. Why V1.1 was needed

Manual review of Deep Author Dataset V1 found repeated SCP website and template
material in some rendered page text. Examples included authorship information,
image attribution, related-reading links, credit-module controls, and embedded
credit-module JavaScript or CSS. The retained works also mixed Traditional and
Simplified Chinese. Finally, V1 could retain punctuation and other non-Han text
inside stored context, although the simulated Pinyin compositions themselves
were intended to represent Chinese-character input.

V1.1 corrects only those representation issues. The six authors, 2014–2021
window, authorship checks, included works, Pinyin convention, Short definition,
Multi definition, and interaction identity hierarchy remain unchanged. The raw
V1 snapshot and V1 report remain preserved.

## 2. Metadata cleaning

A cross-author audit inspected source, rendered text, and V1-cleaned text for
representative works. Automatic deletion was limited to credit/template prefixes
confirmed by explicit `credit:start` and `credit:end` structures in the preserved
Wikidot source. The corresponding rendered prefix was located using the first
stable author-text line after the source block. This also removes leaked modal
JavaScript and CSS when they belong to the same confirmed credit structure.

The rule removed 63 blocks and 43,037 rendered characters. Similar-looking text
outside a confirmed structure was not deleted. Six detected blocks in four works
were retained and recorded for human review. Removed samples are stored for each
author where the confirmed structure occurred.

## 3. Simplified-Chinese normalization

After metadata cleaning, retained text was converted with
`opencc-python-reimplemented` 0.1.7 using the deterministic `t2s` configuration.
The immutable raw source was not changed. Each processed work stores the original
cleaned form, the final simplified form, and a SHA-256 hash for each. Across the
282 works, 89 works changed and 46,076 character positions changed.

Before conversion, the cleaned corpus contained 205 Simplified, 58 mixed, and 19
Traditional works. After `t2s`, the classifier reported 271 Simplified and 11
mixed works. The remaining mixed labels reflect content the standard `t2s`
conversion does not classify as purely Simplified; no manual lexical rewriting
was introduced.

## 4. Han-only IME boundaries

Segmentation now begins from maximal consecutive Han spans after metadata
cleaning and `t2s` conversion. Punctuation, brackets, dashes, Latin text, digits,
emoji, whitespace, and other symbols are excluded from model-facing context and
Gold. They are hard boundaries, rather than deleted bridges. Thus:

```text
这个方法非常实用，而且没有那么复杂。
这个方法非常实用 | 而且没有那么复杂
```

A Multi composition cannot contain `实用而且没有` because that would cross
the comma. The corpus contains 127,908 non-Han boundary sequences. These include
113,468 sequences containing punctuation, 24,346 containing Latin characters or
digits, 33,472 containing whitespace, and 2,027 containing other non-Han symbols;
categories can overlap.

Jieba 0.42.1 and pypinyin 0.55.0 remain unchanged. Initial Pinyin still uses the
first letter of each full syllable: `shi zhong chi` becomes `s z c`. Context is
still bounded to 512 stored Han characters. This is a storage limit, not a claim
about PinyinGPT tokenizer length.

## 5. Dataset changes

| Measure | V1 | V1.1 |
|---|---:|---:|
| Included works | 282 | 282 |
| Han characters | 1,016,100 | 1,010,544 |
| Segmented Han tokens | 883,512 (all V1 segments) | 596,958 |
| Short interactions | 601,393 | 596,650 |
| Multi interactions | 472,639 | 468,901 |
| Total interactions | 1,074,032 | 1,065,551 |
| Alignment failures | 26 | 63 |

The higher recorded failure count follows from checking rare unsupported Han
characters in both Short and eligible Multi candidates after the new span-based
segmentation. Failures remain excluded and are listed rather than silently
corrected. No exact raw duplicates, final cleaned-text duplicates, or duplicate
work IDs were found.

## 6. Quality checks

The audit directory contains per-author and per-work statistics, removed metadata
samples, retained uncertain blocks, script-conversion samples, alignment failures,
duplicate checks, checksums, and a manual interaction sample. Review fields are
blank so the decisions can be checked independently. Every generated Gold and
stored normalized context was programmatically confirmed to contain only Han
characters. Interaction records retain raw-source offsets, processed offsets,
boundary-span identity, the raw source hash, and both processed text hashes.

## 7. Effect on later evaluation

T1 should use V1.1 rather than V1. V1 remains a recoverable historical checkpoint
at `deep-author-dataset-preparation-v1`. No PinyinGPT inference, candidate ranking,
T1 result, personalisation, T2, T3, or RT4 work was run during this correction.

## 8. Limitations

The authors remain proxy users, Pinyin remains reconstructed, and composition
boundaries remain simulated. Conservative structural cleaning deliberately keeps
ambiguous blocks for human review, so V1.1 does not claim perfect removal of all
possible site metadata.
