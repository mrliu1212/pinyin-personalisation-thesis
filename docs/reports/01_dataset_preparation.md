# Deep Author Dataset Preparation

## 1. Purpose

The thesis needs long text histories from identifiable individual writers to
study long-term, user-specific personalisation for contextual Pinyin candidate
ranking. The Deep Author Dataset uses writers as proxy users because each writer
has multiple modern Chinese works with preserved author and work boundaries.
The source has clearer public licensing and provenance than random novel mirrors,
and the selected 2014–2021 window largely predates widespread generative-AI
writing. These authors are not claimed to be real IME users. Their Pinyin input
is reconstructed from published Chinese text.

## 2. Data Source

The source family is the SCP Chinese Branch (SCP-CN). SCPPER-CN supplied
structured page metadata, creation dates, attributions, rendered text and
Wikidot source snapshots. The original SCP-CN URL was retained for every page.
SCP-CN material is generally distributed under CC BY-SA 3.0; page-specific
compatible notices remain part of the preserved raw source.

The six fixed primary authors were Re_spectators, MScarlet, Etinjat, Agent
Phage, QBLevi and breaddddd. Only pages first created from 1 January 2014 through
31 December 2021 were considered. Reserve authors were not used.

## 3. Download and Provenance

Pages were discovered through author-filtered SCPPER-CN queries. Each candidate
was recorded, including excluded pages. Plausible original works were checked
against exact page attributions. Immutable raw JSON bundles preserve search
metadata, page text and attribution records. The work manifest records source
URLs, dates, inclusion decisions, access dates, byte sizes and SHA-256 hashes.
No piracy mirror, repost site or unauthorized aggregator was used.

## 4. Text Cleaning

The pipeline used the SCPPER-CN rendered text field and applied NFC Unicode
normalization, uniform line endings and conservative whitespace cleanup. Known
interface-only lines were removed. It did not paraphrase, spell-correct,
modernize or otherwise rewrite prose. Paragraph and work boundaries remained
separate. Structural component pages, translations, co-authored pages and pages
without an original-Chinese tag were excluded and retained in the manifest.

The corpus contains 205 works classified as Simplified Chinese, 62 mixed works
and 15 Traditional Chinese works. No Traditional-to-Simplified conversion was
applied.

## 5. Chinese Segmentation

Cleaned text was segmented deterministically with Jieba 0.42.1 in default mode,
without a custom dictionary. Each token retains its index and absolute character
start and end offsets. For example, `这个方法非常实用` may be divided into
`这个 / 方法 / 非常 / 实用`. Punctuation remains available for sentence and
composition boundaries.

## 6. Pinyin Reconstruction

Every eligible all-Han target was converted with pypinyin 0.55.0, NORMAL style,
strict mode, lowercase output and no tones. One Chinese character must align to
one Pinyin syllable. For example, `实用` becomes `shi yong`. Official-style
initial input uses the first letter of each syllable, so `实用` becomes `s y`;
`sh`, `zh` and `ch` are not treated as initials. The 26 failed alignments were
recorded and excluded rather than corrected. This is reconstructed Pinyin, not
logged keystroke data.

## 7. Short and Multi-token Interactions

A Short interaction contains one eligible segmented token. A Multi interaction
starts at the same source offset and keeps the same preceding context, then
merges two to four consecutive eligible tokens. It stops at sentence-ending
punctuation, paragraph boundaries and non-Han interruptions. Thus a Short Gold
such as `实用` may have a Multi counterpart such as `实用而且没有`. Multi is a
controlled simulated composition that creates a harder, longer input condition;
it is not evidence of actual user composition behavior.

Context contains only text preceding the target. To match the accepted generic
backend's bounded contextual input and avoid duplicating entire work prefixes in
every record, at most the previous 512 source characters are stored. Absolute
context and target offsets plus the full cleaned work allow exact recovery.

## 8. Dataset Statistics

| Author | Works | Han characters | Short interactions | Multi interactions |
|---|---:|---:|---:|---:|
| Re_spectators | 16 | 62,232 | 36,660 | 29,629 |
| MScarlet | 34 | 151,542 | 92,541 | 75,055 |
| Etinjat | 84 | 208,443 | 122,480 | 89,314 |
| Agent Phage | 56 | 255,215 | 150,242 | 122,379 |
| QBLevi | 37 | 62,974 | 36,496 | 28,700 |
| breaddddd | 55 | 275,694 | 162,974 | 127,562 |
| **Total** | **282** | **1,016,100** | **601,393** | **472,639** |

The corpus has 883,512 segmented tokens and 1,074,032 total interactions. Exact
raw, cleaned-text and work-ID duplicates were absent after structural pages were
excluded.

## 9. How It Supports Evaluation

The processed dataset is the fixed input for the next stage, T1 Generic
Evaluation. It supplies source-traceable context, Pinyin and Gold targets without
running or tuning the model. Later work may use the same author histories for T2
long-history personal learning and T3 same-Pinyin context disambiguation, but
neither personalisation nor those evaluations are implemented here.

## 10. Limitations

Authors are proxy users rather than real IME users. Pinyin is reconstructed and
composition boundaries are simulated. Genre and domain differences may affect
author-specific results. The available public works determine corpus depth, and
mixed-script material remains unchanged pending research-design review.
