# Phase 4A Corpus Summary

## Status

Phase 4A — Real Author Corpus Preparation: **COMPLETE**

This is a corpus-level outcome record. No personalisation or candidate-ranking
performance was measured.

## Source and Acquisition

- Author: 朱自清 (`zhu_ziqing`)
- Source: Chinese Wikisource
- Interface: MediaWiki Action API `action=parse`
- Requested data: canonical rendered HTML, wikitext, and revision ID
- Retrieval timestamp: `2026-08-10T16:47:01.529036+00:00`
- Raw storage: immutable JSON responses keyed by catalog work, revision, and
  response hash
- Provenance: source URL, request URL, page ID, revision ID, retrieval timestamp,
  and SHA-256 checksum in the acquisition/processed manifests

## Included Works and Chronology

| Order | Work | Chronology | Precision | Chinese characters |
| ---: | --- | --- | --- | ---: |
| 1 | 匆匆 | 1922-03-28 | day | 545 |
| 2 | 槳聲燈影裏的秦淮河 | 1924-01-25 | day | 4,935 |
| 3 | 背影 | 1925-10 | month | 1,146 |
| 4 | 阿河 | 1926-01-11 | day | 4,069 |
| 5 | 荷塘月色 | 1927-07 | month | 1,192 |
| 6 | 給亡婦 | 1932-10-11 | day | 2,441 |
| 7 | 春 | 1933-07 | month | 652 |

Included works: **7**

Total Chinese characters: **14,980**

No final history/test boundary was selected.

## Excluded Works

| Work | Reason |
| --- | --- |
| 獨自 | Poetry; the pilot is restricted to prose. |
| 《背影》序 | Book paratext rather than standalone prose. |
| 背影（散文集） | Collection/container page; component works require individual provenance. |
| 歐遊雜記 | Collection/container page; component works require individual provenance and date review. |

Excluded works: **4**

## Data-Quality Observations

- The seven included works have usable, explicitly sourced chronology.
- `背影`, `荷塘月色`, and `春` have month rather than day precision.
- `荷塘月色` has no reliable first-publication date in the selected Wikisource
  page metadata; this is explicitly flagged even though its work chronology is
  known.
- Work lengths are uneven: the two longest works contribute a large share of
  the pilot corpus.
- There is a substantial chronological gap between July 1927 and October 1932.
- Cleaned text retains canonical source script, wording, punctuation, unusual
  vocabulary, and authorial date/location colophons.

## Issues Carried into Phase 4B

- Review whether seven works provide enough chronological coverage for a pilot
  history/test split.
- Select and prepare a second author before a wrong-author control is possible.
- Define how text becomes evaluation interactions without introducing Pinyin,
  segmentation, or candidate-source leakage.
- Decide how missing candidate coverage will affect eligible interactions.
- Perform an additional scholarly review of transcription quality and date
  evidence before treating the corpus as a final benchmark.
