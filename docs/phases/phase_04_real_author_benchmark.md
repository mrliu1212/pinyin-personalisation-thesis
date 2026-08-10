# Phase 4 — Real Author Benchmark

Status: Phase 4A complete; later Phase 4 sub-phases not started.

## Objective

Test whether an author's own earlier language use helps rank lexical choices in
later unseen writing better than Base ranking or history from a different
author:

```text
past writings of Author A -> personalisation history
future unseen writings of Author A -> candidate-ranking evaluation
```

The current implementation covers Phase 4A corpus preparation only. It does not
produce Pinyin interactions, candidates, or benchmark metrics.

## Phase 4A Scope / Work Performed

- Curated a prose-only pilot catalog for 朱自清 using the Chinese Wikisource
  author/work pages.
- Acquired 11 catalog pages through the public MediaWiki Action API using
  `action=parse` with rendered HTML, wikitext, and revision provenance.
- Stored canonical API responses by revision ID without requesting script
  conversion.
- Preserved page/revision identifiers, request URLs, retrieval timestamps, and
  raw/processed SHA-256 checksums.
- Conservatively removed Wikisource headers, navigation aids, page-number
  markers, and license containers while preserving wording and punctuation.
- Produced cleaned text for seven included prose works and explicit reasons for
  four exclusions.
- Produced a chronological machine-readable manifest and corpus diagnostics.

The acquisition uses the official MediaWiki Action API pattern and parse
operation documented at:

- <https://www.mediawiki.org/wiki/API:Action_API>
- <https://www.mediawiki.org/wiki/API:Parsing_wikitext>

The pilot source index is:

- <https://zh.wikisource.org/wiki/Author:%E6%9C%B1%E8%87%AA%E6%B8%85>

## Required / Verified Behaviours

- No author text is fabricated, paraphrased, or manually substituted.
- Every catalog entry records inclusion status, genre, chronology metadata,
  source page, and an exclusion reason when excluded.
- Raw source responses and cleaned text occupy separate directories.
- Processing verifies the raw checksum and does not modify raw files.
- Cleaning is deterministic and does not modernize, correct, or script-convert
  the work.
- Chronology retains day/month/year precision and explicitly represents missing
  or uncertain dates.
- Ordinary tests use local fixtures and require no network access.
- Diagnostics report included/excluded counts, corpus size, per-work size,
  chronology, exclusions, and date metadata needing review.

## Completion Criteria

- A reproducible command acquires the curated Wikisource pages and provenance.
- Included raw responses can be converted deterministically into separate
  cleaned files.
- The pilot manifest presents a usable chronology without inventing dates or
  selecting a premature train/test boundary.
- Excluded pages and reasons are machine-readable.
- All existing and Phase 4A tests pass without live-network requirements.

These Phase 4A criteria were met.

## Important Design Decisions

- The pipeline is a narrow Chinese Wikisource ingestion tool, not a general web
  crawler.
- The work catalog is manually curated metadata; author prose is always
  acquired from the recorded source revision.
- Canonical MediaWiki output is requested without a language-variant parameter,
  preventing silent Simplified/Traditional conversion.
- Composition chronology is preferred when explicitly stated. If it is not
  available, a reliable publication date may be used with that basis recorded;
  a date mentioned only as part of the narrative is not treated as composition
  evidence.
- Partial dates remain partial. No default month or day is inserted.
- Collection/container pages are excluded rather than concatenated because
  their component works need separate provenance and dates.
- The final real-data train/test boundary remains undecided until corpus
  coverage and chronology are reviewed.

## Manual Commands

```bash
python3 -m unittest discover -s tests -v
python3 -m corpus.acquire
python3 -m corpus.prepare
python3 -m corpus.diagnostics
```

## Known Limitations / Deferred Questions

- A single author and seven works are insufficient for a final generalisation
  claim.
- The pilot has uneven work lengths and a large chronology gap between 1927 and
  1932.
- `荷塘月色` lacks a reliable first-publication date in the selected page
  metadata, although its July 1927 work chronology is explicit.
- Page transcription quality and metadata bases need further scholarly review.
- Phase 4B must decide the interaction extraction unit, candidate-source
  coverage policy, treatment of punctuation/context boundaries, and missing
  candidate cases without leaking later works into history.
- A wrong-author control requires at least one separately prepared comparison
  author corpus; no second author has been selected.
- Pinyin generation, word segmentation, candidate generation, scoring changes,
  profiles/interventions, and real-data ranking metrics are deferred.

