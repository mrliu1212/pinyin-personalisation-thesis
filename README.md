# Transparent, User-Controllable Pinyin Personalisation

## Current Phase

Phase 4A — Real Author Corpus Preparation

## Current Objective

The current repository builds a reproducible, provenance-preserving pilot
corpus of chronological Chinese prose by 朱自清 (Zhu Ziqing). This prepares the
real author data needed for a later benchmark; it does not yet create Pinyin
interactions or evaluate candidate ranking.

The eventual Phase 4 comparison is:

```text
earlier writings by Author A -> personalisation history
later unseen writings by Author A -> candidate-ranking evaluation
```

Author A's own earlier writing will eventually be compared with Base ranking
and history from the wrong author.

## Current Pipeline

```text
curated Wikisource work catalog
        ↓
MediaWiki Action API acquisition
        ↓
revision-pinned raw JSON + provenance
        ↓
conservative text extraction
        ↓
chronological processed manifest
        ↓
corpus diagnostics
```

Raw API responses are never overwritten during processing. Each response is
stored under its MediaWiki revision ID with a SHA-256 checksum. Cleaned text is
kept separately under `data/processed/` and no script conversion, vocabulary
modernisation, spelling correction, segmentation, or Pinyin generation is
performed.

## Why This Phase

Real-data evaluation is meaningful only if the history/test chronology and text
provenance can be audited. Phase 4A therefore establishes which pages are
eligible prose works, preserves reliable date precision and uncertainty,
records exclusions, and removes only identifiable Wikisource presentation
artifacts. It deliberately shows the available chronology before any
train/test boundary is selected.

## Pilot Corpus

The pilot author is 朱自清. The curated included works are:

1. `匆匆` — 1922-03-28
2. `槳聲燈影裏的秦淮河` — 1924-01-25
3. `背影` — 1925-10 (month precision)
4. `阿河` — 1926-01-11
5. `荷塘月色` — 1927-07 (month precision)
6. `給亡婦` — 1932-10-11
7. `春` — 1933-07 (month precision)

The curated catalog also records excluded poetry, preface, and collection pages
with explicit reasons. Source titles, URLs, page IDs, revision IDs, retrieval
timestamps, date bases, and checksums are retained in machine-readable JSON.

## How to Run

Run all Phase 1–4A tests from the repository root:

```bash
python3 -m unittest discover -s tests -v
```

Fetch or update the curated pilot pages from the Chinese Wikisource MediaWiki
API:

```bash
python3 -m corpus.acquire
```

The command requests each catalog page, stores a new file when Wikisource has a
new revision, and updates `data/raw/authors/zhu_ziqing/acquisition_manifest.json`.
Existing revision-pinned raw files are not overwritten.

Prepare cleaned text and the chronological processed manifest:

```bash
python3 -m corpus.prepare
```

Print corpus diagnostics:

```bash
python3 -m corpus.diagnostics
```

The diagnostic output reports included and excluded counts, total and per-work
Chinese character counts, chronological order, exclusion reasons, and missing
or uncertain date metadata.

## Data Layout

```text
data/
├── manifests/
│   └── zhu_ziqing_works.json
├── raw/authors/zhu_ziqing/
│   ├── acquisition_manifest.json
│   └── <work_id>__rev_<revision_id>__sha256_<digest>.json
└── processed/authors/zhu_ziqing/
    ├── manifest.json
    └── <work_id>.txt
```

## Current Limitations

- The pilot contains one author and seven included prose works.
- `荷塘月色` has usable month-level chronology but no reliable first-publication
  date in the selected page metadata.
- Several dates have only month precision.
- Wikisource metadata and transcription quality still require scholarly review
  before a final benchmark is claimed.
- Exact train/test boundaries have not been selected.
- No Pinyin conversion, segmentation, candidate generation, or author-ranking
  evaluation exists in Phase 4A.

## Next Planned Step

Phase 4B will decide how eligible corpus text is converted into auditable
Pinyin-IME evaluation interactions and how candidate availability is checked.
That work must preserve the chronology established here and will not begin
automatically.

## Phase Documentation

- Phase design history: [`docs/phases/`](docs/phases/)
- Completed phase outcomes: [`results/phases/`](results/phases/)
- Repository workflow: [`docs/WORKFLOW.md`](docs/WORKFLOW.md)

The accepted Phase 3 snapshot remains tagged `phase-03`. No Phase 4 tag has
been created.
