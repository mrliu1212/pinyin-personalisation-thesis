# Transparent, User-Controllable Pinyin Personalisation

## Current Phase

Phase 4B — Real Pinyin-IME Interaction Construction and Base Candidate Coverage

## Current Objective

This version converts the provenance-preserving Zhu Ziqing corpus prepared in
Phase 4A into reproducible lexical Pinyin-IME interactions. It measures whether
a real Base candidate generator contains each author's target often enough for
later reranking research. It does **not** evaluate personalisation.

The current research question is: can chronological author text be converted
into realistic interactions, and what is the target coverage of the Base
generator?

## Current Pipeline

```text
Phase 4A cleaned work + chronology + provenance
        ↓
Jieba lexical segmentation
        ↓
2–4 character all-Chinese targets
        ↓
tone-free full Pinyin (pypinyin)
        ↓
pinned Luna Pinyin schema through librime
        ↓
ordered Base Top-10 candidates
        ↓
traceable JSONL interactions + coverage diagnostics
```

Each interaction retains the work identity and date, source offsets, complete
preceding `raw_context`, a transparent 12-Chinese-character
`derived_context`, target, full Pinyin, candidate order, and target rank or an
explicit missing-target value. Rime exposes order here but not a meaningful
numeric score, so candidates store `base_rank` and `base_score: null` rather
than an invented score.

## Why This Phase

The later real-author benchmark can only rerank targets that the Base generator
retrieves. Phase 4B establishes that coverage, exposes conversion and
segmentation limitations, and makes each interaction auditable before any
history/test boundary or personalised evaluation is selected.

## Dependencies and Setup

The processing layer adds two pinned Python dependencies:

- `jieba==0.42.1` for established lightweight Chinese word segmentation;
- `pypinyin==0.55.0` for normalized tone-free full Pinyin.

The Base source is Homebrew `librime` with the `luna_pinyin` schema. The exact
official Rime data repository commits are locked in
`config/rime/sources.json`. On macOS, from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-phase4b.txt
brew install librime
.venv/bin/python -m interactions.setup_rime
make rime-adapter
```

`interactions.setup_rime` fetches only the recorded commits, deploys the schema
under ignored `data/rime/`, and records the local setup. `make rime-adapter`
builds the small command-line bridge under ignored `.build/`. Both are required
before real candidate generation.

## How to Run

Run all Phase 1–4B tests (ordinary tests are offline and use a fake candidate
adapter):

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Generate a small, 50-interaction pilot from `匆匆`:

```bash
.venv/bin/python -m interactions.generate \
  --work-id congcong \
  --max-interactions 50 \
  --output-dir data/processed/interactions/zhu_ziqing_pilot
```

Generate all eligible interactions from the seven included works:

```bash
.venv/bin/python -m interactions.generate \
  --output-dir data/processed/interactions/zhu_ziqing
```

Print the full Base coverage diagnostics:

```bash
.venv/bin/python -m interactions.diagnostics
```

Generation writes `interactions.jsonl` and a machine-readable `manifest.json`
containing preprocessing versions and policies, candidate-source configuration,
exclusion counts, and coverage. Re-running the full command replaces only the
derived Phase 4B outputs; it does not alter the Phase 4A corpus.

## Interaction Policy

- Jieba default-mode tokens are the lexical units.
- Targets must contain only Chinese characters and be 2–4 characters long.
- Single characters, units longer than four characters, punctuation, Latin
  text, numbers, and other non-Chinese units are counted by exclusion reason.
- Pinyin uses `pypinyin` normal style, strict mode, no tones, and concatenated
  syllables; abbreviated Pinyin is excluded.
- Potentially polyphonic characters are flagged for review; the generated
  reading is not claimed to be perfect.
- Candidate retrieval uses at most 10 entries in the exact order returned by
  librime. Missing targets remain in the dataset and coverage denominator.
- No final history/test split is selected and no future text is added to an
  earlier interaction's context.

## Data Layout

```text
data/processed/interactions/zhu_ziqing/
├── interactions.jsonl
└── manifest.json
```

The Phase 4A raw and cleaned corpus remains unchanged in its existing
directories.

## Current Limitations

- The corpus still contains one author and seven prose works.
- Jieba boundaries are automatic and may not always match realistic IME input
  units, especially for historical or literary wording.
- Automatic polyphonic-character readings need review; the flag is deliberately
  broad and is not a correctness judgment.
- Luna Pinyin and the canonical corpus both use Traditional Chinese here;
  candidate coverage depends on this schema, dictionary snapshot, and Top-10
  limit.
- Targets absent from Top-10 cannot be helped by a reranker unless retrieval is
  expanded later.
- The exact 12-character context representation is a preparation choice, not an
  optimized context model.
- A second author, wrong-user control, and final chronological history/test
  boundary have not been prepared.

## Next Planned Phase

Phase 4C will define the chronological boundary and evaluate Base versus
correct-user and wrong-user personalisation on real interactions. It must not
start until the Phase 4B dataset and coverage decisions are reviewed.

## Project History

- Phase design history: [`docs/phases/`](docs/phases/)
- Completed phase outcomes: [`results/phases/`](results/phases/)
- Repository workflow: [`docs/WORKFLOW.md`](docs/WORKFLOW.md)

Accepted snapshots are preserved by Git commits/tags. Existing tags, including
`phase-04a`, are not modified and this workflow does not create tags
automatically.
