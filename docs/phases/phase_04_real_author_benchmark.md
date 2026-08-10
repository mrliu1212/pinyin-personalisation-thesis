# Phase 4 — Real Author Benchmark

Status: Phase 4A complete; Phase 4B complete; Phase 4C deferred.

## Objective

Establish whether an author's earlier language use can later improve candidate
ranking in unseen writing relative to Base ranking and another author's
history. Phase 4 is split so corpus provenance, candidate retrieval coverage,
and personalisation evaluation are validated separately.

## Scope

### Phase 4A — Real Author Corpus Preparation

- Curated a prose-only pilot catalog for 朱自清 from Chinese Wikisource.
- Preserved revision-pinned raw responses, page and revision identifiers,
  retrieval timestamps, checksums, date bases, and date precision.
- Conservatively prepared seven cleaned prose works and recorded four excluded
  pages with reasons.
- Produced chronological corpus metadata and diagnostics without selecting a
  train/test boundary.

### Phase 4B — Interaction Construction and Base Coverage

- Segments the Phase 4A text into lightweight lexical units with Jieba.
- Retains all-Chinese targets of 2–4 characters and reports every excluded-token
  category.
- Converts targets to normalized tone-free full Pinyin with `pypinyin`, while
  surfacing potentially polyphonic cases for review.
- Preserves complete raw preceding context plus a deterministic short context
  suffix, work chronology, source offsets, and source provenance.
- Retrieves ordered Top-10 candidates from a real librime adapter using a
  reproducibly pinned Luna Pinyin schema.
- Records Base ranks without inventing numeric scores and reports target
  coverage, missing targets, candidate-list sizes, and counts by work/length.

### Phase 4C — Real Personalised Evaluation (Planned)

- Review interaction quality and determine an auditable chronological
  history/test boundary.
- Prepare a genuinely independent author for the wrong-user control.
- Evaluate Base, correct-user, and wrong-user conditions using the existing
  Phase 2 model and Phase 3 metrics.
- Retain strict earlier-than-test history filtering and explicit missing-target
  handling.

Phase 4C is not implemented by Phase 4B.

## Required / Verified Behaviours

- Author text, targets, and candidates are never fabricated or rewritten to
  improve coverage.
- Phase 4A raw and cleaned sources remain separate and unchanged by interaction
  generation.
- Each interaction is traceable to an author, work, date, source revision/file,
  and character offsets.
- Lexical targets are used rather than decomposing words character by character.
- Raw context is retained whenever a derived context is added.
- Candidate ordering is the order returned by librime; absent numeric scores
  remain null.
- Missing targets remain in both the interaction output and coverage
  denominator.
- Fixed source/configuration produces deterministic interaction records.
- Ordinary processing tests are offline; the real adapter is isolated behind a
  candidate-generator interface.
- Phase 2 scoring and Phase 3 evaluation code remain unchanged.

## Completion Criteria

### Phase 4A

- Reproducible, provenance-preserving acquisition and cleaning produce a usable
  seven-work chronology with explicit exclusions and uncertainties. Complete.

### Phase 4B

- Real corpus text is reproducibly converted through lexical extraction, full
  Pinyin, real Base candidates, traceable JSONL, and complete coverage
  diagnostics. Complete.
- A one-work pilot is inspected before full-corpus generation, and all prior
  plus new processing tests pass. Complete.

### Phase 4C

- A reviewed chronological split and second-author control exist.
- Base/correct-user/wrong-user results are evaluated without future leakage.
- Outcomes and limitations are recorded independently of the phase design.

These Phase 4C criteria remain open.

## Important Design Decisions

- Chinese Wikisource remains a narrow, auditable source rather than making the
  ingestion layer a general crawler.
- Phase 4A preserves canonical source script and partial dates; it does not
  modernize text or invent missing date components.
- Phase 4B uses established lightweight linguistic libraries instead of custom
  segmentation or Pinyin decoding.
- Full Pinyin is concatenated and tone-free. Jianpin remains out of scope.
- The candidate source is Luna Pinyin through librime, with exact Rime data
  commits recorded in a lock file and local deployment manifest.
- The Base adapter is separate from interaction construction so candidate
  sources can be reviewed or replaced without coupling research logic to Rime.
- Top-10 is an explicit retrieval limit. Rime iterator order is authoritative;
  no artificial probabilities are assigned.
- Exact context length, target-length policy, and dependency versions are
  recorded but not tuned on Zhu Ziqing coverage.
- No final chronological split is selected until interaction quality and
  coverage have been reviewed.

## Known Limitations / Deferred Questions

- Seven works from one author cannot support a generalisation claim.
- Automatic segmentation may produce targets unlike users' actual input units.
- Polyphonic conversion flags are intentionally conservative and require a
  review policy before Phase 4C.
- Top-10 misses may reflect schema vocabulary, orthographic variation,
  segmentation, Pinyin conversion, or retrieval depth; these causes are not yet
  separated.
- The derived exact-context suffix is transparent but not validated as the best
  representation for literary text.
- Whether Top-K should be expanded before evaluation remains unresolved.
- Phase 4C requires a second author selected and prepared under comparable
  provenance and chronology rules.
- The train/test date boundary and handling of month-precision chronology need
  explicit decisions before evaluation.
- Recency, confidence-aware scoring, semantic context, model-weight changes,
  profiles/interventions, UI, and commercial IME comparisons remain deferred.
