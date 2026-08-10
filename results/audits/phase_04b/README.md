# Phase 4B Data-Quality Audit

This directory contains a read-only, fixed-seed audit of the existing 4,531
Zhu Ziqing interactions. It does not classify errors, change interactions, or
perform personalised evaluation.

Run from the repository root:

```bash
.venv/bin/python -m audits.phase_04b
```

The default seed is `40402` and the sample size is 100 per stratum. To state
the defaults explicitly:

```bash
.venv/bin/python -m audits.phase_04b \
  --input data/processed/interactions/zhu_ziqing/interactions.jsonl \
  --output-dir results/audits/phase_04b \
  --seed 40402 \
  --sample-size 100
```

Outputs:

- `polyphonic_flagged_sample.jsonl`: 100 flagged interactions;
- `polyphonic_unflagged_sample.jsonl`: 100 unflagged interactions;
- `top10_missing_sample.jsonl`: 100 interactions whose target is absent;
- `top10_missing_diagnostics.json`: complete factual missing-target aggregates;
- `audit_summary.md`: concise human-readable diagnostics;
- `audit_manifest.json`: input checksum and sampling configuration.

Each sample record retains the review fields requested for manual inspection,
including full raw context, derived context, source position, Pinyin and
syllables, flags, ordered candidates, and target rank/presence. Overlap between
the missing-target sample and either polyphonic stratum is intentional because
the three samples answer separate review questions.

Potential causes such as conversion ambiguity, segmentation, character forms,
rare vocabulary, or Top-10 truncation must be assessed by a human reviewer.

## Manual Review CSV

`manual_review.csv` contains the existing 300 deterministic sample records in
three 100-row groups. The four human-review columns are blank when prepared;
the preparation command does not resample or infer labels.

To reproduce the blank review sheet from the existing JSONL samples:

```bash
.venv/bin/python -m audits.phase_04b_manual_review prepare
```

Review the CSV in an editor that preserves UTF-8 CSV quoting. Full raw context
may contain line breaks, so do not split records by physical text lines.
Use only these values, leaving a cell blank until it has been reviewed:

- `pinyin_judgement`: `correct`, `incorrect`, `uncertain`
- `segmentation_judgement`: `reasonable`, `unreasonable`, `uncertain`
- `missing_cause`: `proper_name`, `rare_or_literary_vocabulary`,
  `traditional_or_variant_form`, `segmentation_problem`, `pinyin_problem`,
  `likely_rank_beyond_top10`, `other`, `uncertain`
- `notes`: free text

For `polyphonic_flagged` and `polyphonic_unflagged`, complete the Pinyin
judgement. For `top10_missing`, complete the segmentation judgement and missing
cause. You may use `notes` for supporting observations. Because the audit
strata intentionally overlap, the same interaction can occur in more than one
sample group; review each CSV row in its stated `sample_type`.

After manual review, print the requested aggregate distributions:

```bash
.venv/bin/python -m audits.phase_04b_manual_review summarize \
  --csv results/audits/phase_04b/manual_review.csv
```

The summarizer validates entered labels, counts blank cells separately, and
computes percentages using only manually labelled rows. It never fills labels
or infers a cause. To also save the aggregates as JSON, add:

```bash
--json-output results/audits/phase_04b/manual_review_summary.json
```
