# Phase 4B.7 Manual Data-Quality Review

This audit uses the finalized 4,691-interaction Phase 4B.6 benchmark. It does
not change interactions, Rime, Pinyin, segmentation, or personalisation.

## Prepare the Review Files

From the repository root:

```bash
.venv/bin/python -m audits.phase_04b7_manual_review prepare
```

The fixed seed is `40407`. Selection uses a SHA-256 ordering of the seed,
sample name, and interaction ID, without replacement. The audit manifest stores
the source interaction checksum and selected IDs.

Outputs:

- `polyphonic_review_sample.csv`: 100 of 2,664 flagged interactions;
- `missing_review_sample.csv`: 100 of 511 Top-10 misses;
- `audit_manifest.json`: source checksum, populations, seed, and selected IDs.

Both CSVs are UTF-8 and may be opened in a spreadsheet editor. Do not rerun
`prepare` after entering labels unless the completed files have been backed up.

## Polyphonic Review

For every row, judge:

> Is the generated pronunciation correct for this occurrence in context?

Fill `pinyin_judgement` with exactly one of:

- `correct`
- `incorrect`
- `uncertain`

The flag means pypinyin knows more than one possible reading for at least one
character. It does not itself mean the generated contextual reading is wrong.
Use `notes` for supporting observations; otherwise leave it blank.

## Missing-Target Review

Inspect the simplified target, generated Pinyin, context, and ordered Base
candidates. Fill `missing_cause` with exactly one of:

- `proper_name`
- `rare_or_literary_vocabulary`
- `segmentation_problem`
- `pinyin_problem`
- `candidate_coverage_problem`
- `traditional_variant_residual`
- `other`
- `uncertain`

Do not label a cause without human review. Use `notes` for evidence or nuance.

## Summarize Completed Labels

```bash
.venv/bin/python -m audits.phase_04b7_manual_review summarize
```

The command validates allowed values, reports blanks separately, and computes
percentages over manually labelled rows only. It never fills or infers labels.

Optionally save the aggregate result:

```bash
.venv/bin/python -m audits.phase_04b7_manual_review summarize \
  --json-output results/audits/phase_04b7/manual_review_summary.json
```
