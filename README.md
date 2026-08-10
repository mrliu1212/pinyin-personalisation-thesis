# Transparent, User-Controllable Pinyin Personalisation

## Current Phase

Phase 4B.7 — Final Data Quality Audit

## Current Objective

The current repository prepares deterministic human-review samples from the
final Phase 4B.6 benchmark. The goal is to check whether remaining problems are
dominated by contextual Pinyin errors, segmentation, residual script mismatch,
or Base candidate coverage before any personalisation experiment begins.

Phase 4B.7 does not assign labels automatically and does not start Phase 4C.

## Final Benchmark Representation

The accepted data representation under review is:

- OpenCC Simplified Chinese corpus;
- unchanged Jieba lexical segmentation and pypinyin generation;
- Luna Pinyin with engine-side `zh_hans` Simplified output;
- 4,691 interactions;
- preserved original/normalized provenance and chronology.

Coverage is:

| Metric | Result |
| --- | ---: |
| Top-1 | 72.12% |
| Top-3 | 85.44% |
| Top-5 | 88.23% |
| Top-10 | 89.11% |
| Missing | 511 (10.89%) |

The earlier Phase 4B.5 script mismatch is resolved: corpus and Rime output now
share the Simplified convention. These remain Base coverage results, not
personalisation performance.

## Current Audit Pipeline

```text
unchanged Phase 4B.6 interactions
        ├── 2,664 polyphonic-flagged rows
        │       ↓ fixed-seed sample of 100
        │       ↓ human contextual pronunciation judgement
        │
        └── 511 Top-10 missing rows
                ↓ fixed-seed sample of 100
                ↓ human missing-cause judgement
                        ↓
              labelled-only aggregate summary
```

The fixed seed is `40407`. Sampling uses SHA-256 ordering of the seed, sample
name, and interaction ID, without replacement. The audit manifest records the
source checksum and selected IDs.

## Review Files

- [`polyphonic_review_sample.csv`](results/audits/phase_04b7/polyphonic_review_sample.csv)
- [`missing_review_sample.csv`](results/audits/phase_04b7/missing_review_sample.csv)
- [`review instructions`](results/audits/phase_04b7/README.md)
- [`audit manifest`](results/audits/phase_04b7/audit_manifest.json)

All human label fields are currently blank. No data-quality conclusion should
be drawn until manual review is completed.

## How to Run

Install/setup the existing Phase 4B.6 environment if needed:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-phase4b.txt
brew install opencc librime
.venv/bin/python -m interactions.setup_rime
make rime-adapter
```

Reproduce the blank deterministic audit samples:

```bash
.venv/bin/python -m audits.phase_04b7_manual_review prepare
```

After entering human labels, summarize them:

```bash
.venv/bin/python -m audits.phase_04b7_manual_review summarize
```

Optionally save a JSON summary:

```bash
.venv/bin/python -m audits.phase_04b7_manual_review summarize \
  --json-output results/audits/phase_04b7/manual_review_summary.json
```

Run the full test suite:

```bash
python3 -m unittest discover -s tests -v
```

## Manual Review Policy

For the polyphonic sample, judge whether the generated pronunciation is correct
for that occurrence in context: `correct`, `incorrect`, or `uncertain`.

For missing targets, choose only after inspection: `proper_name`,
`rare_or_literary_vocabulary`, `segmentation_problem`, `pinyin_problem`,
`candidate_coverage_problem`, `traditional_variant_residual`, `other`, or
`uncertain`.

Blank labels are reported separately. Percentages use manually labelled rows
only. The summarizer never infers or fills labels.

## Data Safeguards

- Phase 4B, Phase 4B.5, and Phase 4B.6 data remain unchanged.
- Rime, OpenCC, segmentation, Pinyin generation, personalisation, and evaluation
  code are outside this audit's mutation scope.
- All Phase 4B.7 outputs live under `results/audits/phase_04b7/`.
- The source interaction checksum is verified before sampling.

## Next Step

Complete the two human reviews and run the summarizer. Phase 4C remains deferred
until the audit result is reviewed and the benchmark data is explicitly
accepted.

## Project History

- Phase specifications: [`docs/phases/`](docs/phases/)
- Completed phase outcomes: [`results/phases/`](results/phases/)
- Audit outputs: [`results/audits/`](results/audits/)
- Workflow: [`docs/WORKFLOW.md`](docs/WORKFLOW.md)

Existing Git tags are unchanged. Tags are not created automatically.
