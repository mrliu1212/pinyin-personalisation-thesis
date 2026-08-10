可以，下面这个版本适合作为 `results/phases/phase_04b7_data_quality_audit_summary.md`，保留论文需要的信息，但去掉过度解释。

直接复制：

```markdown
# Phase 4B.7 — Final Data Quality Audit

## Status

Phase 4B.7 is complete.

Human review was performed on:

- 100 polyphonic-flagged interactions;
- 100 Top-10 missing interactions.

No changes were made to:

- Phase 4B.6 interaction data;
- Rime configuration;
- Pinyin generation;
- personalisation model;
- evaluation framework.

Phase 4C has not started.

---

## Motivation

Phase 4B.6 established the final benchmark representation:

- OpenCC Traditional-to-Simplified corpus normalization;
- Luna Pinyin with engine-side `zh_hans` output;
- 4,691 interactions;
- Top-10 coverage: 89.11%;
- Missing targets: 511.

Before personalisation evaluation, a focused audit was performed to determine
whether remaining errors were mainly caused by:

- pronunciation errors;
- segmentation problems;
- vocabulary coverage limitations;
- script mismatch.

---

## Polyphonic Pronunciation Audit

A deterministic sample of 100 interactions was selected from the 2,664
polyphonic-flagged interactions.

Question:

> Is the generated pronunciation correct for this occurrence in context?

Results:

| Judgement | Count | Rate |
| --- | ---: | ---: |
| Correct | 97 | 97.00% |
| Incorrect | 3 | 3.00% |
| Uncertain | 0 | 0.00% |

### Conclusion

Polyphonic flags are not equivalent to pronunciation errors.

In the reviewed sample, most flagged interactions had contextually correct
Pinyin generation. Therefore, polyphonic ambiguity is not considered a dominant
data-quality issue for the Phase 4B.6 benchmark.

---

## Top-10 Missing Target Audit

A deterministic sample of 100 interactions was selected from the 511 Phase 4B.6
Top-10 misses.

Results:

| Cause | Count | Rate |
| --- | ---: | ---: |
| rare_or_literary_vocabulary | 26 | 26.00% |
| segmentation_problem | 26 | 26.00% |
| candidate_coverage_problem | 24 | 24.00% |
| pinyin_problem | 15 | 15.00% |
| proper_name | 9 | 9.00% |
| traditional_variant_residual | 0 | 0.00% |
| other | 0 | 0.00% |
| uncertain | 0 | 0.00% |

### Conclusion

Remaining Top-10 misses are caused by multiple factors rather than one dominant
failure mode.

The largest categories are:

- segmentation problems;
- rare or literary vocabulary;
- base candidate coverage limitations.

No reviewed miss was attributed to remaining Traditional/Simplified mismatch,
supporting the decision to use the Phase 4B.6 script-aligned benchmark.

Pinyin errors exist but represent a smaller subset of difficult missing cases.
The 15% value applies only to the sampled Top-10-missing subset and should not
be interpreted as the overall benchmark Pinyin error rate.

---

## Benchmark Decision

Based on this audit, the Phase 4B.6 benchmark is retained for Phase 4C.

The complete benchmark will be preserved rather than filtering difficult
examples.

Later personalisation evaluation should distinguish:

- end-to-end performance over all interactions;
- reranking performance when the target exists in the candidate list.

This distinction is necessary because a reranker cannot recover candidates that
are absent from the base candidate generator.

---

## Reproducibility

Source dataset:

Phase 4B.6 interaction JSONL

SHA-256:

```text
2d0df837fed3cf6b1a141b9f43677733671cf1f08cb72ca3b9e2f0f2f13f5077
```

Sampling:

- fixed seed: `40407`
- SHA-256 deterministic ranking
- without replacement

Commands:

```bash
.venv/bin/python -m audits.phase_04b7_manual_review summarize
```

Run tests:

```bash
python3 -m unittest discover -s tests -v
```

---

## Files

- `results/audits/phase_04b7/polyphonic_review_sample.csv`
- `results/audits/phase_04b7/missing_review_sample.csv`
- `results/audits/phase_04b7/manual_review_summary.json`
- `audits/phase_04b7_manual_review.py`

---

## Conclusion

Phase 4B.7 completes the final data-quality validation before personalisation
evaluation.

The Phase 4B.6 simplified and script-aligned benchmark is accepted as the basis
for Phase 4C.

Phase 4C has not started.
