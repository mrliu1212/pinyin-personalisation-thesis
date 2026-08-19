# PV1/PV2 vs EM-1 Same-Surface Comparison Addendum

**Date:** 2026-08-19
**Status:** CLOSED POST-HOC EXPLANATORY AUDIT
**Frozen PV result changed:** No
**Frozen EM-1 result changed:** No
**New Test tuning:** No
**New PinyinGPT Test inference:** No

## 1. Purpose

This audit connects the earlier Personal Vocabulary PV1/PV2 experiments with
the later EM-1 recovery experiment.

PV1 and EM-1 both perform personal candidate recovery, but they were developed
at different stages and do not use exactly the same recovery scoring procedure.

The purpose of this audit is to determine how their behaviour differs on the
same Test population without reopening either frozen experiment.

## 2. Same-Surface Population

The comparison is restricted to the three authors used by EM-1:

- Etinjat
- Re_spectators
- breaddddd

The population contains:

- Full+Short
- H5000
- 3,000 Test rows
- 1,000 rows per author

Exact row alignment was verified:

```text
PV.condition_id == EM1.row_id
```

The Generic and Frequency rank vectors were also verified to be exactly equal
between the two artifacts:

```text
G_rank_exact_equal = True
F_rank_exact_equal = True
```

The three-author F-H5000 Macro-author Top-1 recomputed from the original PV
artifact is:

```text
0.810666666667
```

which exactly matches the independently recorded EM-1 F baseline.

## 3. Method Relationship

PV1 uses the bounded H5000 Personal Vocabulary state and selects at most one
personal-only candidate because the frozen Dev-selected value is:

```text
Kpv = 1
lambda_pv = 4.0
```

For a personal-only candidate, PV1 uses the Generic boundary approximation:

```text
Score_PV1(c) = B + lambda_pv * F(c)
```

where `B` is the minimum normalized Generic score on the current Generic
candidate surface.

PV1 does not obtain a new exact PinyinGPT Generic score for the injected
personal-only candidate.

EM-1 later formalises recovery separately from reranking. Recovered candidates
are scored by the frozen PinyinGPT backend using exact candidate scoring before
being combined with Frequency evidence.

Therefore, EM-1 is not merely a renamed PV1 experiment. It changes the
recovered-candidate scoring procedure and, in a small subset of cases, the
selected recovery candidate.

## 4. Same-Surface Macro Top-1

| Method | Macro Top-1 |
| --- | ---: |
| G | 0.776000 |
| F | 0.810667 |
| PV1 | 0.804000 |
| PV2 | 0.803333 |
| EM1-R | 0.777000 |
| EM1-R+F | 0.810333 |

Relative to PV1:

```text
PV1         = 0.804000
EM1-R+F     = 0.810333
difference  = +0.006333
```

This is a +0.6333 percentage-point Macro-author Top-1 difference.

## 5. Paired PV1 to EM1-R+F Result

Across the aligned 3,000 Test rows:

```text
helped            = 26
harmed            = 7
unchanged_correct = 2405
unchanged_wrong   = 562
net_help          = +19
```

The +19 net paired difference exactly corresponds to the +0.6333 percentage-
point Macro Top-1 difference.

## 6. Generic-Missing Recovery Comparison

There are 132 Generic-missing rows on the three-author surface.

PV1:

```text
Top-10 recovered = 24
Top-3 recovered  = 23
Top-1 recovered  = 16
```

EM1-R+F:

```text
Top-10 recovered = 21
Top-3 recovered  = 15
Top-1 recovered  = 10
```

PV1 therefore performs more aggressive recovery of originally missing Gold
targets, while EM1-R+F achieves better overall Top-1 accuracy.

Recovery quantity alone is therefore not sufficient to explain overall
personalisation quality.

## 7. Provenance

The EM-1 evaluation records the following hashes:

```text
Generic predictions SHA256
764db39887f3db04b913d1739d9dbd46295f0e46e5a2bffa649f1563b56ee4e2

Recovered candidate scores SHA256
5151d462bd3594fe63d81b244083ec557886d2018f286a92a105c334b307185d

Test states SHA256
2912d32b8cd88843e825cb5592dfbc0a06e88e4a58831c632a126d2b8452b061
```

The recovered-score hash resolves locally to:

```text
results/personalisation/external_memory/
em1_recovered_scores_test/recovered_candidate_scores.jsonl
```

The Test-state hash resolves exactly to the earlier PV state artifact:

```text
C:\Users\chiar\Desktop\LBH\thesis-personalisation\
results\personalisation\personal_vocabulary_h5000\
cache\test_states.jsonl
```

This verifies that EM-1 reused the same frozen PV Test state surface.

## 8. Candidate Identity Audit

EM-1 exact-scored 306 recovery rows.

Comparing the EM-1 recovered candidate against the first PV1 personal-only
candidate from the same frozen state:

```text
same candidate      = 290
different candidate = 16
identity rate       = 0.9477124183
                    = 94.77%
```

Therefore PV1 and EM-1 do not use exactly the same recovery candidate in every
case.

All 16 candidate-identity differences occur for Etinjat in this audit.

Examples include:

```text
PV1 禘 -> EM1 递
PV1 瀴 -> EM1 迎
PV1 嚚 -> EM1 银
```

EM-1 records some selected candidates with raw personal rank 2 or 3.

Therefore the PV1 to EM-1 comparison must not be described as a pure
approximate-score-versus-exact-score substitution.

## 9. Candidate-Identity Split

The paired Top-1 difference was decomposed according to whether PV1 and EM-1
selected the same recovered candidate.

### Same candidate

```text
rows              = 290
helped            = 24
harmed            = 7
unchanged_correct = 161
unchanged_wrong   = 98
net_help          = +17
```

### Different candidate

```text
rows              = 16
helped            = 1
harmed            = 0
unchanged_correct = 10
unchanged_wrong   = 5
net_help          = +1
```

### No EM-1 recovery score

```text
rows              = 2694
helped            = 1
harmed            = 0
unchanged_correct = 2234
unchanged_wrong   = 459
net_help          = +1
```

The complete accounting is:

```text
+17 same-candidate
 +1 different-candidate
 +1 no-recovery-score
----------------------
+19 total
```

Thus 17 of the 19 net Top-1 improvements occur when PV1 and EM-1 use the same
personal candidate.

This is approximately 89.5% of the total net improvement.

## 10. Interpretation

The audit strongly suggests that EM-1's safer recovered-candidate scoring is the
main source of its improvement over the earlier PV1 prototype.

The evidence is:

1. the Test rows are exactly aligned;
2. G ranks are exactly identical;
3. F ranks are exactly identical;
4. EM-1 reuses the same frozen PV Test state;
5. 94.77% of exact-scored recovery cases use the same recovered candidate;
6. 17 of the total 19 net Top-1 improvements occur inside this same-candidate
   subset.

At the same time, PV1 recovers more originally missing Gold targets, while
EM1-R+F produces better overall Top-1.

This is consistent with the interpretation that the PV1 Generic-boundary
approximation promotes personal candidates more aggressively, whereas the
later exact PinyinGPT candidate scoring provides a more conservative Generic
plausibility signal.

## 11. Interpretation Boundary

This audit does not prove that exact PinyinGPT scoring is the sole causal source
of the +0.6333 percentage-point difference.

Sixteen exact-scored recovery rows use a different candidate, and other merge or
normalisation implementation details may also differ between stages.

The correct conclusion is therefore:

> The paired same-surface evidence strongly suggests that exact recovered-
> candidate scoring is the principal source of EM-1's safer overall behaviour,
> rather than candidate substitution, but the result is not a formal
> single-variable causal ablation.

## 12. Reproducibility

Audit runner:

```text
experiments/external_memory/em1_pv_same_surface_audit.py
```

Generated audit outputs:

```text
results/personalisation/external_memory/em1_pv_same_surface_audit/
    summary.json
    rows.jsonl
    candidate_identity_split.json
```

These generated outputs are LOCAL-ONLY evidence and are not committed as normal
Git source files.

The audit performs:

- no parameter tuning;
- no new model training;
- no new Test PinyinGPT inference;
- no modification of PV1/PV2 frozen results;
- no modification of EM-1 frozen results.

## 13. Final Status

**CLOSED.**

This addendum explains the relationship between the earlier PV1/PV2 candidate-
recovery prototype and the later EM-1 recovery mechanism.

The next research stage remains EM-3.
