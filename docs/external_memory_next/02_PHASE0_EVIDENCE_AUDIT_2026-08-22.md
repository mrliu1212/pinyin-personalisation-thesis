# 02 - Phase 0 Evidence Audit

Date: 2026-08-22

Status: **COMPLETE / TRAIN-FIT AND TRAIN-VAL ONLY**

## 1. What was done

The real Full+Short Clean3 Train-Fit and Train-Val manifests, Generic
predictions, Full RetunedFinal Stage-1 features, Stage-2 supports, selected
configuration, and selected predictions were inspected directly. Every input
was SHA256-checked before it was read. The audit runner rejects Test rows and
does not accept a Dev3000 path.

The runner also recomputed the frozen Full RetunedFinal metrics from the
selected Train-Val predictions as an initial baseline gate.

## 2. Why it matters

The proposed smoothing and learned-fusion studies depend on the actual support
regime. In particular, a shrinkage strength that is reasonable at `N=5` is not
equivalent to one at `N=100`, and a population prior is not useful if most
Personal-K5 candidates have no prior support.

## 3. Inputs and provenance

All input artifacts are local-only in the read-only
`thesis-context-compare` worktree.

| Input | Rows | SHA256 |
|---|---:|---|
| Clean3 Train-Fit | 144,526 | `547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6` |
| Clean3 Train-Val | 34,416 | `d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220` |
| Train-Val Generic predictions | 34,416 | `cf4ae382fa23e5ec1154bf28320d13ac1d6ca9600e9dcf8a6aa599600bc28eab` |
| Retuned Stage-1 features | 34,416 | `e311261cb0c5ea252ce593fdaa43ff87826d19eef440f39e2efc67ddc4310b13` |
| Retuned Stage-2 supports | 34,416 | `d413d02650a759c0a759e1845212e68b2d948c1a1d20fc88abfd89ea7973bc64` |
| Retuned selected predictions | 34,416 | `f3e902e5a9e7d25e62799b9abb719026c336381eacc42999d1e7edccf2731b22` |
| Retuned selected configuration | n/a | `3dc3fb908aeeaa853526ad71cf85de7400f47d261ed7c09acdd8197446f5fa3d` |

## 4. Data audit results

### Population and history

| Population | Rows | Agent Phage | Etinjat | breaddddd | History available |
|---|---:|---:|---:|---:|---:|
| Train-Fit | 144,526 | 55,926 | 32,906 | 55,694 | 108,837 |
| Train-Val | 34,416 | 13,741 | 8,030 | 12,645 | 26,077 |

Train-Val same-Pinyin visible-history count `N` has mean 32.66, median 5,
P75 27, P90 83, P95 196, P99 349, and maximum 378. Counts by predeclared
diagnostic bin are:

| N | 0 | 1 | 2 | 3-5 | 6-10 | 11-20 | 21-50 | 51-100 | >100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train-Val rows | 8,339 | 3,364 | 2,035 | 3,689 | 3,259 | 3,785 | 4,122 | 3,019 | 2,804 |

Thus 24.23% of Train-Val has no same-Pinyin history, while 50% has at most
five observations. This is a genuine sparse-evidence regime.

### Candidate surface and recovery

- Generic candidate count: mean 9.679; median and P99 are 10.
- Personal-K5 count: mean 0.202; P90/P95 are 1, P99 and maximum are 5.
- Rows with a personal injection: 3,556.
- Retuned Stage-1 source instances: 326,363 Generic-frequency and 6,736
  personal-recovery candidates.
- Generic Missing: 2,382 rows (6.921%).
- Generic-Missing rows whose gold is in Personal-K5: 620 (26.03% of missing).
- Per-author missing/recoverable counts are Agent Phage 404/162, Etinjat
  1,708/357, and breaddddd 270/101.

### Choice Share and concentration

Across 6,942 Personal-K5 candidate instances, raw Choice Share has mean
0.1735, median 0.0488, P75 0.1538, P90 0.6084, and P95/P99/max 1.0.
Entropy concentration has mean 0.6101, median 0.8722, and P75 through maximum
equal to 1.0. The frequent unit-valued tail combined with the low median `N`
supports testing the overconfidence hypothesis; it does not by itself prove
that smoothing improves ranking.

### Existing contextual supports

- Personal `P_NG`: mean 0.5122, median 0.4239, P75 1.0.
- Stage-2 NGram effective order: 24,495 at order 0, 6,724 at order 1,
  and 3,197 at order 2.
- NGram matched-history count: median 2, P75 9, P90 33, maximum 378.
- NGram top margin: mean 0.6879, median 1.0.
- BGE candidate-history count: mean 32.59, median 5, P75 26, P90 83.
- BGE top margin: mean 0.5939, median 0.8354.

These highly concentrated support distributions argue for reporting support
and margin strata in a learned model, not merely raw scores.

### Causal prior feasibility

For a Personal-K5 candidate on Train-Val:

| Prior | Candidate seen | Rows with any candidate seen | Median prior value | P90 |
|---|---:|---:|---:|---:|
| All-author Train-Fit `P(target|Pinyin)` | 5,729 / 6,942 (82.53%) | 3,145 / 3,556 | 0.00662 | 0.22414 |
| Other-author Train-Fit `P(target|Pinyin)` | 2,931 / 6,942 (42.22%) | 1,999 / 3,556 | 0 | 0.12 |

All Train-Fit interactions precede Train-Val. Therefore the all-author
Train-Fit prior is causal for this Train-Val ablation and has substantially
better coverage. The other-author-only prior is too sparse to be the primary
estimator. Even the all-author prior requires an explicit unseen policy.

The proposed conservative unseen policy is zero prior mass for an unseen
candidate/Pinyin pair. In the shrinkage feature this yields
`n_c / (N + alpha)`, shrinking unsupported personal evidence toward the
Generic decision boundary without inventing candidate support.

## 5. Frozen baseline check

The selected configuration was verified as:

```text
w_P=2, w_CS=6, w_E=4, lambda_N=6, lambda_B=6
```

Recomputed Train-Val Full RetunedFinal metrics:

| Metric | Value |
|---|---:|
| Macro-author Top1 | 0.7960049265502147 |
| Micro Top1 | 0.8249941887494189 |
| Top3 | 0.9120757787075778 |
| Top5 | 0.9307589493258950 |
| MRR@10 | 0.8713778728793548 |
| Missing@10 | 0.0519816364481636 |

Per-author Top1 is 0.8872716687 (Agent Phage), 0.6014943960 (Etinjat), and
0.8992487149 (breaddddd). The values match the frozen selected configuration
to an absolute tolerance of `1e-15`.

## 6. Prior-system evidence inspected

The underlying Initial-Pinyin, standardized comparison, Full transfer, EM2,
and EM3 records/runners were inspected in addition to the new row-level audit.
They establish the following design constraints:

- On Full Train-Val, zero-shot transferred Stage1 raised Macro Top1 from
  Frequency `.77688` to `.78852`, and fixed-surface Stage2 raised it to
  `.79537`. Recovery therefore changes availability while contextual evidence
  changes ordering.
- The transferred Conflict result exposed an arbitration failure: Stage1
  reached only `.15280` Macro Top1, Stage2 corrected it to `.22897`, but M1
  remained higher at `.25854` and Generic at `.32293`.
- EM2's task-native hidden representation improved retrieval-level evidence,
  but the unchanged M1 decision rule converted little of that retrieval gain
  into final ranking gain. The EM2 handoff explicitly identifies historical
  relevance/decision, rather than retrieval alone, as the remaining problem.
- In the standardized Train-Val pair-scored systems, M2, Hidden-M2, and EM3
  clustered at Macro `.77692`, `.77630`, and `.77715`. A more expensive
  candidate-aware scorer did not automatically dominate the transparent
  methods.

These findings justify testing candidate-level nonlinear evidence interaction
before a new representation. Historical Dev3000 values were read only as
already-observed provenance and are not used to choose features, grids, or
methods in this phase.

## 7. Reusable artifacts and limitations

The Train-Val Stage-1 feature and Stage-2 support artifacts are complete and
safe for arithmetic-only fixed-surface ablations. No reusable Full+Short
Train-Fit Generic candidate surface was found. Consequently, a learned ranker
must not be fitted yet: its training query groups and runtime feature surface
do not exist. Creating that surface is a separate, provenance-controlled GPU
generation step, not something that may be imputed from Train-Val.

The historical Stage-2 supports cover the frozen RetunedFinal Top10 surface,
not every possible Personal-K5 candidate. The first smoothing experiment must
therefore keep that surface fixed. If smoothing is promising, a later dynamic
surface experiment must explicitly compute any missing NGram/BGE supports.

## 8. Reproduction

From the isolated worktree:

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  -m experiments.external_memory_next.audit_phase0_evidence_v1 `
  --fit 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_v2\clean3_train_fit_v1.jsonl' `
  --val 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_v2\clean3_train_val_v1.jsonl' `
  --generic 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_v2\train_val_generic\predictions.jsonl' `
  --stage1 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_stage1_features.jsonl' `
  --stage2 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_stage2_supports.jsonl' `
  --predictions 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_selected_predictions.jsonl' `
  --config 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\selected_config.json' `
  --output-root '.\results\personalisation\external_memory_next\phase0_evidence_audit_v1'
```

Runner SHA256 is
`96598bd3e62e1eb864f1ec21cd5728b3992251432f3bb647c84c8006d0c26f04`.
Current generated outputs:

- `audit.json`: `10561f524884cdc4f27e5bc32d3c770f3f789cf0433b1eb768ac50ccef868db1`
- `artifact_checksums.json`: `2e3b2d57c733e72f2c557bd248fc35f7af3bc781e05a36963b5a60b16d854f09`

## 9. Current decision

Phase 0 supports a controlled all-author Train-Fit empirical-Bayes smoothing
ablation over strengths spanning the observed sparse-history regime. It does
not support starting learned fusion until a causal Train-Fit Generic candidate
surface exists. Dev3000 and Test remain unused.
