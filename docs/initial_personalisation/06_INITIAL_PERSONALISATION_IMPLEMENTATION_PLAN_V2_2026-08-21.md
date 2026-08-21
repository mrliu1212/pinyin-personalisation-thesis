# Initial Personalisation — Implementation Plan V2

Date: 2026-08-21
Status: execution plan before new-method implementation
Workspace target: `C:\Users\chiar\Desktop\LBH\thesis-initial-research`
Current standardized result root: `results\personalisation\initial_recovery_comparison_v1`

## 1. Protocol constraints

All new development in this plan must remain on standardized Clean3 Train-Fit / Train-Val until the method family is frozen.

Pipeline:

\[
\text{Clean3 Train}\rightarrow\text{Train-Fit/Train-Val}\rightarrow\text{method development}\rightarrow\text{PRE-DEV FREEZE}\rightarrow\text{Dev3000}\rightarrow\text{final freeze}\rightarrow\text{Test once}.
\]

Current rules:

- Dev3000 must not be used for method choice, threshold choice, K choice, lambda choice, context window choice, smoothing choice, or ablation selection.
- Test remains closed for current standardized development.
- Historical Test results are contextual evidence only and must not be mixed into the current Train-Val leaderboard.
- Same-author causal history must remain strictly prior.
- H5000 raw cap is applied before Initial/Pinyin matching.
- Do not overwrite historical outputs.
- Every result must have a reproducible command, input/output paths, hashes, protocol flags, and expected summary metrics.

---

## 2. Phase 0 — Freeze the diagnostic starting point

### Goal

Create a durable diagnostic package summarizing the observations that motivate the new method family.

### Inputs

Reuse current artifacts only:

- standardized Initial Train-Fit / Train-Val;
- B1 frozen candidate surface;
- B2 Frequency/PV1 predictions and grid;
- B2 harm audit;
- B3 EM1 exact scores and same-surface evaluation.

### Required outputs

Create a new versioned directory, for example:

`results\personalisation\initial_multisignal_v2\diagnostics`

with:

- `baseline_metrics.json`;
- `k_recovery_comparison.csv`;
- `pv1_rescue_harm_summary.json`;
- `em1_conservativeness_summary.json`;
- `subset_counts.json`;
- `provenance.json`;
- `reproduction.md`.

### Required checks

- verify row IDs align across B1/B2/B3;
- verify no Dev3000/Test paths are read;
- verify B1 candidate-surface SHA;
- verify current B2/B3 prediction hashes;
- verify formal Ambiguous/Conflict definitions are consistent with the evaluator used for the new method family.

### Important audit item

Before freezing the new diagnostic package, explicitly reconcile any earlier prep-level subset counts with the same-surface evaluator counts. Use one canonical definition/count for the new method family and document why.

---

## 3. Phase 1 — Distribution diagnostic before changing the ranker

### Goal

Test whether support, concentration, and candidate distinctiveness actually explain PV1 rescue/harm patterns before putting them into the model.

### New per-row statistics

For current Pinyin \(p\) under legal H5000 causal history, define:

\[
N_u(p)=\sum_{h\in H_t}\mathbf{1}[p_h=p],
\qquad
N_u(c,p)=\sum_{h\in H_t}\mathbf{1}[p_h=p,y_h=c].
\]

\[
P_u(c\mid p)=\frac{N_u(c,p)}{N_u(p)},
\qquad
\text{support\_rate}(p)=\frac{N_u(p)}{|H_t|}.
\]

For concentration diagnostics:

\[
M_u(p)=P_u(c_{(1)}\mid p)-P_u(c_{(2)}\mid p),
\]

\[
H_u^{norm}(p)=
\frac{-\sum_c P_u(c\mid p)\log P_u(c\mid p)}{\log |C_p|}
\quad (|C_p|>1).
\]

Record:

- `history_raw_n = |H_t|`;
- `support_n = N_u(p)`;
- `support_rate = support_n / history_raw_n` (diagnostic only);
- per-candidate `count`;
- per-candidate `share = count / support_n`;
- smoothed share variants;
- distribution entropy;
- normalized entropy;
- top1-top2 margin;
- number of distinct historical targets;
- winner count/share;
- candidate rank by personal count.

### Analysis populations

Compare distributions for:

- F -> PV1 rescue;
- F -> PV1 harm;
- unchanged correct;
- unchanged wrong;
- Conflict;
- Ambiguous non-Conflict;
- recoverable Generic-missing rows.

### Questions to answer

1. Do harms have lower margin or higher entropy than rescues?
2. Are rescues enriched for moderate-support/high-concentration personal targets?
3. Are some harms caused by extremely common majority targets rather than distinctive personal vocabulary?
4. Does support add useful information after concentration is known?
5. Is `support_rate` redundant with `support_n` on saturated H5000 rows?

### Output

`results\personalisation\initial_multisignal_v2\distribution_audit`

with:

- row-level feature table;
- grouped summary CSV/JSON;
- rescue-vs-harm quantiles;
- per-author summaries;
- no model tuning yet.

### Decision rule

Only promote entropy/margin from diagnostics into scoring if the audit shows meaningful separation relevant to rescue/harm or Conflict.

---

## 4. Phase 2 — Contribution A: Broad recovery surface

### Goal

Define the broader candidate surface for new-method development.

### Existing evidence

K=1,3,5 already show large Missing@10 differences with unchanged Top1 at lambda=4.

### New-method choice

Use K=5 as the candidate-capacity setting for the new method family, with K=1 and K=3 retained as ablations.

This is not a change to historical PV1; it is a new pre-Dev design choice.

### Implementation requirement

Prefer reusing the existing B1 candidate surface if it already contains enough ordered personal-only candidates to reconstruct K=5 exactly. Do not rerun Generic inference if no new Generic information is needed.

### Outputs

- frozen K5 candidate view or deterministic derivation manifest;
- K1/K3/K5 coverage table;
- exact source SHA and construction code SHA;
- proof Gold was not used for candidate admission.

---

## 5. Phase 3 — Contribution B: Personal Choice Distribution

### B1. Baseline variants

Compare on the same K5 surface:

1. current count-based personal score;
2. raw conditional share `P(c|p)`;
3. smoothed conditional share \(\widetilde P_u(c\mid p)\);
4. smoothed share plus any concentration modifier justified by Phase 1.

### Smoothing

Use the predeclared family:

\[
\widetilde P_u(c\mid p)=
\frac{N_u(c,p)+\alpha P_0(c\mid p)}{N_u(p)+\alpha}.
\]

Freeze a small search space before running results.

Recommended first grid:

- smoothing strength \(\alpha\) in a small predeclared set, e.g. `{1, 2, 5, 10}`;
- if a minimum-evidence eligibility rule is needed, predeclare a small set such as `{2, 3, 5}` and treat it as a separate ablation.

Prefer smoothing over a hard threshold whenever possible because it naturally distinguishes 2/2 from 7/7 from 90/100 without declaring low support universally bad.

### Concentration

If Phase 1 supports it, test separately:

- margin-based confidence;
- normalized-entropy confidence;
- margin + entropy.

Do not silently combine both without an ablation.

### Metrics

Report Overall / Ambiguous / Conflict / Ambiguous-non-Conflict, plus rescue/harm/net against both F and PV1 where meaningful.

---

## 6. Phase 4 — Contribution B2: Background-Normalised Personal Frequency

### Goal

Test whether a target that is very frequent for the current user is genuinely personal or simply common language behaviour for the same Pinyin.

This phase is now part of the planned contribution set. It is not treated as an optional side note, although its final weight must still pass stability and ablation checks.

### Construction

Using legal Train-Fit information only for fitted background statistics, estimate for every user \(v\):

\[
\widetilde P_v(c\mid p)=
\frac{N_v(c,p)+\alpha P_0(c\mid p)}{N_v(p)+\alpha}.
\]

For current user \(u\), let \(U_{p,-u}\) be the set of other users with legal support for \(p\). Compute the leave-one-user-out macro background:

\[
P_{bg,-u}(c\mid p)=
\frac{1}{|U_{p,-u}|}
\sum_{v\in U_{p,-u}}\widetilde P_v(c\mid p).
\]

Then compute candidate-level lift:

\[
L_u(c,p)=
\log\frac{\widetilde P_u(c\mid p)+\epsilon}
{P_{bg,-u}(c\mid p)+\epsilon}.
\]

### Important fairness rule

Do **not** pool raw cross-user counts. Every background user is normalized first, then users are macro-averaged so a prolific author cannot dominate simply by having more text.

### Required diagnostics

Record for every row/candidate:

- \(|U_{p,-u}|\), number of background users supporting the Pinyin;
- background probability \(P_{bg,-u}(c\mid p)\);
- personal probability \(\widetilde P_u(c\mid p)\);
- lift \(L_u(c,p)\);
- per-author lift distributions;
- cases with high personal frequency but lift near zero;
- rare personal targets with large positive lift.

Also test whether PV1 harms have lower personal lift than rescues.

### Scoring ablation

On the same K5 surface compare:

1. personal distribution only;
2. background lift only;
3. personal distribution + background lift;
4. no-background control.

Use an independently frozen weight grid, for example:

\[
\lambda_B\in\{0.25,0.5,1.0,2.0\}.
\]

### Stability rule

Do not make a strong background-frequency claim if the signal depends on one author or if \(|U_{p,-u}|\) is too small for most relevant rows. The contribution remains in the plan, but the final report must state any observed stability limitation explicitly.

---

## 7. Phase 5 — Contribution C: relative recovered-candidate PinyinGPT scoring

### Goal

Reuse current-context PinyinGPT information without repeating EM1's overly conservative absolute scoring behavior.

### GPU requirement

Existing B3 exact scores cover the current selected K1 recovery setup. A true K5 experiment may require exact scoring for additional recovered personal candidates.

Before GPU inference:

- identify exactly which K5 personal candidates do not already have cached exact scores;
- deduplicate fixed-candidate requests;
- run a small numerical preflight against the frozen backend;
- create resumable cache and provenance manifest.

### Variants

Evaluate:

1. boundary-only;
2. absolute exact score reference;
3. pool-relative softmax exact score
   \[
   Q_{rel}(c)=\frac{\exp(Q(c)/T)}{\sum_{r\in R}\exp(Q(r)/T)};
   \]
4. pool-relative z-normalized exact score
   \[
   Q_z(c)=\frac{Q(c)-\mu_R}{\sigma_R+\epsilon};
   \]
5. boundary + relative exact adjustment.

Freeze the relative-score temperature / weight grid before looking at outcomes.

Recommended compact first grid for one adjustment weight:

`lambda_Q in {0.25, 0.5, 1.0, 2.0}`

Keep any temperature grid equally small and predeclared.

### Key diagnostics

- how many K5 recoverable Gold candidates remain Top10/Top3/Top1;
- whether Conflict harm decreases;
- whether non-Conflict rescue is preserved;
- whether exact scoring is still suppressing candidate coverage.

---

## 8. Phase 6 — Contribution D: position-aware local context

### D1. Character-level positional similarity

Start with character-level context to avoid segmentation dependence.

For each historical same-Pinyin row, compare the suffix before the input point using:

\[
S_{pos}(q,h)=\sum_{d=1}^{L}\alpha^{d-1}\mathbf{1}[x^q_{-d}=x^h_{-d}].
\]

Predeclare a compact grid, for example:

- local window L in `{4, 8, 16}` characters;
- distance decay alpha in `{0.5, 0.7, 0.85}`.

The exact search grid must be frozen in a config before execution.

### D2. N-gram backoff baseline

Implement a deterministic character suffix baseline:

- exact 4-gram if sufficient evidence;
- else 3-gram;
- else 2-gram;
- else 1-gram;
- else long-term distribution.

Predefine what "sufficient evidence" means before comparing outcomes.

### D3. Recency

Use interaction-distance recency, not wall-clock time:

\[
age(h)=\#\{\text{same-author interactions between }h\text{ and the current row}\}.
\]

Use:

\[
R(h)=\exp\left(-\frac{age(h)}{\tau}\right)
\]

and test a compact predeclared grid, for example \(\tau\in\{100,500,2000\}\).

### D4. Local distribution

For every candidate \(c\) under the same Pinyin:

\[
w_h=S_{pos}(q,h)R(h),
\]

\[
P_{local}(c\mid q,p)=\frac{\sum_{h:p_h=p,y_h=c}w_h}{\sum_{h:p_h=p}w_h}.
\]

### D5. Semantic cosine ablation

Only after the cheap lexical experiments are complete:

- reuse existing compatible embeddings where legally possible;
- otherwise create a versioned cache;
- compare semantic cosine to position/recency rather than assuming one is better.

Required variants:

- position only;
- position + recency;
- n-gram backoff;
- semantic cosine;
- position + recency + semantic cosine.

---

## 9. Phase 7 — Combination ablation

Do not jump directly to all features.

Recommended sequence:

- A: K5 broad recovery;
- A+B: + personal choice distribution;
- A+B+B2: + background-normalised frequency/lift;
- A+B+B2+C: + relative PinyinGPT scoring;
- A+B+B2+D: + position/recency context;
- A+B+B2+C+D: planned full model.

For completeness, also keep A+B+C+D as a no-background ablation so the contribution of background frequency is directly measurable.

For every transition report paired rescue/harm/net.

The planned full-model scoring family is:

\[
S(c)=B(c)
+\lambda_D\log(\widetilde P_u(c\mid p)+\epsilon)
+\lambda_B L_u(c,p)
+\lambda_Q Q_{rel}(c)
+\lambda_C P_{local}(c\mid q,p).
\]

All \(\lambda\) values must be selected only on Train-Val from predeclared grids.

### Final selection principle

Primary model selection remains Macro-author Top1 on Train-Val, but a proposed contribution should not be described as solving Conflict unless it also shows the expected paired Conflict behavior.

Do not optimize a hidden composite objective after seeing the outcomes.

---

## 10. Unified result schema

Every method result should expose at least:

- method ID/version;
- input hashes;
- candidate-surface hash;
- code/config hashes;
- hyperparameters;
- `dev3000_used=false`;
- `test_used=false`;
- overall Macro/Micro Top1;
- Top3;
- MRR@10;
- Missing@10;
- Ambiguous metrics;
- Conflict metrics;
- Ambiguous-non-Conflict metrics;
- rescue/harm/net vs declared baseline;
- recoverable Gold Top10/Top3/Top1 when relevant;
- per-author metrics;
- runtime/device details where inference occurs.

---

## 11. Statistics

For row-level Top1 comparisons:

- exact McNemar test for paired correctness;
- paired bootstrap 95% confidence interval for relevant metric deltas.

Do not treat small point improvements as established without uncertainty analysis.

---

## 12. Reproducibility files to maintain after every phase

Each phase should update or create:

- `manifest.json`;
- `config.json`;
- `artifact_checksums.json`;
- `summary.json`;
- row-level output / predictions;
- `reproduction.md` with exact PowerShell commands;
- a human-readable interpretation note separating confirmed finding from hypothesis.

Traceability target:

`claim -> summary -> row-level artifact -> command -> frozen input`

---

## 13. PRE-DEV freeze condition

Do not create PRE-DEV freeze until:

1. diagnostic subset definitions are reconciled;
2. all intended A/B/B2/C/D ablations are complete on Train-Val;
3. the final method formula and all hyperparameters are selected;
4. all configs and artifact hashes are written;
5. no unresolved implementation bug affects ranking semantics;
6. the final planned Dev evaluation table is specified in advance.

Then create:

- machine-readable registry/config with hashes;
- human-readable PRE_DEV_FREEZE document;
- explicit `used_test=false` and current Dev usage status.

Only after that should Dev3000 be evaluated.

---

## 14. Suggested first implementation action

The safest next coding task is **Phase 1 distribution diagnostic**, because it is CPU-only, does not alter the frozen candidate surface, and directly tests whether the proposed support/margin/entropy interpretation is visible in existing PV1 rescue/harm data.

After that, freeze the K5 new-method surface and proceed contribution by contribution.
