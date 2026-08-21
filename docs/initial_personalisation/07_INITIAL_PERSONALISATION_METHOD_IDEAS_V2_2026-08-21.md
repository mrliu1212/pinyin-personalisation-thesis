# Initial Personalisation — Recommended Method Ideas V2

Date: 2026-08-21
Status: method-design note before implementation freeze
Scope: new pre-Dev method family; does not retroactively redefine PV1 or EM1

## 1. Design principle

The next method family should not be described as a single opaque "gate". Each part should have its own research contribution, motivation, ablation, and interpretation.

Recommended contribution structure:

1. **Broad Personal Candidate Recovery**
2. **Personal Choice Distribution**
3. **Background-Normalised Personal Frequency / Distinctiveness**
4. **Relative Recovered-Candidate Scoring**
5. **Position- and Recency-Aware Personal Context**

The background-frequency component is now part of the planned method family rather than only an optional side analysis. It must still be validated independently because the benchmark has only a small number of deep proxy users. The final system combines the validated parts, but every part must remain independently measurable.

---

## 2. Contribution A — Broad Personal Candidate Recovery

### Where the idea came from

PV1 K=1, K=3, and K=5 have the same observed Top1 at lambda=4, but larger K substantially improves Missing@10 and MRR.

This suggests that limiting recovery to one personal candidate hides potentially useful targets.

### Recommended method

Use the existing causal personal vocabulary ordered without Gold leakage and expose up to K=5 personal-only candidates to the next-stage scorer.

Recommended new-method starting point:

- broad candidate pool: K=5;
- retain K=1 and K=3 as formal ablations.

### Interpretation

This contribution is about **coverage**, not about claiming that K=5 itself improves Top1.

Success means:

- more legally recoverable Gold enters the candidate list;
- Missing@10 decreases;
- Top3/MRR improve or remain stable;
- later scoring modules can exploit the additional candidates.

---

## 3. Contribution B — Personal Choice Distribution

### Where the idea came from

Frequency is already a strong signal, but Conflict results show that treating the historical winner as uniformly reliable is unsafe.

A scalar count loses information about:

- how often this Pinyin pattern has been observed;
- how concentrated the user's choices are under that Pinyin;
- whether the winner barely beats the runner-up or dominates it;
- whether an apparently rare Pinyin consistently maps to one special target.

### Core representation

For current Pinyin/Initial pattern \(p\) and candidate \(c\), compute within the current causal same-Pinyin history:

\[
N_u(p)=\sum_{h\in H_t}\mathbf{1}[p_h=p]
\]

\[
N_u(c,p)=\sum_{h\in H_t}\mathbf{1}[p_h=p,\ y_h=c]
\]

The unsmoothed conditional choice distribution is:

\[
P_u(c\mid p)=\frac{N_u(c,p)}{N_u(p)}.
\]

We additionally record normalized entropy and the winner-vs-runner-up margin.

### Support vs exposure

Do not keep exposure as a separate primary feature. The main evidence amount is:

\[
\text{support}_n(p)=N_u(p).
\]

The scale-normalized diagnostic is:

\[
\text{support\_rate}(p)=\frac{N_u(p)}{|H_t|}.
\]

When \(|H_t|=5000\), support rate is only support divided by a constant, so it should not be counted as a separate contribution.

### Smoothing

Observed proportions should not be used unsmoothed when support is tiny. Recommended family:

\[
\widetilde P_u(c\mid p)=
\frac{N_u(c,p)+\alpha P_0(c\mid p)}{N_u(p)+\alpha}.
\]

Here \(P_0\) is a prior distribution defined without future leakage and \(\alpha>0\) controls shrinkage.

If a neutral prior is used in the first experiment, document it clearly. If a background distribution is used, it must be fitted from legal Train-Fit information only.

### How to interpret support x concentration

Do not use the rule "low support = bad".

Interpret the joint state:

| Support | Concentration | Interpretation |
|---|---|---|
| high | high | stable long-term preference |
| high | low | genuinely ambiguous Pinyin for this user |
| moderate/low but above minimal evidence range | high | potentially valuable rare/special personal vocabulary |
| extremely low | high | attractive but uncertain; smoothing should prevent overconfidence |
| low | low | little reliable personal evidence |

This is important because special personal words may be rare globally and rare even within one user's total text, yet highly consistent whenever that Pinyin occurs.

### Entropy and margin

For the top two candidates \(c_{(1)}\) and \(c_{(2)}\), define margin as:

\[
M_u(p)=\widetilde P_u(c_{(1)}\mid p)-\widetilde P_u(c_{(2)}\mid p).
\]

Define distribution entropy as:

\[
H_u(p)=-\sum_{c\in C_p}\widetilde P_u(c\mid p)\log \widetilde P_u(c\mid p),
\]

and, when \(|C_p|>1\), normalized entropy as:

\[
H_u^{norm}(p)=\frac{H_u(p)}{\log |C_p|}.
\]

Recommended first role: **diagnostic variables**.

Before putting them directly in the ranking score, test whether they separate:

- PV1 rescue vs harm;
- Conflict vs non-Conflict;
- correct vs incorrect personal overrides.

If diagnostic separation exists, then define a new scoring ablation using one or both.

### Candidate distribution score

The first clean replacement for count-based preference should be candidate-specific \(\widetilde P_u(c\mid p)\) or a monotonic transformed version of it, for example \(D_u(c,p)=\log(\widetilde P_u(c\mid p)+\epsilon)\).

Keep the transformation simple and interpretable.

---

## 4. Contribution B2 — Background-Normalised Personal Frequency / Distinctiveness

### Where the idea came from

A very frequent target may reflect normal Chinese syntax/usage rather than a personal habit. For example, a user selecting `是` very frequently under `shi` is not necessarily evidence that `是` is distinctive for that user if comparable users show almost the same conditional distribution.

This component is therefore intended to answer a different question from the within-user choice distribution:

> Is this candidate frequent **because this user prefers it**, or because it is common language behaviour for this Pinyin?

### Problem with raw cross-user counts

Users/authors have very different amounts of text. Pooling all raw interactions would let large authors dominate the background. We therefore normalize within each background user first and only then average users.

### Per-user conditional distributions

For every background user \(v\), estimate a smoothed conditional distribution:

\[
\widetilde P_v(c\mid p)=
\frac{N_v(c,p)+\alpha P_0(c\mid p)}{N_v(p)+\alpha}.
\]

Let \(U_{p,-u}\) be the set of other users with legal prior support for \(p\). The leave-one-user-out macro background is:

\[
P_{bg,-u}(c\mid p)=
\frac{1}{|U_{p,-u}|}
\sum_{v\in U_{p,-u}}\widetilde P_v(c\mid p).
\]

Every user therefore receives one vote in the background distribution regardless of how many total interactions they contributed.

### Candidate-level personal lift

Define:

\[
L_u(c,p)=
\log\frac{\widetilde P_u(c\mid p)+\epsilon}
{P_{bg,-u}(c\mid p)+\epsilon}.
\]

Interpretation:

- \(L_u(c,p)\gg0\): candidate is unusually preferred by this user;
- \(L_u(c,p)\approx0\): candidate is common for the user but similarly common in the background;
- \(L_u(c,p)<0\): candidate is less characteristic of this user than of comparable users.

This directly addresses the earlier “grammar/common word” concern. A target may have high \(P_u(c\mid p)\) but low lift if everyone else behaves similarly.

### Optional Pinyin-level diagnostic

A distribution-level divergence such as Jensen-Shannon divergence can be recorded as a diagnostic:

\[
JS\!\left(\widetilde P_u(\cdot\mid p),P_{bg,-u}(\cdot\mid p)\right).
\]

This says how unusual the user's whole choice distribution is for that Pinyin, but candidate ranking should initially use the more directly interpretable candidate-level lift.

### Stability requirement

Background frequency is now part of the planned contribution set and must be implemented and ablated. However, because the benchmark contains only a few deep proxy users, we must report \(|U_{p,-u}|\), background support, and per-author stability. We should not claim the background estimate is reliable unless those diagnostics support it.

---

## 5. Contribution C — Relative Recovered-Candidate Scoring

### Where the idea came from

EM1-R exact fixed-candidate scoring is dramatically more conservative than PV1 on the same recoverable surface. However, EM1-R+F improves some Conflict transitions relative to PV1.

This suggests that exact PinyinGPT likelihood contains current-context information but is currently playing the wrong role.

### Do not use absolute exact score as the recovery authority

Avoid a design where a personal-only candidate must directly compete using its raw absolute PinyinGPT log probability against Generic beam candidates.

That risks recreating the same Generic preference that excluded the candidate.

### Recommended role

Use exact PinyinGPT candidate scores to compare **recovered personal candidates with one another**.

For recovered pool \(R\), compute exact fixed-candidate log probability \(Q(c)\), then normalize only within the recovered pool. A softmax form is:

\[
Q_{rel}(c)=\frac{\exp(Q(c)/T)}{\sum_{r\in R}\exp(Q(r)/T)}.
\]

A z-normalized alternative is:

\[
Q_z(c)=\frac{Q(c)-\mu_R}{\sigma_R+\epsilon}.
\]

Interpretation:

> Among the personal candidates that history says are worth considering, which ones are more linguistically plausible in the current Pinyin/context?

### Required ablation

Compare:

1. PV1-style boundary score;
2. absolute exact score (EM1-like reference);
3. relative exact score;
4. boundary + weighted relative exact adjustment.

The goal is to keep broad recovery while extracting useful current-context discrimination from PinyinGPT.

---

## 6. Contribution D — Position- and Recency-Aware Personal Context

### Where the idea came from

Conflict means the long-term historical winner is wrong for the current row. Therefore a useful corrective signal must represent **what is relevant now**, not only what is frequent historically.

Previous semantic-retrieval experiments showed that better retrieval discrimination does not automatically translate to better end-to-end ranking. We therefore want a simpler candidate-specific local signal first.

### D1. Position-aware local lexical similarity

For current context q and historical context h, compare characters backward from the current Pinyin insertion point.

A simple transparent score:

\[
S_{pos}(q,h)=\sum_{d=1}^{L}\alpha^{d-1}\mathbf{1}[x^q_{-d}=x^h_{-d}].
\]

where:

- d=1 is the character immediately before the Pinyin input;
- larger d means farther away;
- 0 < alpha < 1 makes closer context more important.

Interpretation:

> A matching character immediately next to the Pinyin input is stronger grammatical/collocational evidence than a matching character far away.

This is a soft positional n-gram rather than a hard match/no-match rule.

### D2. Recency

For each legal historical interaction h, define interaction age as the number of same-author interactions between h and the current row.

Use a temporal decay such as:

\[
R(h)=\exp\left(-\frac{age(h)}{\tau}\right).
\]

This avoids depending on inconsistent wall-clock gaps and matches the causal chronological interaction sequence.

### D3. Local target distribution

Combine local position and recency:

\[
w_h=S_{pos}(q,h)R(h).
\]

Then form a target distribution:

\[
P_{local}(c\mid q,p)=\frac{\sum_{h:p_h=p,\ y_h=c}w_h}{\sum_{h:p_h=p}w_h}.
\]

Interpretation:

> What does this user choose for the same Pinyin in recent, locally similar expressions?

### Why a distribution, not one retrieved history row

A single nearest history example is fragile. Aggregating weighted evidence over multiple historical instances gives a more stable candidate-level signal and preserves transparency.

### D4. N-gram backoff baseline

Run a cheap lexical baseline in parallel:

- longest matching suffix / character n-gram;
- back off 4 -> 3 -> 2 -> 1 -> long-term distribution.

This lets us compare:

- hard local lexical matching;
- soft positional matching.

### D5. Semantic cosine as a separate ablation

Do not discard cosine similarity. It captures broader semantic/topic similarity that local n-grams may miss.

Test separately:

- position only;
- position + recency;
- suffix n-gram backoff;
- semantic cosine;
- position + recency + semantic cosine.

This keeps the contribution interpretable and tells us whether local lexical structure or broader semantics is more useful for Conflict correction.

---

## 7. Recommended full system

The final candidate model should be assembled only after the individual contributions are evaluated. The planned full candidate contains all five evidence sources:

1. recover up to \(K=5\) legal personal-only candidates;
2. estimate the user's long-term smoothed choice distribution;
3. estimate leave-one-user-out background frequency and personal lift;
4. compute relative PinyinGPT plausibility within the recovered personal pool;
5. compute a recent local-context distribution using position-aware similarity and recency.

Define the long-term personal term:

\[
D_u(c,p)=\log\left(\widetilde P_u(c\mid p)+\epsilon\right),
\]

and the background-distinctiveness term:

\[
L_u(c,p)=
\log\frac{\widetilde P_u(c\mid p)+\epsilon}
{P_{bg,-u}(c\mid p)+\epsilon}.
\]

A transparent additive final score is:

\[
S(c)=B(c)
+\lambda_D D_u(c,p)
+\lambda_B L_u(c,p)
+\lambda_Q Q_{rel}(c)
+\lambda_C P_{local}(c\mid q,p).
\]

Here:

- \(B(c)\) is normalized Generic score for Generic candidates and the frozen recovery boundary for personal-only candidates;
- \(D_u(c,p)\) captures absolute long-term within-user preference;
- \(L_u(c,p)\) captures how much that preference exceeds background usage;
- \(Q_{rel}(c)\) captures current PinyinGPT plausibility among recovered candidates;
- \(P_{local}(c\mid q,p)\) captures recent, locally similar personal behaviour.

Because \(D_u\) and \(L_u\) both depend on \(\widetilde P_u\), their combination must be justified by ablation rather than assumed. The specific background weight \(\lambda_B\) is therefore independently tuned/frozen on Train-Val.

No hidden gate is required. Every term has a row-level interpretation.

---

## 8. How the new indicators should be interpreted

### support_n

Question: "How much same-Pinyin evidence do we have?"

High does not mean the winner is correct. It only means the distribution estimate is based on more observations.

### support_rate

Question: "What fraction of the currently visible raw causal history matches this Pinyin?"

Diagnostic only. Mostly redundant with support once H5000 is saturated.

### \(\widetilde P_u(c\mid p)\)

Question: "When this user types this Pinyin, how often do they choose this candidate after smoothing small-sample uncertainty?"

This is the main long-term candidate-preference statistic.

### margin

Question: "How far ahead is the historical winner from the runner-up?"

Small margin means the winner is fragile even if raw count is large.

### entropy

Question: "How intrinsically spread/ambiguous is this user's target distribution under this Pinyin?"

High entropy means several targets have meaningful probability.

### \(L_u(c,p)\): background-normalised personal lift

Question: "Is this candidate unusually preferred by this user, or is it simply common language?"

Large positive lift supports genuine personal distinctiveness.

### \(Q_{rel}(c)\)

Question: "Among the personal candidates already admitted by history, which is more plausible to PinyinGPT in the current context?"

It should not be interpreted as an absolute permission to recover.

### \(P_{local}(c\mid q,p)\)

Question: "In recent locally similar contexts, what target does this user choose?"

This is the main candidate-specific corrective signal for Conflict.

---

## 9. Main expected research story

The method family should be presented as a chain of evidence, not as feature accumulation:

- Initial causes a major candidate-coverage problem.
- Broad personal recovery exposes more legally supported candidates.
- Historical frequency helps strongly but can over-trust a majority under Conflict.
- A conditional choice distribution describes how stable or ambiguous that preference really is.
- Background-normalised frequency tests whether a high personal frequency is genuinely user-specific or merely common language behaviour.
- PinyinGPT exact scoring is useful as relative plausibility but too conservative as an absolute recovery authority.
- Local position and recency estimate which historical choice is relevant in the current expression.

The target outcome is:

> More useful recovery, retained rescues, and fewer frequency-driven Conflict harms, with every improvement traceable to an interpretable evidence source.

---

## 10. Literature mapping boundary

Conceptual links to personalized language modeling, cache/recency methods, n-gram/context distributions, and retrieval-based personalization can be added later.

Do not use external literature as proof that this exact method will work. The method is motivated first by our own failure analysis; literature should be used to position related ideas after exact bibliographic verification.
