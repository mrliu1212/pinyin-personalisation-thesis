# Initial Personalisation — Diagnostic Findings and Working Hypotheses

Date: 2026-08-21
Status: pre-Dev research note
Scope: standardized Initial + Short Clean3 Train-Val only unless explicitly marked historical
Primary population: 34,416 Train-Val rows
Protocol boundary: Dev3000 not used for the results summarized here; Test not used for current standardized development.

## 1. Purpose

This document separates **confirmed observations from current artifacts** from **working hypotheses inspired by those observations**. It is the diagnostic starting point for the next Initial-personalisation method family.

The central research objective is:

> Preserve the large number of useful personalisation rescues while reducing errors caused by over-trusting historical frequency when the current intended target differs from the historical frequency winner.

We intentionally do **not** define a gate here. The next research stage should test several interpretable contributions separately.

---

## 2. Current same-surface baseline results

On the standardized 34,416-row Initial Train-Val surface:

| Method | Macro Top1 | Micro Top1 | Top3 | MRR@10 | Missing@10 |
|---|---:|---:|---:|---:|---:|
| G | 0.307099 | 0.330573 | 0.491341 | 0.426472 | 0.365092 |
| F | 0.382495 | 0.408473 | 0.555120 | 0.489432 | 0.365092 |
| PV1 | **0.401872** | **0.426749** | **0.598907** | **0.524450** | **0.291144** |
| EM1-R | 0.310531 | 0.334292 | 0.500349 | 0.434164 | 0.345857 |
| EM1-R+F | 0.394722 | 0.421199 | 0.592719 | 0.517784 | 0.300616 |

Confirmed paired transitions:

- F -> PV1: rescue 933, harm 304, net +629.
- F -> EM1-R+F: rescue 735, harm 297, net +438.
- PV1 -> EM1-R+F: rescue 492, harm 683, net -191.
- On the formal Conflict subset, PV1 -> EM1-R+F gives rescue 195, harm 171, net +24.

### Immediate interpretation

1. **Personal frequency is already a strong signal.** G -> F gives a large gain without changing candidate coverage.
2. **Candidate recovery adds further value.** F -> PV1 produces +629 net correct rows and substantially lowers Missing@10.
3. **PV1 is not uniformly safe.** Its overall gain coexists with a clear failure mode on Conflict cases.
4. **Current-context PinyinGPT evidence contains useful information, but using exact absolute candidate likelihood as the main recovery score is too conservative overall.**

---

## 3. Ambiguity and Conflict are not the same problem

For the current B2 same-surface evaluation:

- Ambiguous subset: n = 30,527.
- Conflict subset: n = 16,192.

Definitions used by the same-surface evaluator:

- **Ambiguous**: the same user has at least two historical targets under the same current Initial/Pinyin condition.
- **Conflict**: the row is ambiguous, there is a unique historical frequency winner, and the current Gold target is not that historical winner.

Therefore:

> Conflict is a harder subset of Ambiguous cases. Ambiguity means multiple historical choices exist; Conflict means historical majority preference actively points away from the current target.

### Confirmed performance pattern

On Ambiguous rows, PV1 still improves over F overall. On Conflict rows, PV1 is worse than F.

This is an important distinction:

> The problem is not simply that Initial input is ambiguous. The specific risk is that frequency-based personalisation can over-trust a historical majority when the present context calls for a different target.

### Working hypothesis H1

A better method should preserve the large non-Conflict recovery benefit while using richer evidence to reduce frequency-driven errors in Conflict cases.

---

## 4. Broader K exposes a coverage/ranking separation

With frozen frequency lambda = 4.0 and PV lambda = 4.0:

| K | Macro Top1 | Micro Top1 | Top3 | MRR@10 | Missing@10 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.401872 | 0.426749 | 0.598907 | 0.524450 | 0.291144 |
| 3 | 0.401872 | 0.426749 | 0.601377 | 0.531101 | 0.253516 |
| 5 | 0.401872 | 0.426749 | 0.601377 | 0.532265 | 0.243201 |

Key observation:

- K=1 -> K=5 leaves Top1 unchanged on this grid point.
- Missing@10 drops from 29.11% to 24.32%.
- MRR@10 improves from 0.52445 to 0.53226.

### Interpretation

> A broader personal candidate pool can recover substantially more useful candidates without automatically improving Top1. This suggests that recovery capacity and ranking quality are separate bottlenecks.

### Working hypothesis H2

Use a broader recovery surface for the next method family, then improve candidate ranking rather than forcing K=1 merely because K=1 won a previous Top1 tie-break.

Important protocol note: choosing K=5 for a new method family is a **new pre-Dev method design decision motivated by Train-Val diagnostics**, not a retroactive change to PV1.

---

## 5. EM1 is too conservative for recovery

On the selected K=1 recoverable surface, n = 2,652 Gold targets are recoverable from personal history.

| Method | Gold in Top10 | Gold in Top3 | Gold at Top1 |
|---|---:|---:|---:|
| PV1 | 2652 | 2417 | 933 |
| EM1-R | 678 | 329 | 135 |
| EM1-R+F | 2300 | 1531 | 675 |

### Confirmed observation

Exact fixed-candidate PinyinGPT scoring alone suppresses many recovered personal candidates. Adding frequency recovers much of that loss, but EM1-R+F still remains below PV1 overall.

### Plausible explanation — not yet a proven causal mechanism

Recovered candidates are, by definition, candidates that Generic beam search did not already place in its Top10. Reusing absolute PinyinGPT likelihood as the main recovered-candidate score may systematically recreate the same preference that excluded them in the first place.

This is a hypothesis, not a causal conclusion.

### Working hypothesis H3

PinyinGPT exact candidate likelihood may be more useful as a **relative plausibility signal among recovered personal candidates** than as an absolute authority determining whether a personal candidate can compete at all.

The Conflict result supports this direction: EM1-R+F is worse overall than PV1, yet PV1 -> EM1-R+F has a small positive net transition on Conflict (+24). That indicates current-context likelihood contains some useful disambiguating signal even though the current scoring role is too conservative.

---

## 6. Frequency count is not the same as preference strength

Current F/PV1 methods use historical frequency support. The next question is whether a single scalar frequency score hides important structure.

For a current input p = `sy`, consider two users or two historical states:

- Case A: `sy` occurred 100 times; target `使用` occurred 90 times.
- Case B: `sy` occurred 2 times; target `使用` occurred 2 times.

Both have a very concentrated observed choice distribution, but they do not carry the same amount of evidence.

This motivates separating:

- **support**: how many same-Pinyin observations exist, \(N_u(p)\);
- **choice distribution**: \(P_u(c\mid p)\);
- **concentration**: how sharply that distribution prefers one target.

### Important refinement

Low support is not automatically bad. Once support is above a defensible minimal range, a rare Pinyin pattern with highly concentrated target choices may be exactly the kind of special personal vocabulary we want to recover.

Example:

- `sy` appears only 7 times, but all 7 map to a rare user-specific target.

This can be a strong personal signal even though `sy` is uncommon in the user's overall history.

### Working hypothesis H4

The useful distinction is not simply high-support vs low-support. It is the interaction between:

1. support size;
2. distribution concentration;
3. candidate-specific personal distinctiveness;
4. current local context.

---

## 7. Support vs exposure: decision for the next method family

Earlier discussion used both:

- support:
  \[
  \text{support}_n(p)=N_u(p),
  \]
  the number of same-Pinyin historical interactions;
- exposure/support rate:
  \[
  \text{support\_rate}(p)=\frac{N_u(p)}{|H_t|},
  \]
  the fraction of the current raw causal history window matching \(p\).

Under the current protocol, H_t is capped at the latest 5,000 raw prior same-author interactions before Initial/Pinyin matching.

### Decision

**Do not treat exposure as a separate primary feature.**

For most rows where |H_t| = 5,000, exposure is just support divided by a constant. It therefore contains almost the same ordering information as support.

Retain only:

- \(\text{support}_n(p)=N_u(p)\) as the primary evidence-count variable;
- optional \(\text{support\_rate}(p)=N_u(p)/|H_t|\) as a diagnostic for early-history rows where \(|H_t|<5000\) or for scale-normalized reporting.

This avoids double-counting essentially the same evidence.

---

## 8. Entropy and margin: useful diagnostics, not automatically useful ranking features

For the same-Pinyin target distribution P_u(c | p):

### Margin

Let \(c_{(1)}\) and \(c_{(2)}\) be the two highest-probability historical targets. Define:

\[
M_u(p)=P_u(c_{(1)}\mid p)-P_u(c_{(2)}\mid p).
\]

Interpretation:

- large margin: the historical winner clearly dominates the runner-up;
- small margin: the apparent winner is fragile.

### Entropy

Entropy measures how spread the full target distribution is:

\[
H_u(p)=-\sum_{c\in C_p}P_u(c\mid p)\log P_u(c\mid p).
\]

For comparability across different numbers of targets, use normalized entropy when \(|C_p|>1\):

\[
H_u^{norm}(p)=\frac{H_u(p)}{\log |C_p|}.
\]

Interpretation:

- low entropy: target choices are concentrated;
- high entropy: the same Pinyin maps to many targets with substantial mass.

These two measures are complementary:

- margin focuses on winner-vs-runner-up separation;
- entropy describes the entire distribution.

### Research discipline

Do **not** assume they should enter the ranking formula immediately.

First audit whether rescue, harm, Conflict, and non-Conflict rows show different support/margin/entropy distributions. If they do, then test them as interpretable scoring modifiers in a separately defined experiment.

### Cross-entropy

Cross-entropy is not the first metric we need here. The immediate within-user uncertainty statistic is ordinary entropy. If we later construct a reliable background distribution, then divergence to that background (for example candidate-level log-lift or a symmetric distribution divergence) becomes more directly interpretable than raw cross-entropy alone.

---

## 9. Very frequent targets may be ordinary language, not personal preference

A high historical frequency does not necessarily imply a user-specific habit.

Example conceptually:

- the user often maps `shi` to `是`;
- but almost every comparable user also maps `shi` to `是` at a similar rate.

Then high frequency may reflect normal language usage rather than strong personal distinctiveness.

### Proposed background frequency

For every other user \(v\), first estimate a smoothed within-user conditional distribution \(\widetilde P_v(c\mid p)\). Do **not** pool raw counts across users.

Let \(U_{p,-u}\) be other users with legal prior support for \(p\). Define the leave-one-user-out macro background:

\[
P_{bg,-u}(c\mid p)=
\frac{1}{|U_{p,-u}|}
\sum_{v\in U_{p,-u}}\widetilde P_v(c\mid p).
\]

Then define candidate-level personal lift:

\[
L_u(c,p)=
\log\frac{\widetilde P_u(c\mid p)+\epsilon}
{P_{bg,-u}(c\mid p)+\epsilon}.
\]

Interpretation:

- large positive \(L_u(c,p)\): unusually preferred by this user;
- near-zero \(L_u(c,p)\): frequent for this user but similarly frequent in the background;
- negative \(L_u(c,p)\): less characteristic of this user than of the background.

### Working hypothesis H5

A personalisation score should distinguish **common-language frequency** from **user-specific over-representation**. Background-normalised frequency is therefore part of the planned method family, not merely an optional afterthought.

Because the current standardized author count is small, its reliability must be audited using background-user count, per-author stability, and Train-Val-only ablations before making a strong claim.

---

## 10. Current-context evidence should include local position and recency

PV1 uses historical preference but not current candidate-specific context. Prior context-aware work shows that stronger retrieval discrimination does not automatically improve end-to-end ranking, so simply replacing frequency with cosine similarity is not enough.

A new idea motivated by the Conflict failure is to model local context more explicitly:

> Characters or tokens closer to the current Pinyin input position may carry stronger grammatical/collocational evidence than more distant context.

For current context ending before `sy`, compare the immediately preceding characters with historical same-`sy` contexts using distance-decayed positional weights:

\[
S_{pos}(q,h)=\sum_{d=1}^{L}\alpha^{d-1}\mathbf{1}[x^q_{-d}=x^h_{-d}].
\]

Then combine this with chronological recency:

\[
R(h)=\exp\left(-\frac{age(h)}{\tau}\right),\qquad w_h=S_{pos}(q,h)R(h).
\]

- closer-to-input context match -> stronger local lexical evidence;
- more recent history -> larger temporal weight.

This creates a local candidate distribution:

\[
P_{local}(c\mid q,p)=\frac{\sum_{h:p_h=p,\ y_h=c}w_h}{\sum_{h:p_h=p}w_h}.
\]

This answers answering:

> When this user recently typed the same Pinyin in a locally similar expression, which target did they choose?

### Working hypothesis H6

Position-aware local lexical context may be especially useful for correcting frequency-driven Conflict errors because it can capture grammatical/collocational evidence that long-term frequency ignores.

Semantic cosine similarity should remain a separate ablation because it captures broader topical/paraphrastic similarity rather than exact local lexical structure.

---

## 11. Core hypotheses to test next

H1. PV1's main risk is not ambiguity in general but frequency-driven Conflict.

H2. Broad recovery (K up to 5) provides valuable coverage that should be retained while ranking is improved.

H3. Absolute exact PinyinGPT scoring is too conservative, but relative recovered-candidate plausibility may still help.

H4. Personal frequency should be represented as a conditional choice distribution with support and concentration information, rather than only a scalar count.

H5. Very frequent ordinary-language targets should be distinguished from genuinely user-specific choices using a carefully normalized background comparison.

H6. Position-aware local context plus recency may help the model choose a non-majority target when the current expression differs from the user's long-term majority pattern.

---

## 12. Evaluation policy for all next-stage methods

Primary metric:

- Macro-author Top1.

Always report secondary metrics:

- Micro Top1;
- Top3;
- MRR@10;
- Missing@10.

Always report paired diagnostics against the relevant baseline:

- rescue;
- harm;
- net.

Always stratify by:

- Overall;
- Ambiguous;
- Conflict;
- Ambiguous-but-non-Conflict.

Recovery methods must additionally report:

- recoverable Gold at Top10;
- Top3;
- Top1;
- candidate-pool coverage by K.

When row-level comparison is available, use paired uncertainty/significance analysis such as exact McNemar correctness and paired bootstrap confidence intervals.

---

## 13. Evidence boundary

The numeric findings in Sections 2–5 are established current Train-Val observations. Sections 6–10 contain research interpretations and hypotheses that still require dedicated diagnostics or ablations.

Do not present a hypothesis as a confirmed mechanism until the planned experiments separate it from competing explanations.
