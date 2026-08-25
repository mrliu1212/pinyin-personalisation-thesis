# R4: Controlled and Reversible Lexical Preference Adaptation

## 1. Research Question

This experiment investigates whether the model-level user Adapter can learn, reverse, and subsequently recover a lexical preference under controlled longitudinal exposure.

The central question is:

> **Can a persistent user-specific Adapter modify the lexical preference of a frozen generic Pinyin-to-Chinese model in response to changing user evidence, and can this preference be reversed when the user's behaviour changes again?**

For an ambiguous Pinyin input associated with two competing Chinese candidates $A$ and $B$, the experiment tests whether the Adapter can follow the sequence:

$$
\text{Generic} \rightarrow A \rightarrow B \rightarrow A
$$

The purpose is therefore to evaluate **preference plasticity and reversibility**, rather than ordinary static prediction accuracy.

---

## 2. Model Setting

The experiment uses the frozen PinyinGPT2-Concat model as the Generic Pinyin-to-Chinese model. User-level adaptation is implemented through the previously defined model-level serial bottleneck Adapter.

The Generic model parameters remain frozen throughout the experiment. Only the user-specific Adapter parameters are updated.

The same Adapter is preserved across the three stages:

$$
\text{Generic}
\rightarrow
\text{Stage 1}
\rightarrow
\text{Stage 2}
\rightarrow
\text{Stage 3}.
$$

Importantly, Stage 2 does **not** restart from Generic, and Stage 3 does **not** restart from Generic or Stage 1. Instead:

- Stage 1 begins from the unpersonalised Generic state.
- Stage 2 continues from the Stage 1 Adapter.
- Stage 3 continues from the Stage 2 Adapter.

This creates a true longitudinal preference trajectory. The optimizer is reset between stages, while the learned Adapter parameters are retained.

---

## 3. Controlled Candidate Pairs

A total of **15 candidate pairs** were selected. Each pair has a unique Pinyin sequence, avoiding intervention interference between multiple controlled pairs sharing the same Pinyin.

| Pair | Pinyin | Candidate A | Candidate B |
|---|---|---|---|
| R4_PAIR_005 | shi | 时 | 使 |
| R4_PAIR_006 | yu | 与 | 于 |
| R4_PAIR_009 | zhi hui | 指挥 | 只会 |
| R4_PAIR_011 | yi jian | 一件 | 一间 |
| R4_PAIR_012 | zheng shi | 正式 | 正是 |
| R4_PAIR_013 | shi qu | 失去 | 逝去 |
| R4_PAIR_014 | zu zhi | 组织 | 阻止 |
| R4_PAIR_017 | yi | 一 | 以 |
| R4_PAIR_018 | ji | 及 | 极 |
| R4_PAIR_019 | yi zhi | 一只 | 一支 |
| R4_PAIR_020 | hui | 会 | 回 |
| R4_PAIR_024 | ci | 此 | 次 |
| R4_PAIR_027 | xiang zhe | 想着 | 向着 |
| R4_PAIR_028 | zhi | 只 | 至 |
| R4_PAIR_030 | tong yi | 统一 | 同一 |

The purpose is not to select only pairs that are perfectly balanced under the Generic model. Some pairs are initially ambiguous, whereas others exhibit stronger pretrained Generic preferences. This allows the experiment to test both:

1. preference formation and reversal for relatively ambiguous choices; and
2. attenuation or override of stronger Generic lexical priors.

---

## 4. Frozen Evaluation Set

Evaluation data were frozen **before training**.

For every candidate pair, eight real Agent Phage examples were held out:

$$
4\ \text{A-origin examples} + 4\ \text{B-origin examples}.
$$

Therefore, the contextual evaluation set contains:

$$
15 \times 8 = 120
$$

held-out examples.

These 120 rows are excluded from all three training stages. Consequently, the contextual evaluation measures generalisation of the learned preference rather than memorisation of the same examples used during intervention training.

No Test split was used.

---

## 5. Training Intervention

Each stage contains:

$$
5000
$$

training rows, consisting of:

$$
600\ \text{controlled preference exposures}
$$

and

$$
4400\ \text{background examples}.
$$

With 15 candidate pairs, the controlled portion corresponds to:

$$
\frac{600}{15} = 40
$$

controlled exposures per pair per stage.

### Stage 1 - A Exposure

For every pair, the model receives 40 controlled examples of candidate $A$.

The expected effect is an increased preference for $A$.

### Stage 2 - B Exposure

The same persistent Adapter is then trained on 40 controlled examples of candidate $B$ per pair.

The expected effect is to attenuate or reverse the Stage 1 preference for $A$.

### Stage 3 - A Recovery

Finally, the same Adapter receives another 40 controlled examples of candidate $A$ per pair.

The expected effect is recovery toward $A$.

The intervention therefore follows:

$$
A \rightarrow B \rightarrow A.
$$

---

## 6. Background Data Policy

The background data remain ordinary real Agent Phage training examples.

However, for each controlled Pinyin, background rows corresponding directly to the controlled candidates $A$ and $B$ are removed. Other naturally occurring candidates sharing the same Pinyin remain in the background.

For example, for:

$$
\text{yi}:\quad A=\text{一},\quad B=\text{以},
$$

background examples corresponding to `yi -> 一` and `yi -> 以` are excluded, while naturally occurring alternatives such as `yi -> 已` may remain.

This prevents the background stream from adding uncontrolled direct evidence for $A$ or $B$, while preserving natural same-Pinyin lexical competition.

The three background blocks are deterministic and disjoint across stages.

---

## 7. What Is Being Measured?

This experiment does **not** primarily measure Top-1 accuracy. Instead, it directly measures the model's **relative preference between candidate $A$ and candidate $B$**.

For a candidate pair, the model independently scores both candidates and produces their log probabilities.

The preference margin is defined as:

$$
m = \log P(A) - \log P(B).
$$

Therefore:

- if $m > 0$, the model prefers $A$;
- if $m < 0$, the model prefers $B$;
- if $m \approx 0$, the model is approximately indifferent between $A$ and $B$.

The absolute magnitude of the margin indicates the strength of the relative preference.

---

## 8. Example of Margin Calculation

Suppose a held-out context produces:

$$
\log P(A) = -3.2
$$

and

$$
\log P(B) = -3.8.
$$

Then:

$$
m = -3.2 - (-3.8) = +0.6.
$$

The model therefore prefers $A$ in this context.

Alternatively, suppose:

$$
\log P(A) = -5.1
$$

and

$$
\log P(B) = -4.3.
$$

Then:

$$
m = -5.1 - (-4.3) = -0.8.
$$

The model therefore prefers $B$.

The experiment examines how this same $A-B$ margin changes under Generic, Stage 1, Stage 2, and Stage 3 model states.

---

## 9. Pinyin-Only Evaluation

The first evaluation removes preceding linguistic context.

For each pair $p$ and model state $s$:

$$
M_p^{(s)} =
\log P_s(A_p \mid \text{Pinyin}_p)
-
\log P_s(B_p \mid \text{Pinyin}_p).
$$

This produces one margin per candidate pair.

The reported Pinyin-only mean is the macro-average across all 15 pairs:

$$
M^{(s)} = \frac{1}{15}\sum_{p=1}^{15} M_p^{(s)}.
$$

Each candidate pair therefore contributes equally to the final mean, regardless of its frequency in the original dataset.

---

## 10. Held-Out Context Evaluation

The second evaluation includes real preceding linguistic context.

For candidate pair $p$, held-out context $c$, and model state $s$:

$$
m_{p,c}^{(s)} =
\log P_s(A_p \mid c, \text{Pinyin}_p)
-
\log P_s(B_p \mid c, \text{Pinyin}_p).
$$

Each pair contains exactly eight frozen contexts. The pair-level contextual margin is therefore:

$$
M_p^{(s)} = \frac{1}{8}\sum_{c=1}^{8}m_{p,c}^{(s)}.
$$

The final reported contextual mean is then the macro-average over the 15 pairs:

$$
M^{(s)} = \frac{1}{15}\sum_{p=1}^{15}M_p^{(s)}.
$$

Thus, the final number is computed in two stages:

1. average the eight held-out contexts within each candidate pair;
2. average the resulting 15 pair-level margins.

This prevents high-frequency candidate pairs from dominating the aggregate result.

---

## 11. Pinyin-Only Results

The Pinyin-only mean margins were:

| Model state | Mean A-B margin |
|---|---:|
| Generic | **+1.095** |
| Stage 1 A | **+3.685** |
| Stage 2 B | **-1.053** |
| Stage 3 A | **+3.494** |

The trajectory is therefore:

$$
+1.095 \rightarrow +3.685 \rightarrow -1.053 \rightarrow +3.494.
$$

This corresponds to:

$$
\text{Generic} \rightarrow A \rightarrow B \rightarrow A.
$$

Stage 1 increases the average preference for $A$. Stage 2 produces a substantial reversal toward $B$. Stage 3 restores a strong preference for $A$.

---

## 12. Pinyin-Only Transition Magnitudes

The Stage 1 to Stage 2 change is:

$$
\Delta_{A\rightarrow B} = M_{S2} - M_{S1}.
$$

Observed:

$$
-1.053 - 3.685 = -4.737.
$$

Therefore:

$$
\Delta_{A\rightarrow B} = -4.737.
$$

The Stage 2 to Stage 3 recovery is:

$$
\Delta_{B\rightarrow A} = M_{S3} - M_{S2}.
$$

Observed:

$$
3.494 - (-1.053) = +4.547.
$$

Therefore:

$$
\Delta_{B\rightarrow A} = +4.547.
$$

The similar magnitude of the two directional shifts indicates substantial continued Adapter plasticity after the initial preference has been learned.

---

## 13. Pinyin-Only Pair-Level Reversal

At Stage 1, all 15 pairs preferred $A$.

At Stage 2, 9 of 15 pairs preferred $B$.

At Stage 3, 13 of 15 pairs preferred $A$ again.

A strict full ABA success requires:

$$
M_{S1} > 0,
$$

$$
M_{S2} < 0,
$$

and

$$
M_{S3} > 0.
$$

Under this strict zero-crossing criterion, 7 of 15 pairs completed the full $A \rightarrow B \rightarrow A$ sequence in the Pinyin-only condition.

Therefore, the Pinyin-only evaluation shows a strong aggregate directional effect, although not every individual pair crosses the zero decision boundary at every stage.

---

## 14. Held-Out Context Results

The held-out contextual results are substantially stronger.

| Model state | Mean A-B margin |
|---|---:|
| Generic | **+0.015** |
| Stage 1 A | **+3.184** |
| Stage 2 B | **-2.097** |
| Stage 3 A | **+2.230** |

The trajectory is:

$$
+0.015 \rightarrow +3.184 \rightarrow -2.097 \rightarrow +2.230.
$$

The Generic baseline is nearly neutral at the aggregate level:

$$
+0.015 \approx 0.
$$

After Stage 1, the mean becomes strongly positive, indicating preference for $A$. After Stage 2, the sign reverses and becomes strongly negative, indicating preference for $B$. After Stage 3, it returns to a strongly positive value, indicating recovery toward $A$.

---

## 15. Held-Out Context Transition Magnitudes

The mean Stage 1 to Stage 2 shift is:

$$
-2.097 - 3.184 = -5.281.
$$

Therefore:

$$
\Delta_{A\rightarrow B} = -5.281.
$$

The Stage 2 to Stage 3 recovery is:

$$
2.230 - (-2.097) = +4.327.
$$

Therefore:

$$
\Delta_{B\rightarrow A} = +4.327.
$$

These values show that the model does not merely weaken an existing preference. The average preference moves from strongly favouring $A$, through zero, to strongly favouring $B$, and then returns toward $A$.

---

## 16. Pair-Level Contextual ABA Success

The contextual evaluation provides the strongest pair-level result.

- Stage 1: 15/15 pairs prefer $A$.
- Stage 2: 14/15 pairs prefer $B$.
- Stage 3: 15/15 pairs prefer $A$.

Therefore, 14 of 15 candidate pairs satisfy the complete strict ABA condition:

$$
\frac{14}{15} \times 100 = 93.3\%.
$$

Thus:

$$
\text{Held-out contextual full ABA success} = 93.3\%.
$$

This is particularly important because these contextual examples were frozen before training and never used as intervention rows. The result therefore indicates that the learned preference dynamics generalise to unseen real contexts.

---

## 17. Interpreting the Magnitude of the Margin

The margin is a log-probability difference:

$$
m = \log P(A) - \log P(B).
$$

Equivalently:

$$
m = \log\frac{P(A)}{P(B)}.
$$

Therefore:

$$
\frac{P(A)}{P(B)} = e^m.
$$

For intuition only, the held-out-context aggregate margins correspond approximately to:

### Generic

$$
e^{0.015} \approx 1.02.
$$

This is approximately neutral between $A$ and $B$.

### Stage 1

$$
e^{3.184} \approx 24.1.
$$

The relative score strongly favours $A$.

### Stage 2

$$
e^{-2.097} \approx 0.123.
$$

Equivalently:

$$
\frac{P(B)}{P(A)} \approx 8.1.
$$

The relative score strongly favours $B$.

### Stage 3

$$
e^{2.230} \approx 9.3.
$$

The relative score again strongly favours $A$.

These ratios are useful only as an intuitive interpretation of the log margin. The primary reported quantity remains the log-probability margin itself.

---

## 18. Why the Generic Context Margin Is Near Zero

The Generic contextual mean is:

$$
+0.015.
$$

This does **not** mean that every candidate pair is individually balanced.

For example, some Pinyin-only Generic margins are substantially positive or negative:

- `一 / 以`: $+2.586$
- `及 / 极`: $-2.221$
- `指挥 / 只会`: $+3.361$

The value $+0.015$ is the macro-average after first averaging the eight contexts within each pair and then averaging across the 15 pairs. Positive and negative Generic preferences therefore partly cancel.

The appropriate interpretation is:

> Across the balanced set of candidate pairs and frozen held-out contexts, the Generic model has almost no systematic overall preference toward the arbitrarily designated A or B side.

This makes the subsequent directional $A \rightarrow B \rightarrow A$ trajectory particularly informative.

---

## 19. Initial Generic Preference Strength

Not all candidate pairs begin with the same Generic preference strength.

For analysis, pairs can be divided according to the absolute Pinyin-only Generic margin.

### Primary / Lower-Prior Group

$$
|M_{\text{Generic}}| \leq 2.
$$

These candidate pairs are relatively more balanced at baseline.

### Strong-Prior Challenge Group

$$
|M_{\text{Generic}}| > 2.
$$

These pairs already exhibit a substantial Generic lexical preference before personalisation.

This distinction matters because strict zero-crossing is more difficult when the Generic prior is strong.

For strong-prior cases, two complementary effects should therefore be considered:

1. **preference attenuation** - how much the margin moves toward the newly exposed candidate; and
2. **preference reversal** - whether the margin actually crosses zero.

---

## 20. Preference Attenuation vs Preference Reversal

Suppose Stage 1 produces:

$$
M_{S1} = +4.0
$$

and Stage 2 produces:

$$
M_{S2} = +0.5.
$$

Candidate $B$ is still not preferred because the Stage 2 margin remains positive. However:

$$
\Delta = +0.5 - 4.0 = -3.5
$$

shows a large movement toward $B$.

This is **preference attenuation**.

If Stage 2 instead produces:

$$
M_{S2} = -1.0,
$$

the preference crosses the zero boundary and $B$ becomes preferred. This is **preference reversal** or **override**.

For this reason, continuous margin changes should be reported alongside strict binary flip success.

---

## 21. Main Interpretation

The combined results indicate that the user-level Adapter is capable of rapid and reversible lexical preference adaptation.

The held-out contextual trajectory is especially clear:

$$
0.015 \rightarrow 3.184 \rightarrow -2.097 \rightarrow 2.230.
$$

This provides evidence for three distinct properties.

### 21.1 Preference Acquisition

Stage 1 moves the model from an aggregate near-neutral state to a strong preference for $A$.

This demonstrates that the Adapter can acquire a controlled user-specific lexical preference.

### 21.2 Preference Reversal

Stage 2 moves the same persistent Adapter from a strong preference for $A$ to a strong preference for $B$.

This demonstrates that previously learned personalisation does not permanently lock the model into its earlier preference.

### 21.3 Preference Recovery

Stage 3 moves the preference back toward $A$.

This demonstrates that the Adapter retains sufficient plasticity to respond again when the user's behaviour changes.

Together, these results demonstrate **longitudinal preference plasticity**, rather than simple one-shot memorisation.

---

## 22. Why the Held-Out Context Result Matters

The held-out context result is substantially stronger than the context-free Pinyin-only result:

- Pinyin-only strict full ABA: **7/15**.
- Held-out contextual strict full ABA: **14/15**.

This suggests that the learned user preference is expressed particularly strongly when realistic preceding linguistic context is available.

The contextual condition captures the interaction between:

- the frozen base model's contextual representation;
- the current Pinyin input; and
- the learned user-specific Adapter.

Because the 120 contextual probes were frozen before training, the result also shows that the preference change is not restricted to exact controlled intervention rows.

---

## 23. Limitations

The result should not be overstated.

This experiment does **not** show that a naturally evolving real user's typing history will always generate equally clean $A \rightarrow B \rightarrow A$ dynamics.

The intervention is deliberately controlled:

- each pair receives a fixed number of controlled exposures;
- stage boundaries are explicit;
- direct uncontrolled $A/B$ background evidence is removed;
- each candidate pair receives the same intervention budget.

The experiment therefore primarily demonstrates the **capacity** of the Adapter to learn, revise, and recover lexical preferences under controlled user evidence.

Broader longitudinal experiments are more appropriate for evaluating adaptation under naturally evolving user histories.

---

## 24. Main Result Table

| Evaluation | Generic | Stage 1 A | Stage 2 B | Stage 3 A | S1 to S2 | S2 to S3 | Full ABA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pinyin-only | +1.095 | +3.685 | -1.053 | +3.494 | -4.737 | +4.547 | 7/15 |
| Held-out context | +0.015 | +3.184 | -2.097 | +2.230 | -5.281 | +4.327 | **14/15 (93.3%)** |

All margin values are defined as:

$$
\log P(A) - \log P(B).
$$

---

## 25. Thesis-Ready Results Paragraph

**Controlled preference dynamics.** To evaluate whether model-level personalisation can track changing lexical preferences, we constructed a three-stage $A \rightarrow B \rightarrow A$ intervention over 15 unique-Pinyin candidate pairs. The same user-specific Adapter was updated sequentially across all stages, while the underlying PinyinGPT2-Concat model remained frozen. For each pair, 40 controlled exposures were provided per stage, alongside 4,400 background examples, while four A-origin and four B-origin examples were frozen as held-out contextual probes before training. Preference was measured as the log-probability margin $\log P(A)-\log P(B)$. In the Pinyin-only condition, the mean margin changed from +1.095 at the Generic baseline to +3.685 after A exposure, -1.053 after B exposure, and +3.494 following A recovery. The corresponding mean Stage-1-to-Stage-2 shift was -4.737 and the Stage-2-to-Stage-3 recovery was +4.547. Seven of fifteen pairs crossed the decision boundary in the complete $A \rightarrow B \rightarrow A$ sequence. The effect was substantially stronger in balanced held-out contexts: the mean margin changed from +0.015 at Generic to +3.184, -2.097, and +2.230 across the three intervention stages. Fourteen of fifteen pairs (93.3%) exhibited the complete $A \rightarrow B \rightarrow A$ preference reversal on unseen contextual examples. These results indicate that the Adapter can acquire, reverse, and recover user-specific lexical preferences without modifying the frozen Generic model.

---

## 26. Thesis-Ready Interpretation Paragraph

The controlled ABA experiment provides evidence that model-level personalisation is not limited to accumulating static frequency information. The learned user representation remains plastic: recent evidence can substantially attenuate or override a previously learned preference, while subsequent evidence can restore the earlier preference. This property is important for practical personalised input methods because user preferences may evolve over time. The strong contextual reversal rate further suggests that the adaptation generalises beyond the specific controlled training examples and interacts meaningfully with the contextual representations supplied by the frozen base model.

---

## 27. Concise Main Finding

> **The model-level Adapter learned a reversible user-specific lexical preference: on frozen held-out contexts, the mean A-B margin followed a near-neutral -> A -> B -> A trajectory (+0.015 -> +3.184 -> -2.097 -> +2.230), with 14 of 15 candidate pairs completing the full reversal-and-recovery pattern.**

---

## 28. Reproducibility Notes

- Number of controlled candidate pairs: **15**.
- Unique controlled Pinyin sequences: **15**.
- Frozen contextual probes: **120** total.
- Contexts per pair: **8 = 4 A-origin + 4 B-origin**.
- Training rows per stage: **5,000**.
- Controlled rows per stage: **600**.
- Controlled exposures per pair per stage: **40**.
- Background rows per stage: **4,400**.
- Background policy: exclude controlled A/B targets only; retain other naturally occurring candidates sharing the same Pinyin.
- Sequential model path: **Generic -> Stage 1 A -> Stage 2 B -> Stage 3 A**.
- Base PinyinGPT2-Concat parameters remain frozen.
- The same user Adapter persists across stages.
- Optimizer is reset between stages.
- Test split used: **false**.
