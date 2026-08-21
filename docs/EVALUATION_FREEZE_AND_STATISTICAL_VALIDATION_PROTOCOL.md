# Evaluation, Freeze, and Statistical Validation Protocol

**Scope:** General project-wide protocol for model development, validation, and final evaluation.

## 1. Dataset roles

The evaluation pipeline is:

```text
Train-Fit
    ->
Train-Val
    ->
method development / hyperparameter selection
    ->
PRE-DEV FREEZE
    ->
Dev3000
    ->
FINAL FREEZE
    ->
Test
```

### Train-Fit
Used to construct training-time or history-derived information permitted by the experiment protocol.

### Train-Val
Used for:
- method development;
- hyperparameter selection;
- ablations;
- diagnostics;
- preliminary paired statistical analysis.

Because Train-Val participates in method selection, statistical tests on Train-Val are **development-stage stability analyses**, not fully independent confirmatory evidence.

### Dev3000
Used only after the method, coefficients, evaluation metrics, and comparator set have been frozen.

Dev3000 is used to check whether the frozen conclusions generalize beyond the development sample. It must not be used to reopen parameter tuning or select a new primary method.

### Test
Used only after the final method specification has been frozen.

Test is the final unbiased evaluation set. Test results must not be used to modify the method.

---

## 2. Statistical tests

### McNemar test

Use McNemar's test for paired binary outcomes such as Top-1 correctness.

For two methods A and B on the same queries, the relevant counts are:

```text
A correct, B wrong
A wrong,   B correct
```

The test asks whether these two disagreement counts are systematically different.

McNemar therefore evaluates whether one method has a reliable paired Top-1 advantage over another on the evaluated sample.

### Paired bootstrap

Use paired bootstrap to estimate uncertainty in metric differences such as:

- Macro-author Top1;
- Micro Top1;
- Top3;
- Top5;
- MRR@10;
- Missing@10.

The same sampled query must be included for both methods so that the paired structure is preserved.

For Macro-author metrics, bootstrap **within each author independently**, recompute the metric for each author, and then take the equal-weight macro average. This keeps the resampling procedure aligned with the definition of Macro-author evaluation.

A bootstrap confidence interval crossing zero means that the observed metric difference is not stable enough to establish a directional advantage under that resampling analysis.

---

## 3. Interpretation by evaluation stage

### Train-Val statistical analysis

Train-Val McNemar tests and bootstrap intervals should be described as:

> pre-freeze development stability analysis

They can be used to determine whether an observed development gain is fragile or reasonably stable, but they should not be presented as fully independent confirmatory significance tests because the same Train-Val data were used during method selection.

### Dev3000 statistical analysis

After PRE-DEV FREEZE, the same paired comparisons may be run on Dev3000.

Because Dev3000 did not participate in method selection, agreement between Train-Val and Dev3000 provides stronger evidence that the frozen result generalizes beyond the development sample.

### Test statistical analysis

If paired statistical tests are reported on Test, they apply only to the already frozen final methods and must not trigger any further method changes.

---

## 4. Freeze rules

Before opening Dev3000, freeze:

- primary method;
- comparator methods;
- all coefficients and hyperparameters;
- candidate-surface definition;
- history semantics;
- primary and secondary metrics;
- evaluation populations;
- exact scripts and relevant SHA256 hashes;
- statistical comparison plan.

After PRE-DEV FREEZE:

- do not tune coefficients on Dev3000;
- do not select a different primary method because of Dev3000 performance;
- do not change the primary metric;
- do not introduce new post-hoc comparator configurations to improve the result.

After FINAL FREEZE:

- open Test once for final evaluation;
- do not modify the method based on Test outcomes.

---

## 5. Thesis-safe interpretation

Use wording such as:

> Statistical comparisons on Train-Val are treated as development-stage stability analyses because Train-Val was also used for method selection. Confirmatory evidence is obtained only after the method specification is frozen and evaluated on previously untouched data.

and:

> McNemar's test evaluates paired Top-1 correctness differences, while paired bootstrap intervals quantify uncertainty in broader ranking metrics. For Macro-author metrics, resampling is performed independently within each author before equal-weight macro aggregation.

The central principle is:

```text
Train-Val = develop and diagnose
Dev3000   = independently confirm the frozen method
Test      = final unbiased evaluation
```
