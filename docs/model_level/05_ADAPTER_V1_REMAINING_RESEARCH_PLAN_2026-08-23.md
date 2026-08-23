# Model-Level Adapter V1: Remaining Research Plan

Date: 2026-08-23

## 1. Objective

The remaining Model-Level research will focus on completing and evaluating the
existing Adapter V1 approach rather than expanding the architecture search.

The main questions are:

1. Does Model-Level Adapter personalisation remain effective on the complete
   held-out development population?
2. How does performance change as the amount of per-user training history
   increases?
3. How does Model-Level personalisation compare with the existing
   external-memory / candidate-level personalisation system?
4. Are Model-Level and external-memory personalisation complementary when
   combined in a Hybrid system?
5. Do the final conclusions hold on the untouched Test partition?

No broad search over alternative Adapter architectures, LoRA variants,
bottleneck sizes, placements, or training-mixture families is planned for the
current thesis scope.

---

## 2. Current Adapter V1 Configuration

The current Adapter V1 architecture remains fixed unless a critical failure is
identified.

### Base model

- PinyinGPT2-Concat
- Base model frozen
- 12 Transformer blocks
- One residual serial bottleneck Adapter after each completed block
- Hidden size: 768
- Bottleneck size: 48
- ReLU activation
- Frozen final LayerNorm and tied LM head
- Adapter parameters per user: 894,528

### Current training protocol

- Train-Fit is the only partition used to update Adapter parameters.
- Train-Val is held out for development, model selection, and calibration.
- Test remains untouched until the final protocol is frozen.
- Current Pinyin exposure:
  - Full : Initial : Mixed = 3 : 1 : 2
- Current target setting:
  - Short targets
- Current primary LR candidate:
  - 5e-4
- Base model remains frozen throughout Adapter training.

The current multi-condition training protocol involving joint Short and
Multi1-Multi5 targets is deferred to Future Work.

---

## 3. Stage A: Complete Full Agent Phage Validation

The immediate task is to complete the currently running full Train-Val
confirmation for the LR=5e-4 Adapter.

### Population

- Author: Agent Phage
- Full Agent Phage Train-Val population: 13,741 rows
- Full Pinyin evaluation
- Short targets
- Beam size: 16
- Top-K: 10

### Existing Generic full-validation baseline

N = 13,741

- Top1 = 0.8199548796
- Top3 = 0.9402518012
- Top5 = 0.9581544284
- Top10 = 0.9705989375
- MRR = 0.8820930699
- Missing@10 = 0.0294010625

### Decision

If LR=5e-4 remains clearly beneficial on the full Agent Phage Train-Val
population, freeze LR=5e-4 for Adapter V1.

Do not perform full-validation reruns for 5e-5, 1e-4, or 2e-4 unless the 5e-4
confirmation exposes an unexpected failure.

---

## 4. Stage B: Training-Data Learning Curve

After LR confirmation, measure how Model-Level personalisation depends on the
amount of available user history.

This experiment is performed only for Agent Phage.

### Training sizes

Use deterministic nested Train-Fit populations:

- 2,048 rows
- 8,192 rows
- 32,768 rows
- 55,926 rows (full Agent Phage Train-Fit)

The subsets should satisfy:

2,048 subset 8,192 subset 32,768 subset Full

where each smaller population is contained in the next larger population.

### Fixed variables

Across the learning-curve models, keep fixed:

- Adapter architecture
- bottleneck = 48
- LR = 5e-4
- optimizer
- batch size
- seed
- Pinyin exposure policy
- loss definition
- training epoch count

Only the amount of Train-Fit data should change.

### Evaluation

Use the existing frozen 2,048-row Agent Phage Train-Val subset for efficient
comparison of all learning-curve models.

Compare:

- 2K training
- 8K training
- 32K training
- Full Train-Fit training

on exactly the same 2,048 development rows.

The full Train-Fit model should additionally be evaluated on all 13,741 Agent
Phage Train-Val rows.

### Purpose

This experiment should answer:

- how much user history is needed before Adapter personalisation becomes useful;
- whether performance continues to improve with more history;
- whether gains begin to saturate;
- whether Model-Level personalisation appears data-hungry relative to
  external-memory personalisation.

---

## 5. Stage C: Epoch / Training-Duration Decision

Do not perform a broad epoch search by default.

Start with the current one-epoch protocol.

If the learning curve and full Train-Fit result show no clear evidence of
under-training, retain one epoch.

Only if the full-data model appears clearly under-trained should an additional
two-epoch experiment be run.

The goal is to avoid turning training duration into another large
hyperparameter search.

---

## 6. Stage D: Freeze Adapter V1 Protocol

After LR and training-duration decisions, freeze the Model-Level Adapter V1
protocol.

The frozen protocol should include:

- architecture
- Adapter placement
- bottleneck size
- activation
- LR
- optimizer
- batch size
- epoch count
- random seed / deterministic policy
- Pinyin exposure policy
- loss
- Beam16 / Top10 evaluation protocol
- Train-Fit / Train-Val / Test partition semantics

No Test result may be used to modify this protocol.

---

## 7. Stage E: Formal Three-User Adapter Training

Train one independent Adapter per user using only that user's complete
Train-Fit population.

Current Train-Fit sizes:

- Agent Phage: 55,926 rows
- Etinjat: 32,906 rows
- breaddddd: 55,694 rows

The frozen PinyinGPT2-Concat base model is shared across users.

Only the per-user Adapter parameters are updated.

Train-Val is not merged back into training.

---

## 8. Stage F: Three-User Full Train-Val Evaluation

Evaluate the formal per-user Adapters on the complete held-out Train-Val
population.

Primary comparison:

1. Generic PinyinGPT2-Concat
2. External-memory / candidate-level personalisation
3. Model-Level Adapter

Report at least:

- Top1
- Top3
- Top5
- Top10
- MRR@10
- Missing@10
- micro aggregate
- equally weighted macro-user aggregate

Where row-level predictions are available, additionally analyse:

- rescue
- harm
- net rescue
- overlap between errors corrected by external memory and Adapter
- cases corrected only by Adapter
- cases corrected only by external memory

This overlap analysis should guide the Hybrid experiment.

---

## 9. Stage G: Hybrid Personalisation

Hybrid is a planned main experiment rather than only Future Work.

The intended architecture is:

Pinyin + preceding context
    ->
Personalised Adapter PinyinGPT2-Concat
    ->
Adapter-specific autoregressive Beam16 candidate generation
    ->
External-memory / Final reranker
    ->
Hybrid final ranking

The Adapter must generate its own candidate set because Model-Level
personalisation changes Transformer hidden states, token probabilities, beam
pruning, and therefore the generated candidate population.

Generic Beam caches must not be treated as Adapter-generated candidates.

### Main Hybrid question

Determine whether:

- Model-Level Adapter personalisation and
- external-memory personalisation

capture complementary user-specific information.

A strong Hybrid gain would support complementarity.

A small or negligible Hybrid gain would suggest that the two methods capture
largely overlapping personalisation signals.

Both outcomes are scientifically meaningful.

---

## 10. Stage H: Final Untouched Test

Only after all development decisions are frozen should the Test partition be
used.

The intended final comparison is:

1. Generic
2. Best external-memory method
3. Model-Level Adapter
4. Hybrid

The final per-user Adapters are trained using Train-Fit only.

Train-Val remains a development/model-selection population and is not used for
final parameter updates.

After Test evaluation begins, no architecture, LR, epoch, training mixture, or
ranking rule may be changed based on Test results.

---

## 11. Explicitly Deferred Work

The following are outside the current main experimental scope and are recorded
as Future Work:

- Full-only versus Full+Initial+Mixed training ablation
- alternative Full / Initial / Mixed exposure ratios
- joint Short + Multi training
- Multi1-Multi5 exposure-ratio optimisation
- unified multi-condition Adapter training
- alternative Adapter bottleneck sizes
- alternative Adapter placement
- alternative Adapter activation functions
- LoRA versus Adapter architecture search
- large optimizer search
- broad LR search beyond the existing calibration
- major batched-generation engineering unless evaluation throughput becomes a
  blocking issue

These questions remain scientifically interesting but are deliberately deferred
to preserve a focused and feasible thesis scope.

---

## 12. Remaining Main Pipeline

The intended remaining sequence is:

1. Complete LR=5e-4 full Agent Phage Train-Val confirmation.
2. Freeze LR if confirmation succeeds.
3. Run Agent Phage training-data learning curve:
   2K -> 8K -> 32K -> Full.
4. Decide whether one epoch is sufficient.
5. Freeze Adapter V1 training protocol.
6. Train formal Adapters for all three users.
7. Run complete three-user Train-Val evaluation.
8. Compare Generic, external memory, and Model-Level Adapter.
9. Analyse rescue/error overlap.
10. Build and evaluate Hybrid.
11. Freeze all final methods.
12. Run untouched Test once.
13. Produce final thesis tables, figures, interpretation, and limitations.

The guiding principle for the remaining work is to finish the existing
Model-Level approach rigorously rather than continuously expanding the search
space.
