# 14 - Task-Specific Bi-Encoder Design and Cost Gate

Date: 2026-08-22

Status: **DESIGNED / TRAINING DEFERRED ON CURRENT EVIDENCE**

## 1. Research question

Can a cacheable representation trained for IME historical utility improve on
generic BGE retrieval, and does that retrieval gain survive the candidate-level
decision layer?

This is a new named experiment, not a modification of M1, EM2, EM3, or the
completed LambdaMART result.

## 2. Evidence considered after Experiments A and B

- Experiment A produced only +0.0511 Macro percentage points and its
  prior-specific contribution was weak relative to simple Choice Share
  suppression.
- Experiment B produced +0.2834 Macro points and +4.396 Conflict points. Its
  largest contributions were the frozen score, score gap, base score, and
  history depth. BGE support/gap were useful but secondary.
- Historical EM2 showed that the frozen PinyinGPT task-native state improved
  retrieval discrimination over BGE, including Conflict Recall@10 from 84.39%
  to 93.17% on that historical surface, yet Hidden-M1 and Original-M1 were
  essentially tied end to end. Better retrieval alone did not solve the
  decision problem.
- The current fixed surface has 5.198% Missing@10. A retriever cannot change
  that without a separately designed candidate-generation experiment.

This evidence does not say a task-specific bi-encoder cannot help. It says its
expected marginal value is lower and less direct than nonlinear decision
fusion, which has already produced the reproducible improvement sought in this
phase.

## 3. Scientifically clean future design

### Encoder and serialization

- Initialize from the full-precision `BAAI/bge-small-zh-v1.5` checkpoint, not
  the inference-only Q8 GGUF. The official model card records 24M parameters
  and 512-dimensional embeddings.
- Use one shared-weight encoder for query and historical context so all history
  vectors remain independently cacheable.
- Preserve the current BGE input ablation: encode only the most recent 64
  characters of current or historical context. Exact segmented Pinyin remains
  an external legal-memory filter, not a learned shortcut.
- Do not serialize current gold, historical target, author identity, future
  rows, or rescue/harm labels into either encoder input. Historical target is
  used only to define supervision and to aggregate retrieved runtime evidence.

### Training population and objective

Reuse the audited Clean3 Train-Fit causal pair infrastructure:

```text
same author
-> strictly prior
-> latest H5000 raw interactions
-> exact segmented-Pinyin
-> matching-target positive / wrong-target hard negative
```

The existing generator produced 35,290 eligible queries, 99,671 positive
pairs, 169,400 negative pairs, and 269,071 unique query/history pairs with
zero non-prior pairs. Group one positive with its query-local hard negatives
and train a temperature-scaled listwise contrastive loss.

Do not use unrestricted in-batch negatives in the first experiment. Different
queries can make another query's positive a false negative, while different
Pinyin groups are often trivially easy. Query-local same-Pinyin wrong-target
negatives already match the deployed confound. A later masked in-batch
ablation would require a separate predeclared design.

This follows the general dense-retrieval evidence for independently cacheable
dual encoders and informative hard negatives, while keeping the repository's
stricter IME causal rule. Relevant primary references are
[DPR](https://arxiv.org/abs/2004.04906),
[ANCE](https://arxiv.org/abs/2007.00808), and
[Sentence-BERT](https://arxiv.org/abs/1908.10084). The model and official
fine-tuning entry points are documented by
[BAAI's BGE model card](https://huggingface.co/BAAI/bge-small-zh-v1.5).

### Evaluation isolation

1. Create a chronological per-author inner gate wholly inside Train-Fit for
   optimizer/schedule checks; do not inspect Train-Val during those choices.
2. Freeze one checkpoint and retrieval configuration.
3. On Train-Val, first compare generic BGE and task-specific retrieval with
   identical candidate/history populations and Recall@1/5/10 plus target-level
   support diagnostics.
4. Substitute only the BGE representation in the existing fixed aggregation
   and frozen linear fusion. This is the primary representation ablation.
5. Only if retrieval and fixed-fusion gates pass, run a separately named
   LambdaMART refit using the new support. That is a representation-plus-fusion
   experiment, not the primary ablation.

Dev3000 remains already observed and unavailable for design or selection.
Test remains closed. A later confirmatory claim requires a new holdout proposal
before any data is opened.

## 4. Cost estimate on the current Windows machine

This is an engineering estimate, not a measured run:

| Component | Estimate |
|---|---|
| Full-precision BGE checkpoint | about 100 MB model weights; new pinned download/hash required |
| Training examples | 99,671 grouped queries / 269,071 pair rows |
| RTX 4060 8 GB training | feasible with FP16 and bounded batches; approximately 1-2 GPU hours for a small fixed schedule, including inner-gate scoring |
| History/query embedding cache | 143,891 required contexts on the current surface; about 295 MB raw float32 vectors before SQLite overhead |
| Cache generation | likely minutes with batched HF inference; must be measured and recorded |
| Fixed-fusion evaluation | no new PinyinGPT Generic inference; arithmetic/support regeneration only |

The full-precision checkpoint, optimizer state, embeddings, logs, and model
outputs would remain local-only.

## 5. Decision

Do **not** start the expensive training run in this phase. The reason is
scientific, not infrastructural:

1. A reproducible gain has already been found at the decision layer.
2. Existing task-native retrieval evidence did not reliably convert to final
   ranking improvement.
3. BGE-derived features are not the dominant learned contributions.
4. Repeatedly adding methods on the same observed Train-Val surface increases
   selection risk without a new confirmatory holdout.

The bi-encoder remains a well-specified next experiment if representation
learning is required as a thesis ablation. Before training, freeze the inner
Train-Fit split, checkpoint revision, exact loss, temperature, schedule, batch
policy, and one-shot Train-Val evaluation in a new design record. No Dev3000
or Test data was used for this gate.
