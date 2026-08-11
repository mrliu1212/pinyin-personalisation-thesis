# Transparent, User-Controllable Pinyin Personalisation

## Current Phase

Phase 4E — Hybrid Neural-Transparent Personalisation

## Current Objective

Phase 4E implements a frozen hybrid architecture motivated by Phase 4D's
lexical-context limitations. It combines Luna candidates, causal-LM semantic
context scores, semantic retrieval from frozen user history, explicit
behavioural features, and transparent personal-vocabulary injection through an
interpretable linear pairwise reranker.

The implementation and engineering smoke test are complete. The full Zhu
Ziqing Phase 4E evaluation has deliberately not been run, so no performance
improvement is claimed.

## Frozen Design

For each interaction, the existing 12-character context remains available for
audit while neural features use only the final 64 Chinese characters preceding
the target:

```text
Luna Top-10 ────────────────→ Base features
64-character context ───────→ frozen Qwen causal-LM features
same-Pinyin user history ───→ frozen Qwen semantic Top-5 memory
chronological user history ─→ behaviour + optional personal vocabulary
                                      ↓
                         standardized linear pairwise reranker
                                      ↓
                     factor-decomposable candidate ranking
```

The frozen models are `Qwen/Qwen3-0.6B-Base` and
`Qwen/Qwen3-Embedding-0.6B`, pinned to exact repository revisions in the Phase
4E model manifest. They remain frozen and are used only for scoring and
embedding—not generation. Retrieval is exact-same-Pinyin with `K=5`, personal
vocabulary injection is capped at three, and the logistic-regression/scaling
configuration is fixed rather than tuned on the test set.

## Evaluation

Phase 4E reuses the exact Phase 4C splits and prepares exactly seven conditions:

1. Base Luna;
2. existing Phase 4D no-gate correct-user;
3. Phase 4E generic neural context;
4. Phase 4E hybrid fixed-pool correct-user;
5. Phase 4E hybrid fixed-pool wrong-user;
6. Phase 4E hybrid augmented-pool correct-user;
7. Phase 4E hybrid augmented-pool wrong-user.

Metrics cover the full Zhu test benchmark and its original Luna-rerankable
subset, with separate reporting for both test works. Besides Top-K, MRR, mean
rank, missing targets, coverage, and rank changes, the framework records neural,
semantic-memory, behaviour, vocabulary-recovery, learned-weight, McNemar, and
paired-bootstrap diagnostics.

Correct-user memory contains only Zhu training interactions. Wrong-user memory
contains only Lu training interactions. No test or future interaction enters
either memory, and no test-time update occurs.

## Transparency

Every evaluated query stores the complete semantic retrieval trace and, for
every candidate:

- Base rank, source, ordinal utility, and structural features;
- causal-LM conditional score, prior, context gain, and normalized values;
- semantic-memory evidence plus all retrieved cases and similarities;
- behavioural counts, shares, seen status, and recency;
- standardized values, learned coefficients, exact feature/factor
  contributions, final score, and final rank.

The recorded fields reconstruct every ranking decision. Selected examples also
support factor-removal and individual-memory-deletion counterfactuals without
retraining.

## How to Run

Run all tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Prepare or inspect the pinned model manifest:

```bash
.venv/bin/python -m experiments.exp_phase_04e_hybrid_personalisation --prepare-models
```

Run the small engineering-only neural smoke test:

```bash
.venv/bin/python -m experiments.exp_phase_04e_hybrid_personalisation --smoke-test
```

After accepting the frozen implementation, manually run the final experiment:

```bash
.venv/bin/python -m experiments.exp_phase_04e_hybrid_personalisation
```

The final command writes `results/experiments/phase_04e/evaluation.json`. That
complete evaluation has not been run in this implementation checkpoint.

## Current Limitations

- Pretrained-data membership for the historical authors is unknown, so this is
  not a clean unseen-text generalisation benchmark.
- Neural feature extractors are internally opaque; auditability is at the
  explicit feature/factor and retrieved-case decision layer.
- Semantic memory remains restricted to exact normalized Pinyin.
- Personal vocabulary can inject only candidates observed in frozen personal
  history and cannot invent unseen vocabulary.
- The design is a frozen post-hoc extension because the Zhu test set was
  observed in earlier phases.

## Project History

- Phase specifications: [`docs/phases/`](docs/phases/)
- Completed results: [`results/experiments/`](results/experiments/)
- Audit outputs: [`results/audits/`](results/audits/)
- Workflow rules: [`docs/WORKFLOW.md`](docs/WORKFLOW.md)

Git tags are not created automatically.
