# Transparent, User-Controllable Pinyin Personalisation

## Current Phase

Phase 4D — Transparent Contextual Memory Retrieval

## Current Objective

Phase 4D evaluates a transparent contextual-memory model motivated by the
completed Phase 4C negative result. Phase 4C found that frequency
personalisation did not outperform the strong Luna baseline and that exact
context evidence was usually absent.

The new model retrieves similar contexts from a frozen user's history while
preserving explicit frequency fallback, confidence, and candidate-level score
traces. The implementation is ready for testing, but the final Phase 4D
experiment has not been run and no improvement is claimed.

## Frozen Design

For each existing 12-character derived context and normalized Pinyin:

```text
same-Pinyin frozen user history
        ↓
character TF-IDF cosine, n-grams (1,2)
        ↓
Top-5 positive-similarity contexts
        ↓
context evidence C(y), confidence q
        +
frequency fallback F(y)
        ↓
U(y) = (1-q)F(y) + qC(y)
        ↓
0.5 normalized Luna Base + 0.5 U(y)
```

The no-gate ablation uses `U(y)=C(y)`. Parameters are frozen: `K=5`,
global/Pinyin frequency weights `0.25/0.75`, and Base blending `alpha=0.5`.
Final-score ties retain Base order.

## Evaluation

Phase 4D reuses the exact Phase 4C splits and compares:

1. Base Luna;
2. Phase 4C frequency personalisation;
3. Phase 4D no-gate correct-user;
4. Phase 4D full correct-user;
5. Phase 4D full wrong-user.

Metrics cover the full 926-interaction Zhu test benchmark and its Base-rerankable
subset. In addition to Top-K, MRR, mean rank, and rank-change counts, Phase 4D
reports same-Pinyin eligibility, non-zero contextual matches, similarity
statistics, and context-involved improved/harmed counts.

Correct-user memory contains only Zhu training interactions. Wrong-user memory
contains only Lu training interactions. No test or future interaction enters
either memory, and no test-time update occurs.

## Transparency

Every evaluated query stores the full retrieval trace and, for every candidate:

- Base rank and normalized ordinal utility;
- raw and normalized global/Pinyin counts and `F(y)`;
- retrieved contributors, similarity-weighted `C(y)`, and `q`;
- final personal evidence `U(y)`, final score, and final rank.

The recorded fields reconstruct every ranking decision.

## How to Run

Run all tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

After accepting the implementation, manually run Phase 4D:

```bash
.venv/bin/python -m experiments.exp_phase_04d_context_personalisation
```

The experiment writes `results/experiments/phase_04d/evaluation.json`. The
directory currently contains only a placeholder; no Phase 4D result has been
generated.

## Current Limitations

- Character n-gram similarity captures lexical overlap, not semantics.
- The context remains the existing 12-character preceding string.
- Retrieval cannot use histories with a different normalized Pinyin.
- All parameters are frozen and untuned.
- Reranking cannot recover candidates absent from Base Top-10.

## Project History

- Phase specifications: [`docs/phases/`](docs/phases/)
- Completed results: [`results/experiments/`](results/experiments/)
- Audit outputs: [`results/audits/`](results/audits/)
- Workflow rules: [`docs/WORKFLOW.md`](docs/WORKFLOW.md)

Git tags are not created automatically.
