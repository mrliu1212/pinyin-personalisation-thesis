# EM-2 -> EM-3 Technical Handoff - 2026-08-19

## Stage state

EM-2 is **CLOSED**.

Do not reopen EM-2 to:

- try a different hidden layer;
- try a different pooling rule;
- extend the Adaptive gate;
- retune M1/M2;
- open EM-2 Test and then revise Dev choices.

The new active stage is **EM-3**.

## What EM-2 established

1. Frozen PinyinGPT final-[SEP] hidden states are valid task-native memory keys.
2. Hidden-state retrieval is stronger than BGE retrieval on the same Dev retrieval surface.
3. Stronger retrieval does not materially improve the unchanged M1 decision rule.
4. Hidden retrieval slightly helps the generic M2 Cross-Encoder, but M2 still does not beat the M1 family.
5. Fixed Frequency + Hidden Context fusion does not add meaningful Overall value.
6. The tested transparent count-aware adaptive gate is worse; the no-count control is only marginally interesting.
7. The remaining research problem is historical **relevance/decision**, not simply history retrieval.

## EM-3 research question

> Can a task-specific learned historical relevance model predict which strictly-prior personal historical interactions are useful for the current Pinyin candidate decision?

## EM-3 design constraints to preserve

- same-user / same-author causal history;
- strictly-prior information only;
- no Test tuning;
- no Gold in inference-time features;
- exact definition of training labels must be frozen before training/evaluation;
- train/dev/test separation must prevent leakage through historical interactions;
- candidate/history pair construction must be documented;
- task-specific Cross-Encoder provenance must distinguish prior generic Cross-Encoder work from this thesis adaptation;
- evaluation must separate retrieval/relevance quality from final candidate ranking.

## Useful EM-2 assets

- frozen legal `HistoryIndex` semantics;
- Frozen Generic candidates;
- hidden-state cache and hidden retrieval runner;
- Ambiguous and Conflict diagnostic definitions;
- Original M2 pair construction / Cross-Encoder infrastructure;
- same-surface comparison utilities.

Reuse infrastructure where appropriate, but do not relabel changed EM-3 semantics as EM-2.

## First EM-3 action

Before implementation:

1. define the training target for a `(current query, historical interaction, candidate)` relevance example;
2. define negative sampling;
3. define causal split/leakage constraints;
4. define Dev selection metric;
5. define comparison against the frozen generic M2 Cross-Encoder;
6. write the EM-3 design/protocol document.
