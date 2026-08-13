# Research Targets

This document is the authoritative direction for future thesis work. Earlier
phases remain valid historical research artifacts, but they do not override the
direction recorded here.

## 1. Current Research Question

How can long-term, user-specific history improve Chinese candidate ranking for
Pinyin input—especially Top-1 accuracy—over the same strong generic Pinyin
backend without personalisation?

## 2. Task Definition

The task is **long-term user-specific personalisation for Pinyin candidate
ranking**:

```text
preceding Chinese context
        +
Pinyin currently typed by the user
        +
the user's past input history
        ↓
ranked valid Chinese candidates
        ↓
place the user's intended Chinese candidate as high as possible,
ideally Top-1
```

For example, given context `这个工具真的很`, Pinyin `shiyong`, and candidates
such as `使用`, `实用`, `适用`, and `试用`, the objective is to rank the intended
target `实用` as highly as possible for that specific user.

## 3. Explicit Non-goals

The current task is not:

- next-word prediction;
- sentence completion;
- conversational response generation;
- GhostText;
- generic AI autocomplete.

## 4. Primary Technical Research Targets

### RT1 — Ranking Accuracy

Use correct-user historical information to improve the rank of the intended
Chinese candidate, with Top-1 accuracy as the primary objective. The
personalised system should outperform the same strong generic Pinyin backend
without personalisation.

### RT2 — Personal Vocabulary

Learn user-specific words and expressions that a generic system may rank very
low or omit from its normal candidate pool. Past user selections should allow
relevant personal vocabulary to enter the candidate pool and potentially reach
Top-1.

### RT3 — Context-Sensitive Personal Preference

Do not reduce personalisation to simple same-Pinyin frequency. The same user
may intend different Chinese targets for the same Pinyin in different contexts:

```text
这个工具非常 + shiyong -> 实用
已经开始     + shiyong -> 使用
免费         + shiyong -> 试用
```

The personalisation system should use both current context and user-specific
history to distinguish these cases.

### RT4 — Temporal Adaptation

User preferences change over time. Recent behaviour should be able to modify
or supersede older habits instead of allowing historical frequency to dominate
forever. The system should demonstrate useful adaptation as longitudinal user
history accumulates.

## 5. Evaluation Principles

The exact evaluation protocol is not frozen yet. The following principles are
authoritative.

The primary metric is **Top-1 candidate accuracy**. Important secondary metrics
include Top-3, Top-5, MRR, mean target rank, and candidate coverage/missing
targets.

Evaluation must eventually measure:

1. generic backend versus correct-user personalisation;
2. correct-user versus wrong-user history;
3. the proposed method versus simple same-Pinyin frequency personalisation;
4. improved versus harmed predictions;
5. recovery of personal vocabulary;
6. same-Pinyin multi-target and context-sensitive cases;
7. performance as user history accumulates over time.

Strict chronology is mandatory: past user history may influence only future
held-out interactions. Future or test selections must never influence the
personal model. The final test set must not be used for model design or
hyperparameter tuning.

## 6. Current PinyinGPT Reference-Backend Direction

The current leading reference-backend candidate is PinyinGPT, particularly
PinyinGPT2-Concat, from *Exploring and Adapting Chinese GPT to Pinyin Input
Method* (ACL 2022).

PinyinGPT is not yet permanently frozen as the final backend. It must first
undergo a reproduction and technical audit. The intended high-level direction
is a strong generic contextual Pinyin model combined with long-term,
user-specific personalisation to improve candidate ranking. The generic model
and personalisation layer should preferably remain conceptually separable.
Per-user fine-tuning must not be assumed to be the final personalisation
approach.

## 7. Current Model Work and Later Transparency/Control Work

The current implementation priority is to establish a strong personalised
Pinyin-ranking model. Transparency and controllability remain important later
thesis contributions, including explaining promotions, identifying influential
history, disabling personal influence, deleting history, correcting learned
preferences, resetting personalisation, counterfactual reranking, and preserving
useful personalisation while correcting harmful effects.

These controls are not part of the current implementation stage. The model
architecture should avoid unnecessarily blocking them, but no UI or control
layer should be implemented yet.

## 8. Historical Status of HuoziIME

HuoziIME remains historical Phase 4F work and must not be deleted or rewritten.
Its memory and provenance ideas may provide useful later inspiration, but its
contextual AI-completion task is not the thesis's main Pinyin candidate-ranking
task. The previously planned HuoziIME-based Phase 5 must not continue as the
main research direction.

## 9. Required Development Order

1. Freeze research targets.
2. Audit/reproduce PinyinGPT baseline.
3. Design and freeze evaluation protocol.
4. Design personalisation architecture.
5. Develop/tune using development data only.
6. Freeze final model.
7. Run untouched final evaluation.
8. Add/evaluate transparency and controllability.

## Candidate Personalisation Directions — Not Yet Frozen

These are research candidates, not final architecture decisions. They may later
be compared experimentally; external reranking is not assumed to be the final
solution, and later transparency and controllability remain relevant when
choosing between them. Simple frequency or recency may be useful as a basic
evaluation baseline, but is not a major research direction.

### 1. Context-Aware Memory Ranking

Store historical same-user interactions containing context, Pinyin, and the
candidate actually selected. For a new context and Pinyin query, retrieve
relevant history and use that contextual user-specific evidence—not only
frequency counts—to influence candidate ranking.

### 2. Personal Vocabulary Augmentation

Allow user-specific words, names, terminology, and expressions learned from
past selections to enter the candidate pool when compatible with the typed
Pinyin. This addresses intended targets absent from the generic candidate set,
not merely targets ranked too low.

### 3. Internal PinyinGPT Personalisation

Investigate personalisation that affects PinyinGPT prediction or constrained
decoding directly rather than only post-hoc reranking. Possible later
mechanisms include user-specific output bias, lightweight adapters, or other
small internal adaptations; no mechanism is frozen yet.

### 4. User Representation / User Embedding

Investigate a learned representation of long-term user behaviour derived from
historical inputs, so prediction may condition on current context, current
Pinyin, and a user representation. That representation may eventually be
dynamic and history-derived rather than a fixed user-ID embedding.
