# EM-2 Hidden-State kNN Retrieval Design

Status: DESIGN FROZEN BEFORE IMPLEMENTATION
Date: 2026-08-19

## Research question

Can the Frozen PinyinGPT model's own hidden representation retrieve more
useful same-user historical examples than the existing BGE semantic embedding?

EM-2 begins as a retrieval diagnostic, not an end-to-end reranking experiment.

## Scope

Condition:
- Full+Short

History:
- H5000

Authors:
- Etinjat
- Re_spectators
- breaddddd

Partition:
- Dev only during method development

No Test evaluation or Test-based method decisions are permitted.

## Legal memory

For each query, retrieval may use only:
- same user;
- strictly-prior history;
- within H5000;
- exact same segmented Pinyin.

## Representation

Query key:

Frozen PinyinGPT(
    preceding context + current segmented Pinyin
)

Historical key:

Frozen PinyinGPT(
    historical preceding context + historical segmented Pinyin
)

The representation is the final-layer hidden state at the prompt position
used to predict the first target Chinese character.

The target character itself must not be part of the key input.

Historical target is stored only as the memory value.

key   = hidden representation
value = historical target

PinyinGPT remains fully frozen.

## Similarity

Cosine similarity.

No learned similarity function is introduced in EM-2.

## Evaluation

Gold is used only after retrieval to evaluate whether a retrieved historical
target matches the current Gold.

Gold must not affect:
- representation extraction;
- legal-history filtering;
- similarity;
- retrieval order.

Primary metric:
- Macro-author Ambiguous R@1

Secondary:
- Overall R@1
- Conflict R@1
- Overall/Ambiguous/Conflict R@5
- Overall/Ambiguous/Conflict R@10

## Baseline

Compare against the frozen BGE retrieval baseline:
- bge-small-zh-v1.5
- ctx64
- cosine similarity
- H5000
- exact same segmented Pinyin

## No-tuning rule

Do not:
- inspect Test while developing EM-2;
- sweep hidden layers after seeing retrieval results;
- change extraction position because retrieval performance is poor;
- tune similarity using Gold.

Any extraction change must be justified by an independent engineering error.

## EM-2 stages

EM-2A:
Hidden-state extraction engineering gate.

EM-2B:
Dev hidden-state cache.

EM-2C:
Frozen cosine kNN retrieval diagnostic.

EM-2D:
BGE ctx64 vs PinyinGPT hidden-state comparison.

EM-2E:
Only if Dev retrieval is useful, design end-to-end reranking.

## Relationship to External Memory stages

EM-1:
Candidate recovery.

EM-2:
Frozen task-native hidden-state retrieval.

EM-3:
Task-specific supervised historical-evidence matcher.

EM-4:
Final External Memory fusion.

## Research provenance and inspiration

EM-2 is not presented as inventing nearest-neighbour memory for language
models.

Its main methodological precedent is:

Khandelwal et al. (2020), "Generalization through Memorization:
Nearest Neighbor Language Models".

kNN-LM constructs a key-value datastore in which a pretrained language
model representation of a context is used as the key and the corresponding
target token is stored as the value. At inference time, the representation
of the current context is used to retrieve similar historical contexts,
without retraining the underlying language model.

A second relevant precedent is:

Khandelwal et al. (2021), "Nearest Neighbor Machine Translation".

kNN-MT demonstrates the use of task-model internal representations as keys
for an external datastore in a conditional generation task, again enabling
nearest-neighbour retrieval without retraining the base model.

The Pinyin-conditioned task model used in this project comes from:

Tan et al. (2022), "Exploring and Adapting Chinese GPT to Pinyin Input
Method", ACL 2022.

PinyinGPT provides the task-specific mapping from Chinese preceding context
and Pinyin to Chinese output. PinyinGPT itself is not claimed to provide the
same-user causal external-memory method proposed here.

### Adaptation made in this thesis

EM-2 adapts the general kNN-LM / kNN-MT principle to transparent
user-specific Pinyin input personalisation.

The datastore is not a generic training corpus.

For every user interaction:

key =
    Frozen PinyinGPT representation of
    preceding context + segmented Pinyin

value =
    historical Chinese target

The legal datastore is restricted to:

- the same user;
- strictly-prior interactions;
- H5000 history;
- exact same segmented Pinyin.

The project additionally evaluates retrieval specifically on History
Available, Ambiguous, and Conflict cases in order to distinguish simple
memorisation of a frequent personal target from context-sensitive
personalisation.

These user-specific causal restrictions and evaluation subsets are part of
the experimental design of this thesis and should not be attributed to
kNN-LM, kNN-MT, or PinyinGPT.

### Relationship to EM-3

EM-3 is conceptually related to neural Cross-Encoder reranking, including:

Nogueira and Cho (2019), "Passage Re-ranking with BERT".

Rather than relying only on distance between independently computed
representations, EM-3 will investigate a task-specific model that jointly
examines the current Pinyin-input situation and a historical personal
example and predicts their relevance.

The specific causal personal-history training construction used by EM-3 is
a thesis-specific adaptation rather than a claim that prior passage
reranking work studied Pinyin personalisation.

## References for methodological provenance

Khandelwal, U., Levy, O., Jurafsky, D., Zettlemoyer, L., and Lewis, M.
(2020). Generalization through Memorization: Nearest Neighbor Language
Models. ICLR 2020. arXiv:1911.00172.

Khandelwal, U., Fan, A., Jurafsky, D., Zettlemoyer, L., and Lewis, M.
(2021). Nearest Neighbor Machine Translation. ICLR 2021.
arXiv:2010.00710.

Tan, M., Dai, Y., Tang, D., Feng, Z., Huang, G., Jiang, J., Li, J.,
and Shi, S. (2022). Exploring and Adapting Chinese GPT to Pinyin Input
Method. ACL 2022, pp. 1899-1909.
doi:10.18653/v1/2022.acl-long.133.

Nogueira, R. and Cho, K. (2019). Passage Re-ranking with BERT.
arXiv:1901.04085.
