# EM-3 Progress Note — 2026-08-19

## Status

EM-3 is in the design and data-audit stage.

No EM-3 model has been trained.
No new EM-3 Test evaluation has been performed.

The immediate goal is to construct and validate a strictly causal supervised training and development protocol for a task-specific Cross-Encoder.

---

## 1. Motivation

Earlier methods used personal history without task-specific Cross-Encoder training.

- M1 uses generic BGE embedding similarity.
- M2 uses a generic pretrained Cross-Encoder.
- EM-2 uses frozen PinyinGPT hidden-state kNN retrieval.

EM-2 improved history retrieval quality over the BGE retrieval baseline, but this improvement did not materially improve the final end-to-end Top-1 ranking.

This motivates EM-3:

> learn task-specific history relevance for the Pinyin personalisation task rather than relying only on generic semantic relevance.

EM-3 therefore focuses on the decision problem:

> Given the current context and Pinyin, which strictly-prior same-Pinyin historical interactions should be trusted as evidence for the current candidate?

---

## 2. Important Train / History Distinction

Previous personalisation stages did not train a new neural model.

Therefore, the large historical interaction pool was previously used mainly as external user memory rather than as a conventional supervised Train split.

EM-3 is the first stage in which these historical interactions are reused to construct supervised training queries.

The roles are chronology-dependent rather than file-dependent.

For one historical interaction used as a current Train query:

- only strictly earlier interactions from the same author may be used as history;
- the current interaction is not visible in its own history;
- future interactions are never visible.

Therefore:

> "training query" and "history interaction" are roles determined by chronology, not two disjoint source files.

This distinction must be preserved in the final method description.

---

## 3. Frozen History Semantics

EM-3 reuses the existing personalisation history semantics:

- same user / author;
- strictly-prior history only;
- H5000 budget;
- H5000 applied before Pinyin filtering;
- exact same segmented Pinyin;
- current interaction is inserted into memory only after it has been processed as a query.

These semantics should reuse the existing HistoryIndex / equivalent implementation wherever possible.

---

## 4. Current Data Sources

Full+Short historical interaction manifest:

`results/personalisation/reranking_matrix/manifests/history_full_short.jsonl`

Full+Short Dev manifest:

`results/personalisation/reranking_matrix/manifests/dev_full_short.jsonl`

Frozen T1 Test condition manifest:

`results/evaluation/deep_author_v2/design/t1_condition_manifest.jsonl`

### Six-author population

Historical interactions: 248,082

Dev queries: 32,212

Full+Short Test queries: 6,000

### Three-author exploratory population

Authors:

- Etinjat
- Re_spectators
- breaddddd

Historical interactions: 120,272

Dev queries: 13,895

Full+Short Test queries: 3,000

The three-author population is currently preferred for the first EM-3 experiment because it aligns with the later Context / EM-2 exploratory comparison.

---

## 5. Train Population Audit

A read-only EM-3 Train population audit was completed using the Full+Short historical interaction pool.

Source SHA-256:

`6d32d44189c0824d7973a5a9a50359dce3fb8111f6f7a9078580eb69fac58597`

Generated local-only audit:

`results/personalisation/external_memory/em3_train_population_audit/`

### Three-author results

| Population | Count |
|---|---:|
| All historical rows | 120,272 |
| Same-Pinyin history available | 89,375 |
| Positive history available | 84,568 |
| Hard-negative history available | 35,775 |
| Positive + hard negative available | 30,968 |
| Ambiguous | 32,861 |
| Conflict | 6,178 |

Potential interaction pairs before sampling:

- Positive interaction pairs: 3,424,295
- Negative interaction pairs: 355,856

### Six-author results

| Population | Count |
|---|---:|
| All historical rows | 248,082 |
| Same-Pinyin history available | 185,530 |
| Positive history available | 177,063 |
| Hard-negative history available | 71,485 |
| Positive + hard negative available | 63,018 |
| Ambiguous | 65,994 |
| Conflict | 11,953 |

Potential interaction pairs before sampling:

- Positive interaction pairs: 8,160,575
- Negative interaction pairs: 741,245

The full Cartesian history-pair population will not be used directly because repeated frequent targets would dominate training.

---

## 6. EM-3 Label Definition

For a current training query Q:

### Positive history

A legal strictly-prior same-Pinyin history interaction H is positive when:

`H.target == Q.gold`

### Hard-negative history

A legal strictly-prior same-Pinyin history interaction H is negative when:

`H.target != Q.gold`

These negatives are naturally hard because they:

- belong to the same user;
- have the same segmented Pinyin;
- were genuinely observed in the user's history;
- differ in target;
- often require context to distinguish.

The same label construction rule may be used for Dev evaluation, but Dev labels must never be used for gradient training.

Test labels are reserved for final frozen evaluation only.

---

## 7. Proposed Cross-Encoder Relation to M1 / M2

M1:

- generic BGE bi-encoder;
- independent context embeddings;
- cosine similarity;
- no IME-specific training.

M2:

- generic pretrained Cross-Encoder;
- joint query-history encoding;
- no IME-specific training.

EM-3:

- task-specific Cross-Encoder;
- trained using causal same-Pinyin positive / hard-negative supervision from this dataset.

A preferred initialisation is the same pretrained reranker family used by M2 where practical, because this gives a cleaner comparison between:

> generic pretrained relevance

and

> task-specific fine-tuned relevance.

---

## 8. Sampling Discussion — Provisional, Not Frozen

The complete history-pair population is too large and strongly imbalanced.

Current intended structure:

For each eligible current query:

- sample up to P positive historical interactions;
- sample up to N hard-negative historical interactions;
- every selected `(current query, history interaction)` is one independent Cross-Encoder training pair;
- negatives are selected once for the current query rather than duplicated separately for every positive;
- unused negatives are not currently recycled into additional groups.

This first version is intentionally simple because the current eligible Train population is already large.

Possible later extension:

- additional groups using previously unused negative histories;
- rotating negatives across epochs;
- hard-negative mining.

These are deferred until the basic EM-3 approach has been validated.

The exact values of P and N are not frozen yet.

Possible Dev-selected ranges currently under discussion include:

- P in {1, 3, 5}
- N in {1, 2, 3}

This must not be reported as a final choice until Dev selection is complete.

---

## 9. Current Next Step

Before constructing training pairs or fine-tuning a model:

1. audit the complete Dev population;
2. use the same causal history semantics;
3. allow each Dev query to see the historical pool plus strictly earlier Dev interactions;
4. count History Available, Positive Available, Hard Negative, Pair-Trainable, Ambiguous, and Conflict subsets;
5. verify that Dev remains separate from supervised Train;
6. then freeze the EM-3A v1 data protocol.

No Test prediction should be inspected during this process.

---

## 10. Dev Population Audit

The complete Full+Short Dev population was audited using the same causal history semantics as the Train population.

For each Dev query:

- the earlier historical interaction pool is available;
- only the latest H5000 strictly-prior same-user interactions are retained before Pinyin filtering;
- earlier Dev interactions become history only after they have been evaluated;
- the current and future Dev interactions are never visible.

Dev source SHA-256:

`a62cb7bcc25c3c6938e5ab1d9b789a83bf0a2c506ee1765dfe82ab043d800235`

Generated local-only audit:

`results/personalisation/external_memory/em3_dev_population_audit/`

### Three-author Dev population

| Population | Count |
|---|---:|
| All Dev queries | 13,895 |
| Same-Pinyin history available | 10,137 |
| Positive history available | 9,447 |
| Hard-negative history available | 4,266 |
| Positive + hard negative available | 3,576 |
| Ambiguous | 3,892 |
| Conflict | 840 |

Potential unsampled history pairs:

- Positive interaction pairs: 366,788
- Negative interaction pairs: 43,379

### Six-author Dev population

| Population | Count |
|---|---:|
| All Dev queries | 32,212 |
| Same-Pinyin history available | 24,025 |
| Positive history available | 22,728 |
| Hard-negative history available | 10,376 |
| Positive + hard negative available | 9,079 |
| Ambiguous | 9,587 |
| Conflict | 1,692 |

Potential unsampled history pairs:

- Positive interaction pairs: 1,074,910
- Negative interaction pairs: 110,424

### Proposed Dev evaluation semantics

The primary EM-3 history-discrimination population is the pair-trainable subset:

- at least one legal Gold-supporting history;
- at least one legal same-Pinyin competing-target history.

For the three-author population this contains 3,576 Dev queries.

The intended primary metric is Macro-author R@1 on this population.

Secondary diagnostics should include R@3, R@5, per-author results, and Conflict results conditional on Gold being present in legal history.

This preserves comparability with the earlier retrieval diagnostics, where retrieval quality was evaluated only when Gold-supporting history was available.

The full Dev population remains the source population for later end-to-end evaluation.

The existing EM-2 Dev tuning workload should be inspected for possible reuse before a new EM-3 tuning subset is created.

---

## 11. Reuse of the Existing Dev Tune Partition

The existing EM-2 Dev tune workload was verified against the Full+Short Dev manifest.

It corresponds exactly to rows with:

`pilot_partition == "tune"`

within the three-author exploratory population.

Per-author counts:

- Etinjat: 3,047
- breaddddd: 2,362
- Re_spectators: 199

Total:

- 5,608 Dev tune queries

This confirms that the EM-2 tuning workload was inherited from an earlier fixed Dev partition rather than selected according to later EM-2 model outcomes.

Decision:

> EM-3 will reuse the same 5,608-query Dev tune partition for checkpoint and hyperparameter selection where practical.

The complete 13,895-query three-author Dev population remains available for broader Dev confirmation after tuning.

No new EM-3-specific Dev subset will be created unless a later methodological need is identified and documented.

---

## 13. Revised EM-3A v1 Sampling Decision

The earlier provisional P=3, N=3 sampling proposal is superseded.

The current EM-3A v1 sampling design is:

- up to 3 sampling rounds per eligible Train query;
- each round uses 1 unused positive history;
- each round uses up to 3 unused hard-negative history interactions;
- the same history interaction should not be reused across rounds when unused alternatives exist;
- negative targets may repeat across rounds because different historical contexts from the same competing target may still provide different evidence;
- if fewer than 3 unused negatives remain, the round uses the available negatives only;
- if fewer than 3 positive histories exist, fewer than 3 rounds are created;
- no more than 3 rounds are created even when additional positive histories remain.

Thus, one query contributes at most:

- 3 positive pairs;
- 9 negative pairs;
- 12 total Cross-Encoder pairs.

This keeps each sampling round approximately 1:3 positive-to-negative while allowing multiple positive and negative contexts to be observed.

The design is intentionally limited to three rounds because the existing eligible Train population is already large.


---

## 14. Final EM-3A v1 Sampling Audit

The revised three-round 1:3 sampling design was audited on the three-author eligible Train population.

Results:

| Item | Count |
|---|---:|
| Eligible Train queries | 30,968 |
| Positive pairs | 86,959 |
| Negative pairs | 146,195 |
| Total training pairs | 233,154 |

Round distribution:

- 1 round: 2,227 queries
- 2 rounds: 1,491 queries
- 3 rounds: 27,250 queries

Per author:

- Etinjat: 91,556 total pairs
- Re_spectators: 13,830 total pairs
- breaddddd: 127,768 total pairs

The realised aggregate positive-to-negative ratio is approximately 1:1.68 rather than 1:3 because later rounds may have fewer than three unused negative histories available.

### Frozen EM-3A v1 sampling semantics

For each eligible Train query:

- create at most 3 rounds;
- each round selects 1 unused positive history;
- each round selects up to 3 unused hard-negative history interactions;
- a concrete history interaction is not reused across rounds;
- negative targets may repeat across rounds when represented by different historical contexts;
- insufficient negatives are not duplicated merely to fill the 1:3 maximum;
- unused histories beyond the three rounds are not used in EM-3A v1;
- selection is deterministic and reproducible.

This produces 233,154 supervised Cross-Encoder training pairs.

This sampling protocol is now frozen for EM-3A v1 unless a correctness problem is discovered.
