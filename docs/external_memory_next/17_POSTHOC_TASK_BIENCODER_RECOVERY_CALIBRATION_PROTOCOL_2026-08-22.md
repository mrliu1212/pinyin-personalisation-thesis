# 17 - Post-hoc Task-BiEncoder Recovery and Calibration Protocol

Date: 2026-08-22

Status: **FROZEN BEFORE NEW GRID EVALUATION**

## 1. Questions and boundary

This controlled post-hoc diagnostic asks whether the completed task-specific
bi-encoder needs different downstream calibration, benefits from the existing
frozen recency mechanism, or is more useful for Personal-K5 recovery scoring
than for fixed-surface final reranking. It also compares the resulting
accuracy/latency trade-off with the historical Q8/Q8+F scorer.

The task encoder is not retrained. Generic generation, Personal-K5 membership,
history construction, candidate merge/deduplication, and evaluation semantics
remain frozen. Dev3000 and Test are rejected by every new runner.

Initial-Pinyin and Full-Pinyin are separate tracks. They have separate inputs,
hashes, candidate surfaces, grids, selections, predictions, and claims.

## 2. Shared causal and support semantics

Every runtime history lookup is:

```text
same author
-> strictly prior interactions
-> latest H5000 raw interactions before Pinyin filtering
-> exact track-specific segmented-Pinyin match
```

No author ID, Gold, correctness, or future row enters a runtime score. Empty
Generic surfaces retain the frozen no-op behavior.

For both Generic BGE and Task-BiEncoder, context is the last 64 Unicode code
points. Candidate-conditioned histories are ranked by cosine similarity alone;
the deterministic tie-break is chronological position then row ID. The first
five histories are aggregated as either:

```text
plain:   sum(max(0, cosine))
recency: sum(max(0, cosine) * exp(-age / 2048))
```

Candidate supports are normalized over the current candidate set. The only
difference between Generic and Task support is the encoder checkpoint.

## 3. Initial-Pinyin track

Population: the canonical 34,416-row Initial Train-Val manifest. Personal-K5
is the frozen, backend-compatible, frequency-ordered personal-only pool. The
three frozen Stage-1 recovery surfaces remain separate:

- `K5+Entropy`;
- `4P+4CS+2E` (primary balanced recovery);
- `6P+2CS+.25E` (front-rank recovery).

The Stage-1 interpolated personal `P_NG` and Stage-2 hard-backoff
NGramRecency are related lexical evidence, not independent evidence families.
They retain their different frozen definitions and locations in the pipeline.

Fixed-surface context methods for every Stage-1 surface:

1. Stage-1 only;
2. NGramRecency only;
3. Generic BGE plain only;
4. Generic BGE-Recency only;
5. Task-BiEncoder plain only;
6. Task-BiEncoder-Recency only;
7. NGramRecency + Generic BGE-Recency;
8. NGramRecency + Task-BiEncoder-Recency.

Frozen equal calibration grids:

```text
lambda_N = [0, .25, .5, 1, 2, 4, 6, 8, 12]
lambda_E = [0, .25, .5, 1, 2, 4, 6, 8, 12, 16]
```

Generic and Task encoders receive exactly the same single-evidence and joint
grid. No boundary extension is permitted after results are observed.

Personal-K5 candidate scoring uses the same frozen K5 membership and reports
Frequency, interpolated `P_NG`, Generic plain/recency, Task plain/recency,
joint `P_NG` + Generic-recency, joint `P_NG` + Task-recency, Q8, and Q8+F.
The joint encoder grids above are shared. The historical Q8+F convex grid is
preserved exactly:

```text
alpha_F = [0, .25, .5, .75, 1]
score = (1-alpha_F) * softmax(Q8 mean log probability)
        + alpha_F * normalized log-frequency
```

Candidate-only metrics use the historical candidate-only population and remain
separate from end-to-end metrics. The primary recovery denominator is the
frozen Generic-missing population whose Gold is in Personal-K5; all methods use
the same denominator. Generic-covered diagnostics are reported separately.

## 4. Full-Pinyin track

Population: the canonical 34,416-row Full Train-Val manifest and frozen
RetunedStage1 surface. Historical comparators are frozen RetunedFinal,
generic-BGE LambdaMART, and Task-BiEncoder fixed fusion.

Fixed-surface methods are the same eight methods listed for Initial. The equal
historical Full calibration grids are:

```text
lambda_N = [0, 2, 4, 6, 8]
lambda_E = [0, 2, 4, 6, 8]
```

The Full Personal-K5 recovery-scoring diagnostic may run only if the audit can
reconstruct the existing Full merge and K5 semantics without importing Initial
formulas. Full Q8 may run only if the existing exact candidate scorer accepts
the frozen Full segmented Pinyin and candidate surface unchanged and its
preflight cost is reasonable. Otherwise the omission and exact cost/command
must be recorded; no new Full-Q8 protocol may be invented.

## 5. Selection and tie-breaking

Within each track and method family, select on Train-Val by:

1. higher Macro-author Top1;
2. higher MRR@10;
3. smaller total added evidence weight;
4. smaller encoder weight;
5. smaller NGram weight.

For candidate-only selection, use Macro-author Recovery@1, then Recovery MRR,
then the same smaller-weight rules. Original candidate order followed by text
is the deterministic score tie-break. Initial and Full selections never cross.

## 6. Comparability gates

Before interpreting a new grid:

- verify all frozen input hashes and exact row order;
- reproduce Initial Stage-1 and completed Generic-BGE Stage-2 operating points;
- reproduce the historical Q8 candidate-only result from its durable scores;
- reproduce Full RetunedFinal and the completed Task fixed-fusion result;
- require candidate-set equality for every fixed-surface rerank;
- require Missing@10 invariance for every pure rerank;
- audit chronology and H5000-before-Pinyin behavior;
- require `used_dev3000=false` and `used_test=false`.

Failure of a comparator reconstruction stops that track before new tuning.

## 7. Primary outputs

The primary end-to-end metric is Macro-author Top1. Also report Micro Top1,
Top3, Top5, MRR@10, Missing@10, mean present-Gold rank, per-author Top1,
Ambiguous, Conflict, recoverable-only, Generic-missing, Generic-covered, and
paired rescue/harm/net against one frozen comparator per track.

Recovery reports Recovery@1/3/5/10, MRR, mean rank, per-author Recovery@5,
Generic-missing recoverable, and Generic-covered harm diagnostics. Intrinsic
Generic-vs-Task retrieval remains the completed same-history comparison from
record 16.

Latency reports distinguish offline embedding construction from warm online
query embedding/retrieval/ranking. Q8 online score-call latency is never
compared as if it included the same work as a precomputed embedding lookup.

## 8. Post-hoc interpretation

This is explicitly a post-hoc calibration diagnostic. No significance claim
will be made without a valid paired analysis. A negative downstream result is
retained if the task representation improves retrieval but not candidate
support, recovery, or final ranking.

```text
used_dev3000 = false
used_test = false
```
