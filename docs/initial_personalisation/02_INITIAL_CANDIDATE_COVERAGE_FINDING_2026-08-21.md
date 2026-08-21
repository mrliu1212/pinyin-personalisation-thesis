# Initial candidate-coverage finding — standardized Train-Val

Status: **A0/A1/A2 complete on standardized Initial+Short Train-Val; paired standardized Full-vs-Initial audit pending.**

## Research question

Why does Initial-only Pinyin perform so much worse than Full Pinyin, and is part of the gap caused before reranking because the Generic model fails to keep the Gold target in its Top-10 candidate surface?

## Standardized Initial result

Population: 34,416 Clean3 Train-Val interactions, deterministically transformed from Full Pinyin to Initial Pinyin. No resampling. Dev3000 and Test were not used.

Frozen Generic backend: PinyinGPT2-Concat, beam 16, Top-10, production-compatible context semantics.

A1 Generic predictions SHA256:

`bd0fb4dc304e0b266b90fae6fe3ac65424d2f52b23fedfa881212706ba2c2873`

A2 headline results:

| Metric | Initial+Short standardized Train-Val |
|---|---:|
| Rows | 34,416 |
| Generic Top1 | 33.0573% |
| Generic Missing@10 | 36.5092% |
| Generic Recall@10 | 63.4908% |
| Raw recoverable among Generic-missing | 47.7199% (5,996 / 12,565) |
| Backend-compatible recoverable among Generic-missing | 47.0275% (5,909 / 12,565) |
| Compatible recoverable@1 among Generic-missing | 21.1062% (2,652 / 12,565) |
| Compatible recoverable@3 among Generic-missing | 33.6570% (4,229 / 12,565) |
| Compatible recoverable@5 among Generic-missing | 39.0768% (4,910 / 12,565) |

Only 87 raw-recoverable Generic-missing cases were removed by backend compatibility filtering, so backend incompatibility is not the main bottleneck in this Initial recovery opportunity.

## Interpretation

The Initial condition appears to suffer from a substantial **candidate-generation / candidate-coverage problem**, not only a reranking problem. For 36.51% of standardized Initial Train-Val queries, the Gold target is absent from the frozen Generic Top-10, so no reranker restricted to that Top-10 can recover those cases.

This suggests the following decomposition:

\[
\text{Final error}
=
\text{candidate-coverage failure}
+
\text{within-surface ranking failure}.
\]

Initial abbreviation discards most of the phonetic information in each syllable. A plausible mechanism is therefore:

\[
\text{Full Pinyin}
\rightarrow
\text{stronger phonetic constraint}
\rightarrow
\text{smaller candidate ambiguity}
\rightarrow
\text{higher Generic Top-10 coverage},
\]

whereas

\[
\text{Initial Pinyin}
\rightarrow
\text{phonetic information loss}
\rightarrow
\text{larger compatible candidate space}
\rightarrow
\text{more Gold targets lost from Generic Top-10}.
\]

The A2 result adds a second important observation: **47.03% of Initial Generic-missing targets are nevertheless present in the same user's legal H5000 history and are compatible with the frozen PinyinGPT backend.** Long-term personal history can therefore restore part of the information lost by abbreviation at the candidate-surface stage.

## Historical Full comparison — supporting evidence only

Historical frozen Full+Short Test had Generic Missing@10 of approximately 8.97%, compared with 37.50% for historical Initial+Short. The new standardized Initial Train-Val Missing@10 of 36.51% replicates the qualitative Initial coverage-collapse pattern.

However, historical Full Test and new standardized Initial Train-Val are different populations, so the historical comparison is **not** the final apples-to-apples evidence.

## Required paired standardized audit

Run Full and Initial on the **same 34,416 Train-Val anchors** and compare:

- Generic Top1 / Top3 / MRR@10 / Missing@10;
- Full vs Initial Missing@10 delta and ratio;
- exact paired transition `Full covered -> Initial missing`;
- `Full missing -> Initial covered` sanity cases;
- how many `Full covered -> Initial missing` targets are recoverable from legal Initial H5000 personal history;
- Full vs Initial compatible recoverability@1/@3/@5.

The strongest explanatory diagnostic is:

\[
P(\text{Initial missing} \land \text{Full covered}),
\]

because these are the exact same anchors for which the Full input kept Gold in the candidate surface but the deterministic Initial abbreviation lost it.

If this paired audit shows a large Full-to-Initial candidate-loss transition, it supports the thesis claim that **a substantial part of Initial-input degradation arises before personalised reranking, through candidate-coverage collapse caused by abbreviation**. It should still not be described as the only cause of the Top1 gap.

## Research boundary

This is Train-Val analysis only. It does not use standardized Dev3000 or Test. The finding may motivate candidate-recovery method development on Train-Val, but current method choices must be frozen before standardized Dev3000 evaluation.
