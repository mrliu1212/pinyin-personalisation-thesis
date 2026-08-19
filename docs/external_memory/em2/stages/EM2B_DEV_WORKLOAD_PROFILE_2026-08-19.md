# EM-2B Dev Workload Profile

Status: FROZEN DESCRIPTIVE RECORD

Date: 2026-08-19

## Purpose

Record the structure of the legal Dev retrieval workload before inspecting
any EM-2 retrieval result.

These statistics describe the available personal-history surface only.

They are not evidence that hidden-state retrieval is effective.

## Frozen scope

Condition:
- Full+Short

History:
- H5000

Authors:
- Etinjat
- Re_spectators
- breaddddd

Partition:
- Dev tune only

History construction:
- same user;
- strictly prior;
- H5000 budget applied before Pinyin filtering;
- exact same segmented Pinyin.

Gold inspected:
- No

Retrieval metrics inspected:
- No

## Query population

Total three-author Dev tune queries:

- 5,608

Per author:

- Etinjat: 3,047
- Re_spectators: 199
- breaddddd: 2,362

The author populations are therefore substantially imbalanced.

This is one reason that Macro-author metrics remain important in addition
to Micro metrics.

## History availability

Queries with at least one legal same-Pinyin H5000 historical interaction:

- 3,625 / 5,608
- 64.64%

Queries with no legal same-Pinyin H5000 history:

- 1,983 / 5,608
- 35.36%

Therefore EM-2 retrieval has no historical evidence to retrieve for more
than one third of the Dev queries under the frozen exact-Pinyin H5000
memory definition.

This is a workload property, not a retrieval-model failure.

## Visible history depth

Number of legal same-Pinyin historical interactions visible per query:

- mean: 21.767
- median: 2
- maximum: 435

The large difference between the mean and median indicates a strongly
uneven history-depth distribution.

A typical query has only a small number of same-Pinyin historical examples,
while a smaller number of queries have very large historical sets.

This observation should be treated descriptively unless later analysis
explicitly studies the history-depth distribution.

## Query-history edge surface

Total legal query -> historical-example relationships:

- 122,067

Mean legal historical edges per query:

- 21.767

Per author:

- Etinjat:
  - queries: 3,047
  - legal history edges: 40,679
  - mean edges/query: 13.35

- Re_spectators:
  - queries: 199
  - legal history edges: 7,795
  - mean edges/query: 39.17

- breaddddd:
  - queries: 2,362
  - legal history edges: 73,593
  - mean edges/query: 31.16

The authors therefore differ not only in query count but also in the
amount of legal historical evidence available per query.

## Unique representation workload

Unique interaction rows required across current queries and all legal
historical examples:

- 11,475

Unique (context, segmented-Pinyin) representation inputs:

- 11,475

The equality of these two counts means that there are no additional
duplicate representation inputs to collapse within this workload.

Although there are 122,067 query-history edges, historical interactions
can be reused by multiple later queries.

Therefore only 11,475 Frozen PinyinGPT hidden representations need to be
computed and cached.

## Unique historical interactions used

Per author:

- Etinjat: 3,694
- Re_spectators: 1,366
- breaddddd: 3,759

These are unique historical interaction rows that participate in at least
one legal retrieval set.

## Interpretation boundary

The following observations are supported at this stage:

1. Exact-Pinyin personal history is available for a majority, but not all,
   Dev queries.

2. Available history depth is highly uneven: the median is only 2 while the
   maximum reaches 435.

3. Query populations and history density differ substantially across
   authors.

4. Many query-history relationships reuse the same historical interactions,
   making representation caching substantially cheaper than recomputing
   representations per retrieval edge.

The following claims are NOT supported yet:

- that PinyinGPT hidden states retrieve better than BGE;
- that more history improves accuracy;
- that large history sets are beneficial or harmful;
- that the uneven history distribution explains any downstream model result.

Those questions require the subsequent EM-2 retrieval evaluation.

## Next stage

EM-2B computes and caches the 11,475 frozen PinyinGPT representations.

EM-2C will then evaluate kNN retrieval over the 122,067 legal query-history
relationships.
