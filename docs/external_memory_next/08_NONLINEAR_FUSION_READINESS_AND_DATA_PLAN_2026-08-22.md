# 08 - Nonlinear Fusion Readiness and Data Plan

Date: 2026-08-22

Status: **DATA GENERATION REQUIRED BEFORE FITTING**

## 1. Evidence-driven gate

The durable Train-Val feature/support tables are sufficient for evaluation but
not for fitting a learned ranker. A repository-wide local-artifact search found
no Full+Short Train-Fit Generic Top10 prediction surface. Training on Train-Val
would violate the required Train-Fit/Train-Val boundary, so no learned model
may be fitted until this input exists.

## 2. Required new artifact

Generate the frozen PinyinGPT Generic Top10 for all 144,526 hash-pinned Clean3
Train-Fit rows using exactly the standardized Full+Short semantics:

- checkpoint revision `76dd20dc92d8236a350fb732e99dde6fa15e2263`;
- official code revision `8f1573ed0bd4d1f3d8d3f10a05f7e870725646f1`;
- beam size 16 and Top10;
- existing long-context `truncate_context_for_generation` policy;
- deterministic equal-prompt/equal-target-length buckets;
- resumable row-ID cache;
- no gold in generation and no Test acceptance.

The existing Train-Val run processed 34,416 rows at 29.37 rows/s on the same
RTX 4060 Laptop GPU, implying about 82 minutes for 144,526 Train-Fit rows.

## 3. Provenance of reused orchestration

The shape-safe resumable helper was inspected in the read-only
`thesis-context-compare` worktree at HEAD `fb09ca2`. It is an untracked source
referenced by that worktree's living indexes, not a committed blob. Its exact
SHA256 is `30aa98c1a41afdc9954bcbb3b1cbcfb18a634b222388c7119d6a3532683b84a0`.
The helper will be copied byte-for-byte into the isolated worktree and its
provenance retained; the new phase-specific runner will independently pin the
Train-Fit manifest hash and population.

Closeout removed one superfluous blank line at EOF before committing the
helper. The tracked, behavior-identical helper SHA256 is
`67bcba87f766ba3c39fa3b3ca69693ec580143dabc4448507c93c9f96a7518d2`;
the original run-time hash above remains the exact generated-artifact record.

The same frozen Full runner also imports the sibling worktree's untracked
`standardized_reranking.py`. It was copied byte-for-byte with SHA256
`8d01d57fc8666f8a8f23a3d24664e9ddfe5545732a18638231359fd571814028`.
The frozen Full-transfer source is checked after CRLF-to-LF normalization as
`f75d40f381e966f85cd4b20647ba7dc6a95df9116ad8657ca9a07505949a37b0`,
so Windows checkout line endings cannot masquerade as a semantic change.

## 4. Learned-fusion plan after generation

After Generic generation, construct causal Train-Fit query groups using only
strictly prior same-author H5000 raw history and exact segmented Pinyin. Runtime
features will exclude author identity and all correctness/post-hoc labels.

The first model comparison should use a small, interpretable tree ranker with
query grouping. Candidate groups whose gold is absent remain valid runtime
groups but contain no positive label; the exact training-library behavior for
zero-positive groups must be tested and documented before fitting. A linear or
logistic control on the same feature table is required so any nonlinear gain
is attributable to interactions rather than a new data surface.

## 5. Candidate library check

LightGBM 4.7.0 was installed into the isolated local-only directory
`.build/external_memory_next_deps/`; the shared virtual environment was not
modified. The official Python documentation defines query groups as contiguous
candidate counts summing to the number of samples and exposes the `lambdarank`
objective through `LGBMRanker`/the core training API:

- <https://lightgbm.readthedocs.io/en/stable/Python-Intro.html>
- <https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html>

A synthetic core-API check confirmed that version 4.7.0 accepts a query group
with no positive label. Such a group has no supervised ordering signal; the
planned policy is to report and exclude zero-positive groups from fitting while
retaining every query during Train-Val evaluation. This policy will be frozen
with the final feature-table audit.

The current frozen staged linear formula is the same-surface linear control.
Library objective, deterministic seed, exact runtime feature list, and the
small tree grid will be frozen only after the generated Train-Fit table is
audited. Dev3000 and Test remain excluded.
