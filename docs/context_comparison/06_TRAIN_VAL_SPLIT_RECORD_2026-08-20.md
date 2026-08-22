# Standardized Train-Fit / Train-Val split record — 2026-08-20

Status: **frozen**
Test used: **false**

## Authoritative source

- Path: `C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\reranking_matrix\manifests\history_full_short.jsonl`
- SHA256: `6d32d44189c0824d7973a5a9a50359dce3fb8111f6f7a9078580eb69fac58597`
- Rows: 178,942
- Authors: Agent Phage 69,667; Etinjat 40,936; breaddddd 68,339
- Complete works: Agent Phage 45; Etinjat 63; breaddddd 45
- Duplicate row IDs: 0
- Test rows: 0

## Deterministic rule

Within each author, works are ordered by `work_chronological_index`, then
`work_creation_date`, then `work_id`. Train-Val is the latest contiguous
complete-work suffix whose row share is in 15–20% and is closest to 20%.
An exact tie prefers the larger validation share, then the earlier boundary.
No work is split and no interaction is shuffled.

## Frozen manifests

| Partition | Rows | Agent Phage | Etinjat | breaddddd | Works (A/E/B) | Pair-trainable | SHA256 |
|---|---:|---:|---:|---:|---:|---:|---|
| Train-Fit | 144,526 | 55,926 | 32,906 | 55,694 | 31 / 56 / 40 | 35,290 | `547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6` |
| Train-Val | 34,416 | 13,741 | 8,030 | 12,645 | 14 / 7 / 5 | 9,498 | `d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220` |

Train-Fit path:
`results/personalisation/context_comparison_v2/clean3_train_fit_v1.jsonl`.

Train-Val path:
`results/personalisation/context_comparison_v2/clean3_train_val_v1.jsonl`.

## Boundaries

| Author | Last Train-Fit work | First Train-Val work | Train-Val share |
|---|---|---|---:|
| Agent Phage | `da-work-857753354` | `da-work-985862865` | 19.7238% |
| Etinjat | `da-work-1133955432` | `da-work-1148395730` | 19.6160% |
| breaddddd | `da-work-1024189828` | `da-work-1024284927` | 18.5033% |

The complete work lists and per-work row counts are machine-recorded in
`train_val_split_audit_v1.json`.

## Population and causal-history audit

- Train-Fit Ambiguous: 36,908; formal Conflict: 6,731.
- Train-Val Ambiguous: 10,053; formal Conflict: 1,865.
- Train-Fit history-depth buckets: 0 = 3; 1–499 = 1,497;
  500–1,999 = 4,500; 2,000–4,999 = 9,000; >=5,000 = 129,526.
- Train-Val history depth is exactly 5,000 for all 34,416 rows.
- The total pair-trainable count is 44,788, matching the audited Clean3
  reference population; 35,290 belong to Train-Fit and 9,498 to Train-Val.

Assertions passed: row-ID disjointness, complete row coverage, whole-work
disjointness, chronological suffix ordering, same-author history, strict
priority, no self/future visibility, and H5000-before-Pinyin filtering.
