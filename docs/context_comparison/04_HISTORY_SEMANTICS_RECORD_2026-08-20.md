# History semantics record — 2026-08-20

## Resolution

The authoritative/frozen Dev behavior is **rolling causal online memory**, and
it matches the standardized experiment intent. No history-semantic change is
required and no user choice is pending.

For query `q_t` from author `a`:

```text
Prior(a, q_t) = same-author interactions strictly earlier than q_t
H_t = latest min(5000, |Prior(a, q_t)|) raw interactions
H_t_same_pinyin = exact segmented-Pinyin matches inside H_t
```

The raw 5000 limit is applied before Pinyin filtering. History is query
specific; a GPU batch never owns or shares one history window.

## Repository evidence

- `src/personalisation/pilot_a.py` constructs Dev history from
  `HistoryIndex(history + dev, H5000)`.
- the reranking-matrix Dev tuning path also passes `history + dev`;
- historical Hidden-M1, Hidden-M2, and EM3 runners construct the same combined
  index;
- `em2_cache_hidden_dev.py` explicitly describes reuse of frozen
  `HistoryIndex(history + dev, H5000)`;
- historical `em3_dev_population_audit/summary.json` records:
  same-user true, strictly-prior true, budget-before-Pinyin-filter true,
  exact-segmented-Pinyin true, earlier-Dev-becomes-history true, and current
  Dev added only after evaluation.

This establishes both historical fidelity and agreement with the online causal
memory requirement.

## Train-Fit

Train-Fit rows are chronological supervised queries even when fewer than 5000
prior personal interactions exist. The visible window uses all strictly prior
rows until it reaches 5000; thereafter it is the latest raw 5000. A query is
pair-trainable only when causal exact-Pinyin history contains valid positive
and negative target evidence. No fake pair is created to fill a quota.

## Train-Val

Train-Val is a later complete-work suffix. Its history may contain prior
Train-Fit rows and strictly earlier Train-Val rows. It never contains the
current row, later Train-Val rows, Dev, or Test. Adding the current row to
memory happens only for subsequent queries.

## Dev3000 audit

The immutable Dev3000 contains 3,000 rows, 1,000 per Clean3 author, and has
SHA256 `9181f895eb19d0c36852e511263bfaefb34459dcd44efa6f45a44252e6b03f93`.
All 3,000 have at least 5,000 prior interactions and therefore belong to the
descriptive Mature-H5000 subset.

Preliminary reproducible comparison of an incorrect fixed-pre-Dev alternative
against the frozen rolling behavior found changed rows:

| Difference | Rows |
|---|---:|
| raw H5000 window | 3,000 |
| exact-Pinyin history membership | 2,845 |
| exact-Pinyin history count | 2,660 |
| unique Frequency winner | 811 |

Per-author changed membership/count/winner rows were Agent Phage
971/900/188, Etinjat 902/849/375, and breaddddd 972/911/248. The versioned
machine audit regenerates and freezes these counts from the authoritative
history and Dev manifests.

Fixed pre-Dev history would therefore materially change the experiment and is
not adopted.

Test used: **false**.
