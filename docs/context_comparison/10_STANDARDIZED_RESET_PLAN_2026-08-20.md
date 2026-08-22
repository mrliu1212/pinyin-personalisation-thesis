# Standardized context-model comparison reset plan — 2026-08-20

## Status

Status: **Train split and pre-tuning freeze in progress**  
Protocol: `standardized-context-comparison-v1`  
Branch: `work/context-model-comparison`  
Base/HEAD at start: `80b053764e70ee2f2886892ba516a6b9e2470e59`  
Test used: **false**

This version supersedes the earlier Dev-preparation-only comparison plan. The
historical artifacts and the previously frozen balanced Dev3000 remain intact,
but historical hyperparameters are not carried forward as standardized choices.

## Scientific pipeline

```text
authoritative Clean3 Train
  -> deterministic chronological whole-work Train-Fit / Train-Val
  -> Train-Val-only system retuning and EM3 checkpoint selection
  -> PRE_DEV_FREEZE
  -> one sealed balanced Dev3000 evaluation
  -> final method freeze
  -> STOP before Test
```

The authors are Agent Phage, Etinjat, and breaddddd. Macro-author Top1 is the
primary selection and evaluation metric. Micro Top1, per-author Top1, Top3,
Top10, exact MRR@10, Missing@10, Ambiguous, formal Conflict, Mature-H5000,
behavior transitions, the complete G/F/C eight-way table, rescue/harm/net, and
the failure funnel are required secondary outputs.

## Frozen boundaries

- Generic remains the frozen production-compatible PinyinGPT2-Concat route at
  revision `76dd20dc92d8236a350fb732e99dde6fa15e2263`, beam 16, Top-10.
- M1 BGE, M2 generic Cross-Encoder, Hidden-M1 PinyinGPT representation, and
  Hidden-M2 generic Cross-Encoder neural weights remain frozen.
- EM3-Clean3 is the only newly trained neural model. It initializes from the
  pinned `BAAI/bge-reranker-base`, not from the historical EM3 checkpoint.
- All personal methods use the same query-specific causal history boundary:
  same author, strictly prior, latest up to 5000 raw interactions, then exact
  segmented-Pinyin filtering.
- Frozen Dev3000 path:
  `results/personalisation/context_comparison_v1/clean3_history_balanced_3000.jsonl`.
- Frozen Dev3000 SHA256:
  `9181f895eb19d0c36852e511263bfaefb34459dcd44efa6f45a44252e6b03f93`.
- Dev3000 is not historically virgin. It is sealed from this standardized
  reset onward and cannot be used for standardized selection.
- Test remains closed and is not an input to preparation, tuning, training,
  history construction, or Dev evaluation.

## Execution gates

1. Verify worktree, environment, source hashes, and historical read-only state.
2. Reproduce the exact historical 5,608-row evaluator eight-way counts.
3. Resolve and record history semantics.
4. Freeze deterministic whole-work Train-Fit/Train-Val manifests and hashes.
5. Freeze model identities, bounded search spaces, tie breaks, and cache rules
   before observing new Train-Val results.
6. Audit cache coverage and compute the minimum new forward workload.
7. Generate causal EM3 Train-Fit pairs and train from the pinned base recipe.
8. Retune every allowed system setting on Train-Val only.
9. Write and validate human- and machine-readable PRE_DEV_FREEZE artifacts.
10. Evaluate every frozen system once on the unchanged Dev3000.
11. Document results and stop before Test.

Generated JSONL, SQLite, caches, model checkpoints, and logs remain local-only.
No commit, push, or tag is authorized.

