# Standardized workload and cache audit — 2026-08-20

Status: **pre-forward audit complete; pair-level counts pending frozen Stage-1 materialization**
Test used: **false**

## Train-Val surface

- Queries: 34,416 (Agent Phage 13,741; Etinjat 8,030; breaddddd 12,645).
- Exact-Pinyin causal history edges: 1,124,083 (510,576 / 138,684 /
  474,823 by author).
- Unique rows required by current queries plus their exact-Pinyin histories:
  42,454 (16,414 / 10,427 / 15,613 by author).
- Unique context keys: 42,358. Fewer context keys than rows reflects identical
  context strings, not row conflation.

## Frozen Generic

No compatible historical Generic predictions exist for the new Train-Val row
surface. Exactly 34,416 new predictions are required. The worker uses the
frozen checkpoint, production long-context semantics, beam 16, Top-10,
equal-shape GPU buckets, durable partial output, and deterministic final row
order. Completed rows are reused on resume.

## BGE / M1

- Read-only historical cache:
  `C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\cache\embedding_cache.sqlite3`.
- Cache rows: 251,234.
- Required unique standardized context keys: 42,358.
- Exact compatible hits: 39,993.
- True misses: 2,365.

Compatibility requires the pinned BGE SHA, recent-510-token preprocessing,
mean pooling, L2 normalization, and exact context. Historical cache files are
not modified; misses use a new versioned overlay/cache.

## PinyinGPT hidden representations

- Read-only historical cache:
  `C:\Users\chiar\Desktop\LBH\thesis-context-lab\results\personalisation\external_memory\em2_hidden_dev\hidden_states.sqlite3`.
- Historical cache rows: 11,475.
- Required standardized rows: 42,454.
- Exact compatible row-ID hits: 0.
- True misses: 42,454.

The representation remains the frozen PinyinGPT final-layer hidden state at
the final prompt `[SEP]`; only platform/runtime orchestration and cache
namespace differ.

## Generic Cross-Encoder and EM3

Exact generic-CE pair counts and historical exact-key hits are computed after
BGE and Hidden Stage-1 Top-20 surfaces are materialized. No estimate will be
silently reported as an exact workload.

Old generic-CE scores may be reused only for exact current/history/candidate,
model/tokenizer revision, template, truncation, max-length, and dtype keys.
Old EM3 scores are categorically incompatible with EM3-Clean3 because the new
checkpoint is trained from base on the frozen Train-Fit manifest; reuse count
is therefore zero.

Machine-readable record:
`results/personalisation/context_comparison_v2/workload_cache_audit_v1.json`.
