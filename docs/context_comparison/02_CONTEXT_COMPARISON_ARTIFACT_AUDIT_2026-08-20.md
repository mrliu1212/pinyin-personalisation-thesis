# Context comparison artifact audit - 2026-08-20

## Status, question, and boundary

Status: **DEV ARTIFACT/CACHE AUDIT COMPLETE; NO INFERENCE**.

The audit asks which existing Original-M1, Original-M2, Hidden-M1,
Hidden-M2, and EM3 artifacts can support a controlled Full+Short/H5000
comparison on the frozen Clean3 balanced-3000 Dev manifest. External artifacts
under `C:\Users\chiar\Desktop\LBH\thesis-context-lab\results` and
`C:\Users\chiar\Desktop\LBH\thesis-personalisation\results` were read-only.
No Test row, model inference, GPU job, training, or score-driven sampling was
used.

## Frozen model registry

The machine-readable registry is local-only at
`results/personalisation/context_comparison_v1/model_registry.json`.

| ID | Exact route | Frozen K/Top-N, lambda | Existing end-to-end output |
|---|---|---|---|
| `original_m1` | full-context BGE cosine retrieval; positive-cosine target support plus normalized Generic | Top-N 5, lambda 4 | compatible legacy 5,608 ranks; other rows require CPU reconstruction when vectors exist |
| `original_m2` | Original-M1 BGE Stage-1; pretrained `BAAI/bge-reranker-base`; M2 target support plus normalized Generic | K=20, lambda 4 | compatible legacy 5,608 ranks; other rows require complete BGE and pair caches |
| `hidden_m1` | frozen PinyinGPT final-layer hidden state at final prompt `[SEP]`; cosine target support plus normalized Generic | Top-N 3, lambda 4 | compatible legacy 5,608 full rankings |
| `hidden_m2` | Hidden-M1 Stage-1; pretrained `BAAI/bge-reranker-base`; M2 support plus normalized Generic | K=10, lambda 4 | no row-level final file; legacy pairs are cache-reconstructable |
| `em3` | Hidden-M1 Stage-1; EM3-BCE v1 task-fine-tuned history-utility Cross-Encoder; M2-style support plus normalized Generic | K=10, lambda 4 | compatible legacy 5,608 rank-only rows |

All five use the frozen Generic Top-10 candidate surface, Full+Short, same-user
strictly-prior history, and H5000 applied before exact segmented-Pinyin
filtering. Frequency is not fused into any of the five Context routes. G and F
remain separate shared baselines.

The pretrained M2 scorer is
`BAAI/bge-reranker-base@2cfc18c9415c912f9d8155881c133215df768a70`,
model SHA256 `ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd`,
tokenizer SHA256 `9eb652ac4e40cc093272bbbe0f55d521cf67570060227109b5cdc20945a4489e`,
paired max length 512. EM3 uses the local final checkpoint under
`em3_cross_encoder_v1/train/final`; its config SHA256 is
`b654d6598b95be4656a4eefd389695542aa4bf30c7eb24378f2e9da8abcfcaa5`.
The 1.11 GB model tensor was not redundantly hashed during this audit; the
training config/result hashes are `59d5f134...` and `ee722d6f...` and the
checkpoint path is recorded exactly.

## Source inventory and reuse decision

| Artifact | Verified content/provenance | Decision |
|---|---|---|
| `.../reranking_matrix/cache/dev_generic/full_short.jsonl` | 32,212 Full+Short Dev rows; frozen Top-10; SHA256 `cb7e304d1b97cb4ddec492aabb5299c2e1be223c1786dfad098d0ef1339590f9` | compatible shared candidate surface |
| `.../pilot_a_context_memory/cache/embedding_cache.sqlite3` | 251,234 BGE vectors; 512 dimensions; mean pooling, L2; preprocessing `bge-gguf-left-truncate-most-recent-510-v1`; model SHA above | compatible only on exact cache-key hits |
| `.../em2_hidden_dev/hidden_states.sqlite3` | 11,475 rows, 768 dimensions, old Etinjat/Re_spectators/breaddddd tune surface plus required history; SHA256 `9a80a3314c184ccf3f0540916203c651474fad162dc3dab1fc97f7451f441df1` | compatible exact-row hidden states; no Agent Phage query states |
| `.../em2_four_way_dev_compare/rows.jsonl` | 5,608 old-surface final G/F/Original-M1/Hidden-M1 ranks; SHA256 `7bc20cddc5a772e7c1f9fb3fdd60ec17e8c2813667b7c32ec835b4cbc15d87d7` | direct rank reuse after identity bridge |
| `.../em2_hidden_m1_dev/selected_rows.jsonl` | 5,608 final Hidden-M1 rankings; SHA256 `3540dfebb532a54922450f02308d813975a6140e83e9c48ecac31c44cf197631` | direct final reuse |
| `.../em2_original_m2_dev/rows.jsonl` | 5,608 final Original-M2 ranks; SHA256 `502b86e93e30cd2ffba25b32dbcce23e21dc28ca2a4a28fe44aecaaf6b700ae5` | direct rank reuse |
| `.../em2_original_m2_dev/cache/pair_scores.sqlite3` | 836,399 exact pair logits; frozen pretrained scorer metadata | cache-only reuse by exact identity key |
| `.../em2_hidden_m2_dev/cache/pair_scores.sqlite3` | 39,415 unique pairs; completed selected legacy K=10/K=20 grid | reconstructable on compatible legacy hidden rows; not a final-prediction artifact |
| `.../em3_hidden_dev/selected_rows.jsonl` | 5,608 final EM3 candidate ranks; SHA256 `2078ed9bbf41879a4c81ddd06576e30dce402d89e48fc877dc787408d53a11a2` | direct rank reuse |
| `.../em3_bce_v1_final_dev_tune/scores.jsonl` | 96,950 pair scores on 1,609 pair-trainable legacy queries | history-discrimination evidence only; not candidate Top1 |
| `.../em3_bce_v1_final_compare/*` | mixed diagnostic report with history-discrimination fields | not a canonical source of end-to-end comparison predictions |

The inherited 5,608 population is Etinjat 3,047, breaddddd 2,362, and
Re_spectators 199. It is a legacy regression surface, not Clean3.

## Frozen balanced-3000 cache coverage

The manifest was frozen first (SHA256
`9181f895eb19d0c36852e511263bfaefb34459dcd44efa6f45a44252e6b03f93`).
Only then were exact immutable SQLite keys inspected.

| Model | Direct | Reconstructable | Partial | True inference | Unresolved | Usable now |
|---|---:|---:|---:|---:|---:|---:|
| Original-M1 | 996 | 573 | 955 | 476 | 0 | 52.30% |
| Original-M2 | 996 | 572 | 956 | 476 | 0 | 52.27% |
| Hidden-M1 | 996 | 0 | 566 | 1,438 | 0 | 33.20% |
| Hidden-M2 | 0 | 996 | 566 | 1,438 | 0 | 33.20% |
| EM3 | 996 | 0 | 566 | 1,438 | 0 | 33.20% |

Per-author states are preserved in `cache_coverage.json` and in the database.
Agent Phage has no direct hidden-route rows: Hidden-M1, Hidden-M2, and EM3 each
require 1,000 new query hidden states there. Etinjat contributes 737 direct
legacy rows and breaddddd 259. Partial means that some compatible intermediate
state exists but an end-to-end prediction cannot be produced without filling
true misses; it is never counted as a hit.

## Interpretation and non-conclusions

This audit proves provenance and computation requirements, not comparative
performance. It does not infer results from smoke tests or history
discrimination, and it does not authorize cache-biased row selection. The next
safe action is to validate the future evaluator on the independent legacy
5,608 regression surface, then request explicit approval before filling true
Dev cache misses. Test remains closed.
