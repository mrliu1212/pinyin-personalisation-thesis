# LiveChat Generic PinyinGPT Baseline V1 Checkpoint

- Checkpoint recorded: `2026-08-13T20:11:53.1117281+08:00`
- Branch before checkpoint: `main`
- Commit before checkpoint: `d23e9a767f258750e1160ae28594e550286a1e49`
- Result directory: `results/experiments/livechat_pinyingpt_generic_baseline_v1/`
- Development benchmark: Frozen LiveChat Development Evaluation Set V1
- Chronology grade: `C`
- Split: deterministic response-level `non-temporal proxy split`, seed `40408`
- Selected users: 100
- Frozen interactions: 10,000, exactly 100 per selected user
- Personalisation implemented: no

## Frozen artifact hashes

| Artifact | SHA-256 |
|---|---|
| `configs/livechat_pinyingpt_generic_baseline_v1.json` | `aa5320be9c2e900e2da98a2f85c55b238e432c335795c04de021af108b0ce461` |
| `frozen_interactions.jsonl` | `305a2fe72054e448e8de0143a91c25fe9b9caa3c66ed129869aec270fd4dfcec` |
| `pinyin_only_predictions.jsonl` | `72b4344bc0fb1b978d62bf656cbff76273abc6fd2ba6240e3de22e98fc020571` |
| `contextual_predictions.jsonl` | `8e89c2a4ddb88ea2cb329e196651a72c99d99a0eb9e8314d0ac14883cbf555a7` |

The prediction and interaction artifacts remain in the result directory. Their hashes are the scientific identity of the accepted baseline even if a future repository policy excludes large artifacts from Git.

## Accepted baseline metrics

| Condition | Top-1 | Top-3 | Top-5 | Top-10 / Coverage@10 | MRR@10 | MeanRank\|Top10 | Missing@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pinyin-only, micro | 0.6006 | 0.8426 | 0.8876 | 0.9619 | 0.718229 | 2.061025 | 381 / 3.81% |
| Contextual Full-Pinyin, micro | 0.8819 | 0.9634 | 0.9770 | 0.9866 | 0.923883 | 1.212548 | 134 / 1.34% |

Primary contextual macro-user Top-1 is `0.8819`. No unrestricted rank or MRR is claimed beyond the returned Top-10.

## Frozen model and environment

- Checkpoint: `aihijo/transformers4ime-pinyingpt-concat`
- Checkpoint revision: `76dd20dc92d8236a350fb732e99dde6fa15e2263`
- Official code revision: `8f1573ed0bd4d1f3d8d3f10a05f7e870725646f1`
- Platform: `Windows-11-10.0.26200-SP0`
- Python: `3.12.13`
- PyTorch: `2.11.0+cu128`
- CUDA runtime: `12.8`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Transformers: `4.57.6`
- Jieba: `0.42.1`
- pypinyin: `0.55.0`

## Verification before checkpoint

- Targeted LiveChat/PinyinGPT tests: `28/28` passed.
- Python compilation: passed.
- Frozen interaction and prediction alignment: passed.
- Six result-manifest SHA-256 entries: verified.
- `git diff --check`: exit `0` (only Git's informational LF-to-CRLF warnings).
- External raw data, model checkpoints, build trees, virtualenv files, and caches were not staged.
