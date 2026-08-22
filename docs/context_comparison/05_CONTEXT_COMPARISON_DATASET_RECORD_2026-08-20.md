# Context comparison dataset record - 2026-08-20

## Frozen dataset

Status: **FROZEN DEV MANIFEST / TEST NOT USED**.

Local-only manifest:
`results/personalisation/context_comparison_v1/clean3_history_balanced_3000.jsonl`

SHA256:
`9181f895eb19d0c36852e511263bfaefb34459dcd44efa6f45a44252e6b03f93`

The parent is the canonical Full+Short Dev manifest
`reranking_matrix/manifests/dev_full_short.jsonl`, 32,212 rows, SHA256
`a62cb7bcc25c3c6938e5ab1d9b789a83bf0a2c506ee1765dfe82ab043d800235`.
The EM3 population audit SHA256 is
`0c79db7a7f6fad2bee30b2cae82b1327f022ed4beeb53aa56af8055eea604059`.

## Verified Clean3 accounting

| Population | N | History available | Ambiguous | Conflict | Pair-trainable |
|---|---:|---:|---:|---:|---:|
| Clean3 Dev parent | 22,723 | 16,794 (73.91%) | 6,570 (28.91%) | 1,235 (5.44%) | 6,180 (27.20%) |
| History-eligible pool | 16,794 | 16,794 (100%) | 6,570 (39.12%) | 1,235 (7.35%) | 6,180 (36.80%) |
| Balanced frozen sample | 3,000 | 3,000 (100%) | 1,330 (44.33%) | 310 (10.33%) | 1,204 (40.13%) |

Parent rows are Agent Phage 10,110, Etinjat 4,045, and breaddddd 8,568.
History-eligible pools are 7,702, 2,409, and 6,683 respectively. The sample is
exactly 1,000 per author. It contains 1,568 tune-partition and 1,432
evaluation-partition Dev rows.

Same-Pinyin history counts in the sample are min 1, mean 40.547, median 11,
p90 96, max 349. In the complete eligible pool they are mean 43.993, median
11, p90 109, max 351. The balanced author weighting raises ambiguity/conflict
shares relative to the unbalanced eligible pool; this is a documented property
of the pre-registered per-author balance, not a score-driven adjustment. No
post-hoc resampling was performed.

The existing Clean3 Train audit was also verified: 178,942 rows, 44,788
pair-trainable, 46,961 ambiguous, and 8,596 conflict (sums of Agent Phage,
Etinjat, and breaddddd in `em3_train_population_audit/summary.json`). No
training was run.

## Identity bridge

Canonical and Pilot chronological positions use different namespaces, so raw
`row_id` and chronology were not joined directly. The frozen bridge key is:

```text
author + work_id + source_position_start + source_position_end
+ anchor_id + condition_id + pinyin_segments + gold
```

Across both 32,212-row Dev manifests: matched 32,212; missing 0; duplicate
strong keys 0; duplicate matches 0; author mismatches 0; Pinyin mismatches 0;
Gold mismatches 0. Historical IDs were not mutated.

## Sampling method and command

Within each author's history-available pool, rows were ordered by
`SHA256("context-comparison-v1|20260820|" + canonical_row_id)`, then canonical
ID, and the first 1,000 were selected. Cache availability and model outputs
were not read until after the manifest was written and hashed.

```powershell
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
& $python -m experiments.context_comparison.prepare_context_comparison `
  --personalisation-root 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation' `
  --external-root 'C:\Users\chiar\Desktop\LBH\thesis-context-lab\results\personalisation\external_memory' `
  --output-root 'results\personalisation\context_comparison_v1'
```

Reruns byte-validate frozen JSON/JSONL and refuse to overwrite differences.
The sample is the primary fast Dev comparison surface; full Clean3 Dev remains
the canonical parent/possible secondary surface. Test was not opened.
