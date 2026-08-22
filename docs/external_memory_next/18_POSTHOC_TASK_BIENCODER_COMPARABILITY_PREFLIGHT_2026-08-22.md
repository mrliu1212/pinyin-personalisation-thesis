# 18 - Post-hoc Task-BiEncoder Comparability and Runtime Preflight

Date: 2026-08-22

Status: **COMPLETE / BOTH TRACKS AUTHORIZED BY COST GATE**

## 1. Hard comparator gate

The Full RetunedFinal arithmetic was reconstructed before new tuning:

```text
rows = 34,416
candidate/rank equality = exact
Macro-author Top1 = .7960049265502147
Micro Top1 = .8249941887494189
Top3 = .9120757787075778
Top5 = .9307589493258950
MRR@10 = .8713778728793548
Missing@10 = .05198163644816364
```

The durable Initial population and controls were also verified:

```text
Train-Val rows = 34,416
Generic-missing rows = 12,565
Gold in frozen Personal-K5 among Generic-missing = 4,910
Personal-K5 candidate pairs = 123,738 across 30,509 nonempty rows
Q8 candidate-only Macro Top1 on K2+ = .6365306668058097
Q8+F@.75 candidate-only Macro Top1 on K2+ = .6691641408627556
Q8 online mean = 32.45296358195237 ms/query
Q8 p95 = 54.540200042538345 ms/query
```

The Initial manifest, Stage-1, and BGERecency files have identical row-ID sets.
The historical Stage-1/support files are row-ID sorted rather than manifest
ordered. The new runner records this fact and restores canonical Train-Val
manifest order before writing any new artifact. Full inputs are already in
manifest order.

## 2. Initial-Pinyin preflight

```text
rows = 34,416
unique contexts required by the initial Stage-1-only preflight = 46,434
Generic cache reusable = 42,717
Generic vectors missing = 3,717
Task cache reusable = 42,129
Task vectors missing = 4,305
bounded Task timing sample = 256 contexts / 4.8537 s including model load
conservative Task estimate = 81.62 s
Generic estimate at historical 2 ms/context = 7.43 s
total incremental vector estimate = 89.06 s
estimated vector disk = 15,809,472 bytes
estimated support JSONL = 120,456,000 bytes
```

The final recovery-surface coverage audit additionally included the original
frozen Generic Frequency/PV1 surface, because downstream recovery can retain a
Generic candidate absent from all three Stage-1 Top10 surfaces. This raised the
exact union to **46,452** contexts. The accepted support refresh reused all
required vectors locally (`fresh=0`) and reproduced historical Generic-BGE
support within maximum absolute difference `4.04e-08`. This is a support-union
orchestration correction, not a candidate or scoring change.

The frozen Generic history cache, completed Full task-vector cache, and frozen
task checkpoint are reused read-only. Fresh vectors go only to the new
versioned local result namespace.

## 3. Full-Pinyin preflight

```text
rows = 34,416
unique contexts required = 42,281
Generic cache reusable = 42,278
Generic vectors missing = 3
Task cache reusable = 42,278
Task vectors missing = 3
bounded Task timing sample = 3 contexts / 6.1025 s including model load
estimated vector disk = 11,712 bytes
estimated support JSONL = 120,456,000 bytes
```

The three additional contexts arise only because the recovery audit covers the
union of frozen Generic and Personal-K5 candidates, while the completed task
cache covered the frozen RetunedStage1 surface.

## 4. Full Q8 portability and cost

The existing Q8 scorer accepts arbitrary frozen segmented-Pinyin sequences and
candidate strings through the same `score_candidates` interface; it has no
Initial-only model architecture or target serialization. Full therefore needs
no protocol change. The frozen Full Personal-K5 surface contains:

```text
nonempty rows = 3,556
candidate pairs = 6,942
K distribution = {0: 30,860, 1: 2,109, 2: 519, 3: 323, 4: 199, 5: 406}
estimated Q8 score time from verified Initial mean = 115.40 s
```

This is reasonably practical, so both Initial and Full will run. Initial Q8
scores are reused exactly; only Full needs fresh Q8 scoring.

## 5. Resource boundary

```text
used_dev3000 = false
used_test = false
Task-BiEncoder retrained = false
Generic predictions regenerated = false
historical caches modified = false
```

All new vectors, supports, Q8 scores, predictions, results, and logs remain
local-only under
`results/personalisation/external_memory_next/posthoc_task_biencoder_calibration_v1/`.
