# EM3 Dev closeout — 2026-08-20

## Status

**DEV-STAGE CHECKPOINT. EM3 is paused, not finished or finally frozen.**

Benchmark Test was not used for this checkpoint. No new GPU inference or heavy
EM3 training was run during closeout. Existing frozen artifacts were not
modified.

## Current scientific state

The canonical diagnostic is Full+Short / H5000 on the old exploratory
three-author Dev surface: Etinjat, Re_spectators, breaddddd, 5,608 rows.
These are diagnostic Micro counts, not replacements for the primary
Macro-author evaluation table.

| G | F | Hidden-M1 | Rows |
|---|---|---|---:|
| ✓ | ✓ | ✓ | 3,361 |
| ✓ | ✓ | ✗ | 24 |
| ✓ | ✗ | ✓ | 42 |
| ✓ | ✗ | ✗ | 100 |
| ✗ | ✓ | ✓ | 403 |
| ✗ | ✓ | ✗ | 35 |
| ✗ | ✗ | ✓ | 45 |
| ✗ | ✗ | ✗ | 1,598 |

Micro correctness is Generic `3527/5608 = 62.89%`, Frequency
`3823/5608 = 68.17%`, and Hidden-M1 `3851/5608 = 68.67%`. Relative to F,
Context rescues 87, harms 59, and nets `+28` rows (about `+0.50` percentage
points Micro Top1).

The 1,598 all-wrong rows contain:

- 1,275 with no Gold in same-Pinyin history;
- 323 with Gold in same-Pinyin history;
- exactly 53.9% of those 323 also have Gold in the current candidate set.

Therefore approximately 174 rows are ranking/fusion opportunities and 149 are
candidate-recovery/EM1-type failures. The values 174 and 149 are approximate
derivations, not separately counted exact integers.

The most informative subsets are:

- `G✗F✗H✓ = 45`: Context-only rescues. Gold is always in history and the
  candidate set, but is the raw history winner only 20% of the time.
- `G✓F✓H✗ = 24`: pure harmful Context overrides.
- `G✗F✓H✗ = 35`: F rescues G but Context loses the rescue. Compared with the
  403 successful `G✗F✓H✓` cases, this group has lower history-winner dominance,
  higher entropy, and more distinct same-Pinyin targets.
- `G✓F✗H✗ = 100`: raw history can override a correct Generic prediction; Gold
  appears in same-Pinyin history in only 39% and is never the raw winner.

Gold, Gold count/share, and correctness groups are oracle analysis fields, not
runtime features. Prediction-visible candidates include history counts,
frequency distribution/winner/share/margin/entropy, Generic confidence,
retrieval/context support, and method agreement.

## Working method direction

Raw user frequency is not identical to personal preference. A common user
target may also be globally common. The next Dev work should investigate:

```text
PersonalLift(c,p) = log(P_user(c|p) / P_global(c|p))
Final(c) = Generic(c) + PersonalScore(c)
PersonalScore(c) ≈ PersonalLift(c) × ContextUtility(query, candidate, history)
```

This is a research direction, not a frozen architecture. Candidate-specific
historical utility is required because Context sometimes correctly selects a
minority historical target. A single global Context/frequency weight would lose
that mechanism.

## Development population

The EM3-v2 main development population is frozen as clean3:

- Agent Phage;
- Etinjat;
- breaddddd.

Re_spectators remains valid provenance for the old 5,608-row diagnostic but has
weak usable Dev supervision. MScarlet remains excluded because of the known
script-normalization confound. QBLevi is smaller than Agent Phage.

## Canonical files

- first-read index: `docs/CURRENT_RESEARCH_INDEX_2026-08-20.md`;
- detailed distribution:
  `docs/external_memory/em3/EM3_ALL_OUTCOME_DISTRIBUTION_RECORD_2026-08-20.md`;
- failure analysis:
  `docs/external_memory/em3/EM3_V2_FAILURE_AUDIT_2026-08-20.md`;
- concise progress record:
  `docs/external_memory/em3/EM3_PROGRESS_2026-08-20.md`;
- data policy:
  `docs/external_memory/em3/EM3_V2_DATA_PREPARATION_2026-08-20.md`;
- method options:
  `docs/external_memory/em3/EM3_V2_METHOD_OPTIONS_2026-08-20.md`;
- execution plan:
  `docs/external_memory/em3/EM3_V2_EXECUTION_PLAN_2026-08-20.md`;
- consolidated runner: `experiments/external_memory/em3_all_outcome_audit.py`;
- formal pair generator:
  `experiments/external_memory/em3_generate_train_pairs.py`.

Earlier context example/cluster scripts remain useful provenance helpers:

- `em3_context_failure_examples.py`;
- `em3_context_failure_cluster_audit.py`;
- `em3_context_outcome_examples.py`.

## Consolidated audit provenance

Output root:

```text
results\personalisation\external_memory\em3_all_outcome_audit\
```

Expected outputs: `summary.json`, `provenance.json`, `report.txt`,
`all_rows.jsonl`, `groups\`, and `focused_subsets\`.

| Input | SHA256 |
|---|---|
| `history_manifest.jsonl` | `7c85c38728d03985856d742f452992b3b3072af5f1c07845e099d9d07854da68` |
| `dev_manifest.jsonl` | `cf072d9323328b77e3d47d8a0c1beed8c40edc8767e075fb58593d6b72120606` |
| `em2_four_way_dev_compare/rows.jsonl` | `7bc20cddc5a772e7c1f9fb3fdd60ec17e8c2813667b7c32ec835b4cbc15d87d7` |
| `em2_fixed_gfc_dev/selected_rows.jsonl` | `6e4007b2ba7cd0bffea4c869a7860cc08c3671bf078c22e957ad09d6ce18ea25` |

```powershell
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
& $python -m experiments.external_memory.em3_all_outcome_audit
```

## Pair-generator regression checkpoint

The canonical runner regression-reproduces the frozen EM3-BCE v1 counts on the
old exploratory authors:

| Item | Count |
|---|---:|
| Eligible queries | 30,968 |
| Positive pairs | 86,959 |
| Negative pairs | 146,195 |
| Total pairs | 233,154 |

It also reproduces the round distribution `2227 / 1491 / 27250` for one/two/
three rounds and per-author totals `91,556 / 13,830 / 127,768` for Etinjat,
Re_spectators, and breaddddd. The audit reports zero non-prior pairs and zero
query-history reuse.

```powershell
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
& $python -m experiments.external_memory.em3_generate_train_pairs `
  --authors Etinjat Re_spectators breaddddd `
  --audit-only `
  --output-root results\personalisation\external_memory\em3_train_pairs_v1_regression_audit
```

Source manifest SHA256:
`6d32d44189c0824d7973a5a9a50359dce3fb8111f6f7a9078580eb69fac58597`.

Frozen existing v1 pair manifest SHA256:
`8729f0db9ea2d4cd5c82ef812d743cdb37f551b6ddfa591b3d788b42d5a8dee2`.

Frozen existing v1 summary SHA256:
`c9161b187e4cace65d8c33e55b96c2a109e5aecabba525107c3e2fa89f6fc0bd`.

The audit-only run does not rewrite or duplicate the 708 MB frozen pair file.
A full future clean3 generation records source, runner, and output hashes.

## Independent Codex Dev result

The isolated branch `codex/em3-performance-research-20260820` reported a
prediction-visible G/F/M1 utility gate on Initial+Short clean3. On 3,296
untouched confirmation rows, Macro-author Top1 was Gate `0.349221` versus F
`0.346863` (`+0.002357`); Micro/Ambiguous/Conflict deltas were `+0.002731`,
`+0.002702`, and `+0.004177`; rescue/harm/net was `80/71/+9`.

The bootstrap 95% interval `[-0.005365, 0.010024]` crosses zero and exact
McNemar is `p=0.5152`. Full+Short support was `+0.001633`. This is a promising
frozen Dev candidate, not a statistically conclusive general advance. No Test
or new GPU inference was used. Do not merge the isolated branch or copy its
generated result trees into this worktree.

## Exact resume order

1. Re-read this closeout, the current index, distribution record, and failure
   audit.
2. Reconfirm the old pair-generator counts before altering sampling semantics.
3. Generate/audit clean3 pairs and preserve all hashes; do not train on a
   failed audit.
4. Freeze a prediction-visible global-frequency source and PersonalLift
   definition.
5. Run a small Dev-only candidate-specific utility experiment emphasizing the
   45 rescue and 24/35 harm mechanisms.
6. Rebuild the pointwise BCE baseline on clean3 if supervised comparison is
   pursued.
7. Use a separate Dev confirmation surface and paired statistics. Keep Test
   closed until a later complete method freeze.

## Git closeout policy

Stage only human-authored documentation, canonical scripts, and tests using
explicit paths. Keep all `results/`, caches, JSONL, SQLite, logs, checkpoints,
temporary files, and duplicate `*_fixed.py` files local and unstaged. The
planned local checkpoint commit is `EM3: freeze Dev diagnostics and v2
preparation`. Do not push or create the planned
`external-memory-em3-dev-audit-20260820` tag without explicit approval.
