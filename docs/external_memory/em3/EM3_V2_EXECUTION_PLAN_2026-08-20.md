# EM3-v2 Execution Plan — 2026-08-20

## Goal

Build a cleaner and more rigorous second EM3 experiment.

### 2026-08-20 closeout status

This plan remains a preserved method-development plan, not a frozen final
architecture. The clean3 author decision is complete (Agent Phage, Etinjat,
breaddddd), the formal pair generator now regression-reproduces the old v1
counts, and the full 5,608-row outcome audit shifted the immediate focus toward
user-vs-global frequency lift and candidate-specific historical utility. The
canonical current status and resume order are in
`EM3_DEV_CLOSEOUT_2026-08-20.md`; where this earlier sequence differs, the
closeout note controls. No heavy training should start before clean3 pair and
provenance audits pass.

The objective is not merely to improve a single Dev Top1 number. The next experiment must make it possible to answer:

1. Does task-specific memory ranking outperform generic memory ranking?
2. Does it improve final candidate ranking, not only history discrimination?
3. Are improvements stable across authors?
4. Does it improve difficult ambiguous and conflict cases?
5. Are observed gains statistically reliable?

## Step 1 — Replace the weak author

The current three-author set contains `Re_spectators`, whose usable EM3 Dev population is too small for a stable author-level comparison.

Next action:
- inspect all six authors using the existing population audits;
- choose a replacement with substantially larger pair-trainable Train and Dev support;
- prefer a cleaner author without the known MScarlet script-normalisation confound.

Current likely clean candidates to compare:
- Etinjat
- Agent Phage
- breaddddd

Before freezing, produce one compact author-selection table:
- Train history rows
- Train pair-trainable queries
- Dev tune queries
- Dev pair-trainable queries
- ambiguous count
- conflict count
- known script / reconstruction concerns

Freeze the new three-author set before retraining.

## Step 2 — Rebuild the BCE baseline on the new author set

If the author set changes, retrain the pointwise BCE baseline on exactly that new set.

Reason:
The old BCE model was trained on the old three-author population. Comparing a new method trained on different authors against the old model would not be fair.

Keep the old BCE run as the v1 exploratory baseline.

## Step 3 — Execute the next EM3 method

Start from the already-written EM3-v2 method options rather than inventing another ad-hoc variant.

Priority direction:

### Preferred v2
A ranking-oriented Cross-Encoder objective using positive vs hard-negative histories.

Candidate options already discussed:
- LCE / listwise Cross-Entropy
- pairwise RankNet-style loss
- LSEPair-style ranking loss

The next method should directly reward:
`positive history score > hard-negative history score`

Hard negatives should emphasize:
- same Pinyin
- different target
- contextually plausible histories
- conflict-like cases where simple frequency is misleading

Do not simply add more ordinary random negatives.

## Step 4 — Use the stronger frozen retrieval surface

For the main comparison, use frozen PinyinGPT hidden-state retrieval.

Reason:
Earlier EM2 retrieval evaluation showed that hidden-state kNN had stronger history retrieval discrimination than BGE retrieval.

Main controlled comparison:

- Hidden-M1:
  hidden retrieval -> simple memory support
- Hidden-M2:
  hidden retrieval -> generic pretrained Cross-Encoder -> target aggregation -> Generic fusion
- EM3-v2-Hidden:
  same hidden retrieval -> task-specific Cross-Encoder -> same target aggregation -> same Generic fusion

This makes Hidden-M2 vs EM3-v2-Hidden a clean supervision comparison.

Original BGE-M2 should remain as an ablation / secondary baseline.

## Step 5 — Freeze a rigorous comparison protocol

### Primary end-to-end metric
- Macro-author Top1

### Secondary end-to-end metrics
- Micro Top1
- Ambiguous Top1
- Conflict Top1
- candidate-level MRR, if complete candidate ranks are stored consistently

### Paired diagnostic metrics
- rescue
- harm
- net rescue
- G wrong / F right
- G right / F wrong
- Hidden-M1 vs Hidden-M2 disagreement
- Hidden-M2 vs EM3-v2 disagreement

### Statistical reliability
Add:
- paired bootstrap 95% confidence interval for Top1 difference
- McNemar test for paired Top1 correctness
- optionally paired bootstrap p-value / probability of improvement

Primary statistical comparison:
`EM3-v2-Hidden vs Hidden-M2`

Secondary:
- EM3-v2-Hidden vs Hidden-M1
- EM3-v2-Hidden vs F
- Hidden-M2 vs Hidden-M1

## Step 6 — Persist row-level outputs for every method

Every future evaluation runner must save row-level results, not only print summaries.

Minimum row schema:
- row_id
- anchor_id if available
- author
- gold
- rank
- correctness
- history_available
- ambiguous
- conflict
- selected hyperparameters / method provenance

For memory methods, also preserve:
- retrieved history IDs
- history targets
- retrieval similarities
- Cross-Encoder scores
- aggregated target support

Reason:
This avoids repeated repository searches and makes later paired analysis purely offline.

## Step 7 — Dev / Test discipline

Do not touch Test yet.

Use Dev for:
- model / loss selection
- K selection
- fusion lambda selection
- diagnostics

Once the new author set, objective, retrieval surface, aggregation and hyperparameters are frozen:
- run the final Test exactly once;
- report Macro-author primary results;
- include 95% CI / paired significance where appropriate;
- do not tune from Test.

## Current working conclusion

The current EM3-BCE v1 does not provide a convincing end-to-end improvement.

This does not yet show that task-specific memory ranking is ineffective.

The experiment exposed two concrete weaknesses that the next stage must control:

1. unstable / noisy author data;
2. a pointwise BCE objective that is weak on conflict and relative ranking.

EM3-v2 should therefore be treated as a cleaner, better-controlled experiment rather than another small hyperparameter tweak of v1.
