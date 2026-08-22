# Standardized reset decision log — 2026-08-20

Every decision below has `used_test=false`. Dev3000 was used only for immutable
identity/history audit where explicitly stated, never for selection.

## Preserve historical artifacts but reset historical tuning

- **Motivation / question:** obtain an apples-to-apples comparison without
  destroying provenance or inheriting differently selected parameters.
- **Evidence:** historical methods used overlapping but not identical
  populations, caches, and development selections.
- **Options:** overwrite/reinterpret history; retain history and reuse its
  settings; retain history but reselect settings under one new Train-Val.
- **Chosen:** preserve every historical artifact and independently retune the
  allowed system parameters on the new Train-Val.
- **Rejected:** overwrite loses provenance; automatic parameter reuse defeats
  the standardized reset.
- **Consequence:** prior artifacts are regression/cache candidates only when
  exact scientific identities match.
- **Next:** freeze split, registry, and workload audit.

## Retain the frozen balanced Dev3000, but do not call it virgin

- **Motivation / question:** define one sealed, balanced evaluation surface.
- **Evidence:** the manifest is already frozen at 1,000 rows per Clean3 author
  with SHA256 `9181f895...b03f93`; some earlier exploratory development
  overlapped it.
- **Options:** construct a strict-fresh Dev; reuse Dev3000 while falsely calling
  it untouched; reuse it with explicit exposure wording.
- **Chosen:** keep the exact manifest and state that it was sealed before the
  standardized protocol, not historically virgin.
- **Rejected:** a new Dev changes the benchmark; virgin wording is inaccurate.
- **Consequence:** no standardized training, tuning, grid expansion, checkpoint
  selection, or gating may use Dev3000.
- **Next:** PRE_DEV_FREEZE must precede any standardized Dev predictions.

## Use rolling causal H5000 before exact-Pinyin filtering

- **Motivation / question:** establish the personal evidence visible to each
  query independently of batching.
- **Evidence:** `HistoryIndex(history + dev, H5000)` is used by Pilot/matrix and
  historical hidden M1/M2/EM3 Dev runners. The historical population audit
  asserts same user, strict priority, raw-budget-before-Pinyin, and earlier Dev
  becoming history. See the dedicated history record.
- **Options:** fixed pre-Dev history; rolling causal Train plus earlier Dev.
- **Chosen:** preserve rolling causal history. For every query, select same-user
  strictly prior rows, take the latest up to 5000 raw interactions, then filter
  exact segmented Pinyin.
- **Rejected:** fixed pre-Dev history contradicts the frozen Dev implementation
  and the intended online-memory setting.
- **Consequence:** Train-Val may see earlier Train-Fit and earlier Train-Val;
  no row sees itself or future rows. Early Train-Fit queries may use fewer than
  5000 prior interactions.
- **Next:** assert these invariants in manifests and tests.

## Deterministic whole-work Train-Fit/Train-Val rule

- **Motivation / question:** obtain a chronological validation surface without
  within-work leakage.
- **Evidence:** authoritative Train rows contain stable author, work ID, work
  chronological index, creation date, and interaction position.
- **Options:** random row split; partial final work; latest complete-work suffix.
- **Chosen:** per author, choose the latest contiguous complete-work suffix whose
  row share lies in 15–20%, minimizing distance to 20%; ties prefer the larger
  validation share, then the earlier deterministic boundary.
- **Rejected:** row shuffling and partial works weaken chronology/isolation.
- **Consequence:** exact shares may differ across authors while scientific
  whole-work separation takes priority.
- **Next:** freeze versioned manifests and audit all overlaps/hashes.

## Freeze neural identities; retrain only EM3-Clean3 from base

- **Motivation / question:** separate representation comparison from arbitrary
  retraining differences.
- **Evidence:** Generic, BGE, and generic CE are method-defining frozen models;
  EM3 is explicitly the task-specific learned utility scorer.
- **Chosen:** freeze Generic/M1/M2/Hidden-M1/Hidden-M2 weights. Train one shared
  cross-user EM3 from pinned generic base on causal Clean3 Train-Fit pairs.
- **Rejected:** initialize from old EM3 or fit one checkpoint per author.
- **Consequence:** user specificity comes from external same-user memory, while
  EM3 parameters learn population-level pair utility.
- **Next:** freeze pair recipe before generation and record its manifest SHA.

## Freeze bounded search spaces before new Train-Val results

- **Motivation / question:** prevent outcome-driven grid growth.
- **Evidence:** historical grids support bounded K/Top-N and lambda values.
- **Chosen:** freeze the machine-readable grids in
  `search_space_registry_v1.json`, primary Macro-author Top1, tie-break lower
  lambda then lower K/Top-N, with no boundary expansion.
- **Rejected:** dynamic expansion after observing results.
- **Consequence:** Train-Val alone selects system configurations.
- **Next:** do not alter grids after the registry is written.

## Allow only exact cache reuse

- **Motivation / question:** reduce compute without changing meaning.
- **Evidence:** historical caches vary by model revision, preprocessing,
  query/history identities, K, and serialization.
- **Chosen:** reuse only when model/tokenizer revision, source identity,
  context preprocessing, H5000 semantics, exact candidate/pair identity, and
  score meaning match. Reconstruct deterministic CPU results where possible.
- **Rejected:** substitute old EM3 scores for newly trained EM3-Clean3.
- **Consequence:** semantic changes naturally create distinct cache identities;
  true misses alone require forward work.
- **Next:** produce exact row/pair coverage counts before GPU execution.

