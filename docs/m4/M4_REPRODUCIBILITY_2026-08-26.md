# M4 Reproducibility Record

Date: 2026-08-26

## Git lineage

This M4 checkpoint was promoted into a dedicated Git worktree from the
`work/context-model-comparison` research line because M4 directly extends the
Frozen Full/context-comparison pipeline.

The exact base commit is recorded in `M4_CHECKPOINT_MANIFEST_2026-08-26.json`.

## Local execution environment

Primary Python environment:

`C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe`

Standalone M4 development workspace:

`C:\Users\chiar\Desktop\LBH\thesis-m4-recovery`

LightGBM:

- 4.7.0

Large generated JSONL files, caches, model-input populations, and local result
artifacts remain local-only. Git retains the code, small freeze manifests,
small summaries, control curves, exact SHA256 identifiers, and experimental
protocol needed to identify the frozen run.

## Data contract

Train-Fit:

- raw rows = 144,526
- Multi rows = 411,105

Train-Val:

- raw rows = 34,416
- Multi rows = 94,590
- families = 18,918
- M1-M5 = 18,918 each

Frozen Train-Val Multi SHA256:

`9789aaa3d5f5a2276f31b18b8047c43c210061a13fd7a2383f0b04947c2d7c9b`

Frozen Generic M0 rows SHA256:

`59cb59170e348c60747522fef2c0af94deaaa0cc9cd022db273063249618915e`

Split invariants:

- raw Fit/Val overlap = 0
- Multi Fit/Val overlap = 0
- missing anchors = 0
- R2_TRAIN_SPLIT = TRAIN_FIT
- R2_EVAL_SPLIT = TRAIN_VAL
- DEV3000 = CLOSED
- TEST = CLOSED

## Generic model

Checkpoint:

`aihijo/transformers4ime-pinyingpt-concat`

Checkpoint revision:

`76dd20dc92d8236a350fb732e99dde6fa15e2263`

Official code revision:

`8f1573ed0bd4d1f3d8d3f10a05f7e870725646f1`

Frozen decoding:

- Beam16
- TopK10
- n_positions = 1024

## Personal-history contract

- same-author history is strictly causal
- H5000 is measured in raw history events
- H5000 is applied before exact segmented-Pinyin filtering
- M1-C derived entries do not consume raw H5000 slots
- within-event exact-Pinyin duplicates are removed
- Personal-K5 removes Generic overlap before K5
- compatibility filtering occurs before K5
- Personal-K5 is a Personal-only novel recovery budget

Gold is never used for runtime candidate generation, runtime scoring, fusion
selection, or held-out ranking. Gold is used only for documented training
labels and post-hoc evaluation/tracing.

## Frozen Full LambdaMART

Model SHA256:

`406b1693e5b8bb10b0af92c6bb31f494f8a78a13590d47ec5bf138fdba18df4e`

Configuration:

- objective = LambdaRank
- max_depth = 5
- num_leaves = 31
- min_data_in_leaf = 500
- rounds = 100
- learning_rate = 0.05
- seed = 1729
- deterministic

## Frozen M4 identifiers

### R0

Script SHA256:

`8ce15e772a9f088bdb4690caeac5cfa75b2fe1941829c39bde1e77b998398c66`

Frozen decision:

- per-span Top3 for R1+

### R1

Runner SHA256:

`65793fce44cd8765e936bddcd24c0cdaa257bec0cf65eaebc87b0a417ab869b7`

Rows SHA256:

`8e352875d8c8d93f0479fcaf35db68f46c44e1d5468cc7a0794ae05dd7864356`

Summary SHA256:

`531b8cc474ce9eede796770274d8e77bc9230b783dcc49271fa39c3f7485129f`

### R2

Composition LambdaMART model SHA256:

`9917cd51a47f591e2170bd52671f06ccbced49492ede30980a7993cb3c9e5cb4`

Training:

- Train-Fit only
- 15 cheap Composition features
- fixed C1 configuration
- no hyperparameter sweep

### R3

Composition calibration:

- deterministic 5-fold family OOF Platt
- Train-Val evaluation uses OOF probability
- final all-Val calibrator retained only for future unseen data
- ranking unchanged

Final all-Val Platt:

- a = 1.2331601816929783
- b = -2.25760956676882

### R4

Runner SHA256:

`a61f30d32ba799aea70c616072e29779534551b7483bcedf911d0afa43b4a50a`

Rows SHA256:

`7cfe23b1aaa8a64f02432938bdb8f84fbae0d8a4bbcb791105f1488b6fa323dd`

The current Windows reconstruction produces M1 Top1 16,664/18,918 whereas
the historical HPC record was 16,663/18,918. M2-M5 reproduce exactly. The
one-row difference is preserved as provenance and is not silently reconciled.

### R4B

Runner SHA256:

`be007d70b45bfbe81472d1407235046170ce37bab24740fb1e2897ff923d4b02`

Calibrated rows SHA256:

`dd569f0baa8a89faac5352cfb77d3261561be3f81ec5f0c4ba8344d49184b8a7`

Exact calibration population:

- candidates = 3,099
- positives = 684
- active queries = 2,096

Final Platt:

- a = 1.9456577892364941
- b = -1.9237066877193154

### R5

Runner SHA256:

`5a59af81ca7fe30b0b069da803740e75488b6049a4f85caccc98000b2a339943`

Fusion rows SHA256:

`be09bc541f635e0a44fd7662714b3afc96488e6f5c1523895c0016d2be5fbd1a`

Summary SHA256:

`94aaae56c22250f54145db3cc974adcc2dfc2b2b59cd41756381eb5b5b22917c`

Freeze SHA256:

`6c51da9bcbe677b922e7b8b8276dc8f762e65aab766898f2a40f010a596ca4f5`

Frozen fusion:

- same-text max(p_exact, p_composition)
- Train-Val uses branch-specific OOF probabilities
- Generic Top10 overlap removed before recovery allocation
- Personal-only Top5
- final pool <=15 unique candidates

### R6-v2

Runner SHA256:

`b8ee6989058a8ba15f187eff0b72cf52354389a53b406c706462814579bf838f`

Final model SHA256:

`db2072cef940eab98295262610fd3df45baae729e5ec98737e950113381aa87d`

Summary SHA256:

`b0fa14cdeca6b0caf5292d1171e569d2dc163a5eea54fe4cebb9e6b48ef07cf1`

Freeze SHA256:

`be001bab70c56dbedfc59af7587facdd3d8ed7cce44b9547361ef3ef61fec4b6`

Protocol:

- 5-fold outer family OOF
- branch calibrators fitted only on outer training families
- held-out R5 pools reproduced 94,590/94,590
- no R6 hyperparameter sweep
- DEV3000 closed
- Test closed

### R7-A

Runner SHA256:

`a18de46673d2f5d3689049fd91c9dc293b9d2484061ca11fdfe196cbaa313218`

Control-surface SHA256:

`4841a20c150c17c3057c180d333dad1efa25f4b12960790a51be3f7f745d918b`

Summary SHA256:

`9aaf8b2ec1a4ea74eb41d00ea5fbf0b50acc784a0e3e00107e81e5c711e17db5`

Frozen R6 and calibrated-rank reproduction:

- 94,590/94,590

### R7-B

Runner SHA256:

`4a7bade3252d7b13e4c3643925c28fefc91be466f620dc7df48b6512f669117c`

Control curves SHA256:

`229d2191bb06e1615b42170e13217c33b04f01a6197c6c80e1fbfcc9d4dc07a5`

Summary SHA256:

`2f6eca0f07fdbaf90619da06b0c6ef0ac19d3032c9502637a828f37001399ef2`

R7 freeze SHA256:

`b02904430a67df6b581a335a63f558bd71539a3f8ed925f0887c0682d73cda83`

All three controls have zero query-level monotonicity violations and u=0
exactly preserves the final M4 ranking.

## Current frozen headline

Frozen Generic Top1:

- **48.346548%**

Final M4 default Top1:

- **59.800190%**

Absolute improvement:

- **+11.453642 pp**

Architecture status:

- Recovery CLOSED
- Final ranking CLOSED
- Controllability CLOSED

Next work:

1. end-to-end runtime evaluation
2. latency decomposition
3. final closed-split evaluation
4. thesis tables/figures/writing

Do not reopen R0-R7 for accuracy tuning without a new explicit scientific
reason.
