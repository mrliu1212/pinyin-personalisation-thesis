# M4 Final Architecture and Research Record

Date: 2026-08-26

Status: **Frozen through R7 controllability.**

M4 architecture/accuracy tuning is closed. The next stage is end-to-end
runtime measurement and final closed-split evaluation.

## Final architecture

```text
context + full Pinyin
        |
        v
Frozen PinyinGPT Beam16
        |
        +----------------------------+
        |                            |
        v                            v
Generic Top10                 Personal Recovery
                              /             \
                             /               \
                         Exact             Composition
                           |                   |
                  Frozen Full stack     bounded Top3/span
                  NGram + BGE +         Beam16 DP/Viterbi
                  25-feature LMART      cheap features + LMART
                           |                   |
                     Exact Platt        Composition Platt
                           \                   /
                            \                 /
                             calibrated fusion
                      max(p_exact, p_composition)
                                  |
                    remove Generic Top10 overlap
                                  |
                         Personal-only Top5
                                  |
                                  v
                  Generic Top10 + Personal Top5
                         <=15 unique candidates
                                  |
                                  v
                    R6-v2 full-signal LambdaMART
                                  |
                                  v
                          frozen final ranking
                                  |
                                  v
                      explicit post-model control
```

The architecture separates candidate recovery, final ranking, and explicit
user control.

Principle:

> Learned model learns; explicit Control Layer controls.

The frozen LambdaMART feature values and tree thresholds are never scaled to
simulate control.

## Headline accuracy

Train-Val population:

- Queries: 94,590
- Families: 18,918
- 18,918 queries for each M1-M5 length

| System | Micro Top-1 |
|---|---:|
| Frozen Generic | 48.346548% |
| Frozen Full | 51.236917% |
| Final M4 R6-v2 | **59.800190%** |

Final M4 gain over Frozen Generic:

- **+11.453642 percentage points**
- approximately **+23.69% relative Top-1 improvement**
- **+10,834 net Top-1-correct queries**

Final M4 gain over Frozen Full:

- **+8.563273 percentage points**
- rescue = 9,925
- harm = 1,825
- net = **+8,100 Top-1-correct queries**

## Per-length Top-1

| Length | Generic | Frozen Full | Final M4 | M4 vs Generic |
|---|---:|---:|---:|---:|
| M1 | 79.2843% | 88.0854% | **88.0748%** | **+8.7905 pp** |
| M2 | 60.2178% | 63.3682% | **72.4601%** | **+12.2423 pp** |
| M3 | 44.5396% | 45.7977% | **57.9131%** | **+13.3735 pp** |
| M4 | 32.8735% | 33.6875% | **45.6391%** | **+12.7656 pp** |
| M5 | 24.8176% | 25.2458% | **34.9138%** | **+10.0962 pp** |

The strongest gains therefore occur on multi-token inputs rather than only on
M1.

## R5 recovery

Frozen Generic Top10 recall:

- **71.470557%**

Generic Top10 + Personal Top5 candidate-pool recall:

- **78.620362%**

Candidate-pool gain:

- **+7.149804 pp**

Generic-pruned population:

- N = 25,443
- Exact Recall@5 = 2.389655%
- Composition Recall@5 = 23.574264%
- dual-branch Fusion Recall@5 = **24.403569%**
- recovered Gold in Personal Top5 = **6,209**

Composition recovered 5,998 of the Generic-pruned cases. Exact contributed an
additional **211** cases beyond Composition.

## R6-v2 final ranking

Evaluation protocol:

- 5-fold outer family OOF
- Exact calibration fitted only on each outer training split
- Composition calibration fitted only on each outer training split
- held-out R5 Personal Top5 reproduced exactly
- no hyperparameter sweep
- DEV3000 closed
- Test closed

Held-out R5 pool reproduction:

- **94,590 / 94,590**

Final metrics:

- Micro Top-1 = **59.800190%**
- Macro author Top-1 = **54.636086%**
- Top-3 = **71.748599%**
- Top-5 = **75.031187%**
- Top-10 = **78.029390%**
- MRR = **66.411185%**

Among the 6,209 Generic-pruned Gold candidates restored by R5:

- conditional Top-1 = **64.857465%**
- conditional Top-3 = **81.559027%**
- conditional Top-5 = **90.062812%**

R6 ranker-only latency over an already constructed <=15 pool:

- mean = 0.119033 ms/query
- p95 = 0.1936 ms/query

This is not complete end-to-end latency.

## R7 controllability

R7-A maps the frozen R6 score to a calibrated OOF log-odds coordinate:

`z_base = logit(p_base)`

Frozen R6 rank reproduction:

- **94,590 / 94,590**

Calibration rank reproduction:

- **94,590 / 94,590**

Control formula:

`z_ctrl = z_base + ln(2) * u * phi`

Levels:

`u in {-2, -1, 0, +1, +2}`

At `u=0`, the final ranking exactly reproduces frozen M4 Top-1 = 59.800190%.

Three user-facing control concepts were validated:

1. Personalisation Strength
2. Exact vs Composition Preference
3. Composition Tolerance

All three had **0 query-level monotonicity violations** over the full 94,590
query evaluation population.

Personal Top1 selection rate for Personalisation Strength:

- u=-2: 7.4130%
- u=0: 8.5643%
- u=+2: 9.4069%

For Exact vs Composition control:

- Exact-supported Top1: 0.7770% -> 0.8785% from u=-2 to u=+2
- Composition-supported Top1: 44.1791% -> 42.3322%

This establishes controllability behaviorally without modifying the learned
ranker.

## Frozen stage status

- R0: retrieval feasibility - FROZEN
- R1: bounded composition - FROZEN
- R2: Composition LambdaMART - FROZEN
- R3: Composition calibration - FROZEN
- R4: Frozen Full / Exact raw reconstruction - FROZEN
- R4B: Exact calibration - FROZEN
- R5: calibrated dual-branch recovery - FROZEN
- R6-v2: final <=15-candidate ranker - FROZEN
- R7: explicit controllability - FROZEN

Next stage:

1. end-to-end runtime evaluation
2. latency decomposition
3. final closed-split evaluation
4. thesis tables, figures, and writing
