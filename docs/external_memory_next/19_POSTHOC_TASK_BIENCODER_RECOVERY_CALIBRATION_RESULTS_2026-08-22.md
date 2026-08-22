# 19 - Post-hoc Task-BiEncoder Recovery and Calibration Results

Date: 2026-08-22

Status: **COMPLETE / TRAIN-VAL POST-HOC DIAGNOSTIC / DEV3000 AND TEST CLOSED**

## 1. Outcome

Equal downstream calibration and the frozen Recency mechanism did not convert
the task-specific encoder's intrinsic history-retrieval gain into an overall
IME gain. Generic BGE remained better on candidate-support Top1 and on the
primary fixed-surface end-to-end metric in both tracks. LambdaMART remains the
best Full Train-Val result.

The one narrow Task advantage was Personal-K5 discrimination after combining
with `P_NG`: on Initial K2+, `P_NG+Task-R` reached `.593512` versus `.593282`
for `P_NG+Generic-R`; on Full K2+, both had Macro `.814801`, with Task having
slightly higher MRR. These did not produce an overall end-to-end win. No
significance claim is made.

```text
used_dev3000 = false
used_test = false
Task-BiEncoder retrained = false
Generic candidate surface changed = false
```

This post-hoc Train-Val diagnostic asked whether Task-BiEncoder needs different
downstream calibration, benefits from frozen Recency, is more useful for
Personal-K5 recovery than fixed-surface reranking, complements NGram evidence
better than Generic BGE, or approaches Q8/Q8+F accuracy at lower latency.
Initial and Full were strictly separate tracks, each with exactly **34,416**
canonical Train-Val interactions. No result here is a new confirmatory model
selection; the very small Generic recovery-remerge increments are diagnostic.

The combined evidence is consistent with the remaining bottleneck lying more
in history-to-candidate evidence aggregation and conditional decision/fusion
than in retrieval representation alone. This is not causal proof, and no
statistical-significance claim is made.

## 2. Fidelity and comparability gates

- Both tracks contain exactly 34,416 canonical Train-Val rows.
- History is same author, strictly prior, latest H5000 raw interactions before
  exact segmented-Pinyin filtering.
- Generic and Task supports use the same context64, cosine-only Top5,
  candidate conditioning, `max(0, cosine)`, `exp(-age/2048)`, and surface
  normalization.
- Frozen Initial Generic operating points reproduced for all three Stage-1
  surfaces.
- Historical Initial Q8 K2+ population reproduced exactly: 4,471 rows, Q8
  Macro `.636530666806`, Q8+F alpha `.75` Macro `.669164140863`.
- Full RetunedFinal reproduced exactly: Macro `.796004926550`, Micro
  `.824994188749`, MRR `.871377872879`, Missing `.051981636448`.
- Generic BGE support reconstruction maximum error was `4.04e-08` for Initial
  and zero for Full.
- Historical Initial Stage-1 and support JSONL are row-ID sorted, whereas the
  canonical manifest is in interaction order. Row-ID populations were exactly
  equal, and every new artifact restored manifest order deterministically.
- Frozen Generic-BGE fidelity required the exact batched
  `history_matrix @ query_vector` operation. Independent vector dot products
  can round near-ties differently, changing cosine-only Top5 membership before
  recency aggregation.
- Historical Q8+F uses a sum-normalized log-frequency distribution over its
  candidate surface. It is distinct from both Stage-1 Choice Share and the
  recovery-stage max-normalized frequency signal.
- A coverage audit found that Initial downstream recovery may retain original
  Generic candidates absent from all three Stage-1 Top10 surfaces. The support
  union was corrected to cover frozen Generic union Personal-K5 union all
  required Stage-1 surfaces. This uses the hash-validated Frequency/PV1
  Generic surface, added 18 required contexts (46,434 to 46,452), changed no
  candidate or scoring semantics, and required zero fresh embeddings on the
  accepted refresh. Final Initial support SHA256 is
  `2c2bc7faddab4c032baf58d23a0767e6c31881bdd4c08f348cb865f43e4fced3`.

## 3. Initial-Pinyin

### 3.1 Fixed-surface calibration: primary balanced surface

| Method | lambda_N | lambda_E | Macro Top1 | Micro Top1 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Stage1 `4P+4CS+2E` | 0 | 0 | .404807 | .429364 | .537433 |
| NGramRecency | 6 | 0 | .432451 | .455718 | .556494 |
| Generic BGE | 0 | 4 | .410464 | .434420 | .542158 |
| Generic BGE-R | 0 | 12 | .419559 | .443195 | .548043 |
| Task Bi-Encoder | 0 | 4 | .408881 | .432706 | .541168 |
| Task Bi-Encoder-R | 0 | 12 | .416505 | .440115 | .546435 |
| NGram + Generic BGE-R | 4 | 6 | **.437058** | **.460571** | **.559755** |
| NGram + Task Bi-Encoder-R | 6 | 6 | .435646 | .458653 | .558564 |

The equal grid selected the frozen Generic operating point exactly. Task-R did
not outperform Generic-R alone or jointly. Task joint support was better on
the recoverable subset (`.445510` versus `.421864` Macro Top1), but worse on
Generic-covered rows (`.606972` versus `.615512`), producing a lower overall
score. Missing@10 was identical (`.243172`) because candidate surfaces were
fixed.

The other frozen surfaces led to the same conclusion:

| Surface | Generic joint selected | Generic Macro | Task joint selected | Task Macro |
|---|---:|---:|---:|---:|
| K5+Entropy | N6/E8 | .436767 | N6/E1 | .435478 |
| 4P+4CS+2E | N4/E6 | .437058 | N6/E6 | .435646 |
| 6P+2CS+.25E | N4/E6 | .436477 | N4/E6 | .435342 |

Primary fixed-surface per-author Top1:

| Method | Agent Phage | Etinjat | breaddddd |
|---|---:|---:|---:|
| NGram + Generic BGE-R | .493923 | .275218 | .542032 |
| NGram + Task Bi-Encoder-R | .490430 | .277086 | .539423 |

Primary fixed-surface subset Macro Top1:

| Subset | n | Generic joint | Task joint |
|---|---:|---:|---:|
| Ambiguous | 30,527 | .441157 | .439552 |
| Conflict | 15,353 | .177684 | .183407 |
| Recoverable | 4,910 | .421864 | .445510 |
| Generic-missing | 12,565 | .172686 | .181832 |
| Generic-covered | 21,851 | .615512 | .606972 |

### 3.2 Personal-K5 candidate discrimination

Primary comparable table: Generic-missing, Gold in Personal-K5, K>=2,
`n=4,471`.

| Scorer | Selected | Macro Top1 | Micro Top1 | Top3 | MRR |
|---|---:|---:|---:|---:|---:|
| Frequency | - | .491082 | .494968 | .847685 | .683822 |
| P_NG | - | .590783 | .594051 | .898009 | .750760 |
| Generic BGE | - | .536167 | .538582 | .885484 | .718840 |
| Generic BGE-R | - | .539075 | .542161 | .879669 | .718355 |
| Task Bi-Encoder | - | .535574 | .538135 | .881682 | .717498 |
| Task Bi-Encoder-R | - | .528805 | .532096 | .877432 | .710851 |
| P_NG + Generic BGE-R | N1/E.25 | .593282 | .596511 | .897338 | .752602 |
| P_NG + Task Bi-Encoder-R | N1/E.25 | .593512 | .596735 | .897338 | .752382 |
| Q8 | - | .636531 | .642362 | .912995 | .783568 |
| Q8+F | alpha_F=.75 | **.669164** | **.675688** | **.925744** | **.804015** |

K>=2 per-author Top1:

| Scorer | Agent Phage | Etinjat | breaddddd |
|---|---:|---:|---:|
| P_NG + Generic BGE-R | .565305 | .563967 | .650573 |
| P_NG + Task Bi-Encoder-R | .564635 | .564724 | .651177 |
| Q8 | .680509 | .537472 | .691611 |
| Q8+F | .705291 | .564724 | .737477 |

On the complete Generic-missing recoverable denominator (`n=4,910`), Macro
Recovery@1 was `.628423` for `P_NG+Generic-R`, `.628642` for
`P_NG+Task-R`, `.666821` for Q8, and `.696537` for Q8+F. All methods have
Recovery@5=1 because Gold is defined to be inside the same Personal-K5 pool.

### 3.3 Recovery remerge and end-to-end result

Both recovery grids selected N6/E0: encoder evidence was rejected at the
recovery boundary. Reconnecting the selected recovery surface to the frozen
downstream context pipeline gave:

| Variant | Macro Top1 | Micro Top1 | MRR@10 | rescue | harm | net |
|---|---:|---:|---:|---:|---:|---:|
| Frozen balanced Generic system | .437058 | .460571 | .559755 | - | - | - |
| Recovery remerge + Generic downstream | **.437311** | **.460832** | .559282 | 148 | 139 | +9 |
| Recovery remerge + Task downstream | .435988 | .459147 | .558364 | 452 | 501 | -49 |

The small Generic Macro increase is a controlled post-hoc diagnostic, not a
new predeclared winner. It trades lower MRR and higher Missing@10 (`.245729`
versus `.243172`).

Recovery-remerge downstream subset Macro Top1:

| Subset | Generic downstream | Task downstream |
|---|---:|---:|
| Ambiguous | .441442 | .439943 |
| Conflict | .184032 | .187373 |
| Recoverable | .416906 | .439299 |
| Generic-missing | .169707 | .178781 |
| Generic-covered | .616052 | .608387 |

## 4. Full-Pinyin

### 4.1 Fixed-surface calibration

| Method | lambda_N | lambda_E | Macro Top1 | Micro Top1 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| RetunedStage1 | 0 | 0 | .789424 | .819038 | .867756 |
| NGramRecency | 6 | 0 | .794988 | .823803 | .870676 |
| Generic BGE-R | 0 | 4 | .792633 | .822292 | .869739 |
| Task Bi-Encoder-R | 0 | 6 | .792140 | .821769 | .869526 |
| NGram + Generic BGE-R | 6 | 6 | **.796005** | **.824994** | **.871378** |
| NGram + Task Bi-Encoder-R | 8 | 8 | .795748 | .824616 | .871178 |
| Historical LambdaMART | nonlinear | - | **.798839** | **.827784** | **.873043** |

Overall per-author Top1:

| Method | Agent Phage | Etinjat | breaddddd |
|---|---:|---:|---:|
| NGram + Generic BGE-R | .887272 | .601494 | .899249 |
| NGram + Task Bi-Encoder-R | .886398 | .601993 | .898853 |
| Historical LambdaMART | .890401 | .604732 | .901384 |

Fixed-surface subset Macro Top1:

| Subset | n | Generic joint | Task joint |
|---|---:|---:|---:|
| Ambiguous | 10,053 | .803894 | .802658 |
| Conflict | 1,865 | .252588 | .257843 |
| Recoverable | 620 | .863694 | .869487 |
| Generic-missing | 2,382 | .291378 | .292984 |
| Generic-covered | 32,034 | .845657 | .845198 |

Task recalibration reduced the old fixed-fusion gap from about `.000293` Macro
to `.000257`, but did not reverse it. Task joint support again improved the
recoverable subset (`.869487` versus `.863694`) and Conflict (`.257843`
versus `.252588`) while slightly harming Generic-covered rows.

### 4.2 Personal-K5 candidate discrimination

Full has 620 Generic-missing recoverable rows, including 196 with K>=2. On
K>=2, `P_NG+Generic-R` and `P_NG+Task-R` tied at Macro `.814801`; Task had
MRR `.882653` versus `.881803`. Full Q8 was directly portable without a new
serialization protocol. Q8+F alpha `.75` was best on K>=2 at Macro `.901900`.
Per-author Top1 for both NGram combinations was Agent Phage `.750000`, Etinjat
`.785311`, and breaddddd `.909091`; Q8+F was `1.000000`, `.796610`, and
`.909091`, respectively.

On all 620 recoverable rows, Q8+F Macro Recovery@1 was `.963086`, while
`P_NG+Generic-R` and `P_NG+Task-R` tied at `.957104`. This Full population is
small and heavily contains K=1 rows; it must not be numerically ranked against
the Initial K>=2 table as if populations were identical.

### 4.3 Recovery remerge and end-to-end result

Both recovery grids selected N4/E0. The Generic downstream variant reached
Macro `.796248` (22 rescues, 14 harms, net +8) with slightly worse Missing;
the Task downstream variant reproduced `.795712` (54 rescues, 66 harms, net
-12). LambdaMART remains clearly best at `.798839`.

Recovery-remerge downstream subset Macro Top1:

| Subset | Generic downstream | Task downstream |
|---|---:|---:|
| Ambiguous | .804666 | .802711 |
| Conflict | .255359 | .260550 |
| Recoverable | .846942 | .862951 |
| Generic-missing | .286542 | .291618 |
| Generic-covered | .846531 | .845489 |

## 5. Intrinsic retrieval versus final ranking

On the identical 9,325-query legal history surface, Task improved Macro
Recall@1 from `.778944` to `.810971` and Micro Recall@1 from `.788204` to
`.821233`. Recall@5 was `.962252` versus `.959786`. However candidate-level
target-support Top1 fell from `.796009` to `.772902` Macro (`.803861` to
`.781233` Micro). The task representation retrieves the correct interaction
more often but its aggregated candidate evidence is less discriminative. Equal
calibration, Recency, and recovery-stage use did not remove this mismatch.

## 6. Latency and Pareto

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU, PyTorch `2.11.0+cu128`, CUDA
12.8. Task was measured on 500 canonical Initial Train-Val queries after 20
warmups, batch size 1, with cached history embeddings but fresh online query
embedding.

| Method/component | Mean ms | p50 ms | p95 ms | Source |
|---|---:|---:|---:|---|
| Interpolated P_NG | .0249 | - | .0770 | historical exact online run |
| Generic BGE query+lookup | 2.1362 | 1.9875 | 3.0524 | historical exact online run |
| Task query embedding | 4.5374 | 3.9321 | 9.9237 | current bounded benchmark |
| Task cached Top5 retrieval | 20.8949 | 6.0686 | 101.8200 | current bounded SQLite-backed benchmark |
| Task total | 25.4323 | 11.0063 | 104.8052 | current bounded benchmark |
| Q8 score call | 32.4530 | 33.8382 | 54.5402 | historical exact online run |

The task retrieval benchmark uses direct SQLite vector reads and therefore has
a long-tail storage cost; it is not an optimized deployment benchmark. Q8 is
a score-call measurement, while embedding history construction is offline.
The candidate-scoring Pareto set is machine-recorded in `latency_pareto.json`.
Task-R alone is dominated. `P_NG+Task-R` gains only `.000230` Macro over the
Generic combination at roughly 12x the component-sum mean latency. Q8/Q8+F
remain materially more accurate; Q8+F is the accuracy endpoint.

## 7. Answers to the primary questions

**A - Calibration:** No. Generic beat Task after equal calibration in both
tracks.

**B - Recency:** Recency improved Task over plain Task in fixed-surface IME
ranking, but harmed Task candidate-only recovery scoring.

**C - Recovery:** Only narrowly. Task-R added a tiny benefit when combined
with `P_NG` on Personal-K5 discrimination, but its recovery-grid coefficient
was selected as zero and it did not improve overall end-to-end ranking.

**D - Complementarity:** No overall. The Task joint system was lower on both
tracks. It helped recoverable/Conflict subsets while harming Generic-covered
rows.

**E - Q8 ceiling:** No. Q8/Q8+F remained substantially more accurate for
candidate discrimination. Task was not a compelling latency substitute in the
measured SQLite-backed implementation.

**F - Initial versus Full:** The conclusion is consistent. Greater Initial
ambiguity makes absolute recovery harder and the Task-versus-Generic gap more
visible; Full has a much smaller, easier recovery pool, but still does not
produce an overall Task win.

## 8. Artifacts and hashes

Track inputs remained separate and were verified as follows:

| Track/input | SHA256 |
|---|---|
| Initial Train-Fit | `162f5c98daa86cc69947571e6d8f20fc401f0a82cdd3fd6e517eb7be2addbdb4` |
| Initial Train-Val | `d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4` |
| Initial Stage-1 | `54e60073daabb14bb7cf43136a335216888ea03c06d078a4eec56e5775a0cfbc` |
| Initial NGramRecency | `03858de42c41a26c4134d4b069b61ab2a5468c24cbd70a71958d600e448a97e1` |
| Initial Frequency/PV1 | `7fd8aa158d8cd50bced36b55610f8d932bc65e3aae1dbbd5bd65907ff1707ea7` |
| Initial Q8 scores | `3de538021fe0ac8a55de95b927e671a64cc0071d3abf915cf4b59f57f1ca561f` |
| Full Train-Fit | `547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6` |
| Full Train-Val | `d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220` |
| Full Generic predictions | `cf4ae382fa23e5ec1154bf28320d13ac1d6ca9600e9dcf8a6aa599600bc28eab` |
| Full Stage-1 | `e311261cb0c5ea252ce593fdaa43ff87826d19eef440f39e2efc67ddc4310b13` |
| Full Stage-2 | `d413d02650a759c0a759e1845212e68b2d948c1a1d20fc88abfd89ea7973bc64` |
| Full frozen predictions | `f3e902e5a9e7d25e62799b9abb719026c336381eacc42999d1e7edccf2731b22` |

Output root:

```text
results/personalisation/external_memory_next/posthoc_task_biencoder_calibration_v1/
```

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `evaluation/result.json` | 227,542 | `5b622c0288c482adee584857801cf13358db3a86ea7171735edc0a83b98d4eac` |
| `evaluation/grid.json` | 635,740 | `e8799c1ff765e05089db025de7ecfbb91fd6805877bb677bb60fdbd1663267f3` |
| `evaluation/selected_predictions.jsonl` | 37,492,212 | `85eac5f6533de3f439f289811e2524f42b8e8146cf5a4b5fdc6c04e13184386c` |
| `support/initial_support.jsonl` | 48,853,320 | `2c2bc7faddab4c032baf58d23a0767e6c31881bdd4c08f348cb865f43e4fced3` |
| `support/full_support.jsonl` | 31,321,549 | `564ec16bd623d722a42b17eec5ee05daffff1918049cce6eaf8bb4ef9902ef4a` |
| `q8_full/full_q8_scores.jsonl` | 1,763,192 | `99b9e7095cd6a6a457366aaddd185cf781f9e04dfccb60fdf939bdd0ff957ab6` |
| `latency/task_biencoder_latency.json` | 1,366 | `e430b029b8adda3a6b09b14d9e3aa2771d800512e9d650ff3662e550b4e73f24` |
| `latency/latency_pareto.json` | 4,424 | `f9df5dafcc6af85fcc15e6745dbacf9371de60474b5f9d3aed311a72e8bc49ec` |
| `latency/accuracy_latency_pareto.png` | 59,783 | `7d5c1efff99ab75f47f89b4d8c9165981021188970cd2f7c68b6a748557f5e42` |

Authored implementation SHA256 values:

| File | SHA256 |
|---|---|
| `src/personalisation/posthoc_context_calibration.py` | `fc8cd40b2499fefbb1f003f9c515f9edbc11dea9f7fef23b45ec96af05fc8a40` |
| `prepare_posthoc_context_support_v1.py` | `56013a7b84634e50c3212c7b1ba055bc9dba15f9b925f747c13eac704739e3bc` |
| `score_full_personal_k5_q8_v1.py` | `1bd6c955904c906e84f386debd7e87a9d51f9e162b6e4884e588148eed927cfa` |
| `evaluate_posthoc_task_biencoder_calibration_v1.py` | `472fa695df9e10b0285f2e3b34850cdac5e0f5a3f466a55ffe25d562de9d6bff` |
| `benchmark_posthoc_task_latency_v1.py` | `b26fac052a71d0d71df13b872690dcb807e3c2b9155c62ac91bb36d51aba0581` |
| `finalize_posthoc_latency_pareto_v1.py` | `cc67a08ea412c6b550d3d90a31c638ca94ecbe45b16ea95bb58783c6c7df962c` |
| `audit_posthoc_closeout_v1.py` | `fe121e3f1944f970c4c89c6cc7b90229db9f0a565f1ba66109c4c8f508b1e14c` |

Task checkpoint tree SHA256:
`f9b87af11fcff692ad7c25fb6330f44f9f23ffedb480af9aec36af0e7cd08a8e`.
All generated artifacts above are local-only and must not be staged.

## 9. Reproduction commands

Run from `C:\Users\chiar\Desktop\LBH\thesis-external-memory-next` in PowerShell:

```powershell
$py = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$initial = 'C:\Users\chiar\Desktop\LBH\thesis-initial-research\results\personalisation\initial_recovery_comparison_v1'
$full = 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation'
$root = '.\results\personalisation\external_memory_next\posthoc_task_biencoder_calibration_v1'
$task = '.\results\personalisation\external_memory_next\task_specific_biencoder_v1'
$checkpoint = "$task\training\final_refit\epoch_2"

# Initial support preflight; change --phase to score to materialize/refresh.
& $py -m experiments.external_memory_next.prepare_posthoc_context_support_v1 `
  --phase preflight --track initial `
  --fit "$initial\initial_train_fit_v1.jsonl" --val "$initial\initial_train_val_v1.jsonl" `
  --stage1 "$initial\recovery_ngram_context_fusion_v1\stage1_frozen.jsonl" `
  --existing-support "$initial\recovery_bge_ngram_context_fusion_v2\bge_recency_support.jsonl" `
  --frequency-predictions "$initial\frequency_pv1\predictions.jsonl" `
  --generic-seed-cache "$initial\recovery_bge_ngram_context_fusion_v2\bge_history_embedding_cache.sqlite3" `
  --task-seed-cache "$task\evaluation\task_vectors.sqlite3" --task-checkpoint $checkpoint `
  --generic-bge-model 'C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf' `
  --output-root "$root\support"

# Full support uses the same runner and --phase preflight/score.
& $py -m experiments.external_memory_next.prepare_posthoc_context_support_v1 `
  --phase preflight --track full `
  --fit "$full\context_comparison_v2\clean3_train_fit_v1.jsonl" `
  --val "$full\context_comparison_v2\clean3_train_val_v1.jsonl" `
  --stage1 "$full\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_stage1_features.jsonl" `
  --existing-support "$full\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_stage2_supports.jsonl" `
  --generic-seed-cache "$full\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\bge_context_cache.sqlite3" `
  --task-seed-cache "$task\evaluation\task_vectors.sqlite3" --task-checkpoint $checkpoint `
  --generic-bge-model 'C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf' `
  --output-root "$root\support"

# Resumable Full Q8 scoring.
& $py -m experiments.external_memory_next.score_full_personal_k5_q8_v1 `
  --val "$full\context_comparison_v2\clean3_train_val_v1.jsonl" `
  --features "$full\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_stage1_features.jsonl" `
  --generic "$full\context_comparison_v2\train_val_generic\predictions.jsonl" `
  --checkpoint 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat' `
  --output-root "$root\q8_full" --device cuda
```

The evaluation, bounded latency, and Pareto commands are long but fully
captured by each runner's `--help`; their exact accepted invocations are also
listed in `docs/REPRODUCIBILITY_INDEX.md`. Re-running the evaluator performs
only cached CPU grid arithmetic; it does not retrain or rescore the encoders.

## 10. Limitations

- This is controlled post-hoc Train-Val diagnosis, not untouched holdout
  evidence.
- No paired significance test was predeclared or run.
- Full K2+ recovery contains only 196 rows.
- Component-sum fusion latency is not a jointly timed deployment measurement.
- The task retrieval latency includes SQLite-backed vector reads and can be
  optimized independently of scientific ranking semantics.
