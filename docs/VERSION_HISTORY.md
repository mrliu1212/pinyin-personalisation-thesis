# Project Version History

This is a navigation index for frozen research and software checkpoints. Detailed protocols and results remain in their linked reports.

| Checkpoint | Purpose and main change | Commit or tag | Detailed record |
| --- | --- | --- | --- |
| Context Strengthening | Formally selected ctx64 on Dev; it improved retrieval-level contextual discrimination, but did not materially improve final M1 ranking over the previous Full-context M1 and remained below Frequency | `context-strengthening-complete-20260819` / `a9a9351c85fe7f40f17c5232e5f77b6c84e7b35c` | [Completed report](context_lab/CONTEXT_STRENGTHENING_REPORT_2026-08-18.md) |
| Context Diagnostic A | Exploratory three-author Full+Short/H5000 diagnostic, not a final six-author evaluation | `context-diagnostic-a-complete-20260818` / `54f05b76fa7b553baec62e260b1c74ed72a83e0f` | [Completed diagnostic report](context_lab/CONTEXT_DIAGNOSTIC_A_REPORT_2026-08-18.md) |
| Reranking Personalisation Matrix implementation | Resumable F/M1/M2 matrix over four frozen T1 conditions and H500/H5000/HFull, with shared Generic/BGE/M2 caches and a focused wrong-user control; final long run pending | branch `work/reranking-matrix`; `reranking-personalisation-matrix-implementation-v1` | [Method](research/reranking_personalisation_matrix.md) · [Pending report](reports/07_reranking_personalisation_matrix.md) |
| Personal Vocabulary H5000 completed result | Exact 6,000-anchor result; PV0 found 160/538 Generic-missing targets recoverable, PV1 reached `0.779` Macro-author Top-1 and `0.0661667` Missing@10, while PV2 context produced net Top-1 help `-5` | `personal-vocabulary-h5000-result-v1` | [Method](research/personal_vocabulary.md) · [Completed report](reports/06_personal_vocabulary_h5000.md) |
| Deep Author Dataset V1 | Initial six-author corpus and interaction build used as the Evaluation V2 development source | `deep-author-dataset-preparation-v1` / `d886b65` | [Dataset preparation](reports/01_dataset_preparation.md) |
| Deep Author Dataset V1.1 | Preserved targeted corpus-cleaning correction; not substituted into frozen T1 | `deep-author-dataset-preparation-v1.1` / `d871f1f` | [Dataset V1.1](reports/01b_dataset_preparation_v1_1.md) |
| Evaluation V2 Design | Frozen chronological six-author, 6,000-anchor, four-condition T1 protocol | `deep-author-evaluation-v2-design` / `b145f2d` | [Evaluation V2 design](reports/02_deep_author_evaluation_v2.md) |
| IME Simulator v0.1 | Full-Pinyin-only interactive CUDA simulator checkpoint | `ime-simulator-v0.1` / `6750bee` | `ime-simulator-v0.1:docs/tools/ime_simulator.md` |
| IME Simulator v0.2 | Unified mixed Full/Abbreviated Pinyin constraints, parsing, and one-search simulator integration | `ime-simulator-v0.2` / `ad69543` | `ime-simulator-v0.2:docs/research/mixed_pinyin_extension.md` |
| Evaluation V2 T1 Generic Baseline | Completed 24,000-condition Dataset V1 development baseline using semantic-equivalent KV-cache inference | `deep-author-evaluation-v2-t1`; implementation `8c608f1`, `5d270cd` | [T1 Generic PinyinGPT Baseline](reports/03_t1_generic_pinyingpt_baseline.md) |
| Personalisation Pilot A - Context-Aware Memory Implementation | Dev-only Full+Short chronological frequency/context-memory ranking, resumable caches, and manual runner | branch `work/personalisation-pilot-a`; `f335715`, `22eb1e0`; `personalisation-pilot-a-implementation-v1` | [Method](research/context_aware_personal_memory.md) · [Report](reports/04_personalisation_pilot_a_context_memory.md) |
| Personalisation Pilot A - Completed M1 H5000 result | Exact 6,000-anchor T1 Full+Short evaluation; G0 `0.7231666666666667`, Frequency `0.7718333333333334`, M1 `0.7675000000000001` Overall Macro-author Top-1 | implementation `personalisation-pilot-a-h5000-implementation-v1`; durable local result completed 2026-08-17 | [Method](research/context_aware_personal_memory.md) · [Completed report](reports/04_personalisation_pilot_a_context_memory.md) |
| Personalisation M2 H5000 implementation | Pretrained candidate-aware BGE reranker over unchanged BGE Stage-1 retrieval, frozen T1/M1 population and candidate surface; final result pending background run | branch `work/personalisation-pilot-a`; `personalisation-m2-h5000-implementation-v1` | [M2 method](research/candidate_aware_personal_memory_m2.md) · [Pending report](reports/05_personalisation_m2_h5000.md) |
| Personalisation M2 H5000 completed result | Exact 6,000-anchor result; M2 Overall Macro-author Top-1 `0.765`; unchanged candidate pool and 538 Missing@10; Test did not select parameters | `personalisation-m2-h5000-result-v1` | [M2 method](research/candidate_aware_personal_memory_m2.md) · [Completed report](reports/05_personalisation_m2_h5000.md) |
| Personal Vocabulary H5000 implementation | Bounded PV0 recoverability, PV1 frequency injection, and PV2 reused-BGE context injection over frozen T1/M1/M2; final result pending at this checkpoint | branch `work/personal-vocabulary`; `personal-vocabulary-h5000-implementation-v1` | [Method](research/personal_vocabulary.md) · [Report](reports/06_personal_vocabulary_h5000.md) |


## 2026-08-19 - EM-1 completed

Completed EM-1 External Memory Recovery + Frequency Fusion.

### Why

Earlier frequency and context-memory methods could only rerank the Frozen
Generic candidate surface. EM-1 tested whether strictly-prior personal
history could recover valid personal targets omitted from Generic Top10.

### What changed

Added exact-scored personal candidate recovery.

Personal-only candidates are taken from H5000 history, filtered against the
Frozen PinyinGPT constrained vocabulary, and scored using the same Frozen
PinyinGPT backend through fixed-candidate teacher-forced scoring.

A compatibility gate first confirmed that fixed-candidate scores match
cached Generic beam scores within the frozen 1e-4 engineering tolerance.

### Dev selection

Three-author Full+Short Dev selected:

- recovery K = 1
- frequency lambda = 4

Selection metric:
Macro-author Overall Top1.

The selection was frozen before formal Test evaluation.

### Frozen Test result

Three-author Test:

- G0 Top1: 77.600%
- F Top1: 81.067%
- R Top1: 77.700%
- R+F Top1: 81.033%

R+F improves candidate depth over F:

- Top3: 92.200% -> 92.700%
- MRR@10: 0.8685 -> 0.8708
- Missing@10: 4.400% -> 3.733%

But R+F does not improve overall Top1 over F:

- rescue: 10
- harm: 11
- net: -1

Recovery restored 23 backend-reachable Generic-missing Gold targets into
the unified pool; 21 reached Top10, 15 Top3, and 10 Top1.

### Interpretation

EM-1 validates recovery as a candidate-coverage mechanism, not as a
context-sensitive conflict resolver.

This motivates EM-2 and EM-3.

Detailed records:

- `docs/external_memory/EM1_DEV_SELECTION_2026-08-19.md`
- `docs/external_memory/EM1_TEST_RESULT_2026-08-19.md`
- `docs/external_memory/EM1_REPRODUCIBILITY_2026-08-19.md`

<!-- EM2-2026-08-19-INDEX -->
## 2026-08-19 - External Memory EM-2 progress

- validated Frozen PinyinGPT final-layer final-[SEP] hidden state as a task-native 768-d retrieval key;
- cached 11,475 required Dev hidden states with zero additional context truncation;
- showed stronger same-surface hidden-state retrieval than BGE Full and BGE ctx64;
- found that Hidden-M1 was almost identical to Original M1 end-to-end despite stronger retrieval;
- found that Hidden-M2 improved slightly over Original M2 but did not beat the M1 family;
- tested fixed `G + F + C_hidden` fusion; selected `lambda_F=0.5`, `lambda_C=4`, but obtained essentially no gain over Hidden-M1;
- preserved Test closure for the new EM-2 methods;
- next planned step: transparent prediction-visible Adaptive Fusion on Dev.


<!-- EM2-FINAL-CLOSE-2026-08-19 -->
## 2026-08-19 - EM-2 External Memory closed

- validated and froze the PinyinGPT final-layer final-[SEP] 768-d hidden representation;
- cached the required causal H5000 Dev hidden states with zero additional model-limit truncation;
- showed stronger same-surface hidden-state retrieval than BGE Full and BGE ctx64;
- found Hidden-M1 approximately tied with Original M1 end-to-end despite stronger retrieval;
- found Hidden-M2 slightly stronger than Original M2 but still below the M1 family;
- found fixed G+F+C approximately tied with Hidden-M1;
- found the registered count-aware Adaptive Fusion worse than Hidden-M1/Fixed GFC;
- retained the no-count adaptive variant as a diagnostic only;
- closed EM-2 without opening new Test results;
- handed the unresolved history-to-candidate-decision problem to EM-3.
