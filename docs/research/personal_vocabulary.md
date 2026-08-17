# Bounded Personal Vocabulary H5000

## Motivation

Completed `F-H5000`, `M1-H5000`, and `M2-H5000` are ranking-personalisation
methods. They can reorder only candidates already returned by frozen Generic
PinyinGPT, so their candidate set equals Generic Top-10. Consequently G0, F,
M1, and M2 all miss the same 538 Gold targets among the exact 6,000 frozen T1
Full+Short Test anchors.

Personal Vocabulary is candidate-set personalisation. It asks whether a target
omitted by Generic Top-10 can be supplied by strictly prior same-user history,
then inserted and ranked without retraining or rerunning PinyinGPT. This stage
separates three questions:

- PV0: is a Generic-missing target available in legal personal history?
- PV1: can personal frequency inject and rank it safely?
- PV2: does contextual similarity improve personal-only ranking beyond PV1?

## Frozen Population and History Boundary

The Test population is the exact frozen 6,000 T1 Full+Short anchors: 1,000 per
proxy user. Generic predictions are read from the completed T1 cache; Test
PinyinGPT inference is zero.

For every query, the canonical `HistoryIndex` applies the existing H5000 rule:

1. isolate the same user;
2. retain only strictly prior legal records;
3. take the 5,000 most recent records;
4. then filter exact segmented Pinyin.

Current Gold, future text, future interactions, and other-user vocabulary are
unavailable during construction and ranking. Dev selects parameters. Test Gold
is used only after final candidates and ranks exist.

## Bounded Personal Lexicon

The active lexicon is the set of distinct selected targets in the legal H5000
same-Pinyin bucket. It is not an unlimited lifetime vocabulary. Each entry
contains:

- user ID;
- segmented Pinyin;
- target surface;
- count inside visible legal history;
- first and last relevant interaction ID and chronological position;
- all contributing historical interaction IDs.

Entries are ordered by descending count and then surface text as a deterministic
non-recency tie-break. PV1/PV2 add at most Kpv entries, selected from `{1, 3,
5}`. H5000 and Kpv bound active vocabulary without introducing recency decay,
lifetime eviction, or RT4 temporal adaptation.

## Shared Per-query State

PV0, PV1, and PV2 use one prediction-visible state per query. It contains the
frozen Generic surface and scores, legal same-Pinyin history, Personal Lexicon,
personal-only membership, the existing F-ranked Generic rows, Generic boundary
score, personal frequency support, and cached BGE context support. It contains
no Gold field.

DEV and Test states are resumable JSONL caches:

- `results/personalisation/personal_vocabulary_h5000/cache/dev_states.jsonl`
- `results/personalisation/personal_vocabulary_h5000/cache/test_states.jsonl`

State identity fixes the bounded-H5000 lexicon and PV2 context versions.
Interrupted construction resumes by row ID and refuses duplicate or
manifest-incompatible rows.

## PV0 — Recoverability

PV0 is candidate availability, not a ranking system. For every Test query it
constructs `Generic Top10 ∪ distinct legal Personal Lexicon targets` without
Gold. Gold is inspected afterward.

Among the original 538 Generic-missing rows:

```text
Recoverable Missing = Gold absent from Generic Top10
                      and present in legal prior same-Pinyin lexicon

Unrecoverable Missing = Gold absent from Generic Top10
                        and absent from that lexicon

recoverability_rate = Recoverable Missing / 538
```

PV0 reports per-author recoverability, the historical occurrence-count
distribution for recoverable Gold targets, and mean, median, p90, and maximum
lexicon size. It never reports Top-1.

## Exact Frequency Reuse

The existing F implementation exposes its exact shared helper. For candidate
surface `S`:

```text
raw_frequency(c) = log(1 + count(c))
F(c) = raw_frequency(c) / max(raw_frequency(x) for x in S)
```

If all counts are zero, support is zero. Generic candidates are scored by the
existing `rank_frequency` function with frozen `lambda_frequency = 4.0`; their
scores are therefore byte-for-value equivalent to F-H5000. Personal-only
candidates call the same helper over the frequency-ordered personal-only
surface. No second frequency formula exists.

## PV1 — Frequency Candidate Injection

A personal-only candidate is a legal Personal Lexicon target not already in
Generic Top-10. The top Kpv personal-only targets are selected by lexicon
frequency order. Generic candidates retain their exact frozen F-H5000 scores.

Let `B` be the minimum normalized Generic score in the current Generic surface.
For a personal-only candidate:

```text
Score_PV1_personal_only(c) = B + lambda_pv * F(c)
```

Dev selects:

- `Kpv ∈ {1, 3, 5}`;
- `lambda_pv ∈ {0.5, 1, 2, 4}`.

Selection maximizes Macro-author Top-1 on the existing chronological
16,171-row Dev-tune partition. Ties choose lower lambda and then lower Kpv.
The primary Test comparison is PV1-H5000 versus F-H5000.

## PV2 — Frequency plus Context Candidate Injection

PV2 keeps PV1-selected Kpv and `lambda_pv` frozen. It does not use M2 or any
Cross-Encoder. For each personal-only candidate, it reuses the M1 BGE model,
normalized embeddings, positive cosine rule, and frozen Top-N = 5.

For candidate `c`, its own historical interactions are sorted by decreasing
cosine, then chronology and stable ID. Let:

```text
raw_context(c) = sum(max(cosine(current, history), 0)
                     for the best five histories whose target = c)

C(c) = raw_context(c) / sum(raw_context(x) over personal-only candidates)
```

If total positive support is zero, every `C(c)` is zero. PV2 uses:

```text
Score_PV2_personal_only(c)
    = B + lambda_pv * F(c) + lambda_ctx * C(c)
```

Only `lambda_ctx ∈ {0.5, 1, 2, 4}` is selected on Dev Macro-author Top-1. Ties
choose lower lambda. Kpv, `lambda_pv`, and M1 Top-N remain frozen. The primary
Test comparison is PV2-H5000 versus PV1-H5000.

## Merge and Deduplication

F-scored Generic candidates and scored personal-only candidates are combined
once by exact surface. Generic wins any defensive duplicate. Rows are sorted
by final score. Exact-score ties conservatively prefer Generic, then preserve
Generic rank or deterministic personal frequency order. The merged surface is
truncated once to final Top-10.

## Efficiency and Cache Reuse

- T1 Generic predictions are loaded once; Test inference remains zero.
- The existing `HistoryIndex` supplies legal H5000 same-Pinyin records.
- The existing F helper supplies all frequency components.
- The shared BGE SQLite cache remains profile-neutral and contains no PV label.
- One query state drives PV0/PV1/PV2 and every grid value.
- Grid search is arithmetic over cached states; embeddings are not recomputed
  per lambda.
- No M2 pair score, Cross-Encoder, neural training, or new PinyinGPT inference
  is required.

The preparation audit required 39,680 unique contexts and found 39,680 cache
hits, zero misses, and zero newly computed embeddings.

## Metrics and Paired Analysis

PV1 and PV2 report Macro-author Top-1, Top-3, MRR@10, Missing@10, and
MeanRank|Top10, plus per-author results. For the fixed original 538 missing
rows they report recovered-to-Top-10/Top-3/Top-1, missing recovery rate, and
recoverable recovery rate relative to PV0.

PV1 reports paired F→PV1 helped, harmed, unchanged-correct, and
unchanged-wrong counts and `net_help = helped - harmed`. It also reports harm
among 5,462 originally covered rows. PV2 reports the corresponding PV1→PV2
context transitions and `net_context_help`.

## Information Boundary and Reproducibility

PV0 Test recoverability is descriptive and cannot select PV1/PV2 parameters.
PV1 selects Kpv and `lambda_pv` on Dev only. PV2 freezes those values and
selects only `lambda_ctx` on Dev. The runner records zero Test rows during
selection and validates T1, M1, and M2 artifact hashes before and after final
evaluation.

Phases are `prepare`, `pv0`, `dev-states`, `tune`, `evaluate`, `smoke`, and
`all`. From the worktree:

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
& C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe `
  -m experiments.personal_vocabulary_h5000 `
  --phase all `
  --dataset-root C:\Users\chiar\Desktop\LBH\thesis-deep-author\.build\dataset-v1-reconstruction `
  --pinyingpt-model C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat `
  --embedding-model C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf `
  --t1-predictions C:\Users\chiar\Desktop\LBH\thesis-deep-author\results\evaluation\deep_author_v2\t1\predictions.jsonl
```

Durable outputs are under
`results/personalisation/personal_vocabulary_h5000/`. The shared BGE cache is
`results/personalisation/pilot_a_context_memory/cache/embedding_cache.sqlite3`.

## Limitations and Future Work

The method uses a fixed H5000 window, frequency-biased Kpv pruning, proxy users,
reconstructed Pinyin, and an approximate Generic boundary for candidates that
have no PinyinGPT score. Cold-start and genuinely unseen personal vocabulary
remain unrecoverable. Erroneous historical selections may be injected. Generic
BGE similarity may still be too coarse, and temporal drift is not modelled.

The provenance-bearing lexicon can later support explanations, deletion,
correction, and user control, but no Transparency or Control interface is part
of this stage. Recency, temporal adaptation, lifetime eviction, and RT4 remain
future research rather than hidden additions to PV0/PV1/PV2.
