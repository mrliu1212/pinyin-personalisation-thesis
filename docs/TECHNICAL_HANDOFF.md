# Technical Handoff: Deep Author Pinyin Personalisation

> **Historical scope notice (2026-08-19):** this handoff is a frozen operator
> snapshot of the old `work/reranking-matrix` worktree. It is not the current
> Context Lab / External Memory handoff. Use [FILE_INDEX.md](FILE_INDEX.md) and
> [REPRODUCIBILITY_INDEX.md](REPRODUCIBILITY_INDEX.md) for current repository
> navigation and checkpoint reproduction.

This is the operator/developer handoff for continuing the thesis repository without Codex. It records the implementation and local Windows environment as inspected on **2026-08-18**. Research intent remains governed by [`RESEARCH_TARGETS.md`](../RESEARCH_TARGETS.md); the frozen matrix protocol is [`docs/research/reranking_personalisation_matrix.md`](research/reranking_personalisation_matrix.md).

## Cheat sheet

| Item | Current value |
| --- | --- |
| Repository | `C:\Users\chiar\Desktop\LBH\thesis-personalisation` |
| Branch / inspected HEAD | `work/reranking-matrix` / `4c95c54ef8c95e9b08c37cc17eeef154462a908b` |
| Python | `C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe` |
| Matrix result root | `results\personalisation\reranking_matrix` |
| Frozen Test condition manifest | `results\evaluation\deep_author_v2\design\t1_condition_manifest.jsonl` |
| Frozen Test predictions | `C:\Users\chiar\Desktop\LBH\thesis-deep-author\results\evaluation\deep_author_v2\t1\predictions.jsonl` |
| Matrix manifest | `results\personalisation\reranking_matrix\matrix_manifest.json` |
| Completion marker | `results\personalisation\reranking_matrix\COMPLETE.json` |
| Live logs | `matrix_resume_stdout.log`, `matrix_resume_stderr.log` under the matrix root |
| Current report | `docs\reports\07_reranking_personalisation_matrix.md` |

```powershell
Set-Location C:\Users\chiar\Desktop\LBH\thesis-personalisation
git status --short --branch
& C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe -m pytest tests -q
Get-Content results\personalisation\reranking_matrix\matrix_resume_stdout.log -Tail 30 -Wait
Get-Process -Id 16396 -ErrorAction SilentlyContinue
```

The PID is a snapshot, not a stable identifier. Re-check the process before acting.

## 1. Git and version state

At inspection, `work/reranking-matrix` was 18 commits ahead of local `main` and 19 ahead of `origin/main`; both are ancestors of the matrix branch. Local `main` points to `c594e2a` (LiveChat generic baseline checkpoint), while `origin/main` points to the older Windows Phase 4F compatibility commit `d23e9a7`. Do not assume local `main` and GitHub `main` are synchronized.

Recent relevant commits, newest first:

- `4c95c54` - release inactive PyTorch CUDA blocks after owned PinyinGPT Dev inference, before llama.cpp/BGE.
- `2318dbb` - bucket variable-length Dev Generic requests by prompt/target token shape; preserve frozen row order and crash-safe journals.
- `4e2c6ea` - implement the 36-cell resumable reranking matrix.
- `a83757e` / `483aa73` - Personal Vocabulary H5000 result / implementation.
- `fb7abca` / `dd0753b` - M2 H5000 result / implementation.
- `8e26cdb` and earlier Pilot A commits - M1/F H5000 and Dev pilot.
- `14d584a`, `5d270cd`, `8c608f1`, `b145f2d` - T1 result, resumable inference, backend integration, Evaluation V2 design.

Relevant tags are `reranking-personalisation-matrix-implementation-v1`, `personal-vocabulary-h5000-result-v1`, `personalisation-m2-h5000-result-v1`, `personalisation-pilot-a-h5000-implementation-v1`, `deep-author-evaluation-v2-t1`, `deep-author-evaluation-v2-design`, and the Dataset V1/V1.1 tags. Use `git show <tag>` to resolve annotated tags to commits.

The tracked tree was clean before this handoff; large local result/cache directories were untracked. This document and `docs/technical_handoff_manifest.json` are the only intended new tracked modifications. Never use `git add .` here: it risks staging large local results.

Other worktrees/branches:

- `C:\Users\chiar\Desktop\LBH\thesis` - `work/livechat-multitoken-abbreviation-audit`.
- `C:\Users\chiar\Desktop\LBH\thesis-deep-author` - `work/deep-author-evaluation-v2`; owns the frozen T1 prediction cache used by the matrix.
- `C:\Users\chiar\Desktop\LBH\thesis-ime-simulator` - `work/ime-simulator`.
- Historical personalisation branches remain navigable by tags and `work/personalisation-pilot-a`, `work/personal-vocabulary`.

Safe Git commands:

```powershell
git status --short --branch
git diff --stat
git diff --check
git log --oneline --decorate -15
git branch -avv
git tag --list --sort=-creatordate

# Prefer a separate worktree while the matrix worker uses this worktree.
git worktree add ..\thesis-context-128 -b work/context-128 work/reranking-matrix

# Deliberately stage only named files.
git add -- src\path.py tests\test_path.py docs\report.md
git diff --cached
git commit -m "descriptive message"
git tag -a experiment-name-v1 -m "Experiment Name v1"

# Inspect an old version without disturbing this worktree.
git worktree add --detach ..\thesis-old experiment-tag
```

Do not switch this worktree while its worker is alive. A new experimental branch should start from the desired frozen commit, not automatically from `main`.

## 2. Repository map

### Dataset and Evaluation V2

- `experiments/prepare_deep_author_dataset.py` - CLI for dataset acquisition/preparation.
- `src/datasets/deep_author/pipeline.py` - provenance, cleaning, tokenization, Pinyin conversion, Dataset V1.1 interactions, and the `CONTEXT_CHARACTER_LIMIT`.
- `config/deep_author/run_config.yaml` - Dataset V1.1 frozen preparation parameters.
- `experiments/deep_author_evaluation_v2.py` - Evaluation V2 design/T1 CLI.
- `src/evaluation/deep_author_v2.py` - frozen anchors, Full/Initial and Short/Multi3 conditions, chronological split, T1 inference, validation, and generic metrics.
- `config/deep_author/evaluation_v2.yaml` - authoritative Dataset V1 source/hash, six authors, split/sampling rules, model revisions, beam 16, Top-10.
- `results/evaluation/deep_author_v2/design/work_split_manifest.csv` - per-work chronological History/Dev/Test assignment.
- `results/evaluation/deep_author_v2/design/t1_anchor_manifest.csv` - 6,000 frozen Test anchors.
- `results/evaluation/deep_author_v2/design/t1_condition_manifest.jsonl` - 24,000 paired Test conditions (four per anchor), normalized SHA-256 `45b9caf...0d39`.
- `C:\...\thesis-deep-author\results\evaluation\deep_author_v2\t1\predictions.jsonl` - immutable 24,000 Generic Test predictions, SHA-256 `764db398...ee4e2`.

### Generic PinyinGPT

- `src/reference_backend_pinyingpt/backend.py` - checkpoint/tokenizer loading, Pinyin segmentation, Concat prompt, fit-to-position truncation, single/batched beam decoding, fixed-candidate scoring.
- `experiments/exp_pinyingpt_reference.py` - small reference-backend experiment/smoke CLI.
- `docs/third_party/pinyingpt.md` - checkpoint/paper/code provenance and reproduction audit.
- `src/evaluation/deep_author_v2.py:T1Runner` - durable frozen T1 cache and metrics.
- `src/personalisation/pilot_a.py:PilotRunner.generic` - Dev-only Generic generation for Pilot A.
- `src/personalisation/reranking_matrix.py:ensure_dev_generic` - condition-specific Dev Generic cache/journal, length-compatible batching, row-order materialization, CUDA transition cleanup.

### Personalisation

- `src/personalisation/context_memory.py` - shared F and M1 formulas, exact visible-history logic, subset labels, rank metrics, and macro-author aggregation.
- `src/personalisation/pilot_a.py` - BGE GGUF embedder/cache, budget-before-Pinyin `HistoryIndex`, Pilot A manifests and orchestration.
- `src/personalisation/h5000.py` - frozen Full+Short H5000 M1/F runner and T1 hash validation.
- `src/personalisation/candidate_memory_m2.py` - M2 pair identity/template, balanced recent-context truncation, Cross-Encoder runtime, pair cache, and final M2 support formula.
- `src/personalisation/m2_h5000.py` - M2 grids, Stage-1 BGE retrieval, pair scoring and completed H5000 runner.
- `src/personalisation/pv_h5000.py` - separate completed Personal Vocabulary experiment; not part of the matrix candidate fusion.

### Matrix

- `experiments/reranking_personalisation_matrix.py` - CLI phases: `audit`, `smoke`, `run`, `finalize`.
- `src/personalisation/reranking_matrix.py` - 36-cell runner, manifests, shared caches, Dev selection, evaluation, diagnostics, resume and finalization.
- `results/personalisation/reranking_matrix/matrix_manifest.json` - authoritative cell state.
- `.../cells/<condition>/<budget>/<method>/` - row predictions plus `result.json` for each completed new cell.
- `.../selections/<condition>/<budget>/` - Dev grid CSVs and selected parameters.
- `.../COMPLETE.json` - authoritative aggregate completion marker; `incomplete` until all cells finish.
- `matrix_resume_stdout.log` / `matrix_resume_stderr.log` - current detached worker logs. Detached launching is operational PowerShell, not a repository service.

### Documentation record

- `RESEARCH_TARGETS.md` - current thesis direction and non-goals.
- `docs/VERSION_HISTORY.md` - chronological checkpoint index.
- `docs/research/*.md` - method/protocol definitions.
- `docs/reports/01...07_*.md` - chronological dataset, evaluation, baseline, and experiment reports.
- `README.md` - manual commands and operator workflow.

## 3. High-risk functions and classes

Line numbers are approximate at commit `4c95c54`.

| Location | Symbol | Role and risks |
| --- | --- | --- |
| `pipeline.py:35, make_interactions:398` | `CONTEXT_CHARACTER_LIMIT`, `make_interactions` | Dataset V1.1 preceding Han-context storage. Changing it changes dataset identities and invalidates downstream data. |
| `deep_author_v2.py:68` | `valid_anchors_for_work` | Builds Evaluation V2 anchors, including 512-character preceding suffix and Short/Multi3 targets. Changes invalidate the frozen condition manifest and T1. |
| `deep_author_v2.py:176` | `conditions_for_anchor` | Maps each anchor to Full/Initial × Short/Multi3 and stable condition IDs. Never alter for a context-only follow-up. |
| `backend.py:172` | `PinyinGPTConcatBackend._prompt` | `[CLS] + context + [SEP] + Pinyin tokens + [SEP]` and aligned positions. Any alteration changes Generic semantics. |
| `backend.py:200` | `effective_context` | Leading token window helper. It is currently not called by production T1/Pilot/matrix inference. |
| `backend.py:212` | `truncate_context_for_generation` | Keeps the most recent context suffix fitting full Concat generation. Called by T1 and Dev Generic paths. |
| `backend.py:314` | `generate_batch` | Shared-forward exact constrained beam search; requires equal prompt and target lengths per call. |
| `pilot_a.py:571` | `PilotRunner.generic` | Dev Generic inference and cache. Uses truncation, shape buckets, beam 16, Top-10. |
| `context_memory.py:31` | `visible_same_pinyin_history` | Reference same-user, exact-Pinyin, strictly-prior filter used by formulas/tests. |
| `pilot_a.py:435` | `HistoryIndex` | Optimized legal history. Applies H500/H5000 to the latest prior same-user records before Pinyin matching. |
| `context_memory.py:104` | `rank_frequency` | F: Generic z-score + lambda × normalized log-frequency support. |
| `pilot_a.py:278` | `BGEContextEmbedder` | Mean-pooled, normalized 512-D BGE embedding with most-recent 510-token truncation. |
| `pilot_a.py:346` | `EmbeddingCache` | Provenance-checked SQLite vectors; key includes model/preprocess/context. |
| `context_memory.py:156,186` | `retrieve_memory`, `rank_from_retrieved` | M1 cosine retrieval and positive normalized target support. |
| `candidate_memory_m2.py:74` | `CandidateAwareTemplate.prepare` | M2 serialization and balanced current/history recent-suffix truncation to 512 total tokens. |
| `candidate_memory_m2.py:161` | `PairScoreCache` | Raw Cross-Encoder logit cache; key includes every semantic pair input and model/template provenance. |
| `candidate_memory_m2.py:273` | `BGEReranker` | Local Transformers FP16 CUDA Cross-Encoder runtime. |
| `candidate_memory_m2.py:385` | `rank_m2` | Sigmoid raw logits, aggregate by historical target, normalize, combine with Generic z-score. |
| `reranking_matrix.py:385` | `ensure_dev_generic` | Resume-safe Dev Generic inference. Primary JSONL is frozen-order; `.partial.jsonl` is the crash journal. |
| `reranking_matrix.py:488` | `_ensure_embedding_values` | Reuses BGE entries and computes only missing exact contexts. |
| `reranking_matrix.py:591` | `_prepare_rows` | Constructs query, legal visible history, Generic candidate surface, BGE retrieval, and diagnostic flags once per group. |
| `reranking_matrix.py:606` | `_tune` | Dev-only F/M1/M2 grid selection by Macro-author Top-1 with deterministic lower-complexity ties. |
| `reranking_matrix.py:649` | `_evaluate_method` | Test ranking and row-level outputs; asserts candidate pool and Missing@10 invariants. |
| `reranking_matrix.py:699` | `run_cell_group` | State transitions and group transaction. Incorrect edits can recompute frozen cells or leak Test. |
| `reranking_matrix.py:800,855` | `finalize`, `all` | Aggregate outputs, wrong-user control, hashes, completion marker, deterministic group order and failure continuation. |

## 4. Current context logic

There are two related dataset builders. Dataset V1.1 `pipeline.make_interactions` selects the last 512 **Han positions** before a token and joins them; non-Han text acts as a hard boundary in the V1.1 representation. Frozen Evaluation V2, however, deliberately uses verified Dataset V1 and `deep_author_v2.valid_anchors_for_work`: `context = text[max(0, start - 512):start]`. This is the immediate preceding character suffix from the cleaned work and can contain punctuation, Latin text and newlines.

The frozen manifest has 24,000 rows, context mean 469.8017 characters, median 512, and 19,872/24,000 (82.80%) exactly 512 characters.

`effective_context(token_limit=512)` keeps a **leading** tokenizer window, but current production paths do not call it. T1 (`deep_author_v2.py:476`), Pilot Dev Generic (`pilot_a.py:602`) and matrix Dev Generic (`reranking_matrix.py:408`) call `truncate_context_for_generation`, which keeps the **most recent suffix** only if required.

The checkpoint has `n_positions = n_ctx = 1024`. For `k` segmented Pinyin elements, the final decoding forward must fit context plus Pinyin and target positions. The implementation allows `1024 - (2 + 2k)` context tokens. This accounts for the maximum forward sequence: prompt specials/Pinyin plus up to `k-1` previously generated characters; the final emitted character is not fed back. The prompt itself contains `[CLS]`, two `[SEP]` tokens and `k` Pinyin tokens.

In frozen Test, none of 24,000 rows was truncated: mean stored and used tokenizer length are both 425.5365 and the minimum used/original ratio is 1.0. Thus current Generic inference receives all stored context in every frozen Test row.

M1 embeds current and historical context strings independently. BGE keeps the most recent 510 content tokens if a context exceeds its 512-token model input. M2 serializes current context + Pinyin + candidate as one side and historical context + selected target as the other; mandatory fields are preserved, remaining tokens are split approximately half/half, unused capacity transfers across sides, and both context suffixes are safety-trimmed from the oldest end until the serialized pair is ≤512 tokens.

For a future 128-total-position policy, decide which layer owns the policy. Changing only PinyinGPT truncation does not change BGE/M2 context; changing stored manifest context changes Generic, BGE and M2 identities and requires a new experiment/cache namespace.

## 5. Data formats and schemas

Real examples are in the paths below. IDs and hashes are frozen; scores, ranks, flags and runtime fields are derived.

### Deep Author interaction

Path: `C:\...\dataset-v1-reconstruction\data\processed\deep_author\interactions_t1_ready.jsonl`.

```json
{"interaction_id":"da-int-...","author_id":"...","author_name":"...","work_id":"da-work-...","context":"preceding text","gold":"目标","full_pinyin":"mu biao","initial_pinyin":"m b","composition_type":"short|multi","token_count":1,"source_position_start":123,"source_position_end":125,"source_creation_date":"...","source_hash":"..."}
```

### T1 condition row

Path: `results/evaluation/deep_author_v2/design/t1_condition_manifest.jsonl`.

```json
{"condition_id":"da-v2-condition-...","anchor_id":"da-v2-anchor-...","author":"...","work_id":"da-work-...","condition":"full_short","target_type":"short","pinyin_type":"full","context":"...","pinyin_input":"ci","gold":"此","gold_char_length":1,"source_position_start":123,"source_hash":"...","cleaned_text_hash":"..."}
```

`condition_id`, `anchor_id`, work/source IDs/hashes and population are frozen. `condition`, Pinyin input and Gold are deterministic derivatives of the anchor.

### Generic prediction

Path: frozen Test `...\thesis-deep-author\results\evaluation\deep_author_v2\t1\predictions.jsonl`; Dev equivalents under `reranking_matrix/cache/dev_generic/`.

Important fields: condition/anchor IDs, context/Pinyin/Gold, `model_used_context`, original/used context tokens, `context_truncated`, `top10_candidates[{text,rank,log_probability,mean_log_probability,compatible}]`, `gold_rank`, correctness metrics, beam/top-k, checkpoint/code/runtime provenance.

### F/M1/M2 row

Path: `reranking_matrix/cells/<condition>/<budget>/<method>/predictions.jsonl`.

```json
{"condition_id":"...","anchor_id":"...","author":"...","work_id":"...","condition":"full_short","history_budget":"H500","method":"F|M1|M2","gold":"...","gold_rank":1,"history_available":true,"visible_history_count":3,"distinct_historical_targets":2,"ambiguous":true,"frequency_winner":"...","frequency_winner_tied":false,"conflict":true,"selected_hyperparameters":{},"candidates":[],"wrong_user_control":false}
```

Candidate rows always retain `generic_rank`, `generic_score`, `normalized_generic_score`, `final_score`, `rank`. F adds `frequency_count`/`personal_score`; M1 uses contextual `personal_score`; M2 exposes `m2_support`.

### Matrix cell and result

`matrix_manifest.json` cells contain `condition`, `history_budget`, `method`, `state`, `output_path`, `existing_artifact_path`, selected parameters and error traceback. States are `pending`, `running`, `failed`, `complete`, `reused_complete`.

Each `result.json` records status, 6,000 rows, per-author row counts, selected parameters, metrics by Overall/History Available/Ambiguous/Conflict, subset sizes, candidate-pool invariant, Generic reuse count, Test-inference count and pair-cache statistics.

### SQLite caches

BGE table: `embeddings(cache_key PRIMARY KEY, context_sha256, vector BLOB)`. Vector is normalized little-endian float32 × 512.

M2 table: `pair_scores(cache_key PRIMARY KEY, current_sha256, historical_sha256, candidate_sha256, raw_score, input_tokens, current_context_truncated, historical_context_truncated)`.

## 6. Frozen F, M1 and M2 definitions

All methods start from the same frozen Generic Top-10 surface. Per-query Generic log probabilities are population-z-normalized. Personalisation may reorder but never inject/remove candidates in this matrix.

**Legal history:** same user; chronological position strictly less than the query; take the latest H500/H5000/all records at the author level; then retain exact tuple equality of segmented Pinyin. Dev selection can see earlier History plus chronologically earlier Dev rows. Test sees History only, never Dev/Test selections.

**F:** count visible historical targets. Candidate support is `log(1+count)` divided by the maximum candidate raw support. Final score is `z(Generic) + lambda_frequency × support`; generic rank breaks score ties. Grid: lambda `{0, .25, .5, 1, 2, 4}`.

**M1:** embed current/historical contexts with pinned `bge-small-zh-v1.5` GGUF; L2-normalized mean pooling. Sort visible same-Pinyin history by descending cosine similarity, then chronology and ID. Weight is `max(cosine,0)`. Top-N evidence weights are normalized by total weight and summed by historical target. Final score is `z(Generic) + lambda_memory × target_support`. Grid: Top-N `{1,3,5,10,20}`, lambda `{0,.25,.5,1,2,4}`.

**M2:** Stage 1 is unchanged M1 BGE retrieval. For K `{10,20}`, create one semantic pair per retrieved history record using current ID/context/Pinyin, historical ID/context/target, and that historical target as the candidate to score. `BAAI/bge-reranker-base` produces a raw sequence-classification logit. Apply sigmoid, sum support by historical target, divide by total retrieved sigmoid support, then `z(Generic) + lambda_m2 × support`. Lambda grid `{.5,1,2,4}`. Pair scores deduplicate across budgets/cells whenever the full semantic pair and provenance match.

Frozen semantics are the eligibility/order, candidate surface, formulas, grids, Dev-only selection and Test population. Batching, SQLite transactions, logs and CUDA cleanup are implementation details.

## 7. Cache architecture

### Frozen Test Generic

- Path: `C:\...\thesis-deep-author\results\evaluation\deep_author_v2\t1\predictions.jsonl`.
- Format: canonical JSONL; 24,000 rows; SHA-256 `764db398...ee4e2`.
- Identity: frozen condition manifest, checkpoint/code revisions, context/Pinyin, beam 16, Top-10.
- Never regenerate for the matrix. A context/model/prompt change requires a new result namespace.

### Matrix Dev Generic

- Paths: `reranking_matrix/cache/dev_generic/<condition>.jsonl` and `<condition>.partial.jsonl`.
- Primary is exact frozen Dev-row order; partial is a crash journal. Loader deduplicates identical rows.
- Compatible across resumes only for unchanged row IDs/context/Pinyin/model/decoding. Do not copy it into a changed-context experiment merely because row IDs match.

### BGE

- Path: `results/personalisation/pilot_a_context_memory/cache/embedding_cache.sqlite3`.
- Key: SHA-256 of model hash + preprocessing version + pooling + normalization + exact context text.
- Safe across budgets/methods/conditions when context bytes are identical. Changed context creates a new key automatically. Model/runtime/provenance metadata mismatch blocks opening.
- Inspected count was dynamic (251,234 during the running matrix); query it rather than trusting this snapshot.

### M2 pair scores

- Path: `results/personalisation/m2_h5000/cache/pair_scores.sqlite3`.
- Key includes current ID/context/Pinyin, historical ID/context/target, candidate, model/revision/tokenizer hashes, template/truncation versions and max length.
- Safe across budgets/cells only for identical semantic pairs. Full/Initial or Short/Multi3 typically change IDs/Pinyin/target and therefore keys. A changed context/model/template/max length creates a different key or provenance rejection.
- Inspected count was 814,959 and growing.

### Other durable artifacts

- `required_contexts_<condition>.json` caches exact context inventories; do not reuse in a changed-context experiment.
- matrix manifests, selections and completed cell outputs are resume state, not disposable logs.
- `generic_predictions.jsonl` under Pilot A is the older Full+Short Dev Generic cache.

Inspection commands:

```powershell
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
& $python -c "import sqlite3; p=r'results\personalisation\pilot_a_context_memory\cache\embedding_cache.sqlite3'; c=sqlite3.connect(p); print(c.execute('select count(*) from embeddings').fetchone()[0]); print(dict(c.execute('select key,value from metadata'))); c.close()"
& $python -c "import sqlite3; p=r'results\personalisation\m2_h5000\cache\pair_scores.sqlite3'; c=sqlite3.connect(p); print(c.execute('select count(*) from pair_scores').fetchone()[0]); print(dict(c.execute('select key,value from metadata'))); c.close()"
(Get-Content results\personalisation\reranking_matrix\cache\dev_generic\initial_multi3.jsonl | Measure-Object -Line).Lines
```

Never casually delete these caches: rebuilding them costs GPU time. Back up the SQLite database together with `-wal`/`-shm` files only after stopping the writer or using SQLite's backup API.

## 8. Models and environment

- Python 3.12.13, executable `C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe`.
- PyTorch `2.11.0+cu128`; CUDA available, toolkit/runtime 12.8.
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB).
- Transformers `4.57.6`; llama-cpp-python `0.3.16`; NumPy `2.5.2`.

PinyinGPT2-Concat is `aihijo/transformers4ime-pinyingpt-concat`, checkpoint revision `76dd20...263`, official code `8f1573...6f1`, local `C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat`. It uses `BertTokenizer`, `GPT2LMHeadModel`, 12 layers, 768 hidden, 12 heads, `n_positions=n_ctx=1024`, oracle segmented Pinyin, CUDA eval, beam 16/Top-10.

BGE is `bge-small-zh-v1.5-q8_0.gguf` at `C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\`, SHA-256 `5a88d266...12039`, 512 dimensions. llama.cpp parameters: `n_ctx=512`, `n_gpu_layers=-1`, `embedding=True`, mean pooling; output is L2-normalized.

M2 is `BAAI/bge-reranker-base`, revision `2cfc18...a70`, local `C:\Users\chiar\Desktop\LBH\thesis\.build\bge-reranker-base`, Transformers sequence classification, FP16 CUDA, batch 32, maximum serialized pair length 512. Model hash `ced967...3fbd`; tokenizer hash `9eb652...89e`.

```powershell
& C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe -c "import sys,torch,transformers,llama_cpp; print(sys.version); print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_name(0)); print(transformers.__version__,llama_cpp.__version__)"
nvidia-smi
Get-FileHash C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf -Algorithm SHA256
```

## 9. Manual command cookbook

```powershell
Set-Location C:\Users\chiar\Desktop\LBH\thesis-personalisation
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'

# Tests and syntax
& $python -m pytest tests -q
& $python -m pytest tests\test_reranking_matrix.py -q
& $python -m py_compile src\personalisation\reranking_matrix.py src\personalisation\context_memory.py src\personalisation\candidate_memory_m2.py experiments\reranking_personalisation_matrix.py
git diff --check

# Git and data/result inspection
git status --short --branch
git diff --stat
(Get-Content results\evaluation\deep_author_v2\design\t1_condition_manifest.jsonl | Measure-Object -Line).Lines
Get-Content results\evaluation\deep_author_v2\design\t1_condition_manifest.jsonl -First 1
Get-Content -Raw results\personalisation\reranking_matrix\matrix_manifest.json
Get-Content -Raw results\personalisation\reranking_matrix\COMPLETE.json
Get-Content -Raw results\personalisation\pilot_a_context_memory\h5000\metrics_summary.json

# Process/log monitoring
Get-Process -Id 16396 -ErrorAction SilentlyContinue
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,StartTime,CPU,Path
Get-Content results\personalisation\reranking_matrix\matrix_resume_stdout.log -Tail 30 -Wait
Get-Content results\personalisation\reranking_matrix\matrix_resume_stderr.log -Tail 50

# Stop only after confirming this exact PID is the intended worker.
Stop-Process -Id 16396
```

Matrix audit/resume parameters:

```powershell
$common = @(
  '--dataset-root','C:\Users\chiar\Desktop\LBH\thesis-deep-author\.build\dataset-v1-reconstruction',
  '--pinyingpt-model','C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat',
  '--embedding-model','C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models\bge-small-zh-v1.5-q8_0.gguf',
  '--reranker-model','C:\Users\chiar\Desktop\LBH\thesis\.build\bge-reranker-base',
  '--t1-predictions','C:\Users\chiar\Desktop\LBH\thesis-deep-author\results\evaluation\deep_author_v2\t1\predictions.jsonl'
)
& $python -m experiments.reranking_personalisation_matrix --phase audit @common
& $python -u -m experiments.reranking_personalisation_matrix --phase run @common
```

`run` is resume. There is no cell-specific CLI and no metrics-only matrix phase; do not invent one. `finalize` only succeeds when all cells are complete/reused.

For detached execution, use `Start-Process` as documented in `docs/research/reranking_personalisation_matrix.md`, with `CUDA_PATH` set and the exact log paths above. `RedirectStandardOutput` overwrites an existing file; archive old logs first or launch a PowerShell wrapper with deliberate append redirection.

## 10. Long-context matrix status snapshot

At **2026-08-18 12:08 CEST**, worker PID 16396 (venv launcher 20444) was alive. `COMPLETE.json` was `incomplete`; stderr contained llama.cpp informational embedding messages but zero `Traceback` entries.

- complete: 12 - Full+Short H500/HFull F/M1/M2 and Initial+Short H500/H5000 F/M1/M2.
- reused_complete: 3 - Full+Short H5000 F/M1/M2.
- running: 3 - Initial+Short HFull F/M1/M2.
- failed: 18 - Full+Multi3 and Initial+Multi3 groups; these are historical BGE-load failure states awaiting deterministic retry, not current active errors.

Latest inspected stdout had finished `test-initial_short-H5000` M2 pair scoring (43,514/43,514) and entered Initial+Short HFull with Dev Generic 16,171/16,171 reused. Re-read the manifest/log for current truth.

Launch command is the `--phase run` command above, detached with stdout/stderr at the resume paths. Frozen Full+Short completed/reused cells are skipped because `run_cell_group` excludes `complete` and returns immediately for `REUSED_CELL`.

## 11. Known incidents and fixes

1. **Windows Phase 4F infrastructure:** Unix-only telemetry, offline model reuse, Windows librime/adapter paths and platform-specific smoke results were handled before this branch; tags `phase-04f-windows-compat` and `phase-04f` preserve that history.
2. **Dev Generic variable-length batch failure:** adjacent Dev rows had different prompt/target token lengths. Commit `2318dbb` groups by both lengths, batches compatible rows, journals results and restores frozen row order. Tests cover mixed lengths, resume and no recomputation.
3. **PinyinGPT->BGE transition:** after full Dev Generic CUDA work, llama.cpp reported a generic model-load failure although the pinned GGUF was healthy. Standalone CUDA loading and embedding succeeded. Commit `4c95c54` drops the owned PinyinGPT backend and calls garbage collection plus `torch.cuda.empty_cache()` before llama.cpp. The bounded real cache path and detached resume passed; BGE progressed beyond the old failure.
4. **`LlamaModel.sampler` cleanup AttributeError:** this occurred after failed constructor cleanup in llama-cpp-python 0.3.16. It was secondary noise, not evidence of GGUF damage. A current traceback or failed load still requires diagnosis.
5. **llama.cpp `init: embeddings required...` messages:** repeated informational stderr output during BGE embedding, not a Python exception.
6. **SQLite under sandbox:** read-only tooling may fail to open a live WAL database if it cannot create/access shared state. Manual local PowerShell outside a sandbox does not have this Codex restriction.

## 12. Metrics and analysis

Generic T1 metrics are in `deep_author_v2.metric_values/aggregate_metrics`; personalised metrics are in `context_memory.metric_values/macro_author_metrics`.

- Top1: fraction with Gold rank 1.
- Top3: fraction with non-missing rank ≤3.
- MRR@10: mean `1/rank`; missing contributes zero.
- Missing@10: fraction with Gold absent from the fixed Top-10.
- MeanRank|Top10: mean rank only among found rows.
- Macro-author: calculate each metric per author, then equally average authors with defined values. `n` remains row count.
- Overall: all Test rows in a cell.
- History Available: at least one legal visible exact-Pinyin record.
- Ambiguous: at least two distinct historical targets.
- Conflict: Ambiguous, exactly one target has the maximum frequency, and Gold differs from that winner. Frequency ties are excluded.

Per-author metrics live within each cell `result.json` and completed historical CSVs. Row-level predictions are `cells/.../predictions.jsonl`. Join F/M1/M2 by `condition_id` to compute transitions:

- F wrong -> M2 correct: `F.gold_rank != 1 and M2.gold_rank == 1`.
- F correct -> M2 wrong: inverse.
- Oracle headroom: Gold rank 1 in any candidate method versus the chosen system. Missing@10 cannot change in this matrix because candidate surfaces are identical.

Do not derive final matrix values from smoke results. Final aggregate files are written only by successful `finalize`.

## 13. Documentation workflow

Use the existing system:

1. `RESEARCH_TARGETS.md` only for authoritative thesis-direction changes.
2. `docs/research/<experiment>.md` for frozen design/semantics before running.
3. `docs/reports/<NN>_<experiment>.md` for implementation status, then measured results/decision.
4. `docs/VERSION_HISTORY.md` for the new implementation/result checkpoint and tag.
5. `README.md` for stable manual operator commands, not transient observations.
6. Result roots retain machine-readable manifests, selections, metrics, checksums, runtime and completion marker.

Repeatable future checklist:

- **DESIGN:** write population, leakage barriers, method, grids, cache identity, outputs and stop criteria.
- **IMPLEMENTATION:** branch/worktree, minimal code, atomic/resumable persistence.
- **VALIDATION:** focused/full tests, compile, diff check, hashes, bounded real smoke.
- **HUMAN AUDIT:** inspect representative rows, chronology, context, candidate surfaces and logs.
- **RUN:** record exact command/environment/PID/logs; detached and resumable.
- **RESULTS:** require completion marker; produce metrics, checksums, transition diagnostics and limitations.
- **DECISION:** update report/version history and tag without rewriting prior artifacts.

## 14. Manual-edit safety guide

### A. Change PinyinGPT effective input budget

Likely files: `backend.py` truncation helper, the relevant runner call sites, a new experiment config/report, and tests in `test_pinyingpt_reference.py` plus matrix tests. Decide whether the budget is total positions or context tokens. Do not alter beam, Top-10, prompt positions, Pinyin compatibility or frozen files in place. A changed used context invalidates Generic predictions; create new Test/Dev Generic result roots. If stored context also changes, BGE/M2 keys change automatically but manifests/required-context caches must be new.

### B. Change M1/M2 context preprocessing

M1: `BGEContextEmbedder._model_text`, `EMBEDDING_PREPROCESSING_VERSION`, and cache tests. M2: `CandidateAwareTemplate.prepare`, `TRUNCATION_VERSION`, pair-key metadata and tests. Do not silently reuse old SQLite metadata or overwrite frozen results. Changed exact context/preprocessing invalidates BGE vectors; changed M2 serialization/max length invalidates pair scores.

### C. Add a diagnostic column

For row diagnostics, modify `_evaluate_method` output and finalizer CSV construction; add tests asserting values but do not feed the column into ranking/selection. Existing completed cells will not acquire the column unless a separate metrics/export pass is implemented; never pretend mixed schemas are complete.

### D. Add a human-audit exporter

Create a read-only experiment script that joins existing JSONL predictions by stable IDs and writes a new diagnostics/audit path. Do not load models, mutate manifests, or write into frozen cell directories. Test deterministic sampling and leakage-safe fields.

### E. Start a new experimental version

Wait for or separately isolate the current worker. Create a new worktree/branch, design doc, unique result/cache namespace and completion marker. Pin inputs/model/config; copy no context-dependent cache blindly. Run focused/full tests and bounded smoke; record command; only tag implementation/result after audit. Never edit old tags or principal results.

## 15. Recovery and backup

- **Worker crash / PC restart:** verify no old worker; read log tail and manifest; rerun the exact `--phase run` command. Complete/reused cells skip, failed/running cells retry, Dev Generic partial journals and SQLite caches resume.
- **Interrupted matrix:** do not edit states manually just to force progress. A stale `running` state is treated as incomplete by `run_cell_group` and retried because only `complete` skips.
- **Wrong Git change:** inspect `git diff`; restore only named tracked files after backing up intentional work. Avoid `reset --hard`. Use a new corrective commit if already committed/shared.
- **Partially written result:** manifest should not be `complete`; rerun overwrites that cell's predictions/result. Atomic JSON metadata uses temporary sibling replacement. Preserve the file for diagnosis first.
- **Cache exists but manifest failed:** this is normal after a later-stage failure. Rerun; cache keys and provenance reuse valid entries.
- **Abandon an experiment:** stop only its verified PID, archive logs/results, record the decision, and preserve the branch/tag/cache. Start the replacement in a new namespace.
- **Backup SQLite:** stop the writer and copy database plus WAL/SHM, or use SQLite `.backup`; never copy only the main file while writes are active.

## 16. Uncertainties and dynamic facts

- The matrix is active, so cell states and BGE/M2 counts in this document are snapshots.
- The exact peak CUDA allocation that caused the original transition failure was not captured before that process exited. The diagnosis is supported by a healthy pinned GGUF, successful standalone and short same-process transitions, the long-run boundary, and successful allocator-release resume; it is an infrastructure diagnosis, not a model-content change.
- The CLI has no cell-only or metrics-only phase. Add one only as a separately reviewed orchestration feature if genuinely needed.


## EM-1 frozen conclusion

EM-1 is complete.

Current frozen EM-1 configuration:

- condition: Full+Short
- history: H5000
- authors: Etinjat, Re_spectators, breaddddd
- recovery K: 1
- frequency lambda: 4

R now means exact-scored recovery-only, not the older PV1 approximate
boundary-score recovery.

R+F uses the same exact-scored unified candidate pool plus frequency.

Frozen Test:

- G0 Top1 77.600%
- F Top1 81.067%
- R Top1 77.700%
- R+F Top1 81.033%

EM-1 improves candidate coverage/ranking depth but not overall Top1 beyond
F. Do not retune K or lambda using Test.

Recovery analysis must distinguish raw Generic Missing from
backend-reachable Generic Missing.

Backend-unreachable Gold rows remain in primary metrics and are documented
under `docs/data_quality/KNOWN_ISSUES.md`.

Next research stage:

EM-2 - Frozen PinyinGPT hidden-state kNN retrieval.

EM-2 should begin as a retrieval diagnostic rather than immediate
end-to-end reranking. Compare the frozen task-native PinyinGPT
representation against the existing BGE cosine retrieval baseline,
especially on Ambiguous and Conflict subsets.

Reproduction:

`docs/external_memory/EM1_REPRODUCIBILITY_2026-08-19.md`

<!-- EM2-FINAL-CLOSE-2026-08-19 -->
## Active-stage update - EM-2 closed, EM-3 next

EM-2 External Memory is frozen and closed.

Canonical closure:

- `docs/external_memory/em2/EM2_FINAL_REPORT_2026-08-19.md`
- `docs/external_memory/em2/EM2_REPRODUCIBILITY_2026-08-19.md`

Next active research stage:

**EM-3 - task-specific learned historical relevance.**

Start from:

`docs/external_memory/em2/EM2_TO_EM3_HANDOFF_2026-08-19.md`

Do not reopen EM-2 for hidden-layer sweeps, further M1/M2 tuning, new adaptive-gate engineering, or Test-driven redesign.

<!-- EM1_PV_AUDIT_20260819_START -->
## Closed explanatory audit before EM-3

The PV1/PV2 -> EM-1 relationship has been audited and is now closed.

Do not describe PV1 -> EM-1 as a pure scoring-only substitution.

Frozen explanatory facts:

```text
same Test rows: 3000
same-candidate scored recovery rows: 290 / 306 = 94.77%
different-candidate scored recovery rows: 16 / 306 = 5.23%

PV1 -> EM1-R+F overall:
helped 26
harmed 7
net +19

same-candidate subset:
helped 24
harmed 7
net +17
```

PV1 recovered more Generic-missing Gold targets, while EM1-R+F achieved better
overall Top-1. The evidence strongly suggests that exact PinyinGPT recovered-
candidate scoring is the principal source of EM-1's safer behaviour, but not
the sole formally isolated causal difference.

The audit changed no frozen PV or EM-1 result, performed no tuning, and ran no
new Test PinyinGPT inference.

Detailed record:
`docs/external_memory/EM1_PV_COMPARISON_ADDENDUM_2026-08-19.md`.

Next research stage: EM-3.
<!-- EM1_PV_AUDIT_20260819_END -->
