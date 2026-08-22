# Standardized reset execution log — 2026-08-20

All stages in this log have `used_test=false`.

## Environment and safety audit

- Timestamp/date: 2026-08-20, Europe/Berlin.
- Worktree: `C:\Users\chiar\Desktop\LBH\thesis-context-compare`.
- Branch: `work/context-model-comparison`.
- Git HEAD: `80b053764e70ee2f2886892ba516a6b9e2470e59`.
- Python: `C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe`.
- Python: 3.12.13, MSC v.1944, AMD64.
- OS: Windows 11 (`Windows-11-10.0.26200-SP0`).
- PyTorch: `2.11.0+cu128`; PyTorch CUDA `12.8`.
- Transformers: `4.57.6`.
- llama-cpp-python: `0.3.16`.
- NumPy: `2.5.2`.
- sentence-transformers and scikit-learn are not installed in this environment.
- CUDA available: true.
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, driver 566.07, 8188 MiB.
- Historical reference worktree:
  `C:\Users\chiar\Desktop\LBH\thesis-context-lab`, read-only for this task.

The first environment-version probe had a PowerShell/Python quote-escaping
SyntaxError. It was rerun with corrected quoting; no artifact or state was
changed by the failed read-only probe.

## Focused invariant implementation

- Added `src/personalisation/standardized_context_comparison.py` for exact
  MRR@10, macro/micro metrics, conditional metrics, G/F/C eight-way counts,
  behavior/failure diagnostics, deterministic whole-work splitting, and causal
  rolling H5000 annotation.
- Added focused tests under `tests/context_comparison/`.
- Optimized partition visibility counters to O(1) updates inside the rolling
  window; this changes orchestration cost, not history membership.

Commands:

```powershell
$env:PYTHONPYCACHEPREFIX='C:\Users\chiar\Desktop\LBH\thesis\.tmp_context_compare_pyc'
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' -m py_compile src/personalisation/standardized_context_comparison.py experiments/context_comparison/prepare_standardized_reset.py tests/context_comparison/test_standardized_context_comparison.py
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests/context_comparison/test_standardized_context_comparison.py -q
```

Observed: 10 focused tests passed. Direct default `py_compile`/pytest cache
creation was blocked by isolated-worktree sandbox permissions, so transient
bytecode was redirected to the writable main-worktree temp directory and the
pytest cache provider was disabled. Source/result semantics are unchanged.

## Standardized preparation worker

The versioned preparation command verifies the Train, legacy regression, and
Dev3000 hashes before writing. It streams the large authoritative Train source
per author to bound memory. Outputs are new under
`results/personalisation/context_comparison_v2/`; no historical artifact is
overwritten.

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' -m experiments.context_comparison.prepare_standardized_reset `
  --train-manifest 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\reranking_matrix\manifests\history_full_short.jsonl' `
  --legacy-rows 'C:\Users\chiar\Desktop\LBH\thesis-context-lab\results\personalisation\external_memory\em2_four_way_dev_compare\rows.jsonl' `
  --dev3000 'results\personalisation\context_comparison_v1\clean3_history_balanced_3000.jsonl' `
  --pilot-history 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\history_manifest.jsonl' `
  --pilot-dev 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\dev_manifest.jsonl' `
  --output-root 'results\personalisation\context_comparison_v2'
```

The command was launched hidden with durable stdout/stderr logs because the
522 MB source requires a multi-pass preparation. Final hashes/counts are added
to the split and history records after successful completion.

Before pair generation or training, the pre-retune model registry seed was
corrected from the experiment-date placeholder `20260820` to historical recipe
seed `42`. Repository evidence is the frozen EM3 generator/trainer, both of
which use seed 42. No Train-Val or Dev result had been produced or inspected;
this was a provenance transcription fix, not outcome-driven tuning.

## Frozen split and evaluator outcome

- Legacy 5,608 regression: PASS, exact eight-way counts
  `3361, 24, 42, 100, 403, 35, 45, 1598`.
- Clean3 source: 178,942 rows at SHA `6d32d441...58597`.
- Train-Fit: 144,526 rows at SHA `547a4f817...0c8a6`.
- Train-Val: 34,416 rows at SHA `d7ae1cc21...f2220`.
- All split, whole-work, chronology, no-Test, and causal-history assertions
  passed. See the split/history machine audits.

## Workload audit

The read-only audit found 1,124,083 exact-Pinyin causal Train-Val history edges
and 42,454 required unique rows. BGE has 39,993 exact hits among 42,358 unique
context keys, leaving 2,365 misses. The historical hidden cache has zero exact
row-ID hits, leaving 42,454 new hidden representations. No inference was run by
the audit itself.

## EM3 pair generation

The first detached invocation explicitly passed `--authors Agent Phage ...`.
PowerShell `Start-Process` split the spaced author into `Agent` and `Phage`; the
generator stopped with `authors absent from source` before writing any planned
pair/audit/provenance artifact. It was rerun using the script's frozen default
Clean3 tuple. This is an orchestration correction only.

Successful command parameters: Train-Fit source, H5000, max rounds 3,
negatives/round 3, seed 42. Outcome:

- eligible queries: 35,290;
- positive pairs: 99,671;
- negative pairs: 169,400;
- total pairs: 269,071;
- query/history duplicate pairs: 0;
- non-prior pairs: 0;
- source rows: 144,526;
- Test used: false.

Pair path:
`results/personalisation/context_comparison_v2/em3_train_pairs_v1/train_pairs.jsonl`.
Its final SHA is recorded in the generated provenance and the PRE_DEV_FREEZE.

## Train-Val completion and sealed Dev3000 — 2026-08-21

- Generic CE pair scoring completed and passed exact cache-integrity checks for
  all `381,295 / 381,295` registered Train-Val pairs.
- Frozen EM3 scored its exact `302,649 / 302,649` registered subset on GPU.
- M2, Hidden-M2, and EM3 grids completed without using Dev3000 or Test.
- The machine freeze was written before any standardized Dev3000 inference:
  `results/personalisation/context_comparison_v2/pre_dev_freeze_v1.json`
  (SHA256 `7c0fcf69823f0b4b7d8b914a81ea54a097e12c03cb61c515c2400be46df46824`).
- The sealed Dev3000 run used the unchanged 3,000-row manifest at SHA256
  `9181f895eb19d0c36852e511263bfaefb34459dcd44efa6f45a44252e6b03f93`.
- The pre-existing Pilot Generic cache supplied 1,568 exact Dev rows. Only its
  1,432 missing rows were generated with the frozen PinyinGPT configuration;
  the candidate surface was not changed.
- Dev representation and pair caches were filled resumably, then the seven
  frozen systems were evaluated once. Test remained closed and
  `used_test=false`.
- Final machine result:
  `results/personalisation/context_comparison_v2/dev3000/standardized_dev3000_result.json`.
- Final human report:
  `docs/context_comparison/14_STANDARDIZED_DEV3000_RESULT_2026-08-21.md`.
