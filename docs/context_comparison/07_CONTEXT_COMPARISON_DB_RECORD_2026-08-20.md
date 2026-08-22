# Context comparison database record - 2026-08-20

## Purpose and policy

Status: **LOCAL-ONLY DEV DATABASE / NO INFERENCE**.

Path: `results/personalisation/context_comparison_v1/dev_context_eval_v1.sqlite`

The database provides one canonical join surface for the frozen balanced-3000
manifest, exact cache coverage, model definitions, and surviving compatible
legacy predictions. It does not fabricate missing ranks and does not run a
model. The SQLite file is generated/local-only and must not be staged.

## Schema and validated counts

| Table | Purpose | Rows | Join key |
|---|---|---:|---|
| `interactions` | canonical/Pilot identity, author/work/chronology, context, Pinyin, Gold | 3,000 | `canonical_row_id` |
| `subset_membership` | history/ambiguity/formal Conflict/pair-trainable/balanced/legacy flags | 3,000 | `canonical_row_id` |
| `runtime_history` | prediction-visible same-Pinyin count, distinct targets, frequency winner | 3,000 | `canonical_row_id` |
| `oracle_diagnostics` | segregated Gold-in-history/count post-hoc fields | 3,000 | `canonical_row_id` |
| `model_registry` | exact five-route configuration JSON | 5 | `model_id` |
| `predictions` | compatible persisted final ranks/Top-K where actually available | 3,984 | `(canonical_row_id, model_id)` |
| `cache_coverage` | one state and reason for each row/model | 15,000 | `(canonical_row_id, model_id)` |
| `provenance` | manifest/source hashes, bridge/schema/Test status | 5 | `key` |

Prediction rows are 996 each for Original-M1, Original-M2, Hidden-M1, and
EM3. Hidden-M2 has no persisted final-row artifact, so its 996 compatible rows
are marked reconstructable rather than inserted as predictions. Rank-only
source artifacts leave `top1`/`topk_json` null; only Hidden-M1 supplies a full
legacy ranking. This preserves the distinction between correctness evidence
and candidate-text availability.

Runtime-visible fields and Gold-dependent diagnostics are deliberately in
different tables. `conflict` remains the project's formal analysis subset and
is not redefined as `G != F`.

## Inputs and regeneration

The builder is
`experiments/context_comparison/prepare_context_comparison.py`. Inputs and the
exact PowerShell command are recorded in the Dataset Record and
`docs/REPRODUCIBILITY_INDEX.md`. External SQLite inputs are opened using
immutable read-only URIs. If the database already exists, the preparation
runner validates its 3,000 interaction rows rather than silently rebuilding it.

The database enables later cache-only reconstruction and evaluator work. It is
not a completed five-model comparison, contains no newly inferred prediction,
and uses no Test data.
