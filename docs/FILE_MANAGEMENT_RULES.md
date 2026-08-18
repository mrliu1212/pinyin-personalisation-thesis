# File Management Rules

## 1. Purpose

This document defines the long-term file, directory, naming, lifecycle, and documentation conventions for this thesis repository.

The goals are to:

- keep the repository understandable as experiments accumulate;
- preserve frozen research evidence and provenance;
- distinguish reusable code, experiment runners, documentation, helpers, and generated artifacts;
- make old checkpoints understandable and reproducible without relying on memory;
- avoid uncontrolled proliferation of duplicate, temporary, or ambiguously named files.

---

## 2. Canonical directory responsibilities

### `src/`

Reusable implementation and core research logic.

Examples:

- ranking formulas;
- history indexing;
- model wrappers;
- candidate recovery logic;
- shared metrics;
- reusable caches and data structures.

`src/` should not contain one-off experiment orchestration or generated result files.

### `experiments/`

Runnable research experiment entry points.

Formal experiment runners should answer a specific research question or reproduce a formal experimental stage.

Stage-specific directories are preferred when useful, for example:

```text
experiments/context_lab/
experiments/external_memory/
```

### `experiments/helpers/`

One-off engineering, inspection, migration, debugging, path-finding, or support utilities.

Helpers are not formal experiment entry points.

### `config/`

Explicit dataset or experiment configuration.

Frozen configuration should not be silently modified after the checkpoint it defines.

### `docs/research/`

Method definitions, protocols, information boundaries, grids, selection rules, and evaluation design.

### `docs/reports/`

Completed experimental results and interpretation.

### Stage-specific documentation directories

Directories such as:

```text
docs/context_lab/
docs/external_memory/
```

may contain plans, diagnostics, stage summaries, and stage-specific documentation.

### `results/`

Generated experiment outputs, such as:

- predictions;
- JSON/JSONL results;
- CSV grids;
- metrics;
- manifests;
- logs;
- caches.

Generated result files are not source code and should not be manually edited to alter an experimental result.

### `tests/`

Automated tests of reusable implementation or research invariants.

---

## 3. General separation of responsibilities

Use the following rule whenever practical:

```text
docs explain
experiments run
src implements
results store generated output
tests verify
```

Each human-authored file should have one clear primary responsibility.

Avoid combining implementation, experiment orchestration, method definition, result interpretation, and temporary debugging in one file unless there is a strong reason.

---

## 4. Naming rules

Names should describe the actual research object or function.

Good examples:

```text
recovery_reranking_fusion.py
hidden_state_knn_retrieval.py
ime_specific_cross_encoder.py
ctx64_m1_test.py
personal_vocabulary_h5000.py
```

Avoid ambiguous names:

```text
test2.py
new.py
new_run.py
temp.py
try_again.py
final2.py
final_final.py
latest.py
```

Avoid meaningless suffix chains:

```text
_new
_new2
_fixed
_latest
_final_final
```

Semantic suffixes are allowed when they encode a real distinction:

```text
_v2
_h5000
_ctx64
_dev
_test
_2026-08-19
```

---

## 5. Git is the primary version history

Do not use duplicate filenames as a substitute for version control.

Avoid:

```text
report.md
report_updated.md
report_final.md
report_final2.md
```

A formal report should normally keep one canonical path.

Example:

```text
CONTEXT_STRENGTHENING_REPORT_2026-08-18.md
```

Git history preserves revisions.

Create a new `v2` file only when the underlying research object, protocol, interface, or schema genuinely becomes a new version.

---

## 6. File lifecycle statuses

Use the following statuses consistently.

### ACTIVE

Currently part of active research or implementation.

### FROZEN

Completed implementation, protocol, or report associated with a frozen checkpoint.

Frozen scientific logic should not be silently changed for additional tuning.

### HELPER

One-off engineering, debugging, inspection, migration, path-finding, or support utility.

A HELPER is not a canonical formal experiment entry point.

### LEGACY

Historical implementation retained for provenance but superseded by a later formal version.

### DEFERRED

Planned or intentionally preserved work that is not currently active.

### GENERATED

Machine-generated result or artifact.

### LOCAL-ONLY

Important local artifact, cache, large result, or model data intentionally not tracked in Git.

---

## 7. Frozen experiment rule

Once an experiment is formally frozen:

- preserve its code;
- preserve its report;
- preserve its commit/tag;
- preserve result provenance;
- do not change its experimental logic and continue calling it the same experiment.

If the scientific method changes, create a new version, experiment, or output namespace.

Small documentation corrections are allowed when they do not misrepresent what was originally run.

---

## 8. Generated results must not be manually repaired

Do not manually edit generated JSON, CSV, prediction, or metric files to make an experiment appear correct.

If a bug affects a formal experiment:

```text
preserve original artifact
fix implementation
create a new version/result namespace
rerun as a separate experiment
document the relationship
```

Do not overwrite historical evidence that was already treated as a formal result.

---

## 9. Result and cache namespace rule

Different experimental semantics require different output namespaces.

Do not reuse an existing result directory after changing:

- method logic;
- preprocessing;
- population;
- context policy;
- history semantics;
- scoring definition;
- candidate construction;
- model revision;
- evaluation protocol.

A cache may only be reused when its provenance is compatible with the new experiment semantics.

---

## 10. Large local artifacts

Large files should normally remain untracked, including:

- prediction caches;
- model caches;
- embeddings;
- pair-score databases;
- large JSONL outputs;
- logs;
- generated matrices.

Important local-only artifacts must still be documented.

Where relevant, record:

- path;
- purpose;
- producer;
- consumer;
- whether it can be regenerated;
- whether future reproduction depends on preserving it.

Untracked does not mean undocumented.

---

## 11. Helper policy

Helpers should be explicitly identifiable as non-authoritative.

Typical helpers include:

- path discovery;
- one-time conversion;
- debugging;
- temporary environment inspection;
- result comparison utilities.

Do not automatically delete helpers after use.

First classify them in `docs/FILE_INDEX.md`, then make cleanup decisions separately.

---

## 12. Documentation roles

### `docs/FILE_MANAGEMENT_RULES.md`

Defines how repository files are created, named, classified, and maintained.

### `docs/FILE_INDEX.md`

Answers:

> What is this file or directory for, and what is its current status?

### `docs/REPRODUCIBILITY_INDEX.md`

Answers:

> How could this frozen checkpoint be rerun later?

### `docs/VERSION_HISTORY.md`

Answers:

> What happened when?

It is the chronological checkpoint navigation.

### `docs/TECHNICAL_HANDOFF.md`

Answers:

> How should an operator continue working with the current repository and environment?

### `docs/research/`

Defines methods and protocols.

### `docs/reports/`

Records completed results and interpretation.

Avoid creating multiple competing sources of truth for the same information.

---

## 13. Plan and report are different document types

A plan describes:

- research question;
- intended method;
- intended population;
- parameter/selection rules;
- intended evaluation.

A completed report describes:

- what was actually executed;
- population;
- results;
- interpretation;
- limitations;
- provenance.

A completed report should not remain written mainly as a future plan.

---

## 14. Formal research document structure

Where applicable, formal research-stage documents should make their role obvious near the beginning and include:

```text
Purpose
Status
Scope
Related checkpoint/tag
```

Formal reports should distinguish, where applicable:

```text
Research Question / Hypothesis
Method
Population
Results
Interpretation
Limitations
Provenance / Reproduction
```

---

## 15. Hypothesis and verified result must be separated

Diagnostic explanations and hypotheses must be labelled as such.

Do not rewrite an unverified explanation as an established conclusion.

---

## 16. Numerical results must include scope

Avoid isolated statements such as:

```text
Top-1 = 79.77%
```

Prefer:

```text
Full+Short / H5000 / 3-author Test:
Top-1 = 79.77%
```

Where relevant, clearly distinguish:

- Dev tune;
- Dev evaluation;
- exploratory Test;
- final Test.

---

## 17. Provenance requirement

Formal reports should contain or link provenance where available, including:

- commit/tag;
- input artifacts;
- model/checkpoint revision;
- important parameters;
- result directory;
- hashes or manifest identity where relevant.

---

## 18. Reproduction information

Formal frozen experiments should have a documented reproduction path where possible.

The canonical reproduction information may live in:

```text
docs/REPRODUCIBILITY_INDEX.md
```

Never invent missing historical commands, arguments, paths, hashes, or dependencies.

If exact reproduction cannot be established, document that limitation explicitly.

---

## 19. File index requirement

Every meaningful human-authored source, experiment, config, test, or documentation file should eventually be represented in:

```text
docs/FILE_INDEX.md
```

Generated result trees should be indexed at meaningful directory or canonical-artifact level rather than exhaustively enumerated.

The index is the canonical answer to:

> Is this file still relevant, and what is it for?

---

## 20. Reproducibility index requirement

Every meaningful frozen research checkpoint should eventually be represented in:

```text
docs/REPRODUCIBILITY_INDEX.md
```

Use these reproducibility statuses:

```text
COMPLETE
PARTIAL
RESULT-ONLY
LOCAL-ARTIFACT-DEPENDENT
LEGACY
```

`COMPLETE` means a reproduction path can be established from preserved evidence.

It does not mean the experiment has been rerun during documentation audit.

---

## 21. No destructive cleanup during documentation audits

Repository documentation audits must not automatically:

- delete files;
- move files;
- rename files;
- clean untracked results;
- rewrite experiment outputs;
- refactor research code.

Audits classify first.

Cleanup decisions are made separately after human review.

---

## 22. Git safety

Avoid broad destructive or indiscriminate Git operations around research artifacts.

Do not casually use:

```text
git add .
git clean
git reset --hard
```

Prefer explicit staging of reviewed files.

Preserve untracked research results and caches until their role is understood.

---

## 23. New-file decision rule

Before creating a new file, determine whether it is:

```text
reusable source
experiment runner
helper
configuration
research documentation
result report
generated artifact
test
```

If its category and responsibility are unclear, do not create it yet.

---

## 24. Recommended new-stage structure

Where appropriate:

```text
docs/
  <stage>/
    <STAGE>_PLAN_<date>.md
    <STAGE>_REPORT_<date>.md

experiments/
  <stage>/
    <formal experiment runners>

src/
  <shared reusable implementation>

results/
  <research area>/
    <stage outputs>
```

Not every stage needs every file, but responsibility boundaries should remain clear.

---

## 25. Source-of-truth principle

Do not create multiple competing authoritative definitions.

Prefer:

- method/protocol definition → research/config/source;
- frozen result → report/result metadata;
- chronology → `VERSION_HISTORY`;
- reproduction path → `REPRODUCIBILITY_INDEX`;
- file role/status → `FILE_INDEX`.

Other documents should link to the authoritative source rather than redefine it.

---

## 26. General principle

When convenience conflicts with preserving repository clarity and research provenance:

> preserve provenance first.

Git history should preserve historical versions.

Documentation should preserve meaning.

Generated artifacts should preserve evidence.

The active tree should remain understandable without erasing the past.
