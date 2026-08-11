# Third-Party Provenance — Phase 4F HuoziIME

## HuoziIME

- Repository: [Shan-HIT/HuoziIME](https://github.com/Shan-HIT/HuoziIME)
- Pinned audited SHA: `63f249e711f6501169e6baafec7e12318b3c765b`
- Release inspected: `v1.0.1-beta`
- Repository license: GNU General Public License v3.0 (`GPL-3.0`)

The repository is not vendored into this project. Most local backend code is an
independent Python adaptation across a clean desktop boundary, based on
documented behaviour in the official paper and source. The source file paths
and adaptations are recorded row by row in the reproduction matrix.

The small default business prompt, memory-worker prompt, and ChatML template are
copied/integrated from upstream in `src/reference_backend/official_prompts.py`.
That boundary carries an SPDX `GPL-3.0-only` notice and records the pinned source
SHA. No upstream notice was removed. No large Kotlin/C++ source file is copied.
Because this module is imported by the backend, distribution of the combined
work should be reviewed for GPLv3 compliance rather than assuming that file
isolation removes the upstream licence obligations.

## Official Release Assets

The official APK is downloaded during preparation and retained only under the
Git-ignored `.cache/phase_04f/` directory.

- APK SHA-256:
  `6ce98a804e503aa2d6dc426ff6284d5064ffff09c9527dcaacfc050f6ab99207`
- Generation GGUF SHA-256:
  `2012b7aa860674e5f2b9fc0c90cc4828b7e5f50f7be4069fa0122685956416a5`
- Embedding GGUF SHA-256:
  `5a88d266870fbd27c6f329df60de80e2d4cf3bbd5e6f080bd5c1b2e5abb12039`

The generation GGUF contains `Apache License 2.0` in model metadata. That
metadata is recorded as model-provided information; it is not used to infer a
license for unrelated APK/repository content. The embedding file has no
independent revision/model card in the release. Neither model file is committed
to Git.

## Runtime and Retrieval Dependencies

- `llama-cpp-python==0.3.16` (MIT metadata): linked/wrapped as the desktop GGUF runtime. Its
  Python and bundled llama.cpp code are not copied into the repository.
- `hnswlib==0.8.0`: linked through its Python binding. The local index is an
  independent adaptation of the audited HNSW configuration, not a copy of the
  Android wrapper.
- NumPy: existing project/runtime dependency used for vector normalization.

HuoziIME vendors llama.cpp and hnswlib source without recording their original
upstream Git SHAs. The exact HuoziIME files inspected are therefore pinned by
the parent repository SHA and, for auditability, local file checksums in the
backend manifest. No unsupported dependency SHA is invented.

## Declared Submodule

`.gitmodules` declares `paper` at
`git@github.com:Shan-HIT/ScirIME-Paper.git`, but the audited Git tree contains no
paper gitlink and therefore no submodule SHA. The published ACL paper is used
directly. The missing submodule revision is classified as not reproducible from
public artifacts.

## Local Modifications and Boundaries

Local adaptations include desktop Python orchestration, sequential candidate
branches, deterministic benchmark seeds, per-user stores/indexes, explicit
provenance/activity fields, strict input-only mapping, and frozen per-work
training trajectories. Android UI, MCP transport, scheduling, and KV-splice are
omitted. These differences are documented and are not presented as upstream
code.
