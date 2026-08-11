# Transparent, User-Controllable Pinyin Personalisation

## Current Phase

Phase 4F — HuoziIME Reference Backend Reproduction

## Current Objective

Phase 4F establishes a published, modern personalisation backend before the
thesis begins its transparency and control contribution. It implements the
strongest supportable desktop adaptation of
[HuoziIME](https://github.com/Shan-HIT/HuoziIME), grounded in the official
[ACL 2026 paper](https://aclanthology.org/2026.acl-demo.32/) and release.

The official audit, implementation, frozen training-state preparation, and
engineering smoke test are complete. The full Zhu Ziqing evaluation has
deliberately not been run, so Phase 4F is not final and no benchmark claim is
made.

## Reference Status

Recommended classification: **B. Faithful HuoziIME reference-backend
adaptation**.

- Audited repository commit: `63f249e711f6501169e6baafec7e12318b3c765b`
- Audited release: `v1.0.1-beta`
- Generation model: official bundled `scirime_grpo_v2_744-q4_0.gguf`, Q4_0
- Embedding model: official bundled `bge-small-zh-v1.5-q8_0.gguf`, Q8_0
- Desktop runtime: `llama-cpp-python==0.3.16`, Metal on Apple M1
- Retrieval: per-user HNSW over L2-normalized 512-dimensional embeddings

The complete component classification and all asset hashes are recorded in
[`results/audits/phase_04f/reproduction_matrix.md`](results/audits/phase_04f/reproduction_matrix.md)
and
[`results/experiments/phase_04f/backend_manifest.json`](results/experiments/phase_04f/backend_manifest.json).

## Current Backend

```text
own preceding text + traced Pinyin/keystrokes
                    ↓
       official lightweight LLM generation
                    ↓
        official learned retrieval action?
             no ↙             ↘ yes
      direct candidates    per-user HNSW
                                ↓
                    selected plaintext memory
                                ↓
                    memory-grounded generation
                                ↓
                   candidates + decision trace

completed training history → separate background memory worker → frozen L2/L3
```

The backend includes:

- direct LLM candidate generation rather than Luna/Phase 4E reranking;
- the official checkpoint's selective `<MEM_RETRIEVAL>` action;
- authoritative, individually addressable plaintext L2 memories;
- per-user HNSW indexes that map every vector to a memory ID;
- memory-grounded LLM reruns with supplied memory IDs and text recorded;
- separate chronological L3 interaction/decision traces;
- explicit foreground prediction and background memory processing;
- strict Zhu/Lu user isolation and frozen training histories.

Phase 4F does not import Phase 4E frequency, semantic, vocabulary, or linear
reranking features.

## Input-Only Benchmark Mode

The frozen corpus has the user's own preceding writing, not an interlocutor's
message. The benchmark therefore passes the final 100 characters of raw
preceding text, records normalized tone-free Pinyin/keystrokes, and always uses
`external_context=None`. It never fabricates a dialogue partner.

The audited Android generation path reads text before the cursor; it does not
expose a separate Pinyin decoder constraint. Pinyin is therefore accepted and
traced by the API but is not injected into the LLM prompt. Exact target ranking
metrics are only generative sanity/reference metrics, not directly equivalent
to the earlier Luna candidate-ranking experiments.

## Memory and Retrieval

Training histories are adapted into per-work chronological trajectories capped
at the upstream 4,000-character memory-worker buffer. The official checkpoint,
prompt, greedy 192-token generation path, JSON schema, and upstream skip
semantics produce frozen memories. Each memory records its stable ID, user,
plaintext, chronology, source interaction IDs, vector label, and provenance.

Retrieval follows the audited release: HNSW inner product over L2-normalized
embeddings, `max_elements=2048`, `M=16`, `ef_construction=200`,
`ef_search=64`, vector Top-20, raw-cosine threshold `0.4`, and one selected
memory after the official vector/lexical reranking formula. Correct-user state
contains only Zhu training history; wrong-user state contains only Lu training
history. No test-time updates occur.

## How to Run

Install the additional Phase 4F runtime dependencies:

```bash
.venv/bin/pip install -r requirements-phase4f.txt
```

On the audited Apple Silicon host, Command Line Tools did not expose libc++
headers to Python extension builds automatically. The exact successful native
build commands were:

```bash
ARCHFLAGS='-arch arm64' CXXFLAGS='-isystem /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1' .venv/bin/pip install hnswlib==0.8.0
ARCHFLAGS='-arch arm64' CFLAGS='-isystem /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1' CXXFLAGS='-isystem /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1' CMAKE_ARGS='-DCMAKE_OSX_ARCHITECTURES=arm64 -DGGML_METAL=ON' .venv/bin/pip install llama-cpp-python==0.3.16
```

Run the complete unit-test suite:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Audit manifests, hashes, and prior artifacts without loading the models:

```bash
.venv/bin/python -m experiments.exp_phase_04f_reference_backend --audit
```

Download/extract the pinned official release and prepare frozen training-only
Zhu/Lu memories and indexes:

```bash
.venv/bin/python -m experiments.exp_phase_04f_reference_backend --prepare
```

Run the small real-model engineering path:

```bash
.venv/bin/python -m experiments.exp_phase_04f_reference_backend --smoke-test
```

After reviewing and accepting the implementation, manually run the final
input-only evaluation:

```bash
.venv/bin/python -m experiments.exp_phase_04f_reference_backend
```

The final command writes
`results/experiments/phase_04f/evaluation.json`. It has not been run in this
checkpoint.

## Phase 4E vs Phase 4F

Phase 4E remains the previous hybrid neural-transparent research prototype:
Luna candidates plus frozen Qwen features and a learned linear reranker. Phase
4F is additive and separate: it adapts the published HuoziIME LLM-generation,
plaintext-memory, selective-retrieval, and grounded-generation backend. No
Phase 4E implementation or results were rewritten.

## Current Limitations

- The official merged checkpoint is public only inside the APK; there is no
  independently versioned model repository or complete training provenance.
- The paper evaluation datasets/results and a runnable post-training pipeline
  are not public, so published experiments cannot be reproduced from the
  available artifacts.
- The thesis corpus tests only input-only operation, not HuoziIME's complete
  cross-application conversation-context capability.
- Pinyin is trace metadata rather than an upstream-documented generation
  constraint, limiting comparability with Luna Top-K metrics.
- Exact Android KV-splice/prefix caching, mobile scheduling, frontend, and MCP
  transport are omitted; desktop latency is not compared with phone latency.
- The engineering smoke is not a performance or accuracy result. The final
  benchmark remains pending manual execution and review.

## Project History

- Phase specifications: [`docs/phases/`](docs/phases/)
- Completed and engineering results: [`results/experiments/`](results/experiments/)
- Audit outputs: [`results/audits/`](results/audits/)
- Workflow rules: [`docs/WORKFLOW.md`](docs/WORKFLOW.md)

Git tags are not created automatically.
