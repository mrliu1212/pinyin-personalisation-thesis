# Phase 4F — HuoziIME Reference Backend Reproduction

## Objective

Establish an academically honest desktop reference backend for the published
HuoziIME personalisation architecture. The phase is infrastructure for later
work on inspectable learned state, provenance, decision transparency, deletion,
disabling, correction, reset, and counterfactual verification.

> The thesis contribution is not the HuoziIME personalisation architecture.
> Phase 4F establishes a published modern reference backend. Subsequent phases
> study transparency and controllability of learned personalisation.

Phase 4F replaces the project's attempt to invent another personalisation
algorithm with a pinned implementation of an existing peer-reviewed reference
system. Fidelity takes precedence over Zhu Ziqing accuracy.

## Reference and Audit

- Paper: [HuoziIME: An On-Device LLM-Enhanced Input Method for Deep
  Personalization](https://aclanthology.org/2026.acl-demo.32/), ACL 2026 System
  Demonstrations.
- Repository: [Shan-HIT/HuoziIME](https://github.com/Shan-HIT/HuoziIME).
- Audited repository commit: `63f249e711f6501169e6baafec7e12318b3c765b`.
- Audited release: `v1.0.1-beta`.

The audit covered the paper, repository tree/history, releases, submodule
declaration, Android/Kotlin code, JNI/llama.cpp integration, prompt/template,
generation and post-processing, memory trigger, memory worker, plaintext store,
HNSW retrieval, model assets, and available evaluation/training artifacts.

The final classification is **B. Faithful HuoziIME reference-backend
adaptation**. The authoritative per-component A/B/C/D/E classification and
deviations are in the Phase 4F reproduction matrix. This wording is deliberate:
the personalisation backend is faithfully adapted, but the Android product and
unpublished experimental/training artifacts are not reproduced.

## Scope and Seven Core Capabilities

1. **LLM candidate generation.** The official post-trained Q4_0 checkpoint
   directly produces short completions. No Luna candidates or Phase 4E reranker
   enter the path. Frozen release decoding values are Top-K 20, Top-P 0.8,
   temperature 0.7, repeat penalty 1.2 over 16 tokens, and at most 8 generated
   tokens. Desktop candidate branches are sequential, with deterministic
   recorded benchmark seeds.
2. **Plaintext personal memory.** L2 is an authoritative, append-only per-user
   JSONL store. Every record has a stable ID, plaintext, chronology, source
   interaction IDs, activity state, vector label, and provenance.
3. **Semantic retrieval.** The official BGE Q8_0 asset produces 512-dimensional
   vectors. Per-user HNSW uses inner product over L2-normalized vectors,
   `max_elements=2048`, `M=16`, `ef_construction=200`, and `ef_search=64`.
   Retrieval uses Top-20, a raw-cosine threshold of 0.4, and the audited
   vector/lexical reranking rule before supplying one memory.
4. **Selective retrieval.** The public merged checkpoint is available, so its
   emitted `<MEM_RETRIEVAL> query="..." </MEM_RETRIEVAL>` action is treated as
   `OFFICIAL_POLICY`; the official runtime parser is adapted exactly. A
   separately named no-retrieval fallback exists for test/non-official runtimes
   and is labelled `ARCHITECTURAL_FALLBACK`.
5. **Memory-grounded generation.** A selected plaintext memory is inserted into
   the official `<memory>` prompt section and the LLM generation path is rerun.
   If retrieval finds nothing, `<NO_MEM>` is supplied. IDs, text, scores, raw
   outputs, prompts hashes, seeds, candidates, and timings are traced.
6. **Hierarchical state.** L2 plaintext/HNSW is fully addressable. L3 stores
   chronological prediction and background-processing traces. Mobile L1
   per-style/per-memory KV blobs and optional KV-splice are omitted; ordinary
   resident-runtime cache telemetry is not claimed as L1 equivalence.
7. **Foreground/background separation.** `predict` never extracts or rebuilds
   memory. A separate background processor consumes completed training
   trajectories, runs the official memory-worker prompt/schema, embeds accepted
   memories, and updates the per-user index. Final evaluation opens this state
   frozen and performs no updates.

## Model and Checkpoint Status

The official release APK is checksum-pinned and kept outside Git. Its runtime-
selected generation asset is
`scirime_grpo_v2_744-q4_0.gguf` (468,700,896 bytes, SHA-256
`2012b7aa860674e5f2b9fc0c90cc4828b7e5f50f7be4069fa0122685956416a5`).
GGUF metadata identifies Qwen3, Q4_0, 40,960-token training context, and the
name `Scirime_Grpo_V2_744_`.

The embedding asset is `bge-small-zh-v1.5-q8_0.gguf` (26,472,640 bytes,
SHA-256 `5a88d266870fbd27c6f329df60de80e2d4cf3bbd5e6f080bd5c1b2e5abb12039`),
with BERT architecture, 512-token context, 512 dimensions, and Q8_0
quantization.

The released model is therefore used as the official post-trained checkpoint,
but it is an opaque APK artifact: no independent model repository/revision,
model card, or complete reproducible training pipeline is public. The bundled
12,000-row `mixed_robust.jsonl` lacks sufficient provenance and runnable
training configuration, so it is not presented as a training reproduction.

## Desktop Adaptation

The backend runs through `llama-cpp-python==0.3.16` with Metal on macOS 26.2,
Apple M1 (7-core GPU), 8 GB RAM, arm64. Generation Q4_0 and embedding Q8_0 are
unchanged. This is algorithm/runtime adaptation, not exact Android runtime
reproduction. Phone-specific scheduling, latency, KV-splice, and frontend
behaviour are not claimed.

The engineering smoke verified checksum-bound model loading, real embedding,
HNSW-to-plaintext mapping, and a real grounded completion. The smoke is not a
benchmark result. Practical desktop timings remain hardware-specific and are
stored only in the smoke artifact.

## Input-Only Benchmark Mode

The benchmark supplies only the user's own preceding text, normalized Pinyin
or keystrokes, and frozen personal state. `external_context` is always `None`;
the adapter rejects a non-null value and never constructs imaginary partner
messages.

The context is the final 100 characters of each interaction's `raw_context`,
strictly before the target, matching the audited Android
`getTextBeforeCursor(100)` path. The upstream LLM prompt does not expose a
separate Pinyin decoder field, so Pinyin is not inserted into that prompt. It is
consumed by the separate conventional Pinyin decoder described below.

“This thesis evaluates the input-only operating mode; cross-application
conversation-context synchronisation is outside the benchmark scope.”

## Pinyin Integration Correction

The pre-evaluation Phase 4F.1 audit found a task mismatch in the first desktop
adapter: `normalized_pinyin` was accepted and traced, but only preceding text
entered HuoziIME generation. This finding does not invalidate the HuoziIME
personalisation reproduction. It shows that HuoziIME's LLM is a contextual
completion capability, not the conventional keystroke decoder.

The pinned Android source confirms that the original product contains both:

1. YuyanIME `RimeEngine`/Rime JNI for ordinary Pinyin-to-Chinese conversion and
   the standard candidate bar; and
2. HuoziIME generation, optional personal-memory retrieval, and a separate
   GhostText/AI suggestion surface.

No official numerical fusion rule or common candidate score was found. Phase
4F.1 therefore preserves separate channels and records overlap/provenance rather
than forcing Pinyin candidates and completions into a fabricated rank.

The local Pinyin layer is a **faithful desktop adaptation** using the project's
pinned `librime 1.17.0_2`, Luna Pinyin schema, `zh_hans` engine option (OpenCC
`t2s.json`), and Top-10 candidate iterator order. It genuinely consumes the
normalized tone-free Pinyin. It is not claimed to reproduce Yuyan's exact
dictionary or order: the APK's `pinyin` assets were compiled with Rime 1.11.2,
its Android ELF library cannot run on macOS, the compiled table did not load in
the desktop librime build, and the source dictionary is not public.

The HuoziIME prompt remains unchanged and still receives preceding text rather
than raw Pinyin. Results expose `PINYIN_DECODER`, `HUOZIIME_DIRECT`, and
`HUOZIIME_MEMORY_GROUNDED` sources; identical text across channels retains all
sources and grounded memory IDs.

Final evaluation is separated into three layers:

- Pinyin conversion: Top-1/3/5/10, MRR, coverage, and explicit missing targets
  from decoder ranks only;
- HuoziIME personalisation: trigger, retrieval, grounded-generation, direct vs
  grounded output, correct/wrong-user differences, memory provenance, and
  latency, without requiring an unconstrained completion to equal the Pinyin
  target;
- integrated backend: decoder-only and decoder plus generic/correct/wrong-user
  HuoziIME, retaining separate channels and reporting overlap rather than fake
  unified Top-K metrics.

No final Phase 4F test result was observed before or during this correction.

## Frozen Personal State and Isolation

- Correct-user memory: Zhu Ziqing Phase 4C training interactions only.
- Wrong-user memory: Lu Xun Phase 4C training interactions only.
- Test: unchanged Zhu `to_my_late_wife` and `spring` interactions.
- Generic condition: same model, runtime, prompt context, decoding, and seed,
  with an empty personal store.

Training interactions are grouped into chronological per-work trajectories and
capped at the upstream 4,000-character buffer before the official memory-worker
prompt. The prepared state contains four indexed Zhu memories and four indexed
Lu memories; each has traceable source interaction IDs. Both test-overlap counts
are zero. There are no test-time updates, future reads, retraining, or mixed
user stores.

## Evaluation and Transparency

The manual final command compares decoder-only, decoder plus generic/no memory,
decoder plus correct-user Zhu memory, and decoder plus wrong-user Lu memory.
The Pinyin decoder is run once per interaction and its result is identical
across memory conditions. Exact target Top-1/3/5/10, MRR, coverage, and missing
count belong only to Pinyin conversion. HuoziIME reports trigger/retrieval/
grounded-generation rates, output differences, provenance, and latency.

Every prediction records query/user/input, external context, candidate text,
rank, seed, raw generation, mean token log-probability when exposed, trigger
decision/method/raw evidence, retrieved memory IDs and vector/lexical scores,
source interaction IDs, supplied plaintext, model hashes/runtime, prompt hashes,
raw action/final outputs, timing, cache status, and path. The decision is
inspectable without hidden global mutable state.

The final evaluation has not been run automatically. Its output is reserved for
manual execution at `results/experiments/phase_04f/evaluation.json`.

## Omitted and Unavailable Components

Omitted as mobile/frontend only: Android keyboard/settings UI, GhostText,
companion chat, MCP transport, Android lifecycle/sandbox integration,
phone-specific scheduling, exact CursorKV/session snapshots, per-memory KV
blobs, and optional KV-splice.

Not reproducible from public artifacts: paper evaluation data/results, complete
post-training data provenance and runnable scripts/configuration, an independent
checkpoint repository/revision, the declared paper submodule SHA, and pinned
upstream revisions for vendored llama.cpp/hnswlib.

## Completion Criteria and Gates

Implementation, deterministic tests, asset preparation, and the engineering
smoke are complete. Gates 1 and 3–13 pass; Gate 2 passes with the explicit
limitation that upstream did not record vendored dependency revisions. Phase
4F remains non-final until the user manually runs and reviews the final
evaluation.

## Known Limitations and Non-Claims

- The Zhu/Lu benchmark is not equivalent to the paper's evaluation setup.
- Exact-target metrics are not assigned to unconstrained HuoziIME completions;
  Pinyin target metrics are computed only from the conventional decoder.
- The desktop Luna dictionary is not bit-identical to Yuyan's compiled APK
  dictionary, so candidate ordering may differ from the Android product.
- The smoke query did not emit the optional retrieval action; HNSW retrieval
  and grounded generation were therefore also exercised explicitly with the
  official models. No trigger or parameter was tuned in response.
- Desktop latency must not be compared numerically with mobile latency.
- This phase does not demonstrate performance improvement or research success.
- This phase does not implement Phase 5 control operations or UI.

Future phases may use the preserved memory IDs, source provenance, active state,
index mapping, and deterministic prediction API to omit, deactivate, remove,
rebuild, reset, and counterfactually evaluate personal state.
