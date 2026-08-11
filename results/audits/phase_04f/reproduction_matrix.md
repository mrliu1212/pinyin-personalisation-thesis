# Phase 4F reproduction matrix

Audit target: HuoziIME at commit `63f249e711f6501169e6baafec7e12318b3c765b`, official release `v1.0.1-beta`, and the ACL 2026 demonstration paper. The machine-readable matrix is authoritative.

| Component | Class | Local treatment | Material limitation |
|---|---:|---|---|
| LLM base model | A | Official released merged checkpoint on desktop llama.cpp | Separate base checkpoint not shipped |
| Post-trained IME model | A | Official APK Q4_0 asset, checksum-pinned | Opaque APK asset; no model repository/card/training provenance |
| Prompt/template | A | Isolated, provenance-tracked upstream prompt and ChatML envelope | Input-only mapping sets no invented partner context |
| Candidate generation | B | Direct LLM completion with frozen upstream decoding values | Sequential desktop branches; deterministic recorded benchmark seeds |
| Special action tokens | B | Ported retrieval/no-memory parser | Kotlin/native parser adapted to Python |
| Memory trigger | B | Official checkpoint action plus ported runtime parser | Learned policy is opaque; fallback is separately named |
| Memory extraction | B | Official prompt/schema/checkpoint and validation | Run as explicit desktop preparation, not Android idle coroutine |
| L1 | D | Ordinary runtime cache metadata only | Mobile KV blobs/radix/KV-splice not reproduced or claimed |
| L2 | B | Per-user plaintext JSONL plus HNSW mapping | Adds user/provenance fields for research addressability |
| L3 | B | Separate chronological interaction/decision JSONL | No unavailable online fine-tuning pipeline |
| Plaintext memory | B | Stable, individually inspectable authoritative records | Adds source interaction IDs and active state |
| Embedding model | A | Official APK BGE Q8_0 asset | Desktop binding instead of JNI |
| HNSW | B | Python hnswlib, IP metric, L2 normalization, upstream parameters | Upstream vendored hnswlib SHA is absent |
| Memory-grounded generation | B | Retrieved plaintext is inserted and the LLM reruns | Mobile KV-splice omitted |
| Asynchronous memory update | B | Separate background processor; frozen evaluation state | Explicit desktop command instead of Android lifecycle |
| Quantization | A | Official Q4_0 generation and Q8_0 embedding files | Desktop kernels differ |
| KV/prefix caching | D | Resident-runtime reuse is only measured | No Android CursorKV/session equivalence claim |
| Mobile scheduling | D | Omitted | Phone-specific runtime only |
| Android UI | D | Omitted | Frontend only |
| MCP/chat context | D | Optional API field; benchmark forces `None` | Input-only benchmark does not test cross-app context |
| Pinyin decoder | B | Desktop librime 1.17.0_2, pinned Luna Pinyin, `zh_hans`, Top-10 | Same mature Rime conversion role, but not Yuyan's unpublished dictionary/order |
| Candidate-surface integration | B | Structured separate Pinyin and HuoziIME channels with multi-source provenance | Android widgets omitted; no unsupported shared score or fusion |
| Evaluation assets | E | No paper-result reproduction claim | Paper datasets and result artifacts are not public |
| Post-training pipeline/data | E | No replacement training | APK has an undocumented 12,000-row mixed dataset, but no complete provenance, scripts or configuration |
| Paper source submodule | E | ACL PDF audited directly | Declared submodule has no gitlink/SHA in the public tree |

## Audit conclusion

Recommended label: **B — Faithful HuoziIME reference-backend adaptation**.

The official release supplies an apparently runnable merged generation checkpoint and embedding checkpoint inside the APK. That supports direct reuse of the learned artifact, but not reproduction of its training. The desktop backend therefore adapts the public runtime architecture, keeps missing artifacts marked E, and makes no claim to reproduce Android mobile execution or the paper’s unavailable evaluation.

The Phase 4F.1 audit found that YuyanIME converts Pinyin with Rime while HuoziIME consumes text before the cursor and presents AI completions on a separate GhostText/AI surface. The corrected desktop backend now sends normalized Pinyin through a mature Luna/librime `zh_hans` decoder and keeps the unchanged HuoziIME path separate. Pinyin target metrics come only from decoder ranks; HuoziIME is measured through personalisation diagnostics. There is no invented unified ranking.

The exact APK `pinyin` dictionary could not be reused: its Android ELF runtime is not a macOS library, its Rime-1.11.2 compiled table is incompatible with the pinned desktop librime 1.17.0_2, and its source dictionary is not public. This deviation is why the label remains B rather than an exact reproduction claim.
