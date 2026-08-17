# Project Version History

This is a navigation index for frozen research and software checkpoints. Detailed protocols and results remain in their linked reports.

| Checkpoint | Purpose and main change | Commit or tag | Detailed record |
| --- | --- | --- | --- |
| Deep Author Dataset V1 | Initial six-author corpus and interaction build used as the Evaluation V2 development source | `deep-author-dataset-preparation-v1` / `d886b65` | [Dataset preparation](reports/01_dataset_preparation.md) |
| Deep Author Dataset V1.1 | Preserved targeted corpus-cleaning correction; not substituted into frozen T1 | `deep-author-dataset-preparation-v1.1` / `d871f1f` | [Dataset V1.1](reports/01b_dataset_preparation_v1_1.md) |
| Evaluation V2 Design | Frozen chronological six-author, 6,000-anchor, four-condition T1 protocol | `deep-author-evaluation-v2-design` / `b145f2d` | [Evaluation V2 design](reports/02_deep_author_evaluation_v2.md) |
| IME Simulator v0.1 | Full-Pinyin-only interactive CUDA simulator checkpoint | `ime-simulator-v0.1` / `6750bee` | `ime-simulator-v0.1:docs/tools/ime_simulator.md` |
| IME Simulator v0.2 | Unified mixed Full/Abbreviated Pinyin constraints, parsing, and one-search simulator integration | `ime-simulator-v0.2` / `ad69543` | `ime-simulator-v0.2:docs/research/mixed_pinyin_extension.md` |
| Evaluation V2 T1 Generic Baseline | Completed 24,000-condition Dataset V1 development baseline using semantic-equivalent KV-cache inference | `deep-author-evaluation-v2-t1`; implementation `8c608f1`, `5d270cd` | [T1 Generic PinyinGPT Baseline](reports/03_t1_generic_pinyingpt_baseline.md) |
| Personalisation Pilot A — Context-Aware Memory Implementation | Dev-only Full+Short chronological frequency/context-memory ranking, resumable caches, and manual runner; research results pending | branch `work/personalisation-pilot-a`; `f335715`, `22eb1e0`; `personalisation-pilot-a-implementation-v1` | [Method](research/context_aware_personal_memory.md) · [Pending report](reports/04_personalisation_pilot_a_context_memory.md) |
| Personalisation Pilot A — M1 H5000 / T1-aligned evaluation implementation | Dev-only parameter selection plus exact 6,000-anchor T1 Full+Short evaluation; read-only T1 G0 reuse and shared profile-neutral BGE cache; results pending manual run | branch `work/personalisation-pilot-a`; `personalisation-pilot-a-h5000-implementation-v1` | [Method](research/context_aware_personal_memory.md) · [Pending report](reports/04_personalisation_pilot_a_context_memory.md) |
