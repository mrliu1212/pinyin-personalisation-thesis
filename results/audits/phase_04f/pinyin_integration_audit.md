# Phase 4F.1 Pinyin integration audit

Audit date: 2026-08-11  
Pinned upstream: HuoziIME commit `63f249e711f6501169e6baafec7e12318b3c765b` and release `v1.0.1-beta`

## Conclusion

The official Android product has two cooperating but distinct paths. Ordinary
Chinese input is decoded by YuyanIME's Rime engine and shown through the normal
candidate bar. HuoziIME uses text before the cursor, plus optional contextual
and personal memory, to generate AI suggestions shown through GhostText and an
AI suggestion container. The audited source exposes no numerical score or
reranking function that fuses the two surfaces.

Phase 4F.1 therefore uses **separate channels**. It does not feed raw Pinyin to
the LLM and does not invent a shared score. The local integration remains
**B — Faithful HuoziIME reference-backend adaptation**.

## Explicit audit answers

1. **What converts Pinyin to Chinese?** YuyanIME's `RimeEngine`, through the
   `Rime` Kotlin/JNI boundary and the bundled `libyuyanime.so`, processes key
   events and reads `RimeContext.candidates`. The normal full-Pinyin schema ID
   is `pinyin`.
2. **Is it Rime/librime-based?** Yes. `Rime.kt` exposes the Rime lifecycle,
   key-processing, context, schema, candidate-selection, and option APIs. The
   release's compiled `pinyin.schema.yaml` records Rime `1.11.2`.
3. **Does HuoziIME replace or augment the decoder?** It augments the ordinary
   Rime IME. Rime remains the keystroke-to-Chinese converter; the LLM provides
   contextual continuation suggestions.
4. **One list or separate surfaces?** Separate surfaces in the audited source.
   `DecodingInfo.candidatesLiveData` updates the standard candidate bar.
   `ImeService.showAiSuggestion` calls `InputView.showAiSuggestion`, which
   maintains GhostText and a distinct AI suggestion container/list.
5. **Official numerical fusion rule?** None was found. No common score, weight,
   or shared sorting routine connects the ordinary candidate list and AI
   suggestions. Candidate overlap is possible, but it is not evidence of a
   unified rank.
6. **Does the LLM receive raw Pinyin/keystrokes?** No documented path does.
   `ImeLLMManager.requestCompletion` uses an explicit text prefix when supplied,
   otherwise `getTextBeforeCursor(100)`, then passes that text to the ChatML
   prompt builder. Rime key state is not added to this prompt.
7. **What is the LLM's role?** It predicts short contextual text completions,
   selectively requests personal-memory retrieval, and can rerun generation
   with retrieved plaintext memory. This is an auxiliary personalised
   completion capability alongside conventional Pinyin conversion.

## Decoder assets and desktop decision

The official APK bundles a compiled full-Pinyin setup:

- schema ID: `pinyin` (`全键拼音`);
- dictionary: bundled compiled `pinyin.table.bin` (the source dictionary and
  its independent version are not public);
- compiled schema Rime version: `1.11.2`;
- native engine: APK arm64 Android `libyuyanime.so`, SHA-256
  `e8b5c6f74ecebb005148ae55c25260f5344dbff36d1caa8ef93573aaea20feb8`;
- script mode: Simplified by default; the `traditionalization` option enables
  the `s2t.json` filter, and the app preference defaults to `false`;
- Android page size: the JNI wrapper requests 100 candidates while composing.

Direct desktop reuse was tested before choosing the local decoder. The APK's
Android native library is an ELF/Android binary and cannot be loaded as a macOS
library. Its compiled `pinyin.table.bin` also failed to load under the pinned
desktop librime `1.17.0_2`; the table decompiler reported an invalid Marisa
payload and a Rime session could not select the schema. Rebuilding the exact
Yuyan dictionary is impossible because its source dictionary is not published.

The local correction consequently uses the project's already established
desktop `librime 1.17.0_2` adapter, pinned Luna Pinyin sources, schema
`luna_pinyin`, and engine option `zh_hans` (OpenCC `t2s.json`), returning Top-10
candidates. This is functionally equivalent at the mature Rime full-Pinyin
decoder boundary—it consumes the normalized Pinyin and returns ordered Chinese
candidates—but it is **not** claimed to reproduce Yuyan's exact dictionary or
candidate order. Its status is `FAITHFUL DESKTOP ADAPTATION`.

## Source evidence

- `yuyansdk/.../inputmethod/core/Rime.kt`: JNI Rime lifecycle and key/context/
  candidate/schema APIs; page size 100.
- `yuyansdk/.../inputmethod/RimeEngine.kt`: keys enter Rime and normal candidates
  come from `getRimeContext().candidates`.
- `yuyansdk/.../inputmethod/core/Kernel.kt`: `traditionalization` and `emoji`
  Rime options.
- `yuyansdk/.../imemodule/prefs/AppPrefs.kt`: Traditional output defaults off.
- `yuyansdk/.../imemodule/service/DecodingInfo.kt` and
  `keyboard/InputView.kt`: normal candidate bar path.
- `yuyansdk/.../service/ImeService.kt` and
  `service/manager/ImeLLMManager.kt`: separate AI suggestion/GhostText path and
  text-before-cursor LLM input.
- Official APK `assets/rime/build/pinyin.schema.yaml`: schema, translator,
  dictionary, Traditional conversion option, and Rime build version.

No final Phase 4F evaluation was run during this audit.
