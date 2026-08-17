# Candidate-Aware Personal Memory M2

## Motivation

M1 retrieves histories whose preceding contexts have high general BGE cosine
similarity. It does not directly ask whether a historical selection supports a
particular candidate now. M2 tests a stronger pretrained, candidate-aware
second stage while preserving the frozen T1/M1 population, chronology,
candidate pool, and Generic score.

## Architecture

Stage 1 takes the 5,000 most recent strictly prior legal History records for
the same author, then filters exact segmented Pinyin. The existing pinned
`bge-small-zh-v1.5-q8_0.gguf` embeddings and cosine ordering retrieve Top-K
history. K is selected on Dev from `{10, 20}`. K=10 is a deterministic prefix
of K=20.

Stage 2 uses the pretrained Cross-Encoder `BAAI/bge-reranker-base`, revision
`2cfc18c9415c912f9d8155881c133215df768a70`, under the MIT license. The frozen
weight artifact is `model.safetensors`, SHA-256
`ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd`.
The tokenizer is `XLMRobertaTokenizerFast`; `tokenizer.json` has SHA-256
`9eb652ac4e40cc093272bbbe0f55d521cf67570060227109b5cdc20945a4489e`.
Inference uses Transformers 4.57.6, PyTorch 2.11.0+cu128, CUDA, and float16.

## Candidate-Aware Input

Template version `candidate-aware-current-history-v1` is:

```text
[CURRENT]
context: <current preceding context>
pinyin: <current segmented Pinyin>
candidate: <candidate>

[HISTORY]
context: <historical preceding context>
selected: <historical selected target>
```

For each retrieved interaction, the candidate field is its historical selected
target. Thus the current state, candidate, historical context, and historical
selection jointly participate in the Cross-Encoder score. Current Gold, future
text, future interactions, Test labels, and author name are absent.

## Input Length

The tokenizer and model limit is 512 tokens. Truncation version
`paired-recent-context-balanced-v1` first reserves all template labels, current
Pinyin, current candidate, and historical selected target. The remaining token
budget is split equally between current and historical context, with unused
capacity transferred to the other context. Only the most recent context tokens
are retained. Because SentencePiece decode/encode can expand a suffix slightly,
the implementation then removes additional oldest context tokens in a fixed
balanced order until the serialized pair is at most 512 tokens. Mandatory
fields are never truncated, and both context truncation counts are recorded.

## Support and Final Score

The raw Cross-Encoder logit is mapped by the monotonic non-negative sigmoid.
Within each query:

```text
raw_support(h) = sigmoid(cross_encoder_logit(h))
M2_support(c) = sum(raw_support(h) where historical_target(h) = c)
                / sum(raw_support(h) over all retrieved h)
Score_M2(c) = Z_generic(c) + lambda_m2 * M2_support(c)
```

Histories whose target is outside the Generic surface remain in the
normalizing denominator but cannot add a candidate. If there is no history,
M2 returns exact Generic order. Final-score ties preserve Generic rank.
`lambda_m2` is selected on Dev from `{0.5, 1, 2, 4}` by Macro-author Top-1;
ties choose lower lambda and then lower K. The Generic population z-score is
unchanged from M1.

## Pair-Score Cache

Raw logits are stored in a provenance-checked SQLite cache. A key includes the
current interaction ID and context, segmented Pinyin, historical interaction ID
and context, historical selected target, candidate, reranker revision and
weight SHA-256, tokenizer SHA-256, template version, truncation version, and
maximum length. It excludes split labels, history-budget names, profile names,
and run numbers. Compatible H500, H5000, HFull, T3, or later analyses can
therefore reuse the same tuple. Cache provenance mismatch stops reuse, and
`INSERT OR IGNORE` makes interruption/resume idempotent.

## Information and Selection Boundary

Only the 16,171-row chronological Dev-tune partition selects K and lambda.
Runtime batch size is chosen from a real DEV-only semantic-equivalence
benchmark, not accuracy. Test remains untouched until selection is written.
The final evaluation reuses the exact 6,000 T1 Full+Short anchors and completed
G0/F/M1 artifacts. It asserts identical candidate sets, Missing@10 status, and
History-available/Ambiguous/Conflict row membership.

## Limitations

The reranker is pretrained rather than trained for this task, proxy authors are
not observed IME users, reconstructed Pinyin remains a dataset limitation, and
fixed candidates cannot recover personal vocabulary. M2 does not test H500,
HFull, wrong-user history, temporal adaptation, M3 training, transparency, or
user control.

## Completed H5000 Result

M2-H5000 completed on the exact 6,000 frozen T1 Full+Short Test anchors. Dev
Macro-author Top-1 selected `retrieval_k = 20` and `lambda_m2 = 4.0` from the
frozen grids. Selection used 16,171 chronological Dev-tune rows and zero Test
rows.

Overall Macro-author Top-1 was `0.765`, compared with
`0.7231666666666667` for G0, `0.7718333333333334` for F-H5000, and
`0.7675000000000001` for M1-H5000. M2 also remained below Frequency and M1 on
History Available and Ambiguous rows, and achieved `0.1643947485347861` on
Conflict. Candidate and Missing@10 invariance held: every ranking-only method
had 538 missing targets.

M2 is therefore a completed negative and diagnostic result. A stronger
pretrained candidate-aware semantic scorer did not automatically learn a
stronger personal preference signal. This supports the next additive research
stage: candidate-set personalisation through bounded Personal Vocabulary. The
completed report is [Personalisation M2-H5000](../reports/05_personalisation_m2_h5000.md).
