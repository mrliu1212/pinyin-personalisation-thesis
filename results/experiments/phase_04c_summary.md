# Phase 4C — Baseline Personalisation Evaluation

## Research question

Does transparent frequency-based personalisation improve Chinese Pinyin
candidate ranking compared with a strong baseline IME?

## Experimental setup

Zhu Ziqing is the target user. His five pre-1930 works provide correct-user
history, and the 926 interactions in his two post-1930 works form the test
benchmark. Lu Xun's five pre-1930 works provide the wrong-user control.

The split is chronological, not random. Both personal models are frozen before
the first Zhu test work: no test interaction is added to history, and no future
interaction can affect a prediction. The same Zhu test interactions are scored
under Base, correct-user, and wrong-user conditions. Results are reported for
the full benchmark and for the subset whose target is present in Base Top-10.

## Results

### Full benchmark — 926 interactions

| Condition | Top-1 | Top-3 | Top-5 | Top-10 | MRR | Mean target rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 77.97% | 90.06% | 92.01% | 92.55% | 0.8401 | 1.2707 |
| Correct-user | 75.27% | 90.17% | 92.12% | 92.55% | 0.8277 | 1.2905 |
| Wrong-user | 74.95% | 90.17% | 92.01% | 92.55% | 0.8252 | 1.3011 |

| Personalised condition | Improved | Unchanged | Harmed |
| --- | ---: | ---: | ---: |
| Correct-user | 25 | 846 | 55 |
| Wrong-user | 24 | 849 | 53 |

### Rerankable subset — 857 interactions

| Condition | Top-1 | Top-3 | Top-5 | Top-10 | MRR | Mean target rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 84.25% | 97.32% | 99.42% | 100.00% | 0.9077 | 1.2707 |
| Correct-user | 81.33% | 97.43% | 99.53% | 100.00% | 0.8943 | 1.2905 |
| Wrong-user | 80.98% | 97.43% | 99.42% | 100.00% | 0.8917 | 1.3011 |

The 69 Base Top-10 misses remain in the full benchmark but are absent from the
rerankable subset. The reranker cannot introduce a candidate that Base did not
generate.

## Interpretation

The current transparent frequency model does not outperform the strong Luna
`zh_hans` baseline. Correct-user personalisation lowered full-benchmark Top-1
and MRR, with 55 harmed cases compared with 25 improved cases.

Frequency evidence alone is insufficient to model the distinctions required
by this benchmark. Correct-user and wrong-user results are also close: their
Top-1 and MRR values, as well as improved and harmed counts, differ only
slightly. This suggests that the current evidence behaves more like broad
lexical frequency than a strong representation of user-specific preference.

The context component is exact context-plus-Pinyin frequency. Repeated exact
contexts are uncommon in later prose, so context evidence is sparse and often
cannot distinguish candidates. The negative result is retained without
filtering difficult cases or hiding regressions.

## Limitations

- Context matching requires an exact match of the current context string.
- Contextual evidence is consequently sparse across chronologically separated
  works.
- Personalisation is based only on raw global, Pinyin, and exact-context
  frequencies.
- The model has no semantic-similarity representation.
- Candidate generation remains fixed, so personalisation cannot recover the 69
  targets absent from Base Top-10.

This summary records the completed Phase 4C result. It does not specify or
implement a replacement model.
