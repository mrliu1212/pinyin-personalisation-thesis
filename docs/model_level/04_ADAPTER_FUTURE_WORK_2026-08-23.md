# Model-Level Adapter Future Work

## Unified Multi-Condition Adapter Training

A unified Adapter that jointly trains on Full, Initial, and Mixed Pinyin
together with Short and Multi1-Multi5 targets is intentionally deferred
from the current study.

The current Model-Level Adapter experiments retain a controlled and
comparatively narrow training protocol so that the contribution of
model-level personalisation can be evaluated without introducing a large
additional hyperparameter search over heterogeneous training mixtures.

Future work should investigate controlled training mixtures across:

- Full Pinyin
- Initial / abbreviated Pinyin
- Mixed Full / Initial Pinyin
- Short targets
- Multi1
- Multi2
- Multi3
- Multi4
- Multi5

Rather than allowing the natural frequency of each condition to determine
the training distribution, these conditions could be assigned explicit,
fixed exposure proportions.

One possible design would separately control:

1. the proportion of Short versus Multi training examples;
2. the proportions of Multi1-Multi5 within the Multi portion; and
3. the proportions of Full, Initial, and Mixed Pinyin within each
   target-length condition.

A key research question is whether broader multi-condition training improves
robustness across Pinyin input styles and target lengths without degrading
performance on the standard Full-Pinyin condition.

Useful controlled ablations would include:

1. Full-only versus Full+Initial+Mixed training;
2. Short-only versus joint Short+Multi training; and
3. alternative fixed exposure ratios across Pinyin modes and Multi target
   lengths.

These experiments are deliberately outside the current Adapter V1 scope in
order to avoid expanding the present thesis into a large training-mixture
hyperparameter study.

## Hybrid Extension

A future unified Adapter could also serve as the model-level component of a
hybrid personalisation system.

In such a system:

1. the personalised Adapter changes the internal Transformer
   representations and therefore the candidate-generation distribution;
2. Adapter-specific candidates are generated using the normal autoregressive
   decoding procedure; and
3. external-memory personalisation subsequently reranks those candidates.

This would allow model-level personalisation to influence candidate
generation while explicit user-memory signals influence the final ranking.

The resulting system would provide a natural basis for studying whether
model-level and external-memory personalisation capture complementary user
signals.
