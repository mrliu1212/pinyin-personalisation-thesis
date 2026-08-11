# Phase 4E — Pretrained-Model Contamination Risk

Zhu Ziqing and Lu Xun are famous public authors whose works are widely
available online. Membership of the selected benchmark texts in the training
data of `Qwen/Qwen3-0.6B-Base` or `Qwen/Qwen3-Embedding-0.6B` is unknown.

Absolute neural-context performance on these historical works may therefore
partially reflect exposure during pretraining. Phase 4E must not be described
as a clean evaluation of generalisation to unseen text, and its semantic-model
results must be interpreted with this limitation visible.

No claim about dataset membership is made without evidence. Future validation
on genuinely held-out modern writing is recommended.
