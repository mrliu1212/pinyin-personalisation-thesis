# PinyinGPT Technical Audit and Minimal Reproduction

## Scope and sources

This audit concerns PinyinGPT2-Concat as a **generic contextual Pinyin-to-Chinese
candidate-ranking backend**. It does not implement personalisation or define the
thesis evaluation protocol.

Primary sources:

- Tan et al., [*Exploring and Adapting Chinese GPT to Pinyin Input Method*](https://aclanthology.org/2022.acl-long.133/), ACL 2022;
- official [VisualJoyce/Transformers4IME](https://github.com/VisualJoyce/Transformers4IME) repository, audited at commit `8f1573ed0bd4d1f3d8d3f10a05f7e870725646f1`;
- project-linked [PinyinGPT2-Concat checkpoint](https://huggingface.co/aihijo/transformers4ime-pinyingpt-concat), revision `76dd20dc92d8236a350fb732e99dde6fa15e2263`.

## Finding

**Suitability: B — strong fit with minor adaptation.** PinyinGPT2-Concat
directly consumes preceding Chinese context plus segmented Pinyin and ranks
fixed-length, Pinyin-compatible Chinese sequences. This is substantially closer
to the thesis task than free-form completion. Minor adaptation is required for
modern inference, explicit fixed-candidate scoring, raw-Pinyin segmentation at
a product boundary, and eventually expanding the candidate pool with personal
vocabulary.

**Reproduction difficulty: MODERATE.** Real inference is reproducible, but the
old package cannot be installed unchanged on Python 3.12 and the full published
benchmark data/configuration is incomplete.

## What PinyinGPT2-Concat is

The checkpoint is a character-level decoder-only GPT-2 language model:

- 12 transformer layers;
- hidden size 768;
- 12 attention heads;
- context limit 1,024 positions;
- 21,571 effective tokens: the 21,128-token BERT Chinese vocabulary plus 443
  published Pinyin special tokens;
- 102,408,960 parameters (about 102M, approximately 0.1B);
- tied input and output embeddings.

The paper starts from a standard 12-layer character-level Chinese GPT, further
pretrains its stronger generic model on a private 800GB corpus, and then trains
PinyinGPT-Concat with Pinyin context and a Pinyin-constrained loss. The
available checkpoint uses standard `GPT2LMHeadModel` tensors and a
`BertTokenizer` character vocabulary.

### Exact input and prediction

For segmented Pinyin `shi yong`, the implemented segmented Concat prompt is:

```text
[CLS] preceding Chinese context [SEP] [shi] [yong] [SEP]
```

“Concat” therefore means horizontal token-sequence concatenation of all Pinyin
tokens after the Chinese context. It is not concatenation of hidden vectors.
Generated target characters are autoregressive and use position IDs aligned
with the Pinyin-token positions, matching the official training and inference
code. The target length equals the number of segmented Pinyin elements, and
the model predicts one Chinese character per element. It does not generate an
unconstrained longer continuation in this mode.

Only preceding context and the current Pinyin are present when generation
starts. There is no future/target input. When scoring a supplied candidate,
teacher forcing exposes only that candidate's earlier characters to score its
next character, which is the ordinary autoregressive factorization and does not
leak later characters.

The paper and implementation assume **oracle Pinyin segmentation**. Raw input
such as `jianshi` can mean `jian shi` or `ji an shi`; segmentation was explicitly
left outside the work. The adapter accepts a raw spelling only if the published
vocabulary yields one segmentation and otherwise requires spaces.

## Constrained decoding and candidate ranking

At output step `i`, the official implementation obtains the GPT logits and
masks every vocabulary entry except characters listed in `pinyin2char.json` for
Pinyin element `p_i`. Pinyin compatibility is therefore enforced at every
position, not checked after generation.

The official benchmark uses beam search with beam size 16 and returns at most
10 sequences. Candidate order is beam order by cumulative autoregressive log
probability. All candidates contain the same number of characters, so any
common length normalization does not change their ordering. Logits and token
log-probabilities are available from the model; the historical JSON output does
not save beam scores, but the modern adapter exposes cumulative and mean token
log-probabilities.

The code's inference beam applies `log_softmax` over the full vocabulary and
then masks incompatible tokens. The adapter deliberately reproduces that
criterion. The paper's Pinyin-constrained *training loss* instead normalizes
over the compatible-character subset; these two facts should not be conflated.

## Fixed-candidate scoring

An arbitrary candidate can receive an exact comparable generic score when:

1. it has one tokenizer character per segmented Pinyin element; and
2. every character occurs in the checkpoint's compatibility set for the
   corresponding Pinyin element.

For candidate `w_1...w_k`, the adapter computes:

```text
sum_i log P_model(w_i | preceding context, all Pinyin, w_<i)
```

using the same Concat prompt, aligned positions, learned weights, and
full-vocabulary token probabilities used to order official beams. This is exact
teacher-forced sequence log-probability, not an approximation. It makes scores
comparable for `使用`, `实用`, `适用`, and `试用` under the same `shi yong` query.

A compatible personal candidate omitted by normal beam search can therefore be
scored and inserted into a future expanded pool, provided all its characters
are represented by the tokenizer and compatibility map. This checkpoint does
not itself solve unseen-character, nonstandard-pronunciation, word-level-token,
or segmentation problems.

## Future extension points (not implemented)

- **External reranking:** cleanly supported. Preserve the frozen generic
  sequence log-probability and combine it later with separately computed
  personal evidence.
- **Personal vocabulary augmentation:** feasible for compatible character
  sequences by adding candidates before fixed-candidate scoring. Truly absent
  characters or pronunciations require an explicit vocabulary policy.
- **Output-logit or decoding bias:** technically direct immediately before the
  compatibility mask/beam update, but its semantics and training must be
  designed and evaluated later.
- **Adapters or other internal updates:** standard GPT blocks provide clean
  insertion points, but this changes learned inference and is not selected.
- **User representation:** not supported by the frozen input format. It could
  later enter through new conditioning tokens, embeddings, or adapter inputs,
  which would require trained architectural adaptation rather than a zero-cost
  inference switch.

No one intervention is preferred or frozen by this audit.

## Official benchmark audit

The paper evaluates complete Pinyin-to-Chinese **sequence conversion**, not a
single-character candidate bar. P@K is the percentage of examples whose exact
full target sequence appears among the top K generated beams. It is analogous
to thesis Top-K ranking at the candidate-sequence level, but the paper's targets
can be much longer and its results do not measure user personalisation.

### Datasets and settings

- **PD:** People's Daily material from 1992–1998. The paper reports 5.04M
  training segments and 2,000 test segments. Test context is empty and input is
  perfect Pinyin. The repository README links a Baidu archive and expects
  `data/benchmarks/PD/samples_0.json`, but no PD data is committed.
- **WD:** a paper-created evaluation set sampled from WuDaoCorpora: 15 domains,
  nine context/target length configurations, 2,000 examples per configuration,
  270,000 examples total. It tests perfect and abbreviated Pinyin. Context
  ranges are 0–3, 4–9, and 10+ words; target ranges use the same bins. WD is not
  committed, and `benchmarks.sh` points to private `/apdcephfs/...` paths.
- **TP:** neither the paper nor official repository defines a TP dataset. It
  must not be attributed to this work without another primary source.

Abbreviation modes in code are perfect Pinyin (`none`), first Pinyin complete
and remaining syllables initial-only (`xone`), and every syllable initial-only
(`full`). The paper evaluates perfect and fully abbreviated settings. It assumes
oracle segmentation and uses beam size 16. Metrics are exact-sequence P@1,
P@5, and P@10. It does not report MRR, mean rank, coverage, user comparisons,
or chronology.

### Recoverability

**Benchmark recoverability: PARTIAL.** The model, tokenizer, compatibility map,
beam logic, metric calculation, and PD/WD protocol are public. The official
repository does not contain either dataset, the exact WD sample JSON files, or
portable benchmark commands; its script embeds private filesystem paths. The
README's Baidu bundle may permit PD recovery, but the complete 270K WD sample
set and exact sampling provenance are not available in the audited Git tree.
Consequently the paper tables cannot presently be claimed as fully
reproducible from Git plus the checkpoint alone.

## Checkpoint and environment audit

The canonical project-linked artifact is
`aihijo/transformers4ime-pinyingpt-concat`, uploaded by the repository author
account shown on the model history. It contains:

- `pytorch_model.bin` — PyTorch pickle state dict, 488,536,999 bytes;
- `config.json` — GPT-2 architecture metadata;
- `vocab.txt` and BERT tokenizer metadata;
- `additional_special_tokens.json` — 443 Pinyin tokens;
- `pinyin2char.json` — Pinyin/abbreviation compatibility map.

Checkpoint license metadata states **CC BY-NC-SA 4.0**. The source files carry
MIT header notices, but the repository has no root license file; redistribution
or broader use should therefore be reviewed conservatively.

Checkpoint SHA256 values used in the smoke run:

```text
pytorch_model.bin  1c5ebb9e7b15d75ea8899b914fc8363f4745703115253071f7834780263c74bb
config.json         86bb492283d576cc845cd2e9f2b67f8e07423a546659c5f1d427f146dd492cb4
vocab.txt           f7863b040bae29ac474065729355252248c92d41141c1e09fbf21dd3e593a238
pinyin2char.json     f344e2dd29253b50b8cc7a512b793996427d1400d7c2d0c79e0e12ff3628e142
```

The historical environment pins Python-incompatible or obsolete dependencies:
PyTorch 1.10.2/CUDA 11.3, Transformers 4.15, `dataclasses`, Horovod, old
WebDataset, and removed Transformers generation import paths/APIs. Training
also depends on private corpora, TexSmart paths, and multi-node infrastructure.

For inference, no upstream source patch is necessary. The modern adapter uses
PyTorch `2.11.0+cu128` and Transformers `4.57.6` on Python 3.12. Two packaging
compatibility steps are required:

1. `config.json` records the base vocabulary size 21,128, while the learned
   embedding/head tensors contain 21,571 rows. Adding the published 443 tokens
   in file order and setting the in-memory config to 21,571 resolves this with
   exact tensor shapes.
2. Twenty-four serialized GPT-2 causal-mask buffers from Transformers 4.15 are
   omitted during loading because modern GPT-2 recreates those non-learned
   buffers. Every learned parameter then loads strictly.

The RTX 4060 8GB is comfortably sufficient: the model is about 102M parameters
(roughly 0.41GB in FP32 before runtime overhead), and the real smoke test ran on
CUDA 12.8. Training the paper configuration is a separate, much larger task and
was not attempted.

## Minimal reproduction

Download and run:

```powershell
git clone https://huggingface.co/aihijo/transformers4ime-pinyingpt-concat .build/pinyingpt2-concat
.\.venv\Scripts\python.exe -m pip install -r requirements-pinyingpt.txt
.\.venv\Scripts\python.exe -m experiments.exp_pinyingpt_reference
```

The Windows CUDA smoke artifact records exact prompts, segmented Pinyin,
ranked candidates, scores, beam configuration, hashes, and runtime. It is an
engineering reproduction only—not the final thesis benchmark.
