# LiveChat Personalised Pinyin Evaluation V1

## Scope

This document freezes the reusable **generic** LiveChat development benchmark.
It implements E0 only: Pinyin-only and ordinary same-response contextual
PinyinGPT2-Concat. It does not implement or evaluate personalisation.

LiveChat is used because its processed release attributes real-life Chinese
livestream responses to stable anonymized `streamer_id` values and contains a
deep-user population. For the IME interpretation, `streamer_id` is the user and
`streamer_response` is text produced by that user. `audience_comment` is retained
only as provenance metadata. It is excluded from the primary context because
the task simulates Chinese text already produced in the current response plus
the current Pinyin, not dialogue-response generation.

## Provenance and license caution

- Official repository: <https://github.com/gaojingsheng/LiveChat>
- Inspected revision: `d06c90aae0cedc1d75792c84e6bc140828c90ded`
- Processed-data source: the Google Drive folder linked by
  `Dataset/README.md`, folder ID
  `1q2GXfeNRN5bOr2Hc5aDneiBXXVfGN45V`
- Development source: `RawDialogueData/train_data.pk` only
- Official `RawDialogueData/dev_data.pk` and `test_data.pk` are audited but not
  used or scored.

The repository root contains an MIT license for software and associated
documentation. It does not explicitly distinguish or establish terms for the
third-party source content represented by the processed data. Therefore,
**dataset usage terms are not explicitly established by the repository license
alone**.

Exact downloaded filenames, byte sizes, SHA-256 values, object types, and row
counts are saved in `provenance.json` and `dataset_audit.json`.

## Released schema and chronology

Each official RawDialogueData pickle unpickles to a Python `list`. Every
inspected row is a three-element `list[str]`:

```text
[streamer_id, audience_comment, streamer_response]
```

There is no timestamp, sequence, session, or hidden order field. The official
repository contains downstream data loaders but not the dataset construction
or pickle-serialization code, and its documentation does not state that
within-streamer serialization order preserves the original timestamps. File
order may look plausible, but empirical appearance is not proof.

The frozen chronology grade is therefore **Grade C**. The split is explicitly
a **non-temporal proxy split**, never a chronological split. A stable SHA-256
hash with seed `40408` assigns each whole source response/session to the 70%
history or 30% evaluation partition. Target spans from one response cannot
cross partitions. RT4 and E7 temporal/prequential adaptation are unavailable
for this release.

## User and interaction selection

Usable depth is the number of non-empty `streamer_response` rows in official
`RawDialogueData/train_data.pk`, before target expansion. Users need at least
2,400 usable responses. At most 100 are selected by descending response count,
with lexical `streamer_id` as the deterministic tie break. Selection does not
use model performance.

Jieba `0.42.1`, default mode and dictionary, segments each evaluation-region
response and supplies exact character offsets. No custom or user vocabulary is
added. From every response with eligible spans, exactly one span is chosen by a
stable hash of streamer, stable source-response ID, span, and seed. Grade C uses
stable-hash sampling to retain at most 100 interactions per selected user.
This artifact is named **Frozen LiveChat Development Evaluation Set V1**.

The interaction ID hashes the source file, stable source-response ID, target
offsets, and target text. `frozen_interactions.jsonl` retains the full
segmentation, response and audience metadata, target, same-response preceding
context, effective model context, reconstructed Pinyin, ambiguity, script and
split provenance needed for later verification.

## Pinyin and context reconstruction

LiveChat contains final Chinese text, not keyboard logs. Pinyin is reconstructed
with pypinyin `0.55.0`, `Style.NORMAL`, `strict=True`, lowercase and tone-free,
with exactly one syllable per target character. Each character must occur in
the frozen checkpoint's `pinyin2char.json` entry for its reconstructed syllable.
Conversion, alignment, and compatibility failures are excluded and counted;
individual readings are never manually corrected after seeing scores.

PinyinGPT composition boundaries are reconstructed from Jieba spans, not
observed IME boundaries. Polyphonic flags come from pypinyin's deterministic
heteronym inventory and are audit flags, not corrected labels. Traditional text
is not converted to Simplified Chinese.

For a target at offsets `[start, end)`, primary context is exactly
`streamer_response[:start]`. It contains no future target/suffix text, audience
comment, later response, history, or profile. The official PinyinGPT benchmark
leading-context window of at most 512 tokenizer tokens is preserved, and the
effective context is persisted.

## Frozen generic model

- Checkpoint: `aihijo/transformers4ime-pinyingpt-concat`
- Revision: `76dd20dc92d8236a350fb732e99dde6fa15e2263`
- Official code audit revision:
  `8f1573ed0bd4d1f3d8d3f10a05f7e870725646f1`
- Beam size: 16
- Returned candidates: Top-10
- Exact gold score: cumulative full-vocabulary teacher-forced autoregressive
  log probability under the same condition, even when the gold is absent from
  the beam Top-10

Condition A uses empty context. Condition B uses only the effective preceding
same-response text. Model weights, candidate constraints, beam settings, and
all other inputs are identical. No user information enters either condition.

## Metrics and diagnostics

For both conditions, micro and equally weighted macro-user metrics are:

- Top-1, Top-3, Top-5 and Top-10/Coverage@10;
- MRR@10, with reciprocal rank zero when gold is absent;
- MeanRank|Top10, conditional on gold appearing in Top-10;
- Missing@10 count and rate.

No unrestricted rank, MRR, or mean rank is inferred for missing candidates.
The primary generic summary is contextual macro-user Top-1.

Context gain uses paired identical interactions. It reports metric differences,
rescued/harmed/both-correct/both-wrong counts, and a paired macro-user Top-1
bootstrap with 10,000 resamples, seed `40408`, labeled development analysis.

Context length bins are fixed at 0, 1-5, 6-10, 11-20 and 21+ effective Chinese
characters. Pinyin ambiguity is
`sum(log2(number of frozen-compatible characters at each position))`; quartile
boundaries are computed from frozen interactions before inference. Target
lengths are separate except that, starting at the maximum length, contiguous
longest lengths are merged into `N+` until the tail contains at least 100 frozen
interactions. This deterministic rule is based only on frozen input counts, not
outcomes.

## Future evaluation layers

- **E0 Generic baseline:** implemented now. Frozen Pinyin-only and contextual
  PinyinGPT Top-10 candidates, scores and exact gold scores.
- **E1 End-to-end personalised ranking:** future actual final ranking versus
  frozen contextual PinyinGPT, primary macro-user Top-1.
- **E2 Fixed generic pool reranking:** future reranking of today's exact
  contextual Top-10, restricted to cases where gold is present.
- **E3 Correct-user versus wrong-user:** future controls with matching based
  only on permitted non-evaluation information.
- **E4 History-attested recovery:** future compatible gold absent from generic
  Top-10 but attested for the same Pinyin in allowed same-user history.
- **E5 Same-Pinyin difficult subset:** future same-user Pinyin with at least two
  targets each observed at least twice before the event.
- **E6 History depth:** future fixed total-history and query-relevant history
  bins specified in the experiment protocol.
- **E7 Temporal/prequential adaptation:** unavailable for the Grade-C public
  processed release; fake time must not be simulated.
- **E8 Architecture comparison:** future frozen-benchmark comparison of
  separately specified personalisation architectures.

The Grade-C history partition can support non-temporal development studies for
E1-E6/E8, subject to a separately frozen personalisation method. It does not
validate temporal adaptation.

## Manual Windows PowerShell commands

Run these from the repository root. They use the locally verified virtualenv
Python executable.

Install exact additional dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-livechat-baseline.txt
```

Audit/download and prepare only:

```powershell
.\.venv\Scripts\python.exe -m experiments.exp_livechat_pinyingpt_generic_baseline --download --prepare-only
```

Run the full generic baseline (download resumes/skips existing files):

```powershell
.\.venv\Scripts\python.exe -m experiments.exp_livechat_pinyingpt_generic_baseline --download --resume
```

Resume an interrupted run without reacquiring the dataset:

```powershell
.\.venv\Scripts\python.exe -m experiments.exp_livechat_pinyingpt_generic_baseline --resume
```

Recompute metrics, plots, analysis and checksums from frozen predictions without
GPU inference:

```powershell
.\.venv\Scripts\python.exe -m experiments.exp_livechat_pinyingpt_generic_baseline --recompute-metrics
```

Optionally discard/recompute prediction outputs for both conditions under the
same configuration:

```powershell
.\.venv\Scripts\python.exe -m experiments.exp_livechat_pinyingpt_generic_baseline --force-recompute
```

## Limitations

The release lacks provable chronology. Pinyin and composition boundaries are
reconstructed. Polyphonic conversion is deterministic but not manually
disambiguated against real keystrokes. Public responses can include informal,
mixed-script and noisy livestream language. Audience comments are useful
dialogue metadata but intentionally excluded from the primary IME context.
The benchmark is development-only, capped at 100 deep users and 100 interactions
per user. It establishes no personalisation result and no final thesis evidence.
