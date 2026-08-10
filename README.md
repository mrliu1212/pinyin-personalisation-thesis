# Transparent, User-Controllable Pinyin Personalisation

## Current Phase

Phase 4C — Personalisation Evaluation

## Current Objective

The repository is prepared to evaluate whether transparent, user-specific
history improves future Chinese Pinyin candidate ranking. It compares:

1. the original Luna `zh_hans` Base ranking;
2. correct-user personalisation from earlier Zhu Ziqing interactions;
3. wrong-user personalisation from earlier Lu Xun interactions.

The experiment setup is complete. The final comparison has not yet been run,
so the repository does not claim that personalisation improves performance.

## Research Design

```text
(context, tone-free Pinyin)
        ↓
Luna zh_hans Top-10 candidates
        ├── Base order
        ├── Zhu pre-1930 history → correct-user reranking
        └── Lu pre-1930 history  → wrong-user reranking
                                ↓
             later Zhu test interactions
                                ↓
        full + rerankable metrics and evidence traces
```

The wrong-user condition tests whether any observed change is specific to Zhu
history rather than a generic benefit from accumulating more text statistics.

## Accepted Data Representation

Both authors use the same processing path:

- revision-pinned Chinese Wikisource raw pages;
- conservative text cleaning;
- OpenCC Traditional-to-Simplified conversion without overwriting raw text;
- Jieba default segmentation;
- normalized, tone-free pypinyin;
- Luna Pinyin candidates with engine-side `zh_hans` conversion;
- Top-10 candidate order and source/normalization provenance.

The accepted Zhu Phase 4B.6 benchmark remains unchanged at 4,691 interactions.
The separate Lu Xun dataset contains 7,593 interactions.

## Frozen Chronological Splits

No random split is used.

| User | History/train works | Test/held-out works | Counts |
| --- | --- | --- | ---: |
| Zhu Ziqing | `congcong`, `qinhuai_river`, `beiying`, `ahe`, `moonlight_over_lotus_pond` (before 1930) | `to_my_late_wife`, `spring` (after 1930) | 3,765 / 926 |
| Lu Xun | `madmans_diary`, `kong_yiji`, `medicine`, `hometown`, `new_years_sacrifice` (before 1930) | `takeism`, `have_chinese_lost_self_confidence` (1934) | 7,014 / 579 |

Only the Lu training partition is used for wrong-user personalisation. Both
personal models are frozen before the first Zhu test work; neither Zhu test
interactions nor Lu held-out interactions are added to history.

## Personal Model and Base Scores

The unchanged interpretable personal model combines:

- global candidate frequency, weight `0.1`;
- exact-Pinyin candidate frequency, weight `0.3`;
- exact-context-plus-Pinyin candidate frequency, weight `0.6`.

The interpolation remains `alpha=0.5`; these are configuration defaults, not
optimised parameters.

Librime supplies an ordered candidate list but no numeric score through the
current adapter. Phase 4C therefore represents that order as the ordinal
utility `candidate_count - base_rank + 1` before the existing min-max
interpolation. This preserves Base order and is not a probability estimate.

## Evaluation Outputs

For the full Zhu test benchmark and the Base-rerankable subset, each condition
reports:

- Top-1, Top-3, Top-5, and Top-10 accuracy;
- Mean Reciprocal Rank;
- mean target rank;
- explicit missing-target counts.

Personalised conditions also report improved, unchanged, and harmed counts
relative to Base. The rerankable subset isolates ranking behavior because a
reranker cannot recover targets absent from the candidate generator.
Missing targets count as incorrect for Top-K and as zero reciprocal rank; mean
target rank is calculated over present targets only.

Selected examples preserve context, Pinyin, target, Base rank/ordinal score,
global/Pinyin/context evidence, combined personal score, final score, and final
rank.

## How to Run

Install the existing Phase 4 toolchain if needed:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-phase4b.txt
brew install opencc librime
.venv/bin/python -m interactions.setup_rime
make rime-adapter
```

Rebuild the Lu corpus and interactions from the checked-in revision-pinned raw
responses:

```bash
.venv/bin/python -m corpus.prepare_phase_04c_lu_xun
```

To deliberately refresh the raw pages to current Wikisource revisions, add
`--acquire`; that is not required for reproduction of the accepted setup.

Run all tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

After reviewing and accepting the setup, run the final Phase 4C comparison:

```bash
.venv/bin/python -m experiments.exp_phase_04c_personalisation
```

The command prints both subsets and writes the full configuration, metrics, and
transparency examples to `results/experiments/phase_04c/evaluation.json`.

## Current Limitations

- Both author corpora use a small, curated selection of works.
- The wrong-user authors differ in genre and historical vocabulary as well as
  personal style.
- Exact context matching may be sparse.
- Evidence weights and interpolation have not been optimised.
- Ordinal Base utility does not encode unknown confidence gaps between Luna
  candidates.
- Base Top-10 misses cannot be recovered by reranking.
- Phase 4B.7 manual audit labels are not used to filter the benchmark.

## Project History

- Phase specifications: [`docs/phases/`](docs/phases/)
- Completed setup/outcome summaries: [`results/phases/`](results/phases/)
- Data-quality audits: [`results/audits/`](results/audits/)
- Workflow rules: [`docs/WORKFLOW.md`](docs/WORKFLOW.md)

Git tags are not created automatically.
