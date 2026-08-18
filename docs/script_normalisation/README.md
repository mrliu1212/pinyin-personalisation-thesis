# Chinese Script Normalisation Audit

**Status:** Known dataset/evaluation issue — formal repair deferred  
**Date identified:** 2026-08-18  
**Affected baseline:** Frozen Dataset V1 and experiments derived from it  
**Planned resolution:** Unified Simplified-Chinese canonicalisation before final formal evaluation

---

## 1. Summary

During preparation of the Context Personalisation Lab, an audit was performed to determine whether the frozen Dataset V1 and the PinyinGPT candidate generator use a consistent Chinese writing system.

The audit found a substantial Simplified/Traditional Chinese inconsistency.

This is not merely a cosmetic corpus property. It directly affects exact-string evaluation because a Traditional-Chinese Gold target and its Simplified-Chinese equivalent are currently treated as different candidates.

For example:

```text
Gold:      觸發
Candidate: 触发

Gold:      點擊
Candidate: 点击

Gold:      離開
Candidate: 离开
```

Under the existing exact-string evaluation:

```text
觸發 != 触发
點擊 != 点击
離開 != 离开
```

even though the candidate represents the same lexical item and the same intended Pinyin input.

The issue is also strongly author-dependent. In particular, MScarlet uses substantially more Traditional Chinese than the other five authors, creating a potential author-specific script confound in a personalisation experiment.

For this reason:

- the existing Dataset V1 and all existing results are preserved unchanged as historical baselines;
- the current exploratory Context Lab temporarily excludes MScarlet;
- final formal evaluation will return to all six authors after script normalisation;
- a new script-normalised dataset version will be derived from frozen Dataset V1 rather than overwriting it.

---

## 2. Why This Matters

The thesis studies personalised Chinese Pinyin input.

The main intended personalisation signals are lexical and contextual preferences, such as:

```text
使用 vs 实用
different words sharing the same Pinyin
different Full-Pinyin expansions under Initial input
context-dependent candidate preferences
personal vocabulary
frequency and recency
```

Chinese script preference itself is not currently intended to be the principal personalisation variable.

If one author writes predominantly in Traditional Chinese while most other authors write predominantly in Simplified Chinese, the personalisation system may partially learn:

> this author tends to use Traditional characters

instead of learning only the lexical/contextual preferences that the experiment is intended to study.

Script therefore becomes an unintended proxy for author identity.

The problem may affect several components of the current pipeline:

- Generic candidate accuracy;
- Missing@K;
- frequency history;
- personal vocabulary construction;
- exact historical target matching;
- ambiguity statistics;
- conflict subsets;
- context-memory retrieval and matching;
- final personalised evaluation.

Therefore, script consistency should be treated as an evaluation/preprocessing requirement rather than a minor display preference.

---

## 3. Frozen Dataset V1 Audit

The audited dataset was the reconstructed frozen Dataset V1:

```text
C:\Users\chiar\Desktop\LBH\thesis-deep-author\.build\dataset-v1-reconstruction\data\processed\deep_author\interactions_t1_ready.jsonl
```

Total interactions:

```text
1,074,032
```

OpenCC `t2s` was applied in read-only mode to each row's:

- `gold`
- `context`

The original dataset was not modified.

### 3.1 Overall Results

| Metric | Count | Rate |
|---|---:|---:|
| Total interactions | 1,074,032 | 100% |
| Gold changed by t2s | 102,982 | 9.5884% |
| Context changed by t2s | 226,817 | 21.1183% |
| Both Gold and context changed | 102,614 | 9.5541% |

Thus, almost one in ten Gold targets in Dataset V1 contains text affected by the selected Traditional-to-Simplified canonicalisation policy.

More than one fifth of all interaction contexts are also changed.

---

## 4. Per-Author Dataset Results

| Author | Rows | Gold t2s-change | Gold rate | Context t2s-change | Context rate |
|---|---:|---:|---:|---:|---:|
| Agent Phage | 272,621 | 81 | 0.0297% | 5,055 | 1.8542% |
| Etinjat | 211,794 | 987 | 0.4660% | 33,136 | 15.6454% |
| MScarlet | 167,596 | 96,653 | 57.6702% | 165,310 | 98.6360% |
| QBLevi | 65,196 | 7 | 0.0107% | 546 | 0.8375% |
| Re_spectators | 66,289 | 91 | 0.1373% | 2,375 | 3.5828% |
| breaddddd | 290,536 | 5,163 | 1.7771% | 20,395 | 7.0198% |

The distribution is highly uneven.

The most important case is MScarlet:

```text
Gold t2s-change:     57.6702%
Context t2s-change:  98.6360%
```

In comparison, all other authors have Gold t2s-change rates below 2%.

This makes Chinese script strongly correlated with author identity in the frozen Dataset V1.

---

## 5. Example Dataset Changes

Examples observed during the audit include:

```text
第一分部門 → 第一分部门
門         → 门
蒐集       → 搜集
```

These examples also show that OpenCC canonicalisation is not always a simple one-character graphical substitution.

For example:

```text
蒐集 → 搜集
```

is a lexical/variant normalisation performed by the selected OpenCC configuration.

For this reason, the statistics in this document should be described as:

> t2s-change rate

rather than interpreted as a perfect linguistic classifier for the percentage of Traditional Chinese.

---

## 6. PinyinGPT Compatibility Dictionary

The PinyinGPT compatibility dictionary used by the local frozen system was found at:

```text
C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat\pinyin2char.json
```

It contains both Simplified and Traditional character forms.

Examples include:

```text
wei  → 韦 / 韋
wei  → 为 / 為 / 爲
men  → 们 / 們
wang → 网 / 網
xue  → 学 / 學
xi   → 习 / 習
```

Therefore, the Pinyin compatibility layer itself does not enforce a Simplified-only output space.

A character can be Pinyin-compatible regardless of whether its surface form is Simplified or Traditional.

---

## 7. Actual PinyinGPT Candidate-Surface Audit

The analysis was then extended from the compatibility dictionary to actual candidate surfaces appearing in an existing Full + Short / H5000 prediction artifact.

Audited candidate occurrences:

```text
58,091
```

Results:

| Candidate category | Count | Rate |
|---|---:|---:|
| Simplified-marked | 24,009 | 41.33% |
| Traditional-marked | 1,649 | 2.84% |
| Both-changing / complex variant | 226 | 0.39% |
| Script-neutral | 32,207 | 55.44% |
| Total | 58,091 | 100% |

Examples of Traditional-marked candidate surfaces include:

```text
盃 → 杯
併 → 并
鈣 → 钙
熱 → 热
內 → 内
説 → 说
說 → 说
碩 → 硕
是因爲 → 是因为
```

The actual PinyinGPT output space is therefore mixed-script but strongly biased toward Simplified Chinese rather than being strictly Simplified-only.

---

## 8. Direct Effect on Generic Missing@10

A direct evaluation audit was performed on the existing Full + Short Test set.

Test rows:

```text
6,000
```

Original exact-string Missing@10:

```text
538
```

For each missing case, Gold and every candidate were independently canonicalised using OpenCC `t2s`.

A case was counted as script-equivalent recoverable when:

```text
Gold not in original candidates
```

but:

```text
t2s(Gold) == t2s(candidate)
```

for at least one candidate.

Results:

| Metric | Count / Rate |
|---|---:|
| Test rows | 6,000 |
| Exact Missing@10 | 538 |
| t2s-equivalent candidate found | 186 |
| Share of Missing explained by t2s equivalence | 34.57% |
| Share of all Test rows | 3.10% |

Thus:

```text
538 exact Missing cases
       ↓
186 contain a t2s-equivalent candidate
       ↓
352 remain missing after equivalence-only canonicalisation
```

The corresponding diagnostic Missing count would change from:

```text
538 → 352
```

if script-equivalent candidates were treated as the same canonical target.

This demonstrates that the mixed-script issue materially affects the current evaluation rather than being only a corpus-formatting concern.

---

## 9. Examples of False Missing Cases

Observed examples include:

```text
Gold: 並在
t2s:  并在
Candidates: 併在, 并在
```

```text
Gold: 變幻
t2s:  变幻
Candidate: 变幻
```

```text
Gold: 觸發
t2s:  触发
Candidate: 触发
```

```text
Gold: 金麥
t2s:  金麦
Candidate: 金麦
```

```text
Gold: 咒語
t2s:  咒语
Candidate: 咒语
```

```text
Gold: 點擊
t2s:  点击
Candidate: 点击
```

```text
Gold: 離開
t2s:  离开
Candidate: 离开
```

These examples are evaluated as incorrect by the current raw exact-string comparison.

---

## 10. Current Experimental Decision

The script issue was discovered through a dataset-level preprocessing audit, independently of comparative personalisation-method Test performance.

For the current exploratory Context Personalisation Lab, the three exploratory authors are therefore:

```text
Etinjat
Re_spectators
breaddddd
```

MScarlet is temporarily excluded.

This is not a permanent author exclusion and is not based on selecting authors according to favourable model performance.

The reason is the independently identified script confound:

```text
MScarlet Gold t2s-change     = 57.6702%
MScarlet context t2s-change = 98.6360%
```

Using MScarlet in the current mixed-script exploratory diagnostic could make it difficult to distinguish genuine context-personalisation behaviour from Chinese-script effects.

After formal script normalisation, final evaluation must return to all six authors.

---

## 11. Existing Results Policy

No existing result should be overwritten.

The following existing experimental families remain historical artifacts of the mixed-script Dataset V1 configuration:

```text
Generic T1
Frequency F
M1
M2
Personal Vocabulary
Initial-condition analyses
previous diagnostic results
```

They should be preserved with their existing caches, outputs, logs, commits and tags.

They should not silently be relabelled as results from a script-normalised dataset.

If the formal dataset is changed, it must receive a new explicit dataset/version identifier and new experiment output directories.

---

## 12. Planned Formal Repair

The intended future pipeline is:

```text
Frozen Dataset V1
        ↓
new versioned derived dataset
        ↓
OpenCC t2s canonicalisation
        ↓
context
Gold
history targets
        ↓
Simplified-Chinese canonical interaction representation
        ↓
PinyinGPT raw generation
        ↓
OpenCC t2s(candidate)
        ↓
canonical candidate deduplication
        ↓
Top-K unique Simplified-Chinese candidates
        ↓
Generic / F / M1 / M2 / PV
        ↓
formal six-author evaluation
```

The new dataset should be derived directly from frozen Dataset V1.

It should **not** reintroduce the earlier Dataset V1.1 metadata-deletion changes, because those changes had separate concerns regarding possible removal of legitimate source text.

The new revision should therefore isolate script normalisation from unrelated corpus-cleaning changes.

---

## 13. Candidate Canonicalisation Policy

Simply deleting every Traditional candidate is not the preferred solution.

For example:

```text
raw candidate: 觸發
canonical:     触发
```

The model has identified the correct lexical candidate even though it generated a different script from the chosen canonical representation.

Therefore the preferred operation is:

```text
candidate
→ OpenCC t2s
→ canonical candidate
```

followed by deduplication.

Example:

```text
raw ranking:

1. 觸發
2. 触发
3. A
4. B

canonicalised:

1. 触发
2. 触发
3. A
4. B

deduplicated:

1. 触发
2. A
3. B
```

Ranking information associated with equivalent surfaces should be preserved according to a documented deterministic rule.

Because canonicalisation can reduce the number of unique candidates, the final implementation may need to generate more than K raw candidates and continue until K unique canonical candidates are available.

This behaviour must be specified before the new formal evaluation is run.

---

## 14. Reporting Guidance

For thesis/report writing, preferred terminology is:

- mixed-script Dataset V1;
- Simplified/Traditional Chinese inconsistency;
- OpenCC t2s canonicalisation;
- t2s-change rate;
- script-equivalent candidate;
- script-normalised dataset version.

Avoid interpreting:

```text
t2s(text) != text
```

as a perfect classifier meaning:

```text
text is Traditional Chinese
```

because OpenCC may also perform character-variant or lexical mappings.

Likewise, the PinyinGPT candidate categories used in this audit are diagnostic categories rather than a complete linguistic taxonomy of Chinese script.

The strongest empirical conclusion currently supported is:

> Frozen Dataset V1 contains substantial and strongly author-dependent text that changes under the planned t2s canonicalisation policy, and this mismatch materially affects exact candidate evaluation.

The Full + Short diagnostic specifically found that 186 of 538 exact Missing@10 cases (34.57%) already contain a candidate equivalent to Gold after t2s canonicalisation.

---

## 15. Next Action

Do not modify or overwrite Dataset V1.

Planned future work:

1. freeze the script-normalisation design;
2. create a new versioned dataset/output namespace;
3. generate the script-normalised dataset;
4. audit the transformed dataset before any expensive model run;
5. audit PinyinGPT canonical candidate behaviour;
6. obtain human approval;
7. rerun the formal six-author evaluation.

---

# Audit Code

The sections below record the code used to obtain the audit evidence.

The implementation is deliberately separated from the interpretation above.

---

## Code 1 — Frozen Dataset V1 Script Audit

File used during the audit:

```text
experiments/context_lab/audit_dataset_v1_script.py
```

```python
import json
from collections import Counter, defaultdict
from pathlib import Path
from opencc import OpenCC

DATASET = Path(
    r"C:\Users\chiar\Desktop\LBH\thesis-deep-author\.build\dataset-v1-reconstruction"
    r"\data\processed\deep_author\interactions_t1_ready.jsonl"
)

cc = OpenCC("t2s")

total = 0
gold_changed = 0
context_changed = 0
both_changed = 0

by_author = defaultdict(lambda: Counter(
    total=0,
    gold_changed=0,
    context_changed=0,
    both_changed=0,
))

gold_examples = []
context_examples = []

with DATASET.open("r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        author = str(row["author_name"])
        gold = str(row["gold"])
        context = str(row["context"])

        gold_t2s = cc.convert(gold)
        context_t2s = cc.convert(context)

        g = gold_t2s != gold
        c = context_t2s != context

        total += 1
        by_author[author]["total"] += 1

        if g:
            gold_changed += 1
            by_author[author]["gold_changed"] += 1

            if len(gold_examples) < 20:
                gold_examples.append(
                    (author, gold, gold_t2s)
                )

        if c:
            context_changed += 1
            by_author[author]["context_changed"] += 1

            if len(context_examples) < 20:
                context_examples.append(
                    (author, context[-100:], context_t2s[-100:])
                )

        if g and c:
            both_changed += 1
            by_author[author]["both_changed"] += 1

        if total % 100000 == 0:
            print(
                f"rows scanned = {total:,}",
                flush=True
            )

print("\n=== DATASET V1 SCRIPT AUDIT ===")

print("rows =", total)

print(
    "gold_changed =",
    gold_changed,
    f"({gold_changed / total:.4%})"
)

print(
    "context_changed =",
    context_changed,
    f"({context_changed / total:.4%})"
)

print(
    "both_changed =",
    both_changed,
    f"({both_changed / total:.4%})"
)

print("\n=== BY AUTHOR ===")

for author in sorted(by_author):
    x = by_author[author]

    print(
        author,
        "rows=", x["total"],
        "gold_changed=", x["gold_changed"],
        f"({x['gold_changed'] / x['total']:.4%})",
        "context_changed=", x["context_changed"],
        f"({x['context_changed'] / x['total']:.4%})",
    )

print("\n=== GOLD EXAMPLES ===")

for x in gold_examples:
    print(repr(x))

print("\n=== CONTEXT EXAMPLES ===")

for x in context_examples:
    print(repr(x))
```

Execution command:

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
    experiments\context_lab\audit_dataset_v1_script.py |
Tee-Object `
    -FilePath results\personalisation\context_lab\script_audit_v1\dataset_v1_script_audit.log
```

---

## Code 2 — PinyinGPT Candidate-Surface Audit

This audit classified actual candidate occurrences using OpenCC `t2s` and `s2t`.

```python
import json
from opencc import OpenCC

PREDICTIONS = (
    r"C:\Users\chiar\Desktop\LBH\thesis-personalisation"
    r"\results\personalisation\pilot_a_context_memory"
    r"\h5000\memory_predictions.jsonl"
)

t2s = OpenCC("t2s")
s2t = OpenCC("s2t")

candidate_occurrences = 0
traditional_marked = 0
simplified_marked = 0
both_changing = 0
script_neutral = 0

traditional_examples = []
simplified_examples = []

with open(PREDICTIONS, encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        for candidate_row in row["candidates"]:
            text = candidate_row["candidate"]

            candidate_occurrences += 1

            changes_t2s = t2s.convert(text) != text
            changes_s2t = s2t.convert(text) != text

            if changes_t2s and not changes_s2t:
                traditional_marked += 1

                if len(traditional_examples) < 15:
                    traditional_examples.append(
                        (text, t2s.convert(text))
                    )

            elif changes_s2t and not changes_t2s:
                simplified_marked += 1

                if len(simplified_examples) < 15:
                    simplified_examples.append(
                        (text, s2t.convert(text))
                    )

            elif changes_t2s and changes_s2t:
                both_changing += 1

            else:
                script_neutral += 1

print("=== CANDIDATE SURFACE AUDIT ===")

print(
    "candidate_occurrences =",
    candidate_occurrences
)

print(
    "simplified-marked =",
    simplified_marked,
    f"({simplified_marked / candidate_occurrences:.2%})"
)

print(
    "traditional-marked =",
    traditional_marked,
    f"({traditional_marked / candidate_occurrences:.2%})"
)

print(
    "both-changing =",
    both_changing,
    f"({both_changing / candidate_occurrences:.2%})"
)

print(
    "script-neutral =",
    script_neutral,
    f"({script_neutral / candidate_occurrences:.2%})"
)

print(
    "traditional examples =",
    traditional_examples
)

print(
    "simplified examples =",
    simplified_examples
)
```

---

## Code 3 — Missing@10 t2s-Equivalence Audit

```python
import json
from opencc import OpenCC

PREDICTIONS = (
    r"C:\Users\chiar\Desktop\LBH\thesis-personalisation"
    r"\results\personalisation\pilot_a_context_memory"
    r"\h5000\memory_predictions.jsonl"
)

t2s = OpenCC("t2s")

rows = 0
exact_missing = 0
t2s_equivalent_found = 0
examples = []

with open(PREDICTIONS, encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        rows += 1

        gold = row["gold"]

        candidates = [
            item["candidate"]
            for item in row["candidates"]
        ]

        if gold not in candidates:
            exact_missing += 1

            matches = [
                candidate
                for candidate in candidates
                if t2s.convert(candidate) == t2s.convert(gold)
            ]

            if matches:
                t2s_equivalent_found += 1

                if len(examples) < 20:
                    examples.append(
                        (
                            gold,
                            t2s.convert(gold),
                            matches,
                        )
                    )

print("=== MISSING@10 SCRIPT-EQUIVALENCE AUDIT ===")

print("rows =", rows)
print("exact_missing =", exact_missing)

print(
    "t2s_equivalent_found =",
    t2s_equivalent_found
)

print(
    "share_of_missing =",
    (
        f"{t2s_equivalent_found / exact_missing:.2%}"
        if exact_missing
        else None
    )
)

print(
    "share_of_all =",
    f"{t2s_equivalent_found / rows:.2%}"
)

print("examples =")

for example in examples:
    print(example)
```