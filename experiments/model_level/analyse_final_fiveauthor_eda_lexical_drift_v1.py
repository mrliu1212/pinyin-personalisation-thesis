#!/usr/bin/env python3

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "results/finalmodel_fiveauthor_v1"
    / "dataset_preparation_v1"
    / "final_fiveauthor_manifest_frozen_v1.jsonl"
)

OUT = (
    ROOT
    / "results/finalmodel_fiveauthor_v1"
    / "eda_v1"
)

EXPECTED_TOTAL = 210_000
EXPECTED_SPLITS = {
    "Fit": 150_000,
    "Validation": 20_000,
    "Test": 40_000,
}
EXPECTED_AUTHORS = {
    "Re_spectators",
    "Etinjat",
    "Agent Phage",
    "QBLevi",
    "breaddddd",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_split(v):
    s = str(v).strip().lower()
    mapping = {
        "fit": "Fit",
        "train": "Fit",
        "training": "Fit",
        "val": "Validation",
        "valid": "Validation",
        "validation": "Validation",
        "dev": "Validation",
        "test": "Test",
    }
    if s not in mapping:
        raise ValueError(f"Unknown split value: {v!r}")
    return mapping[s]


def first_present(row, names):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def get_author(row):
    v = first_present(
        row,
        [
            "author_name",
            "author",
            "proxy_user",
            "user_name",
            "user",
        ],
    )
    if v is None:
        raise KeyError(
            f"Could not find author field. Keys={sorted(row.keys())}"
        )
    return str(v).strip()


def get_split(row):
    v = first_present(
        row,
        [
            "split",
            "dataset_split",
            "partition",
        ],
    )
    if v is None:
        raise KeyError(
            f"Could not find split field. Keys={sorted(row.keys())}"
        )
    return norm_split(v)


def get_gold(row):
    v = first_present(
        row,
        [
            "gold",
            "target",
            "gold_target",
            "text",
        ],
    )
    if v is None:
        raise KeyError(
            f"Could not find gold field. Keys={sorted(row.keys())}"
        )
    return str(v).strip()


def get_full_pinyin(row):
    """
    Return canonical Full-Pinyin key as a space-separated string.

    Prefer full_segments because that is the natural syllable-level
    representation used by the final dataset.
    """
    v = first_present(
        row,
        [
            "full_segments",
            "full_pinyin",
            "pinyin_full",
            "canonical_full_pinyin",
        ],
    )

    if v is None:
        return None

    if isinstance(v, list):
        parts = [str(x).strip().lower() for x in v if str(x).strip()]
        return " ".join(parts) if parts else None

    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None

        # Some JSONL generators may serialize a list-like field as JSON text.
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    parts = [
                        str(x).strip().lower()
                        for x in parsed
                        if str(x).strip()
                    ]
                    return " ".join(parts) if parts else None
            except Exception:
                pass

        return " ".join(s.lower().split())

    return str(v).strip().lower() or None


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def js_divergence(counter_a, counter_b):
    """
    Jensen-Shannon divergence in bits.
    Range: [0, 1] for two discrete distributions using log2.
    """
    total_a = sum(counter_a.values())
    total_b = sum(counter_b.values())

    if total_a == 0 or total_b == 0:
        return None

    keys = set(counter_a) | set(counter_b)
    js = 0.0

    for k in keys:
        pa = counter_a.get(k, 0) / total_a
        pb = counter_b.get(k, 0) / total_b
        m = 0.5 * (pa + pb)

        if pa > 0:
            js += 0.5 * pa * math.log2(pa / m)
        if pb > 0:
            js += 0.5 * pb * math.log2(pb / m)

    return js


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


print("INPUT =", INPUT)

if not INPUT.exists():
    raise FileNotFoundError(INPUT)

OUT.mkdir(parents=True, exist_ok=True)

input_sha = sha256_file(INPUT)
print("INPUT_SHA256 =", input_sha)

# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------

rows = []
with INPUT.open(encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue

        raw = json.loads(line)

        author = get_author(raw)
        split = get_split(raw)
        gold = get_gold(raw)
        full_pinyin = get_full_pinyin(raw)

        if not gold:
            raise ValueError(f"Empty gold at line {line_no}")

        rows.append(
            {
                "author": author,
                "split": split,
                "gold": gold,
                "full_pinyin": full_pinyin,
            }
        )

print("ROWS =", len(rows))

# ------------------------------------------------------------------
# Strong frozen-dataset gates
# ------------------------------------------------------------------

if len(rows) != EXPECTED_TOTAL:
    raise AssertionError(
        f"Expected {EXPECTED_TOTAL} rows, got {len(rows)}"
    )

authors = sorted({r["author"] for r in rows})
if set(authors) != EXPECTED_AUTHORS:
    raise AssertionError(
        f"Unexpected authors: {authors}"
    )

split_counts = Counter(r["split"] for r in rows)
if dict(split_counts) != EXPECTED_SPLITS:
    raise AssertionError(
        f"Unexpected split counts: {dict(split_counts)}"
    )

author_counts = Counter(r["author"] for r in rows)
for author in EXPECTED_AUTHORS:
    if author_counts[author] != 42_000:
        raise AssertionError(
            f"{author}: expected 42000 rows, "
            f"got {author_counts[author]}"
        )

print("FROZEN_DATASET_GATE=PASS")

# ------------------------------------------------------------------
# Containers
# ------------------------------------------------------------------

target_counts = defaultdict(Counter)
target_sets = defaultdict(set)

# key = (scope, author)
# scopes: All, Fit, Validation, Test
for r in rows:
    author = r["author"]
    split = r["split"]
    gold = r["gold"]

    for scope in ("All", split):
        target_counts[(scope, author)][gold] += 1
        target_sets[(scope, author)].add(gold)

# split-level global
split_target_counts = defaultdict(Counter)
split_target_sets = defaultdict(set)

for r in rows:
    split_target_counts[r["split"]][r["gold"]] += 1
    split_target_sets[r["split"]].add(r["gold"])

# author x split
author_split_counts = defaultdict(Counter)
author_split_sets = defaultdict(set)

for r in rows:
    key = (r["author"], r["split"])
    author_split_counts[key][r["gold"]] += 1
    author_split_sets[key].add(r["gold"])

# ------------------------------------------------------------------
# 1. Author vocabulary summary
# ------------------------------------------------------------------

author_summary = []

for author in authors:
    for scope in ("All", "Fit", "Validation", "Test"):
        c = target_counts[(scope, author)]

        n_rows = sum(c.values())
        vocab_size = len(c)
        singleton_types = sum(1 for v in c.values() if v == 1)

        author_summary.append(
            {
                "author": author,
                "scope": scope,
                "N_rows": n_rows,
                "target_vocabulary_size": vocab_size,
                "type_token_ratio": (
                    vocab_size / n_rows if n_rows else ""
                ),
                "singleton_target_types": singleton_types,
                "singleton_type_share": (
                    singleton_types / vocab_size
                    if vocab_size
                    else ""
                ),
                "top1_target_frequency_share": (
                    max(c.values()) / n_rows
                    if n_rows
                    else ""
                ),
            }
        )

write_csv(
    OUT / "author_vocabulary_summary_v1.csv",
    [
        "author",
        "scope",
        "N_rows",
        "target_vocabulary_size",
        "type_token_ratio",
        "singleton_target_types",
        "singleton_type_share",
        "top1_target_frequency_share",
    ],
    author_summary,
)

# ------------------------------------------------------------------
# 2. Top targets by author
# ------------------------------------------------------------------

top_rows = []

for author in authors:
    for scope in ("All", "Fit", "Validation", "Test"):
        c = target_counts[(scope, author)]
        total = sum(c.values())

        for rank, (target, count) in enumerate(
            sorted(
                c.items(),
                key=lambda x: (-x[1], x[0]),
            )[:100],
            1,
        ):
            top_rows.append(
                {
                    "author": author,
                    "scope": scope,
                    "rank": rank,
                    "target": target,
                    "count": count,
                    "share": count / total if total else "",
                }
            )

write_csv(
    OUT / "author_top_targets_v1.csv",
    [
        "author",
        "scope",
        "rank",
        "target",
        "count",
        "share",
    ],
    top_rows,
)

# ------------------------------------------------------------------
# 3. Pairwise author vocabulary Jaccard
# ------------------------------------------------------------------

jaccard_rows = []

for scope in ("All", "Fit", "Validation", "Test"):
    for a, b in combinations(authors, 2):
        va = target_sets[(scope, a)]
        vb = target_sets[(scope, b)]

        jaccard_rows.append(
            {
                "scope": scope,
                "author_a": a,
                "author_b": b,
                "vocab_a": len(va),
                "vocab_b": len(vb),
                "intersection": len(va & vb),
                "union": len(va | vb),
                "jaccard": jaccard(va, vb),
            }
        )

write_csv(
    OUT / "author_vocabulary_jaccard_v1.csv",
    [
        "scope",
        "author_a",
        "author_b",
        "vocab_a",
        "vocab_b",
        "intersection",
        "union",
        "jaccard",
    ],
    jaccard_rows,
)

# ------------------------------------------------------------------
# 4. Distinctive author targets
#
# Simple interpretable score:
# log2(
#   (author_count + 1) / author_N
#   --------------------------------
#   (other_count + 1) / other_N
# )
#
# Only targets occurring >= 5 times for that author.
# This is exploratory ranking, not an inferential significance test.
# ------------------------------------------------------------------

all_author_counts = {
    a: target_counts[("All", a)]
    for a in authors
}

all_rows_per_author = {
    a: sum(all_author_counts[a].values())
    for a in authors
}

distinctive_rows = []

for author in authors:
    own = all_author_counts[author]
    own_n = all_rows_per_author[author]

    other = Counter()
    for b in authors:
        if b != author:
            other.update(all_author_counts[b])

    other_n = sum(other.values())

    ranked = []

    for target, own_count in own.items():
        if own_count < 5:
            continue

        other_count = other.get(target, 0)

        own_rate = (own_count + 1) / (own_n + 1)
        other_rate = (other_count + 1) / (other_n + 1)

        log2_ratio = math.log2(own_rate / other_rate)

        ranked.append(
            (
                log2_ratio,
                target,
                own_count,
                other_count,
                own_count / own_n,
                other_count / other_n,
            )
        )

    ranked.sort(key=lambda x: (-x[0], -x[2], x[1]))

    for rank, item in enumerate(ranked[:100], 1):
        (
            score,
            target,
            own_count,
            other_count,
            own_rate_raw,
            other_rate_raw,
        ) = item

        distinctive_rows.append(
            {
                "author": author,
                "rank": rank,
                "target": target,
                "author_count": own_count,
                "other_authors_count": other_count,
                "author_rate": own_rate_raw,
                "other_authors_rate": other_rate_raw,
                "log2_rate_ratio_smoothed": score,
            }
        )

write_csv(
    OUT / "author_distinctive_targets_v1.csv",
    [
        "author",
        "rank",
        "target",
        "author_count",
        "other_authors_count",
        "author_rate",
        "other_authors_rate",
        "log2_rate_ratio_smoothed",
    ],
    distinctive_rows,
)

# ------------------------------------------------------------------
# 5. Split vocabulary summary
# ------------------------------------------------------------------

split_summary_rows = []

for split in ("Fit", "Validation", "Test"):
    c = split_target_counts[split]
    n = sum(c.values())
    vocab = len(c)

    split_summary_rows.append(
        {
            "scope": "GLOBAL",
            "author": "ALL",
            "split": split,
            "N_rows": n,
            "target_vocabulary_size": vocab,
            "type_token_ratio": vocab / n if n else "",
            "singleton_target_types": sum(
                1 for x in c.values() if x == 1
            ),
        }
    )

    for author in authors:
        c2 = author_split_counts[(author, split)]
        n2 = sum(c2.values())
        vocab2 = len(c2)

        split_summary_rows.append(
            {
                "scope": "AUTHOR",
                "author": author,
                "split": split,
                "N_rows": n2,
                "target_vocabulary_size": vocab2,
                "type_token_ratio": (
                    vocab2 / n2 if n2 else ""
                ),
                "singleton_target_types": sum(
                    1 for x in c2.values() if x == 1
                ),
            }
        )

write_csv(
    OUT / "split_vocabulary_summary_v1.csv",
    [
        "scope",
        "author",
        "split",
        "N_rows",
        "target_vocabulary_size",
        "type_token_ratio",
        "singleton_target_types",
    ],
    split_summary_rows,
)

# ------------------------------------------------------------------
# 6. Split vocabulary overlap
# ------------------------------------------------------------------

split_overlap_rows = []

split_pairs = [
    ("Fit", "Validation"),
    ("Fit", "Test"),
    ("Validation", "Test"),
    ("Fit+Validation", "Test"),
]

def get_split_set(author, label):
    if label == "Fit+Validation":
        if author == "ALL":
            return (
                split_target_sets["Fit"]
                | split_target_sets["Validation"]
            )
        return (
            author_split_sets[(author, "Fit")]
            | author_split_sets[(author, "Validation")]
        )

    if author == "ALL":
        return split_target_sets[label]

    return author_split_sets[(author, label)]


for author in ["ALL"] + authors:
    for left, right in split_pairs:
        a = get_split_set(author, left)
        b = get_split_set(author, right)

        split_overlap_rows.append(
            {
                "author": author,
                "left_split": left,
                "right_split": right,
                "left_vocab": len(a),
                "right_vocab": len(b),
                "intersection": len(a & b),
                "union": len(a | b),
                "jaccard": jaccard(a, b),
                "right_seen_in_left_types": len(a & b),
                "right_unseen_in_left_types": len(b - a),
                "right_unseen_type_rate": (
                    len(b - a) / len(b)
                    if b
                    else ""
                ),
            }
        )

write_csv(
    OUT / "split_vocabulary_overlap_v1.csv",
    [
        "author",
        "left_split",
        "right_split",
        "left_vocab",
        "right_vocab",
        "intersection",
        "union",
        "jaccard",
        "right_seen_in_left_types",
        "right_unseen_in_left_types",
        "right_unseen_type_rate",
    ],
    split_overlap_rows,
)

# ------------------------------------------------------------------
# 7. Test unseen target rows
#
# Two references:
#   Fit only
#   Fit + Validation
# ------------------------------------------------------------------

unseen_target_rows = []

for author in authors:
    fit_vocab = author_split_sets[(author, "Fit")]
    fitval_vocab = (
        fit_vocab
        | author_split_sets[(author, "Validation")]
    )

    test_rows = [
        r
        for r in rows
        if r["author"] == author
        and r["split"] == "Test"
    ]

    for reference_name, reference_vocab in [
        ("Fit", fit_vocab),
        ("Fit+Validation", fitval_vocab),
    ]:
        unseen = [
            r
            for r in test_rows
            if r["gold"] not in reference_vocab
        ]

        unseen_types = {r["gold"] for r in unseen}

        unseen_target_rows.append(
            {
                "author": author,
                "reference_history": reference_name,
                "test_N": len(test_rows),
                "unseen_test_rows": len(unseen),
                "unseen_test_row_rate": (
                    len(unseen) / len(test_rows)
                    if test_rows
                    else ""
                ),
                "unseen_test_target_types": len(unseen_types),
            }
        )

# global totals are computed as author-conditioned unseen status
for reference_name in ("Fit", "Fit+Validation"):
    per_author = [
        x
        for x in unseen_target_rows
        if x["reference_history"] == reference_name
    ]

    test_n = sum(x["test_N"] for x in per_author)
    unseen_n = sum(x["unseen_test_rows"] for x in per_author)

    unseen_target_rows.append(
        {
            "author": "ALL",
            "reference_history": reference_name,
            "test_N": test_n,
            "unseen_test_rows": unseen_n,
            "unseen_test_row_rate": unseen_n / test_n,
            "unseen_test_target_types": "",
        }
    )

write_csv(
    OUT / "unseen_test_targets_v1.csv",
    [
        "author",
        "reference_history",
        "test_N",
        "unseen_test_rows",
        "unseen_test_row_rate",
        "unseen_test_target_types",
    ],
    unseen_target_rows,
)

# ------------------------------------------------------------------
# 8. Test unseen Full-Pinyin -> Gold pairs
#
# Pair definition:
#   (canonical full pinyin, gold target)
#
# Missing Full-Pinyin rows are reported separately.
# ------------------------------------------------------------------

pair_sets = defaultdict(set)

for r in rows:
    if r["full_pinyin"] is None:
        continue

    pair_sets[(r["author"], r["split"])].add(
        (r["full_pinyin"], r["gold"])
    )

unseen_pair_rows = []

for author in authors:
    fit_pairs = pair_sets[(author, "Fit")]
    fitval_pairs = (
        fit_pairs
        | pair_sets[(author, "Validation")]
    )

    test_rows = [
        r
        for r in rows
        if r["author"] == author
        and r["split"] == "Test"
    ]

    valid_test = [
        r
        for r in test_rows
        if r["full_pinyin"] is not None
    ]

    missing_pinyin = len(test_rows) - len(valid_test)

    for reference_name, reference_pairs in [
        ("Fit", fit_pairs),
        ("Fit+Validation", fitval_pairs),
    ]:
        unseen = 0

        for r in valid_test:
            pair = (r["full_pinyin"], r["gold"])
            if pair not in reference_pairs:
                unseen += 1

        unseen_pair_rows.append(
            {
                "author": author,
                "reference_history": reference_name,
                "test_N": len(test_rows),
                "test_rows_with_full_pinyin": len(valid_test),
                "test_rows_missing_full_pinyin": missing_pinyin,
                "unseen_test_pinyin_target_rows": unseen,
                "unseen_test_pinyin_target_rate": (
                    unseen / len(valid_test)
                    if valid_test
                    else ""
                ),
            }
        )

for reference_name in ("Fit", "Fit+Validation"):
    xs = [
        x
        for x in unseen_pair_rows
        if x["reference_history"] == reference_name
    ]

    valid_n = sum(
        x["test_rows_with_full_pinyin"]
        for x in xs
    )
    unseen_n = sum(
        x["unseen_test_pinyin_target_rows"]
        for x in xs
    )

    unseen_pair_rows.append(
        {
            "author": "ALL",
            "reference_history": reference_name,
            "test_N": sum(x["test_N"] for x in xs),
            "test_rows_with_full_pinyin": valid_n,
            "test_rows_missing_full_pinyin": sum(
                x["test_rows_missing_full_pinyin"]
                for x in xs
            ),
            "unseen_test_pinyin_target_rows": unseen_n,
            "unseen_test_pinyin_target_rate": (
                unseen_n / valid_n
                if valid_n
                else ""
            ),
        }
    )

write_csv(
    OUT / "unseen_test_pinyin_target_pairs_v1.csv",
    [
        "author",
        "reference_history",
        "test_N",
        "test_rows_with_full_pinyin",
        "test_rows_missing_full_pinyin",
        "unseen_test_pinyin_target_rows",
        "unseen_test_pinyin_target_rate",
    ],
    unseen_pair_rows,
)

# ------------------------------------------------------------------
# 9. Split frequency distribution shift via JSD
# ------------------------------------------------------------------

jsd_rows = []

for author in ["ALL"] + authors:
    for left, right in combinations(
        ["Fit", "Validation", "Test"],
        2,
    ):
        if author == "ALL":
            a = split_target_counts[left]
            b = split_target_counts[right]
        else:
            a = author_split_counts[(author, left)]
            b = author_split_counts[(author, right)]

        jsd_rows.append(
            {
                "author": author,
                "left_split": left,
                "right_split": right,
                "js_divergence_bits": js_divergence(a, b),
            }
        )

write_csv(
    OUT / "split_target_jsd_v1.csv",
    [
        "author",
        "left_split",
        "right_split",
        "js_divergence_bits",
    ],
    jsd_rows,
)

# ------------------------------------------------------------------
# Summary JSON
# ------------------------------------------------------------------

summary = {
    "analysis": "final_fiveauthor_eda_lexical_drift_v1",
    "input": str(INPUT.relative_to(ROOT)),
    "input_sha256": input_sha,
    "N": len(rows),
    "authors": authors,
    "split_counts": dict(split_counts),
    "author_counts": dict(author_counts),
    "definitions": {
        "vocabulary_unit": (
            "Exact Gold target string. This is target-span vocabulary, "
            "not an external Chinese word segmentation."
        ),
        "unseen_target": (
            "A Test Gold target absent from the same author's specified "
            "earlier split vocabulary."
        ),
        "unseen_pinyin_target_pair": (
            "A canonical Full-Pinyin + Gold pair absent from the same "
            "author's specified earlier split history."
        ),
        "jaccard": (
            "Set intersection divided by set union."
        ),
        "js_divergence_bits": (
            "Jensen-Shannon divergence between empirical target-frequency "
            "distributions, using log base 2."
        ),
        "distinctive_target_score": (
            "Smoothed log2 rate ratio of one author's target frequency "
            "against all other authors combined; exploratory ranking only."
        ),
    },
    "outputs": [
        "author_vocabulary_summary_v1.csv",
        "author_top_targets_v1.csv",
        "author_vocabulary_jaccard_v1.csv",
        "author_distinctive_targets_v1.csv",
        "split_vocabulary_summary_v1.csv",
        "split_vocabulary_overlap_v1.csv",
        "unseen_test_targets_v1.csv",
        "unseen_test_pinyin_target_pairs_v1.csv",
        "split_target_jsd_v1.csv",
    ],
    "gates": {
        "expected_total_210000": True,
        "expected_five_authors": True,
        "expected_split_counts": True,
        "expected_42000_per_author": True,
    },
}

with (
    OUT
    / "eda_lexical_drift_summary_v1.json"
).open("w", encoding="utf-8") as f:
    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2,
    )

print()
print("===== OUTPUTS =====")
for name in summary["outputs"]:
    print(OUT / name)

print(
    OUT
    / "eda_lexical_drift_summary_v1.json"
)

print()
print("EDA_LEXICAL_DRIFT_GATE=PASS")
