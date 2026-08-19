"""Same-surface Dev comparison:
G vs F vs Original M1 vs Hidden-M1.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.external_memory import em2_hidden_m1_dev as em2

from src.personalisation.context_memory import (
    macro_author_metrics,
    rank_frequency,
    rank_from_retrieved,
    rank_of,
    retrieve_memory,
    subset_membership,
)

from src.personalisation.pilot_a import (
    EmbeddingCache,
    EmbeddingLookup,
    HistoryIndex,
    PilotRunner,
)


PILOT_ROOT = Path(
    r"C:\Users\chiar\Desktop\LBH\thesis-personalisation"
    r"\results\personalisation\pilot_a_context_memory"
)

GENERIC_CACHE = (
    PILOT_ROOT
    / "cache"
    / "generic_predictions.jsonl"
)

BGE_CACHE = (
    PILOT_ROOT
    / "cache"
    / "embedding_cache.sqlite3"
)

HIDDEN_ROWS = Path(
    r"results\personalisation\external_memory"
    r"\em2_hidden_m1_dev_boundary8"
    r"\selected_rows.jsonl"
)

AUTHORS = em2.AUTHORS

HISTORY_BUDGET = 5000

# Frozen parameters.
LAMBDA_F = 4.0

ORIGINAL_M1_TOP_N = 5
ORIGINAL_M1_LAMBDA = 4.0

HIDDEN_M1_TOP_N = 3
HIDDEN_M1_LAMBDA = 4.0


def subset(rows, name):
    if name == "overall":
        return list(rows)

    return [
        row
        for row in rows
        if bool(row[name])
    ]


def macro_top1(rows, name):
    selected = subset(rows, name)

    return macro_author_metrics(
        selected,
        "rank",
    )["macro_author"]["top1"]


history = em2.read_jsonl(
    PILOT_ROOT / "history_manifest.jsonl"
)

dev = em2.read_jsonl(
    PILOT_ROOT / "dev_manifest.jsonl"
)

tune = [
    row
    for row in dev
    if (
        row.get("pilot_partition") == "tune"
        and str(row["author"]) in AUTHORS
    )
]

if len(tune) != 5608:
    raise RuntimeError(
        f"Unexpected tune size: {len(tune)}"
    )

tune_ids = {
    str(row["row_id"])
    for row in tune
}

generic = em2.load_generic(
    GENERIC_CACHE,
    tune_ids,
)

history_index = HistoryIndex(
    history + dev,
    HISTORY_BUDGET,
)

hidden_by_id = {}

with HIDDEN_ROWS.open(
    encoding="utf-8"
) as source:
    for line in source:
        if line.strip():
            row = json.loads(line)
            hidden_by_id[
                str(row["row_id"])
            ] = row

if len(hidden_by_id) != 5608:
    raise RuntimeError(
        "Hidden-M1 selected-row surface changed: "
        f"{len(hidden_by_id)}"
    )

g_rows = []
f_rows = []
m1_rows = []
hidden_rows = []

cache = EmbeddingCache(BGE_CACHE)
lookup = EmbeddingLookup(cache)

try:
    for number, row in enumerate(
        tune,
        start=1,
    ):
        query = PilotRunner._query(row)

        visible = history_index.visible(
            query
        )

        gold = em2.gold_of(row)

        candidates = PilotRunner._candidates(
            generic[query.row_id]
        )

        flags = subset_membership(
            query,
            gold,
            visible,
        )

        common = {
            "row_id": query.row_id,
            "author": query.author,
            "history_available": bool(
                flags["history_available"]
            ),
            "ambiguous": bool(
                flags["ambiguous"]
            ),
            "conflict": bool(
                flags["conflict"]
            ),
        }

        # G: use the Frozen Generic cached Gold rank directly.
        g_rank = generic[query.row_id]["gold_rank"]

        g_rows.append({
            **common,
            "rank": g_rank,
        })

        # F: frozen lambda = 4
        f_ranked = rank_frequency(
            query,
            candidates,
            visible,
            lambda_frequency=LAMBDA_F,
        )

        f_rows.append({
            **common,
            "rank": rank_of(
                f_ranked,
                gold,
            ),
        })

        # Original M1:
        # full-context BGE retrieval,
        # frozen Top-N=5, lambda=4.
        retrieved = (
            retrieve_memory(
                query,
                visible,
                lookup,
            )
            if visible
            else ()
        )

        m1_ranked = rank_from_retrieved(
            candidates,
            retrieved[
                :ORIGINAL_M1_TOP_N
            ],
            lambda_memory=(
                ORIGINAL_M1_LAMBDA
            ),
        )

        m1_rows.append({
            **common,
            "rank": rank_of(
                m1_ranked,
                gold,
            ),
        })

        # Hidden-M1 frozen selected result.
        hidden = hidden_by_id[
            query.row_id
        ]

        hidden_rows.append({
            **common,
            "rank": hidden["rank"],
        })

        if (
            number % 500 == 0
            or number == len(tune)
        ):
            print(
                f"Four-way comparison: "
                f"{number}/{len(tune)}",
                flush=True,
            )

finally:
    cache.close()


methods = {
    "G": g_rows,
    "F": f_rows,
    "Original-M1": m1_rows,
    "Hidden-M1": hidden_rows,
}

print()
print(
    "=== SAME-SURFACE DEV FOUR-WAY ==="
)

print(
    "Original M1: BGE Full, "
    "TopN=5, lambda=4"
)

print(
    "Hidden-M1: PinyinGPT hidden, "
    "TopN=3, lambda=4"
)

print()

header = (
    f"{'Method':14s}"
    f"{'Overall':>12s}"
    f"{'History':>12s}"
    f"{'Ambiguous':>12s}"
    f"{'Conflict':>12s}"
)

print(header)
print("-" * len(header))

table = {}

for method, rows in methods.items():
    values = {
        name: macro_top1(
            rows,
            name,
        )
        for name in (
            "overall",
            "history_available",
            "ambiguous",
            "conflict",
        )
    }

    table[method] = values

    print(
        f"{method:14s}"
        f"{values['overall']:12.6f}"
        f"{values['history_available']:12.6f}"
        f"{values['ambiguous']:12.6f}"
        f"{values['conflict']:12.6f}"
    )


def transition(before, after, subset_name):
    before_by_id = {
        row["row_id"]: row
        for row in before
    }

    after_by_id = {
        row["row_id"]: row
        for row in after
    }

    ids = [
        row["row_id"]
        for row in before
        if (
            subset_name == "overall"
            or bool(row[subset_name])
        )
    ]

    rescue = 0
    harm = 0

    for row_id in ids:
        b = (
            before_by_id[row_id]["rank"]
            == 1
        )

        a = (
            after_by_id[row_id]["rank"]
            == 1
        )

        if not b and a:
            rescue += 1

        if b and not a:
            harm += 1

    return rescue, harm


print()
print(
    "=== ORIGINAL M1 -> HIDDEN-M1 ==="
)

for name in (
    "overall",
    "history_available",
    "ambiguous",
    "conflict",
):
    rescue, harm = transition(
        m1_rows,
        hidden_rows,
        name,
    )

    print(
        f"{name:18s} "
        f"rescue={rescue} "
        f"harm={harm} "
        f"net={rescue-harm:+d}"
    )


print()
print(
    "=== F -> HIDDEN-M1 ==="
)

for name in (
    "overall",
    "history_available",
    "ambiguous",
    "conflict",
):
    rescue, harm = transition(
        f_rows,
        hidden_rows,
        name,
    )

    print(
        f"{name:18s} "
        f"rescue={rescue} "
        f"harm={harm} "
        f"net={rescue-harm:+d}"
    )

print()
print("Test used: False")
