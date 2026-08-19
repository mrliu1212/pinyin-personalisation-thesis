"""Original M2 on the exact EM-2 three-author Dev tune surface.

Frozen Original M2:
- BGE Full Stage-1 retrieval
- K=20
- frozen bge-reranker-base pair scores
- lambda_m2=4

No Test.
"""

from pathlib import Path

from experiments.external_memory import em2_hidden_m1_dev as em2
from experiments.external_memory import em2_hidden_m2_dev as hm2

from src.personalisation.candidate_memory_m2 import (
    PairScoreCache,
    RERANKER_MODEL_SHA256,
    RERANKER_REVISION,
    RERANKER_TOKENIZER_SHA256,
    rank_m2,
)

from src.personalisation.context_memory import (
    macro_author_metrics,
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

PAIR_CACHE = Path(
    r"results\personalisation\external_memory"
    r"\em2_original_m2_dev"
    r"\cache\pair_scores.sqlite3"
)

AUTHORS = em2.AUTHORS
HISTORY_BUDGET = 5000

K = 20
LAMBDA_M2 = 4.0


def subset(rows, name):
    if name == "overall":
        return list(rows)

    return [
        row for row in rows
        if bool(row[name])
    ]


def macro_top1(rows, name):
    return macro_author_metrics(
        subset(rows, name),
        "rank",
    )["macro_author"]["top1"]


history = em2.read_jsonl(
    PILOT_ROOT / "history_manifest.jsonl"
)

dev = em2.read_jsonl(
    PILOT_ROOT / "dev_manifest.jsonl"
)

if any(
    str(row.get("source_split")) == "test"
    for row in history + dev
):
    raise RuntimeError("STOP: Test row detected.")

tune = [
    row for row in dev
    if (
        row.get("pilot_partition") == "tune"
        and str(row["author"]) in AUTHORS
    )
]

if len(tune) != 5608:
    raise RuntimeError(
        f"Unexpected tune population: {len(tune)}"
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

embedding_cache = EmbeddingCache(
    BGE_CACHE
)

lookup = EmbeddingLookup(
    embedding_cache
)

pair_cache = PairScoreCache(
    PAIR_CACHE,
    model_revision=RERANKER_REVISION,
    model_sha256=RERANKER_MODEL_SHA256,
    tokenizer_sha256=RERANKER_TOKENIZER_SHA256,
    max_length=512,
    dtype="float16",
)

rows = []
required_pairs = 0
missing_pairs = 0

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

        # Frozen Original M2 Stage-1:
        # full-context BGE retrieval.
        retrieved = (
            retrieve_memory(
                query,
                visible,
                lookup,
            )
            if visible
            else ()
        )

        visible_by_id = {
            str(item["row_id"]): item
            for item in visible
        }

        stage1 = tuple(
            visible_by_id[
                str(
                    item[
                        "historical_interaction_id"
                    ]
                )
            ]
            for item in retrieved[:K]
        )

        evidence = []

        for historical in stage1:
            required_pairs += 1

            pair = hm2.pair_of(
                query,
                historical,
            )

            score = pair_cache.get(pair)

            if score is None:
                missing_pairs += 1
                continue

            evidence.append(
                {
                    "historical_interaction_id":
                        pair.historical_id,
                    "historical_target":
                        pair.historical_target,
                    "raw_score":
                        score["raw_score"],
                    "input_tokens":
                        score["input_tokens"],
                    "current_context_truncated":
                        score[
                            "current_context_truncated"
                        ],
                    "historical_context_truncated":
                        score[
                            "historical_context_truncated"
                        ],
                }
            )

        if len(evidence) != len(stage1):
            continue

        ranked = rank_m2(
            candidates,
            tuple(evidence),
            lambda_m2=LAMBDA_M2,
        )

        rows.append(
            {
                "row_id": query.row_id,
                "author": query.author,
                "rank": rank_of(
                    ranked,
                    gold,
                ),
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
        )

        if (
            number % 500 == 0
            or number == len(tune)
        ):
            print(
                f"Original-M2: "
                f"{number}/{len(tune)}",
                flush=True,
            )

finally:
    pair_cache.close()
    embedding_cache.close()


print()
print("=== ORIGINAL M2 CACHE CHECK ===")
print("Required pair uses:", required_pairs)
print("Missing pair scores:", missing_pairs)

if missing_pairs:
    raise RuntimeError(
        "Old Original-M2 cache does not fully cover "
        "the requested Dev surface."
    )

if len(rows) != 5608:
    raise RuntimeError(
        f"Unexpected evaluated rows: {len(rows)}"
    )


print()
print(
    "=== ORIGINAL M2 SAME-SURFACE DEV ==="
)

print(
    "BGE Full / K=20 / lambda_m2=4"
)

for name in (
    "overall",
    "history_available",
    "ambiguous",
    "conflict",
):
    print(
        f"{name:18s} "
        f"{macro_top1(rows, name):.6f}"
    )

print()
print("Test used: False")
