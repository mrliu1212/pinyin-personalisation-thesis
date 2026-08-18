import json
from collections import defaultdict
from pathlib import Path

from src.personalisation.context_memory import (
    Candidate,
    PredictionQuery,
    macro_author_metrics,
    metric_values,
    rank_from_retrieved,
    rank_of,
    retrieve_memory,
    subset_membership,
)
from src.personalisation.pilot_a import (
    EmbeddingCache,
    EmbeddingLookup,
    HistoryIndex,
)

AUTHORS = {"Etinjat", "Re_spectators", "breaddddd"}

WINDOW = 64
HISTORY_BUDGET = 5000
TOP_N = 3
LAMBDA_MEMORY = 4.0

TEST_MANIFEST = Path(
    r"C:\Users\chiar\Desktop\LBH\thesis-personalisation"
    r"\results\personalisation\pilot_a_context_memory"
    r"\h5000\test_manifest.jsonl"
)

HISTORY_MANIFEST = Path(r"C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\history_manifest.jsonl")

FROZEN_PRED = Path(
    r"C:\Users\chiar\Desktop\LBH\thesis-personalisation"
    r"\results\personalisation\reranking_matrix"
    r"\cells\full_short\HFull\M1\predictions.jsonl"
)

EMBED_CACHE = Path(
    r"results\personalisation\context_lab"
    r"\local_context_retrieval\ctx64\cache"
    r"\embedding_cache.sqlite3"
)

OUTPUT = Path(
    r"results\personalisation\context_lab"
    r"\ctx64_m1_test_h5000"
)


def read_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def local_context(text):
    return str(text)[-WINDOW:]


def localise_record(row):
    row = dict(row)
    row["context"] = local_context(row["context"])
    return row


def micro(rows, key):
    return metric_values([row[key] for row in rows])


def pct(x):
    return "n/a" if x is None else f"{100*x:.2f}%"


for path in (TEST_MANIFEST, HISTORY_MANIFEST, FROZEN_PRED, EMBED_CACHE):
    if not path.exists():
        raise FileNotFoundError(path)

test_rows = [
    r for r in read_jsonl(TEST_MANIFEST)
    if r["author"] in AUTHORS
]

history_rows = [
    localise_record(r)
    for r in read_jsonl(HISTORY_MANIFEST)
    if r["author"] in AUTHORS
]

frozen_rows = [
    r for r in read_jsonl(FROZEN_PRED)
    if r["author"] in AUTHORS
]

if len(test_rows) != 3000:
    raise RuntimeError(f"Expected 3000 Test rows, got {len(test_rows)}")

if len(frozen_rows) != 3000:
    raise RuntimeError(f"Expected 3000 frozen prediction rows, got {len(frozen_rows)}")

frozen_by_anchor = {r["anchor_id"]: r for r in frozen_rows}

if len(frozen_by_anchor) != 3000:
    raise RuntimeError("Frozen predictions contain duplicate anchor_id values")

missing_frozen = [
    r["anchor_id"] for r in test_rows
    if r["anchor_id"] not in frozen_by_anchor
]
if missing_frozen:
    raise RuntimeError(
        f"{len(missing_frozen)} Test anchors missing frozen Generic predictions"
    )

index = HistoryIndex(history_rows, HISTORY_BUDGET)

cache = EmbeddingCache(EMBED_CACHE)
lookup = EmbeddingLookup(cache)

metric_rows = []
prediction_rows = []

try:
    for i, row in enumerate(test_rows, 1):
        query = PredictionQuery(
            row_id=str(row["row_id"]),
            author=str(row["author"]),
            work_id=str(row["work_id"]),
            chronological_position=int(row["chronological_position"]),
            context=local_context(row["context"]),
            pinyin=tuple(row["pinyin_segments"]),
        )

        visible = index.visible(query)

        old = frozen_by_anchor[row["anchor_id"]]

        frozen_candidates = sorted(
            old["candidates"],
            key=lambda c: int(c["generic_rank"]),
        )

        candidates = tuple(
            Candidate(
                text=str(c["candidate"]),
                generic_rank=int(c["generic_rank"]),
                generic_score=float(c["generic_score"]),
            )
            for c in frozen_candidates
        )

        # Frozen Generic rank reconstructed from the original candidate pool.
        generic_rank = next(
            (
                c.generic_rank
                for c in candidates
                if c.text == row["gold"]
            ),
            None,
        )

        if visible:
            try:
                retrieved = retrieve_memory(query, visible, lookup)
            except KeyError as e:
                raise RuntimeError(
                    "ctx64 embedding cache is incomplete. "
                    f"Missing context key at Test row {i}: {e}"
                ) from e
        else:
            retrieved = ()

        selected = retrieved[:TOP_N]

        ranked = rank_from_retrieved(
            candidates,
            selected,
            lambda_memory=LAMBDA_MEMORY,
        )

        memory_rank = rank_of(ranked, row["gold"])

        # Candidate pool is frozen, therefore Missing@10 must not change.
        if (generic_rank is None) != (memory_rank is None):
            raise AssertionError(
                f"Candidate-pool invariance failed at {row['anchor_id']}"
            )

        flags = subset_membership(query, row["gold"], visible)

        metric = {
            "anchor_id": row["anchor_id"],
            "author": row["author"],
            "generic_rank": generic_rank,
            "memory_rank": memory_rank,
            **flags,
        }
        metric_rows.append(metric)

        prediction_rows.append({
            "anchor_id": row["anchor_id"],
            "author": row["author"],
            "work_id": row["work_id"],
            "gold": row["gold"],
            "history_budget": HISTORY_BUDGET,
            "context_window_chars": WINDOW,
            "top_n": TOP_N,
            "lambda_memory": LAMBDA_MEMORY,
            **flags,
            "generic_rank": generic_rank,
            "gold_rank": memory_rank,
            "retrieved_evidence": list(selected),
            "candidates": list(ranked),
        })

        if i % 250 == 0 or i == len(test_rows):
            print(f"ctx64 M1 Test {i}/{len(test_rows)}", flush=True)

finally:
    cache.close()

subsets = {
    "overall": metric_rows,
    "history_available": [
        r for r in metric_rows if r["history_available"]
    ],
    "ambiguous": [
        r for r in metric_rows if r["ambiguous"]
    ],
    "conflict": [
        r for r in metric_rows if r["conflict"]
    ],
}

summary = {
    "status": "complete",
    "population": "Test Full+Short, 3 exploratory authors",
    "authors": sorted(AUTHORS),
    "rows": len(metric_rows),
    "history_budget": HISTORY_BUDGET,
    "context_window_chars": WINDOW,
    "top_n": TOP_N,
    "lambda_memory": LAMBDA_MEMORY,
    "generic_source": str(FROZEN_PRED),
    "embedding_cache": str(EMBED_CACHE),
    "test_manifest": str(TEST_MANIFEST),
    "test_tuning_performed": False,
    "metrics": {},
}

print()
print("=== ctx64 M1 FINAL TEST ===")
print(
    f"window={WINDOW}  H={HISTORY_BUDGET}  "
    f"top_n={TOP_N}  lambda={LAMBDA_MEMORY}"
)
print()

print(
    f"{'subset':18s} {'n':>6s} "
    f"{'Generic Top1':>13s} {'M1 Top1':>10s} "
    f"{'M1 Top3':>9s} {'M1 MRR@10':>11s}"
)
print("-" * 76)

for name, rows in subsets.items():
    generic_micro = micro(rows, "generic_rank")
    memory_micro = micro(rows, "memory_rank")
    generic_macro = macro_author_metrics(
        rows, "generic_rank"
    )
    memory_macro = macro_author_metrics(
        rows, "memory_rank"
    )

    summary["metrics"][name] = {
        "rows": len(rows),
        "generic": {
            "micro": generic_micro,
            **generic_macro,
        },
        "M1": {
            "micro": memory_micro,
            **memory_macro,
        },
    }

    print(
        f"{name:18s} {len(rows):6d} "
        f"{pct(generic_micro['top1']):>13s} "
        f"{pct(memory_micro['top1']):>10s} "
        f"{pct(memory_micro['top3']):>9s} "
        f"{memory_micro['mrr_at_10']:.4f}"
    )

print()
print("=== Macro-author Top1 ===")
for name, values in summary["metrics"].items():
    g = values["generic"]["macro_author"]["top1"]
    m = values["M1"]["macro_author"]["top1"]
    print(
        f"{name:18s} Generic={pct(g)}  M1={pct(m)}  "
        f"delta={100*(m-g):+.2f}pp"
    )

OUTPUT.mkdir(parents=True, exist_ok=True)

with (OUTPUT / "predictions.jsonl").open(
    "w", encoding="utf-8", newline="\n"
) as f:
    for row in prediction_rows:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )

(OUTPUT / "result.json").write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print()
print("Saved:")
print(OUTPUT / "predictions.jsonl")
print(OUTPUT / "result.json")

