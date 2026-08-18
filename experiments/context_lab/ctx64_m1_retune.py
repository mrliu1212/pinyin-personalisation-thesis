from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.context_lab.diagnostic_a_retrieval import AUTHORS, HISTORY_BUDGET
from src.personalisation.context_memory import (
    Candidate,
    PredictionQuery,
    macro_author_metrics,
    rank_from_retrieved,
    rank_of,
    retrieve_memory,
)
from src.personalisation.pilot_a import (
    MEMORY_LAMBDAS,
    MEMORY_TOP_NS,
    EmbeddingCache,
    EmbeddingLookup,
    HistoryIndex,
    read_jsonl,
)

WINDOW = 64
OLD_PILOT_ROOT = Path(
    r"C:\Users\chiar\Desktop\LBH\thesis-personalisation"
    r"\results\personalisation\pilot_a_context_memory"
)
HISTORY_MANIFEST = OLD_PILOT_ROOT / "history_manifest.jsonl"
DEV_MANIFEST = OLD_PILOT_ROOT / "dev_manifest.jsonl"
GENERIC_CACHE = OLD_PILOT_ROOT / "cache" / "generic_predictions.jsonl"
LOCAL_ROOT = Path(
    r"results\personalisation\context_lab\local_context_retrieval_dev"
)
OUTPUT_ROOT = Path(
    r"results\personalisation\context_lab\ctx64_m1_retune"
)


def local_context(text: str) -> str:
    return text[-WINDOW:]


def localize_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{**dict(row), "context": local_context(str(row["context"]))} for row in rows]


def query_from_row(row: Mapping[str, Any]) -> PredictionQuery:
    return PredictionQuery(
        row_id=str(row["row_id"]),
        author=str(row["author"]),
        work_id=str(row["work_id"]),
        chronological_position=int(row["chronological_position"]),
        context=local_context(str(row["context"])),
        pinyin=tuple(row["pinyin_segments"]),
    )


def load_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    history = [r for r in read_jsonl(HISTORY_MANIFEST) if str(r["author"]) in AUTHORS]
    dev = [r for r in read_jsonl(DEV_MANIFEST) if str(r["author"]) in AUTHORS]
    if any(str(r["source_split"]) != "dev" for r in dev):
        raise RuntimeError("Dev manifest contains non-Dev rows")
    return history, dev


def load_generic() -> dict[str, dict[str, Any]]:
    if not GENERIC_CACHE.is_file():
        raise FileNotFoundError(GENERIC_CACHE)
    rows = read_jsonl(GENERIC_CACHE)
    selected = {
        str(r["row_id"]): r
        for r in rows
        if str(r["author"]) in AUTHORS
    }
    return selected


def candidates(generic_row: Mapping[str, Any]) -> tuple[Candidate, ...]:
    return tuple(
        Candidate(str(r["text"]), int(r["rank"]), float(r["log_probability"]))
        for r in generic_row["top10_candidates"]
    )


def cache_path(partition: str) -> Path:
    return LOCAL_ROOT / partition / "ctx64" / "cache" / "embedding_cache.sqlite3"


def indexed_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], HistoryIndex]:
    history, dev = load_inputs()
    localized = localize_records(history + dev)
    return history, dev, HistoryIndex(localized, HISTORY_BUDGET)


def tune() -> dict[str, Any]:
    _, dev, index = indexed_inputs()
    rows = [r for r in dev if str(r["pilot_partition"]) == "tune"]
    generic = load_generic()
    missing_generic = [r["row_id"] for r in rows if str(r["row_id"]) not in generic]
    if missing_generic:
        raise RuntimeError(f"Generic Dev cache incomplete: {len(missing_generic)} missing")

    path = cache_path("tune")
    if not path.is_file():
        raise FileNotFoundError(
            f"ctx64 Dev-tune embedding cache absent: {path}. "
            "Run local_context_retrieval_dev.py --window 64 --partition tune --phase all first."
        )
    cache = EmbeddingCache(path)
    lookup = EmbeddingLookup(cache)
    grid: dict[tuple[int, float], list[dict[str, Any]]] = {
        (top_n, lam): [] for top_n in MEMORY_TOP_NS for lam in MEMORY_LAMBDAS
    }
    started = time.perf_counter()
    try:
        for number, row in enumerate(rows, start=1):
            query = query_from_row(row)
            visible = index.visible(query)
            cand = candidates(generic[str(row["row_id"])])
            retrieved = retrieve_memory(query, visible, lookup) if visible else ()
            for top_n in MEMORY_TOP_NS:
                selected = retrieved[:top_n]
                for lam in MEMORY_LAMBDAS:
                    ranked = rank_from_retrieved(cand, selected, lambda_memory=float(lam))
                    grid[(int(top_n), float(lam))].append(
                        {"author": str(row["author"]), "rank": rank_of(ranked, str(row["gold"]))}
                    )
            if number % 250 == 0 or number == len(rows):
                elapsed = time.perf_counter() - started
                print(f"ctx64 tune {number}/{len(rows)} ({number/elapsed:.2f} rows/s)", flush=True)
    finally:
        cache.close()

    search = []
    for top_n in MEMORY_TOP_NS:
        for lam in MEMORY_LAMBDAS:
            metrics = macro_author_metrics(grid[(int(top_n), float(lam))], "rank")["macro_author"]
            search.append({"top_n": int(top_n), "lambda_memory": float(lam), **metrics})

    selected = max(
        search,
        key=lambda r: (float(r["top1"]), -float(r["lambda_memory"]), -int(r["top_n"])),
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "hyperparameter_search.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(search[0]))
        writer.writeheader()
        writer.writerows(search)
    result = {
        "status": "complete",
        "context_window": WINDOW,
        "window_unit": "characters",
        "selection_population": "Dev tune partition",
        "selection_metric": "Macro-author Top-1",
        "tie_break": "lower lambda_memory, then lower top_n",
        "authors": list(AUTHORS),
        "history_budget": HISTORY_BUDGET,
        "memory_top_n_grid": list(MEMORY_TOP_NS),
        "memory_lambda_grid": list(MEMORY_LAMBDAS),
        "rows": len(rows),
        "selected": selected,
    }
    (OUTPUT_ROOT / "selected_hyperparameters.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("\n=== ctx64 M1 Dev-tune selection ===")
    print(json.dumps(selected, ensure_ascii=False, indent=2))
    return result


def evaluate() -> dict[str, Any]:
    selection_path = OUTPUT_ROOT / "selected_hyperparameters.json"
    if not selection_path.is_file():
        raise RuntimeError("Run --phase tune first")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))["selected"]
    top_n = int(selection["top_n"])
    lam = float(selection["lambda_memory"])

    _, dev, index = indexed_inputs()
    rows = [r for r in dev if str(r["pilot_partition"]) == "evaluation"]
    generic = load_generic()
    missing_generic = [r["row_id"] for r in rows if str(r["row_id"]) not in generic]
    if missing_generic:
        raise RuntimeError(f"Generic Dev cache incomplete: {len(missing_generic)} missing")

    path = cache_path("evaluation")
    if not path.is_file():
        raise FileNotFoundError(
            f"ctx64 Dev-evaluation embedding cache absent: {path}. "
            "Run local_context_retrieval_dev.py --window 64 --partition evaluation --phase all first."
        )
    cache = EmbeddingCache(path)
    lookup = EmbeddingLookup(cache)
    metric_rows = []
    outputs = []
    started = time.perf_counter()
    try:
        for number, row in enumerate(rows, start=1):
            query = query_from_row(row)
            visible = index.visible(query)
            cand = candidates(generic[str(row["row_id"])])
            retrieved = retrieve_memory(query, visible, lookup) if visible else ()
            ranked = rank_from_retrieved(cand, retrieved[:top_n], lambda_memory=lam)
            rank = rank_of(ranked, str(row["gold"]))
            metric_rows.append({"author": str(row["author"]), "rank": rank})
            outputs.append({
                "row_id": str(row["row_id"]),
                "author": str(row["author"]),
                "work_id": str(row["work_id"]),
                "gold": str(row["gold"]),
                "gold_rank": rank,
                "context_window": WINDOW,
                "top_n": top_n,
                "lambda_memory": lam,
            })
            if number % 250 == 0 or number == len(rows):
                elapsed = time.perf_counter() - started
                print(f"ctx64 evaluation {number}/{len(rows)} ({number/elapsed:.2f} rows/s)", flush=True)
    finally:
        cache.close()

    metrics = macro_author_metrics(metric_rows, "rank")
    with (OUTPUT_ROOT / "evaluation_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    result = {
        "status": "complete",
        "context_window": WINDOW,
        "top_n": top_n,
        "lambda_memory": lam,
        "population": "chronologically later Dev evaluation partition",
        "rows": len(rows),
        "metrics": metrics,
    }
    (OUTPUT_ROOT / "evaluation_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("\n=== ctx64 M1 Dev evaluation ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("tune", "evaluate", "all"), default="all")
    args = parser.parse_args()
    if args.phase in ("tune", "all"):
        tune()
    if args.phase in ("evaluate", "all"):
        evaluate()


if __name__ == "__main__":
    main()
