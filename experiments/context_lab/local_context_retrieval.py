from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.context_lab.diagnostic_a_retrieval import (
    AUTHORS,
    HISTORY_BUDGET,
    history_flags,
    load_history,
    load_test_rows,
    metrics_for_rows,
)
from src.personalisation.context_memory import PredictionQuery, retrieve_memory
from src.personalisation.pilot_a import (
    BGEContextEmbedder,
    EmbeddingCache,
    EmbeddingLookup,
    HistoryIndex,
)

CONDITION = "full_short"
WINDOWS = (16, 64)

DEFAULT_MODEL = Path(
    r"C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models"
    r"\bge-small-zh-v1.5-q8_0.gguf"
)

OUTPUT_ROOT = Path(
    r"results\personalisation\context_lab\local_context_retrieval"
)

FULL_BASELINE_METRICS = Path(
    r"results\personalisation\context_lab\diagnostic_a1_retrieval"
    r"\metrics_full_short.json"
)


def local_context(text: str, window: int) -> str:
    return text[-window:]


def localize_history(rows: Sequence[Mapping[str, Any]], window: int) -> list[dict[str, Any]]:
    return [{**dict(row), "context": local_context(str(row["context"]), window)} for row in rows]


def local_query(row: Mapping[str, Any], window: int) -> PredictionQuery:
    return PredictionQuery(
        row_id=str(row["row_id"]),
        author=str(row["author"]),
        work_id=str(row["work_id"]),
        chronological_position=int(row["chronological_position"]),
        context=local_context(str(row["context"]), window),
        pinyin=tuple(row["pinyin_segments"]),
    )


def paths_for(window: int) -> tuple[Path, Path]:
    root = OUTPUT_ROOT / f"ctx{window}"
    return root, root / "cache" / "embedding_cache.sqlite3"


def selected_test_rows() -> list[dict[str, Any]]:
    return [row for row in load_test_rows()[CONDITION] if row["author"] in AUTHORS]


def required_contexts(window: int) -> tuple[set[str], list[dict[str, Any]], HistoryIndex]:
    history = localize_history(load_history(CONDITION), window)
    index = HistoryIndex(history, HISTORY_BUDGET)
    tests = selected_test_rows()
    required: set[str] = set()

    for number, row in enumerate(tests, start=1):
        query = local_query(row, window)
        required.add(query.context)
        visible = index.visible(query)
        required.update(str(item["context"]) for item in visible)
        if number % 500 == 0 or number == len(tests):
            print(f"[ctx{window}] required contexts {number}/{len(tests)}", flush=True)

    return required, tests, index


def embed_missing(window: int, model_path: Path) -> dict[str, Any]:
    root, cache_path = paths_for(window)
    root.mkdir(parents=True, exist_ok=True)
    required, _, _ = required_contexts(window)
    cache = EmbeddingCache(cache_path)
    missing = [text for text in required if cache.get(text) is None]

    print(
        f"[ctx{window}] unique={len(required)} cached={len(required)-len(missing)} missing={len(missing)}",
        flush=True,
    )

    started = time.perf_counter()
    try:
        if missing:
            embedder = BGEContextEmbedder(model_path)
            for number, text in enumerate(missing, start=1):
                cache.put(text, embedder.embed(text))
                if number % 250 == 0 or number == len(missing):
                    cache.commit()
                    elapsed = time.perf_counter() - started
                    rate = number / elapsed if elapsed else 0.0
                    print(f"[ctx{window}] embedded {number}/{len(missing)} ({rate:.2f}/s)", flush=True)
    finally:
        cache.close()

    summary = {
        "condition": CONDITION,
        "authors": list(AUTHORS),
        "history_budget": HISTORY_BUDGET,
        "window_unit": "characters",
        "window": window,
        "window_semantics": f"most recent {window} characters",
        "required_unique_contexts": len(required),
        "cache_missing_before": len(missing),
        "cache_path": str(cache_path),
        "embedding_model": str(model_path),
        "generic_pinyingpt_changed": False,
        "full_baseline_recomputed": False,
    }
    (root / "embedding_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run_window(window: int) -> dict[str, Any]:
    root, cache_path = paths_for(window)
    root.mkdir(parents=True, exist_ok=True)

    history = localize_history(load_history(CONDITION), window)
    index = HistoryIndex(history, HISTORY_BUDGET)
    tests = selected_test_rows()
    cache = EmbeddingCache(cache_path)
    embeddings = EmbeddingLookup(cache)
    outputs: list[dict[str, Any]] = []

    started = time.perf_counter()
    try:
        for number, row in enumerate(tests, start=1):
            query = local_query(row, window)
            visible = index.visible(query)
            flags = history_flags(visible, str(row["gold"]))
            retrieved = retrieve_memory(query, visible, embeddings) if visible else ()
            gold_rank = next(
                (
                    rank
                    for rank, item in enumerate(retrieved, start=1)
                    if str(item["historical_target"]) == str(row["gold"])
                ),
                None,
            )
            outputs.append(
                {
                    "condition": CONDITION,
                    "window": window,
                    "window_unit": "characters",
                    "row_id": str(row["row_id"]),
                    "anchor_id": str(row["anchor_id"]),
                    "author": str(row["author"]),
                    "work_id": str(row["work_id"]),
                    "chronological_position": int(row["chronological_position"]),
                    "pinyin_segments": list(row["pinyin_segments"]),
                    "gold": str(row["gold"]),
                    **flags,
                    "gold_retrieval_rank": gold_rank,
                }
            )
            if number % 250 == 0 or number == len(tests):
                elapsed = time.perf_counter() - started
                rate = number / elapsed if elapsed else 0.0
                print(f"[ctx{window}] retrieval {number}/{len(tests)} ({rate:.2f} rows/s)", flush=True)
    finally:
        cache.close()

    metrics = metrics_for_rows(outputs)
    rows_path = root / "rows_full_short.jsonl"
    with rows_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    (root / "metrics_full_short.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "condition": CONDITION,
                "authors": list(AUTHORS),
                "history_budget": HISTORY_BUDGET,
                "window": window,
                "window_unit": "characters",
                "generic_pinyingpt_changed": False,
                "full_baseline_recomputed": False,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics


def recall(metrics: Mapping[str, Any], subset: str, k: int) -> float | None:
    value = metrics[subset]["micro"].get(f"recall_at_{k}")
    return None if value is None else float(value)


def pct(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:.2f}%"


def print_comparison(local_metrics: dict[int, Mapping[str, Any]]) -> None:
    if not FULL_BASELINE_METRICS.is_file():
        print(f"\nFull baseline not found at {FULL_BASELINE_METRICS}", flush=True)
        return

    full = json.loads(FULL_BASELINE_METRICS.read_text(encoding="utf-8"))
    print("\n=== Full+Short Local Context Retrieval ===")
    print("setting   subset      R@1      R@5      R@10")
    print("-" * 53)
    settings: list[tuple[str, Mapping[str, Any]]] = [("Full", full)]
    settings.extend(
        (f"ctx{window}", local_metrics[window])
        for window in sorted(local_metrics)
        if window in local_metrics
    )
    for label, metrics in settings:
        for subset in ("overall", "ambiguous", "conflict"):
            print(
                f"{label:<9} {subset:<10} "
                f"{pct(recall(metrics, subset, 1)):>8} "
                f"{pct(recall(metrics, subset, 5)):>8} "
                f"{pct(recall(metrics, subset, 10)):>8}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, choices=WINDOWS, required=True)
    parser.add_argument("--phase", choices=("embed", "run", "all"), default="all")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    local_metrics: dict[int, Mapping[str, Any]] = {}
    if args.phase in ("embed", "all"):
        embed_missing(args.window, args.model)
    if args.phase in ("run", "all"):
        local_metrics[args.window] = run_window(args.window)
        print_comparison(local_metrics)


if __name__ == "__main__":
    main()
