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
    metrics_for_rows,
)
from src.personalisation.context_memory import PredictionQuery, retrieve_memory
from src.personalisation.pilot_a import (
    BGEContextEmbedder,
    EmbeddingCache,
    EmbeddingLookup,
    HistoryIndex,
    read_jsonl,
)

CONDITION = "full_short"
WINDOWS = ("full", "64", "16", "8")
PARTITIONS = ("tune", "evaluation")

OLD_PILOT_ROOT = Path(
    r"C:\Users\chiar\Desktop\LBH\thesis-personalisation"
    r"\results\personalisation\pilot_a_context_memory"
)
HISTORY_MANIFEST = OLD_PILOT_ROOT / "history_manifest.jsonl"
DEV_MANIFEST = OLD_PILOT_ROOT / "dev_manifest.jsonl"
FULL_CACHE = OLD_PILOT_ROOT / "cache" / "embedding_cache.sqlite3"

DEFAULT_MODEL = Path(
    r"C:\Users\chiar\Desktop\LBH\thesis\.cache\phase_04f\models"
    r"\bge-small-zh-v1.5-q8_0.gguf"
)

OUTPUT_ROOT = Path(
    r"results\personalisation\context_lab\local_context_retrieval_dev"
)


def parse_window(value: str) -> int | None:
    return None if value == "full" else int(value)


def label(window: int | None) -> str:
    return "Full" if window is None else f"ctx{window}"


def slug(window: int | None) -> str:
    return "full" if window is None else f"ctx{window}"


def local_context(text: str, window: int | None) -> str:
    return text if window is None else text[-window:]


def localize_records(
    rows: Sequence[Mapping[str, Any]], window: int | None
) -> list[dict[str, Any]]:
    if window is None:
        return [dict(row) for row in rows]
    return [
        {**dict(row), "context": local_context(str(row["context"]), window)}
        for row in rows
    ]


def query_from_row(row: Mapping[str, Any], window: int | None) -> PredictionQuery:
    return PredictionQuery(
        row_id=str(row["row_id"]),
        author=str(row["author"]),
        work_id=str(row["work_id"]),
        chronological_position=int(row["chronological_position"]),
        context=local_context(str(row["context"]), window),
        pinyin=tuple(row["pinyin_segments"]),
    )


def load_inputs(partition: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not HISTORY_MANIFEST.is_file():
        raise FileNotFoundError(HISTORY_MANIFEST)
    if not DEV_MANIFEST.is_file():
        raise FileNotFoundError(DEV_MANIFEST)

    history = [
        row
        for row in read_jsonl(HISTORY_MANIFEST)
        if str(row["author"]) in AUTHORS
    ]
    dev = [
        row
        for row in read_jsonl(DEV_MANIFEST)
        if str(row["author"]) in AUTHORS
    ]
    queries = [row for row in dev if str(row["pilot_partition"]) == partition]

    if not queries:
        raise RuntimeError(f"No selected Dev rows for partition={partition}")
    if any(str(row["source_split"]) != "dev" for row in queries):
        raise RuntimeError("Selected query population contains non-Dev rows")

    # The original PilotRunner Dev semantics allow each Dev query to see
    # strictly-prior same-author History + earlier Dev interactions.
    records = history + dev
    return records, queries


def paths_for(partition: str, window: int | None) -> tuple[Path, Path]:
    root = OUTPUT_ROOT / partition / slug(window)
    if window is None:
        return root, FULL_CACHE
    return root, root / "cache" / "embedding_cache.sqlite3"


def required_contexts(
    partition: str, window: int | None
) -> tuple[set[str], list[dict[str, Any]], HistoryIndex]:
    records, queries = load_inputs(partition)
    localized = localize_records(records, window)
    index = HistoryIndex(localized, HISTORY_BUDGET)
    required: set[str] = set()

    for number, row in enumerate(queries, start=1):
        query = query_from_row(row, window)
        required.add(query.context)
        visible = index.visible(query)
        required.update(str(item["context"]) for item in visible)
        if number % 500 == 0 or number == len(queries):
            print(
                f"[{partition}/{label(window)}] required contexts "
                f"{number}/{len(queries)}",
                flush=True,
            )

    return required, queries, index


def embed_or_audit(
    partition: str, window: int | None, model_path: Path
) -> dict[str, Any]:
    root, cache_path = paths_for(partition, window)
    root.mkdir(parents=True, exist_ok=True)
    required, queries, _ = required_contexts(partition, window)

    if not cache_path.is_file() and window is None:
        raise FileNotFoundError(
            f"Frozen Full-context cache is absent: {cache_path}"
        )

    cache = EmbeddingCache(cache_path)
    missing = [text for text in required if cache.get(text) is None]
    print(
        f"[{partition}/{label(window)}] rows={len(queries)} "
        f"unique={len(required)} cached={len(required)-len(missing)} "
        f"missing={len(missing)}",
        flush=True,
    )

    started = time.perf_counter()
    try:
        if window is None:
            if missing:
                raise RuntimeError(
                    "Frozen Full-context cache has unresolved misses; "
                    "do not modify it in this experiment."
                )
        elif missing:
            embedder = BGEContextEmbedder(model_path)
            for number, text in enumerate(missing, start=1):
                cache.put(text, embedder.embed(text))
                if number % 250 == 0 or number == len(missing):
                    cache.commit()
                    elapsed = time.perf_counter() - started
                    rate = number / elapsed if elapsed else 0.0
                    print(
                        f"[{partition}/{label(window)}] embedded "
                        f"{number}/{len(missing)} ({rate:.2f}/s)",
                        flush=True,
                    )
    finally:
        cache.close()

    summary = {
        "condition": CONDITION,
        "population": f"Dev {partition} partition",
        "authors": list(AUTHORS),
        "history_budget": HISTORY_BUDGET,
        "history_semantics": (
            "strictly-prior same-author History + earlier Dev interactions; "
            "H5000 applied before exact segmented-Pinyin filtering"
        ),
        "window": "full" if window is None else window,
        "window_unit": "characters" if window is not None else None,
        "window_semantics": (
            "full stored context"
            if window is None
            else f"most recent {window} characters"
        ),
        "query_rows": len(queries),
        "required_unique_contexts": len(required),
        "cache_missing_before": len(missing),
        "cache_path": str(cache_path),
        "embedding_model": str(model_path),
        "generic_pinyingpt_changed": False,
        "full_cache_modified": False,
    }
    (root / "embedding_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def run_window(partition: str, window: int | None) -> dict[str, Any]:
    root, cache_path = paths_for(partition, window)
    root.mkdir(parents=True, exist_ok=True)

    records, queries = load_inputs(partition)
    localized = localize_records(records, window)
    index = HistoryIndex(localized, HISTORY_BUDGET)
    cache = EmbeddingCache(cache_path)
    embeddings = EmbeddingLookup(cache)
    outputs: list[dict[str, Any]] = []

    started = time.perf_counter()
    try:
        for number, row in enumerate(queries, start=1):
            query = query_from_row(row, window)
            visible = index.visible(query)
            gold = str(row["gold"])
            flags = history_flags(visible, gold)
            retrieved = retrieve_memory(query, visible, embeddings) if visible else ()
            gold_rank = next(
                (
                    rank
                    for rank, item in enumerate(retrieved, start=1)
                    if str(item["historical_target"]) == gold
                ),
                None,
            )
            outputs.append(
                {
                    "condition": CONDITION,
                    "population": f"dev_{partition}",
                    "window": "full" if window is None else window,
                    "window_unit": "characters" if window is not None else None,
                    "row_id": str(row["row_id"]),
                    "author": str(row["author"]),
                    "work_id": str(row["work_id"]),
                    "chronological_position": int(row["chronological_position"]),
                    "pinyin_segments": list(row["pinyin_segments"]),
                    "gold": gold,
                    **flags,
                    "gold_retrieval_rank": gold_rank,
                }
            )
            if number % 250 == 0 or number == len(queries):
                elapsed = time.perf_counter() - started
                rate = number / elapsed if elapsed else 0.0
                print(
                    f"[{partition}/{label(window)}] retrieval "
                    f"{number}/{len(queries)} ({rate:.2f} rows/s)",
                    flush=True,
                )
    finally:
        cache.close()

    metrics = metrics_for_rows(outputs)
    rows_path = root / "rows_full_short.jsonl"
    with rows_path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in outputs:
            destination.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )

    (root / "metrics_full_short.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "condition": CONDITION,
                "population": f"Dev {partition} partition",
                "authors": list(AUTHORS),
                "history_budget": HISTORY_BUDGET,
                "history_semantics": (
                    "strictly-prior same-author History + earlier Dev interactions; "
                    "H5000 applied before exact segmented-Pinyin filtering"
                ),
                "window": "full" if window is None else window,
                "window_unit": "characters" if window is not None else None,
                "generic_pinyingpt_changed": False,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics


def recall(
    metrics: Mapping[str, Any], subset: str, k: int, aggregate: str
) -> float | None:
    value = metrics[subset][aggregate].get(f"recall_at_{k}")
    return None if value is None else float(value)


def pct(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:.2f}%"


def load_available_metrics(partition: str) -> dict[str, Mapping[str, Any]]:
    found: dict[str, Mapping[str, Any]] = {}
    for name in ("full", "ctx64", "ctx16", "ctx8"):
        path = OUTPUT_ROOT / partition / name / "metrics_full_short.json"
        if path.is_file():
            found[name] = json.loads(path.read_text(encoding="utf-8"))
    return found


def print_comparison(partition: str) -> None:
    available = load_available_metrics(partition)
    if not available:
        return

    print(f"\n=== Full+Short Dev {partition} Local Context Retrieval ===")
    print("setting   subset      micro R@1  macro R@1  micro R@5  micro R@10")
    print("-" * 72)
    for name in ("full", "ctx64", "ctx16", "ctx8"):
        if name not in available:
            continue
        metrics = available[name]
        shown = "Full" if name == "full" else name
        for subset in ("overall", "ambiguous", "conflict"):
            print(
                f"{shown:<9} {subset:<10} "
                f"{pct(recall(metrics, subset, 1, 'micro')):>10} "
                f"{pct(recall(metrics, subset, 1, 'macro_author')):>10} "
                f"{pct(recall(metrics, subset, 5, 'micro')):>10} "
                f"{pct(recall(metrics, subset, 10, 'micro')):>11}"
            )

    print(
        "\nPre-registered window-selection view: "
        "macro-author Ambiguous R@1 primary; Conflict R@1 and Overall R@1 secondary."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", choices=WINDOWS, required=True)
    parser.add_argument("--partition", choices=PARTITIONS, default="tune")
    parser.add_argument("--phase", choices=("embed", "run", "all"), default="all")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    window = parse_window(args.window)
    if window is not None and not args.model.is_file():
        raise FileNotFoundError(args.model)

    if args.phase in ("embed", "all"):
        embed_or_audit(args.partition, window, args.model)
    if args.phase in ("run", "all"):
        run_window(args.partition, window)
        print_comparison(args.partition)


if __name__ == "__main__":
    main()
