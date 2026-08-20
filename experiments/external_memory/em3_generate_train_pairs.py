from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCE = Path(
    r"C:\Users\chiar\Desktop\LBH\thesis-personalisation"
    r"\results\personalisation\reranking_matrix\manifests"
    r"\history_full_short.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"results\personalisation\external_memory\em3_train_pairs_v2_clean3"
)
DEFAULT_AUTHORS = ("Agent Phage", "Etinjat", "breaddddd")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def row_id(row: dict[str, Any]) -> str:
    value = row.get("row_id", row.get("condition_id"))
    if value is None:
        raise KeyError("row is missing row_id/condition_id")
    return str(value)


def author(row: dict[str, Any]) -> str:
    value = row.get("author", row.get("user_id"))
    if value is None:
        raise KeyError("row is missing author/user_id")
    return str(value)


def chronological_position(row: dict[str, Any], fallback: int) -> int:
    for key in (
        "chronological_position",
        "position",
        "interaction_position",
        "query_position",
    ):
        if row.get(key) is not None:
            return int(row[key])
    return fallback


def pinyin_segments(row: dict[str, Any]) -> tuple[str, ...]:
    value = row.get("pinyin_segments", row.get("segmented_pinyin"))
    if value is None:
        value = row.get("pinyin")
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    if value is None:
        raise KeyError("row is missing segmented Pinyin")
    return (str(value),)


def target(row: dict[str, Any]) -> str:
    for key in ("target", "gold", "current_gold", "target_candidate"):
        if row.get(key) is not None:
            return str(row[key])
    raise KeyError("row is missing target/gold")


def context(row: dict[str, Any]) -> str:
    for key in ("context", "preceding_context", "current_context"):
        if row.get(key) is not None:
            return str(row[key])
    return ""


def query_rng(seed: int, query_row_id: str) -> random.Random:
    material = f"em3-v1|{seed}|{query_row_id}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))


def select_rounds(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    *,
    seed: int,
    query_row_id: str,
    max_rounds: int,
    negatives_per_round: int,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    rng = query_rng(seed, query_row_id)
    available_positives = list(positives)
    available_negatives = list(negatives)
    rng.shuffle(available_positives)
    rng.shuffle(available_negatives)

    rounds: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    negative_offset = 0
    for positive in available_positives[:max_rounds]:
        selected_negatives = available_negatives[
            negative_offset : negative_offset + negatives_per_round
        ]
        negative_offset += len(selected_negatives)
        rounds.append((positive, selected_negatives))
    return rounds


def pair_record(
    query: dict[str, Any],
    history: dict[str, Any],
    *,
    round_number: int,
    label: int,
    query_position: int,
    history_position: int,
) -> dict[str, Any]:
    return {
        "query_row_id": row_id(query),
        "history_row_id": row_id(history),
        "author": author(query),
        "round": round_number,
        "label": label,
        "current_context": context(query),
        "pinyin_segments": list(pinyin_segments(query)),
        "current_gold": target(query),
        "history_context": context(history),
        "history_target": target(history),
        "query_position": query_position,
        "history_position": history_position,
    }


def refuse_test_input(path: Path, row: dict[str, Any] | None = None) -> None:
    if "test" in path.name.casefold():
        raise RuntimeError(f"benchmark-Test input refused: {path}")
    if row is None:
        return
    for key in ("source_split", "split", "pilot_partition"):
        value = row.get(key)
        if value is not None and str(value).casefold() == "test":
            raise RuntimeError(
                f"benchmark-Test row refused: {row_id(row)} ({key}={value})"
            )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic causal EM3 positive/hard-negative pairs."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--authors", nargs="+", default=list(DEFAULT_AUTHORS))
    parser.add_argument("--history-budget", type=int, default=5000)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--negatives-per-round", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Compute and persist counts/provenance without writing pair JSONL.",
    )
    args = parser.parse_args()

    if args.history_budget <= 0:
        parser.error("--history-budget must be positive")
    if args.max_rounds <= 0:
        parser.error("--max-rounds must be positive")
    if args.negatives_per_round <= 0:
        parser.error("--negatives-per-round must be positive")

    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    refuse_test_input(source)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    planned = [
        output_root / "summary.json",
        output_root / "provenance.json",
        output_root / "audit.json",
    ]
    if not args.audit_only:
        planned.append(output_root / "train_pairs.jsonl")
    existing = [path for path in planned if path.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing output: {names}")

    selected_authors = tuple(dict.fromkeys(str(item) for item in args.authors))
    author_set = set(selected_authors)
    rows_by_author: dict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(
        list
    )
    source_rows = 0
    selected_rows = 0
    for input_index, row in enumerate(iter_jsonl(source)):
        source_rows += 1
        refuse_test_input(source, row)
        row_author = author(row)
        if row_author not in author_set:
            continue
        selected_rows += 1
        rows_by_author[row_author].append(
            (chronological_position(row, input_index), input_index, row)
        )

    missing_authors = [name for name in selected_authors if not rows_by_author[name]]
    if missing_authors:
        raise ValueError(f"authors absent from source: {missing_authors}")

    pair_path = output_root / "train_pairs.jsonl"
    pair_file = None if args.audit_only else pair_path.open("wb")
    pair_digest = hashlib.sha256()
    stats = Counter()
    rounds_by_query = Counter()
    per_author: dict[str, Counter[str]] = defaultdict(Counter)
    seen_query_history: set[tuple[str, str]] = set()
    reused_query_history = 0
    non_prior_pairs = 0

    try:
        for selected_author in selected_authors:
            ordered = sorted(rows_by_author[selected_author], key=lambda item: item[:2])
            # The full deque enforces H5000 before Pinyin filtering.  The
            # secondary index only accelerates exact-Pinyin lookup; entries
            # are evicted from it precisely when they leave the full window.
            visible: deque[tuple[int, int, dict[str, Any]]] = deque()
            visible_by_pinyin: dict[
                tuple[str, ...], deque[tuple[int, int, dict[str, Any]]]
            ] = defaultdict(deque)
            index = 0
            while index < len(ordered):
                current_position = ordered[index][0]
                end = index
                block: list[tuple[int, int, dict[str, Any]]] = []
                while end < len(ordered) and ordered[end][0] == current_position:
                    block.append(ordered[end])
                    end += 1

                for query_position, _, query in block:
                    query_pinyin = pinyin_segments(query)
                    query_target = target(query)
                    same_pinyin = [
                        (history_position, history)
                        for history_position, _, history in visible_by_pinyin.get(
                            query_pinyin, ()
                        )
                    ]
                    positives = [
                        history
                        for _, history in same_pinyin
                        if target(history) == query_target
                    ]
                    negatives = [
                        history
                        for _, history in same_pinyin
                        if target(history) != query_target
                    ]
                    if not positives or not negatives:
                        continue

                    stats["eligible_queries"] += 1
                    per_author[selected_author]["eligible_queries"] += 1
                    history_positions = {
                        row_id(history): history_position
                        for history_position, history in same_pinyin
                    }
                    rounds = select_rounds(
                        positives,
                        negatives,
                        seed=args.seed,
                        query_row_id=row_id(query),
                        max_rounds=args.max_rounds,
                        negatives_per_round=args.negatives_per_round,
                    )
                    rounds_by_query[len(rounds)] += 1
                    per_author[selected_author][f"queries_with_{len(rounds)}_rounds"] += 1

                    for round_index, (positive, round_negatives) in enumerate(
                        rounds, start=1
                    ):
                        selected = [(positive, 1), *[(item, 0) for item in round_negatives]]
                        for history, label in selected:
                            history_id = row_id(history)
                            history_position = history_positions[history_id]
                            key = (row_id(query), history_id)
                            if key in seen_query_history:
                                reused_query_history += 1
                            seen_query_history.add(key)
                            if history_position >= query_position:
                                non_prior_pairs += 1

                            stat_name = "positive_pairs" if label else "negative_pairs"
                            stats[stat_name] += 1
                            stats["total_pairs"] += 1
                            per_author[selected_author][stat_name] += 1
                            per_author[selected_author]["total_pairs"] += 1

                            if pair_file is not None:
                                record = pair_record(
                                    query,
                                    history,
                                    round_number=round_index,
                                    label=label,
                                    query_position=query_position,
                                    history_position=history_position,
                                )
                                encoded = (
                                    json.dumps(record, ensure_ascii=False) + "\n"
                                ).encode("utf-8")
                                pair_file.write(encoded)
                                pair_digest.update(encoded)

                # Equal-position rows become visible only after the whole block.
                for row_position, input_index, row in block:
                    if len(visible) >= args.history_budget:
                        old_position, old_index, old_row = visible.popleft()
                        old_pinyin = pinyin_segments(old_row)
                        indexed = visible_by_pinyin[old_pinyin].popleft()
                        if indexed[:2] != (old_position, old_index):
                            raise RuntimeError("H5000 Pinyin index lost synchronization")
                        if not visible_by_pinyin[old_pinyin]:
                            del visible_by_pinyin[old_pinyin]
                    entry = (row_position, input_index, row)
                    visible.append(entry)
                    visible_by_pinyin[pinyin_segments(row)].append(entry)
                index = end
    finally:
        if pair_file is not None:
            pair_file.close()

    summary = {
        "schema_version": 1,
        "experiment": "em3_generate_train_pairs",
        "status": "audit_only" if args.audit_only else "complete",
        "authors": list(selected_authors),
        "history_budget": args.history_budget,
        "max_rounds": args.max_rounds,
        "negatives_per_round": args.negatives_per_round,
        "seed": args.seed,
        "source_split": "history",
        "test_used": False,
        "stats": dict(stats),
        "round_distribution": {
            str(rounds): count for rounds, count in sorted(rounds_by_query.items())
        },
        "per_author": {
            name: dict(per_author[name]) for name in selected_authors
        },
    }
    audit = {
        "schema_version": 1,
        "test_used": False,
        "source_rows": source_rows,
        "selected_author_rows": selected_rows,
        "selected_authors": list(selected_authors),
        "same_author_required": True,
        "strictly_prior_required": True,
        "history_budget_before_pinyin_filter": True,
        "exact_segmented_pinyin": True,
        "same_position_rows_hidden_from_each_other": True,
        "query_history_reuse_count": reused_query_history,
        "non_prior_pair_count": non_prior_pairs,
        "passed": reused_query_history == 0 and non_prior_pairs == 0,
    }
    provenance = {
        "schema_version": 1,
        "test_used": False,
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "output": {
            "path": str(pair_path),
            "written": not args.audit_only,
            "sha256": pair_digest.hexdigest() if not args.audit_only else None,
        },
        "sampling": (
            "query-local SHA256-derived RNG; shuffle without replacement; "
            "one unused positive and up to N unused negatives per round"
        ),
        "command": "python -m experiments.external_memory.em3_generate_train_pairs",
    }
    write_json(output_root / "summary.json", summary)
    write_json(output_root / "audit.json", audit)
    write_json(output_root / "provenance.json", provenance)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise RuntimeError("pair-generation audit failed")


if __name__ == "__main__":
    main()
