"""EM-2B: cache Frozen PinyinGPT hidden representations on Dev.

Representation:
    final-layer hidden state at the final prompt [SEP] token.

History semantics:
    reuse the frozen HistoryIndex(history + dev, H5000).

This stage:
- uses Dev only;
- does not inspect Gold;
- does not compute retrieval metrics;
- does not use target text to construct representations.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import time
from typing import Any, Mapping

import numpy as np

from src.personalisation.pilot_a import (
    HistoryIndex,
    PilotRunner,
)
from src.reference_backend_pinyingpt.backend import (
    PinyinGPTConcatBackend,
)


AUTHORS = (
    "Etinjat",
    "Re_spectators",
    "breaddddd",
)

HISTORY_BUDGET = 5000

# Frozen from the EM-2B workload preflight.
EXPECTED_TUNE_QUERIES = 5608
EXPECTED_HISTORY_EDGES = 122067
EXPECTED_ROWS_WITH_HISTORY = 3625
EXPECTED_ROWS_WITHOUT_HISTORY = 1983
EXPECTED_REQUIRED_ROWS = 11475

HIDDEN_SIZE = 768


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(
            f"Required manifest missing: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as source:
        return [
            json.loads(line)
            for line in source
            if line.strip()
        ]


def register_required(
    required: dict[str, Mapping[str, Any]],
    row: Mapping[str, Any],
) -> None:
    row_id = str(row["row_id"])

    existing = required.get(row_id)

    if existing is not None:
        if (
            str(existing["author"])
            != str(row["author"])
            or int(existing["chronological_position"])
            != int(row["chronological_position"])
            or str(existing["context"])
            != str(row["context"])
            or tuple(existing["pinyin_segments"])
            != tuple(row["pinyin_segments"])
        ):
            raise RuntimeError(
                f"Conflicting duplicate row_id: {row_id}"
            )

        return

    required[row_id] = row


def build_surface(
    history: list[dict[str, Any]],
    dev: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[str, Any],
]:
    if any(
        row.get("source_split") == "test"
        for row in history + dev
    ):
        raise RuntimeError(
            "STOP: Test row detected in EM-2B inputs."
        )

    tune = [
        row
        for row in dev
        if row.get("pilot_partition") == "tune"
        and str(row["author"]) in AUTHORS
    ]

    if len(tune) != EXPECTED_TUNE_QUERIES:
        raise RuntimeError(
            "Frozen three-author Dev tune population "
            f"changed: {len(tune)}"
        )

    query_counts = Counter(
        str(row["author"])
        for row in tune
    )

    index = HistoryIndex(
        history + dev,
        HISTORY_BUDGET,
    )

    required: dict[
        str,
        Mapping[str, Any],
    ] = {}

    visible_counts = []
    edge_count = 0
    history_rows = {
        author: set()
        for author in AUTHORS
    }

    edge_counts = Counter()

    for row in tune:
        query = PilotRunner._query(row)

        register_required(
            required,
            row,
        )

        visible = index.visible(query)

        visible_counts.append(
            len(visible)
        )

        edge_count += len(visible)
        edge_counts[
            query.author
        ] += len(visible)

        for item in visible:
            # Independent legality assertions.
            if str(item["author"]) != query.author:
                raise RuntimeError(
                    "HistoryIndex returned wrong author."
                )

            if (
                tuple(item["pinyin_segments"])
                != query.pinyin
            ):
                raise RuntimeError(
                    "HistoryIndex returned wrong Pinyin."
                )

            if (
                int(item["chronological_position"])
                >= query.chronological_position
            ):
                raise RuntimeError(
                    "HistoryIndex returned non-prior row."
                )

            register_required(
                required,
                item,
            )

            history_rows[
                query.author
            ].add(
                str(item["row_id"])
            )

    rows_with_history = sum(
        value > 0
        for value in visible_counts
    )

    rows_without_history = sum(
        value == 0
        for value in visible_counts
    )

    if edge_count != EXPECTED_HISTORY_EDGES:
        raise RuntimeError(
            "History edge surface changed: "
            f"{edge_count} != "
            f"{EXPECTED_HISTORY_EDGES}"
        )

    if (
        rows_with_history
        != EXPECTED_ROWS_WITH_HISTORY
    ):
        raise RuntimeError(
            "Rows-with-history surface changed: "
            f"{rows_with_history}"
        )

    if (
        rows_without_history
        != EXPECTED_ROWS_WITHOUT_HISTORY
    ):
        raise RuntimeError(
            "Rows-without-history surface changed: "
            f"{rows_without_history}"
        )

    if len(required) != EXPECTED_REQUIRED_ROWS:
        raise RuntimeError(
            "Required interaction surface changed: "
            f"{len(required)} != "
            f"{EXPECTED_REQUIRED_ROWS}"
        )

    summary = {
        "tune_queries": len(tune),
        "queries_per_author": dict(
            query_counts
        ),
        "history_edges": edge_count,
        "history_edges_per_author": dict(
            edge_counts
        ),
        "rows_with_history": (
            rows_with_history
        ),
        "rows_without_history": (
            rows_without_history
        ),
        "mean_visible_history": (
            statistics.fmean(
                visible_counts
            )
        ),
        "median_visible_history": (
            statistics.median(
                visible_counts
            )
        ),
        "max_visible_history": max(
            visible_counts
        ),
        "required_interaction_rows": (
            len(required)
        ),
        "unique_history_rows_per_author": {
            author: len(history_rows[author])
            for author in AUTHORS
        },
    }

    return tune, required, summary


def open_cache(
    path: Path,
) -> sqlite3.Connection:
    connection = sqlite3.connect(path)

    connection.execute(
        "PRAGMA journal_mode=WAL"
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS hidden_states (
            row_id TEXT PRIMARY KEY,
            author TEXT NOT NULL,
            chronological_position INTEGER NOT NULL,
            context_sha256 TEXT NOT NULL,
            used_context_sha256 TEXT NOT NULL,
            pinyin_json TEXT NOT NULL,
            original_context_tokens INTEGER NOT NULL,
            used_context_tokens INTEGER NOT NULL,
            context_truncated INTEGER NOT NULL,
            prompt_length INTEGER NOT NULL,
            hidden_size INTEGER NOT NULL,
            vector BLOB NOT NULL
        )
        """
    )

    connection.commit()

    return connection


def existing_ids(
    connection: sqlite3.Connection,
) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT row_id FROM hidden_states"
        )
    }


def validate_existing(
    connection: sqlite3.Connection,
    required: Mapping[
        str,
        Mapping[str, Any],
    ],
) -> None:
    cached_ids = existing_ids(
        connection
    )

    stale = cached_ids - set(required)

    if stale:
        raise RuntimeError(
            "Cache contains rows outside the frozen "
            f"EM-2B surface: {len(stale)}"
        )

    for (
        row_id,
        author,
        chronological_position,
        context_hash,
        pinyin_json,
        hidden_size,
    ) in connection.execute(
        """
        SELECT
            row_id,
            author,
            chronological_position,
            context_sha256,
            pinyin_json,
            hidden_size
        FROM hidden_states
        """
    ):
        source = required[
            str(row_id)
        ]

        expected_pinyin = json.dumps(
            list(
                source[
                    "pinyin_segments"
                ]
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        )

        if (
            str(author)
            != str(source["author"])
            or int(chronological_position)
            != int(
                source[
                    "chronological_position"
                ]
            )
            or str(context_hash)
            != sha256_text(
                str(source["context"])
            )
            or str(pinyin_json)
            != expected_pinyin
            or int(hidden_size)
            != HIDDEN_SIZE
        ):
            raise RuntimeError(
                "Existing cache provenance differs "
                f"for row {row_id}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pilot-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size must be positive"
        )

    history_path = (
        args.pilot_root
        / "history_manifest.jsonl"
    )

    dev_path = (
        args.pilot_root
        / "dev_manifest.jsonl"
    )

    history = read_jsonl(
        history_path
    )

    dev = read_jsonl(
        dev_path
    )

    (
        tune,
        required,
        surface,
    ) = build_surface(
        history,
        dev,
    )

    del tune

    print(
        "=== EM-2B frozen workload ==="
    )
    print(
        f"Required rows: "
        f"{len(required)}"
    )
    print(
        f"Legal history edges: "
        f"{surface['history_edges']}"
    )
    print(
        f"Rows with history: "
        f"{surface['rows_with_history']}"
    )
    print(
        "Gold inspected: False"
    )
    print(
        "Retrieval metrics inspected: False"
    )
    print()

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = (
        args.output_root
        / "hidden_states.sqlite3"
    )

    connection = open_cache(
        cache_path
    )

    try:
        validate_existing(
            connection,
            required,
        )

        cached = existing_ids(
            connection
        )

        missing_ids = (
            set(required) - cached
        )

        print(
            f"Already cached: "
            f"{len(cached)}"
        )
        print(
            f"Pending: "
            f"{len(missing_ids)}"
        )
        print()

        if missing_ids:
            print(
                "Loading Frozen PinyinGPT..."
            )

            backend = (
                PinyinGPTConcatBackend(
                    args.checkpoint,
                    device=args.device,
                )
            )

            if (
                int(
                    backend.model.config.n_embd
                )
                != HIDDEN_SIZE
            ):
                raise RuntimeError(
                    "Frozen PinyinGPT hidden size "
                    "changed."
                )

            # Prepare prompts first, then group by
            # exact prompt length. This avoids padding
            # changing the final-token semantics.
            prepared_by_length: dict[
                int,
                list[dict[str, Any]],
            ] = defaultdict(list)

            truncated_count = 0

            for row_id in sorted(
                missing_ids,
                key=lambda value: (
                    hashlib.sha256(
                        value.encode("utf-8")
                    ).hexdigest()
                ),
            ):
                row = required[
                    row_id
                ]

                context = str(
                    row["context"]
                )

                pinyin = tuple(
                    str(value)
                    for value in row[
                        "pinyin_segments"
                    ]
                )

                (
                    used_context,
                    original_tokens,
                    used_tokens,
                    truncated,
                ) = (
                    backend
                    .truncate_context_for_generation(
                        context,
                        pinyin,
                    )
                )

                if truncated:
                    truncated_count += 1

                (
                    prompt_ids,
                    prompt_positions,
                ) = backend._prompt(
                    used_context,
                    pinyin,
                )

                prepared = {
                    "row_id": row_id,
                    "author": str(
                        row["author"]
                    ),
                    "chronological_position": int(
                        row[
                            "chronological_position"
                        ]
                    ),
                    "context_sha256": (
                        sha256_text(
                            context
                        )
                    ),
                    "used_context_sha256": (
                        sha256_text(
                            used_context
                        )
                    ),
                    "pinyin_json": json.dumps(
                        list(pinyin),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "original_context_tokens": int(
                        original_tokens
                    ),
                    "used_context_tokens": int(
                        used_tokens
                    ),
                    "context_truncated": int(
                        bool(truncated)
                    ),
                    "prompt_ids": prompt_ids,
                    "prompt_positions": (
                        prompt_positions
                    ),
                }

                prepared_by_length[
                    len(prompt_ids)
                ].append(
                    prepared
                )

            total_pending = sum(
                len(values)
                for values
                in prepared_by_length.values()
            )

            processed = 0
            started = time.perf_counter()

            torch = backend.torch

            for prompt_length in sorted(
                prepared_by_length
            ):
                group = (
                    prepared_by_length[
                        prompt_length
                    ]
                )

                for start in range(
                    0,
                    len(group),
                    args.batch_size,
                ):
                    batch = group[
                        start:
                        start
                        + args.batch_size
                    ]

                    input_ids = torch.tensor(
                        [
                            item["prompt_ids"]
                            for item in batch
                        ],
                        device=backend.device,
                    )

                    position_ids = torch.tensor(
                        [
                            item[
                                "prompt_positions"
                            ]
                            for item in batch
                        ],
                        device=backend.device,
                    )

                    with torch.inference_mode():
                        output = backend.model(
                            input_ids=input_ids,
                            position_ids=(
                                position_ids
                            ),
                            output_hidden_states=True,
                            return_dict=True,
                        )

                    hidden = (
                        output.hidden_states[-1]
                        [:, -1, :]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(
                            "<f4",
                            copy=False,
                        )
                    )

                    if hidden.shape != (
                        len(batch),
                        HIDDEN_SIZE,
                    ):
                        raise RuntimeError(
                            "Unexpected hidden-state "
                            f"shape: {hidden.shape}"
                        )

                    payload = []

                    for item, vector in zip(
                        batch,
                        hidden,
                    ):
                        payload.append(
                            (
                                item["row_id"],
                                item["author"],
                                item[
                                    "chronological_position"
                                ],
                                item[
                                    "context_sha256"
                                ],
                                item[
                                    "used_context_sha256"
                                ],
                                item[
                                    "pinyin_json"
                                ],
                                item[
                                    "original_context_tokens"
                                ],
                                item[
                                    "used_context_tokens"
                                ],
                                item[
                                    "context_truncated"
                                ],
                                prompt_length,
                                HIDDEN_SIZE,
                                sqlite3.Binary(
                                    vector.tobytes()
                                ),
                            )
                        )

                    connection.executemany(
                        """
                        INSERT INTO hidden_states (
                            row_id,
                            author,
                            chronological_position,
                            context_sha256,
                            used_context_sha256,
                            pinyin_json,
                            original_context_tokens,
                            used_context_tokens,
                            context_truncated,
                            prompt_length,
                            hidden_size,
                            vector
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        payload,
                    )

                    connection.commit()

                    processed += len(batch)

                    if (
                        processed % 250 < len(batch)
                        or processed
                        == total_pending
                    ):
                        elapsed = (
                            time.perf_counter()
                            - started
                        )

                        rate = (
                            processed / elapsed
                            if elapsed
                            else 0.0
                        )

                        print(
                            f"EM-2B hidden cache: "
                            f"{processed}/"
                            f"{total_pending} "
                            f"({rate:.2f} rows/s)",
                            flush=True,
                        )

            print()
            print(
                "New rows whose context required "
                f"model truncation: {truncated_count}"
            )

        final_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM hidden_states
            """
        ).fetchone()[0]

        if (
            int(final_count)
            != EXPECTED_REQUIRED_ROWS
        ):
            raise RuntimeError(
                "Final cache surface incomplete: "
                f"{final_count}"
            )

        dimensions = {
            int(row[0])
            for row in connection.execute(
                """
                SELECT DISTINCT hidden_size
                FROM hidden_states
                """
            )
        }

        if dimensions != {HIDDEN_SIZE}:
            raise RuntimeError(
                "Hidden dimension provenance "
                f"differs: {dimensions}"
            )

        vector_lengths = {
            int(row[0])
            for row in connection.execute(
                """
                SELECT DISTINCT length(vector)
                FROM hidden_states
                """
            )
        }

        expected_bytes = (
            HIDDEN_SIZE * 4
        )

        if vector_lengths != {
            expected_bytes
        }:
            raise RuntimeError(
                "Stored vector byte length differs: "
                f"{vector_lengths}"
            )

        total_truncated = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM hidden_states
                WHERE context_truncated = 1
                """
            ).fetchone()[0]
        )

        prompt_stats = (
            connection.execute(
                """
                SELECT
                    MIN(prompt_length),
                    MAX(prompt_length),
                    AVG(prompt_length)
                FROM hidden_states
                """
            ).fetchone()
        )

        connection.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        )

        connection.commit()

    finally:
        connection.close()

    cache_hash = sha256_file(
        cache_path
    )

    summary = {
        "schema_version": 1,
        "experiment": (
            "em2b_dev_hidden_state_cache"
        ),
        "partition": "dev_tune_only",
        "authors": list(AUTHORS),
        "condition": "Full+Short",
        "history_budget": (
            HISTORY_BUDGET
        ),
        "representation": (
            "Frozen PinyinGPT final-layer "
            "hidden state at final prompt [SEP]"
        ),
        "hidden_size": HIDDEN_SIZE,
        "dtype": "float32 little-endian",
        "history_semantics": (
            "HistoryIndex(history + dev, H5000); "
            "budget before exact-Pinyin filtering"
        ),
        "surface": surface,
        "cached_rows": (
            EXPECTED_REQUIRED_ROWS
        ),
        "context_truncated_rows": (
            total_truncated
        ),
        "prompt_length": {
            "min": int(
                prompt_stats[0]
            ),
            "max": int(
                prompt_stats[1]
            ),
            "mean": float(
                prompt_stats[2]
            ),
        },
        "used_gold": False,
        "target_used_in_representation": (
            False
        ),
        "retrieval_metrics_inspected": (
            False
        ),
        "provenance": {
            "history_manifest_sha256": (
                sha256_file(
                    history_path
                )
            ),
            "dev_manifest_sha256": (
                sha256_file(
                    dev_path
                )
            ),
            "cache_sha256_local": (
                cache_hash
            ),
        },
    }

    with (
        args.output_root
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as destination:
        json.dump(
            summary,
            destination,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

        destination.write("\n")

    print()
    print(
        "=== EM-2B Dev Hidden-State Cache ==="
    )
    print(
        f"Cached rows: "
        f"{summary['cached_rows']}"
    )
    print(
        f"Hidden size: "
        f"{summary['hidden_size']}"
    )
    print(
        "Context-truncated rows: "
        f"{summary['context_truncated_rows']}"
    )
    print(
        "Prompt length: "
        f"min={summary['prompt_length']['min']} "
        f"mean={summary['prompt_length']['mean']:.2f} "
        f"max={summary['prompt_length']['max']}"
    )
    print(
        f"SQLite SHA256: "
        f"{cache_hash}"
    )
    print(
        f"Gold used: "
        f"{summary['used_gold']}"
    )
    print(
        "Target used in representation: "
        f"{summary['target_used_in_representation']}"
    )
    print(
        "Retrieval metrics inspected: "
        f"{summary['retrieval_metrics_inspected']}"
    )
    print()
    print(
        "PASS: EM-2B Dev representation "
        "cache is complete."
    )


if __name__ == "__main__":
    main()
