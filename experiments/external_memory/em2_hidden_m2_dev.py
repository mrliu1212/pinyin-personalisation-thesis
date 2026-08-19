"""EM-2E2: Hidden-M2 Dev experiment.

Replace only M2 Stage-1 BGE retrieval with frozen PinyinGPT hidden-state
retrieval. Keep the original pretrained Cross-Encoder, pair template,
support aggregation, candidate surface, and M2 search grid unchanged.

Dev tune only. No Test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from experiments.external_memory import em2_hidden_m1_dev as hidden_m1

from src.evaluation.deep_author_v2 import sha256_file
from src.personalisation.candidate_memory_m2 import (
    BGEReranker,
    PairIdentity,
    PairScoreCache,
    RERANKER_MODEL_SHA256,
    RERANKER_REVISION,
    RERANKER_TOKENIZER_SHA256,
    rank_m2,
)
from src.personalisation.context_memory import (
    macro_author_metrics,
    rank_frequency,
    rank_of,
    subset_membership,
)
from src.personalisation.pilot_a import (
    HistoryIndex,
    PilotRunner,
)


AUTHORS = hidden_m1.AUTHORS
HISTORY_BUDGET = 5000

RETRIEVAL_KS = (10, 20)
LAMBDAS = (0.5, 1.0, 2.0, 4.0)

EXPECTED_QUERIES = 5608

LAMBDA_F = 4.0


def pair_of(
    query: Any,
    history: Mapping[str, Any],
) -> PairIdentity:
    return PairIdentity.from_query_history(
        query,
        history,
        str(history["target"]),
    )


def hidden_stage1(
    query: Any,
    visible: Sequence[Mapping[str, Any]],
    vectors: Mapping[str, Any],
    k: int,
) -> tuple[Mapping[str, Any], ...]:
    if not visible:
        return ()

    retrieved = hidden_m1.retrieve_hidden(
        query,
        visible,
        vectors,
    )

    by_id = {
        str(row["row_id"]): row
        for row in visible
    }

    return tuple(
        by_id[
            str(item["historical_interaction_id"])
        ]
        for item in retrieved[:k]
    )


def evidence(
    query: Any,
    histories: Sequence[Mapping[str, Any]],
    cache: PairScoreCache,
) -> tuple[dict[str, Any], ...]:
    values = []

    for history in histories:
        pair = pair_of(query, history)

        score = cache.get(pair)

        if score is None:
            raise RuntimeError(
                "Required Hidden-M2 pair score is absent."
            )

        values.append(
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
                    score["current_context_truncated"],
                "historical_context_truncated":
                    score["historical_context_truncated"],
            }
        )

    return tuple(values)


def subset_rows(rows, name):
    if name == "overall":
        return list(rows)

    return [
        row for row in rows
        if bool(row[name])
    ]


def macro_top1(rows, name):
    selected = subset_rows(rows, name)

    return macro_author_metrics(
        selected,
        "rank",
    )["macro_author"]["top1"]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pilot-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--generic-cache",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--hidden-cache",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--reranker-model",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )

    args = parser.parse_args()

    # Frozen reranker provenance.
    model_file = (
        args.reranker_model
        / "model.safetensors"
    )

    tokenizer_file = (
        args.reranker_model
        / "tokenizer.json"
    )

    if sha256_file(model_file) != RERANKER_MODEL_SHA256:
        raise RuntimeError(
            "Frozen M2 model hash differs."
        )

    if (
        sha256_file(tokenizer_file)
        != RERANKER_TOKENIZER_SHA256
    ):
        raise RuntimeError(
            "Frozen M2 tokenizer hash differs."
        )

    history = hidden_m1.read_jsonl(
        args.pilot_root
        / "history_manifest.jsonl"
    )

    dev = hidden_m1.read_jsonl(
        args.pilot_root
        / "dev_manifest.jsonl"
    )

    if any(
        str(row.get("source_split")) == "test"
        for row in history + dev
    ):
        raise RuntimeError(
            "STOP: Test row detected."
        )

    tune = [
        row
        for row in dev
        if (
            row.get("pilot_partition")
            == "tune"
            and str(row["author"])
            in AUTHORS
        )
    ]

    if len(tune) != EXPECTED_QUERIES:
        raise RuntimeError(
            f"Unexpected tune size: {len(tune)}"
        )

    tune_ids = {
        str(row["row_id"])
        for row in tune
    }

    print("Loading Frozen Generic...")
    generic = hidden_m1.load_generic(
        args.generic_cache,
        tune_ids,
    )

    print("Loading Frozen hidden states...")
    vectors = hidden_m1.load_hidden(
        args.hidden_cache
    )

    index = HistoryIndex(
        history + dev,
        HISTORY_BUDGET,
    )

    # -------------------------------------------------
    # Build frozen Hidden Stage-1 retrieval surface.
    # -------------------------------------------------

    states = []
    requested_pairs = {}
    max_k = max(RETRIEVAL_KS)

    for number, row in enumerate(
        tune,
        start=1,
    ):
        query = PilotRunner._query(row)
        visible = index.visible(query)

        stage1 = hidden_stage1(
            query,
            visible,
            vectors,
            max_k,
        )

        gold = hidden_m1.gold_of(row)

        flags = subset_membership(
            query,
            gold,
            visible,
        )

        candidates = PilotRunner._candidates(
            generic[query.row_id]
        )

        for history_row in stage1:
            pair = pair_of(
                query,
                history_row,
            )

            # Key is deterministic after the cache
            # is opened below. Keep logical identity
            # here to deduplicate first.
            logical = (
                pair.current_id,
                pair.historical_id,
                pair.historical_target,
            )

            requested_pairs.setdefault(
                logical,
                pair,
            )

        states.append(
            {
                "row": row,
                "query": query,
                "gold": gold,
                "visible": visible,
                "stage1": stage1,
                "candidates": candidates,
                "history_available":
                    bool(flags["history_available"]),
                "ambiguous":
                    bool(flags["ambiguous"]),
                "conflict":
                    bool(flags["conflict"]),
            }
        )

        if (
            number % 500 == 0
            or number == len(tune)
        ):
            print(
                f"Hidden-M2 Stage1: "
                f"{number}/{len(tune)}",
                flush=True,
            )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    pair_cache_path = (
        args.output_root
        / "cache"
        / "pair_scores.sqlite3"
    )

    cache = PairScoreCache(
        pair_cache_path,
        model_revision=RERANKER_REVISION,
        model_sha256=RERANKER_MODEL_SHA256,
        tokenizer_sha256=RERANKER_TOKENIZER_SHA256,
        max_length=512,
        dtype="float16",
    )

    pairs = list(requested_pairs.values())

    hits = sum(
        cache.get(pair) is not None
        for pair in pairs
    )

    pending = [
        pair
        for pair in pairs
        if cache.get(pair) is None
    ]

    print()
    print("=== Hidden-M2 Preflight ===")
    print(f"Queries: {len(states)}")
    print(
        f"Requested unique Top20 pairs: "
        f"{len(pairs)}"
    )
    print(f"Cache hits: {hits}")
    print(f"Missing pair scores: {len(pending)}")
    print("Test used: False")

    if args.preflight_only:
        cache.close()
        return

    # -------------------------------------------------
    # Cross-Encoder scoring. Resume-safe cache.
    # -------------------------------------------------

    reranker = BGEReranker(
        args.reranker_model,
        revision=RERANKER_REVISION,
        model_sha256=RERANKER_MODEL_SHA256,
        tokenizer_sha256=RERANKER_TOKENIZER_SHA256,
        batch_size=args.batch_size,
        max_length=512,
    )

    reranker.load()

    started = time.perf_counter()
    added = 0

    try:
        for start in range(
            0,
            len(pending),
            args.batch_size,
        ):
            batch = pending[
                start:start + args.batch_size
            ]

            prepared = [
                reranker.prepare(pair)
                for pair in batch
            ]

            scores = reranker.score_prepared(
                prepared
            )

            for pair, prepared_pair, score in zip(
                batch,
                prepared,
                scores,
            ):
                cache.put(
                    pair,
                    prepared_pair,
                    score,
                )
                added += 1

            if (
                added % (args.batch_size * 10)
                == 0
                or added == len(pending)
            ):
                cache.commit()

                elapsed = (
                    time.perf_counter()
                    - started
                )

                rate = (
                    added / elapsed
                    if elapsed
                    else 0.0
                )

                remaining = (
                    len(pending)
                    - added
                )

                eta = (
                    remaining / rate
                    if rate
                    else 0.0
                )

                print(
                    f"Hidden-M2 pair scoring: "
                    f"{added}/{len(pending)} "
                    f"rate={rate:.2f}/s "
                    f"eta={eta:.1f}s",
                    flush=True,
                )

        cache.commit()

        # ---------------------------------------------
        # Dev grid.
        # ---------------------------------------------

        rows_by_grid = {
            (k, value): []
            for k in RETRIEVAL_KS
            for value in LAMBDAS
        }

        f_rows = []
        g_rows = []

        for number, state in enumerate(
            states,
            start=1,
        ):
            query = state["query"]
            candidates = state["candidates"]
            gold = state["gold"]

            common = {
                "row_id": query.row_id,
                "author": query.author,
                "history_available":
                    state["history_available"],
                "ambiguous":
                    state["ambiguous"],
                "conflict":
                    state["conflict"],
            }

            g_rows.append(
                {
                    **common,
                    "rank":
                        generic[
                            query.row_id
                        ]["gold_rank"],
                }
            )

            f_ranked = rank_frequency(
                query,
                candidates,
                state["visible"],
                lambda_frequency=LAMBDA_F,
            )

            f_rows.append(
                {
                    **common,
                    "rank": rank_of(
                        f_ranked,
                        gold,
                    ),
                }
            )

            for k in RETRIEVAL_KS:
                ev = evidence(
                    query,
                    state["stage1"][:k],
                    cache,
                )

                for value in LAMBDAS:
                    ranked = rank_m2(
                        candidates,
                        ev,
                        lambda_m2=value,
                    )

                    rows_by_grid[
                        (k, value)
                    ].append(
                        {
                            **common,
                            "rank": rank_of(
                                ranked,
                                gold,
                            ),
                        }
                    )

            if (
                number % 500 == 0
                or number == len(states)
            ):
                print(
                    f"Hidden-M2 tune: "
                    f"{number}/{len(states)}",
                    flush=True,
                )

        search = []

        for k in RETRIEVAL_KS:
            for value in LAMBDAS:
                rows = rows_by_grid[
                    (k, value)
                ]

                metric = macro_top1(
                    rows,
                    "overall",
                )

                search.append(
                    {
                        "retrieval_k": k,
                        "lambda_m2": value,
                        "macro_author_overall_top1":
                            metric,
                    }
                )

                print(
                    f"K={k:>2} "
                    f"lambda={value:>3} "
                    f"MacroTop1={metric:.6f}"
                )

        selected = max(
            search,
            key=lambda row: (
                float(
                    row[
                        "macro_author_overall_top1"
                    ]
                ),
                -float(row["lambda_m2"]),
                -int(row["retrieval_k"]),
            ),
        )

        selected_rows = rows_by_grid[
            (
                int(selected["retrieval_k"]),
                float(selected["lambda_m2"]),
            )
        ]

        print()
        print(
            "=== EM-2E2 Hidden-M2 Dev Selection ==="
        )
        print(
            "Selected K:",
            selected["retrieval_k"],
        )
        print(
            "Selected lambda_m2:",
            selected["lambda_m2"],
        )

        print()
        print(
            f"{'Method':12s}"
            f"{'Overall':>12s}"
            f"{'History':>12s}"
            f"{'Ambiguous':>12s}"
            f"{'Conflict':>12s}"
        )

        methods = {
            "G": g_rows,
            "F": f_rows,
            "Hidden-M2": selected_rows,
        }

        metrics = {}

        for name, rows in methods.items():
            values = {
                subset: macro_top1(
                    rows,
                    subset,
                )
                for subset in (
                    "overall",
                    "history_available",
                    "ambiguous",
                    "conflict",
                )
            }

            metrics[name] = values

            print(
                f"{name:12s}"
                f"{values['overall']:12.6f}"
                f"{values['history_available']:12.6f}"
                f"{values['ambiguous']:12.6f}"
                f"{values['conflict']:12.6f}"
            )

        summary = {
            "status": "dev_selection_complete",
            "experiment":
                "em2e2_hidden_m2_dev",
            "queries": len(states),
            "authors": list(AUTHORS),
            "history_budget":
                HISTORY_BUDGET,
            "stage1":
                "Frozen PinyinGPT hidden cosine",
            "stage2":
                "Frozen pretrained bge-reranker-base",
            "retrieval_k_grid":
                list(RETRIEVAL_KS),
            "lambda_m2_grid":
                list(LAMBDAS),
            "selected": selected,
            "metrics": metrics,
            "requested_unique_pairs":
                len(pairs),
            "pair_cache_hits_before":
                hits,
            "pair_scores_added":
                added,
            "test_used": False,
        }

        (
            args.output_root
            / "summary.json"
        ).write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        (
            args.output_root
            / "grid.json"
        ).write_text(
            json.dumps(
                search,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print()
        print("Test used: False")

    finally:
        cache.close()


if __name__ == "__main__":
    main()
