from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

import audit_m2_runtime_candidate_lattice_v1 as m2rt
import run_multi_m1c_pv_expansion_frozen_v2 as m1c


SCHEMA_VERSION = 1
SPAN_TOP_K = 3
BEAM_SIZE = 16

EXPECTED = {
    "fit": "547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6",
    "val": "d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220",
    "fit_multi": "cb9f02ababbb7e4272a3dd39314425f56c0263ce35d48dd075c2969a1e13beb1",
    "val_multi": "9789aaa3d5f5a2276f31b18b8047c43c210061a13fd7a2383f0b04947c2d7c9b",
    "m0_rows": "59cb59170e348c60747522fef2c0af94deaaa0cc9cd022db273063249618915e",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                ) + "\n"
            )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def entropy_concentration(
    counts: Mapping[str, int],
) -> float:
    """
    Frozen Full definition:
      0 if empty
      1 if one distinct target
      otherwise 1 - normalized entropy
    """
    values = [
        int(v)
        for v in counts.values()
        if int(v) > 0
    ]

    if not values:
        return 0.0

    if len(values) == 1:
        return 1.0

    total = float(sum(values))

    probs = [
        float(v) / total
        for v in values
    ]

    entropy = -sum(
        p * math.log(p)
        for p in probs
    )

    normalized = entropy / math.log(len(values))

    return max(
        0.0,
        min(1.0, 1.0 - normalized),
    )


def quantile(
    values: Sequence[float],
    q: float,
) -> float | None:
    if not values:
        return None

    ordered = sorted(float(x) for x in values)

    if len(ordered) == 1:
        return ordered[0]

    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return ordered[lo]

    frac = pos - lo

    return (
        ordered[lo] * (1.0 - frac)
        + ordered[hi] * frac
    )


def stats(
    values: Sequence[int | float],
) -> dict[str, float | None]:
    vals = [float(x) for x in values]

    if not vals:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }

    return {
        "mean": statistics.fmean(vals),
        "median": quantile(vals, 0.50),
        "p95": quantile(vals, 0.95),
        "max": max(vals),
    }


def c0_score(
    *,
    weighted_log_frequency: float,
    weighted_choice_share: float,
    weighted_entropy_concentration: float,
    longest_span_ratio: float,
    fragmentation_ratio: float,
) -> float:
    """
    Provisional M4 Composition C0.

    No PinyinGPT / NGram / BGE / frozen 25-feature LambdaMART.
    """
    return (
        weighted_log_frequency
        + 6.0 * weighted_choice_share
        + 4.0 * weighted_entropy_concentration
        + 2.0 * longest_span_ratio
        - 2.0 * fragmentation_ratio
    )


def state_sort_key(
    state: Mapping[str, Any],
) -> tuple[Any, ...]:
    return (
        -float(state["c0_score"]),
        int(state["piece_count"]),
        -float(state["longest_span_ratio"]),
        str(state["text"]),
        tuple(
            (
                int(piece["start"]),
                int(piece["end"]),
                int(piece["frequency_rank"]),
                str(piece["target"]),
            )
            for piece in state["pieces"]
        ),
    )


def build_composition_edges(
    *,
    lookup: m2rt.RuntimeExpandedLookup,
    author: str,
    position: int,
    pinyin: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Gold-blind Composition retrieval.

    Every contiguous SHORTER span is considered.
    Whole-query exact span is excluded.
    Each span first takes frozen frequency Top3.
    """
    n = len(pinyin)

    edges: list[dict[str, Any]] = []

    counters = Counter(
        spans_considered=0,
        whole_exact_span_excluded=0,
        spans_with_history=0,
        retained_targets_before_compatibility=0,
        incompatible_targets_removed=0,
        retained_targets_after_compatibility=0,
    )

    for start in range(n):
        for end in range(start + 1, n + 1):
            counters["spans_considered"] += 1

            if start == 0 and end == n:
                counters["whole_exact_span_excluded"] += 1
                continue

            span = pinyin[start:end]

            counts = lookup.target_counts(
                author=author,
                position=position,
                pinyin=span,
            )

            if not counts:
                continue

            counters["spans_with_history"] += 1

            total = int(sum(counts.values()))

            concentration = entropy_concentration(
                counts
            )

            # Preserve R0 Top3 semantics first.
            ranked = m2rt.ranked_targets(
                counts,
                SPAN_TOP_K,
            )

            counters[
                "retained_targets_before_compatibility"
            ] += len(ranked)

            retained = []

            for item in ranked:
                target = str(item["target"])

                # Runtime compatibility only.
                # Gold is never consulted.
                if len(target) != end - start:
                    counters[
                        "incompatible_targets_removed"
                    ] += 1
                    continue

                frequency = int(item["frequency"])

                retained.append(
                    {
                        "target": target,
                        "frequency": frequency,
                        "frequency_rank": int(
                            item["frequency_rank"]
                        ),
                        "choice_share": (
                            float(frequency)
                            / float(total)
                        ),
                        "entropy_concentration": (
                            concentration
                        ),
                    }
                )

            if not retained:
                continue

            counters[
                "retained_targets_after_compatibility"
            ] += len(retained)

            edges.append(
                {
                    "start": start,
                    "end": end,
                    "syllable_length": end - start,
                    "pinyin": list(span),
                    "history_total": total,
                    "distinct_history_targets": (
                        len(counts)
                    ),
                    "retained_targets": retained,
                }
            )

    return edges, dict(counters)


def extend_state(
    state: Mapping[str, Any],
    edge: Mapping[str, Any],
    target: Mapping[str, Any],
    full_length: int,
) -> dict[str, Any]:
    span_length = int(
        edge["syllable_length"]
    )

    weight = (
        float(span_length)
        / float(full_length)
    )

    piece = {
        "start": int(edge["start"]),
        "end": int(edge["end"]),
        "syllable_length": span_length,
        "pinyin": list(edge["pinyin"]),
        "target": str(target["target"]),
        "frequency": int(target["frequency"]),
        "frequency_rank": int(
            target["frequency_rank"]
        ),
        "choice_share": float(
            target["choice_share"]
        ),
        "entropy_concentration": float(
            target["entropy_concentration"]
        ),
    }

    pieces = [
        *state["pieces"],
        piece,
    ]

    weighted_log_frequency = (
        float(
            state["weighted_log_frequency"]
        )
        + weight
        * math.log1p(
            int(target["frequency"])
        )
    )

    weighted_choice_share = (
        float(
            state["weighted_choice_share"]
        )
        + weight
        * float(target["choice_share"])
    )

    weighted_entropy_concentration = (
        float(
            state[
                "weighted_entropy_concentration"
            ]
        )
        + weight
        * float(
            target["entropy_concentration"]
        )
    )

    piece_count = len(pieces)

    longest_span = max(
        int(state["longest_span"]),
        span_length,
    )

    longest_span_ratio = (
        float(longest_span)
        / float(full_length)
    )

    fragmentation_ratio = (
        float(piece_count - 1)
        / float(full_length - 1)
        if full_length > 1
        else 0.0
    )

    frequencies = [
        int(x["frequency"])
        for x in pieces
    ]

    choice_shares = [
        float(x["choice_share"])
        for x in pieces
    ]

    entropies = [
        float(
            x["entropy_concentration"]
        )
        for x in pieces
    ]

    score = c0_score(
        weighted_log_frequency=(
            weighted_log_frequency
        ),
        weighted_choice_share=(
            weighted_choice_share
        ),
        weighted_entropy_concentration=(
            weighted_entropy_concentration
        ),
        longest_span_ratio=(
            longest_span_ratio
        ),
        fragmentation_ratio=(
            fragmentation_ratio
        ),
    )

    return {
        "position": int(edge["end"]),
        "text": (
            str(state["text"])
            + str(target["target"])
        ),
        "pieces": pieces,
        "piece_count": piece_count,
        "longest_span": longest_span,
        "weighted_log_frequency": (
            weighted_log_frequency
        ),
        "weighted_choice_share": (
            weighted_choice_share
        ),
        "weighted_entropy_concentration": (
            weighted_entropy_concentration
        ),
        "longest_span_ratio": (
            longest_span_ratio
        ),
        "fragmentation_ratio": (
            fragmentation_ratio
        ),
        "average_piece_length": (
            float(full_length)
            / float(piece_count)
        ),
        "min_piece_frequency": min(
            frequencies
        ),
        "max_piece_frequency": max(
            frequencies
        ),
        "min_piece_choice_share": min(
            choice_shares
        ),
        "max_piece_choice_share": max(
            choice_shares
        ),
        "min_piece_entropy": min(
            entropies
        ),
        "max_piece_entropy": max(
            entropies
        ),
        "number_of_derivations": int(
            state["number_of_derivations"]
        ),
        "c0_score": score,
    }


def same_text_viterbi(
    states: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """
    Deterministically keep the best state for each produced text.

    Derivation multiplicity is accumulated as a diagnostic feature.
    """
    grouped: dict[
        str,
        list[Mapping[str, Any]],
    ] = defaultdict(list)

    for state in states:
        grouped[str(state["text"])].append(
            state
        )

    unique = []

    duplicates_removed = 0

    for text, variants in grouped.items():
        ordered = sorted(
            variants,
            key=state_sort_key,
        )

        best = dict(ordered[0])

        best["number_of_derivations"] = sum(
            int(
                x.get(
                    "number_of_derivations",
                    1,
                )
            )
            for x in variants
        )

        unique.append(best)

        duplicates_removed += (
            len(variants) - 1
        )

    unique.sort(
        key=state_sort_key
    )

    return unique, duplicates_removed


def bounded_decode(
    *,
    n: int,
    edges: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, int],
]:
    """
    Bounded endpoint DP / beam.

    Beam16 is applied at each Pinyin endpoint.
    Same-text Viterbi happens before endpoint pruning.
    """
    by_start: dict[
        int,
        list[Mapping[str, Any]],
    ] = defaultdict(list)

    for edge in edges:
        by_start[
            int(edge["start"])
        ].append(edge)

    for start in by_start:
        by_start[start].sort(
            key=lambda edge: (
                -int(
                    edge["syllable_length"]
                ),
                int(edge["end"]),
                tuple(
                    (
                        int(
                            t["frequency_rank"]
                        ),
                        str(t["target"]),
                    )
                    for t
                    in edge[
                        "retained_targets"
                    ]
                ),
            )
        )

    initial = {
        "position": 0,
        "text": "",
        "pieces": [],
        "piece_count": 0,
        "longest_span": 0,
        "weighted_log_frequency": 0.0,
        "weighted_choice_share": 0.0,
        "weighted_entropy_concentration": 0.0,
        "longest_span_ratio": 0.0,
        "fragmentation_ratio": 0.0,
        "average_piece_length": 0.0,
        "min_piece_frequency": 0,
        "max_piece_frequency": 0,
        "min_piece_choice_share": 0.0,
        "max_piece_choice_share": 0.0,
        "min_piece_entropy": 0.0,
        "max_piece_entropy": 0.0,
        "number_of_derivations": 1,
        "c0_score": 0.0,
    }

    pending: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    pending[0].append(initial)

    counters = Counter(
        expanded_states=0,
        target_expansions=0,
        same_text_duplicates_removed=0,
        beam_pruned_states=0,
    )

    for pos in range(n):
        raw_states = pending.get(
            pos,
            [],
        )

        states, removed = (
            same_text_viterbi(
                raw_states
            )
        )

        counters[
            "same_text_duplicates_removed"
        ] += removed

        if len(states) > BEAM_SIZE:
            counters[
                "beam_pruned_states"
            ] += (
                len(states) - BEAM_SIZE
            )
            states = states[:BEAM_SIZE]

        counters[
            "expanded_states"
        ] += len(states)

        for state in states:
            for edge in by_start.get(
                pos,
                (),
            ):
                for target in edge[
                    "retained_targets"
                ]:
                    counters[
                        "target_expansions"
                    ] += 1

                    pending[
                        int(edge["end"])
                    ].append(
                        extend_state(
                            state,
                            edge,
                            target,
                            n,
                        )
                    )

    finals, removed = same_text_viterbi(
        pending.get(n, [])
    )

    counters[
        "same_text_duplicates_removed"
    ] += removed

    if len(finals) > BEAM_SIZE:
        counters[
            "beam_pruned_states"
        ] += (
            len(finals)
            - BEAM_SIZE
        )

        finals = finals[:BEAM_SIZE]

    finals.sort(
        key=state_sort_key
    )

    for rank, row in enumerate(
        finals,
        start=1,
    ):
        row["rank"] = rank

    counters[
        "final_unique_candidates"
    ] = len(finals)

    return finals, dict(counters)


def gold_composable(
    *,
    n: int,
    gold: str,
    edges: Sequence[Mapping[str, Any]],
) -> bool:
    """
    Evaluation only.

    Candidate generation and scoring have already finished without gold.
    """
    by_start: dict[
        int,
        list[Mapping[str, Any]],
    ] = defaultdict(list)

    for edge in edges:
        by_start[
            int(edge["start"])
        ].append(edge)

    reachable = [False] * (n + 1)
    reachable[0] = True

    for pos in range(n):
        if not reachable[pos]:
            continue

        for edge in by_start.get(
            pos,
            (),
        ):
            end = int(edge["end"])

            expected = gold[pos:end]

            if any(
                str(target["target"])
                == expected
                for target
                in edge[
                    "retained_targets"
                ]
            ):
                reachable[end] = True

    return bool(reachable[n])


def gold_rank(
    candidates: Sequence[
        Mapping[str, Any]
    ],
    gold: str,
) -> int | None:
    for row in candidates:
        if str(row["text"]) == gold:
            return int(row["rank"])

    return None


def summarize(
    rows: Sequence[
        Mapping[str, Any]
    ],
) -> dict[str, Any]:
    n = len(rows)

    if n == 0:
        return {"n": 0}

    ranks = [
        row["gold_rank"]
        for row in rows
    ]

    result: dict[str, Any] = {
        "n": n,
        "candidate_nonempty": sum(
            int(row["candidate_count"]) > 0
            for row in rows
        ),
        "candidate_nonempty_rate": (
            sum(
                int(
                    row[
                        "candidate_count"
                    ]
                ) > 0
                for row in rows
            )
            / n
        ),
        "candidate_count": stats(
            [
                int(
                    row[
                        "candidate_count"
                    ]
                )
                for row in rows
            ]
        ),
        "k3_gold_composable": sum(
            bool(
                row[
                    "k3_gold_composable"
                ]
            )
            for row in rows
        ),
    }

    result[
        "k3_gold_composable_rate"
    ] = (
        result["k3_gold_composable"]
        / n
    )

    for k in (1, 3, 5, 10):
        hits = sum(
            rank is not None
            and int(rank) <= k
            for rank in ranks
        )

        result[f"hits_at_{k}"] = hits
        result[f"recall_at_{k}"] = (
            hits / n
        )

    result["recall_any"] = (
        sum(
            rank is not None
            for rank in ranks
        )
        / n
    )

    result["mrr"] = (
        sum(
            (
                1.0 / int(rank)
                if rank is not None
                else 0.0
            )
            for rank in ranks
        )
        / n
    )

    composable = int(
        result[
            "k3_gold_composable"
        ]
    )

    if composable:
        result[
            "recall_at_5_given_composable"
        ] = (
            sum(
                bool(
                    row[
                        "k3_gold_composable"
                    ]
                )
                and row[
                    "gold_rank"
                ] is not None
                and int(
                    row[
                        "gold_rank"
                    ]
                ) <= 5
                for row in rows
            )
            / composable
        )
    else:
        result[
            "recall_at_5_given_composable"
        ] = None

    return result


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--fit",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--val",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--fit-multi",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--val-multi",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--m0-rows",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--max-rows",
        type=int,
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=100,
    )

    args = ap.parse_args()

    checks = (
        ("fit", args.fit),
        ("val", args.val),
        ("fit_multi", args.fit_multi),
        ("val_multi", args.val_multi),
        ("m0_rows", args.m0_rows),
    )

    for name, path in checks:
        got = sha256_file(path)

        expected = EXPECTED[name]

        if got != expected:
            raise RuntimeError(
                f"SHA mismatch {name}: "
                f"got={got} "
                f"expected={expected}"
            )

        print(
            f"SHA PASS {name} {got}",
            flush=True,
        )

    # This R1 smoke is defined on frozen
    # Train-Val Multi manifest only.
    manifest_sha = sha256_file(
        args.manifest
    )

    if manifest_sha != EXPECTED[
        "val_multi"
    ]:
        raise RuntimeError(
            "R1 manifest is not the frozen "
            "Train-Val Multi population"
        )

    if (
        args.output_root.exists()
        and any(
            args.output_root.iterdir()
        )
    ):
        raise RuntimeError(
            "Refusing to overwrite "
            f"non-empty output: "
            f"{args.output_root}"
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    fit_rows = read_jsonl(
        args.fit
    )
    val_rows = read_jsonl(
        args.val
    )
    fit_multi_rows = read_jsonl(
        args.fit_multi
    )
    val_multi_rows = read_jsonl(
        args.val_multi
    )
    manifest_rows = read_jsonl(
        args.manifest
    )
    m0_rows = read_jsonl(
        args.m0_rows
    )

    selected = manifest_rows

    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise ValueError(
                "--max-rows must be positive"
            )

        selected = selected[
            :args.max_rows
        ]

    val_by_id = {
        str(row["row_id"]): row
        for row in val_rows
    }

    m0_by_id = {
        str(row["row_id"]): row
        for row in m0_rows
    }

    print(
        "Building frozen M1-C "
        "ExpandedCausalHistoryIndex ...",
        flush=True,
    )

    history = (
        m1c.ExpandedCausalHistoryIndex(
            [
                *fit_rows,
                *val_rows,
            ],
            [
                *fit_multi_rows,
                *val_multi_rows,
            ],
        )
    )

    lookup = (
        m2rt.RuntimeExpandedLookup(
            history
        )
    )

    outputs = []

    retrieval_ms = []
    search_ms = []

    wall_started = (
        time.perf_counter()
    )

    for number, multi in enumerate(
        selected,
        start=1,
    ):
        row_id = str(
            multi["row_id"]
        )

        source_id = str(
            multi[
                "source_standard_row_id"
            ]
        )

        anchor = val_by_id.get(
            source_id
        )

        if anchor is None:
            raise RuntimeError(
                "Missing Train-Val "
                f"anchor: {source_id}"
            )

        m0 = m0_by_id.get(
            row_id
        )

        if m0 is None:
            raise RuntimeError(
                f"Missing M0 row: "
                f"{row_id}"
            )

        qrow = m1c.make_query_row(
            multi,
            anchor,
        )

        author = str(
            qrow["author"]
        )

        position = int(
            qrow[
                "chronological_position"
            ]
        )

        pinyin = tuple(
            str(x)
            for x in (
                multi.get(
                    "pinyin_segments"
                )
                or str(
                    multi[
                        "pinyin_input"
                    ]
                ).split()
            )
        )

        gold = str(
            multi["gold"]
        )

        n = len(pinyin)

        if len(gold) != n:
            raise RuntimeError(
                "Gold/Pinyin alignment "
                f"mismatch: {row_id}"
            )

        start_raw, stop_raw = (
            lookup.raw_window(
                author=author,
                position=position,
            )
        )

        raw_count = (
            stop_raw - start_raw
        )

        if raw_count != (
            history.raw_visible_count(
                author=author,
                position=position,
            )
        ):
            raise RuntimeError(
                "H5000 mismatch: "
                f"{row_id}"
            )

        retrieval_started = (
            time.perf_counter()
        )

        edges, edge_counters = (
            build_composition_edges(
                lookup=lookup,
                author=author,
                position=position,
                pinyin=pinyin,
            )
        )

        retrieval_ms.append(
            1000.0
            * (
                time.perf_counter()
                - retrieval_started
            )
        )

        raw_derivations = (
            m2rt.derivation_count(
                n,
                edges,
            )
        )

        search_started = (
            time.perf_counter()
        )

        candidates, search_counters = (
            bounded_decode(
                n=n,
                edges=edges,
            )
        )

        search_ms.append(
            1000.0
            * (
                time.perf_counter()
                - search_started
            )
        )

        # Gold begins here only.
        # Everything above is runtime.
        composable = gold_composable(
            n=n,
            gold=gold,
            edges=edges,
        )

        rank = gold_rank(
            candidates,
            gold,
        )

        generic_survived = (
            m2rt.m0_gold_survived(
                m0
            )
        )

        output = {
            "schema_version": (
                SCHEMA_VERSION
            ),
            "experiment": (
                "m4_r1_bounded_"
                "composition_beam16_v1"
            ),
            "row_id": row_id,
            "family_id": str(
                multi["family_id"]
            ),
            "author": author,
            "multi_token_length": int(
                multi[
                    "multi_token_length"
                ]
            ),
            "pinyin_syllable_length": (
                n
            ),
            "raw_h5000_count": (
                raw_count
            ),
            "generic_gold_survived_final_beam": (
                bool(generic_survived)
            ),
            "generic_beam_pruned": (
                not bool(
                    generic_survived
                )
            ),
            "span_top_k": (
                SPAN_TOP_K
            ),
            "beam_size": (
                BEAM_SIZE
            ),
            "composition_span_edges": (
                len(edges)
            ),
            "composition_retained_target_edges": (
                sum(
                    len(
                        edge[
                            "retained_targets"
                        ]
                    )
                    for edge in edges
                )
            ),
            "raw_composition_derivation_count": (
                int(
                    raw_derivations
                )
            ),
            "edge_counters": (
                edge_counters
            ),
            "search_counters": (
                search_counters
            ),
            "candidate_count": (
                len(candidates)
            ),
            "top10": (
                candidates[:10]
            ),
            "gold_rank": (
                rank
            ),
            "k3_gold_composable": (
                bool(composable)
            ),
            "gold_used_for_candidate_generation": False,
            "gold_used_for_candidate_scoring": False,
            "gold_used_for_recoverability_audit_only": True,
            "composition_uses_pinyingpt": False,
            "composition_uses_ngram": False,
            "composition_uses_bge": False,
            "composition_uses_frozen_25_feature_lambdamart": False,
            "multi_training": False,
            "multi_tuning": False,
            "used_dev3000": False,
            "used_test": False,
        }

        outputs.append(
            output
        )

        if (
            args.progress_every > 0
            and (
                number
                % args.progress_every
                == 0
                or number
                == len(selected)
            )
        ):
            print(
                f"processed="
                f"{number}/"
                f"{len(selected)}",
                flush=True,
            )

    wall_seconds = (
        time.perf_counter()
        - wall_started
    )

    gp_rows = [
        row
        for row in outputs
        if bool(
            row[
                "generic_beam_pruned"
            ]
        )
    ]

    summary = {
        "schema_version": (
            SCHEMA_VERSION
        ),
        "experiment": (
            "m4_r1_bounded_"
            "composition_beam16_v1"
        ),
        "runner_sha256": (
            sha256_file(
                Path(__file__)
            )
        ),
        "n": len(outputs),
        "span_top_k": (
            SPAN_TOP_K
        ),
        "beam_size": (
            BEAM_SIZE
        ),
        "candidate_generation": (
            "strict-prior same-author "
            "frozen M1-C expanded PV; "
            "H5000 raw interactions "
            "before exact-Pinyin lookup; "
            "all contiguous shorter spans; "
            "frequency Top3 per span; "
            "whole-query exact edge excluded"
        ),
        "c0_formula": (
            "weighted_log_frequency "
            "+ 6*weighted_choice_share "
            "+ 4*weighted_entropy_concentration "
            "+ 2*longest_span_ratio "
            "- 2*fragmentation_ratio"
        ),
        "overall": summarize(
            outputs
        ),
        "generic_pruned": summarize(
            gp_rows
        ),
        "by_multi_token_length": {},
        "retrieval_latency_ms": stats(
            retrieval_ms
        ),
        "search_latency_ms": stats(
            search_ms
        ),
        "wall_seconds": (
            wall_seconds
        ),
        "structure": {
            "span_edges": stats(
                [
                    row[
                        "composition_span_edges"
                    ]
                    for row in outputs
                ]
            ),
            "retained_target_edges": stats(
                [
                    row[
                        "composition_retained_target_edges"
                    ]
                    for row in outputs
                ]
            ),
            "raw_derivation_count": stats(
                [
                    row[
                        "raw_composition_derivation_count"
                    ]
                    for row in outputs
                ]
            ),
            "total_target_expansions": sum(
                int(
                    row[
                        "search_counters"
                    ][
                        "target_expansions"
                    ]
                )
                for row in outputs
            ),
            "total_same_text_duplicates_removed": sum(
                int(
                    row[
                        "search_counters"
                    ][
                        "same_text_duplicates_removed"
                    ]
                )
                for row in outputs
            ),
            "total_beam_pruned_states": sum(
                int(
                    row[
                        "search_counters"
                    ][
                        "beam_pruned_states"
                    ]
                )
                for row in outputs
            ),
            "incompatible_targets_removed": sum(
                int(
                    row[
                        "edge_counters"
                    ][
                        "incompatible_targets_removed"
                    ]
                )
                for row in outputs
            ),
        },
        "gold_used_for_candidate_generation": False,
        "gold_used_for_candidate_scoring": False,
        "gold_used_for_recoverability_audit_only": True,
        "used_dev3000": False,
        "used_test": False,
    }

    for length in range(1, 6):
        subset = [
            row
            for row in outputs
            if int(
                row[
                    "multi_token_length"
                ]
            ) == length
        ]

        gp_subset = [
            row
            for row in subset
            if bool(
                row[
                    "generic_beam_pruned"
                ]
            )
        ]

        summary[
            "by_multi_token_length"
        ][str(length)] = {
            "overall": summarize(
                subset
            ),
            "generic_pruned": summarize(
                gp_subset
            ),
        }

    # Regression against frozen R0 smoke1000.
    # This ensures our Top3 retrieval ceiling
    # did not silently change.
    if len(outputs) == 1000:
        if summary[
            "overall"
        ][
            "k3_gold_composable"
        ] != 312:
            raise RuntimeError(
                "R0 Top3 composable "
                "regression failed: "
                f"{summary['overall']['k3_gold_composable']} "
                "!= 312"
            )

        if len(gp_rows) != 300:
            raise RuntimeError(
                "R0 generic-pruned "
                f"regression failed: "
                f"{len(gp_rows)} != 300"
            )

        if summary[
            "generic_pruned"
        ][
            "k3_gold_composable"
        ] != 86:
            raise RuntimeError(
                "R0 GP Top3 composable "
                "regression failed: "
                f"{summary['generic_pruned']['k3_gold_composable']} "
                "!= 86"
            )

        summary[
            "r0_k3_ceiling_regression"
        ] = "PASS"

    write_jsonl(
        args.output_root
        / "rows.jsonl",
        outputs,
    )

    write_json(
        args.output_root
        / "summary.json",
        summary,
    )

    overall = summary[
        "overall"
    ]

    gp = summary[
        "generic_pruned"
    ]

    print()
    print(
        "===== M4-R1 BEAM16 "
        "HEADLINE ====="
    )

    print(
        "Overall:",
        f"R1={overall['recall_at_1']:.6f}",
        f"R3={overall['recall_at_3']:.6f}",
        f"R5={overall['recall_at_5']:.6f}",
        f"R10={overall['recall_at_10']:.6f}",
        f"Any={overall['recall_any']:.6f}",
        f"MRR={overall['mrr']:.6f}",
    )

    print(
        "Generic-pruned:",
        f"R1={gp['recall_at_1']:.6f}",
        f"R3={gp['recall_at_3']:.6f}",
        f"R5={gp['recall_at_5']:.6f}",
        f"R10={gp['recall_at_10']:.6f}",
        f"Any={gp['recall_any']:.6f}",
    )

    print(
        "Top3 theoretical ceiling:",
        overall[
            "k3_gold_composable"
        ],
        "/",
        overall["n"],
    )

    print(
        "GP Top3 theoretical ceiling:",
        gp[
            "k3_gold_composable"
        ],
        "/",
        gp["n"],
    )

    print(
        "R5_given_composable=",
        overall[
            "recall_at_5_given_composable"
        ],
    )

    print(
        "GP_R5_given_composable=",
        gp[
            "recall_at_5_given_composable"
        ],
    )

    print(
        "retrieval_latency_ms=",
        json.dumps(
            summary[
                "retrieval_latency_ms"
            ],
            sort_keys=True,
        ),
    )

    print(
        "search_latency_ms=",
        json.dumps(
            summary[
                "search_latency_ms"
            ],
            sort_keys=True,
        ),
    )

    print(
        "wall_seconds=",
        wall_seconds,
    )

    print(
        "R0_K3_CEILING_REGRESSION=",
        summary.get(
            "r0_k3_ceiling_regression"
        ),
    )

    print(
        "M4_R1_BOUNDED_"
        "COMPOSITION_BEAM16=PASS"
    )


if __name__ == "__main__":
    main()
