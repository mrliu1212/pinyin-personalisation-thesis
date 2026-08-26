from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve()
M4 = HERE.parents[1]
DST = M4.parent
M2 = DST / "m2_lattice_v1"

sys.path.insert(0, str(DST))
sys.path.insert(0, str(M2))

import run_multi_m1c_pv_expansion_frozen_v2 as m1c
import audit_m2_runtime_candidate_lattice_v1 as m2rt


KS = (1, 3, 5)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def target_rank(counts: Counter[str], target: str) -> int | None:
    if target not in counts:
        return None
    ordered = sorted(
        counts.items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    )
    for rank, (text, _) in enumerate(ordered, start=1):
        if text == target:
            return rank
    raise AssertionError("target present but rank missing")


def build_span_stats(
    *,
    lookup: m2rt.RuntimeExpandedLookup,
    author: str,
    position: int,
    pinyin: tuple[str, ...],
    gold: str,
) -> list[dict[str, Any]]:
    """
    Candidate retrieval is gold-blind.

    Gold is consulted only after each span's complete historical target-count
    distribution has been retrieved, to measure recoverability/rank.
    """
    stats: list[dict[str, Any]] = []
    n = len(pinyin)

    for start in range(n):
        for end in range(start + 1, n + 1):
            span = pinyin[start:end]

            counts = lookup.target_counts(
                author=author,
                position=position,
                pinyin=span,
            )

            if not counts:
                continue

            gold_slice = gold[start:end]

            stats.append(
                {
                    "start": start,
                    "end": end,
                    "syllable_length": end - start,
                    "distinct_targets": len(counts),
                    "history_occurrences": int(sum(counts.values())),
                    "gold_target_rank_by_frequency": target_rank(
                        counts,
                        gold_slice,
                    ),
                }
            )

    return stats


def retained_count(edge: Mapping[str, Any], k: int | None) -> int:
    distinct = int(edge["distinct_targets"])
    return distinct if k is None else min(distinct, int(k))


def candidate_derivation_count(
    *,
    n: int,
    edges: Sequence[Mapping[str, Any]],
    k: int | None,
    exclude_whole: bool,
) -> int:
    by_start: dict[int, list[Mapping[str, Any]]] = {}

    for edge in edges:
        start = int(edge["start"])
        end = int(edge["end"])

        if exclude_whole and start == 0 and end == n:
            continue

        by_start.setdefault(start, []).append(edge)

    dp = [0] * (n + 1)
    dp[0] = 1

    for pos in range(n):
        if dp[pos] == 0:
            continue

        for edge in by_start.get(pos, ()):
            end = int(edge["end"])
            multiplicity = retained_count(edge, k)

            if multiplicity:
                dp[end] += dp[pos] * multiplicity

    return int(dp[n])


def best_gold_composition_path(
    *,
    n: int,
    edges: Sequence[Mapping[str, Any]],
    k: int | None,
) -> list[dict[str, Any]] | None:
    """
    Evaluation only.

    Finds the minimum-piece derivation that reconstructs gold from retained
    shorter-span candidates. Whole-query exact edges are deliberately excluded.
    """
    by_start: dict[int, list[dict[str, Any]]] = {}

    for raw in edges:
        edge = dict(raw)
        start = int(edge["start"])
        end = int(edge["end"])

        if start == 0 and end == n:
            continue

        rank = edge.get("gold_target_rank_by_frequency")
        if rank is None:
            continue

        if k is not None and int(rank) > int(k):
            continue

        by_start.setdefault(start, []).append(edge)

    for start in by_start:
        by_start[start].sort(
            key=lambda edge: (
                -int(edge["syllable_length"]),
                int(edge["end"]),
                int(edge["gold_target_rank_by_frequency"]),
            )
        )

    best: dict[int, list[dict[str, Any]]] = {0: []}

    def key(path: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
        return (
            len(path),
            tuple(
                (
                    -int(e["syllable_length"]),
                    int(e["start"]),
                    int(e["end"]),
                    int(e["gold_target_rank_by_frequency"]),
                )
                for e in path
            ),
        )

    for pos in range(n):
        prefix = best.get(pos)
        if prefix is None:
            continue

        for edge in by_start.get(pos, ()):
            end = int(edge["end"])
            candidate = [*prefix, edge]
            old = best.get(end)

            if old is None or key(candidate) < key(old):
                best[end] = candidate

    result = best.get(n)

    if result is None or len(result) < 2:
        return None

    return result


def whole_gold_retained(
    *,
    n: int,
    edges: Sequence[Mapping[str, Any]],
    k: int | None,
) -> bool:
    for edge in edges:
        if int(edge["start"]) != 0 or int(edge["end"]) != n:
            continue

        rank = edge.get("gold_target_rank_by_frequency")
        if rank is None:
            return False

        return k is None or int(rank) <= int(k)

    return False


def shape(path: Sequence[Mapping[str, Any]] | None) -> str | None:
    if not path:
        return None
    return "+".join(str(int(edge["syllable_length"])) for edge in path)


def pct(num: int, den: int) -> float:
    return 0.0 if den == 0 else num / den


def summarize_subset(
    rows: Sequence[Mapping[str, Any]],
    prefix: str,
) -> dict[str, Any]:
    n = len(rows)

    exact = sum(bool(x[f"{prefix}_whole_exact"]) for x in rows)
    comp = sum(bool(x[f"{prefix}_composable"]) for x in rows)
    adds = sum(bool(x[f"{prefix}_composition_adds_over_exact"]) for x in rows)
    either = sum(bool(x[f"{prefix}_either"]) for x in rows)

    pruned = [x for x in rows if bool(x["generic_beam_pruned"])]
    p_n = len(pruned)

    p_exact = sum(bool(x[f"{prefix}_whole_exact"]) for x in pruned)
    p_comp = sum(bool(x[f"{prefix}_composable"]) for x in pruned)
    p_adds = sum(
        bool(x[f"{prefix}_composition_adds_over_exact"])
        for x in pruned
    )
    p_either = sum(bool(x[f"{prefix}_either"]) for x in pruned)

    pieces = Counter(
        str(x[f"{prefix}_min_composition_pieces"])
        for x in rows
        if x[f"{prefix}_min_composition_pieces"] is not None
    )

    shapes = Counter(
        str(x[f"{prefix}_composition_shape"])
        for x in rows
        if x[f"{prefix}_composition_shape"] is not None
    )

    return {
        "n": n,
        "whole_exact": exact,
        "whole_exact_rate": pct(exact, n),
        "composable": comp,
        "composable_rate": pct(comp, n),
        "composition_adds_over_exact": adds,
        "composition_adds_over_exact_rate": pct(adds, n),
        "either": either,
        "either_rate": pct(either, n),
        "min_composition_pieces_distribution": dict(sorted(pieces.items())),
        "composition_shape_distribution": dict(sorted(shapes.items())),
        "generic_pruned_n": p_n,
        "generic_pruned_whole_exact": p_exact,
        "generic_pruned_whole_exact_rate": pct(p_exact, p_n),
        "generic_pruned_composable": p_comp,
        "generic_pruned_composable_rate": pct(p_comp, p_n),
        "generic_pruned_composition_adds_over_exact": p_adds,
        "generic_pruned_composition_adds_over_exact_rate": pct(
            p_adds,
            p_n,
        ),
        "generic_pruned_either": p_either,
        "generic_pruned_either_rate": pct(p_either, p_n),
    }


def latency_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}

    ordered = sorted(float(x) for x in values)

    def quantile(q: float) -> float:
        idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
        return ordered[idx]

    return {
        "mean_ms": 1000.0 * statistics.fmean(ordered),
        "median_ms": 1000.0 * statistics.median(ordered),
        "p95_ms": 1000.0 * quantile(0.95),
        "max_ms": 1000.0 * max(ordered),
    }


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--fit", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--fit-multi", type=Path, required=True)
    ap.add_argument("--val-multi", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--m0-rows", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--max-rows", type=int)
    ap.add_argument("--progress-every", type=int, default=100)

    args = ap.parse_args()

    fit_rows = m2rt.read_jsonl(args.fit)
    val_rows = m2rt.read_jsonl(args.val)
    fit_multi_rows = m2rt.read_jsonl(args.fit_multi)
    val_multi_rows = m2rt.read_jsonl(args.val_multi)
    manifest_rows = m2rt.read_jsonl(args.manifest)
    m0_rows = m2rt.read_jsonl(args.m0_rows)

    val_by_id = {
        str(row["row_id"]): row
        for row in val_rows
    }

    m0_by_id = {
        str(row["row_id"]): row
        for row in m0_rows
    }

    print("Building frozen M1-C ExpandedCausalHistoryIndex ...", flush=True)

    history = m1c.ExpandedCausalHistoryIndex(
        [*fit_rows, *val_rows],
        [*fit_multi_rows, *val_multi_rows],
    )

    lookup = m2rt.RuntimeExpandedLookup(history)

    selected = manifest_rows

    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise ValueError("--max-rows must be positive")
        selected = selected[: args.max_rows]

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite non-empty output: {args.output_root}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)

    outputs: list[dict[str, Any]] = []

    wall_start = time.perf_counter()

    for number, multi in enumerate(selected, start=1):
        row_id = str(multi["row_id"])
        source_id = str(multi["source_standard_row_id"])

        anchor = val_by_id.get(source_id)
        if anchor is None:
            raise RuntimeError(f"Missing Train-Val raw anchor: {source_id}")

        m0 = m0_by_id.get(row_id)
        if m0 is None:
            raise RuntimeError(f"Missing M0 row: {row_id}")

        qrow = m1c.make_query_row(multi, anchor)

        author = str(qrow["author"])
        position = int(qrow["chronological_position"])

        pinyin = tuple(
            str(x)
            for x in (
                multi.get("pinyin_segments")
                or str(multi["pinyin_input"]).split()
            )
        )

        gold = str(multi["gold"])

        if len(gold) != len(pinyin):
            raise RuntimeError(
                f"Gold/Pinyin alignment mismatch: {row_id}"
            )

        start_raw, stop_raw = lookup.raw_window(
            author=author,
            position=position,
        )

        raw_count = stop_raw - start_raw

        if raw_count != history.raw_visible_count(
            author=author,
            position=position,
        ):
            raise RuntimeError(f"H5000 raw count mismatch: {row_id}")

        t0 = time.perf_counter()

        span_stats = build_span_stats(
            lookup=lookup,
            author=author,
            position=position,
            pinyin=pinyin,
            gold=gold,
        )

        lookup_seconds = time.perf_counter() - t0

        out: dict[str, Any] = {
            "schema_version": 1,
            "experiment": "m4_r0_compositional_retrieval_v1",
            "row_id": row_id,
            "family_id": str(multi["family_id"]),
            "author": author,
            "multi_token_length": int(multi["multi_token_length"]),
            "pinyin_syllable_length": len(pinyin),
            "raw_h5000_count": raw_count,
            "generic_gold_survived_final_beam": m2rt.m0_gold_survived(m0),
            "generic_beam_pruned": not m2rt.m0_gold_survived(m0),
            "history_supported_spans": len(span_stats),
            "history_supported_shorter_spans": sum(
                not (
                    int(edge["start"]) == 0
                    and int(edge["end"]) == len(pinyin)
                )
                for edge in span_stats
            ),
            "lookup_seconds": lookup_seconds,
        }

        settings: list[tuple[str, int | None]] = [
            ("all", None),
            *[(f"k{k}", k) for k in KS],
        ]

        for prefix, k in settings:
            whole = whole_gold_retained(
                n=len(pinyin),
                edges=span_stats,
                k=k,
            )

            path = best_gold_composition_path(
                n=len(pinyin),
                edges=span_stats,
                k=k,
            )

            composable = path is not None

            out[f"{prefix}_whole_exact"] = whole
            out[f"{prefix}_composable"] = composable
            out[f"{prefix}_composition_adds_over_exact"] = (
                composable and not whole
            )
            out[f"{prefix}_either"] = whole or composable

            out[f"{prefix}_min_composition_pieces"] = (
                len(path) if path is not None else None
            )
            out[f"{prefix}_composition_shape"] = shape(path)

            out[f"{prefix}_candidate_derivations_composition_only"] = (
                candidate_derivation_count(
                    n=len(pinyin),
                    edges=span_stats,
                    k=k,
                    exclude_whole=True,
                )
            )

        out.update(
            {
                "gold_used_for_history_construction": False,
                "gold_used_for_candidate_generation": False,
                "gold_used_for_candidate_scoring": False,
                "gold_used_for_recoverability_audit_only": True,
                "multi_training": False,
                "multi_tuning": False,
                "used_dev3000": False,
                "used_test": False,
            }
        )

        outputs.append(out)

        if args.progress_every and number % args.progress_every == 0:
            print(
                f"processed={number}/{len(selected)}",
                flush=True,
            )

    wall_seconds = time.perf_counter() - wall_start

    by_length: dict[str, Any] = {}

    for length in sorted(
        {int(row["multi_token_length"]) for row in outputs}
    ):
        subset = [
            row
            for row in outputs
            if int(row["multi_token_length"]) == length
        ]

        by_length[str(length)] = {
            prefix: summarize_subset(subset, prefix)
            for prefix in ("all", "k1", "k3", "k5")
        }

    summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "m4_r0_compositional_retrieval_v1",
        "rows": len(outputs),
        "history_budget": 5000,
        "history_budget_unit": "raw_interaction",
        "history_semantics": (
            "strictly-prior same-author frozen M1-C expanded PV; "
            "derived entries remain attached to source raw event"
        ),
        "composition_definition": (
            "contiguous shorter-Pinyin history edges only; "
            "whole-query exact edge excluded; minimum-piece gold "
            "composition evaluated after gold-blind span retrieval"
        ),
        "diagnostic_per_span_frequency_caps": ["all", 1, 3, 5],
        "overall": {
            prefix: summarize_subset(outputs, prefix)
            for prefix in ("all", "k1", "k3", "k5")
        },
        "by_multi_token_length": by_length,
        "lookup_latency": latency_summary(
            [float(row["lookup_seconds"]) for row in outputs]
        ),
        "wall_seconds": wall_seconds,
        "inputs": {
            "fit_sha256": m2rt.sha256_file(args.fit),
            "val_sha256": m2rt.sha256_file(args.val),
            "fit_multi_sha256": m2rt.sha256_file(args.fit_multi),
            "val_multi_sha256": m2rt.sha256_file(args.val_multi),
            "manifest_sha256": m2rt.sha256_file(args.manifest),
            "m0_rows_sha256": m2rt.sha256_file(args.m0_rows),
        },
        "gold_used_for_history_construction": False,
        "gold_used_for_candidate_generation": False,
        "gold_used_for_candidate_scoring": False,
        "gold_used_for_recoverability_audit_only": True,
        "used_dev3000": False,
        "used_test": False,
    }

    write_jsonl(args.output_root / "rows.jsonl", outputs)
    write_json(args.output_root / "summary.json", summary)

    print()
    print("===== M4-R0 SUMMARY =====")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print()
    print("lookup_latency =", json.dumps(
        summary["lookup_latency"],
        ensure_ascii=False,
        sort_keys=True,
    ))
    print("wall_seconds =", wall_seconds)
    print("M4_R0_COMPOSITIONAL_RETRIEVAL=PASS")


if __name__ == "__main__":
    main()
