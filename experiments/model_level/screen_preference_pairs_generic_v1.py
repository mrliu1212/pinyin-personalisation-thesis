from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.reference_backend_pinyingpt.backend import PinyinGPTConcatBackend


AUTHOR = "Agent Phage"

DEFAULT_FIT = Path(
    "/home/3160454/work/model-level-assets-20260822/"
    "clean3_train_fit_v1.jsonl"
)

DEFAULT_VAL = Path(
    "/home/3160454/work/model-level-assets-20260822/"
    "clean3_train_val_v1.jsonl"
)

DEFAULT_CHECKPOINT = Path(
    "/home/3160454/work/model-level-assets-20260822/"
    "pinyingpt2-concat"
)

DEFAULT_STRUCTURE_CSV = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "pair_screening_structure_v1/"
    "preference_pairs_structure_all_v1.csv"
)

DEFAULT_OUTPUT = Path(
    "results/model_level/"
    "controlled_preference_dynamics_v1/"
    "pair_screening_generic_v1"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--fit", type=Path, default=DEFAULT_FIT)
    p.add_argument("--val", type=Path, default=DEFAULT_VAL)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    p.add_argument(
        "--structure-csv",
        type=Path,
        default=DEFAULT_STRUCTURE_CSV,
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument(
        "--contexts-per-side",
        type=int,
        default=8,
    )
    p.add_argument("--validate-only", action="store_true")

    return p.parse_args()


def get_pinyin(row: dict[str, Any]) -> str:
    value = (
        row.get("segmented_pinyin")
        or row.get("pinyin_input")
        or row.get("pinyin")
        or row.get("typed_pinyin")
    )

    if isinstance(value, list):
        return " ".join(
            str(x).strip().lower()
            for x in value
            if str(x).strip()
        )

    if isinstance(value, str):
        return " ".join(value.strip().lower().split())

    raise RuntimeError(
        f"Cannot identify pinyin field: {row.get('row_id')}"
    )


def get_target(row: dict[str, Any]) -> str:
    for key in ("target", "gold", "text"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value

    raise RuntimeError(
        f"Cannot identify target field: {row.get('row_id')}"
    )


def get_context(row: dict[str, Any]) -> str:
    value = row.get("context", "")
    if value is None:
        return ""
    return str(value)


def load_agent_rows(
    path: Path,
    partition: str,
) -> list[dict[str, Any]]:
    rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if str(
                row.get("source_split", "")
            ).lower() == "test":
                raise RuntimeError(
                    "STOP: Test row detected"
                )

            if row.get("author") != AUTHOR:
                continue

            if row.get("target_type") not in (
                None,
                "short",
            ):
                continue

            if row.get("condition") not in (
                None,
                "full_short",
            ):
                continue

            item = dict(row)
            item["_partition"] = partition
            item["_pinyin"] = get_pinyin(row)
            item["_target"] = get_target(row)
            item["_context"] = get_context(row)

            rows.append(item)

    return rows


def chronological_key(
    row: dict[str, Any],
) -> tuple[str, int, str]:
    date = str(row.get("date") or "")

    work = (
        row.get("work_id")
        if row.get("work_id") is not None
        else row.get("work")
    )

    try:
        work_int = int(work)
    except Exception:
        work_int = -1

    row_id = str(row.get("row_id") or "")

    return date, work_int, row_id


def deterministic_spread(
    rows: list[dict[str, Any]],
    n: int,
) -> list[dict[str, Any]]:
    rows = sorted(rows, key=chronological_key)

    if len(rows) <= n:
        return rows

    if n == 1:
        return [rows[len(rows) // 2]]

    indices = []

    for i in range(n):
        x = i * (len(rows) - 1) / (n - 1)
        indices.append(round(x))

    # Defensive unique-preserving pass.
    seen = set()
    selected = []

    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        selected.append(rows[idx])

    return selected


def load_structure_pairs(
    path: Path,
    min_count: int,
) -> list[dict[str, Any]]:
    pairs = []

    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            item: dict[str, Any] = dict(row)

            item["count_a"] = int(item["count_a"])
            item["count_b"] = int(item["count_b"])
            item["min_count"] = int(item["min_count"])

            if item["min_count"] < min_count:
                continue

            pairs.append(item)

    return pairs


def score_pair_context(
    backend: PinyinGPTConcatBackend,
    context: str,
    pinyin: str,
    a: str,
    b: str,
) -> tuple[float, float]:
    scores = backend.score_candidates(
        context,
        pinyin,
        [a, b],
    )

    if len(scores) != 2:
        raise RuntimeError(
            f"Expected 2 scores, got {len(scores)}"
        )

    return float(scores[0].log_probability), float(scores[1].log_probability)


def main() -> None:
    args = parse_args()

    rows = (
        load_agent_rows(args.fit, "train_fit")
        + load_agent_rows(args.val, "train_val")
    )

    pair_defs = load_structure_pairs(
        args.structure_csv,
        args.min_count,
    )

    by_key: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        by_key[
            (
                row["_pinyin"],
                row["_target"],
            )
        ].append(row)

    prepared = []

    for idx, pair in enumerate(pair_defs, 1):
        pinyin = str(pair["pinyin"])
        a = str(pair["candidate_a"])
        b = str(pair["candidate_b"])

        rows_a = by_key.get(
            (pinyin, a),
            [],
        )
        rows_b = by_key.get(
            (pinyin, b),
            [],
        )

        selected_a = deterministic_spread(
            rows_a,
            args.contexts_per_side,
        )
        selected_b = deterministic_spread(
            rows_b,
            args.contexts_per_side,
        )

        if not selected_a or not selected_b:
            continue

        prepared.append(
            {
                "pair_id": f"pair_{idx:04d}",
                "pinyin": pinyin,
                "candidate_a": a,
                "candidate_b": b,
                "count_a": pair["count_a"],
                "count_b": pair["count_b"],
                "contexts_a": selected_a,
                "contexts_b": selected_b,
            }
        )

    print(
        "===== CONTROLLED PREFERENCE GENERIC SCREEN GATE ====="
    )
    print("author =", AUTHOR)
    print("test_used = False")
    print("source_rows =", len(rows))
    print("structural_pairs =", len(pair_defs))
    print("prepared_pairs =", len(prepared))
    print("min_count =", args.min_count)
    print(
        "contexts_per_side =",
        args.contexts_per_side,
    )
    print(
        "max_scores =",
        len(prepared)
        * args.contexts_per_side
        * 2,
    )

    if args.validate_only:
        print("backend_loaded = False")
        print("STATUS = VALIDATED_ONLY")
        return

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Loading frozen Generic backend...")

    backend = PinyinGPTConcatBackend(
        args.checkpoint,
        device=args.device,
    )

    result_rows = []
    context_rows = []

    for pair_index, pair in enumerate(
        prepared,
        1,
    ):
        pinyin = pair["pinyin"]
        a = pair["candidate_a"]
        b = pair["candidate_b"]

        margins = []
        valid_a_contexts = 0
        valid_b_contexts = 0
        invalid = 0

        for source_side, selected in (
            ("A", pair["contexts_a"]),
            ("B", pair["contexts_b"]),
        ):
            for row in selected:
                context = row["_context"]

                try:
                    score_a, score_b = (
                        score_pair_context(
                            backend,
                            context,
                            pinyin,
                            a,
                            b,
                        )
                    )
                except Exception as exc:
                    invalid += 1

                    context_rows.append(
                        {
                            "pair_id": pair["pair_id"],
                            "pinyin": pinyin,
                            "candidate_a": a,
                            "candidate_b": b,
                            "source_side": source_side,
                            "row_id": row.get("row_id"),
                            "context": context,
                            "valid": False,
                            "error": repr(exc),
                        }
                    )
                    continue

                margin = score_a - score_b
                margins.append(margin)

                if source_side == "A":
                    valid_a_contexts += 1
                else:
                    valid_b_contexts += 1

                context_rows.append(
                    {
                        "pair_id": pair["pair_id"],
                        "pinyin": pinyin,
                        "candidate_a": a,
                        "candidate_b": b,
                        "source_side": source_side,
                        "row_id": row.get("row_id"),
                        "context": context,
                        "valid": True,
                        "score_a": score_a,
                        "score_b": score_b,
                        "margin_a_minus_b": margin,
                    }
                )

        if not margins:
            continue

        mean_margin = statistics.mean(margins)
        median_margin = statistics.median(margins)

        if len(margins) > 1:
            std_margin = statistics.stdev(margins)
        else:
            std_margin = 0.0

        a_wins = sum(x > 0 for x in margins)
        b_wins = sum(x < 0 for x in margins)
        ties = sum(x == 0 for x in margins)

        n = len(margins)

        if abs(mean_margin) <= 0.5:
            generic_band = "balanced"
        elif mean_margin > 0:
            generic_band = "A_favoured"
        else:
            generic_band = "B_favoured"

        result_rows.append(
            {
                "pair_id": pair["pair_id"],
                "pinyin": pinyin,
                "candidate_a": a,
                "candidate_b": b,
                "count_a": pair["count_a"],
                "count_b": pair["count_b"],
                "valid_contexts": n,
                "valid_a_contexts": valid_a_contexts,
                "valid_b_contexts": valid_b_contexts,
                "invalid_contexts": invalid,
                "mean_margin_a_minus_b": mean_margin,
                "median_margin_a_minus_b": median_margin,
                "std_margin": std_margin,
                "abs_mean_margin": abs(mean_margin),
                "a_win_rate": a_wins / n,
                "b_win_rate": b_wins / n,
                "tie_rate": ties / n,
                "generic_band": generic_band,
            }
        )

        print(
            f"[{pair_index:03d}/{len(prepared):03d}] "
            f"{pinyin} | {a} vs {b} | "
            f"margin={mean_margin:+.4f} | "
            f"Awin={a_wins/n:.3f} | "
            f"N={n} invalid={invalid}",
            flush=True,
        )

    result_rows.sort(
        key=lambda x: (
            x["abs_mean_margin"],
            x["std_margin"],
            -min(
                x["count_a"],
                x["count_b"],
            ),
            x["pinyin"],
        )
    )

    pair_csv = (
        args.output_root
        / "preference_pairs_generic_all_v1.csv"
    )

    if result_rows:
        with pair_csv.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    result_rows[0].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(result_rows)

    contexts_jsonl = (
        args.output_root
        / "generic_context_scores_v1.jsonl"
    )

    with contexts_jsonl.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in context_rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    shortlist = [
        row
        for row in result_rows
        if row["valid_a_contexts"] >= 4
        and row["valid_b_contexts"] >= 4
        and row["invalid_contexts"] == 0
        and row["abs_mean_margin"] <= 2.0
    ]

    shortlist_csv = (
        args.output_root
        / "preference_pairs_generic_shortlist_v1.csv"
    )

    if shortlist:
        with shortlist_csv.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    shortlist[0].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(shortlist)

    audit = {
        "schema_version": 1,
        "author": AUTHOR,
        "test_used": False,
        "adapter_loaded": False,
        "training": False,
        "generic_backend_only": True,
        "min_count": args.min_count,
        "contexts_per_side": (
            args.contexts_per_side
        ),
        "structural_pairs": len(pair_defs),
        "prepared_pairs": len(prepared),
        "scored_pairs": len(result_rows),
        "shortlist_pairs": len(shortlist),
        "balanced_pairs_abs_margin_le_0_5": sum(
            row["abs_mean_margin"] <= 0.5
            for row in result_rows
        ),
        "moderate_pairs_abs_margin_le_1": sum(
            row["abs_mean_margin"] <= 1.0
            for row in result_rows
        ),
        "moderate_pairs_abs_margin_le_2": sum(
            row["abs_mean_margin"] <= 2.0
            for row in result_rows
        ),
        "output_pairs_csv": str(pair_csv),
        "output_shortlist_csv": str(
            shortlist_csv
        ),
        "output_context_scores": str(
            contexts_jsonl
        ),
    }

    (
        args.output_root
        / "screening_audit_generic_v1.json"
    ).write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("===== GENERIC SCREEN SUMMARY =====")

    for key, value in audit.items():
        print(f"{key} = {value}")

    print()
    print("===== TOP 30 CLOSEST GENERIC PAIRS =====")

    for n, row in enumerate(
        result_rows[:30],
        1,
    ):
        print(
            f"{n:02d}. "
            f"{row['pinyin']} | "
            f"{row['candidate_a']} vs "
            f"{row['candidate_b']} | "
            f"mean={row['mean_margin_a_minus_b']:+.4f} "
            f"std={row['std_margin']:.4f} "
            f"Awin={row['a_win_rate']:.3f} "
            f"N={row['valid_contexts']}"
        )


if __name__ == "__main__":
    main()
