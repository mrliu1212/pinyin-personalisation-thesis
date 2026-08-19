from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.reference_backend_pinyingpt.backend import PinyinGPTConcatBackend


AUTHORS = (
    "Etinjat",
    "Re_spectators",
    "breaddddd",
)

K_VALUES = (1, 3, 5)
HISTORY_BUDGET = 5000


def load_generic(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            row = json.loads(line)

            if row.get("pilot_partition") != "tune":
                continue

            author = str(row.get("author", ""))
            if author not in AUTHORS:
                continue

            row_id = str(row["row_id"])

            if row_id in rows:
                raise RuntimeError(
                    f"Duplicate Generic row_id at line "
                    f"{line_number}: {row_id}"
                )

            rows[row_id] = row

    if not rows:
        raise RuntimeError(
            "No matching three-author Dev tune Generic rows found"
        )

    return rows


def load_states(path: Path) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            state = json.loads(line)

            author = str(state.get("author", ""))
            if author not in AUTHORS:
                continue

            row_id = str(state["row_id"])

            if row_id in states:
                raise RuntimeError(
                    f"Duplicate PV state row_id at line "
                    f"{line_number}: {row_id}"
                )

            states[row_id] = state

    if not states:
        raise RuntimeError(
            "No matching three-author PV Dev states found"
        )

    return states


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def is_backend_compatible(
    backend: PinyinGPTConcatBackend,
    target: str,
    pinyin: tuple[str, ...],
) -> bool:
    characters = list(target)

    if len(characters) != len(pinyin):
        return False

    token_ids = backend.tokenizer.convert_tokens_to_ids(
        characters
    )

    for token_id, segment in zip(
        token_ids,
        pinyin,
    ):
        if token_id not in backend.allowed_token_ids.get(
            segment,
            (),
        ):
            return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "EM-1 H5000 Dev recovery coverage audit with "
            "Frozen PinyinGPT backend compatibility filtering."
        )
    )

    parser.add_argument(
        "--generic-cache",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dev-states",
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
        default="cpu",
    )

    args = parser.parse_args()

    generic = load_generic(args.generic_cache)
    states = load_states(args.dev_states)

    if set(generic) != set(states):
        raise RuntimeError(
            "Three-author Generic/PV row IDs differ: "
            f"generic_only={len(set(generic) - set(states))} "
            f"state_only={len(set(states) - set(generic))}"
        )

    print(
        "Loading Frozen PinyinGPT only to apply its "
        "tokenizer/Pinyin compatibility rules..."
    )

    backend = PinyinGPTConcatBackend(
        args.checkpoint,
        device=args.device,
    )

    rows_output: list[dict[str, Any]] = []

    raw_sizes: list[int] = []
    compatible_sizes: list[int] = []

    rows_with_raw_personal = 0
    rows_with_compatible_personal = 0
    rows_with_incompatible = 0

    incompatible_candidates_total = 0

    generic_missing = 0

    raw_recoverable_any = 0
    compatible_recoverable_any = 0

    raw_recoverable_at = Counter()
    compatible_recoverable_at = Counter()

    per_author = defaultdict(
        lambda: {
            "rows": 0,
            "generic_missing": 0,
            "raw_recoverable_any": 0,
            "compatible_recoverable_any": 0,
            "incompatible_candidates": 0,
            **{
                f"raw_recoverable_at_{k}": 0
                for k in K_VALUES
            },
            **{
                f"compatible_recoverable_at_{k}": 0
                for k in K_VALUES
            },
        }
    )

    for number, row_id in enumerate(
        sorted(generic),
        start=1,
    ):
        row = generic[row_id]
        state = states[row_id]

        author = str(row["author"])
        gold = str(row["target"])

        pinyin = tuple(
            str(value)
            for value in row["pinyin_segments"]
        )

        generic_candidates = {
            str(candidate["text"])
            for candidate in row["top10_candidates"]
        }

        raw_personal = tuple(
            str(value)
            for value in state.get(
                "personal_only_targets",
                []
            )
        )

        overlap = generic_candidates.intersection(
            raw_personal
        )

        if overlap:
            raise RuntimeError(
                f"{row_id}: personal-only overlaps Generic: "
                f"{sorted(overlap)}"
            )

        compatible_personal: list[str] = []
        incompatible_personal: list[str] = []

        for target in raw_personal:
            if is_backend_compatible(
                backend,
                target,
                pinyin,
            ):
                compatible_personal.append(target)
            else:
                incompatible_personal.append(target)

        compatible_personal_tuple = tuple(
            compatible_personal
        )

        raw_sizes.append(len(raw_personal))
        compatible_sizes.append(
            len(compatible_personal_tuple)
        )

        if raw_personal:
            rows_with_raw_personal += 1

        if compatible_personal_tuple:
            rows_with_compatible_personal += 1

        if incompatible_personal:
            rows_with_incompatible += 1
            incompatible_candidates_total += len(
                incompatible_personal
            )

        is_generic_missing = (
            gold not in generic_candidates
        )

        if is_generic_missing:
            generic_missing += 1

        raw_any = (
            is_generic_missing
            and gold in raw_personal
        )

        compatible_any = (
            is_generic_missing
            and gold in compatible_personal_tuple
        )

        if raw_any:
            raw_recoverable_any += 1

        if compatible_any:
            compatible_recoverable_any += 1

        stats = per_author[author]

        stats["rows"] += 1
        stats["incompatible_candidates"] += len(
            incompatible_personal
        )

        if is_generic_missing:
            stats["generic_missing"] += 1

        if raw_any:
            stats["raw_recoverable_any"] += 1

        if compatible_any:
            stats["compatible_recoverable_any"] += 1

        raw_at: dict[int, bool] = {}
        compatible_at: dict[int, bool] = {}

        for k in K_VALUES:
            raw_value = (
                is_generic_missing
                and gold in raw_personal[:k]
            )

            compatible_value = (
                is_generic_missing
                and gold in compatible_personal_tuple[:k]
            )

            raw_at[k] = raw_value
            compatible_at[k] = compatible_value

            if raw_value:
                raw_recoverable_at[k] += 1
                stats[
                    f"raw_recoverable_at_{k}"
                ] += 1

            if compatible_value:
                compatible_recoverable_at[k] += 1
                stats[
                    f"compatible_recoverable_at_{k}"
                ] += 1

        rows_output.append(
            {
                "row_id": row_id,
                "author": author,
                "gold": gold,
                "pinyin_segments": list(pinyin),
                "generic_missing": is_generic_missing,
                "raw_personal_only_count": len(
                    raw_personal
                ),
                "compatible_personal_only_count": len(
                    compatible_personal_tuple
                ),
                "incompatible_personal_only_count": len(
                    incompatible_personal
                ),
                "raw_personal_only_targets": list(
                    raw_personal
                ),
                "compatible_personal_only_targets": list(
                    compatible_personal_tuple
                ),
                "incompatible_personal_only_targets": list(
                    incompatible_personal
                ),
                "raw_recoverable_any": raw_any,
                "compatible_recoverable_any": compatible_any,
                **{
                    f"raw_recoverable_at_{k}": raw_at[k]
                    for k in K_VALUES
                },
                **{
                    f"compatible_recoverable_at_{k}": (
                        compatible_at[k]
                    )
                    for k in K_VALUES
                },
            }
        )

        if number % 1000 == 0 or number == len(generic):
            print(
                f"Compatibility checked: "
                f"{number}/{len(generic)}",
                flush=True,
            )

    total_rows = len(rows_output)

    summary = {
        "schema_version": 2,
        "experiment": (
            "em1_recovery_coverage_h5000_dev"
        ),
        "status": "audit_complete",
        "condition": "Full+Short",
        "partition": "dev_tune",
        "authors": list(AUTHORS),
        "history_budget": HISTORY_BUDGET,
        "k_values": list(K_VALUES),
        "rows": total_rows,

        "raw_personal_only": {
            "rows_with_candidates": (
                rows_with_raw_personal
            ),
            "mean_per_row": statistics.fmean(
                raw_sizes
            ),
            "median_per_row": statistics.median(
                raw_sizes
            ),
            "max_per_row": max(raw_sizes),
        },

        "backend_compatible_personal_only": {
            "rows_with_candidates": (
                rows_with_compatible_personal
            ),
            "mean_per_row": statistics.fmean(
                compatible_sizes
            ),
            "median_per_row": statistics.median(
                compatible_sizes
            ),
            "max_per_row": max(
                compatible_sizes
            ),
        },

        "backend_incompatibility": {
            "rows_with_incompatible_candidates": (
                rows_with_incompatible
            ),
            "incompatible_candidates_total": (
                incompatible_candidates_total
            ),
        },

        "generic_missing": generic_missing,

        "raw_recoverable_any": (
            raw_recoverable_any
        ),

        "compatible_recoverable_any": (
            compatible_recoverable_any
        ),

        "compatible_recoverable_any_rate_of_generic_missing": (
            safe_rate(
                compatible_recoverable_any,
                generic_missing,
            )
        ),

        "raw_recoverable_at_k": {
            str(k): raw_recoverable_at[k]
            for k in K_VALUES
        },

        "compatible_recoverable_at_k": {
            str(k): {
                "count": (
                    compatible_recoverable_at[k]
                ),
                "rate_of_generic_missing": safe_rate(
                    compatible_recoverable_at[k],
                    generic_missing,
                ),
                "rate_of_compatible_recoverable_any": safe_rate(
                    compatible_recoverable_at[k],
                    compatible_recoverable_any,
                ),
            }
            for k in K_VALUES
        },

        "per_author": {
            author: dict(per_author[author])
            for author in AUTHORS
        },

        "test_rows_used": 0,
        "model_forward_scoring_calls": 0,
        "gold_used_for_candidate_construction": False,
        "gold_used_only_for_posthoc_coverage_measurement": True,
    }

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        args.output_root / "rows.jsonl"
    ).open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as destination:
        for row in rows_output:
            destination.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    with (
        args.output_root / "summary.json"
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
        "=== EM-1 Backend-Compatible Recovery Coverage ==="
    )
    print(f"Rows: {total_rows}")

    print(
        "Raw rows with personal-only: "
        f"{rows_with_raw_personal}"
    )

    print(
        "Compatible rows with personal-only: "
        f"{rows_with_compatible_personal}"
    )

    print(
        "Rows containing incompatible candidates: "
        f"{rows_with_incompatible}"
    )

    print(
        "Incompatible candidates total: "
        f"{incompatible_candidates_total}"
    )

    print()
    print(
        f"Generic Missing: {generic_missing}"
    )

    print(
        "Raw Recoverable-any: "
        f"{raw_recoverable_any} "
        f"({100 * safe_rate(raw_recoverable_any, generic_missing):.2f}% "
        "of Generic Missing)"
    )

    print(
        "Compatible Recoverable-any: "
        f"{compatible_recoverable_any} "
        f"({100 * safe_rate(compatible_recoverable_any, generic_missing):.2f}% "
        "of Generic Missing)"
    )

    for k in K_VALUES:
        count = compatible_recoverable_at[k]

        print(
            f"Compatible Recoverable@{k}: "
            f"{count} "
            f"({100 * safe_rate(count, generic_missing):.2f}% "
            f"of Generic Missing; "
            f"{100 * safe_rate(count, compatible_recoverable_any):.2f}% "
            f"of Compatible Recoverable-any)"
        )

    print()
    print("Per author:")

    for author in AUTHORS:
        values = per_author[author]

        print(
            f"  {author}: "
            f"rows={values['rows']} "
            f"missing={values['generic_missing']} "
            f"raw_any={values['raw_recoverable_any']} "
            f"compatible_any={values['compatible_recoverable_any']} "
            + " ".join(
                f"R@{k}="
                f"{values[f'compatible_recoverable_at_{k}']}"
                for k in K_VALUES
            )
            + " "
            f"incompatible="
            f"{values['incompatible_candidates']}"
        )

    print()
    print(
        "No K selected. Coverage audit only."
    )


if __name__ == "__main__":
    main()
