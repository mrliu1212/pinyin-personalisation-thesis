from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.reference_backend_pinyingpt.backend import PinyinGPTConcatBackend


AUTHORS = (
    "Etinjat",
    "Re_spectators",
    "breaddddd",
)


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
                    f"Duplicate row_id at line {line_number}: {row_id}"
                )

            rows[row_id] = row

    if not rows:
        raise RuntimeError("No matching Dev tune rows found")

    return rows


def gold_compatibility(
    backend: PinyinGPTConcatBackend,
    gold: str,
    pinyin: tuple[str, ...],
) -> tuple[bool, str]:
    characters = list(gold)

    if len(characters) != len(pinyin):
        return (
            False,
            "character_count_mismatch",
        )

    token_ids = backend.tokenizer.convert_tokens_to_ids(
        characters
    )

    for index, (
        character,
        token_id,
        segment,
    ) in enumerate(
        zip(
            characters,
            token_ids,
            pinyin,
        )
    ):
        if token_id == backend.tokenizer.unk_token_id:
            return (
                False,
                f"tokenizer_unknown_at_{index}",
            )

        allowed = backend.allowed_token_ids.get(
            segment,
            (),
        )

        if token_id not in allowed:
            return (
                False,
                f"pinyin_incompatible_at_{index}",
            )

    return True, "compatible"


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether Dev Gold targets are reachable "
            "under the Frozen PinyinGPT constrained candidate space."
        )
    )

    parser.add_argument(
        "--generic-cache",
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

    rows = load_generic(args.generic_cache)

    print(
        "Loading Frozen PinyinGPT for Gold "
        "candidate-space compatibility only..."
    )

    backend = PinyinGPTConcatBackend(
        args.checkpoint,
        device=args.device,
    )

    total = len(rows)

    compatible = 0
    incompatible = 0

    generic_missing = 0
    generic_missing_compatible = 0
    generic_missing_incompatible = 0

    reason_counts = Counter()

    per_author = defaultdict(
        lambda: {
            "rows": 0,
            "gold_compatible": 0,
            "gold_incompatible": 0,
            "generic_missing": 0,
            "generic_missing_gold_compatible": 0,
            "generic_missing_gold_incompatible": 0,
        }
    )

    output_rows: list[dict[str, Any]] = []

    for number, row_id in enumerate(
        sorted(rows),
        start=1,
    ):
        row = rows[row_id]

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

        is_generic_missing = (
            gold not in generic_candidates
        )

        is_compatible, reason = gold_compatibility(
            backend,
            gold,
            pinyin,
        )

        stats = per_author[author]
        stats["rows"] += 1

        if is_compatible:
            compatible += 1
            stats["gold_compatible"] += 1
        else:
            incompatible += 1
            stats["gold_incompatible"] += 1
            reason_counts[reason] += 1

        if is_generic_missing:
            generic_missing += 1
            stats["generic_missing"] += 1

            if is_compatible:
                generic_missing_compatible += 1
                stats[
                    "generic_missing_gold_compatible"
                ] += 1
            else:
                generic_missing_incompatible += 1
                stats[
                    "generic_missing_gold_incompatible"
                ] += 1

        output_rows.append(
            {
                "row_id": row_id,
                "author": author,
                "gold": gold,
                "pinyin_segments": list(pinyin),
                "gold_backend_compatible": (
                    is_compatible
                ),
                "compatibility_reason": reason,
                "generic_missing": (
                    is_generic_missing
                ),
            }
        )

        if number % 1000 == 0 or number == total:
            print(
                f"Checked: {number}/{total}",
                flush=True,
            )

    summary = {
        "schema_version": 1,
        "experiment": (
            "em1_gold_backend_reachability_dev"
        ),
        "condition": "Full+Short",
        "partition": "dev_tune",
        "authors": list(AUTHORS),
        "rows": total,

        "gold_compatible": compatible,
        "gold_compatible_rate": safe_rate(
            compatible,
            total,
        ),

        "gold_incompatible": incompatible,
        "gold_incompatible_rate": safe_rate(
            incompatible,
            total,
        ),

        "incompatibility_reasons": dict(
            reason_counts
        ),

        "generic_missing": generic_missing,

        "generic_missing_gold_compatible": (
            generic_missing_compatible
        ),

        "generic_missing_gold_incompatible": (
            generic_missing_incompatible
        ),

        "generic_missing_gold_incompatible_rate": (
            safe_rate(
                generic_missing_incompatible,
                generic_missing,
            )
        ),

        "per_author": {
            author: dict(per_author[author])
            for author in AUTHORS
        },

        "model_forward_calls": 0,
        "test_rows_used": 0,
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
        for row in output_rows:
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
        "=== Gold Backend Reachability — "
        "Full+Short / Dev ==="
    )

    print(f"Rows: {total}")

    print(
        f"Gold compatible: {compatible} "
        f"({100 * safe_rate(compatible, total):.2f}%)"
    )

    print(
        f"Gold incompatible: {incompatible} "
        f"({100 * safe_rate(incompatible, total):.2f}%)"
    )

    print()
    print(
        f"Generic Missing: {generic_missing}"
    )

    print(
        "Generic Missing + Gold compatible: "
        f"{generic_missing_compatible}"
    )

    print(
        "Generic Missing + Gold incompatible: "
        f"{generic_missing_incompatible} "
        f"({100 * safe_rate(generic_missing_incompatible, generic_missing):.2f}% "
        "of Generic Missing)"
    )

    print()
    print("Incompatibility reasons:")

    for reason, count in sorted(
        reason_counts.items()
    ):
        print(
            f"  {reason}: {count}"
        )

    print()
    print("Per author:")

    for author in AUTHORS:
        stats = per_author[author]

        print(
            f"  {author}: "
            f"rows={stats['rows']} "
            f"compatible={stats['gold_compatible']} "
            f"incompatible={stats['gold_incompatible']} "
            f"missing={stats['generic_missing']} "
            f"missing_compatible="
            f"{stats['generic_missing_gold_compatible']} "
            f"missing_incompatible="
            f"{stats['generic_missing_gold_incompatible']}"
        )


if __name__ == "__main__":
    main()
