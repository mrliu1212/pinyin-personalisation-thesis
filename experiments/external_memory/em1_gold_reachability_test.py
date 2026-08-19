from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.reference_backend_pinyingpt.backend import PinyinGPTConcatBackend


FROZEN_T1_SHA256 = (
    "764db39887f3db04b913d1739d9dbd46295f0e46e5a2bffa649f1563b56ee4e2"
)

EXPECTED_FULL_SHORT_ROWS = 6000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_full_short_test(
    path: Path,
) -> list[dict[str, Any]]:
    actual_hash = sha256_file(path)

    if actual_hash != FROZEN_T1_SHA256:
        raise RuntimeError(
            "Frozen T1 Test prediction SHA mismatch:\n"
            f"expected={FROZEN_T1_SHA256}\n"
            f"actual={actual_hash}"
        )

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)

            if row.get("condition") != "full_short":
                continue

            rows.append(row)

    if len(rows) != EXPECTED_FULL_SHORT_ROWS:
        raise RuntimeError(
            "Unexpected Full+Short Test row count: "
            f"expected={EXPECTED_FULL_SHORT_ROWS} "
            f"actual={len(rows)}"
        )

    return rows


def recover_pinyin_segments(
    backend: PinyinGPTConcatBackend,
    typed_pinyin: str,
    expected_segments: int,
) -> tuple[str, ...]:
    normalized = backend._normalize_pinyin(
        typed_pinyin
    )

    if " " in normalized:
        segments = tuple(normalized.split())

        if len(segments) != expected_segments:
            raise RuntimeError(
                f"Stored spaced Pinyin has "
                f"{len(segments)} segments but Gold "
                f"requires {expected_segments}: "
                f"{typed_pinyin!r}"
            )

        return segments

    paths: dict[int, list[tuple[str, ...]]] = {
        0: [()]
    }

    for start in range(len(normalized)):
        for prefix in paths.get(start, []):
            if len(prefix) >= expected_segments:
                continue

            for end in range(
                start + 1,
                len(normalized) + 1,
            ):
                syllable = normalized[start:end]

                if syllable not in backend.allowed_token_ids:
                    continue

                candidate = prefix + (syllable,)

                if len(candidate) > expected_segments:
                    continue

                paths.setdefault(end, []).append(
                    candidate
                )

    alternatives = [
        segments
        for segments in paths.get(
            len(normalized),
            [],
        )
        if len(segments) == expected_segments
    ]

    alternatives = list(
        dict.fromkeys(alternatives)
    )

    if not alternatives:
        raise RuntimeError(
            "No backend-valid Pinyin segmentation "
            f"with Gold length={expected_segments}: "
            f"{typed_pinyin!r}"
        )

    if len(alternatives) != 1:
        rendered = [
            " ".join(value)
            for value in alternatives
        ]

        raise RuntimeError(
            "Gold length does not uniquely recover "
            "Pinyin segmentation: "
            f"{typed_pinyin!r} -> {rendered}"
        )

    return alternatives[0]

def compatibility(
    backend: PinyinGPTConcatBackend,
    gold: str,
    pinyin: tuple[str, ...],
) -> tuple[bool, str]:
    characters = list(gold)

    if len(characters) != len(pinyin):
        return False, "character_count_mismatch"

    token_ids = backend.tokenizer.convert_tokens_to_ids(
        characters
    )

    for index, (
        token_id,
        segment,
    ) in enumerate(
        zip(token_ids, pinyin)
    ):
        if token_id == backend.tokenizer.unk_token_id:
            return (
                False,
                f"tokenizer_unknown_at_{index}",
            )

        if token_id not in backend.allowed_token_ids.get(
            segment,
            (),
        ):
            return (
                False,
                f"pinyin_incompatible_at_{index}",
            )

    return True, "compatible"


def rate(a: int, b: int) -> float:
    return a / b if b else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
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

    rows = load_full_short_test(
        args.predictions
    )

    print(
        "Loading Frozen PinyinGPT for "
        "Test Gold reachability audit..."
    )

    backend = PinyinGPTConcatBackend(
        args.checkpoint,
        device=args.device,
    )

    gold_compatible = 0
    gold_incompatible = 0

    generic_missing = 0
    missing_compatible = 0
    missing_incompatible = 0

    reasons = Counter()

    per_author = defaultdict(
        lambda: {
            "rows": 0,
            "compatible": 0,
            "incompatible": 0,
            "generic_missing": 0,
            "missing_compatible": 0,
            "missing_incompatible": 0,
        }
    )

    for number, row in enumerate(
        rows,
        start=1,
    ):
        author = str(row["author"])

        gold = str(
            row.get(
                "target",
                row.get("gold"),
            )
        )

        pinyin = recover_pinyin_segments(
            backend,
            str(row["pinyin_input"]),
            len(gold),
        )

        candidates = {
            str(candidate["text"])
            for candidate in row["top10_candidates"]
        }

        is_missing = gold not in candidates

        is_compatible, reason = compatibility(
            backend,
            gold,
            pinyin,
        )

        stats = per_author[author]
        stats["rows"] += 1

        if is_compatible:
            gold_compatible += 1
            stats["compatible"] += 1
        else:
            gold_incompatible += 1
            stats["incompatible"] += 1
            reasons[reason] += 1

        if is_missing:
            generic_missing += 1
            stats["generic_missing"] += 1

            if is_compatible:
                missing_compatible += 1
                stats["missing_compatible"] += 1
            else:
                missing_incompatible += 1
                stats["missing_incompatible"] += 1

        if number % 1000 == 0:
            print(
                f"Checked: {number}/{len(rows)}",
                flush=True,
            )

    summary = {
        "schema_version": 1,
        "experiment": (
            "full_short_test_gold_backend_reachability"
        ),
        "purpose": (
            "Backend/data compatibility audit only; "
            "not used for EM-1 parameter selection."
        ),
        "rows": len(rows),
        "gold_compatible": gold_compatible,
        "gold_compatible_rate": rate(
            gold_compatible,
            len(rows),
        ),
        "gold_incompatible": gold_incompatible,
        "gold_incompatible_rate": rate(
            gold_incompatible,
            len(rows),
        ),
        "generic_missing": generic_missing,
        "generic_missing_gold_compatible": (
            missing_compatible
        ),
        "generic_missing_gold_incompatible": (
            missing_incompatible
        ),
        "generic_missing_gold_incompatible_rate": (
            rate(
                missing_incompatible,
                generic_missing,
            )
        ),
        "incompatibility_reasons": dict(
            reasons
        ),
        "per_author": {
            author: dict(values)
            for author, values in sorted(
                per_author.items()
            )
        },
        "frozen_t1_predictions_sha256": (
            FROZEN_T1_SHA256
        ),
        "used_for_parameter_tuning": False,
    }

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
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
        "Full+Short / Test ==="
    )

    print(f"Rows: {len(rows)}")

    print(
        f"Gold compatible: {gold_compatible} "
        f"({100 * rate(gold_compatible, len(rows)):.2f}%)"
    )

    print(
        f"Gold incompatible: {gold_incompatible} "
        f"({100 * rate(gold_incompatible, len(rows)):.2f}%)"
    )

    print()
    print(
        f"Generic Missing: {generic_missing}"
    )

    print(
        "Generic Missing + Gold compatible: "
        f"{missing_compatible}"
    )

    print(
        "Generic Missing + Gold incompatible: "
        f"{missing_incompatible} "
        f"({100 * rate(missing_incompatible, generic_missing):.2f}% "
        "of Generic Missing)"
    )

    print()
    print("Incompatibility reasons:")

    for reason, count in sorted(
        reasons.items()
    ):
        print(f"  {reason}: {count}")

    print()
    print("Per author:")

    for author, stats in sorted(
        per_author.items()
    ):
        print(
            f"  {author}: "
            f"rows={stats['rows']} "
            f"compatible={stats['compatible']} "
            f"incompatible={stats['incompatible']} "
            f"missing={stats['generic_missing']} "
            f"missing_compatible="
            f"{stats['missing_compatible']} "
            f"missing_incompatible="
            f"{stats['missing_incompatible']}"
        )


if __name__ == "__main__":
    main()


