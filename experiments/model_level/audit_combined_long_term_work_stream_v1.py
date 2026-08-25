from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


AUTHOR = "Agent Phage"

TRAIN_FIT = Path(
    "/home/3160454/work/model-level-assets-20260822/"
    "clean3_train_fit_v1.jsonl"
)

TRAIN_VAL = Path(
    "/home/3160454/work/model-level-assets-20260822/"
    "clean3_train_val_v1.jsonl"
)

CHECKPOINT = Path(
    "/home/3160454/work/model-level-assets-20260822/"
    "pinyingpt2-concat"
)

OUT = Path(
    "results/model_level/"
    "long_term_qualification_v1/"
    "combined_work_audit_v1"
)

BLOCK_SPANS = [1, 2, 3, 5]
MIN_COUNTS = [2, 3, 5, 10]
MIN_SHARES = [0.50, 0.60, 0.70, 0.80]


def chrono_key(r: dict[str, Any]) -> tuple:
    return (
        int(r["work_chronological_index"]),
        int(r["chronological_position"]),
        int(r["source_position_start"]),
        int(r["source_position_end"]),
        str(r["row_id"]),
    )


def pinyin_key(r: dict[str, Any]) -> str:
    segments = r.get("pinyin_segments")
    if isinstance(segments, list):
        return " ".join(map(str, segments))

    return str(r.get("pinyin_input", ""))


def target_of(r: dict[str, Any]) -> str:
    return str(r.get("gold", r.get("target", "")))


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_author(
    path: Path,
    source_partition: str,
) -> list[dict[str, Any]]:
    rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            x = json.loads(line)

            if x.get("author") != AUTHOR:
                continue

            x = dict(x)
            x["_long_term_source_partition"] = (
                source_partition
            )

            rows.append(x)

    return rows


def tokenizer_compatible(
    tokenizer,
    target: str,
) -> tuple[bool, list[str]]:
    bad = []

    for char in target:
        token_id = tokenizer.convert_tokens_to_ids(char)

        if token_id == tokenizer.unk_token_id:
            bad.append(char)

    return len(bad) == 0, bad


def block_preferences(
    rows_by_work: dict[int, list[dict[str, Any]]],
    first_work: int,
    last_work: int,
) -> dict[str, dict[str, Any]]:
    pair_counts = Counter()
    totals = Counter()

    for work in range(first_work, last_work + 1):
        for row in rows_by_work.get(work, []):
            if not row["_tokenizer_compatible"]:
                continue

            pinyin = pinyin_key(row)
            target = target_of(row)

            pair_counts[(pinyin, target)] += 1
            totals[pinyin] += 1

    by_pinyin = defaultdict(list)

    for (pinyin, target), count in pair_counts.items():
        by_pinyin[pinyin].append(
            (target, count)
        )

    result = {}

    for pinyin, candidates in by_pinyin.items():
        candidates.sort(
            key=lambda x: (-x[1], x[0])
        )

        target, count = candidates[0]
        total = totals[pinyin]

        runner_count = (
            candidates[1][1]
            if len(candidates) > 1
            else 0
        )

        result[pinyin] = {
            "candidate": target,
            "count": count,
            "total": total,
            "share": count / total,
            "runner_up_count": runner_count,
            "runner_up_share": (
                runner_count / total
                if total
                else 0.0
            ),
            "distinct_candidates": len(candidates),
        }

    return result


def qualified_candidate(
    pref: dict[str, Any] | None,
    min_count: int,
    min_share: float,
) -> str | None:
    if pref is None:
        return None

    if pref["count"] < min_count:
        return None

    if pref["share"] < min_share:
        return None

    return str(pref["candidate"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading Agent chronology...")

    fit = load_author(
        TRAIN_FIT,
        "train_fit",
    )

    val = load_author(
        TRAIN_VAL,
        "train_val",
    )

    fit_ids = {str(x["row_id"]) for x in fit}
    val_ids = {str(x["row_id"]) for x in val}

    assert not (fit_ids & val_ids)

    rows = fit + val
    rows.sort(key=chrono_key)

    work_indices = sorted(
        {
            int(x["work_chronological_index"])
            for x in rows
        }
    )

    assert work_indices == list(range(45)), work_indices

    print("Loading frozen tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT,
        local_files_only=True,
    )

    compatibility = Counter()
    bad_rows = []

    for row in rows:
        target = target_of(row)

        ok, bad_chars = tokenizer_compatible(
            tokenizer,
            target,
        )

        row["_tokenizer_compatible"] = ok

        partition = row[
            "_long_term_source_partition"
        ]

        compatibility[
            (partition, "compatible" if ok else "invalid")
        ] += 1

        if not ok:
            bad_rows.append(
                {
                    "row_id": row["row_id"],
                    "partition": partition,
                    "work_index": (
                        row["work_chronological_index"]
                    ),
                    "date": row["work_creation_date"],
                    "pinyin": pinyin_key(row),
                    "target": target,
                    "unknown_chars": "".join(bad_chars),
                    "unknown_codepoints": " ".join(
                        f"U+{ord(c):04X}"
                        for c in bad_chars
                    ),
                }
            )

    rows_by_work = defaultdict(list)

    for row in rows:
        rows_by_work[
            int(row["work_chronological_index"])
        ].append(row)

    print()
    print("============================================")
    print(" COMBINED STREAM")
    print("============================================")
    print("Train-Fit rows =", len(fit))
    print("Train-Val rows =", len(val))
    print("Total nominal  =", len(rows))
    print("Works          =", len(work_indices))
    print(
        "Dates          =",
        rows[0]["work_creation_date"],
        "->",
        rows[-1]["work_creation_date"],
    )
    print("row_id overlap = 0")

    print()
    print("============================================")
    print(" TOKENIZER COMPATIBILITY")
    print("============================================")

    for partition in ["train_fit", "train_val"]:
        print(
            partition,
            "compatible=",
            compatibility[(partition, "compatible")],
            "invalid=",
            compatibility[(partition, "invalid")],
        )

    work_summary = []

    for work in work_indices:
        wr = rows_by_work[work]

        partitions = Counter(
            x["_long_term_source_partition"]
            for x in wr
        )

        compatible = sum(
            bool(x["_tokenizer_compatible"])
            for x in wr
        )

        pinyins = {
            pinyin_key(x)
            for x in wr
            if x["_tokenizer_compatible"]
        }

        targets = {
            target_of(x)
            for x in wr
            if x["_tokenizer_compatible"]
        }

        work_summary.append(
            {
                "work_index": work,
                "work_id": wr[0].get("work_id", ""),
                "date": wr[0]["work_creation_date"],
                "first_chrono": min(
                    x["chronological_position"]
                    for x in wr
                ),
                "last_chrono": max(
                    x["chronological_position"]
                    for x in wr
                ),
                "nominal_rows": len(wr),
                "compatible_rows": compatible,
                "invalid_rows": (
                    len(wr) - compatible
                ),
                "unique_pinyin": len(pinyins),
                "unique_targets": len(targets),
                "train_fit_rows": partitions["train_fit"],
                "train_val_rows": partitions["train_val"],
            }
        )

    write_csv(
        OUT / "work_summary.csv",
        work_summary,
    )

    write_csv(
        OUT / "tokenizer_invalid_rows.csv",
        bad_rows,
    )

    print()
    print("============================================")
    print(" WORK-LEVEL CHRONOLOGY")
    print("============================================")

    for x in work_summary:
        print(
            f'work={x["work_index"]:2d} '
            f'date={x["date"]} '
            f'rows={x["nominal_rows"]:5d} '
            f'compatible={x["compatible_rows"]:5d} '
            f'pinyin={x["unique_pinyin"]:4d} '
            f'partition='
            f'{"fit" if x["train_fit_rows"] else "val"}'
        )

    # ---------------------------------------------------------
    # Build block preference cache.
    # ---------------------------------------------------------

    cache = {}

    def get_block(first_work: int, span: int):
        key = (first_work, span)

        if key not in cache:
            cache[key] = block_preferences(
                rows_by_work,
                first_work,
                first_work + span - 1,
            )

        return cache[key]

    grid_rows = []
    example_rows = []

    # Scan every possible chronological A/B/C triplet where
    # each block contains 'span' consecutive works.
    for span in BLOCK_SPANS:
        max_start = 45 - 3 * span

        for min_count in MIN_COUNTS:
            for min_share in MIN_SHARES:
                type_counts = Counter()
                type_pinyin = defaultdict(set)
                stored_examples = Counter()

                for start in range(max_start + 1):
                    b1_first = start
                    b2_first = start + span
                    b3_first = start + 2 * span

                    p1 = get_block(b1_first, span)
                    p2 = get_block(b2_first, span)
                    p3 = get_block(b3_first, span)

                    all_pinyin = (
                        set(p1)
                        | set(p2)
                        | set(p3)
                    )

                    for pinyin in all_pinyin:
                        a = qualified_candidate(
                            p1.get(pinyin),
                            min_count,
                            min_share,
                        )

                        b = qualified_candidate(
                            p2.get(pinyin),
                            min_count,
                            min_share,
                        )

                        c = qualified_candidate(
                            p3.get(pinyin),
                            min_count,
                            min_share,
                        )

                        kind = None

                        if (
                            a is not None
                            and a == b == c
                        ):
                            kind = "durable_AAA"

                        elif (
                            a is not None
                            and b is not None
                            and c is not None
                            and a != b
                            and b == c
                        ):
                            kind = "persistent_ABB"

                        elif (
                            a is not None
                            and b is not None
                            and c is not None
                            and a == c
                            and a != b
                        ):
                            kind = "temporary_ABA"

                        elif (
                            a is None
                            and b is not None
                            and b == c
                        ):
                            kind = "emerging_noneBB"

                        elif (
                            a is not None
                            and a == b
                            and c is None
                        ):
                            kind = "stale_AA_none"

                        if kind is None:
                            continue

                        type_counts[kind] += 1
                        type_pinyin[kind].add(pinyin)

                        if stored_examples[kind] >= 40:
                            continue

                        stored_examples[kind] += 1

                        def fields(
                            pref,
                            prefix,
                        ):
                            if pref is None:
                                return {
                                    f"{prefix}_candidate": "",
                                    f"{prefix}_count": 0,
                                    f"{prefix}_total": 0,
                                    f"{prefix}_share": 0.0,
                                }

                            return {
                                f"{prefix}_candidate": (
                                    pref["candidate"]
                                ),
                                f"{prefix}_count": (
                                    pref["count"]
                                ),
                                f"{prefix}_total": (
                                    pref["total"]
                                ),
                                f"{prefix}_share": (
                                    pref["share"]
                                ),
                            }

                        example = {
                            "span_works": span,
                            "min_count": min_count,
                            "min_share": min_share,
                            "transition_type": kind,
                            "pinyin": pinyin,
                            "block1_works": (
                                f"{b1_first}-"
                                f"{b1_first + span - 1}"
                            ),
                            "block2_works": (
                                f"{b2_first}-"
                                f"{b2_first + span - 1}"
                            ),
                            "block3_works": (
                                f"{b3_first}-"
                                f"{b3_first + span - 1}"
                            ),
                            "qualified_A": a or "",
                            "qualified_B": b or "",
                            "qualified_C": c or "",
                        }

                        example.update(
                            fields(
                                p1.get(pinyin),
                                "block1",
                            )
                        )

                        example.update(
                            fields(
                                p2.get(pinyin),
                                "block2",
                            )
                        )

                        example.update(
                            fields(
                                p3.get(pinyin),
                                "block3",
                            )
                        )

                        example_rows.append(example)

                grid_rows.append(
                    {
                        "span_works": span,
                        "min_count": min_count,
                        "min_share": min_share,
                        "durable_AAA_events": (
                            type_counts["durable_AAA"]
                        ),
                        "durable_AAA_unique_pinyin": len(
                            type_pinyin["durable_AAA"]
                        ),
                        "persistent_ABB_events": (
                            type_counts["persistent_ABB"]
                        ),
                        "persistent_ABB_unique_pinyin": len(
                            type_pinyin["persistent_ABB"]
                        ),
                        "temporary_ABA_events": (
                            type_counts["temporary_ABA"]
                        ),
                        "temporary_ABA_unique_pinyin": len(
                            type_pinyin["temporary_ABA"]
                        ),
                        "emerging_noneBB_events": (
                            type_counts["emerging_noneBB"]
                        ),
                        "emerging_noneBB_unique_pinyin": len(
                            type_pinyin["emerging_noneBB"]
                        ),
                        "stale_AA_none_events": (
                            type_counts["stale_AA_none"]
                        ),
                        "stale_AA_none_unique_pinyin": len(
                            type_pinyin["stale_AA_none"]
                        ),
                    }
                )

    write_csv(
        OUT / "transition_threshold_grid.csv",
        grid_rows,
    )

    write_csv(
        OUT / "transition_examples.csv",
        example_rows,
    )

    summary = {
        "schema_version": 1,
        "status": "AUDIT_ONLY_NOT_FROZEN",
        "author": AUTHOR,
        "test_used": False,
        "stream": {
            "train_fit_nominal": len(fit),
            "train_val_nominal": len(val),
            "combined_nominal": len(rows),
            "works": len(work_indices),
            "first_work": work_indices[0],
            "last_work": work_indices[-1],
            "first_date": rows[0]["work_creation_date"],
            "last_date": rows[-1]["work_creation_date"],
            "row_id_overlap_fit_val": 0,
        },
        "tokenizer_compatibility": {
            "train_fit_compatible": compatibility[
                ("train_fit", "compatible")
            ],
            "train_fit_invalid": compatibility[
                ("train_fit", "invalid")
            ],
            "train_val_compatible": compatibility[
                ("train_val", "compatible")
            ],
            "train_val_invalid": compatibility[
                ("train_val", "invalid")
            ],
        },
        "transition_audit": {
            "block_spans": BLOCK_SPANS,
            "min_counts": MIN_COUNTS,
            "min_shares": MIN_SHARES,
            "definition": (
                "Sliding three-block chronological scan over "
                "consecutive works. No threshold or block span "
                "is frozen by this audit."
            ),
        },
        "fixed_dev1000_policy": (
            "Existing Agent Dev1000 is a Train-Val subset and "
            "is not added to the combined stream."
        ),
    }

    dump_json(
        OUT / "audit_summary.json",
        summary,
    )

    print()
    print("============================================")
    print(" TRANSITION GRID")
    print("============================================")

    for x in grid_rows:
        if (
            x["min_share"] in {0.60, 0.70}
            and x["min_count"] in {3, 5}
        ):
            print(
                f'span={x["span_works"]} '
                f'count>={x["min_count"]} '
                f'share>={x["min_share"]:.2f} '
                f'AAA={x["durable_AAA_events"]}'
                f'/{x["durable_AAA_unique_pinyin"]}u '
                f'ABB={x["persistent_ABB_events"]}'
                f'/{x["persistent_ABB_unique_pinyin"]}u '
                f'ABA={x["temporary_ABA_events"]}'
                f'/{x["temporary_ABA_unique_pinyin"]}u '
                f'new={x["emerging_noneBB_events"]}'
                f'/{x["emerging_noneBB_unique_pinyin"]}u '
                f'stale={x["stale_AA_none_events"]}'
                f'/{x["stale_AA_none_unique_pinyin"]}u'
            )

    print()
    print("============================================")
    print(" OUTPUT")
    print("============================================")
    print(OUT)
    print("AUDIT COMPLETE -- NOTHING FROZEN")


if __name__ == "__main__":
    main()
