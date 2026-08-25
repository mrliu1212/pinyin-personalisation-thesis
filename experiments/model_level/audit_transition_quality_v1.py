from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


AUTHOR = "Agent Phage"

TRAIN_FIT = Path(
    "/home/3160454/work/model-level-assets-20260822/"
    "clean3_train_fit_v1.jsonl"
)

TRAIN_VAL = Path(
    "/home/3160454/work/model-level-assets-20260822/"
    "clean3_train_val_v1.jsonl"
)

BASE_AUDIT = Path(
    "results/model_level/"
    "long_term_qualification_v1/"
    "combined_work_audit_v1"
)

TRANSITIONS = BASE_AUDIT / "transition_examples.csv"
INVALID = BASE_AUDIT / "tokenizer_invalid_rows.csv"

OUT = Path(
    "results/model_level/"
    "long_term_qualification_v1/"
    "transition_quality_audit_v1"
)

SPAN = 3
MIN_COUNT = 3
MIN_SHARE = 0.60

# Audit-only quality criteria. NOT frozen.
MIN_DISTINCT_WORK_SUPPORT = 2
DOMINANT_WORK_WARNING = 0.80
CONTEXT_SAMPLES_PER_STATE = 3


def pinyin_key(r: dict[str, Any]) -> str:
    seg = r.get("pinyin_segments")

    if isinstance(seg, list):
        return " ".join(map(str, seg))

    return str(r.get("pinyin_input", ""))


def target_of(r: dict[str, Any]) -> str:
    return str(r.get("gold", r.get("target", "")))


def chrono_key(r: dict[str, Any]) -> tuple:
    return (
        int(r["work_chronological_index"]),
        int(r["chronological_position"]),
        int(r["source_position_start"]),
        int(r["source_position_end"]),
        str(r["row_id"]),
    )


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


def parse_work_range(s: str) -> tuple[int, int]:
    a, b = s.split("-")
    return int(a), int(b)


def clean_context(text: Any, limit: int = 220) -> str:
    if text is None:
        return ""

    s = str(text)
    s = " ".join(s.split())

    if len(s) > limit:
        s = "..." + s[-limit:]

    return s


def load_rows() -> list[dict[str, Any]]:
    invalid_ids = set()

    if INVALID.exists():
        with INVALID.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                invalid_ids.add(r["row_id"])

    rows = []

    for path, partition in [
        (TRAIN_FIT, "train_fit"),
        (TRAIN_VAL, "train_val"),
    ]:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                x = json.loads(line)

                if x.get("author") != AUTHOR:
                    continue

                if str(x["row_id"]) in invalid_ids:
                    continue

                x = dict(x)
                x["_partition"] = partition
                rows.append(x)

    rows.sort(key=chrono_key)

    return rows


def state_stats(
    rows_by_work: dict[int, list[dict[str, Any]]],
    pinyin: str,
    candidate: str,
    first_work: int,
    last_work: int,
) -> dict[str, Any]:
    candidate_counts = Counter()
    total_counts = Counter()

    for work in range(first_work, last_work + 1):
        for r in rows_by_work.get(work, []):
            if pinyin_key(r) != pinyin:
                continue

            total_counts[work] += 1

            if target_of(r) == candidate:
                candidate_counts[work] += 1

    candidate_count = sum(candidate_counts.values())
    total = sum(total_counts.values())

    distinct_candidate_works = sum(
        1
        for count in candidate_counts.values()
        if count > 0
    )

    distinct_pinyin_works = sum(
        1
        for count in total_counts.values()
        if count > 0
    )

    dominant_work_count = (
        max(candidate_counts.values())
        if candidate_counts
        else 0
    )

    dominant_work_fraction = (
        dominant_work_count / candidate_count
        if candidate_count
        else 0.0
    )

    return {
        "candidate_count": candidate_count,
        "pinyin_total": total,
        "choice_share": (
            candidate_count / total
            if total
            else 0.0
        ),
        "candidate_distinct_works": (
            distinct_candidate_works
        ),
        "pinyin_distinct_works": distinct_pinyin_works,
        "dominant_work_count": dominant_work_count,
        "dominant_work_fraction": (
            dominant_work_fraction
        ),
        "candidate_per_work": json.dumps(
            dict(sorted(candidate_counts.items())),
            ensure_ascii=False,
        ),
        "pinyin_total_per_work": json.dumps(
            dict(sorted(total_counts.items())),
            ensure_ascii=False,
        ),
    }


def top_state(
    rows_by_work: dict[int, list[dict[str, Any]]],
    pinyin: str,
    first_work: int,
    last_work: int,
) -> dict[str, Any]:
    pair_counts = Counter()
    per_candidate_works = defaultdict(Counter)
    total = 0

    for work in range(first_work, last_work + 1):
        for r in rows_by_work.get(work, []):
            if pinyin_key(r) != pinyin:
                continue

            candidate = target_of(r)
            pair_counts[candidate] += 1
            per_candidate_works[candidate][work] += 1
            total += 1

    if not pair_counts:
        return {
            "candidate": "",
            "count": 0,
            "total": 0,
            "share": 0.0,
            "distinct_works": 0,
            "dominant_work_fraction": 0.0,
            "qualified_basic": False,
            "qualified_cross_work": False,
        }

    candidate, count = sorted(
        pair_counts.items(),
        key=lambda x: (-x[1], x[0]),
    )[0]

    works = per_candidate_works[candidate]

    distinct_works = len(works)
    dominant = max(works.values()) / count

    share = count / total

    qualified_basic = (
        count >= MIN_COUNT
        and share >= MIN_SHARE
    )

    qualified_cross_work = (
        qualified_basic
        and distinct_works >= MIN_DISTINCT_WORK_SUPPORT
    )

    return {
        "candidate": candidate,
        "count": count,
        "total": total,
        "share": share,
        "distinct_works": distinct_works,
        "dominant_work_fraction": dominant,
        "qualified_basic": qualified_basic,
        "qualified_cross_work": qualified_cross_work,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    rows = load_rows()

    rows_by_work = defaultdict(list)
    work_sizes = Counter()

    for r in rows:
        work = int(r["work_chronological_index"])
        rows_by_work[work].append(r)
        work_sizes[work] += 1

    median_work = statistics.median(work_sizes.values())

    giant_works = {
        work
        for work, n in work_sizes.items()
        if n >= 5000
    }

    print("==============================================")
    print(" TRANSITION QUALITY AUDIT")
    print("==============================================")
    print("effective rows =", len(rows))
    print("median work rows =", median_work)
    print(
        "giant works =",
        {
            w: work_sizes[w]
            for w in sorted(giant_works)
        },
    )

    raw = list(
        csv.DictReader(
            TRANSITIONS.open(encoding="utf-8")
        )
    )

    raw = [
        r
        for r in raw
        if int(r["span_works"]) == SPAN
        and int(r["min_count"]) == MIN_COUNT
        and abs(
            float(r["min_share"]) - MIN_SHARE
        ) < 1e-12
        and r["transition_type"] in {
            "temporary_ABA",
            "persistent_ABB",
        }
    ]

    print()
    print("raw transition detections =", len(raw))

    # ---------------------------------------------------------
    # Annotate every raw transition with cross-work support.
    # ---------------------------------------------------------

    quality_rows = []
    context_rows = []

    for r in raw:
        pinyin = r["pinyin"]

        states = [
            (
                "A",
                r["qualified_A"],
                r["block1_works"],
            ),
            (
                "B",
                r["qualified_B"],
                r["block2_works"],
            ),
            (
                "C",
                r["qualified_C"],
                r["block3_works"],
            ),
        ]

        q = dict(r)

        all_cross_work = True
        any_dominance_warning = False
        contains_giant_work = False

        quality_score_parts = []

        for label, candidate, work_range in states:
            first, last = parse_work_range(
                work_range
            )

            stats = state_stats(
                rows_by_work,
                pinyin,
                candidate,
                first,
                last,
            )

            for key, value in stats.items():
                q[f"{label}_{key}"] = value

            cross_work = (
                stats["candidate_distinct_works"]
                >= MIN_DISTINCT_WORK_SUPPORT
            )

            dominance_warning = (
                stats["dominant_work_fraction"]
                > DOMINANT_WORK_WARNING
            )

            q[f"{label}_cross_work_supported"] = (
                cross_work
            )

            q[f"{label}_dominance_warning"] = (
                dominance_warning
            )

            all_cross_work = (
                all_cross_work
                and cross_work
            )

            any_dominance_warning = (
                any_dominance_warning
                or dominance_warning
            )

            if any(
                w in giant_works
                for w in range(first, last + 1)
            ):
                contains_giant_work = True

            quality_score_parts.append(
                (
                    stats["candidate_distinct_works"],
                    stats["candidate_count"],
                    stats["choice_share"],
                    -stats["dominant_work_fraction"],
                )
            )

            # Context samples for the winning candidate.
            matches = []

            for work in range(first, last + 1):
                for row in rows_by_work[work]:
                    if (
                        pinyin_key(row) == pinyin
                        and target_of(row) == candidate
                    ):
                        matches.append(row)

            # Prefer coverage across works rather than first N
            # rows from one work.
            chosen = []
            seen_works = set()

            for row in matches:
                work = int(
                    row["work_chronological_index"]
                )

                if work in seen_works:
                    continue

                chosen.append(row)
                seen_works.add(work)

                if (
                    len(chosen)
                    >= CONTEXT_SAMPLES_PER_STATE
                ):
                    break

            if (
                len(chosen)
                < CONTEXT_SAMPLES_PER_STATE
            ):
                for row in matches:
                    if row in chosen:
                        continue

                    chosen.append(row)

                    if (
                        len(chosen)
                        >= CONTEXT_SAMPLES_PER_STATE
                    ):
                        break

            for i, row in enumerate(chosen, 1):
                context_rows.append(
                    {
                        "transition_type": (
                            r["transition_type"]
                        ),
                        "pinyin": pinyin,
                        "A": r["qualified_A"],
                        "B": r["qualified_B"],
                        "C": r["qualified_C"],
                        "state": label,
                        "state_candidate": candidate,
                        "state_work_range": work_range,
                        "sample_index": i,
                        "row_id": row["row_id"],
                        "partition": row["_partition"],
                        "work_index": (
                            row[
                                "work_chronological_index"
                            ]
                        ),
                        "date": (
                            row["work_creation_date"]
                        ),
                        "gold": target_of(row),
                        "context_tail": clean_context(
                            row.get("context")
                        ),
                    }
                )

        q["all_states_cross_work_supported"] = (
            all_cross_work
        )

        q["any_dominance_warning"] = (
            any_dominance_warning
        )

        q["contains_giant_work"] = (
            contains_giant_work
        )

        q["audit_robust_candidate"] = (
            all_cross_work
            and not any_dominance_warning
        )

        q["_quality_score"] = min(
            quality_score_parts
        )

        quality_rows.append(q)

    # ---------------------------------------------------------
    # Deduplicate overlapping sliding detections.
    #
    # Same transition type + pinyin + A/B/C is treated as one
    # candidate episode. Keep strongest observed window.
    # ---------------------------------------------------------

    dedup = {}

    for r in quality_rows:
        key = (
            r["transition_type"],
            r["pinyin"],
            r["qualified_A"],
            r["qualified_B"],
            r["qualified_C"],
        )

        score = (
            bool(r["audit_robust_candidate"]),
            int(
                r[
                    "A_candidate_distinct_works"
                ]
            ),
            int(
                r[
                    "B_candidate_distinct_works"
                ]
            ),
            int(
                r[
                    "C_candidate_distinct_works"
                ]
            ),
            min(
                int(r["A_candidate_count"]),
                int(r["B_candidate_count"]),
                int(r["C_candidate_count"]),
            ),
            min(
                float(r["A_choice_share"]),
                float(r["B_choice_share"]),
                float(r["C_choice_share"]),
            ),
            -max(
                float(
                    r[
                        "A_dominant_work_fraction"
                    ]
                ),
                float(
                    r[
                        "B_dominant_work_fraction"
                    ]
                ),
                float(
                    r[
                        "C_dominant_work_fraction"
                    ]
                ),
            ),
        )

        old = dedup.get(key)

        if old is None or score > old[0]:
            dedup[key] = (score, r)

    dedup_rows = [
        r
        for _, r in dedup.values()
    ]

    dedup_rows.sort(
        key=lambda r: (
            r["transition_type"],
            r["pinyin"],
            int(
                r["block1_works"].split("-")[0]
            ),
        )
    )

    for r in quality_rows:
        r.pop("_quality_score", None)

    for r in dedup_rows:
        r.pop("_quality_score", None)

    write_csv(
        OUT / "raw_transition_quality.csv",
        quality_rows,
    )

    write_csv(
        OUT / "deduplicated_transition_quality.csv",
        dedup_rows,
    )

    write_csv(
        OUT / "context_samples.csv",
        context_rows,
    )

    # ---------------------------------------------------------
    # Full rolling trajectories for all transition pinyins.
    # ---------------------------------------------------------

    target_pinyins = sorted(
        {
            r["pinyin"]
            for r in raw
        }
    )

    trajectory_rows = []

    for pinyin in target_pinyins:
        for start in range(0, 45 - SPAN + 1):
            end = start + SPAN - 1

            state = top_state(
                rows_by_work,
                pinyin,
                start,
                end,
            )

            trajectory_rows.append(
                {
                    "pinyin": pinyin,
                    "window_start": start,
                    "window_end": end,
                    "works": f"{start}-{end}",
                    **state,
                }
            )

    write_csv(
        OUT / "rolling_state_trajectories.csv",
        trajectory_rows,
    )

    # ---------------------------------------------------------
    # Collapse consecutive cross-work-qualified states.
    # A gap / unqualified window breaks a run.
    # ---------------------------------------------------------

    collapsed_rows = []

    by_pinyin = defaultdict(list)

    for r in trajectory_rows:
        by_pinyin[r["pinyin"]].append(r)

    for pinyin, traj in by_pinyin.items():
        current = None
        run_id = 0

        def flush():
            nonlocal current

            if current is not None:
                collapsed_rows.append(current)
                current = None

        for state in traj:
            qualified = (
                str(
                    state["qualified_cross_work"]
                ).lower()
                == "true"
                if isinstance(
                    state["qualified_cross_work"],
                    str,
                )
                else bool(
                    state[
                        "qualified_cross_work"
                    ]
                )
            )

            candidate = (
                state["candidate"]
                if qualified
                else None
            )

            if candidate is None:
                flush()
                continue

            if (
                current is not None
                and current["candidate"]
                == candidate
                and int(
                    current[
                        "last_window_start"
                    ]
                )
                + 1
                == int(
                    state["window_start"]
                )
            ):
                current["last_window_start"] = (
                    state["window_start"]
                )
                current["last_window_end"] = (
                    state["window_end"]
                )
                current["n_windows"] += 1
                current["max_count"] = max(
                    current["max_count"],
                    state["count"],
                )
                current["min_share"] = min(
                    current["min_share"],
                    state["share"],
                )
                current[
                    "max_dominant_work_fraction"
                ] = max(
                    current[
                        "max_dominant_work_fraction"
                    ],
                    state[
                        "dominant_work_fraction"
                    ],
                )
            else:
                flush()
                run_id += 1

                current = {
                    "pinyin": pinyin,
                    "run_id": run_id,
                    "candidate": candidate,
                    "first_window_start": (
                        state["window_start"]
                    ),
                    "first_window_end": (
                        state["window_end"]
                    ),
                    "last_window_start": (
                        state["window_start"]
                    ),
                    "last_window_end": (
                        state["window_end"]
                    ),
                    "n_windows": 1,
                    "max_count": state["count"],
                    "min_share": state["share"],
                    "max_dominant_work_fraction": (
                        state[
                            "dominant_work_fraction"
                        ]
                    ),
                }

        flush()

    write_csv(
        OUT / "collapsed_state_runs.csv",
        collapsed_rows,
    )

    # ---------------------------------------------------------
    # Console summary.
    # ---------------------------------------------------------

    print()
    print("==============================================")
    print(" DEDUPLICATED QUALITY SUMMARY")
    print("==============================================")

    for kind in [
        "temporary_ABA",
        "persistent_ABB",
    ]:
        xs = [
            r
            for r in dedup_rows
            if r["transition_type"] == kind
        ]

        robust = [
            r
            for r in xs
            if r["audit_robust_candidate"]
        ]

        cross = [
            r
            for r in xs
            if r[
                "all_states_cross_work_supported"
            ]
        ]

        print()
        print(kind)
        print("deduplicated =", len(xs))
        print(
            "all states supported by >=2 works =",
            len(cross),
        )
        print(
            "provisional robust candidates =",
            len(robust),
        )

        for r in xs:
            print(
                f'  {r["pinyin"]!r:18s} '
                f'{r["qualified_A"]!r}'
                f' -> {r["qualified_B"]!r}'
                f' -> {r["qualified_C"]!r} '
                f'works='
                f'{r["block1_works"]}/'
                f'{r["block2_works"]}/'
                f'{r["block3_works"]} '
                f'support='
                f'{r["A_candidate_distinct_works"]}/'
                f'{r["B_candidate_distinct_works"]}/'
                f'{r["C_candidate_distinct_works"]} '
                f'dom='
                f'{float(r["A_dominant_work_fraction"]):.2f}/'
                f'{float(r["B_dominant_work_fraction"]):.2f}/'
                f'{float(r["C_dominant_work_fraction"]):.2f} '
                f'giant={r["contains_giant_work"]} '
                f'robust={r["audit_robust_candidate"]}'
            )

    print()
    print("==============================================")
    print(" COLLAPSED STATE TRAJECTORIES")
    print("==============================================")

    for pinyin in target_pinyins:
        runs = [
            r
            for r in collapsed_rows
            if r["pinyin"] == pinyin
        ]

        if not runs:
            continue

        print()
        print("pinyin =", repr(pinyin))

        for r in runs:
            print(
                f'  {r["candidate"]!r} '
                f'windows='
                f'{r["first_window_start"]}'
                f'-{r["last_window_start"]} '
                f'n={r["n_windows"]} '
                f'min_share='
                f'{float(r["min_share"]):.3f}'
            )

    summary = {
        "schema_version": 1,
        "status": "AUDIT_ONLY_NOT_FROZEN",
        "author": AUTHOR,
        "test_used": False,
        "effective_combined_rows": len(rows),
        "primary_audit_definition": {
            "span_works": SPAN,
            "minimum_count": MIN_COUNT,
            "minimum_choice_share": MIN_SHARE,
            "minimum_distinct_work_support": (
                MIN_DISTINCT_WORK_SUPPORT
            ),
            "dominant_work_warning_threshold": (
                DOMINANT_WORK_WARNING
            ),
            "note": (
                "All quality criteria remain provisional."
            ),
        },
        "giant_works": {
            str(w): work_sizes[w]
            for w in sorted(giant_works)
        },
        "raw_transition_detections": len(raw),
        "deduplicated_transition_candidates": len(
            dedup_rows
        ),
        "temporary_deduplicated": sum(
            r["transition_type"]
            == "temporary_ABA"
            for r in dedup_rows
        ),
        "persistent_deduplicated": sum(
            r["transition_type"]
            == "persistent_ABB"
            for r in dedup_rows
        ),
        "provisional_robust_temporary": sum(
            r["transition_type"]
            == "temporary_ABA"
            and bool(r["audit_robust_candidate"])
            for r in dedup_rows
        ),
        "provisional_robust_persistent": sum(
            r["transition_type"]
            == "persistent_ABB"
            and bool(r["audit_robust_candidate"])
            for r in dedup_rows
        ),
    }

    dump_json(
        OUT / "audit_summary.json",
        summary,
    )

    print()
    print("==============================================")
    print(" OUTPUT FILES")
    print("==============================================")
    print(OUT / "audit_summary.json")
    print(OUT / "deduplicated_transition_quality.csv")
    print(OUT / "rolling_state_trajectories.csv")
    print(OUT / "collapsed_state_runs.csv")
    print(OUT / "context_samples.csv")
    print()
    print("AUDIT COMPLETE -- NOTHING FROZEN")


if __name__ == "__main__":
    main()
