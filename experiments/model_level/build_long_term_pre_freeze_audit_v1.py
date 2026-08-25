from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SEED = 20260822
AUTHOR = "Agent Phage"
N_WINDOWS = 9
PROBE_MOD = 10

TRAIN_FIT = Path(
    "/home/3160454/work/model-level-assets-20260822/"
    "clean3_train_fit_v1.jsonl"
)

TOKENIZER_AUDIT = Path(
    "results/model_level/"
    "train_fit_tokenizer_unknown_audit_v1.json"
)

OUT = Path(
    "results/model_level/"
    "long_term_qualification_v1/"
    "audit_pre_freeze_v1"
)

MIN_COUNTS = [3, 5, 10, 20]
MIN_SHARES = [0.50, 0.60, 0.70, 0.80]


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: list[str] = []
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
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def chrono_key(r: dict[str, Any]) -> tuple:
    return (
        r["work_chronological_index"],
        r["chronological_position"],
        r["source_position_start"],
        r["source_position_end"],
        r["row_id"],
    )


def pinyin_key(r: dict[str, Any]) -> str:
    segments = r.get("pinyin_segments")
    if isinstance(segments, list) and segments:
        return " ".join(map(str, segments))
    return str(r.get("pinyin_input", ""))


def target_of(r: dict[str, Any]) -> str:
    return str(r.get("gold", r.get("target", "")))


def is_probe(row_id: str) -> bool:
    digest = hashlib.sha256(
        f"{SEED}:probe:{row_id}".encode("utf-8")
    ).digest()

    value = int.from_bytes(digest[:8], "big")

    return value % PROBE_MOD == 0


def load_effective_agent_rows() -> tuple[
    list[dict[str, Any]],
    list[str],
]:
    audit = json.loads(
        TOKENIZER_AUDIT.read_text(encoding="utf-8")
    )

    bad_ids = {
        str(x["row_id"])
        for x in audit["bad_rows"]
        if x["author"] == AUTHOR
    }

    rows = []

    with TRAIN_FIT.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if row.get("author") != AUTHOR:
                continue

            if str(row["row_id"]) in bad_ids:
                continue

            rows.append(row)

    rows.sort(key=chrono_key)

    return rows, sorted(bad_ids)


def equal_row_windows(
    rows: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    n = len(rows)
    windows = []

    for i in range(N_WINDOWS):
        start = (i * n) // N_WINDOWS
        end = ((i + 1) * n) // N_WINDOWS
        windows.append(rows[start:end])

    return windows


def work_boundary_windows(
    rows: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    grouped: list[list[dict[str, Any]]] = []

    current_work = None
    current_rows: list[dict[str, Any]] = []

    for row in rows:
        work = row["work_chronological_index"]

        if current_work is None:
            current_work = work

        if work != current_work:
            grouped.append(current_rows)
            current_rows = []
            current_work = work

        current_rows.append(row)

    if current_rows:
        grouped.append(current_rows)

    total = len(rows)
    cumulative = []

    running = 0
    for group in grouped:
        running += len(group)
        cumulative.append(running)

    cut_after_group: list[int] = []
    previous = -1

    for k in range(1, N_WINDOWS):
        target = total * k / N_WINDOWS

        min_j = previous + 1
        max_j = len(grouped) - (N_WINDOWS - k) - 1

        candidates = list(range(min_j, max_j + 1))

        best = min(
            candidates,
            key=lambda j: (
                abs(cumulative[j] - target),
                j,
            ),
        )

        cut_after_group.append(best)
        previous = best

    windows: list[list[dict[str, Any]]] = []
    start_group = 0

    for cut in cut_after_group + [len(grouped) - 1]:
        window = []

        for group in grouped[start_group : cut + 1]:
            window.extend(group)

        windows.append(window)
        start_group = cut + 1

    assert len(windows) == N_WINDOWS
    assert sum(map(len, windows)) == len(rows)

    return windows


def window_summary(
    scheme: str,
    windows: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    out = []

    for i, rows in enumerate(windows, 1):
        works = sorted(
            {
                int(r["work_chronological_index"])
                for r in rows
            }
        )

        probe_n = sum(
            is_probe(str(r["row_id"]))
            for r in rows
        )

        out.append(
            {
                "scheme": scheme,
                "window": f"W{i}",
                "phase": (
                    "T1"
                    if i <= 3
                    else "T2"
                    if i <= 6
                    else "T3"
                ),
                "rows": len(rows),
                "stream_rows": len(rows) - probe_n,
                "probe_rows": probe_n,
                "probe_fraction": (
                    probe_n / len(rows)
                    if rows
                    else 0.0
                ),
                "first_date": rows[0]["work_creation_date"],
                "last_date": rows[-1]["work_creation_date"],
                "first_work": works[0],
                "last_work": works[-1],
                "n_works": len(works),
                "starts_mid_work": False,
                "ends_mid_work": False,
            }
        )

    work_to_windows: dict[int, set[int]] = defaultdict(set)

    for i, rows in enumerate(windows, 1):
        for row in rows:
            work_to_windows[
                int(row["work_chronological_index"])
            ].add(i)

    split_works = {
        work: sorted(ws)
        for work, ws in work_to_windows.items()
        if len(ws) > 1
    }

    for i, row in enumerate(out, 1):
        window_works = {
            int(r["work_chronological_index"])
            for r in windows[i - 1]
        }

        split_here = {
            w
            for w in window_works
            if w in split_works
        }

        row["split_work_count"] = len(split_here)
        row["split_works"] = ",".join(
            map(str, sorted(split_here))
        )

    return out


def save_assignments(
    scheme: str,
    windows: list[list[dict[str, Any]]],
) -> None:
    path = OUT / f"{scheme}_window_assignments.jsonl"

    with path.open("w", encoding="utf-8") as f:
        for i, rows in enumerate(windows, 1):
            for row in rows:
                x = {
                    "row_id": row["row_id"],
                    "window": f"W{i}",
                    "phase": (
                        "T1"
                        if i <= 3
                        else "T2"
                        if i <= 6
                        else "T3"
                    ),
                    "probe": is_probe(
                        str(row["row_id"])
                    ),
                    "work_chronological_index": (
                        row["work_chronological_index"]
                    ),
                    "chronological_position": (
                        row["chronological_position"]
                    ),
                    "work_creation_date": (
                        row["work_creation_date"]
                    ),
                    "pinyin": pinyin_key(row),
                    "target": target_of(row),
                }

                f.write(
                    json.dumps(
                        x,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )


def top_preferences(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    totals: Counter[str] = Counter()

    for row in rows:
        if is_probe(str(row["row_id"])):
            continue

        pinyin = pinyin_key(row)
        target = target_of(row)

        pair_counts[(pinyin, target)] += 1
        totals[pinyin] += 1

    by_pinyin: dict[str, list[tuple[str, int]]] = defaultdict(list)

    for (pinyin, target), count in pair_counts.items():
        by_pinyin[pinyin].append((target, count))

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
            "share": (
                count / total
                if total
                else 0.0
            ),
            "runner_up_count": runner_count,
            "runner_up_share": (
                runner_count / total
                if total
                else 0.0
            ),
            "distinct_candidates": len(candidates),
        }

    return result


def qualified(
    item: dict[str, Any] | None,
    min_count: int,
    min_share: float,
) -> str | None:
    if item is None:
        return None

    if item["count"] < min_count:
        return None

    if item["share"] < min_share:
        return None

    return str(item["candidate"])


def transition_grid(
    scheme: str,
    windows: list[list[dict[str, Any]]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    prefs = [
        top_preferences(rows)
        for rows in windows
    ]

    all_pinyin = sorted(
        set().union(
            *(set(p) for p in prefs)
        )
    )

    grid_rows = []
    example_rows = []

    for min_count in MIN_COUNTS:
        for min_share in MIN_SHARES:
            counts = Counter()

            examples_by_type: dict[
                str,
                list[dict[str, Any]],
            ] = defaultdict(list)

            for pinyin in all_pinyin:
                q = [
                    qualified(
                        prefs[i].get(pinyin),
                        min_count,
                        min_share,
                    )
                    for i in range(N_WINDOWS)
                ]

                w7, w8, w9 = q[6], q[7], q[8]

                kind = None

                if (
                    w7 is not None
                    and w7 == w8 == w9
                ):
                    kind = "durable_T3_AAA"

                elif (
                    w7 is not None
                    and w8 is not None
                    and w9 is not None
                    and w7 != w8
                    and w8 == w9
                ):
                    kind = "persistent_switch_ABB"

                elif (
                    w7 is not None
                    and w8 is not None
                    and w9 is not None
                    and w7 == w9
                    and w7 != w8
                ):
                    kind = "temporary_switch_ABA"

                elif (
                    w7 is None
                    and w8 is not None
                    and w8 == w9
                ):
                    kind = "emerging_noneBB"

                elif (
                    w7 is not None
                    and w8 == w7
                    and w9 is None
                ):
                    kind = "stale_AA_none"

                if kind is not None:
                    counts[kind] += 1

                    if len(examples_by_type[kind]) < 25:
                        examples_by_type[kind].append(
                            {
                                "scheme": scheme,
                                "min_count": min_count,
                                "min_share": min_share,
                                "transition_type": kind,
                                "pinyin": pinyin,
                                "W7": w7,
                                "W8": w8,
                                "W9": w9,
                                "W7_count": (
                                    prefs[6]
                                    .get(pinyin, {})
                                    .get("count")
                                ),
                                "W7_share": (
                                    prefs[6]
                                    .get(pinyin, {})
                                    .get("share")
                                ),
                                "W8_count": (
                                    prefs[7]
                                    .get(pinyin, {})
                                    .get("count")
                                ),
                                "W8_share": (
                                    prefs[7]
                                    .get(pinyin, {})
                                    .get("share")
                                ),
                                "W9_count": (
                                    prefs[8]
                                    .get(pinyin, {})
                                    .get("count")
                                ),
                                "W9_share": (
                                    prefs[8]
                                    .get(pinyin, {})
                                    .get("share")
                                ),
                            }
                        )

            row = {
                "scheme": scheme,
                "min_count": min_count,
                "min_share": min_share,
                "durable_T3_AAA": counts[
                    "durable_T3_AAA"
                ],
                "persistent_switch_ABB": counts[
                    "persistent_switch_ABB"
                ],
                "temporary_switch_ABA": counts[
                    "temporary_switch_ABA"
                ],
                "emerging_noneBB": counts[
                    "emerging_noneBB"
                ],
                "stale_AA_none": counts[
                    "stale_AA_none"
                ],
            }

            grid_rows.append(row)

            for rows_ in examples_by_type.values():
                example_rows.extend(rows_)

    return grid_rows, example_rows


def build_history_sets(
    rows: list[dict[str, Any]],
) -> dict[str, list[str]]:
    oldest = rows[:5000]
    recent = rows[-5000:]

    rng = random.Random(SEED)
    random_rows = rng.sample(rows, 5000)
    random_rows.sort(key=chrono_key)

    sets = {
        "oldest5000": [
            str(r["row_id"])
            for r in oldest
        ],
        "random5000": [
            str(r["row_id"])
            for r in random_rows
        ],
        "recent5000": [
            str(r["row_id"])
            for r in recent
        ],
    }

    return sets


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    rows, bad_ids = load_effective_agent_rows()

    assert len(rows) == 55925, len(rows)
    assert len(bad_ids) == 1, bad_ids

    print("effective Agent rows =", len(rows))
    print("excluded tokenizer-invalid =", bad_ids)

    history_sets = build_history_sets(rows)

    for name, ids in history_sets.items():
        dump_json(
            OUT / f"{name}_row_ids.json",
            {
                "schema_version": 1,
                "status": "AUDIT_ONLY_NOT_FROZEN",
                "author": AUTHOR,
                "seed": SEED,
                "population": name,
                "n": len(ids),
                "row_ids": ids,
            },
        )

    set_names = list(history_sets)
    overlap_rows = []

    for i, a in enumerate(set_names):
        for b in set_names[i + 1 :]:
            overlap_rows.append(
                {
                    "set_a": a,
                    "set_b": b,
                    "overlap": len(
                        set(history_sets[a])
                        & set(history_sets[b])
                    ),
                }
            )

    write_csv(
        OUT / "history_set_overlap.csv",
        overlap_rows,
    )

    schemes = {
        "equal_rows": equal_row_windows(rows),
        "work_boundary": work_boundary_windows(rows),
    }

    all_window_summary = []
    all_grid = []
    all_examples = []

    for scheme, windows in schemes.items():
        summary = window_summary(
            scheme,
            windows,
        )
        all_window_summary.extend(summary)

        save_assignments(
            scheme,
            windows,
        )

        grid, examples = transition_grid(
            scheme,
            windows,
        )

        all_grid.extend(grid)
        all_examples.extend(examples)

    write_csv(
        OUT / "window_scheme_comparison.csv",
        all_window_summary,
    )

    write_csv(
        OUT / "transition_threshold_grid.csv",
        all_grid,
    )

    write_csv(
        OUT / "transition_examples.csv",
        all_examples,
    )

    summary = {
        "schema_version": 1,
        "status": "AUDIT_ONLY_NOT_FROZEN",
        "author": AUTHOR,
        "seed": SEED,
        "nominal_rows": 55926,
        "effective_rows": len(rows),
        "tokenizer_invalid_excluded": len(bad_ids),
        "excluded_row_ids": bad_ids,
        "chronology": {
            "first_date": rows[0]["work_creation_date"],
            "last_date": rows[-1]["work_creation_date"],
            "first_work": rows[0][
                "work_chronological_index"
            ],
            "last_work": rows[-1][
                "work_chronological_index"
            ],
            "unique_works": len(
                {
                    r["work_chronological_index"]
                    for r in rows
                }
            ),
        },
        "probe_policy": {
            "status": "PROVISIONAL",
            "method": (
                "sha256(seed:probe:row_id) modulo 10 == 0"
            ),
            "target_fraction": 0.10,
        },
        "window_schemes": [
            "equal_rows",
            "work_boundary",
        ],
        "transition_threshold_grid": {
            "min_counts": MIN_COUNTS,
            "min_shares": MIN_SHARES,
        },
        "note": (
            "No window scheme, probe policy, or transition "
            "threshold is frozen by this audit."
        ),
    }

    dump_json(
        OUT / "audit_summary.json",
        summary,
    )

    print()
    print("===== WINDOW SCHEME COMPARISON =====")

    for row in all_window_summary:
        print(
            f'{row["scheme"]:14s} '
            f'{row["window"]:2s} '
            f'rows={row["rows"]:5d} '
            f'probe={row["probe_rows"]:4d} '
            f'works={row["n_works"]:2d} '
            f'split_works={row["split_work_count"]:2d} '
            f'{row["first_date"]} -> {row["last_date"]}'
        )

    print()
    print("===== HISTORY SET OVERLAP =====")

    for row in overlap_rows:
        print(
            row["set_a"],
            "<->",
            row["set_b"],
            "=",
            row["overlap"],
        )

    print()
    print("===== TRANSITION THRESHOLD GRID =====")

    for row in all_grid:
        print(
            f'{row["scheme"]:14s} '
            f'count>={row["min_count"]:2d} '
            f'share>={row["min_share"]:.2f} '
            f'AAA={row["durable_T3_AAA"]:4d} '
            f'ABB={row["persistent_switch_ABB"]:4d} '
            f'ABA={row["temporary_switch_ABA"]:4d} '
            f'noneBB={row["emerging_noneBB"]:4d} '
            f'AAnone={row["stale_AA_none"]:4d}'
        )

    print()
    print("===== OUTPUT =====")
    print(OUT)
    print("AUDIT COMPLETE -- NOTHING FROZEN")
    

if __name__ == "__main__":
    main()
