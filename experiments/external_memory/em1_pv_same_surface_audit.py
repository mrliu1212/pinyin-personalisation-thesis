from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


AUTHORS = (
    "Etinjat",
    "Re_spectators",
    "breaddddd",
)

EXPECTED_ROWS = 3000
EXPECTED_PER_AUTHOR = 1000
EXPECTED_F_MACRO = 0.8106666666666666


def read_jsonl(path: Path):
    rows = []

    with path.open(
        encoding="utf-8"
    ) as source:
        for line_no, line in enumerate(
            source,
            start=1,
        ):
            if not line.strip():
                continue

            value = json.loads(line)

            if not isinstance(
                value,
                dict,
            ):
                raise RuntimeError(
                    f"Invalid row "
                    f"{path}:{line_no}"
                )

            rows.append(value)

    return rows


def macro_top1(
    rows,
    rank_key,
):
    values = []

    for author in AUTHORS:
        selected = [
            row
            for row in rows
            if row["author"] == author
        ]

        if (
            len(selected)
            != EXPECTED_PER_AUTHOR
        ):
            raise RuntimeError(
                f"{rank_key}: "
                f"{author} has "
                f"{len(selected)} rows"
            )

        values.append(
            sum(
                row.get(rank_key) == 1
                for row in selected
            )
            / len(selected)
        )

    return sum(values) / len(values)


def per_author_top1(
    rows,
    rank_key,
):
    out = {}

    for author in AUTHORS:
        selected = [
            row
            for row in rows
            if row["author"] == author
        ]

        out[author] = (
            sum(
                row.get(rank_key) == 1
                for row in selected
            )
            / len(selected)
        )

    return out


def transitions(
    rows,
    before_key,
    after_key,
):
    helped = 0
    harmed = 0
    unchanged_correct = 0
    unchanged_wrong = 0

    for row in rows:
        before = (
            row.get(before_key) == 1
        )

        after = (
            row.get(after_key) == 1
        )

        if not before and after:
            helped += 1
        elif before and not after:
            harmed += 1
        elif before and after:
            unchanged_correct += 1
        else:
            unchanged_wrong += 1

    return {
        "helped": helped,
        "harmed": harmed,
        "net": helped - harmed,
        "unchanged_correct":
            unchanged_correct,
        "unchanged_wrong":
            unchanged_wrong,
    }


def find_em1_key(
    sample,
    candidates,
):
    for key in candidates:
        if key in sample:
            return key

    return None


def find_shared_id(
    pv,
    em1,
):
    candidates = (
        "condition_id",
        "anchor_id",
        "row_id",
    )

    for pk in candidates:
        if pk not in pv[0]:
            continue

        pset = {
            str(row[pk])
            for row in pv
        }

        for ek in candidates:
            if ek not in em1[0]:
                continue

            eset = {
                str(row[ek])
                for row in em1
            }

            if (
                pset == eset
                and len(pset)
                == EXPECTED_ROWS
            ):
                return pk, ek

    raise RuntimeError(
        "Could not find exact shared ID.\n"
        f"PV keys: "
        f"{sorted(pv[0])}\n"
        f"EM1 keys: "
        f"{sorted(em1[0])}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pv-predictions",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--em1-rows",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    pv_all = read_jsonl(
        args.pv_predictions
    )

    em1_all = read_jsonl(
        args.em1_rows
    )

    pv = [
        row
        for row in pv_all
        if row.get("author")
        in AUTHORS
    ]

    em1 = [
        row
        for row in em1_all
        if row.get("author")
        in AUTHORS
    ]

    for label, rows in (
        ("PV", pv),
        ("EM1", em1),
    ):
        counts = Counter(
            row.get("author")
            for row in rows
        )

        if len(rows) != EXPECTED_ROWS:
            raise RuntimeError(
                f"{label}: "
                f"expected 3000 rows, "
                f"found {len(rows)}"
            )

        expected = Counter(
            {
                author:
                    EXPECTED_PER_AUTHOR
                for author in AUTHORS
            }
        )

        if counts != expected:
            raise RuntimeError(
                f"{label}: "
                f"author counts differ: "
                f"{dict(counts)}"
            )

    required_pv = {
        "author",
        "gold",
        "generic_rank",
        "frequency_rank",
        "pv1_rank",
        "pv2_rank",
    }

    missing = (
        required_pv
        - set(pv[0])
    )

    if missing:
        raise RuntimeError(
            "PV fields missing: "
            f"{sorted(missing)}"
        )

    pv_f_macro = macro_top1(
        pv,
        "frequency_rank",
    )

    print(
        "PV three-author "
        f"F Macro Top1 = "
        f"{pv_f_macro:.12f}"
    )

    if abs(
        pv_f_macro
        - EXPECTED_F_MACRO
    ) > 1e-9:
        raise RuntimeError(
            "Same-surface F sanity "
            "check failed."
        )

    pv_id, em_id = (
        find_shared_id(
            pv,
            em1,
        )
    )

    print(
        f"Shared ID: "
        f"PV.{pv_id} "
        f"== EM1.{em_id}"
    )

    sample = em1[0]

    ranks = sample.get("ranks")

    if not isinstance(ranks, dict):
        raise RuntimeError(
            "EM1 row does not contain "
            "a ranks object."
        )

    expected_rank_keys = {
        "G0",
        "F",
        "R",
        "R+F",
    }

    missing_rank_keys = (
        expected_rank_keys
        - set(ranks)
    )

    if missing_rank_keys:
        raise RuntimeError(
            "EM1 ranks missing keys: "
            f"{sorted(missing_rank_keys)}"
        )

    pv_by_id = {
        str(row[pv_id]):
            row
        for row in pv
    }

    em_by_id = {
        str(row[em_id]):
            row
        for row in em1
    }

    merged = []

    for shared_id in sorted(
        pv_by_id
    ):
        p = pv_by_id[shared_id]
        e = em_by_id[shared_id]

        if (
            p["author"]
            != e["author"]
        ):
            raise RuntimeError(
                "Author mismatch: "
                f"{shared_id}"
            )

        if (
            "gold" in e
            and str(p["gold"])
            != str(e["gold"])
        ):
            raise RuntimeError(
                "Gold mismatch: "
                f"{shared_id}"
            )

        merged.append(
            {
                "shared_id":
                    shared_id,
                "author":
                    p["author"],
                "gold":
                    p["gold"],

                "generic_rank":
                    p["generic_rank"],
                "frequency_rank":
                    p["frequency_rank"],
                "pv1_rank":
                    p["pv1_rank"],
                "pv2_rank":
                    p["pv2_rank"],

                "em1_g_rank":
                    e["ranks"]["G0"],
                "em1_f_rank":
                    e["ranks"]["F"],
                "em1_r_rank":
                    e["ranks"]["R"],
                "em1_rf_rank":
                    e["ranks"]["R+F"],
            }
        )

    baseline_checks = {}

    baseline_checks[
        "G_rank_exact_equal"
    ] = all(
        row["generic_rank"]
        == row["em1_g_rank"]
        for row in merged
    )

    baseline_checks[
        "F_rank_exact_equal"
    ] = all(
        row["frequency_rank"]
        == row["em1_f_rank"]
        for row in merged
    )

    methods = {
        "G":
            "generic_rank",
        "F":
            "frequency_rank",
        "PV1":
            "pv1_rank",
        "PV2":
            "pv2_rank",
        "EM1-R+F":
            "em1_rf_rank",
    }

    methods[
        "EM1-R"
    ] = "em1_r_rank"

    metrics = {}

    for name, key in methods.items():
        metrics[name] = {
            "macro_top1":
                macro_top1(
                    merged,
                    key,
                ),
            "per_author_top1":
                per_author_top1(
                    merged,
                    key,
                ),
        }

    paired = {
        "PV1_to_EM1_R+F":
            transitions(
                merged,
                "pv1_rank",
                "em1_rf_rank",
            ),

        "F_to_PV1":
            transitions(
                merged,
                "frequency_rank",
                "pv1_rank",
            ),

        "F_to_EM1_R+F":
            transitions(
                merged,
                "frequency_rank",
                "em1_rf_rank",
            ),
    }

    missing_rows = [
        row
        for row in merged
        if row["generic_rank"]
        is None
    ]

    missing_summary = {
        "rows":
            len(missing_rows),

        "pv1_top10":
            sum(
                row["pv1_rank"]
                is not None
                for row in missing_rows
            ),

        "pv1_top3":
            sum(
                row["pv1_rank"]
                is not None
                and row["pv1_rank"] <= 3
                for row in missing_rows
            ),

        "pv1_top1":
            sum(
                row["pv1_rank"] == 1
                for row in missing_rows
            ),

        "em1_rf_top10":
            sum(
                row["em1_rf_rank"]
                is not None
                for row in missing_rows
            ),

        "em1_rf_top3":
            sum(
                row["em1_rf_rank"]
                is not None
                and row["em1_rf_rank"] <= 3
                for row in missing_rows
            ),

        "em1_rf_top1":
            sum(
                row["em1_rf_rank"] == 1
                for row in missing_rows
            ),
    }

    summary = {
        "status":
            "complete",

        "audit_type":
            "post_hoc_cross_stage_explanatory_audit",

        "research_result_changed":
            False,

        "parameter_tuning":
            False,

        "new_pinyingpt_inference":
            False,

        "authors":
            list(AUTHORS),

        "rows":
            len(merged),

        "shared_id": {
            "pv": pv_id,
            "em1": em_id,
        },

        "em1_rank_fields": {
            "G": "ranks.G0",
            "F": "ranks.F",
            "R": "ranks.R",
            "R+F": "ranks.R+F",
        },

        "baseline_checks":
            baseline_checks,

        "metrics":
            metrics,

        "paired_transitions":
            paired,

        "generic_missing":
            missing_summary,

        "interpretation_boundary":
            (
                "Performance differences "
                "cannot automatically be "
                "attributed only to exact "
                "PinyinGPT scoring unless "
                "candidate construction and "
                "all other ranking semantics "
                "are also verified identical."
            ),
    }

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    with (
        args.output_root
        / "rows.jsonl"
    ).open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as destination:
        for row in merged:
            destination.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    print()
    print(
        "=== PV vs EM1 "
        "SAME-SURFACE AUDIT ==="
    )

    print(
        f"{'Method':12s}"
        f"{'MacroTop1':>12s}"
    )

    for name, values in (
        metrics.items()
    ):
        print(
            f"{name:12s}"
            f"{values['macro_top1']:12.6f}"
        )

    print()
    print(
        "=== PV1 -> EM1 R+F ==="
    )

    for key, value in (
        paired[
            "PV1_to_EM1_R+F"
        ].items()
    ):
        print(
            f"{key}: {value}"
        )

    print()
    print(
        "=== GENERIC MISSING ==="
    )

    for key, value in (
        missing_summary.items()
    ):
        print(
            f"{key}: {value}"
        )

    print()
    print(
        "=== BASELINE CHECKS ==="
    )

    if baseline_checks:
        for key, value in (
            baseline_checks.items()
        ):
            print(
                f"{key}: {value}"
            )
    else:
        print(
            "No EM1 baseline rank "
            "fields exposed."
        )

    print()
    print(
        "No tuning. "
        "No new PinyinGPT inference."
    )


if __name__ == "__main__":
    main()
