#!/usr/bin/env python3
from __future__ import annotations

import bisect
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(
    "/home/3160454/work/"
    "thesis-finalmodel-sixauthor-dataset-v1"
)

FINAL_ROOT = (
    ROOT
    / "results/finalmodel_fiveauthor_v1"
)

INPUT_ROOT = (
    FINAL_ROOT
    / "adapter_input_preparation_v1"
)

DATASET_ROOT = (
    FINAL_ROOT
    / "dataset_preparation_v1"
)

OUT_ROOT = (
    FINAL_ROOT
    / "h5000_preparation_v1"
)

FIT = (
    INPUT_ROOT
    / "adapter_fit_manifest_frozen_v1.jsonl"
)

VAL = (
    INPUT_ROOT
    / "adapter_val_manifest_frozen_v1.jsonl"
)

WORK_SPLIT = (
    DATASET_ROOT
    / "work_split_manifest_frozen_v1.csv"
)

FULL_MANIFEST = (
    DATASET_ROOT
    / "final_fiveauthor_manifest_frozen_v1.jsonl"
)

RAW = Path(
    "/home/3160454/work/"
    "final-sixauthor-source-v1/"
    "data/processed/deep_author/"
    "interactions_t1_ready.jsonl"
)

BOUNDARIES_OUT = (
    OUT_ROOT
    / "h5000_query_boundaries_frozen_v1.jsonl"
)

SUMMARY_OUT = (
    OUT_ROOT
    / "h5000_query_boundaries_summary_v1.json"
)

AUTHORS = (
    "Re_spectators",
    "Etinjat",
    "Agent Phage",
    "QBLevi",
    "breaddddd",
)

EXPECTED_FIT = 150_000
EXPECTED_VAL = 20_000
EXPECTED_RAW = 904_162

HISTORY_BUDGET = 5000


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# ------------------------------------------------------------
# Frozen work chronology
# ------------------------------------------------------------

work_meta = {}

with WORK_SPLIT.open(
    encoding="utf-8-sig",
    newline="",
) as f:

    for r in csv.DictReader(f):

        work_id = str(r["work_id"])

        if work_id in work_meta:
            raise RuntimeError(
                f"duplicate work: {work_id}"
            )

        work_meta[work_id] = {
            "author":
                str(r["author_name"]),
            "split":
                str(r["split"]).lower(),
            "creation_date":
                str(r["creation_date"]),
            "chronological_index":
                int(r["chronological_index"]),
        }


if len(work_meta) != 248:
    raise RuntimeError(
        f"expected 248 works, "
        f"got {len(work_meta)}"
    )


# Check chronological_index uniqueness per author.
for author in AUTHORS:

    values = [
        m["chronological_index"]
        for m in work_meta.values()
        if m["author"] == author
    ]

    if len(values) != len(set(values)):
        raise RuntimeError(
            f"{author}: duplicate chronological_index"
        )


print("WORK_CHRONOLOGY_GATE=PASS")


# ------------------------------------------------------------
# Build raw timeline keys.
#
# Only final five-author eligible works are admitted.
# This prevents excluded/ineligible works from entering memory.
# ------------------------------------------------------------

raw_keys = {
    author: []
    for author in AUTHORS
}

raw_counts = Counter()

for r in read_jsonl(RAW):

    author = str(r["author_name"])
    work_id = str(r["work_id"])

    if author not in AUTHORS:
        continue

    meta = work_meta.get(work_id)

    if meta is None:
        # Not one of the frozen 248 eligible works.
        continue

    if meta["author"] != author:
        raise RuntimeError(
            f"raw/work author mismatch: {work_id}"
        )

    key = (
        int(meta["chronological_index"]),
        int(r["source_position_start"]),
        int(r["source_position_end"]),
        str(r["interaction_id"]),
    )

    raw_keys[author].append(key)
    raw_counts[author] += 1


for author in AUTHORS:
    raw_keys[author].sort()


raw_total = sum(raw_counts.values())

print("RAW_ELIGIBLE_FIVE_AUTHOR_ROWS =", raw_total)

for author in AUTHORS:
    print(
        author,
        "=",
        raw_counts[author],
    )


# We expect the historical 904,162 population to already
# correspond to the frozen final five-author eligible works.
if raw_total != EXPECTED_RAW:
    raise RuntimeError(
        f"eligible raw population changed: "
        f"{raw_total} != {EXPECTED_RAW}"
    )


print("RAW_TIMELINE_GATE=PASS")


# ------------------------------------------------------------
# Recover full frozen query metadata.
#
# Adapter Fit/Val manifests intentionally omit source positions.
# The frozen 210k manifest is used only as a row_id metadata map.
# Test rows are ignored and never added to the query population.
# ------------------------------------------------------------

full_meta = {}

for r in read_jsonl(FULL_MANIFEST):

    rid = str(r["row_id"])
    split = str(r["split"]).lower()

    if split not in {"fit", "val", "test"}:
        raise RuntimeError(
            f"unexpected full-manifest split: {split}"
        )

    # Test metadata exists in the already-frozen dataset,
    # but Test is not part of H5000 development.
    # Do not retain Test rows as query metadata.
    if split == "test":
        continue

    if rid in full_meta:
        raise RuntimeError(
            f"duplicate row_id in Fit/Val full manifest: {rid}"
        )

    full_meta[rid] = {
        "split":
            split,
        "author_name":
            str(r["author_name"]),
        "work_id":
            str(r["work_id"]),
        "source_creation_date":
            str(r["source_creation_date"]),
        "source_position_start":
            int(r["source_position_start"]),
        "source_position_end":
            int(r["source_position_end"]),
        "boundary_span_id":
            int(r["boundary_span_id"]),
    }


if len(full_meta) != 170_000:
    raise RuntimeError(
        f"expected 170000 Fit+Val metadata rows, "
        f"got {len(full_meta)}"
    )


print("FULL_MANIFEST_METADATA_GATE=PASS")


# ------------------------------------------------------------
# Query population
# ------------------------------------------------------------

queries = []

for expected_split, path in (
    ("fit", FIT),
    ("val", VAL),
):

    count = 0

    for r in read_jsonl(path):

        if str(r["split"]).lower() != expected_split:
            raise RuntimeError(
                f"wrong split in {path}: "
                f"{r['split']}"
            )

        author = str(r["author_name"])
        work_id = str(r["work_id"])

        if author not in AUTHORS:
            raise RuntimeError(
                f"unexpected author: {author}"
            )

        meta = work_meta.get(work_id)

        if meta is None:
            raise RuntimeError(
                f"query work not frozen: {work_id}"
            )

        if meta["author"] != author:
            raise RuntimeError(
                f"query author/work mismatch: "
                f"{r['row_id']}"
            )

        if meta["split"] != expected_split:
            raise RuntimeError(
                f"query/work split mismatch: "
                f"{r['row_id']}"
            )

        rid = str(r["row_id"])

        fm = full_meta.get(rid)

        if fm is None:
            raise RuntimeError(
                f"query row missing from full frozen manifest: {rid}"
            )

        if fm["split"] != expected_split:
            raise RuntimeError(
                f"full-manifest split mismatch: {rid}"
            )

        if fm["author_name"] != author:
            raise RuntimeError(
                f"full-manifest author mismatch: {rid}"
            )

        if fm["work_id"] != work_id:
            raise RuntimeError(
                f"full-manifest work mismatch: {rid}"
            )

        q = dict(r)

        q["source_position_start"] = int(
            fm["source_position_start"]
        )

        q["source_position_end"] = int(
            fm["source_position_end"]
        )

        q["boundary_span_id"] = int(
            fm["boundary_span_id"]
        )

        q["_work_chronological_index"] = int(
            meta["chronological_index"]
        )

        queries.append(q)
        count += 1

    expected = (
        EXPECTED_FIT
        if expected_split == "fit"
        else EXPECTED_VAL
    )

    if count != expected:
        raise RuntimeError(
            f"{expected_split}: "
            f"{count} != {expected}"
        )


print("QUERY_POPULATION_GATE=PASS")


# ------------------------------------------------------------
# Compute strictly-prior raw boundary.
#
# We deliberately use the earliest possible raw key at the
# query's source start:
#
# (work_index, source_start, -1, "")
#
# Therefore every raw interaction beginning at the SAME source
# position as the query is excluded from its history.
#
# This is conservative and prevents the query target itself,
# or another generated span sharing its starting anchor,
# from leaking into personal memory.
# ------------------------------------------------------------

stats = defaultdict(Counter)

with BOUNDARIES_OUT.open(
    "w",
    encoding="utf-8",
) as out:

    for number, q in enumerate(
        queries,
        start=1,
    ):

        author = str(q["author_name"])
        split = str(q["split"]).lower()

        source_start = q.get(
            "source_position_start"
        )

        if source_start is None:
            raise RuntimeError(
                f"source_position_start recovery failed: "
                f"{q['row_id']}"
            )

        work_index = int(
            q["_work_chronological_index"]
        )

        boundary_key = (
            work_index,
            int(source_start),
            -1,
            "",
        )

        keys = raw_keys[author]

        stop = bisect.bisect_left(
            keys,
            boundary_key,
        )

        start = max(
            0,
            stop - HISTORY_BUDGET,
        )

        visible = stop - start

        if visible < 0:
            raise RuntimeError(
                "negative visible count"
            )

        if visible > HISTORY_BUDGET:
            raise RuntimeError(
                "H5000 overflow"
            )

        row = {
            "row_id":
                str(q["row_id"]),
            "author_name":
                author,
            "split":
                split,
            "work_id":
                str(q["work_id"]),
            "work_chronological_index":
                work_index,
            "source_position_start":
                int(source_start),
            "raw_prior_count":
                int(stop),
            "history_start_ordinal":
                int(start),
            "history_stop_ordinal":
                int(stop),
            "visible_raw_history_count":
                int(visible),
            "history_budget":
                HISTORY_BUDGET,
            "strictly_prior":
                True,
            "same_author":
                True,
            "test_used":
                False,
        }

        out.write(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )

        stats[
            (split, author)
        ]["rows"] += 1

        stats[
            (split, author)
        ]["visible_sum"] += visible

        if visible == HISTORY_BUDGET:
            stats[
                (split, author)
            ]["full_h5000"] += 1

        if visible == 0:
            stats[
                (split, author)
            ]["zero_history"] += 1

        if number % 20_000 == 0:
            print(
                "BOUNDARIES",
                number,
                "/",
                len(queries),
            )


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

summary_stats = {}

for split in ("fit", "val"):

    summary_stats[split] = {}

    for author in AUTHORS:

        x = stats[
            (split, author)
        ]

        n = int(x["rows"])

        summary_stats[
            split
        ][author] = {
            "rows":
                n,
            "mean_visible_raw_history":
                (
                    x["visible_sum"] / n
                    if n
                    else 0.0
                ),
            "full_h5000_rows":
                int(x["full_h5000"]),
            "zero_history_rows":
                int(x["zero_history"]),
        }


summary = {
    "schema_version": 1,
    "experiment":
        "final_fiveauthor_h5000_query_boundaries_v1",

    "status":
        "FROZEN_BOUNDARIES_READY",

    "history_policy":
        (
            "same author -> strictly prior -> "
            "latest H5000 raw interactions"
        ),

    "history_budget":
        HISTORY_BUDGET,

    "chronology": {
        "work_order":
            "frozen work_split chronological_index",

        "within_work_order":
            (
                "source_position_start -> "
                "source_position_end -> "
                "interaction_id"
            ),

        "query_boundary":
            (
                "exclude all raw interactions "
                "starting at the query source position"
            ),
    },

    "population": {
        "raw_five_author_rows":
            raw_total,
        "fit_queries":
            EXPECTED_FIT,
        "val_queries":
            EXPECTED_VAL,
        "test_queries":
            0,
    },

    "per_split_author":
        summary_stats,

    "inputs": {
        "fit_sha256":
            sha256(FIT),
        "val_sha256":
            sha256(VAL),
        "work_split_sha256":
            sha256(WORK_SPLIT),
        "full_manifest_sha256":
            sha256(FULL_MANIFEST),
        "raw_path":
            str(RAW),
        "raw_size_bytes":
            RAW.stat().st_size,
    },

    "artifact": {
        "boundaries_path":
            str(BOUNDARIES_OUT),
        "boundaries_sha256":
            sha256(BOUNDARIES_OUT),
    },

    "test_status":
        "CLOSED",
}


with SUMMARY_OUT.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

    f.write("\n")


print()
print("===== H5000 BOUNDARY SUMMARY =====")

for split in ("fit", "val"):

    for author in AUTHORS:

        x = summary_stats[
            split
        ][author]

        print(
            split,
            author,
            x,
        )


print()
print(
    "BOUNDARIES_SHA256 =",
    sha256(BOUNDARIES_OUT),
)

print(
    "SUMMARY_SHA256 =",
    sha256(SUMMARY_OUT),
)

print("TEST_ROWS_USED_AS_QUERIES=0")
print("TEST_MODEL_EVALUATION=FALSE")
print("H5000_QUERY_BOUNDARIES=PASS")
