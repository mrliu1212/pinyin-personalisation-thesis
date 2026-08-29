from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(
    "/home/3160454/work/"
    "thesis-finalmodel-sixauthor-dataset-v1"
)

WORK_SPLIT = (
    ROOT
    / "results/finalmodel_fiveauthor_v1/"
      "dataset_preparation_v1/"
      "work_split_manifest_frozen_v1.csv"
)

FULL_MANIFEST = (
    ROOT
    / "results/finalmodel_fiveauthor_v1/"
      "dataset_preparation_v1/"
      "final_fiveauthor_manifest_frozen_v1.jsonl"
)

BOUNDARIES = (
    ROOT
    / "results/finalmodel_fiveauthor_v1/"
      "h5000_preparation_v1/"
      "h5000_query_boundaries_frozen_v1.jsonl"
)

RAW = Path(
    "/home/3160454/work/"
    "final-sixauthor-source-v1/"
    "data/processed/deep_author/"
    "interactions_t1_ready.jsonl"
)

OUTDIR = (
    ROOT
    / "results/finalmodel_fiveauthor_v1/"
      "h5000_preparation_v1"
)

SUMMARY = (
    OUTDIR
    / "h5000_fitval_continuity_audit_v1.json"
)

ROWS_OUT = (
    OUTDIR
    / "h5000_val_continuity_rows_v1.jsonl"
)

AUTHORS = {
    "Re_spectators",
    "Etinjat",
    "Agent Phage",
    "QBLevi",
    "breaddddd",
}

EXPECTED_BOUNDARY_SHA256 = (
    "5e7f472b82853e01ff432d5189bd32ade"
    "488e2ffcc0eda324e48a6119afb5b51"
)


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


# ------------------------------------------------------------
# SHA gate
# ------------------------------------------------------------

boundary_sha = sha256(BOUNDARIES)

print(
    "BOUNDARY_SHA256 =",
    boundary_sha,
)

if boundary_sha != EXPECTED_BOUNDARY_SHA256:
    raise RuntimeError(
        "Frozen H5000 boundary SHA changed"
    )


# ------------------------------------------------------------
# Frozen work metadata
# ------------------------------------------------------------

work_meta = {}

with WORK_SPLIT.open(
    encoding="utf-8",
    newline="",
) as f:

    for r in csv.DictReader(f):

        author = str(r["author_name"])

        if author not in AUTHORS:
            continue

        wid = str(r["work_id"])

        work_meta[wid] = {
            "author_name": author,
            "chronological_index":
                int(r["chronological_index"]),
            "split":
                str(r["split"]).lower(),
        }


print(
    "FROZEN_WORKS =",
    len(work_meta),
)


# ------------------------------------------------------------
# Query metadata — Fit/Val only.
# Test rows are skipped.
# ------------------------------------------------------------

query_meta = {}

for r in read_jsonl(FULL_MANIFEST):

    split = str(r["split"]).lower()

    if split == "test":
        continue

    if split not in {"fit", "val"}:
        raise RuntimeError(
            f"Unexpected split: {split}"
        )

    rid = str(r["row_id"])

    query_meta[rid] = {
        "split": split,
        "author_name":
            str(r["author_name"]),
        "work_id":
            str(r["work_id"]),
    }


if len(query_meta) != 170000:
    raise RuntimeError(
        f"Expected 170000 Fit+Val queries, "
        f"got {len(query_meta)}"
    )


# ------------------------------------------------------------
# Reconstruct exact raw timeline used by boundary builder.
# ------------------------------------------------------------

raw_keyed = defaultdict(list)

for r in read_jsonl(RAW):

    author = str(r["author_name"])

    if author not in AUTHORS:
        continue

    wid = str(r["work_id"])

    wm = work_meta.get(wid)

    if wm is None:
        continue

    if wm["author_name"] != author:
        raise RuntimeError(
            f"Author/work mismatch: {wid}"
        )

    key = (
        int(wm["chronological_index"]),
        int(r["source_position_start"]),
        int(r["source_position_end"]),
        str(r["interaction_id"]),
    )

    raw_keyed[author].append(
        (
            key,
            wm["split"],
        )
    )


# ------------------------------------------------------------
# Build split prefix sums for O(1) window audits.
# ------------------------------------------------------------

prefix = {}
raw_counts = {}

for author in sorted(AUTHORS):

    xs = sorted(
        raw_keyed[author],
        key=lambda x: x[0],
    )

    raw_counts[author] = len(xs)

    pf = [0]
    pv = [0]
    pt = [0]

    for _key, split in xs:

        pf.append(
            pf[-1]
            + int(split == "fit")
        )

        pv.append(
            pv[-1]
            + int(split == "val")
        )

        pt.append(
            pt[-1]
            + int(split == "test")
        )

    prefix[author] = {
        "fit": pf,
        "val": pv,
        "test": pt,
    }


print(
    "RAW_TOTAL =",
    sum(raw_counts.values()),
)

for author in sorted(AUTHORS):
    print(
        author,
        raw_counts[author],
    )


# ------------------------------------------------------------
# Audit frozen boundaries.
# ------------------------------------------------------------

totals = Counter()

val_by_author = defaultdict(Counter)

val_examples = []

ROWS_OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with ROWS_OUT.open(
    "w",
    encoding="utf-8",
) as fout:

    for b in read_jsonl(BOUNDARIES):

        rid = str(b["row_id"])

        q = query_meta.get(rid)

        if q is None:
            raise RuntimeError(
                f"Boundary row not Fit/Val: {rid}"
            )

        author = str(
            b["author_name"]
        )

        if author != q["author_name"]:
            raise RuntimeError(
                f"Boundary/query author mismatch: "
                f"{rid}"
            )

        start = int(
            b["history_start_ordinal"]
        )

        stop = int(
            b["history_stop_ordinal"]
        )

        if not bool(
            b["strictly_prior"]
        ):
            raise RuntimeError(
                f"Not strictly prior: {rid}"
            )

        if bool(
            b.get("test_used", False)
        ):
            raise RuntimeError(
                f"Frozen boundary marks Test used: "
                f"{rid}"
            )

        p = prefix[author]

        fit_n = (
            p["fit"][stop]
            - p["fit"][start]
        )

        val_n = (
            p["val"][stop]
            - p["val"][start]
        )

        test_n = (
            p["test"][stop]
            - p["test"][start]
        )

        split = q["split"]

        totals[
            f"{split}_queries"
        ] += 1

        totals[
            f"{split}_history_fit"
        ] += fit_n

        totals[
            f"{split}_history_val"
        ] += val_n

        totals[
            f"{split}_history_test"
        ] += test_n

        if test_n != 0:
            raise RuntimeError(
                f"TEST history visible to {rid}: "
                f"{test_n}"
            )

        # Fit queries must never see Val or Test.
        if split == "fit":
            if val_n != 0:
                raise RuntimeError(
                    f"Fit query sees Val history: "
                    f"{rid}, n={val_n}"
                )

        # Val is allowed to see prior Fit + prior Val.
        if split == "val":

            a = val_by_author[author]

            a["queries"] += 1

            if fit_n > 0:
                a[
                    "queries_with_fit_history"
                ] += 1

            if val_n > 0:
                a[
                    "queries_with_prior_val_history"
                ] += 1

            if (
                fit_n > 0
                and val_n > 0
            ):
                a[
                    "queries_with_fit_and_val_history"
                ] += 1

            a[
                "visible_fit_events"
            ] += fit_n

            a[
                "visible_prior_val_events"
            ] += val_n

            out = {
                "row_id": rid,
                "author_name": author,
                "query_split": split,
                "work_id": q["work_id"],
                "history_start_ordinal":
                    start,
                "history_stop_ordinal":
                    stop,
                "history_count":
                    stop - start,
                "visible_fit_history":
                    fit_n,
                "visible_prior_val_history":
                    val_n,
                "visible_test_history":
                    test_n,
                "strictly_prior": True,
            }

            fout.write(
                json.dumps(
                    out,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

            if (
                len(val_examples) < 20
                and fit_n > 0
            ):
                val_examples.append(out)


# ------------------------------------------------------------
# Required scientific gates.
# ------------------------------------------------------------

if totals["fit_queries"] != 150000:
    raise RuntimeError(
        "Fit query count changed"
    )

if totals["val_queries"] != 20000:
    raise RuntimeError(
        "Val query count changed"
    )

if totals["fit_history_val"] != 0:
    raise RuntimeError(
        "Fit history contains Val events"
    )

if totals["fit_history_test"] != 0:
    raise RuntimeError(
        "Fit history contains Test events"
    )

if totals["val_history_test"] != 0:
    raise RuntimeError(
        "Val history contains Test events"
    )


val_with_fit = sum(
    x["queries_with_fit_history"]
    for x in val_by_author.values()
)

if val_with_fit == 0:
    raise RuntimeError(
        "No Val queries can see Fit history; "
        "Fit→Val continuity appears broken"
    )


rows_sha = sha256(ROWS_OUT)

summary = {
    "schema_version": 1,

    "status": "PASS",

    "scientific_policy": (
        "H5000 memory is continuous across the "
        "chronological Fit-to-Val transition. "
        "Val queries may use same-author strictly-prior "
        "Fit history and strictly-prior Val history. "
        "Val supervision is not used to train Adapter, "
        "Platt calibrators, or LambdaMART. "
        "Test history is never visible."
    ),

    "boundary_sha256":
        boundary_sha,

    "query_counts": {
        "fit":
            totals["fit_queries"],
        "val":
            totals["val_queries"],
    },

    "history_event_totals": {
        "fit_queries_visible_fit":
            totals["fit_history_fit"],

        "fit_queries_visible_val":
            totals["fit_history_val"],

        "fit_queries_visible_test":
            totals["fit_history_test"],

        "val_queries_visible_fit":
            totals["val_history_fit"],

        "val_queries_visible_prior_val":
            totals["val_history_val"],

        "val_queries_visible_test":
            totals["val_history_test"],
    },

    "val_by_author": {
        a: dict(v)
        for a, v in sorted(
            val_by_author.items()
        )
    },

    "val_queries_with_fit_history":
        val_with_fit,

    "examples":
        val_examples,

    "rows_output":
        str(ROWS_OUT),

    "rows_sha256":
        rows_sha,

    "test_rows_used_as_queries":
        0,

    "test_model_evaluation":
        False,
}


SUMMARY.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


print()
print(
    "FIT_QUERIES =",
    totals["fit_queries"],
)

print(
    "VAL_QUERIES =",
    totals["val_queries"],
)

print(
    "FIT_VISIBLE_VAL_HISTORY =",
    totals["fit_history_val"],
)

print(
    "FIT_VISIBLE_TEST_HISTORY =",
    totals["fit_history_test"],
)

print(
    "VAL_VISIBLE_FIT_HISTORY =",
    totals["val_history_fit"],
)

print(
    "VAL_VISIBLE_PRIOR_VAL_HISTORY =",
    totals["val_history_val"],
)

print(
    "VAL_VISIBLE_TEST_HISTORY =",
    totals["val_history_test"],
)

print(
    "VAL_QUERIES_WITH_FIT_HISTORY =",
    val_with_fit,
)

print()
print(
    "ROWS_SHA256 =",
    rows_sha,
)

print(
    "SUMMARY =",
    SUMMARY,
)

print()
print(
    "FIT_VAL_H5000_CONTINUITY_AUDIT=PASS"
)

print(
    "TEST_ROWS_USED_AS_QUERIES=0"
)

print(
    "TEST_MODEL_EVALUATION=FALSE"
)
