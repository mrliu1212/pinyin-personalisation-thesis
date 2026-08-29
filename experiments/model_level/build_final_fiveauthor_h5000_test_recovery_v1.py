from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import importlib.util
import json
import time
from collections import Counter, defaultdict
from pathlib import Path


AUTHORS = [
    "Re_spectators",
    "Etinjat",
    "Agent Phage",
    "QBLevi",
    "breaddddd",
]

HISTORY_LIMIT = 5000
EXPECTED_TEST_ROWS = 40_000

EXPECTED_FULL_MANIFEST_SHA256 = (
    "e05b07d5020ebf0ccf090e98b34e4c66"
    "db7dab9b565541d39feb2d6f1d1bb20b"
)


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_frozen_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        "frozen_h5000_dev_v1",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import frozen builder: {path}"
        )

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--frozen-builder",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--full-manifest",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--raw-history",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--work-split",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--summary",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--artifact-mode",
        choices=("audit", "compact"),
        default="compact",
    )

    args = ap.parse_args()
    t0 = time.time()

    dev = load_frozen_module(
        args.frozen_builder
    )

    full_sha = sha256_file(
        args.full_manifest
    )

    print(
        "FULL_MANIFEST_SHA256 =",
        full_sha,
        flush=True,
    )

    if full_sha != EXPECTED_FULL_MANIFEST_SHA256:
        raise RuntimeError(
            "Frozen full-manifest SHA mismatch"
        )

    # --------------------------------------------------------
    # Work chronology
    # --------------------------------------------------------

    work_meta = {}
    chrono_seen = defaultdict(set)

    with args.work_split.open(
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        for r in reader:
            author = str(r["author_name"])

            if author not in AUTHORS:
                continue

            wid = str(r["work_id"])
            ci = int(r["chronological_index"])

            if wid in work_meta:
                raise RuntimeError(
                    f"Duplicate work_id: {wid}"
                )

            if ci in chrono_seen[author]:
                raise RuntimeError(
                    f"Duplicate chronological index "
                    f"{author}: {ci}"
                )

            chrono_seen[author].add(ci)

            work_meta[wid] = {
                "author_name": author,
                "chronological_index": ci,
                "split": str(
                    r.get("split")
                    or r.get("work_split")
                    or ""
                ).lower(),
            }

    print(
        "FROZEN_WORKS =",
        len(work_meta),
        flush=True,
    )

    # --------------------------------------------------------
    # Test queries
    # --------------------------------------------------------

    queries = []
    author_counts = Counter()

    with args.full_manifest.open(
        encoding="utf-8",
    ) as f:

        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)

            if str(r["split"]).lower() != "test":
                continue

            author = str(r["author_name"])

            if author not in AUTHORS:
                raise RuntimeError(
                    f"Unexpected author: {author}"
                )

            wid = str(r["work_id"])

            wm = work_meta.get(wid)

            if wm is None:
                raise RuntimeError(
                    f"Test work missing chronology: {wid}"
                )

            if wm["author_name"] != author:
                raise RuntimeError(
                    f"Work/author mismatch: {wid}"
                )

            q = dict(r)

            q["_author"] = author
            q["_segments"] = dev.query_segments(r)
            q["_work_index"] = int(
                wm["chronological_index"]
            )

            queries.append(q)
            author_counts[author] += 1

    if len(queries) != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            f"Test rows={len(queries)} "
            f"!= {EXPECTED_TEST_ROWS}"
        )

    for author in AUTHORS:
        if author_counts[author] != 8000:
            raise RuntimeError(
                f"{author}: "
                f"{author_counts[author]} != 8000"
            )

    print(
        "TEST_QUERY_GATE=PASS",
        flush=True,
    )

    # --------------------------------------------------------
    # Raw history, streamed
    # --------------------------------------------------------

    raw_keyed = defaultdict(list)

    with args.raw_history.open(
        encoding="utf-8",
    ) as f:

        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)

            author = dev.author_of(r)

            if author not in AUTHORS:
                continue

            wid = str(
                r.get("work_id")
                or ""
            )

            wm = work_meta.get(wid)

            if wm is None:
                continue

            if wm["author_name"] != author:
                raise RuntimeError(
                    f"Raw/work author mismatch: {wid}"
                )

            iid = str(
                r.get("interaction_id")
                or dev.row_id(r)
            )

            if not iid:
                raise RuntimeError(
                    f"Raw interaction missing id: {wid}"
                )

            source_start = int(
                r["source_position_start"]
            )

            source_end = int(
                r["source_position_end"]
            )

            key = (
                int(wm["chronological_index"]),
                source_start,
                source_end,
                iid,
            )

            event = {
                "interaction_id": iid,
                "target": dev.target_of(r),
                "context": dev.context_of(r),
                "segments": dev.raw_segments(r),
                "work_id": wid,
                "work_chronological_index":
                    int(wm["chronological_index"]),
                "source_position_start":
                    source_start,
                "source_position_end":
                    source_end,
                "boundary_span_id":
                    r.get("boundary_span_id"),
                "work_split":
                    wm["split"],
                "_sort_key":
                    key,
            }

            raw_keyed[author].append(
                (key, event)
            )

    raw_by_author = {}
    keys_by_author = {}
    prefix_split = {}

    for author in AUTHORS:
        keyed = raw_keyed[author]

        keyed.sort(
            key=lambda x: x[0]
        )

        for i in range(1, len(keyed)):
            if keyed[i - 1][0] == keyed[i][0]:
                raise RuntimeError(
                    f"Duplicate raw sort key "
                    f"{author}: {keyed[i][0]}"
                )

        keys = [
            key for key, _event in keyed
        ]

        events = [
            event for _key, event in keyed
        ]

        raw_by_author[author] = events
        keys_by_author[author] = keys

        pref = {
            "fit": [0],
            "val": [0],
            "test": [0],
        }

        for event in events:
            split = event["work_split"]

            for s in pref:
                pref[s].append(
                    pref[s][-1]
                    + int(split == s)
                )

        prefix_split[author] = pref

    total_raw = sum(
        len(v)
        for v in raw_by_author.values()
    )

    print(
        "RAW_ELIGIBLE_FIVE_AUTHOR_ROWS =",
        total_raw,
        flush=True,
    )

    if total_raw != 904162:
        raise RuntimeError(
            f"Raw rows={total_raw} != 904162"
        )

    print(
        "RAW_TIMELINE_GATE=PASS",
        flush=True,
    )

    # --------------------------------------------------------
    # Recovery
    # --------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.summary.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed = 0
    exact_active = 0
    composition_active = 0
    queries_with_prior_test = 0

    visible_totals = Counter()

    snapshot_cache = {}

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as fout:

        for q in queries:
            author = q["_author"]

            keys = keys_by_author[author]
            raw = raw_by_author[author]

            # Frozen boundary semantics:
            # anything at the current source start or later
            # is excluded.
            query_key = (
                int(q["_work_index"]),
                int(q["source_position_start"]),
                -1,
                "",
            )

            stop = bisect.bisect_left(
                keys,
                query_key,
            )

            start = max(
                0,
                stop - HISTORY_LIMIT,
            )

            if stop - start > HISTORY_LIMIT:
                raise RuntimeError(
                    "H5000 exceeded"
                )

            # Strict-prior audit.
            if stop > 0:
                if not keys[stop - 1] < query_key:
                    raise RuntimeError(
                        f"Strict-prior failure: "
                        f"{q['row_id']}"
                    )

            if stop < len(keys):
                if keys[stop] < query_key:
                    raise RuntimeError(
                        f"Boundary failure: "
                        f"{q['row_id']}"
                    )

            pref = prefix_split[author]

            split_visible = {}

            for s in ("fit", "val", "test"):
                split_visible[s] = (
                    pref[s][stop]
                    - pref[s][start]
                )
                visible_totals[s] += (
                    split_visible[s]
                )

            if split_visible["test"] > 0:
                queries_with_prior_test += 1

            cache_key = (
                author,
                start,
                stop,
            )

            if cache_key in snapshot_cache:
                index = snapshot_cache[
                    cache_key
                ]
            else:
                visible = raw[start:stop]

                index = dev.build_window_index(
                    visible
                )

                if len(snapshot_cache) > 256:
                    snapshot_cache.clear()

                snapshot_cache[
                    cache_key
                ] = index

            parts = q["_segments"]
            qctx = dev.context_of(q)

            exact = dev.rank_exact(
                index=index,
                parts=parts,
                query_context=qctx,
            )

            comp = dev.composition_candidates(
                index=index,
                parts=parts,
                query_context=qctx,
            )

            exact_active += int(bool(exact))
            composition_active += int(bool(comp))

            if args.artifact_mode == "compact":
                exact_out = dev.compact_candidates(
                    exact
                )
                comp_out = dev.compact_candidates(
                    comp
                )
            else:
                exact_out = exact
                comp_out = comp

            if [
                x["text"]
                for x in exact_out
            ] != [
                x["text"]
                for x in exact
            ]:
                raise RuntimeError(
                    "Exact compaction identity failure"
                )

            if [
                x["text"]
                for x in comp_out
            ] != [
                x["text"]
                for x in comp
            ]:
                raise RuntimeError(
                    "Composition compaction identity failure"
                )

            out = {
                "schema_version": 1,
                "experiment":
                    "final_fiveauthor_h5000_test_recovery_v1",

                "row_id":
                    str(q["row_id"]),

                "split":
                    "test",

                "author_name":
                    author,

                "M":
                    int(q["M"]),

                "typing_mode":
                    q.get("typing_mode"),

                "effective_typing_mode":
                    q.get("effective_typing_mode"),

                "pinyin_input":
                    q.get("pinyin_input"),

                "gold":
                    dev.target_of(q),

                "history_start_ordinal":
                    start,

                "history_stop_ordinal":
                    stop,

                "history_raw_count":
                    stop - start,

                "visible_fit_raw":
                    split_visible["fit"],

                "visible_val_raw":
                    split_visible["val"],

                "visible_prior_test_raw":
                    split_visible["test"],

                "strictly_prior":
                    True,

                "same_author":
                    True,

                "exact_candidates":
                    exact_out,

                "composition_candidates":
                    comp_out,

                "artifact_mode":
                    args.artifact_mode,

                "used_test":
                    True,

                "used_current_test":
                    False,

                "used_future_test":
                    False,
            }

            fout.write(
                json.dumps(
                    out,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

            processed += 1

            if (
                processed <= 10
                or processed % 1000 == 0
            ):
                elapsed = (
                    time.time() - t0
                )

                print(
                    f"{processed}/40000 "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    if processed != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            f"Processed={processed}"
        )

    summary = {
        "schema_version": 1,
        "experiment":
            "final_fiveauthor_h5000_test_recovery_v1",

        "queries": processed,
        "history_limit": HISTORY_LIMIT,

        "exact_active_queries":
            exact_active,

        "composition_active_queries":
            composition_active,

        "queries_with_strictly_prior_test_history":
            queries_with_prior_test,

        "visible_raw_totals_by_split":
            dict(visible_totals),

        "current_test_visible":
            0,

        "future_test_visible":
            0,

        "same_author_only":
            True,

        "strictly_prior":
            True,

        "used_test_as_query":
            True,

        "output_path":
            str(args.output),

        "output_sha256":
            sha256_file(args.output),
    }

    args.summary.write_text(
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
        "TEST_H5000_QUERY_ROWS = 40000",
        flush=True,
    )
    print(
        "CURRENT_TEST_VISIBLE = 0",
        flush=True,
    )
    print(
        "FUTURE_TEST_VISIBLE = 0",
        flush=True,
    )
    print(
        "H5000_TEST_CAUSAL_GATE=PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
