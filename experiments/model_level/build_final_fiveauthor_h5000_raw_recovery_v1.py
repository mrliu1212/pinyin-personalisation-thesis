from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path


# ================================================================
# Frozen final-study configuration
# ================================================================

HISTORY_LIMIT = 5000
TAU = 2048.0
MAX_N = 2

SPAN_TOPK = 5
COMPOSITION_BEAM = 300

EPS = 1e-12

EXPECTED_BOUNDARY_SHA256 = (
    "5e7f472b82853e01ff432d5189bd32ade"
    "488e2ffcc0eda324e48a6119afb5b51"
)

EXPECTED_FIT_SHA256 = (
    "789d0e47e18120d1a9c5136c4f185d2"
    "ceb7c760b00f2c9796d4d356032cd4739"
)

EXPECTED_VAL_SHA256 = (
    "5c00277aaa989026520584614f02a32500"
    "545e38fdd0fb8f9bf2517a121cfdbe"
)

AUTHORS = {
    "Re_spectators",
    "Etinjat",
    "Agent Phage",
    "QBLevi",
    "breaddddd",
}


# ================================================================
# Utilities
# ================================================================

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise RuntimeError(
                    f"Bad JSON at {path}:{line_no}"
                ) from e


def row_id(r):
    return str(
        r.get("row_id")
        or r.get("interaction_id")
        or r.get("raw_row_id")
        or ""
    )


def author_of(r):
    return str(
        r.get("author_name")
        or r.get("author")
        or r.get("user_id")
        or ""
    )


def target_of(r):
    x = (
        r.get("gold")
        or r.get("target")
        or r.get("selected_target")
    )
    if x is None:
        raise RuntimeError(
            f"Missing target: {row_id(r)}"
        )
    return str(x)


def context_of(r):
    x = r.get("context")
    if x is None:
        return ""
    return str(x)


def split_segments(x):
    if x is None:
        return ()

    if isinstance(x, list):
        return tuple(
            str(z).strip().lower()
            for z in x
            if str(z).strip()
        )

    return tuple(
        z.strip().lower()
        for z in str(x).split()
        if z.strip()
    )


def query_segments(r):
    # Preserve the actual frozen typing representation.
    for key in (
        "pinyin_segments",
        "segmented_pinyin",
        "pinyin_input",
        "typed_pinyin",
    ):
        if r.get(key):
            parts = split_segments(r[key])
            if parts:
                return parts

    raise RuntimeError(
        f"No query pinyin segments: {row_id(r)}"
    )


def raw_segments(r):
    # Historical memory stores canonical full Pinyin.
    for key in (
        "pinyin_segments",
        "segmented_pinyin",
        "full_pinyin",
        "pinyin_input",
    ):
        if r.get(key):
            parts = split_segments(r[key])
            if parts:
                return parts

    raise RuntimeError(
        f"No raw pinyin segments: {row_id(r)}"
    )


def recency_weight(age: int) -> float:
    return math.exp(-float(age) / TAU)


def suffix_match(a: str, b: str, n: int) -> bool:
    if n <= 0:
        return True

    # Historical implementation used sequence suffix matching.
    # Context here is character text, so suffix length is character based.
    return (
        len(a) >= n
        and len(b) >= n
        and a[-n:] == b[-n:]
    )


# ================================================================
# History representation
# ================================================================

def build_window_index(events):
    """
    events: visible causal H5000 raw events, oldest -> newest.

    Return:
      exact pinyin tuple -> [(event, age), ...]
    where age=0 is newest.
    """

    n = len(events)
    idx = defaultdict(list)

    for pos, event in enumerate(events):
        age = n - 1 - pos
        idx[event["segments"]].append(
            (event, age)
        )

    return idx


# ================================================================
# Exact deterministic recovery
# ================================================================

def pinyin_segment_compatible(
    query_segment: str,
    full_segment: str,
) -> bool:
    """Match frozen typed Pinyin against canonical Full Pinyin."""

    q = str(query_segment).strip().lower()
    f = str(full_segment).strip().lower()

    if not q or not f:
        return False

    return (
        q == f
        or (
            len(q) == 1
            and q == f[0]
        )
    )


def pinyin_sequence_compatible(
    query_parts,
    full_parts,
) -> bool:

    q = tuple(map(str, query_parts))
    f = tuple(map(str, full_parts))

    if len(q) != len(f):
        return False

    return all(
        pinyin_segment_compatible(qi, fi)
        for qi, fi in zip(q, f)
    )


def compatible_history(
    index,
    parts,
):
    """Find canonical Full-Pinyin history compatible with typed query."""

    query = tuple(parts)

    result = list(
        index.get(query, [])
    )

    seen = {
        event["interaction_id"]
        for event, _age in result
    }

    for full_parts, items in index.items():

        if full_parts == query:
            continue

        if len(full_parts) != len(query):
            continue

        if not pinyin_sequence_compatible(
            query,
            full_parts,
        ):
            continue

        for event, age in items:
            iid = event["interaction_id"]

            if iid in seen:
                continue

            seen.add(iid)

            result.append(
                (event, age)
            )

    return result


def rank_exact(
    index,
    parts,
    query_context,
):
    exact = compatible_history(
        index,
        parts,
    )

    if not exact:
        return []

    candidates = sorted({
        event["target"]
        for event, _age in exact
    })

    matched = exact
    effective_n = 0

    for n in range(MAX_N, 0, -1):
        current = [
            (event, age)
            for event, age in exact
            if suffix_match(
                query_context,
                event["context"],
                n,
            )
        ]

        if current:
            matched = current
            effective_n = n
            break

    scores = {
        text: 0.0
        for text in candidates
    }

    for event, age in matched:
        scores[event["target"]] += (
            recency_weight(age)
        )

    total = sum(scores.values())

    if total > 0:
        scores = {
            k: v / total
            for k, v in scores.items()
        }

    last_age = {
        text: 10**18
        for text in candidates
    }

    for event, age in exact:
        t = event["target"]
        last_age[t] = min(
            last_age[t],
            age,
        )

    ranked = sorted(
        candidates,
        key=lambda text: (
            -scores[text],
            last_age[text],
            text,
        ),
    )

    return [
        {
            "text": text,

            # Final-v1 deterministic branch score.
            "exact_retrieval_score":
                float(scores[text]),

            "exact_effective_n":
                int(effective_n),

            "exact_last_seen_age":
                int(last_age[text]),
        }
        for text in ranked
    ]


# ================================================================
# Composition deterministic recovery
# ================================================================

def composition_candidates(
    index,
    parts,
    query_context,
):
    n = len(parts)

    if n < 2:
        return []

    spans = {}

    for start in range(n):
        for end in range(
            start + 1,
            n + 1,
        ):
            # Proper spans only.
            if start == 0 and end == n:
                continue

            ranked = rank_exact(
                index=index,
                parts=parts[start:end],
                query_context=query_context,
            )

            if ranked:
                spans[(start, end)] = (
                    ranked[:SPAN_TOPK]
                )

    states = {
        0: [{
            "text": "",
            "log_score": 0.0,
            "pieces": [],
        }]
    }

    for pos in range(n):

        current = sorted(
            states.get(pos, []),
            key=lambda s: (
                -(
                    s["log_score"]
                    / max(pos, 1)
                ),
                len(s["pieces"]),
                s["text"],
            ),
        )[:COMPOSITION_BEAM]

        if not current:
            continue

        for state in current:

            for end in range(
                pos + 1,
                n + 1,
            ):
                edge = spans.get(
                    (pos, end)
                )

                if not edge:
                    continue

                span_len = end - pos

                for c in edge:
                    score = max(
                        float(
                            c[
                                "exact_retrieval_score"
                            ]
                        ),
                        EPS,
                    )

                    states.setdefault(
                        end,
                        [],
                    ).append({
                        "text":
                            state["text"]
                            + c["text"],

                        "log_score":
                            state["log_score"]
                            + span_len
                            * math.log(score),

                        "pieces":
                            state["pieces"]
                            + [{
                                "start": pos,
                                "end": end,
                                "text": c["text"],
                                "score": score,
                            }],
                    })

    best = {}

    for state in states.get(n, []):

        pieces = state["pieces"]

        if len(pieces) < 2:
            continue

        text = state["text"]

        raw = (
            state["log_score"]
            / max(n, 1)
        )

        spans_len = [
            p["end"] - p["start"]
            for p in pieces
        ]

        cand = {
            "text": text,

            # IMPORTANT:
            # This is deterministic composition raw evidence.
            # It is NOT historical R3 C1 score and is not yet
            # a calibrated probability.
            "composition_retrieval_log_score":
                float(raw),

            "piece_count":
                int(len(pieces)),

            "fragmentation_ratio":
                float(
                    len(pieces)
                    / max(n, 1)
                ),

            "longest_span_ratio":
                float(
                    max(spans_len)
                    / max(n, 1)
                ),

            "pieces": pieces,
        }

        old = best.get(text)

        if (
            old is None
            or cand["composition_retrieval_log_score"]
            > old["composition_retrieval_log_score"]
        ):
            best[text] = cand

    return sorted(
        best.values(),
        key=lambda x: (
            -x["composition_retrieval_log_score"],
            x["piece_count"],
            x["text"],
        ),
    )


# ================================================================
# Main
# ================================================================


_COMPACT_DROP = object()


def compact_json_value(value):
    """Preserve cheap scalar evidence; drop nested debug structures."""

    if (
        value is None
        or isinstance(
            value,
            (str, int, float, bool),
        )
    ):
        return value

    if isinstance(value, (list, tuple)):
        if all(
            (
                x is None
                or isinstance(
                    x,
                    (str, int, float, bool),
                )
            )
            for x in value
        ):
            return list(value)

    return _COMPACT_DROP


def compact_candidate(candidate):
    """Preserve candidate text/order and every scalar evidence field."""

    if "text" not in candidate:
        raise RuntimeError(
            "Candidate missing text"
        )

    out = {}

    for key, value in candidate.items():

        compacted = compact_json_value(
            value
        )

        if compacted is _COMPACT_DROP:
            continue

        out[key] = compacted

    if out.get("text") != candidate["text"]:
        raise RuntimeError(
            "Compaction changed candidate text"
        )

    return out


def compact_candidates(candidates):
    return [
        compact_candidate(c)
        for c in candidates
    ]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--fit",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--val",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--boundaries",
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
        help=(
            "Frozen work split manifest containing "
            "author_name, work_id, chronological_index."
        ),
    )
    ap.add_argument(
        "--full-manifest",
        type=Path,
        required=True,
        help=(
            "Frozen full 210k manifest used only to recover "
            "Fit/Val source positions and boundary_span_id."
        ),
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
        "--limit",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--selected-row-ids",
        type=Path,
        default=None,
        help=(
            "Optional JSON file containing an exact list "
            "of row_ids to evaluate."
        ),
    )

    ap.add_argument(
        "--artifact-mode",
        choices=("audit", "compact"),
        default="audit",
        help=(
            "audit keeps full nested recovery evidence; "
            "compact keeps candidate identity/order and "
            "scalar evidence while dropping nested debug "
            "structures such as composition piece paths."
        ),
    )

    args = ap.parse_args()

    t0 = time.time()

    # ------------------------------------------------------------
    # SHA gates
    # ------------------------------------------------------------

    print("===== SHA GATES =====")

    bsha = sha256(args.boundaries)
    fsha = sha256(args.fit)
    vsha = sha256(args.val)

    print("BOUNDARY_SHA256 =", bsha)
    print("FIT_SHA256 =", fsha)
    print("VAL_SHA256 =", vsha)

    if bsha != EXPECTED_BOUNDARY_SHA256:
        raise RuntimeError(
            "Boundary SHA mismatch."
        )

    if fsha != EXPECTED_FIT_SHA256:
        raise RuntimeError(
            "Fit SHA mismatch."
        )

    if vsha != EXPECTED_VAL_SHA256:
        raise RuntimeError(
            "Val SHA mismatch."
        )

    # ------------------------------------------------------------
    # Query rows
    # ------------------------------------------------------------

    print()
    print("===== LOAD FROZEN FIT / VAL METADATA =====")

    full_meta = {}

    for r in load_jsonl(args.full_manifest):
        split = str(r["split"]).lower()

        if split not in {"fit", "val", "test"}:
            raise RuntimeError(
                f"Unexpected full-manifest split: {split}"
            )

        # Test metadata already exists in the frozen dataset,
        # but Test is not part of H5000 development.
        if split == "test":
            continue

        rid = str(r["row_id"])

        if rid in full_meta:
            raise RuntimeError(
                f"Duplicate Fit/Val row_id in full manifest: {rid}"
            )

        full_meta[rid] = {
            "split": split,
            "author_name": str(r["author_name"]),
            "work_id": str(r["work_id"]),
            "source_position_start":
                int(r["source_position_start"]),
            "source_position_end":
                int(r["source_position_end"]),
            "boundary_span_id":
                int(r["boundary_span_id"]),
        }

    if len(full_meta) != 170000:
        raise RuntimeError(
            f"Expected 170000 Fit+Val metadata rows, "
            f"got {len(full_meta)}"
        )

    print(
        "FULL_MANIFEST_FITVAL_METADATA =",
        len(full_meta),
    )
    print(
        "FULL_MANIFEST_METADATA_GATE=PASS"
    )

    print()
    print("===== LOAD FIT / VAL QUERIES =====")

    query = {}

    split_counts = defaultdict(int)

    for split, path in (
        ("fit", args.fit),
        ("val", args.val),
    ):
        for r in load_jsonl(path):
            rid = row_id(r)

            if not rid:
                raise RuntimeError(
                    "Query without row_id"
                )

            if rid in query:
                raise RuntimeError(
                    f"Duplicate query row_id: {rid}"
                )

            a = author_of(r)

            if a not in AUTHORS:
                raise RuntimeError(
                    f"Unexpected author: {a}"
                )

            fm = full_meta.get(rid)

            if fm is None:
                raise RuntimeError(
                    f"Query missing from full frozen manifest: {rid}"
                )

            if fm["split"] != split:
                raise RuntimeError(
                    f"Full-manifest split mismatch: {rid}"
                )

            if fm["author_name"] != a:
                raise RuntimeError(
                    f"Full-manifest author mismatch: {rid}"
                )

            if fm["work_id"] != str(r["work_id"]):
                raise RuntimeError(
                    f"Full-manifest work mismatch: {rid}"
                )

            q = dict(r)

            q["source_position_start"] = (
                fm["source_position_start"]
            )
            q["source_position_end"] = (
                fm["source_position_end"]
            )
            q["boundary_span_id"] = (
                fm["boundary_span_id"]
            )

            q["_split"] = split
            q["_author"] = a
            q["_segments"] = query_segments(r)

            query[rid] = q
            split_counts[split] += 1

    print(
        "FIT_QUERIES =",
        split_counts["fit"],
    )
    print(
        "VAL_QUERIES =",
        split_counts["val"],
    )

    if split_counts["fit"] != 150000:
        raise RuntimeError(
            "Unexpected Fit count"
        )

    if split_counts["val"] != 20000:
        raise RuntimeError(
            "Unexpected Val count"
        )

    # ------------------------------------------------------------
    # Boundary artifact
    # ------------------------------------------------------------

    print()
    print("===== LOAD H5000 BOUNDARIES =====")

    boundary = {}

    max_stop_by_author = defaultdict(int)

    for r in load_jsonl(args.boundaries):
        rid = str(r["row_id"])

        if rid not in query:
            raise RuntimeError(
                "Boundary contains non-Fit/Val query: "
                + rid
            )

        if rid in boundary:
            raise RuntimeError(
                f"Duplicate boundary: {rid}"
            )

        boundary[rid] = r

        a = str(r["author_name"])

        stop = int(
            r.get("history_stop_ordinal")
            or r.get("raw_history_stop")
            or r.get("visible_stop")
            or r.get("stop_ordinal")
            or 0
        )

        max_stop_by_author[a] = max(
            max_stop_by_author[a],
            stop,
        )

    if len(boundary) != 170000:
        raise RuntimeError(
            "Expected 170000 boundaries, got "
            f"{len(boundary)}"
        )

    missing = set(query) - set(boundary)

    if missing:
        raise RuntimeError(
            f"Missing boundaries: {len(missing)}"
        )

    print(
        "BOUNDARY_ROWS =",
        len(boundary),
    )

    # ------------------------------------------------------------
    # Raw history
    # ------------------------------------------------------------

    print()
    print("===== LOAD FROZEN WORK CHRONOLOGY =====")

    work_meta = {}
    chronological_seen = defaultdict(set)

    with args.work_split.open(
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        required = {
            "author_name",
            "work_id",
            "chronological_index",
        }

        missing = required - set(
            reader.fieldnames or []
        )

        if missing:
            raise RuntimeError(
                "Work-split CSV missing columns: "
                + ", ".join(sorted(missing))
            )

        for r in reader:
            a = str(
                r.get("author_name")
                or r.get("author")
                or ""
            )

            if a not in AUTHORS:
                continue

            wid = str(r["work_id"])

            if wid in work_meta:
                raise RuntimeError(
                    f"Duplicate work_id in work split: {wid}"
                )

            ci = int(r["chronological_index"])

            if ci in chronological_seen[a]:
                raise RuntimeError(
                    f"Duplicate chronological_index "
                    f"for {a}: {ci}"
                )

            chronological_seen[a].add(ci)

            work_meta[wid] = {
                "author_name": a,
                "chronological_index": ci,
                "split": str(
                    r.get("split")
                    or r.get("work_split")
                    or ""
                ).lower(),
            }

    if not work_meta:
        raise RuntimeError(
            "No five-author frozen works loaded"
        )

    print(
        "FROZEN_FIVE_AUTHOR_WORKS =",
        len(work_meta),
    )

    print()
    print("===== LOAD RAW FIVE-AUTHOR HISTORY =====")

    # Store (authoritative_sort_key, event), then sort exactly
    # as the frozen boundary builder did:
    #
    # (
    #   work chronological_index,
    #   source_position_start,
    #   source_position_end,
    #   interaction_id
    # )
    #
    # This is critical because history_start_ordinal /
    # history_stop_ordinal refer to this exact ordering.
    raw_keyed_by_author = defaultdict(list)
    raw_counts = defaultdict(int)

    for r in load_jsonl(args.raw_history):

        a = author_of(r)

        if a not in AUTHORS:
            continue

        wid = str(
            r.get("work_id")
            or ""
        )

        meta = work_meta.get(wid)

        # Match the frozen boundary builder:
        # raw interactions outside the frozen eligible works
        # are not part of the timeline.
        if meta is None:
            continue

        if meta["author_name"] != a:
            raise RuntimeError(
                f"Raw/work author mismatch: {wid}"
            )

        interaction_id = str(
            r.get("interaction_id")
            or row_id(r)
        )

        if not interaction_id:
            raise RuntimeError(
                f"Raw interaction without interaction_id: "
                f"{wid}"
            )

        source_start = int(
            r["source_position_start"]
        )
        source_end = int(
            r["source_position_end"]
        )

        key = (
            int(meta["chronological_index"]),
            source_start,
            source_end,
            interaction_id,
        )

        event = {
            "interaction_id":
                interaction_id,

            "target":
                target_of(r),

            "context":
                context_of(r),

            "segments":
                raw_segments(r),

            "work_id":
                wid,

            "work_chronological_index":
                int(meta["chronological_index"]),

            "source_position_start":
                source_start,

            "source_position_end":
                source_end,

            "boundary_span_id":
                r.get("boundary_span_id"),

            "_sort_key":
                key,
        }

        raw_keyed_by_author[a].append(
            (key, event)
        )

        raw_counts[a] += 1


    raw_by_author = {}

    for a in AUTHORS:

        keyed = raw_keyed_by_author[a]

        keyed.sort(
            key=lambda x: x[0]
        )

        # Defensive uniqueness check. If two raw interactions had
        # the exact same full ordering key, the boundary ordinal
        # would not be well-defined.
        for i in range(1, len(keyed)):
            if keyed[i - 1][0] == keyed[i][0]:
                raise RuntimeError(
                    f"Duplicate raw sort key for {a}: "
                    f"{keyed[i][0]}"
                )

        raw_by_author[a] = [
            event
            for _key, event in keyed
        ]


    total_raw = sum(
        len(v)
        for v in raw_by_author.values()
    )

    print(
        "RAW_ELIGIBLE_FIVE_AUTHOR_ROWS =",
        total_raw,
    )

    for a in sorted(AUTHORS):
        print(
            "RAW",
            a,
            len(raw_by_author[a]),
        )

    if total_raw != 904162:
        raise RuntimeError(
            "Expected 904162 frozen eligible "
            "five-author raw rows, got "
            f"{total_raw}"
        )

    print("RAW_TIMELINE_GATE=PASS")


    # ------------------------------------------------------------
    # Boundary ordinal consistency audit
    # ------------------------------------------------------------

    print()
    print("===== BOUNDARY ORDINAL CONSISTENCY =====")

    checked = 0

    for rid, b in boundary.items():

        a = str(b["author_name"])
        start = int(
            b["history_start_ordinal"]
        )
        stop = int(
            b["history_stop_ordinal"]
        )

        visible = int(
            b["visible_raw_history_count"]
        )
        raw_prior = int(
            b["raw_prior_count"]
        )

        if stop - start != visible:
            raise RuntimeError(
                f"Visible count mismatch {rid}: "
                f"{start}:{stop} -> {stop-start}, "
                f"artifact={visible}"
            )

        expected_start = max(
            0,
            raw_prior - HISTORY_LIMIT,
        )

        if start != expected_start:
            raise RuntimeError(
                f"H5000 start mismatch {rid}: "
                f"{start} != {expected_start}"
            )

        if stop != raw_prior:
            raise RuntimeError(
                f"Raw prior stop mismatch {rid}: "
                f"{stop} != {raw_prior}"
            )

        if not bool(b["strictly_prior"]):
            raise RuntimeError(
                f"Boundary not strictly prior: {rid}"
            )

        if not bool(b["same_author"]):
            raise RuntimeError(
                f"Boundary not same-author: {rid}"
            )

        if bool(b.get("test_used", False)):
            raise RuntimeError(
                f"Test boundary unexpectedly used: {rid}"
            )

        if not (
            0 <= start <= stop
            <= len(raw_by_author[a])
        ):
            raise RuntimeError(
                f"Boundary outside raw timeline "
                f"{rid}: {start}:{stop}/"
                f"{len(raw_by_author[a])}"
            )

        checked += 1

    if checked != 170000:
        raise RuntimeError(
            f"Boundary audit count changed: "
            f"{checked} != 170000"
        )

    print(
        "BOUNDARY_ORDINAL_ROWS_CHECKED =",
        checked,
    )
    print(
        "BOUNDARY_ORDINAL_CONSISTENCY=PASS"
    )



    # ------------------------------------------------------------
    # Process
    # ------------------------------------------------------------

    selected_row_ids = None

    if args.selected_row_ids is not None:
        payload = json.loads(
            args.selected_row_ids.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(payload, list):
            raise RuntimeError(
                "--selected-row-ids must contain a JSON list"
            )

        selected_row_ids = {
            str(x)
            for x in payload
        }

        if len(selected_row_ids) != len(payload):
            raise RuntimeError(
                "Duplicate selected row_ids"
            )

        missing = (
            selected_row_ids
            - set(query)
        )

        if missing:
            raise RuntimeError(
                "Selected row_ids missing from Fit/Val: "
                f"{list(sorted(missing))[:10]}"
            )

        print(
            "SELECTED_ROW_IDS =",
            len(selected_row_ids),
        )

    print()
    print("===== BUILD RAW RECOVERY =====")

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = 0
    exact_active = 0
    composition_active = 0
    exact_candidates_total = 0
    composition_candidates_total = 0

    by_split = defaultdict(
        lambda: defaultdict(int)
    )

    snapshot_cache = {}

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as fout:

        for rid, q in query.items():

            if (
                selected_row_ids is not None
                and rid not in selected_row_ids
            ):
                continue

            if (
                args.limit is not None
                and total >= args.limit
            ):
                break

            b = boundary[rid]
            a = q["_author"]
            split = q["_split"]

            # Prefer explicit start ordinal from the frozen
            # boundary artifact.
            start = b.get(
                "history_start_ordinal"
            )

            stop = b.get(
                "history_stop_ordinal"
            )

            if start is None or stop is None:
                # Allow alternative names only if present.
                start = b.get(
                    "raw_history_start"
                )
                stop = b.get(
                    "raw_history_stop"
                )

            if start is None or stop is None:
                raise RuntimeError(
                    "Boundary artifact does not expose "
                    "history_start_ordinal/"
                    "history_stop_ordinal for "
                    f"{rid}. Inspect schema before full run."
                )

            start = int(start)
            stop = int(stop)

            if not (
                0 <= start <= stop
                <= len(raw_by_author[a])
            ):
                raise RuntimeError(
                    f"Bad window {rid}: "
                    f"{start}:{stop} / "
                    f"{len(raw_by_author[a])}"
                )

            if stop - start > HISTORY_LIMIT:
                raise RuntimeError(
                    f"H5000 exceeded: {rid}"
                )

            cache_key = (
                a,
                start,
                stop,
            )

            if cache_key in snapshot_cache:
                index = snapshot_cache[
                    cache_key
                ]
            else:
                visible = (
                    raw_by_author[a][
                        start:stop
                    ]
                )

                index = build_window_index(
                    visible
                )

                # Small bounded cache.
                if len(snapshot_cache) > 256:
                    snapshot_cache.clear()

                snapshot_cache[
                    cache_key
                ] = index

            parts = q["_segments"]
            qctx = context_of(q)

            exact = rank_exact(
                index=index,
                parts=parts,
                query_context=qctx,
            )

            comp = composition_candidates(
                index=index,
                parts=parts,
                query_context=qctx,
            )

            if exact:
                exact_active += 1
                by_split[split][
                    "exact_active"
                ] += 1

            if comp:
                composition_active += 1
                by_split[split][
                    "composition_active"
                ] += 1

            exact_candidates_total += len(
                exact
            )

            composition_candidates_total += (
                len(comp)
            )

            by_split[split][
                "queries"
            ] += 1

            by_split[split][
                "exact_candidates"
            ] += len(exact)

            by_split[split][
                "composition_candidates"
            ] += len(comp)

            family_key = [
                a,
                str(q.get("work_id") or ""),
                int(q["boundary_span_id"]),
                int(
                    q["source_position_start"]
                ),
            ]

            if args.artifact_mode == "compact":
                exact_out = compact_candidates(
                    exact
                )
                comp_out = compact_candidates(
                    comp
                )
            else:
                exact_out = exact
                comp_out = comp

            # Compaction must never change candidate identity
            # or ordering.
            if [
                x["text"]
                for x in exact_out
            ] != [
                x["text"]
                for x in exact
            ]:
                raise RuntimeError(
                    f"Exact compaction changed candidates: {rid}"
                )

            if [
                x["text"]
                for x in comp_out
            ] != [
                x["text"]
                for x in comp
            ]:
                raise RuntimeError(
                    f"Composition compaction changed candidates: {rid}"
                )

            out = {
                "schema_version": 1,

                "row_id": rid,
                "split": split,
                "author_name": a,

                "family_key":
                    family_key,

                "M": int(q["M"]),

                "typing_mode":
                    q.get("typing_mode"),

                "effective_typing_mode":
                    q.get(
                        "effective_typing_mode"
                    ),

                "pinyin_input":
                    q.get("pinyin_input"),

                "gold":
                    target_of(q),

                "history_start_ordinal":
                    start,

                "history_stop_ordinal":
                    stop,

                "history_raw_count":
                    stop - start,

                "exact_candidates":
                    exact_out,

                "composition_candidates":
                    comp_out,

                "artifact_mode":
                    args.artifact_mode,

                # Test guard.
                "used_test": False,
            }

            fout.write(
                json.dumps(
                    out,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

            total += 1

            if (
                total <= 10
                or total % 1000 == 0
            ):
                elapsed = (
                    time.time() - t0
                )

                print(
                    f"{total}/"
                    f"{min(len(query), args.limit or len(query))}"
                    f" elapsed={elapsed:.1f}s"
                    f" exact_active={exact_active}"
                    f" comp_active={composition_active}",
                    flush=True,
                )

    # ------------------------------------------------------------
    # Summary / SHA
    # ------------------------------------------------------------

    output_sha = sha256(args.output)

    summary = {
        "schema_version": 1,

        "status": "complete",

        "artifact_mode":
            args.artifact_mode,

        "compact_policy": (
            "Candidate text, ordering, and scalar evidence are "
            "preserved; nested debug structures are omitted."
            if args.artifact_mode == "compact"
            else "Complete audit evidence retained."
        ),

        "scientific_role":
            "final five-author deterministic "
            "H5000 retrieval evidence",

        "historical_difference": {
            "strict_old_r3_r4_branch_scorer_reproduced":
                False,

            "reason":
                "Historical R3/R4 branch scorer "
                "producer and OOF input artifacts "
                "were not retained in the current "
                "workspace; final v1 therefore uses "
                "fully auditable deterministic H5000 "
                "retrieval evidence. These values are "
                "not historical R5 branch raw scores "
                "and must not be described as such.",
        },

        "constants": {
            "history_limit":
                HISTORY_LIMIT,

            "tau":
                TAU,

            "max_context_suffix_n":
                MAX_N,

            "span_topk":
                SPAN_TOPK,

            "composition_beam":
                COMPOSITION_BEAM,
        },

        "queries_written":
            total,

        "exact_active_queries":
            exact_active,

        "composition_active_queries":
            composition_active,

        "exact_candidate_total":
            exact_candidates_total,

        "composition_candidate_total":
            composition_candidates_total,

        "by_split": {
            k: dict(v)
            for k, v in by_split.items()
        },

        "hashes": {
            "fit":
                fsha,

            "val":
                vsha,

            "boundaries":
                bsha,

            "output":
                output_sha,
        },

        "test_rows_used_as_queries":
            0,

        "test_model_evaluation":
            False,

        "runtime_seconds":
            time.time() - t0,
    }

    args.summary.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    print("===== DONE =====")
    print(
        "OUTPUT =",
        args.output,
    )
    print(
        "OUTPUT_SHA256 =",
        output_sha,
    )
    print(
        "SUMMARY =",
        args.summary,
    )
    print(
        "TEST_ROWS_USED_AS_QUERIES = 0"
    )
    print(
        "TEST_MODEL_EVALUATION = FALSE"
    )


if __name__ == "__main__":
    main()
