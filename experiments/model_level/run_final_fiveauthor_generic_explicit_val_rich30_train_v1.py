from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np


AUTHORS = [
    "Re_spectators",
    "Etinjat",
    "Agent Phage",
    "QBLevi",
    "breaddddd",
]

PERSONAL_BUDGET = 5
NGRAM_MAX_N = 2
NGRAM_TAU = 2048.0

BASE_FEATURES = [
    "has_adapter",
    "adapter_rank",
    "adapter_reciprocal_rank",
    "adapter_score",
    "adapter_mean_score",
    "adapter_gap_to_top",
    "is_personal",
    "personal_rank",
    "has_exact",
    "exact_raw_score",
    "p_exact",
    "has_composition",
    "composition_raw_score",
    "p_composition",
    "recovery_confidence",
    "composition_piece_count",
    "composition_fragmentation_ratio",
    "composition_longest_span_ratio",
    "adapter_personal_overlap",
    "multi_token_length",
    "pool_size",
]

RICH_FEATURES = [
    "history_frequency_count",
    "history_choice_share",
    "history_entropy_concentration",
    "history_log1p_same_pinyin_count",
    "history_log1p_raw_count",
    "history_ngram_support",
    "history_ngram_effective_n",
    "history_log1p_ngram_matched_count",
    "history_ngram_gap_to_top",
]

FEATURES = BASE_FEATURES + RICH_FEATURES

PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "max_depth": 5,
    "num_leaves": 31,
    "min_data_in_leaf": 500,
    "learning_rate": 0.05,
    "lambda_l2": 0.0,
    "verbosity": -1,
    "seed": 1729,
    "feature_fraction_seed": 1729,
    "bagging_seed": 1729,
    "data_random_seed": 1729,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 4,
}

ROUNDS = 100


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe(x):
    if x is None:
        return 0.0
    x = float(x)
    if not math.isfinite(x):
        raise RuntimeError(f"Non-finite feature: {x}")
    return x


def normalize(values, uniform_if_zero=True):
    clipped = [
        max(0.0, float(x))
        for x in values
    ]
    total = sum(clipped)
    if total <= 0.0:
        if uniform_if_zero and clipped:
            return [1.0 / len(clipped)] * len(clipped)
        return [0.0] * len(clipped)
    return [x / total for x in clipped]


def suffix_match(a: str, b: str, n: int) -> bool:
    return (
        n <= 0
        or (
            len(a) >= n
            and len(b) >= n
            and a[-n:] == b[-n:]
        )
    )


def recency_weight(age: int) -> float:
    return math.exp(-float(age) / NGRAM_TAU)


def entropy_concentration(counts):
    positive = [
        int(v)
        for v in counts.values()
        if int(v) > 0
    ]
    total = sum(positive)
    distinct = len(positive)

    if total <= 0 or distinct == 0:
        return 0.0
    if distinct == 1:
        return 1.0

    shares = [
        x / total
        for x in positive
    ]
    entropy = -sum(
        p * math.log(p)
        for p in shares
        if p > 0.0
    )
    entropy_norm = (
        entropy
        / math.log(distinct)
    )
    return max(
        0.0,
        min(
            1.0,
            1.0 - entropy_norm,
        ),
    )


def pinyin_segment_compatible(query, full):
    query = str(query).lower()
    full = str(full).lower()

    return (
        query == full
        or (
            len(query) == 1
            and bool(full)
            and query == full[0]
        )
    )


def pinyin_sequence_compatible(query, full):
    if len(query) != len(full):
        return False
    return all(
        pinyin_segment_compatible(q, f)
        for q, f in zip(query, full)
    )



def map_generic_predictions(path: Path):
    rows = load_jsonl(path)

    if len(rows) != 20000:
        raise RuntimeError(
            f"Generic Val rows={len(rows)} != 20000"
        )

    out = {}

    for r in rows:
        rid = str(r["row_id"])

        if rid in out:
            raise RuntimeError(
                f"Duplicate Generic row_id: {rid}"
            )

        if str(r["split"]).lower() != "val":
            raise RuntimeError(
                f"{rid}: non-Val row in Generic predictions"
            )

        if bool(r.get("used_adapter", False)):
            raise RuntimeError(
                f"{rid}: Generic prediction used adapter"
            )

        if bool(r.get("used_h5000", False)):
            raise RuntimeError(
                f"{rid}: Generic prediction used H5000"
            )

        if bool(r.get("used_rich30", False)):
            raise RuntimeError(
                f"{rid}: Generic prediction used Rich30"
            )

        out[rid] = r

    return out


def map_adapter_predictions(root: Path):
    out = {}

    for author in AUTHORS:
        safe_name = (
            author.lower()
            .replace(" ", "_")
        )

        path = (
            root
            / f"{safe_name}_val_top10_v1.jsonl"
        )

        rows = load_jsonl(path)

        if len(rows) != 4000:
            raise RuntimeError(
                f"{author}: adapter rows "
                f"{len(rows)} != 4000"
            )

        for r in rows:
            rid = str(r["row_id"])

            if rid in out:
                raise RuntimeError(
                    f"Duplicate adapter row: {rid}"
                )

            if str(r["split"]).lower() != "val":
                raise RuntimeError(
                    "Non-Val adapter row"
                )

            if r.get("used_test"):
                raise RuntimeError(
                    "Adapter artifact used Test"
                )

            out[rid] = r

    if len(out) != 20_000:
        raise RuntimeError(
            f"Adapter total={len(out)}"
        )

    return out


def merge_personal(row):
    merged = {}

    for x in row.get(
        "exact_candidates",
        [],
    ):
        text = str(x["text"])

        score = float(
            x["exact_retrieval_score"]
        )

        merged[text] = {
            "text": text,
            "exact_raw_score":
                score,
            "p_exact":
                score,
            "composition_raw_score":
                None,
            "p_composition":
                None,
            "piece_count":
                None,
            "fragmentation_ratio":
                None,
            "longest_span_ratio":
                None,
            "recovery_confidence":
                score,
        }

    for x in row.get(
        "composition_candidates",
        [],
    ):
        text = str(x["text"])

        raw = float(
            x[
                "composition_retrieval_log_score"
            ]
        )

        p_comp = math.exp(
            max(
                -50.0,
                min(0.0, raw),
            )
        )

        if text not in merged:
            merged[text] = {
                "text":
                    text,
                "exact_raw_score":
                    None,
                "p_exact":
                    None,
                "composition_raw_score":
                    raw,
                "p_composition":
                    p_comp,
                "piece_count":
                    int(x["piece_count"]),
                "fragmentation_ratio":
                    float(
                        x[
                            "fragmentation_ratio"
                        ]
                    ),
                "longest_span_ratio":
                    float(
                        x[
                            "longest_span_ratio"
                        ]
                    ),
                "recovery_confidence":
                    p_comp,
            }

        else:
            y = merged[text]

            y[
                "composition_raw_score"
            ] = raw

            y[
                "p_composition"
            ] = p_comp

            y[
                "piece_count"
            ] = int(
                x["piece_count"]
            )

            y[
                "fragmentation_ratio"
            ] = float(
                x[
                    "fragmentation_ratio"
                ]
            )

            y[
                "longest_span_ratio"
            ] = float(
                x[
                    "longest_span_ratio"
                ]
            )

            y[
                "recovery_confidence"
            ] = max(
                float(
                    y["p_exact"]
                    or 0.0
                ),
                p_comp,
            )

    ranked = sorted(
        merged.values(),
        key=lambda x: (
            -x[
                "recovery_confidence"
            ],
            x["text"],
        ),
    )

    for rank, x in enumerate(
        ranked,
        start=1,
    ):
        x[
            "personal_rank_raw"
        ] = rank

    return ranked


def load_raw_history(
    raw_path: Path,
    work_order,
):
    """Rebuild the exact frozen H5000 raw author timelines.

    Ordering and raw-Pinyin semantics mirror
    build_final_fiveauthor_h5000_raw_recovery_v1.py.
    """

    grouped = {
        author: []
        for author in AUTHORS
    }

    def split_segments(value):
        if value is None:
            return ()

        if isinstance(value, list):
            return tuple(
                str(x).strip().lower()
                for x in value
                if str(x).strip()
            )

        return tuple(
            x.strip().lower()
            for x in str(value).split()
            if x.strip()
        )

    def raw_segments(row):
        # Historical memory stores canonical Full Pinyin.
        for key in (
            "pinyin_segments",
            "segmented_pinyin",
            "full_pinyin",
            "pinyin_input",
        ):
            if row.get(key):
                parts = split_segments(
                    row[key]
                )

                if parts:
                    return parts

        raise RuntimeError(
            "Raw interaction has no "
            "canonical Pinyin segments: "
            f"{row.get('interaction_id')}"
        )

    def raw_target(row):
        # Match target_of() semantics needed by
        # the current raw history asset.
        value = row.get("target")

        if value is None:
            value = row.get("gold")

        if value is None:
            raise RuntimeError(
                "Raw interaction has no target/gold: "
                f"{row.get('interaction_id')}"
            )

        return str(value)

    with raw_path.open(
        encoding="utf-8"
    ) as f:

        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)

            author = str(
                r.get("author_name")
                or r.get("author")
                or ""
            )

            if author not in grouped:
                continue

            work_id = str(
                r.get("work_id")
                or ""
            )

            work_key = (
                author,
                work_id,
            )

            # Match frozen boundary/H5000 builder:
            # interactions outside frozen eligible
            # author/work pairs are excluded.
            if work_key not in work_order:
                continue

            interaction_id = str(
                r.get("interaction_id")
                or r.get("row_id")
                or ""
            )

            if not interaction_id:
                raise RuntimeError(
                    "Raw interaction missing "
                    f"interaction_id: {work_id}"
                )

            source_start = int(
                r[
                    "source_position_start"
                ]
            )

            source_end = int(
                r[
                    "source_position_end"
                ]
            )

            work_index = int(
                work_order[
                    work_key
                ]
            )

            grouped[
                author
            ].append({
                "work_index":
                    work_index,

                "source_start":
                    source_start,

                "source_end":
                    source_end,

                "interaction_id":
                    interaction_id,

                "target":
                    raw_target(r),

                "context":
                    str(
                        r.get("context")
                        or ""
                    ),

                # Keep this field name because the
                # Rich30 compatibility code expects it.
                "pinyin":
                    raw_segments(r),

                "work_id":
                    work_id,
            })

    # Exact authoritative raw-timeline ordering.
    for author in AUTHORS:
        grouped[author].sort(
            key=lambda x: (
                x["work_index"],
                x["source_start"],
                x["source_end"],
                x["interaction_id"],
            )
        )

    return grouped


def load_work_order(path: Path):
    import csv

    out = {}

    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        for r in reader:
            author = str(
                r["author_name"]
            )
            work_id = str(
                r["work_id"]
            )
            out[
                (author, work_id)
            ] = int(
                r[
                    "chronological_index"
                ]
            )

    return out


def history_features_for_query(
    h5000_row,
    full_meta,
    pool,
    raw_by_author,
):
    """Build the frozen Rich30 H5000 history features.

    Important semantics:
    - the raw visible window is taken DIRECTLY from the frozen
      history_start_ordinal/history_stop_ordinal carried by the
      H5000 retrieval artifact;
    - recency age is measured on the same-author RAW interaction
      stream, not on the compatible/same-Pinyin substream;
    - Full / Initial / Mixed compatibility is applied only after
      the raw H5000 window and raw ages are fixed.
    """

    rid = str(
        h5000_row["row_id"]
    )

    meta = full_meta[rid]

    author = str(
        meta["author_name"]
    )

    if (
        str(h5000_row["author_name"])
        != author
    ):
        raise RuntimeError(
            f"{rid}: H5000/full-manifest "
            "author mismatch"
        )

    if (
        str(h5000_row["split"]).lower()
        != "val"
    ):
        raise RuntimeError(
            f"{rid}: non-Val H5000 row"
        )

    query_parts = tuple(
        str(
            meta["pinyin_input"]
        ).split()
    )

    query_context = str(
        meta.get("context")
        or ""
    )

    start = int(
        h5000_row[
            "history_start_ordinal"
        ]
    )

    stop = int(
        h5000_row[
            "history_stop_ordinal"
        ]
    )

    if start < 0 or stop < start:
        raise RuntimeError(
            f"{rid}: invalid frozen "
            f"H5000 window [{start}, {stop})"
        )

    raw = raw_by_author[
        author
    ]

    if stop > len(raw):
        raise RuntimeError(
            f"{rid}: H5000 stop ordinal "
            f"{stop} > raw author "
            f"timeline {len(raw)}"
        )

    visible_raw = raw[
        start:stop
    ]

    raw_count = (
        stop - start
    )

    if len(visible_raw) != raw_count:
        raise RuntimeError(
            f"{rid}: frozen raw-window "
            "length mismatch"
        )

    expected_raw_count = (
        h5000_row.get(
            "history_raw_count"
        )
    )

    if (
        expected_raw_count
        is not None
        and int(expected_raw_count)
        != raw_count
    ):
        raise RuntimeError(
            f"{rid}: H5000 history_raw_count "
            f"{expected_raw_count} != "
            f"{raw_count}"
        )

    # Preserve RAW-stream age before filtering by
    # query-Pinyin compatibility.
    #
    # global raw ordinal = start + local_ordinal
    # age = stop - 1 - global_raw_ordinal
    raw_with_age = [
        (
            event,
            stop
            - 1
            - (
                start
                + local_ordinal
            ),
        )
        for local_ordinal, event
        in enumerate(
            visible_raw
        )
    ]

    # Final-v1 extension:
    # canonical Full history is visible to
    # Full / Initial / Mixed query forms.
    visible = [
        (
            event,
            age,
        )
        for event, age
        in raw_with_age
        if pinyin_sequence_compatible(
            query_parts,
            event["pinyin"],
        )
    ]

    counts = Counter(
        event["target"]
        for event, _age
        in visible
    )

    same_count = len(
        visible
    )

    entropy = (
        entropy_concentration(
            counts
        )
    )

    candidate_set = set(
        pool
    )

    candidate_history = [
        (
            event,
            age,
        )
        for event, age
        in visible
        if event["target"]
        in candidate_set
    ]

    effective_n = 0

    matched = (
        candidate_history
    )

    for order in range(
        min(
            NGRAM_MAX_N,
            len(query_context),
        ),
        0,
        -1,
    ):
        current = [
            (
                event,
                age,
            )
            for event, age
            in candidate_history
            if suffix_match(
                query_context,
                event["context"],
                order,
            )
        ]

        if current:
            effective_n = order
            matched = current
            break

    raw_scores = {
        candidate: 0.0
        for candidate
        in pool
    }

    for event, age in matched:
        raw_scores[
            event["target"]
        ] += recency_weight(
            age
        )

    values = normalize(
        [
            raw_scores[
                candidate
            ]
            for candidate
            in pool
        ],
        uniform_if_zero=True,
    )

    ngram = dict(
        zip(
            pool,
            values,
        )
    )

    top_ngram = max(
        ngram.values(),
        default=0.0,
    )

    total = sum(
        counts.values()
    )

    result = {}

    for text in pool:
        freq = int(
            counts.get(
                text,
                0,
            )
        )

        choice = (
            freq / total
            if total
            else 0.0
        )

        ng = float(
            ngram.get(
                text,
                0.0,
            )
        )

        result[text] = {
            "history_frequency_count":
                float(freq),

            "history_choice_share":
                float(choice),

            "history_entropy_concentration":
                float(entropy),

            "history_log1p_same_pinyin_count":
                math.log1p(
                    same_count
                ),

            "history_log1p_raw_count":
                math.log1p(
                    raw_count
                ),

            "history_ngram_support":
                ng,

            "history_ngram_effective_n":
                float(
                    effective_n
                ),

            "history_log1p_ngram_matched_count":
                math.log1p(
                    len(matched)
                ),

            "history_ngram_gap_to_top":
                float(
                    top_ngram
                    - ng
                ),
        }

    return result


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--generic-predictions",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--h5000",
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
        "--output-root",
        type=Path,
        required=True,
    )

    args = ap.parse_args()

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    adapter = (
        map_generic_predictions(
            args.generic_predictions
        )
    )

    h5000_rows = load_jsonl(
        args.h5000
    )

    if len(h5000_rows) != 20_000:
        raise RuntimeError(
            f"H5000 rows="
            f"{len(h5000_rows)} "
            "!= 20000"
        )

    h5000 = {
        str(r["row_id"]): r
        for r in h5000_rows
    }

    if len(h5000) != 20_000:
        raise RuntimeError(
            "Duplicate H5000 row ids"
        )

    # Recover full frozen metadata only for Val rows.
    full_meta = {}

    with args.full_manifest.open(
        encoding="utf-8"
    ) as f:
        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)

            if (
                str(
                    r["split"]
                ).lower()
                != "val"
            ):
                continue

            rid = str(
                r["row_id"]
            )

            full_meta[rid] = r

    if len(full_meta) != 20_000:
        raise RuntimeError(
            f"Full Val metadata="
            f"{len(full_meta)}"
        )

    if (
        set(adapter)
        != set(h5000)
        or set(adapter)
        != set(full_meta)
    ):
        raise RuntimeError(
            "Val row-id universe mismatch"
        )

    work_order = load_work_order(
        args.work_split
    )

    print(
        "===== LOAD RAW HISTORY =====",
        flush=True,
    )

    raw_by_author = (
        load_raw_history(
            args.raw_history,
            work_order,
        )
    )

    print(
        "RAW_HISTORY_READY=PASS",
        flush=True,
    )

    X = []
    y = []
    groups = []

    positive_queries = 0
    total_candidates = 0

    surface_path = (
        args.output_root
        / "generic_explicit_val_rich30_training_surface_v1.jsonl"
    )

    with surface_path.open(
        "w",
        encoding="utf-8",
    ) as surface:

        for number, rid in enumerate(
            sorted(adapter),
            start=1,
        ):
            a = adapter[rid]
            h = h5000[rid]

            personal = (
                merge_personal(
                    h
                )
            )

            adapter_texts = list(
                map(
                    str,
                    a[
                        "top10_candidates"
                    ],
                )
            )[:10]

            adapter_scores = list(
                map(
                    float,
                    a[
                        "top10_candidate_scores"
                    ],
                )
            )[:10]

            adapter_map = {
                text: {
                    "rank": rank,
                    "score": score,
                }
                for rank, (
                    text,
                    score,
                )
                in enumerate(
                    zip(
                        adapter_texts,
                        adapter_scores,
                    ),
                    start=1,
                )
            }

            personal_map = {
                x["text"]: x
                for x in personal
            }

            personal_only = [
                x for x in personal
                if x["text"]
                not in adapter_map
            ]

            selected = (
                personal_only[
                    :PERSONAL_BUDGET
                ]
            )

            pool = list(
                adapter_texts
            )

            for x in selected:
                if x["text"] not in pool:
                    pool.append(
                        x["text"]
                    )

            if len(pool) > 15:
                raise RuntimeError(
                    f"{rid}: "
                    f"pool={len(pool)}"
                )

            top_score = (
                adapter_scores[0]
                if adapter_scores
                else 0.0
            )

            pool_size = len(pool)

            history_ev = (
                history_features_for_query(
                    h,
                    full_meta,
                    pool,
                    raw_by_author,
                )
            )

            features = []

            for text in pool:
                ad = (
                    adapter_map.get(
                        text
                    )
                )

                pe = (
                    personal_map.get(
                        text
                    )
                )

                has_adapter = (
                    ad is not None
                )

                is_personal = (
                    pe is not None
                )

                if has_adapter:
                    arank = int(
                        ad["rank"]
                    )
                    ascore = float(
                        ad["score"]
                    )
                    arr = (
                        1.0
                        / arank
                    )
                    amean = (
                        ascore
                        / max(
                            1,
                            len(text),
                        )
                    )
                    agap = (
                        top_score
                        - ascore
                    )
                else:
                    arank = 0
                    ascore = 0.0
                    arr = 0.0
                    amean = 0.0
                    agap = 0.0

                if is_personal:
                    prank = int(
                        pe[
                            "personal_rank_raw"
                        ]
                    )
                    has_exact = (
                        pe[
                            "p_exact"
                        ]
                        is not None
                    )
                    has_comp = (
                        pe[
                            "p_composition"
                        ]
                        is not None
                    )
                else:
                    prank = 0
                    has_exact = False
                    has_comp = False

                base = [
                    float(
                        has_adapter
                    ),
                    float(arank),
                    float(arr),
                    float(ascore),
                    float(amean),
                    float(agap),
                    float(
                        is_personal
                    ),
                    float(prank),
                    float(
                        has_exact
                    ),
                    safe(
                        pe[
                            "exact_raw_score"
                        ]
                        if pe
                        else None
                    ),
                    safe(
                        pe[
                            "p_exact"
                        ]
                        if pe
                        else None
                    ),
                    float(
                        has_comp
                    ),
                    safe(
                        pe[
                            "composition_raw_score"
                        ]
                        if pe
                        else None
                    ),
                    safe(
                        pe[
                            "p_composition"
                        ]
                        if pe
                        else None
                    ),
                    safe(
                        pe[
                            "recovery_confidence"
                        ]
                        if pe
                        else None
                    ),
                    safe(
                        pe[
                            "piece_count"
                        ]
                        if pe
                        else None
                    ),
                    safe(
                        pe[
                            "fragmentation_ratio"
                        ]
                        if pe
                        else None
                    ),
                    safe(
                        pe[
                            "longest_span_ratio"
                        ]
                        if pe
                        else None
                    ),
                    float(
                        has_adapter
                        and is_personal
                    ),
                    float(
                        a["M"]
                    ),
                    float(
                        pool_size
                    ),
                ]

                ev = (
                    history_ev[text]
                )

                rich = [
                    safe(
                        ev[name]
                    )
                    for name
                    in RICH_FEATURES
                ]

                feat = (
                    base
                    + rich
                )

                if (
                    len(feat)
                    != 30
                ):
                    raise RuntimeError(
                        "Feature width "
                        "changed"
                    )

                features.append(
                    feat
                )

            gold = str(
                a["gold"]
            )

            labels = [
                1
                if text == gold
                else 0
                for text in pool
            ]

            positive_queries += int(
                any(labels)
            )

            total_candidates += (
                len(pool)
            )

            X.extend(
                features
            )

            y.extend(
                labels
            )

            groups.append(
                len(pool)
            )

            surface.write(
                json.dumps(
                    {
                        "schema_version":
                            1,
                        "row_id":
                            rid,
                        "author_name":
                            a[
                                "author_name"
                            ],
                        "split":
                            "val",
                        "gold":
                            gold,
                        "adapter_top10":
                            adapter_texts,
                        "personal_recovery_top5":
                            selected,
                        "final_pool":
                            pool,
                        "pool_size":
                            pool_size,
                        "gold_in_pool":
                            any(
                                labels
                            ),
                        "used_test":
                            False,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(
                        ",",
                        ":",
                    ),
                )
                + "\n"
            )

            if (
                number % 250
                == 0
                or number
                == 20_000
            ):
                print(
                    "RICH30",
                    number,
                    "/20000",
                    flush=True,
                )

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y = np.asarray(
        y,
        dtype=np.int32,
    )

    if len(groups) != 20_000:
        raise RuntimeError(
            "Group count changed"
        )

    if X.shape[1] != 30:
        raise RuntimeError(
            "Feature count changed"
        )

    if not np.isfinite(
        X
    ).all():
        raise RuntimeError(
            "Non-finite Rich30 matrix"
        )

    print(
        "TRAIN_QUERIES =",
        len(groups),
    )

    print(
        "TRAIN_POSITIVE_QUERIES =",
        positive_queries,
    )

    print(
        "TRAIN_CANDIDATES =",
        total_candidates,
    )

    print(
        "FEATURE_COUNT =",
        X.shape[1],
    )

    print(
        "RICH30_MATRIX_GATE=PASS",
        flush=True,
    )

    dataset = lgb.Dataset(
        X,
        label=y,
        group=groups,
        feature_name=FEATURES,
        free_raw_data=False,
    )

    model = lgb.train(
        PARAMS,
        dataset,
        num_boost_round=ROUNDS,
    )

    model_path = (
        args.output_root
        / "final_fiveauthor_generic_explicit_rich30_lambdamart_v1.txt"
    )

    model.save_model(
        str(model_path)
    )

    summary = {
        "schema_version":
            1,
        "experiment":
            "final_fiveauthor_val_rich30_train_v1",
        "training_population":
            "Val",
        "train_queries":
            20_000,
        "train_positive_queries":
            positive_queries,
        "train_candidates":
            total_candidates,
        "feature_count":
            30,
        "base_feature_count":
            21,
        "rich_feature_count":
            9,
        "features":
            FEATURES,
        "params":
            PARAMS,
        "rounds":
            ROUNDS,
        "personal_budget":
            PERSONAL_BUDGET,
        "max_pool_size":
            15,
        "model_path":
            str(model_path),
        "model_sha256":
            sha256_file(
                model_path
            ),
        "surface_path":
            str(surface_path),
        "surface_sha256":
            sha256_file(
                surface_path
            ),
        "used_test":
            False,
        "test_rows_used_as_queries":
            0,
        "test_model_evaluation":
            False,
    }

    summary_path = (
        args.output_root
        / "final_fiveauthor_generic_explicit_val_rich30_training_summary_v1.json"
    )

    summary_path.write_text(
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
        "RICH30_LAMBDAMART_TRAINING=PASS",
        flush=True,
    )

    print(
        "MODEL =",
        model_path,
        flush=True,
    )

    print(
        "MODEL_SHA256 =",
        summary[
            "model_sha256"
        ],
        flush=True,
    )

    print(
        "TEST_ROWS_USED_AS_QUERIES = 0",
        flush=True,
    )


if __name__ == "__main__":
    main()
