"""Final Generic PinyinGPT2-Concat baseline on frozen Val/Test.

No Adapter.
No H5000.
No personal memory.
No Rich30.

The frozen per-row pinyin_input is used exactly as stored, including
Full / Initial / Mixed conditions.

Test is explicitly opened for FINAL baseline evaluation only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.reference_backend_pinyingpt.backend import (
    PinyinGPTConcatBackend,
)


EXPECTED_MANIFEST_SHA256 = (
    "e05b07d5020ebf0ccf090e98b34e4c66"
    "db7dab9b565541d39feb2d6f1d1bb20b"
)

EXPECTED_VAL_ROWS = 20_000
EXPECTED_TEST_ROWS = 40_000

EXPECTED_AUTHORS = {
    "Re_spectators",
    "Etinjat",
    "Agent Phage",
    "QBLevi",
    "breaddddd",
}

EXPECTED_PER_AUTHOR = {
    "val": 4_000,
    "test": 8_000,
}

TOP_K = 10
BEAM_SIZE = 16

UNSUPPORTED_PREFIXES = (
    "no tokenizer candidates for Pinyin ",
    "unsupported Pinyin input: ",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def read_rows(
    path: Path,
    split: str,
) -> list[dict[str, Any]]:

    rows = []
    per_author = Counter()

    with path.open(
        encoding="utf-8"
    ) as f:

        for number, line in enumerate(f, 1):

            if not line.strip():
                continue

            r = json.loads(line)

            row_split = str(
                r["split"]
            ).lower()

            if row_split != split:
                continue

            author = str(
                r["author_name"]
            )

            if author not in EXPECTED_AUTHORS:
                raise RuntimeError(
                    f"Unexpected author at line "
                    f"{number}: {author}"
                )

            pinyin_input = str(
                r["pinyin_input"]
            ).strip()

            segments = tuple(
                x
                for x in pinyin_input.split()
                if x
            )

            if not segments:
                raise RuntimeError(
                    f"Empty pinyin_input: "
                    f"{r['row_id']}"
                )

            gold = str(r["gold"])

            if len(gold) != len(segments):
                raise RuntimeError(
                    "Gold/Pinyin segment-count "
                    f"mismatch: {r['row_id']} "
                    f"gold={len(gold)} "
                    f"segments={len(segments)}"
                )

            rows.append({
                "row_id":
                    str(r["row_id"]),

                "split":
                    split,

                "author_name":
                    author,

                "work_id":
                    str(r["work_id"]),

                "M":
                    int(r["M"]),

                "context":
                    str(r.get("context") or ""),

                "gold":
                    gold,

                "pinyin_input":
                    pinyin_input,

                "pinyin_segments":
                    segments,

                "typing_mode":
                    str(r["typing_mode"]),

                "effective_typing_mode":
                    str(
                        r["effective_typing_mode"]
                    ),
            })

            per_author[author] += 1

    expected = (
        EXPECTED_VAL_ROWS
        if split == "val"
        else EXPECTED_TEST_ROWS
    )

    if len(rows) != expected:
        raise RuntimeError(
            f"{split}: expected {expected}, "
            f"got {len(rows)}"
        )

    for author in EXPECTED_AUTHORS:
        expected_author = (
            EXPECTED_PER_AUTHOR[split]
        )

        if (
            per_author[author]
            != expected_author
        ):
            raise RuntimeError(
                f"{split}/{author}: "
                f"{per_author[author]} != "
                f"{expected_author}"
            )

    return rows


def predict_one(
    backend: PinyinGPTConcatBackend,
    row: dict[str, Any],
) -> dict[str, Any]:

    started = time.perf_counter()

    generated = None
    unsupported_reason = None

    with torch.inference_mode():
        try:
            generated = backend.generate(
                row["context"],
                list(row["pinyin_segments"]),
                top_k=TOP_K,
                beam_size=BEAM_SIZE,
            )

        except ValueError as error:

            message = str(error)

            if not any(
                message.startswith(prefix)
                for prefix in UNSUPPORTED_PREFIXES
            ):
                raise

            # Frozen denominator:
            # known PinyinGPT representation failures are
            # explicit misses and are never dropped.
            unsupported_reason = message

    elapsed = (
        time.perf_counter()
        - started
    )

    if generated is None:
        candidates = []
        scores = []
        runtime_device = str(
            getattr(
                backend,
                "device",
                "unknown",
            )
        )

    else:
        candidates = [
            c.text
            for c
            in generated.candidates
        ]

        scores = [
            float(c.log_probability)
            for c
            in generated.candidates
        ]

        runtime_device = str(
            generated.runtime_device
        )

    gold = row["gold"]

    rank = (
        candidates.index(gold) + 1
        if gold in candidates
        else None
    )

    return {
        "schema_version": 1,

        "experiment":
            "final_generic_baseline_v1",

        "condition":
            "generic",

        "split":
            row["split"],

        "row_id":
            row["row_id"],

        "author_name":
            row["author_name"],

        "work_id":
            row["work_id"],

        "M":
            row["M"],

        "typing_mode":
            row["typing_mode"],

        "effective_typing_mode":
            row["effective_typing_mode"],

        "context":
            row["context"],

        "gold":
            gold,

        "pinyin_input":
            row["pinyin_input"],

        "segmented_pinyin":
            list(
                row["pinyin_segments"]
            ),

        "top10_candidates":
            candidates,

        "top10_candidate_scores":
            scores,

        "gold_top10_rank":
            rank,

        "top1_correct":
            rank == 1,

        "top3_correct":
            rank is not None
            and rank <= 3,

        "top5_correct":
            rank is not None
            and rank <= 5,

        "top10_present":
            rank is not None,

        "reciprocal_rank_at_10":
            (
                0.0
                if rank is None
                else 1.0 / rank
            ),

        "unsupported_input":
            unsupported_reason
            is not None,

        "unsupported_reason":
            unsupported_reason,

        "inference_seconds":
            elapsed,

        "beam_size":
            BEAM_SIZE,

        "top_k":
            TOP_K,

        "runtime_device":
            runtime_device,

        "used_adapter":
            False,

        "used_h5000":
            False,

        "used_rich30":
            False,
    }


def metric_block(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:

    n = len(rows)

    if n == 0:
        return {
            "rows": 0,
        }

    def mean_bool(key: str) -> float:
        return (
            sum(
                bool(r[key])
                for r in rows
            )
            / n
        )

    total_seconds = sum(
        float(r["inference_seconds"])
        for r in rows
    )

    return {
        "rows":
            n,

        "top1":
            mean_bool("top1_correct"),

        "top3":
            mean_bool("top3_correct"),

        "top5":
            mean_bool("top5_correct"),

        "top10":
            mean_bool("top10_present"),

        "mrr_at_10":
            sum(
                float(
                    r[
                        "reciprocal_rank_at_10"
                    ]
                )
                for r in rows
            )
            / n,

        "missing_at_10_rate":
            1.0
            - mean_bool(
                "top10_present"
            ),

        "unsupported_input_rate":
            sum(
                bool(r["unsupported_input"])
                for r in rows
            )
            / n,

        "unsupported_input_count":
            sum(
                bool(r["unsupported_input"])
                for r in rows
            ),

        "inference_seconds_total":
            total_seconds,

        "mean_inference_ms":
            1000.0
            * total_seconds
            / n,
    }


def compute_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:

    by_author = defaultdict(list)
    by_mode = defaultdict(list)
    by_m = defaultdict(list)

    for r in rows:

        by_author[
            r["author_name"]
        ].append(r)

        by_mode[
            r["effective_typing_mode"]
        ].append(r)

        by_m[
            str(r["M"])
        ].append(r)

    return {
        "micro":
            metric_block(rows),

        "by_author": {
            k: metric_block(v)
            for k, v
            in sorted(
                by_author.items()
            )
        },

        "by_effective_typing_mode": {
            k: metric_block(v)
            for k, v
            in sorted(
                by_mode.items()
            )
        },

        "by_M": {
            k: metric_block(v)
            for k, v
            in sorted(
                by_m.items(),
                key=lambda x: int(x[0]),
            )
        },
    }


def run_split(
    *,
    backend: PinyinGPTConcatBackend,
    rows: list[dict[str, Any]],
    split: str,
    output_root: Path,
    log_every: int,
) -> dict[str, Any]:

    predictions_path = (
        output_root
        / f"generic_{split}_predictions.jsonl"
    )

    metrics_path = (
        output_root
        / f"generic_{split}_metrics.json"
    )

    if predictions_path.exists():
        raise RuntimeError(
            f"Refusing overwrite: "
            f"{predictions_path}"
        )

    started = time.time()

    predictions = []

    with predictions_path.open(
        "w",
        encoding="utf-8",
    ) as out:

        for i, row in enumerate(
            rows,
            start=1,
        ):

            pred = predict_one(
                backend,
                row,
            )

            predictions.append(pred)

            out.write(
                canonical_json(pred)
                + "\n"
            )

            if (
                i == 1
                or i % log_every == 0
                or i == len(rows)
            ):
                elapsed = (
                    time.time()
                    - started
                )

                print(
                    f"{split} "
                    f"{i}/{len(rows)} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    metrics = compute_metrics(
        predictions
    )

    metrics["split"] = split

    metrics[
        "wall_runtime_seconds"
    ] = (
        time.time()
        - started
    )

    metrics[
        "predictions_sha256"
    ] = sha256_file(
        predictions_path
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"===== GENERIC {split.upper()} ====="
    )

    micro = metrics["micro"]

    print(
        f"Top1={micro['top1']:.6f} "
        f"Top3={micro['top3']:.6f} "
        f"Top5={micro['top5']:.6f} "
        f"Top10={micro['top10']:.6f} "
        f"MRR10={micro['mrr_at_10']:.6f} "
        f"Missing={micro['missing_at_10_rate']:.6f} "
        f"Unsupported={micro['unsupported_input_rate']:.6f}"
    )

    return metrics


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--manifest",
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
        default="cuda",
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--open-test",
        action="store_true",
        help=(
            "Required explicit acknowledgement "
            "that Test is being opened for final "
            "evaluation."
        ),
    )

    args = parser.parse_args()

    if not args.open_test:
        raise RuntimeError(
            "STOP: --open-test is required."
        )

    if TOP_K != 10:
        raise RuntimeError(
            "top_k freeze changed"
        )

    if BEAM_SIZE != 16:
        raise RuntimeError(
            "beam_size freeze changed"
        )

    manifest_sha = sha256_file(
        args.manifest
    )

    print(
        "MANIFEST_SHA256 =",
        manifest_sha,
    )

    if (
        manifest_sha
        != EXPECTED_MANIFEST_SHA256
    ):
        raise RuntimeError(
            "Frozen final manifest SHA changed"
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = list(
        args.output_root.iterdir()
    )

    if existing:
        raise RuntimeError(
            "Refusing to use non-empty "
            f"output directory: "
            f"{args.output_root}"
        )

    print(
        "===== LOAD VAL =====",
        flush=True,
    )

    val_rows = read_rows(
        args.manifest,
        "val",
    )

    print(
        "VAL_ROWS =",
        len(val_rows),
    )

    print(
        "===== LOAD TEST =====",
        flush=True,
    )

    test_rows = read_rows(
        args.manifest,
        "test",
    )

    print(
        "TEST_ROWS =",
        len(test_rows),
    )

    config = {
        "schema_version": 1,

        "experiment":
            "final_generic_baseline_v1",

        "status":
            "FINAL_EVALUATION",

        "checkpoint_path":
            str(
                args.checkpoint.resolve()
            ),

        "checkpoint_name":
            "aihijo/"
            "transformers4ime-pinyingpt-concat",

        "checkpoint_revision":
            "76dd20dc92d8236a350fb732e99dde6fa15e2263",

        "manifest_path":
            str(
                args.manifest.resolve()
            ),

        "manifest_sha256":
            manifest_sha,

        "val_rows":
            len(val_rows),

        "test_rows":
            len(test_rows),

        "beam_size":
            BEAM_SIZE,

        "top_k":
            TOP_K,

        "pinyin_policy":
            "use frozen pinyin_input exactly",

        "conditions":
            [
                "full",
                "initial",
                "mixed",
            ],

        "generic_only":
            True,

        "used_adapter":
            False,

        "used_h5000":
            False,

        "used_rich30":
            False,

        "test_opened":
            True,

        "test_opening_scope":
            (
                "Final Generic baseline "
                "evaluation only"
            ),

        "post_test_policy":
            (
                "Generic Test results must "
                "not be used to change Adapter, "
                "H5000, Personal5, Rich30, "
                "LambdaMART, architecture, or "
                "hyperparameters."
            ),

        "started_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    (
        args.output_root
        / "evaluation_config.json"
    ).write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    global_started = time.time()

    print()
    print(
        "===== LOAD GENERIC BACKEND =====",
        flush=True,
    )

    backend = PinyinGPTConcatBackend(
        checkpoint=args.checkpoint,
        device=args.device,
    )

    print(
        "GENERIC_BACKEND_LOAD=PASS",
        flush=True,
    )

    # Val first.
    val_metrics = run_split(
        backend=backend,
        rows=val_rows,
        split="val",
        output_root=args.output_root,
        log_every=args.log_every,
    )

    # Then final Test.
    test_metrics = run_split(
        backend=backend,
        rows=test_rows,
        split="test",
        output_root=args.output_root,
        log_every=args.log_every,
    )

    del backend
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "schema_version": 1,

        "experiment":
            "final_generic_baseline_v1",

        "status":
            "complete",

        "val":
            val_metrics,

        "test":
            test_metrics,

        "total_wall_runtime_seconds":
            time.time()
            - global_started,

        "manifest_sha256":
            manifest_sha,

        "test_opened":
            True,

        "test_used_for_model_selection":
            False,
    }

    result_path = (
        args.output_root
        / "final_generic_result.json"
    )

    result_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "================================"
    )
    print(
        "FINAL GENERIC BASELINE COMPLETE"
    )
    print(
        "================================"
    )
    print(
        "VAL_ROWS =",
        len(val_rows),
    )
    print(
        "TEST_ROWS =",
        len(test_rows),
    )
    print(
        "BEAM_SIZE = 16"
    )
    print(
        "TOP_K = 10"
    )
    print(
        "TEST_OPENED = TRUE"
    )
    print(
        "TEST_USED_FOR_MODEL_SELECTION = FALSE"
    )
    print(
        "RESULT =",
        result_path,
    )


if __name__ == "__main__":
    main()
