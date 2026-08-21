from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

# Make `src/...` importable when this file is run from experiments/...
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.personalisation.pilot_a import HistoryIndex


EXPECTED_TRAIN_FIT_ROWS = 144_526
EXPECTED_TRAIN_VAL_ROWS = 34_416

EXPECTED_TRAIN_FIT_SHA256 = (
    "547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6"
)
EXPECTED_TRAIN_VAL_SHA256 = (
    "d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220"
)

HISTORY_BUDGET = 5000
OUTPUT_VERSION = "initial-standardized-h5000-a0-v1"

DEFAULT_FIT = Path(
    r"C:\Users\chiar\Desktop\LBH\thesis-context-compare"
    r"\results\personalisation\context_comparison_v2\clean3_train_fit_v1.jsonl"
)
DEFAULT_VAL = Path(
    r"C:\Users\chiar\Desktop\LBH\thesis-context-compare"
    r"\results\personalisation\context_comparison_v2\clean3_train_val_v1.jsonl"
)
DEFAULT_OUTPUT = ROOT / "results/personalisation/initial_recovery_comparison_v1"

# These fields may carry Full-Pinyin history semantics in the standardized source.
# They must not survive into the Initial query rows without recomputation.
STALE_HISTORY_FIELDS = {
    "history_available",
    "same_pinyin_history_count",
    "same_initial_history_count",
    "visible_raw_history_count",
    "raw_prior_count",
    "distinct_history_targets",
    "distinct_initial_targets",
    "ambiguous",
    "conflict",
    "frequency_winner",
    "frequency_winner_tied",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(canonical_json(row) + "\n")


def row_id(row: Mapping[str, Any]) -> str:
    for key in ("row_id", "condition_id", "anchor_id"):
        value = row.get(key)
        if value is not None and str(value):
            return str(value)
    raise RuntimeError(f"Cannot resolve row ID; keys={sorted(row.keys())}")


def gold(row: Mapping[str, Any]) -> str:
    for key in ("gold", "target"):
        value = row.get(key)
        if value is not None:
            return str(value)
    raise RuntimeError(f"{row_id(row)}: missing gold/target")


def initial_segments(segments: Sequence[Any]) -> tuple[str, ...]:
    output: list[str] = []
    for raw in segments:
        segment = str(raw).strip().lower()
        if not segment:
            raise RuntimeError("Empty Pinyin segment")
        # Historical Initial definition: first letter of each syllable.
        first = segment[0]
        if not ("a" <= first <= "z"):
            raise RuntimeError(f"Unexpected Pinyin segment: {segment!r}")
        output.append(first)
    if not output:
        raise RuntimeError("Empty Pinyin sequence")
    return tuple(output)


def transform_row(row: Mapping[str, Any], *, partition: str) -> dict[str, Any]:
    if "pinyin_segments" not in row:
        raise RuntimeError(f"{row_id(row)}: missing pinyin_segments")

    full_segments = tuple(str(value).lower() for value in row["pinyin_segments"])
    initials = initial_segments(full_segments)

    if len(initials) != len(full_segments):
        raise AssertionError("Initial transform changed syllable count")

    value = dict(row)
    for key in STALE_HISTORY_FIELDS:
        value.pop(key, None)

    value["source_condition"] = str(row.get("condition", "full_short"))
    value["source_full_pinyin_input"] = str(
        row.get("pinyin_input", " ".join(full_segments))
    )
    value["source_full_pinyin_segments"] = list(full_segments)

    value["condition"] = "initial_short"
    value["pinyin_segments"] = list(initials)
    value["pinyin_input"] = " ".join(initials)
    value["standardized_partition"] = partition
    value["initial_transform_version"] = "first-letter-per-syllable-v1"
    return value


def require_schema(rows: Sequence[Mapping[str, Any]], label: str) -> None:
    required = {
        "author",
        "work_id",
        "chronological_position",
        "context",
        "pinyin_segments",
    }
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise RuntimeError(
                f"{label} row {index} / {row_id(row)} missing fields: "
                f"{sorted(missing)}"
            )
        gold(row)


def assert_unique_ids(
    fit_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
) -> None:
    fit_ids = [row_id(row) for row in fit_rows]
    val_ids = [row_id(row) for row in val_rows]

    if len(fit_ids) != len(set(fit_ids)):
        raise RuntimeError("Duplicate row IDs inside Train-Fit")
    if len(val_ids) != len(set(val_ids)):
        raise RuntimeError("Duplicate row IDs inside Train-Val")

    overlap = set(fit_ids).intersection(val_ids)
    if overlap:
        sample = sorted(overlap)[:10]
        raise RuntimeError(f"Train-Fit/Train-Val row-ID overlap: {sample}")


def assert_chronology(
    fit_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_author_fit: dict[str, list[int]] = defaultdict(list)
    by_author_val: dict[str, list[int]] = defaultdict(list)

    for row in fit_rows:
        by_author_fit[str(row["author"])].append(int(row["chronological_position"]))
    for row in val_rows:
        by_author_val[str(row["author"])].append(int(row["chronological_position"]))

    authors = sorted(set(by_author_fit) | set(by_author_val))
    per_author: dict[str, Any] = {}

    for author in authors:
        fit_positions = sorted(by_author_fit[author])
        val_positions = sorted(by_author_val[author])
        combined = fit_positions + val_positions

        if len(combined) != len(set(combined)):
            raise RuntimeError(f"{author}: duplicate chronological_position")

        # Whole-work standardized split should put Train-Fit before Train-Val
        # within each author. Fail rather than invent chronology if this is false.
        ordered_fit_before_val = True
        if fit_positions and val_positions:
            ordered_fit_before_val = max(fit_positions) < min(val_positions)
            if not ordered_fit_before_val:
                raise RuntimeError(
                    f"{author}: Train-Fit is not strictly earlier than Train-Val; "
                    "do not continue until split provenance is checked"
                )

        per_author[author] = {
            "train_fit_rows": len(fit_positions),
            "train_val_rows": len(val_positions),
            "fit_min_position": min(fit_positions) if fit_positions else None,
            "fit_max_position": max(fit_positions) if fit_positions else None,
            "val_min_position": min(val_positions) if val_positions else None,
            "val_max_position": max(val_positions) if val_positions else None,
            "fit_strictly_before_val": ordered_fit_before_val,
        }

    return per_author


def build_author_positions(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[str(row["author"])].append(int(row["chronological_position"]))
    return {
        author: tuple(sorted(positions))
        for author, positions in grouped.items()
    }


def query_object(row: Mapping[str, Any]) -> SimpleNamespace:
    # HistoryIndex.visible() only reads these three attributes.
    return SimpleNamespace(
        author=str(row["author"]),
        chronological_position=int(row["chronological_position"]),
        pinyin=tuple(str(value) for value in row["pinyin_segments"]),
    )


def unique_frequency_winner(
    visible: Sequence[Mapping[str, Any]],
) -> tuple[str | None, bool, Counter[str]]:
    counts: Counter[str] = Counter(gold(row) for row in visible)
    if not counts:
        return None, False, counts

    maximum = max(counts.values())
    winners = sorted(target for target, count in counts.items() if count == maximum)
    if len(winners) == 1:
        return winners[0], False, counts
    return None, True, counts


def recompute_val_history(
    fit_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    combined = list(fit_rows) + list(val_rows)
    index = HistoryIndex(combined, HISTORY_BUDGET)
    positions = build_author_positions(combined)

    outputs: list[dict[str, Any]] = []

    aggregate = Counter()
    per_author: dict[str, Counter[str]] = defaultdict(Counter)

    for number, row in enumerate(val_rows, start=1):
        author = str(row["author"])
        position = int(row["chronological_position"])
        author_positions = positions.get(author, ())

        stop = bisect.bisect_left(author_positions, position)
        raw_prior_count = stop
        visible_raw_history_count = min(HISTORY_BUDGET, raw_prior_count)

        visible = index.visible(query_object(row))
        winner, winner_tied, counts = unique_frequency_winner(visible)

        target = gold(row)
        same_initial_count = len(visible)
        distinct_targets = len(counts)
        initial_history_available = same_initial_count > 0
        ambiguous = distinct_targets >= 2
        conflict = ambiguous and winner is not None and target != winner

        value = dict(row)
        value.update(
            {
                "raw_prior_count": raw_prior_count,
                "visible_raw_history_count": visible_raw_history_count,
                # `history_available` here deliberately means usable same-Initial
                # personal history, matching the personalization subset semantics.
                "history_available": initial_history_available,
                "initial_history_available": initial_history_available,
                "same_initial_history_count": same_initial_count,
                "distinct_initial_targets": distinct_targets,
                "ambiguous": ambiguous,
                "frequency_winner": winner,
                "frequency_winner_tied": winner_tied,
                "conflict": conflict,
            }
        )
        outputs.append(value)

        flags = {
            "rows": 1,
            "raw_history_available": int(raw_prior_count > 0),
            "initial_history_available": int(initial_history_available),
            "ambiguous": int(ambiguous),
            "conflict": int(conflict),
            "frequency_winner_tied": int(winner_tied),
        }
        aggregate.update(flags)
        per_author[author].update(flags)

        if number % 5000 == 0 or number == len(val_rows):
            print(
                f"Initial H5000 audit: {number}/{len(val_rows)}",
                flush=True,
            )

    def render(counter: Counter[str]) -> dict[str, Any]:
        rows = counter["rows"]
        return {
            "rows": rows,
            "raw_history_available": counter["raw_history_available"],
            "raw_history_available_rate": (
                counter["raw_history_available"] / rows if rows else None
            ),
            "initial_history_available": counter["initial_history_available"],
            "initial_history_available_rate": (
                counter["initial_history_available"] / rows if rows else None
            ),
            "ambiguous": counter["ambiguous"],
            "ambiguous_rate": counter["ambiguous"] / rows if rows else None,
            "conflict": counter["conflict"],
            "conflict_rate": counter["conflict"] / rows if rows else None,
            "frequency_winner_tied": counter["frequency_winner_tied"],
            "frequency_winner_tied_rate": (
                counter["frequency_winner_tied"] / rows if rows else None
            ),
        }

    summary = {
        "schema_version": 1,
        "status": "complete",
        "version": OUTPUT_VERSION,
        "condition": "initial_short",
        "partition": "train_val",
        "history_budget": HISTORY_BUDGET,
        "history_semantics": (
            "same author -> strictly prior -> latest up-to-5000 raw interactions "
            "-> exact segmented Initial match"
        ),
        "history_budget_applied_before_initial_filter": True,
        "strictly_prior": True,
        "train_val_rolling_history": True,
        "overall": render(aggregate),
        "per_author": {
            author: render(per_author[author])
            for author in sorted(per_author)
        },
    }
    return outputs, summary


def deterministic_audit_ids(
    rows: Sequence[Mapping[str, Any]],
    n: int = 200,
) -> set[str]:
    ranked = sorted(
        (hashlib.sha256(row_id(row).encode("utf-8")).hexdigest(), row_id(row))
        for row in rows
    )
    return {identifier for _, identifier in ranked[: min(n, len(ranked))]}


def reference_history_audit(
    fit_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Cross-check HistoryIndex against a simple reference implementation."""
    combined = list(fit_rows) + list(val_rows)
    by_author: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in combined:
        by_author[str(row["author"])].append(row)
    for author in by_author:
        by_author[author].sort(key=lambda row: int(row["chronological_position"]))

    index = HistoryIndex(combined, HISTORY_BUDGET)
    selected = deterministic_audit_ids(val_rows, n=200)

    checked = 0
    for row in val_rows:
        identifier = row_id(row)
        if identifier not in selected:
            continue

        author_rows = by_author[str(row["author"])]
        position = int(row["chronological_position"])

        prior = [
            item
            for item in author_rows
            if int(item["chronological_position"]) < position
        ]
        raw_window = prior[-HISTORY_BUDGET:]
        expected = [
            item
            for item in raw_window
            if tuple(item["pinyin_segments"]) == tuple(row["pinyin_segments"])
        ]
        actual = list(index.visible(query_object(row)))

        expected_ids = [row_id(item) for item in expected]
        actual_ids = [row_id(item) for item in actual]
        if expected_ids != actual_ids:
            raise RuntimeError(
                f"HistoryIndex mismatch at {identifier}: "
                f"expected={expected_ids[:10]} actual={actual_ids[:10]}"
            )
        checked += 1

    return {
        "status": "passed",
        "deterministic_sample_rows": checked,
        "reference_rule": (
            "strictly prior same-author rows -> last 5000 raw -> "
            "exact segmented Initial match"
        ),
        "history_index_exact_match": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "A0 standardized Initial+Short transform and rolling causal H5000 audit. "
            "No Generic inference, Dev3000, or Test."
        )
    )
    parser.add_argument("--train-fit", type=Path, default=DEFAULT_FIT)
    parser.add_argument("--train-val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-source-hash-check",
        action="store_true",
        help="Debug only. Do not use for the formal standardized run.",
    )
    args = parser.parse_args()

    fit_path = args.train_fit.resolve()
    val_path = args.train_val.resolve()
    output_root = args.output_root.resolve()

    if not fit_path.is_file():
        raise FileNotFoundError(fit_path)
    if not val_path.is_file():
        raise FileNotFoundError(val_path)

    fit_sha = sha256_file(fit_path)
    val_sha = sha256_file(val_path)

    if not args.skip_source_hash_check:
        if fit_sha.lower() != EXPECTED_TRAIN_FIT_SHA256:
            raise RuntimeError(
                "Train-Fit SHA256 mismatch:\n"
                f"expected={EXPECTED_TRAIN_FIT_SHA256}\nactual={fit_sha}"
            )
        if val_sha.lower() != EXPECTED_TRAIN_VAL_SHA256:
            raise RuntimeError(
                "Train-Val SHA256 mismatch:\n"
                f"expected={EXPECTED_TRAIN_VAL_SHA256}\nactual={val_sha}"
            )

    print("Reading standardized source files...", flush=True)
    fit_source = read_jsonl(fit_path)
    val_source = read_jsonl(val_path)

    if len(fit_source) != EXPECTED_TRAIN_FIT_ROWS:
        raise RuntimeError(
            f"Train-Fit rows differ: {len(fit_source)} != {EXPECTED_TRAIN_FIT_ROWS}"
        )
    if len(val_source) != EXPECTED_TRAIN_VAL_ROWS:
        raise RuntimeError(
            f"Train-Val rows differ: {len(val_source)} != {EXPECTED_TRAIN_VAL_ROWS}"
        )

    require_schema(fit_source, "Train-Fit")
    require_schema(val_source, "Train-Val")
    assert_unique_ids(fit_source, val_source)
    chronology = assert_chronology(fit_source, val_source)

    print("Transforming Full Pinyin -> Initial...", flush=True)
    fit_initial = [
        transform_row(row, partition="train_fit")
        for row in fit_source
    ]
    val_initial_base = [
        transform_row(row, partition="train_val")
        for row in val_source
    ]

    # Strong transform assertions.
    for source, transformed in zip(fit_source + val_source, fit_initial + val_initial_base):
        source_segments = tuple(str(x).lower() for x in source["pinyin_segments"])
        transformed_segments = tuple(transformed["pinyin_segments"])
        if len(source_segments) != len(transformed_segments):
            raise AssertionError("Syllable count changed")
        if transformed_segments != tuple(segment[0] for segment in source_segments):
            raise AssertionError("Initial transform differs from first-letter rule")

    print("Recomputing rolling Initial H5000 history...", flush=True)
    val_initial, history_summary = recompute_val_history(
        fit_initial,
        val_initial_base,
    )

    print("Running independent HistoryIndex reference audit...", flush=True)
    reference_audit = reference_history_audit(
        fit_initial,
        val_initial_base,
    )

    output_root.mkdir(parents=True, exist_ok=True)

    fit_out = output_root / "initial_train_fit_v1.jsonl"
    val_out = output_root / "initial_train_val_v1.jsonl"
    transform_out = output_root / "initial_transform_audit.json"
    history_out = output_root / "history_semantics_audit.json"
    manifest_out = output_root / "manifest.json"

    write_jsonl(fit_out, fit_initial)
    write_jsonl(val_out, val_initial)

    transform_summary = {
        "schema_version": 1,
        "status": "passed",
        "version": OUTPUT_VERSION,
        "source_condition": "full_short",
        "target_condition": "initial_short",
        "rule": "first letter of every Pinyin syllable",
        "examples": {
            "shi yong": "s y",
            "zhong guo": "z g",
            "chi fan": "c f",
        },
        "sh_zh_ch_special_initials_used": False,
        "train_fit_rows": len(fit_initial),
        "train_val_rows": len(val_initial),
        "syllable_count_preserved": True,
        "source_full_pinyin_preserved_in_output": True,
        "stale_full_history_fields_reused": False,
    }

    write_json(transform_out, transform_summary)
    write_json(
        history_out,
        {
            **history_summary,
            "reference_implementation_audit": reference_audit,
            "chronology_audit": chronology,
        },
    )

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "version": OUTPUT_VERSION,
        "dev3000_used": False,
        "test_used": False,
        "generic_inference_performed": False,
        "source": {
            "train_fit": {
                "path": str(fit_path),
                "rows": len(fit_source),
                "sha256": fit_sha,
            },
            "train_val": {
                "path": str(val_path),
                "rows": len(val_source),
                "sha256": val_sha,
            },
        },
        "outputs": {
            "initial_train_fit_v1.jsonl": {
                "path": str(fit_out),
                "rows": len(fit_initial),
                "sha256": sha256_file(fit_out),
            },
            "initial_train_val_v1.jsonl": {
                "path": str(val_out),
                "rows": len(val_initial),
                "sha256": sha256_file(val_out),
            },
            "initial_transform_audit.json": {
                "path": str(transform_out),
                "sha256": sha256_file(transform_out),
            },
            "history_semantics_audit.json": {
                "path": str(history_out),
                "sha256": sha256_file(history_out),
            },
        },
    }
    write_json(manifest_out, manifest)

    print()
    print("=== A0 INITIAL STANDARDIZED PREPARATION COMPLETE ===")
    print(f"Train-Fit: {len(fit_initial):,}")
    print(f"Train-Val: {len(val_initial):,}")
    print(f"Output: {output_root}")
    print()
    print("Train-Val Initial H5000:")
    for key, value in history_summary["overall"].items():
        print(f"  {key}: {value}")
    print()
    print(f"Reference HistoryIndex audit rows: {reference_audit['deterministic_sample_rows']}")
    print(f"Manifest: {manifest_out}")


if __name__ == "__main__":
    main()
