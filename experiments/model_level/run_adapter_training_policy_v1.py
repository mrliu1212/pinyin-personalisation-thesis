"""Policy wrapper around the validated Adapter V1 training runner.

Adds two controlled experiment dimensions without duplicating the canonical
training loop:

1. Full-only versus the canonical Full:Initial:Mixed=3:1:2 exposure.
2. Most-recent-N history selection versus the canonical deterministic subset.

The underlying model, loss, optimizer, batching, checkpoint validation,
deterministic epoch ordering, Adapter implementation, and save format remain
those of run_adapter_training_v1.py.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from experiments.model_level import run_adapter_training_v1 as base


ORIGINAL_CHOOSE_POPULATION = base.choose_training_population
ORIGINAL_CHOOSE_REPRESENTATION = base.choose_pinyin_representation

SELECTION_AUDIT: dict[str, Any] = {}


def chronological_key(row: Mapping[str, Any]) -> tuple[int, int, int, int, str]:
    required = (
        "work_chronological_index",
        "chronological_position",
        "source_position_start",
        "source_position_end",
        "row_id",
    )
    missing = [key for key in required if row.get(key) is None]
    if missing:
        raise RuntimeError(
            f"Cannot construct recent-history ordering; "
            f"missing {missing} for row {row.get('row_id')!r}"
        )

    return (
        int(row["work_chronological_index"]),
        int(row["chronological_position"]),
        int(row["source_position_start"]),
        int(row["source_position_end"]),
        str(row["row_id"]),
    )


def make_recent_population_selector():
    signature = inspect.signature(ORIGINAL_CHOOSE_POPULATION)

    def patched(*args: Any, **kwargs: Any):
        bound = signature.bind_partial(*args, **kwargs)

        row_name = None
        for candidate in ("values", "rows", "author_rows", "population"):
            if candidate in bound.arguments:
                row_name = candidate
                break

        if row_name is None:
            raise RuntimeError(
                "STOP: could not identify row population argument in "
                f"choose_training_population{signature}"
            )

        rows = list(bound.arguments[row_name])
        max_rows = bound.arguments.get("max_rows")

        if max_rows is None:
            selected = sorted(rows, key=chronological_key)
        else:
            max_rows = int(max_rows)
            if max_rows < 1:
                raise RuntimeError("--max-rows must be positive")
            if max_rows > len(rows):
                raise RuntimeError(
                    f"--max-rows={max_rows} exceeds author population {len(rows)}"
                )

            ordered = sorted(rows, key=chronological_key)
            selected = ordered[-max_rows:]

        if len({str(row["row_id"]) for row in selected}) != len(selected):
            raise RuntimeError("Duplicate row_id detected in recent-history subset")

        SELECTION_AUDIT.clear()
        SELECTION_AUDIT.update(
            {
                "selection": "most_recent",
                "source_population_rows": len(rows),
                "selected_rows": len(selected),
                "ordering": [
                    "work_chronological_index",
                    "chronological_position",
                    "source_position_start",
                    "source_position_end",
                    "row_id",
                ],
                "oldest_selected": (
                    {
                        "row_id": str(selected[0]["row_id"]),
                        "work_chronological_index": int(
                            selected[0]["work_chronological_index"]
                        ),
                        "chronological_position": int(
                            selected[0]["chronological_position"]
                        ),
                    }
                    if selected
                    else None
                ),
                "newest_selected": (
                    {
                        "row_id": str(selected[-1]["row_id"]),
                        "work_chronological_index": int(
                            selected[-1]["work_chronological_index"]
                        ),
                        "chronological_position": int(
                            selected[-1]["chronological_position"]
                        ),
                    }
                    if selected
                    else None
                ),
            }
        )

        return selected

    return patched


def make_full_only_representation_selector():
    signature = inspect.signature(ORIGINAL_CHOOSE_REPRESENTATION)

    if "forced_mode" not in signature.parameters:
        raise RuntimeError(
            "STOP: choose_pinyin_representation no longer exposes forced_mode"
        )

    if "weights" not in signature.parameters:
        raise RuntimeError(
            "STOP: choose_pinyin_representation no longer exposes weights"
        )

    def patched(*args: Any, **kwargs: Any):
        bound = signature.bind_partial(*args, **kwargs)
        bound.arguments["forced_mode"] = "full"
        bound.arguments["weights"] = (1, 0, 0)
        return ORIGINAL_CHOOSE_REPRESENTATION(
            *bound.args,
            **bound.kwargs,
        )

    return patched


def cli_value(arguments: list[str], flag: str) -> str | None:
    try:
        index = arguments.index(flag)
    except ValueError:
        return None

    if index + 1 >= len(arguments):
        raise RuntimeError(f"Missing value after {flag}")

    return arguments[index + 1]


def main() -> None:
    wrapper = argparse.ArgumentParser(add_help=False)

    wrapper.add_argument(
        "--training-pinyin-policy",
        choices=("full_only", "mixed_3_1_2"),
        required=True,
    )
    wrapper.add_argument(
        "--history-selection",
        choices=("recent", "deterministic"),
        default="deterministic",
    )

    wrapper_args, remaining = wrapper.parse_known_args()

    max_rows_raw = cli_value(remaining, "--max-rows")

    if wrapper_args.history_selection == "recent":
        if max_rows_raw is None:
            raise RuntimeError(
                "Recent-history experiment requires explicit --max-rows"
            )
        base.choose_training_population = make_recent_population_selector()

    if wrapper_args.training_pinyin_policy == "full_only":
        base.choose_pinyin_representation = (
            make_full_only_representation_selector()
        )

    output_root_raw = cli_value(remaining, "--output-root")
    if output_root_raw is None:
        raise RuntimeError("--output-root is required")

    output_root = Path(output_root_raw)

    sys.argv = [sys.argv[0], *remaining]

    base.main()

    result_path = output_root / "training_result.json"
    if not result_path.is_file():
        raise RuntimeError(f"Missing training result: {result_path}")

    result = json.loads(result_path.read_text(encoding="utf-8"))

    requested = result.get("requested_mode_counts", {})

    if wrapper_args.training_pinyin_policy == "full_only":
        if int(requested.get("initial", 0)) != 0:
            raise RuntimeError(
                f"Full-only validation failed: initial count={requested.get('initial')}"
            )
        if int(requested.get("mixed", 0)) != 0:
            raise RuntimeError(
                f"Full-only validation failed: mixed count={requested.get('mixed')}"
            )
        if int(requested.get("full", 0)) < 1:
            raise RuntimeError("Full-only validation failed: no Full examples")

    provenance = {
        "schema_version": 1,
        "wrapper": "run_adapter_training_policy_v1",
        "training_pinyin_policy": wrapper_args.training_pinyin_policy,
        "history_selection": wrapper_args.history_selection,
        "max_rows": (
            int(max_rows_raw)
            if max_rows_raw is not None
            else None
        ),
        "selection_audit": SELECTION_AUDIT or None,
        "requested_mode_counts": requested,
        "effective_mode_counts": result.get("effective_mode_counts"),
    }

    (output_root / "training_policy_wrapper.json").write_text(
        json.dumps(
            provenance,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("===== POLICY WRAPPER VALIDATION =====")
    print(
        "training_pinyin_policy="
        f"{wrapper_args.training_pinyin_policy}"
    )
    print(
        "history_selection="
        f"{wrapper_args.history_selection}"
    )
    print(f"requested_mode_counts={requested}")

    if SELECTION_AUDIT:
        print(
            "selected_rows="
            f"{SELECTION_AUDIT['selected_rows']}"
        )
        print(
            "oldest_selected="
            f"{SELECTION_AUDIT['oldest_selected']}"
        )
        print(
            "newest_selected="
            f"{SELECTION_AUDIT['newest_selected']}"
        )


if __name__ == "__main__":
    main()
