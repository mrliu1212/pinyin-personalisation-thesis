"""Prepare and gate the focused Multi3 128-position H5000 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.personalisation.multi3_128 import Multi3AuditRunner


ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--phase", choices=("prepare-audit", "preflight", "run"), required=True
    )
    value.add_argument("--root", type=Path, default=ROOT)
    value.add_argument("--dataset-root", type=Path, required=True)
    value.add_argument("--pinyingpt-model", type=Path, required=True)
    value.add_argument("--t1-predictions", type=Path, required=True)
    value.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/personalisation/multi3_128_h5000_v2",
    )
    return value


def main() -> None:
    args = parser().parse_args()
    runner = Multi3AuditRunner(
        root=args.root.resolve(),
        dataset_root=args.dataset_root.resolve(),
        pinyingpt_model=args.pinyingpt_model.resolve(),
        t1_predictions=args.t1_predictions.resolve(),
        output_root=args.output_root.resolve(),
    )
    action = {
        "prepare-audit": runner.prepare_audit,
        "preflight": runner.preflight,
        "run": runner.formal_run,
    }[args.phase]
    print(json.dumps(action(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
