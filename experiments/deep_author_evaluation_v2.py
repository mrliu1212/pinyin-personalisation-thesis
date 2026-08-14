"""Build the frozen Deep Author Evaluation V2 design or run its T1 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.deep_author_v2 import DesignBuilder


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("design", "t1", "metrics"), required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.phase != "design":
        raise SystemExit("T1 runner is enabled only after the design checkpoint is frozen")
    result = DesignBuilder(args.root.resolve()).run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
