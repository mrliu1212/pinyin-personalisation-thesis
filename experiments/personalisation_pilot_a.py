"""Run or resume Personalisation Pilot A phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.personalisation.pilot_a import PilotRunner


ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--phase", choices=("prepare", "generic", "embeddings", "tune", "evaluate", "smoke", "all"), required=True)
    value.add_argument("--root", type=Path, default=ROOT)
    value.add_argument("--dataset-root", type=Path, required=True)
    value.add_argument("--pinyingpt-model", type=Path, required=True)
    value.add_argument("--embedding-model", type=Path, required=True)
    value.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/personalisation/pilot_a_context_memory",
    )
    return value


def main() -> None:
    args = parser().parse_args()
    runner = PilotRunner(
        root=args.root.resolve(),
        dataset_root=args.dataset_root.resolve(),
        pinyingpt_model=args.pinyingpt_model.resolve(),
        embedding_model=args.embedding_model.resolve(),
        output_root=args.output_root.resolve(),
    )
    result = getattr(runner, args.phase)()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
