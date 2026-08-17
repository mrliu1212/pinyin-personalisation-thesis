"""Run or resume the T1-aligned Personalisation Pilot A M1-H5000 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.personalisation.h5000 import H5000Runner


ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--phase", choices=("prepare", "dev-generic", "dev-embeddings", "tune", "test-embeddings", "evaluate", "smoke", "all"), required=True)
    value.add_argument("--root", type=Path, default=ROOT)
    value.add_argument("--dataset-root", type=Path, required=True)
    value.add_argument("--pinyingpt-model", type=Path, required=True)
    value.add_argument("--embedding-model", type=Path, required=True)
    value.add_argument("--t1-predictions", type=Path, required=True)
    value.add_argument("--output-root", type=Path, default=ROOT / "results/personalisation/pilot_a_context_memory")
    return value


def main() -> None:
    args = parser().parse_args()
    runner = H5000Runner(
        root=args.root.resolve(),
        dataset_root=args.dataset_root.resolve(),
        pinyingpt_model=args.pinyingpt_model.resolve(),
        embedding_model=args.embedding_model.resolve(),
        pilot_root=args.output_root.resolve(),
        t1_predictions=args.t1_predictions.resolve(),
    )
    phases = {
        "prepare": runner.prepare,
        "dev-generic": runner.dev_runner.generic,
        "dev-embeddings": runner.dev_runner.embeddings,
        "tune": runner.tune,
        "test-embeddings": runner.embeddings,
        "evaluate": runner.evaluate,
        "smoke": runner.smoke,
        "all": runner.all,
    }
    print(json.dumps(phases[args.phase](), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
