"""Run or resume candidate-aware Personalisation M2-H5000."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.personalisation.h5000 import H5000Runner
from src.personalisation.m2_h5000 import M2H5000Runner


ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--phase",
        choices=("prepare", "benchmark", "dev-scores", "tune", "test-scores", "evaluate", "smoke", "all"),
        required=True,
    )
    value.add_argument("--root", type=Path, default=ROOT)
    value.add_argument("--dataset-root", type=Path, required=True)
    value.add_argument("--pinyingpt-model", type=Path, required=True)
    value.add_argument("--embedding-model", type=Path, required=True)
    value.add_argument("--reranker-model", type=Path, required=True)
    value.add_argument("--t1-predictions", type=Path, required=True)
    value.add_argument(
        "--m1-root",
        type=Path,
        default=ROOT / "results/personalisation/pilot_a_context_memory",
    )
    value.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/personalisation/m2_h5000",
    )
    value.add_argument("--batch-size", type=int, default=32)
    value.add_argument("--max-length", type=int, default=512)
    value.add_argument("--benchmark-queries", type=int, default=64)
    return value


def main() -> None:
    args = parser().parse_args()
    m1 = H5000Runner(
        root=args.root.resolve(),
        dataset_root=args.dataset_root.resolve(),
        pinyingpt_model=args.pinyingpt_model.resolve(),
        embedding_model=args.embedding_model.resolve(),
        pilot_root=args.m1_root.resolve(),
        t1_predictions=args.t1_predictions.resolve(),
    )
    runner = M2H5000Runner(
        m1,
        args.reranker_model.resolve(),
        args.output_root.resolve(),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    phases = {
        "prepare": runner.prepare,
        "benchmark": lambda: runner.benchmark(args.benchmark_queries),
        "dev-scores": runner.dev_scores,
        "tune": runner.tune,
        "test-scores": runner.test_scores,
        "evaluate": runner.evaluate,
        "smoke": runner.smoke,
        "all": runner.all,
    }
    print(json.dumps(phases[args.phase](), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
