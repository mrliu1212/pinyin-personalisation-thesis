"""Audit, smoke, run, or resume the frozen reranking-personalisation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.personalisation.reranking_matrix import RerankingMatrixRunner


ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--phase", choices=("audit", "smoke", "run", "finalize"), required=True)
    value.add_argument("--root", type=Path, default=ROOT)
    value.add_argument("--dataset-root", type=Path, required=True)
    value.add_argument("--pinyingpt-model", type=Path, required=True)
    value.add_argument("--embedding-model", type=Path, required=True)
    value.add_argument("--reranker-model", type=Path, required=True)
    value.add_argument("--t1-predictions", type=Path, required=True)
    value.add_argument("--output-root", type=Path, default=ROOT / "results/personalisation/reranking_matrix")
    value.add_argument("--m1-root", type=Path, default=ROOT / "results/personalisation/pilot_a_context_memory")
    value.add_argument("--m2-root", type=Path, default=ROOT / "results/personalisation/m2_h5000")
    value.add_argument("--batch-size", type=int, default=32)
    value.add_argument("--max-length", type=int, default=512)
    return value


def main() -> None:
    args = parser().parse_args()
    runner = RerankingMatrixRunner(
        root=args.root.resolve(),
        dataset_root=args.dataset_root.resolve(),
        pinyingpt_model=args.pinyingpt_model.resolve(),
        embedding_model=args.embedding_model.resolve(),
        reranker_model=args.reranker_model.resolve(),
        t1_predictions=args.t1_predictions.resolve(),
        output_root=args.output_root.resolve(),
        m1_root=args.m1_root.resolve(),
        m2_root=args.m2_root.resolve(),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    action = {"audit": runner.audit, "smoke": runner.smoke, "run": runner.all, "finalize": runner.finalize}[args.phase]
    print(json.dumps(action(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
