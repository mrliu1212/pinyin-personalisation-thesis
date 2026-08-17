"""Run or resume frozen Personal Vocabulary PV0/PV1/PV2 H5000 evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.personalisation.h5000 import H5000Runner
from src.personalisation.pv_h5000 import PersonalVocabularyH5000Runner


ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--phase", choices=("prepare", "pv0", "dev-states", "tune", "evaluate", "smoke", "all"), required=True)
    value.add_argument("--root", type=Path, default=ROOT)
    value.add_argument("--dataset-root", type=Path, required=True)
    value.add_argument("--pinyingpt-model", type=Path, required=True)
    value.add_argument("--embedding-model", type=Path, required=True)
    value.add_argument("--t1-predictions", type=Path, required=True)
    value.add_argument("--m1-root", type=Path, default=ROOT / "results/personalisation/pilot_a_context_memory")
    value.add_argument("--m2-root", type=Path, default=ROOT / "results/personalisation/m2_h5000")
    value.add_argument("--output-root", type=Path, default=ROOT / "results/personalisation/personal_vocabulary_h5000")
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
    runner = PersonalVocabularyH5000Runner(m1, args.m2_root.resolve(), args.output_root.resolve())
    phases = {
        "prepare": runner.prepare,
        "pv0": runner.pv0,
        "dev-states": lambda: {"status": "complete", "rows": len(runner.dev_states()[1])},
        "tune": runner.tune,
        "evaluate": runner.evaluate,
        "smoke": runner.smoke,
        "all": runner.all,
    }
    print(json.dumps(phases[args.phase](), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
