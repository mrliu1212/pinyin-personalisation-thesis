"""Run the frozen Phase 4C correct-user and wrong-user comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.phase_04c_evaluation import evaluate_phase_04c, read_jsonl


DEFAULT_ZHU = Path(
    "data/processed/interactions/zhu_ziqing_simplified_rime/interactions.jsonl"
)
DEFAULT_LU = Path(
    "data/processed/interactions/lu_xun_simplified_rime/interactions.jsonl"
)
DEFAULT_OUTPUT = Path("results/experiments/phase_04c/evaluation.json")


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _print_condition(name: str, condition: dict[str, Any]) -> None:
    metrics = condition["metrics"]
    mean_rank = metrics["mean_target_rank"]
    print(
        f"{name}: Top-1={_percent(metrics['top1_accuracy'])}, "
        f"Top-3={_percent(metrics['top3_accuracy'])}, "
        f"Top-5={_percent(metrics['top5_accuracy'])}, "
        f"Top-10={_percent(metrics['top10_accuracy'])}, "
        f"MRR={metrics['mrr']:.4f}, "
        f"mean rank={'n/a' if mean_rank is None else f'{mean_rank:.4f}'}"
    )
    if "rank_changes" in condition:
        changes = condition["rank_changes"]
        print(
            "  rank changes: "
            f"improved={changes['improved']}, "
            f"unchanged={changes['unchanged']}, harmed={changes['harmed']}"
        )


def run(zhu_path: Path, lu_path: Path, output_path: Path) -> dict[str, Any]:
    result = evaluate_phase_04c(read_jsonl(zhu_path), read_jsonl(lu_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for subset_name, subset in result["subsets"].items():
        print(f"\n{subset_name.replace('_', ' ').title()}")
        _print_condition("Base", subset["base"])
        _print_condition("Correct-user", subset["correct_user"])
        _print_condition("Wrong-user", subset["wrong_user"])
    print(f"\nDetailed output: {output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zhu-interactions", type=Path, default=DEFAULT_ZHU)
    parser.add_argument("--lu-interactions", type=Path, default=DEFAULT_LU)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.zhu_interactions, args.lu_interactions, args.output)


if __name__ == "__main__":
    main()
