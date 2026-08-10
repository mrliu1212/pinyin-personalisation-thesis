"""Run the Phase 3 chronological synthetic evaluation."""

from src.evaluation import ConditionEvaluation, evaluate_chronologically

from .synthetic_phase_03 import (
    WRONG_USER_BY_USER,
    build_base_ranker,
    build_interactions,
)


def print_condition(label: str, condition: ConditionEvaluation) -> None:
    metrics = condition.metrics
    mean_rank = (
        "n/a" if metrics.mean_target_rank is None else f"{metrics.mean_target_rank:.3f}"
    )
    print(f"{label}:")
    print(f"  Top-1: {metrics.top1_accuracy:.3f}")
    print(f"  Top-3: {metrics.top3_accuracy:.3f}")
    print(f"  MRR: {metrics.mrr:.3f}")
    print(f"  Mean target rank: {mean_rank}")
    if condition.reranking_counts is not None:
        counts = condition.reranking_counts
        print(
            "  Reranking: "
            f"helpful={counts.helpful}, harmful={counts.harmful}, "
            f"unchanged={counts.unchanged}"
        )


def main() -> None:
    comparison = evaluate_chronologically(
        build_interactions(),
        build_base_ranker(),
        WRONG_USER_BY_USER,
        alpha=0.4,
    )
    print_condition("Base", comparison.base)
    print_condition("Correct-user personalised", comparison.correct_user)
    print_condition("Wrong-user personalised", comparison.wrong_user)


if __name__ == "__main__":
    main()

