"""Exact-scored External Memory candidate recovery and frequency fusion."""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence


def unified_pool(
    generic_candidates: Sequence[Mapping[str, Any]],
    recovered_scores: Sequence[Mapping[str, Any]],
    *,
    k_recovery: int,
) -> tuple[dict[str, Any], ...]:
    """Combine Frozen Generic surface with up to K exact-scored personal candidates."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for candidate in generic_candidates:
        text = str(candidate["text"])

        if text in seen:
            continue

        seen.add(text)

        rows.append(
            {
                "candidate": text,
                "source": "generic",
                "generic_rank": int(candidate["rank"]),
                "personal_rank": None,
                "log_probability": float(
                    candidate["log_probability"]
                ),
            }
        )

    for candidate in recovered_scores:
        personal_rank = int(
            candidate["personal_candidate_rank"]
        )

        if personal_rank > k_recovery:
            continue

        text = str(candidate["candidate"])

        if text in seen:
            raise AssertionError(
                f"Recovered candidate already exists in Generic: {text!r}"
            )

        seen.add(text)

        rows.append(
            {
                "candidate": text,
                "source": "personal_memory",
                "generic_rank": None,
                "personal_rank": personal_rank,
                "log_probability": float(
                    candidate["fixed_log_probability"]
                ),
            }
        )

    return tuple(rows)


def generic_reference_normalisation(
    pool: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Normalize all candidates using only the original Generic score surface."""

    generic_scores = [
        float(row["log_probability"])
        for row in pool
        if row["source"] == "generic"
    ]

    if not generic_scores:
        raise ValueError("Unified pool contains no Generic candidates")

    mean = statistics.fmean(generic_scores)
    deviation = statistics.pstdev(generic_scores)

    if deviation == 0.0:
        return {
            str(row["candidate"]): 0.0
            for row in pool
        }

    return {
        str(row["candidate"]): (
            float(row["log_probability"]) - mean
        ) / deviation
        for row in pool
    }


def frequency_support(
    pool: Sequence[Mapping[str, Any]],
    history_counts: Mapping[str, int],
) -> dict[str, float]:
    """Normalized log(1+count) over the complete unified candidate pool."""

    raw = {
        str(row["candidate"]): math.log1p(
            int(
                history_counts.get(
                    str(row["candidate"]),
                    0,
                )
            )
        )
        for row in pool
    }

    maximum = max(raw.values(), default=0.0)

    if maximum == 0.0:
        return {
            candidate: 0.0
            for candidate in raw
        }

    return {
        candidate: value / maximum
        for candidate, value in raw.items()
    }


def _finish_ranking(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    rows.sort(
        key=lambda row: (
            -float(row["final_score"]),
            -float(row["log_probability"]),
            0 if row["source"] == "generic" else 1,
            (
                int(row["generic_rank"])
                if row["generic_rank"] is not None
                else 10**9
            ),
            (
                int(row["personal_rank"])
                if row["personal_rank"] is not None
                else 10**9
            ),
            str(row["candidate"]),
        )
    )

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return tuple(rows[:10])


def rank_recovery_only(
    pool: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """R: exact Frozen PinyinGPT score only."""

    rows = []

    for row in pool:
        value = dict(row)
        value["frequency_count"] = None
        value["frequency_support"] = 0.0
        value["normalized_generic_score"] = None
        value["final_score"] = float(
            row["log_probability"]
        )
        rows.append(value)

    return _finish_ranking(rows)


def rank_recovery_frequency(
    pool: Sequence[Mapping[str, Any]],
    history_counts: Mapping[str, int],
    *,
    lambda_frequency: float,
) -> tuple[dict[str, Any], ...]:
    """R+F: Generic-reference z-score plus normalized log frequency."""

    generic_component = (
        generic_reference_normalisation(pool)
    )

    personal_component = frequency_support(
        pool,
        history_counts,
    )

    rows = []

    for row in pool:
        candidate = str(row["candidate"])

        value = dict(row)
        value["normalized_generic_score"] = (
            generic_component[candidate]
        )
        value["frequency_count"] = int(
            history_counts.get(candidate, 0)
        )
        value["frequency_support"] = (
            personal_component[candidate]
        )
        value["final_score"] = (
            generic_component[candidate]
            + lambda_frequency
            * personal_component[candidate]
        )

        rows.append(value)

    return _finish_ranking(rows)


def rank_of(
    rows: Sequence[Mapping[str, Any]],
    target: str,
) -> int | None:
    return next(
        (
            int(row["rank"])
            for row in rows
            if str(row["candidate"]) == target
        ),
        None,
    )
