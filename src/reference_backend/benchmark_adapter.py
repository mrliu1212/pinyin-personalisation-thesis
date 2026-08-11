"""Strict input-only adapter from frozen Phase 4C interactions to Phase 4F."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from src.phase_04c_evaluation import parse_work_date


@dataclass(frozen=True)
class ReferenceRequest:
    benchmark_interaction_id: str
    target_user_id: str
    personal_state_user_id: str
    preceding_text: str
    pinyin_or_keystrokes: str
    top_k: int
    external_context: None
    target: str
    chronological_position: str
    work_id: str


@dataclass(frozen=True)
class TrainingTrajectory:
    user_id: str
    work_id: str
    chronological_position: str
    text: str
    source_interaction_ids: tuple[str, ...]


class Phase04FBenchmarkAdapter:
    CONTEXT_CHARACTERS = 100
    TRAJECTORY_CHARACTERS = 4000

    def request(
        self,
        record: dict[str, Any],
        *,
        personal_state_user_id: str,
        top_k: int = 10,
        external_context: str | None = None,
    ) -> ReferenceRequest:
        if external_context is not None:
            raise ValueError("Phase 4F Zhu/Lu benchmark is input-only")
        raw_context = str(record["raw_context"])
        position = (
            f"{record['work_date']}|{int(record['source_start_offset']):012d}|"
            f"{record['interaction_id']}"
        )
        return ReferenceRequest(
            benchmark_interaction_id=record["interaction_id"],
            target_user_id=record["author_id"],
            personal_state_user_id=personal_state_user_id,
            preceding_text=raw_context[-self.CONTEXT_CHARACTERS :],
            pinyin_or_keystrokes=record["pinyin"],
            top_k=top_k,
            external_context=None,
            target=record["target_candidate"],
            chronological_position=position,
            work_id=record["work_id"],
        )

    def training_trajectories(
        self,
        records: Sequence[dict[str, Any]],
        *,
        user_id: str,
    ) -> tuple[TrainingTrajectory, ...]:
        if any(record["author_id"] != user_id for record in records):
            raise ValueError("training trajectories cannot mix users")
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            grouped.setdefault(record["work_id"], []).append(record)
        trajectories = []
        for work_id, work_records in grouped.items():
            ordered = sorted(
                work_records,
                key=lambda item: (int(item["source_start_offset"]), item["interaction_id"]),
            )
            final = ordered[-1]
            complete = str(final["raw_context"]) + str(final["target_candidate"])
            retained_start = max(0, int(final["source_end_offset"]) - self.TRAJECTORY_CHARACTERS)
            source_ids = tuple(
                record["interaction_id"]
                for record in ordered
                if int(record["source_end_offset"]) > retained_start
            )
            trajectories.append(
                TrainingTrajectory(
                    user_id=user_id,
                    work_id=work_id,
                    chronological_position=(
                        f"{final['work_date']}|{int(final['source_end_offset']):012d}|{work_id}"
                    ),
                    text=complete[-self.TRAJECTORY_CHARACTERS :],
                    source_interaction_ids=source_ids,
                )
            )
        return tuple(
            sorted(
                trajectories,
                key=lambda item: (
                    parse_work_date(item.chronological_position.split("|", 1)[0]),
                    item.chronological_position,
                ),
            )
        )


def target_rank(target: str, candidates: Sequence[str]) -> int | None:
    try:
        return list(candidates).index(target) + 1
    except ValueError:
        return None
