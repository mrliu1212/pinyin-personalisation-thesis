"""Bounded Personal Vocabulary state and PV0/PV1/PV2 operations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import statistics
from typing import Any, Mapping, Sequence

from src.personalisation.context_memory import (
    Candidate,
    PredictionQuery,
    cosine_similarity,
    frequency_support,
    normalize_generic_scores,
    rank_frequency,
    visible_same_pinyin_history,
)


FROZEN_FREQUENCY_LAMBDA = 4.0
FROZEN_M1_TOP_N = 5
PV1_K_GRID = (1, 3, 5)
PV1_LAMBDA_GRID = (0.5, 1.0, 2.0, 4.0)
PV2_CONTEXT_LAMBDA_GRID = (0.5, 1.0, 2.0, 4.0)
PERSONAL_VOCABULARY_VERSION = "bounded-h5000-exact-pinyin-v1"
PV2_CONTEXT_VERSION = "target-conditioned-positive-cosine-top5-normalized-v1"


@dataclass(frozen=True)
class PersonalLexiconEntry:
    user_id: str
    pinyin: tuple[str, ...]
    target: str
    count: int
    first_history_id: str
    last_history_id: str
    first_chronological_position: int
    last_chronological_position: int
    interaction_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "pinyin": list(self.pinyin),
            "target": self.target,
            "count": self.count,
            "first_history_id": self.first_history_id,
            "last_history_id": self.last_history_id,
            "first_chronological_position": self.first_chronological_position,
            "last_chronological_position": self.last_chronological_position,
            "interaction_ids": list(self.interaction_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PersonalLexiconEntry":
        return cls(
            user_id=str(value["user_id"]),
            pinyin=tuple(value["pinyin"]),
            target=str(value["target"]),
            count=int(value["count"]),
            first_history_id=str(value["first_history_id"]),
            last_history_id=str(value["last_history_id"]),
            first_chronological_position=int(value["first_chronological_position"]),
            last_chronological_position=int(value["last_chronological_position"]),
            interaction_ids=tuple(str(item) for item in value["interaction_ids"]),
        )


@dataclass(frozen=True)
class PersonalVocabularyState:
    """Prediction-visible shared state. Current Gold is deliberately absent."""

    row_id: str
    author: str
    pinyin: tuple[str, ...]
    generic_candidates: tuple[Candidate, ...]
    generic_frequency_ranked: tuple[Mapping[str, Any], ...]
    lexicon: tuple[PersonalLexiconEntry, ...]
    personal_only_targets: tuple[str, ...]
    personal_frequency_support: Mapping[str, float]
    personal_context_support: Mapping[str, float]
    generic_boundary_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "state_version": PERSONAL_VOCABULARY_VERSION,
            "context_version": PV2_CONTEXT_VERSION,
            "row_id": self.row_id,
            "author": self.author,
            "pinyin": list(self.pinyin),
            "generic_candidates": [
                {"text": value.text, "generic_rank": value.generic_rank, "generic_score": value.generic_score}
                for value in self.generic_candidates
            ],
            "generic_frequency_ranked": [dict(value) for value in self.generic_frequency_ranked],
            "lexicon": [value.to_dict() for value in self.lexicon],
            "personal_only_targets": list(self.personal_only_targets),
            "personal_frequency_support": dict(self.personal_frequency_support),
            "personal_context_support": dict(self.personal_context_support),
            "generic_boundary_score": self.generic_boundary_score,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PersonalVocabularyState":
        if value.get("state_version") != PERSONAL_VOCABULARY_VERSION or value.get("context_version") != PV2_CONTEXT_VERSION:
            raise RuntimeError("Personal Vocabulary state cache provenance differs")
        return cls(
            row_id=str(value["row_id"]),
            author=str(value["author"]),
            pinyin=tuple(value["pinyin"]),
            generic_candidates=tuple(
                Candidate(str(row["text"]), int(row["generic_rank"]), float(row["generic_score"]))
                for row in value["generic_candidates"]
            ),
            generic_frequency_ranked=tuple(dict(row) for row in value["generic_frequency_ranked"]),
            lexicon=tuple(PersonalLexiconEntry.from_dict(row) for row in value["lexicon"]),
            personal_only_targets=tuple(str(item) for item in value["personal_only_targets"]),
            personal_frequency_support={str(key): float(item) for key, item in value["personal_frequency_support"].items()},
            personal_context_support={str(key): float(item) for key, item in value["personal_context_support"].items()},
            generic_boundary_score=float(value["generic_boundary_score"]),
        )


def build_personal_lexicon(
    query: PredictionQuery,
    history: Sequence[Mapping[str, Any]],
) -> tuple[PersonalLexiconEntry, ...]:
    """Build a traceable lexicon from prediction-visible history only."""

    visible = visible_same_pinyin_history(query, history)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in visible:
        grouped.setdefault(str(row["target"]), []).append(row)
    entries = []
    for target, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: (int(row["chronological_position"]), str(row["row_id"])))
        entries.append(
            PersonalLexiconEntry(
                user_id=query.author,
                pinyin=query.pinyin,
                target=target,
                count=len(ordered),
                first_history_id=str(ordered[0]["row_id"]),
                last_history_id=str(ordered[-1]["row_id"]),
                first_chronological_position=int(ordered[0]["chronological_position"]),
                last_chronological_position=int(ordered[-1]["chronological_position"]),
                interaction_ids=tuple(str(row["row_id"]) for row in ordered),
            )
        )
    return tuple(sorted(entries, key=lambda entry: (-entry.count, entry.target)))


def personal_context_support(
    query: PredictionQuery,
    personal_entries: Sequence[PersonalLexiconEntry],
    history: Sequence[Mapping[str, Any]],
    embeddings: Mapping[str, Sequence[float]],
    *,
    top_n: int = FROZEN_M1_TOP_N,
) -> dict[str, float]:
    """Reuse M1 positive cosine with target-conditioned Top-5 normalization."""

    by_id = {str(row["row_id"]): row for row in history}
    query_vector = embeddings[query.context]
    raw: dict[str, float] = {}
    for entry in personal_entries:
        scored = []
        for interaction_id in entry.interaction_ids:
            row = by_id[interaction_id]
            similarity = cosine_similarity(query_vector, embeddings[str(row["context"])])
            scored.append(
                (
                    max(similarity, 0.0),
                    similarity,
                    int(row["chronological_position"]),
                    interaction_id,
                )
            )
        scored.sort(key=lambda value: (-value[1], value[2], value[3]))
        raw[entry.target] = sum(value[0] for value in scored[:top_n])
    total = sum(raw.values())
    return {target: value / total for target, value in raw.items()} if total else {}


def prepare_personal_vocabulary_state(
    query: PredictionQuery,
    candidates: Sequence[Candidate],
    history: Sequence[Mapping[str, Any]],
    embeddings: Mapping[str, Sequence[float]],
) -> PersonalVocabularyState:
    """Prepare PV0/PV1/PV2 state once without consulting current Gold."""

    visible = visible_same_pinyin_history(query, history)
    lexicon = build_personal_lexicon(query, visible)
    generic_texts = {candidate.text for candidate in candidates}
    personal = tuple(entry for entry in lexicon if entry.target not in generic_texts)
    personal_targets = tuple(entry.target for entry in personal)
    _, personal_frequency = frequency_support(personal_targets, visible)
    context = personal_context_support(query, personal, visible, embeddings) if personal else {}
    generic_frequency = rank_frequency(
        query,
        candidates,
        visible,
        lambda_frequency=FROZEN_FREQUENCY_LAMBDA,
    )
    return PersonalVocabularyState(
        row_id=query.row_id,
        author=query.author,
        pinyin=query.pinyin,
        generic_candidates=tuple(candidates),
        generic_frequency_ranked=generic_frequency,
        lexicon=lexicon,
        personal_only_targets=personal_targets,
        personal_frequency_support=personal_frequency,
        personal_context_support=context,
        generic_boundary_score=min(normalize_generic_scores(candidates)),
    )


def _merge(
    state: PersonalVocabularyState,
    *,
    k_pv: int,
    lambda_pv: float,
    lambda_ctx: float,
) -> tuple[dict[str, Any], ...]:
    if k_pv <= 0:
        raise ValueError("k_pv must be positive")
    entries = {entry.target: entry for entry in state.lexicon}
    rows: list[dict[str, Any]] = []
    for generic in state.generic_frequency_ranked:
        row = dict(generic)
        row["source"] = "generic"
        row["frequency_support"] = float(row["personal_score"])
        row["context_support"] = 0.0
        rows.append(row)
    for personal_rank, target in enumerate(state.personal_only_targets[:k_pv], start=1):
        frequency = float(state.personal_frequency_support.get(target, 0.0))
        context = float(state.personal_context_support.get(target, 0.0))
        rows.append(
            {
                "candidate": target,
                "generic_rank": None,
                "generic_score": None,
                "normalized_generic_score": state.generic_boundary_score,
                "personal_score": frequency,
                "frequency_support": frequency,
                "context_support": context,
                "frequency_count": entries[target].count,
                "final_score": state.generic_boundary_score + lambda_pv * frequency + lambda_ctx * context,
                "source": "personal_vocabulary",
                "personal_candidate_rank": personal_rank,
                "lexicon_provenance": entries[target].to_dict(),
            }
        )
    deduplicated: dict[str, dict[str, Any]] = {}
    for row in rows:
        target = str(row["candidate"])
        if target not in deduplicated or row["source"] == "generic":
            deduplicated[target] = row
    values = list(deduplicated.values())
    values.sort(
        key=lambda row: (
            -float(row["final_score"]),
            0 if row["source"] == "generic" else 1,
            int(row["generic_rank"] or row.get("personal_candidate_rank", 0)),
            str(row["candidate"]),
        )
    )
    values = values[:10]
    for rank, row in enumerate(values, start=1):
        row["rank"] = rank
    return tuple(values)


def rank_pv1(
    state: PersonalVocabularyState,
    *,
    k_pv: int,
    lambda_pv: float,
) -> tuple[dict[str, Any], ...]:
    return _merge(state, k_pv=k_pv, lambda_pv=lambda_pv, lambda_ctx=0.0)


def rank_pv2(
    state: PersonalVocabularyState,
    *,
    k_pv: int,
    lambda_pv: float,
    lambda_ctx: float,
) -> tuple[dict[str, Any], ...]:
    return _merge(state, k_pv=k_pv, lambda_pv=lambda_pv, lambda_ctx=lambda_ctx)


def lexicon_size_statistics(sizes: Sequence[int]) -> dict[str, float | int]:
    ordered = sorted(sizes)
    if not ordered:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0}
    index = max(0, min(len(ordered) - 1, int(0.9 * len(ordered) + 0.999999) - 1))
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p90": float(ordered[index]),
        "max": max(ordered),
    }


def transition_counts(
    baseline_ranks: Sequence[int | None],
    new_ranks: Sequence[int | None],
    *,
    helped_name: str = "helped",
    harmed_name: str = "harmed",
) -> dict[str, int]:
    if len(baseline_ranks) != len(new_ranks):
        raise ValueError("paired rank vectors differ in length")
    values = Counter()
    for baseline, new in zip(baseline_ranks, new_ranks):
        baseline_correct = baseline == 1
        new_correct = new == 1
        if not baseline_correct and new_correct:
            values[helped_name] += 1
        elif baseline_correct and not new_correct:
            values[harmed_name] += 1
        elif baseline_correct and new_correct:
            values["unchanged_correct"] += 1
        else:
            values["unchanged_wrong"] += 1
    return dict(values)
