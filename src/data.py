"""Core data records used by the minimal pipeline."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Interaction:
    user_id: str
    timestamp: datetime
    context: str
    pinyin: str
    target_candidate: str


@dataclass(frozen=True)
class BaseCandidate:
    text: str
    base_score: float


@dataclass(frozen=True)
class RankedCandidate:
    text: str
    base_score: float
    personal_score: float
    final_score: float
    global_evidence: float = 0.0
    pinyin_evidence: float = 0.0
    context_evidence: float = 0.0
