"""Run the minimal pipeline on a manually inspectable synthetic example."""

from datetime import datetime, timezone

from src.base_ranker import InMemoryBaseRanker
from src.data import BaseCandidate, Interaction
from src.personal_model import FrequencyPersonalModel
from src.reranker import LinearReranker


base_ranker = InMemoryBaseRanker(
    {
        "shiyong": [
            BaseCandidate("实用", 0.90),
            BaseCandidate("使用", 0.80),
            BaseCandidate("试用", 0.70),
        ]
    }
)
history = [
    Interaction("user-a", datetime(2026, 1, day, tzinfo=timezone.utc), "我们可以", "shiyong", "使用")
    for day in range(1, 4)
]
model = FrequencyPersonalModel().fit(history, user_id="user-a")
ranking = LinearReranker(base_ranker, model, alpha=0.4).rank(
    context="我们可以", pinyin="shiyong", top_k=3
)

for position, candidate in enumerate(ranking, start=1):
    print(
        position,
        candidate.text,
        f"base={candidate.base_score:.2f}",
        f"global={candidate.global_evidence:.0f}",
        f"pinyin={candidate.pinyin_evidence:.0f}",
        f"context={candidate.context_evidence:.0f}",
        f"personal={candidate.personal_score:.2f}",
        f"final={candidate.final_score:.2f}",
    )
