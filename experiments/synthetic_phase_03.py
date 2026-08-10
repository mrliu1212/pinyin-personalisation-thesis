"""Deterministic multi-user data for validating the Phase 3 evaluator."""

from datetime import datetime, timezone

from src.base_ranker import InMemoryBaseRanker
from src.data import BaseCandidate, Interaction


def build_base_ranker() -> InMemoryBaseRanker:
    return InMemoryBaseRanker(
        {
            "shiyong": [
                BaseCandidate("实用", 0.90),
                BaseCandidate("使用", 0.80),
                BaseCandidate("试用", 0.70),
            ],
            "xing": [
                BaseCandidate("行", 0.90),
                BaseCandidate("型", 0.80),
                BaseCandidate("性", 0.70),
            ],
            "hang": [
                BaseCandidate("行", 0.90),
                BaseCandidate("航", 0.80),
                BaseCandidate("杭", 0.70),
            ],
            "zhongyao": [
                BaseCandidate("重要", 0.90),
                BaseCandidate("中药", 0.80),
            ],
        }
    )


def build_interactions() -> list[Interaction]:
    """Return chronological preferences for two deliberately distinct users."""

    def item(
        day: int, user: str, context: str, pinyin: str, candidate: str
    ) -> Interaction:
        return Interaction(
            user_id=user,
            timestamp=datetime(2026, 2, day, tzinfo=timezone.utc),
            context=context,
            pinyin=pinyin,
            target_candidate=candidate,
        )

    return [
        # User A: early "行" selections make it globally strong. Later exact
        # context evidence for "型" competes with that broader signal.
        item(1, "user-a", "银行这一", "hang", "行"),
        item(2, "user-a", "排行这一", "hang", "行"),
        item(3, "user-a", "同行这一", "hang", "行"),
        item(4, "user-a", "这个款式", "xing", "型"),
        item(5, "user-a", "这个款式", "xing", "型"),
        item(6, "user-a", "这样也", "xing", "行"),
        item(7, "user-a", "这个款式", "xing", "型"),
        item(8, "user-a", "我们可以", "shiyong", "使用"),
        item(9, "user-a", "我们可以", "shiyong", "使用"),
        item(10, "user-a", "这个软件很", "shiyong", "实用"),
        item(11, "user-a", "这个软件很", "shiyong", "实用"),
        item(12, "user-a", "我们可以", "shiyong", "使用"),
        item(13, "user-a", "这个软件很", "shiyong", "实用"),
        item(14, "user-a", "这件事很", "zhongyao", "重要"),
        item(15, "user-a", "这件事很", "zhongyao", "重要"),
        item(16, "user-a", "需要服用", "zhongyao", "中药"),
        item(17, "user-a", "需要服用", "zhongyao", "中药"),
        # User B supplies intentionally different preferences for the
        # wrong-user control.
        item(1, "user-b", "航空", "hang", "航"),
        item(2, "user-b", "航空", "hang", "航"),
        item(3, "user-b", "人的个", "xing", "性"),
        item(4, "user-b", "人的个", "xing", "性"),
        item(5, "user-b", "这样可", "xing", "行"),
        item(6, "user-b", "这样可", "xing", "行"),
        item(7, "user-b", "我们可以", "shiyong", "实用"),
        item(8, "user-b", "我们可以", "shiyong", "实用"),
        item(9, "user-b", "这个软件很", "shiyong", "使用"),
        item(10, "user-b", "这个软件很", "shiyong", "使用"),
        item(11, "user-b", "我们可以", "shiyong", "实用"),
        item(12, "user-b", "这个软件很", "shiyong", "使用"),
        item(13, "user-b", "需要服用", "zhongyao", "中药"),
        item(14, "user-b", "需要服用", "zhongyao", "中药"),
        item(15, "user-b", "这件事很", "zhongyao", "重要"),
        item(16, "user-b", "这件事很", "zhongyao", "重要"),
    ]


WRONG_USER_BY_USER = {"user-a": "user-b", "user-b": "user-a"}

