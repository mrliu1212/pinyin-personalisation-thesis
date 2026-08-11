# SPDX-License-Identifier: GPL-3.0-only
"""Small prompt assets directly integrated from HuoziIME.

Source: Shan-HIT/HuoziIME at
63f249e711f6501169e6baafec7e12318b3c765b. This file is deliberately isolated
to make the GPL provenance of the copied prompt/template text unambiguous.
"""

from __future__ import annotations


DEFAULT_BUSINESS_SYSTEM_PROMPT = """你是职场沟通与决策的得力干将（偏总监/高级经理风格）。
原则：平视、专业、直接、结果导向；不卑微、不客服腔、不复读。
输出要求：
- 先给结论/建议，再给1-3条关键理由或风险点。
- 语句短、信息密度高；必要时用条列。
- 避免空泛鼓励，强调可执行下一步。
重要：当前任务是‘对话续写/补全’。
- 如果 <instruction> 里给了‘要补全的前缀’，你只输出续写部分，不要复述前缀。
- 如果 <instruction> 里的前缀为空，你输出一条完整回复。
- 只有在必须依赖记忆库中的过去信息、且当前 <memory> 为空/缺失导致无法继续时，才允许输出工具调用：<MEM_RETRIEVAL> query="..." </MEM_RETRIEVAL>。
- 若 <memory> 内出现 <NO_MEM>，表示已尝试检索但无结果，你必须继续正常续写，禁止输出任何 <MEM_RETRIEVAL> 工具调用。
- 若不需要检索，正常续写即可，不要输出任何工具标记。"""


MEMORY_WORKER_SYSTEM_PROMPT = """[MEMORY_WORKER] 结构化提取记忆。
输出格式：JSON。若无有效信息，输出<NO_MEM>。
字段约束：
- 必有：summary
- 可选：datetime, location, participants(数组), item, detail
决策规则：
- 只在出现明确可执行计划/约定时输出 JSON（只包含原文中能直接支持的字段）。
- 其余情况一律输出<NO_MEM>。
严格要求：只输出 JSON 或 <NO_MEM>，不要任何额外解释。"""


def build_typing_prompt(
    preceding_text: str,
    *,
    memory: str | None = None,
    external_context: str | None = None,
    history: str | None = None,
) -> str:
    prefix = preceding_text[-100:]
    history_text = history.strip() if history and history.strip() else "无"
    last_message = (
        f"[对方]: {external_context.strip()}"
        if external_context and external_context.strip()
        else "无"
    )
    memory_text = memory.strip() if memory and memory.strip() else "无"
    user = (
        "<env>\n<history>\n"
        f"{history_text}\n</history>\n<last_msg>\n{last_message}\n</last_msg>\n"
        f"<memory>\n{memory_text}\n</memory></env>\n<instruction>\n"
        f"请根据记忆和LastMsg，补全我的回复：\n{prefix}\n</instruction>"
    )
    return (
        f"<|im_start|>system\n{DEFAULT_BUSINESS_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n{prefix}"
    )


def build_memory_worker_prompt(user_text: str) -> str:
    return (
        f"<|im_start|>system\n{MEMORY_WORKER_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
