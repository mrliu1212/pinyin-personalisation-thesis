"""Exact target-only teacher forcing for frozen PinyinGPT2-Concat semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from .pinyin_modes import PinyinRepresentation


IGNORE_INDEX = -100


@dataclass(frozen=True)
class PreparedConcatExample:
    row_id: str
    author: str
    context: str
    used_context: str
    original_context_tokens: int
    used_context_tokens: int
    context_truncated: bool
    target: str
    pinyin: PinyinRepresentation
    input_ids: tuple[int, ...]
    position_ids: tuple[int, ...]
    loss_positions: tuple[int, ...]
    target_token_ids: tuple[int, ...]

    def audit_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["pinyin"] = self.pinyin.to_dict()
        for key in ("input_ids", "position_ids", "loss_positions", "target_token_ids"):
            value[key] = list(value[key])
        return value


def truncate_recent_context(
    backend: Any,
    context: str,
    pinyin: Sequence[str],
) -> tuple[str, int, int, bool]:
    """Reproduce the standardized Generic 1024-position recent-context policy.

    The teacher-forcing input is ``prompt + target[:-1]``, which has the same
    maximum sequence length as the final generation forward.  Therefore the
    frozen generation budget ``n_positions - (2 + 2*m)`` is exact here too.
    """

    segments = backend.segment_pinyin(tuple(pinyin))
    original_ids = backend.tokenizer.encode(context, add_special_tokens=False)
    maximum = int(backend.model.config.n_positions)
    available = maximum - (2 + 2 * len(segments))
    if available < 0:
        raise ValueError("Pinyin target exceeds the model position limit")
    if len(original_ids) <= available:
        return context, len(original_ids), len(original_ids), False
    low, high = 0, len(context)
    while low < high:
        middle = (low + high) // 2
        length = len(backend.tokenizer.encode(context[middle:], add_special_tokens=False))
        if length <= available:
            high = middle
        else:
            low = middle + 1
    used = context[low:]
    used_tokens = len(backend.tokenizer.encode(used, add_special_tokens=False))
    return used, len(original_ids), used_tokens, True


def prepare_concat_example(
    backend: Any,
    row: Mapping[str, Any],
    representation: PinyinRepresentation,
) -> PreparedConcatExample:
    row_id = str(row["row_id"])
    if row_id != representation.row_id:
        raise ValueError("row and Pinyin representation IDs differ")
    if str(row.get("condition")) != "full_short" or str(row.get("target_type")) != "short":
        raise ValueError("Adapter V1 accepts only canonical Full+Short underlying rows")
    full_segments = tuple(map(str, row["pinyin_segments"]))
    if full_segments != representation.full_segments:
        raise ValueError("representation was not derived from the row's canonical Full Pinyin")
    target = str(row["gold"])
    if target != str(row.get("target", target)):
        raise ValueError("row target and Gold differ")
    if len(target) != len(representation.segmented_input):
        raise ValueError("target character count and Pinyin segment count differ")

    segments = backend.segment_pinyin(representation.segmented_input)
    target_ids = tuple(backend.tokenizer.convert_tokens_to_ids(list(target)))
    if backend.tokenizer.unk_token_id in target_ids:
        raise ValueError("target contains a tokenizer-unknown character")
    for character, token_id, segment in zip(target, target_ids, segments):
        if backend.tokenizer.convert_ids_to_tokens(token_id) != character:
            raise ValueError(f"target character is not an exact tokenizer token: {character!r}")
        if token_id not in backend.allowed_token_ids[segment]:
            raise ValueError(f"Gold is incompatible with Pinyin segment {segment!r}")

    context = str(row["context"])
    used, original_tokens, used_tokens, truncated = truncate_recent_context(
        backend, context, segments
    )
    prompt_ids, prompt_positions = backend._prompt(used, segments)
    output_positions = tuple(
        prompt_positions[-len(segments) - 1] + offset
        for offset in range(len(target_ids))
    )

    # The last Gold token is a label, not an input.  Each loss position predicts
    # the corresponding Gold token, exactly matching score_candidates().
    input_ids = tuple(prompt_ids) + target_ids[:-1]
    position_ids = tuple(prompt_positions) + output_positions[:-1]
    loss_positions = tuple(len(prompt_ids) - 1 + step for step in range(len(target_ids)))
    if len(input_ids) != len(position_ids):
        raise AssertionError("input/position alignment failed")
    if loss_positions[-1] >= len(input_ids):
        raise AssertionError("target supervision points outside the causal input")
    if len(input_ids) > int(backend.model.config.n_positions):
        raise AssertionError("teacher-forcing input exceeds n_positions")

    return PreparedConcatExample(
        row_id=row_id,
        author=str(row["author"]),
        context=context,
        used_context=used,
        original_context_tokens=original_tokens,
        used_context_tokens=used_tokens,
        context_truncated=truncated,
        target=target,
        pinyin=representation,
        input_ids=input_ids,
        position_ids=position_ids,
        loss_positions=loss_positions,
        target_token_ids=target_ids,
    )


def collate_concat_examples(
    examples: Sequence[PreparedConcatExample],
    *,
    pad_token_id: int,
    device: torch.device | str,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    maximum = max(len(example.input_ids) for example in examples)
    batch_size = len(examples)
    input_ids = torch.full((batch_size, maximum), pad_token_id, dtype=torch.long)
    position_ids = torch.zeros((batch_size, maximum), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, maximum), dtype=torch.long)
    loss_labels = torch.full((batch_size, maximum), IGNORE_INDEX, dtype=torch.long)
    for row_index, example in enumerate(examples):
        length = len(example.input_ids)
        input_ids[row_index, :length] = torch.tensor(example.input_ids, dtype=torch.long)
        position_ids[row_index, :length] = torch.tensor(example.position_ids, dtype=torch.long)
        attention_mask[row_index, :length] = 1
        for position, token_id in zip(example.loss_positions, example.target_token_ids):
            loss_labels[row_index, position] = token_id
    return {
        "input_ids": input_ids.to(device),
        "position_ids": position_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "loss_labels": loss_labels.to(device),
        "row_ids": [example.row_id for example in examples],
        "supervised_tokens": sum(len(example.target_token_ids) for example in examples),
    }


def compute_target_loss(model: Any, batch: Mapping[str, Any]) -> torch.Tensor:
    output = model(
        input_ids=batch["input_ids"],
        position_ids=batch["position_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    labels = batch["loss_labels"]
    return F.cross_entropy(
        output.logits.float().reshape(-1, output.logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="mean",
    )


def per_example_log_probabilities(model: Any, batch: Mapping[str, Any]) -> list[float]:
    """Return summed Gold log-probability for exact scorer regression gates."""

    with torch.no_grad():
        output = model(
            input_ids=batch["input_ids"],
            position_ids=batch["position_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        log_probabilities = torch.log_softmax(output.logits.float(), dim=-1)
        labels = batch["loss_labels"]
        totals = []
        for row_index in range(labels.shape[0]):
            mask = labels[row_index] != IGNORE_INDEX
            selected = log_probabilities[row_index, mask]
            targets = labels[row_index, mask]
            totals.append(selected.gather(1, targets.unsqueeze(1)).sum().item())
    return totals
