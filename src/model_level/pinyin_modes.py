"""Deterministic Full/Initial/Mixed Pinyin manifestations for Adapter training.

Each underlying interaction is exposed exactly once per epoch.  The default
3:1:2 mode weights mean Full=1/2, Initial=1/6, and Mixed=1/3.  Mixed masks are
uniform over every non-degenerate binary mask, so a multi-syllable Mixed row
always contains at least one Full and one Initial unit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Literal, Sequence


PinyinMode = Literal["full", "initial", "mixed"]
DEFAULT_MODE_WEIGHTS: tuple[int, int, int] = (3, 1, 2)
MANIFEST_SCHEMA_VERSION = 1


def _stable_integer(*parts: object, seed: int) -> int:
    payload = json.dumps(
        {"parts": parts, "seed": seed},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _validate_full_segments(segments: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(value).strip().lower() for value in segments)
    if not values or any(re.fullmatch(r"[a-zv]+", value) is None for value in values):
        raise ValueError("Full Pinyin segments must be non-empty Latin syllables")
    return values


def full_to_initial(segments: Sequence[str]) -> tuple[str, ...]:
    """Apply the official PinyinGPT ``abbr_mode=full`` first-letter rule."""

    return tuple(value[0] for value in _validate_full_segments(segments))


def select_mode(
    row_id: str,
    epoch: int,
    *,
    seed: int,
    weights: tuple[int, int, int] = DEFAULT_MODE_WEIGHTS,
) -> PinyinMode:
    if not row_id:
        raise ValueError("row_id must be non-empty")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if len(weights) != 3 or any(not isinstance(value, int) or value < 0 for value in weights):
        raise ValueError("mode weights must be three non-negative integers")
    total = sum(weights)
    if total <= 0:
        raise ValueError("at least one mode weight must be positive")
    slot = _stable_integer("pinyin-mode", row_id, epoch, seed=seed) % total
    if slot < weights[0]:
        return "full"
    if slot < weights[0] + weights[1]:
        return "initial"
    return "mixed"


def _mixed_mask(length: int, row_id: str, epoch: int, *, seed: int) -> tuple[bool, ...]:
    """Return True for abbreviated positions.

    For length >= 2, the integer code is uniform over [1, 2**length-2], which
    is exactly independent Bernoulli(0.5) conditioned on a non-degenerate mask.
    A one-syllable requested Mixed row uses a deterministic 50/50 fallback.
    """

    if length < 1:
        raise ValueError("Pinyin length must be positive")
    value = _stable_integer("mixed-mask", row_id, epoch, seed=seed)
    if length == 1:
        return (bool(value % 2),)
    code = 1 + value % ((1 << length) - 2)
    mask = tuple(bool((code >> index) & 1) for index in range(length))
    if not any(mask) or all(mask):
        raise AssertionError("conditioned Mixed mask is degenerate")
    return mask


@dataclass(frozen=True)
class PinyinRepresentation:
    schema_version: int
    row_id: str
    epoch: int
    seed: int
    requested_mode: PinyinMode
    effective_mode: PinyinMode
    full_segments: tuple[str, ...]
    segmented_input: tuple[str, ...]
    abbreviated_mask: tuple[bool, ...]
    abbreviated_positions: tuple[int, ...]
    mode_weights: tuple[int, int, int]
    selection_policy: str

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for key in (
            "full_segments",
            "segmented_input",
            "abbreviated_mask",
            "abbreviated_positions",
            "mode_weights",
        ):
            value[key] = list(value[key])
        return value


def choose_pinyin_representation(
    full_segments: Sequence[str],
    *,
    row_id: str,
    epoch: int,
    seed: int,
    forced_mode: PinyinMode | None = None,
    weights: tuple[int, int, int] = DEFAULT_MODE_WEIGHTS,
) -> PinyinRepresentation:
    """Choose one reproducible Pinyin manifestation for an underlying row."""

    full = _validate_full_segments(full_segments)
    mode = forced_mode or select_mode(row_id, epoch, seed=seed, weights=weights)
    if mode not in {"full", "initial", "mixed"}:
        raise ValueError(f"unsupported Pinyin mode: {mode!r}")

    if mode == "full":
        mask = (False,) * len(full)
    elif mode == "initial":
        mask = (True,) * len(full)
    else:
        mask = _mixed_mask(len(full), row_id, epoch, seed=seed)

    initials = full_to_initial(full)
    segmented = tuple(
        initials[index] if abbreviated else full[index]
        for index, abbreviated in enumerate(mask)
    )
    abbreviated_positions = tuple(index for index, abbreviated in enumerate(mask) if abbreviated)
    if not abbreviated_positions:
        effective: PinyinMode = "full"
    elif len(abbreviated_positions) == len(full):
        effective = "initial"
    else:
        effective = "mixed"

    return PinyinRepresentation(
        schema_version=MANIFEST_SCHEMA_VERSION,
        row_id=row_id,
        epoch=epoch,
        seed=seed,
        requested_mode=mode,
        effective_mode=effective,
        full_segments=full,
        segmented_input=segmented,
        abbreviated_mask=mask,
        abbreviated_positions=abbreviated_positions,
        mode_weights=weights,
        selection_policy=(
            "sha256_mode_3_1_2;mixed_uniform_non_degenerate_binary_mask;"
            "single_syllable_mixed_sha256_50_50_fallback"
        ),
    )
