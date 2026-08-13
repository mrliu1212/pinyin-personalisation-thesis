"""Faithful modern-inference adapter for the published PinyinGPT2-Concat model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Sequence
import unicodedata


CHECKPOINT_ID = "aihijo/transformers4ime-pinyingpt-concat"
CHECKPOINT_REVISION = "76dd20dc92d8236a350fb732e99dde6fa15e2263"
OFFICIAL_CODE_REVISION = "8f1573ed0bd4d1f3d8d3f10a05f7e870725646f1"


@dataclass(frozen=True)
class CandidateScore:
    text: str
    rank: int
    log_probability: float
    mean_log_probability: float
    compatible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PinyinGPTResult:
    context: str
    typed_pinyin: str
    segmented_pinyin: tuple[str, ...]
    model_input_tokens: tuple[str, ...]
    candidates: tuple[CandidateScore, ...]
    beam_size: int
    runtime_device: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["segmented_pinyin"] = list(self.segmented_pinyin)
        value["model_input_tokens"] = list(self.model_input_tokens)
        value["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return value


class PinyinGPTConcatBackend:
    """Character-level constrained decoding with the official Concat input layout.

    The ACL 2022 work assumes oracle Pinyin segmentation. A raw, unspaced input
    is accepted only when the published Pinyin vocabulary gives one segmentation;
    callers must add spaces when multiple segmentations are possible.
    """

    def __init__(self, checkpoint: Path, device: str = "auto") -> None:
        self.checkpoint = checkpoint.resolve()
        if not (self.checkpoint / "pytorch_model.bin").is_file():
            raise FileNotFoundError(
                f"PinyinGPT2-Concat checkpoint is incomplete: {self.checkpoint}"
            )
        try:
            import torch
            from transformers import BertTokenizer, GPT2Config, GPT2LMHeadModel
        except ImportError as error:  # pragma: no cover - exercised by environment setup
            raise RuntimeError(
                "PinyinGPT dependencies are missing; install requirements-pinyingpt.txt"
            ) from error

        self.torch = torch

        additional_tokens = json.loads(
            (self.checkpoint / "additional_special_tokens.json").read_text(
                encoding="utf-8"
            )
        )
        self.tokenizer = BertTokenizer.from_pretrained(self.checkpoint)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": additional_tokens}
        )

        config = GPT2Config.from_pretrained(self.checkpoint)
        config.vocab_size = len(self.tokenizer)
        self.model = GPT2LMHeadModel(config)
        state = torch.load(
            self.checkpoint / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )
        obsolete_buffers = [
            name
            for name in state
            if name.endswith(".attn.bias") or name.endswith(".attn.masked_bias")
        ]
        for name in obsolete_buffers:
            state.pop(name)
        self.model.load_state_dict(state, strict=True)
        self.model.tie_weights()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.device = torch.device(device)
        self.model.to(self.device).eval()

        self.pinyin2char: dict[str, list[str]] = json.loads(
            (self.checkpoint / "pinyin2char.json").read_text(encoding="utf-8")
        )
        self.allowed_token_ids = self._build_allowed_token_ids()
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())

    def _build_allowed_token_ids(self) -> dict[str, tuple[int, ...]]:
        allowed: dict[str, tuple[int, ...]] = {}
        for pinyin, characters in self.pinyin2char.items():
            ids = []
            for character in characters:
                token_id = self.tokenizer.convert_tokens_to_ids(character)
                if (
                    token_id != self.tokenizer.unk_token_id
                    and self.tokenizer.convert_ids_to_tokens(token_id) == character
                ):
                    ids.append(token_id)
            allowed[pinyin] = tuple(sorted(set(ids)))
        return allowed

    @staticmethod
    def _normalize_pinyin(value: str) -> str:
        value = value.lower().replace("ü", "v").replace("u:", "v")
        value = "".join(
            character
            for character in unicodedata.normalize("NFD", value)
            if not unicodedata.combining(character)
        )
        value = re.sub(r"[1-5]", "", value)
        value = value.replace("'", " ")
        if not re.fullmatch(r"[a-zv\s]+", value):
            raise ValueError(f"unsupported Pinyin input: {value!r}")
        return " ".join(value.split())

    def segment_pinyin(self, typed_pinyin: str | Sequence[str]) -> tuple[str, ...]:
        if not isinstance(typed_pinyin, str):
            segments = tuple(self._normalize_pinyin(item) for item in typed_pinyin)
        else:
            normalized = self._normalize_pinyin(typed_pinyin)
            if " " in normalized:
                segments = tuple(normalized.split())
            else:
                paths: dict[int, list[tuple[str, ...]]] = {0: [()]}
                for start in range(len(normalized)):
                    for prefix in paths.get(start, []):
                        for end in range(start + 1, len(normalized) + 1):
                            syllable = normalized[start:end]
                            if syllable in self.allowed_token_ids:
                                paths.setdefault(end, []).append(prefix + (syllable,))
                                if len(paths[end]) > 2:
                                    paths[end] = paths[end][:2]
                alternatives = paths.get(len(normalized), [])
                if not alternatives:
                    raise ValueError(f"Pinyin cannot be segmented: {typed_pinyin!r}")
                if len(alternatives) != 1:
                    rendered = [" ".join(item) for item in alternatives]
                    raise ValueError(
                        "ambiguous Pinyin segmentation; supply spaces explicitly: "
                        + " or ".join(rendered)
                    )
                segments = alternatives[0]
        if not segments or any(segment not in self.allowed_token_ids for segment in segments):
            raise ValueError(f"unknown Pinyin segment in {segments!r}")
        return segments

    def _prompt(self, context: str, pinyin: Sequence[str]) -> tuple[list[int], list[int]]:
        context_ids = self.tokenizer.encode(
            context,
            add_special_tokens=False,
        )
        pinyin_ids = self.tokenizer.convert_tokens_to_ids(
            [f"[{segment}]" for segment in pinyin]
        )
        if self.tokenizer.unk_token_id in pinyin_ids:
            raise ValueError(f"checkpoint has no token for one of {pinyin!r}")
        prompt_ids = [
            self.tokenizer.cls_token_id,
            *context_ids,
            self.tokenizer.sep_token_id,
            *pinyin_ids,
            self.tokenizer.sep_token_id,
        ]
        first_separator = 1 + len(context_ids)
        prompt_positions = list(range(first_separator + 1)) + list(
            range(first_separator + 1, first_separator + len(pinyin) + 2)
        )
        if len(prompt_ids) != len(prompt_positions):
            raise AssertionError("Concat prompt/position alignment failed")
        return prompt_ids, prompt_positions

    def _model_input_tokens(self, prompt_ids: Sequence[int]) -> tuple[str, ...]:
        return tuple(self.tokenizer.convert_ids_to_tokens(list(prompt_ids)))

    def effective_context(self, context: str, *, token_limit: int = 512) -> str:
        """Apply the official benchmark's fixed leading-context token window."""

        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        if len(context_ids) <= token_limit:
            return context
        return self.tokenizer.decode(
            context_ids[:token_limit],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def generate(
        self,
        context: str,
        typed_pinyin: str | Sequence[str],
        *,
        top_k: int = 10,
        beam_size: int = 16,
    ) -> PinyinGPTResult:
        if top_k < 1 or beam_size < top_k:
            raise ValueError("beam_size must be at least top_k, and top_k must be positive")
        pinyin = self.segment_pinyin(typed_pinyin)
        prompt_ids, prompt_positions = self._prompt(context, pinyin)
        beams: list[tuple[list[int], float]] = [([], 0.0)]

        with self.torch.inference_mode():
            for step, segment in enumerate(pinyin):
                sequences = [prompt_ids + generated for generated, _ in beams]
                positions = [
                    prompt_positions
                    + [prompt_positions[-len(pinyin) - 1] + offset for offset in range(len(generated))]
                    for generated, _ in beams
                ]
                input_ids = self.torch.tensor(sequences, device=self.device)
                position_ids = self.torch.tensor(positions, device=self.device)
                logits = self.model(input_ids=input_ids, position_ids=position_ids).logits[:, -1]
                log_probabilities = self.torch.log_softmax(logits.float(), dim=-1)
                allowed = self.torch.tensor(
                    self.allowed_token_ids[segment], device=self.device
                )
                if allowed.numel() == 0:
                    raise ValueError(f"no tokenizer candidates for Pinyin {segment!r}")
                constrained = log_probabilities.index_select(1, allowed)
                prior = self.torch.tensor(
                    [score for _, score in beams], device=self.device
                ).unsqueeze(1)
                combined = constrained + prior
                keep = min(beam_size, combined.numel())
                values, flat_indices = combined.flatten().topk(keep)
                next_beams = []
                width = allowed.numel()
                for value, flat_index in zip(values.tolist(), flat_indices.tolist()):
                    beam_index = flat_index // width
                    token_index = flat_index % width
                    next_beams.append(
                        (beams[beam_index][0] + [allowed[token_index].item()], value)
                    )
                beams = next_beams

        candidates = []
        seen = set()
        for generated, score in beams:
            text = "".join(self.tokenizer.convert_ids_to_tokens(generated))
            if text in seen:
                continue
            seen.add(text)
            candidates.append(
                CandidateScore(
                    text=text,
                    rank=len(candidates) + 1,
                    log_probability=score,
                    mean_log_probability=score / len(pinyin),
                )
            )
            if len(candidates) == top_k:
                break
        typed = typed_pinyin if isinstance(typed_pinyin, str) else " ".join(typed_pinyin)
        return PinyinGPTResult(
            context=context,
            typed_pinyin=typed,
            segmented_pinyin=pinyin,
            model_input_tokens=self._model_input_tokens(prompt_ids),
            candidates=tuple(candidates),
            beam_size=beam_size,
            runtime_device=str(self.device),
        )

    def score_candidates(
        self,
        context: str,
        typed_pinyin: str | Sequence[str],
        candidates: Sequence[str],
    ) -> tuple[CandidateScore, ...]:
        pinyin = self.segment_pinyin(typed_pinyin)
        prompt_ids, prompt_positions = self._prompt(context, pinyin)
        rows = []
        for candidate in candidates:
            characters = list(candidate)
            if len(characters) != len(pinyin):
                raise ValueError(
                    f"candidate {candidate!r} has {len(characters)} characters; "
                    f"expected {len(pinyin)}"
                )
            ids = self.tokenizer.convert_tokens_to_ids(characters)
            for character, token_id, segment in zip(characters, ids, pinyin):
                if token_id not in self.allowed_token_ids[segment]:
                    raise ValueError(
                        f"candidate {candidate!r} is incompatible: {character!r} != {segment!r}"
                    )
            rows.append((candidate, ids))

        scored = []
        with self.torch.inference_mode():
            for candidate, ids in rows:
                sequence = prompt_ids + ids
                output_positions = [
                    prompt_positions[-len(pinyin) - 1] + offset
                    for offset in range(len(ids))
                ]
                position_ids = prompt_positions + output_positions
                logits = self.model(
                    input_ids=self.torch.tensor([sequence], device=self.device),
                    position_ids=self.torch.tensor([position_ids], device=self.device),
                ).logits[0]
                total = 0.0
                for step, token_id in enumerate(ids):
                    distribution = self.torch.log_softmax(
                        logits[len(prompt_ids) - 1 + step].float(), dim=-1
                    )
                    total += distribution[token_id].item()
                scored.append((candidate, total))
        scored.sort(key=lambda item: item[1], reverse=True)
        return tuple(
            CandidateScore(
                text=text,
                rank=index + 1,
                log_probability=score,
                mean_log_probability=score / len(pinyin),
            )
            for index, (text, score) in enumerate(scored)
        )

    def runtime_info(self) -> dict[str, Any]:
        torch = self.torch
        return {
            "checkpoint": CHECKPOINT_ID,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "official_code_revision": OFFICIAL_CODE_REVISION,
            "architecture": "GPT2LMHeadModel",
            "parameter_count": self.parameter_count,
            "layers": self.model.config.n_layer,
            "hidden_size": self.model.config.n_embd,
            "attention_heads": self.model.config.n_head,
            "vocabulary_size": self.model.config.vocab_size,
            "device": str(self.device),
            "device_name": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else "CPU"
            ),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        }
