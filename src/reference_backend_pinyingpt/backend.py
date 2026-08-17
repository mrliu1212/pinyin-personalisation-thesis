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

    def truncate_context_for_generation(
        self, context: str, typed_pinyin: str | Sequence[str]
    ) -> tuple[str, int, int, bool]:
        """Keep the most recent context that fits the complete Concat generation."""

        pinyin = self.segment_pinyin(typed_pinyin)
        original_ids = self.tokenizer.encode(context, add_special_tokens=False)
        maximum = int(self.model.config.n_positions)
        # [CLS], context, [SEP], Pinyin, [SEP], and generated target positions.
        available = maximum - (2 + 2 * len(pinyin))
        if available < 0:
            raise ValueError("Pinyin target exceeds the model position limit")
        if len(original_ids) <= available:
            return context, len(original_ids), len(original_ids), False
        low, high = 0, len(context)
        while low < high:
            middle = (low + high) // 2
            length = len(self.tokenizer.encode(context[middle:], add_special_tokens=False))
            if length <= available:
                high = middle
            else:
                low = middle + 1
        used = context[low:]
        used_tokens = len(self.tokenizer.encode(used, add_special_tokens=False))
        return used, len(original_ids), used_tokens, True

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

    def generate_batch(
        self,
        requests: Sequence[tuple[str, Sequence[str]]],
        *,
        top_k: int = 10,
        beam_size: int = 16,
    ) -> tuple[PinyinGPTResult, ...]:
        """Decode independent requests in shared forwards without changing beams."""

        if not requests:
            return ()
        if top_k < 1 or beam_size < top_k:
            raise ValueError("beam_size must be at least top_k, and top_k must be positive")
        prepared = []
        for context, typed in requests:
            pinyin = self.segment_pinyin(typed)
            prompt_ids, prompt_positions = self._prompt(context, pinyin)
            prepared.append((context, tuple(typed), pinyin, prompt_ids, prompt_positions))
        lengths = {len(item[2]) for item in prepared}
        sequence_lengths = {len(item[3]) for item in prepared}
        if len(lengths) != 1 or len(sequence_lengths) != 1:
            raise ValueError("batched requests must have equal target and prompt token lengths")
        target_length = lengths.pop()
        beams: list[list[tuple[list[int], float]]] = [[([], 0.0)] for _ in prepared]

        with self.torch.inference_mode():
            prompt_output = self.model(
                input_ids=self.torch.tensor([item[3] for item in prepared], device=self.device),
                position_ids=self.torch.tensor([item[4] for item in prepared], device=self.device),
                use_cache=True,
            )
            logits = prompt_output.logits[:, -1]
            past_key_values = prompt_output.past_key_values
            for step in range(target_length):
                log_probabilities = self.torch.log_softmax(logits.float(), dim=-1)
                next_all = []
                selected_rows = []
                row_start = 0
                for request_index, item_beams in enumerate(beams):
                    segment = prepared[request_index][2][step]
                    allowed = self.torch.tensor(self.allowed_token_ids[segment], device=self.device)
                    row_indices = list(range(row_start, row_start + len(item_beams)))
                    constrained = log_probabilities[row_indices].index_select(1, allowed)
                    prior = self.torch.tensor([score for _, score in item_beams], device=self.device).unsqueeze(1)
                    combined = constrained + prior
                    keep = min(beam_size, combined.numel())
                    values, flat_indices = combined.flatten().topk(keep)
                    width = allowed.numel()
                    next_beams = []
                    for value, flat_index in zip(values.tolist(), flat_indices.tolist()):
                        beam_index = flat_index // width
                        token_index = flat_index % width
                        next_beams.append((item_beams[beam_index][0] + [allowed[token_index].item()], value))
                        selected_rows.append(row_start + beam_index)
                    next_all.append(next_beams)
                    row_start += len(item_beams)
                beams = next_all
                if step + 1 == target_length:
                    break
                selection = self.torch.tensor(selected_rows, device=self.device)
                past_key_values.batch_select_indices(selection)
                previous_tokens = [[generated[-1]] for item_beams in beams for generated, _ in item_beams]
                next_positions = []
                for request_index, item_beams in enumerate(beams):
                    pinyin = prepared[request_index][2]
                    prompt_positions = prepared[request_index][4]
                    position = prompt_positions[-len(pinyin) - 1] + step
                    next_positions.extend([position] for _ in item_beams)
                incremental_output = self.model(
                    input_ids=self.torch.tensor(previous_tokens, device=self.device),
                    position_ids=self.torch.tensor(next_positions, device=self.device),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = incremental_output.logits[:, -1]
                past_key_values = incremental_output.past_key_values

        results = []
        for request_index, item_beams in enumerate(beams):
            context, typed, pinyin, prompt_ids, _ = prepared[request_index]
            candidates = []
            seen = set()
            for generated, score in item_beams:
                value = "".join(self.tokenizer.convert_ids_to_tokens(generated))
                if value in seen:
                    continue
                seen.add(value)
                candidates.append(CandidateScore(text=value, rank=len(candidates) + 1, log_probability=score, mean_log_probability=score / len(pinyin)))
                if len(candidates) == top_k:
                    break
            results.append(PinyinGPTResult(context=context, typed_pinyin=" ".join(typed), segmented_pinyin=pinyin, model_input_tokens=self._model_input_tokens(prompt_ids), candidates=tuple(candidates), beam_size=beam_size, runtime_device=str(self.device)))
        return tuple(results)

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
