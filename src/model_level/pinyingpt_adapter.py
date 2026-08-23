"""Serial bottleneck Adapters for a frozen Hugging Face GPT-2 model.

The frozen checkpoint must be loaded before calling :func:`install_adapters`.
Only Adapter tensors are exported; wrapped base-model tensors are never saved
as part of a per-user Adapter bundle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from torch import nn


ADAPTER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SerialAdapterConfig:
    reduction_factor: int = 16
    activation: str = "relu"
    residual_scale: float = 1.0
    zero_init_up_projection: bool = True
    extra_layer_norm: bool = False
    dropout: float = 0.0
    gate: bool = False

    def validate(self, hidden_size: int) -> None:
        if self.reduction_factor < 1 or hidden_size % self.reduction_factor:
            raise ValueError("hidden size must be divisible by the positive reduction factor")
        if self.activation != "relu":
            raise ValueError("Adapter V1 freezes activation='relu'")
        if self.residual_scale != 1.0:
            raise ValueError("Adapter V1 freezes residual_scale=1")
        if not self.zero_init_up_projection:
            raise ValueError("Adapter V1 requires zero-initialized W_up")
        if self.extra_layer_norm or self.dropout != 0.0 or self.gate:
            raise ValueError("Adapter V1 has no extra LayerNorm, dropout, or gate")


class BottleneckAdapter(nn.Module):
    def __init__(self, hidden_size: int, config: SerialAdapterConfig) -> None:
        super().__init__()
        config.validate(hidden_size)
        bottleneck = hidden_size // config.reduction_factor
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck
        self.residual_scale = config.residual_scale
        self.down = nn.Linear(hidden_size, bottleneck, bias=True)
        self.activation = nn.ReLU()
        self.up = nn.Linear(bottleneck, hidden_size, bias=True)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        update = self.up(self.activation(self.down(hidden_states)))
        return hidden_states + self.residual_scale * update


class AdapterWrappedBlock(nn.Module):
    """Preserve the GPT2Block interface while adapting its completed output."""

    def __init__(self, block: nn.Module, adapter: BottleneckAdapter) -> None:
        super().__init__()
        self.block = block
        self.adapter = adapter

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        output = self.block(*args, **kwargs)
        if output is None:
            return None
        if isinstance(output, tuple):
            if not output or not isinstance(output[0], torch.Tensor):
                raise TypeError("GPT2Block returned an unsupported tuple")
            return (self.adapter(output[0]), *output[1:])
        if isinstance(output, torch.Tensor):
            return self.adapter(output)
        raise TypeError(f"GPT2Block returned unsupported output type {type(output)!r}")


def _block_list(model: nn.Module) -> nn.ModuleList:
    transformer = getattr(model, "transformer", None)
    blocks = getattr(transformer, "h", None)
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError("expected GPT2LMHeadModel.transformer.h to be an nn.ModuleList")
    return blocks


def iter_adapter_blocks(model: nn.Module) -> Iterator[tuple[int, AdapterWrappedBlock]]:
    for index, block in enumerate(_block_list(model)):
        if isinstance(block, AdapterWrappedBlock):
            yield index, block


def install_adapters(
    model: nn.Module,
    config: SerialAdapterConfig = SerialAdapterConfig(),
    *,
    expected_layers: int = 12,
) -> dict[str, int]:
    """Freeze a loaded base model and install one independent Adapter per block."""

    blocks = _block_list(model)
    if any(isinstance(block, AdapterWrappedBlock) for block in blocks):
        raise RuntimeError("Adapters are already installed")
    if len(blocks) != expected_layers:
        raise ValueError(f"expected {expected_layers} Transformer blocks, found {len(blocks)}")
    hidden_size = int(getattr(model.config, "n_embd"))
    config.validate(hidden_size)

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for index, block in enumerate(tuple(blocks)):
        reference_parameter = next(block.parameters(), None)
        if reference_parameter is None:
            reference_parameter = next(model.parameters())
        adapter = BottleneckAdapter(hidden_size, config).to(
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )
        blocks[index] = AdapterWrappedBlock(block, adapter)

    # Keep every frozen dropout in evaluation mode.  Adapter V1 has no dropout.
    set_adapter_training_mode(model, enabled=False)
    base_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".adapter." not in name
    )
    adapter_parameters = sum(
        parameter.numel()
        for _, wrapped in iter_adapter_blocks(model)
        for parameter in wrapped.adapter.parameters()
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameters != adapter_parameters:
        raise AssertionError("trainable parameter surface is not Adapter-only")
    return {
        "layers": len(blocks),
        "hidden_size": hidden_size,
        "bottleneck_size": hidden_size // config.reduction_factor,
        "base_parameters": base_parameters,
        "adapter_parameters": adapter_parameters,
        "trainable_parameters": trainable_parameters,
    }


def set_adapter_training_mode(model: nn.Module, *, enabled: bool) -> None:
    """Keep the frozen base in eval mode while toggling only Adapter modules."""

    model.eval()
    found = 0
    for _, wrapped in iter_adapter_blocks(model):
        wrapped.block.eval()
        wrapped.adapter.train(enabled)
        found += 1
    if not found:
        raise RuntimeError("no installed Adapters found")


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    blocks = list(iter_adapter_blocks(model))
    if not blocks:
        raise RuntimeError("no installed Adapters found")
    for index, wrapped in blocks:
        for name, tensor in wrapped.adapter.state_dict().items():
            state[f"layers.{index}.{name}"] = tensor.detach().cpu().contiguous()
    return state


def load_adapter_state_dict(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    strict: bool = True,
) -> None:
    expected = adapter_state_dict(model)
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if strict and (missing or unexpected):
        raise RuntimeError(f"Adapter state mismatch: missing={missing}, unexpected={unexpected}")
    for index, wrapped in iter_adapter_blocks(model):
        prefix = f"layers.{index}."
        layer_state = {
            key[len(prefix) :]: tensor
            for key, tensor in state.items()
            if key.startswith(prefix)
        }
        wrapped.adapter.load_state_dict(layer_state, strict=strict)


def save_adapter_bundle(
    model: nn.Module,
    weights_path: Path,
    *,
    config: SerialAdapterConfig,
    metadata: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Write Adapter-only safetensors plus deterministic JSON metadata."""

    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - environment gate
        raise RuntimeError("safetensors is required for Adapter persistence") from error
    if weights_path.suffix != ".safetensors":
        raise ValueError("Adapter weight path must end in .safetensors")
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(adapter_state_dict(model), str(weights_path))
    metadata_path = weights_path.with_suffix(".json")
    payload = {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "architecture": "serial_bottleneck_residual_adapter_after_each_gpt2_block",
        "adapter_config": asdict(config),
        **dict(metadata),
    }
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return weights_path, metadata_path


def load_adapter_bundle(model: nn.Module, weights_path: Path) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover - environment gate
        raise RuntimeError("safetensors is required for Adapter persistence") from error
    metadata_path = weights_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != ADAPTER_SCHEMA_VERSION:
        raise RuntimeError("unsupported Adapter bundle schema")
    load_adapter_state_dict(model, load_file(str(weights_path), device="cpu"), strict=True)
    return metadata


def gradient_audit(model: nn.Module) -> dict[str, object]:
    adapter_with_gradient: list[str] = []
    adapter_nonzero_gradient: list[str] = []
    base_with_gradient: list[str] = []
    for name, parameter in model.named_parameters():
        if ".adapter." in name:
            if parameter.grad is not None:
                adapter_with_gradient.append(name)
                if torch.count_nonzero(parameter.grad.detach()).item() > 0:
                    adapter_nonzero_gradient.append(name)
        elif parameter.grad is not None:
            base_with_gradient.append(name)
    return {
        "adapter_parameters_with_gradient": adapter_with_gradient,
        "adapter_parameters_with_nonzero_gradient": adapter_nonzero_gradient,
        "base_parameters_with_gradient": base_with_gradient,
        "adapter_gradient_present": bool(adapter_with_gradient),
        "adapter_nonzero_gradient_present": bool(adapter_nonzero_gradient),
        "base_gradient_absent": not base_with_gradient,
    }
