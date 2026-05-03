"""Shared activation helpers used across model components."""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

_ACTIVATION_FNS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
}

_ACTIVATION_MODULES: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


def _normalize_activation_name(name: str) -> str:
    normalized = name.lower()
    if normalized not in _ACTIVATION_FNS:
        msg = f"Unsupported activation {name!r}. Choose from {sorted(_ACTIVATION_FNS)}."
        raise ValueError(msg)
    return normalized


def build_activation(name: str) -> nn.Module:
    """Instantiate a standard pointwise activation module by name."""
    normalized = _normalize_activation_name(name)
    return _ACTIVATION_MODULES[normalized]()


def resolve_activation_fn(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Resolve a standard pointwise activation function by name."""
    normalized = _normalize_activation_name(name)
    return _ACTIVATION_FNS[normalized]


def swiglu(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Apply activation-only SwiGLU to matching gate and value tensors."""
    if gate.shape != value.shape:
        msg = (
            "SwiGLU requires gate and value tensors with matching shapes, got "
            f"{tuple(gate.shape)} and {tuple(value.shape)}."
        )
        raise ValueError(msg)

    return F.silu(gate) * value


class SwiGLU(nn.Module):
    """Parameterless module wrapper for activation-only SwiGLU."""

    def forward(self, gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return swiglu(gate, value)
