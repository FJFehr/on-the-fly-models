"""
Canon Layer: depthwise (optionally causal) short convolution for transformer token-mixing.

Based on https://github.com/facebookresearch/PhysicsLM4
Paper: "Physics of Language Models: Part 4.1, Architecture Design and the Magic
of Canon Layers", Zeyuan Allen-Zhu, NeurIPS 2025. Apache 2.0 license.

A Canon Layer is a nn.Conv1d with groups=hidden_size (depthwise), a small
causal kernel, and an optional SiLU activation. It can be placed at four
positions (A/B/C/D) inside a transformer block to add short-range token mixing
at near-zero parameter cost.

Positions in a standard pre-norm transformer block:
  A — before attention, applied to norm(x)
  B — inside attention, on concatenated [q, k, v] after linear projections
  C — before FFN, applied to norm(h)
  D — inside SwiGLU FFN, on concatenated gate tensors [x1, x3]
"""

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    import torch._dynamo

    @torch._dynamo.disable
    def _fast_causal_conv1d(*args, **kwargs):
        return causal_conv1d_fn(*args, **kwargs)

    _HAS_CAUSAL_CONV1D = True
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None
    _HAS_CAUSAL_CONV1D = False

# causal-conv1d CUDA kernel supports kernel sizes 2, 3, 4 only.
_FAST_KERNEL_SIZES = frozenset({2, 3, 4})


class CanonLayer(nn.Conv1d):
    """Depthwise short 1D convolution — the Canon Layer from PhysicsLM4.

    Input/output: [batch, seq_len, hidden_size]  (channels-last)

    Args:
        hidden_size: channel count (each channel gets its own kernel — depthwise).
        kernel_size: receptive field width (default 4).
        bias: learnable bias term (default False, consistent with original).
        activation: pointwise activation after the conv; 'silu'/'swish' or None.
        residual: if True, output = x + conv(x); if False, output = conv(x).
        causal: if True (default), only past tokens are visible (left-padded trim).
                if False, a centered trim gives roughly symmetric context.
        use_fast_conv1d: use causal-conv1d CUDA kernel when available (causal=True only).
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        bias: bool = False,
        activation: Optional[str] = "silu",
        residual: bool = True,
        causal: bool = True,
        use_fast_conv1d: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,  # depthwise: one kernel per channel
            bias=bias,
            padding=kernel_size - 1,  # left-pad; trimmed to seq_len in forward
            device=device,
            dtype=dtype,
        )
        self.hidden_size = hidden_size
        self.causal = causal

        if activation is not None and activation not in ("silu", "swish"):
            raise ValueError(
                f"Unsupported activation '{activation}'. Choose 'silu', 'swish', or None."
            )
        self.activation = activation
        self.residual = residual

        # Fast causal-conv1d kernel is only valid in causal mode.
        if not causal:
            use_fast_conv1d = False
        if use_fast_conv1d:
            if not _HAS_CAUSAL_CONV1D:
                warnings.warn(
                    "causal-conv1d is not installed; falling back to pure PyTorch. "
                    "Run `pip install causal-conv1d>=1.4.0` for faster kernels.",
                    ImportWarning,
                    stacklevel=2,
                )
                use_fast_conv1d = False
            elif kernel_size not in _FAST_KERNEL_SIZES:
                warnings.warn(
                    f"causal-conv1d kernel supports sizes {sorted(_FAST_KERNEL_SIZES)} only; "
                    f"kernel_size={kernel_size} will fall back to pure PyTorch.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                use_fast_conv1d = False
        self.use_fast_conv1d = use_fast_conv1d

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B, T, D] input tensor.
            mask: optional [B, T] binary mask (1 = valid token, 0 = padding).
            cache: optional [B, D, kernel_size] conv state for autoregressive decoding.
                   Updated in-place when provided.
            output_final_state: allocate and return a state tensor even when cache
                                 is None (useful to seed future step-by-step decoding).

        Returns:
            (output [B, T, D], state [B, D, kernel_size] or None)
        """
        B, T, D = x.shape

        if mask is not None:
            x = x * mask.unsqueeze(-1)

        residual = x  # captured after masking so residual doesn't restore padded positions

        if output_final_state and cache is None:
            cache = x.new_zeros(B, D, self.kernel_size[0])

        # Single-token path for autoregressive decoding.
        if cache is not None and T == 1:
            out, cache = self._step(x, cache)
            if self.residual:
                out = residual + out
            return out, cache

        # Full-sequence path.
        x_chf = x.transpose(1, 2)  # [B, D, T] — channels-first for Conv1d

        # Save state for streaming (last kernel_size tokens).
        if cache is not None:
            cache.copy_(F.pad(x_chf, (self.kernel_size[0] - T, 0)))

        if self.use_fast_conv1d:
            x_chf = _fast_causal_conv1d(
                x=x_chf,
                weight=self.weight.squeeze(1),  # [D, W]
                bias=self.bias,
                activation=self.activation,
            )
        else:
            raw = self._conv_forward(x_chf, self.weight, self.bias)
            if self.causal:
                # Left-pad only: trim right side to enforce causality.
                x_chf = raw[..., :T]
            else:
                # Centered trim: roughly symmetric past/future context.
                left = (self.kernel_size[0] - 1) // 2
                x_chf = raw[..., left : left + T]
            if self.activation is not None:
                x_chf = F.silu(x_chf)

        out = x_chf.transpose(1, 2)  # [B, T, D]
        if self.residual:
            out = residual + out
        return out, cache

    # ------------------------------------------------------------------
    # Single-token autoregressive step
    # ------------------------------------------------------------------

    def _step(
        self, x: torch.Tensor, cache: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update rolling cache with one new token and return the output."""
        if not self.causal:
            raise NotImplementedError(
                "Step-mode autoregressive decoding is not supported for non-causal CanonLayer "
                "(causal=False). Use full-sequence forward instead."
            )
        assert x.shape[1] == 1, "_step only supports T=1"
        x = x.squeeze(1)  # [B, D]

        if self.use_fast_conv1d:
            x = causal_conv1d_update(
                x=x,
                conv_state=cache,
                weight=self.weight.squeeze(1),  # [D, W]
                bias=self.bias,
                activation=self.activation,
            )
        else:
            dtype = x.dtype
            # Shift oldest token out, append new token at the end.
            cache.copy_(torch.roll(cache, shifts=-1, dims=-1))
            cache[:, :, -1] = x
            # Weighted sum over the sliding window.
            x = (cache * self.weight.squeeze(1)).sum(dim=-1)
            if self.bias is not None:
                x = x + self.bias
            if self.activation is not None:
                x = F.silu(x).to(dtype)

        return x.unsqueeze(1), cache  # [B, 1, D]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def state_size(self) -> int:
        """Number of elements in the rolling cache (hidden_size × kernel_size)."""
        return self.hidden_size * self.kernel_size[0]

    def __repr__(self) -> str:
        return (
            f"CanonLayer(hidden_size={self.hidden_size}, "
            f"kernel_size={self.kernel_size[0]}, "
            f"causal={self.causal}, "
            f"activation={self.activation!r}, "
            f"residual={self.residual}, "
            f"bias={self.bias is not None})"
        )
