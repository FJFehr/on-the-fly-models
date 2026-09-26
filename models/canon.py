"""Canon layer: a short depthwise convolution for local token mixing.

From "Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon
Layers" (Allen-Zhu, NeurIPS 2025), https://github.com/facebookresearch/PhysicsLM4.

Each channel gets its own small kernel, so the layer adds short-range mixing between
neighbouring tokens at almost no parameter cost. Here it is non-causal: the kernel is
centred, so a token sees roughly as many tokens to its left as to its right.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CanonConv(nn.Conv1d):
    """Depthwise 1D convolution over the sequence, with SiLU and a residual connection.

    Input and output shape: (batch, seq_len, dim).
    """

    def __init__(self, dim: int, kernel_size: int):
        super().__init__(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=dim,
            bias=False,
            padding=kernel_size - 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        # Conv1d pads kernel_size - 1 on both sides; keep the centred seq_len window.
        conv = super().forward(x.transpose(1, 2))
        start = (self.kernel_size[0] - 1) // 2
        conv = conv[..., start : start + seq_len]
        return x + F.silu(conv).transpose(1, 2)
