"""Sandwich recursive transformer: fixed pre/post layers with a looped middle layer.

Architecture:
    input_projection  →  pre_layer (1×)  →  middle_layer (n_loops×, shared weights)  →  post_layer (1×)  →  final_norm

The baseline (n_loops=1) is a plain 3-layer transformer at identical parameter cost.
Increasing n_loops adds effective depth with no extra parameters:
    n_loops=1  →  3 effective layers  (baseline)
    n_loops=2  →  4 effective layers
    n_loops=4  →  6 effective layers
"""

import torch
import torch.nn as nn

from models.activations import resolve_activation_fn
from models.canon_transformer import CanonBlock


class SandwichTransformer(nn.Module):
    """Transformer with fixed pre/post layers and a weight-shared looped middle layer.

    Parameter count equals a plain 3-layer transformer regardless of n_loops.

    Interface matches models.recursive_transformer.RecursiveTransformer:
        forward(inputs, src_key_padding_mask=None) -> (batch, seq_len, hidden_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        output_dim: int,
        n_loops: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
        bias: bool = False,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.n_loops = n_loops

        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=bias)

        layer_kwargs = dict(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation=resolve_activation_fn(activation),
            batch_first=True,
            norm_first=True,
            bias=bias,
        )
        self.pre_layer = nn.TransformerEncoderLayer(**layer_kwargs)
        self.middle_layer = nn.TransformerEncoderLayer(**layer_kwargs)
        self.post_layer = nn.TransformerEncoderLayer(**layer_kwargs)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.input_projection(inputs)
        h = self.pre_layer(h, src_key_padding_mask=src_key_padding_mask)
        for _ in range(self.n_loops):
            h = self.middle_layer(h, src_key_padding_mask=src_key_padding_mask)
        h = self.post_layer(h, src_key_padding_mask=src_key_padding_mask)
        return self.final_norm(h)

    def __repr__(self) -> str:
        return (
            f"SandwichTransformer(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"heads={self.num_heads}, n_loops={self.n_loops})"
        )


class CanonSandwichTransformer(nn.Module):
    """Sandwich transformer where all three layers are CanonBlocks.

    Mirrors SandwichTransformer but uses CanonBlock (depthwise 1D conv at
    positions A/B/C/D) for pre, middle, and post layers.

    Parameter count equals a Canon 3-layer transformer regardless of n_loops.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        output_dim: int,
        n_loops: int = 1,
        dropout: float = 0.0,
        bias: bool = False,
        canon_set: str = "ABCD",
        canon_kernel: int = 4,
        canon_activation: bool = True,
        canon_residual: bool = True,
        canon_causal: bool = False,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        invalid = set(canon_set) - set("ABCD")
        if invalid:
            raise ValueError(
                f"canon_set contains invalid positions {invalid}. Use only 'A', 'B', 'C', 'D'."
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.n_loops = n_loops
        self.canon_set = canon_set
        self.canon_kernel = canon_kernel
        self.canon_causal = canon_causal

        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=bias)

        canon_kwargs = dict(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            causal=False,
            block_size=2048,
            canon_set=canon_set,
            canon_kernel=canon_kernel,
            canon_activation=canon_activation,
            canon_residual=canon_residual,
            canon_causal=canon_causal,
        )
        self.pre_layer = CanonBlock(**canon_kwargs)
        self.middle_layer = CanonBlock(**canon_kwargs)
        self.post_layer = CanonBlock(**canon_kwargs)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.input_projection(inputs)
        h = self.pre_layer(h)
        for _ in range(self.n_loops):
            h = self.middle_layer(h)
        h = self.post_layer(h)
        return self.final_norm(h)

    def __repr__(self) -> str:
        return (
            f"CanonSandwichTransformer(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"heads={self.num_heads}, n_loops={self.n_loops}, "
            f"canon_set={self.canon_set!r}, canon_kernel={self.canon_kernel}, "
            f"canon_causal={self.canon_causal})"
        )
