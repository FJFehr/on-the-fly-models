"""Transformer with Canon layers at positions A, B, C, and/or D.

Canon layers are depthwise short 1D convolutions placed at specific positions
inside each transformer block. See models/canon_layer.py and:
https://github.com/facebookresearch/PhysicsLM4

Position map (pre-norm block):
  A — applied to norm(x) before self-attention         hidden_dim
  B — applied to concatenated QKV after projection     3 × hidden_dim
  C — applied to norm(h) before the FFN                hidden_dim
  D — applied to fc1(x) before GELU inside the FFN     4 × hidden_dim

The set of active positions is controlled by the `canon_set` string (e.g.
"AC", "ABCD", ""). The transformer interface is a superset of
models.transformer.Transformer.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.activations import swiglu
from models.canon_layer import CanonLayer
from models.transformer import LayerNorm


def _make_canon(
    hidden_size: int,
    canon_kernel: int,
    canon_activation: bool,
    canon_residual: bool,
    canon_causal: bool = False,
) -> CanonLayer:
    return CanonLayer(
        hidden_size=hidden_size,
        kernel_size=canon_kernel,
        bias=False,
        activation="silu" if canon_activation else None,
        residual=canon_residual,
        causal=canon_causal,
        use_fast_conv1d=True,  # falls back gracefully if causal-conv1d not installed
    )


class CanonMLP(nn.Module):
    """GELU MLP with an optional Canon layer at position D (pre-GELU)."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        bias: bool,
        canon_set: str,
        canon_kernel: int,
        canon_activation: bool,
        canon_residual: bool,
        canon_causal: bool = False,
    ):
        super().__init__()
        self.c_fc = nn.Linear(hidden_dim, 4 * hidden_dim, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * hidden_dim, hidden_dim, bias=bias)
        self.c_proj._is_residual_proj = True
        self.dropout = nn.Dropout(dropout)

        self.canon_d = (
            _make_canon(4 * hidden_dim, canon_kernel, canon_activation, canon_residual, canon_causal)
            if "D" in canon_set
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c_fc(x)
        if self.canon_d is not None:
            h, _ = self.canon_d(h)
        h = self.gelu(h)
        return self.dropout(self.c_proj(h))


class CanonZhuMLP(nn.Module):
    """CanonMLP, with an optional SwiGLU path (Zhu/PhysicsLM4-style "tricks").

    use_swiglu=False: identical structure and behaviour to CanonMLP (single c_fc -> Canon-D
    -> GELU -> c_proj). use_swiglu=True: gate_up_proj -> Canon-D on the concatenated
    [gate, up] tensor (matching canon_layer.py's own documented position for Canon-D inside a
    SwiGLU FFN) -> split -> SwiGLU -> down_proj. intermediate_dim defaults to the standard
    LLaMA-style int(2/3 * 4 * hidden_dim), rounded to the nearest multiple of 8, to keep total
    FFN parameter count close to the GELU MLP's (SwiGLU has 3 weight matrices instead of 2).
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        bias: bool,
        canon_set: str,
        canon_kernel: int,
        canon_activation: bool,
        canon_residual: bool,
        canon_causal: bool = False,
        use_swiglu: bool = False,
        intermediate_dim: int | None = None,
    ):
        super().__init__()
        self.use_swiglu = use_swiglu
        self.dropout = nn.Dropout(dropout)

        if not use_swiglu:
            self.c_fc = nn.Linear(hidden_dim, 4 * hidden_dim, bias=bias)
            self.gelu = nn.GELU()
            self.c_proj = nn.Linear(4 * hidden_dim, hidden_dim, bias=bias)
            self.c_proj._is_residual_proj = True
            self.canon_d = (
                _make_canon(4 * hidden_dim, canon_kernel, canon_activation, canon_residual, canon_causal)
                if "D" in canon_set
                else None
            )
            return

        if intermediate_dim is None:
            intermediate_dim = round(2 / 3 * 4 * hidden_dim / 8) * 8
        self.intermediate_dim = intermediate_dim
        self.gate_up_proj = nn.Linear(hidden_dim, 2 * intermediate_dim, bias=bias)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=bias)
        self.down_proj._is_residual_proj = True
        self.canon_d = (
            _make_canon(2 * intermediate_dim, canon_kernel, canon_activation, canon_residual, canon_causal)
            if "D" in canon_set
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_swiglu:
            h = self.c_fc(x)
            if self.canon_d is not None:
                h, _ = self.canon_d(h)
            h = self.gelu(h)
            return self.dropout(self.c_proj(h))

        gate_up = self.gate_up_proj(x)
        if self.canon_d is not None:
            gate_up, _ = self.canon_d(gate_up)
        gate, up = gate_up.split(self.intermediate_dim, dim=-1)
        h = swiglu(gate, up)
        return self.dropout(self.down_proj(h))


class CanonSelfAttention(nn.Module):
    """Multi-head self-attention with an optional Canon layer at position B (post-QKV projection)."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        bias: bool,
        causal: bool,
        block_size: int,
        canon_set: str,
        canon_kernel: int,
        canon_activation: bool,
        canon_residual: bool,
        canon_causal: bool = False,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.c_attn = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.c_proj._is_residual_proj = True
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.causal = causal
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        if self.causal and not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            )

        self.canon_b = (
            _make_canon(3 * hidden_dim, canon_kernel, canon_activation, canon_residual, canon_causal)
            if "B" in canon_set
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.c_attn(x)  # [B, T, 3*C]
        if self.canon_b is not None:
            qkv, _ = self.canon_b(qkv)

        query, key, value = qkv.split(self.hidden_dim, dim=2)

        head_dim = C // self.num_heads
        key = key.view(B, T, self.num_heads, head_dim).transpose(1, 2)
        query = query.view(B, T, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(B, T, self.num_heads, head_dim).transpose(1, 2)

        if self.flash:
            attended = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal,
            )
        else:
            scores = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
            if self.causal:
                scores = scores.masked_fill(
                    self.bias[:, :, :T, :T] == 0, float("-inf")
                )
            attended = self.attn_dropout(F.softmax(scores, dim=-1)) @ value

        attended = attended.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(attended))


class CanonBlock(nn.Module):
    """Pre-norm transformer block with Canon layers at positions A, B, C, D."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        bias: bool,
        causal: bool,
        block_size: int,
        canon_set: str,
        canon_kernel: int,
        canon_activation: bool,
        canon_residual: bool,
        canon_causal: bool = False,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(hidden_dim, bias=bias)
        self.attn = CanonSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            causal=causal,
            block_size=block_size,
            canon_set=canon_set,
            canon_kernel=canon_kernel,
            canon_activation=canon_activation,
            canon_residual=canon_residual,
            canon_causal=canon_causal,
        )
        self.ln_2 = LayerNorm(hidden_dim, bias=bias)
        self.mlp = CanonMLP(
            hidden_dim=hidden_dim,
            dropout=dropout,
            bias=bias,
            canon_set=canon_set,
            canon_kernel=canon_kernel,
            canon_activation=canon_activation,
            canon_residual=canon_residual,
            canon_causal=canon_causal,
        )

        self.canon_a = (
            _make_canon(hidden_dim, canon_kernel, canon_activation, canon_residual, canon_causal)
            if "A" in canon_set
            else None
        )
        self.canon_c = (
            _make_canon(hidden_dim, canon_kernel, canon_activation, canon_residual, canon_causal)
            if "C" in canon_set
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention path with optional position A.
        xx = self.ln_1(x)
        if self.canon_a is not None:
            xx, _ = self.canon_a(xx)
        x = x + self.attn(xx)

        # FFN path with optional position C.
        hh = self.ln_2(x)
        if self.canon_c is not None:
            hh, _ = self.canon_c(hh)
        x = x + self.mlp(hh)

        return x


class CanonTransformer(nn.Module):
    """Transformer encoder with Canon layers at configurable positions A/B/C/D.

    Interface is a superset of models.transformer.Transformer.

    Args:
        input_dim: dimension of raw input features.
        hidden_dim: transformer hidden width (must be divisible by num_heads).
        num_layers: number of CanonBlock layers.
        num_heads: number of attention heads.
        output_dim: output feature dimension (only used when use_output_head=True).
        dropout: dropout probability.
        bias: whether linear layers include bias terms.
        use_output_head: if False, returns hidden states instead of projected output.
        block_size: maximum sequence length (needed for causal mask fallback).
        causal: whether self-attention is causally masked.
        canon_set: string subset of "ABCD" selecting active Canon positions.
        canon_kernel: kernel size for all Canon layers (default 4).
        canon_activation: if True, apply SiLU inside each Canon conv.
        canon_residual: if True, Canon output is x + conv(x); else conv(x) only.
        canon_causal: if False (default), Canon convs see both past and future tokens.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        output_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_output_head: bool = True,
        block_size: int = 2048,
        causal: bool = False,
        canon_set: str = "AC",
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
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.dropout = dropout
        self.canon_set = canon_set
        self.canon_kernel = canon_kernel
        self.canon_causal = canon_causal
        self.use_output_head = use_output_head

        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=bias)

        canon_kwargs = dict(
            canon_set=canon_set,
            canon_kernel=canon_kernel,
            canon_activation=canon_activation,
            canon_residual=canon_residual,
            canon_causal=canon_causal,
        )
        self.blocks = nn.ModuleList([
            CanonBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                causal=causal,
                block_size=block_size,
                **canon_kwargs,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

        if use_output_head:
            self.output_head = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.input_projection(inputs)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        if self.use_output_head:
            h = self.output_head(h)
        return h

    def __repr__(self) -> str:
        return (
            f"CanonTransformer(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"layers={self.num_layers}, heads={self.num_heads}, output={self.output_dim}, "
            f"canon_set={self.canon_set!r}, canon_kernel={self.canon_kernel}, "
            f"canon_causal={self.canon_causal}, dropout={self.dropout})"
        )


class CanonRecursiveTransformer(nn.Module):
    """Weight-shared transformer whose block includes Canon layers at A/B/C/D.

    Mirrors models.recursive_transformer.RecursiveTransformer, replacing the
    nn.TransformerEncoderLayer block with CanonBlock so that all active Canon
    positions recur on every loop iteration.

    Interface is a superset of RecursiveTransformer: same positional args plus
    canon_set / canon_kernel / canon_activation / canon_residual / canon_causal.

    Args:
        input_dim: dimension of raw input features.
        hidden_dim: transformer hidden width (must be divisible by num_heads).
        num_layers: number of CanonBlocks in the shared block.
        num_heads: number of attention heads.
        output_dim: accepted for interface compatibility; unused (no output head).
        n_loops: how many times the shared block is applied per forward pass.
        dropout: dropout probability.
        bias: whether linear layers include bias terms.
        canon_set: string subset of "ABCD" selecting active Canon positions.
        canon_kernel: depthwise conv kernel size for all Canon layers.
        canon_activation: if True, apply SiLU inside each Canon conv.
        canon_residual: if True, Canon output = x + conv(x).
        canon_causal: if False (default), Canon convs see both past and future tokens.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        output_dim: int,
        n_loops: int = 4,
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
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.n_loops = n_loops
        self.dropout = dropout
        self.canon_set = canon_set
        self.canon_kernel = canon_kernel
        self.canon_causal = canon_causal

        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=bias)

        canon_kwargs = dict(
            canon_set=canon_set,
            canon_kernel=canon_kernel,
            canon_activation=canon_activation,
            canon_residual=canon_residual,
            canon_causal=canon_causal,
        )
        # Shared block — weights are reused across all n_loops iterations.
        self.block = nn.ModuleList([
            CanonBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                causal=False,
                block_size=2048,
                **canon_kwargs,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.input_projection(inputs)
        for _ in range(self.n_loops):
            for layer in self.block:
                h = layer(h)
        return self.final_norm(h)

    def __repr__(self) -> str:
        return (
            f"CanonRecursiveTransformer(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"layers={self.num_layers}, heads={self.num_heads}, n_loops={self.n_loops}, "
            f"canon_set={self.canon_set!r}, canon_kernel={self.canon_kernel}, "
            f"canon_causal={self.canon_causal}, dropout={self.dropout})"
        )
