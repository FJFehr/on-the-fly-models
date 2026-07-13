"""RoPE Canon Looped Transformer.

Looped architecture (fixed pre + looped middle + fixed post) where every
CanonBlock uses RoPE (rotary position embedding) inside attention instead of
the additive sinusoidal PE that lives in the input embedder.

Classes mirror their counterparts in models/canon_transformer.py and
models/looped_transformer.py, with the single addition of RoPE applied to
Q and K after the per-head reshape, before the dot product.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.canon_layer import CanonLayer
from models.canon_transformer import CanonMLP, _make_canon
from models.rope import RoPE
from models.transformer import LayerNorm


class RoPECanonSelfAttention(nn.Module):
    """Multi-head self-attention with optional Canon-B and RoPE on Q/K."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        bias: bool,
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
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        self.canon_b = (
            _make_canon(
                3 * hidden_dim, canon_kernel, canon_activation, canon_residual, canon_causal
            )
            if "B" in canon_set
            else None
        )

        head_dim = hidden_dim // num_heads
        self.rope = RoPE(head_dim=head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.c_attn(x)
        if self.canon_b is not None:
            qkv, _ = self.canon_b(qkv)

        query, key, value = qkv.split(self.hidden_dim, dim=2)

        head_dim = C // self.num_heads
        key = key.view(B, T, self.num_heads, head_dim).transpose(1, 2)
        query = query.view(B, T, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(B, T, self.num_heads, head_dim).transpose(1, 2)

        # Apply RoPE to Q and K before attention.
        query, key = self.rope(query, key)

        if self.flash:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            scores = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
            attended = self.attn_dropout(F.softmax(scores, dim=-1)) @ value

        attended = attended.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(attended))


class RoPECanonBlock(nn.Module):
    """Pre-norm transformer block with Canon layers (A/B/C/D) and RoPE attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        bias: bool,
        block_size: int,
        canon_set: str,
        canon_kernel: int,
        canon_activation: bool,
        canon_residual: bool,
        canon_causal: bool = False,
        use_block_skip: bool = False,
    ):
        super().__init__()
        self.use_block_skip = use_block_skip
        self.ln_1 = LayerNorm(hidden_dim, bias=bias)
        self.attn = RoPECanonSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
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
        x_input = x
        xx = self.ln_1(x)
        if self.canon_a is not None:
            xx, _ = self.canon_a(xx)
        x = x + self.attn(xx)

        hh = self.ln_2(x)
        if self.canon_c is not None:
            hh, _ = self.canon_c(hh)
        x = x + self.mlp(hh)
        if self.use_block_skip:
            x = x + x_input
        return x


class RoPECanonTransformer(nn.Module):
    """Flat (non-looped) transformer with RoPE attention and Canon layers (A/B/C/D).

    Structurally identical to models.canon_transformer.CanonTransformer, with
    RoPECanonBlock (RoPE + optional Canon) in place of CanonBlock -- the flat sibling of
    RoPECanonLoopedTransformer, matching how CanonTransformer/CanonRecursiveTransformer
    already coexist as flat/looped siblings in models/canon_transformer.py.
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
        self.output_dim = output_dim
        self.dropout = dropout
        self.canon_set = canon_set
        self.canon_kernel = canon_kernel
        self.use_output_head = use_output_head

        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.blocks = nn.ModuleList(
            [
                RoPECanonBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    bias=bias,
                    block_size=block_size,
                    canon_set=canon_set,
                    canon_kernel=canon_kernel,
                    canon_activation=canon_activation,
                    canon_residual=canon_residual,
                    canon_causal=canon_causal,
                )
                for _ in range(num_layers)
            ]
        )
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
            f"RoPECanonTransformer(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"layers={self.num_layers}, heads={self.num_heads}, output={self.output_dim}, "
            f"canon_set={self.canon_set!r}, canon_kernel={self.canon_kernel})"
        )


class RoPECanonLoopedTransformer(nn.Module):
    """Looped transformer with RoPE Canon blocks.

    Architecture (uniform width):
        input_projection → pre (1×) → middle (n_loops×, shared) → post (1×) → final_norm

    Architecture (wide middle, inner_dim != hidden_dim):
        input_projection → pre (1×) → up_proj → middle (n_loops×, shared) → down_proj → post (1×) → final_norm

    Position is encoded via RoPE inside each block's attention (applied to Q and K).
    The input embedder should be configured with use_sinusoidal_pe=False.

    Parameter count is independent of n_loops in both variants.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
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
        inner_dim: Optional[int] = None,
        inner_num_heads: Optional[int] = None,
        use_block_skip: bool = False,
        use_loop_skip: bool = False,
        use_inner_bypass: bool = False,
        use_outer_bypass: bool = False,
        use_output_head: bool = True,
    ):
        super().__init__()
        inner_dim = inner_dim if inner_dim is not None else hidden_dim
        inner_num_heads = inner_num_heads if inner_num_heads is not None else num_heads
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if inner_dim % inner_num_heads != 0:
            raise ValueError("inner_dim must be divisible by inner_num_heads.")
        invalid = set(canon_set) - set("ABCD")
        if invalid:
            raise ValueError(
                f"canon_set contains invalid positions {invalid}. Use only 'A', 'B', 'C', 'D'."
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.inner_dim = inner_dim
        self.inner_num_heads = inner_num_heads
        self.n_loops = n_loops
        self.canon_set = canon_set
        self.canon_kernel = canon_kernel
        self.canon_causal = canon_causal
        self.has_wide_middle = inner_dim != hidden_dim
        self.use_loop_skip = use_loop_skip
        self.use_inner_bypass = use_inner_bypass
        self.use_outer_bypass = use_outer_bypass
        self.use_output_head = use_output_head

        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=bias)

        outer_block_kwargs = dict(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            block_size=2048,
            canon_set=canon_set,
            canon_kernel=canon_kernel,
            canon_activation=canon_activation,
            canon_residual=canon_residual,
            canon_causal=canon_causal,
            use_block_skip=use_block_skip,
        )
        inner_block_kwargs = dict(
            hidden_dim=inner_dim,
            num_heads=inner_num_heads,
            dropout=dropout,
            bias=bias,
            block_size=2048,
            canon_set=canon_set,
            canon_kernel=canon_kernel,
            canon_activation=canon_activation,
            canon_residual=canon_residual,
            canon_causal=canon_causal,
            use_block_skip=use_block_skip,
        )
        self.pre_layer = RoPECanonBlock(**outer_block_kwargs)
        self.middle_layer = RoPECanonBlock(**inner_block_kwargs)
        self.post_layer = RoPECanonBlock(**outer_block_kwargs)
        self.final_norm = nn.LayerNorm(hidden_dim)

        if self.has_wide_middle:
            self.up_proj = nn.Linear(hidden_dim, inner_dim, bias=bias)
            self.down_proj = nn.Linear(inner_dim, hidden_dim, bias=bias)

        if use_output_head:
            self.output_head = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.input_projection(inputs)
        h_outer = h  # outer bypass anchor (hidden_dim)
        h = self.pre_layer(h)
        if self.has_wide_middle:
            h = self.up_proj(h)
        h_inner = h  # inner bypass anchor (inner_dim)
        for _ in range(self.n_loops):
            h = self.middle_layer(h)
            if self.use_loop_skip:  # per-iteration h0 injection (old behaviour)
                h = h + h_inner
        if self.use_inner_bypass:
            h = h + h_inner  # single skip over all N loops
        if self.has_wide_middle:
            h = self.down_proj(h)
        h = self.post_layer(h)
        if self.use_outer_bypass:
            h = h + h_outer  # single skip over all 3 blocks
        h = self.final_norm(h)
        if self.use_output_head:
            h = self.output_head(h)
        return h

    def __repr__(self) -> str:
        mid = f"inner={self.inner_dim}" if self.has_wide_middle else f"hidden={self.hidden_dim}"
        return (
            f"RoPECanonLoopedTransformer(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"{mid}, heads={self.num_heads}, n_loops={self.n_loops}, "
            f"canon_set={self.canon_set!r}, canon_kernel={self.canon_kernel})"
        )
