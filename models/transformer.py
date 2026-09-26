"""Pre-norm transformer with rotary position embeddings and optional Canon layers.

This one class is used everywhere in the paper: as the directly trained model, as the
target model whose weights the hypernetwork generates, and as the hypernetwork's encoder.

Canon layers (models/canon.py) can sit at four positions in each block, chosen by the
letters in `canon_set` ("ABCD" for all four, "" for none):
  A: on the normed input, before attention      (hidden_dim channels)
  B: on the concatenated query/key/value         (3 * hidden_dim channels)
  C: on the normed input, before the feed-forward (hidden_dim channels)
  D: inside the feed-forward, before the GELU    (4 * hidden_dim channels)

Attention is bidirectional and has no padding mask: padded positions reach the model as
zero vectors (see the Lightning modules) and are ignored by the loss, but real tokens can
still attend to them. Every paper result was produced this way.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.canon import CanonConv
from models.rope import RoPE


class LayerNorm(nn.Module):
    """LayerNorm with a learned scale and no bias."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, None, 1e-5)


class SelfAttention(nn.Module):
    """Multi-head self-attention with RoPE, and Canon B on the query/key/value."""

    def __init__(
        self, dim: int, num_heads: int, dropout: float, canon_set: str, canon_kernel: int
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"hidden_dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.dropout = dropout
        self.c_attn = nn.Linear(dim, 3 * dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.canon_b = CanonConv(3 * dim, canon_kernel) if "B" in canon_set else None
        self.rope = RoPE(head_dim=dim // num_heads)
        # The hypernetwork sets this to False: the fused attention kernel's backward pass
        # does not work under torch.vmap, while the plain matmul/softmax version does.
        self.flash = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        head_dim = dim // self.num_heads

        qkv = self.c_attn(x)
        if self.canon_b is not None:
            qkv = self.canon_b(qkv)
        query, key, value = (
            t.view(batch, seq_len, self.num_heads, head_dim).transpose(1, 2)
            for t in qkv.split(dim, dim=2)
        )
        query, key = self.rope(query, key)

        if self.flash:
            dropout_p = self.dropout if self.training else 0.0
            out = F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p)
        else:
            scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)
            out = self.attn_dropout(F.softmax(scores, dim=-1)) @ value

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.resid_dropout(self.c_proj(out))


class FeedForward(nn.Module):
    """GELU feed-forward (4x expansion), with Canon D before the GELU."""

    def __init__(self, dim: int, dropout: float, canon_set: str, canon_kernel: int):
        super().__init__()
        self.c_fc = nn.Linear(dim, 4 * dim, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.canon_d = CanonConv(4 * dim, canon_kernel) if "D" in canon_set else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c_fc(x)
        if self.canon_d is not None:
            h = self.canon_d(h)
        return self.dropout(self.c_proj(self.gelu(h)))


class Block(nn.Module):
    """Pre-norm transformer block, with Canon A before attention and Canon C before the MLP."""

    def __init__(
        self, dim: int, num_heads: int, dropout: float, canon_set: str, canon_kernel: int
    ):
        super().__init__()
        # Submodules are created in this order so a given seed gives the same initial
        # weights as the models the paper results were first produced with.
        self.ln_1 = LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, dropout, canon_set, canon_kernel)
        self.ln_2 = LayerNorm(dim)
        self.mlp = FeedForward(dim, dropout, canon_set, canon_kernel)
        self.canon_a = CanonConv(dim, canon_kernel) if "A" in canon_set else None
        self.canon_c = CanonConv(dim, canon_kernel) if "C" in canon_set else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_1(x)
        if self.canon_a is not None:
            h = self.canon_a(h)
        x = x + self.attn(h)

        h = self.ln_2(x)
        if self.canon_c is not None:
            h = self.canon_c(h)
        return x + self.mlp(h)


class Transformer(nn.Module):
    """input projection -> num_layers blocks -> final LayerNorm -> optional output head.

    With `output_dim=None` there is no output head and the model returns hidden states.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        output_dim: int | None = None,
        dropout: float = 0.0,
        canon_set: str = "ABCD",
        canon_kernel: int = 5,
    ):
        super().__init__()
        if set(canon_set) - set("ABCD"):
            raise ValueError(
                f"canon_set may only contain the letters A, B, C, D, got {canon_set!r}."
            )
        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.blocks = nn.ModuleList(
            Block(hidden_dim, num_heads, dropout, canon_set, canon_kernel)
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_head = (
            nn.Linear(hidden_dim, output_dim, bias=False) if output_dim is not None else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        if self.output_head is not None:
            h = self.output_head(h)
        return h
