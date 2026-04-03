"""Simple transformer used by the simplified hypermodel path."""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias parameter."""

    def __init__(self, hidden_dim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim)) if bias else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(inputs, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    """Multi-head self-attention over token features."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        bias: bool,
        causal: bool,
        block_size: int,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            msg = "hidden_dim must be divisible by num_heads."
            raise ValueError(msg)

        self.c_attn = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
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

    def forward(
        self,
        inputs: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = inputs.shape

        # One projection produces query, key, and value for every token.
        query, key, value = self.c_attn(inputs).split(self.hidden_dim, dim=2)

        # Split the hidden dimension into attention heads before computing scores.
        head_dim = hidden_dim // self.num_heads
        key = key.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        query = query.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)

        attention_weights = None
        if self.flash and not return_attention_weights:
            attended = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal,
            )
        else:
            scores = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
            if self.causal:
                scores = scores.masked_fill(
                    self.bias[:, :, :seq_len, :seq_len] == 0,
                    float("-inf"),
                )
            attention_weights = F.softmax(scores, dim=-1)
            attended = self.attn_dropout(attention_weights) @ value

        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        output = self.resid_dropout(self.c_proj(attended))
        if return_attention_weights:
            return output, attention_weights
        return output


class MLP(nn.Module):
    """Position-wise feed-forward network inside each block."""

    def __init__(self, hidden_dim: int, dropout: float, bias: bool):
        super().__init__()
        self.c_fc = nn.Linear(hidden_dim, 4 * hidden_dim, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * hidden_dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc(inputs)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return self.dropout(hidden_states)


class Block(nn.Module):
    """Pre-norm transformer block with residual attention and MLP paths."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        bias: bool,
        causal: bool,
        block_size: int,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(hidden_dim, bias=bias)
        self.attn = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            causal=causal,
            block_size=block_size,
        )
        self.ln_2 = LayerNorm(hidden_dim, bias=bias)
        self.mlp = MLP(hidden_dim=hidden_dim, dropout=dropout, bias=bias)

    def forward(
        self,
        inputs: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attention_outputs = self.attn(
            self.ln_1(inputs),
            return_attention_weights=return_attention_weights,
        )
        if return_attention_weights:
            attention_update, attention_weights = attention_outputs
        else:
            attention_update = attention_outputs
            attention_weights = None

        # Attention updates the token states first, then the MLP adds a second residual update.
        hidden_states = inputs + attention_update
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))

        if return_attention_weights:
            return hidden_states, attention_weights
        return hidden_states


class Transformer(nn.Module):
    """Transformer encoder over dense per-token features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.output_dim = output_dim
        # This is just the maximum sequence length this simplified encoder accepts.
        self.block_size = 8192

        bias = False
        causal = False

        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                Block(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    bias=bias,
                    causal=causal,
                    block_size=self.block_size,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = LayerNorm(hidden_dim, bias=bias)
        self.output_head = nn.Linear(hidden_dim, output_dim, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        inputs: torch.Tensor,
        return_attentions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        _, seq_len, _ = inputs.shape
        if seq_len > self.block_size:
            msg = (
                f"Cannot forward sequence of length {seq_len}, block size is only "
                f"{self.block_size}."
            )
            raise ValueError(msg)

        # Project raw input features into the transformer hidden width.
        hidden_states = self.input_projection(inputs)
        hidden_states = self.dropout(hidden_states)

        attentions = []
        # Each block refines the token states with attention, then an MLP update.
        for block in self.blocks:
            if return_attentions:
                hidden_states, attention_weights = block(
                    hidden_states,
                    return_attention_weights=True,
                )
                attentions.append(attention_weights)
            else:
                hidden_states = block(hidden_states)

        # Final normalization and output projection produce per-token outputs.
        hidden_states = self.ln_f(hidden_states)
        outputs = self.output_head(hidden_states)
        if return_attentions:
            return outputs, attentions
        return outputs

    def __repr__(self) -> str:
        return (
            f"Transformer(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"layers={self.num_layers}, heads={self.num_heads}, output={self.output_dim})"
        )
